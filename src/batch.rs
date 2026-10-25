//! Dedicated CUDA worker for d3-turbo state-once requests.
//!
//! One engine owns one CUDA context/stream and cannot run concurrent forwards.
//! The HTTP reactor tokenizes and builds question blocks; this thread:
//!
//! 1. coalesces requests that arrive within a short window;
//! 2. groups them by exact state token ids;
//! 3. keeps a pinned exact-state GPU cache plus a host-memory L2 for overflow;
//! 4. runs one state capture per distinct cold state and one combined question
//!    pass per state group.
//!
//! Repeated or concurrent copies of the same state therefore pay the state
//! read once and share the question GEMMs; distinct states remain serialized
//! by the single CUDA stream.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use decidra_native::Engine;
use decidra_native::HostStateCache;
use decidra_native::cuda::PhaseTimer;
use decidra_native::engine::StateCache;
use decidra_native::vision::{VisionGrid, VisionInput};
use tokio::sync::oneshot;

use crate::error::EngineError;
use crate::image_media::ImageMedia;

pub struct BatchHandle {
    tx: SyncSender<Job>,
    queue_slots: Arc<tokio::sync::Semaphore>,
    counters: Arc<QueueCounters>,
}

struct Job {
    state: Vec<u32>,
    questions: Vec<Vec<u32>>,
    images: Arc<Vec<ImageMedia>>,
    state_positions: Arc<Vec<[u32; 3]>>,
    image_spans: Arc<Vec<(usize, usize)>>,
    digest: [u8; 32],
    _queue_slot: tokio::sync::OwnedSemaphorePermit,
    submitted: Instant,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
}

struct CachedState {
    state: Vec<u32>,
    digest: [u8; 32],
    cache: StateCache,
    bytes: usize,
}

struct HostCachedState {
    state: Vec<u32>,
    digest: [u8; 32],
    cache: HostStateCache,
    bytes: usize,
}

#[derive(Clone, Copy)]
struct CacheLimits {
    entries: usize,
    bytes: usize,
    host_bytes: usize,
}

#[derive(Clone, Copy)]
enum CacheResult {
    GpuHit,
    HostHit,
    Miss,
}

impl CacheResult {
    fn label(self) -> &'static str {
        match self {
            Self::GpuHit => "gpu_hit",
            Self::HostHit => "host_hit",
            Self::Miss => "miss",
        }
    }

    fn hit(self) -> bool {
        !matches!(self, Self::Miss)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GroupOutcome {
    Success,
    Failed,
}

impl GroupOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failed => "failed",
        }
    }
}

/// Queue/running/channel counters are the source of truth for the gauges.
/// Saturating decrements keep a worker reset from racing a submit rollback and
/// publishing a negative gauge.
#[derive(Default)]
struct QueueCounters {
    waiting: AtomicUsize,
    running: AtomicUsize,
    channel: AtomicUsize,
}

impl QueueCounters {
    fn inc(counter: &AtomicUsize, gauge: &'static str) {
        let value = counter.fetch_add(1, Ordering::SeqCst) + 1;
        metrics::gauge!(gauge).set(value as f64);
    }

    fn dec(counter: &AtomicUsize, gauge: &'static str) {
        let mut current = counter.load(Ordering::SeqCst);
        loop {
            let next = current.saturating_sub(1);
            match counter.compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => {
                    metrics::gauge!(gauge).set(next as f64);
                    break;
                }
                Err(actual) => current = actual,
            }
        }
    }

    fn waiting_inc(&self) {
        Self::inc(&self.waiting, "decidra_engine_requests_waiting");
    }

    fn waiting_dec(&self) {
        Self::dec(&self.waiting, "decidra_engine_requests_waiting");
    }

    fn running_inc(&self) {
        Self::inc(&self.running, "decidra_engine_requests_running");
    }

    fn running_dec(&self) {
        Self::dec(&self.running, "decidra_engine_requests_running");
    }

    fn channel_inc(&self) {
        Self::inc(&self.channel, "decidra_engine_channel_occupancy");
    }

    fn channel_dec(&self) {
        Self::dec(&self.channel, "decidra_engine_channel_occupancy");
    }

    fn reset(&self) {
        self.waiting.store(0, Ordering::SeqCst);
        self.running.store(0, Ordering::SeqCst);
        self.channel.store(0, Ordering::SeqCst);
        metrics::gauge!("decidra_engine_requests_waiting").set(0.0);
        metrics::gauge!("decidra_engine_requests_running").set(0.0);
        metrics::gauge!("decidra_engine_channel_occupancy").set(0.0);
    }
}

struct WorkerGuard {
    counters: Arc<QueueCounters>,
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        metrics::gauge!("decidra_engine_worker_alive").set(0.0);
        self.counters.reset();
    }
}

struct RunningGuard<'a> {
    counters: &'a QueueCounters,
    count: usize,
}

impl Drop for RunningGuard<'_> {
    fn drop(&mut self) {
        for _ in 0..self.count {
            self.counters.running_dec();
        }
    }
}

/// Leave room for the question-pass scratch (attention scores/masks, GDN
/// recurrence, INT8 accumulation) before launching a group. The pool is
/// trimmed first; cached states are offloaded to the host L2 if needed.
const MIN_PASS_RESERVE: usize = 1536 * 1024 * 1024;

/// Capture and question passes allocate several per-token buffers in addition
/// to the cached K/V. Scale the reserve with the state length; a fixed reserve
/// is enough for 2K but not for 16K-32K states.
fn pass_reserve(state_len: usize) -> usize {
    state_len
        .saturating_mul(192 * 1024)
        .saturating_add(1024 * 1024 * 1024)
        .max(MIN_PASS_RESERVE)
}

/// State cache is dominated by 16 full-attention K/V layers (64 KiB/token)
/// plus the fixed GDN recurrent state.
fn state_cache_bytes_for(state_len: usize) -> usize {
    state_len * 65_536 + 160 * 1024 * 1024
}

fn ratio(used: usize, capacity: usize) -> f64 {
    if capacity == 0 {
        0.0
    } else {
        used as f64 / capacity as f64
    }
}

fn refresh_cache_gauges(
    cache: &[CachedState],
    host_cache: &[HostCachedState],
    limits: CacheLimits,
) {
    let gpu_bytes: usize = cache.iter().map(|entry| entry.bytes).sum();
    let host_bytes: usize = host_cache.iter().map(|entry| entry.bytes).sum();
    metrics::gauge!("decidra_state_cache_entries", "tier" => "gpu").set(cache.len() as f64);
    metrics::gauge!("decidra_state_cache_entries", "tier" => "host").set(host_cache.len() as f64);
    metrics::gauge!("decidra_state_cache_bytes", "tier" => "gpu").set(gpu_bytes as f64);
    metrics::gauge!("decidra_state_cache_bytes", "tier" => "host").set(host_bytes as f64);
    let gpu_ratio = ratio(cache.len(), limits.entries).max(ratio(gpu_bytes, limits.bytes));
    metrics::gauge!("decidra_state_cache_usage_ratio", "tier" => "gpu").set(gpu_ratio);
    metrics::gauge!("decidra_state_cache_usage_ratio", "tier" => "host")
        .set(ratio(host_bytes, limits.host_bytes));
}

fn fail_questions(jobs: &[Job]) {
    let questions: usize = jobs.iter().map(|job| job.questions.len()).sum();
    metrics::counter!("decidra_engine_questions_total", "outcome" => "failed")
        .increment(questions as u64);
}

fn stop_timer(timer: &mut Option<PhaseTimer>, gpu: &decidra_native::cuda::Gpu) {
    let error = timer.as_mut().and_then(|timer| timer.stop(gpu).err());
    if let Some(error) = error {
        tracing::warn!(%error, "metrics: phase timer stop failed");
        *timer = None;
    }
}

fn timer_seconds(timer: &Option<PhaseTimer>) -> Option<f64> {
    timer.as_ref().and_then(|timer| match timer.seconds() {
        Ok(seconds) => Some(seconds),
        Err(error) => {
            tracing::warn!(%error, "metrics: phase timer read failed");
            None
        }
    })
}

impl BatchHandle {
    /// Spawn the single CUDA worker.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        mut engine: Engine,
        max_queue: usize,
        batch_requests: usize,
        batch_rows: usize,
        batch_window_us: u64,
        state_cache_entries: usize,
        state_cache_bytes: usize,
        host_state_cache_bytes: usize,
    ) -> Self {
        let limits = CacheLimits {
            entries: state_cache_entries,
            bytes: state_cache_bytes,
            host_bytes: host_state_cache_bytes,
        };
        let (tx, rx) = mpsc::sync_channel::<Job>(max_queue);
        let queue_slots = Arc::new(tokio::sync::Semaphore::new(max_queue));
        let counters = Arc::new(QueueCounters::default());
        metrics::gauge!("decidra_engine_worker_alive").set(1.0);
        let worker_counters = counters.clone();
        thread::Builder::new()
            .name("decidra-cuda".to_owned())
            .spawn(move || {
                let _worker_guard = WorkerGuard {
                    counters: worker_counters.clone(),
                };
                if let Err(error) = engine.rebind_blas() {
                    tracing::error!(%error, "failed to rebind cuBLAS on CUDA worker");
                }
                let mut cache: Vec<CachedState> = Vec::with_capacity(state_cache_entries);
                let mut host_cache: Vec<HostCachedState> = Vec::with_capacity(state_cache_entries);
                loop {
                    let first = match rx.recv() {
                        Ok(job) => {
                            worker_counters.channel_dec();
                            job
                        }
                        Err(_) => break,
                    };
                    // Budget the *sum* of question rows in the group, not just
                    // rows of the first request. A 1-question request followed
                    // by 64-question requests must not turn a 32-row budget into
                    // a 2,048-row pass.
                    let mut rows = first.questions.len().max(1);
                    let mut jobs = vec![first];
                    let deadline = Instant::now() + Duration::from_micros(batch_window_us);
                    while jobs.len() < batch_requests && rows < batch_rows {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        match rx.recv_timeout(deadline - now) {
                            Ok(job) => {
                                worker_counters.channel_dec();
                                rows = rows.saturating_add(job.questions.len().max(1));
                                jobs.push(job);
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => break,
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                // Flush the jobs already collected, then stop.
                                run_jobs(
                                    &engine,
                                    jobs,
                                    &mut cache,
                                    &mut host_cache,
                                    limits,
                                    &worker_counters,
                                );
                                return;
                            }
                        }
                    }
                    run_jobs(
                        &engine,
                        jobs,
                        &mut cache,
                        &mut host_cache,
                        limits,
                        &worker_counters,
                    );
                }
            })
            .expect("spawning the decidra CUDA worker");
        Self {
            tx,
            queue_slots,
            counters,
        }
    }

    /// Run one state-once request. The receiver resolves when the worker has
    /// finished the request's question batch.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_state_once(
        &self,
        state: Vec<u32>,
        questions: Vec<Vec<u32>>,
        images: Vec<ImageMedia>,
        state_positions: Vec<[u32; 3]>,
        image_spans: Vec<(usize, usize)>,
        digest: [u8; 32],
    ) -> Result<oneshot::Receiver<Result<Vec<Vec<f32>>, String>>, EngineError> {
        let queue_slot = match self.queue_slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => {
                metrics::counter!(
                    "decidra_engine_admission_rejections_total",
                    "reason" => "worker_stopped"
                )
                .increment(1);
                return Err(EngineError::Internal(
                    "d3-turbo CUDA worker stopped".to_owned(),
                ));
            }
        };
        let (reply, receiver) = oneshot::channel();
        let job = Job {
            state,
            questions,
            images: Arc::new(images),
            state_positions: Arc::new(state_positions),
            image_spans: Arc::new(image_spans),
            digest,
            _queue_slot: queue_slot,
            submitted: Instant::now(),
            reply,
        };
        // Count before the send so a fast worker cannot decrement a job that
        // the HTTP thread has not counted yet.
        self.counters.waiting_inc();
        self.counters.channel_inc();
        match self.tx.try_send(job) {
            Ok(()) => Ok(receiver),
            Err(TrySendError::Full(_)) => {
                self.counters.waiting_dec();
                self.counters.channel_dec();
                metrics::counter!(
                    "decidra_engine_admission_rejections_total",
                    "reason" => "queue_full"
                )
                .increment(1);
                Err(EngineError::Overloaded(
                    "d3-turbo queue is full; retry shortly".to_owned(),
                ))
            }
            Err(TrySendError::Disconnected(_)) => {
                self.counters.waiting_dec();
                self.counters.channel_dec();
                metrics::counter!(
                    "decidra_engine_admission_rejections_total",
                    "reason" => "worker_stopped"
                )
                .increment(1);
                Err(EngineError::Internal(
                    "d3-turbo CUDA worker stopped".to_owned(),
                ))
            }
        }
    }
}

fn run_jobs(
    engine: &Engine,
    jobs: Vec<Job>,
    cache: &mut Vec<CachedState>,
    host_cache: &mut Vec<HostCachedState>,
    limits: CacheLimits,
    counters: &QueueCounters,
) {
    // A client that disconnected before the worker starts must not pay for a
    // cold image encode or state capture.
    let mut live = Vec::with_capacity(jobs.len());
    for job in jobs {
        if job.reply.is_closed() {
            counters.waiting_dec();
            metrics::counter!(
                "decidra_engine_requests_cancelled_total",
                "stage" => "queued"
            )
            .increment(1);
            metrics::counter!("decidra_engine_questions_total", "outcome" => "cancelled")
                .increment(job.questions.len() as u64);
        } else {
            live.push(job);
        }
    }
    if live.is_empty() {
        return;
    }
    // Preserve first-seen state order so co-batched replies stay deterministic.
    let mut groups: Vec<(Vec<u32>, [u8; 32], Vec<Job>)> = Vec::new();
    for job in live {
        match groups
            .iter_mut()
            .find(|(state, digest, _)| *state == job.state && *digest == job.digest)
        {
            Some((_, _, members)) => members.push(job),
            None => groups.push((job.state.clone(), job.digest, vec![job])),
        }
    }
    // Cost-aware scheduling, opt-in for A/B: shortest estimated GPU work
    // first. A group older than 500 ms jumps the queue to avoid starvation.
    if std::env::var("DECIDRA_SCHED_PRIORITY")
        .map(|value| value != "0")
        .unwrap_or(false)
    {
        groups.sort_by_key(|(state, digest, jobs)| {
            let age_ms = jobs
                .iter()
                .map(|job| job.submitted.elapsed().as_millis() as usize)
                .max()
                .unwrap_or(0);
            if age_ms >= 500 {
                return 0;
            }
            let questions = jobs
                .iter()
                .flat_map(|job| job.questions.iter())
                .map(|question| question.len())
                .sum::<usize>();
            let state_cost = if cache
                .iter()
                .any(|entry| same_state_identity(state, digest, &entry.state, &entry.digest))
            {
                0
            } else {
                state.len()
            };
            state_cost + questions
        });
    }
    for (state, digest, jobs) in groups {
        run_group(
            engine, &state, digest, jobs, cache, host_cache, limits, counters,
        );
    }
}

fn same_state_identity(
    state: &[u32],
    digest: &[u8; 32],
    cached: &[u32],
    cached_digest: &[u8; 32],
) -> bool {
    state == cached && digest == cached_digest
}

#[allow(clippy::too_many_arguments)]
fn run_group(
    engine: &Engine,
    state: &[u32],
    digest: [u8; 32],
    jobs: Vec<Job>,
    cache: &mut Vec<CachedState>,
    host_cache: &mut Vec<HostCachedState>,
    limits: CacheLimits,
    counters: &QueueCounters,
) {
    let started = Instant::now();
    let job_count = jobs.len();
    let question_count: usize = jobs.iter().map(|job| job.questions.len()).sum();
    for job in &jobs {
        counters.waiting_dec();
        counters.running_inc();
        metrics::histogram!("decidra_request_queue_seconds")
            .record(job.submitted.elapsed().as_secs_f64());
    }
    let running_guard = RunningGuard {
        counters,
        count: job_count,
    };
    metrics::histogram!("decidra_engine_group_requests").record(job_count as f64);
    metrics::histogram!("decidra_engine_group_questions").record(question_count as f64);
    let (cache_result, outcome) =
        run_group_inner(engine, state, digest, jobs, cache, host_cache, limits);
    drop(running_guard);
    let cache_label = cache_result.map(CacheResult::label).unwrap_or("unresolved");
    metrics::histogram!(
        "decidra_engine_group_seconds",
        "cache" => cache_label
    )
    .record(started.elapsed().as_secs_f64());
    metrics::counter!(
        "decidra_engine_groups_total",
        "outcome" => outcome.as_str()
    )
    .increment(1);
    refresh_cache_gauges(cache, host_cache, limits);
    if let Ok((free, _)) = engine.gpu.mem_info() {
        metrics::gauge!("decidra_gpu_memory_free_bytes", "device" => "0").set(free as f64);
    }
    tracing::debug!(
        jobs = job_count,
        rows = question_count,
        state = state.len(),
        cache = cache_label,
        outcome = outcome.as_str(),
        elapsed_ms = started.elapsed().as_secs_f64() * 1e3,
        "state-once group"
    );
}

#[allow(clippy::too_many_arguments)]
fn run_group_inner(
    engine: &Engine,
    state: &[u32],
    digest: [u8; 32],
    jobs: Vec<Job>,
    cache: &mut Vec<CachedState>,
    host_cache: &mut Vec<HostCachedState>,
    limits: CacheLimits,
) -> (Option<CacheResult>, GroupOutcome) {
    let mut cached = cache
        .iter()
        .position(|entry| same_state_identity(state, &digest, &entry.state, &entry.digest));
    let mut host_cached = if cached.is_none() {
        host_cache
            .iter()
            .position(|entry| same_state_identity(state, &digest, &entry.state, &entry.digest))
    } else {
        None
    };
    let pin_cache = std::env::var("DECIDRA_STATE_CACHE_PIN")
        .map(|v| v != "0")
        .unwrap_or(true);
    let entry_bytes = state_cache_bytes_for(state.len());
    if !pin_cache {
        // Preserve the hit itself while evicting older entries. Removing from
        // the front shifts the hit index, and the byte total must be
        // recomputed after each removal.
        let preserved = cached.map(|index| cache.remove(index));
        let max_len = limits
            .entries
            .saturating_sub(usize::from(preserved.is_some()));
        let byte_budget = limits.bytes.saturating_sub(entry_bytes);
        let mut bytes: usize = cache.iter().map(|entry| entry.bytes).sum();
        while !cache.is_empty() && (cache.len() > max_len || bytes > byte_budget) {
            let reason = if cache.len() > max_len {
                "entry_limit"
            } else {
                "byte_limit"
            };
            let removed = cache.remove(0);
            bytes = bytes.saturating_sub(removed.bytes);
            metrics::counter!(
                "decidra_state_cache_evictions_total",
                "tier" => "gpu",
                "reason" => reason
            )
            .increment(1);
        }
        if let Some(entry) = preserved {
            cache.push(entry);
            cached = Some(cache.len() - 1);
        }
    }
    // A question pass allocates substantially more scratch than a cache hit
    // suggests: attention scores/masks, GDN recurrence and INT8 accumulators.
    // Reclaim pooled scratch and, if necessary, move other states to the host
    // L2 before either restoring/capturing a state or launching questions.
    let pass_need = pass_reserve(state.len());
    let state_need = entry_bytes.saturating_add(pass_need);
    if cached.is_none() {
        if let Err(message) = ensure_headroom(
            engine,
            cache,
            host_cache,
            state_need,
            limits.host_bytes,
            "state capture/restore",
        ) {
            metrics::counter!("decidra_engine_failures_total", "stage" => "headroom").increment(1);
            refresh_cache_gauges(cache, host_cache, limits);
            fail_questions(&jobs);
            for job in jobs {
                let _ = job.reply.send(Err(message.clone()));
            }
            return (None, GroupOutcome::Failed);
        }
        // Offloading other states may have evicted this one from the host L2.
        if host_cached.is_some() {
            host_cached = host_cache
                .iter()
                .position(|entry| same_state_identity(state, &digest, &entry.state, &entry.digest));
        }
        refresh_cache_gauges(cache, host_cache, limits);
    }
    let cache_result = if cached.is_some() {
        CacheResult::GpuHit
    } else if host_cached.is_some() {
        CacheResult::HostHit
    } else {
        CacheResult::Miss
    };
    for _ in &jobs {
        metrics::counter!(
            "decidra_state_cache_requests_total",
            "result" => cache_result.label()
        )
        .increment(1);
        if cache_result.hit() {
            metrics::counter!("decidra_state_cache_reused_tokens_total")
                .increment(state.len() as u64);
        }
    }
    let mut vision_timer = None;
    let mut capture_timer = None;
    let mut restore_timer = None;
    let from_host = host_cached.is_some();

    let entry = if let Some(index) = cached {
        cache.remove(index)
    } else if let Some(index) = host_cached {
        restore_timer = engine.gpu.phase_timer().ok();
        let host_entry = &host_cache[index];
        let restored = StateCache::restore(&host_entry.cache, &engine.gpu);
        stop_timer(&mut restore_timer, &engine.gpu);
        match restored {
            Ok(restored) => {
                metrics::counter!("decidra_state_cache_restores_total").increment(1);
                metrics::counter!("decidra_state_cache_restore_bytes_total")
                    .increment(entry_bytes as u64);
                CachedState {
                    state: state.to_vec(),
                    digest,
                    cache: restored,
                    bytes: entry_bytes,
                }
            }
            Err(error) => {
                metrics::counter!(
                    "decidra_engine_failures_total",
                    "stage" => "state_restore"
                )
                .increment(1);
                let message = format!("state restore failed: {error}");
                fail_questions(&jobs);
                for job in jobs {
                    let _ = job.reply.send(Err(message.clone()));
                }
                return (Some(CacheResult::HostHit), GroupOutcome::Failed);
            }
        }
    } else {
        let mut vision_features = None;
        if !jobs[0].images.is_empty() {
            vision_timer = engine.gpu.phase_timer().ok();
            let inputs: Vec<VisionInput<'_>> = jobs[0]
                .images
                .iter()
                .map(|image| VisionInput {
                    patches: &image.patches,
                    grid: VisionGrid {
                        grid_t: image.grid_t,
                        grid_h: image.grid_h,
                        grid_w: image.grid_w,
                    },
                })
                .collect();
            let encoded = engine.vision_encode(&inputs);
            stop_timer(&mut vision_timer, &engine.gpu);
            vision_features = match encoded {
                Ok(value) => Some(value),
                Err(error) => {
                    metrics::counter!(
                        "decidra_engine_failures_total",
                        "stage" => "vision_encode"
                    )
                    .increment(1);
                    let message = format!("image encoding failed: {error}");
                    fail_questions(&jobs);
                    for job in jobs {
                        let _ = job.reply.send(Err(message.clone()));
                    }
                    return (Some(CacheResult::Miss), GroupOutcome::Failed);
                }
            };
        }
        if jobs[0].state_positions.len() != state.len() && vision_features.is_some() {
            metrics::counter!("decidra_engine_failures_total", "stage" => "positions").increment(1);
            let message = "state positions do not match the state tokens".to_owned();
            fail_questions(&jobs);
            for job in jobs {
                let _ = job.reply.send(Err(message.clone()));
            }
            return (Some(CacheResult::Miss), GroupOutcome::Failed);
        }
        capture_timer = engine.gpu.phase_timer().ok();
        let mut captured = match vision_features.as_ref() {
            Some(value) => engine.capture_state_mm(
                state,
                &jobs[0].state_positions,
                &jobs[0].image_spans,
                Some(&value.features),
            ),
            None => engine.capture_state(state),
        };
        if captured
            .as_ref()
            .is_err_and(|error| error.to_string().contains("out of memory"))
        {
            // A long state can still exceed the scaled estimate. Drop all
            // reclaimable scratch/host-cached states and retry once.
            metrics::counter!("decidra_engine_oom_retries_total", "stage" => "state_capture")
                .increment(1);
            let _ = engine.trim_pools();
            let _ = ensure_headroom(
                engine,
                cache,
                host_cache,
                state_need,
                limits.host_bytes,
                "state capture retry",
            );
            captured = match vision_features.as_ref() {
                Some(value) => engine.capture_state_mm(
                    state,
                    &jobs[0].state_positions,
                    &jobs[0].image_spans,
                    Some(&value.features),
                ),
                None => engine.capture_state(state),
            };
        }
        stop_timer(&mut capture_timer, &engine.gpu);
        match captured {
            Ok(captured) => {
                metrics::counter!("decidra_engine_state_captures_total").increment(1);
                metrics::counter!("decidra_engine_tokens_total", "phase" => "state")
                    .increment(state.len() as u64);
                CachedState {
                    state: state.to_vec(),
                    digest,
                    cache: captured,
                    bytes: entry_bytes,
                }
            }
            Err(error) => {
                metrics::counter!(
                    "decidra_engine_failures_total",
                    "stage" => "state_capture"
                )
                .increment(1);
                let free_mib = engine
                    .gpu
                    .mem_info()
                    .map(|(free, _)| free / (1024 * 1024))
                    .unwrap_or(0);
                tracing::error!(
                    state_tokens = state.len(),
                    free_mib,
                    cached_states = cache.len(),
                    host_states = host_cache.len(),
                    pass_need_mib = pass_need / (1024 * 1024),
                    %error,
                    "state capture failed"
                );
                let message = format!("state capture failed: {error}");
                fail_questions(&jobs);
                for job in jobs {
                    let _ = job.reply.send(Err(message.clone()));
                }
                return (Some(CacheResult::Miss), GroupOutcome::Failed);
            }
        }
    };

    let mut questions: Vec<&[u32]> = Vec::new();
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(jobs.len());
    for job in &jobs {
        let start = questions.len();
        questions.extend(job.questions.iter().map(|question| question.as_slice()));
        ranges.push((start, questions.len()));
    }
    if let Err(message) = ensure_headroom(
        engine,
        cache,
        host_cache,
        pass_need,
        limits.host_bytes,
        "question pass",
    ) {
        metrics::counter!("decidra_engine_failures_total", "stage" => "headroom").increment(1);
        refresh_cache_gauges(cache, host_cache, limits);
        fail_questions(&jobs);
        for job in jobs {
            let _ = job.reply.send(Err(message.clone()));
        }
        return (Some(cache_result), GroupOutcome::Failed);
    }
    refresh_cache_gauges(cache, host_cache, limits);
    let pass_started = Instant::now();
    let mut rows = engine.readout_state_once_batch(&entry.cache, &questions);
    if rows
        .as_ref()
        .is_err_and(|error| error.to_string().contains("out of memory"))
    {
        // A failed allocation can leave fragmented/pooled blocks behind even
        // when the preflight looked healthy. Reclaim once and retry before
        // failing the request.
        metrics::counter!("decidra_engine_oom_retries_total", "stage" => "question_pass")
            .increment(1);
        let _ = engine.trim_pools();
        let _ = ensure_headroom(
            engine,
            cache,
            host_cache,
            pass_need,
            limits.host_bytes,
            "question pass retry",
        );
        rows = engine.readout_state_once_batch(&entry.cache, &questions);
    }
    let pass_seconds = pass_started.elapsed().as_secs_f64();
    if rows.is_ok() {
        metrics::histogram!(
            "decidra_engine_question_pass_seconds",
            "cache" => cache_result.label()
        )
        .record(pass_seconds);
        if let Some(seconds) = timer_seconds(&capture_timer) {
            metrics::histogram!("decidra_engine_state_capture_seconds").record(seconds);
        }
        if let Some(seconds) = timer_seconds(&vision_timer) {
            metrics::histogram!("decidra_engine_vision_encode_seconds").record(seconds);
        }
        if let Some(seconds) = timer_seconds(&restore_timer) {
            metrics::histogram!("decidra_state_cache_restore_seconds").record(seconds);
        }
    }
    let question_tokens: usize = questions.iter().map(|question| question.len()).sum();
    match rows {
        Ok(rows) if rows.len() == questions.len() => {
            metrics::counter!("decidra_engine_tokens_total", "phase" => "question")
                .increment(question_tokens as u64);
            metrics::counter!("decidra_engine_questions_total", "outcome" => "answered")
                .increment(questions.len() as u64);
            for (job, (start, end)) in jobs.into_iter().zip(ranges) {
                if job.reply.send(Ok(rows[start..end].to_vec())).is_err() {
                    metrics::counter!(
                        "decidra_engine_requests_cancelled_total",
                        "stage" => "reply"
                    )
                    .increment(1);
                }
            }
            if limits.entries > 0 {
                let cached_bytes: usize = cache.iter().map(|entry| entry.bytes).sum();
                let has_room = cache.len() < limits.entries
                    && cached_bytes.saturating_add(entry.bytes) <= limits.bytes;
                if has_room {
                    cache.push(entry);
                    if from_host {
                        host_cache.retain(|cached| {
                            !same_state_identity(state, &digest, &cached.state, &cached.digest)
                        });
                    }
                } else if !from_host && limits.host_bytes > 0 {
                    host_insert(engine, host_cache, entry, entry_bytes, limits.host_bytes);
                }
            }
            (Some(cache_result), GroupOutcome::Success)
        }
        Ok(rows) => {
            metrics::counter!("decidra_engine_failures_total", "stage" => "rows").increment(1);
            let message = format!(
                "state-once returned {} rows for {} questions",
                rows.len(),
                questions.len()
            );
            fail_questions(&jobs);
            for job in jobs {
                let _ = job.reply.send(Err(message.clone()));
            }
            (Some(cache_result), GroupOutcome::Failed)
        }
        Err(error) => {
            metrics::counter!("decidra_engine_failures_total", "stage" => "question_pass")
                .increment(1);
            let message = format!("state-once question pass failed: {error}");
            fail_questions(&jobs);
            for job in jobs {
                let _ = job.reply.send(Err(message.clone()));
            }
            (Some(cache_result), GroupOutcome::Failed)
        }
    }
}

fn ensure_headroom(
    engine: &Engine,
    cache: &mut Vec<CachedState>,
    host_cache: &mut Vec<HostCachedState>,
    need: usize,
    host_state_cache_bytes: usize,
    context: &str,
) -> std::result::Result<(), String> {
    let mut free = engine
        .gpu
        .mem_info()
        .map(|(free, _)| free)
        .map_err(|error| format!("failed to query GPU memory: {error}"))?;
    metrics::gauge!("decidra_gpu_memory_free_bytes", "device" => "0").set(free as f64);
    if free < need {
        // Only pay for a synchronize + CUDA mempool trim when the hot path
        // cannot satisfy the request. Unconditional trimming on every group
        // returns scratch to the driver only to reallocate it immediately.
        if let Err(error) = engine.trim_pools() {
            return Err(format!("failed to trim scratch pools: {error}"));
        }
        free = engine
            .gpu
            .mem_info()
            .map(|(free, _)| free)
            .map_err(|error| format!("failed to query GPU memory: {error}"))?;
        metrics::gauge!("decidra_gpu_memory_free_bytes", "device" => "0").set(free as f64);
    }
    let mut offloaded = 0usize;
    while free < need && !cache.is_empty() {
        let entry = cache.remove(0);
        let bytes = entry.bytes;
        host_insert(engine, host_cache, entry, bytes, host_state_cache_bytes);
        offloaded += 1;
        // The CUDA mempool retains freed blocks; trim before re-querying.
        if let Err(error) = engine.trim_pools() {
            return Err(format!("failed to trim after state offload: {error}"));
        }
        free = engine
            .gpu
            .mem_info()
            .map(|(free, _)| free)
            .map_err(|error| format!("failed to query GPU memory: {error}"))?;
        metrics::gauge!("decidra_gpu_memory_free_bytes", "device" => "0").set(free as f64);
    }
    if free < need {
        return Err(format!(
            "insufficient GPU memory for {context}: {} MiB free, {} MiB required",
            free / (1024 * 1024),
            need / (1024 * 1024)
        ));
    }
    if offloaded > 0 {
        tracing::warn!(
            offloaded,
            free_mib = free / (1024 * 1024),
            context,
            "offloaded cached states to fit GPU scratch"
        );
    }
    tracing::debug!(
        free_mib = free / (1024 * 1024),
        cached_states = cache.len(),
        context,
        "GPU memory preflight"
    );
    Ok(())
}

fn host_insert(
    engine: &Engine,
    host_cache: &mut Vec<HostCachedState>,
    entry: CachedState,
    bytes: usize,
    budget: usize,
) {
    let started = Instant::now();
    let host = entry.cache.offload(&engine.gpu);
    metrics::histogram!("decidra_state_cache_offload_seconds")
        .record(started.elapsed().as_secs_f64());
    let host = match host {
        Ok(host) => host,
        Err(error) => {
            metrics::counter!("decidra_engine_failures_total", "stage" => "state_offload")
                .increment(1);
            metrics::counter!(
                "decidra_state_cache_evictions_total",
                "tier" => "gpu",
                "reason" => "offload_failed"
            )
            .increment(1);
            tracing::warn!(%error, "state host offload failed");
            return;
        }
    };
    metrics::counter!("decidra_state_cache_offloads_total").increment(1);
    metrics::counter!("decidra_state_cache_offload_bytes_total").increment(bytes as u64);
    host_cache.push(HostCachedState {
        state: entry.state,
        digest: entry.digest,
        cache: host,
        bytes,
    });
    let mut total: usize = host_cache.iter().map(|entry| entry.bytes).sum();
    while total > budget && !host_cache.is_empty() {
        let removed = host_cache.remove(0);
        total = total.saturating_sub(removed.bytes);
        metrics::counter!(
            "decidra_state_cache_evictions_total",
            "tier" => "host",
            "reason" => "host_budget"
        )
        .increment(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{QueueCounters, same_state_identity};
    use std::sync::atomic::Ordering;

    #[test]
    fn image_identity_includes_the_digest() {
        let tokens = [1, 2, 3];
        let red = [7u8; 32];
        let blue = [8u8; 32];
        assert!(same_state_identity(&tokens, &red, &tokens, &red));
        assert!(!same_state_identity(&tokens, &red, &tokens, &blue));
        assert!(!same_state_identity(&tokens, &red, &[1, 2, 4], &red));
    }

    #[test]
    fn queue_counters_saturate_at_zero() {
        let counters = QueueCounters::default();
        counters.running_inc();
        counters.running_dec();
        counters.running_dec();
        assert_eq!(counters.running.load(Ordering::SeqCst), 0);
        counters.channel_inc();
        counters.channel_dec();
        assert_eq!(counters.channel.load(Ordering::SeqCst), 0);
        counters.waiting_dec();
        assert_eq!(counters.waiting.load(Ordering::SeqCst), 0);
    }
}
