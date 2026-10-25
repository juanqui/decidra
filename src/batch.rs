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

use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use decidra_native::Engine;
use decidra_native::HostStateCache;
use decidra_native::engine::StateCache;
use decidra_native::vision::{VisionGrid, VisionInput};
use tokio::sync::oneshot;

use crate::error::EngineError;
use crate::image_media::ImageMedia;

pub struct BatchHandle {
    tx: SyncSender<Job>,
    queue_slots: std::sync::Arc<tokio::sync::Semaphore>,
}

struct Job {
    state: Vec<u32>,
    questions: Vec<Vec<u32>>,
    images: std::sync::Arc<Vec<ImageMedia>>,
    state_positions: std::sync::Arc<Vec<[u32; 3]>>,
    image_spans: std::sync::Arc<Vec<(usize, usize)>>,
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

impl BatchHandle {
    /// Spawn the single CUDA worker.
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
        let (tx, rx) = mpsc::sync_channel::<Job>(max_queue);
        let queue_slots = std::sync::Arc::new(tokio::sync::Semaphore::new(max_queue));
        thread::Builder::new()
            .name("decidra-cuda".to_owned())
            .spawn(move || {
                if let Err(error) = engine.rebind_blas() {
                    tracing::error!(%error, "failed to rebind cuBLAS on CUDA worker");
                }
                let mut cache: Vec<CachedState> = Vec::with_capacity(state_cache_entries);
                let mut host_cache: Vec<HostCachedState> = Vec::with_capacity(state_cache_entries);
                loop {
                    let first = match rx.recv() {
                        Ok(job) => job,
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
                                    state_cache_entries,
                                    state_cache_bytes,
                                    host_state_cache_bytes,
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
                        state_cache_entries,
                        state_cache_bytes,
                        host_state_cache_bytes,
                    );
                }
            })
            .expect("spawning the decidra CUDA worker");
        Self { tx, queue_slots }
    }

    /// Run one state-once request. The receiver resolves when the worker has
    /// finished the request's question batch.
    pub async fn submit_state_once(
        &self,
        state: Vec<u32>,
        questions: Vec<Vec<u32>>,
        images: Vec<ImageMedia>,
        state_positions: Vec<[u32; 3]>,
        image_spans: Vec<(usize, usize)>,
        digest: [u8; 32],
    ) -> Result<oneshot::Receiver<Result<Vec<Vec<f32>>, String>>, EngineError> {
        let queue_slot = self
            .queue_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| EngineError::Internal("d3-turbo CUDA worker stopped".to_owned()))?;
        let (reply, receiver) = oneshot::channel();
        let job = Job {
            state,
            questions,
            images: std::sync::Arc::new(images),
            state_positions: std::sync::Arc::new(state_positions),
            image_spans: std::sync::Arc::new(image_spans),
            digest,
            _queue_slot: queue_slot,
            submitted: Instant::now(),
            reply,
        };
        match self.tx.try_send(job) {
            Ok(()) => Ok(receiver),
            Err(TrySendError::Full(_)) => Err(EngineError::Overloaded(
                "d3-turbo queue is full; retry shortly".to_owned(),
            )),
            Err(TrySendError::Disconnected(_)) => Err(EngineError::Internal(
                "d3-turbo CUDA worker stopped".to_owned(),
            )),
        }
    }
}

fn run_jobs(
    engine: &Engine,
    jobs: Vec<Job>,
    cache: &mut Vec<CachedState>,
    host_cache: &mut Vec<HostCachedState>,
    state_cache_entries: usize,
    state_cache_bytes: usize,
    host_state_cache_bytes: usize,
) {
    // A client that disconnected before the worker starts must not pay for a
    // cold image encode or state capture.
    let jobs: Vec<Job> = jobs
        .into_iter()
        .filter(|job| !job.reply.is_closed())
        .collect();
    if jobs.is_empty() {
        return;
    }
    // Preserve first-seen state order so co-batched replies stay deterministic.
    let mut groups: Vec<(Vec<u32>, [u8; 32], Vec<Job>)> = Vec::new();
    for job in jobs {
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
            engine,
            &state,
            digest,
            jobs,
            cache,
            host_cache,
            state_cache_entries,
            state_cache_bytes,
            host_state_cache_bytes,
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

fn run_group(
    engine: &Engine,
    state: &[u32],
    digest: [u8; 32],
    jobs: Vec<Job>,
    cache: &mut Vec<CachedState>,
    host_cache: &mut Vec<HostCachedState>,
    state_cache_entries: usize,
    state_cache_bytes: usize,
    host_state_cache_bytes: usize,
) {
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
    let hit = cached.is_some() || host_cached.is_some();
    let pin_cache = std::env::var("DECIDRA_STATE_CACHE_PIN")
        .map(|v| v != "0")
        .unwrap_or(true);
    let entry_bytes = state_cache_bytes_for(state.len());
    if !pin_cache {
        // Preserve the hit itself while evicting older entries. Removing from
        // the front shifts the hit index, and the byte total must be
        // recomputed after each removal.
        let preserved = cached.map(|index| cache.remove(index));
        let max_len = state_cache_entries.saturating_sub(usize::from(preserved.is_some()));
        let byte_budget = state_cache_bytes.saturating_sub(entry_bytes);
        let mut bytes: usize = cache.iter().map(|entry| entry.bytes).sum();
        while !cache.is_empty() && (cache.len() > max_len || bytes > byte_budget) {
            let removed = cache.remove(0);
            bytes = bytes.saturating_sub(removed.bytes);
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
            host_state_cache_bytes,
            "state capture/restore",
        ) {
            for job in jobs {
                let _ = job.reply.send(Err(message.clone()));
            }
            return;
        }
        // Offloading other states may have evicted this one from the host L2.
        if host_cached.is_some() {
            host_cached = host_cache
                .iter()
                .position(|entry| same_state_identity(state, &digest, &entry.state, &entry.digest));
        }
    }
    let capture_started = Instant::now();
    let from_host = host_cached.is_some();

    let entry = if let Some(index) = cached {
        cache.remove(index)
    } else if let Some(index) = host_cached {
        let host_entry = &host_cache[index];
        match StateCache::restore(&host_entry.cache, &engine.gpu) {
            Ok(restored) => CachedState {
                state: state.to_vec(),
                digest,
                cache: restored,
                bytes: entry_bytes,
            },
            Err(error) => {
                let message = format!("state restore failed: {error}");
                for job in jobs {
                    let _ = job.reply.send(Err(message.clone()));
                }
                return;
            }
        }
    } else {
        let mut vision_features = None;
        if !jobs[0].images.is_empty() {
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
            vision_features = match engine.vision_encode(&inputs) {
                Ok(value) => Some(value),
                Err(error) => {
                    let message = format!("image encoding failed: {error}");
                    for job in jobs {
                        let _ = job.reply.send(Err(message.clone()));
                    }
                    return;
                }
            };
        }
        let mut captured = match vision_features.as_ref() {
            Some(v) => {
                if jobs[0].state_positions.len() != state.len() {
                    let message = "state positions do not match the state tokens".to_owned();
                    for job in jobs {
                        let _ = job.reply.send(Err(message.clone()));
                    }
                    return;
                }
                engine.capture_state_mm(
                    state,
                    &jobs[0].state_positions,
                    &jobs[0].image_spans,
                    Some(&v.features),
                )
            }
            None => engine.capture_state(state),
        };
        if captured
            .as_ref()
            .is_err_and(|error| error.to_string().contains("out of memory"))
        {
            // A long state can still exceed the scaled estimate. Drop all
            // reclaimable scratch/host-cached states and retry once.
            metrics::counter!("semel_state_capture_oom_retries_total").increment(1);
            let _ = engine.trim_pools();
            let _ = ensure_headroom(
                engine,
                cache,
                host_cache,
                state_need,
                host_state_cache_bytes,
                "state capture retry",
            );
            captured = match vision_features.as_ref() {
                Some(v) => engine.capture_state_mm(
                    state,
                    &jobs[0].state_positions,
                    &jobs[0].image_spans,
                    Some(&v.features),
                ),
                None => engine.capture_state(state),
            };
        }
        match captured {
            Ok(captured) => CachedState {
                state: state.to_vec(),
                digest,
                cache: captured,
                bytes: entry_bytes,
            },
            Err(error) => {
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
                for job in jobs {
                    let _ = job.reply.send(Err(message.clone()));
                }
                return;
            }
        }
    };
    let capture_ms = capture_started.elapsed().as_secs_f64() * 1e3;
    let wait_max_ms = jobs
        .iter()
        .map(|job| job.submitted.elapsed().as_secs_f64() * 1e3)
        .fold(0.0_f64, f64::max);
    let job_count = jobs.len();

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
        host_state_cache_bytes,
        "question pass",
    ) {
        for job in jobs {
            let _ = job.reply.send(Err(message.clone()));
        }
        return;
    }
    let pass_started = Instant::now();
    let mut rows = engine.readout_state_once_batch(&entry.cache, &questions);
    if rows
        .as_ref()
        .is_err_and(|error| error.to_string().contains("out of memory"))
    {
        // A failed allocation can leave fragmented/pooled blocks behind even
        // when the preflight looked healthy. Reclaim once and retry before
        // failing the request.
        metrics::counter!("semel_question_pass_oom_retries_total").increment(1);
        let _ = engine.trim_pools();
        let _ = ensure_headroom(
            engine,
            cache,
            host_cache,
            pass_need,
            host_state_cache_bytes,
            "question pass retry",
        );
        rows = engine.readout_state_once_batch(&entry.cache, &questions);
    }
    let pass_ms = pass_started.elapsed().as_secs_f64() * 1e3;
    metrics::counter!(if hit {
        "semel_state_cache_hits_total"
    } else {
        "semel_state_cache_misses_total"
    })
    .increment(1);
    metrics::histogram!("semel_group_jobs").record(job_count as f64);
    metrics::histogram!("semel_group_rows").record(questions.len() as f64);
    metrics::histogram!("semel_capture_seconds").record(capture_ms / 1e3);
    metrics::histogram!("semel_queue_wait_seconds").record(wait_max_ms / 1e3);
    metrics::histogram!("semel_pass_seconds").record(pass_ms / 1e3);
    tracing::debug!(
        jobs = job_count,
        rows = questions.len(),
        state = state.len(),
        cache_hit = hit,
        wait_max_ms,
        capture_ms,
        pass_ms,
        "state-once group"
    );
    match rows {
        Ok(rows) if rows.len() == questions.len() => {
            for (job, (start, end)) in jobs.into_iter().zip(ranges) {
                let _ = job.reply.send(Ok(rows[start..end].to_vec()));
            }
        }
        Ok(rows) => {
            let message = format!(
                "state-once returned {} rows for {} questions",
                rows.len(),
                questions.len()
            );
            for job in jobs {
                let _ = job.reply.send(Err(message.clone()));
            }
        }
        Err(error) => {
            let message = format!("state-once question pass failed: {error}");
            for job in jobs {
                let _ = job.reply.send(Err(message.clone()));
            }
        }
    }

    metrics::gauge!("semel_state_cache_states").set(cache.len() as f64);
    metrics::gauge!("semel_state_cache_bytes")
        .set(cache.iter().map(|entry| entry.bytes).sum::<usize>() as f64);
    metrics::gauge!("semel_host_cache_states").set(host_cache.len() as f64);
    if state_cache_entries > 0 {
        let cached_bytes: usize = cache.iter().map(|entry| entry.bytes).sum();
        let has_room = cache.len() < state_cache_entries
            && cached_bytes.saturating_add(entry.bytes) <= state_cache_bytes;
        if has_room {
            cache.push(entry);
            if from_host {
                host_cache.retain(|cached| {
                    !same_state_identity(state, &digest, &cached.state, &cached.digest)
                });
            }
        } else if !from_host && host_state_cache_bytes > 0 {
            host_insert(
                engine,
                host_cache,
                entry,
                entry_bytes,
                host_state_cache_bytes,
            );
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
    metrics::gauge!("semel_gpu_free_bytes").set(free as f64);
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
        metrics::gauge!("semel_gpu_free_bytes").set(free as f64);
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
        metrics::gauge!("semel_gpu_free_bytes").set(free as f64);
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
    let host = match entry.cache.offload(&engine.gpu) {
        Ok(host) => host,
        Err(error) => {
            tracing::warn!(%error, "state host offload failed");
            return;
        }
    };
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
    }
}

#[cfg(test)]
mod tests {
    use super::same_state_identity;

    #[test]
    fn image_identity_includes_the_digest() {
        let tokens = [1, 2, 3];
        let red = [7u8; 32];
        let blue = [8u8; 32];
        assert!(same_state_identity(&tokens, &red, &tokens, &red));
        assert!(!same_state_identity(&tokens, &red, &tokens, &blue));
        assert!(!same_state_identity(&tokens, &red, &[1, 2, 4], &red));
    }
}
