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
use tokio::sync::oneshot;

use crate::error::EngineError;

pub struct BatchHandle {
    tx: SyncSender<Job>,
}

struct Job {
    state: Vec<u32>,
    questions: Vec<Vec<u32>>,
    submitted: Instant,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
}

struct CachedState {
    state: Vec<u32>,
    cache: StateCache,
    bytes: usize,
}

struct HostCachedState {
    state: Vec<u32>,
    cache: HostStateCache,
    bytes: usize,
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
                    let first_rows = first.questions.len().max(1);
                    let max_jobs = (batch_rows / first_rows).max(1).min(batch_requests);
                    let mut jobs = vec![first];
                    let deadline = Instant::now() + Duration::from_micros(batch_window_us);
                    while jobs.len() < max_jobs {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        match rx.recv_timeout(deadline - now) {
                            Ok(job) => jobs.push(job),
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
        Self { tx }
    }

    /// Run one state-once request. The receiver resolves when the worker has
    /// finished the request's question batch.
    pub fn submit_state_once(
        &self,
        state: Vec<u32>,
        questions: Vec<Vec<u32>>,
    ) -> Result<oneshot::Receiver<Result<Vec<Vec<f32>>, String>>, EngineError> {
        let (reply, receiver) = oneshot::channel();
        let job = Job {
            state,
            questions,
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
    // Preserve first-seen state order so co-batched replies stay deterministic.
    let mut groups: Vec<(Vec<u32>, Vec<Job>)> = Vec::new();
    for job in jobs {
        match groups.iter_mut().find(|(state, _)| *state == job.state) {
            Some((_, members)) => members.push(job),
            None => groups.push((job.state.clone(), vec![job])),
        }
    }
    // Cost-aware scheduling, opt-in for A/B: shortest estimated GPU work
    // first. A group older than 500 ms jumps the queue to avoid starvation.
    if std::env::var("DECIDRA_SCHED_PRIORITY")
        .map(|value| value != "0")
        .unwrap_or(false)
    {
        groups.sort_by_key(|(state, jobs)| {
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
            let state_cost = if cache.iter().any(|entry| entry.state == *state) {
                0
            } else {
                state.len()
            };
            state_cost + questions
        });
    }
    for (state, jobs) in groups {
        run_group(
            engine,
            &state,
            jobs,
            cache,
            host_cache,
            state_cache_entries,
            state_cache_bytes,
            host_state_cache_bytes,
        );
    }
}

fn run_group(
    engine: &Engine,
    state: &[u32],
    jobs: Vec<Job>,
    cache: &mut Vec<CachedState>,
    host_cache: &mut Vec<HostCachedState>,
    state_cache_entries: usize,
    state_cache_bytes: usize,
    host_state_cache_bytes: usize,
) {
    let cached = cache.iter().position(|entry| entry.state == state);
    let host_cached = if cached.is_none() {
        host_cache.iter().position(|entry| entry.state == state)
    } else {
        None
    };
    let hit = cached.is_some() || host_cached.is_some();
    let pin_cache = std::env::var("DECIDRA_STATE_CACHE_PIN")
        .map(|v| v != "0")
        .unwrap_or(true);
    let entry_bytes = state_cache_bytes_for(state.len());
    if cached.is_none() && host_cached.is_none() {
        let cached_bytes: usize = cache.iter().map(|entry| entry.bytes).sum();
        if !pin_cache {
            while !cache.is_empty()
                && (cache.len() >= state_cache_entries
                    || cached_bytes.saturating_add(entry_bytes) > state_cache_bytes)
            {
                cache.remove(0);
            }
        }
        if let Ok((free, _)) = engine.gpu.mem_info() {
            let reserve = 2 * 1024 * 1024 * 1024;
            if free < entry_bytes.saturating_add(reserve) {
                cache.clear();
                let _ = engine.trim_pools();
                if let Ok((free, _)) = engine.gpu.mem_info() {
                    if free < entry_bytes.saturating_add(reserve) {
                        let message = format!(
                            "insufficient GPU memory for a {}-token state (need ~{} MiB, {} MiB free)",
                            state.len(),
                            entry_bytes / (1024 * 1024),
                            free / (1024 * 1024)
                        );
                        for job in jobs {
                            let _ = job.reply.send(Err(message.clone()));
                        }
                        return;
                    }
                }
            }
        }
    }
    let capture_started = Instant::now();
    let from_host = host_cached;

    let entry = if let Some(index) = cached {
        cache.remove(index)
    } else if let Some(index) = host_cached {
        let host_entry = &host_cache[index];
        match StateCache::restore(&host_entry.cache, &engine.gpu) {
            Ok(restored) => CachedState {
                state: state.to_vec(),
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
        match engine.capture_state(state) {
            Ok(captured) => CachedState {
                state: state.to_vec(),
                cache: captured,
                bytes: entry_bytes,
            },
            Err(error) => {
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
    let pass_started = Instant::now();
    let rows = engine.readout_state_once_batch(&entry.cache, &questions);
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

    if state_cache_entries > 0 {
        let cached_bytes: usize = cache.iter().map(|entry| entry.bytes).sum();
        let has_room = cache.len() < state_cache_entries
            && cached_bytes.saturating_add(entry.bytes) <= state_cache_bytes;
        if has_room {
            cache.push(entry);
            if let Some(index) = from_host {
                host_cache.remove(index);
            }
        } else if from_host.is_none() && host_state_cache_bytes > 0 {
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
        cache: host,
        bytes,
    });
    let mut total: usize = host_cache.iter().map(|entry| entry.bytes).sum();
    while total > budget && !host_cache.is_empty() {
        let removed = host_cache.remove(0);
        total = total.saturating_sub(removed.bytes);
    }
}
