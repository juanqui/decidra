//! FIFO batching queue for the native backend.
//!
//! One dedicated thread owns `decidra_native::Engine` and executes requests in
//! batches. The worker blocks for the first request, then drains additional
//! requests until the batch window expires, `MAX_BATCH` is reached, or the next
//! request would exceed `MAX_BATCH_TOKENS`. A request larger than the token cap
//! runs alone. Order is strict FIFO; a batch is never split.
//!
//! `BatchExecutor` is abstract so the queue can be unit-tested without a GPU.

use std::{
    collections::VecDeque,
    panic::AssertUnwindSafe,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        mpsc::{self, Receiver, Sender},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use tokio::sync::oneshot;

use crate::error::EngineError;

/// Result type returned by an executor for one request.
pub type ExecResult = Result<Vec<f32>, String>;

/// A request waiting for a forward.
pub struct PendingRequest {
    pub tokens: Vec<u32>,
    pub ids: Vec<u32>,
    pub reply: oneshot::Sender<ExecResult>,
    enqueued_at: Instant,
    /// Pushed back from a sealed batch (`Full`/`NoFit`). The next batch starts
    /// immediately instead of waiting for the window again: the request has
    /// already paid its batching latency.
    deferred: bool,
}

/// Batching policy. `window` is the maximum time the worker waits after the
/// first request of a batch for more work.
#[derive(Debug, Clone, Copy)]
pub struct BatchOptions {
    pub max_batch: usize,
    pub max_batch_tokens: usize,
    pub window: Duration,
}

impl BatchOptions {
    pub fn new(max_batch: usize, max_batch_tokens: usize, window: Duration) -> Self {
        Self {
            max_batch: max_batch.max(1),
            max_batch_tokens: max_batch_tokens.max(1),
            window,
        }
    }

    /// Batching disabled: one request per forward and no window wait.
    pub fn single(max_batch_tokens: usize) -> Self {
        Self::new(1, max_batch_tokens, Duration::ZERO)
    }
}

/// Executes one batch. Implementations must return exactly one result per
/// request, in input order.
pub trait BatchExecutor: Send + 'static {
    fn execute(&mut self, tokens: &[&[u32]], ids: &[&[u32]]) -> Vec<ExecResult>;
}

struct QueueState {
    pending: VecDeque<PendingRequest>,
    stopping: bool,
}

struct Shared {
    state: Mutex<QueueState>,
    ready: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }
}

/// Handle to the engine thread. `submit` is non-blocking; the caller awaits
/// the returned oneshot receiver.
pub struct BatchHandle {
    shared: Arc<Shared>,
    done: Mutex<Option<Receiver<()>>>,
    join: Option<JoinHandle<()>>,
}

impl BatchHandle {
    /// Start the engine thread. `executor` is moved in and owns the engine.
    pub fn spawn<E: BatchExecutor>(executor: E, options: BatchOptions) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(QueueState {
                pending: VecDeque::new(),
                stopping: false,
            }),
            ready: Condvar::new(),
        });
        let (done_tx, done_rx) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let join = std::thread::Builder::new()
            .name("decidra-native-batch".to_owned())
            .spawn(move || run_worker(executor, options, worker_shared, done_tx))
            .expect("spawning the native batch worker");
        Self {
            shared,
            done: Mutex::new(Some(done_rx)),
            join: Some(join),
        }
    }

    /// Enqueue a request. Fails after shutdown has been requested.
    pub fn submit(
        &self,
        tokens: Vec<u32>,
        ids: Vec<u32>,
    ) -> Result<oneshot::Receiver<ExecResult>, EngineError> {
        let (reply, receiver) = oneshot::channel();
        let request = PendingRequest {
            tokens,
            ids,
            reply,
            enqueued_at: Instant::now(),
            deferred: false,
        };
        {
            let mut state = self.shared.lock();
            if state.stopping {
                return Err(EngineError::Internal(
                    "native batch worker is shutting down".to_owned(),
                ));
            }
            state.pending.push_back(request);
            metrics::gauge!("decidra_native_batch_queue_depth").set(state.pending.len() as f64);
        }
        self.shared.ready.notify_one();
        Ok(receiver)
    }

    /// Current number of queued (not yet executing) requests.
    pub fn depth(&self) -> usize {
        self.shared.lock().pending.len()
    }
}

impl Drop for BatchHandle {
    fn drop(&mut self) {
        {
            let mut state = self.shared.lock();
            state.stopping = true;
            // Close queued replies so callers cannot wait forever.
            state.pending.clear();
        }
        self.shared.ready.notify_all();

        let stopped = match self.done.lock() {
            Ok(mut guard) => guard
                .take()
                .is_some_and(|done| done.recv_timeout(Duration::from_secs(30)).is_ok()),
            Err(_) => false,
        };
        match self.join.take() {
            Some(join) if stopped => {
                let _ = join.join();
            }
            Some(_) => {
                tracing::warn!(
                    "native batch worker did not stop within 30s; detaching the engine thread"
                );
            }
            None => {}
        }
    }
}

/// Outcome of trying to add one request to the current batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddOutcome {
    /// The request joined the batch.
    Added,
    /// A hard cap was reached; seal the batch.
    Full,
    /// The request would exceed the token cap; it was pushed back and the
    /// batch must be sealed (waiting cannot help: the head can never fit).
    NoFit,
}

/// Add one request if it fits. `Full` and `NoFit` both mean the batch is
/// sealed; `NoFit` guarantees the batch is non-empty (a lone oversize request
/// is accepted).
fn add_to_batch(
    state: &mut QueueState,
    batch: &mut Vec<PendingRequest>,
    batch_tokens: &mut usize,
    request: PendingRequest,
    options: &BatchOptions,
) -> AddOutcome {
    if request.reply.is_closed() {
        // Dropped client: discard it and keep the batch open.
        return AddOutcome::Added;
    }
    if batch.len() >= options.max_batch {
        let mut request = request;
        request.deferred = true;
        state.pending.push_front(request);
        return AddOutcome::Full;
    }
    let tokens = request.tokens.len();
    if !batch.is_empty() && *batch_tokens + tokens > options.max_batch_tokens {
        // FIFO: keep it at the head of the queue for the next batch.
        let mut request = request;
        request.deferred = true;
        state.pending.push_front(request);
        return AddOutcome::NoFit;
    }
    *batch_tokens += tokens;
    batch.push(request);
    if batch.len() >= options.max_batch || *batch_tokens >= options.max_batch_tokens {
        AddOutcome::Full
    } else {
        AddOutcome::Added
    }
}

/// Take everything that fits from `state` starting with `first`. A request that
/// does not fit is pushed back at the head, preserving FIFO. Closed receivers
/// are skipped. The worker uses [`add_to_batch`] directly with its window; this
/// helper exists for deterministic batch-policy tests.
#[cfg(test)]
fn take_batch(
    state: &mut QueueState,
    first: PendingRequest,
    options: &BatchOptions,
) -> Vec<PendingRequest> {
    let mut batch = Vec::with_capacity(options.max_batch);
    let mut batch_tokens = 0usize;
    let mut candidate = Some(first);
    while let Some(request) = candidate.take() {
        if add_to_batch(state, &mut batch, &mut batch_tokens, request, options) != AddOutcome::Added
        {
            break;
        }
        candidate = state.pending.pop_front();
    }
    batch
}

fn run_worker<E: BatchExecutor>(
    mut executor: E,
    options: BatchOptions,
    shared: Arc<Shared>,
    done: Sender<()>,
) {
    loop {
        // Block until there is work or shutdown is requested.
        let mut state = shared.lock();
        while state.pending.is_empty() && !state.stopping {
            state = shared
                .ready
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        let Some(first) = state.pending.pop_front() else {
            drop(state);
            break;
        };
        metrics::gauge!("decidra_native_batch_queue_depth").set(state.pending.len() as f64);

        // A request that arrives alone only needs a short grace period to pick
        // up a burst; a full window is for a queue that already has waiters.
        // This keeps a lone request's latency low without losing coalescing.
        const LONE_WINDOW: Duration = Duration::from_micros(500);
        let window = if state.pending.is_empty() {
            options.window.min(LONE_WINDOW)
        } else {
            options.window
        };
        let deadline = Instant::now()
            + if first.deferred {
                Duration::ZERO
            } else {
                window
            };
        let mut batch = Vec::with_capacity(options.max_batch);
        let mut batch_tokens = 0usize;
        let mut candidate = Some(first);
        let mut open = true;
        while open {
            // Drain whatever is already queued. A hard cap, a no-fit head, or
            // stopping all seal the batch immediately: waiting cannot make the
            // head fit, it only adds latency.
            while let Some(request) = candidate.take().or_else(|| state.pending.pop_front()) {
                match add_to_batch(&mut state, &mut batch, &mut batch_tokens, request, &options) {
                    AddOutcome::Added => {}
                    AddOutcome::Full | AddOutcome::NoFit => {
                        open = false;
                        break;
                    }
                }
            }
            if !open || state.stopping {
                break;
            }
            // Wait for arrivals until the window expires.
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let (guard, timed_out) = shared
                .ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(|error| error.into_inner());
            state = guard;
            if state.pending.is_empty() {
                if timed_out.timed_out() || state.stopping {
                    break;
                }
                continue;
            }
            candidate = state.pending.pop_front();
        }

        metrics::gauge!("decidra_native_batch_queue_depth").set(state.pending.len() as f64);
        drop(state);

        if batch.is_empty() {
            continue;
        }

        let waits: Vec<f64> = batch
            .iter()
            .map(|request| request.enqueued_at.elapsed().as_secs_f64())
            .collect();
        metrics::histogram!("decidra_native_batch_size").record(batch.len() as f64);
        for wait in &waits {
            metrics::histogram!("decidra_native_batch_wait_seconds").record(*wait);
        }

        let tokens: Vec<&[u32]> = batch
            .iter()
            .map(|request| request.tokens.as_slice())
            .collect();
        let ids: Vec<&[u32]> = batch.iter().map(|request| request.ids.as_slice()).collect();
        let mut results =
            match std::panic::catch_unwind(AssertUnwindSafe(|| executor.execute(&tokens, &ids))) {
                Ok(results) => results,
                Err(_) => {
                    tracing::error!(batch = batch.len(), "native batch executor panicked");
                    vec![Err("native batch executor panicked".to_owned()); batch.len()]
                }
            };
        results.resize(batch.len(), Err("missing batch result".to_owned()));
        results.truncate(batch.len());
        for (request, result) in batch.into_iter().zip(results) {
            let _ = request.reply.send(result);
        }
    }
    let _ = done.send(());
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    use super::{
        AddOutcome, BatchExecutor, BatchHandle, BatchOptions, ExecResult, PendingRequest,
        QueueState,
    };
    use tokio::sync::oneshot;

    #[derive(Clone, Default)]
    struct Recorder {
        batches: Arc<Mutex<Vec<Vec<Vec<u32>>>>>,
    }

    impl Recorder {
        fn batches(&self) -> Vec<Vec<Vec<u32>>> {
            self.batches.lock().unwrap().clone()
        }
    }

    struct Fake {
        recorder: Recorder,
        fail_on: Option<u32>,
        delay: Duration,
        executed: Arc<AtomicUsize>,
    }

    impl BatchExecutor for Fake {
        fn execute(&mut self, tokens: &[&[u32]], _ids: &[&[u32]]) -> Vec<ExecResult> {
            self.executed.fetch_add(tokens.len(), Ordering::SeqCst);
            self.recorder
                .batches
                .lock()
                .unwrap()
                .push(tokens.iter().map(|tokens| tokens.to_vec()).collect());
            if !self.delay.is_zero() {
                thread::sleep(self.delay);
            }
            tokens
                .iter()
                .map(|tokens| {
                    if self.fail_on == tokens.first().copied() {
                        Err("fake failure".to_owned())
                    } else {
                        Ok(tokens.iter().map(|value| *value as f32).collect())
                    }
                })
                .collect()
        }
    }

    fn fake(recorder: &Recorder, fail_on: Option<u32>) -> Fake {
        Fake {
            recorder: recorder.clone(),
            fail_on,
            delay: Duration::ZERO,
            executed: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn request(tokens: Vec<u32>) -> (PendingRequest, oneshot::Receiver<ExecResult>) {
        let (reply, receiver) = oneshot::channel();
        (
            PendingRequest {
                tokens,
                ids: vec![1],
                reply,
                enqueued_at: Instant::now(),
                deferred: false,
            },
            receiver,
        )
    }

    fn state_with(requests: Vec<PendingRequest>) -> QueueState {
        QueueState {
            pending: requests.into(),
            stopping: false,
        }
    }

    #[tokio::test]
    async fn window_fills_a_batch_from_queued_requests() {
        let recorder = Recorder::default();
        let handle = BatchHandle::spawn(
            fake(&recorder, None),
            BatchOptions::new(8, 16384, Duration::from_millis(20)),
        );
        let mut receivers = Vec::new();
        for value in 0..4u32 {
            receivers.push(handle.submit(vec![value], vec![1]).unwrap());
        }
        for (value, receiver) in (0..4u32).zip(receivers) {
            let logits = receiver.await.unwrap().unwrap();
            assert_eq!(logits, vec![value as f32]);
        }
        let batches = recorder.batches();
        assert_eq!(batches.iter().map(Vec::len).sum::<usize>(), 4);
        assert!(batches.iter().all(|batch| batch.len() <= 8));
        assert_eq!(batches[0].len(), 4, "batches: {batches:?}");
    }

    #[tokio::test]
    async fn max_batch_splits_into_fifo_batches() {
        let recorder = Recorder::default();
        let handle = BatchHandle::spawn(
            fake(&recorder, None),
            BatchOptions::new(2, 16384, Duration::from_millis(20)),
        );
        let mut receivers = Vec::new();
        for value in 0..5u32 {
            receivers.push(handle.submit(vec![value], vec![1]).unwrap());
        }
        for (value, receiver) in (0..5u32).zip(receivers) {
            assert_eq!(receiver.await.unwrap().unwrap(), vec![value as f32]);
        }
        let batches = recorder.batches();
        assert!(batches.iter().all(|batch| batch.len() <= 2));
        assert_eq!(batches.iter().map(Vec::len).sum::<usize>(), 5);
        let order: Vec<u32> = batches
            .iter()
            .flat_map(|batch| batch.iter().map(|tokens| tokens[0]))
            .collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4], "FIFO order");
    }

    #[test]
    fn max_batch_tokens_keeps_fifo_and_never_splits() {
        let (a, _ra) = request(vec![1; 400]);
        let (b, _rb) = request(vec![2; 400]);
        let (c, _rc) = request(vec![3; 100]);
        let mut state = state_with(vec![b, c]);
        let options = BatchOptions::new(8, 512, Duration::from_millis(1));
        let batch = super::take_batch(&mut state, a, &options);
        assert_eq!(batch.len(), 1, "400 + 400 exceeds 512");
        assert_eq!(state.pending.len(), 2, "the next request is kept");
        assert_eq!(batch[0].tokens.len(), 400);

        let first = state.pending.pop_front().unwrap();
        let batch = super::take_batch(&mut state, first, &options);
        assert_eq!(batch.len(), 2, "400 + 100 fits");
        assert_eq!(batch[0].tokens[0], 2);
        assert_eq!(batch[1].tokens[0], 3);
        assert!(state.pending.is_empty());
    }

    #[test]
    fn single_request_larger_than_token_cap_runs_alone() {
        let (big, _r) = request(vec![9; 2000]);
        let (next, _r2) = request(vec![1; 10]);
        let mut state = state_with(vec![next]);
        let options = BatchOptions::new(8, 512, Duration::from_millis(1));
        let batch = super::take_batch(&mut state, big, &options);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].tokens.len(), 2000);
        assert_eq!(state.pending.len(), 1, "next request stays queued");
    }

    #[test]
    fn dropped_receivers_are_skipped() {
        let (dropped, receiver) = request(vec![1; 10]);
        drop(receiver);
        let (kept, _kept_rx) = request(vec![2; 10]);
        let mut state = state_with(vec![dropped, kept]);
        let (first, _first_rx) = request(vec![3; 10]);
        let options = BatchOptions::new(8, 16384, Duration::from_millis(1));
        let batch = super::take_batch(&mut state, first, &options);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].tokens[0], 3);
        assert_eq!(batch[1].tokens[0], 2);
        assert!(state.pending.is_empty());
    }

    #[tokio::test]
    async fn error_propagation_is_per_request() {
        let recorder = Recorder::default();
        let handle = BatchHandle::spawn(
            fake(&recorder, Some(7)),
            BatchOptions::new(8, 16384, Duration::from_millis(20)),
        );
        let good = handle.submit(vec![1], vec![1]).unwrap();
        let bad = handle.submit(vec![7], vec![1]).unwrap();
        assert_eq!(good.await.unwrap().unwrap(), vec![1.0]);
        assert_eq!(bad.await.unwrap().unwrap_err(), "fake failure");
    }

    #[tokio::test]
    async fn dropped_receiver_does_not_execute() {
        let recorder = Recorder::default();
        let executed = Arc::new(AtomicUsize::new(0));
        let handle = BatchHandle::spawn(
            Fake {
                recorder: recorder.clone(),
                fail_on: None,
                delay: Duration::ZERO,
                executed: executed.clone(),
            },
            BatchOptions::new(8, 16384, Duration::from_millis(5)),
        );
        let dropped = handle.submit(vec![4], vec![1]).unwrap();
        drop(dropped);
        let live = handle.submit(vec![5], vec![1]).unwrap();
        assert_eq!(live.await.unwrap().unwrap(), vec![5.0]);
        assert_eq!(executed.load(Ordering::SeqCst), 1);
        let batches = recorder.batches();
        assert!(batches.iter().all(|batch| batch[0][0] != 4), "{batches:?}");
    }

    #[tokio::test]
    async fn window_expires_without_waiting_forever() {
        let recorder = Recorder::default();
        let handle = BatchHandle::spawn(
            fake(&recorder, None),
            BatchOptions::new(8, 16384, Duration::from_millis(5)),
        );
        let first = handle.submit(vec![1], vec![1]).unwrap();
        assert_eq!(first.await.unwrap().unwrap(), vec![1.0]);
        let second = handle.submit(vec![2], vec![1]).unwrap();
        assert_eq!(second.await.unwrap().unwrap(), vec![2.0]);
        let batches = recorder.batches();
        assert_eq!(batches.len(), 2, "separate arrivals form separate batches");
    }

    #[test]
    fn no_fit_seals_the_batch_immediately() {
        let (first, _r1) = request(vec![1; 600]);
        let (next, _r2) = request(vec![2; 600]);
        let mut state = state_with(vec![next]);
        let options = BatchOptions::new(8, 1000, Duration::from_millis(1));
        let mut batch = Vec::new();
        let mut batch_tokens = 0usize;
        assert_eq!(
            super::add_to_batch(&mut state, &mut batch, &mut batch_tokens, first, &options),
            AddOutcome::Added
        );
        let head = state.pending.pop_front().unwrap();
        assert_eq!(
            super::add_to_batch(&mut state, &mut batch, &mut batch_tokens, head, &options),
            AddOutcome::NoFit
        );
        assert_eq!(batch.len(), 1);
        assert_eq!(
            state.pending.len(),
            1,
            "second request stays for the next batch"
        );
    }

    #[tokio::test]
    async fn no_fit_pair_does_not_wait_for_the_window() {
        let recorder = Recorder::default();
        // A long window would add 500 ms per batch if the no-fit head did not
        // seal it immediately.
        let handle = BatchHandle::spawn(
            fake(&recorder, None),
            BatchOptions::new(8, 1000, Duration::from_millis(500)),
        );
        let started = Instant::now();
        let first = handle.submit(vec![1; 600], vec![1]).unwrap();
        let second = handle.submit(vec![2; 600], vec![1]).unwrap();
        assert_eq!(first.await.unwrap().unwrap().len(), 600);
        assert_eq!(second.await.unwrap().unwrap().len(), 600);
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(250),
            "no-fit pair waited for the window: {elapsed:?}"
        );
        let batches = recorder.batches();
        assert_eq!(
            batches.len(),
            2,
            "600 + 600 cannot share a 1000-token batch"
        );
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn shutdown_does_not_hang_with_no_work() {
        let recorder = Recorder::default();
        let handle = BatchHandle::spawn(
            fake(&recorder, None),
            BatchOptions::new(2, 16384, Duration::from_millis(1)),
        );
        let started = Instant::now();
        drop(handle);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(recorder.batches().is_empty());
    }
}
