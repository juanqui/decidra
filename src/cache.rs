//! Exact-decision logits cache.
//!
//! Keyed by the exact `(token ids, head ids)` pair and holding the raw logits
//! returned by the engine. The cache is deliberately exact: it never matches a
//! shared prefix or a re-tokenized prompt. Prefix/KV caching remains future work.
//!
//! Bounded by entry count (LRU). A capacity of zero disables the cache.

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard},
};

/// Cache key: token ids and the requested head ids, in order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    tokens: Vec<u32>,
    ids: Vec<u32>,
}

struct Entry {
    logits: Vec<f32>,
    /// Monotonic access tick used to pick the least-recently-used entry.
    tick: u64,
}

struct Lru {
    capacity: usize,
    clock: u64,
    entries: HashMap<Key, Entry>,
}

impl Lru {
    fn get(&mut self, tokens: &[u32], ids: &[u32]) -> Option<Vec<f32>> {
        let key = Key {
            tokens: tokens.to_vec(),
            ids: ids.to_vec(),
        };
        let clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(&key)?;
        entry.tick = clock;
        self.clock = clock;
        Some(entry.logits.clone())
    }

    fn insert(&mut self, tokens: &[u32], ids: &[u32], logits: Vec<f32>) {
        if self.capacity == 0 {
            return;
        }
        let key = Key {
            tokens: tokens.to_vec(),
            ids: ids.to_vec(),
        };
        let clock = self.clock.wrapping_add(1);
        self.clock = clock;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.logits = logits;
            entry.tick = clock;
            return;
        }
        if self.entries.len() >= self.capacity {
            if let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.tick)
                .map(|(key, _)| key.clone())
            {
                self.entries.remove(&oldest);
            }
        }
        self.entries.insert(
            key,
            Entry {
                logits,
                tick: clock,
            },
        );
    }
}

/// Thread-safe exact-decision cache. All public methods are non-blocking apart
/// from the internal mutex; cloning a hit is the only allocation.
pub struct DecisionCache {
    inner: Mutex<Lru>,
}

impl DecisionCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Lru {
                capacity,
                clock: 0,
                entries: HashMap::new(),
            }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.lock().capacity > 0
    }

    pub fn capacity(&self) -> usize {
        self.lock().capacity
    }

    /// Return a clone of the cached logits when the exact key is present.
    pub fn get(&self, tokens: &[u32], ids: &[u32]) -> Option<Vec<f32>> {
        self.lock().get(tokens, ids)
    }

    /// Insert or refresh an entry. No-op when the cache is disabled.
    pub fn insert(&self, tokens: &[u32], ids: &[u32], logits: Vec<f32>) {
        self.lock().insert(tokens, ids, logits)
    }

    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> MutexGuard<'_, Lru> {
        self.inner.lock().unwrap_or_else(|error| error.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENS: &[u32] = &[1, 2, 3];
    const IDS: &[u32] = &[3721, 1802];

    #[test]
    fn hit_returns_the_exact_logits() {
        let cache = DecisionCache::new(4);
        assert!(cache.get(TOKENS, IDS).is_none());
        cache.insert(TOKENS, IDS, vec![0.25, -1.5]);
        assert_eq!(cache.get(TOKENS, IDS), Some(vec![0.25, -1.5]));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn miss_on_different_tokens_or_ids() {
        let cache = DecisionCache::new(4);
        cache.insert(TOKENS, IDS, vec![1.0]);
        assert!(cache.get(&[1, 2, 4], IDS).is_none());
        assert!(cache.get(TOKENS, &[1802, 3721]).is_none());
        assert!(cache.get(TOKENS, &[3721]).is_none());
        assert_eq!(cache.get(TOKENS, IDS), Some(vec![1.0]));
    }

    #[test]
    fn evicts_least_recently_used_entry() {
        let cache = DecisionCache::new(2);
        cache.insert(&[1], &[10], vec![1.0]);
        cache.insert(&[2], &[10], vec![2.0]);
        // Touch the first entry so the second is now least-recently used.
        assert_eq!(cache.get(&[1], &[10]), Some(vec![1.0]));
        cache.insert(&[3], &[10], vec![3.0]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&[1], &[10]), Some(vec![1.0]));
        assert!(cache.get(&[2], &[10]).is_none());
        assert_eq!(cache.get(&[3], &[10]), Some(vec![3.0]));
    }

    #[test]
    fn zero_capacity_disables_and_never_stores() {
        let cache = DecisionCache::new(0);
        assert!(!cache.enabled());
        assert!(cache.get(TOKENS, IDS).is_none());
        cache.insert(TOKENS, IDS, vec![1.0]);
        assert!(cache.get(TOKENS, IDS).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn duplicate_insert_refreshes_without_growing() {
        let cache = DecisionCache::new(2);
        cache.insert(TOKENS, IDS, vec![1.0]);
        cache.insert(TOKENS, IDS, vec![2.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(TOKENS, IDS), Some(vec![2.0]));
    }
}
