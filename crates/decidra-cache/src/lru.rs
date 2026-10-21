//! Entry-level LRU with byte or count budgets and pins.

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Slot<K> {
    key: K,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug)]
pub enum LruError {
    NoRoom,
}

#[derive(Debug)]
pub struct Lru<K: Eq + Hash + Clone> {
    slots: Vec<Slot<K>>,
    pins: HashMap<K, usize>,
    bytes: usize,
    budget_bytes: usize,
    budget_entries: usize,
    clock: u64,
}

impl<K: Eq + Hash + Clone> Lru<K> {
    pub fn new_bytes(budget: usize) -> Self {
        Self {
            slots: Vec::new(),
            pins: HashMap::new(),
            bytes: 0,
            budget_bytes: budget,
            budget_entries: usize::MAX,
            clock: 0,
        }
    }
    pub fn new_entries(budget: usize) -> Self {
        Self {
            slots: Vec::new(),
            pins: HashMap::new(),
            bytes: 0,
            budget_bytes: usize::MAX,
            budget_entries: budget,
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn over(&self) -> bool {
        self.bytes > self.budget_bytes || self.slots.len() > self.budget_entries
    }

    fn evict_one(&mut self) -> bool {
        let victim = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| self.pins.get(&s.key).copied().unwrap_or(0) == 0)
            .min_by_key(|(_, s)| s.last_used)
            .map(|(i, _)| i);
        match victim {
            Some(i) => {
                let slot = self.slots.remove(i);
                self.bytes -= slot.bytes;
                true
            }
            None => false,
        }
    }

    /// Insert or replace. Evicts unpinned LRU entries until the budget holds.
    pub fn insert(&mut self, key: K, bytes: usize) -> Result<(), LruError> {
        if let Some(i) = self.slots.iter().position(|s| s.key == key) {
            let old = self.slots.remove(i);
            self.bytes -= old.bytes;
        }
        while self.over()
            || (self.bytes + bytes > self.budget_bytes && self.budget_bytes != usize::MAX)
            || (self.slots.len() + 1 > self.budget_entries && self.budget_entries != usize::MAX)
        {
            if !self.evict_one() {
                return Err(LruError::NoRoom);
            }
        }
        self.clock += 1;
        self.slots.push(Slot {
            key,
            bytes,
            last_used: self.clock,
        });
        self.bytes += bytes;
        Ok(())
    }

    pub fn get(&mut self, key: &K) -> Option<usize> {
        let clock = self.tick();
        let slot = self.slots.iter_mut().find(|s| &s.key == key)?;
        slot.last_used = clock;
        Some(slot.bytes)
    }

    pub fn pin(&mut self, key: &K) -> bool {
        if self.slots.iter().any(|s| &s.key == key) {
            *self.pins.entry(key.clone()).or_insert(0) += 1;
            true
        } else {
            false
        }
    }

    pub fn unpin(&mut self, key: &K) {
        if let Some(count) = self.pins.get_mut(key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.pins.remove(key);
            }
        }
    }

    pub fn remove(&mut self, key: &K) -> bool {
        if let Some(i) = self.slots.iter().position(|s| &s.key == key) {
            let slot = self.slots.remove(i);
            self.bytes -= slot.bytes;
            true
        } else {
            false
        }
    }

    pub fn contains(&self, key: &K) -> bool {
        self.slots.iter().any(|s| &s.key == key)
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_lru_count_eviction() {
        let mut lru = Lru::new_entries(2);
        lru.insert(1u32, 10).unwrap();
        lru.insert(2, 10).unwrap();
        lru.get(&1);
        lru.insert(3, 10).unwrap();
        assert!(!lru.contains(&2), "LRU victim should be 2");
        assert!(lru.contains(&1) && lru.contains(&3));
    }

    #[test]
    fn t_lru_pin_blocks_eviction() {
        let mut lru = Lru::new_entries(2);
        lru.insert(1u32, 10).unwrap();
        lru.insert(2, 10).unwrap();
        assert!(lru.pin(&1));
        lru.get(&2); // 1 is still pinned, 2 is the only victim
        lru.insert(3, 10).unwrap();
        assert!(lru.contains(&1), "pinned entry must survive");
        assert!(!lru.contains(&2));
        lru.unpin(&1);
        lru.get(&1);
        lru.insert(4, 10).unwrap();
        assert!(!lru.contains(&3));
    }

    #[test]
    fn t_lru_bytes() {
        let mut lru = Lru::new_bytes(100);
        lru.insert(1u32, 60).unwrap();
        lru.insert(2, 60).unwrap();
        assert_eq!(lru.len(), 1);
        assert!(lru.contains(&2) && !lru.contains(&1));
        assert_eq!(lru.bytes(), 60);
        // Pin the only entry; a huge insert cannot evict it.
        assert!(lru.pin(&2));
        assert!(lru.insert(3, 200).is_err());
        assert!(lru.contains(&2));
    }

    #[test]
    fn t_lru_replace() {
        let mut lru = Lru::new_entries(2);
        lru.insert(1u32, 10).unwrap();
        lru.insert(1, 20).unwrap();
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.bytes(), 20);
    }
}
