//! Prefix matching. BlockCascade states are matched whole (bidirectional state
//! rows depend on later state tokens, so sub-prefix reuse is invalid); Causal
//! prefixes may use 64-token block chains. Entries own unique storage regions.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub type Hash32 = [u8; 32];

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixKey {
    pub backbone_hash: Hash32,
    pub tokens: Vec<u32>,
    pub pos_ids: Vec<u32>,
    pub split: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixEntry {
    pub key: PrefixKey,
    /// Unique physical K/V + GDN region. Never shared between entries.
    pub region: u64,
    pub prefix_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixMatch {
    /// Whole-prefix hit with the entry's region and length.
    Whole {
        region: u64,
        prefix_len: usize,
    },
    /// Causal block-chain hit.
    Causal {
        region: u64,
        prefix_len: usize,
    },
    None,
}

#[derive(Debug, Default)]
pub struct PrefixIndex {
    whole: HashMap<PrefixKey, PrefixEntry>,
    causal: HashMap<Hash32, PrefixEntry>,
    regions: HashSet<u64>,
    next_region: u64,
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self {
            next_region: 1,
            ..Default::default()
        }
    }

    fn allocate_region(&mut self) -> u64 {
        let region = self.next_region;
        self.next_region += 1;
        debug_assert!(self.regions.insert(region));
        region
    }

    /// Insert a BlockCascade state: matched only on the entire key.
    pub fn insert_whole(&mut self, key: PrefixKey, prefix_len: usize) -> u64 {
        if let Some(entry) = self.whole.get(&key) {
            return entry.region;
        }
        let region = self.allocate_region();
        self.whole.insert(
            key.clone(),
            PrefixEntry {
                key,
                region,
                prefix_len,
            },
        );
        region
    }

    /// Insert a Causal prefix; this stores one whole-prefix key (the Causal
    /// block-chain matching is P6 and not enabled by default).
    pub fn insert_causal(&mut self, key: PrefixKey, prefix_len: usize) -> u64 {
        let block_key = causal_block_key(
            &key.backbone_hash,
            &key.tokens[..prefix_len.min(key.tokens.len())],
        );
        if let Some(entry) = self.causal.get(&block_key) {
            return entry.region;
        }
        let region = self.allocate_region();
        self.causal.insert(
            block_key,
            PrefixEntry {
                key,
                region,
                prefix_len,
            },
        );
        region
    }

    /// Exact whole-prefix lookup for BlockCascade.
    pub fn lookup_whole(&self, key: &PrefixKey) -> PrefixMatch {
        match self.whole.get(key) {
            Some(entry) => PrefixMatch::Whole {
                region: entry.region,
                prefix_len: entry.prefix_len,
            },
            None => PrefixMatch::None,
        }
    }

    /// Whole-prefix lookup for Causal.
    pub fn lookup_causal(&self, key: &PrefixKey, prefix_len: usize) -> PrefixMatch {
        if prefix_len > key.tokens.len() {
            return PrefixMatch::None;
        }
        let block_key = causal_block_key(&key.backbone_hash, &key.tokens[..prefix_len]);
        match self.causal.get(&block_key) {
            Some(entry) => PrefixMatch::Causal {
                region: entry.region,
                prefix_len: entry.prefix_len,
            },
            None => PrefixMatch::None,
        }
    }

    pub fn remove(&mut self, key: &PrefixKey) -> Option<PrefixEntry> {
        self.whole.remove(key)
    }

    pub fn regions(&self) -> &HashSet<u64> {
        &self.regions
    }

    pub fn len(&self) -> usize {
        self.whole.len() + self.causal.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn causal_block_key(backbone: &Hash32, tokens: &[u32]) -> Hash32 {
    let mut h = Sha256::new();
    h.update(backbone);
    h.update((tokens.len() as u64).to_le_bytes());
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    h.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(prefix: &[u32], split: usize) -> PrefixKey {
        PrefixKey {
            backbone_hash: [1; 32],
            tokens: prefix.to_vec(),
            pos_ids: (0..prefix.len() as u32).collect(),
            split,
        }
    }

    #[test]
    fn t_subprefix_blockcascade_must_miss() {
        let mut index = PrefixIndex::new();
        let full = key(&(0..4000u32).collect::<Vec<_>>(), 4000);
        let region = index.insert_whole(full.clone(), 4000);
        assert_eq!(
            index.lookup_whole(&full),
            PrefixMatch::Whole {
                region,
                prefix_len: 4000
            }
        );
        // A different state sharing the first 3,000 tokens is not a hit.
        let mut other = full.clone();
        other.tokens[3000..].reverse();
        assert_eq!(index.lookup_whole(&other), PrefixMatch::None);
        // Nor is a strict sub-prefix of the same state.
        let sub = key(&full.tokens[..3000], 3000);
        assert_eq!(index.lookup_whole(&sub), PrefixMatch::None);
    }

    #[test]
    fn t_storage_alias() {
        let mut index = PrefixIndex::new();
        let a = key(&(0..100u32).collect::<Vec<_>>(), 100);
        let mut b = a.clone();
        b.tokens[50..].reverse();
        let ra = index.insert_whole(a.clone(), 100);
        let rb = index.insert_whole(b.clone(), 100);
        assert_ne!(ra, rb);
        assert_eq!(index.regions().len(), 2);
        // Re-inserting the same key returns the same region, not a second one.
        let again = index.insert_whole(a, 100);
        assert_eq!(again, ra);
        assert_eq!(index.regions().len(), 2);
    }

    #[test]
    fn t_causal_prefix() {
        let mut index = PrefixIndex::new();
        let k = key(&(0..128u32).collect::<Vec<_>>(), 0);
        let region = index.insert_causal(k.clone(), 128);
        assert_eq!(
            index.lookup_causal(&k, 128),
            PrefixMatch::Causal {
                region,
                prefix_len: 128
            }
        );
        // A different prefix is a miss.
        let mut other = k.clone();
        other.tokens[127] = 999;
        assert_eq!(index.lookup_causal(&other, 128), PrefixMatch::None);
    }
}
