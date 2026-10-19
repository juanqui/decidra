//! CPython-compatible MT19937 and `random.sample` permutations.
//!
//! The reference shim seeds `random.Random(zlib.crc32(question))` and calls
//! `rng.sample(range(n), n)` three times for the `permute` strategy. To keep the exact
//! same four option orders (identity plus those three permutations), this module
//! reproduces CPython's integer seeding (via `init_by_array`), `getrandbits`,
//! `_randbelow`, and the selection-sampling branch of `random.sample`. Matching only
//! "some deterministic shuffle" would produce different probabilities.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct PyRng {
    state: [u32; N],
    index: usize,
}

impl PyRng {
    /// Equivalent to `random.Random(seed)` for an unsigned 32-bit integer seed.
    pub fn new(seed: u32) -> Self {
        let mut rng = Self {
            state: [0; N],
            index: N,
        };
        rng.init_genrand(19_650_218);
        // A CRC32 is one 32-bit key word, including the zero seed (CPython uses
        // `keyused = max(1, (bits + 31) / 32)`).
        rng.init_by_array(&[seed]);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..N {
            self.state[index] = 1_812_433_253u32
                .wrapping_mul(self.state[index - 1] ^ (self.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..N.max(key.len()) {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..(N - 1) {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn gen_u32(&mut self) -> u32 {
        if self.index >= N {
            for kk in 0..(N - M) {
                let y = (self.state[kk] & UPPER_MASK) | (self.state[kk + 1] & LOWER_MASK);
                self.state[kk] =
                    self.state[kk + M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.state[kk] & UPPER_MASK) | (self.state[kk + 1] & LOWER_MASK);
                self.state[kk] =
                    self.state[kk + M - N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            let y = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
            self.state[N - 1] =
                self.state[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn getrandbits(&mut self, bits: u32) -> u32 {
        debug_assert!((1..=32).contains(&bits));
        self.gen_u32() >> (32 - bits)
    }

    fn rand_below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        let k = u64::BITS - (n as u64).leading_zeros();
        loop {
            let candidate = self.getrandbits(k) as usize;
            if candidate < n {
                return candidate;
            }
        }
    }

    /// CPython `random.sample(range(n), n)` on the selection-sampling branch.
    ///
    /// CPython fills `result[i]` from `pool[j]`, then moves the last unselected
    /// element into slot `j`. This is not a swap with slot `i`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let j = self.rand_below(n - i);
            result.push(pool[j]);
            pool[j] = pool[n - i - 1];
        }
        result
    }
}

/// `[identity, rng.sample(...), rng.sample(...), rng.sample(...)]`, as the shim builds it.
pub fn permutations(seed: u32, n: usize) -> Vec<Vec<usize>> {
    let mut rng = PyRng::new(seed);
    let mut orders = Vec::with_capacity(4);
    orders.push((0..n).collect());
    for _ in 0..3 {
        orders.push(rng.permutation(n));
    }
    orders
}

/// `zlib.crc32` (CRC-32/ISO-HDLC) of the UTF-8 question bytes.
pub fn question_seed(question: &str) -> u32 {
    crc32fast::hash(question.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{PyRng, permutations, question_seed};

    // Generated with CPython 3.x:
    //   rng = random.Random(zlib.crc32(b"hello"))
    //   [rng.sample(range(n), n) for _ in range(3)]
    #[test]
    fn matches_cpython_permutations_for_17() {
        assert_eq!(question_seed("hello"), 0x3610_a686);
        let orders = permutations(question_seed("hello"), 17);
        assert_eq!(orders[0], (0..17).collect::<Vec<_>>());
        assert_eq!(
            orders[1],
            vec![8, 6, 0, 1, 4, 3, 15, 13, 5, 9, 11, 12, 10, 16, 2, 7, 14]
        );
        assert_eq!(
            orders[2],
            vec![6, 15, 2, 9, 13, 16, 8, 3, 1, 4, 11, 10, 0, 14, 12, 7, 5]
        );
        assert_eq!(
            orders[3],
            vec![1, 5, 6, 3, 8, 16, 12, 4, 0, 2, 7, 11, 9, 15, 13, 10, 14]
        );
    }

    #[test]
    fn matches_cpython_permutations_for_32() {
        let orders = permutations(question_seed("hello"), 32);
        assert_eq!(
            orders[1],
            vec![
                16, 18, 19, 29, 6, 1, 3, 8, 27, 12, 25, 11, 2, 22, 14, 30, 24, 5, 10, 15, 21, 7,
                28, 23, 20, 0, 9, 26, 31, 17, 4, 13
            ]
        );
        assert_eq!(
            orders[2],
            vec![
                28, 22, 20, 1, 25, 5, 12, 6, 16, 3, 17, 9, 31, 26, 8, 18, 23, 4, 19, 27, 21, 11, 7,
                15, 13, 0, 2, 29, 24, 30, 14, 10
            ]
        );
        assert_eq!(
            orders[3],
            vec![
                22, 2, 13, 9, 26, 5, 29, 24, 10, 14, 11, 20, 7, 23, 31, 25, 27, 3, 30, 8, 21, 19,
                17, 16, 15, 28, 18, 0, 6, 1, 12, 4
            ]
        );
    }

    #[test]
    fn matches_cpython_permutations_for_256() {
        let orders = permutations(question_seed("hello"), 256);
        assert_eq!(&orders[1][..8], &[128, 150, 158, 159, 50, 8, 28, 67]);
        assert_eq!(orders[1].len(), 256);
    }

    #[test]
    fn zero_seed_is_cpython_compatible() {
        // random.Random(0).random() == 0.8444218515250481. The top 53 bits of the
        // first two 32-bit draws produce that value.
        let mut rng = PyRng::new(0);
        let a = rng.getrandbits(32) >> 5;
        let b = rng.getrandbits(32) >> 6;
        let value = (a as f64 * 67_108_864.0 + b as f64) / 9_007_199_254_740_992.0;
        assert!((value - 0.844_421_851_525_048_1).abs() < 1e-15);
    }

    #[test]
    fn permutations_are_well_formed_and_deterministic() {
        for n in [2usize, 16, 17, 31, 32, 255, 256] {
            let first = permutations(question_seed("question"), n);
            let second = permutations(question_seed("question"), n);
            assert_eq!(first, second);
            for order in first {
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            }
        }
    }
}
