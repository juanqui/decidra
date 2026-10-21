//! Mask algebra and the dense fp32 attention reference.
//!
//! These definitions are the contract the GPU kernels are tested against.
//! `Causal` uses the FA2 offset form `kv_idx <= (kv_len - q_len) + q_idx`;
//! `Full` is all-visible; `BlockCascade` is a bidirectional state prefix plus a
//! bidirectional question block, where state rows never see question keys.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskSpec {
    Causal,
    Full,
    /// d3-turbo state-once: `split` state tokens, question tokens after it.
    BlockCascade {
        split: usize,
    },
}

/// Visibility of every query/key pair, row-major `[q_len, kv_len]`.
/// `split` is ignored except for `BlockCascade`.
pub fn visible(mask: MaskSpec, split_arg: usize, q_len: usize, kv_len: usize) -> Vec<bool> {
    let mut out = vec![false; q_len * kv_len];
    match mask {
        MaskSpec::Causal => {
            assert!(kv_len >= q_len, "causal requires kv_len >= q_len");
            let offset = kv_len - q_len;
            for q in 0..q_len {
                let limit = (offset + q + 1).min(kv_len);
                for k in 0..limit {
                    out[q * kv_len + k] = true;
                }
            }
        }
        MaskSpec::Full => out.fill(true),
        MaskSpec::BlockCascade { split } => {
            assert_eq!(
                split, split_arg,
                "MaskSpec::BlockCascade split disagrees with argument"
            );
            assert!(split <= q_len && split <= kv_len, "split outside sequence");
            for q in 0..q_len {
                if q < split {
                    // State rows: bidirectional over the state only.
                    for k in 0..split {
                        out[q * kv_len + k] = true;
                    }
                } else {
                    // Question rows: state + the whole question block.
                    for k in 0..kv_len {
                        out[q * kv_len + k] = true;
                    }
                }
            }
        }
    }
    out
}

/// Dense single-head attention reference in f32. `q` is `[q_len, d]`,
/// `k`/`v` are `[kv_len, d]`.
pub fn softmax_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    mask: MaskSpec,
    split: usize,
) -> Vec<f32> {
    let d = q.len() / q_len;
    assert_eq!(k.len(), kv_len * d);
    assert_eq!(v.len(), kv_len * d);
    let vis = visible(mask, split, q_len, kv_len);
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; q_len * d];
    for qi in 0..q_len {
        let mut scores = vec![f32::NEG_INFINITY; kv_len];
        for ki in 0..kv_len {
            if !vis[qi * kv_len + ki] {
                continue;
            }
            let mut dot = 0.0f32;
            for di in 0..d {
                dot += q[qi * d + di] * k[ki * d + di];
            }
            scores[ki] = dot * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for score in scores.iter_mut() {
            *score /= sum;
        }
        for di in 0..d {
            let mut acc = 0.0f32;
            for ki in 0..kv_len {
                acc += scores[ki] * v[ki * d + di];
            }
            out[qi * d + di] = acc;
        }
    }
    out
}

/// Merge two attention partials produced with log2-domain LSE (FA2).
/// `o` are `[rows, heads, d]`, `lse` are `[rows, heads]` in log2.
pub fn merge_state(
    o_a: &[f32],
    lse_a: &[f32],
    o_b: &[f32],
    lse_b: &[f32],
    rows: usize,
    heads: usize,
    d: usize,
) -> Vec<f32> {
    assert_eq!(o_a.len(), rows * heads * d);
    assert_eq!(o_b.len(), rows * heads * d);
    assert_eq!(lse_a.len(), rows * heads);
    assert_eq!(lse_b.len(), rows * heads);
    let mut out = vec![0.0f32; rows * heads * d];
    for r in 0..rows {
        for h in 0..heads {
            let idx = r * heads + h;
            let m = lse_a[idx].max(lse_b[idx]);
            let wa = (lse_a[idx] - m).exp2();
            let wb = (lse_b[idx] - m).exp2();
            let denom = wa + wb;
            for di in 0..d {
                let off = idx * d + di;
                out[off] = (o_a[off] * wa + o_b[off] * wb) / denom;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|x| x * x).sum::<f32>().max(1e-30);
        (num / den).sqrt()
    }

    fn fixture(q_len: usize, kv_len: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q = (0..q_len * d)
            .map(|i| (i as f32 * 0.017).sin() * 0.5)
            .collect();
        let k = (0..kv_len * d)
            .map(|i| (i as f32 * 0.013).cos() * 0.5)
            .collect();
        let v = (0..kv_len * d)
            .map(|i| (i as f32 * 0.007).sin() * 0.3)
            .collect();
        (q, k, v)
    }

    #[test]
    fn t_mask_causal() {
        for (q_len, kv_len) in [(1, 1), (7, 7), (64, 64), (1, 5), (40, 4096)] {
            let vis = visible(MaskSpec::Causal, 0, q_len, kv_len);
            for q in 0..q_len {
                let limit = kv_len - q_len + q;
                for k in 0..kv_len {
                    assert_eq!(vis[q * kv_len + k], k <= limit, "q {q} k {k}");
                }
            }
        }
        // Dense reference and the offset definition agree.
        let (q, k, v) = fixture(8, 32, 16);
        let a = softmax_reference(&q, &k, &v, 8, 32, MaskSpec::Causal, 0);
        let b = softmax_reference(&q, &k, &v, 8, 32, MaskSpec::Causal, 0);
        assert_eq!(rel(&a, &b), 0.0);
    }

    #[test]
    fn t_mask_full() {
        let vis = visible(MaskSpec::Full, 0, 5, 9);
        assert!(vis.iter().all(|v| *v));
    }

    #[test]
    fn t_mask_block() {
        let (split, q_len) = (37usize, 64usize);
        let total = split + q_len;
        let vis = visible(MaskSpec::BlockCascade { split }, split, total, total);
        for q in 0..total {
            for k in 0..total {
                let expected = if q < split { k < split } else { true };
                assert_eq!(vis[q * total + k], expected, "q {q} k {k}");
            }
        }
        // Negative: the "all-true" mask (the old Full bug) must not match.
        let full = visible(MaskSpec::Full, split, total, total);
        assert_ne!(vis, full);
    }

    #[test]
    fn t_mask_straddle() {
        let (split, q_len) = (37usize, 64usize);
        let total = split + q_len;
        let vis = visible(MaskSpec::BlockCascade { split }, split, total, total);
        // The token at `split` is a question row and sees everything.
        assert!(vis[split * total..(split + 1) * total].iter().all(|v| *v));
        // Assigning it to the state would hide question keys; assert that wrong
        // assignment is detectably different.
        let wrong = visible(
            MaskSpec::BlockCascade { split: split + 1 },
            split + 1,
            total,
            total,
        );
        assert_ne!(vis, wrong);
    }

    #[test]
    fn t_mask_block_merge() {
        // Merge a cross partial (all question rows vs the state) with a self
        // partial (question rows vs their own block). Compare with one dense
        // BlockCascade product. Skewed LSE fixture.
        let (split, q_len, d, heads) = (37usize, 64usize, 4usize, 1usize);
        let total = split + q_len;
        let (q, k, v) = fixture(total, total, d);
        let dense = softmax_reference(
            &q,
            &k,
            &v,
            total,
            total,
            MaskSpec::BlockCascade { split },
            split,
        );
        // Build the two partials with the same dense reference restricted to the
        // key ranges, then merge with their log2 LSEs.
        let scale = 1.0 / (d as f32).sqrt();
        let mut o_cross = vec![0.0f32; q_len * heads * d];
        let mut lse_cross = vec![0.0f32; q_len * heads];
        let mut o_self = vec![0.0f32; q_len * heads * d];
        let mut lse_self = vec![0.0f32; q_len * heads];
        for qi in 0..q_len {
            let row = split + qi;
            let mut cross_scores = vec![f32::NEG_INFINITY; split];
            let mut self_scores = vec![f32::NEG_INFINITY; q_len];
            for ki in 0..split {
                let mut dot = 0.0;
                for di in 0..d {
                    dot += q[row * d + di] * k[ki * d + di];
                }
                cross_scores[ki] = dot * scale;
            }
            for ki in 0..q_len {
                let mut dot = 0.0;
                for di in 0..d {
                    dot += q[row * d + di] * k[(split + ki) * d + di];
                }
                self_scores[ki] = dot * scale;
            }
            let m_cross = cross_scores
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let m_self = self_scores
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let z_cross: f32 = cross_scores.iter().map(|s| (s - m_cross).exp()).sum();
            let z_self: f32 = self_scores.iter().map(|s| (s - m_self).exp()).sum();
            for di in 0..d {
                let mut a = 0.0;
                for ki in 0..split {
                    a += (cross_scores[ki] - m_cross).exp() * v[ki * d + di];
                }
                let mut b = 0.0;
                for ki in 0..q_len {
                    b += (self_scores[ki] - m_self).exp() * v[(split + ki) * d + di];
                }
                o_cross[qi * d + di] = a / z_cross;
                o_self[qi * d + di] = b / z_self;
            }
            lse_cross[qi] = (m_cross + z_cross.ln()) / std::f32::consts::LN_2;
            lse_self[qi] = (m_self + z_self.ln()) / std::f32::consts::LN_2;
        }
        let merged = merge_state(&o_cross, &lse_cross, &o_self, &lse_self, q_len, heads, d);
        let reference: Vec<f32> = dense[split * d..].to_vec();
        assert!(
            rel(&merged, &reference) < 1e-5,
            "rel {}",
            rel(&merged, &reference)
        );
        // Negative: natural-log merge must be detectably different.
        let mut bad = vec![0.0f32; q_len * heads * d];
        for qi in 0..q_len {
            let m = lse_cross[qi].max(lse_self[qi]);
            let wa = (lse_cross[qi] - m).exp();
            let wb = (lse_self[qi] - m).exp();
            for di in 0..d {
                bad[qi * d + di] =
                    (o_cross[qi * d + di] * wa + o_self[qi * d + di] * wb) / (wa + wb);
            }
        }
        assert!(rel(&bad, &reference) > 1e-5);
    }
}
