//! Decision-head metadata: bias vectors, trained verbalizers, calibration and softmax math.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use crate::engine::Kind;

/// Per-kind temperatures from `calibration.json`.
#[derive(Debug, Clone)]
pub struct Temperatures {
    pub noul: f64,
    pub choice: f64,
    pub score: f64,
    /// The original `per_kind` object, preserved for `/v1/decide/info`.
    pub raw: Value,
}

impl Temperatures {
    pub fn get(&self, kind: Kind) -> f64 {
        match kind {
            Kind::Noul => self.noul,
            Kind::Choice => self.choice,
            Kind::Score => self.score,
        }
    }
}

/// Optional serving-side calibration produced by `fit_bias.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct FittedKind {
    pub scale: f64,
    pub bias: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct HeadFile {
    bias: Vec<f64>,
    verbalizer_ids: Vec<u32>,
    slots: Slots,
}

#[derive(Debug, Clone, Deserialize)]
struct Slots {
    ranges: Ranges,
    #[serde(default)]
    template_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Ranges {
    noul: [usize; 2],
    score: [usize; 2],
    choice: [usize; 2],
}

/// The parsed decision head. Slot ranges are `[lo, hi)` into both `bias` and
/// `verbalizer_ids`; the choice range carries the 16 trained A-P verbalizers.
#[derive(Debug, Clone)]
pub struct DecisionHead {
    pub bias: Vec<f64>,
    pub verbalizer_ids: Vec<u32>,
    pub noul: [usize; 2],
    pub score: [usize; 2],
    pub choice: [usize; 2],
    pub template_version: Option<String>,
}

impl DecisionHead {
    pub fn range(&self, kind: Kind) -> [usize; 2] {
        match kind {
            Kind::Noul => self.noul,
            Kind::Choice => self.choice,
            Kind::Score => self.score,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (name, range) in [
            ("noul", self.noul),
            ("score", self.score),
            ("choice", self.choice),
        ] {
            let [lo, hi] = range;
            if lo >= hi {
                bail!("decision_head has an empty {name} range [{lo}, {hi})");
            }
            if hi > self.bias.len() || hi > self.verbalizer_ids.len() {
                bail!(
                    "decision_head {name} range [{lo}, {hi}) exceeds bias ({}) or verbalizer_ids ({})",
                    self.bias.len(),
                    self.verbalizer_ids.len()
                );
            }
        }
        if self.noul[1] - self.noul[0] != 2 {
            bail!("decision_head noul range must hold exactly 2 slots");
        }
        if self.score[1] - self.score[0] != 6 {
            bail!("decision_head score range must hold exactly 6 slots");
        }
        if self.choice[1] - self.choice[0] < 16 {
            bail!("decision_head choice range must hold at least 16 slots");
        }
        Ok(())
    }
}

/// Everything loaded from disk that shapes probabilities.
#[derive(Debug, Clone)]
pub struct LoadedHead {
    pub head: DecisionHead,
    pub temperatures: Temperatures,
    pub fitted: Option<HashMap<String, FittedKind>>,
    pub head_path: PathBuf,
    pub calibration_path: PathBuf,
    pub fitted_path: Option<PathBuf>,
}

#[derive(Deserialize)]
struct CalibrationFile {
    per_kind: Value,
}

fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|path| path.is_file()).cloned()
}

fn candidates(root: &Path, relative: &[&str]) -> Vec<PathBuf> {
    relative.iter().map(|path| root.join(path)).collect()
}

/// Resolve the decision head and calibration from the AWQ artifact, falling back to
/// the Hugging Face snapshot for files the quantized artifact does not carry.
pub fn load(
    checkpoint: &Path,
    calibration_override: Option<&Path>,
    fitted_override: Option<&Path>,
    snapshot: &Path,
) -> Result<LoadedHead> {
    let mut head_candidates = candidates(
        checkpoint,
        &[
            "decision_head/decision_head.json",
            "adapter_vllm/decision_head.json",
            "decision_head.json",
        ],
    );
    if checkpoint != snapshot {
        head_candidates.extend(candidates(
            snapshot,
            &[
                "adapter_vllm/decision_head.json",
                "decision_head/decision_head.json",
                "decision_head.json",
            ],
        ));
    }
    let head_path = first_existing(&head_candidates).ok_or_else(|| {
        anyhow::anyhow!(
            "decision_head.json not found; searched: {}",
            display_paths(&head_candidates)
        )
    })?;

    let calibration_candidates: Vec<PathBuf> = match calibration_override {
        Some(path) => vec![path.to_path_buf()],
        None => {
            let mut paths = candidates(
                checkpoint,
                &["decision_head/calibration.json", "calibration.json"],
            );
            if checkpoint != snapshot {
                paths.extend(candidates(snapshot, &["calibration.json"]));
            }
            paths
        }
    };
    let calibration_path = first_existing(&calibration_candidates).ok_or_else(|| {
        anyhow::anyhow!(
            "calibration.json not found; searched: {}",
            display_paths(&calibration_candidates)
        )
    })?;

    let head_file: HeadFile = serde_json::from_slice(
        &std::fs::read(&head_path).with_context(|| format!("reading {}", head_path.display()))?,
    )
    .with_context(|| format!("parsing {}", head_path.display()))?;
    let head = DecisionHead {
        bias: head_file.bias,
        verbalizer_ids: head_file.verbalizer_ids,
        noul: head_file.slots.ranges.noul,
        score: head_file.slots.ranges.score,
        choice: head_file.slots.ranges.choice,
        template_version: head_file.slots.template_version,
    };
    head.validate()?;

    let calibration_file: CalibrationFile = serde_json::from_slice(
        &std::fs::read(&calibration_path)
            .with_context(|| format!("reading {}", calibration_path.display()))?,
    )
    .with_context(|| format!("parsing {}", calibration_path.display()))?;
    if !calibration_file.per_kind.is_object() {
        bail!("{}: per_kind must be an object", calibration_path.display());
    }
    let temperature = |kind: &str| -> Result<f64> {
        let value = calibration_file
            .per_kind
            .get(kind)
            .and_then(Value::as_f64)
            .with_context(|| {
                format!(
                    "{}: missing numeric per_kind.{kind}",
                    calibration_path.display()
                )
            })?;
        if !(value.is_finite() && value > 0.0) {
            bail!(
                "{}: per_kind.{kind} must be a positive finite number, got {value}",
                calibration_path.display()
            );
        }
        Ok(value)
    };
    let temperatures = Temperatures {
        noul: temperature("noul")?,
        choice: temperature("choice")?,
        score: temperature("score")?,
        raw: calibration_file.per_kind,
    };

    let (fitted, fitted_path) = match fitted_override {
        Some(path) if path.is_file() => {
            let parsed: HashMap<String, FittedKind> = serde_json::from_slice(
                &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
            )
            .with_context(|| format!("parsing {}", path.display()))?;
            for (kind, fit) in &parsed {
                if !(fit.scale.is_finite() && fit.scale > 0.0) {
                    bail!(
                        "{}: fitted {kind}.scale must be positive and finite",
                        path.display()
                    );
                }
                if fit.bias.iter().any(|value| !value.is_finite()) {
                    bail!("{}: fitted {kind}.bias must be finite", path.display());
                }
            }
            (Some(parsed), Some(path.to_path_buf()))
        }
        Some(path) => {
            tracing::warn!(
                path = %path.display(),
                "JEV_CALIBRATION_FITTED does not exist; fitted calibration disabled"
            );
            (None, None)
        }
        None => (None, None),
    };

    Ok(LoadedHead {
        head,
        temperatures,
        fitted,
        head_path,
        calibration_path,
        fitted_path,
    })
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Apply head bias, per-kind temperature (and optional fitted calibration), then softmax.
///
/// `logprobs` is the vLLM map from token id to logprob. Tokens the backend did not
/// return fall back to `-1e9`, matching the reference implementation. Fitted bias
/// entries beyond the supplied vector are treated as zero instead of panicking; the
/// shipped fit only covers the native 16-way head, while serving accepts up to 256
/// choices.
pub fn scale_and_softmax(
    logprobs: &HashMap<u32, f64>,
    ids: &[u32],
    bias: &[f64],
    temperature: f64,
    fitted: Option<&FittedKind>,
) -> Vec<f64> {
    let mut scaled: Vec<f64> = ids
        .iter()
        .zip(bias)
        .map(|(token, bias)| {
            let logprob = logprobs.get(token).copied().unwrap_or(-1e9).max(-1e9);
            (logprob + bias) / temperature
        })
        .collect();
    if let Some(fit) = fitted {
        for (index, value) in scaled.iter_mut().enumerate() {
            let raw = *value * temperature;
            let extra_bias = fit.bias.get(index).copied().unwrap_or(0.0);
            *value = raw * fit.scale + extra_bias;
        }
    }
    softmax(&scaled)
}

/// Numerically stable softmax. Empty input yields an empty vector.
pub fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = values.iter().map(|value| (value - top).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    if total.is_finite() && total > 0.0 {
        exponentials
            .into_iter()
            .map(|value| value / total)
            .collect()
    } else {
        vec![1.0 / values.len() as f64; values.len()]
    }
}

/// Argmax with Python `max(range(n), key=...)` tie-breaking: the lowest index wins.
pub fn argmax_first(values: &[f64]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, value) in values.iter().enumerate() {
        match best {
            Some(current) if values[current] >= *value => {}
            _ => best = Some(index),
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{FittedKind, argmax_first, scale_and_softmax, softmax};

    #[test]
    fn softmax_is_normalised_and_stable() {
        let probabilities = softmax(&[1000.0, 1000.0, 999.0]);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probabilities[0] > probabilities[2]);
        assert_eq!(probabilities[0], probabilities[1]);
    }

    #[test]
    fn bias_temperature_and_missing_tokens_match_reference() {
        let mut logprobs = HashMap::new();
        logprobs.insert(1, 0.75f64.ln());
        logprobs.insert(2, 0.25f64.ln());
        // Token 3 is absent: -1e9 dominates the softmax for the other two.
        let probabilities = scale_and_softmax(&logprobs, &[1, 2, 3], &[0.0, 0.0, 0.0], 1.0, None);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((probabilities[0] - 0.75).abs() < 1e-9);
        assert!((probabilities[1] - 0.25).abs() < 1e-9);
        assert!(probabilities[2] < 1e-9);
    }

    #[test]
    fn temperature_changes_sharpness() {
        let mut logprobs = HashMap::new();
        logprobs.insert(1, 0.0);
        logprobs.insert(2, 0.5);
        let cold = scale_and_softmax(&logprobs, &[1, 2], &[0.0, 0.0], 0.5, None);
        let warm = scale_and_softmax(&logprobs, &[1, 2], &[0.0, 0.0], 2.0, None);
        assert!(cold[1] > warm[1]);
    }

    #[test]
    fn fitted_calibration_operates_on_raw_logits() {
        let mut logprobs = HashMap::new();
        logprobs.insert(7, 1.0);
        logprobs.insert(8, 0.0);
        let fitted = FittedKind {
            scale: 2.0,
            bias: vec![0.1],
        };
        // raw = (logit)/T*T = logit; scaled = raw*scale + bias.
        let probabilities = scale_and_softmax(&logprobs, &[7, 8], &[0.0, 0.0], 1.7, Some(&fitted));
        let expected = softmax(&[2.1, 0.0]);
        assert!((probabilities[0] - expected[0]).abs() < 1e-12);
        assert!((probabilities[1] - expected[1]).abs() < 1e-12);
    }

    #[test]
    fn argmax_keeps_first_on_ties() {
        assert_eq!(argmax_first(&[0.1, 0.4, 0.4]), Some(1));
        assert_eq!(argmax_first(&[]), None);
    }
}
