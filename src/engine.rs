//! Decision math: single pass, tournament, and deterministic permute strategies.

use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use futures::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    DEFAULT_STRATEGY, NATIVE_CHOICE_OPTIONS,
    backend::Backend,
    error::EngineError,
    head::{DecisionHead, FittedKind, Temperatures, scale_and_softmax},
    prompt::{self, build_content, content_text, label_names, validate_state},
    rng,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Noul,
    Score,
    Choice,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Score => "score",
            Kind::Choice => "choice",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "noul" => Some(Kind::Noul),
            "score" => Some(Kind::Score),
            "choice" => Some(Kind::Choice),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    #[default]
    Auto,
    Single,
    Tournament,
    Permute,
}

impl Strategy {
    pub fn effective(self) -> Strategy {
        match self {
            Strategy::Auto => Strategy::Tournament,
            other => other,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Strategy::Auto => "auto",
            Strategy::Single => "single",
            Strategy::Tournament => DEFAULT_STRATEGY,
            Strategy::Permute => "permute",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub requests: u64,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.requests += other.requests;
    }

    pub fn total_tokens(self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

#[derive(Debug, Clone)]
pub struct DecideOutcome {
    pub probabilities: Vec<f64>,
    pub usage: Usage,
    pub effective_strategy: Strategy,
}

pub struct DecisionEngine {
    pub head: Arc<DecisionHead>,
    pub temperatures: Arc<Temperatures>,
    pub fitted: Option<Arc<HashMap<String, FittedKind>>>,
    pub label_ids: Arc<Vec<u32>>,
    pub label_names: Arc<Vec<String>>,
    pub backend: Arc<dyn Backend>,
}

impl DecisionEngine {
    pub fn new(
        head: Arc<DecisionHead>,
        temperatures: Arc<Temperatures>,
        fitted: Option<Arc<HashMap<String, FittedKind>>>,
        label_ids: Vec<u32>,
        backend: Arc<dyn Backend>,
    ) -> Self {
        Self {
            head,
            temperatures,
            fitted,
            label_ids: Arc::new(label_ids),
            label_names: Arc::new(label_names()),
            backend,
        }
    }

    pub fn max_options(&self) -> usize {
        self.label_ids.len().min(256)
    }

    /// Run one decision and return the option distribution plus backend usage.
    pub async fn decide(
        &self,
        kind: Kind,
        state: &Value,
        question: &str,
        options: &[String],
        strategy: Strategy,
    ) -> Result<DecideOutcome, EngineError> {
        validate_state(state).map_err(EngineError::Internal)?;
        let effective = strategy.effective();
        metrics::histogram!("jev_decide_options").record(options.len() as f64);
        let (probabilities, usage) = self
            .distribution(kind, state, question, options, effective)
            .await?;
        metrics::counter!(
            "jev_decide_total",
            "kind" => kind.as_str(),
            "strategy" => effective.as_str()
        )
        .increment(1);
        Ok(DecideOutcome {
            probabilities,
            usage,
            effective_strategy: effective,
        })
    }

    async fn distribution(
        &self,
        kind: Kind,
        state: &Value,
        question: &str,
        options: &[String],
        strategy: Strategy,
    ) -> Result<(Vec<f64>, Usage), EngineError> {
        if kind != Kind::Choice
            || options.len() <= NATIVE_CHOICE_OPTIONS
            || strategy == Strategy::Single
        {
            let (probabilities, usage) = self.s1_pass(kind, state, question, options).await?;
            return Ok((probabilities, usage));
        }
        match strategy {
            Strategy::Permute => self.permute(kind, state, question, options).await,
            Strategy::Tournament | Strategy::Auto => {
                self.tournament(kind, state, question, options).await
            }
            Strategy::Single => self.s1_pass(kind, state, question, options).await,
        }
    }

    /// One System 1 read-out over at most 256 options.
    pub async fn s1_pass(
        &self,
        kind: Kind,
        state: &Value,
        question: &str,
        options: &[String],
    ) -> Result<(Vec<f64>, Usage), EngineError> {
        let [lo, hi] = self.head.range(kind);
        let (ids, bias) = if kind == Kind::Choice {
            let count = options.len();
            if count == 0 || count > self.label_ids.len() {
                return Err(EngineError::Internal(format!(
                    "choice pass has {count} options but {} labels are available",
                    self.label_ids.len()
                )));
            }
            let ids = self.label_ids[..count].to_vec();
            let bias = (0..count)
                .map(|index| {
                    if lo + index < hi {
                        self.head.bias[lo + index]
                    } else {
                        0.0
                    }
                })
                .collect();
            (ids, bias)
        } else {
            let count = options.len();
            if count == 0 || lo + count > hi {
                return Err(EngineError::Internal(format!(
                    "{} pass has {count} options outside slot range [{lo}, {hi})",
                    kind.as_str()
                )));
            }
            (
                self.head.verbalizer_ids[lo..lo + count].to_vec(),
                self.head.bias[lo..lo + count].to_vec(),
            )
        };

        let (parts, has_image) = prompt::parts(state);
        if has_image {
            return Err(EngineError::Unsupported(
                "native vision not implemented: image states are not supported".to_owned(),
            ));
        }
        let content = build_content(kind.as_str(), &parts, question, options, &self.label_names);
        let text = content_text(&content);
        let logits = self.backend.logits(&text, &ids).await?;
        if logits.len() != ids.len() {
            return Err(EngineError::BadResponse(format!(
                "backend returned {} logits for {} requested ids",
                logits.len(),
                ids.len()
            )));
        }
        // Raw logits and log-probabilities differ by a constant (log Z); the
        // head softmax normalizer cancels it, so the existing math is unchanged.
        let logprobs: HashMap<u32, f64> = ids
            .iter()
            .copied()
            .zip(logits.iter().map(|value| *value as f64))
            .collect();
        let prompt_tokens = self.backend.prompt_tokens(&text);
        metrics::histogram!("jev_decide_pass_options").record(options.len() as f64);
        metrics::counter!("jev_backend_prompt_tokens_total").increment(prompt_tokens);

        let fitted = self.fitted.as_ref().and_then(|map| map.get(kind.as_str()));
        let probabilities =
            scale_and_softmax(&logprobs, &ids, &bias, self.temperatures.get(kind), fitted);
        // One forward produces one decision read-out position.
        let usage = Usage {
            prompt_tokens,
            completion_tokens: 1,
            requests: 1,
        };
        Ok((probabilities, usage))
    }

    async fn permute(
        &self,
        kind: Kind,
        state: &Value,
        question: &str,
        options: &[String],
    ) -> Result<(Vec<f64>, Usage), EngineError> {
        let n = options.len();
        let seed = rng::question_seed(question);
        let orders = rng::permutations(seed, n);
        let passes = orders.iter().map(|order| {
            let permuted: Vec<String> = order.iter().map(|index| options[*index].clone()).collect();
            async move { self.s1_pass(kind, state, question, &permuted).await }
        });
        let results = join_all(passes).await;
        let mut usage = Usage::default();
        let mut probabilities = vec![0.0; n];
        let mut completed = 0usize;
        for (order, result) in orders.iter().zip(results) {
            let (probs, pass_usage) = result?;
            usage.add(pass_usage);
            for (position, option_index) in order.iter().enumerate() {
                probabilities[*option_index] += probs[position] / orders.len() as f64;
            }
            completed += 1;
        }
        debug_assert_eq!(completed, orders.len());
        Ok((probabilities, usage))
    }

    async fn tournament(
        &self,
        kind: Kind,
        state: &Value,
        question: &str,
        options: &[String],
    ) -> Result<(Vec<f64>, Usage), EngineError> {
        let n = options.len();
        let groups = groups_for(n, NATIVE_CHOICE_OPTIONS);
        let passes = groups.iter().map(|group| {
            let opts: Vec<String> = group.iter().map(|index| options[*index].clone()).collect();
            async move { self.s1_pass(kind, state, question, &opts).await }
        });
        let results = join_all(passes).await;
        let mut usage = Usage::default();
        let mut in_group = vec![0.0; n];
        for (group, result) in groups.iter().zip(results) {
            let (probs, pass_usage) = result?;
            usage.add(pass_usage);
            for (option_index, probability) in group.iter().zip(probs) {
                in_group[*option_index] = probability;
            }
        }

        let mut chosen = Vec::with_capacity(groups.len());
        for group in &groups {
            let mut best = group[0];
            for option in &group[1..] {
                if in_group[*option] > in_group[best] {
                    best = *option;
                }
            }
            chosen.push(best);
        }
        let mut chosen_set = vec![false; n];
        for option in &chosen {
            chosen_set[*option] = true;
        }
        let mut rest: Vec<usize> = (0..n).filter(|index| !chosen_set[*index]).collect();
        rest.sort_by(|left, right| {
            in_group[*right]
                .partial_cmp(&in_group[*left])
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        let take = NATIVE_CHOICE_OPTIONS.saturating_sub(chosen.len());
        let mut finalists = chosen.clone();
        finalists.extend(rest.into_iter().take(take));
        finalists.sort_unstable();

        let final_options: Vec<String> = finalists
            .iter()
            .map(|index| options[*index].clone())
            .collect();
        let (final_probabilities, final_usage) =
            self.s1_pass(kind, state, question, &final_options).await?;
        usage.add(final_usage);

        let mut final_by_option: Vec<Option<f64>> = vec![None; n];
        for (option_index, probability) in finalists.iter().zip(final_probabilities) {
            final_by_option[*option_index] = Some(probability);
        }
        let probabilities = tournament_combination(n, &groups, &in_group, &final_by_option);
        Ok((probabilities, usage))
    }
}

/// `_groups(n, k=16)`: `ceil(n/k)` groups as evenly sized as possible.
pub fn groups_for(n: usize, group_size: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    let group_size = group_size.max(1);
    let count = n.div_ceil(group_size);
    let base = n / count;
    let extra = n % count;
    let mut groups = Vec::with_capacity(count);
    let mut start = 0usize;
    for index in 0..count {
        let size = base + usize::from(index < extra);
        groups.push((start..start + size).collect());
        start += size;
    }
    groups
}

/// Port of the shim's probability combination:
///
/// ```text
/// share[g] = sum of final probabilities of finalists from group g
/// cap[g]   = sum of in-group probabilities of those same finalists
/// among    = sum(share[g] * cap[g])
/// p[o]     = final[o] * among                 if o was a finalist
///          = share[group(o)] * in_group[o]    otherwise
/// ```
pub fn tournament_combination(
    n: usize,
    groups: &[Vec<usize>],
    in_group: &[f64],
    final_by_option: &[Option<f64>],
) -> Vec<f64> {
    let mut group_of = vec![usize::MAX; n];
    for (group_index, group) in groups.iter().enumerate() {
        for option in group {
            group_of[*option] = group_index;
        }
    }
    let mut share = vec![0.0; groups.len()];
    let mut cap = vec![0.0; groups.len()];
    for (option, probability) in final_by_option.iter().enumerate() {
        if let Some(probability) = probability {
            let group = group_of[option];
            share[group] += probability;
            cap[group] += in_group[option];
        }
    }
    let among: f64 = share.iter().zip(&cap).map(|(share, cap)| share * cap).sum();

    let mut probabilities = vec![0.0; n];
    for option in 0..n {
        probabilities[option] = match final_by_option[option] {
            Some(probability) => probability * among,
            None => share[group_of[option]] * in_group[option],
        };
    }
    let total: f64 = probabilities.iter().sum();
    if total.is_finite() && total > 0.0 {
        for probability in &mut probabilities {
            *probability /= total;
        }
    } else {
        probabilities.fill(1.0 / n as f64);
    }
    probabilities
}

#[cfg(test)]
mod tests {
    use super::{Kind, Strategy, groups_for, tournament_combination};

    #[test]
    fn groups_are_even_and_match_reference() {
        assert_eq!(
            groups_for(17, 16),
            vec![(0..9).collect::<Vec<_>>(), (9..17).collect::<Vec<_>>()]
        );
        assert_eq!(
            groups_for(32, 16),
            vec![(0..16).collect::<Vec<_>>(), (16..32).collect::<Vec<_>>()]
        );
        assert_eq!(groups_for(256, 16).len(), 16);
        assert!(groups_for(256, 16).iter().all(|group| group.len() == 16));
        assert_eq!(groups_for(255, 16).len(), 16);
        assert_eq!(groups_for(255, 16).iter().map(Vec::len).sum::<usize>(), 255);
    }

    #[test]
    fn tournament_combination_matches_hand_computation() {
        let groups = vec![vec![0, 1], vec![2, 3]];
        let in_group = [0.6, 0.4, 0.9, 0.1];
        let final_by_option = [Some(0.7), None, Some(0.3), None];
        let probabilities = tournament_combination(4, &groups, &in_group, &final_by_option);
        assert!((probabilities[0] - 0.483).abs() < 1e-12);
        assert!((probabilities[1] - 0.28).abs() < 1e-12);
        assert!((probabilities[2] - 0.207).abs() < 1e-12);
        assert!((probabilities[3] - 0.03).abs() < 1e-12);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kind_and_strategy_round_trip() {
        for kind in [Kind::Noul, Kind::Score, Kind::Choice] {
            assert_eq!(Kind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(Kind::parse("bogus"), None);
        assert_eq!(Strategy::default().effective(), Strategy::Tournament);
        assert_eq!(Strategy::Auto.effective(), Strategy::Tournament);
        assert_eq!(Strategy::Single.effective(), Strategy::Single);
    }
}
