//! Prompt construction and the wide-choice label table.
//!
//! This is a direct port of `decide_shim.py`'s `LABELS`, `_single_token_labels`,
//! `_parts`, `build_content`, and `RAW_TEMPLATE`. The prompt layout is part of the
//! model's trained protocol; changing a space or a newline changes the distribution.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::head::DecisionHead;
use crate::pyjson;

/// The shim's prompt context used to verify that a label is a single token at the
/// decision position.
pub const LABEL_CONTEXT: &str = "x\n{}) y";

/// Raw Qwen chat template that splices vision tokens at image parts. Sent per request
/// to vLLM for image decisions, with `add_generation_prompt: false`.
pub const RAW_TEMPLATE: &str = "{%- for m in messages -%}{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}";

/// `string.ascii_uppercase` then `itertools.product(uppercase, repeat=2)`: A..Z, AA..ZZ.
pub fn label_names() -> Vec<String> {
    let mut labels = Vec::with_capacity(702);
    for byte in b'A'..=b'Z' {
        labels.push((byte as char).to_string());
    }
    for first in b'A'..=b'Z' {
        for second in b'A'..=b'Z' {
            labels.push(format!("{}{}", first as char, second as char));
        }
    }
    labels
}

/// Pure label derivation used by both the real tokenizer and unit tests.
///
/// A label qualifies when its encoding is exactly one token and that token id appears
/// in the encoding of `context.format(label)`.
pub fn derive_label_ids_with<F>(mut encode: F, context: &str, limit: usize) -> Vec<u32>
where
    F: FnMut(&str) -> Vec<u32>,
{
    let mut selected = Vec::with_capacity(limit);
    for label in label_names() {
        let tokenized = encode(&label);
        if tokenized.len() != 1 {
            continue;
        }
        let token = tokenized[0];
        let formatted = if let Some(position) = context.find("{}") {
            format!(
                "{}{}{}",
                &context[..position],
                label,
                &context[position + 2..]
            )
        } else {
            context.to_owned()
        };
        if encode(&formatted).contains(&token) {
            selected.push(token);
            if selected.len() == limit {
                break;
            }
        }
    }
    selected
}

/// Derive the label token ids with the real tokenizer.
pub fn derive_label_ids(tokenizer: &Tokenizer, context: &str, limit: usize) -> Result<Vec<u32>> {
    let mut encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .unwrap_or_default()
    };
    let labels = derive_label_ids_with(&mut encode, context, limit);
    if labels.is_empty() {
        bail!("tokenizer produced no single-token label ids");
    }
    Ok(labels)
}

/// Fail fast when the tokenizer's first 16 labels are not the head's trained A-P
/// verbalizers. A silent mismatch would change every decision.
pub fn validate_label_parity(labels: &[u32], head: &DecisionHead) -> Result<()> {
    head.validate()?;
    let native = head.choice[1] - head.choice[0];
    if labels.len() < native {
        bail!(
            "tokenizer produced {} single-token labels, need at least {native}",
            labels.len()
        );
    }
    let expected = &head.verbalizer_ids[head.choice[0]..head.choice[1]];
    if labels[..native] != *expected {
        bail!(
            "tokenizer label ids {:?} do not match trained choice verbalizers {:?}",
            &labels[..native],
            expected
        );
    }
    Ok(())
}

/// Load `tokenizer.json` from an explicit file/directory, the AWQ checkpoint, or the
/// Hugging Face snapshot (in that order).
pub fn load_tokenizer(
    checkpoint: &Path,
    override_path: Option<&Path>,
    snapshot: &Path,
) -> Result<(Tokenizer, PathBuf)> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = override_path {
        if path.is_dir() {
            candidates.push(path.join("tokenizer.json"));
        } else {
            candidates.push(path.to_path_buf());
        }
    }
    candidates.push(checkpoint.join("tokenizer.json"));
    if checkpoint != snapshot {
        candidates.push(snapshot.join("tokenizer.json"));
    }
    let path = candidates
        .iter()
        .find(|path| path.is_file())
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "tokenizer.json not found; searched: {}",
                candidates
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let tokenizer = Tokenizer::from_file(&path)
        .map_err(|error| anyhow::anyhow!("loading {}: {error}", path.display()))?;
    Ok((tokenizer, path))
}

/// Validate that a request `state` is one of the JSON shapes the protocol accepts.
pub fn validate_state(state: &Value) -> Result<(), String> {
    match state {
        Value::String(_) | Value::Object(_) | Value::Array(_) => Ok(()),
        _ => Err("state must be a string, an object, or an array".to_owned()),
    }
}

/// Count top-level image parts in a state list (the backend reads one image per prompt).
pub fn image_count(state: &Value) -> usize {
    match state {
        Value::Array(items) => items
            .iter()
            .filter(|item| {
                item.as_object().is_some_and(|map| {
                    map.contains_key("image")
                        || map.get("type") == Some(&Value::String("image_url".to_owned()))
                })
            })
            .count(),
        _ => 0,
    }
}

/// Port of `_parts`: turn a state into chat content parts and report whether any part
/// is an image. Images may appear anywhere in a list; the backend accepts one image.
pub fn parts(state: &Value) -> (Vec<Value>, bool) {
    match state {
        Value::String(_) => (vec![text_part(pyjson::py_str(state))], false),
        Value::Object(_) => (vec![text_part(pyjson::dumps(state))], false),
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            let mut has_image = false;
            for item in items {
                match item {
                    Value::String(text) => out.push(text_part(text.clone())),
                    Value::Object(map) if map.contains_key("image") => {
                        let mut image_url = Map::new();
                        image_url.insert(
                            "url".to_owned(),
                            map.get("image").cloned().unwrap_or(Value::Null),
                        );
                        let mut part = Map::new();
                        part.insert("type".to_owned(), Value::String("image_url".to_owned()));
                        part.insert("image_url".to_owned(), Value::Object(image_url));
                        out.push(Value::Object(part));
                        has_image = true;
                    }
                    Value::Object(map)
                        if map.get("type") == Some(&Value::String("image_url".to_owned())) =>
                    {
                        out.push(item.clone());
                        has_image = true;
                    }
                    Value::Object(map)
                        if map.get("type") == Some(&Value::String("text".to_owned())) =>
                    {
                        out.push(item.clone());
                        let _ = map;
                    }
                    other => out.push(text_part(pyjson::dumps(other))),
                }
            }
            (out, has_image)
        }
        other => (vec![text_part(pyjson::dumps(other))], false),
    }
}

fn text_part(text: String) -> Value {
    let mut part = Map::new();
    part.insert("type".to_owned(), Value::String("text".to_owned()));
    part.insert("text".to_owned(), Value::String(text));
    Value::Object(part)
}

/// Port of `build_content`. The returned parts are joined with the empty string for
/// text-only prompts and sent as chat `content` for image prompts.
pub fn build_content(
    kind: &str,
    parts: &[Value],
    question: &str,
    options: &[String],
    labels: &[String],
) -> Vec<Value> {
    let lines = if kind == "choice" {
        options
            .iter()
            .enumerate()
            .map(|(index, option)| {
                let label = labels.get(index).map(String::as_str).unwrap_or("A");
                format!("{label}) {option}")
            })
            .collect::<Vec<_>>()
    } else {
        options.to_vec()
    };
    let mut content = Vec::with_capacity(parts.len() + 2);
    content.push(text_part(format!("[kind] {kind}\n[state] ")));
    content.extend_from_slice(parts);
    content.push(text_part(format!(
        "\n[question] {question}\n[options]\n{}\n[decision]:",
        lines.join("\n")
    )));
    content
}

/// Concatenate text parts exactly as the reference does for `/v1/completions`.
pub fn content_text(content: &[Value]) -> String {
    let mut prompt = String::new();
    for part in content {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            prompt.push_str(text);
        }
    }
    prompt
}

/// Convenience for tests and tooling: the exact text prompt for a string state.
pub fn text_prompt(kind: &str, state: &str, question: &str, options: &[String]) -> String {
    let content = build_content(
        kind,
        &[text_part(state.to_owned())],
        question,
        options,
        &label_names(),
    );
    content_text(&content)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        LABEL_CONTEXT, build_content, content_text, derive_label_ids_with, image_count,
        label_names, parts, text_prompt, validate_label_parity,
    };
    use crate::head::DecisionHead;

    #[test]
    fn exact_native_prompt_for_choice() {
        let options = vec!["red".to_owned(), "green".to_owned(), "blue".to_owned()];
        let prompt = text_prompt("choice", "the sky is", "pick a color", &options);
        assert_eq!(
            prompt,
            "[kind] choice\n[state] the sky is\n[question] pick a color\n[options]\nA) red\nB) green\nC) blue\n[decision]:"
        );
    }

    #[test]
    fn exact_prompt_for_noul_and_score() {
        assert_eq!(
            text_prompt(
                "noul",
                "x",
                "is it true?",
                &["false".to_owned(), "true".to_owned()]
            ),
            "[kind] noul\n[state] x\n[question] is it true?\n[options]\nfalse\ntrue\n[decision]:"
        );
        let score: Vec<String> = (0..6).map(|value| value.to_string()).collect();
        assert_eq!(
            text_prompt("score", "x", "rate it", &score),
            "[kind] score\n[state] x\n[question] rate it\n[options]\n0\n1\n2\n3\n4\n5\n[decision]:"
        );
    }

    #[test]
    fn dict_state_keeps_python_order_and_spacing() {
        let state = json!({"z": 1, "a": "日本語"});
        let (parts, has_image) = parts(&state);
        assert!(!has_image);
        assert_eq!(parts[0]["text"], r#"{"z": 1, "a": "日本語"}"#);
    }

    #[test]
    fn list_state_mixes_text_and_images_in_place() {
        let state = json!([
            "before",
            {"image": "data:image/png;base64,AAAA"},
            {"type": "text", "text": "after"},
            {"type": "image_url", "image_url": {"url": "https://example.test/x.png"}},
            7
        ]);
        let (parts, has_image) = parts(&state);
        assert!(has_image);
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0]["text"], "before");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(parts[2]["text"], "after");
        assert_eq!(parts[3]["type"], "image_url");
        assert_eq!(parts[3]["image_url"]["url"], "https://example.test/x.png");
        assert_eq!(parts[4]["text"], "7");
    }

    #[test]
    fn image_content_splices_parts_between_prefix_and_suffix() {
        let state = json!(["look", {"image": "data:image/png;base64,AAAA"}]);
        let (parts, has_image) = parts(&state);
        assert!(has_image);
        let content = build_content(
            "noul",
            &parts,
            "is it a cat?",
            &["false".to_owned(), "true".to_owned()],
            &label_names(),
        );
        assert_eq!(content[0]["text"], "[kind] noul\n[state] ");
        assert_eq!(content[1]["text"], "look");
        assert_eq!(content[2]["type"], "image_url");
        assert_eq!(
            content[3]["text"],
            "\n[question] is it a cat?\n[options]\nfalse\ntrue\n[decision]:"
        );
        assert!(!content_text(&content).contains("<|image_pad|>"));
    }

    #[test]
    fn counts_images_and_rejects_multiple() {
        assert_eq!(image_count(&json!("text")), 0);
        assert_eq!(image_count(&json!([{"type": "text", "text": "x"}])), 0);
        let two = json!([{"image": "data:image/png;base64,AAAA"}, {"type": "image_url", "image_url": {"url": "u"}}]);
        assert_eq!(image_count(&two), 2);
    }

    #[test]
    fn label_table_matches_python_order() {
        let labels = label_names();
        assert_eq!(labels.len(), 702);
        assert_eq!(&labels[..4], ["A", "B", "C", "D"]);
        assert_eq!(labels[25], "Z");
        assert_eq!(labels[26], "AA");
        assert_eq!(labels[27], "AB");
        assert_eq!(labels[51], "AZ");
        assert_eq!(labels[52], "BA");
    }

    #[test]
    fn label_derivation_filters_multi_token_and_absent_labels() {
        let names = label_names();
        let mut ids = std::collections::HashMap::new();
        for (index, name) in names.iter().take(30).enumerate() {
            ids.insert(name.clone(), 100 + index as u32);
        }
        // "B" and "Z" are not single tokens in this fake vocabulary.
        ids.insert("B".to_owned(), 0);
        let encode = |text: &str| -> Vec<u32> {
            if text.contains('\n') {
                let mut all: Vec<u32> = ids.values().copied().collect();
                all.sort_unstable();
                // Hide the id for Z from every decision context.
                let z = ids.get("Z").copied().unwrap_or(0);
                all.retain(|id| *id != z);
                all
            } else if text == "B" {
                vec![1, 2]
            } else {
                ids.get(text)
                    .copied()
                    .map(|id| vec![id])
                    .unwrap_or_default()
            }
        };
        let derived = derive_label_ids_with(encode, LABEL_CONTEXT, 256);
        assert_eq!(derived.len(), 28);
        assert!(!derived.contains(&ids["B"]));
        assert!(!derived.contains(&ids["Z"]));
        assert_eq!(derived[..3], [ids["A"], ids["C"], ids["D"]]);
    }

    #[test]
    fn label_derivation_stops_at_limit() {
        let encode = |text: &str| -> Vec<u32> {
            if text.contains('\n') {
                (0..702).collect()
            } else {
                vec![
                    text.bytes()
                        .fold(0u32, |acc, byte| acc.wrapping_add(byte as u32)),
                ]
            }
        };
        let derived = derive_label_ids_with(encode, LABEL_CONTEXT, 256);
        assert_eq!(derived.len(), 256);
    }

    #[test]
    fn label_parity_detects_mismatch() {
        let verbalizers: Vec<u32> = (0..24).map(|index| 32 + index).collect();
        let good: Vec<u32> = verbalizers[8..24].to_vec();
        let head = DecisionHead {
            bias: vec![0.0; 24],
            verbalizer_ids: verbalizers,
            noul: [0, 2],
            score: [2, 8],
            choice: [8, 24],
            template_version: None,
        };
        assert!(validate_label_parity(&good, &head).is_ok());
        let mut bad = good;
        bad[0] = 999;
        assert!(validate_label_parity(&bad, &head).is_err());
    }
}
