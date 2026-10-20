//! Qwen3.5 text config parsing (tolerant of text_config nesting).
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<LayerType>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub attn_output_gate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

impl Config {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let v: Value = serde_json::from_str(&raw)?;
        let c = v.get("text_config").unwrap_or(&v);

        let num_hidden_layers = get_usize(c, "num_hidden_layers")?;
        let num_attention_heads = get_usize(c, "num_attention_heads")?;
        let num_key_value_heads = get_usize(c, "num_key_value_heads")
            .or_else(|_| Ok::<usize, anyhow::Error>(num_attention_heads))?;
        let hidden_size = get_usize(c, "hidden_size")?;
        let head_dim = get_usize(c, "head_dim").unwrap_or(hidden_size / num_attention_heads);
        let intermediate_size = get_usize(c, "intermediate_size")?;
        let vocab_size = get_usize(c, "vocab_size")?;
        let rms_norm_eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;

        let layer_types_v = c
            .get("layer_types")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        let mut layer_types = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            let s = layer_types_v
                .get(i)
                .and_then(|x| x.as_str())
                .unwrap_or("full_attention");
            layer_types.push(match s {
                "linear_attention" => LayerType::LinearAttention,
                _ => LayerType::FullAttention,
            });
        }

        let rope = c.get("rope_parameters").unwrap_or(&Value::Null);
        let rope_theta = rope["rope_theta"].as_f64().unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = rope["partial_rotary_factor"]
            .as_f64()
            .or_else(|| c.get("partial_rotary_factor").and_then(|x| x.as_f64()))
            .unwrap_or(1.0) as f32;

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            layer_types,
            linear_num_key_heads: c
                .get("linear_num_key_heads")
                .and_then(|x| x.as_u64())
                .unwrap_or(16) as usize,
            linear_num_value_heads: c
                .get("linear_num_value_heads")
                .and_then(|x| x.as_u64())
                .unwrap_or(48) as usize,
            linear_key_head_dim: c
                .get("linear_key_head_dim")
                .and_then(|x| x.as_u64())
                .unwrap_or(128) as usize,
            linear_value_head_dim: c
                .get("linear_value_head_dim")
                .and_then(|x| x.as_u64())
                .unwrap_or(128) as usize,
            rope_theta,
            partial_rotary_factor,
            attn_output_gate: c
                .get("attn_output_gate")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
        })
    }

    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f32 * self.partial_rotary_factor) as usize) & !1
    }

    pub fn gdn_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn gdn_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
}

fn get_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .with_context(|| format!("config missing {key}"))
}
