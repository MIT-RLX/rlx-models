// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `FlintConfig` — the HF `config.json` of `RecursiveSI/flint`.
//!
//! Mirrors `flint_sglang/configuration_flint.py`, including its
//! `__post_init__` defaults: a missing `layer_types` means every layer is full
//! attention, `use_sliding_window = false` drops the window, and missing
//! `mlp_only_layers` / `no_rope_layer_indices` are empty.
//!
//! Three fields describe the model rather than tune it, and the reference only
//! implements one value of each — `softmax_type = "learnable"` (attention
//! sinks), `router_score_function = "sigmoid"` and `decoder_sparse_step = 1`.
//! [`FlintConfig::validate`] refuses anything else instead of building a
//! different model under the same name.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

/// HF `model_type`.
pub const MODEL_TYPE: &str = "flint";
/// HF `architectures[0]`.
pub const ARCHITECTURE: &str = "FlintForCausalLM";

/// What one decoder layer is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpec {
    /// Sliding-window attention (`layer_types[i] == "sliding_attention"` and a
    /// window is configured). Otherwise full causal attention.
    pub sliding: bool,
    /// Rotary position embedding on q and k. `false` for the layers in
    /// `no_rope_layer_indices` — in the published config, every full layer.
    pub rope: bool,
    /// Dense SwiGLU MLP (`mlp_only_layers`) instead of the sparse MoE block.
    pub dense: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlintConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Dense-layer SwiGLU width.
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// Tokens a sliding layer sees, **counting the current one** — the config's
    /// own convention (SGLang subtracts one). `None` when sliding is disabled.
    pub sliding_window: Option<usize>,
    pub layer_types: Vec<String>,
    pub no_rope_layer_indices: Vec<usize>,
    pub mlp_only_layers: Vec<usize>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    /// `0` = no shared expert.
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
}

impl FlintConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() {
            path.join("config.json")
        } else {
            path.to_path_buf()
        };
        let text =
            std::fs::read_to_string(&file).with_context(|| format!("flint: reading {file:?}"))?;
        Self::from_json(&text).with_context(|| format!("flint: parsing {file:?}"))
    }

    pub fn from_json(text: &str) -> Result<Self> {
        Self::from_value(&serde_json::from_str(text)?)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or("");
        if model_type != MODEL_TYPE {
            bail!("flint: model_type is `{model_type}`, expected `{MODEL_TYPE}`");
        }
        let req = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .with_context(|| format!("flint: config.json has no integer `{k}`"))
        };
        let opt = |k: &str, d: usize| v.get(k).and_then(Value::as_u64).map_or(d, |n| n as usize);
        let s = |k: &str, d: &str| -> String {
            v.get(k).and_then(Value::as_str).unwrap_or(d).to_string()
        };
        let usizes = |k: &str| -> Vec<usize> {
            v.get(k)
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default()
        };
        let ids = |k: &str| -> Vec<u32> {
            match v.get(k) {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect(),
                Some(x) => x.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
                None => Vec::new(),
            }
        };

        // These three name the architecture; reject anything the reference
        // does not implement rather than silently building something else.
        let softmax_type = s("softmax_type", "learnable");
        if softmax_type != "learnable" {
            bail!(
                "flint: softmax_type `{softmax_type}` — only `learnable` (attention sinks) is implemented"
            );
        }
        let score_fn = s("router_score_function", "sigmoid");
        if score_fn != "sigmoid" {
            bail!("flint: router_score_function `{score_fn}` — only `sigmoid` is implemented");
        }
        let sparse_step = opt("decoder_sparse_step", 1);
        if sparse_step != 1 {
            bail!("flint: decoder_sparse_step = {sparse_step} — only 1 is implemented");
        }

        let num_hidden_layers = req("num_hidden_layers")?;
        let hidden_size = req("hidden_size")?;
        let num_attention_heads = req("num_attention_heads")?;
        let use_sliding = v
            .get("use_sliding_window")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let sliding_window = if use_sliding {
            v.get("sliding_window")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .filter(|&n| n > 0)
        } else {
            None
        };
        let layer_types = match v.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|x| x.as_str().unwrap_or("").to_string())
                .collect(),
            None => vec!["full_attention".to_string(); num_hidden_layers],
        };
        // HF moved `rope_theta` under `rope_parameters`; accept both.
        let rope_theta = v
            .get("rope_theta")
            .and_then(Value::as_f64)
            .or_else(|| {
                v.get("rope_parameters")
                    .and_then(|r| r.get("rope_theta"))
                    .and_then(Value::as_f64)
            })
            .unwrap_or(10000.0);

        let cfg = FlintConfig {
            vocab_size: req("vocab_size")?,
            hidden_size,
            intermediate_size: req("intermediate_size")?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: opt("num_key_value_heads", num_attention_heads),
            head_dim: opt("head_dim", hidden_size / num_attention_heads.max(1)),
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            rope_theta,
            sliding_window,
            layer_types,
            no_rope_layer_indices: usizes("no_rope_layer_indices"),
            mlp_only_layers: usizes("mlp_only_layers"),
            num_experts: req("num_experts")?,
            num_experts_per_tok: req("num_experts_per_tok")?,
            moe_intermediate_size: req("moe_intermediate_size")?,
            shared_expert_intermediate_size: opt("shared_expert_intermediate_size", 0),
            norm_topk_prob: v
                .get("norm_topk_prob")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            max_position_embeddings: opt("max_position_embeddings", 262144),
            bos_token_id: ids("bos_token_id").first().copied(),
            eos_token_ids: ids("eos_token_id"),
            pad_token_id: ids("pad_token_id").first().copied(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Structural checks: shapes divide, per-layer lists are in range.
    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "flint: layer_types has {} entries but num_hidden_layers is {}",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        for (i, t) in self.layer_types.iter().enumerate() {
            if t != "full_attention" && t != "sliding_attention" {
                bail!(
                    "flint: layer_types[{i}] = `{t}` is neither full_attention nor sliding_attention"
                );
            }
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            bail!(
                "flint: {} query heads do not divide into {} key/value heads",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if !self.head_dim.is_multiple_of(2) {
            bail!("flint: head_dim {} must be even for RoPE", self.head_dim);
        }
        for (name, list) in [
            ("no_rope_layer_indices", &self.no_rope_layer_indices),
            ("mlp_only_layers", &self.mlp_only_layers),
        ] {
            if let Some(&i) = list.iter().find(|&&i| i >= self.num_hidden_layers) {
                bail!(
                    "flint: {name} names layer {i}, past num_hidden_layers {}",
                    self.num_hidden_layers
                );
            }
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            bail!(
                "flint: num_experts_per_tok {} must be in 1..={}",
                self.num_experts_per_tok,
                self.num_experts
            );
        }
        if !self.norm_topk_prob && self.num_experts_per_tok > 1 {
            // The reference router always renormalizes (`renormalize=False`
            // is passed to TopK *because* the routing function does it).
            bail!("flint: norm_topk_prob = false is not what the reference router computes");
        }
        Ok(())
    }

    /// The layer's attention and FFN kind.
    pub fn layer(&self, i: usize) -> LayerSpec {
        LayerSpec {
            sliding: self.sliding_window.is_some() && self.layer_types[i] == "sliding_attention",
            rope: !self.no_rope_layer_indices.contains(&i),
            dense: self.mlp_only_layers.contains(&i),
        }
    }

    /// `num_attention_heads · head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// `num_key_value_heads · head_dim` — one cached key (or value) row.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Previous tokens a sliding layer still sees — the cache a sliding layer
    /// needs. `sliding_window - 1` (the current token is the other one).
    pub fn sliding_history(&self) -> Option<usize> {
        self.sliding_window.map(|w| w.saturating_sub(1))
    }

    /// Indices of the sparse (MoE) layers.
    pub fn moe_layers(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_hidden_layers).filter(|&i| !self.layer(i).dense)
    }

    /// Keep only the first `n` layers — for running a range-fetched subset of
    /// the checkpoint. Per-layer lists are cut to match.
    pub fn truncated(&self, n: usize) -> Result<Self> {
        if n == 0 || n > self.num_hidden_layers {
            bail!(
                "flint: cannot truncate {} layers to {n}",
                self.num_hidden_layers
            );
        }
        let mut c = self.clone();
        c.num_hidden_layers = n;
        c.layer_types.truncate(n);
        c.no_rope_layer_indices.retain(|&i| i < n);
        c.mlp_only_layers.retain(|&i| i < n);
        Ok(c)
    }

    /// Whether `token` ends generation (any of `eos_token_id`).
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token_ids.contains(&token)
    }
}
