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

//! Parses EAGLE3 speculator `config.json` (the file shipped alongside
//! `model.safetensors` on RedHatAI / vLLM-speculators checkpoints).
//!
//! Schema source: vLLM-speculators `Eagle3SpeculatorConfig` (Python
//! pydantic class) — see crate-level docs for the link.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Inner draft-transformer config — a Llama-shaped 1-layer block.
///
/// We only deserialize the fields the draft graph actually needs.
/// Everything else (HF metadata, padding token, etc.) is ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Eagle3DraftTransformerConfig {
    /// Always `"llama"` for the Gemma 4 + Llama 3 family Eagle3
    /// checkpoints. We don't yet support draft families other than
    /// Llama-style RMSNorm + GQA + SwiGLU.
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rms_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

/// Top-level Eagle3 speculator config — direct port of
/// `Eagle3SpeculatorConfig` fields we depend on.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Eagle3Config {
    /// Number of draft tokens per round (mirrors b9606 `--draft-max`).
    /// Read from `speculators_config.proposal_methods[0].speculative_tokens`.
    #[serde(skip)]
    pub speculative_tokens: usize,

    /// Inner Llama-shaped 1-layer transformer config.
    pub transformer_layer_config: Eagle3DraftTransformerConfig,

    /// Draft vocab is typically 32K — much smaller than target's
    /// (262144 for Gemma 4). The d2t buffer maps draft → target.
    #[serde(default = "default_draft_vocab")]
    pub draft_vocab_size: usize,

    /// Apply `hidden_norm` to verifier hidden states before adding
    /// them to the residual.
    #[serde(default)]
    pub norm_before_residual: bool,

    /// Insert an RMSNorm before `fc` (gpt-oss-style drafts).
    /// Disabled for Gemma 4 / Llama 3 checkpoints.
    #[serde(default)]
    pub norm_before_fc: bool,

    /// Target model's hidden size if it differs from the draft's
    /// `transformer_layer_config.hidden_size`. `None` ⇒ same size.
    #[serde(default)]
    pub target_hidden_size: Option<usize>,

    /// Verifier layer indices to extract auxiliary hidden states from.
    /// `None` ⇒ derive from `num_hidden_layers` (low/mid/high split),
    /// which is what b9606 falls back to when the field is missing.
    #[serde(default)]
    pub eagle_aux_hidden_state_layer_ids: Option<Vec<usize>>,

    #[serde(default)]
    pub embed_requires_grad: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_draft_vocab() -> usize {
    32_000
}

impl Eagle3Config {
    /// Parse from a `config.json` file emitted by RedHatAI /
    /// speculators.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read eagle3 config {path:?}"))?;
        Self::from_bytes(&bytes)
    }

    /// Parse from raw JSON bytes (used by tests + checkpoint loaders).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // The on-disk schema nests `speculators_config.proposal_methods[].speculative_tokens`
        // outside `Eagle3Config`'s flat fields. Pull it out of a
        // generic Value first, then deserialize the rest in place.
        let v: serde_json::Value =
            serde_json::from_slice(bytes).context("parse eagle3 config json")?;
        let speculative_tokens = v
            .get("speculators_config")
            .and_then(|c| c.get("proposal_methods"))
            .and_then(|m| m.as_array())
            .and_then(|arr| arr.first())
            .and_then(|m| m.get("speculative_tokens"))
            .and_then(|t| t.as_u64())
            .map(|t| t as usize)
            .unwrap_or(3);

        let mut cfg: Self = serde_json::from_value(v).context("deserialize eagle3 config")?;
        cfg.speculative_tokens = speculative_tokens;
        Ok(cfg)
    }

    /// Hidden size of the verifier's residual stream.
    /// Defaults to the draft's own hidden_size when `target_hidden_size`
    /// is `None` in the on-disk config.
    pub fn target_hidden_size(&self) -> usize {
        self.target_hidden_size
            .unwrap_or(self.transformer_layer_config.hidden_size)
    }

    /// Resolve the auxiliary-layer-id list. Returns the on-disk
    /// `eagle_aux_hidden_state_layer_ids` if present; otherwise
    /// derives a low/mid/high stratification over `target_num_layers`,
    /// matching b9606's fallback (`(target/4, target/2, 3*target/4)`).
    pub fn resolve_aux_layer_ids(&self, target_num_layers: usize) -> Vec<usize> {
        if let Some(ids) = self.eagle_aux_hidden_state_layer_ids.as_ref() {
            return ids.clone();
        }
        if target_num_layers < 4 {
            return (0..target_num_layers).collect();
        }
        vec![
            target_num_layers / 4,
            target_num_layers / 2,
            (3 * target_num_layers) / 4,
        ]
    }

    pub fn draft_vocab_size(&self) -> usize {
        self.draft_vocab_size
    }

    pub fn target_vocab_size(&self) -> usize {
        self.transformer_layer_config.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim subset of the real RedHatAI/gemma-4-31B-it-speculator.eagle3
    /// `config.json` we fetched from Hugging Face. Tests the full parse
    /// path: nested `speculators_config.proposal_methods[0]`, the
    /// flat `eagle_aux_hidden_state_layer_ids`, the inner
    /// `transformer_layer_config`.
    const REDHATAI_GEMMA4_31B: &str = r#"{
        "architectures": ["Eagle3DraftModel"],
        "draft_vocab_size": 32000,
        "dtype": "bfloat16",
        "eagle_aux_hidden_state_layer_ids": [2, 30, 57],
        "embed_requires_grad": false,
        "norm_before_fc": false,
        "norm_before_residual": true,
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [{
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": 3,
                "verifier_accept_k": 1
            }],
            "verifier": {
                "architectures": [],
                "name_or_path": "google/gemma-4-31B-it"
            }
        },
        "speculators_model_type": "eagle3",
        "target_hidden_size": null,
        "tie_word_embeddings": false,
        "transformer_layer_config": {
            "attention_bias": false,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "max_position_embeddings": 262144,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 1,
            "num_key_value_heads": 16,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "rope_type": "default"
            },
            "tie_word_embeddings": false,
            "vocab_size": 262144
        }
    }"#;

    #[test]
    fn parses_redhatai_gemma4_31b_config() {
        let cfg = Eagle3Config::from_bytes(REDHATAI_GEMMA4_31B.as_bytes()).unwrap();

        assert_eq!(cfg.speculative_tokens, 3);
        assert_eq!(cfg.draft_vocab_size, 32_000);
        assert!(cfg.norm_before_residual);
        assert!(!cfg.norm_before_fc);
        assert_eq!(
            cfg.eagle_aux_hidden_state_layer_ids.as_deref(),
            Some(&[2, 30, 57][..])
        );
        assert_eq!(cfg.target_hidden_size, None);
        // target_hidden_size() falls back to draft hidden_size when null
        assert_eq!(cfg.target_hidden_size(), 5376);

        let tl = &cfg.transformer_layer_config;
        assert_eq!(tl.model_type, "llama");
        assert_eq!(tl.hidden_size, 5376);
        assert_eq!(tl.intermediate_size, 21_504);
        assert_eq!(tl.num_hidden_layers, 1);
        assert_eq!(tl.num_attention_heads, 32);
        assert_eq!(tl.num_key_value_heads, 16);
        assert_eq!(tl.head_dim, 256);
        assert_eq!(tl.vocab_size, 262_144);
        assert_eq!(
            tl.rope_parameters.as_ref().map(|r| r.rope_theta as i32),
            Some(10_000)
        );
    }

    #[test]
    fn resolve_aux_layer_ids_uses_disk_value_when_present() {
        let cfg = Eagle3Config::from_bytes(REDHATAI_GEMMA4_31B.as_bytes()).unwrap();
        // Target num_layers is irrelevant when the field is on disk.
        assert_eq!(cfg.resolve_aux_layer_ids(58), vec![2, 30, 57]);
        assert_eq!(cfg.resolve_aux_layer_ids(0), vec![2, 30, 57]);
    }

    #[test]
    fn resolve_aux_layer_ids_falls_back_to_b9606_stratification() {
        let json = r#"{
            "draft_vocab_size": 32000,
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 4096, "intermediate_size": 11008,
                "num_hidden_layers": 1, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 128256
            }
        }"#;
        let cfg = Eagle3Config::from_bytes(json.as_bytes()).unwrap();
        assert!(cfg.eagle_aux_hidden_state_layer_ids.is_none());
        // For target=32 layers: (32/4, 32/2, 32*3/4) = (8, 16, 24)
        assert_eq!(cfg.resolve_aux_layer_ids(32), vec![8, 16, 24]);
        // For target=2 layers, we have <4 layers — return all of them
        assert_eq!(cfg.resolve_aux_layer_ids(2), vec![0, 1]);
    }

    #[test]
    fn missing_speculative_tokens_defaults_to_three() {
        let json = r#"{
            "draft_vocab_size": 32000,
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 256, "intermediate_size": 1024,
                "num_hidden_layers": 1, "num_attention_heads": 8,
                "num_key_value_heads": 4, "head_dim": 32,
                "vocab_size": 1024
            }
        }"#;
        let cfg = Eagle3Config::from_bytes(json.as_bytes()).unwrap();
        assert_eq!(cfg.speculative_tokens, 3);
    }

    #[test]
    fn target_hidden_size_uses_override_when_set() {
        let json = r#"{
            "draft_vocab_size": 32000,
            "target_hidden_size": 8192,
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 4096, "intermediate_size": 11008,
                "num_hidden_layers": 1, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 128256
            }
        }"#;
        let cfg = Eagle3Config::from_bytes(json.as_bytes()).unwrap();
        assert_eq!(cfg.target_hidden_size(), 8192);
        assert_eq!(cfg.transformer_layer_config.hidden_size, 4096);
    }
}
