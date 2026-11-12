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

//! `config.json` of the dual-AR checkpoints: S2-Pro and Audio8.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

/// One transformer stack of S2 (slow backbone or fast decoder).
#[derive(Clone, Debug, PartialEq)]
pub struct StackConfig {
    pub dim: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub qk_norm: bool,
    /// `attention.wqkv` has a bias (the Audio8 slow stack).
    pub qkv_bias: bool,
}

/// Where a dual-AR checkpoint keeps its tensors. The two prefixes cover
/// `layers.N.…`, `norm.weight` and `embeddings.weight` of each stack, and
/// `output.weight` of the fast one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorLayout {
    pub slow: &'static str,
    pub fast: &'static str,
    pub codebook_embed: &'static str,
}

/// S2-Pro's tensor names.
pub const S2_LAYOUT: TensorLayout = TensorLayout {
    slow: "text_model.model.",
    fast: "audio_decoder.",
    codebook_embed: "audio_decoder.codebook_embeddings.weight",
};

/// Audio8's tensor names (`fast_layers.N.…`, `fast_norm.weight`, …).
pub const AUDIO8_LAYOUT: TensorLayout = TensorLayout {
    slow: "",
    fast: "fast_",
    codebook_embed: "codebook_embeddings.weight",
};

/// Repetition-aware sampling: when the main token repeats one of the last
/// `window` main tokens, a second draw at these settings replaces it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasConfig {
    pub window: usize,
    pub temperature: f32,
    pub top_p: f32,
}

/// A dual-AR checkpoint: S2-Pro (`model_type: fish_qwen3_omni`) or Audio8
/// (`arktts`).
#[derive(Clone, Debug, PartialEq)]
pub struct S2Config {
    pub vocab_size: usize,
    pub slow: StackConfig,
    pub fast: StackConfig,
    pub norm_eps: f32,
    /// Shared by both stacks (`precompute_freqs_cis(.., config.rope_base)`).
    pub rope_base: f64,
    pub tie_word_embeddings: bool,
    pub max_seq_len: usize,
    pub num_codebooks: usize,
    /// Entries per codebook in the LM's tables (the semantic band width).
    pub codebook_size: usize,
    pub semantic_begin: u32,
    pub semantic_end: u32,
    /// `<|im_end|>`: the stop token.
    pub eos: u32,
    pub layout: TensorLayout,
    /// An audio row's embedding sum is scaled by `1/√(N+1)` (S2; Audio8 keeps
    /// the plain sum).
    pub scale_codebook_sum: bool,
    pub ras: RasConfig,
}

fn int(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .with_context(|| format!("config: missing integer {key}"))
}

fn float(v: &Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("config: missing number {key}"))
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn stack(v: &Value) -> Result<StackConfig> {
    let n_head = int(v, "n_head")?;
    let dim = int(v, "dim")?;
    let n_kv = match v.get("n_local_heads").and_then(Value::as_i64) {
        Some(n) if n > 0 => n as usize,
        _ => n_head,
    };
    let head_dim = v
        .get("head_dim")
        .and_then(Value::as_u64)
        .map_or(dim / n_head, |h| h as usize);
    ensure!(
        n_head % n_kv == 0,
        "n_head {n_head} is not a multiple of n_local_heads {n_kv}"
    );
    for key in ["attention_qkv_bias", "attention_o_bias"] {
        if flag(v, key) {
            bail!("config: {key} is not supported");
        }
    }
    Ok(StackConfig {
        dim,
        n_layer: int(v, "n_layer")?,
        n_head,
        n_kv,
        head_dim,
        ffn: int(v, "intermediate_size")?,
        qk_norm: flag(v, "attention_qk_norm"),
        qkv_bias: false,
    })
}

/// One stack of Audio8's flat config: its keys carry `prefix` (`""` or
/// `fast_`), its depth is `layers_key`.
fn arktts_stack(v: &Value, prefix: &str, layers_key: &str) -> Result<StackConfig> {
    let key = |name: &str| format!("{prefix}{name}");
    let n_head = int(v, &key("n_head"))?;
    let n_kv = int(v, &key("n_local_heads"))?;
    ensure!(
        n_kv > 0 && n_head % n_kv == 0,
        "n_head {n_head} is not a multiple of n_local_heads {n_kv}"
    );
    ensure!(
        !flag(v, &key("attention_o_bias")),
        "config: {} is not supported",
        key("attention_o_bias")
    );
    Ok(StackConfig {
        dim: int(v, &key("dim"))?,
        n_layer: int(v, layers_key)?,
        n_head,
        n_kv,
        head_dim: int(v, &key("head_dim"))?,
        ffn: int(v, &key("intermediate_size"))?,
        qk_norm: flag(v, &key("attention_qk_norm")),
        qkv_bias: flag(v, &key("attention_qkv_bias")),
    })
}

impl S2Config {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        Self::from_value(&serde_json::from_slice(&raw)?)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or("");
        if model_type == "arktts" {
            return Self::from_arktts(v);
        }
        ensure!(
            model_type == "fish_qwen3_omni",
            "not a dual-AR checkpoint (model_type {model_type:?})"
        );
        let tc = v.get("text_config").context("config: text_config")?;
        let ac = v
            .get("audio_decoder_config")
            .context("config: audio_decoder_config")?;
        let slow = stack(tc)?;
        let fast = stack(ac)?;
        ensure!(
            fast.dim == slow.dim,
            "fast dim {} != slow dim {}",
            fast.dim,
            slow.dim
        );
        let semantic_begin = int(v, "semantic_start_token_id")? as u32;
        let semantic_end = int(v, "semantic_end_token_id")? as u32;
        let codebook_size = int(ac, "vocab_size")?;
        ensure!(
            (semantic_end - semantic_begin + 1) as usize == codebook_size,
            "semantic band {semantic_begin}..={semantic_end} is not {codebook_size} wide"
        );
        Ok(Self {
            vocab_size: int(tc, "vocab_size")?,
            slow,
            fast,
            norm_eps: float(tc, "norm_eps")? as f32,
            rope_base: float(tc, "rope_base")?,
            tie_word_embeddings: tc
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            max_seq_len: int(tc, "max_seq_len")?,
            num_codebooks: int(ac, "num_codebooks")?,
            codebook_size,
            semantic_begin,
            semantic_end,
            eos: int(v, "eos_token_id")? as u32,
            layout: S2_LAYOUT,
            scale_codebook_sum: true,
            ras: RasConfig {
                window: 10,
                temperature: 1.0,
                top_p: 0.9,
            },
        })
    }

    /// Audio8's flat `config.json`. The reference's `_embed` never scales the
    /// codebook sum, so `scale_codebook_embeddings` is not read.
    fn from_arktts(v: &Value) -> Result<Self> {
        let slow = arktts_stack(v, "", "n_layer")?;
        let fast = arktts_stack(v, "fast_", "n_fast_layer")?;
        // `fast_project_in` is a linear only when the widths differ, and the
        // checkpoint has none.
        ensure!(
            fast.dim == slow.dim,
            "fast dim {} != slow dim {}",
            fast.dim,
            slow.dim
        );
        // The fast stack reads the final-norm hidden state (`SlowState::hidden`).
        ensure!(
            flag(v, "norm_fastlayer_input"),
            "config: norm_fastlayer_input = false is not supported"
        );
        let semantic_begin = int(v, "semantic_begin_id")? as u32;
        let semantic_end = int(v, "semantic_end_id")? as u32;
        let codebook_size = int(v, "codebook_size")?;
        ensure!(
            (semantic_end - semantic_begin + 1) as usize == codebook_size,
            "semantic band {semantic_begin}..={semantic_end} is not {codebook_size} wide"
        );
        Ok(Self {
            vocab_size: int(v, "vocab_size")?,
            slow,
            fast,
            norm_eps: float(v, "norm_eps")? as f32,
            rope_base: float(v, "rope_base")?,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            max_seq_len: int(v, "max_seq_len")?,
            num_codebooks: int(v, "num_codebooks")?,
            codebook_size,
            semantic_begin,
            semantic_end,
            eos: int(v, "eos_token_id")? as u32,
            layout: AUDIO8_LAYOUT,
            scale_codebook_sum: false,
            ras: RasConfig {
                window: int(v, "ras_window_size")?,
                temperature: float(v, "ras_temperature")? as f32,
                top_p: float(v, "ras_top_p")? as f32,
            },
        })
    }

    pub fn is_semantic(&self, token: u32) -> bool {
        (self.semantic_begin..=self.semantic_end).contains(&token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S2: &str = r#"{
      "model_type": "fish_qwen3_omni", "eos_token_id": 151645,
      "semantic_start_token_id": 151678, "semantic_end_token_id": 155773,
      "text_config": {"dim": 2560, "n_layer": 36, "n_head": 32, "n_local_heads": 8,
        "head_dim": 128, "intermediate_size": 9728, "norm_eps": 1e-6, "rope_base": 1000000,
        "attention_qk_norm": true, "tie_word_embeddings": true, "max_seq_len": 32768,
        "vocab_size": 155776},
      "audio_decoder_config": {"dim": 2560, "n_layer": 4, "n_head": 32, "n_local_heads": 8,
        "head_dim": 128, "intermediate_size": 9728, "attention_qk_norm": false,
        "num_codebooks": 10, "vocab_size": 4096}
    }"#;

    #[test]
    fn parses_s2_pro() {
        let c = S2Config::from_value(&serde_json::from_str(S2).unwrap()).unwrap();
        assert_eq!(c.slow.n_layer, 36);
        assert_eq!(c.fast.n_layer, 4);
        assert!(c.slow.qk_norm && !c.fast.qk_norm);
        assert_eq!(c.codebook_size, 4096);
        assert!(c.is_semantic(151678) && !c.is_semantic(c.eos));
        assert_eq!(c.layout, S2_LAYOUT);
        assert!(c.scale_codebook_sum && !c.slow.qkv_bias);
    }

    /// The published `config.json`, verbatim.
    const AUDIO8: &str = include_str!("../tests/fixtures/audio8_config.json");

    #[test]
    fn parses_audio8() {
        let c = S2Config::from_value(&serde_json::from_str(AUDIO8).unwrap()).unwrap();
        assert_eq!((c.slow.n_layer, c.fast.n_layer), (24, 4));
        assert_eq!((c.slow.n_head, c.slow.n_kv, c.slow.head_dim), (14, 2, 64));
        assert!(c.slow.qkv_bias && !c.fast.qkv_bias);
        assert!(!c.slow.qk_norm && !c.fast.qk_norm);
        assert_eq!(c.max_seq_len, 2048);
        assert_eq!((c.codebook_size, c.num_codebooks), (4096, 10));
        assert_eq!(c.layout, AUDIO8_LAYOUT);
        assert!(!c.scale_codebook_sum);
        assert_eq!(c.ras.window, 10);
    }

    #[test]
    fn audio8_rejects_what_the_graph_cannot_express() {
        for (key, value) in [
            ("norm_fastlayer_input", false),
            ("attention_o_bias", true),
            ("fast_attention_o_bias", true),
        ] {
            let mut v: Value = serde_json::from_str(AUDIO8).unwrap();
            v[key] = Value::Bool(value);
            assert!(S2Config::from_value(&v).is_err(), "{key} = {value}");
        }
    }
}
