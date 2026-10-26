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

//! `config.json` of the two fish checkpoints.

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
}

/// S2-Pro (`model_type: fish_qwen3_omni`).
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
        ensure!(
            model_type == "fish_qwen3_omni",
            "not an S2 checkpoint (model_type {model_type:?})"
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
    }
}
