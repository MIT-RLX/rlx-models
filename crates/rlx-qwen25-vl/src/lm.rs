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

//! Qwen2.5 dense LM config from GGUF (reuses `rlx-qwen3` Qwen2 path).

use crate::config::{Qwen25VlLmConfig, mrope_sections_from_gguf};
use anyhow::{Context, Result, bail, ensure};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_qwen3::Qwen3Config;
use std::path::Path;

pub fn load_lm_config_from_gguf(path: &Path) -> Result<(Qwen25VlLmConfig, GgufFile)> {
    let raw = GgufFile::from_path(path).with_context(|| format!("opening GGUF {path:?}"))?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("qwen2");
    ensure!(
        crate::ACCEPTED_LM_ARCHES.contains(&arch),
        "{path:?}: LM arch `{arch}` is not Qwen2/2.5-VL (expected {:?})",
        crate::ACCEPTED_LM_ARCHES
    );
    let mut lm = qwen25_vl_lm_from_gguf(&raw)?;
    if let Some(t) = raw.tensors.get("token_embd.weight") {
        let mut shape = t.shape.clone();
        shape.reverse();
        if shape.len() == 2 {
            let vocab = if shape[1] == lm.hidden_size {
                shape[0]
            } else if shape[0] == lm.hidden_size {
                shape[1]
            } else {
                lm.vocab_size
            };
            lm.vocab_size = vocab;
        }
    }
    let mrope_sections = mrope_sections_from_gguf(&raw);
    let rope_dim_count = lm.head_dim;
    Ok((
        Qwen25VlLmConfig {
            lm,
            mrope_sections,
            rope_dim_count,
        },
        raw,
    ))
}

/// Parse Qwen2 / 2.5 dense LM metadata into [`Qwen3Config`] (QK-norm off, biases on).
pub fn qwen25_vl_lm_from_gguf(raw: &GgufFile) -> Result<Qwen3Config> {
    let arch_prefix = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("qwen2");
    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("qwen3.")?;
            if arch_prefix == "qwen3" {
                None
            } else {
                raw.metadata.get(&format!("{arch_prefix}.{suffix}"))
            }
        })
    };
    let get_u32 = |k: &str| -> Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };

    let hidden_size = get_u32("qwen3.embedding_length")? as usize;
    let num_attention_heads = get_u32("qwen3.attention.head_count")? as usize;
    let head_dim_default = if num_attention_heads > 0 {
        hidden_size / num_attention_heads
    } else {
        128
    };

    Ok(Qwen3Config {
        vocab_size: get_u32("qwen3.vocab_size").unwrap_or(151_936) as usize,
        hidden_size,
        intermediate_size: get_u32("qwen3.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("qwen3.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads: get_u32("qwen3.attention.head_count_kv")? as usize,
        head_dim: get_u32("qwen3.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(head_dim_default),
        attention_bias: true,
        qk_norm: false,
        max_position_embeddings: get_u32("qwen3.context_length").unwrap_or(32_768) as usize,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: get_bool("qwen3.tie_word_embeddings").unwrap_or(true),
        rope_theta: get_f32("qwen3.rope.freq_base").unwrap_or(1_000_000.0) as f64,
        rms_norm_eps: get_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
        use_sliding_window: false,
        hidden_act: "silu".into(),
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    })
}

pub fn assert_lm_gguf(path: &Path) -> Result<()> {
    let (cfg, _) = load_lm_config_from_gguf(path)?;
    if cfg.lm.num_hidden_layers == 0 {
        bail!("{path:?}: invalid layer count");
    }
    Ok(())
}
