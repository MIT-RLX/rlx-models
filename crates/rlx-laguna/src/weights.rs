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

//! HF safetensors key layout for Laguna checkpoints.

use crate::config::{AttnGating, LagunaConfig};
use crate::eager::TextWeights;
use anyhow::Result;
use rlx_core::{GgufLoader, WeightLoader};

/// Canonical HF tensor names expected for a Laguna safetensors checkpoint.
pub fn expected_hf_keys(cfg: &LagunaConfig) -> Vec<String> {
    let mut keys = vec![
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
    ];
    if !cfg.tie_word_embeddings {
        keys.push("lm_head.weight".into());
    }
    for layer in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        keys.push(format!("{p}.input_layernorm.weight"));
        keys.push(format!("{p}.post_attention_layernorm.weight"));
        keys.push(format!("{p}.self_attn.q_proj.weight"));
        keys.push(format!("{p}.self_attn.k_proj.weight"));
        keys.push(format!("{p}.self_attn.v_proj.weight"));
        keys.push(format!("{p}.self_attn.o_proj.weight"));
        keys.push(format!("{p}.self_attn.q_norm.weight"));
        keys.push(format!("{p}.self_attn.k_norm.weight"));
        if cfg.gating != AttnGating::Off {
            keys.push(format!("{p}.self_attn.g_proj.weight"));
        }
        if cfg.is_dense_mlp(layer) {
            keys.push(format!("{p}.mlp.gate_proj.weight"));
            keys.push(format!("{p}.mlp.up_proj.weight"));
            keys.push(format!("{p}.mlp.down_proj.weight"));
        } else {
            keys.push(format!("{p}.mlp.gate.weight"));
            keys.push(format!("{p}.mlp.experts.e_score_correction_bias"));
            keys.push(format!("{p}.mlp.shared_expert.gate_proj.weight"));
            keys.push(format!("{p}.mlp.shared_expert.up_proj.weight"));
            keys.push(format!("{p}.mlp.shared_expert.down_proj.weight"));
            for e in 0..cfg.num_experts {
                keys.push(format!("{p}.mlp.experts.{e}.gate_proj.weight"));
                keys.push(format!("{p}.mlp.experts.{e}.up_proj.weight"));
                keys.push(format!("{p}.mlp.experts.{e}.down_proj.weight"));
            }
        }
    }
    keys
}

/// Map HF tensor name → [`crate::eager::TextWeights`] key (split / packed form).
pub fn hf_to_eager_key(name: &str) -> Option<String> {
    if name == "model.embed_tokens.weight" {
        return Some("embed".into());
    }
    if name == "model.norm.weight" {
        return Some("norm".into());
    }
    if name == "lm_head.weight" {
        return Some("unembed".into());
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (idx_s, rest) = rest.split_once('.')?;
    let layer: usize = idx_s.parse().ok()?;
    let key = match rest {
        "input_layernorm.weight" => format!("layers.{layer}.attn_norm"),
        "post_attention_layernorm.weight" => format!("layers.{layer}.ffn_norm"),
        "self_attn.q_proj.weight" => format!("layers.{layer}.wq"),
        "self_attn.k_proj.weight" => format!("layers.{layer}.wk"),
        "self_attn.v_proj.weight" => format!("layers.{layer}.wv"),
        "self_attn.o_proj.weight" => format!("layers.{layer}.wo"),
        "self_attn.g_proj.weight" => format!("layers.{layer}.wg"),
        "self_attn.q_norm.weight" => format!("layers.{layer}.q_norm"),
        "self_attn.k_norm.weight" => format!("layers.{layer}.k_norm"),
        "mlp.gate_proj.weight" => format!("layers.{layer}.gate"),
        "mlp.up_proj.weight" => format!("layers.{layer}.up"),
        "mlp.down_proj.weight" => format!("layers.{layer}.down"),
        "mlp.gate.weight" => format!("layers.{layer}.gate_weight"),
        "mlp.experts.e_score_correction_bias" | "mlp.gate.e_score_correction_bias" => {
            format!("layers.{layer}.gate_bias")
        }
        "mlp.shared_expert.gate_proj.weight" => format!("layers.{layer}.shared_gate"),
        "mlp.shared_expert.up_proj.weight" => format!("layers.{layer}.shared_up"),
        "mlp.shared_expert.down_proj.weight" => format!("layers.{layer}.shared_down"),
        // Per-expert packs need stacking into expert_*; loader handles that.
        _ if rest.starts_with("mlp.experts.") => return None,
        _ => return None,
    };
    Some(key)
}

/// Drain a Laguna GGUF into eager [`TextWeights`] via quant→F32 `take`.
///
/// Caller must have opted in ([`crate::memory::allow_f32_expand`]). MoE
/// `ffn_*_exps` packs become stacked `expert_{gate,up,down}` tensors.
pub fn load_text_weights_from_gguf_f32(loader: &mut GgufLoader) -> Result<TextWeights> {
    use crate::gguf_layout::gguf_to_eager_key;
    use anyhow::Context;

    let names: Vec<String> = loader.file().tensors.keys().cloned().collect();
    let mut tensors = std::collections::HashMap::new();
    for name in names {
        if let Some(eager) = gguf_to_eager_key(&name) {
            let (data, _shape) = loader
                .take(&name)
                .with_context(|| format!("F32 take {name} → {eager}"))?;
            tensors.insert(eager, data);
            continue;
        }
        // MoE expert packs: store as stacked [ne, …] matching eager moe_mlp.
        let rest = match name.strip_prefix("blk.") {
            Some(r) => r,
            None => continue,
        };
        let (idx_s, rest) = match rest.split_once('.') {
            Some(p) => p,
            None => continue,
        };
        let layer: usize = match idx_s.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let eager = match rest {
            "ffn_gate_exps.weight" => format!("layers.{layer}.expert_gate"),
            "ffn_up_exps.weight" => format!("layers.{layer}.expert_up"),
            "ffn_down_exps.weight" => format!("layers.{layer}.expert_down"),
            _ => continue,
        };
        let (data, _shape) = loader
            .take(&name)
            .with_context(|| format!("F32 take expert pack {name} → {eager}"))?;
        tensors.insert(eager, data);
    }
    Ok(TextWeights { tensors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::tiny_cfg;

    #[test]
    fn expected_keys_cover_dense_and_moe() {
        let cfg = tiny_cfg();
        let keys = expected_hf_keys(&cfg);
        assert!(keys.iter().any(|k| k.contains("layers.0.mlp.gate_proj")));
        assert!(keys.iter().any(|k| k.contains("layers.1.mlp.gate.weight")));
        assert!(keys.iter().any(|k| k.contains("experts.0.gate_proj")));
        assert!(keys.iter().any(|k| k.contains("g_proj")));
    }

    #[test]
    fn hf_renames() {
        assert_eq!(
            hf_to_eager_key("model.layers.1.self_attn.g_proj.weight").as_deref(),
            Some("layers.1.wg")
        );
        assert_eq!(
            hf_to_eager_key("model.layers.1.mlp.gate.weight").as_deref(),
            Some("layers.1.gate_weight")
        );
    }
}
