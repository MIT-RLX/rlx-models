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

//! Checkpoint key layout for Inkling safetensors.
//!
//! HF shards use Thinking Machines packing (`wq_du`, fused `w13_dn`, …).
//! transformers 5.14+ remaps those via `conversion_mapping["inkling_mm_model"]`.
//! This module documents the same renames for RLX loaders.

use crate::config::{InklingTextConfig, MlpLayerType};

/// Rename a single HF tensor name toward the transformers / RLX canonical form.
///
/// Returns `None` when the name is left unchanged by the documented mapping
/// (callers should keep the original). Composite converters (`w13` → gate/up
/// split, interleave) are listed in [`hf_composite_conversions`] and are not
/// expressed as 1:1 renames.
pub fn rename_hf_key(name: &str) -> Option<String> {
    let mut out = name.to_string();
    let before = out.clone();

    // Tower / stem
    out = out.replacen("model.llm.layers", "model.language_model.layers", 1);
    out = out.replacen(
        "model.llm.embed_norm.weight",
        "model.language_model.embed_norm.weight",
        1,
    );
    out = out.replacen(
        "model.llm.embed.weight",
        "model.language_model.embed_tokens.weight",
        1,
    );
    out = out.replacen(
        "model.llm.norm.weight",
        "model.language_model.norm.weight",
        1,
    );
    out = out.replacen("model.llm.unembed.weight", "lm_head.weight", 1);
    out = out.replacen("model.audio.", "model.audio_tower.", 1);
    out = out.replacen("model.visual", "model.vision_tower", 1);

    // Vision / audio internals (after tower rename)
    if let Some(rest) = out.strip_prefix("model.vision_tower.layers.linear_") {
        if let Some((idx, _)) = rest.split_once('.') {
            out = format!("model.vision_tower.encoder_layers.{idx}.projection.weight");
        }
    }
    if let Some(rest) = out.strip_prefix("model.vision_tower.layers.norm_") {
        if let Some((idx, _)) = rest.split_once('.') {
            out = format!("model.vision_tower.encoder_layers.{idx}.layer_norm.weight");
        }
    }
    out = out.replacen(
        "model.audio_tower.encoder.weight",
        "model.audio_tower.embed_audio_tokens.embed_audio_tokens.weight",
        1,
    );
    out = out.replacen(
        "model.audio_tower.final_norm.weight",
        "model.audio_tower.norm.weight",
        1,
    );

    // Per-layer attention / MLP renames (substring, as in transformers)
    out = out.replace("attn.wq_du", "self_attn.q_proj");
    out = out.replace("attn.wk_dv", "self_attn.k_proj");
    out = out.replace("attn.wv_dv", "self_attn.v_proj");
    out = out.replace("attn.wr_du", "self_attn.r_proj");
    out = out.replace("attn.wo_ud", "self_attn.o_proj");
    out = out.replace(".attn.q_norm", ".self_attn.q_norm");
    out = out.replace(".attn.k_norm", ".self_attn.k_norm");
    out = out.replace(".attn.k_sconv", ".self_attn.k_sconv.conv1d");
    out = out.replace(".attn.v_sconv", ".self_attn.v_sconv.conv1d");
    out = out.replace(".attn.rel_logits_proj", ".self_attn.rel_logits_proj");
    if out.ends_with("attn_sconv.weight") {
        out = out.replace("attn_sconv.weight", "attn_sconv.conv1d.weight");
    }
    if out.ends_with("mlp_sconv.weight") {
        out = out.replace("mlp_sconv.weight", "mlp_sconv.conv1d.weight");
    }
    out = out.replace("mlp_norm", "post_attention_layernorm");
    out = out.replace("attn_norm", "input_layernorm");
    out = out.replace("mlp.gate.bias", "mlp.gate.e_score_correction_bias");
    out = out.replace("mlp.experts.w2_weight", "mlp.experts.down_proj");
    out = out.replace("shared_w2_weight", "down_proj");
    out = out.replace("mlp.w2_md.weight", "mlp.down_proj.weight");

    if out != before { Some(out) } else { None }
}

/// Fused tensors that expand into multiple targets (or need layout ops).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeConversion {
    /// `mlp.w13_dn.weight` → `gate_proj` + `up_proj` (interleave + chunk on dim 0).
    DenseW13,
    /// `mlp.experts.w13_weight` → `mlp.experts.gate_up_proj` (interleave dim 1).
    ExpertW13,
    /// `shared_w13_weight` → `gate_proj` + `up_proj` (interleave + chunk dim 1).
    SharedW13,
}

pub fn hf_composite_conversions(name: &str) -> Option<CompositeConversion> {
    if name.contains("mlp.w13_dn.weight") {
        Some(CompositeConversion::DenseW13)
    } else if name.contains("mlp.experts.w13_weight") {
        Some(CompositeConversion::ExpertW13)
    } else if name.contains("shared_w13_weight") {
        Some(CompositeConversion::SharedW13)
    } else {
        None
    }
}

/// Expected HF keys for the text trunk (no vision / audio / MTP).
pub fn expected_text_hf_keys(cfg: &InklingTextConfig) -> Vec<String> {
    let mut keys = vec![
        "model.llm.embed.weight".into(),
        "model.llm.embed_norm.weight".into(),
        "model.llm.norm.weight".into(),
        "model.llm.unembed.weight".into(),
    ];
    for layer in 0..cfg.num_hidden_layers {
        let p = format!("model.llm.layers.{layer}");
        keys.extend([
            format!("{p}.attn_norm.weight"),
            format!("{p}.mlp_norm.weight"),
            format!("{p}.attn_sconv.weight"),
            format!("{p}.mlp_sconv.weight"),
            format!("{p}.attn.q_norm.weight"),
            format!("{p}.attn.k_norm.weight"),
            format!("{p}.attn.k_sconv.weight"),
            format!("{p}.attn.v_sconv.weight"),
            format!("{p}.attn.rel_logits_proj.proj"),
            format!("{p}.attn.wq_du.weight"),
            format!("{p}.attn.wk_dv.weight"),
            format!("{p}.attn.wv_dv.weight"),
            format!("{p}.attn.wr_du.weight"),
            format!("{p}.attn.wo_ud.weight"),
        ]);
        match cfg
            .mlp_layer_types
            .get(layer)
            .copied()
            .unwrap_or(MlpLayerType::Sparse)
        {
            MlpLayerType::Dense => {
                keys.extend([
                    format!("{p}.mlp.global_scale"),
                    format!("{p}.mlp.w13_dn.weight"),
                    format!("{p}.mlp.w2_md.weight"),
                ]);
            }
            MlpLayerType::Sparse => {
                keys.extend([
                    format!("{p}.mlp.experts.w13_weight"),
                    format!("{p}.mlp.experts.w2_weight"),
                    format!("{p}.mlp.gate.bias"),
                    format!("{p}.mlp.gate.global_scale"),
                    format!("{p}.mlp.gate.weight"),
                    format!("{p}.mlp.shared_experts.shared_w13_weight"),
                    format!("{p}.mlp.shared_experts.shared_w2_weight"),
                ]);
            }
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::tiny_cfg;

    #[test]
    fn renames_core_keys() {
        assert_eq!(
            rename_hf_key("model.llm.embed.weight").as_deref(),
            Some("model.language_model.embed_tokens.weight")
        );
        assert_eq!(
            rename_hf_key("model.llm.layers.0.attn.wq_du.weight").as_deref(),
            Some("model.language_model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            rename_hf_key("model.llm.layers.2.mlp.gate.bias").as_deref(),
            Some("model.language_model.layers.2.mlp.gate.e_score_correction_bias")
        );
        assert_eq!(
            hf_composite_conversions("model.llm.layers.0.mlp.w13_dn.weight"),
            Some(CompositeConversion::DenseW13)
        );
    }

    #[test]
    fn expected_keys_cover_dense_and_moe() {
        let cfg = tiny_cfg();
        let keys = expected_text_hf_keys(&cfg);
        assert!(keys.iter().any(|k| k.contains("w13_dn")));
        assert!(keys.iter().any(|k| k.contains("experts.w13")));
        assert!(keys.iter().any(|k| k.contains("shared_w13")));
    }
}
