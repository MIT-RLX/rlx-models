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

//! Expected HF tensor shapes derived from [`InklingTextConfig`].
//!
//! Used to validate safetensors **headers** (bytes, not full shard payloads).
//! Vision HMLP channel schedules are planned dynamically upstream — those
//! keys are listed for presence probes but are **not** shape-checked here.

use crate::config::{InklingConfig, InklingTextConfig, MlpLayerType};
use std::collections::HashMap;

/// Text-trunk HF shapes (strict). Multimodal towers are omitted — HMLP / dMel
/// packing needs the encoder plan, not just `config.json` scalars.
pub fn expected_hf_shapes(cfg: &InklingConfig) -> HashMap<String, Vec<usize>> {
    let mut m = HashMap::new();
    text_shapes(&cfg.text, &mut m);
    m
}

/// Keys we still want to *see* in a remote probe (any shape).
pub fn multimodal_presence_keys() -> &'static [&'static str] {
    &[
        "model.visual.layers.linear_0.weight",
        "model.audio.encoder.weight",
    ]
}

fn text_shapes(cfg: &InklingTextConfig, m: &mut HashMap<String, Vec<usize>>) {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let k = cfg.conv_kernel_size;

    m.insert("model.llm.embed.weight".into(), vec![v, h]);
    m.insert("model.llm.embed_norm.weight".into(), vec![h]);
    m.insert("model.llm.norm.weight".into(), vec![h]);
    m.insert("model.llm.unembed.weight".into(), vec![v, h]);

    for layer in 0..cfg.num_hidden_layers {
        let (n_h, n_kv, hd) = cfg.attn_heads(layer);
        let rel_ext = cfg.rel_extent_for_layer(layer);
        let q_dim = n_h * hd;
        let kv_dim = n_kv * hd;
        let r_dim = n_h * cfg.d_rel;
        let p = format!("model.llm.layers.{layer}");

        m.insert(format!("{p}.attn_norm.weight"), vec![h]);
        m.insert(format!("{p}.mlp_norm.weight"), vec![h]);
        // Conv1d depthwise: stored as [C, 1, K] in the HF / PT packing.
        m.insert(format!("{p}.attn_sconv.weight"), vec![h, 1, k]);
        m.insert(format!("{p}.mlp_sconv.weight"), vec![h, 1, k]);
        m.insert(format!("{p}.attn.q_norm.weight"), vec![hd]);
        m.insert(format!("{p}.attn.k_norm.weight"), vec![hd]);
        m.insert(format!("{p}.attn.k_sconv.weight"), vec![kv_dim, 1, k]);
        m.insert(format!("{p}.attn.v_sconv.weight"), vec![kv_dim, 1, k]);
        m.insert(
            format!("{p}.attn.rel_logits_proj.proj"),
            vec![cfg.d_rel, rel_ext],
        );
        m.insert(format!("{p}.attn.wq_du.weight"), vec![q_dim, h]);
        m.insert(format!("{p}.attn.wk_dv.weight"), vec![kv_dim, h]);
        m.insert(format!("{p}.attn.wv_dv.weight"), vec![kv_dim, h]);
        m.insert(format!("{p}.attn.wr_du.weight"), vec![r_dim, h]);
        m.insert(format!("{p}.attn.wo_ud.weight"), vec![h, q_dim]);

        match cfg
            .mlp_layer_types
            .get(layer)
            .copied()
            .unwrap_or(MlpLayerType::Sparse)
        {
            MlpLayerType::Dense => {
                let inter = cfg.dense_intermediate_size;
                m.insert(format!("{p}.mlp.global_scale"), vec![1]);
                m.insert(format!("{p}.mlp.w13_dn.weight"), vec![2 * inter, h]);
                m.insert(format!("{p}.mlp.w2_md.weight"), vec![h, inter]);
            }
            MlpLayerType::Sparse => {
                let inter = cfg.moe_intermediate_size;
                let ne = cfg.n_routed_experts;
                let ns = cfg.n_shared_experts;
                m.insert(
                    format!("{p}.mlp.experts.w13_weight"),
                    vec![ne, 2 * inter, h],
                );
                m.insert(format!("{p}.mlp.experts.w2_weight"), vec![ne, h, inter]);
                m.insert(format!("{p}.mlp.gate.bias"), vec![ne]);
                m.insert(format!("{p}.mlp.gate.global_scale"), vec![1]);
                m.insert(format!("{p}.mlp.gate.weight"), vec![ne + ns, h]);
                m.insert(
                    format!("{p}.mlp.shared_experts.shared_w13_weight"),
                    vec![ns, 2 * inter, h],
                );
                m.insert(
                    format!("{p}.mlp.shared_experts.shared_w2_weight"),
                    vec![ns, h, inter],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::tiny_mm_cfg;

    #[test]
    fn tiny_shapes_are_consistent() {
        let cfg = tiny_mm_cfg();
        let shapes = expected_hf_shapes(&cfg);
        assert_eq!(
            shapes.get("model.llm.embed.weight").map(Vec::as_slice),
            Some([32, 16].as_slice())
        );
        assert_eq!(
            shapes
                .get("model.llm.layers.0.attn.wq_du.weight")
                .map(Vec::as_slice),
            Some([16, 16].as_slice()) // 4 heads * 4 dim
        );
        assert_eq!(
            shapes
                .get("model.llm.layers.1.mlp.experts.w13_weight")
                .map(Vec::as_slice),
            Some([4, 32, 16].as_slice()) // 4 experts, 2*16, 16
        );
        assert!(!shapes.contains_key("model.visual.layers.linear_0.weight"));
    }
}
