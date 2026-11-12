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
use crate::eager::TextWeights;
use anyhow::{Result, bail};
use rlx_core::weight_map::WeightMap;

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
    if let Some(rest) = out.strip_prefix("model.vision_tower.layers.linear_")
        && let Some((idx, _)) = rest.split_once('.')
    {
        out = format!("model.vision_tower.encoder_layers.{idx}.projection.weight");
    }
    if let Some(rest) = out.strip_prefix("model.vision_tower.layers.norm_")
        && let Some((idx, _)) = rest.split_once('.')
    {
        out = format!("model.vision_tower.encoder_layers.{idx}.layer_norm.weight");
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

/// One step of the HF → eager-schema conversion for the text trunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HfStep {
    /// Same data under a new name.
    Plain { hf: String, eager: String },
    /// Fused gate/up rows stored interleaved (`gate0, up0, gate1, up1, …`);
    /// reordered to `[gate rows; up rows]` per group (routed experts).
    Reorder {
        hf: String,
        eager: String,
        groups: usize,
    },
    /// Interleaved gate/up split into two tensors (dense MLP, shared experts).
    Split {
        hf: String,
        gate: String,
        up: String,
        groups: usize,
    },
}

impl HfStep {
    pub fn hf_key(&self) -> &str {
        match self {
            Self::Plain { hf, .. } | Self::Reorder { hf, .. } | Self::Split { hf, .. } => hf,
        }
    }
}

/// Every `model.llm.*` tensor the text trunk reads and where it goes.
pub fn hf_text_plan(cfg: &InklingTextConfig) -> Vec<HfStep> {
    let plain = |hf: &str, eager: &str| HfStep::Plain {
        hf: hf.into(),
        eager: eager.into(),
    };
    let mut plan = vec![
        plain("model.llm.embed.weight", "embed"),
        plain("model.llm.embed_norm.weight", "embed_norm"),
        plain("model.llm.norm.weight", "norm"),
        plain("model.llm.unembed.weight", "unembed"),
    ];
    for l in 0..cfg.num_hidden_layers {
        let (p, e) = (format!("model.llm.layers.{l}"), format!("layers.{l}"));
        for (hf, eager) in [
            ("attn_norm.weight", "attn_norm"),
            ("mlp_norm.weight", "mlp_norm"),
            ("attn_sconv.weight", "attn_sconv"),
            ("mlp_sconv.weight", "mlp_sconv"),
            ("attn.q_norm.weight", "q_norm"),
            ("attn.k_norm.weight", "k_norm"),
            ("attn.k_sconv.weight", "k_sconv"),
            ("attn.v_sconv.weight", "v_sconv"),
            ("attn.rel_logits_proj.proj", "rel_proj"),
            ("attn.wq_du.weight", "wq"),
            ("attn.wk_dv.weight", "wk"),
            ("attn.wv_dv.weight", "wv"),
            ("attn.wr_du.weight", "wr"),
            ("attn.wo_ud.weight", "wo"),
        ] {
            plan.push(plain(&format!("{p}.{hf}"), &format!("{e}.{eager}")));
        }
        if cfg.is_dense_mlp(l) {
            plan.push(plain(
                &format!("{p}.mlp.global_scale"),
                &format!("{e}.mlp_global_scale"),
            ));
            plan.push(HfStep::Split {
                hf: format!("{p}.mlp.w13_dn.weight"),
                gate: format!("{e}.gate"),
                up: format!("{e}.up"),
                groups: 1,
            });
            plan.push(plain(
                &format!("{p}.mlp.w2_md.weight"),
                &format!("{e}.down"),
            ));
        } else {
            for (hf, eager) in [
                ("mlp.gate.weight", "gate_weight"),
                ("mlp.gate.bias", "gate_bias"),
                ("mlp.gate.global_scale", "gate_global_scale"),
                ("mlp.experts.w2_weight", "expert_w2"),
                ("mlp.shared_experts.shared_w2_weight", "shared_down"),
            ] {
                plan.push(plain(&format!("{p}.{hf}"), &format!("{e}.{eager}")));
            }
            plan.push(HfStep::Reorder {
                hf: format!("{p}.mlp.experts.w13_weight"),
                eager: format!("{e}.expert_w13"),
                groups: cfg.n_routed_experts,
            });
            plan.push(HfStep::Split {
                hf: format!("{p}.mlp.shared_experts.shared_w13_weight"),
                gate: format!("{e}.shared_gate"),
                up: format!("{e}.shared_up"),
                groups: cfg.n_shared_experts,
            });
        }
    }
    plan
}

/// `[groups, 2·r, cols]` with gate/up rows interleaved → `[groups, 2·r, cols]`
/// with all gate rows (even) first, then all up rows (odd), per group.
///
/// This is transformers' `Interleave(dim)` op for `inkling_mm_model`: split the
/// fused dim into `[n/2, 2]` and transpose.
pub fn deinterleave_rows(data: &[f32], groups: usize, rows: usize, cols: usize) -> Vec<f32> {
    assert!(rows.is_multiple_of(2) && data.len() == groups * rows * cols);
    let half = rows / 2;
    let mut out = vec![0.0; data.len()];
    for g in 0..groups {
        let base = g * rows * cols;
        for r in 0..half {
            for (which, dst_row) in [(0, r), (1, half + r)] {
                let src = base + (2 * r + which) * cols;
                let dst = base + dst_row * cols;
                out[dst..dst + cols].copy_from_slice(&data[src..src + cols]);
            }
        }
    }
    out
}

/// Convert HF-named text tensors (drained from `wm`) into the eager schema that
/// [`crate::eager`] and [`crate::graph`] read. Shapes are checked against
/// [`crate::shapes::expected_hf_shapes`] before any data moves.
pub fn text_weights_from_hf(cfg: &InklingTextConfig, wm: &mut WeightMap) -> Result<TextWeights> {
    let model_cfg = crate::config::InklingConfig {
        text: cfg.clone(),
        ..crate::config::InklingConfig::production_preset()
    };
    let shapes = crate::shapes::expected_hf_shapes(&model_cfg);
    let mut tensors = std::collections::HashMap::new();
    for step in hf_text_plan(cfg) {
        let (data, shape) = wm.take(step.hf_key())?;
        if let Some(want) = shapes.get(step.hf_key())
            && *want != shape
        {
            bail!("{}: shape {shape:?}, expected {want:?}", step.hf_key());
        }
        match step {
            HfStep::Plain { eager, .. } => {
                tensors.insert(eager, data);
            }
            HfStep::Reorder { eager, groups, .. } => {
                let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                tensors.insert(eager, deinterleave_rows(&data, groups, rows, cols));
            }
            HfStep::Split {
                gate, up, groups, ..
            } => {
                let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                let ordered = deinterleave_rows(&data, groups, rows, cols);
                let (mut g, mut u) = (Vec::new(), Vec::new());
                for chunk in ordered.chunks(rows * cols) {
                    g.extend_from_slice(&chunk[..rows / 2 * cols]);
                    u.extend_from_slice(&chunk[rows / 2 * cols..]);
                }
                tensors.insert(gate, g);
                tensors.insert(up, u);
            }
        }
    }
    Ok(TextWeights { tensors })
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

    #[test]
    fn deinterleave_puts_even_rows_first() {
        // one group, 4 rows of width 2: g0 u0 g1 u1
        let d = [1., 1., 10., 10., 2., 2., 20., 20.];
        assert_eq!(
            deinterleave_rows(&d, 1, 4, 2),
            [1., 1., 2., 2., 10., 10., 20., 20.]
        );
        // groups are independent
        let d2 = [d.as_slice(), d.as_slice()].concat();
        let out = deinterleave_rows(&d2, 2, 4, 2);
        assert_eq!(out[..8], out[8..]);
    }

    /// Re-interleave the eager tensors into HF layout, convert back, compare.
    #[test]
    fn hf_round_trip_recovers_eager_weights() {
        use crate::synth::synthetic_text_weights;
        let cfg = tiny_cfg();
        let w = synthetic_text_weights(&cfg);
        let interleave = |gate: &[f32], up: &[f32], groups: usize, half: usize, cols: usize| {
            let mut out = Vec::with_capacity(gate.len() * 2);
            for g in 0..groups {
                for r in 0..half {
                    let o = (g * half + r) * cols;
                    out.extend_from_slice(&gate[o..o + cols]);
                    out.extend_from_slice(&up[o..o + cols]);
                }
            }
            out
        };
        let (h, ne, ns) = (cfg.hidden_size, cfg.n_routed_experts, cfg.n_shared_experts);
        let mut hf = std::collections::HashMap::new();
        for step in hf_text_plan(&cfg) {
            let (key, data, shape) = match &step {
                HfStep::Plain { hf, eager } => (
                    hf.clone(),
                    w.tensors[eager].clone(),
                    vec![w.tensors[eager].len()],
                ),
                HfStep::Reorder { hf, eager, groups } => {
                    let t = &w.tensors[eager];
                    let i = cfg.moe_intermediate_size;
                    let per = i * h;
                    let (mut g, mut u) = (Vec::new(), Vec::new());
                    for e in 0..*groups {
                        g.extend_from_slice(&t[e * 2 * per..e * 2 * per + per]);
                        u.extend_from_slice(&t[e * 2 * per + per..(e + 1) * 2 * per]);
                    }
                    (hf.clone(), interleave(&g, &u, ne, i, h), vec![ne, 2 * i, h])
                }
                HfStep::Split {
                    hf,
                    gate,
                    up,
                    groups,
                } => {
                    let i = w.tensors[gate].len() / groups / h;
                    (
                        hf.clone(),
                        interleave(&w.tensors[gate], &w.tensors[up], *groups, i, h),
                        if *groups == 1 {
                            vec![2 * i, h]
                        } else {
                            vec![ns, 2 * i, h]
                        },
                    )
                }
            };
            hf.insert(key, (data, shape));
        }
        // Plain tensors were registered flat; give them the shape the converter checks.
        let shapes = crate::shapes::expected_hf_shapes(&crate::config::InklingConfig {
            text: cfg.clone(),
            ..crate::config::InklingConfig::production_preset()
        });
        for (k, (_, shape)) in hf.iter_mut() {
            if let Some(s) = shapes.get(k) {
                *shape = s.clone();
            }
        }
        let mut wm = WeightMap::from_tensors(hf);
        let back = text_weights_from_hf(&cfg, &mut wm).expect("convert");
        assert_eq!(back.tensors.len(), w.tensors.len());
        for (k, v) in &w.tensors {
            assert_eq!(&back.tensors[k], v, "{k}");
        }
    }
}
