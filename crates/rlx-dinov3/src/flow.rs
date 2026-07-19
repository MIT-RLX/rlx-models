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

//! Native DINOv3 flow — a RoPE ViT assembled from rlx-flow stages plus two
//! tier-2 plugins (RoPE attention, biased MLP / gated MLP).
//!
//! Each block mirrors HF `DINOv3ViTLayer` (pre-norm, LayerScale):
//! ```text
//!   x = x + ls1 · attn_rope(norm1(x))
//!   x = x + ls2 · mlp(norm2(x))
//! ```
//!
//! Two things differ from DINOv2 and force custom plugins:
//!   1. **Attention** uses separate q/k/v/o projections (asymmetric bias:
//!      no key bias) and applies 2-D axial RoPE to Q/K before the softmax.
//!   2. **MLP** carries biases (and is optionally a GeGLU gate). The
//!      existing rlx-flow SwiGLU stages are bias-free / SiLU, so they don't
//!      fit.
//!
//! RoPE cos/sin are host-precomputed (see [`crate::rope`]) and published
//! as a named flow slot (`dv3_cos`/`dv3_sin`) via [`RopeTablesStage`]; the
//! attention plugin reads them by name and applies the stock NeoX `rope`
//! op to the whole sequence (prefix rows are identity — no in-graph slice).

use anyhow::Result;
use rlx_flow::blocks::{LayerScaleStage, RopeTablesStage};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, LayerStack, ModelFlow, plugin_named};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::config::DinoV3Config;
use super::preprocess::DinoV3PreprocessWeights;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

/// DINOv3 self-attention with 2-D axial RoPE.
///
/// `q = x·Wq + bq`, `k = x·Wk` (no bias), `v = x·Wv + bv`; RoPE(Q), RoPE(K)
/// via the shared `dv3_cos`/`dv3_sin` tables; scaled dot-product attention
/// (default scale `head_dim^-0.5`, all-ones encoder mask); `out = attn·Wo + bo`.
fn dinov3_attention(layer_prefix: &str, num_heads: usize, head_dim: usize) -> FlowStage {
    let lp = layer_prefix.to_string();
    plugin_named(format!("{lp}.attention"), move |emit, hidden| {
        let x = hidden.ok_or_else(|| anyhow::anyhow!("dinov3 attention requires hidden input"))?;

        let q_w = emit.load_param(&format!("{lp}.attention.q_proj.weight"), true)?;
        let q_b = emit.load_param(&format!("{lp}.attention.q_proj.bias"), false)?;
        let k_w = emit.load_param(&format!("{lp}.attention.k_proj.weight"), true)?;
        let v_w = emit.load_param(&format!("{lp}.attention.v_proj.weight"), true)?;
        let v_b = emit.load_param(&format!("{lp}.attention.v_proj.bias"), false)?;
        let o_w = emit.load_param(&format!("{lp}.attention.o_proj.weight"), true)?;
        let o_b = emit.load_param(&format!("{lp}.attention.o_proj.bias"), false)?;
        let cos = emit.named("dv3_cos")?;
        let sin = emit.named("dv3_sin")?;
        let mask = emit.named(rlx_flow::blocks::ATTN_MASK)?;

        let out_shape = x.shape().clone();
        let mut gb = HirMut::new(emit.hir());
        let q = {
            let m = gb.mm(x.hir_id(), q_w);
            gb.add(m, q_b)
        };
        let k = gb.mm(x.hir_id(), k_w);
        let v = {
            let m = gb.mm(x.hir_id(), v_w);
            gb.add(m, v_b)
        };
        // NeoX rope over the flat [B, seq, nh·dh] tensor; per-head on dh.
        let q_r = gb.rope(q, cos, sin, head_dim);
        let k_r = gb.rope(k, cos, sin, head_dim);
        let attn = gb.attention_(q_r, k_r, v, mask, num_heads, head_dim);
        let proj = gb.mm(attn, o_w);
        let out = gb.add(proj, o_b);
        Ok(Some(emit.wrap(out, out_shape)))
    })
}

/// Standard DINOv3 MLP: `down(gelu(up(x)))`, both projections biased.
fn dinov3_mlp(layer_prefix: &str, tanh_gelu: bool) -> FlowStage {
    let lp = layer_prefix.to_string();
    plugin_named(format!("{lp}.mlp"), move |emit, hidden| {
        let x = hidden.ok_or_else(|| anyhow::anyhow!("dinov3 mlp requires hidden input"))?;
        let up_w = emit.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
        let up_b = emit.load_param(&format!("{lp}.mlp.up_proj.bias"), false)?;
        let dn_w = emit.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;
        let dn_b = emit.load_param(&format!("{lp}.mlp.down_proj.bias"), false)?;

        let out_shape = x.shape().clone();
        let mut gb = HirMut::new(emit.hir());
        let up = {
            let m = gb.mm(x.hir_id(), up_w);
            gb.add(m, up_b)
        };
        let act = if tanh_gelu {
            gb.gelu_approx(up)
        } else {
            gb.gelu(up)
        };
        let dn = {
            let m = gb.mm(act, dn_w);
            gb.add(m, dn_b)
        };
        Ok(Some(emit.wrap(dn, out_shape)))
    })
}

/// Gated (GeGLU) DINOv3 MLP: `down(gelu(gate(x)) * up(x))`, all biased.
fn dinov3_gated_mlp(layer_prefix: &str, tanh_gelu: bool) -> FlowStage {
    let lp = layer_prefix.to_string();
    plugin_named(format!("{lp}.mlp"), move |emit, hidden| {
        let x = hidden.ok_or_else(|| anyhow::anyhow!("dinov3 gated mlp requires hidden input"))?;
        let gate_w = emit.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
        let gate_b = emit.load_param(&format!("{lp}.mlp.gate_proj.bias"), false)?;
        let up_w = emit.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
        let up_b = emit.load_param(&format!("{lp}.mlp.up_proj.bias"), false)?;
        let dn_w = emit.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;
        let dn_b = emit.load_param(&format!("{lp}.mlp.down_proj.bias"), false)?;

        let out_shape = x.shape().clone();
        let mut gb = HirMut::new(emit.hir());
        let gate = {
            let m = gb.mm(x.hir_id(), gate_w);
            gb.add(m, gate_b)
        };
        let up = {
            let m = gb.mm(x.hir_id(), up_w);
            gb.add(m, up_b)
        };
        let act = if tanh_gelu {
            gb.gelu_approx(gate)
        } else {
            gb.gelu(gate)
        };
        let gated = gb.mul(act, up);
        let dn = {
            let m = gb.mm(gated, dn_w);
            gb.add(m, dn_b)
        };
        Ok(Some(emit.wrap(dn, out_shape)))
    })
}

/// One DINOv3 transformer block.
fn dinov3_layer(
    layer_idx: usize,
    num_heads: usize,
    head_dim: usize,
    use_gated: bool,
    tanh_gelu: bool,
    eps: f32,
) -> FlowStage {
    let lp = format!("layer.{layer_idx}");
    let ffn = if use_gated {
        dinov3_gated_mlp(&lp, tanh_gelu)
    } else {
        dinov3_mlp(&lp, tanh_gelu)
    };
    LayerStack::named(format!("layer{layer_idx}"))
        .residual_save()
        .layer_norm(
            format!("{lp}.norm1.weight"),
            format!("{lp}.norm1.bias"),
            eps,
        )
        .stage(dinov3_attention(&lp, num_heads, head_dim))
        .stage(FlowStage::LayerScale(LayerScaleStage::new(format!(
            "{lp}.layer_scale1.lambda1"
        ))))
        .residual_add()
        .residual_save()
        .layer_norm(
            format!("{lp}.norm2.weight"),
            format!("{lp}.norm2.bias"),
            eps,
        )
        .stage(ffn)
        .stage(FlowStage::LayerScale(LayerScaleStage::new(format!(
            "{lp}.layer_scale2.lambda1"
        ))))
        .residual_add()
        .build()
}

/// Builder handle for the DINOv3 encoder graph, bound to a config + batch.
#[derive(Debug, Clone)]
pub struct DinoV3Flow<'a> {
    cfg: &'a DinoV3Config,
    batch: usize,
}

impl<'a> DinoV3Flow<'a> {
    /// Bind a config and batch size.
    pub fn new(cfg: &'a DinoV3Config, batch: usize) -> Self {
        Self { cfg, batch }
    }

    /// Consume the checkpoint weights and emit the [`DinoV3Built`] graph.
    pub fn build(self, weights: &mut WeightMap) -> Result<DinoV3Built> {
        build_dinov3_built(self.cfg, weights, self.batch)
    }
}

/// The compiled-ready encoder graph plus the host-side preprocessing weights.
pub struct DinoV3Built {
    /// The IR/flow model (final output is `"hidden"` `[batch, seq, E]`).
    pub model: BuiltModel,
    /// Patch-embed + CLS/register weights consumed by
    /// [`crate::preprocess::assemble_hidden`] before each forward.
    pub preprocess: DinoV3PreprocessWeights,
}

/// Build the DINOv3 encoder graph. Input `"hidden"` `[batch, seq, E]` is the
/// host-assembled `[CLS, reg…, patches]` tensor (no pos_embed — see
/// [`crate::preprocess::assemble_hidden`]); output `"hidden"` is the
/// post-final-norm token sequence `[batch, seq, E]`.
pub fn build_dinov3_built(
    cfg: &DinoV3Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<DinoV3Built> {
    let preprocess = super::preprocess::extract_preprocess_weights(weights, cfg)?;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let eps = cfg.layer_norm_eps as f32;
    let seq = cfg.seq_len();
    let use_gated = cfg.use_gated_mlp;
    let tanh_gelu = cfg.gelu_is_tanh();
    let f = DType::F32;

    let n_side = cfg.num_patches_side();
    let num_prefix = cfg.num_prefix_tokens();
    let (cos, sin) = crate::rope::rope_tables(n_side, n_side, head_dim, cfg.rope_theta, num_prefix);
    let half = head_dim / 2;

    let flow = ModelFlow::new("dinov3")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, h], f))
        .attn_mask_ones(batch, seq)
        .rope_tables(RopeTablesStage::param_named("dv3", seq, half, cos, sin))
        .repeat_layers(cfg.num_hidden_layers, move |i| {
            dinov3_layer(i, nh, head_dim, use_gated, tanh_gelu, eps)
        });

    let flow = if cfg.final_layer_norm_affine {
        flow.layer_norm("norm.weight", "norm.bias", eps)
    } else {
        // Match Trellis `DinoV3FeatureExtractor`: F.layer_norm without affine.
        let _ = weights.take("norm.weight")?;
        let _ = weights.take("norm.bias")?;
        flow.stage(plugin_named("final_ln_nonaffine", move |emit, hidden| {
            let x = hidden.ok_or_else(|| anyhow::anyhow!("final ln needs hidden"))?;
            let ones = emit.synth_param("norm.ones", vec![1.0f32; h], Shape::new(&[h], f));
            let zeros = emit.synth_zeros("norm.zeros", h);
            let mut gb = HirMut::new(emit.hir());
            let y = gb.ln(x.hir_id(), ones, zeros, eps);
            Ok(Some(emit.wrap(y, x.shape().clone())))
        }))
    }
    .output("hidden");

    Ok(DinoV3Built {
        model: flow.build_with(&mut WeightMapSource(weights), None)?,
        preprocess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_cfg(use_gated: bool) -> DinoV3Config {
        DinoV3Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            image_size: 32,
            patch_size: 16,
            num_channels: 3,
            hidden_act: "gelu".into(),
            layer_norm_eps: 1e-5,
            rope_theta: 100.0,
            query_bias: true,
            key_bias: false,
            value_bias: true,
            proj_bias: true,
            mlp_bias: true,
            layerscale_value: 1.0,
            use_gated_mlp: use_gated,
            num_register_tokens: 4,
            final_layer_norm_affine: true,
        }
    }

    fn synth_weights(cfg: &DinoV3Config) -> WeightMap {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let seq_patch = cfg.num_patches();
        let _ = seq_patch;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];
        t.insert(
            "embeddings.patch_embeddings.weight".into(),
            (
                z(h * cfg.patch_dim()),
                vec![h, 3, cfg.patch_size, cfg.patch_size],
            ),
        );
        t.insert("embeddings.patch_embeddings.bias".into(), (z(h), vec![h]));
        t.insert("embeddings.cls_token".into(), (z(h), vec![1, 1, h]));
        t.insert("embeddings.mask_token".into(), (z(h), vec![1, 1, h]));
        t.insert(
            "embeddings.register_tokens".into(),
            (
                z(cfg.num_register_tokens * h),
                vec![1, cfg.num_register_tokens, h],
            ),
        );
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("layer.{i}");
            t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.attention.q_proj.weight"),
                (z(h * h), vec![h, h]),
            );
            t.insert(format!("{lp}.attention.q_proj.bias"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.attention.k_proj.weight"),
                (z(h * h), vec![h, h]),
            );
            t.insert(
                format!("{lp}.attention.v_proj.weight"),
                (z(h * h), vec![h, h]),
            );
            t.insert(format!("{lp}.attention.v_proj.bias"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.attention.o_proj.weight"),
                (z(h * h), vec![h, h]),
            );
            t.insert(format!("{lp}.attention.o_proj.bias"), (z(h), vec![h]));
            t.insert(format!("{lp}.layer_scale1.lambda1"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm2.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm2.bias"), (z(h), vec![h]));
            if cfg.use_gated_mlp {
                t.insert(
                    format!("{lp}.mlp.gate_proj.weight"),
                    (z(inter * h), vec![inter, h]),
                );
                t.insert(format!("{lp}.mlp.gate_proj.bias"), (z(inter), vec![inter]));
            }
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (z(inter * h), vec![inter, h]),
            );
            t.insert(format!("{lp}.mlp.up_proj.bias"), (z(inter), vec![inter]));
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (z(h * inter), vec![h, inter]),
            );
            t.insert(format!("{lp}.mlp.down_proj.bias"), (z(h), vec![h]));
            t.insert(format!("{lp}.layer_scale2.lambda1"), (z(h), vec![h]));
        }
        t.insert("norm.weight".into(), (z(h), vec![h]));
        t.insert("norm.bias".into(), (z(h), vec![h]));
        WeightMap::from_tensors(t)
    }

    #[test]
    fn dinov3_flow_builds_standard_mlp() {
        let cfg = tiny_cfg(false);
        let mut wm = synth_weights(&cfg);
        let built = DinoV3Flow::new(&cfg, 1).build(&mut wm).unwrap();
        assert_eq!(built.model.primary_shape().rank(), 3);
        assert_eq!(wm.len(), 0, "all checkpoint tensors should be consumed");
    }

    #[test]
    fn dinov3_flow_builds_gated_mlp() {
        let cfg = tiny_cfg(true);
        let mut wm = synth_weights(&cfg);
        let built = DinoV3Flow::new(&cfg, 1).build(&mut wm).unwrap();
        assert_eq!(built.model.primary_shape().rank(), 3);
        assert_eq!(wm.len(), 0);
    }
}
