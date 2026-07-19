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

//! Native UNI2-h flow — a DINOv2-style ViT-H/14 assembled from rlx-flow's
//! public ViT stages, with a packed-SwiGLU FFN emitted as a tier-2 plugin.
//!
//! Each block mirrors timm's `Block` for this config:
//!
//! ```text
//!   x = x + ls1 · attn(norm1(x))
//!   x = x + ls2 · swiglu_packed(norm2(x))
//! ```
//!
//! The attention, LayerScale and LayerNorm sub-blocks are the exact same
//! reusable stages the DINOv2 encoder uses. Only the FFN differs: UNI2-h
//! uses `timm.layers.SwiGLUPacked` (single `mlp.fc1` → chunk → SiLU gate).

use anyhow::{Result, ensure};
use rlx_flow::blocks::{LayerScaleStage, VitSelfAttnStage};
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, LayerStack, ModelFlow, plugin_named};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::config::Uni2Config;
use super::preprocess::Uni2PreprocessWeights;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

/// Packed SwiGLU FFN (`timm.layers.SwiGLUPacked`, `act_layer=SiLU`,
/// `gate_last=False`).
///
/// timm packs the gate and value projections into a single `mlp.fc1`
/// `[2·inner, in]` weight and splits with `chunk(2, dim=-1)`:
///
/// ```text
///   h = fc1(x); value, gate = h.chunk(2); y = SiLU(value) * gate; out = fc2(y)
/// ```
///
/// We split `fc1` into its value/gate halves **on the host** and emit two
/// plain matmuls rather than an in-graph `narrow` of the activation: that
/// narrow-then-elementwise pattern was silently mis-lowered on the MLX and
/// wgpu backends (CPU/Metal were bit-exact either way). This mirrors the
/// cross-backend-verified [`rlx_flow::blocks::VisionSwiGluFfnStage`] (Nomic)
/// which likewise uses split (`fc11`/`fc12`) projections. `inner` is the
/// SwiGLU inner width (`mlp.fc1` output / 2 = `mlp.fc2` input).
///
/// `fc1`/`fc2` both carry biases (timm `bias=True`).
fn packed_swiglu_ffn(layer_prefix: &str, inner: usize) -> FlowStage {
    let lp = layer_prefix.to_string();
    plugin_named(format!("{lp}.mlp"), move |emit, hidden| {
        let x = hidden.ok_or_else(|| anyhow::anyhow!("uni2 swiglu ffn requires hidden input"))?;

        // Take the packed fc1 [2·inner, in] and split into value (first half)
        // and gate (second half) out-channel blocks, each transposed to
        // [in, inner] for the matmul.
        let (fc1_w, w_shape) = emit.weights.take(&format!("{lp}.mlp.fc1.weight"), false)?;
        ensure!(
            w_shape.len() == 2 && w_shape[0] == 2 * inner,
            "{lp}.mlp.fc1.weight expected [{}, in], got {w_shape:?}",
            2 * inner
        );
        let in_dim = w_shape[1];
        let (fc1_b, b_shape) = emit.weights.take(&format!("{lp}.mlp.fc1.bias"), false)?;
        ensure!(
            b_shape.len() == 1 && b_shape[0] == 2 * inner,
            "{lp}.mlp.fc1.bias expected [{}], got {b_shape:?}",
            2 * inner
        );
        let mut val_wt = vec![0f32; in_dim * inner];
        let mut gate_wt = vec![0f32; in_dim * inner];
        for o in 0..inner {
            let val_row = o * in_dim;
            let gate_row = (o + inner) * in_dim;
            for c in 0..in_dim {
                val_wt[c * inner + o] = fc1_w[val_row + c];
                gate_wt[c * inner + o] = fc1_w[gate_row + c];
            }
        }
        let val_b = fc1_b[0..inner].to_vec();
        let gate_b = fc1_b[inner..2 * inner].to_vec();

        let ws = Shape::new(&[in_dim, inner], DType::F32);
        let bs = Shape::new(&[inner], DType::F32);
        let val_w = emit.synth_param(&format!("{lp}.mlp.fc1_value.weight"), val_wt, ws.clone());
        let gate_w = emit.synth_param(&format!("{lp}.mlp.fc1_gate.weight"), gate_wt, ws);
        let val_bias = emit.synth_param(&format!("{lp}.mlp.fc1_value.bias"), val_b, bs.clone());
        let gate_bias = emit.synth_param(&format!("{lp}.mlp.fc1_gate.bias"), gate_b, bs);
        let fc2_w = emit.load_param(&format!("{lp}.mlp.fc2.weight"), true)?;
        let fc2_b = emit.load_param(&format!("{lp}.mlp.fc2.bias"), false)?;

        let out_shape = x.shape().clone();
        let mut gb = HirMut::new(emit.hir());
        let val = gb.mm(x.hir_id(), val_w);
        let val = gb.add(val, val_bias);
        let gate = gb.mm(x.hir_id(), gate_w);
        let gate = gb.add(gate, gate_bias);
        let act = gb.silu(val);
        let gated = gb.mul(act, gate);
        let fc2_mm = gb.mm(gated, fc2_w);
        let out = gb.add(fc2_mm, fc2_b);
        Ok(Some(emit.wrap(out, out_shape)))
    })
}

/// One UNI2-h transformer block (pre-norm, LayerScale, packed SwiGLU FFN).
fn uni2_layer(
    layer_idx: usize,
    hidden_size: usize,
    num_heads: usize,
    swiglu_inner: usize,
    eps: f32,
) -> FlowStage {
    let lp = format!("blocks.{layer_idx}");
    LayerStack::named(format!("layer{layer_idx}"))
        .residual_save()
        .layer_norm(
            format!("{lp}.norm1.weight"),
            format!("{lp}.norm1.bias"),
            eps,
        )
        .stage(FlowStage::VitSelfAttn(VitSelfAttnStage::dinov2(
            &lp,
            hidden_size,
            num_heads,
        )))
        .stage(FlowStage::LayerScale(LayerScaleStage::new(format!(
            "{lp}.ls1.gamma"
        ))))
        .residual_add()
        .residual_save()
        .layer_norm(
            format!("{lp}.norm2.weight"),
            format!("{lp}.norm2.bias"),
            eps,
        )
        .stage(packed_swiglu_ffn(&lp, swiglu_inner))
        .stage(FlowStage::LayerScale(LayerScaleStage::new(format!(
            "{lp}.ls2.gamma"
        ))))
        .residual_add()
        .build()
}

/// Thin builder over [`build_uni2_built`] for a `cfg` + `batch` pair.
#[derive(Debug, Clone)]
pub struct Uni2Flow<'a> {
    cfg: &'a Uni2Config,
    batch: usize,
}

impl<'a> Uni2Flow<'a> {
    /// Prepare a flow for `cfg` at the given `batch` size.
    pub fn new(cfg: &'a Uni2Config, batch: usize) -> Self {
        Self { cfg, batch }
    }

    /// Extract preprocessing weights and build the encoder graph, draining the
    /// consumed tensors from `weights`.
    pub fn build(self, weights: &mut WeightMap) -> Result<Uni2Built> {
        build_uni2_built(self.cfg, weights, self.batch)
    }
}

/// Output of [`build_uni2_built`]: the compiled-ready model plus the host-side
/// preprocessing weights needed to turn an image into the `"hidden"` input.
pub struct Uni2Built {
    /// The built encoder ([`rlx_flow::BuiltModel`]) ready for `graph_from_built`.
    pub model: BuiltModel,
    /// Patch-projection / token-assembly weights (see [`crate::preprocess`]).
    pub preprocess: Uni2PreprocessWeights,
}

/// Build the UNI2-h encoder graph. Input `"hidden"` `[batch, seq, E]` is
/// the host-assembled `[CLS, reg…, patches] + pos` tensor (see
/// [`crate::preprocess::assemble_hidden`]); output `"hidden"` is the
/// post-final-norm token sequence `[batch, seq, E]`.
pub fn build_uni2_built(
    cfg: &Uni2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<Uni2Built> {
    let preprocess = super::preprocess::extract_preprocess_weights(weights, cfg)?;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.layer_norm_eps as f32;
    let seq = cfg.seq_len();
    let inner = cfg.swiglu_inner();
    let f = DType::F32;

    let flow = ModelFlow::new("uni2")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, h], f))
        .attn_mask_ones(batch, seq)
        .repeat_layers(cfg.num_hidden_layers, move |i| {
            uni2_layer(i, h, nh, inner, eps)
        })
        .layer_norm("norm.weight", "norm.bias", eps)
        .output("hidden");

    Ok(Uni2Built {
        model: flow.build_with(&mut WeightMapSource(weights), None)?,
        preprocess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny UNI2-shaped config (real topology, small dims) for build tests.
    fn tiny_cfg() -> Uni2Config {
        Uni2Config {
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            img_size: 28,
            patch_size: 14,
            mlp_hidden_dim: 32, // inner = 16
            layer_norm_eps: 1e-6,
            num_register_tokens: 8,
        }
    }

    fn synth_weights(cfg: &Uni2Config) -> WeightMap {
        let h = cfg.hidden_size;
        let full = cfg.mlp_hidden_dim;
        let inner = cfg.swiglu_inner();
        let pd = cfg.patch_dim();
        let np = cfg.num_patches();
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];
        t.insert(
            "patch_embed.proj.weight".into(),
            (z(h * pd), vec![h, 3, cfg.patch_size, cfg.patch_size]),
        );
        t.insert("patch_embed.proj.bias".into(), (z(h), vec![h]));
        t.insert("cls_token".into(), (z(h), vec![1, 1, h]));
        t.insert(
            "reg_token".into(),
            (
                z(cfg.num_register_tokens * h),
                vec![1, cfg.num_register_tokens, h],
            ),
        );
        // no_embed_class → pos_embed spans patches only.
        t.insert("pos_embed".into(), (z(np * h), vec![1, np, h]));
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("blocks.{i}");
            t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm2.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.norm2.bias"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.attn.qkv.weight"),
                (z(3 * h * h), vec![3 * h, h]),
            );
            t.insert(format!("{lp}.attn.qkv.bias"), (z(3 * h), vec![3 * h]));
            t.insert(format!("{lp}.attn.proj.weight"), (z(h * h), vec![h, h]));
            t.insert(format!("{lp}.attn.proj.bias"), (z(h), vec![h]));
            t.insert(format!("{lp}.ls1.gamma"), (z(h), vec![h]));
            t.insert(format!("{lp}.ls2.gamma"), (z(h), vec![h]));
            t.insert(format!("{lp}.mlp.fc1.weight"), (z(full * h), vec![full, h]));
            t.insert(format!("{lp}.mlp.fc1.bias"), (z(full), vec![full]));
            t.insert(
                format!("{lp}.mlp.fc2.weight"),
                (z(h * inner), vec![h, inner]),
            );
            t.insert(format!("{lp}.mlp.fc2.bias"), (z(h), vec![h]));
        }
        t.insert("norm.weight".into(), (z(h), vec![h]));
        t.insert("norm.bias".into(), (z(h), vec![h]));
        WeightMap::from_tensors(t)
    }

    #[test]
    fn uni2_flow_builds_and_consumes_all_weights() {
        let cfg = tiny_cfg();
        let mut wm = synth_weights(&cfg);
        let built = Uni2Flow::new(&cfg, 1).build(&mut wm).unwrap();
        assert_eq!(built.model.primary_shape().rank(), 3);
        // Every checkpoint tensor should be claimed by the flow / preprocess.
        assert_eq!(
            wm.len(),
            0,
            "unconsumed weights: {:?}",
            wm.keys().collect::<Vec<_>>()
        );
        assert_eq!(built.preprocess.seq, cfg.seq_len());
        assert_eq!(
            built.preprocess.pos_embed.len(),
            cfg.num_patches() * cfg.hidden_size
        );
        assert_eq!(
            built.preprocess.register_tokens.len(),
            cfg.num_register_tokens * cfg.hidden_size
        );
    }
}
