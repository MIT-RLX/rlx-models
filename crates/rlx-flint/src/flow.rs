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

//! The decoder, as sublayer emitters and one whole-sequence flow.
//!
//! ```text
//!   embed_tokens
//!     → 49 × ( h += attn(input_layernorm(h))
//!              h += ffn(post_attention_layernorm(h)) )    dense MLP or sparse MoE
//!     → norm → lm_head
//! ```
//!
//! The sublayer emitters ([`emit_attn_sublayer`], [`emit_ffn_resident`],
//! [`emit_ffn_pre_route`]) are what both execution paths are built from:
//! [`build_flint_text_flow`] strings all of them into one resident graph, and
//! [`crate::runner`] compiles them per layer around paged experts. Sharing
//! them is what makes the runner-vs-flow equivalence test meaningful.

use anyhow::{Result, anyhow, bail};
use rlx_flow::blocks::LmHeadStage;
use rlx_flow::{BuiltModel, CompileProfile, Emit, FlowStage, ModelFlow, WeightSource};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::{AttnDims, AttnInputs, emit_attention};
use crate::common::{rms_norm, swiglu};
use crate::config::FlintConfig;
use crate::mask::{rope_tables, step_bias};
use crate::moe::{emit_moe_resident, emit_router_scores, emit_shared};

/// HF token-embedding tensor.
pub const EMBED_KEY: &str = "model.embed_tokens.weight";
/// HF final norm.
pub const NORM_KEY: &str = "model.norm.weight";
/// HF LM head (untied in the published checkpoint).
pub const LM_HEAD_KEY: &str = "lm_head.weight";

/// Per-step tensors one attention sublayer reads, by layer kind.
#[derive(Debug, Clone, Copy)]
pub struct StepTensors {
    /// `(cos, sin)`; may be `None` when the layer being built has no RoPE.
    pub rope: Option<(HirNodeId, HirNodeId)>,
    pub bias_full: HirNodeId,
    pub bias_sliding: HirNodeId,
}

/// `h + attn(input_layernorm(h))` for `layer`, returning the new residual and
/// this step's rotated key / value rows.
pub fn emit_attn_sublayer(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
    h: HirNodeId,
    dims: AttnDims,
    step: StepTensors,
    cache: Option<(HirNodeId, HirNodeId)>,
) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
    let spec = cfg.layer(layer);
    let p = format!("model.layers.{layer}");
    let x = rms_norm(
        emit,
        &format!("{p}.input_layernorm.weight"),
        h,
        cfg.hidden_size,
        cfg.rms_norm_eps,
    )?;
    let rope = match (spec.rope, step.rope) {
        (true, None) => bail!("flint: layer {layer} rotates q/k but no RoPE tables were given"),
        (true, r) => r,
        (false, _) => None,
    };
    let io = AttnInputs {
        rope,
        bias: if spec.sliding {
            step.bias_sliding
        } else {
            step.bias_full
        },
        cache,
    };
    let a = emit_attention(emit, layer, x, dims, io)?;
    let mut gb = HirMut::new(emit.hir());
    Ok((gb.add(h, a.out), a.k, a.v))
}

/// `post_attention_layernorm(h)` flattened to `[rows, hidden]`.
fn ffn_input(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
    h: HirNodeId,
) -> Result<HirNodeId> {
    let x = rms_norm(
        emit,
        &format!("model.layers.{layer}.post_attention_layernorm.weight"),
        h,
        cfg.hidden_size,
        cfg.rms_norm_eps,
    )?;
    let mut gb = HirMut::new(emit.hir());
    let rows = gb.shape(h).dim(1).unwrap_static() as i64;
    Ok(gb.reshape_(x, vec![rows, cfg.hidden_size as i64]))
}

/// `h + ffn(post_attention_layernorm(h))` with every expert resident.
pub fn emit_ffn_resident(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
    h: HirNodeId,
) -> Result<HirNodeId> {
    let x2d = ffn_input(emit, cfg, layer, h)?;
    let y = if cfg.layer(layer).dense {
        swiglu(emit, &format!("model.layers.{layer}.mlp"), x2d)?
    } else {
        emit_moe_resident(emit, cfg, layer, x2d)?
    };
    let mut gb = HirMut::new(emit.hir());
    let shape: Vec<i64> = gb
        .shape(h)
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    let y = gb.reshape_(y, shape);
    Ok(gb.add(h, y))
}

/// The half of a sparse FFN that runs before the routed experts are known.
#[derive(Debug, Clone, Copy)]
pub struct PreRoute {
    /// `[rows, hidden]` — the normed input the experts read.
    pub x2d: HirNodeId,
    /// `[rows, hidden]` — `h + shared_expert(x)`: everything but the routed sum.
    pub partial: HirNodeId,
    /// `[rows, num_experts]` — the router's sigmoid scores; the experts are
    /// picked from them on the host ([`crate::moe::select_experts`]).
    pub scores: HirNodeId,
}

/// Norm, router and shared expert for a sparse `layer`. The routed sum is
/// added later, once the fired experts are paged in.
pub fn emit_ffn_pre_route(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
    h: HirNodeId,
) -> Result<PreRoute> {
    if cfg.layer(layer).dense {
        bail!("flint: layer {layer} is dense; it has no router");
    }
    let x2d = ffn_input(emit, cfg, layer, h)?;
    let scores = emit_router_scores(emit, layer, x2d)?;
    let shared = emit_shared(emit, cfg, layer, x2d)?;
    let mut gb = HirMut::new(emit.hir());
    let rows = gb.shape(x2d).dim(0).unwrap_static() as i64;
    let h2d = gb.reshape_(h, vec![rows, cfg.hidden_size as i64]);
    let partial = match shared {
        Some(s) => gb.add(h2d, s),
        None => h2d,
    };
    Ok(PreRoute {
        x2d,
        partial,
        scores,
    })
}

/// Build the whole model as one resident graph over prompt positions
/// `0..seq`: `input_ids [1, seq]` → `logits [1, seq, vocab]`.
///
/// Every routed expert becomes a graph parameter, so this is for synthetic
/// weights, truncated subsets and the equivalence tests — the published
/// checkpoint's experts alone are 74 GB of f32-able bf16. Run that through
/// [`crate::runner::FlintRunner`], which pages them.
pub fn build_flint_text_flow(
    cfg: &FlintConfig,
    weights: &mut dyn WeightSource,
    seq: usize,
) -> Result<BuiltModel> {
    cfg.validate()?;
    if seq == 0 {
        bail!("flint: seq must be >= 1");
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let half = cfg.head_dim / 2;
    let positions: Vec<usize> = (0..seq).collect();
    let (cos, sin) = rope_tables(&positions, cfg.head_dim, cfg.rope_theta);
    let bias_full = step_bias(&[], &positions, None);
    let bias_sliding = step_bias(&[], &positions, cfg.sliding_history());

    let mut flow = ModelFlow::new("flint")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f));

    flow = flow.plugin_named("step_tensors", move |emit, prev| {
        let t = Shape::new(&[seq, half], DType::F32);
        let b = Shape::new(&[1, 1, seq, seq], DType::F32);
        let c = emit.synth_param("flint.rope.cos", cos.clone(), t.clone());
        let s = emit.synth_param("flint.rope.sin", sin.clone(), t);
        let bf = emit.synth_param("flint.bias.full", bias_full.clone(), b.clone());
        let bs = emit.synth_param("flint.bias.sliding", bias_sliding.clone(), b);
        emit.set_named("rope_cos", c);
        emit.set_named("rope_sin", s);
        emit.set_named("bias_full", bf);
        emit.set_named("bias_sliding", bs);
        Ok(prev)
    });
    flow = flow.embed(EMBED_KEY);

    for layer in 0..cfg.num_hidden_layers {
        let cfg = cfg.clone();
        let hs = Shape::new(&[1, seq, hidden], f);
        flow = flow.plugin_named(format!("layer{layer}"), move |emit, prev| {
            let h = prev
                .ok_or_else(|| anyhow!("layer{layer} needs the residual"))?
                .hir_id();
            let step = StepTensors {
                rope: Some((emit.named("rope_cos")?, emit.named("rope_sin")?)),
                bias_full: emit.named("bias_full")?,
                bias_sliding: emit.named("bias_sliding")?,
            };
            let dims = AttnDims::from_config(&cfg, seq, 0);
            let (h, _, _) = emit_attn_sublayer(emit, &cfg, layer, h, dims, step, None)?;
            let h = emit_ffn_resident(emit, &cfg, layer, h)?;
            Ok(Some(emit.wrap(h, hs.clone())))
        });
    }

    let head = if cfg.tie_word_embeddings {
        LmHeadStage::tied(cfg.vocab_size, hidden)
    } else {
        LmHeadStage::separate(LM_HEAD_KEY, cfg.vocab_size, hidden)
    };
    let eps = cfg.rms_norm_eps;
    flow.plugin_named("norm", move |emit, prev| {
        let h = prev.ok_or_else(|| anyhow!("norm needs the residual"))?;
        let shape = h.shape().clone();
        let x = rms_norm(emit, NORM_KEY, h.hir_id(), hidden, eps)?;
        Ok(Some(emit.wrap(x, shape)))
    })
    .raw_stage(FlowStage::LmHead(head))
    .output("logits")
    .build_with(weights, None)
}
