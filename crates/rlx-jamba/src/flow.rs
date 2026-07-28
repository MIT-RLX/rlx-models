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

//! Jamba text decoder flow: interleaved Mamba-1 / attention mixers with a dense
//! SwiGLU FFN, standard pre-norm. Layer `i` is attention when
//! `i % attn_layer_period == attn_layer_offset`, else Mamba. (MoE FFN on the
//! expert layers is a follow-up; v1 uses the dense MLP everywhere.)

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::{JambaAttnDims, emit_jamba_attention};
use crate::mamba::{MambaDims, emit_mamba1_block};

/// Everything the flow needs; `layer_is_attention[i]` selects the mixer.
#[derive(Debug, Clone)]
pub struct JambaFlowDims {
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub seq: usize,
    pub ffn_inter: usize,
    pub tie_word_embeddings: bool,
    pub mamba: MambaDims,
    pub attn: JambaAttnDims,
    pub layer_is_attention: Vec<bool>,
}

fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

fn rmsnorm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{key}.weight"), false)?;
    let zb = emit.synth_param(
        &format!("{key}.zb"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

fn dense_mlp(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let gate = linear(emit, &format!("{prefix}.gate_proj"), x)?;
    let up = linear(emit, &format!("{prefix}.up_proj"), x)?;
    let swiglu = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(gate);
        gb.mul(a, up)
    };
    linear(emit, &format!("{prefix}.down_proj"), swiglu)
}

/// Build the Jamba text prefill graph for a fixed `seq`.
pub fn build_jamba_text_flow(
    d: &JambaFlowDims,
    weights: &mut WeightMap,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let hidden = d.hidden;
    let eps = d.eps;
    let seq = d.seq;

    let mut flow = ModelFlow::new("jamba")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f))
        .zero_beta_named("jamba.zero_beta.hidden", hidden)
        .token_embed();

    let hs = Shape::new(&[1, seq, hidden], f);
    for i in 0..d.layer_is_attention.len() {
        let prefix = format!("model.layers.{i}");
        let is_attn = d.layer_is_attention[i];
        let mamba = d.mamba;
        let attn = d.attn;
        let hs = hs.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a hidden input"))?
                .hir_id();
            let normed = rmsnorm(emit, &format!("{prefix}.input_layernorm"), x, hidden, eps)?;
            let mixer = if is_attn {
                emit_jamba_attention(emit, &format!("{prefix}.self_attn"), normed, attn)?
            } else {
                emit_mamba1_block(emit, &format!("{prefix}.mamba"), normed, mamba)?
            };
            let h = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, mixer)
            };
            let normed2 = rmsnorm(emit, &format!("{prefix}.pre_ff_layernorm"), h, hidden, eps)?;
            let ffn = dense_mlp(emit, &format!("{prefix}.feed_forward"), normed2)?;
            let out = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(h, ffn)
            };
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    flow = flow.final_norm(eps);
    let flow = if with_lm_head {
        flow.lm_head(d.vocab, hidden, d.tie_word_embeddings)
            .output("logits")
    } else {
        flow.output("hidden")
    };
    flow.build_with(&mut WeightMapSource(weights), None)
}
