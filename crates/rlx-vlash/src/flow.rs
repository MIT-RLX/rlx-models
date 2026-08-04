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

//! The VLASH denoise-step graph (design A: prefix + suffix recomputed each
//! step). One graph run maps a noised action chunk `x_t` (+ prefix/state/time)
//! to the flow-matching velocity `v_t = action_out_proj(expert_hidden)`.
//!
//! Inputs (batch 1):
//! ```text
//!   prefix_emb  [1, P, 2048]     assembled image++text embeddings
//!   state       [1, 32]          normalized + padded robot state
//!   actions     [1, C, 32]       noised x_t (padded to max_action_dim)
//!   time_emb    [1, C, 1024]     (π₀) | [1, 1024] (π₀.₅)  host sinusoidal embed
//!   cos, sin    [P+S, 128]       RoPE tables for the full sequence
//!   attn_bias   [1, 8, P+S, P+S] block-causal additive mask
//! ```
//! Output: `velocity [1, C, 32]`.

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use crate::config::{VlashConfig, VlashVariant};
use crate::joint_layer::{emit_final_norm, emit_joint_layer};
use crate::suffix::emit_suffix;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

/// `x @ Wᵀ + b` under `{key}.{weight,bias}` (used for `action_out_proj`).
fn linear_b(emit: &mut Emit<'_>, key: &str, x: rlx_ir::HirNodeId) -> Result<rlx_ir::HirNodeId> {
    let w = emit.load_param(&format!("{key}.weight"), true)?;
    let b = emit.load_param(&format!("{key}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let m = gb.mm(x, w);
    Ok(gb.add(m, b))
}

/// Build the denoise-step flow for `cfg` with a prefix of `prefix_len` tokens
/// (batch 1). `wm` must already be remapped to canonical keys.
pub fn build_denoise_flow(
    cfg: &VlashConfig,
    wm: &mut WeightMap,
    prefix_len: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let batch = 1;
    let vlm_hidden = cfg.vlm.hidden;
    let exp_hidden = cfg.expert.hidden;
    let chunk = cfg.chunk_size;
    let act_dim = cfg.max_action_dim;
    let state_dim = cfg.max_state_dim;
    let head_half = cfg.head_dim() / 2;
    let heads = cfg.heads();
    let suffix_len = cfg.suffix_len();
    let seq = prefix_len + suffix_len;

    let mut flow = ModelFlow::new("vlash_denoise")
        .with_profile(CompileProfile::encoder())
        .input(
            "prefix_emb",
            Shape::new(&[batch, prefix_len, vlm_hidden], f),
        )
        .input("state", Shape::new(&[batch, state_dim], f))
        .input("actions", Shape::new(&[batch, chunk, act_dim], f));
    // time_emb shape differs by variant.
    flow = match cfg.variant {
        VlashVariant::Pi0 => flow.input("time_emb", Shape::new(&[batch, chunk, exp_hidden], f)),
        VlashVariant::Pi05 => flow.input("time_emb", Shape::new(&[batch, exp_hidden], f)),
    };
    flow = flow
        .input("cos", Shape::new(&[seq, head_half], f))
        .input("sin", Shape::new(&[seq, head_half], f))
        .input("attn_bias", Shape::new(&[batch, heads, seq, seq], f));

    let cfg = cfg.clone();
    let out_shape = Shape::new(&[batch, chunk, act_dim], f);
    flow = flow.plugin_named("vlash.model", move |emit, _prev| {
        let prefix = emit.flow_input("prefix_emb")?.hir_id();
        let state = emit.flow_input("state")?.hir_id();
        let actions = emit.flow_input("actions")?.hir_id();
        let time = emit.flow_input("time_emb")?.hir_id();
        let cos = emit.flow_input("cos")?.hir_id();
        let sin = emit.flow_input("sin")?.hir_id();
        let bias = emit.flow_input("attn_bias")?.hir_id();

        // Suffix embedding (+ adaRMS cond for π₀.₅).
        let (mut suffix, cond) = emit_suffix(emit, &cfg, state, actions, time)?;
        let mut prefix = prefix;

        // 18 joint layers.
        for idx in 0..cfg.vlm.layers {
            let (p, s) = emit_joint_layer(
                emit, &cfg, idx, prefix, suffix, cos, sin, bias, cond, batch, prefix_len,
                suffix_len,
            )?;
            prefix = p;
            suffix = s;
        }

        // Final expert norm.
        let suffix = emit_final_norm(
            emit,
            "expert.norm",
            suffix,
            exp_hidden,
            cfg.expert.rms_eps,
            batch,
            cond,
            &cfg.expert,
        )?;

        // Keep the last `chunk` (action) tokens, then project to velocity.
        let start = suffix_len - chunk; // π₀: 1 (drop state token); π₀.₅: 0
        let action_hidden = {
            let mut gb = HirMut::new(emit.hir());
            gb.narrow_(suffix, 1, start, chunk)
        };
        let velocity = linear_b(emit, "action_out_proj", action_hidden)?;
        Ok(Some(emit.wrap(velocity, out_shape.clone())))
    });

    flow.output("velocity")
        .build_with(&mut WeightMapSource(wm), None)
}
