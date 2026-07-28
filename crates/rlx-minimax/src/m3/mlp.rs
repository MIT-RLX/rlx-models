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

//! MiniMax-M3 dense SwiGLU-OAI MLP (`mlp.{gate,up,down}_proj`), used by the
//! first `moe_layer_freq[i]==0` layers and re-used for each MoE block's shared
//! expert.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use super::ops::swiglu_oai;

/// Synthesize the two `[1]` SwiGLU-OAI constants (`alpha`, `1.0`) keyed off `tag`.
pub(crate) fn oai_consts(emit: &mut Emit<'_>, tag: &str, alpha: f32) -> (HirNodeId, HirNodeId) {
    let f = DType::F32;
    let a = emit.synth_param(
        &format!("{tag}.oai_alpha"),
        vec![alpha],
        Shape::new(&[1], f),
    );
    let o = emit.synth_param(&format!("{tag}.oai_one"), vec![1.0], Shape::new(&[1], f));
    (a, o)
}

/// Emit a dense SwiGLU-OAI MLP for `prefix` (`{prefix}.{gate,up,down}_proj`) over
/// a `[1, seq, hidden]` input; returns `[1, seq, hidden]`.
pub fn emit_dense_mlp(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    seq: usize,
    hidden: usize,
    inter: usize,
    alpha: f32,
    limit: f32,
) -> Result<HirNodeId> {
    let (alpha_node, one_node) = oai_consts(emit, prefix, alpha);
    let gate_w = emit.load_param(&format!("{prefix}.gate_proj.weight"), true)?;
    let up_w = emit.load_param(&format!("{prefix}.up_proj.weight"), true)?;
    let down_w = emit.load_param(&format!("{prefix}.down_proj.weight"), true)?;

    let mut gb = HirMut::new(emit.hir());
    let h2d = gb.reshape_(x, vec![seq as i64, hidden as i64]);
    let gate = gb.mm(h2d, gate_w);
    let up = gb.mm(h2d, up_w);
    let hx = swiglu_oai(&mut gb, gate, up, seq, inter, limit, alpha_node, one_node);
    let down = gb.mm(hx, down_w);
    Ok(gb.reshape_(down, vec![1, seq as i64, hidden as i64]))
}
