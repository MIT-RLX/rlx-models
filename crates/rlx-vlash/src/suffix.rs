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

//! The VLASH suffix embedders (π₀ and π₀.₅), emitted in-graph.
//!
//! **π₀** (`PI0SuffixEmbedder`): suffix = `[state_token, action_tokens]`; the
//! sinusoidal time embedding is concatenated to the action embeddings and
//! MLP'd in (`action_time_mlp_{in,out}`, SiLU). Returns `cond = None`.
//! ```text
//!   state_tok = state_proj(state)                                   # [B,1,D]
//!   a  = action_in_proj(x_t)                                        # [B,C,D]
//!   at = action_time_mlp_out(silu(action_time_mlp_in([a, time])))   # [B,C,D]
//!   suffix = concat(state_tok, at)                                  # [B,1+C,D]
//! ```
//!
//! **π₀.₅** (`PI05SuffixEmbedder`): suffix = action tokens only; state + time
//! drive the adaRMS conditioning vector.
//! ```text
//!   t = silu(time_mlp_out(silu(time_mlp_in(time))))                 # [B,D]
//!   cond = t (+ silu(state_mlp_out(silu(state_mlp_in(state_proj(state))))))
//!   suffix = action_in_proj(x_t)                                    # [B,C,D]
//! ```
//! `time` here is the host-computed sinusoidal embedding (see
//! [`crate::util::sinusoidal_time_embedding`]) — `[B,C,D]` for π₀ (already
//! broadcast over the chunk) and `[B,D]` for π₀.₅.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use crate::config::{VlashConfig, VlashVariant};

/// `x @ Wᵀ + b` (HF `nn.Linear` with bias) under `{key}.{weight,bias}`.
fn linear_b(emit: &mut Emit<'_>, key: &str, x: rlx_ir::HirNodeId) -> Result<rlx_ir::HirNodeId> {
    let w = emit.load_param(&format!("{key}.weight"), true)?;
    let b = emit.load_param(&format!("{key}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let m = gb.mm(x, w);
    Ok(gb.add(m, b))
}

/// Emit the suffix embedder. Inputs are HIR node ids:
/// - `state`   `[batch, max_state_dim]`
/// - `actions` `[batch, chunk, max_action_dim]`  (the noisy `x_t`)
/// - `time`    `[batch, chunk, D]` (π₀) or `[batch, D]` (π₀.₅), D = expert hidden
///
/// Returns `(suffix [batch, S, D], cond)`, where `cond = Some([batch, D])` for
/// π₀.₅ (adaRMS) and `None` for π₀.
pub fn emit_suffix(
    emit: &mut Emit<'_>,
    cfg: &VlashConfig,
    state: rlx_ir::HirNodeId,
    actions: rlx_ir::HirNodeId,
    time: rlx_ir::HirNodeId,
) -> Result<(rlx_ir::HirNodeId, Option<rlx_ir::HirNodeId>)> {
    match cfg.variant {
        VlashVariant::Pi0 => {
            // state token
            let state_tok = linear_b(emit, "suffix.state_proj", state)?;
            let d = cfg.suffix_width() as i64;
            let state_tok = {
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(state_tok, vec![-1, 1, d]) // [B,1,D]
            };
            // action + time MLP
            let action_emb = linear_b(emit, "suffix.action_in_proj", actions)?;
            let action_time = {
                let mut gb = HirMut::new(emit.hir());
                gb.concat_(vec![action_emb, time], 2) // [B,C,2D]
            };
            let at = linear_b(emit, "suffix.action_time_mlp_in", action_time)?;
            let at = {
                let mut gb = HirMut::new(emit.hir());
                gb.silu(at)
            };
            let at = linear_b(emit, "suffix.action_time_mlp_out", at)?;
            // suffix = [state_token, action_time]
            let suffix = {
                let mut gb = HirMut::new(emit.hir());
                gb.concat_(vec![state_tok, at], 1) // [B,1+C,D]
            };
            Ok((suffix, None))
        }
        VlashVariant::Pi05 => {
            // time MLP → cond
            let t = linear_b(emit, "suffix.time_mlp_in", time)?;
            let t = {
                let mut gb = HirMut::new(emit.hir());
                gb.silu(t)
            };
            let t = linear_b(emit, "suffix.time_mlp_out", t)?;
            let mut cond = {
                let mut gb = HirMut::new(emit.hir());
                gb.silu(t)
            };
            if cfg.state_cond {
                let s = linear_b(emit, "suffix.state_proj", state)?;
                let s = linear_b(emit, "suffix.state_mlp_in", s)?;
                let s = {
                    let mut gb = HirMut::new(emit.hir());
                    gb.silu(s)
                };
                let s = linear_b(emit, "suffix.state_mlp_out", s)?;
                let s = {
                    let mut gb = HirMut::new(emit.hir());
                    gb.silu(s)
                };
                cond = {
                    let mut gb = HirMut::new(emit.hir());
                    gb.add(cond, s)
                };
            }
            // suffix = action tokens only
            let suffix = linear_b(emit, "suffix.action_in_proj", actions)?;
            Ok((suffix, Some(cond)))
        }
    }
}
