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

//! Jamba Mamba-1 mixer (`JambaMambaMixer`) as a flow subgraph:
//! ```text
//!   x, z = split(in_proj(h))                        # [.,d_inner] each
//!   x = silu( causal_conv1d(x) + conv_bias )
//!   dt, B, C = split(x_proj(x))                      # [dt_rank], [state], [state]
//!   dt, B, C = rms(dt), rms(B), rms(C)               # Jamba-specific norms
//!   dt_raw = dt_proj(dt) + dt_bias
//!   y = selective_scan(x, softplus(dt_raw), -exp(A_log), B, C) + x·D
//!   y = y · silu(z)
//!   out = out_proj(y)
//! ```

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

#[derive(Debug, Clone, Copy)]
pub struct MambaDims {
    pub hidden: usize,
    pub d_inner: usize,
    pub dt_rank: usize,
    pub state: usize,
    pub d_conv: usize,
    pub eps: f32,
    pub seq: usize,
}

fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// Emit the Mamba-1 mixer for `model.layers.{i}.mamba` (`prefix`).
pub fn emit_mamba1_block(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MambaDims,
) -> Result<HirNodeId> {
    rlx_ssm::register_ir_ops();
    let f = DType::F32;
    let s = d.seq;
    let di = d.d_inner;
    let (si, dii, sti) = (s as i64, di as i64, d.state as i64);

    // Params + constants (loaded before the HirMut scope).
    let conv_w = emit.load_param(&format!("{prefix}.conv1d.weight"), false)?;
    let conv_b = emit.load_param(&format!("{prefix}.conv1d.bias"), false)?;
    let dt_ln = emit.load_param(&format!("{prefix}.dt_layernorm.weight"), false)?;
    let b_ln = emit.load_param(&format!("{prefix}.b_layernorm.weight"), false)?;
    let c_ln = emit.load_param(&format!("{prefix}.c_layernorm.weight"), false)?;
    let dt_bias = emit.load_param(&format!("{prefix}.dt_proj.bias"), false)?;
    let a_log = emit.load_param(&format!("{prefix}.A_log"), false)?;
    let d_skip = emit.load_param(&format!("{prefix}.D"), false)?;
    let pad = emit.synth_param(
        &format!("{prefix}.cpad"),
        vec![0.0; (d.d_conv - 1) * di],
        Shape::new(&[1, d.d_conv - 1, di], f),
    );
    let one = emit.synth_param(&format!("{prefix}.one"), vec![1.0], Shape::new(&[1], f));
    let zb_dt = emit.synth_param(
        &format!("{prefix}.zbdt"),
        vec![0.0; d.dt_rank],
        Shape::new(&[d.dt_rank], f),
    );
    let zb_s = emit.synth_param(
        &format!("{prefix}.zbs"),
        vec![0.0; d.state],
        Shape::new(&[d.state], f),
    );

    // in_proj → split x / z
    let proj = linear(emit, &format!("{prefix}.in_proj"), hidden)?; // [1,s,2*d_inner]
    let (x, z) = {
        let mut gb = HirMut::new(emit.hir());
        let x = gb.narrow_(proj, 2, 0, di);
        let z = gb.narrow_(proj, 2, di, di);
        (x, z)
    };
    // causal depthwise conv1d + bias + silu
    let x = {
        let w4 = {
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(conv_w, vec![dii, 1, 1, d.d_conv as i64])
        };
        let out = Shape::new(&[1, s, di], f);
        let c = emit
            .hir()
            .depthwise_conv1d_causal(x, w4, pad, d.d_conv, out);
        let mut gb = HirMut::new(emit.hir());
        let cb = gb.add(c, conv_b);
        gb.silu(cb)
    };
    // x_proj → dt / B / C
    let xp = linear(emit, &format!("{prefix}.x_proj"), x)?; // [1,s,dt_rank+2*state]
    let (dt, bmat, cmat) = {
        let mut gb = HirMut::new(emit.hir());
        let dt = gb.narrow_(xp, 2, 0, d.dt_rank);
        let bmat = gb.narrow_(xp, 2, d.dt_rank, d.state);
        let cmat = gb.narrow_(xp, 2, d.dt_rank + d.state, d.state);
        // Jamba RMSNorms on dt/B/C (weight-only).
        let dt = gb.rms_norm(dt, dt_ln, zb_dt, d.eps);
        let bmat = gb.rms_norm(bmat, b_ln, zb_s, d.eps);
        let cmat = gb.rms_norm(cmat, c_ln, zb_s, d.eps);
        (dt, bmat, cmat)
    };
    // dt_proj (+bias) → dt_raw
    let dt_raw = linear(emit, &format!("{prefix}.dt_proj"), dt)?;
    let y = {
        let mut gb = HirMut::new(emit.hir());
        let dt_raw = gb.add(dt_raw, dt_bias); // [1,s,d_inner]
        // softplus(dt_raw) = log(1 + exp(dt_raw))
        let e = gb.exp(dt_raw);
        let onep = gb.add(e, one);
        let delta = gb.activation(Activation::Log, onep, Shape::new(&[1, s, di], f));
        // A = -exp(A_log)   [d_inner, state]
        let ea = gb.exp(a_log);
        let a = gb.neg(ea);
        let scan = gb.add_node(
            Op::SelectiveScan {
                state_size: d.state,
            },
            vec![x, delta, a, bmat, cmat],
            Shape::new(&[1, s, di], f),
        );
        let skip = gb.mul(x, d_skip);
        let y = gb.add(scan, skip);
        // gate: y · silu(z)
        let gz = gb.silu(z);
        gb.mul(y, gz)
    };
    let _ = (si, sti); // shape helpers
    linear(emit, &format!("{prefix}.out_proj"), y)
}
