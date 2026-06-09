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

//! Backend-agnostic Mamba1 forward driver. Takes a [`MambaBackend`]
//! and a [`Mamba1Block`]'s f32 weight slices; uploads everything once
//! into the backend and returns a closure-ish runner.
//!
//! Each backend implementation only has to fulfil the trait surface in
//! [`crate::backend`] — this module assembles the actual Mamba1
//! computation graph from those primitives. That way the algorithm
//! lives in exactly one place; backends only own their kernels.

use crate::backend::MambaBackend;
use crate::block::Mamba1Block;
use anyhow::Result;

/// Per-backend resident Mamba1 weights. Built once with
/// [`Mamba1ResidentBlock::upload`], then `forward` is cheap (no
/// re-upload of weights).
pub struct Mamba1ResidentBlock<B: MambaBackend> {
    pub cfg: crate::config::Mamba1Config,
    pub in_proj_w: B::Tensor,
    pub in_proj_b: B::Tensor,
    pub conv1d_w: B::Tensor,
    pub conv1d_b: B::Tensor,
    pub x_proj_w: B::Tensor,
    pub dt_proj_w: B::Tensor,
    pub dt_proj_b: B::Tensor,
    pub a_log: B::Tensor,
    pub d_skip: B::Tensor,
    pub out_proj_w: B::Tensor,
    pub out_proj_b: B::Tensor,
    pub has_in_bias: bool,
    pub has_out_bias: bool,
}

impl<B: MambaBackend> Mamba1ResidentBlock<B> {
    /// Upload all weights from a host-side [`Mamba1Block`] into the
    /// backend. The host-side block is unchanged.
    pub fn upload(backend: &mut B, block: &Mamba1Block) -> Result<Self> {
        let has_in_bias = block.cfg.bias;
        let has_out_bias = block.cfg.bias;
        Ok(Self {
            cfg: block.cfg.clone(),
            in_proj_w: backend.upload(&block.in_proj_w)?,
            in_proj_b: backend.upload(&block.in_proj_b)?,
            conv1d_w: backend.upload(&block.conv1d_w)?,
            conv1d_b: backend.upload(&block.conv1d_b)?,
            x_proj_w: backend.upload(&block.x_proj_w)?,
            dt_proj_w: backend.upload(&block.dt_proj_w)?,
            dt_proj_b: backend.upload(&block.dt_proj_b)?,
            a_log: backend.upload(&block.a_log)?,
            d_skip: backend.upload(&block.d)?,
            out_proj_w: backend.upload(&block.out_proj_w)?,
            out_proj_b: backend.upload(&block.out_proj_b)?,
            has_in_bias,
            has_out_bias,
        })
    }
}

/// Run Mamba1 forward on the given backend. `input` is host data
/// `[batch, seq, d_model]`. Returns the host result of the same shape.
pub fn mamba1_forward<B: MambaBackend>(
    backend: &mut B,
    block: &Mamba1ResidentBlock<B>,
    input: &[f32],
    batch: usize,
    seq: usize,
) -> Result<Vec<f32>> {
    let cfg = &block.cfg;
    let m = cfg.d_model;
    let h = cfg.d_inner();
    let n = cfg.d_state;
    let dr = cfg.dt_rank();
    let k = cfg.d_conv;
    let bs = batch * seq;

    let x = backend.upload(input)?;

    // in_proj
    let mut xz = backend.alloc(bs * 2 * h)?;
    let in_bias = if block.has_in_bias {
        Some(&block.in_proj_b)
    } else {
        None
    };
    backend.sgemm_bias(&x, &block.in_proj_w, in_bias, &mut xz, bs, m, 2 * h)?;

    // The split into xs / res is implemented host-side via read-back +
    // re-upload. That's fine for the high-level driver; native backends
    // can later specialize this with a single device-side split kernel.
    let xz_host = backend.read_to_host(&xz)?;
    let mut xs_host = vec![0.0f32; bs * h];
    let mut res_host = vec![0.0f32; bs * h];
    for r in 0..bs {
        let src = &xz_host[r * 2 * h..(r + 1) * 2 * h];
        xs_host[r * h..(r + 1) * h].copy_from_slice(&src[..h]);
        res_host[r * h..(r + 1) * h].copy_from_slice(&src[h..]);
    }
    let xs = backend.upload(&xs_host)?;
    let mut res = backend.upload(&res_host)?;

    // Causal conv1d + SiLU
    let mut conv_out = backend.alloc(bs * h)?;
    backend.causal_conv1d(
        &xs,
        &block.conv1d_w,
        &block.conv1d_b,
        &mut conv_out,
        batch,
        seq,
        h,
        k,
    )?;
    backend.silu_in_place(&mut conv_out, bs * h)?;

    // x_proj: conv_out [bs, h] @ x_proj_w [h, dr + 2n]
    let dn = dr + 2 * n;
    let mut x_dbl = backend.alloc(bs * dn)?;
    backend.sgemm_bias(&conv_out, &block.x_proj_w, None, &mut x_dbl, bs, h, dn)?;

    // Split into delta_raw, b, c
    let xd_host = backend.read_to_host(&x_dbl)?;
    let mut delta_raw_host = vec![0.0f32; bs * dr];
    let mut b_host = vec![0.0f32; bs * n];
    let mut c_host = vec![0.0f32; bs * n];
    for r in 0..bs {
        let row = &xd_host[r * dn..(r + 1) * dn];
        delta_raw_host[r * dr..(r + 1) * dr].copy_from_slice(&row[..dr]);
        b_host[r * n..(r + 1) * n].copy_from_slice(&row[dr..dr + n]);
        c_host[r * n..(r + 1) * n].copy_from_slice(&row[dr + n..]);
    }
    let delta_raw = backend.upload(&delta_raw_host)?;
    let b_mat = backend.upload(&b_host)?;
    let c_mat = backend.upload(&c_host)?;

    let mut dt_pre = backend.alloc(bs * h)?;
    backend.sgemm_bias(
        &delta_raw,
        &block.dt_proj_w,
        Some(&block.dt_proj_b),
        &mut dt_pre,
        bs,
        dr,
        h,
    )?;

    // Selective scan via rlx_ssm flow (softplus inside MambaScanStage).
    let mut y = backend.alloc(bs * h)?;
    backend.selective_scan(
        &conv_out,
        &dt_pre,
        &b_mat,
        &c_mat,
        &block.a_log,
        &block.d_skip,
        &mut y,
        batch,
        seq,
        h,
        n,
    )?;

    // Gate: y *= silu(res)
    backend.silu_in_place(&mut res, bs * h)?;
    let y_in = y;
    let mut y_gated = backend.alloc(bs * h)?;
    backend.mul(&y_in, &res, &mut y_gated, bs * h)?;

    // out_proj
    let mut out = backend.alloc(bs * m)?;
    let out_bias = if block.has_out_bias {
        Some(&block.out_proj_b)
    } else {
        None
    };
    backend.sgemm_bias(&y_gated, &block.out_proj_w, out_bias, &mut out, bs, h, m)?;
    backend.read_to_host(&out)
}
