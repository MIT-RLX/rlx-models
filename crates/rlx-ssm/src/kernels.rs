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

//! Reference CPU kernels for SSM custom ops (parity tests + host fallback).

use anyhow::Result;

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Lightning-attention decode step: updates `[b,h,n,n]` state, emits `y[b,h,n]`.
pub fn execute_lightning_attention_step_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state_in: &[f32],
    y_out: &mut [f32],
    state_out: &mut [f32],
    b: usize,
    h: usize,
    n: usize,
) -> Result<()> {
    let _ = q;
    let scale = 1.0 / (n as f32).sqrt();
    for bi in 0..b {
        for hi in 0..h {
            let off = bi * h * n * n + hi * n * n;
            let beta_h = beta[bi * h + hi];
            let gate_h = gate[bi * h + hi];
            for i in 0..n {
                for j in 0..n {
                    let idx = off + i * n + j;
                    let k_i = k[bi * h * n + hi * n + i];
                    let v_j = v[bi * h * n + hi * n + j];
                    state_out[idx] = beta_h * k_i * v_j + state_in[idx] * (1.0 - beta_h);
                }
            }
            for j in 0..n {
                let mut acc = 0f32;
                for i in 0..n {
                    acc += state_out[off + i * n + j];
                }
                y_out[bi * h * n + hi * n + j] = gate_h * acc * scale;
            }
        }
    }
    Ok(())
}

/// LFM SSM decode step: `state' = a * state + b * x`, `y = sum_n c * state' * gate`.
pub fn execute_lfm_ssm_step_f32(
    x: &[f32],
    a: &[f32],
    b_in: &[f32],
    c_proj: &[f32],
    gate: &[f32],
    state_in: &[f32],
    packed_out: &mut [f32],
    batch: usize,
    channels: usize,
    state_size: usize,
) -> Result<()> {
    let n = state_size;
    let c = channels;
    for bi in 0..batch {
        for ci in 0..c {
            let xv = x[bi * c + ci];
            let g = gate[bi * c + ci];
            let mut y_acc = 0f32;
            for ni in 0..n {
                let s_in = state_in[bi * c * n + ci * n + ni];
                let s_out = a[ci * n + ni] * s_in + b_in[bi * n + ni] * xv;
                packed_out[bi * (c + c * n) + c + ci * n + ni] = s_out;
                y_acc += c_proj[bi * n + ni] * s_out;
            }
            packed_out[bi * (c + c * n) + ci] = y_acc * g;
        }
    }
    Ok(())
}

/// Mamba1 / Mamba2 decode step: selective-scan style update with optional D-skip.
pub fn execute_mamba1_step_f32(
    x: &[f32],
    dt_raw: &[f32],
    a_log: &[f32],
    b_in: &[f32],
    c_proj: &[f32],
    d_skip: &[f32],
    state_in: &[f32],
    packed_out: &mut [f32],
    batch: usize,
    heads: usize,
    state_size: usize,
) -> Result<()> {
    execute_mamba2_step_f32(
        x, dt_raw, a_log, b_in, c_proj, d_skip, state_in, packed_out, batch, heads, state_size,
    )
}

/// Mamba2 decode step: selective-scan style update with optional D-skip.
pub fn execute_mamba2_step_f32(
    x: &[f32],
    dt_raw: &[f32],
    a_log: &[f32],
    b_in: &[f32],
    c_proj: &[f32],
    d_skip: &[f32],
    state_in: &[f32],
    packed_out: &mut [f32],
    batch: usize,
    heads: usize,
    state_size: usize,
) -> Result<()> {
    let n = state_size;
    let h = heads;
    let use_d = !d_skip.is_empty();
    for bi in 0..batch {
        for hi in 0..h {
            let xv = x[bi * h + hi];
            let dt = softplus(dt_raw[bi * h + hi]);
            let mut y = 0f32;
            for ni in 0..n {
                let a = -(a_log[hi * n + ni].exp());
                let s_in = state_in[bi * h * n + hi * n + ni];
                let s_out = (a * dt).exp() * s_in + dt * b_in[bi * n + ni] * xv;
                packed_out[bi * (h + h * n) + h + hi * n + ni] = s_out;
                y += c_proj[bi * n + ni] * s_out;
            }
            if use_d {
                y += d_skip[hi] * xv;
            }
            packed_out[bi * (h + h * n) + hi] = y;
        }
    }
    Ok(())
}
