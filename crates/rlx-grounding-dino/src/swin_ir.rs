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

//! Swin windowed self-attention as an on-device HIR graph.
//!
//! Window partition / cyclic shift / merge are cheap host memory moves and stay
//! on the host; the per-window multi-head attention (the matmul-heavy part) runs
//! as a single **batched** graph over all windows: `[n_win·heads, ws², hd]`
//! `mm` → add relative-position bias → softmax → `mm`. The additive bias input
//! `[n_win, heads, ws², ws²]` already folds in the relative-position table and,
//! for shifted blocks, the region mask. Memory stays bounded (windows are small,
//! `ws² = 144`).

use crate::ir::{self, Params};
use anyhow::Result;
use rlx_ir::{DType, HirGraphExt, HirModule, HirMut, HirNodeId, Shape};
use rlx_runtime::Device;

/// Build batched per-window multi-head attention into a shared graph: projects
/// `win_n [n_win·ws2, dim]`, runs `[n_win·heads, ws2, hd]` `mm` → +bias → softmax
/// → `mm`, and projects back to `[n_win·ws2, dim]`. `bias_n` is
/// `[n_win, heads, ws2, ws2]`. `prefix` namespaces the q/k/v/o params. Shared by
/// the standalone runner and the fused Swin block ([`crate::swin`]).
#[allow(clippy::too_many_arguments)]
pub fn build_window_attn(
    g: &mut HirMut<'_>,
    params: &mut Params,
    w: &WindowAttnWeights,
    prefix: &str,
    win_n: HirNodeId,
    bias_n: HirNodeId,
    n_win: usize,
    ws2: usize,
    dim: usize,
    nh: usize,
) -> HirNodeId {
    let hd = dim / nh;
    let scale = 1.0 / (hd as f32).sqrt();
    let nwh = (n_win * nh) as i64;
    let n = |s: &str| format!("{prefix}{s}");

    let q = ir::linear(g, params, &n("q"), win_n, dim, dim, &w.q_w, &w.q_b, scale);
    let k = ir::linear(g, params, &n("k"), win_n, dim, dim, &w.k_w, &w.k_b, 1.0);
    let v = ir::linear(g, params, &n("v"), win_n, dim, dim, &w.v_w, &w.v_b, 1.0);

    let q = g.reshape_(q, vec![n_win as i64, ws2 as i64, nh as i64, hd as i64]);
    let q = g.transpose_(q, vec![0, 2, 1, 3]);
    let q = g.reshape_(q, vec![nwh, ws2 as i64, hd as i64]);
    let k = g.reshape_(k, vec![n_win as i64, ws2 as i64, nh as i64, hd as i64]);
    let k = g.transpose_(k, vec![0, 2, 3, 1]);
    let k = g.reshape_(k, vec![nwh, hd as i64, ws2 as i64]);
    let v = g.reshape_(v, vec![n_win as i64, ws2 as i64, nh as i64, hd as i64]);
    let v = g.transpose_(v, vec![0, 2, 1, 3]);
    let v = g.reshape_(v, vec![nwh, ws2 as i64, hd as i64]);

    let scores = g.mm(q, k); // [n_win*heads, ws2, ws2]
    let bias_r = g.reshape_(bias_n, vec![nwh, ws2 as i64, ws2 as i64]);
    let scores = g.add(scores, bias_r);
    let probs = g.sm(scores, -1);
    let ctx = g.mm(probs, v); // [n_win*heads, ws2, hd]

    let ctx = g.reshape_(ctx, vec![n_win as i64, nh as i64, ws2 as i64, hd as i64]);
    let ctx = g.transpose_(ctx, vec![0, 2, 1, 3]);
    let ctx = g.reshape_(ctx, vec![(n_win * ws2) as i64, dim as i64]);
    ir::linear(g, params, &n("o"), ctx, dim, dim, &w.o_w, &w.o_b, 1.0)
}

/// Per-window attention projection weights (PyTorch `[dim, dim]`).
#[derive(Clone)]
pub struct WindowAttnWeights {
    pub q_w: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w: Vec<f32>,
    pub v_b: Vec<f32>,
    pub o_w: Vec<f32>,
    pub o_b: Vec<f32>,
}

/// On-device Swin window attention.
pub struct WindowAttnIr {
    w: WindowAttnWeights,
    dim: usize,
    n_heads: usize,
    device: Device,
}

impl WindowAttnIr {
    pub fn new(w: WindowAttnWeights, dim: usize, n_heads: usize, device: Device) -> Self {
        Self {
            w,
            dim,
            n_heads,
            device,
        }
    }

    /// `windows` is `[n_win·ws2, dim]`, `bias` is `[n_win, heads, ws2, ws2]`
    /// (relative-position bias + optional shift mask). Returns `[n_win·ws2, dim]`.
    pub fn forward(
        &self,
        windows: &[f32],
        bias: &[f32],
        n_win: usize,
        ws2: usize,
    ) -> Result<Vec<f32>> {
        let dim = self.dim;
        let nh = self.n_heads;
        let w = &self.w;

        let mut hir = HirModule::new("swin_window_attn");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);

        let win_n = g.input("windows", Shape::new(&[n_win * ws2, dim], DType::F32));
        let bias_n = g.input("bias", Shape::new(&[n_win, nh, ws2, ws2], DType::F32));
        let out = build_window_attn(
            &mut g,
            &mut params,
            w,
            "",
            win_n,
            bias_n,
            n_win,
            ws2,
            dim,
            nh,
        );
        g.set_outputs(vec![out]);

        let outs = ir::compile_and_run(
            hir,
            params,
            self.device,
            &[("windows", windows), ("bias", bias)],
        )?;
        Ok(outs.into_iter().next().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{self, AttnBias};

    fn det(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 13 + seed * 7) % 17) as f32 - 8.0) * 0.02)
            .collect()
    }

    #[test]
    fn window_attn_ir_matches_native() {
        let (dim, nh) = (8usize, 2usize);
        let ws2 = 4usize; // 2x2 window
        let n_win = 3usize;
        let w = WindowAttnWeights {
            q_w: det(dim * dim, 1),
            q_b: vec![0.0; dim],
            k_w: det(dim * dim, 2),
            k_b: vec![0.0; dim],
            v_w: det(dim * dim, 3),
            v_b: vec![0.0; dim],
            o_w: det(dim * dim, 4),
            o_b: vec![0.0; dim],
        };
        let windows = det(n_win * ws2 * dim, 20);
        // Per-window per-head relative-position bias.
        let bias = det(n_win * nh * ws2 * ws2, 30);

        // Native: per-window nn::mha with PerHead bias.
        let mut native = vec![0f32; n_win * ws2 * dim];
        for wi in 0..n_win {
            let win = &windows[wi * ws2 * dim..(wi + 1) * ws2 * dim];
            let wbias = &bias[wi * nh * ws2 * ws2..(wi + 1) * nh * ws2 * ws2];
            let out = nn::mha(
                win,
                win,
                win,
                ws2,
                ws2,
                dim,
                nh,
                &w.q_w,
                &w.q_b,
                &w.k_w,
                &w.k_b,
                &w.v_w,
                &w.v_b,
                &w.o_w,
                &w.o_b,
                AttnBias::PerHead(wbias),
            );
            native[wi * ws2 * dim..(wi + 1) * ws2 * dim].copy_from_slice(&out);
        }

        let ir = WindowAttnIr::new(w, dim, nh, Device::Cpu);
        let got = ir.forward(&windows, &bias, n_win, ws2).unwrap();

        assert_eq!(native.len(), got.len());
        let e = native
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e < 1e-4, "native vs IR window attn max_err={e}");
    }
}
