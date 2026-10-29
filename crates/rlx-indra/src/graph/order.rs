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

//! Reading-order / layout model (`order.onnx`): line features plus Fourier
//! features of the box, a pre-norm transformer encoder, and pairwise heads.
//!
//! Inputs `feats [1, N, 16]` and `key_bias [1, 1, 1, N]` (0 for a line, a large
//! negative number for padding). Outputs, for the first `N` rows only when
//! padded: `a [N, N]` (i before j logits, antisymmetric), `cls [N, C]` (block
//! class), `same [N, N]` (same-block, symmetric), `tab [3, N, N]` (same row /
//! column / cell, symmetric).

use super::{Builder, Weights};
use anyhow::{Result, bail};
use rlx_ir::hir::HirNodeId;
use rlx_ir::{Graph, HirGraphExt};
use std::collections::HashMap;

pub const N_FEATS: usize = 16;
/// Box coordinates (the first four features) that get Fourier features.
const N_FOURIER_IN: usize = 4;
/// Fourier frequencies per coordinate: `(π/2)·2^k`, k = 0..16.
const N_FREQ: usize = 16;
const HEADS: usize = 4;

/// Key bias for padded lines (exp underflows to exactly 0).
pub const PAD_BIAS: f32 = -1e9;

pub fn build(w: &Weights, n: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let layers = (0..)
        .take_while(|i| w.has(&format!("layers.{i}.qkv.weight")))
        .count();
    if layers == 0 {
        bail!("order model has no layers");
    }
    let d = w.shape("inp.2.weight")?[1];
    let dh = d / HEADS;
    let mut b = Builder::new("indra_order", w);
    let feats = b.input("feats", &[1, n, N_FEATS]);
    let key_bias = b.input("key_bias", &[1, 1, 1, n]);

    // Fourier features of x0, y0, x1, y1: sin then cos per coordinate.
    let coords = b.m().narrow_(feats, 2, 0, N_FOURIER_IN);
    let coords = b
        .m()
        .reshape_(coords, vec![1, n as i64, N_FOURIER_IN as i64, 1]);
    let freq: Vec<f32> = (0..N_FREQ)
        .map(|k| std::f32::consts::FRAC_PI_2 * (1u32 << k) as f32)
        .collect();
    let freq = b.constant("order.freq", &[1, 1, 1, N_FREQ], freq);
    let grid = vec![1, n as i64, N_FOURIER_IN as i64, N_FREQ as i64];
    let coords = b.m().expand_(coords, grid.clone());
    let freq = b.m().expand_(freq, grid);
    let ang = b.m().mul(coords, freq);
    let sin = b.act(ang, rlx_ir::op::Activation::Sin);
    let cos = b.act(ang, rlx_ir::op::Activation::Cos);
    let four = b.m().concat_(vec![sin, cos], 3);
    let four = b
        .m()
        .reshape_(four, vec![1, n as i64, (N_FOURIER_IN * N_FREQ * 2) as i64]);
    let mut h = b.m().concat_(vec![feats, four], 2);

    h = b.linear(h, "inp.0")?;
    h = b.m().gelu(h);
    h = b.linear(h, "inp.2")?;

    let scale = 1.0 / (dh as f32).sqrt();
    for l in 0..layers {
        let p = format!("layers.{l}");
        let x = b.layer_norm(h, &format!("{p}.n1"))?;
        let qkv = b.linear(x, &format!("{p}.qkv"))?; // [1, N, 3·D]
        let qkv = b
            .m()
            .reshape_(qkv, vec![n as i64, 3, HEADS as i64, dh as i64]);
        let qkv = b.m().transpose_(qkv, vec![1, 2, 0, 3]); // [3, H, N, dh]
        let q = b.m().narrow_(qkv, 0, 0, 1);
        let k = b.m().narrow_(qkv, 0, 1, 1);
        let v = b.m().narrow_(qkv, 0, 2, 1);
        let hd = vec![HEADS as i64, n as i64, dh as i64];
        let (q, k, v) = (
            b.m().reshape_(q, hd.clone()),
            b.m().reshape_(k, hd.clone()),
            b.m().reshape_(v, hd),
        );
        let kt = b.m().transpose_(k, vec![0, 2, 1]);
        let s = b.m().mm(q, kt); // [H, N, N]
        let s = b.scale(s, scale, &format!("{p}.attn_scale"));
        let s = b.m().reshape_(s, vec![1, HEADS as i64, n as i64, n as i64]);
        let s = b.add_b(s, key_bias);
        let s = b.m().sm(s, -1);
        let s = b.m().reshape_(s, vec![HEADS as i64, n as i64, n as i64]);
        let o = b.m().mm(s, v); // [H, N, dh]
        let o = b.m().transpose_(o, vec![1, 0, 2]);
        let o = b.m().reshape_(o, vec![1, n as i64, d as i64]);
        let o = b.linear(o, &format!("{p}.proj"))?;
        h = b.m().add(h, o);
        let x = b.layer_norm(h, &format!("{p}.n2"))?;
        let f = b.linear(x, &format!("{p}.ff.0"))?;
        let f = b.m().gelu(f);
        let f = b.linear(f, &format!("{p}.ff.2"))?;
        h = b.m().add(h, f);
    }
    let h = b.layer_norm(h, "norm")?;
    let inv = 1.0 / (d as f32).sqrt();

    let pair = |b: &mut Builder<'_>, u: &str, v: &str, key: &str| -> Result<HirNodeId> {
        let qu = b.linear(h, u)?;
        let kv = b.linear(h, v)?;
        let qu = b.m().reshape_(qu, vec![n as i64, d as i64]);
        let kv = b.m().reshape_(kv, vec![n as i64, d as i64]);
        let kt = b.m().transpose_(kv, vec![1, 0]);
        let s = b.m().mm(qu, kt);
        Ok(b.scale(s, inv, key))
    };
    let a = pair(&mut b, "q", "k", "order.a_scale")?;
    let at = b.m().transpose_(a, vec![1, 0]);
    let a = b.m().sub(a, at);
    let same = pair(&mut b, "bu", "bw", "order.same_scale")?;
    let st = b.m().transpose_(same, vec![1, 0]);
    let same = b.m().add(same, st);

    // Table heads: three [N, D] projections each side.
    let tu = b.linear(h, "tu")?;
    let tw = b.linear(h, "tw")?;
    let tu = b.m().reshape_(tu, vec![n as i64, 3, d as i64]);
    let tu = b.m().transpose_(tu, vec![1, 0, 2]); // [3, N, D]
    let tw = b.m().reshape_(tw, vec![n as i64, 3, d as i64]);
    let tw = b.m().transpose_(tw, vec![1, 2, 0]); // [3, D, N]
    let tab = b.m().mm(tu, tw);
    let tab = b.scale(tab, inv, "order.tab_scale");
    let tt = b.m().transpose_(tab, vec![0, 2, 1]);
    let tab = b.m().add(tab, tt);

    let classes = w.shape("cls.bias")?[0];
    let cls = b.linear(h, "cls")?;
    let cls = b.m().reshape_(cls, vec![n as i64, classes as i64]);
    b.finish(vec![a, cls, same, tab])
}
