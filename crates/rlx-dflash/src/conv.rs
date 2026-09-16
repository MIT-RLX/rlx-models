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

//! DFlash2 grouped dynamic depthwise convolution.
//!
//! ```text
//! out[i,c] = Σ_t (base[t,c] + δ[i,t,g(c)]) · x[i−t,c]
//! ```
//!
//! A block drafter predicts every position in one pass, so nothing carries
//! information *along* the block — which is why acceptance decays towards the
//! suffix. This conv is the cheapest possible fix: a 2-tap depthwise filter
//! that lets position `i` see position `i−1`, with a static per-channel kernel
//! `base` plus a content-predicted correction `δ` shared across each group of
//! `conv_group_size` (16) channels.
//!
//! It is applied four times per layer — before and after attention, before and
//! after the FFN — with `side` selecting which half of `base`/`δ` to use. Both
//! sides read the *same* `δ`, projected once from the pre-norm hidden, so a
//! sublayer pair costs one extra GEMM rather than two.
//!
//! Shifts stop at the block boundary. That is not an approximation: the block
//! is laid out `[anchor, MASK, MASK, …]`, so shifting within it hands position
//! 1 the anchor — the last *verified* token's representation — exactly as the
//! reference intends. Only slot 0 reads zeros, and slot 0 is the anchor, not a
//! prediction.
//!
//! No new opcode is needed: a tap is a slice + pad, a group broadcast is
//! [`Op::Expand`], and the rest is `mul`/`add`.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::PadMode;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::config::Dflash2Config;

/// Which of the two convolutions in a sublayer pair to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvSide {
    /// Applied to the normed hidden, before attention / the FFN.
    Pre = 0,
    /// Applied to the sublayer's output, before the residual add.
    Post = 1,
}

/// Emit one grouped dynamic depthwise convolution.
///
/// * `x`     — `[batch, block, hidden]`; `batch` is the number of draft blocks
///   in flight and `block` the tokens per block. rlx has a real batch axis, so
///   the tap shift is a plain slice along axis 1 (the reference has to carve
///   blocks out of a flat token stream via `ubatch.n_seqs_unq`).
/// * `delta` — `[batch, block, dynamic_dim]`, the output of
///   `kernel_projection`, laid out `(side, tap, group)` with `group` fastest —
///   matching the GGUF tensor's `ne = (n_groups, kernel, 2)`.
/// * `base`  — `[2, kernel, hidden]`, the static kernel; GGUF ships
///   `ne = (n_embd, kernel, 2)`, which reverses to exactly this.
pub fn emit_dyn_conv(
    g: &mut Graph,
    x: NodeId,
    delta: NodeId,
    base: NodeId,
    side: ConvSide,
    cfg: &Dflash2Config,
) -> NodeId {
    let xs = g.shape(x).clone();
    assert_eq!(xs.rank(), 3, "emit_dyn_conv expects [batch, block, hidden]");
    let (b, s, h) = (
        xs.dim(0).unwrap_static(),
        xs.dim(1).unwrap_static(),
        xs.dim(2).unwrap_static(),
    );
    let (k, gsz) = (cfg.conv_kernel_size, cfg.conv_group_size);
    let ng = cfg.n_groups(h);
    assert_eq!(ng * gsz, h, "hidden {h} is not a multiple of group {gsz}");
    let side = side as usize;
    let f = DType::F32;

    let d = g.reshape_(delta, vec![b as i64, s as i64, 2, k as i64, ng as i64]);

    let mut acc: Option<NodeId> = None;
    for tap in 0..k {
        // x shifted `tap` positions later within the block, zero at the front.
        let shifted = if tap == 0 {
            x
        } else if tap < s {
            let keep = g.slice_(x, 1, 0, s - tap, 1);
            g.pad_(keep, vec![[0, 0], [tap, 0], [0, 0]], PadMode::Constant(0.0))
        } else {
            // Tap reaches past the whole block: contributes nothing.
            g.zeros(&[b, s, h], f)
        };

        // δ[.., side, tap, :] : [B,S,G] -> broadcast over the group -> [B,S,H]
        let dt = g.slice_(d, 2, side, 1, 1);
        let dt = g.slice_(dt, 3, tap, 1, 1);
        let dt = g.reshape_(dt, vec![b as i64, s as i64, ng as i64, 1]);
        let dt = g.add_node(
            Op::Expand {
                target_shape: vec![b as i64, s as i64, ng as i64, gsz as i64],
            },
            vec![dt],
            Shape::new(&[b, s, ng, gsz], f),
        );
        let dt = g.reshape_(dt, vec![b as i64, s as i64, h as i64]);

        // base[side, tap, :] : [H], broadcasts over [B,S,H].
        let bt = g.slice_(base, 0, side, 1, 1);
        let bt = g.slice_(bt, 1, tap, 1, 1);
        let bt = g.reshape_(bt, vec![h as i64]);

        let w = g.add(dt, bt);
        let term = g.mul(w, shifted);
        acc = Some(match acc {
            None => term,
            Some(a) => g.add(a, term),
        });
    }
    acc.expect("conv_kernel_size must be >= 1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::{Device, Session};

    fn cfg() -> Dflash2Config {
        Dflash2Config {
            conv_kernel_size: 2,
            conv_group_size: 2,
            selector_rank: 4,
            selector_top_k: 2,
        }
    }

    /// Straight transcription of `out[i,c] = Σ_t (base[t,c] + δ[i,t,g(c)]) ·
    /// x[i−t,c]`, written independently of the graph so it can disagree.
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        x: &[f32],
        delta: &[f32],
        base: &[f32],
        side: usize,
        b: usize,
        s: usize,
        h: usize,
        c: &Dflash2Config,
    ) -> Vec<f32> {
        let (k, gsz) = (c.conv_kernel_size, c.conv_group_size);
        let ng = h / gsz;
        let mut out = vec![0f32; b * s * h];
        for bi in 0..b {
            for i in 0..s {
                for ch in 0..h {
                    let grp = ch / gsz;
                    let mut sum = 0f32;
                    for tap in 0..k {
                        if tap > i {
                            continue; // zero-filled before the block start
                        }
                        let d = delta[((bi * s + i) * 2 + side) * k * ng + tap * ng + grp];
                        let bs = base[(side * k + tap) * h + ch];
                        sum += (bs + d) * x[(bi * s + (i - tap)) * h + ch];
                    }
                    out[(bi * s + i) * h + ch] = sum;
                }
            }
        }
        out
    }

    fn deterministic(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (((st >> 33) as f64 / (1u64 << 30) as f64) - 1.0) as f32 * 0.5
            })
            .collect()
    }

    fn run_side(side: ConvSide) {
        let c = cfg();
        let (b, s, h) = (2usize, 4usize, 6usize);
        let dyn_dim = c.dynamic_dim(h);

        let x = deterministic(b * s * h, 0x243f_6a88_85a3_08d3);
        let delta = deterministic(b * s * dyn_dim, 0x1319_8a2e_0370_7344);
        let base = deterministic(2 * c.conv_kernel_size * h, 0xa409_3822_299f_31d0);

        let mut g = Graph::new("dyn_conv_test");
        let xi = g.input("x", Shape::new(&[b, s, h], DType::F32));
        let di = g.input("delta", Shape::new(&[b, s, dyn_dim], DType::F32));
        let bi = g.param("base", Shape::new(&[2, c.conv_kernel_size, h], DType::F32));
        let out = emit_dyn_conv(&mut g, xi, di, bi, side, &c);
        g.set_outputs(vec![out]);

        let mut compiled = Session::new(Device::Cpu).compile(g);
        compiled.set_param("base", &base);
        let got = compiled.run(&[("x", x.as_slice()), ("delta", delta.as_slice())]);

        let want = oracle(&x, &delta, &base, side as usize, b, s, h, &c);
        assert_eq!(got[0].len(), want.len());
        for (i, (a, e)) in got[0].iter().zip(&want).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "{side:?} element {i}: graph {a} vs oracle {e}"
            );
        }
    }

    #[test]
    fn pre_side_matches_oracle() {
        run_side(ConvSide::Pre);
    }

    /// The two sides read different slices of `base`/`δ`; a layout slip that
    /// swapped them would still pass a single-side test.
    #[test]
    fn post_side_matches_oracle() {
        run_side(ConvSide::Post);
    }

    /// Position 0 has no predecessor inside the block, so it must depend on
    /// tap 0 alone — this is what keeps blocks independent.
    #[test]
    fn first_position_ignores_the_shifted_tap() {
        let c = cfg();
        let (b, s, h) = (1usize, 3usize, 4usize);
        let dyn_dim = c.dynamic_dim(h);

        let mut g = Graph::new("first_pos");
        let xi = g.input("x", Shape::new(&[b, s, h], DType::F32));
        let di = g.input("delta", Shape::new(&[b, s, dyn_dim], DType::F32));
        let bi = g.param("base", Shape::new(&[2, c.conv_kernel_size, h], DType::F32));
        let out = emit_dyn_conv(&mut g, xi, di, bi, ConvSide::Pre, &c);
        g.set_outputs(vec![out]);

        let mut compiled = Session::new(Device::Cpu).compile(g);
        // base = 1 everywhere so any leaked tap shows up directly.
        compiled.set_param("base", &vec![1f32; 2 * c.conv_kernel_size * h]);
        let delta = vec![0f32; b * s * dyn_dim];

        // Row 0 is 1s; rows 1..: 0. With tap 1 zero-filled at the block start,
        // out[0] = x[0] = 1s. out[1] = x[1] + x[0] = 0 + 1 = 1s.
        let mut x = vec![0f32; b * s * h];
        x[..h].fill(1.0);
        let got = compiled.run(&[("x", x.as_slice()), ("delta", delta.as_slice())]);

        for ch in 0..h {
            assert!((got[0][ch] - 1.0).abs() < 1e-6, "out[0][{ch}] leaked a tap");
            assert!(
                (got[0][h + ch] - 1.0).abs() < 1e-6,
                "out[1][{ch}] lost the anchor"
            );
            assert!(got[0][2 * h + ch].abs() < 1e-6, "out[2][{ch}] should be 0");
        }
    }
}
