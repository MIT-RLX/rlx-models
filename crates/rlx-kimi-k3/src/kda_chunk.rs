// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! FlashKDA — the **chunked-parallel** forward for Kimi Delta Attention, ported
//! from Moonshot AI's `FlashKDA` CUDA/CUTLASS kernel
//! (<https://github.com/MoonshotAI/FlashKDA>) into portable RLX IR primitives.
//!
//! The native [`rlx_ir::op::Op::GatedDeltaNet`] (`gated_delta_net_pc`) evaluates
//! the *same* per-channel gated delta-rule recurrence, but **sequentially**, one
//! token at a time (llama.cpp autoregressive path). That is dispatch-bound on a
//! GPU: the state matrix carries a data dependency across every token. FlashKDA's
//! insight is to reformulate the recurrence into a **chunked-parallel** form: pick
//! a small chunk `C` (16), do all the intra-chunk work as dense batched matmuls
//! over *every* chunk at once (**K1**), and let only the coarse `T/C` chunk-to-chunk
//! state carry stay sequential (**K2**). This is the WY / DPLR representation of
//! the delta rule.
//!
//! ## Algorithm (matches `FlashKDA/tests/torch_ref.py`, natural-log gate)
//!
//! Inputs are already L2-normed `q,k`, values `v`, the per-channel **log**-gate
//! `g_log` (`≤ 0`), and the post-sigmoid `beta`. For each chunk (rows `0..C`,
//! channels `0..N`), with `Gc = cumsum(g_log)` and `Gtot = Gc[C-1]`:
//!
//! ```text
//!   k_dec = k · exp(Gc)              q_dec = q · exp(Gc) · scale
//!   k_inv = k · exp(−Gc)             k_res = k_inv · exp(Gtot)
//!   L     = tril(k_dec · k_invᵀ, −1) · beta          (strictly lower, C×C)
//!   Mqk   = tril(q_dec · k_invᵀ,  0)                 (lower incl. diag, C×C)
//!   INV   = (I + L)⁻¹   via a Neumann product  (L nilpotent, L^C = 0)
//! ```
//!
//! and the cross-chunk recurrence carrying `H[k,v]` (the state `S[i,j]` of
//! `Op::GatedDeltaNet`):
//!
//! ```text
//!   U     = INV · (beta · (v − k_dec · H))
//!   out   = q_dec · H + Mqk · U
//!   H     = exp(Gtot)·H + k_resᵀ · U
//! ```
//!
//! `scale = 1/√N`. The Neumann product `(I−L)(I+L²)(I+L⁴)(I+L⁸) = (I+L)⁻¹` is
//! exact for `C ≤ 16` (strictly-lower `L` is nilpotent).
//!
//! ## Two K2 implementations (`ChunkDims::use_scan`)
//!
//! * **unroll** (default) — the chunk loop is materialized in the graph. Exact and
//!   simple, but graph size is `O(⌈T/C⌉)`, which blows up at long `T`.
//! * **scan** — the chunk-to-chunk state recurrence is a single
//!   [`rlx_ir::op::Op::Scan`] whose carry is `H`; the scan emits the trajectory of
//!   post-chunk states, and the per-chunk **outputs** are then computed in ONE
//!   fully parallel batched pass from the *pre*-chunk states. Graph size is `O(1)`
//!   in `T` (FlashKDA's K1/K2 split: K1 chunk-parallel, K2 the lone sequential
//!   thread, outputs parallel). Both paths are bit-identical.
//!
//! Everything is a stock RLX primitive (matmul / cumsum / trilu / exp / add / mul /
//! scan), so the graph runs on **all** backends in f32, bit-comparable to the
//! sequential `gated_delta_net_pc`.

use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{BinaryOp, Op};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};

/// Inclusive `cumsum` along `axis` (same shape in/out).
fn cumsum(g: &mut HirMut, x: HirNodeId, axis: i32) -> HirNodeId {
    let sh = g.shape(x).clone();
    g.add_node(Op::Cumsum { axis, exclusive: false }, vec![x], sh)
}

/// Lower-triangular mask keeping entries on/below diagonal `diag`.
fn tril(g: &mut HirMut, x: HirNodeId, diag: i64) -> HirNodeId {
    let sh = g.shape(x).clone();
    g.add_node(Op::Trilu { upper: false, diagonal: diag }, vec![x], sh)
}

/// Embed an f32 constant tensor directly in the graph.
fn f32_const(g: &mut HirMut, dims: &[usize], data: Vec<f32>) -> HirNodeId {
    debug_assert_eq!(data.len(), dims.iter().product::<usize>());
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.add_node(Op::Constant { data: bytes }, vec![], Shape::new(dims, DType::F32))
}

/// Swap the last two axes of a rank-`r` tensor.
fn transpose_last2(g: &mut HirMut, x: HirNodeId, r: usize) -> HirNodeId {
    let mut perm: Vec<usize> = (0..r).collect();
    perm.swap(r - 2, r - 1);
    g.transpose_(x, perm)
}

/// Move axis 0 and axis 1 (`[b, nc, …] → [nc, b, …]`).
fn swap01(g: &mut HirMut, x: HirNodeId, r: usize) -> HirNodeId {
    let mut perm: Vec<usize> = (0..r).collect();
    perm.swap(0, 1);
    g.transpose_(x, perm)
}

/// Split `[b, sp, h, n]` into per-chunk batches `[b, nc, h, c, n]` so `(b, nc, h)`
/// are matmul batch dims and `(c, n)` are the chunk's `(time, channel)` matrix.
fn to_chunks(g: &mut HirMut, x: HirNodeId, b: usize, nc: usize, c: usize, h: usize, n: usize) -> HirNodeId {
    let r = g.reshape_(x, vec![b as i64, nc as i64, c as i64, h as i64, n as i64]);
    g.transpose_(r, vec![0, 1, 3, 2, 4])
}

/// Slice chunk `ci` out of a `[b, nc, h, m, p]` K1 tensor → `[b, h, m, p]`.
fn chunk_of(g: &mut HirMut, x: HirNodeId, ci: usize, b: usize, h: usize, m: usize, p: usize) -> HirNodeId {
    let s = g.narrow_(x, 1, ci, 1);
    g.reshape_(s, vec![b as i64, h as i64, m as i64, p as i64])
}

/// Shape parameters for [`build_kda_chunked_scan`].
#[derive(Debug, Clone, Copy)]
pub struct ChunkDims {
    pub batch: usize,
    pub seq: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Chunk size `C` (16 in FlashKDA). `≤ 16` for the fixed Neumann product.
    pub chunk: usize,
    /// Use the [`rlx_ir::op::Op::Scan`]-based K2 (O(1) graph size) instead of the
    /// unrolled chunk loop. Bit-identical output.
    pub use_scan: bool,
}

/// Intra-chunk (K1) tensors, all `[b, nc, h, c, *]` batched over `(b, nc, h)`.
struct K1 {
    k_dec: HirNodeId,  // [b, nc, h, c, n]
    q_dec: HirNodeId,  // [b, nc, h, c, n]
    inv: HirNodeId,    // [b, nc, h, c, c]
    mqk: HirNodeId,    // [b, nc, h, c, c]
    k_res: HirNodeId,  // [b, nc, h, c, n]
    v5: HirNodeId,     // [b, nc, h, c, n]
    beta5: HirNodeId,  // [b, nc, h, c, 1]
    egtot: HirNodeId,  // [b, nc, h, 1, n]
}

/// FlashKDA K1: all intra-chunk math, fully parallel over `(b, nc, h)`.
#[allow(clippy::too_many_arguments)]
fn kda_k1(
    g: &mut HirMut,
    q: HirNodeId,
    k: HirNodeId,
    v: HirNodeId,
    g_log: HirNodeId,
    beta: HirNodeId,
    b: usize,
    nc: usize,
    c: usize,
    h: usize,
    n: usize,
) -> K1 {
    let q5 = to_chunks(g, q, b, nc, c, h, n);
    let k5 = to_chunks(g, k, b, nc, c, h, n);
    let v5 = to_chunks(g, v, b, nc, c, h, n);
    let g5 = to_chunks(g, g_log, b, nc, c, h, n);
    let beta5 = {
        let r = g.reshape_(beta, vec![b as i64, nc as i64, c as i64, h as i64]);
        let t = g.transpose_(r, vec![0, 1, 3, 2]);
        g.reshape_(t, vec![b as i64, nc as i64, h as i64, c as i64, 1])
    };

    let scale = f32_const(g, &[1], vec![1.0 / (n as f32).sqrt()]);
    // Cumsum over the chunk's time axis (c). CPU cumsum only walks the LAST axis,
    // so transpose (c, n) → (n, c), scan, transpose back.
    let gc = {
        let gt = transpose_last2(g, g5, 5);
        let gct = cumsum(g, gt, 4);
        transpose_last2(g, gct, 5)
    };
    let eg = g.exp(gc);
    let neg_gc = g.neg(gc);
    let eng = g.exp(neg_gc);

    let k_dec = g.mul(k5, eg);
    let q_dec = {
        let qe = g.mul(q5, eg);
        g.mul(qe, scale)
    };
    let k_inv = g.mul(k5, eng);

    let gtot = g.narrow_(gc, 3, c - 1, 1); // [b, nc, h, 1, n]
    let egtot = g.exp(gtot);
    let k_res = g.mul(k_inv, egtot);

    let k_inv_t = transpose_last2(g, k_inv, 5); // [b, nc, h, n, c]
    let l = {
        let l0 = g.mm(k_dec, k_inv_t);
        let lt = tril(g, l0, -1);
        g.mul(lt, beta5)
    };
    let mqk = {
        let m0 = g.mm(q_dec, k_inv_t);
        tril(g, m0, 0)
    };

    let inv = {
        let mut eye = vec![0f32; c * c];
        for i in 0..c {
            eye[i * c + i] = 1.0;
        }
        let eye = f32_const(g, &[c, c], eye);
        let mut inv = g.sub(eye, l);
        let l2 = g.mm(l, l);
        let step = g.mm(inv, l2);
        inv = g.add(inv, step);
        let l4 = g.mm(l2, l2);
        let step = g.mm(inv, l4);
        inv = g.add(inv, step);
        let l8 = g.mm(l4, l4);
        let step = g.mm(inv, l8);
        g.add(inv, step)
    };

    K1 { k_dec, q_dec, inv, mqk, k_res, v5, beta5, egtot }
}

/// Reassemble a per-chunk output `[b, nc, c, h, n]` → `[b, s, h, n]` (padding
/// rows sliced off).
#[allow(clippy::too_many_arguments)]
fn reassemble(g: &mut HirMut, out_bncchn: HirNodeId, b: usize, sp: usize, h: usize, n: usize, s: usize, pad: usize) -> HirNodeId {
    let outr = g.reshape_(out_bncchn, vec![b as i64, sp as i64, h as i64, n as i64]);
    if pad > 0 {
        g.narrow_(outr, 1, 0, s)
    } else {
        outr
    }
}

/// K2 via an unrolled chunk loop. Graph size `O(⌈T/C⌉)`.
#[allow(clippy::too_many_arguments)]
fn k2_unroll(
    g: &mut HirMut,
    k1: &K1,
    mut hstate: HirNodeId,
    b: usize,
    nc: usize,
    c: usize,
    h: usize,
    n: usize,
    sp: usize,
    s: usize,
    pad: usize,
) -> (HirNodeId, HirNodeId) {
    let mut out_chunks: Vec<HirNodeId> = Vec::with_capacity(nc);
    for ci in 0..nc {
        let kd = chunk_of(g, k1.k_dec, ci, b, h, c, n);
        let qd = chunk_of(g, k1.q_dec, ci, b, h, c, n);
        let inv_c = chunk_of(g, k1.inv, ci, b, h, c, c);
        let mqk_c = chunk_of(g, k1.mqk, ci, b, h, c, c);
        let kr = chunk_of(g, k1.k_res, ci, b, h, c, n);
        let vc = chunk_of(g, k1.v5, ci, b, h, c, n);
        let bc = chunk_of(g, k1.beta5, ci, b, h, c, 1);
        let egt = {
            let sl = g.narrow_(k1.egtot, 1, ci, 1); // [b, 1, h, 1, n]
            g.reshape_(sl, vec![b as i64, h as i64, n as i64, 1])
        };

        let sk = g.mm(kd, hstate);
        let vcorr = g.sub(vc, sk);
        let vcorr = g.mul(vcorr, bc);
        let u = g.mm(inv_c, vcorr);

        let qh = g.mm(qd, hstate);
        let mu = g.mm(mqk_c, u);
        let out_c = g.add(qh, mu);
        let out_c = g.reshape_(out_c, vec![b as i64, h as i64, 1, c as i64, n as i64]);
        out_chunks.push(out_c);

        let kr_t = transpose_last2(g, kr, 4);
        let delta = g.mm(kr_t, u);
        let hdec = g.mul(hstate, egt);
        hstate = g.add(hdec, delta);
    }
    let out5 = g.concat_(out_chunks, 2); // [b, h, nc, c, n]
    let outp = g.transpose_(out5, vec![0, 2, 3, 1, 4]); // [b, nc, c, h, n]
    let out = reassemble(g, outp, b, sp, h, n, s, pad);
    (out, hstate)
}

/// Flat-`Graph` body for the K2 state scan: given carry `H [b,h,n,n]` and the
/// per-chunk `(k_dec, inv, k_res, v, beta, egtot)` slices, return the next `H`.
#[allow(clippy::too_many_arguments)]
fn scan_state_body(b: usize, c: usize, h: usize, n: usize) -> Graph {
    let f = DType::F32;
    let bhcn = Shape::new(&[b, h, c, n], f);
    let bhcc = Shape::new(&[b, h, c, c], f);
    let bhc1 = Shape::new(&[b, h, c, 1], f);
    let bhnn = Shape::new(&[b, h, n, n], f);
    let bhnc = Shape::new(&[b, h, n, c], f);
    let bhn1 = Shape::new(&[b, h, n, 1], f);

    let mut body = Graph::new("kda_state_scan");
    let carry = body.input("carry", bhnn.clone()); // H [b,h,n,n]
    let kd = body.input("kd", bhcn.clone());
    let inv = body.input("inv", bhcc);
    let kr = body.input("kr", bhcn.clone());
    let vc = body.input("v", bhcn.clone());
    let bc = body.input("beta", bhc1);
    let egt = body.input("egt", bhn1);

    let sk = body.add_node(Op::MatMul, vec![kd, carry], bhcn.clone()); // kd·H
    let vcorr = body.binary(BinaryOp::Sub, vc, sk, bhcn.clone());
    let vcorr = body.binary(BinaryOp::Mul, vcorr, bc, bhcn.clone());
    let u = body.add_node(Op::MatMul, vec![inv, vcorr], bhcn); // INV·(β·(v−kd·H))
    let kr_t = body.add_node(Op::Transpose { perm: vec![0, 1, 3, 2] }, vec![kr], bhnc);
    let delta = body.add_node(Op::MatMul, vec![kr_t, u], bhnn.clone()); // k_resᵀ·U
    let hdec = body.binary(BinaryOp::Mul, carry, egt, bhnn.clone()); // exp(Gtot)·H
    let h_new = body.binary(BinaryOp::Add, hdec, delta, bhnn);
    body.set_outputs(vec![h_new]);
    body
}

/// K2 via a single [`Op::Scan`] over the chunk axis. Graph size `O(1)` in `T`:
/// the scan threads only the state `H`, and the per-chunk outputs are computed in
/// one fully parallel batched pass from the *pre*-chunk states.
#[allow(clippy::too_many_arguments)]
fn k2_scan(
    g: &mut HirMut,
    k1: &K1,
    init: HirNodeId,
    b: usize,
    nc: usize,
    c: usize,
    h: usize,
    n: usize,
    sp: usize,
    s: usize,
    pad: usize,
) -> (HirNodeId, HirNodeId) {
    let f = DType::F32;
    // xs for the scan: reorder K1 tensors to [nc, b, h, c, *] (chunk axis leading).
    let kd_xs = swap01(g, k1.k_dec, 5); // [nc, b, h, c, n]
    let inv_xs = swap01(g, k1.inv, 5); // [nc, b, h, c, c]
    let kr_xs = swap01(g, k1.k_res, 5); // [nc, b, h, c, n]
    let v_xs = swap01(g, k1.v5, 5); // [nc, b, h, c, n]
    let bc_xs = swap01(g, k1.beta5, 5); // [nc, b, h, c, 1]
    let egt_xs = {
        let t = swap01(g, k1.egtot, 5); // [nc, b, h, 1, n]
        transpose_last2(g, t, 5) // [nc, b, h, n, 1]
    };

    // Sequential state recurrence → trajectory of post-chunk states.
    let body = scan_state_body(b, c, h, n);
    let traj_shape = Shape::new(&[nc, b, h, n, n], f);
    let h_traj = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: nc as u32,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 6,
            num_checkpoints: 0,
        },
        vec![init, kd_xs, inv_xs, kr_xs, v_xs, bc_xs, egt_xs],
        traj_shape,
    ); // [nc, b, h, n, n] — H AFTER each chunk

    // Pre-chunk states Hb: [init, H(0), …, H(nc-2)]  → [nc, b, h, n, n].
    let init5 = g.reshape_(init, vec![1, b as i64, h as i64, n as i64, n as i64]);
    let hb = if nc == 1 {
        init5
    } else {
        let prev = g.narrow_(h_traj, 0, 0, nc - 1); // [nc-1, b, h, n, n]
        g.concat_(vec![init5, prev], 0)
    };

    // Parallel output pass, batched over (nc, b, h). Recomputes U from Hb.
    let qd_xs = swap01(g, k1.q_dec, 5); // [nc, b, h, c, n]
    let mqk_xs = swap01(g, k1.mqk, 5); // [nc, b, h, c, c]
    let sk = g.mm(kd_xs, hb); // [nc, b, h, c, n]
    let vcorr = g.sub(v_xs, sk);
    let vcorr = g.mul(vcorr, bc_xs);
    let u = g.mm(inv_xs, vcorr);
    let out = {
        let qh = g.mm(qd_xs, hb);
        let mu = g.mm(mqk_xs, u);
        g.add(qh, mu)
    }; // [nc, b, h, c, n]

    // [nc, b, h, c, n] → [b, nc, c, h, n] → [b, s, h, n].
    let outp = g.transpose_(out, vec![1, 0, 3, 2, 4]);
    let out = reassemble(g, outp, b, sp, h, n, s, pad);

    // Final state = last trajectory row.
    let final_state = {
        let last = g.narrow_(h_traj, 0, nc - 1, 1); // [1, b, h, n, n]
        g.reshape_(last, vec![b as i64, h as i64, n as i64, n as i64])
    };
    (out, final_state)
}

/// Build FlashKDA's chunked-parallel gated-delta-net forward.
///
/// Drop-in for `HirMut::gated_delta_net_pc`: `q,k,v,g_log` are `[b, s, h, n]`
/// (`q,k` L2-normed, `g_log` the per-channel natural-log gate), `beta` is
/// `[b, s, h]` (post-sigmoid). Returns `(out [b, s, h, n], final_state
/// [b, h, n, n])`. `initial_state` (`[b, h, n, n]`) seeds the recurrence; `None`
/// starts from zero (prefill). `d.use_scan` selects the K2 implementation.
#[allow(clippy::too_many_arguments)]
pub fn build_kda_chunked_scan(
    g: &mut HirMut,
    q: HirNodeId,
    k: HirNodeId,
    v: HirNodeId,
    g_log: HirNodeId,
    beta: HirNodeId,
    d: ChunkDims,
    initial_state: Option<HirNodeId>,
) -> (HirNodeId, HirNodeId) {
    let (b, s, h, n, c) = (d.batch, d.seq, d.heads, d.head_dim, d.chunk);
    assert!((1..=16).contains(&c), "chunk size must be in 1..=16 (got {c})");
    let nc = s.div_ceil(c);
    let sp = nc * c;
    let pad = sp - s;

    // Pad the sequence up to a whole number of chunks with zeros. Padded tokens
    // (k=v=beta=0) contribute nothing to the state and their output rows are
    // sliced off; g_log padded with 0 keeps the cumulative gate flat.
    let (q, k, v, g_log, beta) = if pad > 0 {
        let zqkvg = |g: &mut HirMut| f32_const(g, &[b, pad, h, n], vec![0.0; b * pad * h * n]);
        let z1 = zqkvg(g);
        let q = g.concat_(vec![q, z1], 1);
        let z2 = zqkvg(g);
        let k = g.concat_(vec![k, z2], 1);
        let z3 = zqkvg(g);
        let v = g.concat_(vec![v, z3], 1);
        let z4 = zqkvg(g);
        let g_log = g.concat_(vec![g_log, z4], 1);
        let zb = f32_const(g, &[b, pad, h], vec![0.0; b * pad * h]);
        let beta = g.concat_(vec![beta, zb], 1);
        (q, k, v, g_log, beta)
    } else {
        (q, k, v, g_log, beta)
    };

    let k1 = kda_k1(g, q, k, v, g_log, beta, b, nc, c, h, n);

    let init = initial_state.unwrap_or_else(|| f32_const(g, &[b, h, n, n], vec![0.0; b * h * n * n]));

    if d.use_scan {
        k2_scan(g, &k1, init, b, nc, c, h, n, sp, s, pad)
    } else {
        k2_unroll(g, &k1, init, b, nc, c, h, n, sp, s, pad)
    }
}
