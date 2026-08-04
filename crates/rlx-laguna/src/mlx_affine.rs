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

//! mlx-community **affine** (4/8-bit) packed-linear support for Laguna.
//!
//! Laguna's native packed path is GGUF K-quant only (`packed_forward.rs` host
//! `gguf_matmul_bt` + 2-input device `Op::DequantMatMul`). mlx-community
//! checkpoints ship a *different* codec — packed uint32 codes + per-group f32
//! `scales`/`biases` (`QuantScheme::MlxAffine { bits, group_size }`) — which the
//! GGUF block kernels can't read.
//!
//! This module adds the affine path in the same "keep it packed, dequant on the
//! fly" spirit as the GGUF host path: [`MlxPackedLinear`] tensors stay resident
//! as packed codes+scales+biases, and [`affine_matmul_bt`] dequantizes one
//! weight matrix transiently (via the shared [`dequant_affine_f32`]) and runs a
//! plain F32 GEMM. Correctness-first; no new backend kernel.
//!
//! Scope: dense linears (attention q/k/v/o, shared-expert, embed, lm_head) are
//! covered and unit-tested here. Routed-MoE expert *stacks* need the checkpoint's
//! exact expert tensor naming/layout to finalize — deferred behind an actionable
//! error until a reference Laguna mlx checkpoint is available (256-expert giant;
//! not on the dev box). See [`crate::packed_forward`] call sites.

use anyhow::{Result, bail};
use rlx_core::weight_loader::{MlxPackedLinear, dequant_affine_f32};
use rlx_ir::quant::QuantScheme;

/// Parse a little-endian f32 byte buffer into a `Vec<f32>`.
fn f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dequantize one MLX-affine packed linear to a row-major `[out, in]` F32
/// matrix. Only [`QuantScheme::MlxAffine`] is handled — mxfp variants ship u8
/// scales and are rejected here (Laguna checkpoints are affine 4-bit).
pub fn dequant_linear(p: &MlxPackedLinear) -> Result<(Vec<f32>, usize, usize)> {
    let QuantScheme::MlxAffine { bits, group_size } = p.scheme else {
        bail!(
            "laguna mlx-affine: unsupported scheme {:?} (expected MlxAffine)",
            p.scheme
        );
    };
    if p.out_shape.len() != 2 {
        bail!(
            "laguna mlx-affine: expected rank-2 out_shape, got {:?}",
            p.out_shape
        );
    }
    let out = p.out_shape[0];
    let inn = p.out_shape[1];
    let gs = group_size as usize;
    if gs == 0 || !inn.is_multiple_of(gs) {
        bail!("laguna mlx-affine: in_features {inn} not divisible by group_size {gs}");
    }
    let n_groups = inn / gs;
    let scales = f32_le(&p.scales);
    let biases = f32_le(&p.biases);
    let w = dequant_affine_f32(
        &p.w_q,
        &scales,
        &biases,
        bits as u32,
        group_size,
        out,
        n_groups,
    )?;
    Ok((w, out, inn))
}

/// `y[t, o] = Σ_i x[t, i] · W[o, i]` for a packed MLX-affine weight `W` of
/// logical shape `[out, in]`. `x` is row-major `[seq, in]`; returns `[seq, out]`.
///
/// The weight is dequantized transiently (one matrix at a time — bounded memory,
/// unlike a full-model F32 expand) then multiplied with a straight triple loop,
/// mirroring [`crate::packed_forward`]'s GGUF host GEMM.
pub fn affine_matmul_bt(
    x: &[f32],
    p: &MlxPackedLinear,
    seq: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    let (w, out, inn) = dequant_linear(p)?;
    if out != out_dim || inn != in_dim {
        bail!("laguna mlx-affine: weight [{out},{inn}] != expected out={out_dim} in={in_dim}");
    }
    if x.len() != seq * in_dim {
        bail!(
            "laguna mlx-affine: x len {} != seq*in {}",
            x.len(),
            seq * in_dim
        );
    }
    let mut y = vec![0f32; seq * out_dim];
    for t in 0..seq {
        let xr = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += xr[i] * wr[i];
            }
            y[t * out_dim + o] = acc;
        }
    }
    Ok(y)
}

fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Slice expert `e` out of a **stacked** MLX-affine MoE weight (`switch_mlp.*`,
/// logical `[n_expert, out, in]`) into a per-expert [`MlxPackedLinear`] of shape
/// `[out, in]`. MLX packs codes per output-row over the last (`in`) dim, so each
/// expert is a contiguous byte block of both the codes and the f32 scales/biases
/// — a clean slice, no re-packing.
pub fn expert_slice(
    stacked: &MlxPackedLinear,
    e: usize,
    n_expert: usize,
    out: usize,
    inn: usize,
) -> Result<MlxPackedLinear> {
    if !matches!(stacked.scheme, QuantScheme::MlxAffine { .. }) {
        bail!(
            "laguna mlx-affine expert: unsupported scheme {:?}",
            stacked.scheme
        );
    }
    if e >= n_expert {
        bail!("laguna mlx-affine expert {e} >= {n_expert}");
    }
    let need = |len: usize, name: &str| -> Result<usize> {
        if !len.is_multiple_of(n_expert) {
            bail!(
                "laguna mlx-affine expert: {name} len {len} not divisible by n_expert {n_expert}"
            );
        }
        Ok(len / n_expert)
    };
    let pw = need(stacked.w_q.len(), "codes")?;
    let ps = need(stacked.scales.len(), "scales")?;
    let w_q = stacked.w_q[e * pw..(e + 1) * pw].to_vec();
    let scales = stacked.scales[e * ps..(e + 1) * ps].to_vec();
    let biases = if stacked.biases.is_empty() {
        Vec::new()
    } else {
        let pb = need(stacked.biases.len(), "biases")?;
        stacked.biases[e * pb..(e + 1) * pb].to_vec()
    };
    Ok(MlxPackedLinear {
        w_q,
        scales,
        biases,
        scheme: stacked.scheme,
        out_shape: vec![out, inn],
    })
}

/// SwiGLU for one routed expert on a single token, from **stacked** MLX-affine
/// `switch_mlp` gate/up/down packs: `down( silu(gate·x) * (up·x) )`. `x` is one
/// `[hidden]` row; returns `[hidden]`. Per-expert weights are dequantized
/// transiently (one matrix at a time — bounded memory, never a full-model F32
/// expand), mirroring the GGUF host MoE path.
#[allow(clippy::too_many_arguments)]
pub fn affine_expert_swiglu(
    x: &[f32],
    gate: &MlxPackedLinear,
    up: &MlxPackedLinear,
    down: &MlxPackedLinear,
    e: usize,
    n_expert: usize,
    hidden: usize,
    inter: usize,
) -> Result<Vec<f32>> {
    let ge = expert_slice(gate, e, n_expert, inter, hidden)?;
    let ue = expert_slice(up, e, n_expert, inter, hidden)?;
    let de = expert_slice(down, e, n_expert, hidden, inter)?;
    let g = affine_matmul_bt(x, &ge, 1, inter, hidden)?;
    let u = affine_matmul_bt(x, &ue, 1, inter, hidden)?;
    let mut mid = vec![0f32; inter];
    for i in 0..inter {
        mid[i] = silu_f32(g[i]) * u[i];
    }
    affine_matmul_bt(&mid, &de, 1, hidden, inter)
}

/// Full routed-MoE for one token from **stacked mlx-affine** `switch_mlp` packs:
/// top-k select over `scores_row + gate_bias`, weight by the RAW scores
/// (optionally top-k-normalized) × `scale`, sum the per-expert SwiGLU
/// contributions onto the (already-computed) `shared`-expert output. Mirrors the
/// GGUF `moe_one_token` selection exactly; only the expert compute differs
/// (mlx-affine `affine_expert_swiglu` vs GGUF block GEMM).
#[allow(clippy::too_many_arguments)]
pub fn affine_moe_token(
    scores_row: &[f32],
    x: &[f32],
    shared: &[f32],
    gate: &MlxPackedLinear,
    up: &MlxPackedLinear,
    down: &MlxPackedLinear,
    ne: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
    scale: f32,
    norm_topk: bool,
    gate_bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let mut order: Vec<(usize, f32)> = (0..ne)
        .map(|e| {
            let b = gate_bias.and_then(|b| b.get(e).copied()).unwrap_or(0.0);
            (e, scores_row[e] + b)
        })
        .collect();
    let kth = top_k.min(order.len());
    if kth > 0 && kth < order.len() {
        order.select_nth_unstable_by(kth - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        order[..kth]
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    let mut picks: Vec<(usize, f32)> = order
        .into_iter()
        .take(top_k)
        .map(|(e, _)| (e, scores_row[e]))
        .collect();
    if norm_topk {
        let sum: f32 = picks.iter().map(|(_, w)| *w).sum::<f32>().max(1e-12);
        for p in &mut picks {
            p.1 /= sum;
        }
    }
    let mut acc = shared.to_vec();
    for &(e, rw) in &picks {
        let down_out = affine_expert_swiglu(x, gate, up, down, e, ne, hidden, inter)?;
        let w = rw * scale;
        for o in 0..hidden {
            acc[o] += w * down_out[o];
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack `codes` (row-major `[rows, cols]`, each in `0..2^bits`) into the
    /// mlx uint32 layout: `bits`-wide values LSB-first, `32/bits` per u32,
    /// contiguous per row. Mirrors mlx-lm's `mx.quantize` bit packing so the
    /// test exercises the real `dequant_affine_f32` unpack path.
    fn pack_codes(codes: &[u32], bits: u32) -> Vec<u8> {
        let per_word = 32 / bits as usize;
        assert_eq!(
            codes.len() % per_word,
            0,
            "codes must be a whole number of u32 words"
        );
        let mut out = Vec::with_capacity(codes.len() / per_word * 4);
        for chunk in codes.chunks_exact(per_word) {
            let mut word = 0u32;
            for (j, &c) in chunk.iter().enumerate() {
                word |= (c & ((1 << bits) - 1)) << (j as u32 * bits);
            }
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// End-to-end: build a real affine-packed linear, and check
    /// `affine_matmul_bt` equals `dequant → plain GEMM` computed independently.
    #[test]
    fn affine_matmul_matches_reference() {
        let (out, inn, bits, gs) = (3usize, 64usize, 4u32, 64u32);
        let n_groups = inn / gs as usize;
        // Deterministic codes in [0,15], one scale/bias per (row, group).
        let codes: Vec<u32> = (0..out * inn).map(|i| (i % 13) as u32).collect();
        let w_q = pack_codes(&codes, bits);
        let scales: Vec<f32> = (0..out * n_groups).map(|i| 0.1 + 0.05 * i as f32).collect();
        let biases: Vec<f32> = (0..out * n_groups)
            .map(|i| -0.3 + 0.02 * i as f32)
            .collect();
        let p = MlxPackedLinear {
            w_q,
            scales: scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
            biases: biases.iter().flat_map(|v| v.to_le_bytes()).collect(),
            scheme: QuantScheme::MlxAffine {
                bits: bits as u8,
                group_size: gs,
            },
            out_shape: vec![out, inn],
        };

        // Independent reference: dequant via the shared primitive, GEMM by hand.
        let w_ref = dequant_affine_f32(&p.w_q, &scales, &biases, bits, gs, out, n_groups).unwrap();
        let seq = 2usize;
        let x: Vec<f32> = (0..seq * inn).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut y_ref = vec![0f32; seq * out];
        for t in 0..seq {
            for o in 0..out {
                let mut acc = 0.0f32;
                for i in 0..inn {
                    acc += x[t * inn + i] * w_ref[o * inn + i];
                }
                y_ref[t * out + o] = acc;
            }
        }

        let y = affine_matmul_bt(&x, &p, seq, out, inn).unwrap();
        assert_eq!(y.len(), y_ref.len());
        for (a, b) in y.iter().zip(&y_ref) {
            assert!((a - b).abs() < 1e-5, "affine matmul mismatch: {a} vs {b}");
        }
    }

    /// Build a stacked `[n_expert, out, in]` MLX-affine pack + return each
    /// expert's independently-dequantized `[out, in]` F32 reference.
    fn make_stack(
        out: usize,
        inn: usize,
        ne: usize,
        gs: u32,
        bits: u32,
        seed: usize,
    ) -> (MlxPackedLinear, Vec<Vec<f32>>) {
        let ng = inn / gs as usize;
        let (mut wq, mut sc, mut bi, mut refs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for e in 0..ne {
            let codes: Vec<u32> = (0..out * inn)
                .map(|i| ((i + e * 7 + seed) % 13) as u32)
                .collect();
            let scales: Vec<f32> = (0..out * ng).map(|i| 0.1 + 0.03 * (i + e) as f32).collect();
            let biases: Vec<f32> = (0..out * ng)
                .map(|i| -0.2 + 0.01 * (i + e) as f32)
                .collect();
            let w_q = pack_codes(&codes, bits);
            refs.push(dequant_affine_f32(&w_q, &scales, &biases, bits, gs, out, ng).unwrap());
            wq.extend(w_q);
            sc.extend(scales.iter().flat_map(|v| v.to_le_bytes()));
            bi.extend(biases.iter().flat_map(|v| v.to_le_bytes()));
        }
        let p = MlxPackedLinear {
            w_q: wq,
            scales: sc,
            biases: bi,
            scheme: QuantScheme::MlxAffine {
                bits: bits as u8,
                group_size: gs,
            },
            out_shape: vec![ne, out, inn],
        };
        (p, refs)
    }

    /// Stacked-`switch_mlp` MoE expert SwiGLU (glm/deepseek-style, mlx-affine) —
    /// `affine_expert_swiglu` for a chosen expert equals dequant→SwiGLU from that
    /// expert's independently-built weights. Validates `expert_slice` + the kernel.
    #[test]
    fn stacked_expert_swiglu_matches_reference() {
        let (ne, hidden, inter, bits, gs) = (3usize, 64usize, 64usize, 4u32, 32u32);
        let (gate, gref) = make_stack(inter, hidden, ne, gs, bits, 1);
        let (up, uref) = make_stack(inter, hidden, ne, gs, bits, 2);
        let (down, dref) = make_stack(hidden, inter, ne, gs, bits, 3);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
        let e = 2usize;

        // Reference from expert e's dequantized matrices.
        let mm = |x: &[f32], w: &[f32], out: usize, inn: usize| -> Vec<f32> {
            (0..out)
                .map(|o| (0..inn).map(|i| x[i] * w[o * inn + i]).sum())
                .collect()
        };
        let g = mm(&x, &gref[e], inter, hidden);
        let u = mm(&x, &uref[e], inter, hidden);
        let mid: Vec<f32> = (0..inter).map(|i| silu_f32(g[i]) * u[i]).collect();
        let want = mm(&mid, &dref[e], hidden, inter);

        let got = affine_expert_swiglu(&x, &gate, &up, &down, e, ne, hidden, inter).unwrap();
        assert_eq!(got.len(), hidden);
        for (a, b) in got.iter().zip(&want) {
            assert!(
                (a - b).abs() < 1e-4,
                "stacked expert SwiGLU mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rejects_non_affine_scheme() {
        let p = MlxPackedLinear {
            w_q: vec![0u8; 4],
            scales: vec![0u8; 4],
            biases: vec![0u8; 4],
            scheme: QuantScheme::MlxMxfp4 { group_size: 32 },
            out_shape: vec![1, 32],
        };
        assert!(dequant_linear(&p).is_err());
    }
}
