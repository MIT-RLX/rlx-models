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

//! `Glm5NextTextAttention` — DeepSeek-style multi-head latent attention behind
//! a DSA indexer. 11 of GLM-5.3-Flash's 45 layers.
//!
//! ```text
//!   q_resid = RMSNorm(q_a(x))                     [s, 1536]   ← also feeds the indexer
//!   q       = q_b(q_resid)                        [s, 64·256]
//!   k_pass  = RMSNorm(kv_a_mqa(x))                [s, 512]
//!   k[h]    = k_pass @ k_b[h]                     [s, 256]
//!   v[h]    = k_pass @ v_b[h]ᵀ                    [s, 256]
//!   out     = o_proj(SDPA(q, k, v, dsa_mask))
//! ```
//!
//! **This is pure NoPE.** `qk_rope_head_dim = 0`, so `kv_a_proj_with_mqa`
//! produces only the latent — there is no decoupled RoPE head to split off, and
//! no rotary embedding anywhere in the text model. Position information reaches
//! these layers only through the KDA layers below them.
//!
//! ## The two `kv_b` orientations
//!
//! GGUF splits HF's single `kv_b_proj` into `attn_k_b` and `attn_v_b`, and it
//! does **not** store them the same way round — a detail that silently
//! transposes the key projection if assumed:
//!
//! ```text
//!   attn_k_b   GGML ne {256, 512, 64}  → rlx [64, 512, 256] = [head, kv_lora, nope]
//!   attn_v_b   GGML ne {512, 256, 64}  → rlx [64, 256, 512] = [head, v_head, kv_lora]
//! ```
//!
//! `attn_k_b` is transposed by the converter so that GGML's contraction axis is
//! `nope` — llama.cpp absorbs the query into the latent rather than expanding
//! the key. Read as a plain matrix that same buffer is `[kv_lora, nope]`.
//!
//! ## Prefill expands, decode absorbs
//!
//! Prefill ([`emit_mla_attention`]) expands the latent to per-head keys and
//! values and hands them to the fused `Op::Attention`, which carries a single
//! head count for Q, K and V alike — absorbing there would leave K and V as one
//! shared 512-wide latent that would have to be broadcast to all 64 heads,
//! twice the traffic of the 256-wide expanded keys.
//!
//! Decode ([`emit_mla_decode`]) does the opposite, because there the cache is
//! what matters: keeping the latent is 64× smaller than keeping expanded keys
//! and values. See [`MlaCache`].
//!
//! The two are algebraically identical — `q · (k_pass @ k_b) = (q @ k_bᵀ) ·
//! k_pass` — so `tests/decode_equivalence.rs` checking them against each other
//! is also a check that both orientations above are read correctly.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::{linear, rms_norm};
use crate::indexer::{IndexerDims, emit_dsa_bias};

#[derive(Debug, Clone, Copy)]
pub struct MlaDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub eps: f32,
    pub seq: usize,
}

/// Emit one MLA + DSA layer for GGUF block `prefix` (`blk.{i}`) over
/// `[1, seq, hidden]`. Returns the branch output; the caller owns the residual.
pub fn emit_mla_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MlaDims,
    indexer: IndexerDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (s, h) = (d.seq, d.num_heads);
    let nope = d.qk_nope_head_dim;
    let vd = d.v_head_dim;
    let (si, hi, nopei, vdi) = (s as i64, h as i64, nope as i64, vd as i64);

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![si, d.hidden as i64])
    };

    // ── Q: low-rank, with the intermediate shared with the indexer ──
    let q_a = linear(emit, &format!("{prefix}.attn_q_a.weight"), x2d)?;
    let q_resid = rms_norm(
        emit,
        &format!("{prefix}.attn_q_a_norm"),
        q_a,
        d.q_lora_rank,
        d.eps,
    )?;
    let q = linear(emit, &format!("{prefix}.attn_q_b.weight"), q_resid)?; // [s, h*nope]

    // ── KV: the compressed latent only (NoPE ⇒ nothing to split off) ──
    let ckv = linear(emit, &format!("{prefix}.attn_kv_a_mqa.weight"), x2d)?;
    let k_pass = rms_norm(
        emit,
        &format!("{prefix}.attn_kv_a_norm"),
        ckv,
        d.kv_lora_rank,
        d.eps,
    )?; // [s, kv_lora]

    // Per-head up-projections, kept 3-D (`load_param`'s transpose is 2-D only).
    let k_b = emit.load_param(&format!("{prefix}.attn_k_b.weight"), false)?; // [h, kv_lora, nope]
    let v_b = emit.load_param(&format!("{prefix}.attn_v_b.weight"), false)?; // [h, v_head, kv_lora]

    // ── DSA selection (None ⇒ the selection is the causal mask) ──
    let bias = emit_dsa_bias(emit, prefix, hidden, q_resid, indexer)?;

    let attn = {
        let mut gb = HirMut::new(emit.hir());
        // Expand the latent to per-head keys/values, with the **batched operand
        // on the left**.
        //
        // The natural spelling is `mm(k_pass, k_b)`: `k_pass` is `[s, kv_lora]`
        // with no batch axis, and `matmul_shape` broadcasts it across `k_b`'s
        // leading head axis. Do not do that. A rank-2 lhs against a batched rhs
        // is *silently wrong* — rlx-cpu's dispatch only takes its batched-GEMM
        // path when **both** operands are rank ≥ 3, so this collapses to one
        // `Sgemm` against the rhs's first batch and every head after head 0 gets
        // stale memory. `tests/mm_broadcast.rs` pins the behaviour.
        //
        // Transposing puts the batched operand first, which is the direction
        // that is exercised everywhere (and the one decode already uses), and it
        // costs nothing: the alternative — materializing `k_pass` for all
        // `num_heads` — is 268 MB at `seq = 2048`.
        let k_pass_t = gb.transpose_(k_pass, vec![1, 0]); // [kv_lora, s]
        let k_b_t = gb.transpose_(k_b, vec![0, 2, 1]); // [h, nope, kv_lora]
        let keys = {
            let t = gb.mm(k_b_t, k_pass_t); // [h, nope, s]
            gb.transpose_(t, vec![0, 2, 1]) // [h, s, nope]
        };
        // `v_b` is already `[h, v_head, kv_lora]`, so it needs no transpose here.
        let values = {
            let t = gb.mm(v_b, k_pass_t); // [h, v_head, s]
            gb.transpose_(t, vec![0, 2, 1]) // [h, s, v_head]
        };

        // Pack heads into the last axis, the layout `Op::Attention` expects.
        let kf = {
            let t = gb.transpose_(keys, vec![1, 0, 2]); // [s, h, nope]
            gb.reshape_(t, vec![1, si, hi * nopei])
        };
        let vf = {
            let t = gb.transpose_(values, vec![1, 0, 2]);
            gb.reshape_(t, vec![1, si, hi * vdi])
        };
        let qf = gb.reshape_(q, vec![1, si, hi * nopei]);

        // The default score scale is qk_head_dim^-0.5, which with NoPE is
        // `qk_nope_head_dim^-0.5` — exactly the reference's `self.scaling`.
        let out = match bias {
            None => gb.attention_kind(
                qf,
                kf,
                vf,
                h,
                nope,
                MaskKind::Causal,
                Shape::new(&[1, s, h * vd], f),
            ),
            Some(bias) => {
                // `MaskKind::Bias` reads `[batch, num_heads, query_len,
                // key_len]`. The DSA selection is shared across heads, so the
                // indexer emits `[1, 1, s, s]` — broadcast it up rather than
                // handing the kernel a tensor it will index past the end of.
                let bias = gb.expand_(bias, vec![1, hi, si, si]);
                gb.attention_bias(qf, kf, vf, bias, h, nope, Shape::new(&[1, s, h * vd], f))
            }
        };
        gb.reshape_(out, vec![si, hi * vdi])
    };

    let out = linear(emit, &format!("{prefix}.attn_output.weight"), attn)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.reshape_(out, vec![1, si, d.hidden as i64]))
}

/// The MLA decode cache: the **compressed latent only**.
///
/// `rlx_ling::mla::MlaCache` stores expanded per-head keys and values. GLM-5.3-
/// Flash stores the latent instead, which is the whole point of caching MLA:
///
/// ```text
///   expanded   h·(qk + v) = 64·(256 + 256) = 32768 floats/token/layer
///   latent     kv_lora    =            512         floats/token/layer   → 64×
/// ```
///
/// At `cap = 2048` over the 11 MLA layers that is 46 MB instead of 2.9 GB.
///
/// The cost is that attention has to run **absorbed**: the query is projected
/// into latent space (`q · k_bᵀ`) and the output projected back out
/// (`· v_bᵀ`), so the scores contract `kv_lora = 512` rather than
/// `qk_nope_head_dim = 256` — 2× the score FLOPs per token. At decode's one
/// query that trade is overwhelmingly worth it, and it is exactly the layout
/// llama.cpp's converter already stores `attn_k_b` in.
///
/// Absorbed and un-absorbed are algebraically identical:
///
/// ```text
///   q · (k_pass @ k_b)  =  (q @ k_bᵀ) · k_pass
///   Σₜ pₜ·(k_passₜ @ v_bᵀ)  =  (Σₜ pₜ·k_passₜ) @ v_bᵀ
/// ```
///
/// so the prefill graph (un-absorbed, fused `Op::Attention`) and this one must
/// agree — `tests/decode_equivalence.rs` checks that they do.
#[derive(Debug, Clone, Copy)]
pub struct MlaCache {
    /// `[1, cap, kv_lora_rank]` — the RMSNormed `kv_a_mqa` latents.
    pub latent: HirNodeId,
    /// `[1, cap + 1]` — `1.0` valid, `0.0` ignored. The last slot is this token.
    pub mask: HirNodeId,
    pub cap: usize,
}

/// Emit one MLA **decode step** for a single new token.
///
/// Returns `(out [1, 1, hidden], latent_new [1, kv_lora])`; the caller stores
/// `latent_new` into the cache at the current position.
///
/// There is no indexer here. Decode is only built where DSA is dense (the
/// builder rejects `cap + 1 > index_topk`), so the validity mask is the whole
/// story — with one query there is nothing left to mask causally either.
///
/// The fused `Op::Attention` is not used: it carries a single head count for Q,
/// K and V alike, so the shared latent would have to be broadcast to all 64
/// heads, undoing exactly the saving the latent cache exists for. At one query
/// row the attention is two GEMVs against the cache regardless.
pub fn emit_mla_decode(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    cache: MlaCache,
    d: MlaDims,
) -> Result<(HirNodeId, HirNodeId)> {
    let (h, nope, vd, kvl) = (
        d.num_heads,
        d.qk_nope_head_dim,
        d.v_head_dim,
        d.kv_lora_rank,
    );
    let (hi, nopei, vdi, kvli) = (h as i64, nope as i64, vd as i64, kvl as i64);
    let len = cache.cap + 1;
    debug_assert_eq!(d.seq, 1, "emit_mla_decode processes exactly one token");

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![1, d.hidden as i64])
    };

    // ── Q / KV projections for the new token (identical to prefill) ──
    let q_a = linear(emit, &format!("{prefix}.attn_q_a.weight"), x2d)?;
    let q_resid = rms_norm(
        emit,
        &format!("{prefix}.attn_q_a_norm"),
        q_a,
        d.q_lora_rank,
        d.eps,
    )?;
    let q = linear(emit, &format!("{prefix}.attn_q_b.weight"), q_resid)?;
    let ckv = linear(emit, &format!("{prefix}.attn_kv_a_mqa.weight"), x2d)?;
    let latent_new = rms_norm(emit, &format!("{prefix}.attn_kv_a_norm"), ckv, kvl, d.eps)?; // [1, kv_lora]

    let k_b = emit.load_param(&format!("{prefix}.attn_k_b.weight"), false)?; // [h, kv_lora, nope]
    let v_b = emit.load_param(&format!("{prefix}.attn_v_b.weight"), false)?; // [h, v_head, kv_lora]
    let scale = crate::common::scalar(
        emit,
        &format!("{prefix}.mla.dec.scale"),
        (nope as f32).powf(-0.5),
    );
    let one = crate::common::scalar(emit, &format!("{prefix}.mla.dec.one"), 1.0);
    let penalty = crate::common::scalar(emit, &format!("{prefix}.mla.dec.neg"), -1.0e30);

    let attn = {
        let mut gb = HirMut::new(emit.hir());

        // Absorb the query into latent space: [h, 1, nope] @ [h, nope, kv_lora].
        let q3 = gb.reshape_(q, vec![hi, 1, nopei]);
        let k_b_t = gb.transpose_(k_b, vec![0, 2, 1]); // [h, nope, kv_lora]
        let q_lat = gb.mm(q3, k_b_t); // [h, 1, kv_lora]

        // [cache | this token] — appending at the end is safe because attention
        // is permutation-invariant over keys once the mask is applied, and NoPE
        // means position was never baked into the key.
        let new3 = gb.reshape_(latent_new, vec![1, 1, kvli]);
        let all = gb.concat_(vec![cache.latent, new3], 1); // [1, len, kv_lora]
        let all2 = gb.reshape_(all, vec![len as i64, kvli]);
        let all_t = gb.transpose_(all2, vec![1, 0]); // [kv_lora, len]

        // Scores contract the latent; `all_t` has no batch axis, so it
        // broadcasts across the head axis instead of being materialized per head.
        let scores = gb.mm(q_lat, all_t); // [h, 1, len]
        let scores = gb.mul(scores, scale);
        // Validity mask → additive bias: 0 where valid, -1e30 where not.
        let inv = gb.sub(one, cache.mask);
        let bias = gb.mul(inv, penalty); // [1, len]
        let bias = gb.reshape_(bias, vec![1, 1, len as i64]);
        let scores = gb.add(scores, bias);
        let probs = gb.sm(scores, -1);

        let out_lat = gb.mm(probs, all2); // [h, 1, kv_lora]
        let v_b_t = gb.transpose_(v_b, vec![0, 2, 1]); // [h, kv_lora, v_head]
        let out = gb.mm(out_lat, v_b_t); // [h, 1, v_head]
        gb.reshape_(out, vec![1, hi * vdi])
    };

    let out = linear(emit, &format!("{prefix}.attn_output.weight"), attn)?;
    let mut gb = HirMut::new(emit.hir());
    let out = gb.reshape_(out, vec![1, 1, d.hidden as i64]);
    let latent_new = gb.reshape_(latent_new, vec![1, kvli]);
    Ok((out, latent_new))
}
