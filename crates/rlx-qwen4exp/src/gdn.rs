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

//! GatedDeltaNet linear attention — `Qwen4ExpTextGatedDeltaNet`.
//!
//! Three of every four layers. Inherited verbatim from Qwen3.5 apart from the
//! output gate, so the shape of this emitter follows `rlx_qwen35`'s
//! `build_linear_layer`:
//!
//! ```text
//!   qkv   = x @ attn_qkvᵀ                           [s, 2·key_dim + value_dim]
//!   z     = x @ attn_gateᵀ                           [s, value_dim]
//!   β     = σ(x @ ssm_betaᵀ)                         [s, n_v]
//!   g     = ssm_a · softplus(x @ ssm_alphaᵀ + dt_bias)
//!
//!   q,k,v = split(silu(depthwise_conv1d(qkv)))
//!   q,k   = l2norm(q), l2norm(k)                     per head, over the state
//!   q,k   = tile(q, n_v/n_k), tile(k, n_v/n_k)
//!   y     = GatedDeltaNet(q, k, v, g, β)             [1, s, n_v, state]
//!   out   = (RMSNorm(y) · σ(z)) @ ssm_outᵀ
//! ```
//!
//! Four things are easy to get wrong, three of them because this crate reads
//! the **GGUF** spelling rather than HF's:
//!
//! * **The output gate is a sigmoid, not SiLU.** `output_gate_type = sigmoid`
//!   is the one numerical difference from Qwen3.5's GDN
//!   (`qwen4exp.cpp build_norm_gated` hardcodes it). There is no GGUF key for
//!   it; [`crate::Qwen4ExpConfig::output_gate_sigmoid`] carries it.
//! * **The GQA expansion TILES, it does not interleave.** The reference does
//!   `repeat_interleave`, because HF stores the 48 value heads grouped by their
//!   16 key heads. The GGUF converter permutes them to tiled order
//!   (`_LinearAttentionVReorderBase`, applied to `attn_qkv`'s v rows,
//!   `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`,
//!   `ssm_conv1d`'s v channels and `ssm_out`'s columns) so that a whole-block
//!   repeat is correct and cheap. Interleaving here agrees only for v-heads 0
//!   and 47 and flips the sign of every head in between — the bug
//!   `rlx-qwen35` shipped and fixed.
//! * **`ssm_a` is already `−exp(A_log)`.** The converter precomputes it, so the
//!   decay is `ssm_a · softplus(·)`. Applying another `−exp()` gives a ~24×
//!   too-aggressive decay, which collapses the recurrent state in one step and
//!   turns the layer into a memoryless prompt echo.
//! * **`Op::GatedDeltaNet` already scales the query by `state_size^-0.5`.**
//!   The reference applies that scale inside the delta rule too, so a
//!   pre-scaled query would square it.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::{
    ChunkInputs, act, carried_conv1d, linear, rms_gamma, scalar, sigmoid, unit_rms_norm,
};
use crate::config::Qwen4ExpConfig;

/// Static shape/hyper-parameters of one GatedDeltaNet layer.
#[derive(Debug, Clone, Copy)]
pub struct GdnDims {
    pub hidden: usize,
    /// `linear_num_key_heads`.
    pub n_k_heads: usize,
    /// `linear_num_value_heads`.
    pub n_v_heads: usize,
    /// `linear_key_head_dim` == `linear_value_head_dim` == the scan's state.
    pub state: usize,
    pub conv_kernel: usize,
    /// `false` selects Qwen3.5's SiLU output gate.
    pub gate_sigmoid: bool,
    pub eps: f32,
    pub seq: usize,
}

/// The recurrent state a GatedDeltaNet layer carries between calls.
///
/// Both halves are graph **inputs**, not params updated in place. `Op::GatedDeltaNet`
/// documents that CPU/Metal/wgpu honour a carried state buffer, but MLX
/// substitutes the new state into its per-evaluation environment where it does
/// not survive to the next `run()` — so a param-bound state silently freezes at
/// its initial value there. See [`crate::decode::ScanState`] for the opt-in.
#[derive(Debug, Clone, Copy)]
pub struct GdnState {
    /// `[1, conv_kernel - 1, conv_dim]` — the previous tokens' **pre-conv**
    /// fused `attn_qkv` projection. Pre-conv, not post: the convolution is
    /// causal over its own input, so caching its output would double-filter.
    pub conv: HirNodeId,
    /// `[1, n_v_heads, state, state]` — the delta-rule's recurrent matrix.
    pub scan: HirNodeId,
}

/// What a state-carrying GatedDeltaNet call hands back.
#[derive(Debug, Clone, Copy)]
pub struct GdnStateOut {
    pub conv: HirNodeId,
    /// The post-update scan state, materialized so a discrete-GPU read of it is
    /// ordered after the scan.
    pub scan: HirNodeId,
}

impl GdnDims {
    /// Derive from the model config, for a step of `seq` positions.
    ///
    /// There is exactly one statement of how a config becomes these dimensions, so
    /// prefill, chunked prefill, the decode step, the MTP block and the tests
    /// cannot disagree about what model they are building. They used to hold four
    /// copies of it.
    pub fn from_config(cfg: &Qwen4ExpConfig, seq: usize) -> Self {
        GdnDims {
            hidden: cfg.hidden_size,
            n_k_heads: cfg.n_k_heads(),
            n_v_heads: cfg.n_v_heads(),
            state: cfg.ssm_state_size,
            conv_kernel: cfg.ssm_conv_kernel,
            gate_sigmoid: cfg.output_gate_sigmoid,
            eps: cfg.rms_norm_eps,
            seq,
        }
    }
    pub fn key_dim(&self) -> usize {
        self.state * self.n_k_heads
    }
    pub fn value_dim(&self) -> usize {
        self.state * self.n_v_heads
    }
    /// `attn_qkv`'s output width, and the convolution's channel count.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// `x · rsqrt(Σ x² + eps)` over the last axis.
///
/// Spelled as llama.cpp's `build_gdn_l2_norm` does — a unit-gain RMSNorm at
/// `eps/n` scaled by `n^-0.5`, which is algebraically the FLA `l2norm` the
/// reference calls with `eps = 1e-6` *and* rides the backends' fused RMSNorm
/// kernel. Note this is **not** `x / max(‖x‖, eps)`: the epsilon is inside the
/// square root.
fn l2_norm(emit: &mut Emit<'_>, name: &str, x: HirNodeId, n: usize, eps: f32) -> HirNodeId {
    let normed = unit_rms_norm(emit, name, x, n, eps / n as f32);
    let inv = scalar(emit, &format!("{name}.inv_sqrt_n"), 1.0 / (n as f32).sqrt());
    let mut gb = HirMut::new(emit.hir());
    gb.mul(normed, inv)
}

/// Depthwise causal 1-D convolution over `[1, seq, channels]`.
///
/// Lowered as `Op::Conv` on an NCHW `[1, C, L, 1]` view with a `[k, 1]` kernel
/// and `groups = C`, the layout every backend already has a grouped-conv path
/// for. `ssm_conv1d.weight` arrives as `[channels, k]` (GGUF stores it
/// innermost-first as `[k, channels]`; the loader reverses the dims and leaves
/// the bytes alone), which is exactly `[C, 1, k, 1]` reshaped.
fn depthwise_causal_conv1d(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    channels: usize,
    seq: usize,
    k: usize,
) -> Result<HirNodeId> {
    let w = emit.load_param(key, false)?;
    let pad = emit.synth_zeros(&format!("{key}.causal_pad"), (k - 1) * channels);
    let mut gb = HirMut::new(emit.hir());

    let pad = gb.reshape_(pad, vec![1, (k - 1) as i64, channels as i64]);
    let padded = gb.concat_(vec![pad, x], 1);
    let width = seq + k - 1;

    let bcw = gb.transpose_(padded, vec![0, 2, 1]);
    let nchw = gb.reshape_(bcw, vec![1, channels as i64, width as i64, 1]);
    let w4 = gb.reshape_(w, vec![channels as i64, 1, k as i64, 1]);
    let conv = gb.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: channels,
        },
        vec![nchw, w4],
        Shape::new(&[1, channels, seq, 1], DType::F32),
    );
    let bcs = gb.reshape_(conv, vec![1, channels as i64, seq as i64]);
    Ok(gb.transpose_(bcs, vec![0, 2, 1]))
}

/// Repeat a `[1, seq, in_heads, state]` tensor to `n_v_heads` by **tiling** the
/// whole head block.
///
/// `[h0 … h15, h0 … h15, h0 … h15]`, so value head `h` pairs with key head
/// `h % in_heads`. See the module docs for why this is not an interleave, and
/// `tests/gqa_tile.rs` for the check that it is not.
fn tile_heads(gb: &mut HirMut<'_>, x: HirNodeId, factor: usize) -> HirNodeId {
    if factor <= 1 {
        return x;
    }
    // The repeat tiles the entire block, so the concat is literally `[x; f]` —
    // no per-head narrows, which would turn every layer's q and k into a
    // 48-input concat.
    gb.concat_(vec![x; factor], 2)
}

/// Emit the linear-attention branch for GGUF block `prefix` over
/// `[1, seq, hidden]`, returning `[1, seq, hidden]`.
///
/// Stateless: the convolution zero-pads on the left and the scan starts from a
/// zero matrix, which is the whole-prompt prefill. For a continued prefill or
/// single-token decode use [`emit_gdn_with_state`].
pub fn emit_gdn(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, d: GdnDims) -> Result<HirNodeId> {
    let (out, _) = emit_gdn_with_state(emit, prefix, x, d, None, ChunkInputs::default())?;
    Ok(out)
}

/// [`emit_gdn`] carrying the recurrent state across calls.
///
/// `None` is the stateless path, bit-identical to [`emit_gdn`]. `Some(state)`
/// prepends the carried convolution window and resumes the scan, and returns
/// the updated state alongside the output — so stepping this with `seq = 1`
/// reproduces a prefill of the same tokens. That equivalence is the test that
/// matters here, because the two paths spell the convolution differently (one
/// grouped `Op::Conv` versus a sum of shifted copies) and the scan differently
/// (`gated_delta_net` versus `gated_delta_net_carry`).
pub fn emit_gdn_with_state(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: GdnDims,
    state: Option<GdnState>,
    chunk: ChunkInputs,
) -> Result<(HirNodeId, Option<GdnStateOut>)> {
    let f = DType::F32;
    let (s, si) = (d.seq, d.seq as i64);
    let (nk, nv, n) = (d.n_k_heads, d.n_v_heads, d.state);
    let (key_dim, value_dim, conv_dim) = (d.key_dim(), d.value_dim(), d.conv_dim());

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(x, vec![si, d.hidden as i64])
    };

    let qkv = linear(emit, &format!("{prefix}.attn_qkv.weight"), x2d)?;
    let z = linear(emit, &format!("{prefix}.attn_gate.weight"), x2d)?;
    let beta_pre = linear(emit, &format!("{prefix}.ssm_beta.weight"), x2d)?;
    let alpha = linear(emit, &format!("{prefix}.ssm_alpha.weight"), x2d)?;

    // `ssm_a` is the negative decay coefficient the converter precomputed as
    // `-exp(A_log)`; `ssm_dt.bias` is the reference's `dt_bias`. Both are
    // per-value-head and already in tiled order.
    let ssm_a = emit.load_param(&format!("{prefix}.ssm_a"), false)?;
    let dt_bias = emit.load_param(&format!("{prefix}.ssm_dt.bias"), false)?;

    let (gate_g, beta) = {
        let mut gb = HirMut::new(emit.hir());
        let biased = gb.add(alpha, dt_bias);
        let sp = act(&mut gb, Activation::Softplus, biased);
        let g = gb.mul(ssm_a, sp);
        let b = sigmoid(&mut gb, beta_pre);
        let (g, b) = match chunk.pad_mask {
            // `gate = 0` and `beta = 0` make a scan step an exact no-op, so the
            // exported state is the state after the last REAL token. Harmless
            // for the per-token outputs, which are causal anyway.
            Some(keep) => {
                let keep = gb.reshape_(keep, vec![si, 1]);
                (gb.mul(g, keep), gb.mul(b, keep))
            }
            None => (g, b),
        };
        (
            gb.reshape_(g, vec![1, si, nv as i64]),
            gb.reshape_(b, vec![1, si, nv as i64]),
        )
    };

    let qkv3 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(qkv, vec![1, si, conv_dim as i64])
    };
    let (conv, next_conv) = match state {
        Some(st) => {
            let (c, next) = carried_conv1d(
                emit,
                &format!("{prefix}.ssm_conv1d.weight"),
                qkv3,
                st.conv,
                s,
                conv_dim,
                d.conv_kernel,
                1,
                chunk.conv_shift,
            )?;
            (c, Some(next))
        }
        None => (
            depthwise_causal_conv1d(
                emit,
                &format!("{prefix}.ssm_conv1d.weight"),
                qkv3,
                conv_dim,
                s,
                d.conv_kernel,
            )?,
            None,
        ),
    };

    let (q_part, k_part, v_part) = {
        let mut gb = HirMut::new(emit.hir());
        let silu = gb.silu(conv);
        (
            gb.narrow_(silu, 2, 0, key_dim),
            gb.narrow_(silu, 2, key_dim, key_dim),
            gb.narrow_(silu, 2, 2 * key_dim, value_dim),
        )
    };

    let (q4, k4, v4) = {
        let mut gb = HirMut::new(emit.hir());
        (
            gb.reshape_(q_part, vec![1, si, nk as i64, n as i64]),
            gb.reshape_(k_part, vec![1, si, nk as i64, n as i64]),
            gb.reshape_(v_part, vec![1, si, nv as i64, n as i64]),
        )
    };

    let q_l2 = l2_norm(emit, &format!("{prefix}.gdn.q_l2"), q4, n, d.eps);
    let k_l2 = l2_norm(emit, &format!("{prefix}.gdn.k_l2"), k4, n, d.eps);

    let (scan, next_scan) = {
        let mut gb = HirMut::new(emit.hir());
        let factor = nv / nk;
        let q_rep = tile_heads(&mut gb, q_l2, factor);
        let k_rep = tile_heads(&mut gb, k_l2, factor);
        let out_shape = Shape::new(&[1, s, nv, n], f);
        match state {
            Some(st) => {
                // No pre-scale on q: the op applies `state^-0.5` itself.
                let y =
                    gb.gated_delta_net_carry(q_rep, k_rep, v4, gate_g, beta, st.scan, n, out_shape);
                // Materialize the post-update state into a buffer ordered after
                // the scan. Exporting the raw input node is unreliable on
                // discrete GPUs: a device-to-host read of an Input-as-output can
                // observe the pre-upload zeros even when the op mutated the
                // arena in place (unified memory hides this). `scan · (1 + 0·Σy)`
                // is a broadcast multiply that depends on `y`, and the compiler
                // folds it away — so it costs no thunk and keeps the ordering.
                let total = gb.sum(y, vec![0, 1, 2, 3], false);
                let zero = scalar_in(&mut gb, 0.0);
                let one = scalar_in(&mut gb, 1.0);
                let dep = gb.mul(total, zero);
                let keep = gb.add(one, dep);
                (y, Some(gb.mul(st.scan, keep)))
            }
            None => (
                gb.gated_delta_net(q_rep, k_rep, v4, gate_g, beta, n, out_shape),
                None,
            ),
        }
    };

    // Gated norm: RMSNorm over the state axis, then the output gate. `ssm_norm`
    // is a `RMSNormGated` weight — initialized to ones and used as-is, so it
    // never takes the `1 +` the other norms do.
    let ssm_norm = rms_gamma(emit, &format!("{prefix}.ssm_norm.weight"), n, false)?;
    let ssm_zb = emit.synth_zeros(&format!("{prefix}.ssm_norm.zb"), n);
    let gate_kind = if d.gate_sigmoid {
        Activation::Sigmoid
    } else {
        Activation::Silu
    };

    let flat = {
        let mut gb = HirMut::new(emit.hir());
        let normed = gb.rms_norm(scan, ssm_norm, ssm_zb, d.eps);
        let z4 = gb.reshape_(z, vec![1, si, nv as i64, n as i64]);
        let g = act(&mut gb, gate_kind, z4);
        let gated = gb.mul(normed, g);
        gb.reshape_(gated, vec![si, value_dim as i64])
    };

    let out = linear(emit, &format!("{prefix}.ssm_out.weight"), flat)?;
    let mut gb = HirMut::new(emit.hir());
    let out = gb.reshape_(out, vec![1, si, d.hidden as i64]);
    let st_out = match (next_conv, next_scan) {
        (Some(conv), Some(scan)) => Some(GdnStateOut { conv, scan }),
        _ => None,
    };
    Ok((out, st_out))
}

/// A `[1]` constant built directly on the graph, for use inside a `HirMut`
/// borrow where `Emit` (and therefore `synth_param`) is unavailable.
fn scalar_in(gb: &mut HirMut<'_>, v: f32) -> HirNodeId {
    gb.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}
