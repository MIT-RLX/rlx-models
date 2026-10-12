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

//! Small shared emitters.

use anyhow::Result;
use rlx_flow::{Emit, GgufPackedBank, GgufPackedLinear};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// `x @ Wᵀ` for a GGUF-stored weight, packed when the source offers it.
///
/// Two paths, chosen by the [`rlx_flow::WeightSource`]:
///
/// * **Packed.** When the source hands out a GGUF quant blob for `key` (i.e.
///   the flow was built through `rlx_core::flow_bridge::PackedWeightLoaderSource`),
///   the U8 blob becomes a graph param and the projection a fused
///   `Op::DequantMatMul` — no f32 weight is ever materialized.
/// * **Dense.** Otherwise the weight loads as f32. The GGUF loader reverses
///   GGML's innermost-first dims, so it arrives in torch `[out, in]` order and
///   `transpose = true` turns it into the `[in, out]` operand `mm` contracts
///   against.
///
/// The packed blob is already `[out_dim, in_dim]` and `DequantMatMul`
/// transposes internally, so the two agree without the caller knowing which
/// ran. This mirrors `FlowCtx::linear`, which hand-rolled `HirMut` blocks like
/// this crate's cannot use directly.
pub fn linear(emit: &mut Emit<'_>, key: &str, x: HirNodeId) -> Result<HirNodeId> {
    // A projection already registered by an earlier stage: reuse the param
    // rather than re-`take_packed`-ing a consuming loader.
    if let Some(&(wq, out_dim, scheme)) = emit.state.packed_linears.get(key) {
        let mut gb = HirMut::new(emit.hir());
        return Ok(emit_dequant(&mut gb, x, wq, out_dim, scheme));
    }
    if let Some(packed) = emit.weights.take_packed(key)? {
        let GgufPackedLinear {
            w_q,
            scheme,
            out_dim,
            ..
        } = packed;
        // `\0q` keeps the blob's param name distinct from the f32 key.
        let wq_key = format!("{key}\0q");
        let wq = emit
            .hir()
            .param(&wq_key, Shape::new(&[w_q.len()], DType::U8));
        emit.state.typed_params.push((wq_key, w_q, DType::U8));
        emit.state
            .packed_linears
            .insert(key.to_string(), (wq, out_dim, scheme));
        let mut gb = HirMut::new(emit.hir());
        return Ok(emit_dequant(&mut gb, x, wq, out_dim, scheme));
    }
    let w = emit.load_param(key, true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// `DequantMatMul` with `x`'s shape, last dim replaced by `out_dim` — leading
/// dims are preserved, so one packed graph runs at any sequence length.
fn emit_dequant(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    wq: HirNodeId,
    out_dim: usize,
    scheme: QuantScheme,
) -> HirNodeId {
    let mut dims: Vec<usize> = gb
        .shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let last = dims.len() - 1;
    dims[last] = out_dim;
    let out = Shape::new(&dims, DType::F32);
    gb.0.dequant_matmul(x, wq, None, None, scheme, out)
}

/// A scalar `[1]` synthetic parameter.
pub fn scalar(emit: &mut Emit<'_>, name: &str, v: f32) -> HirNodeId {
    emit.synth_param(name, vec![v], Shape::new(&[1], DType::F32))
}

/// Load an RMSNorm gain, applying the `1 +` the reference folds into every
/// `Qwen4ExpTextRMSNorm`.
///
/// `offset` is [`crate::Qwen4ExpConfig::rms_norm_offset`]: the GGUF converter
/// already added the 1 (`modify_tensors` does it for the norms the inherited
/// rule misses, and the shared rule for the rest), so a GGUF loads the tensor
/// verbatim while a HF checkpoint needs the shift. Getting this backwards does
/// not fail — it scales every activation by roughly the wrong constant.
///
/// `ssm_norm` is NOT one of these. It is a `Qwen4ExpTextRMSNormGated`, whose
/// weight is initialized to ones and used directly; pass `offset = false` for
/// it whatever the checkpoint format.
pub fn rms_gamma(emit: &mut Emit<'_>, key: &str, width: usize, offset: bool) -> Result<HirNodeId> {
    if !offset {
        return emit.load_param(key, false);
    }
    let (data, _) = emit.weights.take(key, false)?;
    if data.len() != width {
        anyhow::bail!(
            "qwen4exp: norm gain `{key}` has {} elements, expected {width}",
            data.len()
        );
    }
    let shifted: Vec<f32> = data.into_iter().map(|w| w + 1.0).collect();
    Ok(emit.synth_param(key, shifted, Shape::new(&[width], DType::F32)))
}

/// RMSNorm over the last axis with a GGUF gain and no bias.
pub fn rms_norm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    width: usize,
    eps: f32,
    offset: bool,
) -> Result<HirNodeId> {
    let g = rms_gamma(emit, &format!("{key}.weight"), width, offset)?;
    let zb = emit.synth_zeros(&format!("{key}.zb"), width);
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

/// `x · rsqrt(mean(x²) + eps)` over the last axis, with no learned gain.
///
/// `Op::RmsNorm` carries the gain and bias as operands, so a unit gain and a
/// zero bias reproduce the parameter-free form exactly while still riding the
/// backends' fused RMSNorm kernels.
pub fn unit_rms_norm(
    emit: &mut Emit<'_>,
    name: &str,
    x: HirNodeId,
    width: usize,
    eps: f32,
) -> HirNodeId {
    let ones = emit.synth_param(
        &format!("{name}.ones"),
        vec![1.0; width],
        Shape::new(&[width], DType::F32),
    );
    let zeros = emit.synth_zeros(&format!("{name}.zb"), width);
    let mut gb = HirMut::new(emit.hir());
    gb.rms_norm(x, ones, zeros, eps)
}

/// The **grouped** RMSNorm `Qwen4ExpTextRMSNorm(dim = groups·width,
/// group_size = width)`.
///
/// Normalization reduces over one group (`width`), but the gain is the full
/// `groups·width` vector — one gamma per (stream, channel). `Op::RmsNorm`'s
/// gain broadcasts over the last axis only, so this is a unit-gain norm on a
/// `[.., groups, width]` view followed by an explicit multiply, exactly as
/// llama.cpp's `ggml_mul(ggml_rms_norm(x), w_norm)` with `w_norm` loaded as
/// `[n_embd, hc]`.
///
/// `x` is `[.., groups · width]` (any leading dims); the result has the same
/// shape.
pub fn grouped_rms_norm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    groups: usize,
    width: usize,
    eps: f32,
    offset: bool,
) -> Result<HirNodeId> {
    let gamma = rms_gamma(emit, &format!("{key}.weight"), groups * width, offset)?;
    let leading: Vec<i64> = {
        let gb = HirMut::new(emit.hir());
        let dims = gb.shape(x).dims();
        dims[..dims.len() - 1]
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect()
    };
    let mut grouped_shape = leading.clone();
    grouped_shape.push(groups as i64);
    grouped_shape.push(width as i64);
    let flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(x, grouped_shape.clone())
    };
    let normed = unit_rms_norm(emit, &format!("{key}.gn"), flat, width, eps);
    let mut gb = HirMut::new(emit.hir());
    let g = gb.reshape_(gamma, vec![groups as i64, width as i64]);
    let scaled = gb.mul(normed, g);
    let mut flat_shape = leading;
    flat_shape.push((groups * width) as i64);
    Ok(gb.reshape_(scaled, flat_shape))
}

/// `sigmoid(x)` with `x`'s shape.
pub fn sigmoid(gb: &mut HirMut<'_>, x: HirNodeId) -> HirNodeId {
    let s = gb.shape(x).clone();
    gb.add_node(Op::Activation(Activation::Sigmoid), vec![x], s)
}

/// An activation node with `x`'s shape.
pub fn act(gb: &mut HirMut<'_>, kind: Activation, x: HirNodeId) -> HirNodeId {
    let s = gb.shape(x).clone();
    gb.add_node(Op::Activation(kind), vec![x], s)
}

/// A packed MoE expert bank registered as a graph param, if the source has one.
///
/// The grouped sibling of [`linear`]: returns the U8 blob's node plus the dims
/// `Op::DequantGroupedMatMul` needs. `None` means the caller should fall back
/// to an F32 `GroupedMatMul` over a dequantized bank.
///
/// GGUF's `[experts, out, in]` order is already the op's slab layout, so unlike
/// the F32 path there is no `[E, N, K] → [E, K, N]` transpose — which also
/// avoids constant-folding a second copy of the bank into the arena.
pub fn packed_bank(
    emit: &mut Emit<'_>,
    key: &str,
) -> Result<Option<(HirNodeId, QuantScheme, usize)>> {
    if let Some(&(wq, out_dim, scheme)) = emit.state.packed_linears.get(key) {
        return Ok(Some((wq, scheme, out_dim)));
    }
    let Some(bank) = emit.weights.take_packed_bank(key)? else {
        return Ok(None);
    };
    let GgufPackedBank {
        w_q,
        scheme,
        out_dim,
        ..
    } = bank;
    let wq_key = format!("{key}\0q");
    let wq = emit
        .hir()
        .param(&wq_key, Shape::new(&[w_q.len()], DType::U8));
    emit.state.typed_params.push((wq_key, w_q, DType::U8));
    // Reuse `packed_linears` as the registry: a bank is keyed the same way and
    // only ever read back as (node, out_dim, scheme).
    emit.state
        .packed_linears
        .insert(key.to_string(), (wq, out_dim, scheme));
    Ok(Some((wq, scheme, out_dim)))
}

/// Runtime tensors a layer needs when its input is a *chunk* of a longer
/// sequence rather than the whole thing.
///
/// Both are `[1, …]` graph inputs, so one compiled graph serves every chunk.
/// A fully-real chunk can leave both `None`, which is bit-identical to the
/// unpadded path.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkInputs {
    /// `[1, seq]` — `1.0` real, `0.0` padding.
    ///
    /// Only the recurrent pieces read this, and only to keep their *exported
    /// state* from advancing over padding: a GatedDeltaNet step with
    /// `gate = 0` and `beta = 0` is an exact no-op, because the recurrence is
    /// `S ← S·exp(g) + k ⊗ ((v − Sᵀk)·β)`. The per-token outputs at real
    /// positions never need it — the convolution and the scan are both causal,
    /// so with **right** padding a real position has only real history.
    pub pad_mask: Option<HirNodeId>,
    /// `[1, 1]` — `real_len − seq`, so `≤ 0`.
    ///
    /// Where the exported convolution window ends. The obvious
    /// `narrow(padded, seq, hist)` takes the *last* `hist` positions, which are
    /// padding when the chunk is short; the next chunk would then resume its
    /// convolution from tokens the prompt never contained. An unbound input
    /// reads `0` and reproduces that narrow exactly.
    pub conv_shift: Option<HirNodeId>,
}

/// A depthwise causal 1-D convolution that **carries** its left context across
/// calls, as a sum of shifted copies.
///
/// `out[c, t] = Σ_j w[j, c] · padded[c, t + j·?]` — precisely,
/// `tap j` reads `(k − 1 − j) · dilation` positions back, with the carried
/// `state` supplying everything before `x`:
///
/// ```text
///   padded = [state (hist) | x (seq)]          hist = (k − 1) · dilation
///   out[t] = Σ_j w[j] · padded[hist − (k−1−j)·dilation + t]
///   next   = padded[seq ..]                    the trailing `hist` positions
/// ```
///
/// Used by the decode path, where `seq = 1` and `k` is 4, so the shifted-copy
/// form costs four broadcast multiplies. The stateless prefill path uses a
/// single grouped `Op::Conv` instead — same function, one kernel instead of
/// `2k − 1` elementwise nodes over a `[1, seq, 10240]` tensor.
///
/// A zero `state` reproduces the stateless path's causal zero-pad exactly,
/// which is what makes a chunked prefill agree with a single-shot one.
#[allow(clippy::too_many_arguments)]
pub fn carried_conv1d(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,     // [1, seq, channels]
    state: HirNodeId, // [1, hist, channels]
    seq: usize,
    channels: usize,
    k: usize,
    dilation: usize,
    shift: Option<HirNodeId>,
) -> Result<(HirNodeId, HirNodeId)> {
    let hist = (k - 1) * dilation;
    let w = emit.load_param(key, false)?;
    // The window-base indices, hoisted before the graph borrow.
    let base = shift.map(|_| {
        let data: Vec<f32> = (0..hist).map(|j| (seq + j) as f32).collect();
        emit.synth_param(
            &format!("{key}.window_base"),
            data,
            Shape::new(&[1, hist], DType::F32),
        )
    });
    let mut gb = HirMut::new(emit.hir());
    let w2 = gb.reshape_(w, vec![channels as i64, k as i64]);
    let padded = gb.concat_(vec![state, x], 1); // [1, hist + seq, channels]
    let mut acc: Option<HirNodeId> = None;
    for j in 0..k {
        let tap = gb.narrow_(w2, 1, j, 1);
        let tap = gb.reshape_(tap, vec![1, 1, channels as i64]);
        let start = hist - (k - 1 - j) * dilation;
        let slice = gb.narrow_(padded, 1, start, seq);
        let term = gb.mul(slice, tap);
        acc = Some(match acc {
            Some(a) => gb.add(a, term),
            None => term,
        });
    }
    let next = match (base, shift) {
        // Gather the window ending at the last REAL position, rather than the
        // last compiled one.
        (Some(base), Some(shift)) => {
            let idx = gb.add(base, shift);
            let w = gb.gather_(padded, idx, 1);
            gb.reshape_(w, vec![1, hist as i64, channels as i64])
        }
        _ => gb.narrow_(padded, 1, seq, hist),
    };
    Ok((acc.expect("conv kernel >= 1"), next))
}
