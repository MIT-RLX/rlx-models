// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// DAC encoder/decoder as rlx-runtime HIR graphs. The convolutional networks
// (which are the bulk of the compute) run natively on every rlx backend
// (cpu/metal/mlx/cuda/rocm/wgpu/vulkan) via a `Session`. The RVQ codebook
// search stays host-side (see `quantize.rs`) — it is a tiny argmin and is
// inherently control-flow heavy.
//
// Layout convention: a `[C, T]` activation is carried as an NCHW tensor
// `[1, C, T, 1]` — the length lives in H and W is the singleton axis, with a
// `[k, 1]` kernel. This is the canonical 1D-conv form rlx's importer emits, and
// the one every backend's `Conv`/`ConvTranspose2d` lowering agrees on: MLX reads
// the length-axis stride/pad/dilation from index 0, so the length MUST sit in H.

use crate::layers::{
    Decoder, DecoderBlock, Encoder, EncoderBlock, ResidualUnit, WnConv1d, WnConvTranspose1d,
};
use anyhow::{Context, Result};
use ndarray::Array2;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

const F32: DType = DType::F32;

/// Accumulates graph params (name → row-major f32 data) while building.
struct Ctx<'a, 'b> {
    g: &'a mut HirMut<'b>,
    params: NamedTensors,
    next: usize,
}

impl<'a, 'b> Ctx<'a, 'b> {
    fn new(g: &'a mut HirMut<'b>) -> Self {
        Self {
            g,
            params: Vec::new(),
            next: 0,
        }
    }

    fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let name = format!("w{}", self.next);
        self.next += 1;
        let n: usize = shape.iter().product();
        debug_assert_eq!(
            data.len(),
            n,
            "param {name} shape {shape:?} vs len {}",
            data.len()
        );
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.push((name, data));
        id
    }

    fn expand(&mut self, id: HirNodeId, target: &[usize]) -> HirNodeId {
        let shape = Shape::new(target, F32);
        let tgt_i64: Vec<i64> = target.iter().map(|&d| d as i64).collect();
        self.g.add_node(
            Op::Expand {
                target_shape: tgt_i64,
            },
            vec![id],
            shape,
        )
    }

    /// `[1, C, T, 1]` bias-add from a `[C]` bias.
    fn add_bias(&mut self, y: HirNodeId, bias: &[f32], c: usize, t: usize) -> HirNodeId {
        let b = self.param(bias.to_vec(), &[1, c, 1, 1]);
        let be = self.expand(b, &[1, c, t, 1]);
        self.g.add(y, be)
    }

    /// Symmetric zero-pad of `pad` columns on each side of the length (H) axis.
    /// Done explicitly (concat of zero tensors) rather than via the conv's own
    /// padding arg, because backends disagree on how they pad the singleton
    /// layout — MLX, for one, reads the padding from the wrong axis.
    fn zero_pad_len(&mut self, x: HirNodeId, c: usize, t: usize, pad: usize) -> (HirNodeId, usize) {
        if pad == 0 {
            return (x, t);
        }
        let zl = self.param(vec![0.0; c * pad], &[1, c, pad, 1]);
        let zr = self.param(vec![0.0; c * pad], &[1, c, pad, 1]);
        let y = self.g.concat_(vec![zl, x, zr], 2);
        (y, t + 2 * pad)
    }

    /// Zero-insert `s-1` samples between each length step (input dilation), so
    /// `[1, C, T, 1]` → `[1, C, (T-1)·s+1, 1]`. Implemented with concat+reshape
    /// so it lowers to ops every backend supports.
    fn inflate_len(&mut self, x: HirNodeId, c: usize, t: usize, s: usize) -> (HirNodeId, usize) {
        if s <= 1 {
            return (x, t);
        }
        let z = self.param(vec![0.0; c * t * (s - 1)], &[1, c, t, s - 1]);
        let cat = self.g.concat_(vec![x, z], 3); // [1, C, T, s]
        let resh = self.g.reshape_(cat, vec![1, c as i64, (t * s) as i64, 1]);
        let lu = (t - 1) * s + 1;
        (self.g.narrow_(resh, 2, 0, lu), lu)
    }
}

fn weight_data(w: &ndarray::Array3<f32>) -> Vec<f32> {
    w.as_standard_layout().iter().copied().collect()
}

/// Emit a `Conv1d` (weight `[C_out, C_in, K]`) on a `[1, x_c, x_t, 1]` tensor.
/// Returns `(node, c_out, t_out)`.
fn conv1d(ctx: &mut Ctx, x: HirNodeId, x_t: usize, conv: &WnConv1d) -> (HirNodeId, usize, usize) {
    let (c_out, c_in, k) = conv.weight.dim();
    let stride = if conv.same_length { 1 } else { conv.stride };
    let dil = conv.dilation.max(1);
    let (xp, t_pad) = ctx.zero_pad_len(x, c_in, x_t, conv.pad);
    let t_out = t_pad.saturating_sub(dil * (k - 1) + 1) / stride + 1;
    let w = ctx.param(weight_data(&conv.weight), &[c_out, c_in, k, 1]);
    let out_shape = Shape::new(&[1, c_out, t_out, 1], F32);
    let mut y = ctx.g.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![stride, 1],
            padding: vec![0, 0],
            dilation: vec![dil, 1],
            groups: 1,
        },
        vec![xp, w],
        out_shape,
    );
    if let Some(b) = &conv.bias {
        y = ctx.add_bias(y, b.as_slice().unwrap(), c_out, t_out);
    }
    (y, c_out, t_out)
}

/// Emit `ConvTranspose1d` (weight `[C_in, C_out, K]`). For the DAC/TSAC upsamplers
/// (`K == 2·stride`) this uses a **polyphase** decomposition — `stride` regular
/// 2-tap convs (one per output phase) interleaved — which touches only real input
/// samples. The generic case falls back to zero-insert + full conv. Both lower to
/// ops every backend implements (wgpu/coreml ship no native `ConvTranspose`) and
/// are **bit-identical**: the polyphase phases compute exactly the nonzero terms
/// the inflated conv sums, and IEEE-754 add is commutative so the inserted `+0`s
/// don't change the result. Polyphase avoids the inflated tensor (≈`stride`× more
/// MACs, almost all against zeros) — the wgpu/Metal decode bottleneck.
/// Returns `(node, c_out, t_out)`.
fn conv_transpose1d(
    ctx: &mut Ctx,
    x: HirNodeId,
    x_t: usize,
    conv: &WnConvTranspose1d,
) -> (HirNodeId, usize, usize) {
    let (c_in, c_out, k) = conv.weight.dim();
    let stride = conv.stride;
    if stride > 1 && k == 2 * stride {
        return conv_transpose1d_polyphase(ctx, x, x_t, conv);
    }
    let pad = conv.pad;

    // ConvTranspose1d(stride s) ≡ conv(stride 1) of the s-inflated input, padded
    // by (k-1) on each side, with the kernel flipped and channels swapped:
    //   W[oc, ic, j] = weight[ic, oc, k-1-j].
    let mut wflip = vec![0f32; c_out * c_in * k];
    for oc in 0..c_out {
        for ic in 0..c_in {
            for j in 0..k {
                wflip[(oc * c_in + ic) * k + j] = conv.weight[[ic, oc, k - 1 - j]];
            }
        }
    }

    let (u, lu) = ctx.inflate_len(x, c_in, x_t, stride);
    let (up, t_pad) = ctx.zero_pad_len(u, c_in, lu, k - 1);
    let t_raw = t_pad - (k - 1); // = (x_t-1)*stride + k
    let w = ctx.param(wflip, &[c_out, c_in, k, 1]);
    let mut y = ctx.g.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![up, w],
        Shape::new(&[1, c_out, t_raw, 1], F32),
    );

    let t_out = t_raw - 2 * pad;
    let crop_start = pad + conv.out_offset;
    if crop_start > 0 {
        y = ctx.g.narrow_(y, 2, crop_start, t_out);
    }
    if let Some(b) = &conv.bias {
        y = ctx.add_bias(y, b.as_slice().unwrap(), c_out, t_out);
    }
    (y, c_out, t_out)
}

/// Polyphase `ConvTranspose1d` for `K == 2·stride` (see [`conv_transpose1d`]).
///
/// Output coord `O = j·s + p` (phase `p = O mod s`) gets contributions only from
/// kernel taps `ki ≡ p (mod s)` — exactly two of them (`p` and `p+s`) since
/// `K = 2s`. So phase `p`'s outputs form a regular 2-tap conv of the input:
///   `y_p[oc, j] = Σ_ic x[ic, j-1]·w[ic, oc, p+s] + x[ic, j]·w[ic, oc, p]`
/// (`j ∈ 0..x_t+1`, out-of-range `x` = 0), realised as a `k=2`, `pad=1`,
/// `stride=1` conv on the once-padded input. Concatenating the `s` phases on the
/// W axis then reshaping interleaves them as `O = j·s + p`, reproducing the full
/// `(x_t+1)·s` uncropped output, after which the same crop/bias apply.
fn conv_transpose1d_polyphase(
    ctx: &mut Ctx,
    x: HirNodeId,
    x_t: usize,
    conv: &WnConvTranspose1d,
) -> (HirNodeId, usize, usize) {
    let (c_in, c_out, k) = conv.weight.dim();
    let s = conv.stride;
    let pad = conv.pad;
    debug_assert_eq!(k, 2 * s);

    let lp = x_t + 1; // per-phase output length
    let (xp, _) = ctx.zero_pad_len(x, c_in, x_t, 1); // [1, c_in, x_t+2, 1]
    let mut phases = Vec::with_capacity(s);
    for p in 0..s {
        // Phase kernel in conv1d `[c_out, c_in, 2]` layout: tap 0 = w[.,.,p+s]
        // (the `x[j-1]` term), tap 1 = w[.,.,p] (the `x[j]` term).
        let mut wp = vec![0f32; c_out * c_in * 2];
        for oc in 0..c_out {
            for ic in 0..c_in {
                wp[(oc * c_in + ic) * 2] = conv.weight[[ic, oc, p + s]];
                wp[(oc * c_in + ic) * 2 + 1] = conv.weight[[ic, oc, p]];
            }
        }
        let w = ctx.param(wp, &[c_out, c_in, 2, 1]);
        let y = ctx.g.add_node(
            Op::Conv {
                kernel_size: vec![2, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![xp, w],
            Shape::new(&[1, c_out, lp, 1], F32),
        );
        phases.push(y);
    }

    // Interleave: concat phases on W → [1, c_out, lp, s], reshape → [1,c_out,lp*s,1]
    // (row-major flatten of (lp, s) gives output index `j*s + p`).
    let cat = ctx.g.concat_(phases, 3);
    let full = ctx
        .g
        .reshape_(cat, vec![1, c_out as i64, (lp * s) as i64, 1]);

    let t_raw = lp * s; // = (x_t + 1) * s, same uncropped length as the inflate path
    let t_out = t_raw - 2 * pad;
    let crop_start = pad + conv.out_offset;
    let mut y = if crop_start > 0 {
        ctx.g.narrow_(full, 2, crop_start, t_out)
    } else {
        full
    };
    if let Some(b) = &conv.bias {
        y = ctx.add_bias(y, b.as_slice().unwrap(), c_out, t_out);
    }
    (y, c_out, t_out)
}

/// Snake activation: `x + sin(α·x)² / (α + 1e-9)`, per channel.
fn snake(ctx: &mut Ctx, x: HirNodeId, c: usize, t: usize, alpha: &[f32]) -> HirNodeId {
    snake_mode(ctx, x, c, t, alpha, false)
}

/// Snake with selectable α convention. `c_style=false` broadcasts α per channel
/// (DAC). `c_style=true` reproduces TSAC's `snake_s`: α indexed by the flat
/// channel-major position `(ci*t + ti) % alpha.len()`, clamped to 1e-6 — built
/// as an explicit `[1,c,t,1]` tensor so it lowers identically on every backend.
fn snake_mode(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    alpha: &[f32],
    c_style: bool,
) -> HirNodeId {
    let full = Shape::new(&[1, c, t, 1], F32);
    let (a_e, ia_e) = if c_style {
        let na = alpha.len();
        let mut af = vec![0f32; c * t];
        let mut iaf = vec![0f32; c * t];
        for ci in 0..c {
            for ti in 0..t {
                let a = alpha[(ci * t + ti) % na].max(1e-6);
                af[ci * t + ti] = a;
                iaf[ci * t + ti] = 1.0 / a;
            }
        }
        (ctx.param(af, &[1, c, t, 1]), ctx.param(iaf, &[1, c, t, 1]))
    } else {
        debug_assert_eq!(alpha.len(), c);
        let inv: Vec<f32> = alpha.iter().map(|&a| 1.0 / (a + 1e-9)).collect();
        let a = ctx.param(alpha.to_vec(), &[1, c, 1, 1]);
        let ia = ctx.param(inv, &[1, c, 1, 1]);
        (ctx.expand(a, &[1, c, t, 1]), ctx.expand(ia, &[1, c, t, 1]))
    };
    let ax = ctx.g.mul(x, a_e);
    let s = ctx.g.activation(Activation::Sin, ax, full);
    let s2 = ctx.g.mul(s, s);
    let scaled = ctx.g.mul(s2, ia_e);
    ctx.g.add(x, scaled)
}

/// ResidualUnit: snake → conv → snake → conv → center-crop add. Channels preserved.
fn residual_unit(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    ru: &ResidualUnit,
    cs: bool,
) -> (HirNodeId, usize, usize) {
    let h = snake_mode(ctx, x, c, t, ru.snake1_alpha.as_slice().unwrap(), cs);
    let (h, c1, t1) = conv1d(ctx, h, t, &ru.conv1);
    let h = snake_mode(ctx, h, c1, t1, ru.snake2_alpha.as_slice().unwrap(), cs);
    let (h, c2, t2) = conv1d(ctx, h, t1, &ru.conv2);
    let x_aligned = if t2 == t {
        x
    } else {
        let start = (t - t2) / 2;
        ctx.g.narrow_(x, 2, start, t2)
    };
    (ctx.g.add(x_aligned, h), c2, t2)
}

fn encoder_block(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    blk: &EncoderBlock,
) -> (HirNodeId, usize, usize) {
    let (mut h, mut hc, mut ht) = residual_unit(ctx, x, c, t, &blk.residual_units[0], false);
    (h, hc, ht) = residual_unit(ctx, h, hc, ht, &blk.residual_units[1], false);
    (h, hc, ht) = residual_unit(ctx, h, hc, ht, &blk.residual_units[2], false);
    let h = snake(ctx, h, hc, ht, blk.snake_alpha.as_slice().unwrap());
    conv1d(ctx, h, ht, &blk.downsample)
}

fn decoder_block(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    blk: &DecoderBlock,
    cs: bool,
) -> (HirNodeId, usize, usize) {
    let h = snake_mode(ctx, x, c, t, blk.snake_alpha.as_slice().unwrap(), cs);
    let (h, hc, ht) = conv_transpose1d(ctx, h, t, &blk.upsample);
    let (mut h, mut hc, mut ht) = residual_unit(ctx, h, hc, ht, &blk.residual_units[0], cs);
    (h, hc, ht) = residual_unit(ctx, h, hc, ht, &blk.residual_units[1], cs);
    residual_unit(ctx, h, hc, ht, &blk.residual_units[2], cs)
}

/// Build the encoder graph for a fixed input PCM length. Input `"pcm"` is
/// `[1, 1, in_len, 1]`; the single output is `[1, latent, t, 1]`.
pub fn build_encoder_graph(
    enc: &Encoder,
    in_len: usize,
) -> Result<(Graph, NamedTensors, (usize, usize))> {
    let mut hir = HirModule::new("dac_encoder");
    let mut g = HirMut::new(&mut hir);
    let mut ctx = Ctx::new(&mut g);

    let in_ch = enc.stem.weight.dim().1; // stem c_in (1 = mono DAC, 2 = stereo TSAC)
    let x = ctx.g.input("pcm", Shape::new(&[1, in_ch, in_len, 1], F32));
    let (mut h, mut c, mut t) = conv1d(&mut ctx, x, in_len, &enc.stem);
    for blk in &enc.blocks {
        (h, c, t) = encoder_block(&mut ctx, h, c, t, blk);
    }
    let h = snake(&mut ctx, h, c, t, enc.snake_alpha.as_slice().unwrap());
    let (out, oc, ot) = conv1d(&mut ctx, h, t, &enc.head);
    let params = ctx.params;
    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("dac encoder lower: {e}"))?;
    Ok((graph, params, (oc, ot)))
}

/// Build the decoder graph for a fixed latent length. Input `"z"` is
/// `[1, in_c, in_t, 1]`; output is the waveform `[1, 1, out_len, 1]`.
pub fn build_decoder_graph(
    dec: &Decoder,
    in_c: usize,
    in_t: usize,
) -> Result<(Graph, NamedTensors, (usize, usize))> {
    let mut hir = HirModule::new("dac_decoder");
    let mut g = HirMut::new(&mut hir);
    let mut ctx = Ctx::new(&mut g);

    let cs = dec.c_snake();
    let x = ctx.g.input("z", Shape::new(&[1, in_c, in_t, 1], F32));
    let (mut h, mut c, mut t) = conv1d(&mut ctx, x, in_t, &dec.stem);
    for blk in &dec.blocks {
        (h, c, t) = decoder_block(&mut ctx, h, c, t, blk, cs);
    }
    let h = snake_mode(&mut ctx, h, c, t, dec.snake_alpha.as_slice().unwrap(), cs);
    let (h, hc, ht) = conv1d(&mut ctx, h, t, &dec.head);
    let out = ctx.g.tanh(h);
    let params = ctx.params;
    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("dac decoder lower: {e}"))?;
    Ok((graph, params, (hc, ht)))
}

/// A compiled encoder/decoder graph plus its output dimensions.
pub struct CodecGraph {
    compiled: CompiledGraph,
    out_c: usize,
    out_t: usize,
    input_name: &'static str,
}

impl CodecGraph {
    fn compile(
        device: Device,
        graph: Graph,
        params: NamedTensors,
        out_c: usize,
        out_t: usize,
        input_name: &'static str,
    ) -> Self {
        let session = Session::new(device);
        let mut compiled = session.compile(graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Self {
            compiled,
            out_c,
            out_t,
            input_name,
        }
    }

    pub fn encoder(device: Device, enc: &Encoder, in_len: usize) -> Result<Self> {
        let (graph, params, (oc, ot)) = build_encoder_graph(enc, in_len)?;
        Ok(Self::compile(device, graph, params, oc, ot, "pcm"))
    }

    pub fn decoder(device: Device, dec: &Decoder, in_c: usize, in_t: usize) -> Result<Self> {
        let (graph, params, (out_c, out_len)) = build_decoder_graph(dec, in_c, in_t)?;
        Ok(Self::compile(device, graph, params, out_c, out_len, "z"))
    }

    /// Run with a flat `[C_in, T_in]`-row-major input, returning `[out_c, out_t]`.
    pub fn run(&mut self, input: &[f32]) -> Result<Array2<f32>> {
        let outs = self.compiled.run(&[(self.input_name, input)]);
        let flat = outs
            .into_iter()
            .next()
            .context("codec graph produced no output")?;
        Array2::from_shape_vec((self.out_c, self.out_t), flat).context("codec graph output reshape")
    }

    pub fn out_dims(&self) -> (usize, usize) {
        (self.out_c, self.out_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{
        Decoder, DecoderBlock, Encoder, EncoderBlock, ResidualUnit, WnConv1d, WnConvTranspose1d,
    };
    use ndarray::{Array1, Array2, Array3};

    /// Tiny deterministic PRNG so the test needs no external rng dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn arr3(&mut self, a: usize, b: usize, c: usize) -> Array3<f32> {
            Array3::from_shape_fn((a, b, c), |_| self.next_f32() * 0.2)
        }
        fn arr1(&mut self, n: usize) -> Array1<f32> {
            Array1::from_shape_fn(n, |_| self.next_f32() * 0.5 + 0.5)
        }
        fn bias(&mut self, n: usize) -> Array1<f32> {
            Array1::from_shape_fn(n, |_| self.next_f32() * 0.1)
        }
    }

    fn conv(
        r: &mut Lcg,
        c_out: usize,
        c_in: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        same: bool,
    ) -> WnConv1d {
        WnConv1d {
            weight: r.arr3(c_out, c_in, k),
            bias: Some(r.bias(c_out)),
            stride,
            pad,
            dilation,
            same_length: same,
        }
    }

    fn residual(r: &mut Lcg, c: usize, dil: usize) -> ResidualUnit {
        let pad = ((7 - 1) * dil) / 2;
        ResidualUnit {
            snake1_alpha: r.arr1(c),
            conv1: conv(r, c, c, 7, 1, pad, dil, true),
            snake2_alpha: r.arr1(c),
            conv2: conv(r, c, c, 1, 1, 0, 1, true),
        }
    }

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        assert_eq!(
            a.dim(),
            b.dim(),
            "shape mismatch {:?} vs {:?}",
            a.dim(),
            b.dim()
        );
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn tiny_encoder(r: &mut Lcg) -> Encoder {
        let c0 = 6;
        let c1 = 12;
        Encoder {
            stem: conv(r, c0, 1, 7, 1, 3, 1, true),
            blocks: vec![EncoderBlock {
                residual_units: [residual(r, c0, 1), residual(r, c0, 3), residual(r, c0, 9)],
                snake_alpha: r.arr1(c0),
                downsample: conv(r, c1, c0, 4, 2, 1, 1, false),
            }],
            snake_alpha: r.arr1(c1),
            head: conv(r, 8, c1, 3, 1, 1, 1, true),
        }
    }

    fn tiny_decoder(r: &mut Lcg) -> Decoder {
        let c0 = 12;
        let c1 = 6;
        Decoder {
            stem: conv(r, c0, 8, 3, 1, 1, 1, true),
            blocks: vec![DecoderBlock {
                snake_alpha: r.arr1(c0),
                upsample: WnConvTranspose1d {
                    weight: r.arr3(c0, c1, 4),
                    bias: Some(r.bias(c1)),
                    stride: 2,
                    pad: 1,
                    out_offset: 0,
                },
                residual_units: [residual(r, c1, 1), residual(r, c1, 3), residual(r, c1, 9)],
            }],
            snake_alpha: r.arr1(c1),
            head: conv(r, 1, c1, 3, 1, 1, 1, true),
            c_snake: false,
        }
    }

    #[test]
    fn encoder_graph_matches_ndarray() {
        let mut r = Lcg(0x1234_5678);
        let enc = tiny_encoder(&mut r);
        let in_len = 320usize;
        let pcm: Vec<f32> = (0..in_len).map(|i| ((i as f32) * 0.05).sin()).collect();
        let input = Array2::from_shape_vec((1, in_len), pcm.clone()).unwrap();

        let reference = enc.forward(input.view());
        let mut g = CodecGraph::encoder(Device::Cpu, &enc, in_len).unwrap();
        let got = g.run(&pcm).unwrap();

        assert_eq!(got.dim(), reference.dim());
        let err = max_abs(&got, &reference);
        assert!(err < 1e-3, "encoder graph vs ndarray max|Δ| = {err}");
    }

    /// GPU backends compiled into this build (CPU is covered separately).
    // Elements are cfg-gated per backend, so `vec![..]` is not an option.
    #[allow(clippy::vec_init_then_push)]
    fn gpu_devices() -> Vec<Device> {
        // `mut` is only exercised by the backend-feature pushes below.
        #[allow(unused_mut)]
        let mut v = Vec::new();
        #[cfg(feature = "metal")]
        v.push(Device::Metal);
        #[cfg(feature = "mlx")]
        v.push(Device::Mlx);
        #[cfg(feature = "gpu")]
        v.push(Device::Gpu);
        v
    }

    #[test]
    fn encoder_graph_cross_backend() {
        let mut r = Lcg(0x1234_5678);
        let enc = tiny_encoder(&mut r);
        let in_len = 320usize;
        let pcm: Vec<f32> = (0..in_len).map(|i| ((i as f32) * 0.05).sin()).collect();
        let input = Array2::from_shape_vec((1, in_len), pcm.clone()).unwrap();
        let reference = enc.forward(input.view());

        for dev in gpu_devices() {
            if !rlx_runtime::is_available(dev) {
                eprintln!("skip {dev:?}: not available");
                continue;
            }
            let mut g = CodecGraph::encoder(dev, &enc, in_len).unwrap();
            let got = g.run(&pcm).unwrap();
            let err = max_abs(&got, &reference);
            assert!(err < 1e-2, "encoder on {dev:?} vs ndarray max|Δ| = {err}");
            eprintln!("encoder {dev:?} parity ok: max|Δ| = {err:.2e}");
        }
    }

    #[test]
    fn decoder_graph_cross_backend() {
        let mut r = Lcg(0x90ab_cdef);
        let dec = tiny_decoder(&mut r);
        let (in_c, in_t) = (8usize, 16usize);
        let z: Vec<f32> = (0..in_c * in_t)
            .map(|i| ((i as f32) * 0.1).cos() * 0.3)
            .collect();
        let z_arr = Array2::from_shape_vec((in_c, in_t), z.clone()).unwrap();
        let reference = dec.forward(z_arr.view());

        for dev in gpu_devices() {
            if !rlx_runtime::is_available(dev) {
                eprintln!("skip {dev:?}: not available");
                continue;
            }
            let mut g = CodecGraph::decoder(dev, &dec, in_c, in_t).unwrap();
            let got = g.run(&z).unwrap();
            let err = max_abs(&got, &reference);
            assert!(err < 1e-2, "decoder on {dev:?} vs ndarray max|Δ| = {err}");
            eprintln!("decoder {dev:?} parity ok: max|Δ| = {err:.2e}");
        }
    }

    #[test]
    fn decoder_graph_matches_ndarray() {
        let mut r = Lcg(0x90ab_cdef);
        let dec = tiny_decoder(&mut r);
        let (in_c, in_t) = (8usize, 16usize);
        let z: Vec<f32> = (0..in_c * in_t)
            .map(|i| ((i as f32) * 0.1).cos() * 0.3)
            .collect();
        let z_arr = Array2::from_shape_vec((in_c, in_t), z.clone()).unwrap();

        let reference = dec.forward(z_arr.view());
        let mut g = CodecGraph::decoder(Device::Cpu, &dec, in_c, in_t).unwrap();
        let got = g.run(&z).unwrap();

        assert_eq!(got.dim(), reference.dim());
        let err = max_abs(&got, &reference);
        assert!(err < 1e-3, "decoder graph vs ndarray max|Δ| = {err}");
    }
}
