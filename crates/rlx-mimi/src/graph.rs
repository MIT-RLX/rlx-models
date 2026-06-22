// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Mimi codec as rlx-runtime HIR graphs running natively on every backend
// (cpu/metal/mlx/cuda/rocm/wgpu/vulkan). The SEANet conv stacks, the frame-rate
// down/up-sampling, and the encoder/decoder transformers are all expressed as a
// single compiled graph per (device, length). The split RVQ codebook search
// stays host-side (see `rvq.rs`) — it is a tiny argmin.
//
// Conventions (shared with rlx-dac, see its graph.rs for the why):
//   * channel-major activations `[C, T]` are carried as NCHW `[1, C, T, 1]`
//     (length in H, singleton W), kernel `[k, 1]`. MLX reads the length-axis
//     conv params from index 0, so the length MUST sit in H.
//   * padding is explicit (concat) — backends disagree on conv `padding`.
//   * transposed conv is decomposed to zero-insert + regular conv (wgpu/coreml
//     have no native ConvTranspose kernel).
//   * the transformer runs in time-major `[T, C]`.

use crate::conv::PadMode;
use crate::seanet::{
    Conv1dLayer, FrameRateDownsample, FrameRateUpsample, ResnetBlock, SeanetDecoder, SeanetEncoder,
    SeanetLayer,
};
use crate::transformer::{Attention, MimiTransformer, TransformerLayer};
use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// Named tensors `(name, data)` produced alongside a built graph.
type NamedTensors = Vec<(String, Vec<f32>)>;

const F32: DType = DType::F32;

/// Builder context: accumulates params (name → row-major f32) and unique names.
pub(crate) struct Ctx<'a, 'b> {
    pub(crate) g: &'a mut HirMut<'b>,
    pub(crate) params: NamedTensors,
    next: usize,
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub(crate) fn new(g: &'a mut HirMut<'b>) -> Self {
        Self {
            g,
            params: Vec::new(),
            next: 0,
        }
    }

    pub(crate) fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let name = format!("w{}", self.next);
        self.next += 1;
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "param {name} shape {shape:?} vs len {}",
            data.len()
        );
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.push((name, data));
        id
    }

    fn expand(&mut self, id: HirNodeId, target: &[usize]) -> HirNodeId {
        let shape = Shape::new(target, F32);
        let tgt: Vec<i64> = target.iter().map(|&d| d as i64).collect();
        self.g
            .add_node(Op::Expand { target_shape: tgt }, vec![id], shape)
    }

    fn scalar(&mut self, v: f32, target: &[usize]) -> HirNodeId {
        let ones = vec![1usize; target.len()];
        let s = self.param(vec![v], &ones);
        self.expand(s, target)
    }

    /// `[1, C, T, 1]` bias-add from a `[C]` bias.
    fn add_bias(&mut self, y: HirNodeId, bias: &[f32], c: usize, t: usize) -> HirNodeId {
        let b = self.param(bias.to_vec(), &[1, c, 1, 1]);
        let be = self.expand(b, &[1, c, t, 1]);
        self.g.add(y, be)
    }

    /// Asymmetric pad on the length (H) axis, zero or edge-replicate.
    fn pad_len(
        &mut self,
        x: HirNodeId,
        c: usize,
        t: usize,
        pl: usize,
        pr: usize,
        mode: PadMode,
    ) -> (HirNodeId, usize) {
        if pl == 0 && pr == 0 {
            return (x, t);
        }
        let mut parts: Vec<HirNodeId> = Vec::new();
        match mode {
            PadMode::Constant => {
                if pl > 0 {
                    parts.push(self.param(vec![0.0; c * pl], &[1, c, pl, 1]));
                }
                parts.push(x);
                if pr > 0 {
                    parts.push(self.param(vec![0.0; c * pr], &[1, c, pr, 1]));
                }
            }
            PadMode::Replicate => {
                if pl > 0 {
                    let first = self.g.narrow_(x, 2, 0, 1);
                    parts.push(self.expand(first, &[1, c, pl, 1]));
                }
                parts.push(x);
                if pr > 0 {
                    let last = self.g.narrow_(x, 2, t - 1, 1);
                    parts.push(self.expand(last, &[1, c, pr, 1]));
                }
            }
        }
        (self.g.concat_(parts, 2), t + pl + pr)
    }

    /// Zero-insert `s-1` samples between length steps: `[1,C,T,1]` → `[1,C,(T-1)s+1,1]`.
    fn inflate_len(&mut self, x: HirNodeId, c: usize, t: usize, s: usize) -> (HirNodeId, usize) {
        if s <= 1 {
            return (x, t);
        }
        let z = self.param(vec![0.0; c * t * (s - 1)], &[1, c, t, s - 1]);
        let cat = self.g.concat_(vec![x, z], 3);
        let resh = self.g.reshape_(cat, vec![1, c as i64, (t * s) as i64, 1]);
        let lu = (t - 1) * s + 1;
        (self.g.narrow_(resh, 2, 0, lu), lu)
    }

    fn conv_raw(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        c_out: usize,
        t_out: usize,
        k: usize,
        stride: usize,
        dil: usize,
        groups: usize,
    ) -> HirNodeId {
        self.g.add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![0, 0],
                dilation: vec![dil, 1],
                groups,
            },
            vec![x, w],
            Shape::new(&[1, c_out, t_out, 1], F32),
        )
    }
}

fn weight_data(w: &Array3<f32>) -> Vec<f32> {
    w.as_standard_layout().iter().copied().collect()
}

/// ELU: `x>=0 ? x : exp(x)-1`, via `relu(x) + exp(min(x,0)) - 1`.
fn elu(ctx: &mut Ctx, x: HirNodeId, c: usize, t: usize) -> HirNodeId {
    let shape = Shape::new(&[1, c, t, 1], F32);
    let r = ctx.g.activation(Activation::Relu, x, shape.clone());
    let negx = ctx.g.activation(Activation::Neg, x, shape.clone());
    let relu_neg = ctx.g.activation(Activation::Relu, negx, shape.clone()); // -min(x,0)
    let minx = ctx.g.activation(Activation::Neg, relu_neg, shape.clone()); // min(x,0)
    let e = ctx.g.activation(Activation::Exp, minx, shape);
    let re = ctx.g.add(r, e);
    let ones = ctx.scalar(1.0, &[1, c, t, 1]);
    ctx.g.sub(re, ones)
}

/// Causal 1D conv matching `mimi_causal_conv1d`. Returns `(node, c_out, t_out)`.
fn causal_conv(
    ctx: &mut Ctx,
    x: HirNodeId,
    t_in: usize,
    conv: &Conv1dLayer,
) -> (HirNodeId, usize, usize) {
    let (c_out, c_in, k) = conv.weight.dim();
    let stride = conv.stride;
    let dil = conv.dilation;
    let effective_k = (k - 1) * dil + 1;
    let padding_total = effective_k.saturating_sub(stride);
    let num = t_in as i64 - effective_k as i64 + padding_total as i64;
    let n_frames = if num <= 0 {
        0
    } else {
        (num as usize).div_ceil(stride)
    };
    let ideal_len = n_frames * stride + effective_k - padding_total;
    let pad_left = padding_total;
    let pad_right = ideal_len.saturating_sub(t_in);

    let (xp, t_pad) = ctx.pad_len(x, c_in, t_in, pad_left, pad_right, conv.pad_mode);
    let t_out = (t_pad - effective_k) / stride + 1;
    let w = ctx.param(weight_data(&conv.weight), &[c_out, c_in, k, 1]);
    let mut y = ctx.conv_raw(xp, w, c_out, t_out, k, stride, dil, 1);
    if let Some(b) = &conv.bias {
        y = ctx.add_bias(y, b.as_slice().unwrap(), c_out, t_out);
    }
    (y, c_out, t_out)
}

/// Transposed conv (`weight [C_in, C_out/groups, K]`) with `trim_right_ratio`,
/// decomposed to zero-insert + regular grouped conv. Returns `(node, c_out, t_out)`.
fn trans_conv(
    ctx: &mut Ctx,
    x: HirNodeId,
    t_in: usize,
    weight: &Array3<f32>,
    bias: Option<&[f32]>,
    stride: usize,
    trim_right_ratio: f32,
    groups: usize,
) -> (HirNodeId, usize, usize) {
    let (in_ch, c_out_per_g, k) = weight.dim();
    let c_out = c_out_per_g * groups;

    // Flip the kernel and swap in/out channels per group:
    //   W[oc, ic_per_g, j] = weight[ic, oc_per_g, k-1-j]
    let c_in_per_g = in_ch / groups;
    let mut wflip = vec![0f32; c_out * c_in_per_g * k];
    for g in 0..groups {
        for ocp in 0..c_out_per_g {
            let oc = g * c_out_per_g + ocp;
            for icp in 0..c_in_per_g {
                let ic = g * c_in_per_g + icp;
                for j in 0..k {
                    wflip[(oc * c_in_per_g + icp) * k + j] = weight[[ic, ocp, k - 1 - j]];
                }
            }
        }
    }

    let (u, lu) = ctx.inflate_len(x, in_ch, t_in, stride);
    let (up, t_pad) = ctx.pad_len(u, in_ch, lu, k - 1, k - 1, PadMode::Constant);
    let t_raw = t_pad - (k - 1); // = (t_in-1)*stride + k
    let w = ctx.param(wflip, &[c_out, c_in_per_g, k, 1]);
    let mut y = ctx.conv_raw(up, w, c_out, t_raw, k, 1, 1, groups);

    let padding_total = k.saturating_sub(stride);
    let padding_right = ((padding_total as f32) * trim_right_ratio).ceil() as usize;
    let padding_left = padding_total - padding_right;
    let t_out = t_raw - padding_total;
    if padding_left > 0 || padding_right > 0 {
        y = ctx.g.narrow_(y, 2, padding_left, t_out);
    }
    if let Some(b) = bias {
        y = ctx.add_bias(y, b, c_out, t_out);
    }
    (y, c_out, t_out)
}

fn resnet_block(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    blk: &ResnetBlock,
) -> (HirNodeId, usize, usize) {
    let h = elu(ctx, x, c, t);
    let (h, hc, ht) = causal_conv(ctx, h, t, &blk.conv_a);
    let h = elu(ctx, h, hc, ht);
    let (h, hc, ht) = causal_conv(ctx, h, ht, &blk.conv_b);
    // mimi resnet keeps length (causal conv), so residual adds directly.
    debug_assert_eq!((hc, ht), (c, t));
    (ctx.g.add(x, h), hc, ht)
}

/// Walk a SEANet layer stack. Returns `(node, c_out, t_out)`.
pub(crate) fn seanet_stack(
    ctx: &mut Ctx,
    mut x: HirNodeId,
    mut c: usize,
    mut t: usize,
    layers: &[SeanetLayer],
) -> (HirNodeId, usize, usize) {
    for layer in layers {
        match layer {
            SeanetLayer::Conv(conv) => {
                (x, c, t) = causal_conv(ctx, x, t, conv);
            }
            SeanetLayer::TransConv {
                weight,
                bias,
                stride,
                trim_right_ratio,
            } => {
                (x, c, t) = trans_conv(
                    ctx,
                    x,
                    t,
                    weight,
                    bias.as_ref().map(|b| b.as_slice().unwrap()),
                    *stride,
                    *trim_right_ratio,
                    1,
                );
            }
            SeanetLayer::Res(blk) => {
                (x, c, t) = resnet_block(ctx, x, c, t, blk);
            }
            SeanetLayer::Elu => {
                x = elu(ctx, x, c, t);
            }
        }
    }
    (x, c, t)
}

pub(crate) fn seanet_encoder(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    enc: &SeanetEncoder,
) -> (HirNodeId, usize, usize) {
    seanet_stack(ctx, x, c, t, &enc.layers)
}

pub(crate) fn seanet_decoder(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    dec: &SeanetDecoder,
) -> (HirNodeId, usize, usize) {
    seanet_stack(ctx, x, c, t, &dec.layers)
}

/// Frame-rate downsample (single stride-2 replicate-padded causal conv).
pub(crate) fn downsample(
    ctx: &mut Ctx,
    x: HirNodeId,
    t: usize,
    conv: &Conv1dLayer,
) -> (HirNodeId, usize, usize) {
    causal_conv(ctx, x, t, conv)
}

/// Frame-rate upsample (depthwise grouped transposed conv, `groups = channels`).
pub(crate) fn upsample(
    ctx: &mut Ctx,
    x: HirNodeId,
    c: usize,
    t: usize,
    weight: &Array3<f32>,
    stride: usize,
    trim_right_ratio: f32,
) -> (HirNodeId, usize, usize) {
    trans_conv(ctx, x, t, weight, None, stride, trim_right_ratio, c)
}

// ------------------------------- transformer -------------------------------
//
// Operates in time-major `[T, C]`. Uses native LayerNorm/Softmax, batched
// matmul for per-head attention, and rotate-half RoPE with host-precomputed
// cos/sin + sliding-window causal mask bound as params.

/// `x [rows, in] @ Wᵀ` where `w_oi` is `[out, in]` (torch Linear weight, no bias).
fn linear(ctx: &mut Ctx, x: HirNodeId, in_d: usize, out_d: usize, w_oi: &Array2<f32>) -> HirNodeId {
    let mut wt = vec![0f32; in_d * out_d];
    for o in 0..out_d {
        for i in 0..in_d {
            wt[i * out_d + o] = w_oi[[o, i]];
        }
    }
    let wp = ctx.param(wt, &[in_d, out_d]);
    ctx.g.mm(x, wp)
}

/// Per-channel scale (`LayerScale`) on a `[T, C]` tensor.
fn layer_scale(ctx: &mut Ctx, x: HirNodeId, t: usize, c: usize, scale: &[f32]) -> HirNodeId {
    let s = ctx.param(scale.to_vec(), &[1, c]);
    let se = ctx.expand(s, &[t, c]);
    ctx.g.mul(x, se)
}

/// Rotate-half RoPE on `[H, T, D]` with cos/sin already shaped `[1, T, D]`.
fn rope(
    ctx: &mut Ctx,
    x: HirNodeId,
    h: usize,
    t: usize,
    d: usize,
    cos_p: HirNodeId,
    sin_p: HirNodeId,
) -> HirNodeId {
    let half = d / 2;
    let cos_e = ctx.expand(cos_p, &[h, t, d]);
    let sin_e = ctx.expand(sin_p, &[h, t, d]);
    let a = ctx.g.narrow_(x, 2, 0, half);
    let b = ctx.g.narrow_(x, 2, half, half);
    let nb = ctx
        .g
        .activation(Activation::Neg, b, Shape::new(&[h, t, half], F32));
    let rot = ctx.g.concat_(vec![nb, a], 2);
    let xc = ctx.g.mul(x, cos_e);
    let rs = ctx.g.mul(rot, sin_e);
    ctx.g.add(xc, rs)
}

#[allow(clippy::too_many_arguments)]
fn attention(
    ctx: &mut Ctx,
    x: HirNodeId,
    t: usize,
    attn: &Attention,
    cos_p: HirNodeId,
    sin_p: HirNodeId,
    mask_p: HirNodeId,
) -> HirNodeId {
    let h = attn.num_heads;
    let d = attn.head_dim;
    let c = h * d;

    let q = linear(ctx, x, c, c, &attn.q_w);
    let k = linear(ctx, x, c, c, &attn.k_w);
    let v = linear(ctx, x, c, c, &attn.v_w);

    // [T, C] → [T, H, D] → [H, T, D]
    let to_htd = |ctx: &mut Ctx, n: HirNodeId| {
        let r = ctx.g.reshape_(n, vec![t as i64, h as i64, d as i64]);
        ctx.g.transpose_(r, vec![1, 0, 2])
    };
    let q = to_htd(ctx, q);
    let k = to_htd(ctx, k);
    let v = to_htd(ctx, v);

    let q = rope(ctx, q, h, t, d, cos_p, sin_p);
    let k = rope(ctx, k, h, t, d, cos_p, sin_p);

    let kt = ctx.g.transpose_(k, vec![0, 2, 1]); // [H, D, T]
    let scores = ctx.g.mm(q, kt); // [H, T, T]
    let sc = ctx.scalar(attn.scaling, &[h, t, t]);
    let scores = ctx.g.mul(scores, sc);
    let mask_e = ctx.expand(mask_p, &[h, t, t]);
    let scores = ctx.g.add(scores, mask_e);
    let probs = ctx.g.sm(scores, -1);
    let out = ctx.g.mm(probs, v); // [H, T, D]

    let out = ctx.g.transpose_(out, vec![1, 0, 2]); // [T, H, D]
    let out = ctx.g.reshape_(out, vec![t as i64, c as i64]);
    linear(ctx, out, c, c, &attn.o_w)
}

fn transformer_layer(
    ctx: &mut Ctx,
    x: HirNodeId,
    t: usize,
    c: usize,
    eps: f32,
    layer: &TransformerLayer,
    cos_p: HirNodeId,
    sin_p: HirNodeId,
    mask_p: HirNodeId,
) -> HirNodeId {
    let ng = ctx.param(layer.input_norm_w.to_vec(), &[c]);
    let nb = ctx.param(layer.input_norm_b.to_vec(), &[c]);
    let n = ctx.g.ln(x, ng, nb, eps);
    let attn_out = attention(ctx, n, t, &layer.attn, cos_p, sin_p, mask_p);
    let attn_out = layer_scale(
        ctx,
        attn_out,
        t,
        c,
        layer.attn_scale.scale.as_slice().unwrap(),
    );
    let h = ctx.g.add(x, attn_out);

    let pg = ctx.param(layer.post_norm_w.to_vec(), &[c]);
    let pb = ctx.param(layer.post_norm_b.to_vec(), &[c]);
    let n2 = ctx.g.ln(h, pg, pb, eps);
    let inter = layer.mlp.fc1_w.dim().0;
    let m = linear(ctx, n2, c, inter, &layer.mlp.fc1_w);
    let m = ctx.g.gelu(m);
    let m = linear(ctx, m, inter, c, &layer.mlp.fc2_w);
    let m = layer_scale(ctx, m, t, c, layer.mlp_scale.scale.as_slice().unwrap());
    ctx.g.add(h, m)
}

/// Build cos/sin `[t, head_dim]` (rotate-half layout) from `inv_freq`.
fn rope_tables(inv_freq: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let d = half * 2;
    let mut cos = vec![0f32; t * d];
    let mut sin = vec![0f32; t * d];
    for ti in 0..t {
        for hi in 0..half {
            let f = ti as f32 * inv_freq[hi];
            let (s, c) = f.sin_cos();
            cos[ti * d + hi] = c;
            cos[ti * d + hi + half] = c;
            sin[ti * d + hi] = s;
            sin[ti * d + hi + half] = s;
        }
    }
    (cos, sin)
}

fn sliding_causal_mask(t: usize, window: usize) -> Vec<f32> {
    let mut m = vec![f32::NEG_INFINITY; t * t];
    for i in 0..t {
        let lo = i.saturating_sub(window - 1);
        for j in lo..=i {
            m[i * t + j] = 0.0;
        }
    }
    m
}

/// `[1, C, T, 1]` → `[T, C]`.
fn ct_to_tc(ctx: &mut Ctx, x: HirNodeId, c: usize, t: usize) -> HirNodeId {
    let r = ctx.g.reshape_(x, vec![c as i64, t as i64]);
    ctx.g.transpose_(r, vec![1, 0])
}

/// `[T, C]` → `[1, C, T, 1]`.
fn tc_to_ct(ctx: &mut Ctx, x: HirNodeId, t: usize, c: usize) -> HirNodeId {
    let r = ctx.g.transpose_(x, vec![1, 0]);
    ctx.g.reshape_(r, vec![1, c as i64, t as i64, 1])
}

/// Run the full transformer over a `[T, C]` node. Returns the `[T, C]` output.
pub(crate) fn transformer(
    ctx: &mut Ctx,
    x: HirNodeId,
    t: usize,
    c: usize,
    tf: &MimiTransformer,
) -> HirNodeId {
    let d = tf.inv_freq.len() * 2;
    let (cos, sin) = rope_tables(tf.inv_freq.as_slice().unwrap(), t);
    let cos_p = ctx.param(cos, &[1, t, d]);
    let sin_p = ctx.param(sin, &[1, t, d]);
    let mask_p = ctx.param(sliding_causal_mask(t, tf.sliding_window), &[1, t, t]);
    let mut h = x;
    for layer in &tf.layers {
        h = transformer_layer(ctx, h, t, c, tf.norm_eps, layer, cos_p, sin_p, mask_p);
    }
    h
}

// ----------------------------- full pipelines -----------------------------

/// Weights for the encode path: SEANet encoder → transformer → downsample.
pub struct EncodeWeights<'a> {
    pub encoder: &'a SeanetEncoder,
    pub transformer: &'a MimiTransformer,
    pub downsample: &'a FrameRateDownsample,
    pub audio_channels: usize,
    pub hidden_size: usize,
}

/// Weights for the decode path: upsample → transformer → SEANet decoder.
pub struct DecodeWeights<'a> {
    pub upsample: &'a FrameRateUpsample,
    pub transformer: &'a MimiTransformer,
    pub decoder: &'a SeanetDecoder,
    pub hidden_size: usize,
}

/// Encode graph: PCM `[1, 1, in_len, 1]` → downsampled latent `[1, hidden, t_ds, 1]`.
pub fn build_encode_graph(
    w: &EncodeWeights,
    in_len: usize,
) -> Result<(Graph, NamedTensors, (usize, usize))> {
    let mut hir = HirModule::new("mimi_encode");
    let mut g = HirMut::new(&mut hir);
    let mut ctx = Ctx::new(&mut g);

    let x = ctx
        .g
        .input("pcm", Shape::new(&[1, w.audio_channels, in_len, 1], F32));
    let (h, c, t) = seanet_encoder(&mut ctx, x, w.audio_channels, in_len, w.encoder);
    let tc = ct_to_tc(&mut ctx, h, c, t);
    let tc = transformer(&mut ctx, tc, t, c, w.transformer);
    let ct = tc_to_ct(&mut ctx, tc, t, c);
    let (out, oc, ot) = downsample(&mut ctx, ct, t, &w.downsample.conv);
    let params = ctx.params;
    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("mimi encode lower: {e}"))?;
    Ok((graph, params, (oc, ot)))
}

/// Decode graph: latent `[1, hidden, in_t, 1]` → waveform `[1, 1, out_len, 1]`.
pub fn build_decode_graph(w: &DecodeWeights, in_t: usize) -> Result<(Graph, NamedTensors, usize)> {
    let mut hir = HirModule::new("mimi_decode");
    let mut g = HirMut::new(&mut hir);
    let mut ctx = Ctx::new(&mut g);

    let x = ctx
        .g
        .input("z", Shape::new(&[1, w.hidden_size, in_t, 1], F32));
    let (h, c, t) = upsample(
        &mut ctx,
        x,
        w.hidden_size,
        in_t,
        &w.upsample.weight,
        w.upsample.stride,
        w.upsample.trim_right_ratio,
    );
    let tc = ct_to_tc(&mut ctx, h, c, t);
    let tc = transformer(&mut ctx, tc, t, c, w.transformer);
    let ct = tc_to_ct(&mut ctx, tc, t, c);
    let (out, _oc, ot) = seanet_decoder(&mut ctx, ct, c, t, w.decoder);
    let params = ctx.params;
    hir.set_outputs(vec![out]);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("mimi decode lower: {e}"))?;
    Ok((graph, params, ot))
}

/// A compiled encode/decode graph plus its `[out_c, out_t]` output dims.
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
        let mut compiled = Session::new(device).compile(graph);
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

    pub fn encoder(device: Device, w: &EncodeWeights, in_len: usize) -> Result<Self> {
        let (graph, params, (oc, ot)) = build_encode_graph(w, in_len)?;
        Ok(Self::compile(device, graph, params, oc, ot, "pcm"))
    }

    pub fn decoder(device: Device, w: &DecodeWeights, in_t: usize) -> Result<Self> {
        let (graph, params, out_len) = build_decode_graph(w, in_t)?;
        Ok(Self::compile(device, graph, params, 1, out_len, "z"))
    }

    /// Run with row-major `[C_in, T_in]` input, returning `[out_c, out_t]`.
    pub fn run(&mut self, input: &[f32]) -> Result<Array2<f32>> {
        let outs = self.compiled.run(&[(self.input_name, input)]);
        let flat = outs
            .into_iter()
            .next()
            .context("mimi graph produced no output")?;
        Array2::from_shape_vec((self.out_c, self.out_t), flat).context("mimi graph output reshape")
    }

    pub fn out_dims(&self) -> (usize, usize) {
        (self.out_c, self.out_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{conv_transpose1d, elu_inplace, grouped_conv_transpose1d};
    use crate::seanet::Conv1dLayer;
    use ndarray::{Array1, Array2, Array3};
    use rlx_ir::Graph;
    use rlx_ir::hir::HirModule;
    use rlx_runtime::{Device, Session};

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn a3(&mut self, a: usize, b: usize, c: usize) -> Array3<f32> {
            Array3::from_shape_fn((a, b, c), |_| self.f() * 0.3)
        }
        fn a1(&mut self, n: usize) -> Array1<f32> {
            Array1::from_shape_fn(n, |_| self.f() * 0.2)
        }
    }

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        assert_eq!(a.dim(), b.dim(), "shape {:?} vs {:?}", a.dim(), b.dim());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Build a one-shot graph from `build`, run on `dev`, return `[c_out, t_out]`.
    fn run_unary(
        dev: Device,
        x: &Array2<f32>,
        build: impl FnOnce(&mut Ctx, HirNodeId, usize, usize) -> (HirNodeId, usize, usize),
    ) -> Array2<f32> {
        let (c, t) = x.dim();
        let mut hir = HirModule::new("t");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);
        let xin = ctx.g.input("x", Shape::new(&[1, c, t, 1], F32));
        let (out, oc, ot) = build(&mut ctx, xin, c, t);
        let params = std::mem::take(&mut ctx.params);
        hir.set_outputs(vec![out]);
        let graph = Graph::from_hir(hir).unwrap();
        let mut compiled = Session::new(dev).compile(graph);
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        compiled.finalize_params();
        let flat: Vec<f32> = x.iter().copied().collect();
        let got = compiled.run(&[("x", &flat)]).into_iter().next().unwrap();
        Array2::from_shape_vec((oc, ot), got).unwrap()
    }

    fn devices() -> Vec<Device> {
        let v = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        if rlx_runtime::is_available(Device::Metal) {
            v.push(Device::Metal);
        }
        #[cfg(feature = "mlx")]
        if rlx_runtime::is_available(Device::Mlx) {
            v.push(Device::Mlx);
        }
        #[cfg(feature = "gpu")]
        if rlx_runtime::is_available(Device::Gpu) {
            v.push(Device::Gpu);
        }
        v
    }

    fn check(
        name: &str,
        x: &Array2<f32>,
        reference: &Array2<f32>,
        build: impl Fn(&mut Ctx, HirNodeId, usize, usize) -> (HirNodeId, usize, usize),
    ) {
        for dev in devices() {
            let got = run_unary(dev, x, &build);
            let err = max_abs(&got, reference);
            assert!(err < 2e-3, "{name} on {dev:?}: max|Δ| = {err}");
            eprintln!("{name} {dev:?} ok: max|Δ| = {err:.2e}");
        }
    }

    fn conv_layer(
        r: &mut Lcg,
        c_out: usize,
        c_in: usize,
        k: usize,
        stride: usize,
        dil: usize,
        mode: PadMode,
    ) -> Conv1dLayer {
        Conv1dLayer {
            weight: r.a3(c_out, c_in, k),
            bias: Some(r.a1(c_out)),
            stride,
            dilation: dil,
            pad_mode: mode,
        }
    }

    #[test]
    fn causal_conv_constant() {
        let mut r = Lcg(1);
        let x = Array2::from_shape_fn((4, 40), |(_, t)| (t as f32 * 0.1).sin());
        let conv = conv_layer(&mut r, 6, 4, 7, 1, 1, PadMode::Constant);
        let reference = conv.forward(x.view());
        check("causal_conv_constant", &x, &reference, |ctx, xin, _c, t| {
            causal_conv(ctx, xin, t, &conv)
        });
    }

    #[test]
    fn causal_conv_dilated() {
        let mut r = Lcg(2);
        let x = Array2::from_shape_fn((5, 50), |(c, t)| ((t + c) as f32 * 0.07).cos());
        let conv = conv_layer(&mut r, 5, 5, 3, 1, 2, PadMode::Constant);
        let reference = conv.forward(x.view());
        check("causal_conv_dilated", &x, &reference, |ctx, xin, _c, t| {
            causal_conv(ctx, xin, t, &conv)
        });
    }

    #[test]
    fn causal_conv_strided_replicate() {
        let mut r = Lcg(3);
        let x = Array2::from_shape_fn((4, 41), |(_, t)| (t as f32 * 0.05).sin());
        let conv = conv_layer(&mut r, 4, 4, 4, 2, 1, PadMode::Replicate);
        let reference = conv.forward(x.view());
        check(
            "causal_conv_strided_replicate",
            &x,
            &reference,
            |ctx, xin, _c, t| causal_conv(ctx, xin, t, &conv),
        );
    }

    #[test]
    fn elu_matches() {
        let x = Array2::from_shape_fn((4, 20), |(c, t)| (t as f32 * 0.3 - 3.0) + c as f32 * 0.5);
        let mut reference = x.clone();
        elu_inplace(&mut reference);
        check("elu", &x, &reference, |ctx, xin, c, t| {
            (elu(ctx, xin, c, t), c, t)
        });
    }

    #[test]
    fn trans_conv_matches() {
        let mut r = Lcg(4);
        let x = Array2::from_shape_fn((6, 16), |(_, t)| (t as f32 * 0.2).sin());
        let weight = r.a3(6, 3, 4); // [C_in, C_out, K]
        let bias = r.a1(3);
        let reference = conv_transpose1d(x.view(), &weight, Some(&bias), 2, 1.0);
        check("trans_conv", &x, &reference, |ctx, xin, _c, t| {
            trans_conv(
                ctx,
                xin,
                t,
                &weight,
                Some(bias.as_slice().unwrap()),
                2,
                1.0,
                1,
            )
        });
    }

    #[test]
    fn grouped_upsample_matches() {
        let mut r = Lcg(5);
        let x = Array2::from_shape_fn((8, 12), |(c, t)| ((t + c) as f32 * 0.15).cos());
        let weight = r.a3(8, 1, 4); // depthwise [channels, 1, K]
        let reference = grouped_conv_transpose1d(x.view(), &weight, 2, 1.0);
        check("grouped_upsample", &x, &reference, |ctx, xin, c, t| {
            upsample(ctx, xin, c, t, &weight, 2, 1.0)
        });
    }

    #[test]
    fn resnet_block_matches() {
        let mut r = Lcg(6);
        let block = ResnetBlock {
            conv_a: conv_layer(&mut r, 4, 4, 3, 1, 1, PadMode::Constant),
            conv_b: conv_layer(&mut r, 4, 4, 1, 1, 1, PadMode::Constant),
        };
        let x = Array2::from_shape_fn((4, 30), |(c, t)| (t as f32 * 0.1).sin() + c as f32 * 0.05);
        let reference = block.forward(x.view());
        check("resnet_block", &x, &reference, |ctx, xin, c, t| {
            resnet_block(ctx, xin, c, t, &block)
        });
    }

    fn a2(r: &mut Lcg, rows: usize, cols: usize, scale: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |_| r.f() * scale)
    }

    fn tiny_transformer(
        r: &mut Lcg,
        c: usize,
        h: usize,
        d: usize,
        inter: usize,
        n_layers: usize,
        window: usize,
    ) -> MimiTransformer {
        use crate::transformer::{LayerScale, Mlp};
        let half = d / 2;
        let inv_freq = Array1::from_shape_fn(half, |i| {
            1.0 / (10000f64.powf(i as f64 / half as f64) as f32)
        });
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            layers.push(TransformerLayer {
                input_norm_w: Array1::from_shape_fn(c, |_| 1.0 + r.f() * 0.1),
                input_norm_b: r.a1(c),
                post_norm_w: Array1::from_shape_fn(c, |_| 1.0 + r.f() * 0.1),
                post_norm_b: r.a1(c),
                attn: Attention {
                    q_w: a2(r, c, c, 0.3),
                    k_w: a2(r, c, c, 0.3),
                    v_w: a2(r, c, c, 0.3),
                    o_w: a2(r, c, c, 0.3),
                    num_heads: h,
                    head_dim: d,
                    scaling: 1.0 / (d as f32).sqrt(),
                },
                attn_scale: LayerScale {
                    scale: Array1::from_shape_fn(c, |_| 0.1 + r.f() * 0.02),
                },
                mlp: Mlp {
                    fc1_w: a2(r, inter, c, 0.3),
                    fc2_w: a2(r, c, inter, 0.3),
                },
                mlp_scale: LayerScale {
                    scale: Array1::from_shape_fn(c, |_| 0.1 + r.f() * 0.02),
                },
            });
        }
        MimiTransformer {
            layers,
            inv_freq,
            sliding_window: window,
            norm_eps: 1e-5,
        }
    }

    fn run_tc(dev: Device, x_tc: &Array2<f32>, tf: &MimiTransformer) -> Array2<f32> {
        let (t, c) = x_tc.dim();
        let mut hir = HirModule::new("tf");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);
        let xin = ctx.g.input("x", Shape::new(&[t, c], F32));
        let out = transformer(&mut ctx, xin, t, c, tf);
        let params = std::mem::take(&mut ctx.params);
        hir.set_outputs(vec![out]);
        let graph = Graph::from_hir(hir).unwrap();
        let mut compiled = Session::new(dev).compile(graph);
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        compiled.finalize_params();
        let flat: Vec<f32> = x_tc.iter().copied().collect();
        let got = compiled.run(&[("x", &flat)]).into_iter().next().unwrap();
        Array2::from_shape_vec((t, c), got).unwrap()
    }

    #[test]
    fn transformer_matches() {
        let mut r = Lcg(7);
        let (c, h, d, inter, t) = (8usize, 2usize, 4usize, 16usize, 12usize);
        let tf = tiny_transformer(&mut r, c, h, d, inter, 2, 100);
        let x = Array2::from_shape_fn((t, c), |(ti, ci)| ((ti + ci) as f32 * 0.13).sin() * 0.5);
        let reference = tf.forward(x.view());
        for dev in devices() {
            let got = run_tc(dev, &x, &tf);
            let err = max_abs(&got, &reference);
            assert!(err < 3e-3, "transformer on {dev:?}: max|Δ| = {err}");
            eprintln!("transformer {dev:?} ok: max|Δ| = {err:.2e}");
        }
    }

    #[test]
    fn transformer_sliding_window() {
        let mut r = Lcg(8);
        let (c, h, d, inter, t) = (8usize, 2usize, 4usize, 16usize, 20usize);
        let tf = tiny_transformer(&mut r, c, h, d, inter, 1, 4); // narrow window
        let x = Array2::from_shape_fn((t, c), |(ti, ci)| ((ti * 2 + ci) as f32 * 0.09).cos() * 0.5);
        let reference = tf.forward(x.view());
        for dev in devices() {
            let got = run_tc(dev, &x, &tf);
            let err = max_abs(&got, &reference);
            assert!(
                err < 3e-3,
                "transformer(window=4) on {dev:?}: max|Δ| = {err}"
            );
            eprintln!("transformer_sliding_window {dev:?} ok: max|Δ| = {err:.2e}");
        }
    }
}
