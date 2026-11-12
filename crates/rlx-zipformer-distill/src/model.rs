// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! The student as an RLX graph: `rlx-kroko-core`'s chunk computation,
//! batched over `B` segments and unrolled over `K` chunks from a zero state.
//!
//! Inputs: `feats [B, segment_frames(K), 80]` and `target [B, K·8, 512]`
//! (the teacher's encoder output). Output: the loss, mean squared error
//! normalized by the target's mean square (1.0 = predicting zero).
//!
//! [`build_kd`] trains on what decides the transcript instead: `target` is
//! `[B, K·8, 1024]`, the teacher's encoder output then the predictor output
//! of its greedy context at each frame, and the loss is the KL divergence
//! from the teacher's joiner distribution to the student's (same predictor
//! and joiner), plus [`Kd::mse_weight`] times the squared error above.
//!
//! Every parameter is declared under the tensor name `rlx-kroko`'s exporter
//! reads, with the checkpoint's shape (`[out, in]` linears, `[C, 1, k]`
//! depthwise convs), so the trained values save as they are.

use rlx_tensor::{DType, Func, GraphScope, Tensor, shape};

use crate::config::{
    CHUNK_SHIFT, CHUNK_SIZE, EMBED_DIM, FRAMES, N_MELS, OUT_FRAMES, POS_DIM, POS_HEAD,
    StudentConfig,
};

/// Additive mask of a not-yet-filled left-context frame (as the core).
const MASKED: f64 = -1000.0;
const C0: usize = 8;
const C4: usize = 32;
const C7: usize = 128;
const HID: usize = 384;
/// Frequency bins after the embed convs.
const F7: usize = ((N_MELS - 3) / 2 + 1 - 3) / 2 + 1;

/// The token-distribution loss of [`build_kd`].
pub struct Kd {
    /// Joiner output weight `[vocab, 512]` (the teacher's, fixed).
    pub out_w: Vec<f32>,
    /// Joiner output bias `[vocab]`.
    pub out_b: Vec<f32>,
    /// Weight of the normalized encoder-output squared error.
    pub mse_weight: f32,
    /// Softmax temperature of both distributions (the KL is scaled by its
    /// square, as usual, so gradients keep their size).
    pub temperature: f32,
    /// Weight of a frame where the teacher emits a token, relative to a
    /// blank frame: each frame counts `1 + (token_weight − 1)·(1 − p_blank)`
    /// under the teacher's distribution. Blank frames are most of the data
    /// and easy, so at 1 the student learns to stay silent through short
    /// tokens (word endings, function words).
    pub token_weight: f32,
    /// The blank token id.
    pub blank: usize,
}

enum Loss<'a> {
    None,
    Mse,
    Kd(&'a Kd),
}

/// Build the training graph for `batch` segments of `k` chunks.
pub fn build(cfg: &StudentConfig, batch: usize, k: usize) -> Func {
    let cfg = cfg.clone();
    Func::new("zipformer-student", move |s| {
        forward(s, &cfg, batch, k, Loss::Mse)
    })
}

/// The training graph with the token-distribution loss (`target [B, K·8, 1024]`).
pub fn build_kd(cfg: &StudentConfig, batch: usize, k: usize, kd: std::sync::Arc<Kd>) -> Func {
    let cfg = cfg.clone();
    Func::new("zipformer-student-kd", move |s| {
        forward(s, &cfg, batch, k, Loss::Kd(&kd))
    })
}

/// The same graph returning the encoder output `[B, K·8, 512]` (no target).
pub fn build_encoder(cfg: &StudentConfig, batch: usize, k: usize) -> Func {
    let cfg = cfg.clone();
    Func::new("zipformer-student-enc", move |s| {
        forward(s, &cfg, batch, k, Loss::None)
    })
}

fn swoosh_l(x: &Tensor) -> Tensor {
    &(&(x - 4.0f32).softplus() - &(x * 0.08f32)) - 0.035f32
}

fn swoosh_r(x: &Tensor) -> Tensor {
    &(&(x - 1.0f32).softplus() - &(x * 0.08f32)) - 0.313_261_7f32
}

fn zeros(s: &mut GraphScope, dims: &[usize]) -> Tensor {
    let n = dims.iter().product();
    s.constant_nd(vec![0.0; n], dims.to_vec(), DType::F32)
}

fn r(x: &Tensor, dims: &[usize]) -> Tensor {
    x.reshape(dims.iter().map(|&d| d as i64).collect::<Vec<_>>())
}

/// `x[N, din] · Wᵀ + b` with `W [dout, din]`.
fn lin(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    &x.matmul_t(w) + b
}

/// BiasNorm over the last axis: `x · e^log_scale / rms(x − bias)`.
fn bias_norm(x: &Tensor, bias: &Tensor, log_scale: &Tensor) -> Tensor {
    let z = x - bias;
    let ms = (&z * &z).mean(vec![x.dims().len() - 1], true);
    &(x * &ms.rsqrt()) * &log_scale.exp()
}

/// `orig + (x − orig) · scale`.
fn bypass(x: &Tensor, orig: &Tensor, scale: &Tensor) -> Tensor {
    orig + &(&(x - orig) * scale)
}

struct Lin {
    w: Tensor,
    b: Tensor,
}

fn p_lin(s: &mut GraphScope, name: &str, o: usize, i: usize) -> Lin {
    Lin {
        w: s.param(format!("{name}.weight"), shape![o, i]),
        b: s.param(format!("{name}.bias"), shape![o]),
    }
}

/// A layer matrix as the MCU export stores it (`rlx_kroko::mcu`): at
/// `bits = Some(4)` one int4 scale per 32 values (16 when 32 does not divide
/// the row, per-row int8 when neither does); `Some(8)` one int8 scale per row.
fn q_lin(s: &mut GraphScope, bits: Option<u8>, name: &str, o: usize, i: usize) -> Lin {
    let Lin { w, b } = p_lin(s, name, o, i);
    let ste = rlx_tensor::SteKind::Identity;
    let w = match bits {
        None => w,
        Some(4) if i.is_multiple_of(16) => {
            let g = if i.is_multiple_of(32) { 32 } else { 16 };
            r(
                &r(&w, &[o * i / g, g]).fake_quantize(4, Some(0), ste),
                &[o, i],
            )
        }
        Some(_) => w.fake_quantize(8, Some(0), ste),
    };
    Lin { w, b }
}

struct ConvW {
    in_proj: Lin,
    causal_w: Tensor,
    causal_b: Tensor,
    chunk_w: Tensor,
    chunk_b: Tensor,
    left: Tensor,
    right: Tensor,
    out_proj: Lin,
}

struct LayerW {
    bypass: Tensor,
    bypass_mid: Tensor,
    norm_bias: Tensor,
    norm_log_scale: Tensor,
    attn_in: Lin,
    linear_pos: Tensor,
    ff: [(Lin, Lin); 3],
    nonlin_in: Lin,
    nonlin_out: Lin,
    sa: [(Lin, Lin); 2],
    conv: [ConvW; 2],
}

fn layer_params(s: &mut GraphScope, cfg: &StudentConfig, st: usize, l: usize) -> LayerW {
    let p = StudentConfig::layer_prefix(st, l);
    let (d, h, k) = (cfg.dims[st], cfg.heads[st], cfg.kernels[st]);
    let (hq, hv, nl) = (h * cfg.qdim, h * cfg.vdim, 3 * d / 4);
    let ff = cfg.ff[st];
    let qb = cfg.qat_bits;
    let mut ffw = |i: usize| {
        (
            q_lin(
                s,
                qb,
                &format!("{p}feed_forward{}.in_proj", i + 1),
                ff[i],
                d,
            ),
            q_lin(
                s,
                qb,
                &format!("{p}feed_forward{}.out_proj", i + 1),
                d,
                ff[i],
            ),
        )
    };
    let ff = [ffw(0), ffw(1), ffw(2)];
    let mut conv = |c: &str| {
        let dc = format!("{p}{c}.depthwise_conv");
        ConvW {
            in_proj: q_lin(s, qb, &format!("{p}{c}.in_proj"), 2 * d, d),
            causal_w: s.param(format!("{dc}.causal_conv.weight"), shape![d, 1, k / 2 + 1]),
            causal_b: s.param(format!("{dc}.causal_conv.bias"), shape![d]),
            chunk_w: s.param(format!("{dc}.chunkwise_conv.weight"), shape![d, 1, k]),
            chunk_b: s.param(format!("{dc}.chunkwise_conv.bias"), shape![d]),
            left: s.param(format!("{dc}.chunk_scale_left"), shape![d, k]),
            right: s.param(format!("{dc}.chunk_scale_right"), shape![d, k]),
            out_proj: q_lin(s, qb, &format!("{p}{c}.out_proj"), d, d),
        }
    };
    let conv = [conv("conv_module1"), conv("conv_module2")];
    LayerW {
        bypass: s.param(format!("{p}bypass.bypass_scale"), shape![d]),
        bypass_mid: s.param(format!("{p}bypass_mid.bypass_scale"), shape![d]),
        norm_bias: s.param(format!("{p}norm.bias"), shape![d]),
        norm_log_scale: s.param(format!("{p}norm.log_scale"), shape![1]),
        attn_in: q_lin(
            s,
            qb,
            &format!("{p}self_attn_weights.in_proj"),
            2 * hq + h * POS_HEAD,
            d,
        ),
        linear_pos: s.param(
            format!("{p}self_attn_weights.linear_pos.weight"),
            shape![h * POS_HEAD, POS_DIM],
        ),
        ff,
        nonlin_in: q_lin(s, qb, &format!("{p}nonlin_attention.in_proj"), 3 * nl, d),
        nonlin_out: q_lin(s, qb, &format!("{p}nonlin_attention.out_proj"), d, nl),
        sa: [
            (
                q_lin(s, qb, &format!("{p}self_attn1.in_proj"), hv, d),
                q_lin(s, qb, &format!("{p}self_attn1.out_proj"), d, hv),
            ),
            (
                q_lin(s, qb, &format!("{p}self_attn2.in_proj"), hv, d),
                q_lin(s, qb, &format!("{p}self_attn2.out_proj"), d, hv),
            ),
        ],
        conv,
    }
}

/// Compact relative positions (as the core), rows for offsets
/// `-(left+t-1) ..= t-1`, gathered for every (query, key) pair:
/// `M[tt, j] = table[(t-1-tt) + j]`, `[t·(left+t), POS_DIM]`. A constant;
/// times `linear_posᵀ` it is the per-pair position projection.
fn shifted_positions(t: usize, left: usize) -> Vec<f64> {
    let len = left + 2 * t - 1;
    let compression = (POS_DIM as f64).sqrt();
    let length_scale = POS_DIM as f64 / (2.0 * std::f64::consts::PI);
    let mut table = vec![0f64; len * POS_DIM];
    for row in 0..len {
        // f32 throughout, as the core computes it.
        let x = row as f32 - (left + t - 1) as f32;
        let c = compression as f32;
        let compressed = c * x.signum() * ((x.abs() + c).ln() - c.ln());
        let compressed = if x == 0.0 { 0.0 } else { compressed };
        let angle = (compressed / length_scale as f32).atan();
        for d in 0..POS_DIM / 2 {
            let f = (d + 1) as f32;
            table[row * POS_DIM + 2 * d] = f64::from((angle * f).cos());
            table[row * POS_DIM + 2 * d + 1] = f64::from((angle * f).sin());
        }
        table[row * POS_DIM + POS_DIM - 1] = 1.0;
    }
    let src = left + t;
    let mut m = vec![0f64; t * src * POS_DIM];
    for tt in 0..t {
        for j in 0..src {
            let row = (t - 1 - tt) + j;
            m[(tt * src + j) * POS_DIM..(tt * src + j + 1) * POS_DIM]
                .copy_from_slice(&table[row * POS_DIM..(row + 1) * POS_DIM]);
        }
    }
    m
}

/// Per-layer caches carried from chunk to chunk.
struct Caches {
    key: Tensor,
    nonlin: Tensor,
    val: [Tensor; 2],
    conv: [Tensor; 2],
}

/// Depthwise conv over time of `x [B, t_in, d]` with weight `[d, 1, k]`,
/// `pad` zeros each side → `[B, t_out, d]`.
fn depthwise(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    bsz: usize,
    d: usize,
    k: usize,
    pad: usize,
) -> Tensor {
    let t_in = x.dims()[1];
    let xn = r(&x.transpose(vec![0, 2, 1]), &[bsz, d, 1, t_in]);
    let wn = r(w, &[d, 1, 1, k]);
    let y = xn.conv2d(&wn, [1, k], [1, 1], [0, pad], [1, 1], d);
    let t_out = y.dims()[3];
    let y = &r(&y, &[bsz, d, t_out]) + &r(b, &[1, d, 1]);
    y.transpose(vec![0, 2, 1])
}

#[allow(clippy::too_many_arguments)]
fn conv_module(
    s: &mut GraphScope,
    x: &Tensor,
    w: &ConvW,
    cache: &Tensor,
    bsz: usize,
    t: usize,
    d: usize,
    k: usize,
) -> (Tensor, Tensor) {
    let half = k / 2;
    let pr = lin(x, &w.in_proj.w, &w.in_proj.b); // [B*t, 2d]
    let a = pr.narrow(1, 0, d);
    let g = pr.narrow(1, d, d);
    let mixed = r(&(&a * &g.sigmoid()), &[bsz, t, d]);
    let ccat = rlx_tensor::cat(&[cache, &mixed], 1); // [B, half+t, d]
    let new_cache = ccat.narrow(1, t, half);
    let y = depthwise(&ccat, &w.causal_w, &w.causal_b, bsz, d, half + 1, 0);
    let z = depthwise(&mixed, &w.chunk_w, &w.chunk_b, bsz, d, k, half);
    // Edge scales: every stack chunk is shorter than its kernel here.
    assert!(t < k, "chunk of {t} frames vs kernel {k}");
    let edge = &w.left.narrow(1, 0, t) + &w.right.narrow(1, k - t, t); // [d, t]
    let edge = r(&edge.transpose(vec![1, 0]), &[1, t, d]);
    let one = s.constant(1.0, DType::F32);
    let out = swoosh_r(&(&y + &(&z * &(&one + &edge))));
    let out = lin(&r(&out, &[bsz * t, d]), &w.out_proj.w, &w.out_proj.b);
    (out, new_cache)
}

#[allow(clippy::too_many_arguments)]
fn layer(
    s: &mut GraphScope,
    cfg: &StudentConfig,
    st: usize,
    w: &LayerW,
    x: Tensor, // [B*t, d]
    caches: &mut Caches,
    bsz: usize,
    t: usize,
    valid_left: usize,
) -> Tensor {
    let (d, h, k, l) = (cfg.dims[st], cfg.heads[st], cfg.kernels[st], cfg.left(st));
    let (q, v) = (cfg.qdim, cfg.vdim);
    let (hq, hv, nl) = (h * q, h * v, 3 * d / 4);
    let src = l + t;
    let orig = x.clone();

    // Attention weights, shared by the non-linear attention and both
    // self-attentions.
    let pr = lin(&x, &w.attn_in.w, &w.attn_in.b); // [B*t, 2hq + h*4]
    let qq = r(&pr.narrow(1, 0, hq), &[bsz, t, h, q]).transpose(vec![0, 2, 1, 3]);
    let kk = r(&pr.narrow(1, hq, hq), &[bsz, t, hq]);
    let pq = r(&pr.narrow(1, 2 * hq, h * POS_HEAD), &[bsz, t, h, POS_HEAD]);
    let kcat = rlx_tensor::cat(&[&caches.key, &kk], 1); // [B, src, hq]
    caches.key = kcat.narrow(1, t, l);
    let kt = r(&kcat, &[bsz, src, h, q]).transpose(vec![0, 2, 3, 1]); // [B,h,q,src]
    let mut scores = qq.matmul(&kt); // [B,h,t,src]
    let m = s.constant_nd(shifted_positions(t, l), vec![t * src, POS_DIM], DType::F32);
    let pshift = r(&m.matmul_t(&w.linear_pos), &[t, src, h, POS_HEAD]);
    let pk = pshift.transpose(vec![0, 2, 3, 1]); // [t,h,4,src]
    let pqt = pq.transpose(vec![1, 2, 0, 3]); // [t,h,B,4]
    let pos = pqt.matmul(&pk).transpose(vec![2, 1, 0, 3]); // [B,h,t,src]
    scores = &scores + &pos;
    let masked = l - valid_left.min(l);
    if masked > 0 {
        let mut mv = vec![0f64; src];
        mv[..masked].fill(MASKED);
        scores = &scores + &s.constant_nd(mv, vec![1, 1, 1, src], DType::F32);
    }
    let attn = scores.softmax(-1);

    let ffw = |x: &Tensor, i: usize| {
        let (a, b) = &w.ff[i];
        x + &lin(&swoosh_l(&lin(x, &a.w, &a.b)), &b.w, &b.b)
    };
    let mut x = ffw(&x, 0);

    // Non-linear attention (head 0's weights).
    let p3 = lin(&x, &w.nonlin_in.w, &w.nonlin_in.b); // [B*t, 3nl]
    let mixed = &p3.narrow(1, nl, nl) * &p3.narrow(1, 0, nl).tanh();
    let ncat = rlx_tensor::cat(&[&caches.nonlin, &r(&mixed, &[bsz, t, nl])], 1);
    caches.nonlin = ncat.narrow(1, t, l);
    let a0 = r(&attn.narrow(1, 0, 1), &[bsz, t, src]);
    let agg = r(&a0.matmul(&ncat), &[bsz * t, nl]);
    let gated = &agg * &p3.narrow(1, 2 * nl, nl);
    x = &x + &lin(&gated, &w.nonlin_out.w, &w.nonlin_out.b);

    let self_attn = |x: &Tensor, i: usize, cache: &mut Tensor| {
        let (a, b) = &w.sa[i];
        let vv = r(&lin(x, &a.w, &a.b), &[bsz, t, hv]);
        let vcat = rlx_tensor::cat(&[&*cache, &vv], 1);
        *cache = vcat.narrow(1, t, l);
        let vh = r(&vcat, &[bsz, src, h, v]).transpose(vec![0, 2, 1, 3]); // [B,h,src,v]
        let o = attn.matmul(&vh).transpose(vec![0, 2, 1, 3]); // [B,t,h,v]
        x + &lin(&r(&o, &[bsz * t, hv]), &b.w, &b.b)
    };
    x = self_attn(&x, 0, &mut caches.val[0]);
    let (c, nc) = conv_module(s, &x, &w.conv[0], &caches.conv[0], bsz, t, d, k);
    caches.conv[0] = nc;
    x = &x + &c;
    x = ffw(&x, 1);
    x = bypass(&x, &orig, &w.bypass_mid);
    x = self_attn(&x, 1, &mut caches.val[1]);
    let (c, nc) = conv_module(s, &x, &w.conv[1], &caches.conv[1], bsz, t, d, k);
    caches.conv[1] = nc;
    x = &x + &c;
    x = ffw(&x, 2);
    x = bias_norm(&x, &w.norm_bias, &w.norm_log_scale);
    bypass(&x, &orig, &w.bypass)
}

struct EmbedW {
    c0: (Tensor, Tensor),
    c4: (Tensor, Tensor),
    c7: (Tensor, Tensor),
    dw: (Tensor, Tensor),
    pw1: (Tensor, Tensor),
    pw2: (Tensor, Tensor),
    out: Lin,
    norm_bias: Tensor,
    norm_log_scale: Tensor,
}

fn embed_params(s: &mut GraphScope) -> EmbedW {
    let e = "encoder.encoder_embed";
    let mut cw = |n: &str, o: usize, i: usize, kh: usize, kw: usize| {
        (
            s.param(format!("{e}.{n}.weight"), shape![o, i, kh, kw]),
            s.param(format!("{e}.{n}.bias"), shape![o]),
        )
    };
    let c0 = cw("conv.0", C0, 1, 3, 3);
    let c4 = cw("conv.4", C4, C0, 3, 3);
    let c7 = cw("conv.7", C7, C4, 3, 3);
    let dw = cw("convnext.depthwise_conv", C7, 1, 7, 7);
    let pw1 = cw("convnext.pointwise_conv1", HID, C7, 1, 1);
    let pw2 = cw("convnext.pointwise_conv2", C7, HID, 1, 1);
    EmbedW {
        c0,
        c4,
        c7,
        dw,
        pw1,
        pw2,
        out: p_lin(s, &format!("{e}.out"), EMBED_DIM, C7 * F7),
        norm_bias: s.param(format!("{e}.out_norm.bias"), shape![EMBED_DIM]),
        norm_log_scale: s.param(format!("{e}.out_norm.log_scale"), shape![1]),
    }
}

/// One chunk's conv embed: `feats [B, 45, 80]` → `[B·16, 192]`; updates the
/// 3-frame conv7 cache `[B, 128, 3, F7]`.
fn embed(e: &EmbedW, feats: &Tensor, cache: &mut Tensor, bsz: usize) -> Tensor {
    let x = r(feats, &[bsz, 1, CHUNK_SIZE, N_MELS]);
    let bias4 = |b: &Tensor, c: usize| r(b, &[1, c, 1, 1]);
    let a0 =
        swoosh_r(&(&x.conv2d(&e.c0.0, [3, 3], [1, 1], [0, 1], [1, 1], 1) + &bias4(&e.c0.1, C0)));
    let a4 =
        swoosh_r(&(&a0.conv2d(&e.c4.0, [3, 3], [2, 2], [0, 0], [1, 1], 1) + &bias4(&e.c4.1, C4)));
    let a7n =
        swoosh_r(&(&a4.conv2d(&e.c7.0, [3, 3], [1, 2], [0, 0], [1, 1], 1) + &bias4(&e.c7.1, C7)));
    let a7 = rlx_tensor::cat(&[&*cache, &a7n], 2); // [B, 128, 3+T7, F7]
    *cache = a7.narrow(2, FRAMES, 3);
    let dw = &a7.conv2d(&e.dw.0, [7, 7], [1, 1], [0, 3], [1, 1], C7) + &bias4(&e.dw.1, C7); // [B,128,16,F7]
    let rows = r(&dw.transpose(vec![0, 2, 3, 1]), &[bsz * FRAMES * F7, C7]);
    let h = swoosh_l(&lin(&rows, &r(&e.pw1.0, &[HID, C7]), &e.pw1.1));
    let o = lin(&h, &r(&e.pw2.0, &[C7, HID]), &e.pw2.1); // [B*16*F7, 128]
    let byp = r(
        &a7.narrow(2, 3, FRAMES).transpose(vec![0, 2, 3, 1]),
        &[bsz * FRAMES * F7, C7],
    );
    let sum = r(&(&o + &byp), &[bsz, FRAMES, F7, C7]);
    let flat = r(&sum.transpose(vec![0, 1, 3, 2]), &[bsz * FRAMES, C7 * F7]); // channel-major
    let y = lin(&flat, &e.out.w, &e.out.b);
    bias_norm(&y, &e.norm_bias, &e.norm_log_scale)
}

fn forward(
    s: &mut GraphScope,
    cfg: &StudentConfig,
    bsz: usize,
    k: usize,
    loss: Loss<'_>,
) -> Tensor {
    let n = cfg.n_stacks();
    let md = cfg.max_dim();
    let seg = StudentConfig::segment_frames(k);
    let feats = s.input("feats", shape![bsz, seg, N_MELS]);

    let ew = embed_params(s);
    let mut lw: Vec<Vec<LayerW>> = Vec::new();
    let mut downs: Vec<Option<(Tensor, Tensor)>> = Vec::new();
    for st in 0..n {
        lw.push(
            (0..cfg.layers[st])
                .map(|l| layer_params(s, cfg, st, l))
                .collect(),
        );
        downs.push((cfg.downs[st] > 1).then(|| {
            (
                s.param(
                    format!("encoder.encoder.encoders.{st}.downsample.weights"),
                    shape![cfg.downs[st]],
                ),
                s.param(
                    format!("encoder.encoder.encoders.{st}.out_combiner.bypass_scale"),
                    shape![cfg.dims[st]],
                ),
            )
        }));
    }
    let down_out = s.param("encoder.encoder.downsample_output.weights", shape![2]);
    let enc_proj = p_lin(s, "encoder.encoder_proj", 512, md);

    // Zero states.
    let mut ecache = zeros(s, &[bsz, C7, 3, F7]);
    let mut caches: Vec<Vec<Caches>> = Vec::new();
    for st in 0..n {
        let (d, h, k_, l) = (cfg.dims[st], cfg.heads[st], cfg.kernels[st], cfg.left(st));
        caches.push(
            (0..cfg.layers[st])
                .map(|_| Caches {
                    key: zeros(s, &[bsz, l, h * cfg.qdim]),
                    nonlin: zeros(s, &[bsz, l, 3 * d / 4]),
                    val: [
                        zeros(s, &[bsz, l, h * cfg.vdim]),
                        zeros(s, &[bsz, l, h * cfg.vdim]),
                    ],
                    conv: [zeros(s, &[bsz, k_ / 2, d]), zeros(s, &[bsz, k_ / 2, d])],
                })
                .collect(),
        );
    }
    // Columns of the final concat each stack contributes (as the core).
    let mut cols = vec![(0usize, 0usize); n];
    let mut cur_dim = cfg.dims[n - 1];
    cols[n - 1] = (0, cur_dim);
    for st in (0..n - 1).rev() {
        if cfg.dims[st] > cur_dim {
            cols[st] = (cur_dim, cfg.dims[st]);
            cur_dim = cfg.dims[st];
        }
    }

    let mut outs = Vec::new();
    for c in 0..k {
        let fc = feats.narrow(1, c * CHUNK_SHIFT, CHUNK_SIZE);
        let mut cur = embed(&ew, &fc, &mut ecache, bsz); // [B*16, 192]
        let mut prev_dim = EMBED_DIM;
        let processed = c * FRAMES;
        let mut parts: Vec<Tensor> = Vec::new();
        for st in 0..n {
            let d = cfg.dims[st];
            let x3 = r(&cur, &[bsz, FRAMES, prev_dim]);
            let sin = if d <= prev_dim {
                x3.narrow(2, 0, d)
            } else {
                x3.pad(2, 0, d - prev_dim, 0.0)
            };
            let ds = cfg.downs[st];
            let t = FRAMES.div_ceil(ds);
            let mut x = match &downs[st] {
                Some((wd, _)) => {
                    let g = r(&sin, &[bsz, t, ds, d]);
                    r(
                        &(&g * &r(wd, &[1, 1, ds, 1])).sum(vec![2], false),
                        &[bsz * t, d],
                    )
                }
                None => r(&sin, &[bsz * t, d]),
            };
            for (li, w) in lw[st].iter().enumerate() {
                x = layer(
                    s,
                    cfg,
                    st,
                    w,
                    x,
                    &mut caches[st][li],
                    bsz,
                    t,
                    processed / ds,
                );
            }
            let y = match &downs[st] {
                Some((_, comb)) => {
                    let up = r(&x, &[bsz, t, 1, d])
                        .broadcast_to(vec![bsz as i64, t as i64, ds as i64, d as i64]);
                    let up = r(&up, &[bsz, FRAMES, d]);
                    &sin + &(&(&up - &sin) * comb)
                }
                None => r(&x, &[bsz, FRAMES, d]),
            };
            let (lo, hi) = cols[st];
            if hi > lo {
                parts.push(y.narrow(2, lo, hi - lo));
            }
            cur = r(&y, &[bsz * FRAMES, d]);
            prev_dim = d;
        }
        // Final concat in column order (stacks were visited first to last,
        // columns run last stack first).
        parts.reverse();
        let refs: Vec<&Tensor> = parts.iter().collect();
        let full = rlx_tensor::cat(&refs, 2); // [B, 16, md]
        let g = r(&full, &[bsz, OUT_FRAMES, 2, md]);
        let pooled = r(
            &(&g * &r(&down_out, &[1, 1, 2, 1])).sum(vec![2], false),
            &[bsz * OUT_FRAMES, md],
        );
        outs.push(r(
            &lin(&pooled, &enc_proj.w, &enc_proj.b),
            &[bsz, OUT_FRAMES, 512],
        ));
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    let enc = rlx_tensor::cat(&refs, 1); // [B, K*8, 512]
    let frames = k * OUT_FRAMES;
    let (t_enc, kd) = match loss {
        Loss::None => return enc,
        Loss::Mse => (s.input("target", shape![bsz, frames, 512]), None),
        Loss::Kd(kd) => {
            let t = s.input("target", shape![bsz, frames, 1024]);
            (t.narrow(2, 0, 512), Some((kd, t.narrow(2, 512, 512))))
        }
    };
    let diff = &enc - &t_enc;
    let mse = &(&diff * &diff).mean_all() / &(&t_enc * &t_enc).mean_all();
    let Some((kd, dec)) = kd else {
        return mse;
    };
    let vocab = kd.out_b.len();
    let to64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let w = s.constant_nd(to64(&kd.out_w), vec![vocab, 512], DType::F32);
    let b = s.constant_nd(to64(&kd.out_b), vec![vocab], DType::F32);
    let rows = bsz * frames;
    let inv_t = 1.0 / kd.temperature;
    let logits = |e: &Tensor| &lin(&r(&(e + &dec).tanh(), &[rows, 512]), &w, &b) * inv_t;
    let zs = logits(&enc);
    let zt = logits(&t_enc).stop_gradient();
    let pt = zt.softmax(1);
    // KL(teacher ‖ student) = cross-entropy − teacher entropy.
    let log_pt = &zt - &zt.logsumexp(1, true);
    let entropy = &(&pt * &log_pt).sum(vec![1], false) * -1.0f32;
    let kl_rows = &zs.cross_entropy(&pt) - &entropy; // [rows]
    let p_blank = r(&pt.narrow(1, kd.blank, 1), &[rows]);
    let weight = &(&(&p_blank * -1.0f32) + 1.0f32) * (kd.token_weight - 1.0) + 1.0f32;
    let kl = &(&kl_rows * &weight).sum_all() / &weight.sum_all();
    let t2 = kd.temperature * kd.temperature;
    &(&kl * t2) + &(&mse * kd.mse_weight)
}
