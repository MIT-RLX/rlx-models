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

//! FastConformer encoder HIR. Layout, per NeMo `ConformerEncoder`:
//!   * `dw_striding` subsampling — three stride-2 2-D convs (first full,
//!     the rest depthwise-separable) → 8× time reduction → linear to d_model;
//!   * N conformer blocks, each
//!     `½·FFN → rel-pos MHSA → ConvModule → ½·FFN → LayerNorm`;
//!   * Transformer-XL relative-position self-attention built explicitly
//!     (content `(q+u)·kᵀ` + position `rel_shift((q+v)·pᵀ)`), since RLX has
//!     no rel-pos attention primitive;
//!   * the conv module's frozen BatchNorm is folded into a per-channel
//!     affine at load time.
//!
//! Built for a single utterance / chunk (`batch == 1`). Cache-aware
//! streaming masks are applied by the runner.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use rlx_flow::WeightSource;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Op, Shape};

use crate::config::AsrConfig;
use crate::weights::keys;

const LN_EPS: f32 = 1e-5;

struct EncoderBuilder<'a> {
    hir: &'a mut HirModule,
    params: &'a mut HashMap<String, Vec<f32>>,
    weights: &'a mut dyn WeightSource,
    cfg: &'a AsrConfig,
    f: DType,
    uid: usize,
}

impl<'a> EncoderBuilder<'a> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    fn fresh(&mut self, tag: &str) -> String {
        self.uid += 1;
        format!("_enc.{tag}.{}", self.uid)
    }

    // ── param helpers ────────────────────────────────────────────────
    fn load(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    fn synth(&mut self, key: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let id = self.hir.param(key, Shape::new(shape, self.f));
        self.params.insert(key.to_string(), data);
        id
    }

    /// Load a param only if the key exists (probe via `take`, since the
    /// `&mut dyn WeightSource` `has()` is unreliable). Returns `None` when
    /// absent — Nemotron's linears/convs are largely bias-free.
    fn load_opt(&mut self, key: &str, transpose: bool) -> Result<Option<HirNodeId>> {
        match self.weights.take(key, transpose) {
            Ok((data, shape)) => {
                let id = self.hir.param(key, Shape::new(&shape, self.f));
                self.params.insert(key.to_string(), data);
                Ok(Some(id))
            }
            Err(_) => Ok(None),
        }
    }

    /// A unique broadcastable scalar param (HIR has no inline constant op).
    fn scalar(&mut self, v: f32) -> HirNodeId {
        let key = self.fresh("scalar");
        self.synth(&key, vec![v], &[1])
    }

    // ── basic layers ─────────────────────────────────────────────────
    /// `x @ Wᵀ (+ b)` for x `[1, T, in]`, weight key stored `[out, in]`.
    /// The bias is included only if the key is given *and present* in the
    /// checkpoint — Nemotron's FastConformer linears are bias-free.
    fn linear(&mut self, x: HirNodeId, w: &str, b: Option<&str>, out: usize) -> Result<HirNodeId> {
        let wt = self.load(w, true)?;
        let mut y = self.g().mm(x, wt);
        if let Some(bk) = b {
            if let Some(bias) = self.load_opt(bk, false)? {
                let b3 = self.g().reshape_(bias, vec![1, 1, out as i64]);
                y = self.g().add(y, b3);
            }
        }
        Ok(y)
    }

    fn layer_norm(&mut self, x: HirNodeId, w: &str, b: &str) -> Result<HirNodeId> {
        let g = self.load(w, false)?;
        let beta = self.load(b, false)?;
        Ok(self.g().ln(x, g, beta, LN_EPS))
    }

    /// σ(x) = 1 / (1 + e^{-x}).
    fn sigmoid(&mut self, x: HirNodeId) -> HirNodeId {
        let neg = self.g().neg(x);
        let e = self.g().exp(neg);
        let one = self.scalar(1.0);
        let den = self.g().add(e, one);
        let onen = self.scalar(1.0);
        self.g().div(onen, den)
    }

    // ── feed-forward (half-step macaron) ─────────────────────────────
    fn feed_forward(
        &mut self,
        x: HirNodeId,
        norm_w: &str,
        norm_b: &str,
        l1_w: &str,
        l1_b: &str,
        l2_w: &str,
        l2_b: &str,
    ) -> Result<HirNodeId> {
        let d = self.cfg.d_model;
        let ff = self.cfg.ff_dim();
        let h = self.layer_norm(x, norm_w, norm_b)?;
        let h = self.linear(h, l1_w, Some(l1_b), ff)?;
        let h = self.g().silu(h); // NeMo conformer FFN uses SiLU/Swish
        let h = self.linear(h, l2_w, Some(l2_b), d)?;
        // half-step residual scaling (macaron): 0.5 * ff
        let half = self.scalar(0.5);
        let h = self.g().mul(h, half);
        Ok(self.g().add(x, h))
    }

    // ── relative-position multi-head self-attention ──────────────────
    fn self_attention(
        &mut self,
        layer: usize,
        x: HirNodeId,
        pos: HirNodeId,
        t: usize,
    ) -> Result<HirNodeId> {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let nh = cfg.n_heads;
        let hd = cfg.head_dim();
        let scale = (hd as f32).powf(-0.5);
        let p = |s: &str| keys::enc_layer(layer, s);

        let normed = self.layer_norm(x, &p(keys::NORM_ATT_W), &p(keys::NORM_ATT_B))?;

        // Q, K, V: [1, T, d] -> [1, nh, T, hd]
        let q = self.linear(normed, &p(keys::ATT_Q_W), Some(&p(keys::ATT_Q_B)), d)?;
        let k = self.linear(normed, &p(keys::ATT_K_W), Some(&p(keys::ATT_K_B)), d)?;
        let v = self.linear(normed, &p(keys::ATT_V_W), Some(&p(keys::ATT_V_B)), d)?;
        let q = self.heads(q, t, nh, hd);
        let k = self.heads(k, t, nh, hd);
        let v = self.heads(v, t, nh, hd);

        // Positional projection: pos [1, 2T-1, d] -> [1, nh, 2T-1, hd]
        let pos_len = 2 * t - 1;
        let pe = self.linear(pos, &p(keys::ATT_POS_W), None, d)?;
        let pe = self.heads(pe, pos_len, nh, hd);

        // pos_bias_u / pos_bias_v: [nh, hd] -> broadcastable [1, nh, 1, hd]
        let bu = self.load(&p(keys::ATT_POS_U), false)?;
        let bu = self.g().reshape_(bu, vec![1, nh as i64, 1, hd as i64]);
        let bv = self.load(&p(keys::ATT_POS_V), false)?;
        let bv = self.g().reshape_(bv, vec![1, nh as i64, 1, hd as i64]);

        let q_u = self.g().add(q, bu); // [1,nh,T,hd]
        let q_v = self.g().add(q, bv);

        // content score: (q+u) @ kᵀ -> [1,nh,T,T]
        let k_t = self.g().transpose_(k, vec![0, 1, 3, 2]);
        let ac = self.g().mm(q_u, k_t);
        // position score: (q+v) @ peᵀ -> [1,nh,T,2T-1] then rel_shift -> [1,nh,T,T]
        let pe_t = self.g().transpose_(pe, vec![0, 1, 3, 2]);
        let bd = self.g().mm(q_v, pe_t);
        let bd = self.rel_shift(bd, nh, t)?;

        let scores = self.g().add(ac, bd);
        let sc = self.scalar(scale);
        let scores = self.g().mul(scores, sc);
        let attn = self.g().sm(scores, -1); // softmax over keys

        let ctx = self.g().mm(attn, v); // [1,nh,T,hd]
        let ctx = self.g().transpose_(ctx, vec![0, 2, 1, 3]); // [1,T,nh,hd]
        let ctx = self.g().reshape_(ctx, vec![1, t as i64, d as i64]);
        let out = self.linear(ctx, &p(keys::ATT_O_W), Some(&p(keys::ATT_O_B)), d)?;
        Ok(self.g().add(x, out))
    }

    /// `[1, T, d] -> [1, nh, T, hd]`.
    fn heads(&mut self, x: HirNodeId, t: usize, nh: usize, hd: usize) -> HirNodeId {
        let x = self
            .g()
            .reshape_(x, vec![1, t as i64, nh as i64, hd as i64]);
        self.g().transpose_(x, vec![0, 2, 1, 3])
    }

    /// Transformer-XL relative shift: `[1,nh,T,2T-1] -> [1,nh,T,T]`.
    fn rel_shift(&mut self, bd: HirNodeId, nh: usize, t: usize) -> Result<HirNodeId> {
        let pos_len = 2 * t - 1;
        // Prepend a zero column on the last axis.
        let zkey = self.fresh("relshift_zeros");
        let zeros = self.synth(&zkey, vec![0.0; nh * t], &[1, nh, t, 1]);
        let padded = self.g().concat_(vec![zeros, bd], 3); // [1,nh,T,2T]
        let padded = self
            .g()
            .reshape_(padded, vec![1, nh as i64, (pos_len + 1) as i64, t as i64]);
        let sliced = self.g().narrow_(padded, 2, 1, pos_len); // [1,nh,2T-1,T]
        let back = self
            .g()
            .reshape_(sliced, vec![1, nh as i64, t as i64, pos_len as i64]);
        Ok(self.g().narrow_(back, 3, 0, t)) // keep first T keys
    }

    // ── convolution module ───────────────────────────────────────────
    fn conv_module(&mut self, layer: usize, x: HirNodeId, t: usize) -> Result<HirNodeId> {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let k = cfg.conv_kernel;
        let p = |s: &str| keys::enc_layer(layer, s);

        let h = self.layer_norm(x, &p(keys::NORM_CONV_W), &p(keys::NORM_CONV_B))?;
        // pointwise_conv1: Conv1d(d, 2d, 1) == linear to 2d, then GLU -> d.
        let h = self.pointwise(
            h,
            &p(keys::CONV_PW1_W),
            Some(&p(keys::CONV_PW1_B)),
            d,
            2 * d,
        )?;
        let a = self.g().narrow_(h, 2, 0, d);
        let b = self.g().narrow_(h, 2, d, d);
        let gate = self.sigmoid(b);
        let h = self.g().mul(a, gate); // GLU -> [1,T,d]

        // depthwise_conv: Conv1d(d, d, k, groups=d), causal padding.
        let h = self.depthwise(h, &p(keys::CONV_DW_W), &p(keys::CONV_DW_B), d, k, t)?;
        // conv_norm_type = layer_norm (no BatchNorm running stats in the
        // checkpoint) — normalize over channels (the last axis of [1,T,d]).
        let h = self.layer_norm(h, &p(keys::CONV_BN_W), &p(keys::CONV_BN_B))?;
        let h = self.g().silu(h);
        // pointwise_conv2: Conv1d(d, d, 1) == linear.
        let h = self.pointwise(h, &p(keys::CONV_PW2_W), Some(&p(keys::CONV_PW2_B)), d, d)?;
        Ok(self.g().add(x, h))
    }

    /// Kernel-1 conv over `[1, T, in]` is a plain linear; weights are
    /// stored `[out, in, 1]` so squeeze to `[out, in]` before transpose.
    fn pointwise(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: Option<&str>,
        in_c: usize,
        out_c: usize,
    ) -> Result<HirNodeId> {
        let (wd, _) = self.weights.take(w, false)?; // [out, in, 1] flattened
        // transpose [out,in] -> [in,out]
        let mut wt = vec![0.0f32; in_c * out_c];
        for o in 0..out_c {
            for i in 0..in_c {
                wt[i * out_c + o] = wd[o * in_c + i];
            }
        }
        let wkey = self.fresh("pw_w");
        let wnode = self.synth(&wkey, wt, &[in_c, out_c]);
        let mut y = self.g().mm(x, wnode);
        if let Some(bk) = b {
            if let Some(bias) = self.load_opt(bk, false)? {
                let b3 = self.g().reshape_(bias, vec![1, 1, out_c as i64]);
                y = self.g().add(y, b3);
            }
        }
        Ok(y)
    }

    /// Depthwise conv1d on `[1, T, d]` (channels-last) with **causal**
    /// padding (`conv_context_size = causal`): pad `k-1` on the left only,
    /// so output length stays `T` and no future frames leak in. Internally
    /// moves to NCHW `[1, d, T+k-1, 1]`, convolves `groups == d`, returns
    /// `[1, T, d]`.
    fn depthwise(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: &str,
        d: usize,
        k: usize,
        t: usize,
    ) -> Result<HirNodeId> {
        let f = self.f;
        let pad_l = k - 1;
        let xt = self.g().transpose_(x, vec![0, 2, 1]); // [1,d,T]
        let nchw = self.g().reshape_(xt, vec![1, d as i64, t as i64, 1]);
        // Explicit left-pad along the time (H) axis for causality.
        let zkey = self.fresh("dw_causal_pad");
        let zeros = self.synth(&zkey, vec![0.0; d * pad_l], &[1, d, pad_l, 1]);
        let padded = self.g().concat_(vec![zeros, nchw], 2); // [1,d,T+k-1,1]
        let (wd, _) = self.weights.take(w, false)?; // [d,1,k]
        let wkey = self.fresh("dw_w");
        let wnode = self.synth(&wkey, wd, &[d, 1, k, 1]);
        let conv = self.g().add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: d,
            },
            vec![padded, wnode],
            Shape::new(&[1, d, t, 1], f),
        );
        let conv = if let Some(bias) = self.load_opt(b, false)? {
            let b4 = self.g().reshape_(bias, vec![1, d as i64, 1, 1]);
            self.g().add(conv, b4)
        } else {
            conv
        };
        let back = self.g().reshape_(conv, vec![1, d as i64, t as i64]);
        Ok(self.g().transpose_(back, vec![0, 2, 1])) // [1,T,d]
    }

    // ── one conformer block ──────────────────────────────────────────
    fn block(&mut self, layer: usize, x: HirNodeId, pos: HirNodeId, t: usize) -> Result<HirNodeId> {
        let p = |s: &str| keys::enc_layer(layer, s);
        let x = self.feed_forward(
            x,
            &p(keys::NORM_FF1_W),
            &p(keys::NORM_FF1_B),
            &p(keys::FF1_L1_W),
            &p(keys::FF1_L1_B),
            &p(keys::FF1_L2_W),
            &p(keys::FF1_L2_B),
        )?;
        let x = self.self_attention(layer, x, pos, t)?;
        let x = self.conv_module(layer, x, t)?;
        let x = self.feed_forward(
            x,
            &p(keys::NORM_FF2_W),
            &p(keys::NORM_FF2_B),
            &p(keys::FF2_L1_W),
            &p(keys::FF2_L1_B),
            &p(keys::FF2_L2_W),
            &p(keys::FF2_L2_B),
        )?;
        self.layer_norm(x, &p(keys::NORM_OUT_W), &p(keys::NORM_OUT_B))
    }

    // ── dw_striding subsampling: mel [1, n_mels, T] -> [1, T/8, d_model] ──
    //
    // Per stride-2 stage (kernel 3): time is **causal** (left-pad k-1) and
    // frequency uses NeMo's `ceil_mode` reduction (`f -> f/2 + 1`). For
    // n_mels=128: 128 -> 65 -> 33 -> 17, so the flattened width fed to
    // `pre_encode.out` is `conv_channels * 17`.
    fn subsample(&mut self, mel: HirNodeId, mel_frames: usize) -> Result<(HirNodeId, usize)> {
        let cfg = self.cfg;
        let c = cfg.subsampling_conv_channels;
        // mel [1, n_mels, T] -> [1, 1, T, n_mels] (channel=1, H=time, W=freq).
        let m = self.g().transpose_(mel, vec![0, 2, 1]); // [1,T,n_mels]
        let mut h = self
            .g()
            .reshape_(m, vec![1, 1, mel_frames as i64, cfg.n_mels as i64]);
        let mut t = mel_frames;
        let mut freq = cfg.n_mels;
        let mut in_c = 1usize;
        let n_stages = (cfg.subsampling_factor as f64).log2().round() as usize;
        for stage in 0..n_stages {
            // time: causal (left k-1, right 0); freq: symmetric 1 + ceil extra.
            let pt = 2usize; // k - 1
            let pf_r = 2 - (freq % 2); // 2 for even, 1 for odd -> ceil reduction
            let t2 = (t + pt - 3) / 2 + 1;
            let f2 = (freq + 1 + pf_r - 3) / 2 + 1;
            let (wkey, bkey) = if stage == 0 {
                (
                    keys::pre_encode_conv(0, "weight"),
                    keys::pre_encode_conv(0, "bias"),
                )
            } else {
                let base = 1 + (stage - 1) * 3; // ReLU(idx base) then dw,pw
                (
                    keys::pre_encode_conv(base + 1, "weight"),
                    keys::pre_encode_conv(base + 1, "bias"),
                )
            };
            let (gin, gout, groups) = if stage == 0 { (1, c, 1) } else { (c, c, c) };
            h = self.conv2d_pad(
                h,
                &wkey,
                Some(&bkey),
                gin,
                gout,
                [t, freq],
                3,
                2,
                [pt, 0, 1, pf_r],
                groups,
            )?;
            if stage > 0 {
                // pointwise C->C (kernel 1, stride 1, no padding).
                let base = 1 + (stage - 1) * 3;
                let pw = keys::pre_encode_conv(base + 2, "weight");
                let pb = keys::pre_encode_conv(base + 2, "bias");
                h = self.conv2d_pad(h, &pw, Some(&pb), c, c, [t2, f2], 1, 1, [0, 0, 0, 0], 1)?;
            }
            h = self.g().relu(h);
            t = t2;
            freq = f2;
            in_c = c;
        }
        // h: [1, c, t, freq] -> [1, t, c*freq] -> linear to d_model.
        let h = self.g().transpose_(h, vec![0, 2, 1, 3]); // [1,t,c,freq]
        let flat = c * freq;
        let h = self.g().reshape_(h, vec![1, t as i64, flat as i64]);
        let _ = in_c;
        let h = self.linear(
            h,
            keys::PRE_ENCODE_OUT_W,
            Some(keys::PRE_ENCODE_OUT_B),
            cfg.d_model,
        )?;
        Ok((h, t))
    }

    /// 2-D conv with explicit asymmetric padding `[top, bottom, left, right]`
    /// applied by zero-concatenation, then `Op::Conv` with zero padding —
    /// the only way to express causal/ceil padding through the symmetric
    /// `Op::Conv` padding field. `x` is `[1, in_c, H, W]`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_pad(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: Option<&str>,
        in_c: usize,
        out_c: usize,
        hw: [usize; 2],
        k: usize,
        stride: usize,
        pad: [usize; 4],
        groups: usize,
    ) -> Result<HirNodeId> {
        let f = self.f;
        let [pt, pb, pl, pr] = pad;
        let [h_in, w_in] = hw;
        // Pad along H (axis 2) then W (axis 3) with zero params.
        let mut node = x;
        let mut h_cur = h_in;
        if pt > 0 {
            let key = self.fresh("pad_top");
            let z = self.synth(&key, vec![0.0; in_c * pt * w_in], &[1, in_c, pt, w_in]);
            node = self.g().concat_(vec![z, node], 2);
            h_cur += pt;
        }
        if pb > 0 {
            let key = self.fresh("pad_bot");
            let z = self.synth(&key, vec![0.0; in_c * pb * w_in], &[1, in_c, pb, w_in]);
            node = self.g().concat_(vec![node, z], 2);
            h_cur += pb;
        }
        if pl > 0 {
            let key = self.fresh("pad_left");
            let z = self.synth(&key, vec![0.0; in_c * h_cur * pl], &[1, in_c, h_cur, pl]);
            node = self.g().concat_(vec![z, node], 3);
        }
        if pr > 0 {
            let key = self.fresh("pad_right");
            let z = self.synth(&key, vec![0.0; in_c * h_cur * pr], &[1, in_c, h_cur, pr]);
            node = self.g().concat_(vec![node, z], 3);
        }
        let h_pad = h_in + pt + pb;
        let w_pad = w_in + pl + pr;
        let h_out = (h_pad - k) / stride + 1;
        let w_out = (w_pad - k) / stride + 1;

        let (wd, _) = self.weights.take(w, false)?;
        let wkey = self.fresh("conv2d_w");
        let wpc = in_c / groups;
        let wnode = self.synth(&wkey, wd, &[out_c, wpc, k, k]);
        let conv = self.g().add_node(
            Op::Conv {
                kernel_size: vec![k, k],
                stride: vec![stride, stride],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups,
            },
            vec![node, wnode],
            Shape::new(&[1, out_c, h_out, w_out], f),
        );
        if let Some(bk) = b {
            if let Some(bias) = self.load_opt(bk, false)? {
                let b4 = self.g().reshape_(bias, vec![1, out_c as i64, 1, 1]);
                return Ok(self.g().add(conv, b4));
            }
        }
        Ok(conv)
    }
}

/// Sinusoidal Transformer-XL relative positional encoding `[2T-1, d]`.
fn rel_pos_encoding(t: usize, d: usize) -> Vec<f32> {
    let pos_len = 2 * t - 1;
    let mut pe = vec![0.0f32; pos_len * d];
    for (row, p) in (0..pos_len).enumerate() {
        // positions run T-1, T-2, …, 0, …, -(T-1).
        let pos = (t as f64 - 1.0) - p as f64;
        for i in (0..d).step_by(2) {
            let div = (-(10000f64.ln()) * i as f64 / d as f64).exp();
            let ang = pos * div;
            pe[row * d + i] = ang.sin() as f32;
            if i + 1 < d {
                pe[row * d + i + 1] = ang.cos() as f32;
            }
        }
    }
    pe
}

/// Build the FastConformer encoder graph for `mel_frames` mel columns.
/// Returns the HIR (output `[1, enc_frames, d_model]`) and its params.
pub fn build_encoder_hir(
    cfg: &AsrConfig,
    weights: &mut dyn WeightSource,
    mel_frames: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, usize)> {
    ensure!(mel_frames > 0, "mel_frames must be > 0");
    let f = DType::F32;
    let mut hir = HirModule::new("nemotron_asr_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[1, cfg.n_mels, mel_frames], f));

    let mut b = EncoderBuilder {
        hir: &mut hir,
        params: &mut params,
        weights,
        cfg,
        f,
        uid: 0,
    };

    let (mut x, t) = b.subsample(mel, mel_frames)?;
    ensure!(
        t > 0,
        "subsampling produced 0 frames for {mel_frames} mel frames"
    );

    // Relative positional encoding param, shared across layers.
    let pe = rel_pos_encoding(t, cfg.d_model);
    let pos = b.synth("_enc.rel_pos_encoding", pe, &[1, 2 * t - 1, cfg.d_model]);

    for layer in 0..cfg.n_layers {
        x = b.block(layer, x, pos, t)?;
    }
    hir.outputs = vec![x];
    Ok((hir, params, t))
}
