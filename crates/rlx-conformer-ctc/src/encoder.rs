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

//! Conformer encoder HIR for NeMo `ConformerEncoder` with **striding**
//! subsampling (classic Conformer-CTC).
//!
//! Layout:
//! - `striding` pre_encode — `log2(factor)` full Conv2d(k=3,s=2,pad=1) + ReLU
//!   → linear to `d_model`, then ×√`d_model` (`xscaling`, required for correct
//!   transcripts)
//! - N conformer blocks, each
//!   `½·FFN → rel-pos MHSA → ConvModule → ½·FFN → LayerNorm`
//! - ConvModule uses **same** depthwise padding and folds BatchNorm into a
//!   per-channel affine at load time
//!
//! Build with [`build_encoder_hir`]. FastConformer `dw_striding` is not
//! implemented here (see `rlx-nemotron-asr`).

use std::collections::HashMap;

use anyhow::{Result, bail, ensure};
use rlx_flow::WeightSource;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Op, Shape};

use crate::config::AsrConfig;
use crate::weights::keys;

const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;

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

    fn scalar(&mut self, v: f32) -> HirNodeId {
        let key = self.fresh("scalar");
        self.synth(&key, vec![v], &[1])
    }

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

    fn sigmoid(&mut self, x: HirNodeId) -> HirNodeId {
        let neg = self.g().neg(x);
        let e = self.g().exp(neg);
        let one = self.scalar(1.0);
        let den = self.g().add(e, one);
        let onen = self.scalar(1.0);
        self.g().div(onen, den)
    }

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
        let h = self.g().silu(h);
        let h = self.linear(h, l2_w, Some(l2_b), d)?;
        let half = self.scalar(0.5);
        let h = self.g().mul(h, half);
        Ok(self.g().add(x, h))
    }

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

        let q = self.linear(normed, &p(keys::ATT_Q_W), Some(&p(keys::ATT_Q_B)), d)?;
        let k = self.linear(normed, &p(keys::ATT_K_W), Some(&p(keys::ATT_K_B)), d)?;
        let v = self.linear(normed, &p(keys::ATT_V_W), Some(&p(keys::ATT_V_B)), d)?;
        let q = self.heads(q, t, nh, hd);
        let k = self.heads(k, t, nh, hd);
        let v = self.heads(v, t, nh, hd);

        let pos_len = 2 * t - 1;
        let pe = self.linear(pos, &p(keys::ATT_POS_W), None, d)?;
        let pe = self.heads(pe, pos_len, nh, hd);

        let bu = self.load(&p(keys::ATT_POS_U), false)?;
        let bu = self.g().reshape_(bu, vec![1, nh as i64, 1, hd as i64]);
        let bv = self.load(&p(keys::ATT_POS_V), false)?;
        let bv = self.g().reshape_(bv, vec![1, nh as i64, 1, hd as i64]);

        let q_u = self.g().add(q, bu);
        let q_v = self.g().add(q, bv);

        let k_t = self.g().transpose_(k, vec![0, 1, 3, 2]);
        let ac = self.g().mm(q_u, k_t);
        let pe_t = self.g().transpose_(pe, vec![0, 1, 3, 2]);
        let bd = self.g().mm(q_v, pe_t);
        let bd = self.rel_shift(bd, nh, t)?;

        let scores = self.g().add(ac, bd);
        let sc = self.scalar(scale);
        let scores = self.g().mul(scores, sc);
        let attn = self.g().sm(scores, -1);

        let ctx = self.g().mm(attn, v);
        let ctx = self.g().transpose_(ctx, vec![0, 2, 1, 3]);
        let ctx = self.g().reshape_(ctx, vec![1, t as i64, d as i64]);
        let out = self.linear(ctx, &p(keys::ATT_O_W), Some(&p(keys::ATT_O_B)), d)?;
        Ok(self.g().add(x, out))
    }

    fn heads(&mut self, x: HirNodeId, t: usize, nh: usize, hd: usize) -> HirNodeId {
        let x = self
            .g()
            .reshape_(x, vec![1, t as i64, nh as i64, hd as i64]);
        self.g().transpose_(x, vec![0, 2, 1, 3])
    }

    fn rel_shift(&mut self, bd: HirNodeId, nh: usize, t: usize) -> Result<HirNodeId> {
        let pos_len = 2 * t - 1;
        let zkey = self.fresh("relshift_zeros");
        let zeros = self.synth(&zkey, vec![0.0; nh * t], &[1, nh, t, 1]);
        let padded = self.g().concat_(vec![zeros, bd], 3);
        let padded = self
            .g()
            .reshape_(padded, vec![1, nh as i64, (pos_len + 1) as i64, t as i64]);
        let sliced = self.g().narrow_(padded, 2, 1, pos_len);
        let back = self
            .g()
            .reshape_(sliced, vec![1, nh as i64, t as i64, pos_len as i64]);
        Ok(self.g().narrow_(back, 3, 0, t))
    }

    fn conv_module(&mut self, layer: usize, x: HirNodeId, t: usize) -> Result<HirNodeId> {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let k = cfg.conv_kernel;
        let p = |s: &str| keys::enc_layer(layer, s);

        let h = self.layer_norm(x, &p(keys::NORM_CONV_W), &p(keys::NORM_CONV_B))?;
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
        let h = self.g().mul(a, gate);

        let h = self.depthwise_same(h, &p(keys::CONV_DW_W), &p(keys::CONV_DW_B), d, k, t)?;
        let h = self.batch_norm_affine(h, layer, d)?;
        let h = self.g().silu(h);
        let h = self.pointwise(h, &p(keys::CONV_PW2_W), Some(&p(keys::CONV_PW2_B)), d, d)?;
        Ok(self.g().add(x, h))
    }

    /// Fold BatchNorm into `y = x * scale + shift` (channels last `[1,T,d]`).
    fn batch_norm_affine(&mut self, x: HirNodeId, layer: usize, d: usize) -> Result<HirNodeId> {
        let p = |s: &str| keys::enc_layer(layer, s);
        let (gamma, _) = self.weights.take(&p(keys::CONV_BN_W), false)?;
        let (beta, _) = self.weights.take(&p(keys::CONV_BN_B), false)?;
        let (mean, _) = self.weights.take(&p(keys::CONV_BN_MEAN), false)?;
        let (var, _) = self.weights.take(&p(keys::CONV_BN_VAR), false)?;
        ensure!(
            gamma.len() == d && beta.len() == d && mean.len() == d && var.len() == d,
            "batch_norm shapes for layer {layer}"
        );
        let mut scale = vec![0.0f32; d];
        let mut shift = vec![0.0f32; d];
        for i in 0..d {
            let s = gamma[i] / (var[i] + BN_EPS).sqrt();
            scale[i] = s;
            shift[i] = beta[i] - mean[i] * s;
        }
        let sk = self.fresh("bn_scale");
        let bk = self.fresh("bn_shift");
        let snode = self.synth(&sk, scale, &[1, 1, d]);
        let bnode = self.synth(&bk, shift, &[1, 1, d]);
        let y = self.g().mul(x, snode);
        Ok(self.g().add(y, bnode))
    }

    fn pointwise(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: Option<&str>,
        in_c: usize,
        out_c: usize,
    ) -> Result<HirNodeId> {
        let (wd, _) = self.weights.take(w, false)?;
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

    /// Depthwise conv1d with **same** padding (offline Conformer).
    fn depthwise_same(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: &str,
        d: usize,
        k: usize,
        t: usize,
    ) -> Result<HirNodeId> {
        let f = self.f;
        let pad = (k - 1) / 2;
        let xt = self.g().transpose_(x, vec![0, 2, 1]);
        let nchw = self.g().reshape_(xt, vec![1, d as i64, t as i64, 1]);
        let zkey_l = self.fresh("dw_pad_l");
        let zkey_r = self.fresh("dw_pad_r");
        let zeros_l = self.synth(&zkey_l, vec![0.0; d * pad], &[1, d, pad, 1]);
        let zeros_r = self.synth(&zkey_r, vec![0.0; d * pad], &[1, d, pad, 1]);
        let padded = self.g().concat_(vec![zeros_l, nchw, zeros_r], 2);
        let (wd, _) = self.weights.take(w, false)?;
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
        Ok(self.g().transpose_(back, vec![0, 2, 1]))
    }

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

    /// Classic `striding` subsampling: full Conv2d per stage (indices 0,2,…).
    fn subsample_striding(
        &mut self,
        mel: HirNodeId,
        mel_frames: usize,
    ) -> Result<(HirNodeId, usize)> {
        let cfg = self.cfg;
        let c = cfg.subsampling_conv_channels;
        let m = self.g().transpose_(mel, vec![0, 2, 1]);
        let mut h = self
            .g()
            .reshape_(m, vec![1, 1, mel_frames as i64, cfg.n_mels as i64]);
        let mut t = mel_frames;
        let mut freq = cfg.n_mels;
        let n_stages = (cfg.subsampling_factor as f64).log2().round() as usize;
        for stage in 0..n_stages {
            let idx = stage * 2; // Sequential: Conv, ReLU, Conv, ReLU, …
            let in_c = if stage == 0 { 1 } else { c };
            let t2 = (t + 2 - 3) / 2 + 1;
            let f2 = (freq + 2 - 3) / 2 + 1;
            let wkey = keys::pre_encode_conv(idx, "weight");
            let bkey = keys::pre_encode_conv(idx, "bias");
            h = self.conv2d_pad(
                h,
                &wkey,
                Some(&bkey),
                in_c,
                c,
                [t, freq],
                3,
                2,
                [1, 1, 1, 1],
                1,
            )?;
            h = self.g().relu(h);
            t = t2;
            freq = f2;
        }
        let h = self.g().transpose_(h, vec![0, 2, 1, 3]);
        let flat = c * freq;
        let h = self.g().reshape_(h, vec![1, t as i64, flat as i64]);
        let h = self.linear(
            h,
            keys::PRE_ENCODE_OUT_W,
            Some(keys::PRE_ENCODE_OUT_B),
            cfg.d_model,
        )?;
        Ok((h, t))
    }

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

fn rel_pos_encoding(t: usize, d: usize) -> Vec<f32> {
    let pos_len = 2 * t - 1;
    let mut pe = vec![0.0f32; pos_len * d];
    for (row, p) in (0..pos_len).enumerate() {
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

/// Build the Conformer encoder HIR for a fixed mel length.
///
/// - Input: `mel` with shape `[1, n_mels, mel_frames]`
/// - Output: encoder features `[1, enc_frames, d_model]` where
///   `enc_frames = cfg.enc_frames(mel_frames)`
///
/// Returns `(hir, params, enc_frames)`. Requires `cfg.subsampling == "striding"`.
pub fn build_encoder_hir(
    cfg: &AsrConfig,
    weights: &mut dyn WeightSource,
    mel_frames: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, usize)> {
    ensure!(mel_frames > 0, "mel_frames must be > 0");
    if cfg.subsampling != "striding" {
        bail!(
            "encoder.subsampling {:?} not implemented in rlx-conformer-ctc (need striding)",
            cfg.subsampling
        );
    }
    let f = DType::F32;
    let mut hir = HirModule::new("conformer_ctc_encoder").with_fusion_policy(FusionPolicy::Direct);
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

    let (mut x, t) = b.subsample_striding(mel, mel_frames)?;
    ensure!(
        t > 0,
        "subsampling produced 0 frames for {mel_frames} mel frames"
    );

    // NeMo `RelPositionalEncoding` / `xscaling=true`: scale embeddings by √d_model
    // before the conformer stack (see ConformerEncoder + RelPositionalEncoding).
    let xscale = (cfg.d_model as f32).sqrt();
    let xs = b.scalar(xscale);
    x = b.g().mul(x, xs);

    let pe = rel_pos_encoding(t, cfg.d_model);
    let pos = b.synth("_enc.rel_pos_encoding", pe, &[1, 2 * t - 1, cfg.d_model]);

    for layer in 0..cfg.n_layers {
        x = b.block(layer, x, pos, t)?;
    }
    hir.outputs = vec![x];
    Ok((hir, params, t))
}
