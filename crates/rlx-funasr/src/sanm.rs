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

//! SAN-M / FSMN HIR building blocks shared by the SAN-M encoder
//! ([`crate::sensevoice`], [`crate::paraformer`], [`crate::punc`]) and the
//! Paraformer SAN-M decoder.
//!
//! Faithful to FunASR `funasr/models/sanm/{attention,encoder,decoder}.py`:
//!
//! * **SAN-M self-attention** — one fused `linear_q_k_v`, scale `q` by
//!   `d_k^-0.5`, scaled-dot-product attention through `linear_out`, **plus** a
//!   parallel FSMN memory branch (a depthwise `Conv1d` over the un-headed `v`
//!   with a residual add) — the two are summed.
//! * **FSMN memory** — depthwise conv with explicit asymmetric left/right
//!   padding (`left = (k-1)/2 + sanm_shfit`, `right = k-1-left`) and an
//!   `+ inputs` residual.
//! * **Encoder layer** — pre-norm; the residual is skipped when `in_size !=
//!   size` (how the first layer changes dimension).
//! * **Sinusoidal position encoding** — `cat([sin, cos])` (sin in the first
//!   half of the channels), positions `1..=T`, applied after scaling the input
//!   by `sqrt(output_size)`.

use std::collections::HashMap;

use anyhow::Result;
use rlx_flow::WeightSource;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Op, Shape};

/// A mutable HIR construction context bound to a weight source.
pub struct Graph<'a> {
    /// The HIR module being built.
    pub hir: &'a mut HirModule,
    /// Collected parameter data, keyed by name.
    pub params: &'a mut HashMap<String, Vec<f32>>,
    /// Source of tensor weights.
    pub weights: &'a mut dyn WeightSource,
    /// Working float dtype (f32).
    pub f: DType,
    uid: usize,
}

impl<'a> Graph<'a> {
    /// Bind a builder to a HIR module and a weight source.
    pub fn new(
        hir: &'a mut HirModule,
        params: &'a mut HashMap<String, Vec<f32>>,
        weights: &'a mut dyn WeightSource,
    ) -> Self {
        Self {
            hir,
            params,
            weights,
            f: DType::F32,
            uid: 0,
        }
    }

    /// A mutable graph view for emitting ops.
    pub fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    /// A fresh unique parameter key with the given tag.
    pub fn fresh(&mut self, tag: &str) -> String {
        self.uid += 1;
        format!("_fa.{tag}.{}", self.uid)
    }

    /// Raw (un-transposed) tensor fetch as a synthesized param of `shape`.
    pub fn synth_weight(&mut self, key: &str, shape: &[usize]) -> Result<HirNodeId> {
        let (data, _) = self.weights.take(key, false)?;
        let k = self.fresh("w");
        Ok(self.synth(&k, data, shape))
    }

    /// `[1, in, T]` full conv1d with weight `[out, in, k]`, symmetric `pad`.
    pub fn conv1d(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        t: usize,
    ) -> Result<HirNodeId> {
        let f = self.f;
        let nchw = self.g().reshape_(x, vec![1, in_c as i64, t as i64, 1]);
        let (wd, _) = self.weights.take(w_key, false)?; // [out,in,k]
        let wk = self.fresh("c1w");
        let wnode = self.synth(&wk, wd, &[out_c, in_c, k, 1]);
        let t_out = (t + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let mut y = self.g().add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![pad, 0],
                dilation: vec![dilation, 1],
                groups: 1,
            },
            vec![nchw, wnode],
            Shape::new(&[1, out_c, t_out, 1], f),
        );
        if let Some(bk) = b_key {
            let bias = self.load(bk, false)?;
            let b4 = self.g().reshape_(bias, vec![1, out_c as i64, 1, 1]);
            y = self.g().add(y, b4);
        }
        Ok(self.g().reshape_(y, vec![1, out_c as i64, t_out as i64]))
    }

    /// Fold a (frozen) BatchNorm into a per-channel affine over `[1, C, T]`:
    /// `y = x·(γ/√(var+ε)) + (β - mean·γ/√(var+ε))`. When `affine` is false the
    /// `weight`/`bias` keys are skipped (γ=1, β=0).
    pub fn batchnorm1d(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        c: usize,
        eps: f32,
        affine: bool,
    ) -> Result<HirNodeId> {
        let (mean, _) = self
            .weights
            .take(&format!("{prefix}.running_mean"), false)?;
        let (var, _) = self.weights.take(&format!("{prefix}.running_var"), false)?;
        let (gamma, beta) = if affine {
            let (g, _) = self.weights.take(&format!("{prefix}.weight"), false)?;
            let (b, _) = self.weights.take(&format!("{prefix}.bias"), false)?;
            (g, b)
        } else {
            (vec![1.0; c], vec![0.0; c])
        };
        let mut scale = vec![0.0f32; c];
        let mut shift = vec![0.0f32; c];
        for i in 0..c {
            let s = gamma[i] / (var[i] + eps).sqrt();
            scale[i] = s;
            shift[i] = beta[i] - mean[i] * s;
        }
        let sk = self.fresh("bn_s");
        let sc = self.synth(&sk, scale, &[c]);
        let sc = self.g().reshape_(sc, vec![1, c as i64, 1]);
        let shk = self.fresh("bn_b");
        let sh = self.synth(&shk, shift, &[c]);
        let sh = self.g().reshape_(sh, vec![1, c as i64, 1]);
        let y = self.g().mul(x, sc);
        Ok(self.g().add(y, sh))
    }

    /// Declare a graph input tensor.
    pub fn input(&mut self, name: &str, shape: &[usize]) -> HirNodeId {
        self.hir.input(name, Shape::new(shape, self.f))
    }

    /// Mark `id` as the graph's single output.
    pub fn set_output(&mut self, id: HirNodeId) {
        self.hir.outputs = vec![id];
    }

    // ── param helpers ─────────────────────────────────────────────────
    /// Load a weight as a param (optionally transposed for `mm`).
    pub fn load(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    /// Register a synthesized constant parameter.
    pub fn synth(&mut self, key: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let id = self.hir.param(key, Shape::new(shape, self.f));
        self.params.insert(key.to_string(), data);
        id
    }

    /// A broadcastable scalar constant parameter.
    pub fn scalar(&mut self, v: f32) -> HirNodeId {
        let key = self.fresh("scalar");
        self.synth(&key, vec![v], &[1])
    }

    /// `x @ Wᵀ (+ b)`; weight stored `[out, in]`, optional bias `[out]`.
    pub fn linear(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: Option<&str>,
        out: usize,
    ) -> Result<HirNodeId> {
        let wt = self.load(w, true)?;
        let mut y = self.g().mm(x, wt);
        if let Some(bk) = b {
            let bias = self.load(bk, false)?;
            let b3 = self.g().reshape_(bias, vec![1, 1, out as i64]);
            y = self.g().add(y, b3);
        }
        Ok(y)
    }

    /// LayerNorm over the last axis using weight/bias keys.
    pub fn layer_norm(&mut self, x: HirNodeId, w: &str, b: &str, eps: f32) -> Result<HirNodeId> {
        let g = self.load(w, false)?;
        let beta = self.load(b, false)?;
        Ok(self.g().ln(x, g, beta, eps))
    }

    /// Sigmoid, built as `1/(1+e^-x)` (HIR has no sigmoid op).
    pub fn sigmoid(&mut self, x: HirNodeId) -> HirNodeId {
        let neg = self.g().neg(x);
        let e = self.g().exp(neg);
        let one = self.scalar(1.0);
        let den = self.g().add(e, one);
        let onen = self.scalar(1.0);
        self.g().div(onen, den)
    }

    /// Reshape `[1, T, n_feat] -> [1, nh, T, hd]`.
    fn heads(&mut self, x: HirNodeId, t: usize, nh: usize, hd: usize) -> HirNodeId {
        let x = self
            .g()
            .reshape_(x, vec![1, t as i64, nh as i64, hd as i64]);
        self.g().transpose_(x, vec![0, 2, 1, 3])
    }

    /// Depthwise `Conv1d` over `[1, T, d]` with explicit `left`/`right` time
    /// padding and a depthwise weight stored `[d, 1, k]`; no residual, no bias.
    pub fn depthwise_conv1d(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        d: usize,
        k: usize,
        left: usize,
        right: usize,
        t: usize,
    ) -> Result<HirNodeId> {
        let f = self.f;
        let xt = self.g().transpose_(x, vec![0, 2, 1]); // [1,d,T]
        let nchw = self.g().reshape_(xt, vec![1, d as i64, t as i64, 1]); // [1,d,T,1]
        // explicit asymmetric pad in the time (H) dim
        let mut seq = Vec::new();
        if left > 0 {
            let zk = self.fresh("padL");
            seq.push(self.synth(&zk, vec![0.0; d * left], &[1, d, left, 1]));
        }
        seq.push(nchw);
        if right > 0 {
            let zk = self.fresh("padR");
            seq.push(self.synth(&zk, vec![0.0; d * right], &[1, d, right, 1]));
        }
        let padded = if seq.len() == 1 {
            seq[0]
        } else {
            self.g().concat_(seq, 2)
        };
        let padded_t = t + left + right;
        let (wd, _) = self.weights.take(w_key, false)?; // [d,1,k]
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
        let _ = padded_t;
        let back = self.g().reshape_(conv, vec![1, d as i64, t as i64]);
        Ok(self.g().transpose_(back, vec![0, 2, 1])) // [1,T,d]
    }

    /// FSMN memory: `depthwise_conv(x) + x`.
    fn fsmn_memory(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        d: usize,
        kernel: usize,
        sanm_shfit: usize,
        t: usize,
    ) -> Result<HirNodeId> {
        let mut left = (kernel - 1) / 2;
        if sanm_shfit > 0 {
            left += sanm_shfit;
        }
        let right = kernel - 1 - left;
        let conv = self.depthwise_conv1d(x, w_key, d, kernel, left, right, t)?;
        Ok(self.g().add(conv, x))
    }

    /// `MultiHeadedAttentionSANM.forward` → `[1, T, n_feat]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sanm_attention(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        n_feat: usize,
        n_heads: usize,
        kernel: usize,
        sanm_shfit: usize,
        t: usize,
    ) -> Result<HirNodeId> {
        let hd = n_feat / n_heads;
        let scale = (hd as f32).powf(-0.5);
        let qkv_w = format!("{prefix}.linear_q_k_v.weight");
        let qkv_b = format!("{prefix}.linear_q_k_v.bias");
        let qkv = self.linear(x, &qkv_w, Some(&qkv_b), 3 * n_feat)?; // [1,t,3*n_feat]
        let q = self.g().narrow_(qkv, 2, 0, n_feat);
        let k = self.g().narrow_(qkv, 2, n_feat, n_feat);
        let v = self.g().narrow_(qkv, 2, 2 * n_feat, n_feat);

        // FSMN over the un-headed v.
        let fsmn_w = format!("{prefix}.fsmn_block.weight");
        let fsmn = self.fsmn_memory(v, &fsmn_w, n_feat, kernel, sanm_shfit, t)?;

        let q = self.heads(q, t, n_heads, hd);
        let k = self.heads(k, t, n_heads, hd);
        let v = self.heads(v, t, n_heads, hd);
        let sc = self.scalar(scale);
        let q = self.g().mul(q, sc);
        let k_t = self.g().transpose_(k, vec![0, 1, 3, 2]);
        let scores = self.g().mm(q, k_t); // [1,nh,t,t]
        let attn = self.g().sm(scores, -1);
        let ctx = self.g().mm(attn, v); // [1,nh,t,hd]
        let ctx = self.g().transpose_(ctx, vec![0, 2, 1, 3]);
        let ctx = self.g().reshape_(ctx, vec![1, t as i64, n_feat as i64]);
        let ow = format!("{prefix}.linear_out.weight");
        let ob = format!("{prefix}.linear_out.bias");
        let att_out = self.linear(ctx, &ow, Some(&ob), n_feat)?;
        Ok(self.g().add(att_out, fsmn))
    }

    /// Cross-attention `MultiHeadedAttentionCrossAtt.forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_attention(
        &mut self,
        x: HirNodeId,
        memory: HirNodeId,
        prefix: &str,
        n_feat: usize,
        n_heads: usize,
        t_q: usize,
        t_kv: usize,
    ) -> Result<HirNodeId> {
        let hd = n_feat / n_heads;
        let scale = (hd as f32).powf(-0.5);
        let qw = format!("{prefix}.linear_q.weight");
        let qb = format!("{prefix}.linear_q.bias");
        let q = self.linear(x, &qw, Some(&qb), n_feat)?;
        let kvw = format!("{prefix}.linear_k_v.weight");
        let kvb = format!("{prefix}.linear_k_v.bias");
        let kv = self.linear(memory, &kvw, Some(&kvb), 2 * n_feat)?;
        let k = self.g().narrow_(kv, 2, 0, n_feat);
        let v = self.g().narrow_(kv, 2, n_feat, n_feat);
        let q = self.heads(q, t_q, n_heads, hd);
        let k = self.heads(k, t_kv, n_heads, hd);
        let v = self.heads(v, t_kv, n_heads, hd);
        let sc = self.scalar(scale);
        let q = self.g().mul(q, sc);
        let k_t = self.g().transpose_(k, vec![0, 1, 3, 2]);
        let scores = self.g().mm(q, k_t); // [1,nh,tq,tkv]
        let attn = self.g().sm(scores, -1);
        let ctx = self.g().mm(attn, v); // [1,nh,tq,hd]
        let ctx = self.g().transpose_(ctx, vec![0, 2, 1, 3]);
        let ctx = self.g().reshape_(ctx, vec![1, t_q as i64, n_feat as i64]);
        let ow = format!("{prefix}.linear_out.weight");
        let ob = format!("{prefix}.linear_out.bias");
        self.linear(ctx, &ow, Some(&ob), n_feat)
    }

    /// Encoder feed-forward (`PositionwiseFeedForward`: w_1 → ReLU → w_2, biases).
    pub fn ffn_encoder(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        d: usize,
        units: usize,
    ) -> Result<HirNodeId> {
        let w1 = format!("{prefix}.w_1.weight");
        let b1 = format!("{prefix}.w_1.bias");
        let w2 = format!("{prefix}.w_2.weight");
        let b2 = format!("{prefix}.w_2.bias");
        let h = self.linear(x, &w1, Some(&b1), units)?;
        let h = self.g().relu(h);
        self.linear(h, &w2, Some(&b2), d)
    }

    /// Decoder feed-forward (`PositionwiseFeedForwardDecoderSANM`: w_1 → ReLU →
    /// LayerNorm(hidden) → w_2 with **no bias**).
    pub fn ffn_decoder(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        d: usize,
        units: usize,
        eps: f32,
    ) -> Result<HirNodeId> {
        let w1 = format!("{prefix}.w_1.weight");
        let b1 = format!("{prefix}.w_1.bias");
        let nw = format!("{prefix}.norm.weight");
        let nb = format!("{prefix}.norm.bias");
        let w2 = format!("{prefix}.w_2.weight");
        let h = self.linear(x, &w1, Some(&b1), units)?;
        let h = self.g().relu(h);
        let h = self.layer_norm(h, &nw, &nb, eps)?;
        self.linear(h, &w2, None, d)
    }

    /// One `EncoderLayerSANM`. Residual on the attention branch is skipped when
    /// `in_size != size`.
    #[allow(clippy::too_many_arguments)]
    pub fn encoder_layer(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        in_size: usize,
        size: usize,
        n_heads: usize,
        units: usize,
        kernel: usize,
        sanm_shfit: usize,
        t: usize,
        eps: f32,
    ) -> Result<HirNodeId> {
        let n1w = format!("{prefix}.norm1.weight");
        let n1b = format!("{prefix}.norm1.bias");
        let n2w = format!("{prefix}.norm2.weight");
        let n2b = format!("{prefix}.norm2.bias");
        let attn_prefix = format!("{prefix}.self_attn");
        let ff_prefix = format!("{prefix}.feed_forward");

        let residual = x;
        let h = self.layer_norm(x, &n1w, &n1b, eps)?;
        let h = self.sanm_attention(h, &attn_prefix, size, n_heads, kernel, sanm_shfit, t)?;
        let x = if in_size == size {
            self.g().add(residual, h)
        } else {
            h
        };
        let residual = x;
        let h = self.layer_norm(x, &n2w, &n2b, eps)?;
        let h = self.ffn_encoder(h, &ff_prefix, size, units)?;
        Ok(self.g().add(residual, h))
    }

    /// `DecoderLayerSANM`: feed-forward first, then optional FSMN self-attn
    /// (residual from the original `tgt`), then optional cross-attention.
    #[allow(clippy::too_many_arguments)]
    pub fn decoder_layer(
        &mut self,
        tgt: HirNodeId,
        memory: Option<HirNodeId>,
        prefix: &str,
        d: usize,
        n_heads: usize,
        units: usize,
        self_kernel: usize,
        self_shfit: usize,
        has_self: bool,
        t_q: usize,
        t_kv: usize,
        eps: f32,
    ) -> Result<HirNodeId> {
        let n1w = format!("{prefix}.norm1.weight");
        let n1b = format!("{prefix}.norm1.bias");
        let ff_prefix = format!("{prefix}.feed_forward");

        let residual = tgt;
        let h = self.layer_norm(tgt, &n1w, &n1b, eps)?;
        let h = self.ffn_decoder(h, &ff_prefix, d, units, eps)?;
        let mut x = h;
        if has_self {
            let n2w = format!("{prefix}.norm2.weight");
            let n2b = format!("{prefix}.norm2.bias");
            let h2 = self.layer_norm(x, &n2w, &n2b, eps)?;
            let fsmn_w = format!("{prefix}.self_attn.fsmn_block.weight");
            let s = self.fsmn_memory(h2, &fsmn_w, d, self_kernel, self_shfit, t_q)?;
            x = self.g().add(residual, s);
        }
        if let Some(mem) = memory {
            let n3w = format!("{prefix}.norm3.weight");
            let n3b = format!("{prefix}.norm3.bias");
            let residual2 = x;
            let h3 = self.layer_norm(x, &n3w, &n3b, eps)?;
            let src_prefix = format!("{prefix}.src_attn");
            let c = self.cross_attention(h3, mem, &src_prefix, d, n_heads, t_q, t_kv)?;
            x = self.g().add(residual2, c);
        }
        Ok(x)
    }

    /// Fold a frozen BatchNorm2d into a per-channel affine over `[1, C, H, W]`.
    pub fn batchnorm2d(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        c: usize,
        eps: f32,
    ) -> Result<HirNodeId> {
        let (mean, _) = self
            .weights
            .take(&format!("{prefix}.running_mean"), false)?;
        let (var, _) = self.weights.take(&format!("{prefix}.running_var"), false)?;
        let (gamma, _) = self.weights.take(&format!("{prefix}.weight"), false)?;
        let (beta, _) = self.weights.take(&format!("{prefix}.bias"), false)?;
        let mut scale = vec![0.0f32; c];
        let mut shift = vec![0.0f32; c];
        for i in 0..c {
            let s = gamma[i] / (var[i] + eps).sqrt();
            scale[i] = s;
            shift[i] = beta[i] - mean[i] * s;
        }
        let sk = self.fresh("bn2_s");
        let sc = self.synth(&sk, scale, &[c]);
        let sc = self.g().reshape_(sc, vec![1, c as i64, 1, 1]);
        let shk = self.fresh("bn2_b");
        let sh = self.synth(&shk, shift, &[c]);
        let sh = self.g().reshape_(sh, vec![1, c as i64, 1, 1]);
        let y = self.g().mul(x, sc);
        Ok(self.g().add(y, sh))
    }

    /// FSMN-VAD memory block (`FSMNBlock`): a causal depthwise `Conv2d`
    /// `[lorder, 1]` with `dilation = [lstride, 1]` over `[1, T, C]` (left-only
    /// padding `(lorder-1)·lstride`), plus an `+ inputs` residual. `rorder` is 0
    /// for the standard VAD config so no future-context branch is built.
    pub fn vad_fsmn(
        &mut self,
        x: HirNodeId,
        c: usize,
        lorder: usize,
        lstride: usize,
        t: usize,
        w_key: &str,
    ) -> Result<HirNodeId> {
        let f = self.f;
        let pad = (lorder - 1) * lstride;
        let xt = self.g().transpose_(x, vec![0, 2, 1]); // [1,c,t]
        let nchw = self.g().reshape_(xt, vec![1, c as i64, t as i64, 1]);
        let padded = if pad > 0 {
            let zk = self.fresh("vadpad");
            let z = self.synth(&zk, vec![0.0; c * pad], &[1, c, pad, 1]);
            self.g().concat_(vec![z, nchw], 2)
        } else {
            nchw
        };
        let (wd, _) = self.weights.take(w_key, false)?; // [c,1,lorder,1]
        let wk = self.fresh("vadw");
        let wnode = self.synth(&wk, wd, &[c, 1, lorder, 1]);
        let conv = self.g().add_node(
            Op::Conv {
                kernel_size: vec![lorder, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![lstride, 1],
                groups: c,
            },
            vec![padded, wnode],
            Shape::new(&[1, c, t, 1], f),
        );
        let back = self.g().reshape_(conv, vec![1, c as i64, t as i64]);
        let bt = self.g().transpose_(back, vec![0, 2, 1]);
        Ok(self.g().add(bt, x))
    }

    /// Scale input by `sqrt(output_size)` and add the sinusoidal position
    /// encoding (depth = the input feature dim). Returns the new node.
    pub fn add_pos_and_scale(
        &mut self,
        x: HirNodeId,
        t: usize,
        feat_dim: usize,
        output_size: usize,
    ) -> HirNodeId {
        let scale = (output_size as f32).sqrt();
        let sc = self.scalar(scale);
        let xs = self.g().mul(x, sc);
        let pe = sinusoidal_pos(t, feat_dim);
        let key = self.fresh("sinpos");
        let pos = self.synth(&key, pe, &[1, t, feat_dim]);
        self.g().add(xs, pos)
    }
}

/// Build a full SAN-M encoder stack and return the output node `[1, T, d]`.
///
/// `encoders0.0` maps `input_size → output_size`; `encoders.{i}` keep
/// `output_size`; `after_norm` follows. With `use_tp`, `tp_encoders.{i}` and a
/// final `tp_norm` are appended (SenseVoice's temporal-processing blocks).
pub fn build_sanm_encoder(
    g: &mut Graph,
    x: HirNodeId,
    cfg: &crate::config::SanmEncoderConfig,
    prefix: &str,
    t: usize,
    use_tp: bool,
) -> Result<HirNodeId> {
    let d = cfg.output_size;
    let nh = cfg.n_heads;
    let units = cfg.linear_units;
    let k = cfg.kernel_size;
    let shfit = cfg.sanm_shfit;
    let eps = cfg.ln_eps;

    let x = g.add_pos_and_scale(x, t, cfg.input_size, d);
    let p0 = format!("{prefix}.encoders0.0");
    let mut h = g.encoder_layer(x, &p0, cfg.input_size, d, nh, units, k, shfit, t, eps)?;
    for i in 0..cfg.num_blocks.saturating_sub(1) {
        let p = format!("{prefix}.encoders.{i}");
        h = g.encoder_layer(h, &p, d, d, nh, units, k, shfit, t, eps)?;
    }
    h = g.layer_norm(
        h,
        &format!("{prefix}.after_norm.weight"),
        &format!("{prefix}.after_norm.bias"),
        eps,
    )?;
    if use_tp && cfg.tp_blocks > 0 {
        for i in 0..cfg.tp_blocks {
            let p = format!("{prefix}.tp_encoders.{i}");
            h = g.encoder_layer(h, &p, d, d, nh, units, k, shfit, t, eps)?;
        }
        h = g.layer_norm(
            h,
            &format!("{prefix}.tp_norm.weight"),
            &format!("{prefix}.tp_norm.bias"),
            eps,
        )?;
    }
    Ok(h)
}

/// FunASR `SinusoidalPositionEncoder`: positions `1..=T`,
/// `inv_timescales[i] = exp(-i·ln(10000)/(depth/2 - 1))`, output
/// `[sin(p·inv) ‖ cos(p·inv)]` (sin in the first half of the channels).
pub fn sinusoidal_pos(t: usize, depth: usize) -> Vec<f32> {
    let half = depth / 2;
    let mut pe = vec![0.0f32; t * depth];
    if half == 0 {
        return pe;
    }
    let incr = (10000f64).ln() / (half as f64 - 1.0).max(1.0);
    let inv: Vec<f64> = (0..half).map(|i| (-(i as f64) * incr).exp()).collect();
    for p in 0..t {
        let pos = (p + 1) as f64; // 1-indexed
        for i in 0..half {
            let scaled = pos * inv[i];
            pe[p * depth + i] = scaled.sin() as f32;
            pe[p * depth + half + i] = scaled.cos() as f32;
        }
    }
    pe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_enc_layout_sin_then_cos() {
        let pe = sinusoidal_pos(3, 8);
        // first row, channel 0 = sin(1*inv0), channel half=4 = cos(1*inv0)
        let incr = (10000f64).ln() / 3.0; // half=4 -> /(4-1)
        let inv0 = (-0.0 * incr).exp(); // =1
        assert!((pe[0] - (1.0f64 * inv0).sin() as f32).abs() < 1e-6);
        assert!((pe[4] - (1.0f64 * inv0).cos() as f32).abs() < 1e-6);
        assert!(pe.iter().all(|x| x.is_finite()));
    }
}
