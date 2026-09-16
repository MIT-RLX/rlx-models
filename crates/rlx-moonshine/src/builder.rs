// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Shared HIR helpers for Moonshine encoder / decoder graphs.

use crate::config::{GROUP_NORM_EPS, LN_EPS, MoonshineConfig};
use crate::weights::MoonshineWeightPrefix;
use anyhow::Result;
use rlx_flow::WeightSource;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, Op, PadMode, RopeStyle};
use rlx_ir::ops::attention::attention_kind_op;
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

pub(crate) struct MoonshineBuilder<'a> {
    pub hir: &'a mut HirModule,
    pub params: &'a mut HashMap<String, Vec<f32>>,
    pub weights: &'a mut dyn WeightSource,
    pub pfx: &'a MoonshineWeightPrefix,
    pub batch: usize,
    pub f: DType,
}

impl<'a> MoonshineBuilder<'a> {
    pub(crate) fn new(
        hir: &'a mut HirModule,
        params: &'a mut HashMap<String, Vec<f32>>,
        weights: &'a mut dyn WeightSource,
        pfx: &'a MoonshineWeightPrefix,
        batch: usize,
    ) -> Self {
        Self {
            hir,
            params,
            weights,
            pfx,
            batch,
            f: DType::F32,
        }
    }

    pub(crate) fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    pub(crate) fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    pub(crate) fn register_param(
        &mut self,
        key: &str,
        data: Vec<f32>,
        dims: &[usize],
    ) -> Result<HirNodeId> {
        let id = self.hir.param(key, Shape::new(dims, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    /// HF `nn.Linear`: weight `[out, in]` → transpose for `x @ W`.
    pub(crate) fn linear(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
    ) -> Result<HirNodeId> {
        let w = self.load_param(w_key, true)?;
        let mut y = self.g().mm(x, w);
        if let Some(bk) = b_key
            && self.weights.has(bk)
        {
            let b = self.load_param(bk, false)?;
            y = self.g().add(y, b);
        }
        Ok(y)
    }

    /// LayerNorm with weight only (`bias=False`); register zero β.
    pub(crate) fn layer_norm_nobias(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        d: usize,
    ) -> Result<HirNodeId> {
        let gamma = self.load_param(w_key, false)?;
        let beta_key = format!("{w_key}.__zero_beta");
        let beta = self.register_param(&beta_key, vec![0.0; d], &[d])?;
        Ok(self.g().ln(x, gamma, beta, LN_EPS))
    }

    /// Conv1d as NCHW Conv2d with kernel `[k, 1]`, optional bias, optional gelu/tanh.
    pub(crate) fn conv1d(
        &mut self,
        input: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
        in_c: usize,
        out_c: usize,
        t_in: usize,
        k: usize,
        stride: usize,
        pad: usize,
        act: ConvAct,
    ) -> Result<(HirNodeId, usize)> {
        let batch = self.batch;
        let f = self.f;
        let t_out = (t_in + 2 * pad - k) / stride + 1;
        let nchw = self
            .g()
            .reshape_(input, vec![batch as i64, in_c as i64, t_in as i64, 1]);
        let (w_data, _) = self.weights.take(w_key, false)?;
        let w = self.register_param(w_key, w_data, &[out_c, in_c, k, 1])?;
        let conv = self.g().add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![pad, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![nchw, w],
            Shape::new(&[batch, out_c, t_out, 1], f),
        );
        let mut out = self
            .g()
            .reshape_(conv, vec![batch as i64, out_c as i64, t_out as i64]);
        if let Some(bk) = b_key {
            let (b_data, _) = self.weights.take(bk, false)?;
            let bias = self.register_param(bk, b_data, &[out_c])?;
            let b3 = self.broadcast_conv_bias(bias, out_c, t_out)?;
            out = self.g().add(out, b3);
        }
        let out = match act {
            ConvAct::None => out,
            ConvAct::Tanh => self.g().tanh(out),
            ConvAct::Gelu => self.g().gelu(out),
        };
        Ok((out, t_out))
    }

    fn broadcast_conv_bias(
        &mut self,
        bias: HirNodeId,
        out_c: usize,
        t_out: usize,
    ) -> Result<HirNodeId> {
        let batch = self.batch as i64;
        let b2 = self.g().reshape_(bias, vec![1, out_c as i64, 1]);
        let ones = self.register_param(
            &format!("conv_bias_ones_{t_out}"),
            vec![1.0; self.batch * t_out],
            &[self.batch, t_out],
        )?;
        let ones3 = self.g().reshape_(ones, vec![batch, 1, t_out as i64]);
        Ok(self.g().mul(b2, ones3))
    }

    /// GroupNorm(groups=1) on `[B,C,T]` (reshape to NCHW).
    pub(crate) fn group_norm_ct(
        &mut self,
        x: HirNodeId,
        c: usize,
        t: usize,
        w_key: &str,
        b_key: &str,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let nchw = self
            .g()
            .reshape_(x, vec![batch as i64, c as i64, t as i64, 1]);
        let gamma = self.load_param(w_key, false)?;
        let beta = self.load_param(b_key, false)?;
        let gn = self.g().group_norm(nchw, gamma, beta, 1, GROUP_NORM_EPS);
        Ok(self
            .g()
            .reshape_(gn, vec![batch as i64, c as i64, t as i64]))
    }

    /// Host-side cos/sin tables `[seq, head_dim/2]` for GPT-J (interleaved) RoPE.
    ///
    /// Table width is always `head_dim/2`; only the leading `n_rot/2` columns
    /// carry frequencies (partial rotary). CPU fused-attn and MLX expect this
    /// layout; Metal RoPE must use `rope_table_stride` (not hardcoded
    /// `n_rot/2`) so partial rotary stays correct.
    pub(crate) fn rope_tables(
        &mut self,
        cfg: &MoonshineConfig,
        seq: usize,
        head_dim: usize,
        tag: &str,
    ) -> Result<(HirNodeId, HirNodeId, usize)> {
        let n_rot = cfg.n_rot(head_dim);
        let tab_half = head_dim / 2;
        let rot_half = n_rot / 2;
        let mut cos = vec![0f32; seq * tab_half];
        let mut sin = vec![0f32; seq * tab_half];
        for pos in 0..seq {
            let base = pos * tab_half;
            for i in 0..rot_half {
                let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / n_rot as f64);
                let angle = pos as f64 * freq;
                let (s, c) = angle.sin_cos();
                cos[base + i] = c as f32;
                sin[base + i] = s as f32;
            }
        }
        let cos_id =
            self.register_param(&format!("rope_cos_{tag}_{seq}"), cos, &[seq, tab_half])?;
        let sin_id =
            self.register_param(&format!("rope_sin_{tag}_{seq}"), sin, &[seq, tab_half])?;
        Ok((cos_id, sin_id, n_rot))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn self_attn_rope(
        &mut self,
        x: HirNodeId,
        pfx: &str,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        n_rot: usize,
        cos: HirNodeId,
        sin: HirNodeId,
        mask: MaskKind,
        attention_bias: bool,
        pad_multiple: Option<usize>,
    ) -> Result<HirNodeId> {
        let q = self.proj(x, &format!("{pfx}.q_proj"), attention_bias)?;
        let k = self.proj(x, &format!("{pfx}.k_proj"), attention_bias)?;
        let v = self.proj(x, &format!("{pfx}.v_proj"), attention_bias)?;
        let q = self
            .g()
            .rope_n_styled(q, cos, sin, head_dim, n_rot, RopeStyle::GptJ);
        let k = self
            .g()
            .rope_n_styled(k, cos, sin, head_dim, n_rot, RopeStyle::GptJ);
        let (q, k, v, attn_hd) =
            self.pad_qkv_heads(q, k, v, seq, seq, n_head, head_dim, pad_multiple)?;
        // HF keeps softmax scale on the *unpadded* head_dim.
        let scale = (head_dim as f32).powf(-0.5);
        let out_shape = Shape::new(&[self.batch, seq, n_head * attn_hd], self.f);
        let mut attn = self.g().add_node(
            attention_kind_op(n_head, attn_hd, None, mask, Some(scale), None),
            vec![q, k, v],
            out_shape,
        );
        if attn_hd != head_dim {
            attn = self.unpad_attn_out(attn, seq, n_head, head_dim, attn_hd)?;
        }
        let o_key = MoonshineWeightPrefix::attn_o_proj(&|k| self.weights.has(k), pfx);
        self.linear(attn, &o_key, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cross_attn(
        &mut self,
        q_src: HirNodeId,
        enc: HirNodeId,
        pfx: &str,
        q_seq: usize,
        enc_seq: usize,
        n_head: usize,
        head_dim: usize,
        attention_bias: bool,
        pad_multiple: Option<usize>,
    ) -> Result<HirNodeId> {
        let q = self.proj(q_src, &format!("{pfx}.q_proj"), attention_bias)?;
        let k = self.proj(enc, &format!("{pfx}.k_proj"), attention_bias)?;
        let v = self.proj(enc, &format!("{pfx}.v_proj"), attention_bias)?;
        let (q, k, v, attn_hd) =
            self.pad_qkv_heads(q, k, v, q_seq, enc_seq, n_head, head_dim, pad_multiple)?;
        let scale = (head_dim as f32).powf(-0.5);
        let out_shape = Shape::new(&[self.batch, q_seq, n_head * attn_hd], self.f);
        let mut attn = self.g().add_node(
            attention_kind_op(n_head, attn_hd, None, MaskKind::None, Some(scale), None),
            vec![q, k, v],
            out_shape,
        );
        if attn_hd != head_dim {
            attn = self.unpad_attn_out(attn, q_seq, n_head, head_dim, attn_hd)?;
        }
        let o_key = MoonshineWeightPrefix::attn_o_proj(&|k| self.weights.has(k), pfx);
        self.linear(attn, &o_key, None)
    }

    /// HF: pad each head's last dim so SDPA kernels see an aligned width.
    #[allow(clippy::too_many_arguments)]
    fn pad_qkv_heads(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        q_seq: usize,
        kv_seq: usize,
        n_head: usize,
        head_dim: usize,
        pad_multiple: Option<usize>,
    ) -> Result<(HirNodeId, HirNodeId, HirNodeId, usize)> {
        let padded = match pad_multiple.filter(|&m| m > 0) {
            Some(m) => m * head_dim.div_ceil(m),
            None => head_dim,
        };
        let pad = padded - head_dim;
        if pad == 0 {
            return Ok((q, k, v, head_dim));
        }
        let batch = self.batch as i64;
        let nh = n_head as i64;
        let hd = head_dim as i64;
        let pad_one = |this: &mut Self, x: HirNodeId, seq: usize| -> HirNodeId {
            let x4 = this.g().reshape_(x, vec![batch, seq as i64, nh, hd]);
            let pads = vec![[0, 0], [0, 0], [0, 0], [0, pad]];
            let x4 = this.g().pad_(x4, pads, PadMode::Constant(0.0));
            this.g()
                .reshape_(x4, vec![batch, seq as i64, nh * padded as i64])
        };
        Ok((
            pad_one(self, q, q_seq),
            pad_one(self, k, kv_seq),
            pad_one(self, v, kv_seq),
            padded,
        ))
    }

    fn unpad_attn_out(
        &mut self,
        attn: HirNodeId,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        padded_hd: usize,
    ) -> Result<HirNodeId> {
        let batch = self.batch as i64;
        let x4 = self.g().reshape_(
            attn,
            vec![batch, seq as i64, n_head as i64, padded_hd as i64],
        );
        let x4 = self.g().narrow_(x4, 3, 0, head_dim);
        Ok(self
            .g()
            .reshape_(x4, vec![batch, seq as i64, (n_head * head_dim) as i64]))
    }

    fn proj(&mut self, x: HirNodeId, base: &str, with_bias: bool) -> Result<HirNodeId> {
        let w = format!("{base}.weight");
        let b = format!("{base}.bias");
        self.linear(x, &w, if with_bias { Some(b.as_str()) } else { None })
    }

    /// Encoder MLP: fc1 → gelu → fc2.
    pub(crate) fn encoder_mlp(&mut self, x: HirNodeId, layer: usize) -> Result<HirNodeId> {
        let p = |s: &str| self.pfx.enc_layer(layer, s);
        let h = self.linear(x, &p("mlp.fc1.weight"), Some(&p("mlp.fc1.bias")))?;
        let h = self.g().gelu(h);
        self.linear(h, &p("mlp.fc2.weight"), Some(&p("mlp.fc2.bias")))
    }

    /// Decoder MLP: fc1 → chunk → silu(gate)*up → fc2 (HF MoonshineDecoderMLP).
    pub(crate) fn decoder_mlp(
        &mut self,
        x: HirNodeId,
        layer: usize,
        intermediate: usize,
    ) -> Result<HirNodeId> {
        let p = |s: &str| self.pfx.dec_layer(layer, s);
        let h = self.linear(x, &p("mlp.fc1.weight"), Some(&p("mlp.fc1.bias")))?;
        // HF: hidden, gate = chunk(2); out = act(gate) * hidden
        let up = self.g().narrow_(h, 2, 0, intermediate);
        let gate = self.g().narrow_(h, 2, intermediate, intermediate);
        let gate = self.g().silu(gate);
        let h = self.g().mul(up, gate);
        self.linear(h, &p("mlp.fc2.weight"), Some(&p("mlp.fc2.bias")))
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ConvAct {
    #[allow(dead_code)]
    None,
    Tanh,
    Gelu,
}
