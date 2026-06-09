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

//! CPU-eager Qwen3-shaped code-predictor (parity reference path).

use crate::config::CodePredictorConfig;
use crate::load::Qwen3TtsWeightStore;
use crate::talker::math::{
    gqa_attention1_cp, linear_logits_flat_unchecked, linear_logits_into, matvec_accumulate_blas,
    matvec_blas, sample_greedy, sample_greedy_vocab,
};
use crate::talker::rope::{build_inv_freq, rope_slice_into};
use anyhow::{Context, Result, ensure};
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Max CP AR depth (prefill 2 + 14 decode steps); bucket attention scratch matches talker.
const CP_MAX_SEQ: usize = 32;
/// Qwen3-TTS 0.6B code predictor depth (unrolled decode micro-kernel).
const CP_DECODE_LAYERS: usize = 5;
/// CustomVoice 0.6B: 16 codec groups → 15 CP lm_head steps after group-0.
const CP_AR_LM_STEPS: usize = 15;
/// Max `head_dim / 2` for CustomVoice CP (128 / 2).
const CP_ROPE_HALF_CAP: usize = 64;

/// Expand 15 CP AR steps at compile time (no loop counter on the hot path).
macro_rules! cp_ar_unroll_15 {
    ($slf:ident, $ge:expr, $lh:expr, $lv:expr, $h:expr, $ce:expr) => {{
        $slf.cp_ar_step(0, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(1, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(2, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(3, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(4, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(5, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(6, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(7, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(8, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(9, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(10, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(11, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(12, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(13, $ge, $lh, $lv, $h, $ce, true);
        $slf.cp_ar_step(14, $ge, $lh, $lv, $h, $ce, false);
    }};
}

pub struct CpEagerModel {
    layers: Vec<CpLayer>,
    norm_weight: Vec<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    head_half: usize,
    hidden: usize,
    kv_dim: usize,
    /// Full tables for multi-token [`Self::forward`].
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    /// Flat `[pos * head_half + i]` cos/sin for [`Self::forward_one`] (no pos indexing).
    rope_cos_bank: Vec<f32>,
    rope_sin_bank: Vec<f32>,
    kv: Vec<LayerKv>,
    past_len: usize,
    norm_eps: f32,
    last_hidden: Vec<f32>,
    logits: Vec<f32>,
    /// Reused codec-group output (`group0` + 15 AR tokens).
    codes_buf: Vec<u32>,
    work_hidden: Vec<f32>,
    work_q: Vec<f32>,
    work_k: Vec<f32>,
    work_v: Vec<f32>,
    work_attn: Vec<f32>,
    work_gate: Vec<f32>,
    work_up: Vec<f32>,
    work_qkv: Vec<f32>,
    work_gate_up: Vec<f32>,
    work_scratch: Vec<f32>,
    attn_weights: Vec<f32>,
    /// Token-1 hidden state during fused 2-token prefill.
    work_x1: Vec<f32>,
    work_norm_b: Vec<f32>,
    work_qkv_b: Vec<f32>,
    work_gu_b: Vec<f32>,
    work_b_stack: Vec<f32>,
    work_gemm2: Vec<f32>,
    /// Gathered K/V head rows for BLAS attention (`2 * CP_MAX_SEQ * head_dim`).
    work_kv_head: Vec<f32>,
}

struct CpLayer {
    wq: Array2<f32>,
    wk: Array2<f32>,
    wv: Array2<f32>,
    /// Stacked `[Q; K; V]` for one matvec in [`CpLayer::forward_one`].
    wqkv: Array2<f32>,
    wo: Array2<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: Array2<f32>,
    up: Array2<f32>,
    /// Stacked `[gate; up]` for one matvec in [`CpLayer::forward_one`].
    gate_up: Array2<f32>,
    down: Array2<f32>,
    /// Optional BF16 copies of the four hot weight matrices, populated when
    /// `RLX_QWEN3_TTS_CP_BF16=1`. Each is row-major same shape as the F32 storage.
    wqkv_bf16: Vec<u16>,
    wo_bf16: Vec<u16>,
    gate_up_bf16: Vec<u16>,
    down_bf16: Vec<u16>,
    q_dim: usize,
    kv_dim: usize,
    inter_dim: usize,
}

#[derive(Default)]
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    n_rows: usize,
}

impl CpEagerModel {
    pub fn open(store: &Qwen3TtsWeightStore, cp: &CodePredictorConfig) -> Result<Self> {
        let keys: Vec<String> = store
            .keys()
            .iter()
            .filter(|k| k.starts_with("talker.code_predictor.model."))
            .cloned()
            .collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let map = store.tensor_snapshot(&key_refs)?;

        let mut layers = Vec::with_capacity(cp.num_hidden_layers);
        for i in 0..cp.num_hidden_layers {
            layers.push(load_layer(&map, i)?);
        }
        let norm_weight = take1d(&map, "talker.code_predictor.model.norm.weight")?;

        let head_half = cp.head_dim / 2;
        let inv_freq = build_inv_freq(cp.head_dim, cp.rope_theta);
        let max_pos = cp.max_position_embeddings.min(4096);
        let (rope_cos, rope_sin) = build_rope_tables(&inv_freq, max_pos, head_half);
        let mut rope_cos_bank = vec![0f32; CP_MAX_SEQ * head_half];
        let mut rope_sin_bank = vec![0f32; CP_MAX_SEQ * head_half];
        for pos in 0..CP_MAX_SEQ {
            let off = pos * head_half;
            rope_slice_into(
                &inv_freq,
                pos,
                cp.head_dim,
                &mut rope_cos_bank[off..off + head_half],
                &mut rope_sin_bank[off..off + head_half],
            );
        }
        let hidden = cp.hidden_size;
        let q_dim = cp.num_attention_heads * cp.head_dim;
        let kv_dim = cp.num_key_value_heads * cp.head_dim;
        let inter_dim = cp.intermediate_size;
        let kv_cap = CP_MAX_SEQ.saturating_mul(kv_dim);
        let mut kv_layers: Vec<LayerKv> = (0..cp.num_hidden_layers)
            .map(|_| LayerKv::default())
            .collect();
        for layer in &mut kv_layers {
            layer.k.resize(kv_cap, 0.0);
            layer.v.resize(kv_cap, 0.0);
        }
        Ok(Self {
            layers,
            norm_weight,
            n_heads: cp.num_attention_heads,
            n_kv_heads: cp.num_key_value_heads,
            head_dim: cp.head_dim,
            head_half,
            hidden,
            kv_dim,
            rope_cos,
            rope_sin,
            rope_cos_bank,
            rope_sin_bank,
            kv: kv_layers,
            past_len: 0,
            norm_eps: cp.rms_norm_eps as f32,
            last_hidden: vec![0f32; hidden],
            logits: vec![0f32; cp.vocab_size],
            codes_buf: Vec::with_capacity(CP_AR_LM_STEPS + 1),
            work_hidden: vec![0f32; hidden],
            work_q: vec![0f32; q_dim],
            work_k: vec![0f32; kv_dim],
            work_v: vec![0f32; kv_dim],
            work_attn: vec![0f32; q_dim],
            work_gate: vec![0f32; inter_dim],
            work_up: vec![0f32; inter_dim],
            work_qkv: vec![0f32; q_dim + 2 * kv_dim],
            work_gate_up: vec![0f32; 2 * inter_dim],
            work_scratch: vec![0f32; hidden],
            attn_weights: vec![0f32; cp.num_attention_heads * CP_MAX_SEQ],
            work_x1: vec![0f32; hidden],
            work_norm_b: vec![0f32; hidden],
            work_qkv_b: vec![0f32; q_dim + 2 * kv_dim],
            work_gu_b: vec![0f32; 2 * inter_dim],
            work_b_stack: vec![0f32; 2 * hidden.max(inter_dim)],
            work_gemm2: vec![0f32; 2 * (2 * inter_dim).max(q_dim + 2 * kv_dim).max(hidden)],
            work_kv_head: vec![0f32; 2 * CP_MAX_SEQ * cp.head_dim],
        })
    }

    pub fn last_hidden(&self) -> &[f32] {
        &self.last_hidden
    }

    #[inline]
    pub fn reset_kv(&mut self) {
        self.past_len = 0;
        for kv in &mut self.kv {
            kv.n_rows = 0;
        }
    }

    /// Run causal forward on `embeds` `[seq, hidden]`; returns all hidden rows.
    pub fn forward(&mut self, embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        let start_pos = self.past_len;
        let mut x = embeds.to_owned();
        let layerdbg = std::env::var("RLX_QWEN3_TTS_CP_LAYERDBG").ok().as_deref() == Some("1");
        for (li, layer) in self.layers.iter().enumerate() {
            x = layer.forward(
                x.view(),
                &mut self.kv[li],
                self.kv_dim,
                &self.rope_cos,
                &self.rope_sin,
                start_pos,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.norm_eps,
            )?;
            if layerdbg {
                eprintln!(
                    "layer{li} hs[0,:8] = {:?}",
                    x.row(0).iter().take(8).collect::<Vec<_>>()
                );
                eprintln!(
                    "layer{li} hs[-1,:8] = {:?}",
                    x.row(x.nrows() - 1).iter().take(8).collect::<Vec<_>>()
                );
            }
        }
        x = rms_norm2(x.view(), &self.norm_weight, self.norm_eps);
        self.past_len = start_pos + seq;
        Ok(x)
    }

    pub(crate) fn forward_one(&mut self, embed: ArrayView1<f32>) -> Result<()> {
        ensure!(embed.len() == self.hidden);
        self.work_hidden.copy_from_slice(embed.as_slice().unwrap());
        self.forward_one_from_work_hidden()
    }

    /// Like [`Self::forward_one`] when [`Self::work_hidden`] already holds the token embed.
    #[inline(always)]
    fn forward_one_from_work_hidden(&mut self) -> Result<()> {
        self.forward_decode_token();
        Ok(())
    }

    /// Single-token decode: 5-layer unrolled micro-kernel (hot CP AR path).
    #[inline(always)]
    fn forward_decode_token(&mut self) {
        let start_pos = self.past_len;
        debug_assert!(start_pos < CP_MAX_SEQ);
        let rh = self.head_half;
        let roff = start_pos * rh;
        let cos = &self.rope_cos_bank[roff..roff + rh];
        let sin = &self.rope_sin_bank[roff..roff + rh];
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let kv_dim = self.kv_dim;
        let eps = self.norm_eps;
        let hidden = self.hidden;

        if self.layers.len() == CP_DECODE_LAYERS {
            let layers = &self.layers;
            let kv = &mut self.kv;
            let x = &mut self.work_hidden;
            let attn = &mut self.work_attn;
            let scratch = &mut self.work_scratch;
            let qkv = &mut self.work_qkv;
            let gu = &mut self.work_gate_up;
            let attn_w = &mut self.attn_weights;
            let kv_scratch = &mut self.work_kv_head;
            layers[0].forward_one(
                x, attn, scratch, qkv, gu, attn_w, kv_scratch, &mut kv[0], kv_dim, cos, sin,
                n_heads, n_kv_heads, head_dim, eps, CP_MAX_SEQ,
            );
            layers[1].forward_one(
                x, attn, scratch, qkv, gu, attn_w, kv_scratch, &mut kv[1], kv_dim, cos, sin,
                n_heads, n_kv_heads, head_dim, eps, CP_MAX_SEQ,
            );
            layers[2].forward_one(
                x, attn, scratch, qkv, gu, attn_w, kv_scratch, &mut kv[2], kv_dim, cos, sin,
                n_heads, n_kv_heads, head_dim, eps, CP_MAX_SEQ,
            );
            layers[3].forward_one(
                x, attn, scratch, qkv, gu, attn_w, kv_scratch, &mut kv[3], kv_dim, cos, sin,
                n_heads, n_kv_heads, head_dim, eps, CP_MAX_SEQ,
            );
            layers[4].forward_one(
                x, attn, scratch, qkv, gu, attn_w, kv_scratch, &mut kv[4], kv_dim, cos, sin,
                n_heads, n_kv_heads, head_dim, eps, CP_MAX_SEQ,
            );
        } else {
            for (li, layer) in self.layers.iter().enumerate() {
                layer.forward_one(
                    &mut self.work_hidden,
                    &mut self.work_attn,
                    &mut self.work_scratch,
                    &mut self.work_qkv,
                    &mut self.work_gate_up,
                    &mut self.attn_weights,
                    &mut self.work_kv_head,
                    &mut self.kv[li],
                    self.kv_dim,
                    cos,
                    sin,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    self.norm_eps,
                    CP_MAX_SEQ,
                );
            }
        }
        rms_norm1_fast(
            &self.work_hidden[..hidden],
            &self.norm_weight,
            eps,
            &mut self.last_hidden[..hidden],
        );
        self.past_len = start_pos + 1;
    }

    /// Two-token CP prefill (talker hidden + group-0 codec embed) via batched `forward_two`.
    pub(crate) fn forward_prefill_two(
        &mut self,
        e0: ArrayView1<f32>,
        e1: ArrayView1<f32>,
    ) -> Result<()> {
        self.forward_prefill_two_slices(e0.as_slice().unwrap(), e1.as_slice().unwrap())
    }

    #[inline(always)]
    pub(crate) fn forward_prefill_two_slices(&mut self, e0: &[f32], e1: &[f32]) -> Result<()> {
        debug_assert!(e0.len() == self.hidden && e1.len() == self.hidden);
        self.work_hidden.copy_from_slice(e0);
        self.work_x1.copy_from_slice(e1);
        let rh = self.head_half;
        debug_assert!(rh <= CP_ROPE_HALF_CAP);
        let mut cos0 = [0f32; CP_ROPE_HALF_CAP];
        let mut sin0 = [0f32; CP_ROPE_HALF_CAP];
        let mut cos1 = [0f32; CP_ROPE_HALF_CAP];
        let mut sin1 = [0f32; CP_ROPE_HALF_CAP];
        cos0[..rh].copy_from_slice(&self.rope_cos_bank[..rh]);
        sin0[..rh].copy_from_slice(&self.rope_sin_bank[..rh]);
        cos1[..rh].copy_from_slice(&self.rope_cos_bank[rh..rh * 2]);
        sin1[..rh].copy_from_slice(&self.rope_sin_bank[rh..rh * 2]);
        if self.layers.len() == CP_DECODE_LAYERS {
            let (c0, s0, c1, s1) = (&cos0[..rh], &sin0[..rh], &cos1[..rh], &sin1[..rh]);
            self.forward_two_layer(0, c0, s0, c1, s1);
            self.forward_two_layer(1, c0, s0, c1, s1);
            self.forward_two_layer(2, c0, s0, c1, s1);
            self.forward_two_layer(3, c0, s0, c1, s1);
            self.forward_two_layer(4, c0, s0, c1, s1);
        } else {
            self.forward_decode_token();
            self.work_hidden.copy_from_slice(e1);
            self.forward_decode_token();
            return Ok(());
        }
        rms_norm1_fast(
            &self.work_x1[..self.hidden],
            &self.norm_weight,
            self.norm_eps,
            &mut self.last_hidden[..self.hidden],
        );
        self.past_len = 2;
        Ok(())
    }

    fn forward_two_layer(
        &mut self,
        li: usize,
        cos0: &[f32],
        sin0: &[f32],
        cos1: &[f32],
        sin1: &[f32],
    ) {
        let layer = &self.layers[li];
        layer.forward_two_fast(
            &mut self.work_hidden,
            &mut self.work_x1,
            &mut self.work_q,
            &mut self.work_k,
            &mut self.work_v,
            &mut self.work_attn,
            &mut self.work_scratch,
            &mut self.work_gate,
            &mut self.work_up,
            &mut self.work_qkv,
            &mut self.work_qkv_b,
            &mut self.work_gate_up,
            &mut self.work_gu_b,
            &mut self.work_norm_b,
            &mut self.work_b_stack,
            &mut self.work_gemm2,
            &mut self.attn_weights,
            &mut self.work_kv_head,
            &mut self.kv[li],
            self.kv_dim,
            cos0,
            sin0,
            cos1,
            sin1,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.norm_eps,
            CP_MAX_SEQ,
        );
    }

    pub fn predict_groups(
        &mut self,
        talker_codec: &Array2<f32>,
        group_embeds: &[Array2<f32>],
        lm_heads: &[Array2<f32>],
        talker_hidden: ArrayView1<f32>,
        group0: u32,
    ) -> Result<Vec<u32>> {
        ensure!(talker_hidden.len() == self.hidden);
        ensure!(
            (group0 as usize) < talker_codec.nrows(),
            "group0 {group0} oob"
        );
        self.reset_kv();
        let e0 = talker_codec.row(group0 as usize);
        self.forward_prefill_two(talker_hidden, e0)?;
        if std::env::var("RLX_QWEN3_TTS_CP_LAYERDBG").ok().as_deref() == Some("1") {
            eprintln!(
                "embeds[0,:8] = {:?}",
                talker_hidden.iter().take(8).collect::<Vec<_>>()
            );
            eprintln!("embeds[1,:8] = {:?}", e0.iter().take(8).collect::<Vec<_>>());
        }
        if std::env::var("RLX_QWEN3_TTS_CP_LAYERDBG").ok().as_deref() == Some("1") {
            eprintln!(
                "cp eager hs[-1,:8] = {:?}",
                self.last_hidden.iter().take(8).collect::<Vec<_>>()
            );
        }
        let mut codes = vec![group0];
        for step in 0..lm_heads.len() {
            let (vocab, _) = lm_heads[step].dim();
            if self.logits.len() != vocab {
                self.logits.resize(vocab, 0.0);
            }
            linear_logits_into(
                ArrayView1::from(&self.last_hidden),
                lm_heads[step].view(),
                &mut self.logits,
            )?;
            let tok = sample_greedy(&self.logits);
            codes.push(tok);
            if step + 1 < lm_heads.len() {
                let row = group_embeds[step].row(tok as usize);
                self.forward_one(row)?;
            }
        }
        Ok(codes)
    }

    /// Like [`Self::predict_groups_fill_emb`] with row-major flat embed + lm_head tables.
    pub fn predict_groups_fill_emb_flat(
        &mut self,
        talker_codec_flat: &[f32],
        group_embed_flat: &[Vec<f32>],
        lm_head_flat: &[Vec<f32>],
        lm_head_vocab: &[usize],
        talker_hidden: ArrayView1<f32>,
        group0: u32,
        pad: &[f32],
        codec_emb: &mut [f32],
        hidden: usize,
    ) -> Result<Vec<u32>> {
        ensure!(codec_emb.len() == hidden);
        ensure!(talker_hidden.len() == hidden);
        ensure!(lm_head_flat.len() == lm_head_vocab.len());
        let g0 = group0 as usize;
        let g0_off = g0 * hidden;
        ensure!(
            g0_off + hidden <= talker_codec_flat.len(),
            "group0 {group0} oob"
        );
        let g0_row = &talker_codec_flat[g0_off..g0_off + hidden];
        codec_emb.copy_from_slice(g0_row);
        for (j, v) in pad.iter().enumerate().take(hidden) {
            codec_emb[j] += *v;
        }

        self.reset_kv();
        self.forward_prefill_two_slices(talker_hidden.as_slice().unwrap(), g0_row)?;

        self.codes_buf.clear();
        self.codes_buf.push(group0);
        let n_steps = lm_head_flat.len();
        if n_steps == CP_AR_LM_STEPS && lm_head_flat.len() == CP_AR_LM_STEPS {
            self.predict_groups_fill_emb_flat_15(
                group_embed_flat,
                lm_head_flat,
                lm_head_vocab,
                hidden,
                codec_emb,
            );
        } else {
            for step in 0..n_steps {
                self.predict_group_step(
                    step,
                    n_steps,
                    &group_embed_flat[step],
                    &lm_head_flat[step],
                    lm_head_vocab[step],
                    hidden,
                    codec_emb,
                )?;
            }
        }
        let out = std::mem::replace(&mut self.codes_buf, Vec::with_capacity(CP_AR_LM_STEPS + 1));
        Ok(out)
    }

    #[inline(always)]
    fn predict_group_step(
        &mut self,
        step: usize,
        n_steps: usize,
        table: &[f32],
        lm_head: &[f32],
        vocab: usize,
        hidden: usize,
        codec_emb: &mut [f32],
    ) -> Result<()> {
        debug_assert!(self.logits.len() >= vocab);
        linear_logits_flat_unchecked(&self.last_hidden, lm_head, vocab, hidden, &mut self.logits);
        let tok = sample_greedy_vocab(&self.logits, vocab);
        self.codes_buf.push(tok);
        let t_off = tok as usize * hidden;
        ensure!(
            t_off + hidden <= table.len(),
            "token {tok} oob for group {}",
            step + 1
        );
        if step + 1 < n_steps {
            let row = &table[t_off..t_off + hidden];
            self.work_hidden.copy_from_slice(row);
            for (d, &v) in codec_emb.iter_mut().zip(row.iter()) {
                *d += v;
            }
            self.forward_decode_token();
        } else {
            for (d, &v) in codec_emb.iter_mut().zip(&table[t_off..t_off + hidden]) {
                *d += v;
            }
        }
        Ok(())
    }

    /// One CP AR lm_head step; `decode` runs the 5-layer decode micro-kernel when true.
    #[inline(always)]
    fn cp_ar_step(
        &mut self,
        step: usize,
        group_embed_flat: &[Vec<f32>],
        lm_head_flat: &[Vec<f32>],
        lm_head_vocab: &[usize],
        hidden: usize,
        codec_emb: &mut [f32],
        decode: bool,
    ) {
        let vocab = lm_head_vocab[step];
        linear_logits_flat_unchecked(
            &self.last_hidden,
            &lm_head_flat[step],
            vocab,
            hidden,
            &mut self.logits,
        );
        let tok = sample_greedy_vocab(&self.logits, vocab);
        self.codes_buf.push(tok);
        let table = group_embed_flat[step].as_slice();
        let t_off = tok as usize * hidden;
        let row = &table[t_off..t_off + hidden];
        if decode {
            self.work_hidden.copy_from_slice(row);
            for (d, &v) in codec_emb.iter_mut().zip(row.iter()) {
                *d += v;
            }
            self.forward_decode_token();
        } else {
            for (d, &v) in codec_emb.iter_mut().zip(row.iter()) {
                *d += v;
            }
        }
    }

    /// Fully unrolled 15-step CP AR for CustomVoice 0.6B (hot synthesis path).
    fn predict_groups_fill_emb_flat_15(
        &mut self,
        group_embed_flat: &[Vec<f32>],
        lm_head_flat: &[Vec<f32>],
        lm_head_vocab: &[usize],
        hidden: usize,
        codec_emb: &mut [f32],
    ) {
        cp_ar_unroll_15!(
            self,
            group_embed_flat,
            lm_head_flat,
            lm_head_vocab,
            hidden,
            codec_emb
        );
    }

    /// Like [`Self::predict_groups`] but also writes codec embed sum + `pad` into `codec_emb`.
    pub fn predict_groups_fill_emb(
        &mut self,
        talker_codec: &Array2<f32>,
        group_embeds: &[Array2<f32>],
        lm_heads: &[Array2<f32>],
        talker_hidden: ArrayView1<f32>,
        group0: u32,
        pad: &[f32],
        codec_emb: &mut [f32],
    ) -> Result<Vec<u32>> {
        ensure!(codec_emb.len() == self.hidden);
        ensure!(talker_hidden.len() == self.hidden);
        ensure!(
            (group0 as usize) < talker_codec.nrows(),
            "group0 {group0} oob"
        );
        codec_emb.fill(0.0);
        for (j, v) in talker_codec.row(group0 as usize).iter().enumerate() {
            codec_emb[j] += *v;
        }
        for (j, v) in pad.iter().enumerate() {
            codec_emb[j] += *v;
        }

        self.reset_kv();
        let e0 = talker_codec.row(group0 as usize);
        self.forward_prefill_two(talker_hidden, e0)?;

        let mut codes = vec![group0];
        for step in 0..lm_heads.len() {
            let (vocab, _) = lm_heads[step].dim();
            if self.logits.len() != vocab {
                self.logits.resize(vocab, 0.0);
            }
            linear_logits_into(
                ArrayView1::from(&self.last_hidden),
                lm_heads[step].view(),
                &mut self.logits,
            )?;
            let tok = sample_greedy(&self.logits);
            codes.push(tok);
            let row = group_embeds[step].row(tok as usize);
            for (j, v) in row.iter().enumerate() {
                codec_emb[j] += *v;
            }
            if step + 1 < lm_heads.len() {
                self.forward_one(row)?;
            }
        }
        Ok(codes)
    }
}

impl LayerKv {
    fn rows(&self, _dim: usize) -> usize {
        self.n_rows
    }
}

impl CpLayer {
    #[inline(always)]
    fn forward_one(
        &self,
        x: &mut [f32],
        attn_out: &mut [f32],
        scratch: &mut [f32],
        work_qkv: &mut [f32],
        work_gate_up: &mut [f32],
        attn_weights: &mut [f32],
        kv_head_scratch: &mut [f32],
        kv: &mut LayerKv,
        kv_dim: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        max_attn: usize,
    ) {
        let hidden = x.len();
        let wqkv = self.wqkv.as_slice().expect("cp wqkv contiguous");
        let wo = self.wo.as_slice().expect("cp wo contiguous");
        let gate_up = self.gate_up.as_slice().expect("cp gate_up contiguous");
        let down = self.down.as_slice().expect("cp down contiguous");
        let use_bf16 = !self.wqkv_bf16.is_empty();

        rms_norm1_fast(x, &self.attn_norm, eps, scratch);
        matvec_maybe_bf16(
            wqkv,
            &self.wqkv_bf16,
            use_bf16,
            scratch,
            work_qkv,
            self.q_dim + 2 * self.kv_dim,
            hidden,
        );
        let (q_part, kv_part) = work_qkv.split_at_mut(self.q_dim);
        let (k_part, v_part) = kv_part.split_at_mut(self.kv_dim);
        qk_norm_heads1(q_part, &self.q_norm, n_heads, head_dim, eps);
        qk_norm_heads1(k_part, &self.k_norm, n_kv_heads, head_dim, eps);
        apply_rope1_flat(q_part, n_heads, head_dim, rope_cos, rope_sin);
        apply_rope1_flat(k_part, n_kv_heads, head_dim, rope_cos, rope_sin);
        append_kv1(kv, kv_dim, k_part, v_part);
        let t_k = kv.n_rows;
        gqa_attention1_cp(
            q_part, &kv.k, &kv.v, t_k, kv_dim, attn_out, n_heads, n_kv_heads, head_dim,
        );
        let _ = (attn_weights, kv_head_scratch, max_attn);
        matvec_maybe_bf16(
            wo,
            &self.wo_bf16,
            use_bf16,
            attn_out,
            scratch,
            hidden,
            self.q_dim,
        );
        for i in 0..hidden {
            scratch[i] += x[i];
        }
        rms_norm1_fast(scratch, &self.ffn_norm, eps, x);
        matvec_maybe_bf16(
            gate_up,
            &self.gate_up_bf16,
            use_bf16,
            x,
            work_gate_up,
            2 * self.inter_dim,
            hidden,
        );
        let (gate_part, up_part) = work_gate_up.split_at_mut(self.inter_dim);
        for i in 0..self.inter_dim {
            gate_part[i] = silu(gate_part[i]) * up_part[i];
        }
        matvec_maybe_bf16(
            down,
            &self.down_bf16,
            use_bf16,
            gate_part,
            x,
            hidden,
            self.inter_dim,
        );
        for i in 0..hidden {
            x[i] += scratch[i];
        }
    }

    #[inline(always)]
    fn forward_two_fast(
        &self,
        x0: &mut [f32],
        x1: &mut [f32],
        _q: &mut [f32],
        _k: &mut [f32],
        _v: &mut [f32],
        attn_out: &mut [f32],
        scratch0: &mut [f32],
        gate0: &mut [f32],
        _up0: &mut [f32],
        qkv0: &mut [f32],
        qkv1: &mut [f32],
        gu0: &mut [f32],
        gu1: &mut [f32],
        norm_b: &mut [f32],
        _b_stack: &mut [f32],
        _gemm2: &mut [f32],
        attn_weights: &mut [f32],
        kv_head_scratch: &mut [f32],
        kv: &mut LayerKv,
        kv_dim: usize,
        cos0: &[f32],
        sin0: &[f32],
        cos1: &[f32],
        sin1: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        max_attn: usize,
    ) {
        let hidden = x0.len();
        let wqkv = self.wqkv.as_slice().expect("cp wqkv contiguous");
        let gate_up = self.gate_up.as_slice().expect("cp gate_up contiguous");
        let down = self.down.as_slice().expect("cp down contiguous");
        rms_norm1_fast(x0, &self.attn_norm, eps, scratch0);
        rms_norm1_fast(x1, &self.attn_norm, eps, norm_b);
        let qkv_dim = self.q_dim + 2 * self.kv_dim;
        matvec_blas(wqkv, scratch0, qkv0, qkv_dim, hidden);
        matvec_blas(wqkv, norm_b, qkv1, qkv_dim, hidden);

        self.forward_two_token_fast(
            x0,
            qkv0,
            attn_out,
            scratch0,
            attn_weights,
            kv_head_scratch,
            kv,
            kv_dim,
            cos0,
            sin0,
            n_heads,
            n_kv_heads,
            head_dim,
            eps,
            max_attn,
            hidden,
        );
        self.forward_two_token_fast(
            x1,
            qkv1,
            attn_out,
            norm_b,
            attn_weights,
            kv_head_scratch,
            kv,
            kv_dim,
            cos1,
            sin1,
            n_heads,
            n_kv_heads,
            head_dim,
            eps,
            max_attn,
            hidden,
        );

        rms_norm1_fast(x0, &self.ffn_norm, eps, scratch0);
        rms_norm1_fast(x1, &self.ffn_norm, eps, norm_b);
        matvec_blas(gate_up, scratch0, gu0, 2 * self.inter_dim, hidden);
        matvec_blas(gate_up, norm_b, gu1, 2 * self.inter_dim, hidden);
        for i in 0..self.inter_dim {
            gate0[i] = silu(gu0[i]) * gu0[self.inter_dim + i];
            gu1[i] = silu(gu1[i]) * gu1[self.inter_dim + i];
        }
        matvec_accumulate_blas(down, gate0, x0, hidden, self.inter_dim);
        matvec_accumulate_blas(down, &gu1[..self.inter_dim], x1, hidden, self.inter_dim);
    }

    #[inline(always)]
    fn forward_two_token_fast(
        &self,
        x: &mut [f32],
        qkv: &mut [f32],
        attn_out: &mut [f32],
        scratch: &mut [f32],
        attn_weights: &mut [f32],
        kv_head_scratch: &mut [f32],
        kv: &mut LayerKv,
        kv_dim: usize,
        cos: &[f32],
        sin: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        max_attn: usize,
        hidden: usize,
    ) {
        let (q_part, kv_part) = qkv.split_at_mut(self.q_dim);
        let (k_part, v_part) = kv_part.split_at_mut(self.kv_dim);
        qk_norm_heads1(q_part, &self.q_norm, n_heads, head_dim, eps);
        qk_norm_heads1(k_part, &self.k_norm, n_kv_heads, head_dim, eps);
        apply_rope1_flat(q_part, n_heads, head_dim, cos, sin);
        apply_rope1_flat(k_part, n_kv_heads, head_dim, cos, sin);
        append_kv1(kv, kv_dim, k_part, v_part);
        let t_k = kv.n_rows;
        gqa_attention1_cp(
            q_part, &kv.k, &kv.v, t_k, kv_dim, attn_out, n_heads, n_kv_heads, head_dim,
        );
        let _ = (attn_weights, kv_head_scratch, max_attn);
        let wo = self.wo.as_slice().expect("cp wo contiguous");
        matvec_blas(wo, attn_out, scratch, hidden, self.q_dim);
        for i in 0..hidden {
            scratch[i] += x[i];
        }
        x.copy_from_slice(scratch);
    }

    fn forward(
        &self,
        x: ArrayView2<f32>,
        kv: &mut LayerKv,
        kv_dim: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        start_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Array2<f32>> {
        let h = rms_norm2(x, &self.attn_norm, eps);
        let mut q = linear2(h.view(), self.wq.view());
        let mut k = linear2(h.view(), self.wk.view());
        let v = linear2(h.view(), self.wv.view());
        qk_norm_heads(&mut q, &self.q_norm, n_heads, head_dim, eps);
        qk_norm_heads(&mut k, &self.k_norm, n_kv_heads, head_dim, eps);
        apply_rope_qk(
            &mut q, &mut k, rope_cos, rope_sin, start_pos, n_heads, n_kv_heads, head_dim,
        );
        append_kv(kv, kv_dim, &k, &v);
        let t_k = kv.rows(kv_dim);
        let k_view = ArrayView2::from_shape((t_k, kv_dim), &kv.k)?;
        let v_view = ArrayView2::from_shape((t_k, kv_dim), &kv.v)?;
        let attn = gqa_attention(q.view(), k_view, v_view, n_heads, n_kv_heads, head_dim);
        let attn_out = linear2(attn.view(), self.wo.view());
        let mut out = x.to_owned() + attn_out;
        let h2 = rms_norm2(out.view(), &self.ffn_norm, eps);
        let gate = linear2(h2.view(), self.gate.view());
        let up = linear2(h2.view(), self.up.view());
        let ff = linear2((silu2(gate.view()) * up).view(), self.down.view());
        out = out + ff;
        Ok(out)
    }
}

fn stack_proj_weights(parts: &[Array2<f32>]) -> Array2<f32> {
    let out_dim: usize = parts.iter().map(|w| w.nrows()).sum();
    let in_dim = parts[0].ncols();
    let mut out = Array2::<f32>::zeros((out_dim, in_dim));
    let out_slice = out.as_slice_mut().expect("contiguous");
    let mut row = 0usize;
    for w in parts {
        debug_assert_eq!(w.ncols(), in_dim);
        let src = w.as_slice().expect("contiguous part");
        let n = w.nrows() * in_dim;
        out_slice[row * in_dim..row * in_dim + n].copy_from_slice(src);
        row += w.nrows();
    }
    out
}

fn load_layer(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, i: usize) -> Result<CpLayer> {
    let p = format!("talker.code_predictor.model.layers.{i}");
    let wq = take2d(map, &format!("{p}.self_attn.q_proj.weight"))?;
    let wk = take2d(map, &format!("{p}.self_attn.k_proj.weight"))?;
    let wv = take2d(map, &format!("{p}.self_attn.v_proj.weight"))?;
    let gate = take2d(map, &format!("{p}.mlp.gate_proj.weight"))?;
    let up = take2d(map, &format!("{p}.mlp.up_proj.weight"))?;
    let q_dim = wq.nrows();
    let kv_dim = wk.nrows();
    let inter_dim = gate.nrows();
    let wqkv = stack_proj_weights(&[wq.clone(), wk.clone(), wv.clone()]);
    let gate_up = stack_proj_weights(&[gate.clone(), up.clone()]);
    let wo = take2d(map, &format!("{p}.self_attn.o_proj.weight"))?;
    let down = take2d(map, &format!("{p}.mlp.down_proj.weight"))?;
    let bf16_on = cp_bf16_enabled();
    let to_bf16 = |arr: &Array2<f32>| -> Vec<u16> {
        if bf16_on {
            arr.as_slice()
                .expect("contiguous")
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect()
        } else {
            Vec::new()
        }
    };
    let wqkv_bf16 = to_bf16(&wqkv);
    let wo_bf16 = to_bf16(&wo);
    let gate_up_bf16 = to_bf16(&gate_up);
    let down_bf16 = to_bf16(&down);
    Ok(CpLayer {
        wqkv,
        gate_up,
        wq,
        wk,
        wv,
        wo,
        q_norm: take1d(map, &format!("{p}.self_attn.q_norm.weight"))?,
        k_norm: take1d(map, &format!("{p}.self_attn.k_norm.weight"))?,
        attn_norm: take1d(map, &format!("{p}.input_layernorm.weight"))?,
        ffn_norm: take1d(map, &format!("{p}.post_attention_layernorm.weight"))?,
        gate,
        up,
        down,
        wqkv_bf16,
        wo_bf16,
        gate_up_bf16,
        down_bf16,
        q_dim,
        kv_dim,
        inter_dim,
    })
}

fn cp_bf16_enabled() -> bool {
    std::env::var("RLX_QWEN3_TTS_CP_BF16").ok().as_deref() == Some("1")
}

/// BF16-weight × F32-input matvec via NEON `vshll_n_u16`. Output is F32.
/// Halves weight memory bandwidth; compute uses standard F32 NEON FMA.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_bf16_neon(w: &[u16], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    use std::arch::aarch64::*;
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);
    let lanes = 8usize;
    let bulk = in_dim - (in_dim % lanes);
    let x_ptr = x.as_ptr();
    // SAFETY: caller upholds the function's safety contract (aarch64 NEON,
    // correct slice lengths). All pointer arithmetic stays in bounds.
    unsafe {
        for co in 0..out_dim {
            let w_row = w.as_ptr().add(co * in_dim);
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut ci = 0usize;
            while ci < bulk {
                let w_u16 = vld1q_u16(w_row.add(ci));
                let w_lo_f32 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(w_u16)));
                let w_hi_f32 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(w_u16));
                let x_lo = vld1q_f32(x_ptr.add(ci));
                let x_hi = vld1q_f32(x_ptr.add(ci + 4));
                acc0 = vfmaq_f32(acc0, w_lo_f32, x_lo);
                acc1 = vfmaq_f32(acc1, w_hi_f32, x_hi);
                ci += lanes;
            }
            let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            while ci < in_dim {
                let bf = *w_row.add(ci);
                let f = f32::from_bits((bf as u32) << 16);
                sum += f * *x_ptr.add(ci);
                ci += 1;
            }
            *out.get_unchecked_mut(co) = sum;
        }
    }
}

/// Dispatch to BF16 NEON if available + enabled, else fall through to BLAS sgemm.
#[inline(always)]
fn matvec_maybe_bf16(
    w_f32: &[f32],
    w_bf16: &[u16],
    use_bf16: bool,
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if use_bf16 && !w_bf16.is_empty() {
        unsafe {
            matvec_bf16_neon(w_bf16, x, out, out_dim, in_dim);
        }
        return;
    }
    let _ = (w_bf16, use_bf16);
    matvec_blas(w_f32, x, out, out_dim, in_dim);
}

fn take2d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 2);
    Array2::from_shape_vec((shape[0], shape[1]), data.clone()).with_context(|| key.to_string())
}

fn take1d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Vec<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 1);
    Ok(data.clone())
}

fn linear2(x: ArrayView2<f32>, w: ArrayView2<f32>) -> Array2<f32> {
    x.dot(&w.t())
}

fn rms_norm2(x: ArrayView2<f32>, weight: &[f32], eps: f32) -> Array2<f32> {
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mut sum = 0f32;
        for v in row.iter() {
            sum += v * v;
        }
        let inv = 1.0 / (sum / d as f32 + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = row[j] * inv * weight[j];
        }
    }
    out
}

fn qk_norm_heads(q: &mut Array2<f32>, weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    let t = q.nrows();
    for ti in 0..t {
        for h in 0..n_heads {
            let off = h * head_dim;
            let mut sum = 0f32;
            for di in 0..head_dim {
                sum += q[[ti, off + di]] * q[[ti, off + di]];
            }
            let inv = 1.0 / (sum / head_dim as f32 + eps).sqrt();
            for di in 0..head_dim {
                q[[ti, off + di]] *= inv * weight[di];
            }
        }
    }
}

fn silu2(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

fn build_rope_tables(inv_freq: &[f64], max_pos: usize, half: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for (i, &freq) in inv_freq.iter().enumerate() {
            let ang = pos as f64 * freq;
            cos[pos * half + i] = ang.cos() as f32;
            sin[pos * half + i] = ang.sin() as f32;
        }
    }
    (cos, sin)
}

fn apply_rope_qk(
    q: &mut Array2<f32>,
    k: &mut Array2<f32>,
    cos: &[f32],
    sin: &[f32],
    start_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    let seq = q.nrows();
    for ti in 0..seq {
        let pos = start_pos + ti;
        for hi in 0..n_heads {
            rotate_row(q, ti, hi * head_dim, cos, sin, pos, half);
        }
        for hi in 0..n_kv_heads {
            rotate_row(k, ti, hi * head_dim, cos, sin, pos, half);
        }
    }
}

fn rotate_row(
    x: &mut Array2<f32>,
    row: usize,
    col_off: usize,
    cos: &[f32],
    sin: &[f32],
    pos: usize,
    half: usize,
) {
    for i in 0..half {
        let c = cos[pos * half + i];
        let s = sin[pos * half + i];
        let x0 = x[[row, col_off + i]];
        let x1 = x[[row, col_off + half + i]];
        x[[row, col_off + i]] = x0 * c - x1 * s;
        x[[row, col_off + half + i]] = x0 * s + x1 * c;
    }
}

#[inline(always)]
fn rms_norm1_fast(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len(), weight.len());
    let d = x.len();
    let mut sum = 0f32;
    for v in x {
        sum += v * v;
    }
    let inv = 1.0 / (sum / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * inv * weight[i];
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn qk_norm_heads1(q: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let off = h * head_dim;
        let mut sum = 0f32;
        for di in 0..head_dim {
            let v = q[off + di];
            sum += v * v;
        }
        let inv = 1.0 / (sum / head_dim as f32 + eps).sqrt();
        for di in 0..head_dim {
            q[off + di] *= inv * weight[di];
        }
    }
}

#[inline]
fn apply_rope1_flat(x: &mut [f32], n_heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let c = cos[i];
            let s = sin[i];
            let x0 = x[off + i];
            let x1 = x[off + half + i];
            x[off + i] = x0 * c - x1 * s;
            x[off + half + i] = x0 * s + x1 * c;
        }
    }
}

#[inline(always)]
fn append_kv1(kv: &mut LayerKv, dim: usize, k: &[f32], v: &[f32]) {
    debug_assert_eq!(k.len(), dim);
    debug_assert_eq!(v.len(), dim);
    let off = kv.n_rows * dim;
    debug_assert!(off + dim <= kv.k.len() && off + dim <= kv.v.len());
    kv.k[off..off + dim].copy_from_slice(k);
    kv.v[off..off + dim].copy_from_slice(v);
    kv.n_rows += 1;
}

fn append_kv(kv: &mut LayerKv, dim: usize, k: &Array2<f32>, v: &Array2<f32>) {
    let (t_new, d) = k.dim();
    debug_assert_eq!(d, dim);
    debug_assert_eq!(v.dim(), (t_new, dim));
    for row in 0..t_new {
        let off = kv.n_rows * dim;
        kv.k[off..off + dim].copy_from_slice(k.row(row).as_slice().unwrap());
        kv.v[off..off + dim].copy_from_slice(v.row(row).as_slice().unwrap());
        kv.n_rows += 1;
    }
}

fn gqa_attention(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    let t_q = q.nrows();
    let t_k = k.nrows();
    let kv_off = t_k.saturating_sub(t_q);
    let repeats = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = Array2::<f32>::zeros((t_q, n_heads * head_dim));
    for qi in 0..t_q {
        let kq = qi + kv_off;
        for hi in 0..n_heads {
            let kv_h = hi / repeats;
            let mut max_w = f32::NEG_INFINITY;
            let mut weights = vec![f32::NEG_INFINITY; t_k];
            for ki in 0..=kq {
                let mut dot = 0f32;
                for di in 0..head_dim {
                    dot += q[[qi, hi * head_dim + di]] * k[[ki, kv_h * head_dim + di]];
                }
                dot *= scale;
                weights[ki] = dot;
                max_w = max_w.max(dot);
            }
            let mut sum = 0f32;
            for w in weights.iter_mut() {
                if w.is_finite() {
                    *w = (*w - max_w).exp();
                    sum += *w;
                } else {
                    *w = 0.0;
                }
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for w in weights.iter_mut() {
                *w *= inv;
            }
            for di in 0..head_dim {
                let mut acc = 0f32;
                for ki in 0..t_k {
                    acc += weights[ki] * v[[ki, kv_h * head_dim + di]];
                }
                out[[qi, hi * head_dim + di]] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod forward_one_tests {
    use super::*;
    use ndarray::Array2;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn forward_one_matches_forward_on_single_token() {
        let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR")
            .ok()
            .map(std::path::PathBuf::from)
        else {
            eprintln!("skip: RLX_QWEN3_TTS_DIR");
            return;
        };
        let cfg = crate::Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
        let store = crate::load::Qwen3TtsWeightStore::open(&model_dir).unwrap();
        let cp_cfg = cfg.code_predictor();
        let key = "talker.code_predictor.model.codec_embedding.0.weight";
        let (data, shape) = store.tensor_snapshot(&[key]).unwrap()[key].clone();
        let table = Array2::from_shape_vec((shape[0], shape[1]), data).unwrap();

        let mut eager = CpEagerModel::open(&store, cp_cfg).unwrap();
        let embed = table.row(1642);
        let emb1 = Array2::from_shape_vec((1, cp_cfg.hidden_size), embed.to_vec()).unwrap();
        let out = eager.forward(emb1.view()).unwrap();
        let via_forward: Vec<f32> = out.row(0).iter().copied().collect();

        eager.reset_kv();
        eager.forward_one(embed).unwrap();
        let d = max_abs(&via_forward, eager.last_hidden());
        eprintln!("forward_one vs forward max_abs = {d}");
        assert!(d < 1e-4, "forward_one diverged (max_abs={d})");
    }
}
