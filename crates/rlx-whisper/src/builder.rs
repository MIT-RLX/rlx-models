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

//! Whisper encoder + decoder HIR graphs (OpenAI architecture).

use crate::config::WhisperConfig;
use crate::fused::{FusedDecoderWeights, FusedEncoderWeights};
use crate::weights::WhisperWeightPrefix;
use anyhow::{Result, ensure};
use rlx_flow::WeightSource;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Op, Shape};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-5;

/// Default encoder self-attention query chunk size (full K/V; lowers peak attn memory).
pub const DEFAULT_ENCODER_ATTN_CHUNK: usize = 256;

/// Graph build options (precision + fused decoder weights).
#[derive(Debug, Clone, Copy)]
pub struct WhisperGraphOpts {
    /// Activation / matmul dtype (F16 for mixed precision).
    pub act_dtype: DType,
    /// LayerNorm and embeddings stay F32 when `use_f16_compute`.
    pub use_f16_compute: bool,
    /// Encoder self-attn: process Q in chunks of this many positions (`0` = one shot).
    pub encoder_attn_chunk: usize,
    /// Decoder cross-attn (prefill): chunk Q along `dec_seq` when `enc_seq` is long (`0` = off).
    pub cross_attn_chunk: usize,
}

impl Default for WhisperGraphOpts {
    fn default() -> Self {
        Self {
            act_dtype: DType::F32,
            use_f16_compute: false,
            encoder_attn_chunk: DEFAULT_ENCODER_ATTN_CHUNK,
            cross_attn_chunk: DEFAULT_ENCODER_ATTN_CHUNK,
        }
    }
}

impl WhisperGraphOpts {
    pub fn f16_mixed() -> Self {
        Self {
            act_dtype: DType::F16,
            use_f16_compute: true,
            encoder_attn_chunk: DEFAULT_ENCODER_ATTN_CHUNK,
            cross_attn_chunk: DEFAULT_ENCODER_ATTN_CHUNK,
        }
    }
}

pub(crate) struct WhisperBuilder<'a> {
    pub hir: &'a mut HirModule,
    pub params: &'a mut HashMap<String, Vec<f32>>,
    pub weights: &'a mut dyn WeightSource,
    pub pfx: &'a WhisperWeightPrefix,
    pub batch: usize,
    pub f: DType,
    pub opts: WhisperGraphOpts,
    pub fused: Option<&'a FusedDecoderWeights>,
    pub fused_enc: Option<&'a FusedEncoderWeights>,
}

impl<'a> WhisperBuilder<'a> {
    pub(crate) fn new(
        hir: &'a mut HirModule,
        params: &'a mut HashMap<String, Vec<f32>>,
        weights: &'a mut dyn WeightSource,
        pfx: &'a WhisperWeightPrefix,
        batch: usize,
        opts: WhisperGraphOpts,
    ) -> Self {
        let f = if opts.use_f16_compute {
            opts.act_dtype
        } else {
            DType::F32
        };
        Self {
            hir,
            params,
            weights,
            pfx,
            batch,
            f,
            opts,
            fused: None,
            fused_enc: None,
        }
    }

    pub(crate) fn with_fused(mut self, fused: &'a FusedDecoderWeights) -> Self {
        self.fused = Some(fused);
        self
    }

    pub(crate) fn with_fused_enc(mut self, fused: &'a FusedEncoderWeights) -> Self {
        self.fused_enc = Some(fused);
        self
    }

    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    fn maybe_cast_compute(&mut self, x: HirNodeId) -> HirNodeId {
        let f = self.f;
        if self.opts.use_f16_compute && f != DType::F32 {
            self.g().cast(x, f)
        } else {
            x
        }
    }

    fn layer_norm_f32(&mut self, x: HirNodeId, w: &str, b: &str) -> Result<HirNodeId> {
        let f = self.f;
        let x32 = if self.opts.use_f16_compute {
            self.g().cast(x, DType::F32)
        } else {
            x
        };
        let gamma = self.load_param(w, false)?;
        let beta = self.load_param(b, false)?;
        let y = self.g().ln(x32, gamma, beta, LN_EPS);
        Ok(if self.opts.use_f16_compute {
            self.g().cast(y, f)
        } else {
            y
        })
    }

    pub(crate) fn emit_encoder(
        &mut self,
        cfg: &WhisperConfig,
        mel: HirNodeId,
        mel_frames: usize,
        enc_seq: usize,
    ) -> Result<HirNodeId> {
        self.emit_encoder_inner(cfg, mel, mel_frames, enc_seq)
    }

    pub(crate) fn emit_decoder(
        &mut self,
        cfg: &WhisperConfig,
        token_ids: HirNodeId,
        encoder_hidden: HirNodeId,
        dec_seq: usize,
        enc_seq: usize,
    ) -> Result<HirNodeId> {
        self.emit_decoder_inner(cfg, token_ids, encoder_hidden, dec_seq, enc_seq)
    }
}

pub fn build_whisper_encoder_hir(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    mel_frames: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let enc_seq = cfg.encoder_seq_len(mel_frames);
    ensure!(
        enc_seq <= cfg.max_source_positions,
        "mel frames {mel_frames} → encoder seq {enc_seq} exceeds max_source_positions {}",
        cfg.max_source_positions
    );
    let f = DType::F32;
    let mut hir = HirModule::new("whisper_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let mel = hir.input("mel", Shape::new(&[batch, cfg.num_mel_bins, mel_frames], f));
    let mut b = WhisperBuilder::new(
        &mut hir,
        &mut params,
        weights,
        pfx,
        batch,
        WhisperGraphOpts::default(),
    );
    let hidden = b.emit_encoder_inner(cfg, mel, mel_frames, enc_seq)?;
    hir.outputs = vec![hidden];
    Ok((hir, params))
}

pub fn build_whisper_decoder_hir(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    ensure!(
        dec_seq <= cfg.max_target_positions,
        "decoder seq {dec_seq} exceeds max_target_positions {}",
        cfg.max_target_positions
    );
    let f = DType::F32;
    let h = cfg.d_model;
    let mut hir = HirModule::new("whisper_decoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let token_ids = hir.input("token_ids", Shape::new(&[batch, dec_seq], f));
    let encoder_hidden = hir.input("encoder_hidden", Shape::new(&[batch, enc_seq, h], f));
    let mut b = make_builder(
        &mut hir,
        &mut params,
        weights,
        pfx,
        batch,
        WhisperGraphOpts::default(),
        None,
    );
    let logits = b.emit_decoder_inner(cfg, token_ids, encoder_hidden, dec_seq, enc_seq)?;
    hir.outputs = vec![logits];
    Ok((hir, params))
}

/// Cross-attention K/V from encoder hidden (computed once per utterance).
pub fn build_whisper_cross_kv_hir(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    enc_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let f = DType::F32;
    let h = cfg.d_model;
    let mut hir = HirModule::new("whisper_cross_kv").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let encoder_hidden = hir.input("encoder_hidden", Shape::new(&[batch, enc_seq, h], f));
    let mut b = make_builder(
        &mut hir,
        &mut params,
        weights,
        pfx,
        batch,
        WhisperGraphOpts::default(),
        None,
    );
    let mut outs = Vec::new();
    for i in 0..cfg.decoder_layers {
        let (k, v) = b.emit_cross_kv_layer(cfg, i, encoder_hidden, enc_seq)?;
        outs.push(k);
        outs.push(v);
    }
    hir.outputs = outs;
    Ok((hir, params))
}

/// Decoder prefill: prompt tokens → logits + per-layer self-attention K/V caches.
pub fn build_whisper_decoder_prefill_hir(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_whisper_decoder_prefill_hir_ext(cfg, weights, pfx, batch, dec_seq, enc_seq, false)
}

pub fn build_whisper_decoder_prefill_hir_ext(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
    use_cross_cache: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_whisper_decoder_prefill_hir_ext_opts(
        cfg,
        weights,
        pfx,
        batch,
        dec_seq,
        enc_seq,
        use_cross_cache,
        WhisperGraphOpts::default(),
        None,
    )
}

pub fn build_whisper_decoder_prefill_hir_ext_opts(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
    use_cross_cache: bool,
    graph_opts: WhisperGraphOpts,
    fused: Option<&FusedDecoderWeights>,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let f = DType::F32;
    let h = cfg.d_model;
    let mut hir =
        HirModule::new("whisper_decoder_prefill").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let token_ids = hir.input("token_ids", Shape::new(&[batch, dec_seq], f));
    let encoder_hidden = if use_cross_cache {
        None
    } else {
        Some(hir.input("encoder_hidden", Shape::new(&[batch, enc_seq, h], f)))
    };
    let mut cross_k = Vec::with_capacity(cfg.decoder_layers);
    let mut cross_v = Vec::with_capacity(cfg.decoder_layers);
    if use_cross_cache {
        for i in 0..cfg.decoder_layers {
            cross_k.push(hir.input(format!("cross_k_{i}"), Shape::new(&[batch, enc_seq, h], f)));
            cross_v.push(hir.input(format!("cross_v_{i}"), Shape::new(&[batch, enc_seq, h], f)));
        }
    }
    let mut b = make_builder(
        &mut hir,
        &mut params,
        weights,
        pfx,
        batch,
        graph_opts,
        fused,
    );
    let (logits, kv) = b.emit_decoder_prefill_inner(
        cfg,
        token_ids,
        encoder_hidden,
        &cross_k,
        &cross_v,
        dec_seq,
        enc_seq,
    )?;
    hir.outputs = vec![logits];
    for (k, v) in kv {
        hir.outputs.push(k);
        hir.outputs.push(v);
    }
    Ok((hir, params))
}

/// Single-token decode step with cached self-attention K/V.
pub fn build_whisper_decode_step_hir(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    past_seq: usize,
    enc_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_whisper_decode_step_hir_ext(cfg, weights, pfx, batch, past_seq, enc_seq, false)
}

/// Decode step with optional bucketed self-attn mask and cross-attn KV inputs.
pub fn build_whisper_decode_step_hir_ext(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    past_seq: usize,
    enc_seq: usize,
    use_custom_mask: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    build_whisper_decode_step_hir_ext_opts(
        cfg,
        weights,
        pfx,
        batch,
        past_seq,
        enc_seq,
        use_custom_mask,
        WhisperGraphOpts::default(),
        None,
    )
}

pub fn build_whisper_decode_step_hir_ext_opts(
    cfg: &WhisperConfig,
    weights: &mut dyn WeightSource,
    pfx: &WhisperWeightPrefix,
    batch: usize,
    past_seq: usize,
    enc_seq: usize,
    use_custom_mask: bool,
    graph_opts: WhisperGraphOpts,
    fused: Option<&FusedDecoderWeights>,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    validate_cfg(cfg)?;
    let f = DType::F32;
    let h = cfg.d_model;
    let mut hir = HirModule::new("whisper_decode_step").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let token_id = hir.input("token_id", Shape::new(&[batch, 1], f));
    let pos_ix = hir.input("pos_ix", Shape::new(&[batch, 1], f));
    let mask = if use_custom_mask {
        Some(hir.input("mask", Shape::new(&[batch, past_seq + 1], f)))
    } else {
        None
    };
    let mut cross_k = Vec::with_capacity(cfg.decoder_layers);
    let mut cross_v = Vec::with_capacity(cfg.decoder_layers);
    for i in 0..cfg.decoder_layers {
        cross_k.push(hir.input(format!("cross_k_{i}"), Shape::new(&[batch, enc_seq, h], f)));
        cross_v.push(hir.input(format!("cross_v_{i}"), Shape::new(&[batch, enc_seq, h], f)));
    }
    let mut past_k = Vec::with_capacity(cfg.decoder_layers);
    let mut past_v = Vec::with_capacity(cfg.decoder_layers);
    for i in 0..cfg.decoder_layers {
        past_k.push(hir.input(format!("past_k_{i}"), Shape::new(&[batch, past_seq, h], f)));
        past_v.push(hir.input(format!("past_v_{i}"), Shape::new(&[batch, past_seq, h], f)));
    }
    let mut b = make_builder(
        &mut hir,
        &mut params,
        weights,
        pfx,
        batch,
        graph_opts,
        fused,
    );
    let (logits, kv) = b.emit_decoder_step_inner(
        cfg,
        token_id,
        Some(pos_ix),
        mask,
        &cross_k,
        &cross_v,
        past_seq,
        enc_seq,
        &past_k,
        &past_v,
    )?;
    hir.outputs = vec![logits];
    for (k, v) in kv {
        hir.outputs.push(k);
        hir.outputs.push(v);
    }
    Ok((hir, params))
}

fn make_builder<'a>(
    hir: &'a mut HirModule,
    params: &'a mut HashMap<String, Vec<f32>>,
    weights: &'a mut dyn WeightSource,
    pfx: &'a WhisperWeightPrefix,
    batch: usize,
    opts: WhisperGraphOpts,
    fused: Option<&'a FusedDecoderWeights>,
) -> WhisperBuilder<'a> {
    let mut b = WhisperBuilder::new(hir, params, weights, pfx, batch, opts);
    if let Some(f) = fused {
        b = b.with_fused(f);
    }
    b
}

impl<'a> WhisperBuilder<'a> {
    /// OpenAI Whisper dot-product scale `(head_dim)^-0.25`, not the default `^-0.5`.
    fn whisper_attention(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        n_head: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        out_shape: Shape,
    ) -> HirNodeId {
        let scale = (head_dim as f32).powf(-0.25);
        self.g().attention_kind_opts(
            q,
            k,
            v,
            n_head,
            head_dim,
            mask_kind,
            out_shape,
            Some(scale),
            None,
        )
    }

    fn whisper_attention_masked(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        n_head: usize,
        head_dim: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        let scale = (head_dim as f32).powf(-0.25);
        self.hir.mir(
            rlx_ir::ops::attention::attention_kind_op(
                n_head,
                head_dim,
                MaskKind::Custom,
                Some(scale),
                None,
            ),
            vec![q, k, v, mask],
            out_shape,
        )
    }

    fn emit_encoder_inner(
        &mut self,
        cfg: &WhisperConfig,
        mel: HirNodeId,
        mel_frames: usize,
        enc_seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let mut x = self.conv1d_gelu(
            mel,
            &self.pfx.enc_conv1_w(),
            &self.pfx.enc_conv1_b(),
            cfg.num_mel_bins,
            d,
            mel_frames,
            3,
            1,
            1,
        )?;
        x = self.conv1d_gelu(
            x,
            &self.pfx.enc_conv2_w(),
            &self.pfx.enc_conv2_b(),
            d,
            d,
            mel_frames,
            3,
            2,
            1,
        )?;
        x = self.maybe_cast_compute(x);
        // [B, C, T] → [B, T, C]
        x = self.g().transpose_(x, vec![0, 2, 1]);
        let pos = self.encoder_sinusoids(cfg.max_source_positions, d)?;
        let pos_slice = self.g().narrow_(pos, 0, 0, enc_seq);
        let pos_bc = self.broadcast_pos(pos_slice, enc_seq, d)?;
        x = self.g().add(x, pos_bc);
        for i in 0..cfg.encoder_layers {
            x = self.residual_block(
                cfg,
                i,
                x,
                None,
                enc_seq,
                cfg.encoder_attention_heads,
                MaskKind::None,
                true,
            )?;
        }
        self.layer_norm_f32(x, &self.pfx.enc_ln_post_w(), &self.pfx.enc_ln_post_b())
    }

    fn emit_decoder_inner(
        &mut self,
        cfg: &WhisperConfig,
        token_ids: HirNodeId,
        encoder_hidden: HirNodeId,
        dec_seq: usize,
        _enc_seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let embed_w = self.load_param(&self.pfx.dec_embed_tokens(), false)?;
        let mut x = self.g().gather_(embed_w, token_ids, 0);
        let pos_w = self.load_param(&self.pfx.dec_embed_positions(), false)?;
        let pos = self.g().narrow_(pos_w, 0, 0, dec_seq);
        let pos_bc = self.broadcast_pos(pos, dec_seq, d)?;
        x = self.g().add(x, pos_bc);
        for i in 0..cfg.decoder_layers {
            x = self.residual_block(
                cfg,
                i,
                x,
                Some(encoder_hidden),
                dec_seq,
                cfg.decoder_attention_heads,
                MaskKind::Causal,
                false,
            )?;
        }
        x = self.layer_norm_f32(x, &self.pfx.dec_ln_w(), &self.pfx.dec_ln_b())?;
        self.emit_logits(x, dec_seq, embed_w)
    }

    fn emit_cross_kv_layer(
        &mut self,
        cfg: &WhisperConfig,
        layer: usize,
        encoder_hidden: HirNodeId,
        enc_seq: usize,
    ) -> Result<(HirNodeId, HirNodeId)> {
        let d = cfg.d_model;
        let layer_pfx = |suffix: &str| self.pfx.dec_layer(layer, suffix);
        let k = self.linear(
            encoder_hidden,
            &layer_pfx("encoder_attn.k_proj.weight"),
            None,
            enc_seq,
            d,
            d,
        )?;
        let v = self.linear(
            encoder_hidden,
            &layer_pfx("encoder_attn.v_proj.weight"),
            Some(layer_pfx("encoder_attn.v_proj.bias").as_str()),
            enc_seq,
            d,
            d,
        )?;
        Ok((k, v))
    }

    fn emit_decoder_prefill_inner(
        &mut self,
        cfg: &WhisperConfig,
        token_ids: HirNodeId,
        encoder_hidden: Option<HirNodeId>,
        cross_k: &[HirNodeId],
        cross_v: &[HirNodeId],
        dec_seq: usize,
        _enc_seq: usize,
    ) -> Result<(HirNodeId, Vec<(HirNodeId, HirNodeId)>)> {
        let d = cfg.d_model;
        let embed_w = self.load_param(&self.pfx.dec_embed_tokens(), false)?;
        let mut x = self.g().gather_(embed_w, token_ids, 0);
        let pos_w = self.load_param(&self.pfx.dec_embed_positions(), false)?;
        let pos = self.g().narrow_(pos_w, 0, 0, dec_seq);
        let pos_bc = self.broadcast_pos(pos, dec_seq, d)?;
        x = self.g().add(x, pos_bc);
        let mut kv = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let cross = if !cross_k.is_empty() {
                (Some(cross_k[i]), Some(cross_v[i]))
            } else {
                (None, None)
            };
            let (x_out, k, v) = self.residual_block_kv(
                cfg,
                i,
                x,
                encoder_hidden,
                cross.0,
                cross.1,
                dec_seq,
                cfg.decoder_attention_heads,
                MaskKind::Causal,
                None,
                None,
                None,
            )?;
            x = x_out;
            kv.push((k, v));
        }
        x = self.layer_norm_f32(x, &self.pfx.dec_ln_w(), &self.pfx.dec_ln_b())?;
        let logits = self.emit_logits(x, dec_seq, embed_w)?;
        Ok((logits, kv))
    }

    fn emit_decoder_step_inner(
        &mut self,
        cfg: &WhisperConfig,
        token_id: HirNodeId,
        pos_ix: Option<HirNodeId>,
        mask: Option<HirNodeId>,
        cross_k: &[HirNodeId],
        cross_v: &[HirNodeId],
        past_seq: usize,
        enc_seq: usize,
        past_k: &[HirNodeId],
        past_v: &[HirNodeId],
    ) -> Result<(HirNodeId, Vec<(HirNodeId, HirNodeId)>)> {
        let d = cfg.d_model;
        let embed_w = self.load_param(&self.pfx.dec_embed_tokens(), false)?;
        let mut x = self.g().gather_(embed_w, token_id, 0);
        let pos_w = self.load_param(&self.pfx.dec_embed_positions(), false)?;
        let pos = match pos_ix {
            Some(ix) => self.g().gather_(pos_w, ix, 0),
            None => self.g().narrow_(pos_w, 0, past_seq, 1),
        };
        let pos_bc = self.broadcast_pos(pos, 1, d)?;
        x = self.g().add(x, pos_bc);
        let mut kv = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let (x_out, k, v) = self.residual_block_kv(
                cfg,
                i,
                x,
                None,
                Some(cross_k[i]),
                Some(cross_v[i]),
                1,
                cfg.decoder_attention_heads,
                MaskKind::None,
                Some(past_k[i]),
                Some(past_v[i]),
                mask,
            )?;
            x = x_out;
            kv.push((k, v));
        }
        x = self.layer_norm_f32(x, &self.pfx.dec_ln_w(), &self.pfx.dec_ln_b())?;
        let logits = self.emit_logits(x, 1, embed_w)?;
        let _ = enc_seq;
        Ok((logits, kv))
    }

    fn residual_block_kv(
        &mut self,
        cfg: &WhisperConfig,
        layer: usize,
        x: HirNodeId,
        xa: Option<HirNodeId>,
        cross_k: Option<HirNodeId>,
        cross_v: Option<HirNodeId>,
        seq: usize,
        n_head: usize,
        self_mask: MaskKind,
        past_self_k: Option<HirNodeId>,
        past_self_v: Option<HirNodeId>,
        self_attn_mask: Option<HirNodeId>,
    ) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
        let d = cfg.d_model;
        let hd = d / n_head;
        let scale = (hd as f32).powf(-0.25);
        let layer_pfx = |suffix: &str| self.pfx.dec_layer(layer, suffix);
        let attn_ln_w = layer_pfx("self_attn_layer_norm.weight");
        let attn_ln_b = layer_pfx("self_attn_layer_norm.bias");
        let ln_x = self.layer_norm_f32(x, &attn_ln_w, &attn_ln_b)?;
        // Fused QKV matmul layout still diverges on bucketed decode; keep separate Q/K/V.
        let (sa, k_full, v_full) = self.mha_kv(
            ln_x,
            ln_x,
            &layer_pfx("self_attn.q_proj.weight"),
            Some(layer_pfx("self_attn.q_proj.bias").as_str()),
            &layer_pfx("self_attn.k_proj.weight"),
            None,
            &layer_pfx("self_attn.v_proj.weight"),
            layer_pfx("self_attn.v_proj.bias").as_str(),
            &layer_pfx("self_attn.out_proj.weight"),
            layer_pfx("self_attn.out_proj.bias").as_str(),
            seq,
            n_head,
            hd,
            scale,
            self_mask,
            past_self_k,
            past_self_v,
            self_attn_mask,
        )?;
        let mut x = self.g().add(x, sa);
        if let (Some(cross_k), Some(cross_v)) = (cross_k, cross_v) {
            let ca_ln_w = layer_pfx("encoder_attn_layer_norm.weight");
            let ca_ln_b = layer_pfx("encoder_attn_layer_norm.bias");
            let ln_x = self.layer_norm_f32(x, &ca_ln_w, &ca_ln_b)?;
            let ca = self.mha_cross_cached(
                ln_x,
                cross_k,
                cross_v,
                &layer_pfx("encoder_attn.q_proj.weight"),
                Some(layer_pfx("encoder_attn.q_proj.bias").as_str()),
                &layer_pfx("encoder_attn.out_proj.weight"),
                layer_pfx("encoder_attn.out_proj.bias").as_str(),
                seq,
                n_head,
                hd,
                scale,
            )?;
            x = self.g().add(x, ca);
        } else if let Some(xa) = xa {
            let ca_ln_w = layer_pfx("encoder_attn_layer_norm.weight");
            let ca_ln_b = layer_pfx("encoder_attn_layer_norm.bias");
            let ln_x = self.layer_norm_f32(x, &ca_ln_w, &ca_ln_b)?;
            let (ca, _, _) = self.mha_kv(
                ln_x,
                xa,
                &layer_pfx("encoder_attn.q_proj.weight"),
                Some(layer_pfx("encoder_attn.q_proj.bias").as_str()),
                &layer_pfx("encoder_attn.k_proj.weight"),
                None,
                &layer_pfx("encoder_attn.v_proj.weight"),
                layer_pfx("encoder_attn.v_proj.bias").as_str(),
                &layer_pfx("encoder_attn.out_proj.weight"),
                layer_pfx("encoder_attn.out_proj.bias").as_str(),
                seq,
                n_head,
                hd,
                scale,
                MaskKind::None,
                None,
                None,
                None,
            )?;
            x = self.g().add(x, ca);
        }
        let mlp_ln_w = layer_pfx("final_layer_norm.weight");
        let mlp_ln_b = layer_pfx("final_layer_norm.bias");
        let ln_x = self.layer_norm_f32(x, &mlp_ln_w, &mlp_ln_b)?;
        let mlp = self.mlp(
            ln_x,
            &layer_pfx("fc1.weight"),
            layer_pfx("fc1.bias").as_str(),
            &layer_pfx("fc2.weight"),
            layer_pfx("fc2.bias").as_str(),
            seq,
            d,
        )?;
        Ok((self.g().add(x, mlp), k_full, v_full))
    }

    fn emit_logits(
        &mut self,
        x: HirNodeId,
        _dec_seq: usize,
        embed_w: HirNodeId,
    ) -> Result<HirNodeId> {
        // Tied LM head: always use embed^T (same as HF); pre-fused logit matrix diverged on real weights.
        let embed_t = self.g().transpose_(embed_w, vec![1, 0]);
        let logits = self.g().mm(x, embed_t);
        Ok(if self.opts.use_f16_compute {
            self.g().cast(logits, DType::F32)
        } else {
            logits
        })
    }

    /// Fused QKV matmul + single-query attention (`seq == 1`, cached decode).
    #[allow(dead_code)]
    fn mha_kv_fused(
        &mut self,
        x: HirNodeId,
        layer: usize,
        q_seq: usize,
        n_head: usize,
        head_dim: usize,
        past_k: Option<HirNodeId>,
        past_v: Option<HirNodeId>,
        custom_mask: Option<HirNodeId>,
    ) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
        let fused = self.fused.expect("fused weights");
        let d = n_head * head_dim;
        let qkv =
            self.linear_fused_qkv(x, fused.qkv_w_key(layer), fused.qkv_b_key(layer), 3 * d)?;
        let q = self.g().narrow_(qkv, 2, 0, d);
        let k_new = self.g().narrow_(qkv, 2, d, d);
        let v_new = self.g().narrow_(qkv, 2, 2 * d, d);
        let (k, v) = match (past_k, past_v) {
            (Some(pk), Some(pv)) => (
                self.g().concat_(vec![pk, k_new], 1),
                self.g().concat_(vec![pv, v_new], 1),
            ),
            _ => (k_new, v_new),
        };
        let out_shape = Shape::new(&[self.batch, q_seq, d], self.f);
        let attn = if let Some(m) = custom_mask {
            self.whisper_attention_masked(q, k, v, m, n_head, head_dim, out_shape)
        } else {
            self.whisper_attention(q, k, v, n_head, head_dim, MaskKind::None, out_shape)
        };
        let layer_pfx = |suffix: &str| self.pfx.dec_layer(layer, suffix);
        let out = self.linear(
            attn,
            &layer_pfx("self_attn.out_proj.weight"),
            Some(layer_pfx("self_attn.out_proj.bias").as_str()),
            q_seq,
            d,
            d,
        )?;
        Ok((out, k, v))
    }

    fn mha_cross_cached(
        &mut self,
        x: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        qw: &str,
        qb: Option<&str>,
        ow: &str,
        ob: &str,
        q_seq: usize,
        n_head: usize,
        head_dim: usize,
        _scale: f32,
    ) -> Result<HirNodeId> {
        let chunk = self.opts.cross_attn_chunk;
        if chunk == 0 || q_seq <= 1 || q_seq <= chunk {
            let d = n_head * head_dim;
            let q = self.linear(x, qw, qb, q_seq, d, d)?;
            let out_shape = Shape::new(&[self.batch, q_seq, d], self.f);
            let attn = self.whisper_attention(q, k, v, n_head, head_dim, MaskKind::None, out_shape);
            return self.linear(attn, ow, Some(ob), q_seq, d, d);
        }
        let d = n_head * head_dim;
        let out_w = self.load_param(ow, true)?;
        let out_b = self.load_param(ob, false)?;
        let mut parts = Vec::new();
        let mut start = 0usize;
        while start < q_seq {
            let len = (q_seq - start).min(chunk);
            let q_slice = self.g().narrow_(x, 1, start, len);
            let q = self.linear(q_slice, qw, qb, len, d, d)?;
            let out_shape = Shape::new(&[self.batch, len, d], self.f);
            let attn = self.whisper_attention(q, k, v, n_head, head_dim, MaskKind::None, out_shape);
            parts.push(self.linear_mm(attn, out_w, Some(out_b), d)?);
            start += len;
        }
        Ok(self.g().concat_(parts, 1))
    }

    fn mha_kv(
        &mut self,
        x: HirNodeId,
        kv_src: HirNodeId,
        qw: &str,
        qb: Option<&str>,
        kw: &str,
        kb: Option<&str>,
        vw: &str,
        vb: &str,
        ow: &str,
        ob: &str,
        q_seq: usize,
        n_head: usize,
        head_dim: usize,
        _scale: f32,
        mask: MaskKind,
        past_k: Option<HirNodeId>,
        past_v: Option<HirNodeId>,
        custom_mask: Option<HirNodeId>,
    ) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
        let d = n_head * head_dim;
        let q = self.linear(x, qw, qb, q_seq, d, d)?;
        let k_new = self.linear(kv_src, kw, kb, self.kv_seq(kv_src), d, d)?;
        let v_new = self.linear(kv_src, vw, Some(vb), self.kv_seq(kv_src), d, d)?;
        let (k, v) = match (past_k, past_v) {
            (Some(pk), Some(pv)) => (
                self.g().concat_(vec![pk, k_new], 1),
                self.g().concat_(vec![pv, v_new], 1),
            ),
            _ => (k_new, v_new),
        };
        let out_shape = Shape::new(&[self.batch, q_seq, d], self.f);
        let attn = if let Some(m) = custom_mask {
            self.whisper_attention_masked(q, k, v, m, n_head, head_dim, out_shape)
        } else {
            self.whisper_attention(q, k, v, n_head, head_dim, mask, out_shape)
        };
        let out = self.linear(attn, ow, Some(ob), q_seq, d, d)?;
        Ok((out, k, v))
    }

    fn residual_block(
        &mut self,
        cfg: &WhisperConfig,
        layer: usize,
        x: HirNodeId,
        xa: Option<HirNodeId>,
        seq: usize,
        n_head: usize,
        self_mask: MaskKind,
        encoder: bool,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let hd = d / n_head;
        let scale = (hd as f32).powf(-0.25);
        let attn_ln_w = if encoder {
            self.pfx.enc_layer(layer, "self_attn_layer_norm.weight")
        } else {
            self.pfx.dec_layer(layer, "self_attn_layer_norm.weight")
        };
        let attn_ln_b = if encoder {
            self.pfx.enc_layer(layer, "self_attn_layer_norm.bias")
        } else {
            self.pfx.dec_layer(layer, "self_attn_layer_norm.bias")
        };
        let ln_x = self.layer_norm(x, &attn_ln_w, &attn_ln_b, seq, d)?;
        let layer_pfx = |suffix: &str| {
            if encoder {
                self.pfx.enc_layer(layer, suffix)
            } else {
                self.pfx.dec_layer(layer, suffix)
            }
        };
        let sa = if encoder {
            self.encoder_self_attn(
                ln_x,
                layer,
                &layer_pfx("self_attn.q_proj.weight"),
                Some(layer_pfx("self_attn.q_proj.bias").as_str()),
                &layer_pfx("self_attn.k_proj.weight"),
                None,
                &layer_pfx("self_attn.v_proj.weight"),
                layer_pfx("self_attn.v_proj.bias").as_str(),
                &layer_pfx("self_attn.out_proj.weight"),
                layer_pfx("self_attn.out_proj.bias").as_str(),
                seq,
                n_head,
                hd,
                scale,
                self_mask,
            )?
        } else {
            self.mha(
                ln_x,
                ln_x,
                &layer_pfx("self_attn.q_proj.weight"),
                Some(layer_pfx("self_attn.q_proj.bias").as_str()),
                &layer_pfx("self_attn.k_proj.weight"),
                None,
                &layer_pfx("self_attn.v_proj.weight"),
                layer_pfx("self_attn.v_proj.bias").as_str(),
                &layer_pfx("self_attn.out_proj.weight"),
                layer_pfx("self_attn.out_proj.bias").as_str(),
                seq,
                n_head,
                hd,
                scale,
                self_mask,
            )?
        };
        let mut x = self.g().add(x, sa);
        if let Some(xa) = xa {
            let ca_ln_w = layer_pfx("encoder_attn_layer_norm.weight");
            let ca_ln_b = layer_pfx("encoder_attn_layer_norm.bias");
            let ln_x = self.layer_norm(x, &ca_ln_w, &ca_ln_b, seq, d)?;
            let ca = self.mha(
                ln_x,
                xa,
                &layer_pfx("encoder_attn.q_proj.weight"),
                Some(layer_pfx("encoder_attn.q_proj.bias").as_str()),
                &layer_pfx("encoder_attn.k_proj.weight"),
                None,
                &layer_pfx("encoder_attn.v_proj.weight"),
                layer_pfx("encoder_attn.v_proj.bias").as_str(),
                &layer_pfx("encoder_attn.out_proj.weight"),
                layer_pfx("encoder_attn.out_proj.bias").as_str(),
                seq,
                n_head,
                hd,
                scale,
                MaskKind::None,
            )?;
            x = self.g().add(x, ca);
        }
        let mlp_ln_w = layer_pfx("final_layer_norm.weight");
        let mlp_ln_b = layer_pfx("final_layer_norm.bias");
        let ln_x = self.layer_norm(x, &mlp_ln_w, &mlp_ln_b, seq, d)?;
        let mlp = self.mlp(
            ln_x,
            &layer_pfx("fc1.weight"),
            layer_pfx("fc1.bias").as_str(),
            &layer_pfx("fc2.weight"),
            layer_pfx("fc2.bias").as_str(),
            seq,
            d,
        )?;
        Ok(self.g().add(x, mlp))
    }

    /// Encoder self-attention with optional Q-axis chunking (full-sequence K/V).
    fn encoder_self_attn(
        &mut self,
        x: HirNodeId,
        layer: usize,
        qw: &str,
        qb: Option<&str>,
        kw: &str,
        kb: Option<&str>,
        vw: &str,
        vb: &str,
        ow: &str,
        ob: &str,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        mask: MaskKind,
    ) -> Result<HirNodeId> {
        let chunk = self.opts.encoder_attn_chunk;
        if false {
            if let Some(fused) = self.fused_enc {
                return self.encoder_self_attn_fused(
                    fused, layer, x, ow, ob, seq, n_head, head_dim, scale, mask, chunk,
                );
            }
        }
        if chunk == 0 || seq <= chunk {
            return self.mha(
                x, x, qw, qb, kw, kb, vw, vb, ow, ob, seq, n_head, head_dim, scale, mask,
            );
        }
        let d = n_head * head_dim;
        let out_w = self.load_param(ow, true)?;
        let out_b = self.load_param(ob, false)?;
        let qw_w = self.load_param(qw, true)?;
        let qb_n = match qb {
            Some(k) => Some(self.load_param(k, false)?),
            None => None,
        };
        let kw_w = self.load_param(kw, true)?;
        let kb_n = match kb {
            Some(k) => Some(self.load_param(k, false)?),
            None => None,
        };
        let vw_w = self.load_param(vw, true)?;
        let vb_n = self.load_param(vb, false)?;
        let _kv_len = self.kv_seq(x);
        let k = self.linear_mm(x, kw_w, kb_n, d)?;
        let v = self.linear_mm(x, vw_w, Some(vb_n), d)?;
        let mut parts = Vec::new();
        let mut start = 0usize;
        while start < seq {
            let len = (seq - start).min(chunk);
            let q_slice = self.g().narrow_(x, 1, start, len);
            let q = self.linear_mm(q_slice, qw_w, qb_n, d)?;
            let out_shape = Shape::new(&[self.batch, len, d], self.f);
            let attn = self.whisper_attention(q, k, v, n_head, head_dim, mask, out_shape);
            parts.push(self.linear_mm(attn, out_w, Some(out_b), d)?);
            start += len;
        }
        Ok(self.g().concat_(parts, 1))
    }

    /// Fused QKV matmul for encoder self-attention (optional Q chunking, shared K/V).
    fn encoder_self_attn_fused(
        &mut self,
        fused: &FusedEncoderWeights,
        layer: usize,
        x: HirNodeId,
        ow: &str,
        ob: &str,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        _scale: f32,
        mask: MaskKind,
        chunk: usize,
    ) -> Result<HirNodeId> {
        let d = n_head * head_dim;
        let qkv =
            self.linear_fused_qkv(x, fused.qkv_w_key(layer), fused.qkv_b_key(layer), 3 * d)?;
        let k = self.g().narrow_(qkv, 2, d, d);
        let v = self.g().narrow_(qkv, 2, 2 * d, d);
        if chunk == 0 || seq <= chunk {
            let q = self.g().narrow_(qkv, 2, 0, d);
            let out_shape = Shape::new(&[self.batch, seq, d], self.f);
            let attn = self.whisper_attention(q, k, v, n_head, head_dim, mask, out_shape);
            return self.linear(attn, ow, Some(ob), seq, d, d);
        }
        let out_w = self.load_param(ow, true)?;
        let out_b = self.load_param(ob, false)?;
        let mut parts = Vec::new();
        let mut start = 0usize;
        while start < seq {
            let len = (seq - start).min(chunk);
            let q_chunk = self.g().narrow_(qkv, 1, start, len);
            let q = self.g().narrow_(q_chunk, 2, 0, d);
            let out_shape = Shape::new(&[self.batch, len, d], self.f);
            let attn = self.whisper_attention(q, k, v, n_head, head_dim, mask, out_shape);
            parts.push(self.linear_mm(attn, out_w, Some(out_b), d)?);
            start += len;
        }
        Ok(self.g().concat_(parts, 1))
    }

    fn mha(
        &mut self,
        x: HirNodeId,
        kv_src: HirNodeId,
        qw: &str,
        qb: Option<&str>,
        kw: &str,
        kb: Option<&str>,
        vw: &str,
        vb: &str,
        ow: &str,
        ob: &str,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        _scale: f32,
        mask: MaskKind,
    ) -> Result<HirNodeId> {
        let out_w = self.load_param(ow, true)?;
        let out_b = self.load_param(ob, false)?;
        self.mha_with_out_proj(
            x, kv_src, qw, qb, kw, kb, vw, vb, out_w, out_b, seq, n_head, head_dim, mask,
        )
    }

    /// Self-attention with QKV linears; output projection weights loaded once (chunked encoder).
    fn mha_with_out_proj(
        &mut self,
        x: HirNodeId,
        kv_src: HirNodeId,
        qw: &str,
        qb: Option<&str>,
        kw: &str,
        kb: Option<&str>,
        vw: &str,
        vb: &str,
        out_w: HirNodeId,
        out_b: HirNodeId,
        seq: usize,
        n_head: usize,
        head_dim: usize,
        mask: MaskKind,
    ) -> Result<HirNodeId> {
        let d = n_head * head_dim;
        let q = self.linear(x, qw, qb, seq, d, d)?;
        let k = self.linear(kv_src, kw, kb, self.kv_seq(kv_src), d, d)?;
        let v = self.linear(kv_src, vw, Some(vb), self.kv_seq(kv_src), d, d)?;
        let out_shape = Shape::new(&[self.batch, seq, d], self.f);
        let attn = self.whisper_attention(q, k, v, n_head, head_dim, mask, out_shape);
        self.linear_mm(attn, out_w, Some(out_b), d)
    }

    fn kv_seq(&self, x: HirNodeId) -> usize {
        self.hir.node(x).shape.dim(1).unwrap_static()
    }

    fn mlp(
        &mut self,
        x: HirNodeId,
        w1: &str,
        b1: &str,
        w2: &str,
        b2: &str,
        seq: usize,
        d: usize,
    ) -> Result<HirNodeId> {
        let mlp_dim = d * 4;
        let h1 = self.linear(x, w1, Some(b1), seq, mlp_dim, d)?;
        let h1 = self.g().gelu(h1);
        self.linear(h1, w2, Some(b2), seq, d, mlp_dim)
    }

    fn linear(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
        _seq: usize,
        out_f: usize,
        _in_f: usize,
    ) -> Result<HirNodeId> {
        self.linear_fused(x, w_key, b_key, _seq, out_f, _in_f)
    }

    fn linear_fused(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
        _seq: usize,
        out_f: usize,
        _in_f: usize,
    ) -> Result<HirNodeId> {
        let w = self.load_param(w_key, true)?;
        let mut y = self.g().mm(x, w);
        if let Some(bk) = b_key {
            let b = self.load_param(bk, false)?;
            let b3 = self.g().reshape_(b, vec![1, 1, out_f as i64]);
            y = self.g().add(y, b3);
        }
        Ok(y)
    }

    /// Matmul with already-loaded weight/bias param nodes (reuse across Q chunks).
    fn linear_mm(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        b: Option<HirNodeId>,
        out_f: usize,
    ) -> Result<HirNodeId> {
        let mut y = self.g().mm(x, w);
        if let Some(b) = b {
            let b3 = self.g().reshape_(b, vec![1, 1, out_f as i64]);
            y = self.g().add(y, b3);
        }
        Ok(y)
    }

    /// Fused QKV weight is already `[d_model, 3*d_model]` for `x @ w` (no transpose).
    fn linear_fused_qkv(
        &mut self,
        x: HirNodeId,
        w_key: &str,
        b_key: Option<&str>,
        out_f: usize,
    ) -> Result<HirNodeId> {
        let w = self.load_param(w_key, false)?;
        let mut y = self.g().mm(x, w);
        if let Some(bk) = b_key {
            let b = self.load_param(bk, false)?;
            let b3 = self.g().reshape_(b, vec![1, 1, out_f as i64]);
            y = self.g().add(y, b3);
        }
        Ok(y)
    }

    fn layer_norm(
        &mut self,
        x: HirNodeId,
        w: &str,
        b: &str,
        _seq: usize,
        _d: usize,
    ) -> Result<HirNodeId> {
        let gamma = self.load_param(w, false)?;
        let beta = self.load_param(b, false)?;
        Ok(self.g().ln(x, gamma, beta, LN_EPS))
    }

    fn broadcast_conv_bias(
        &mut self,
        bias: HirNodeId,
        out_c: usize,
        t_out: usize,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let bias3 = self.g().reshape_(bias, vec![1, out_c as i64, 1]);
        let ones = self.register_param(
            &format!("conv_bias_batch_{out_c}_{t_out}"),
            vec![1.0; batch],
            &[batch],
        )?;
        let ones2 = self.g().reshape_(ones, vec![batch as i64, 1, 1]);
        let bias_bc = self.g().mul(bias3, ones2);
        let time = self.register_param(
            &format!("conv_bias_time_{t_out}"),
            vec![1.0; t_out],
            &[t_out],
        )?;
        let time3 = self.g().reshape_(time, vec![1, 1, t_out as i64]);
        Ok(self.g().mul(bias_bc, time3))
    }

    fn conv1d_gelu(
        &mut self,
        input: HirNodeId,
        w_key: &str,
        b_key: &str,
        in_c: usize,
        out_c: usize,
        t_in: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Result<HirNodeId> {
        let batch = self.batch;
        let f = self.f;
        let t_out = (t_in + 2 * pad - k) / stride + 1;
        let nchw = self
            .g()
            .reshape_(input, vec![batch as i64, in_c as i64, t_in as i64, 1]);
        let (w_data, _) = self.weights.take(w_key, false)?;
        let w = self.register_param(
            w_key,
            pack_conv1d_weight(&w_data, out_c, in_c, k),
            &[out_c, in_c, k, 1],
        )?;
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
        let (b_data, _) = self.weights.take(b_key, false)?;
        let bias = self.register_param(b_key, b_data, &[out_c])?;
        let b3 = self.broadcast_conv_bias(bias, out_c, t_out)?;
        out = self.g().add(out, b3);
        let bcs = out;
        Ok(self.g().gelu(bcs))
    }

    fn encoder_sinusoids(&mut self, length: usize, channels: usize) -> Result<HirNodeId> {
        let data = sinusoids(length, channels);
        let name = "encoder.positional_embedding";
        self.register_param(name, data, &[length, channels])
    }

    fn broadcast_pos(&mut self, pos: HirNodeId, seq: usize, d: usize) -> Result<HirNodeId> {
        let batch = self.batch;
        let pos3 = self.g().reshape_(pos, vec![1, seq as i64, d as i64]);
        let ones =
            self.register_param(&format!("pos_broadcast_{seq}"), vec![1.0; batch], &[batch])?;
        let ones2 = self.g().reshape_(ones, vec![batch as i64, 1, 1]);
        Ok(self.g().mul(pos3, ones2))
    }

    fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.weights.take(key, transpose)?;
        let id = self.hir.param(key, Shape::new(&shape, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }

    fn register_param(
        &mut self,
        key: &str,
        data: Vec<f32>,
        shape_dims: &[usize],
    ) -> Result<HirNodeId> {
        let id = self.hir.param(key, Shape::new(shape_dims, self.f));
        self.params.insert(key.to_string(), data);
        Ok(id)
    }
}

fn pack_conv1d_weight(raw: &[f32], _out_c: usize, _in_c: usize, _k: usize) -> Vec<f32> {
    // PyTorch Conv1d `[out, in, k]` → NCHW `[out, in, k, 1]`.
    raw.to_vec()
}

fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let max_timescale = 10000f32;
    let log_step = max_timescale.ln() / (channels / 2 - 1) as f32;
    let mut out = vec![0f32; length * channels];
    for pos in 0..length {
        for i in 0..channels / 2 {
            let inv = (i as f32 * -log_step).exp();
            let angle = pos as f32 * inv;
            out[pos * channels + i] = angle.sin();
            out[pos * channels + channels / 2 + i] = angle.cos();
        }
    }
    out
}

fn validate_cfg(cfg: &WhisperConfig) -> Result<()> {
    ensure!(
        cfg.d_model.is_multiple_of(cfg.encoder_attention_heads),
        "encoder head dim"
    );
    ensure!(
        cfg.d_model.is_multiple_of(cfg.decoder_attention_heads),
        "decoder head dim"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn synth_weights(cfg: &WhisperConfig, pfx: &WhisperWeightPrefix) -> WeightMap {
        let d = cfg.d_model;
        let m = cfg.num_mel_bins;
        let v = cfg.vocab_size;
        let mlp = d * 4;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.01f32; n];
        t.insert(pfx.enc_conv1_w(), (z(d * m * 3), vec![d, m, 3]));
        t.insert(pfx.enc_conv1_b(), (z(d), vec![d]));
        t.insert(pfx.enc_conv2_w(), (z(d * d * 3), vec![d, d, 3]));
        t.insert(pfx.enc_conv2_b(), (z(d), vec![d]));
        t.insert(pfx.enc_ln_post_w(), (z(d), vec![d]));
        t.insert(pfx.enc_ln_post_b(), (z(d), vec![d]));
        for i in 0..cfg.encoder_layers {
            for name in ["self_attn.q_proj", "self_attn.out_proj", "self_attn.v_proj"] {
                t.insert(
                    pfx.enc_layer(i, &format!("{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
                t.insert(pfx.enc_layer(i, &format!("{name}.bias")), (z(d), vec![d]));
            }
            t.insert(
                pfx.enc_layer(i, "self_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(pfx.enc_layer(i, "fc1.weight"), (z(mlp * d), vec![mlp, d]));
            t.insert(pfx.enc_layer(i, "fc1.bias"), (z(mlp), vec![mlp]));
            t.insert(pfx.enc_layer(i, "fc2.weight"), (z(d * mlp), vec![d, mlp]));
            t.insert(pfx.enc_layer(i, "fc2.bias"), (z(d), vec![d]));
            for n in ["self_attn_layer_norm", "final_layer_norm"] {
                t.insert(pfx.enc_layer(i, &format!("{n}.weight")), (z(d), vec![d]));
                t.insert(pfx.enc_layer(i, &format!("{n}.bias")), (z(d), vec![d]));
            }
        }
        t.insert(pfx.dec_embed_tokens(), (z(v * d), vec![v, d]));
        t.insert(
            pfx.dec_embed_positions(),
            (
                z(cfg.max_target_positions * d),
                vec![cfg.max_target_positions, d],
            ),
        );
        t.insert(pfx.dec_ln_w(), (z(d), vec![d]));
        t.insert(pfx.dec_ln_b(), (z(d), vec![d]));
        for i in 0..cfg.decoder_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.out_proj",
                "self_attn.v_proj",
                "encoder_attn.q_proj",
                "encoder_attn.out_proj",
                "encoder_attn.v_proj",
            ] {
                t.insert(
                    pfx.dec_layer(i, &format!("{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
                t.insert(pfx.dec_layer(i, &format!("{name}.bias")), (z(d), vec![d]));
            }
            t.insert(
                pfx.dec_layer(i, "self_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(
                pfx.dec_layer(i, "encoder_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(pfx.dec_layer(i, "fc1.weight"), (z(mlp * d), vec![mlp, d]));
            t.insert(pfx.dec_layer(i, "fc1.bias"), (z(mlp), vec![mlp]));
            t.insert(pfx.dec_layer(i, "fc2.weight"), (z(d * mlp), vec![d, mlp]));
            t.insert(pfx.dec_layer(i, "fc2.bias"), (z(d), vec![d]));
            for n in [
                "self_attn_layer_norm",
                "encoder_attn_layer_norm",
                "final_layer_norm",
            ] {
                t.insert(pfx.dec_layer(i, &format!("{n}.weight")), (z(d), vec![d]));
                t.insert(pfx.dec_layer(i, &format!("{n}.bias")), (z(d), vec![d]));
            }
        }
        WeightMap::from_tensors(t)
    }

    #[test]
    fn whisper_encoder_hir_builds() {
        use rlx_core::flow_util::WeightMapSource;
        let cfg = WhisperConfig::tiny_synthetic();
        let mel_frames = 8;
        let pfx = WhisperWeightPrefix {
            encoder: "model.encoder".into(),
            decoder: "model.decoder".into(),
            hf_embed_names: true,
        };
        let mut wm = synth_weights(&cfg, &pfx);
        let mut src = WeightMapSource(&mut wm);
        let (hir, _params) =
            build_whisper_encoder_hir(&cfg, &mut src, &pfx, 1, mel_frames).unwrap();
        assert_eq!(hir.outputs.len(), 1);
    }
}
