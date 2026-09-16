// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Moonshine encoder: raw PCM → conv frontend → RoPE transformer layers.

use crate::builder::{ConvAct, MoonshineBuilder};
use crate::config::MoonshineConfig;
use anyhow::{Result, bail};
use rlx_ir::hir::{HirGraphExt, HirNodeId};
use rlx_ir::op::MaskKind;

impl MoonshineBuilder<'_> {
    /// Full encoder: `pcm [B, L]` → `encoder_hidden [B, T, d]`.
    pub(crate) fn emit_encoder(
        &mut self,
        cfg: &MoonshineConfig,
        pcm: HirNodeId,
        audio_len: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.hidden_size;
        let enc_seq = MoonshineConfig::feat_extract_output_length(audio_len);
        if enc_seq == 0 {
            bail!(
                "moonshine: audio length {audio_len} too short for conv frontend (need ≥895 samples @ 16 kHz)"
            );
        }

        // [B, L] → [B, 1, L] for Conv1d(in=1)
        let batch = self.batch as i64;
        let x = self.g().reshape_(pcm, vec![batch, 1, audio_len as i64]);

        // conv1 (no bias) → tanh
        let (mut x, t1) = self.conv1d(
            x,
            &self.pfx.enc_conv1_w(),
            None,
            1,
            d,
            audio_len,
            127,
            64,
            0,
            ConvAct::Tanh,
        )?;
        // groupnorm
        x = self.group_norm_ct(
            x,
            d,
            t1,
            &self.pfx.enc_groupnorm_w(),
            &self.pfx.enc_groupnorm_b(),
        )?;
        // conv2 → gelu
        let (x, t2) = self.conv1d(
            x,
            &self.pfx.enc_conv2_w(),
            Some(&self.pfx.enc_conv2_b()),
            d,
            2 * d,
            t1,
            7,
            3,
            0,
            ConvAct::Gelu,
        )?;
        // conv3 → gelu
        let (x, t3) = self.conv1d(
            x,
            &self.pfx.enc_conv3_w(),
            Some(&self.pfx.enc_conv3_b()),
            2 * d,
            d,
            t2,
            3,
            2,
            0,
            ConvAct::Gelu,
        )?;
        debug_assert_eq!(t3, enc_seq);

        // [B, C, T] → [B, T, C]
        let mut x = self.g().transpose_(x, vec![0, 2, 1]);

        let hd = cfg.enc_head_dim();
        let nh = cfg.encoder_num_attention_heads;
        let (cos, sin, n_rot) = self.rope_tables(cfg, enc_seq, hd, "enc")?;

        for layer in 0..cfg.encoder_num_hidden_layers {
            x = self.encoder_layer(cfg, layer, x, enc_seq, nh, hd, n_rot, cos, sin)?;
        }
        self.layer_norm_nobias(x, &self.pfx.enc_ln_w(), d)
    }

    #[allow(clippy::too_many_arguments)]
    fn encoder_layer(
        &mut self,
        cfg: &MoonshineConfig,
        layer: usize,
        x: HirNodeId,
        seq: usize,
        nh: usize,
        hd: usize,
        n_rot: usize,
        cos: HirNodeId,
        sin: HirNodeId,
    ) -> Result<HirNodeId> {
        let d = cfg.hidden_size;
        let p = |s: &str| self.pfx.enc_layer(layer, s);

        let residual = x;
        let mut h = self.layer_norm_nobias(x, &p("input_layernorm.weight"), d)?;
        h = self.self_attn_rope(
            h,
            &p("self_attn"),
            seq,
            nh,
            hd,
            n_rot,
            cos,
            sin,
            MaskKind::None,
            cfg.attention_bias,
            cfg.pad_head_dim_to_multiple_of,
        )?;
        let x = self.g().add(residual, h);

        let residual = x;
        let h = self.layer_norm_nobias(x, &p("post_attention_layernorm.weight"), d)?;
        let h = self.encoder_mlp(h, layer)?;
        Ok(self.g().add(residual, h))
    }
}
