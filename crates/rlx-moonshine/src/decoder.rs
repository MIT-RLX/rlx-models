// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Moonshine decoder: causal self-attn (RoPE) + cross-attn to encoder + SwiGLU MLP.

use crate::builder::MoonshineBuilder;
use crate::config::MoonshineConfig;
use anyhow::Result;
use rlx_ir::hir::{HirGraphExt, HirNodeId};
use rlx_ir::op::MaskKind;

impl MoonshineBuilder<'_> {
    /// Decoder hidden states from host token embeds + encoder hidden.
    ///
    /// Inputs: `decoder_inputs_embeds [B, T, d]`, `encoder_hidden [B, enc_seq, d]`.
    /// Output: post-norm hidden `[B, T, d]` (LM head applied host-side).
    pub(crate) fn emit_decoder_hidden(
        &mut self,
        cfg: &MoonshineConfig,
        inputs_embeds: HirNodeId,
        encoder_hidden: HirNodeId,
        dec_seq: usize,
        enc_seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.hidden_size;
        let hd = cfg.dec_head_dim();
        let nh = cfg.decoder_num_attention_heads;
        let (cos, sin, n_rot) = self.rope_tables(cfg, dec_seq, hd, "dec")?;

        let mut x = inputs_embeds;
        for layer in 0..cfg.decoder_num_hidden_layers {
            x = self.decoder_layer(
                cfg,
                layer,
                x,
                encoder_hidden,
                dec_seq,
                enc_seq,
                nh,
                hd,
                n_rot,
                cos,
                sin,
            )?;
        }
        self.layer_norm_nobias(x, &self.pfx.dec_norm_w(), d)
    }

    #[allow(clippy::too_many_arguments)]
    fn decoder_layer(
        &mut self,
        cfg: &MoonshineConfig,
        layer: usize,
        x: HirNodeId,
        encoder_hidden: HirNodeId,
        seq: usize,
        enc_seq: usize,
        nh: usize,
        hd: usize,
        n_rot: usize,
        cos: HirNodeId,
        sin: HirNodeId,
    ) -> Result<HirNodeId> {
        let d = cfg.hidden_size;
        let inter = cfg.decoder_intermediate_size;
        let p = |s: &str| self.pfx.dec_layer(layer, s);

        // Causal self-attn (RoPE)
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
            MaskKind::Causal,
            cfg.attention_bias,
            cfg.pad_head_dim_to_multiple_of,
        )?;
        let mut x = self.g().add(residual, h);

        // Cross-attn to encoder (no RoPE)
        let residual = x;
        h = self.layer_norm_nobias(x, &p("post_attention_layernorm.weight"), d)?;
        h = self.cross_attn(
            h,
            encoder_hidden,
            &p("encoder_attn"),
            seq,
            enc_seq,
            nh,
            hd,
            cfg.attention_bias,
            cfg.pad_head_dim_to_multiple_of,
        )?;
        x = self.g().add(residual, h);

        // Decoder MLP
        let residual = x;
        h = self.layer_norm_nobias(x, &p("final_layernorm.weight"), d)?;
        h = self.decoder_mlp(h, layer, inter)?;
        Ok(self.g().add(residual, h))
    }
}
