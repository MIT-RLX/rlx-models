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

//! NLLB / M2M100 encoder + decoder graphs (post-norm, learned positions with
//! offset 2, `layernorm_embedding`, ReLU FFN, tied shared embedding LM head).

use crate::builder::NllbBuilder;
use crate::config::NllbConfig;
use crate::weights::lang as lk;
use anyhow::Result;
use rlx_ir::hir::{HirGraphExt, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{Shape, ops::attention::attention_kind_op};

impl NllbBuilder<'_> {
    /// Encoder over precomputed `inputs_embeds [B, S, d]`. Returns
    /// `encoder_hidden [B, S, d]`.
    pub(crate) fn emit_encoder(
        &mut self,
        cfg: &NllbConfig,
        inputs_embeds: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let pos = self.positional_embeddings(cfg, &lk::enc_embed_positions(), 0, seq, d)?;
        let mut x = self.g().add(inputs_embeds, pos);
        if self.weights.has(&lk::enc_layernorm_embedding_w()) {
            x = self.layer_norm(
                x,
                &lk::enc_layernorm_embedding_w(),
                &lk::enc_layernorm_embedding_b(),
            )?;
        }
        for layer in 0..cfg.encoder_layers {
            x = self.encoder_layer(cfg, layer, x, seq)?;
        }
        // Optional final encoder LayerNorm (`model.encoder.layer_norm`).
        if self.weights.has(&lk::enc_final_layer_norm_w()) {
            x = self.layer_norm(
                x,
                &lk::enc_final_layer_norm_w(),
                &lk::enc_final_layer_norm_b(),
            )?;
        }
        Ok(x)
    }

    fn encoder_layer(
        &mut self,
        cfg: &NllbConfig,
        layer: usize,
        mut x: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let nh = cfg.encoder_attention_heads;
        let hd = cfg.enc_head_dim();
        let p = |s: &str| lk::enc_layer(layer, s);

        if self.m2m100_style() {
            let residual = x;
            let mut h = self.layer_norm(
                x,
                &p("self_attn_layer_norm.weight"),
                &p("self_attn_layer_norm.bias"),
            )?;
            let sa =
                self.self_attention(h, h, &p("self_attn"), seq, seq, nh, hd, MaskKind::None)?;
            x = self.g().add(residual, sa);
            let residual = x;
            h = self.layer_norm(
                x,
                &p("final_layer_norm.weight"),
                &p("final_layer_norm.bias"),
            )?;
            let ff = self.ffn(
                h,
                &p("fc1.weight"),
                &p("fc1.bias"),
                &p("fc2.weight"),
                &p("fc2.bias"),
            )?;
            return Ok(self.g().add(residual, ff));
        }

        let residual = x;
        let sa = self.self_attention(x, x, &p("self_attn"), seq, seq, nh, hd, MaskKind::None)?;
        let mut x = self.g().add(residual, sa);
        x = self.layer_norm(
            x,
            &p("self_attn_layer_norm.weight"),
            &p("self_attn_layer_norm.bias"),
        )?;

        let residual = x;
        let ff = self.ffn(
            x,
            &p("fc1.weight"),
            &p("fc1.bias"),
            &p("fc2.weight"),
            &p("fc2.bias"),
        )?;
        x = self.g().add(residual, ff);
        self.layer_norm(
            x,
            &p("final_layer_norm.weight"),
            &p("final_layer_norm.bias"),
        )
    }

    /// Decoder hidden states (no embed lookup, no LM head). Takes precomputed
    /// `decoder_inputs_embeds [B,dec_seq,d]` plus `encoder_hidden`, and returns
    /// the post-layer hidden `[B,dec_seq,d]`. The LM head is applied host-side.
    pub(crate) fn emit_decoder_hidden(
        &mut self,
        cfg: &NllbConfig,
        inputs_embeds: HirNodeId,
        encoder_hidden: HirNodeId,
        dec_seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let pos = self.positional_embeddings(cfg, &lk::dec_embed_positions(), 0, dec_seq, d)?;
        let mut x = self.g().add(inputs_embeds, pos);
        if self.weights.has(&lk::dec_layernorm_embedding_w()) {
            x = self.layer_norm(
                x,
                &lk::dec_layernorm_embedding_w(),
                &lk::dec_layernorm_embedding_b(),
            )?;
        }
        for layer in 0..cfg.decoder_layers {
            x = self.decoder_layer(cfg, layer, x, encoder_hidden, dec_seq)?;
        }
        if self.weights.has(&lk::dec_final_layer_norm_w()) {
            x = self.layer_norm(
                x,
                &lk::dec_final_layer_norm_w(),
                &lk::dec_final_layer_norm_b(),
            )?;
        }
        Ok(x)
    }

    fn decoder_layer(
        &mut self,
        cfg: &NllbConfig,
        layer: usize,
        mut x: HirNodeId,
        encoder_hidden: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let nh = cfg.decoder_attention_heads;
        let hd = cfg.dec_head_dim();
        let enc_seq = self.kv_seq(encoder_hidden);
        let p = |s: &str| lk::dec_layer(layer, s);

        if self.m2m100_style() {
            let residual = x;
            let mut h = self.layer_norm(
                x,
                &p("self_attn_layer_norm.weight"),
                &p("self_attn_layer_norm.bias"),
            )?;
            let sa =
                self.self_attention(h, h, &p("self_attn"), seq, seq, nh, hd, MaskKind::Causal)?;
            x = self.g().add(residual, sa);

            let residual = x;
            h = self.layer_norm(
                x,
                &p("encoder_attn_layer_norm.weight"),
                &p("encoder_attn_layer_norm.bias"),
            )?;
            let ca =
                self.cross_attention(h, encoder_hidden, &p("encoder_attn"), seq, enc_seq, nh, hd)?;
            x = self.g().add(residual, ca);

            let residual = x;
            h = self.layer_norm(
                x,
                &p("final_layer_norm.weight"),
                &p("final_layer_norm.bias"),
            )?;
            let ff = self.ffn(
                h,
                &p("fc1.weight"),
                &p("fc1.bias"),
                &p("fc2.weight"),
                &p("fc2.bias"),
            )?;
            return Ok(self.g().add(residual, ff));
        }

        // Self attention (causal).
        let residual = x;
        let sa = self.self_attention(x, x, &p("self_attn"), seq, seq, nh, hd, MaskKind::Causal)?;
        let mut x = self.g().add(residual, sa);
        x = self.layer_norm(
            x,
            &p("self_attn_layer_norm.weight"),
            &p("self_attn_layer_norm.bias"),
        )?;

        // Cross attention to encoder.
        let residual = x;
        let ca =
            self.cross_attention(x, encoder_hidden, &p("encoder_attn"), seq, enc_seq, nh, hd)?;
        x = self.g().add(residual, ca);
        x = self.layer_norm(
            x,
            &p("encoder_attn_layer_norm.weight"),
            &p("encoder_attn_layer_norm.bias"),
        )?;

        // FFN (ReLU).
        let residual = x;
        let ff = self.ffn(
            x,
            &p("fc1.weight"),
            &p("fc1.bias"),
            &p("fc2.weight"),
            &p("fc2.bias"),
        )?;
        x = self.g().add(residual, ff);
        self.layer_norm(
            x,
            &p("final_layer_norm.weight"),
            &p("final_layer_norm.bias"),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cross_attention(
        &mut self,
        q_src: HirNodeId,
        enc: HirNodeId,
        pfx: &str,
        q_seq: usize,
        enc_seq: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Result<HirNodeId> {
        self.self_attention(
            q_src,
            enc,
            pfx,
            q_seq,
            enc_seq,
            n_head,
            head_dim,
            MaskKind::None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn self_attention(
        &mut self,
        q_src: HirNodeId,
        kv_src: HirNodeId,
        pfx: &str,
        q_seq: usize,
        kv_seq: usize,
        n_head: usize,
        head_dim: usize,
        mask: MaskKind,
    ) -> Result<HirNodeId> {
        let d = n_head * head_dim;
        let q = self.linear(
            q_src,
            &format!("{pfx}.q_proj.weight"),
            Some(&format!("{pfx}.q_proj.bias")),
        )?;
        let k = self.linear(
            kv_src,
            &format!("{pfx}.k_proj.weight"),
            Some(&format!("{pfx}.k_proj.bias")),
        )?;
        let v = self.linear(
            kv_src,
            &format!("{pfx}.v_proj.weight"),
            Some(&format!("{pfx}.v_proj.bias")),
        )?;
        let out_shape = Shape::new(&[self.batch, q_seq, d], self.f);
        let _ = kv_seq;
        let attn = self.g().add_node(
            attention_kind_op(n_head, head_dim, None, mask, None, None),
            vec![q, k, v],
            out_shape,
        );
        self.linear(
            attn,
            &format!("{pfx}.out_proj.weight"),
            Some(&format!("{pfx}.out_proj.bias")),
        )
    }

    /// M2M100 / NLLB FFN uses **ReLU** (not GELU).
    fn ffn(&mut self, x: HirNodeId, w1: &str, b1: &str, w2: &str, b2: &str) -> Result<HirNodeId> {
        let h = self.linear(x, w1, Some(b1))?;
        let h = self.g().relu(h);
        self.linear(h, w2, Some(b2))
    }

    /// Positional embeddings: learned table when present, else M2M100 sinusoidal.
    fn positional_embeddings(
        &mut self,
        cfg: &NllbConfig,
        key: &str,
        past: usize,
        seq: usize,
        d: usize,
    ) -> Result<HirNodeId> {
        if self.weights.has(key) {
            return self.learned_positions(key, past, seq, d);
        }
        let num_embed = cfg.max_position_embeddings + NllbConfig::POS_OFFSET;
        let table =
            NllbConfig::m2m100_sinusoidal_embedding(num_embed, d, cfg.pad_token_id as usize);
        let start = cfg.pad_token_id as usize + 1 + past;
        let slice: Vec<f32> = table[start * d..(start + seq) * d].to_vec();
        let param_key = format!("@sin_pos_{key}_{past}_{seq}");
        let id = self.hir.param(&param_key, Shape::new(&[1, seq, d], self.f));
        self.params.insert(param_key, slice);
        Ok(id)
    }

    /// Learned positional embeddings with BART/M2M100 offset of 2.
    fn learned_positions(
        &mut self,
        key: &str,
        past: usize,
        seq: usize,
        d: usize,
    ) -> Result<HirNodeId> {
        let pos_w = self.load_param(key, false)?;
        let start = NllbConfig::POS_OFFSET + past;
        let rows = self.g().narrow_(pos_w, 0, start, seq);
        Ok(self.g().reshape_(rows, vec![1, seq as i64, d as i64]))
    }

    pub(crate) fn kv_seq(&self, x: HirNodeId) -> usize {
        self.hir.node(x).shape.dim(1).unwrap_static()
    }
}
