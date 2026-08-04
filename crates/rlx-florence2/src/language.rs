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

//! Florence-2 BART encoder + decoder graphs (post-norm, learned positions
//! with offset 2, `layernorm_embedding`, tied LM head + `final_logits_bias`).

use crate::builder::Florence2Builder;
use crate::config::{Florence2Config, Florence2TextConfig};
use crate::weights::lang as lk;
use anyhow::Result;
use rlx_ir::hir::{HirGraphExt, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{Shape, ops::attention::attention_kind_op};

impl Florence2Builder<'_> {
    /// BART encoder over precomputed `inputs_embeds [B, S, d]` (image ++ text
    /// prompt embeddings). Returns `encoder_hidden [B, S, d]`.
    pub(crate) fn emit_encoder(
        &mut self,
        cfg: &Florence2Config,
        inputs_embeds: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let t = &cfg.text;
        let d = t.d_model;
        let pos = self.learned_positions(&lk::enc_embed_positions(), 0, seq, d)?;
        let mut x = self.g().add(inputs_embeds, pos);
        x = self.layer_norm(
            x,
            &lk::enc_layernorm_embedding_w(),
            &lk::enc_layernorm_embedding_b(),
        )?;
        for layer in 0..t.encoder_layers {
            x = self.encoder_layer(t, layer, x, seq)?;
        }
        Ok(x)
    }

    fn encoder_layer(
        &mut self,
        t: &Florence2TextConfig,
        layer: usize,
        x: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let d = t.d_model;
        let nh = t.encoder_attention_heads;
        let hd = t.enc_head_dim();
        let p = |s: &str| lk::enc_layer(layer, s);

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
        let _ = d;
        self.layer_norm(
            x,
            &p("final_layer_norm.weight"),
            &p("final_layer_norm.bias"),
        )
    }

    /// Decoder over the full target sequence (prefill). `encoder_hidden` is
    /// the encoder output; cross K/V are recomputed from it each call.
    /// Returns logits `[B, dec_seq, vocab]`.
    /// Decoder hidden states (no embed lookup, no LM head). Takes precomputed
    /// `decoder_inputs_embeds [B,dec_seq,d]` (token embeds gathered host-side)
    /// plus `encoder_hidden`, and returns the post-layer hidden `[B,dec_seq,d]`.
    /// The LM head is applied host-side on just the needed row, so neither the
    /// `[vocab,d]` embedding nor a full-vocab logits tensor lives in the graph.
    pub(crate) fn emit_decoder_hidden(
        &mut self,
        cfg: &Florence2Config,
        inputs_embeds: HirNodeId,
        encoder_hidden: HirNodeId,
        dec_seq: usize,
    ) -> Result<HirNodeId> {
        let t = &cfg.text;
        let d = t.d_model;
        let pos = self.learned_positions(&lk::dec_embed_positions(), 0, dec_seq, d)?;
        let mut x = self.g().add(inputs_embeds, pos);
        x = self.layer_norm(
            x,
            &lk::dec_layernorm_embedding_w(),
            &lk::dec_layernorm_embedding_b(),
        )?;
        for layer in 0..t.decoder_layers {
            x = self.decoder_layer(t, layer, x, encoder_hidden, dec_seq)?;
        }
        Ok(x)
    }

    fn decoder_layer(
        &mut self,
        t: &Florence2TextConfig,
        layer: usize,
        x: HirNodeId,
        encoder_hidden: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let nh = t.decoder_attention_heads;
        let hd = t.dec_head_dim();
        let enc_seq = self.kv_seq(encoder_hidden);
        let p = |s: &str| lk::dec_layer(layer, s);

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

        // FFN.
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

    /// Cross attention: Q from decoder, K/V projected from encoder hidden states.
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

    /// Self attention: Q,K,V from the same source.
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

    fn ffn(&mut self, x: HirNodeId, w1: &str, b1: &str, w2: &str, b2: &str) -> Result<HirNodeId> {
        let h = self.linear(x, w1, Some(b1))?;
        let h = self.g().gelu(h);
        self.linear(h, w2, Some(b2))
    }

    /// Learned positional embeddings with BART's offset of 2, broadcast over
    /// the batch. Rows `[offset+past .. offset+past+seq]`.
    fn learned_positions(
        &mut self,
        key: &str,
        past: usize,
        seq: usize,
        d: usize,
    ) -> Result<HirNodeId> {
        let pos_w = self.load_param(key, false)?;
        let start = Florence2TextConfig::POS_OFFSET + past;
        let rows = self.g().narrow_(pos_w, 0, start, seq);
        Ok(self.g().reshape_(rows, vec![1, seq as i64, d as i64]))
    }

    pub(crate) fn kv_seq(&self, x: HirNodeId) -> usize {
        self.hir.node(x).shape.dim(1).unwrap_static()
    }
}
