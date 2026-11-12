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

//! NLLB / M2M100 encoder + decoder graphs, matching HF
//! `M2M100{Encoder,Decoder}`: pre-norm layers, sinusoidal positions offset by
//! `padding_idx + 1`, ReLU FFN, a final `layer_norm` per stack, tied LM head.
//!
//! Two decoder spellings share the layer code: the full-prefix causal pass
//! ([`NllbBuilder::emit_decoder_hidden`], teacher forcing) and the KV-cached
//! single-token step ([`NllbBuilder::emit_decoder_step`]). Each must equal the
//! other, which is what the HF parity test checks.

use crate::builder::NllbBuilder;
use crate::config::NllbConfig;
use crate::weights::lang as lk;
use anyhow::Result;
use rlx_ir::hir::{HirGraphExt, HirNodeId};
use rlx_ir::op::MaskKind;
use rlx_ir::{Shape, ops::attention::attention_kind_op};

/// Key mask for one attention: none, causal, or a `[B, key_len]` keep mask.
#[derive(Clone, Copy)]
pub(crate) enum KeyMask {
    None,
    Causal,
    Keep(HirNodeId),
}

/// Cached K/V a decode step attends to, one entry per decoder layer.
pub(crate) struct StepKv<'a> {
    pub cross_k: &'a [HirNodeId],
    pub cross_v: &'a [HirNodeId],
    pub past_k: &'a [HirNodeId],
    pub past_v: &'a [HirNodeId],
    /// `[B, past + 1]` keep mask over `concat(past_k, k_new)`.
    pub self_mask: HirNodeId,
    /// `[B, src]` keep mask over the (bucket-padded) source.
    pub src_mask: HirNodeId,
}

impl NllbBuilder<'_> {
    /// Encoder over token embeds `[B, S, d]` (already `× embed_scale`);
    /// adds positions in-graph. Returns `encoder_hidden [B, S, d]`.
    pub(crate) fn emit_encoder(
        &mut self,
        cfg: &NllbConfig,
        inputs_embeds: HirNodeId,
        seq: usize,
    ) -> Result<HirNodeId> {
        let pos = self.positional_embeddings(cfg, "enc", seq)?;
        let x = self.g().add(inputs_embeds, pos);
        self.emit_encoder_stack(cfg, x, KeyMask::None)
    }

    /// Encoder layers + final norm over `x = embeds + positions`.
    pub(crate) fn emit_encoder_stack(
        &mut self,
        cfg: &NllbConfig,
        mut x: HirNodeId,
        mask: KeyMask,
    ) -> Result<HirNodeId> {
        let nh = cfg.encoder_attention_heads;
        let hd = cfg.enc_head_dim();
        for layer in 0..cfg.encoder_layers {
            let p = |s: &str| lk::enc_layer(layer, s);
            let h = self.layer_norm(
                x,
                &p("self_attn_layer_norm.weight"),
                &p("self_attn_layer_norm.bias"),
            )?;
            let (q, k, v) = self.qkv(h, h, &p("self_attn"))?;
            let sa = self.attend(q, k, v, &p("self_attn"), nh, hd, mask)?;
            x = self.g().add(x, sa);
            x = self.ffn_block(x, &p)?;
        }
        self.layer_norm(
            x,
            &lk::enc_final_layer_norm_w(),
            &lk::enc_final_layer_norm_b(),
        )
    }

    /// Per decoder layer, the cross-attention `(K, V)` of the encoder output.
    pub(crate) fn emit_cross_kv(
        &mut self,
        cfg: &NllbConfig,
        encoder_hidden: HirNodeId,
    ) -> Result<Vec<(HirNodeId, HirNodeId)>> {
        (0..cfg.decoder_layers)
            .map(|layer| {
                let p = |s: &str| lk::dec_layer(layer, s);
                let k = self.linear(
                    encoder_hidden,
                    &p("encoder_attn.k_proj.weight"),
                    Some(&p("encoder_attn.k_proj.bias")),
                )?;
                let v = self.linear(
                    encoder_hidden,
                    &p("encoder_attn.v_proj.weight"),
                    Some(&p("encoder_attn.v_proj.bias")),
                )?;
                Ok((k, v))
            })
            .collect()
    }

    /// Full-prefix causal decoder over token embeds `[B, T, d]` (already
    /// `× embed_scale`); adds positions in-graph. Returns hidden `[B, T, d]`
    /// after the final norm. The LM head runs host-side.
    pub(crate) fn emit_decoder_hidden(
        &mut self,
        cfg: &NllbConfig,
        inputs_embeds: HirNodeId,
        encoder_hidden: HirNodeId,
        dec_seq: usize,
    ) -> Result<HirNodeId> {
        let pos = self.positional_embeddings(cfg, "dec", dec_seq)?;
        let mut x = self.g().add(inputs_embeds, pos);
        let cross = self.emit_cross_kv(cfg, encoder_hidden)?;
        for (layer, &(ck, cv)) in cross.iter().enumerate() {
            let p = |s: &str| lk::dec_layer(layer, s);
            let h = self.layer_norm(
                x,
                &p("self_attn_layer_norm.weight"),
                &p("self_attn_layer_norm.bias"),
            )?;
            let (q, k, v) = self.qkv(h, h, &p("self_attn"))?;
            let sa = self.attend(
                q,
                k,
                v,
                &p("self_attn"),
                self.dec_nh(cfg),
                cfg.dec_head_dim(),
                KeyMask::Causal,
            )?;
            x = self.g().add(x, sa);
            x = self.cross_block(cfg, x, ck, cv, &p, KeyMask::None)?;
            x = self.ffn_block(x, &p)?;
        }
        self.layer_norm(
            x,
            &lk::dec_final_layer_norm_w(),
            &lk::dec_final_layer_norm_b(),
        )
    }

    /// One KV-cached decoder step over `x [B, 1, d]` (embeds × scale +
    /// position, host-side). Returns the normed hidden `[B, 1, d]` and, per
    /// layer, `concat(past, new)` self-attention `(K, V)` `[B, past + 1, d]`.
    pub(crate) fn emit_decoder_step(
        &mut self,
        cfg: &NllbConfig,
        mut x: HirNodeId,
        kv: &StepKv<'_>,
    ) -> Result<(HirNodeId, Vec<(HirNodeId, HirNodeId)>)> {
        let mut new_kv = Vec::with_capacity(cfg.decoder_layers);
        for layer in 0..cfg.decoder_layers {
            let p = |s: &str| lk::dec_layer(layer, s);
            let h = self.layer_norm(
                x,
                &p("self_attn_layer_norm.weight"),
                &p("self_attn_layer_norm.bias"),
            )?;
            let (q, k_new, v_new) = self.qkv(h, h, &p("self_attn"))?;
            let k = self.g().concat_(vec![kv.past_k[layer], k_new], 1);
            let v = self.g().concat_(vec![kv.past_v[layer], v_new], 1);
            let sa = self.attend(
                q,
                k,
                v,
                &p("self_attn"),
                self.dec_nh(cfg),
                cfg.dec_head_dim(),
                KeyMask::Keep(kv.self_mask),
            )?;
            new_kv.push((k, v));
            x = self.g().add(x, sa);
            x = self.cross_block(
                cfg,
                x,
                kv.cross_k[layer],
                kv.cross_v[layer],
                &p,
                KeyMask::Keep(kv.src_mask),
            )?;
            x = self.ffn_block(x, &p)?;
        }
        let hidden = self.layer_norm(
            x,
            &lk::dec_final_layer_norm_w(),
            &lk::dec_final_layer_norm_b(),
        )?;
        Ok((hidden, new_kv))
    }

    fn dec_nh(&self, cfg: &NllbConfig) -> usize {
        cfg.decoder_attention_heads
    }

    /// `x + encoder_attn(LN(x))` over precomputed cross `(K, V)`.
    fn cross_block(
        &mut self,
        cfg: &NllbConfig,
        x: HirNodeId,
        ck: HirNodeId,
        cv: HirNodeId,
        p: &dyn Fn(&str) -> String,
        mask: KeyMask,
    ) -> Result<HirNodeId> {
        let h = self.layer_norm(
            x,
            &p("encoder_attn_layer_norm.weight"),
            &p("encoder_attn_layer_norm.bias"),
        )?;
        let q = self.linear(
            h,
            &p("encoder_attn.q_proj.weight"),
            Some(&p("encoder_attn.q_proj.bias")),
        )?;
        let ca = self.attend(
            q,
            ck,
            cv,
            &p("encoder_attn"),
            self.dec_nh(cfg),
            cfg.dec_head_dim(),
            mask,
        )?;
        Ok(self.g().add(x, ca))
    }

    /// `x + fc2(relu(fc1(LN(x))))` — M2M100 / NLLB use **ReLU**.
    fn ffn_block(&mut self, x: HirNodeId, p: &dyn Fn(&str) -> String) -> Result<HirNodeId> {
        let h = self.layer_norm(
            x,
            &p("final_layer_norm.weight"),
            &p("final_layer_norm.bias"),
        )?;
        let h = self.linear(h, &p("fc1.weight"), Some(&p("fc1.bias")))?;
        let h = self.g().relu(h);
        let ff = self.linear(h, &p("fc2.weight"), Some(&p("fc2.bias")))?;
        Ok(self.g().add(x, ff))
    }

    fn qkv(
        &mut self,
        q_src: HirNodeId,
        kv_src: HirNodeId,
        pfx: &str,
    ) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
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
        Ok((q, k, v))
    }

    /// Scaled dot-product attention (`head_dim^-0.5`, the op default) +
    /// `out_proj`.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        pfx: &str,
        n_head: usize,
        head_dim: usize,
        mask: KeyMask,
    ) -> Result<HirNodeId> {
        let q_shape = self.hir.node(q).shape.clone();
        let (kind, inputs) = match mask {
            KeyMask::None => (MaskKind::None, vec![q, k, v]),
            KeyMask::Causal => (MaskKind::Causal, vec![q, k, v]),
            KeyMask::Keep(m) => (MaskKind::Custom, vec![q, k, v, m]),
        };
        let attn = self.g().add_node(
            attention_kind_op(n_head, head_dim, None, kind, None, None),
            inputs,
            q_shape,
        );
        self.linear(
            attn,
            &format!("{pfx}.out_proj.weight"),
            Some(&format!("{pfx}.out_proj.bias")),
        )
    }

    /// `[1, seq, d]` slice of the M2M100 sinusoidal table starting at row
    /// `padding_idx + 1` (the first non-pad position).
    fn positional_embeddings(
        &mut self,
        cfg: &NllbConfig,
        tag: &str,
        seq: usize,
    ) -> Result<HirNodeId> {
        let d = cfg.d_model;
        let start = cfg.pad_token_id as usize + 1;
        let table = cfg.sinusoidal_positions(start + seq);
        let slice = table[start * d..(start + seq) * d].to_vec();
        let key = format!("@sin_pos_{tag}_{seq}");
        let id = self.hir.param(&key, Shape::new(&[1, seq, d], self.f));
        self.params.insert(key, slice);
        Ok(id)
    }
}
