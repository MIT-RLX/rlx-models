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

//! Gemma 4 E2B multimodal glue: fuse the (parity-verified) vision/audio soft
//! tokens into the LM token stream and run the E2B text LM over the fused
//! `inputs_embeds`.
//!
//! The E2B text LM is driven by [`crate::builder::build_gemma_graph_sized_packed_ext`].
//! That builder already exposes an `input_embeddings` graph input (its
//! "embed_lazy" path) which it multiplies by `√hidden`. We reuse it for
//! multimodal: feed **pre-scale** embeddings where text rows are the raw
//! `embed_tokens` rows and media rows are `soft_token / √hidden`, so the
//! builder's `× √hidden` yields `text·√hidden` (the Gemma embed-scale) and the
//! media soft tokens unchanged — exactly matching HF, which scales only the
//! text embeddings and `masked_scatter`s the media features in as-is.
//!
//! `embed_lazy` is enabled by passing a one-entry `known_packed` map containing
//! `model.embed_tokens.weight`; E2B's `lm_head` is **not** tied, so none of the
//! tied-head packed paths fire — the entry only flips the input mode.

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::Graph;
use rlx_ir::quant::QuantScheme;

use crate::config::GemmaConfig;
use crate::multimodal::{GemmaMultimodalConfig, fuse_multimodal_embeddings};
use crate::multimodal_mask::{build_vision_bidirectional_mask_2d, expand_attn_bias};
use crate::qat_loader::GemmaQatLoader;

type PackedWeightMap = HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>;

/// Build the **pre-scale** fused `inputs_embeds` `[seq·hidden]` for the
/// embed-lazy LM path: raw `embed_tokens` rows for text positions, and
/// `media_soft_token / √hidden` spliced at the image/audio/video placeholder
/// positions (so the builder's `× √hidden` restores the soft tokens as-is).
///
/// `image_embeds` / `audio_embeds` / `video_embeds` are the LM-space soft
/// tokens from [`crate::gemma4_vision::build_vision_features`] /
/// [`crate::gemma4_audio::build_audio_features`] (flat `[n_tok · hidden]`).
pub fn e2b_fused_inputs_embeds_prescale(
    loader: &GemmaQatLoader,
    cfg: &GemmaConfig,
    mm_cfg: &GemmaMultimodalConfig,
    token_ids: &[u32],
    image_embeds: &[f32],
    audio_embeds: &[f32],
    video_embeds: &[f32],
) -> Result<Vec<f32>> {
    let h = cfg.hidden_size;
    // Raw text rows (no embed-scale; the builder applies √hidden).
    let (mut embeds, dim) =
        loader.dequant_embedding_rows("model.embed_tokens.weight", token_ids)?;
    anyhow::ensure!(dim == h, "embed dim {dim} != hidden {h}");
    // Media soft tokens are in LM space already; pre-divide by √hidden so the
    // builder's × √hidden leaves them unchanged.
    let inv = 1.0f32 / (h as f32).sqrt();
    let scale = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| x * inv).collect() };
    fuse_multimodal_embeddings(
        &mut embeds,
        token_ids,
        h,
        mm_cfg,
        &scale(image_embeds),
        &scale(audio_embeds),
        &scale(video_embeds),
    )?;
    Ok(embeds)
}

/// Additive LM attention bias `[batch, heads, seq, seq]` (0 = allowed,
/// large-negative = blocked): causal everywhere, **plus** full bidirectional
/// attention inside contiguous image/audio/video placeholder spans — matching
/// HF E2B's `mm_token_type_ids` → `block_sequence_ids` masking. Feed as the
/// `attn_bias` graph input. Valid for `seq <= sliding_window` (sliding layers
/// then equal causal, so one bias serves every layer).
pub fn e2b_media_attn_bias(
    token_ids: &[u32],
    cfg: &GemmaConfig,
    mm_cfg: &GemmaMultimodalConfig,
    batch: usize,
) -> Vec<f32> {
    let mask2d = build_vision_bidirectional_mask_2d(token_ids, mm_cfg);
    expand_attn_bias(&mask2d, batch, cfg.num_attention_heads, token_ids.len())
}

/// A `known_packed` marker map: flips the builder's `embed_lazy` (so the LM
/// takes `input_embeddings`) and, when `media_bias`, also `__media_bias__` (so
/// attention reads an `attn_bias` tensor — bidirectional media blocks). The
/// bytes are never read for E2B (lm_head is not tied); only `contains_key` is.
pub fn embed_lazy_marker_with(cfg: &GemmaConfig, media_bias: bool) -> PackedWeightMap {
    let mut m = PackedWeightMap::new();
    m.insert(
        "model.embed_tokens.weight".to_string(),
        (
            Vec::new(),
            QuantScheme::Int8Block { block_size: 32 },
            vec![cfg.vocab_size, cfg.hidden_size],
        ),
    );
    if media_bias {
        m.insert(
            "__media_bias__".to_string(),
            (
                Vec::new(),
                QuantScheme::Int8Block { block_size: 32 },
                vec![],
            ),
        );
    }
    m
}

/// See [`embed_lazy_marker_with`] (no media bias).
pub fn embed_lazy_marker(cfg: &GemmaConfig) -> PackedWeightMap {
    embed_lazy_marker_with(cfg, false)
}

/// Build the E2B LM prefill graph in `inputs_embeds` mode. Graph inputs:
/// `input_embeddings` `[batch, seq, hidden]` (pre-scale; see
/// [`e2b_fused_inputs_embeds_prescale`]) and `per_layer_inputs`. Output: logits.
pub fn build_e2b_multimodal_prefill(
    cfg: &GemmaConfig,
    loader: &mut GemmaQatLoader,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_e2b_multimodal_prefill_ext(cfg, loader, batch, seq, false)
}

/// Build the E2B LM prefill in `inputs_embeds` mode. With `media_bias`, the LM
/// also takes an `attn_bias` input (see [`e2b_media_attn_bias`]) so image/audio
/// blocks attend bidirectionally; otherwise plain causal/sliding masking.
pub fn build_e2b_multimodal_prefill_ext(
    cfg: &GemmaConfig,
    loader: &mut GemmaQatLoader,
    batch: usize,
    seq: usize,
    media_bias: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let marker = embed_lazy_marker_with(cfg, media_bias);
    let mut packed = HashMap::new();
    crate::builder::build_gemma_graph_sized_packed_ext(
        cfg,
        loader,
        batch,
        seq,
        /*with_lm_head*/ true,
        /*last_token_from_input*/ false,
        /*with_kv_outputs*/ false,
        &mut packed,
        Some(&marker),
        None,
    )
}
