// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Graph assembly: encoder and decoder as separate [`BuiltModel`]s.

use crate::builder::MoonshineBuilder;
use crate::config::MoonshineConfig;
use crate::weights::MoonshineWeightPrefix;
use anyhow::Result;
use rlx_core::flow_util::built_from_hir_with_profile;
use rlx_flow::{BuiltModel, CompileProfile, WeightSource};
use rlx_ir::hir::{FusionPolicy, HirModule};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Encoder: `pcm [B, L]` → `encoder_hidden [B, T, d]`.
pub fn build_encoder_built(
    cfg: &MoonshineConfig,
    weights: &mut dyn WeightSource,
    pfx: &MoonshineWeightPrefix,
    batch: usize,
    audio_len: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let mut hir = HirModule::new("moonshine_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let pcm = hir.input("pcm", Shape::new(&[batch, audio_len], f));
    let mut b = MoonshineBuilder::new(&mut hir, &mut params, weights, pfx, batch);
    let hidden = b.emit_encoder(cfg, pcm, audio_len)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// Decoder hidden: `decoder_inputs_embeds [B,T,d]` + `encoder_hidden [B,enc,d]` → `[B,T,d]`.
pub fn build_decoder_hidden_built(
    cfg: &MoonshineConfig,
    weights: &mut dyn WeightSource,
    pfx: &MoonshineWeightPrefix,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.hidden_size;
    let mut hir =
        HirModule::new("moonshine_decoder_hidden").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let inputs_embeds = hir.input("decoder_inputs_embeds", Shape::new(&[batch, dec_seq, d], f));
    let encoder_hidden = hir.input("encoder_hidden", Shape::new(&[batch, enc_seq, d], f));
    let mut b = MoonshineBuilder::new(&mut hir, &mut params, weights, pfx, batch);
    let hidden = b.emit_decoder_hidden(cfg, inputs_embeds, encoder_hidden, dec_seq, enc_seq)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}
