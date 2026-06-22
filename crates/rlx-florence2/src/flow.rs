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

//! Graph assembly: vision encoder, BART text encoder, and BART decoder as
//! separate [`BuiltModel`]s compiled per device.

use crate::builder::Florence2Builder;
use crate::config::Florence2Config;
use anyhow::Result;
use rlx_core::flow_util::built_from_hir_with_profile;
use rlx_flow::{BuiltModel, CompileProfile, WeightSource};
use rlx_ir::hir::{FusionPolicy, HirModule};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Vision tower: `pixel [B,3,H,W]` → `image_features [B, image_seq, proj]`.
pub fn build_vision_built(
    cfg: &Florence2Config,
    weights: &mut dyn WeightSource,
    batch: usize,
    img_h: usize,
    img_w: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let mut hir = HirModule::new("florence2_vision").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let pixel = hir.input("pixel", Shape::new(&[batch, 3, img_h, img_w], f));
    let mut b = Florence2Builder::new(&mut hir, &mut params, weights, batch);
    let feats = b.emit_vision(cfg, pixel, img_h, img_w)?;
    hir.outputs = vec![feats];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// BART encoder over `inputs_embeds [B,S,d]` → `encoder_hidden [B,S,d]`.
pub fn build_encoder_built(
    cfg: &Florence2Config,
    weights: &mut dyn WeightSource,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.text.d_model;
    let mut hir = HirModule::new("florence2_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let inputs_embeds = hir.input("inputs_embeds", Shape::new(&[batch, seq, d], f));
    let mut b = Florence2Builder::new(&mut hir, &mut params, weights, batch);
    let hidden = b.emit_encoder(cfg, inputs_embeds, seq)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// BART decoder hidden states from `decoder_inputs_embeds [B,T,d]` +
/// `encoder_hidden [B,enc_seq,d]` → hidden `[B,T,d]`. Compiled once per
/// (bucket, enc_seq) and reused across decode steps (causal mask keeps each
/// position independent of trailing pad); the LM head runs host-side.
pub fn build_decoder_hidden_built(
    cfg: &Florence2Config,
    weights: &mut dyn WeightSource,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.text.d_model;
    let mut hir =
        HirModule::new("florence2_decoder_hidden").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let inputs_embeds = hir.input("decoder_inputs_embeds", Shape::new(&[batch, dec_seq, d], f));
    let encoder_hidden = hir.input("encoder_hidden", Shape::new(&[batch, enc_seq, d], f));
    let mut b = Florence2Builder::new(&mut hir, &mut params, weights, batch);
    let hidden = b.emit_decoder_hidden(cfg, inputs_embeds, encoder_hidden, dec_seq)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}
