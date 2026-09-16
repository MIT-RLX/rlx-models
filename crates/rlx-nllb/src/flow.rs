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

//! Graph assembly: M2M100 text encoder and decoder as separate [`BuiltModel`]s.

use crate::builder::NllbBuilder;
use crate::config::NllbConfig;
use anyhow::Result;
use rlx_core::flow_util::built_from_hir_with_profile;
use rlx_flow::{BuiltModel, CompileProfile, WeightSource};
use rlx_ir::hir::{FusionPolicy, HirModule};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Encoder over `inputs_embeds [B,S,d]` → `encoder_hidden [B,S,d]`.
pub fn build_encoder_built(
    cfg: &NllbConfig,
    weights: &mut dyn WeightSource,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.d_model;
    let mut hir = HirModule::new("nllb_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let inputs_embeds = hir.input("inputs_embeds", Shape::new(&[batch, seq, d], f));
    let mut b = NllbBuilder::new(&mut hir, &mut params, weights, batch);
    let hidden = b.emit_encoder(cfg, inputs_embeds, seq)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// Decoder hidden states from `decoder_inputs_embeds [B,T,d]` +
/// `encoder_hidden [B,enc_seq,d]` → hidden `[B,T,d]`.
pub fn build_decoder_hidden_built(
    cfg: &NllbConfig,
    weights: &mut dyn WeightSource,
    batch: usize,
    dec_seq: usize,
    enc_seq: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.d_model;
    let mut hir = HirModule::new("nllb_decoder_hidden").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let inputs_embeds = hir.input("decoder_inputs_embeds", Shape::new(&[batch, dec_seq, d], f));
    let encoder_hidden = hir.input("encoder_hidden", Shape::new(&[batch, enc_seq, d], f));
    let mut b = NllbBuilder::new(&mut hir, &mut params, weights, batch);
    let hidden = b.emit_decoder_hidden(cfg, inputs_embeds, encoder_hidden, dec_seq)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}
