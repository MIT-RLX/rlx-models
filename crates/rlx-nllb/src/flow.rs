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
//!
//! The generation path uses two graphs: [`build_encoder_cross_built`] (the
//! encoder and every layer's cross-attention K/V, over a bucket-padded source)
//! and [`build_decode_step_built`] (one KV-cached token for `batch` sequences).
//! [`build_encoder_built`] / [`build_decoder_hidden_built`] are the
//! full-sequence spellings, kept as the independent path tests compare to.

use crate::builder::NllbBuilder;
use crate::config::NllbConfig;
use crate::language::{KeyMask, StepKv};
use anyhow::Result;
use rlx_core::flow_util::built_from_hir_with_profile;
use rlx_flow::{BuiltModel, CompileProfile, WeightSource};
use rlx_ir::hir::{FusionPolicy, HirModule};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

/// Encoder over token embeds `inputs_embeds [B,S,d]` (`× embed_scale`,
/// positions added in-graph) → `encoder_hidden [B,S,d]`.
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
    let mut b = NllbBuilder::new(&mut hir, &mut params, weights);
    let hidden = b.emit_encoder(cfg, inputs_embeds, seq)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// Causal decoder hidden states from token embeds `decoder_inputs_embeds [B,T,d]` +
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
    let mut b = NllbBuilder::new(&mut hir, &mut params, weights);
    let hidden = b.emit_decoder_hidden(cfg, inputs_embeds, encoder_hidden, dec_seq)?;
    hir.outputs = vec![hidden];
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// Name of the decode step's source keep-mask parameter.
pub const SRC_MASK_PARAM: &str = "@src_mask";

/// Parameter names of the cross-attention K/V a decode step reads, per layer.
pub fn cross_kv_param_names(num_layers: usize) -> Vec<String> {
    (0..num_layers)
        .flat_map(|i| [format!("@cross_k_{i}"), format!("@cross_v_{i}")])
        .collect()
}

/// Encoder over `x [1, S, d]` = token embeds × scale + positions (host-side),
/// with a `mask [1, S]` keep mask so `S` can be a padded bucket. Outputs
/// `encoder_hidden [1, S, d]`, then `cross_k_i`, `cross_v_i` `[1, S, d]` for
/// every decoder layer — the cross K/V never change during generation.
pub fn build_encoder_cross_built(
    cfg: &NllbConfig,
    weights: &mut dyn WeightSource,
    src: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.d_model;
    let mut hir = HirModule::new("nllb_encoder_cross").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let x = hir.input("x", Shape::new(&[1, src, d], f));
    let mask = hir.input("mask", Shape::new(&[1, src], f));
    let mut b = NllbBuilder::new(&mut hir, &mut params, weights);
    let hidden = b.emit_encoder_stack(cfg, x, KeyMask::Keep(mask))?;
    let cross = b.emit_cross_kv(cfg, hidden)?;
    hir.outputs = vec![hidden];
    for (k, v) in cross {
        hir.outputs.push(k);
        hir.outputs.push(v);
    }
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// One decoder step for `batch` sequences against `past` cached positions
/// (a bucket upper bound) and a `src`-row source.
///
/// Inputs: `x [B,1,d]` (token embed × scale + position, host-side),
/// `mask [B, past+1]` ([`rlx_runtime::attn_mask::bucket_decode_mask`]),
/// `past_k_i` / `past_v_i [B, past, d]`. The source's [`SRC_MASK_PARAM`]
/// `[B, src]` and cross K/V ([`cross_kv_param_names`], `[B, src, d]`) are
/// **parameters**, set once per source sentence rather than copied into the
/// graph every step (they are ~12 MB at `src = 128`). Outputs: hidden `[B,1,d]`, then per
/// layer `concat(past, new)` K and V `[B, past+1, d]` — the layout
/// `rlx_core::autoregressive::run_bucketed_kv_decode_keyed_batched` expects.
pub fn build_decode_step_built(
    cfg: &NllbConfig,
    weights: &mut dyn WeightSource,
    batch: usize,
    past: usize,
    src: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let d = cfg.d_model;
    let n = cfg.decoder_layers;
    let mut hir = HirModule::new("nllb_decode_step").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let x = hir.input("x", Shape::new(&[batch, 1, d], f));
    let self_mask = hir.input("mask", Shape::new(&[batch, past + 1], f));
    let src_mask = hir.param(SRC_MASK_PARAM, Shape::new(&[batch, src], f));
    let names = cross_kv_param_names(n);
    let (mut cross_k, mut cross_v) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        cross_k.push(hir.param(&names[2 * i], Shape::new(&[batch, src, d], f)));
        cross_v.push(hir.param(&names[2 * i + 1], Shape::new(&[batch, src, d], f)));
    }
    let names = rlx_core::autoregressive::past_kv_input_names(n);
    let (mut past_k, mut past_v) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        past_k.push(hir.input(&names[2 * i], Shape::new(&[batch, past, d], f)));
        past_v.push(hir.input(&names[2 * i + 1], Shape::new(&[batch, past, d], f)));
    }
    let kv = StepKv {
        cross_k: &cross_k,
        cross_v: &cross_v,
        past_k: &past_k,
        past_v: &past_v,
        self_mask,
        src_mask,
    };
    let mut b = NllbBuilder::new(&mut hir, &mut params, weights);
    let (hidden, new_kv) = b.emit_decoder_step(cfg, x, &kv)?;
    hir.outputs = vec![hidden];
    for (k, v) in new_kv {
        hir.outputs.push(k);
        hir.outputs.push(v);
    }
    built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}
