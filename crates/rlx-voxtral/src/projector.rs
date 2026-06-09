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

//! Multimodal projector — 4× temporal downsample + GELU MLP to text hidden size.

use crate::config::VoxtralConfig;
use crate::weights::VoxtralWeightPrefix;
use anyhow::{Result, ensure};
use rlx_flow::WeightSource;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

pub fn build_voxtral_projector_hir(
    cfg: &VoxtralConfig,
    weights: &mut dyn WeightSource,
    batch: usize,
    enc_seq: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
    ensure!(
        enc_seq.is_multiple_of(4),
        "projector requires encoder seq divisible by 4, got {enc_seq}"
    );
    let audio_h = cfg.audio_config.d_model;
    let group = cfg.audio_config.intermediate_size;
    let down_seq = enc_seq / 4;
    let f = DType::F32;

    let mut hir = HirModule::new("voxtral_projector").with_fusion_policy(FusionPolicy::Direct);
    let mut params = HashMap::new();
    let enc = hir.input("encoder_hidden", Shape::new(&[batch, enc_seq, audio_h], f));

    let w1 = load_param(
        &mut hir,
        &mut params,
        weights,
        VoxtralWeightPrefix::projector_linear1(),
        true,
    )?;
    let w2 = load_param(
        &mut hir,
        &mut params,
        weights,
        VoxtralWeightPrefix::projector_linear2(),
        true,
    )?;

    let mut g = HirMut::new(&mut hir);
    let grouped = g.reshape_(enc, vec![batch as i64, down_seq as i64, group as i64]);
    let mm1 = g.mm(grouped, w1);
    let h1 = g.gelu(mm1);
    let out = g.mm(h1, w2);
    hir.outputs = vec![out];
    Ok((hir, params))
}

pub fn build_voxtral_projector_built(
    cfg: &VoxtralConfig,
    weights: &mut rlx_core::weight_map::WeightMap,
    batch: usize,
    enc_seq: usize,
) -> Result<rlx_flow::BuiltModel> {
    use rlx_core::flow_util::WeightMapSource;
    let (hir, params) =
        build_voxtral_projector_hir(cfg, &mut WeightMapSource(weights), batch, enc_seq)?;
    rlx_core::flow_util::built_from_hir(hir, params)
}

fn load_param(
    hir: &mut HirModule,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightSource,
    key: &str,
    transpose: bool,
) -> Result<HirNodeId> {
    let (data, shape) = weights.take(key, transpose)?;
    let id = hir.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}
