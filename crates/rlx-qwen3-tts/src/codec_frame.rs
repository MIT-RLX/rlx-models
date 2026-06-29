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

//! Tier-1 RLX graphs for one codec-AR frame (talker decode + CP prefill/decode building blocks).
//!
//! Production fusion: [`crate::codec_frame_fused::CodecFrameFusedEngine::run_codec_frame`]
//! (host CP greedy + talker decode graph). Optional megagraph:
//! [`crate::codec_frame_fused::build_qwen3_tts_codec_frame_built`]
//! (`RLX_QWEN3_TTS_CODEC_FRAME_MEGAGRAPH=1`).
//! This module centralizes the **RLX graph pieces** shared by talker and code predictor.

use crate::compile_opts::{tune_cp_qwen3_profile, tune_qwen3_profile};
use crate::weights::weight_map_from_cache;
use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::{BuiltModel, CompileProfile};
use rlx_qwen3::{
    Qwen3Config, Qwen3DecodeOpts, Qwen3PrefillOpts, build_qwen3_decode_embeds_built,
    build_qwen3_prefill_embeds_built, qwen3_profile_near_weights,
};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

/// Which Qwen3-shaped subgraph to build (talker 28L vs CP 5L).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3TtsGraphRole {
    Talker,
    CodePredictor,
}

/// Prefill + decode compile profiles tuned for `compile_device`.
#[derive(Debug, Clone)]
pub struct Qwen3TtsGraphProfiles {
    pub prefill: CompileProfile,
    pub decode: CompileProfile,
}

impl Qwen3TtsGraphProfiles {
    pub fn for_role(model_dir: &Path, role: Qwen3TtsGraphRole, compile_device: Device) -> Self {
        let mut prefill = qwen3_profile_near_weights(model_dir, false);
        let mut decode = qwen3_profile_near_weights(model_dir, true);
        match role {
            Qwen3TtsGraphRole::Talker => {
                tune_qwen3_profile(&mut prefill, compile_device);
                tune_qwen3_profile(&mut decode, compile_device);
            }
            Qwen3TtsGraphRole::CodePredictor => {
                tune_cp_qwen3_profile(&mut prefill, compile_device);
                tune_cp_qwen3_profile(&mut decode, compile_device);
            }
        }
        Self { prefill, decode }
    }
}

/// Single-token `inputs_embeds` decode (`[batch, 1, hidden]` → `hidden_states` + K/V).
pub fn build_qwen3_tts_decode_built(
    qwen3: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    past_seq: usize,
    profile: &CompileProfile,
) -> Result<BuiltModel> {
    build_qwen3_decode_embeds_built(
        qwen3,
        weights,
        &Qwen3DecodeOpts {
            batch: 1,
            past_seq,
            dynamic_past: false,
            use_custom_mask: true,
            ragged_rope: false,
            profile: Some(profile.clone()),
        },
    )
}

/// Multi-token `inputs_embeds` prefill with K/V export (no LM head).
pub fn build_qwen3_tts_prefill_built(
    qwen3: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    seq: usize,
    profile: &CompileProfile,
    rope_cos: Option<Vec<f32>>,
    rope_sin: Option<Vec<f32>>,
) -> Result<BuiltModel> {
    build_qwen3_prefill_embeds_built(
        qwen3,
        weights,
        &Qwen3PrefillOpts {
            batch: 1,
            seq,
            with_kv_outputs: true,
            with_qk_outputs: false,
            with_lm_head: false,
            last_logits_only: false,
            profile: Some(profile.clone()),
            rope_cos,
            rope_sin,
        },
    )
}

/// Talker decode graph for one codec frame (codec embed sum → next talker hidden).
pub fn build_qwen3_tts_codec_frame_decode_built(
    qwen3: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    past_seq: usize,
    profile: &CompileProfile,
) -> Result<BuiltModel> {
    build_qwen3_tts_decode_built(qwen3, weights, past_seq, profile)
}

/// Build talker decode `(Graph, params)` from a weight snapshot (bucket `upper`).
pub fn talker_decode_graph_parts(
    qwen3: &Qwen3Config,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    profile: &CompileProfile,
    upper: u64,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let mut wm = weight_map_from_cache(weights)?;
    let built = build_qwen3_tts_decode_built(qwen3, &mut wm, upper as usize, profile)?;
    built.into_graph_parts()
}

/// Build talker decode HIR from a weight snapshot (bucket `upper`).
pub fn talker_decode_hir_parts(
    qwen3: &Qwen3Config,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    profile: &CompileProfile,
    upper: u64,
) -> Result<(rlx_ir::hir::HirModule, HashMap<String, Vec<f32>>)> {
    let mut wm = weight_map_from_cache(weights)?;
    let built = build_qwen3_tts_decode_built(qwen3, &mut wm, upper as usize, profile)?;
    built.into_parts()
}

/// Build CP decode `(Graph, params)` from a weight snapshot.
pub fn cp_decode_graph_parts(
    qwen3: &Qwen3Config,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    profile: &CompileProfile,
    upper: u64,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let mut wm = weight_map_from_cache(weights)?;
    let built = build_qwen3_tts_decode_built(qwen3, &mut wm, upper as usize, profile)?;
    built.into_graph_parts()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Qwen3TtsConfig;
    use crate::load::{Qwen3TtsWeightStore, remap_talker_weights};
    use std::path::PathBuf;

    #[test]
    fn codec_frame_talk_decode_graph_builds() {
        let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
            eprintln!("skip: RLX_QWEN3_TTS_DIR");
            return;
        };
        let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).expect("config");
        let store = Qwen3TtsWeightStore::open(&model_dir).expect("store");
        let mut wm = store.load_talker_backbone().expect("talker wm");
        let weights = remap_talker_weights(&mut wm).expect("remap");
        let qwen3 = cfg.talker().to_qwen3_config();
        let profiles = Qwen3TtsGraphProfiles::for_role(
            store.model_dir(),
            Qwen3TtsGraphRole::Talker,
            Device::Cpu,
        );
        let mut loader = crate::weights::SnapshotLoader::new(weights);
        let built =
            build_qwen3_tts_codec_frame_decode_built(&qwen3, &mut loader, 16, &profiles.decode)
                .expect("codec frame decode built");
        let (graph, params) = built.into_graph_parts().expect("graph parts");
        assert!(!graph.nodes().is_empty());
        assert!(!params.is_empty());
    }
}
