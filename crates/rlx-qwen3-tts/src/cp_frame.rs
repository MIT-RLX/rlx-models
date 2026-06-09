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

//! Code-predictor codec frame: 2-token prefill + up to 15 decode steps (groups 1–15).
//!
//! Greedy lm_head + embed gather stay on host; backbone forward uses tier-1 RLX graphs
//! from [`crate::codec_frame`].

use crate::codec_frame::{build_qwen3_tts_decode_built, build_qwen3_tts_prefill_built};
/// CustomVoice 0.6B: 16 codec groups → 15 CP lm_head steps after group-0.
pub const CP_AR_LM_STEPS: usize = 15;
use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::{BuiltModel, CompileProfile};
use rlx_qwen3::Qwen3Config;

/// CP prefill length: talker hidden + group-0 codec embed.
pub const CP_PREFILL_TWO: usize = 2;

/// CustomVoice 0.6B CP AR depth after group-0.
pub const CP_DECODE_STEPS: usize = CP_AR_LM_STEPS; // 15
/// Decode forwards inside one codec frame (15 lm_head steps → 14 decode steps).
pub const CP_DECODE_BACKBONE_STEPS: usize = CP_AR_LM_STEPS - 1;

/// Two-token CP prefill graph (start of every codec frame).
pub fn build_qwen3_tts_cp_prefill_two_built(
    qwen3: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    profile: &CompileProfile,
    rope_cos: Option<Vec<f32>>,
    rope_sin: Option<Vec<f32>>,
) -> Result<BuiltModel> {
    build_qwen3_tts_prefill_built(qwen3, weights, CP_PREFILL_TWO, profile, rope_cos, rope_sin)
}

/// One CP decode step inside the codec frame (group embed → next hidden).
pub fn build_qwen3_tts_cp_decode_step_built(
    qwen3: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    past_seq: usize,
    profile: &CompileProfile,
) -> Result<BuiltModel> {
    build_qwen3_tts_decode_built(qwen3, weights, past_seq, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Qwen3TtsConfig;
    use crate::codec_frame::{Qwen3TtsGraphProfiles, Qwen3TtsGraphRole};
    use crate::load::{Qwen3TtsWeightStore, remap_code_predictor_weights};
    use crate::talker::rope::{build_inv_freq, rope_prefill_feeds, rope_tables_full};
    use rlx_runtime::Device;
    use std::path::PathBuf;

    #[test]
    fn cp_prefill_two_graph_builds() {
        let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
            eprintln!("skip: RLX_QWEN3_TTS_DIR");
            return;
        };
        let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).expect("config");
        let store = Qwen3TtsWeightStore::open(&model_dir).expect("store");
        let cp = cfg.code_predictor();
        let mut wm = store.load_code_predictor_backbone().expect("cp wm");
        let weights = remap_code_predictor_weights(&mut wm).expect("remap");
        let qwen3 = cp.to_qwen3_config();
        let profiles = Qwen3TtsGraphProfiles::for_role(
            store.model_dir(),
            Qwen3TtsGraphRole::CodePredictor,
            Device::Cpu,
        );
        let _head_half = cp.head_dim / 2;
        let inv_freq = build_inv_freq(cp.head_dim, cp.rope_theta);
        let positions: Vec<usize> = (0..CP_PREFILL_TWO).collect();
        let (rope_cos, rope_sin) = rope_prefill_feeds(&inv_freq, &positions, cp.head_dim);
        let mut loader = crate::weights::SnapshotLoader::new(weights);
        let built = build_qwen3_tts_cp_prefill_two_built(
            &qwen3,
            &mut loader,
            &profiles.prefill,
            Some(rope_cos),
            Some(rope_sin),
        )
        .expect("cp prefill-two built");
        let (graph, params) = built.into_graph_parts().expect("graph parts");
        assert!(!graph.nodes().is_empty());
        assert!(!params.is_empty());
        let _ = rope_tables_full(&inv_freq, 8, cp.head_dim);
    }
}
