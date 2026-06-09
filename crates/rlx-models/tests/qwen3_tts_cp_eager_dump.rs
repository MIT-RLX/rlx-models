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

//! CP eager vs HF on dumped `cp_past_hidden.json` (env `RLX_QWEN3_TTS_DIR`).

use ndarray::Array2;
use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::code_predictor::CodePredictorEngine;
use rlx_qwen3_tts::load::{Qwen3TtsWeightStore, remap_code_predictor_weights};
use rlx_qwen3_tts::talker::engine::TalkerEngine;
use std::path::PathBuf;

#[test]
fn cp_eager_matches_hf_on_dumped_past_hidden() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let dump = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/qwen3-tts/cp_past_hidden.json");
    if !dump.is_file() {
        eprintln!("skip: run qwen3_tts_debug_first with RLX_QWEN3_TTS_DUMP_CP=1 first");
        return;
    }
    let past: Vec<f32> = serde_json::from_slice(&std::fs::read(&dump).unwrap()).unwrap();
    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let mut cp =
        CodePredictorEngine::open(&store, cfg.code_predictor(), rlx_runtime::Device::Cpu).unwrap();
    let groups = cp.predict_groups_slice(&past, 1995).unwrap();
    eprintln!("eager={groups:?}");

    if std::env::var("RLX_QWEN3_TTS_CP_COMPILED").ok().as_deref() == Some("1") {
        let cp_cfg = cfg.code_predictor();
        let mut wm = store.load_code_predictor_backbone().unwrap();
        let weights = remap_code_predictor_weights(&mut wm).unwrap();
        let talker_cfg = cp_cfg.to_talker_config();
        let mut compiled = TalkerEngine::open_with_weights(
            &model_dir,
            &store,
            &talker_cfg,
            weights,
            rlx_runtime::Device::Cpu,
        )
        .unwrap();
        let e0 = store
            .tensor_snapshot(&["talker.model.codec_embedding.weight"])
            .unwrap();
        let (tc, sh) = e0.get("talker.model.codec_embedding.weight").unwrap();
        let codec = Array2::from_shape_vec((sh[0], sh[1]), tc.clone()).unwrap();
        let embeds = Array2::from_shape_vec((2, talker_cfg.hidden_size), {
            let mut v = past.clone();
            v.extend(codec.row(1995).iter().copied());
            v
        })
        .unwrap();
        let h = compiled.prefill(embeds.view()).unwrap();
        eprintln!(
            "compiled hs[-1,:8] = {:?}",
            h.row(h.nrows() - 1).iter().take(8).collect::<Vec<_>>()
        );
    }

    assert_eq!(
        groups[1], 1642,
        "g1 should match HF on same past_hidden (compiled CP)"
    );
}
