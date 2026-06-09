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

//! Layer-0 forward from HF prefill embeds (`hf_ie_full.json`).

use ndarray::Array2;
use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::mrope::talker_rope_index_prefill;
use rlx_qwen3_tts::talker::eager::TalkerEagerModel;
use std::path::PathBuf;

#[test]
fn layer0_from_hf_embeds() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let ie_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/qwen3-tts/hf_ie_full.json");
    let layers_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/qwen3-tts/hf_layer_last.json");
    if !ie_path.is_file() || !layers_path.is_file() {
        eprintln!("skip: run dump_talker_ie_full.py");
        return;
    }
    let ie_j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ie_path).unwrap()).unwrap();
    let layers: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&layers_path).unwrap()).unwrap();
    let seq = ie_j["seq"].as_u64().unwrap() as usize;
    let embeds: Vec<f32> = ie_j["embeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let x = Array2::from_shape_vec((seq, 1024), embeds).unwrap();
    let hf_l0: Vec<f32> = layers["layer_last"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    let mask = vec![1u8; seq];
    let (positions, _) = talker_rope_index_prefill(&mask);
    let out = eager.forward_layer(0, x.view(), &positions, 0).unwrap();
    let native: Vec<f32> = out.row(out.nrows() - 1).iter().copied().collect();
    let mut max_d = 0f32;
    for (a, b) in native.iter().zip(hf_l0.iter()) {
        max_d = max_d.max((a - b).abs());
    }
    eprintln!("layer0 from hf-embeds max_abs={max_d}");
    assert!(
        max_d < 0.002,
        "layer0 from hf embeds diverged (max_abs={max_d})"
    );
}
