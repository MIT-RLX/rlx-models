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

//! CustomVoice greedy synthesis generates the codec frames the official
//! `qwen-tts` does (`scripts/custom_voice_reference.py` wrote
//! `tests/fixtures/custom_voice_greedy`).

use std::path::PathBuf;

use rlx_qwen3_tts::Qwen3TtsWeightStore;
use rlx_qwen3_tts::config::{GenerationConfig, Qwen3TtsConfig};
use rlx_qwen3_tts::synthesize::synthesize_custom_voice_greedy;
use rlx_runtime::Device;

#[test]
fn custom_voice_frames_are_the_reference() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../../weights/tts/qwen3-tts-0.6b-customvoice");
    if !dir.join("model.safetensors").is_file() {
        eprintln!("skip: no Qwen3-TTS CustomVoice checkpoint");
        return;
    }
    let case: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/custom_voice_greedy/case.json")).unwrap(),
    )
    .unwrap();
    let want: Vec<Vec<u32>> = serde_json::from_value(case["codes"].clone()).unwrap();
    let cfg = Qwen3TtsConfig::from_model_dir(&dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&dir).unwrap();
    let gen_cfg = GenerationConfig {
        do_sample: false,
        subtalker_dosample: false,
        repetition_penalty: case["repetition_penalty"].as_f64().unwrap() as f32,
        max_new_tokens: 256,
        ..GenerationConfig::default()
    };
    let got = synthesize_custom_voice_greedy(
        &dir,
        &cfg,
        &store,
        Device::Cpu,
        case["text"].as_str().unwrap(),
        case["speaker"].as_str().unwrap(),
        case["language"].as_str().unwrap(),
        &gen_cfg,
        true,
    )
    .unwrap()
    .codec_frames;
    let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
    eprintln!(
        "{same} of {} frames the reference's ({} from the runner)",
        want.len(),
        got.len()
    );
    assert_eq!(got, want);
}
