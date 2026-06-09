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

//! Native CustomVoice prefill embeds vs HF capture (`hf_ie_full.json`).

use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::text_embed::TextEmbedder;
use std::path::PathBuf;

#[test]
fn native_embeds_match_hf_ie() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        return;
    };
    let ie_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/qwen3-tts/hf_ie_full.json");
    if !ie_path.is_file() {
        eprintln!("skip: hf_ie_full.json");
        return;
    }
    let ie_j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ie_path).unwrap()).unwrap();
    let hf: Vec<f32> = ie_j["embeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).unwrap();
    let store = Qwen3TtsWeightStore::open(&model_dir).unwrap();
    let tokenizer = load_text_tokenizer(&model_dir).unwrap();
    let text_embedder = TextEmbedder::open(&store).unwrap();
    let prompt = build_custom_voice_prompt(
        &cfg,
        &store,
        &text_embedder,
        &tokenizer,
        "Hi.",
        "vivian",
        "english",
    )
    .unwrap();
    let native: Vec<f32> = prompt.embeds.iter().copied().collect();
    assert_eq!(native.len(), hf.len());
    let mut max_d = 0f32;
    let mut worst = 0usize;
    for (i, (a, b)) in native.iter().zip(hf.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_d {
            max_d = d;
            worst = i;
        }
    }
    eprintln!("embeds max_abs={max_d} at flat_idx={worst}");
    assert!(max_d < 1e-5, "embeds differ from HF (max_abs={max_d})");
}
