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

//! Layer-0 attention output (pre-`o_proj`) vs HF (`hf_l0_attn.json`).

use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::eager::TalkerEagerModel;
use rlx_qwen3_tts::text_embed::TextEmbedder;
use std::path::PathBuf;

#[test]
fn layer0_attn_pre_o_matches_hf() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let hf_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/qwen3-tts/hf_l0_attn.json");
    if !hf_path.is_file() {
        eprintln!("skip: run hf_l0_attn dump");
        return;
    }
    let hf: serde_json::Value = serde_json::from_slice(&std::fs::read(&hf_path).unwrap()).unwrap();
    let hf_a: Vec<f32> = hf["attn_pre_o_last16"]
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

    let mut eager = TalkerEagerModel::open(&store, cfg.talker()).unwrap();
    let native = eager
        .layer0_attn_pre_o_last16(prompt.embeds.view())
        .unwrap();
    let mut max_d = 0f32;
    for (a, b) in native.iter().zip(hf_a.iter()) {
        max_d = max_d.max((a - b).abs());
    }
    eprintln!("attn pre-o max_abs={max_d}");
    eprintln!("native[:4]={:?} hf[:4]={:?}", &native[..4], &hf_a[..4]);
    assert!(max_d < 0.006, "attn pre-o diverged (max_abs={max_d})");
}
