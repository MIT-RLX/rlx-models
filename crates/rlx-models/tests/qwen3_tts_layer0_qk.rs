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

//! Layer-0 Q/K after RoPE vs HF (`hf_l0_qk.json`).

use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::eager::TalkerEagerModel;
use rlx_qwen3_tts::text_embed::TextEmbedder;
use std::path::PathBuf;

#[test]
fn layer0_qk_head0_matches_hf() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let hf_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/qwen3-tts/hf_l0_qk.json");
    if !hf_path.is_file() {
        eprintln!("skip: run dump_talker_l0_qk.py");
        return;
    }
    let hf: serde_json::Value = serde_json::from_slice(&std::fs::read(&hf_path).unwrap()).unwrap();
    let hf_q: Vec<f32> = hf["q_head0_last16"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let hf_k: Vec<f32> = hf["k_head0_last16"]
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
    let (rq, rk) = eager.layer0_qk_head0_last(prompt.embeds.view()).unwrap();
    let mut max_q = 0f32;
    let mut max_k = 0f32;
    for (a, b) in rq.iter().zip(hf_q.iter()) {
        max_q = max_q.max((a - b).abs());
    }
    for (a, b) in rk.iter().zip(hf_k.iter()) {
        max_k = max_k.max((a - b).abs());
    }
    eprintln!("layer0 q max_abs={max_q} k max_abs={max_k}");
    eprintln!("native q[:4]={:?} hf q[:4]={:?}", &rq[..4], &hf_q[..4]);
    assert!(max_q < 1e-3, "q after rope diverged (max_abs={max_q})");
    assert!(max_k < 1e-3, "k after rope diverged (max_abs={max_k})");
}
