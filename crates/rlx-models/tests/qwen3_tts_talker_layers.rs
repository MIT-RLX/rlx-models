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

//! Compare eager talker per-layer hidden vs HF dump (`hf_layer_last.json`).

use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::eager::TalkerEagerModel;
use rlx_qwen3_tts::text_embed::TextEmbedder;
use std::path::PathBuf;

#[test]
fn eager_talker_layer_last_tracks_hf() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    let hf_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/qwen3-tts/hf_layer_last.json");
    if !hf_path.is_file() {
        eprintln!("skip: run dump_talker_layer_hiddens.py first");
        return;
    }
    let hf: serde_json::Value = serde_json::from_slice(&std::fs::read(&hf_path).unwrap()).unwrap();
    let hf_rows: Vec<Vec<f32>> = hf["layer_last"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
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
    let native = eager.prefill_layer_last_rows(prompt.embeds.view()).unwrap();
    assert_eq!(native.len(), hf_rows.len());

    let mut first_bad = None;
    for (li, (n, h)) in native.iter().zip(hf_rows.iter()).enumerate() {
        let mut max_d = 0f32;
        for (a, b) in n.iter().zip(h.iter()) {
            max_d = max_d.max((a - b).abs());
        }
        if li < 5 || li + 1 == native.len() {
            eprintln!("layer {li} max_abs={max_d}");
        }
        if max_d > 0.01 && first_bad.is_none() {
            first_bad = Some((li, max_d, n[..8].to_vec(), h[..8].to_vec()));
        }
    }
    if let Some((li, max_d, n8, h8)) = first_bad {
        eprintln!("first divergent layer {li} max_abs={max_d}");
        eprintln!("native[:8]={n8:?}");
        eprintln!("hf[:8]={h8:?}");
        panic!("eager layer {li} diverged from HF (max_abs={max_d})");
    }
}
