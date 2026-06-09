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

//! Debug first greedy codec group-0 vs HF (env `RLX_QWEN3_TTS_DIR`).

use rlx_qwen3_tts::Qwen3TtsConfig;
use rlx_qwen3_tts::code_predictor::CodePredictorEngine;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_custom_voice_prompt, load_text_tokenizer};
use rlx_qwen3_tts::talker::engine::TalkerEngine;
use rlx_qwen3_tts::talker::math::{linear_logits, sample_greedy_talker_codec};
use rlx_qwen3_tts::text_embed::TextEmbedder;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn debug_first_group0_hi() {
    let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: RLX_QWEN3_TTS_DIR");
        return;
    };
    if !model_dir.join("model.safetensors").is_file() {
        eprintln!("skip: weights");
        return;
    }
    let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).expect("config");
    let store = Qwen3TtsWeightStore::open(&model_dir).expect("store");
    let tokenizer = load_text_tokenizer(&model_dir).expect("tok");
    let text_embedder = TextEmbedder::open(&store).expect("text");
    let prompt = build_custom_voice_prompt(
        &cfg,
        &store,
        &text_embedder,
        &tokenizer,
        "Hi.",
        "vivian",
        "english",
    )
    .expect("prompt");
    eprintln!("prefill_rows={}", prompt.embeds.nrows());

    let mut talker =
        TalkerEngine::open(&store, cfg.talker(), rlx_runtime::Device::Cpu).expect("talker");
    let hidden = talker.prefill(prompt.embeds.view()).expect("prefill");
    let h_last: Vec<f32> = hidden.row(hidden.nrows() - 1).iter().copied().collect();
    if let Ok(hf) = std::fs::read(repo_root().join(".cache/qwen3-tts/hf_layer_last.json")) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&hf) {
            if let Some(row) = v["layer_last"].as_array().and_then(|a| a.last()) {
                let hf_last: Vec<f32> = row
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect();
                let mut max_d = 0f32;
                for (a, b) in h_last.iter().zip(hf_last.iter()) {
                    max_d = max_d.max((a - b).abs());
                }
                eprintln!("compiled prefill vs hf norm max_abs={max_d}");
            }
        }
    }
    let h_last = hidden.row(hidden.nrows() - 1);
    if std::env::var("RLX_QWEN3_TTS_DUMP_CP").ok().as_deref() == Some("1") {
        let path = repo_root().join(".cache/qwen3-tts/cp_past_hidden.json");
        std::fs::create_dir_all(path.parent().unwrap()).ok();
        let vec: Vec<f32> = h_last.iter().copied().collect();
        std::fs::write(&path, serde_json::to_string(&vec).unwrap()).ok();
        eprintln!("wrote {}", path.display());
    }
    let logits = linear_logits(h_last, talker.codec_head().view()).expect("logits");
    let g0 = sample_greedy_talker_codec(
        &logits,
        cfg.talker().vocab_size,
        cfg.talker().codec_eos_token_id,
    );
    eprintln!("native_g0={g0} (HF frame0[0]=1995)");

    let mut cp = CodePredictorEngine::open(&store, cfg.code_predictor(), rlx_runtime::Device::Cpu)
        .expect("cp");
    let groups = cp.predict_groups(h_last, g0).expect("cp");
    eprintln!("native_frame0={groups:?}");
    eprintln!(
        "HF frame0=[1995, 1642, 988, 1088, 246, 1543, 1579, 437, 1356, 86, 1042, 248, 1555, 781, 1772, 374]"
    );

    if std::env::var("RLX_QWEN3_TTS_CP_SCAN").ok().as_deref() == Some("1") {
        let golden = [
            1995u32, 1642, 988, 1088, 246, 1543, 1579, 437, 1356, 86, 1042, 248, 1555, 781, 1772,
            374,
        ];
        for ri in 0..hidden.nrows() {
            let h_row = hidden.row(ri);
            let g = cp.predict_groups(h_row, g0).expect("cp scan");
            let ok = g[1] == golden[1];
            eprintln!("row {ri} g1={} match={ok}", g[1]);
        }
    }
}
