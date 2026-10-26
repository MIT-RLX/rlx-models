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

//! An exported Qwen3-TTS CustomVoice (`weights/tts/qwen3-tts-0.6b-customvoice`),
//! opened by rlx alone, speaks as the official `qwen-tts` does: its greedy
//! codec frames are the reference's (`tests/fixtures/custom_voice_greedy`),
//! and its speech the crate's decoder's on them.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_qwen3_tts::export::export_rlxp;
use rlx_qwen3_tts::speech_tokenizer::open_speech_decoder;
use rlx_runtime::Device;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_qwen3_tts_speaks_as_the_reference_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../../weights/tts/qwen3-tts-0.6b-customvoice");
    if !dir.join("tokenizer.json").is_file() {
        eprintln!("skip: no Qwen3-TTS CustomVoice checkpoint with its tokenizer.json");
        return;
    }
    rlx_text::install_model_tokenizer();
    let case: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/custom_voice_greedy/case.json")).unwrap(),
    )
    .unwrap();
    let want: Vec<Vec<u32>> = serde_json::from_value(case["codes"].clone()).unwrap();
    let want_audio = open_speech_decoder(&dir, Device::Cpu)
        .unwrap()
        .decode(&want, Device::Cpu)
        .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen3-tts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "text": case["text"], "speaker": case["speaker"],
                    "language": case["language"], "max_new_tokens": 256}),
        )
        .unwrap();
    let frames: Vec<Vec<u32>> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect::<Vec<_>>()
        .chunks(16)
        .map(|c| c.to_vec())
        .collect();
    let same = frames.iter().zip(&want).take_while(|(a, b)| a == b).count();
    eprintln!(
        "{same} of {} frames the reference's ({} from the package)",
        want.len(),
        frames.len()
    );
    assert_eq!(frames, want);
    let audio: Vec<f32> = got[1]
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(audio.len(), want_audio.len());
    let cos = cosine(&audio, &want_audio);
    eprintln!("{} samples, cosine {cos}", audio.len());
    assert!(cos > 0.9999, "cosine {cos}");
}
