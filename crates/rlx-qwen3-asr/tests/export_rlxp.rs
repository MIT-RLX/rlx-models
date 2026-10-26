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

//! An exported Qwen3-ASR (`weights/asr/qwen3-asr-0.6b`), opened by rlx
//! alone, transcribes as the crate's runner does: its transcript is the
//! runner's, on a clip of one window and on one of two.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_qwen3_asr::export::export_rlxp;
use rlx_qwen3_asr::{AsrRunner, AsrTokenizer};
use rlx_runtime::Device;
use serde_json::json;

const ASR_TEXT: u32 = 151704;
const IM_END: u32 = 151645;

/// A clip at 16 kHz (the assets are 24 kHz).
fn speech(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/jfk")
        .join(name);
    let w = rlx_nemotron_asr::wav::parse(&std::fs::read(path).unwrap()).unwrap();
    rlx_nemotron_asr::wav::resample(&w.samples, w.sample_rate, 16_000)
}

#[test]
fn an_exported_qwen3_asr_transcribes_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/asr/qwen3-asr-0.6b");
    if !dir.join("model.safetensors").is_file() {
        eprintln!("skip: no Qwen3-ASR checkpoint (`just fetch-qwen3-asr`)");
        return;
    }
    rlx_text::install_model_tokenizer();
    let clips = ["jfk_voice_clone.wav", "jfk_rust_speech.wav"].map(speech);
    let runner = AsrRunner::builder()
        .weights(&dir)
        .device(Device::Cpu)
        .build()
        .unwrap();
    let tok = AsrTokenizer::from_model_dir(&dir).unwrap();
    // The runner's transcript: what follows `<asr_text>`.
    let want: Vec<String> = clips
        .iter()
        .map(|pcm| {
            let ids = runner.transcribe_pcm_ids(pcm, "").unwrap();
            let at = ids.iter().position(|&i| i == ASR_TEXT).map_or(0, |p| p + 1);
            let end = ids.iter().position(|&i| i == IM_END).unwrap_or(ids.len());
            tok.decode(&ids[at..end]).unwrap().trim().to_string()
        })
        .collect();
    drop(runner);

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen3-asr.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (pcm, want) in clips.iter().zip(&want) {
        let text = m.transcribe(pcm, 16_000, &json!({})).unwrap();
        eprintln!("runner:  {want}\npackage: {text}");
        assert_eq!(&text, want);
    }
}
