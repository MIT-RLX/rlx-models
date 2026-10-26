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

//! An exported ZipVoice-distill, opened by rlx alone, clones a voice as the
//! runner does with ZipVoice's defaults (`weights/tts/zipvoice-distill`).
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use rlx_zipvoice::export::export_rlxp;
use rlx_zipvoice::{ZipVoice, zipvoice_opts};
use serde_json::json;

const PROMPT_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";

fn prompt() -> Vec<f32> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rlx-luxtts/tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(p).unwrap();
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    r.samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect()
}

#[test]
fn an_exported_zipvoice_clones_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/zipvoice-distill");
    if !dir.join("encoder_body.onnx").is_file() || !dir.join("onnx/vocoder_spec.onnx").is_file() {
        eprintln!("skip: no ZipVoice-distill checkpoint");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("zipvoice.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = ZipVoice::load_on(&dir, Device::Cpu).unwrap();
    let wav = prompt();
    let text = "Seven boats sailed out at dawn, and six came home by night.";
    let want = runner
        .synthesize(text, &wav, PROMPT_TEXT, &zipvoice_opts())
        .unwrap();
    let got = m
        .synthesize(
            text,
            &json!({"prompt": {"audio": wav, "sample_rate": 24000, "text": PROMPT_TEXT}}),
        )
        .unwrap();
    assert_eq!(got.samples.len(), want.len());
    let dot: f64 = got
        .samples
        .iter()
        .zip(&want)
        .map(|(x, y)| f64::from(x * y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    let cos = dot / (norm(&got.samples) * norm(&want));
    eprintln!("{} samples, cosine {cos}", want.len());
    // As LuxTTS: the prompt's log-mel a matmul DFT, the runner's an FFT.
    assert!(cos > 0.999, "cosine {cos}");
}
