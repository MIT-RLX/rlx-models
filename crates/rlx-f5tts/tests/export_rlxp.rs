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

//! An exported F5-TTS, opened by rlx alone, clones a voice as
//! `F5Native::synthesize` does (`weights/tts/f5tts`): its frontend and flow
//! loop a program, its three graphs rlx graphs symbolic in the reference
//! audio, the ids and the frames.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_f5tts::export::export_rlxp;
use rlx_f5tts::model::InferOpts;
use rlx_f5tts::native::F5Native;
use rlx_runtime::Device;
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
fn an_exported_f5tts_clones_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/f5tts");
    if !dir.join("F5_Transformer.onnx").is_file() {
        eprintln!("skip: no F5-TTS checkpoint");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("f5tts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = F5Native::load_on(&dir, Device::Cpu).unwrap();
    let wav = prompt();
    let text = "Seven boats sailed out at dawn.";
    let opts = InferOpts { nfe: 8, speed: 1.0 };
    let want = runner.synthesize(text, &wav, PROMPT_TEXT, &opts).unwrap();
    let got = m
        .synthesize(
            text,
            &json!({"nfe": 8, "prompt": {"audio": wav, "sample_rate": 24000, "text": PROMPT_TEXT}}),
        )
        .unwrap();
    assert_eq!(got.samples.len(), want.len());
    let worst = got
        .samples
        .iter()
        .zip(&want)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    let dot: f64 = got
        .samples
        .iter()
        .zip(&want)
        .map(|(x, y)| f64::from(x * y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    let cos = dot / (norm(&got.samples) * norm(&want));
    eprintln!("{} samples, off by {worst}, cosine {cos}", want.len());
    // The graphs the runner imports, symbolic: the same samples.
    assert!(worst < 1e-6, "off by {worst} (cosine {cos})");
}
