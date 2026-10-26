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

//! An exported OpenVoice, opened by rlx alone, clones a voice as
//! `OpenVoice::synthesize` does (`weights/tts/openvoice`, MeloTTS beside
//! it): MeloTTS' graphs, the spectrogram and the tone converter's, rlx
//! graphs; the glue a program.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_openvoice::export::export_rlxp;
use rlx_openvoice::{DEFAULT_TAU, OpenVoice};
use rlx_runtime::Device;
use serde_json::json;

fn prompt() -> (Vec<f32>, u32) {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rlx-luxtts/tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(p).unwrap();
    let sr = r.spec().sample_rate;
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    (
        r.samples::<i32>()
            .map(|s| s.unwrap() as f32 / max)
            .collect(),
        sr,
    )
}

#[test]
fn an_exported_openvoice_clones_as_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts");
    let (dir, melo) = (root.join("openvoice"), root.join("melotts"));
    if !dir.join("tone_color.onnx").is_file() || !melo.join("onnx/decoder.onnx").is_file() {
        eprintln!("skip: no OpenVoice checkpoint or MeloTTS bundle");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("openvoice.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = OpenVoice::load_on(&melo, &dir, Device::Cpu).unwrap();
    let (wav, sr) = prompt();
    let text = "Seven boats sailed out at dawn.";
    let want = runner.synthesize(text, &wav, sr, DEFAULT_TAU).unwrap();
    let c = |a: &[f32], b: &[f32]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
        let n = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
        dot / (n(a) * n(b))
    };
    // No flow noise too (`tau` 0).
    let zero = runner.synthesize(text, &wav, sr, 0.0).unwrap();
    let got0 = m
        .synthesize(
            text,
            &json!({"tau": 0.0, "prompt": {"audio": wav, "sample_rate": sr}}),
        )
        .unwrap();
    let cos0 = c(&zero, &got0.samples);
    eprintln!("tau 0: cosine {cos0}");
    assert!(cos0 > 0.999, "tau 0: cosine {cos0}");
    let got = m
        .synthesize(text, &json!({"prompt": {"audio": wav, "sample_rate": sr}}))
        .unwrap();
    assert_eq!(got.sample_rate, 22_050);
    assert_eq!(got.samples.len(), want.len());
    let worst = got
        .samples
        .iter()
        .zip(&want)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    let cos = c(&got.samples, &want);
    eprintln!("{} samples, off by {worst}, cosine {cos}", want.len());
    // The spectrograms are matmul DFTs here, FFTs in the runner.
    assert!(cos > 0.999, "cosine {cos}");
}
