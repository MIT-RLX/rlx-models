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

//! An exported Parler-TTS, opened by rlx alone, speaks as
//! `NativeParler::synthesize` does (`weights/tts/parlertts`, DAC beside
//! it): the sampling loop a program, the encoder, the decoder and DAC rlx
//! graphs.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_parlertts::export::export_rlxp;
use rlx_parlertts::native::DEFAULT_DESCRIPTION;
use rlx_parlertts::{InferOpts, NativeParler};
use rlx_runtime::Device;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

#[test]
fn an_exported_parler_speaks_as_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts");
    let (dir, dac) = (root.join("parlertts"), root.join("parler-dac"));
    if !dir.join("onnx/decoder.onnx").is_file() || !dac.join("model.safetensors").is_file() {
        eprintln!("skip: no Parler-TTS checkpoint or DAC");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("parler.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = NativeParler::open(&dir, &dac, Device::Cpu).unwrap();
    let text = "Hello there.";
    for greedy in [true, false] {
        let opts = InferOpts {
            greedy,
            ..InferOpts::default()
        };
        let want = runner.synthesize(text, DEFAULT_DESCRIPTION, &opts).unwrap();
        let got = m.synthesize(text, &json!({"greedy": greedy})).unwrap();
        assert_eq!(got.samples.len(), want.len(), "greedy {greedy}");
        let cos = cosine(&got.samples, &want);
        eprintln!("greedy {greedy}: {} samples, cosine {cos}", want.len());
        assert!(cos > 0.999, "greedy {greedy}: cosine {cos}");
    }
}
