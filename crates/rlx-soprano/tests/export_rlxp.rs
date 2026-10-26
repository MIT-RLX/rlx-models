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

//! An exported Soprano 1.1 (`weights/tts/soprano`), opened by rlx alone,
//! speaks as `NativeSoprano` does: the same audio tokens, sampled and
//! greedy, and the same speech.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use rlx_soprano::export::export_rlxp;
use rlx_soprano::{InferOpts, NativeSoprano};
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_soprano_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/soprano");
    if !dir.join("onnx/soprano_backbone_kv_fp32.onnx").is_file() {
        eprintln!("skip: no Soprano ONNX");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("soprano.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    // Its graphs rlx IR alone: no op an ONNX import adds.
    let bytes = std::fs::read(&out).unwrap();
    let onnx = bytes.windows(5).filter(|w| w == b"onnx.").count();
    assert_eq!(onnx, 0, "an ONNX op in the package");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = NativeSoprano::open_loose(&dir, Device::Cpu).unwrap();

    let text = "Good morning, the tea is ready.";
    for greedy in [false, true] {
        let opts = InferOpts {
            max_new_tokens: 40,
            greedy,
            ..InferOpts::default()
        };
        let want = runner.generate_audio_tokens(text, &opts).unwrap();
        let got = m
            .run(
                &[],
                &json!({"entry": "speak", "text": text, "max_new_tokens": 40, "greedy": greedy}),
            )
            .unwrap();
        let got: Vec<i64> = got[0]
            .data
            .chunks_exact(4)
            .map(|b| i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect();
        eprintln!(
            "greedy {greedy}: {} tokens, the runner's {}",
            got.len(),
            want.len()
        );
        assert_eq!(got, want, "greedy {greedy}");
    }

    let opts = InferOpts {
        max_new_tokens: 40,
        ..InferOpts::default()
    };
    let want = runner.synthesize(text, &opts).unwrap();
    let got = m.synthesize(text, &json!({"max_new_tokens": 40})).unwrap();
    assert_eq!(got.sample_rate, 32_000);
    assert_eq!(got.samples.len(), want.len());
    let cos = cosine(&got.samples, &want);
    eprintln!("{} samples, cosine {cos}", want.len());
    assert!(cos > 0.9999, "cosine {cos}");
}
