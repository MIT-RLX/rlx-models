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

//! An exported Maya1, opened by rlx alone, speaks as `Maya1` does
//! (`weights/tts/maya1`'s GGUF, SNAC in `weights/tts/snac-24khz`): greedy, so
//! the same codes, and SNAC's rlx graphs over them.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_maya1::export::{DEFAULT_DESCRIPTION, ExportOptions, export_rlxp};
use rlx_maya1::{Device, Maya1};
use serde_json::json;

#[test]
fn an_exported_maya1_speaks_as_the_runner_does() {
    let tts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts");
    let gguf = tts.join("maya1/maya1.Q4_K_M.gguf");
    let snac = tts.join("snac-24khz/snac_24khz_decoder.safetensors");
    if !gguf.is_file() || !snac.is_file() {
        eprintln!("skip: no Maya1 GGUF or SNAC decoder");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("maya1.rlxp");
    let opts = ExportOptions {
        uppers: vec![256, 1024],
        ..ExportOptions::default()
    };
    eprintln!("{}", export_rlxp(&gguf, &out, &opts).unwrap());
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    // Four frames.
    let new = 28u32;
    let mut runner = Maya1::load_with_snac(&gguf, &snac, Device::Cpu).unwrap();
    runner.config_mut().max_new_tokens = new;
    let text = "Good morning, the tea is ready.";
    let want = runner.synthesize(DEFAULT_DESCRIPTION, text).unwrap();
    let got = m.synthesize(text, &json!({"max_new_tokens": new})).unwrap();
    assert_eq!(got.sample_rate, 24_000);
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
    assert!(cos > 0.999, "cosine {cos}");
}
