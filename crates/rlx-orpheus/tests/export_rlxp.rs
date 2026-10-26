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

//! An exported Orpheus, opened by rlx alone, speaks as `OrpheusTts` does
//! (`weights/tts/orpheus`'s GGUF, SNAC in `weights/tts/snac-24khz`): the
//! language model a `generate` entry its program samples as the runner
//! does; SNAC rlx graphs.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_orpheus::OrpheusTts;
use rlx_orpheus::export::export_rlxp;
use rlx_runtime::Device;
use serde_json::json;

fn weights() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_orpheus_speaks_as_the_runner_does() {
    let gguf = weights().join("orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf");
    let snac = weights().join("snac-24khz/snac_24khz_decoder.safetensors");
    if !gguf.is_file() || !snac.is_file() {
        eprintln!("skip: no Orpheus GGUF or SNAC decoder");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("orpheus.rlxp");
    let opts = ExportOptions {
        uppers: vec![256, 1024],
        ..ExportOptions::default()
    };
    eprintln!("{}", export_rlxp(&gguf, &out, &opts).unwrap());
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    // Four frames, sampled as the runner samples.
    let new = 28u32;
    let mut runner = OrpheusTts::load_on(&gguf, &snac, Device::Cpu).unwrap();
    runner.config.max_new_tokens = new;
    let text = "Good morning, the tea is ready.";
    let want = runner.synthesize(text, Some("tara")).unwrap();

    // The language model: the runner's codes, until the two LMs' float
    // rounding tips a near tie (a sampled decoder's ordinary divergence).
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "text": text, "voice": "tara", "max_new_tokens": new}),
        )
        .unwrap();
    let codes: Vec<i32> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let same = codes
        .iter()
        .zip(&want.codes)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "codes: {same} of {} the runner's ({} from the package)",
        want.codes.len(),
        codes.len()
    );
    assert!(same >= 14, "{same} codes the same");

    // SNAC: the runner's codes, through the package's graphs.
    let audio = m.synthesize(text, &json!({"codes": want.codes})).unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    assert_eq!(audio.samples.len(), want.samples.len());
    let cos = cosine(&audio.samples, &want.samples);
    eprintln!("{} samples, cosine {cos}", want.samples.len());
    assert!(cos > 0.999, "cosine {cos}");
}
