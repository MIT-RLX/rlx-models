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

//! An exported Pocket TTS (`weights/tts/pocket-tts-kyutai/languages/
//! english_2026-10`), opened by rlx alone, speaks as `PocketTts::generate`
//! does: a released voice, a one-chunk and a two-chunk text, sampled and at
//! temperature 0.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_pocket_tts::PocketTts;
use rlx_pocket_tts::export::export_rlxp;
use rlx_runtime::Device;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_pocket_tts_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../weights/tts/pocket-tts-kyutai/languages/english_2026-10");
    if !dir.join("model.safetensors").is_file() || !dir.join("tokenizer.json").is_file() {
        eprintln!("skip: no Pocket TTS english_2026-10");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("pocket-tts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut tts = PocketTts::open(&dir, "english_2026-10", Device::Cpu).unwrap();
    let voice = tts.load_voice("alba").unwrap();
    let default_temperature = tts.options.temperature;

    let long = "The old lighthouse keeper climbed the spiral stairs every evening, \
                counting each step as he went. At the top he lit the great lamp, \
                and its beam swept across the dark water to guide the fishing boats \
                safely home before the storm arrived.";
    for (text, temperature) in [
        (
            "Hello there, this is a test of the package.",
            default_temperature,
        ),
        ("Hello there, this is a test of the package.", 0.0),
        (long, default_temperature),
    ] {
        tts.options.temperature = temperature;
        let want = tts.generate(text, &voice).unwrap();
        let got = m
            .synthesize(text, &json!({"voice": "alba", "temperature": temperature}))
            .unwrap();
        assert_eq!(got.sample_rate, 24_000);
        let cos = cosine(&got.samples, &want.samples);
        eprintln!(
            "temperature {temperature}, {} chars: {} samples ({} from the package), cosine {cos}",
            text.len(),
            want.samples.len(),
            got.samples.len()
        );
        assert_eq!(got.samples.len(), want.samples.len());
        assert!(cos > 0.9999, "cosine {cos}");
    }
}
