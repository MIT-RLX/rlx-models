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

//! An exported Inflect-Nano (`weights/inflect-nano-rlx`), opened by rlx
//! alone, speaks as the crate's host reference does: the same samples for
//! two texts (one with numbers and a homograph), one faster.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_inflect_nano::InflectNano;
use rlx_inflect_nano::config::InferOpts;
use rlx_inflect_nano::export::export_rlxp;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_inflect_nano_speaks_as_the_reference_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/inflect-nano-rlx");
    if !dir.join("acoustic.safetensors").is_file() {
        eprintln!("skip: no Inflect-Nano bundle");
        return;
    }
    let tts = InflectNano::load_from_dir(&dir).unwrap();
    let cases = [
        ("Hello from a tiny text to speech model.", 1.0f32),
        ("I read 42 books last year, and I will read more.", 1.2),
    ];
    let want: Vec<Vec<f32>> = cases
        .iter()
        .map(|(t, speed)| {
            tts.synthesize(t, &InferOpts::with_speed(*speed))
                .unwrap()
                .samples
        })
        .collect();

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("inflect-nano.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for ((text, speed), want) in cases.iter().zip(&want) {
        let audio = m.synthesize(text, &json!({"speed": speed})).unwrap();
        let cos = cosine(&audio.samples, want);
        let diff = audio
            .samples
            .iter()
            .zip(want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "{text:?}: {} samples ({} from the package), max |d| {diff}, cosine {cos}",
            want.len(),
            audio.samples.len()
        );
        assert_eq!(audio.samples.len(), want.len());
        assert!(cos > 0.9999, "cosine {cos}");
    }
}
