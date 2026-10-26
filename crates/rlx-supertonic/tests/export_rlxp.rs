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

//! An exported Supertonic-3, opened by rlx alone, speaks as
//! `Supertonic::synthesize` does (`weights/tts/supertonic-3`): its frontend
//! and flow-matching loop a program, its four graphs rlx graphs symbolic
//! in the text and latent lengths.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use rlx_supertonic::export::export_rlxp;
use rlx_supertonic::{InferOpts, Supertonic, Voice};
use serde_json::json;

#[test]
fn an_exported_supertonic_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/supertonic-3");
    if !dir.join("onnx/vector_estimator.onnx").is_file() {
        eprintln!("skip: no Supertonic-3 checkpoint");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("supertonic.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = Supertonic::load_on(&dir, Device::Cpu).unwrap();
    for (text, voice, lang, seed) in [
        ("Hello world, this is Supertonic.", "F1", "en", 0u64),
        ("La lectura es un viaje sin moverse.", "M2", "es", 7),
    ] {
        let v = Voice::load(&dir.join(format!("voice_styles/{voice}.json"))).unwrap();
        let opts = InferOpts {
            seed,
            ..InferOpts::default()
        };
        let want = runner.synthesize(text, lang, &v, &opts).unwrap();
        let got = m
            .synthesize(text, &json!({"voice": voice, "lang": lang, "seed": seed}))
            .unwrap();
        assert_eq!(got.samples.len(), want.len(), "{text:?}");
        let worst = got
            .samples
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        eprintln!(
            "{text:?} ({voice}): {} samples, off by {worst} (peak {peak})",
            want.len()
        );
        // The graphs the runner imports, symbolic: the same samples.
        assert!(worst < 1e-6, "{text:?}: off by {worst}");
    }
}
