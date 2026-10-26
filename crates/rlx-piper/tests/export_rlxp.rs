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

//! An exported Piper voice, opened by rlx alone, speaks as
//! `NativeVits::synthesize` does (`weights/tts/piper`): its frontend a
//! program (espeak-ng in WebAssembly), its encoder, duration predictor,
//! prior and decoder rlx graphs symbolic in their lengths.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_piper::NativeVits;
use rlx_piper::export::export_rlxp;
use rlx_runtime::Device;
use serde_json::json;

#[test]
fn an_exported_piper_voice_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/piper");
    if !dir.join("rlx-split/flow_dec.onnx").is_file() {
        eprintln!("skip: no Piper voice");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("piper.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = NativeVits::load(&dir, Device::Cpu).unwrap();
    for (text, scale) in [
        ("Hello world, this is Piper.", None),
        ("The fox read 42 numbers aloud, slowly.", Some(1.2)),
    ] {
        let want = runner.synthesize(text, scale).unwrap();
        let opts = match scale {
            Some(s) => json!({"length_scale": s}),
            None => json!({}),
        };
        let got = m.synthesize(text, &opts).unwrap();
        assert_eq!(got.sample_rate, 22_050);
        assert_eq!(got.samples.len(), want.len(), "{text:?}");
        let worst = got
            .samples
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        eprintln!(
            "{text:?}: {} samples, off by {worst} (peak {peak})",
            want.len()
        );
        assert!(worst < 1e-5, "{text:?}: off by {worst}");
    }
    // Punctuation alone is four ids: an encoder of its own.
    for text in [".", "?!"] {
        match (
            runner.synthesize(text, None),
            m.synthesize(text, &json!({})),
        ) {
            (Ok(want), Ok(got)) => {
                assert_eq!(got.samples.len(), want.len(), "{text:?}");
                let worst = got
                    .samples
                    .iter()
                    .zip(&want)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                eprintln!("{text:?}: {} samples, off by {worst}", want.len());
                assert!(worst < 1e-5, "{text:?}: off by {worst}");
            }
            (Err(a), Err(b)) => eprintln!("{text:?}: both refuse ({a} / {b})"),
            (a, b) => panic!(
                "{text:?}: the runner gives {:?}, the package {:?}",
                a.is_ok(),
                b.is_ok()
            ),
        }
    }
}
