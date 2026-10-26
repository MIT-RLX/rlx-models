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

//! An exported Kokoro package, opened by rlx alone, speaks as
//! `NativeKokoro::generate_from_text` does, from the real checkpoint
//! (`RLX_KOKORO_DIR`, or the workspace's `weights/tts/kokoro-82m`): its
//! frontend a program (espeak-ng in WebAssembly, reading its data from the
//! package), its encoder and decoder rlx graphs symbolic in their lengths.
#![cfg(all(feature = "export", feature = "native"))]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_kokoro::export::export_rlxp;
use rlx_kokoro::native::NativeKokoro;
use rlx_runtime::Device;
use serde_json::json;

fn model_dir() -> Option<PathBuf> {
    let dir = rlx_ir::env::var_os("RLX_KOKORO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/kokoro-82m")
        });
    dir.join("onnx/rlx-split/decoder_raw.onnx")
        .is_file()
        .then_some(dir)
}

#[test]
fn an_exported_kokoro_speaks_as_the_runner_does() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: no Kokoro checkpoint");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("kokoro.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = NativeKokoro::load(&dir, Device::Cpu).unwrap();
    for (text, voice) in [
        ("Hello world, this is Kokoro.", "af_heart"),
        ("The fox read 42 numbers aloud, slowly.", "bf_emma"),
    ] {
        let want = runner.generate_from_text(text, voice, 1.0).unwrap();
        let got = m.synthesize(text, &json!({"voice": voice})).unwrap();
        assert_eq!(got.sample_rate, 24_000);
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
