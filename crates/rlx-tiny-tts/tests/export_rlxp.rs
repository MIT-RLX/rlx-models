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

//! An exported TinyTTS / MeloTTS bundle, opened by rlx alone, speaks as
//! `TinyTts::synthesize` does (`weights/tts/melotts`): its frontend a
//! program, its graphs — the G2P's GRU too, which a made-up word needs —
//! rlx graphs.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_tiny_tts::TinyTts;
use rlx_tiny_tts::export::export_rlxp;
use rlx_tiny_tts::model::InferOpts;
use serde_json::json;

#[test]
fn an_exported_tiny_tts_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/melotts");
    if !dir.join("onnx/decoder.onnx").is_file() {
        eprintln!("skip: no TinyTTS bundle");
        return;
    }
    // The frontend's WordPiece groups: the host's tokenizer.
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("tiny-tts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = TinyTts::load(dir.as_path()).unwrap();
    let opts = InferOpts::from_config(runner.config());
    for text in [
        "Hello world, this is a small voice.",
        "Zorblax met 42 friends on the 3rd of May.",
    ] {
        let want = runner.synthesize(text, &opts).unwrap();
        let got = m.synthesize(text, &json!({})).unwrap();
        assert_eq!(got.sample_rate, want.sample_rate);
        assert_eq!(got.samples.len(), want.samples.len(), "{text:?}");
        let worst = got
            .samples
            .iter()
            .zip(&want.samples)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.samples.iter().fold(0f32, |a, x| a.max(x.abs()));
        eprintln!(
            "{text:?}: {} samples, off by {worst} (peak {peak})",
            want.samples.len()
        );
        assert!(worst < 1e-5, "{text:?}: off by {worst}");
    }
}
