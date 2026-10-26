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

//! An exported LuxTTS, opened by rlx alone, clones the fixture's voice as
//! `LuxTts::synthesize` does (`weights/tts/luxtts`): its frontend and flow
//! loop a program, its graphs — the log-mel and the ISTFT too — rlx graphs
//! symbolic in their lengths.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_luxtts::export::export_rlxp;
use rlx_luxtts::{InferOpts, LuxTts};
use rlx_runtime::Device;
use serde_json::json;

const PROMPT_TEXT: &str = "Hello from Kokoro. This is a test of speech synthesis in Rust.";

fn prompt() -> Vec<f32> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav");
    let mut r = hound::WavReader::open(p).unwrap();
    let max = (1i64 << (r.spec().bits_per_sample - 1)) as f32;
    r.samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect()
}

#[test]
fn an_exported_luxtts_clones_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/luxtts");
    if !dir.join("encoder_body.onnx").is_file() {
        eprintln!("skip: no LuxTTS checkpoint");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("luxtts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let runner = LuxTts::load_on(&dir, Device::Cpu).unwrap();
    let wav = prompt();
    // Texts of a few lengths, so the flow's frames fall in more than one
    // of its decoders (one per frames mod 4).
    for text in [
        "The river ran past the old mill and on toward the sea.",
        "A small bird sang on the fence.",
        "Seven boats sailed out at dawn, and six came home by night.",
    ] {
        let want = runner
            .synthesize(text, &wav, PROMPT_TEXT, &InferOpts::default())
            .unwrap();
        let got = m
            .synthesize(
                text,
                &json!({"prompt": {"audio": wav, "sample_rate": 24000, "text": PROMPT_TEXT}}),
            )
            .unwrap();
        assert_eq!(got.sample_rate, 24_000);
        assert_eq!(got.samples.len(), want.len(), "{text:?}");
        let worst = got
            .samples
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let peak = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        let dot: f64 = got
            .samples
            .iter()
            .zip(&want)
            .map(|(x, y)| f64::from(x * y))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
        let cos = dot / (norm(&got.samples) * norm(&want));
        eprintln!(
            "{text:?}: {} samples ({} vocoder frames), off by {worst} (peak {peak}), cosine {cos}",
            want.len(),
            want.len() / 256 + 1
        );
        // The prompt's log-mel is a matmul DFT here, an FFT in the runner
        // (7e-5 in the log domain): the flow carries it, so the samples are
        // close, not equal.
        assert!(cos > 0.999, "{text:?}: cosine {cos}");
    }
}
