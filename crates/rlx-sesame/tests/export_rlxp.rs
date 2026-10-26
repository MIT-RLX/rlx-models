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

//! An exported Sesame CSM-1B (`weights/tts/sesame`, Mimi from
//! `rlx-mimi/.cache/mimi`), opened by rlx alone, speaks as
//! `SesameSession` does: the frames' codes, greedy and sampled, until a
//! near tie between the runner's eager f32 and the package's graphs; Mimi's
//! speech on the runner's frames.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_sesame::export::export_rlxp;
use rlx_sesame::generate::GenerateOpts;
use rlx_sesame::session::SesameSession;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_sesame_speaks_as_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("weights/tts/sesame");
    let mimi = root.join("crates/rlx-mimi/.cache/mimi");
    if !dir.join("model.safetensors").is_file() || !mimi.join("model.safetensors").is_file() {
        eprintln!("skip: no CSM checkpoint or Mimi");
        return;
    }
    rlx_ir::env::set("RLX_MIMI_DIR", mimi.to_str().unwrap());
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("sesame.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut session = SesameSession::open(&dir, &mimi).unwrap();

    let text = "Good morning, the tea is ready.";
    let frames = 4;
    for greedy in [true, false] {
        let opts = GenerateOpts {
            max_audio_frames: frames,
            greedy,
            ..GenerateOpts::default()
        };
        let want = session.synthesize(text, &opts).unwrap();
        let got = m
            .run(
                &[],
                &json!({"entry": "speak", "text": text, "max_audio_frames": frames,
                        "greedy": greedy}),
            )
            .unwrap();
        let codes: Vec<u32> = got[0]
            .data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
            .collect();
        let want_codes: Vec<u32> = want.audio_frames.iter().flatten().copied().collect();
        let same = codes
            .iter()
            .zip(&want_codes)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "greedy {greedy}: {same} of {} codes the runner's ({} from the package)",
            want_codes.len(),
            codes.len()
        );
        // Until the runner's eager f32 and the package's graphs tip a near
        // tie: greedy, three frames at least; sampled, the first codes.
        assert!(same >= if greedy { 96 } else { 8 }, "{same} codes the same");

        // Mimi on the runner's frames.
        let audio = m
            .synthesize(text, &json!({"frames": want.audio_frames}))
            .unwrap();
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.samples.len(), want.samples.len());
        let cos = cosine(&audio.samples, &want.samples);
        eprintln!(
            "greedy {greedy}: {} samples, cosine {cos}",
            want.samples.len()
        );
        assert!(cos > 0.9999, "cosine {cos}");
    }
}
