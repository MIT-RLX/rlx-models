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

//! An exported Kyutai TTS (`weights/tts/kyutai-tts`, Mimi from
//! `weights/tts/kyutai-mimi`, the alba-mackenna voice), opened by rlx alone,
//! speaks as the crate's runner does: greedy, its Mimi frames and speech
//! the runner's; sampled with the runner's seed, its frames until a near tie.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_kyutai_tts::export::export_with;
use rlx_kyutai_tts::{GenerationConfig, KyutaiTtsSession};
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_kyutai_tts_speaks_as_the_runner_does() {
    let w = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts");
    let (dir, mimi, voices) = (
        w.join("kyutai-tts"),
        w.join("kyutai-mimi"),
        w.join("kyutai-tts-voices"),
    );
    let voice = voices.join("alba-mackenna/casual.wav.1e68beda@240.safetensors");
    if !voice.is_file() || !mimi.join("model.safetensors").is_file() {
        eprintln!("skip: no Kyutai TTS checkpoint, Mimi and voice");
        return;
    }
    rlx_text::install_model_tokenizer();
    let text = "Hello from Kyutai, running on RLX.";
    let greedy = GenerationConfig {
        text_temperature: 0.0,
        audio_temperature: 0.0,
        ..GenerationConfig::default()
    };
    let (sampled, argmax) = {
        let mut s = KyutaiTtsSession::open(&dir, &mimi).unwrap();
        s.set_voices_dir(&voices);
        let a = s.generate(text, &GenerationConfig::default()).unwrap();
        let b = s.generate(text, &greedy).unwrap();
        (a, b)
    };

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("kyutai-tts.rlxp");
    eprintln!(
        "{}",
        export_with(&dir, &mimi, &voice, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut run = |options: serde_json::Value| -> (Vec<Vec<u32>>, Vec<f32>) {
        let mut body = json!({"entry": "speak", "text": text});
        body.as_object_mut()
            .unwrap()
            .extend(options.as_object().unwrap().clone());
        let got = m.run(&[], &body).unwrap();
        let books = got[0].shape[1];
        let frames = got[0]
            .data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
            .collect::<Vec<_>>()
            .chunks(books)
            .map(|c| c.to_vec())
            .collect();
        let audio = got[1]
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        (frames, audio)
    };
    let same = |a: &[Vec<u32>], b: &[Vec<u32>]| a.iter().zip(b).take_while(|(x, y)| x == y).count();

    // Greedy: every frame, and the speech.
    let (frames, audio) = run(json!({"text_temperature": 0.0, "audio_temperature": 0.0}));
    let cos = cosine(&audio, &argmax.samples);
    eprintln!(
        "greedy: {} of {} frames the runner's ({} from the package), cosine {cos}",
        same(&frames, &argmax.audio_frames),
        argmax.audio_frames.len(),
        frames.len()
    );
    assert_eq!(frames, argmax.audio_frames);
    assert!(cos > 0.999, "cosine {cos}");

    // Sampled with the runner's seed: the frames until the graphs'
    // rounding tips a near tie (22 of 34 here), and as many frames.
    let (frames, _) = run(json!({}));
    let n = same(&frames, &sampled.audio_frames);
    eprintln!(
        "sampled: {n} of {} frames the runner's ({} from the package)",
        sampled.audio_frames.len(),
        frames.len()
    );
    assert!(n >= 16, "only {n} frames the runner's");
    assert_eq!(frames.len(), sampled.audio_frames.len());
}
