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

//! An exported Voxtral-4B-TTS (`weights/tts/voxtral-tts`, or
//! `RLX_VOXTRAL_TTS_DIR`), opened by rlx alone, speaks as `VoxtralTtsRunner`
//! does: its frames' codes, until a near tie between the runner's f32 and
//! the package's graphs, and the codec's speech on the runner's frames.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_voxtral_tts::export::export_rlxp;
use rlx_voxtral_tts::speech_tokenizer::SpeechTokenizer;
use rlx_voxtral_tts::{GenerationConfig, VoxtralTtsRunner};
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_voxtral_tts_speaks_as_the_runner_does() {
    let dir = rlx_ir::env::var_result("RLX_VOXTRAL_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/voxtral-tts")
        });
    if !dir.join("consolidated.safetensors").is_file() {
        eprintln!("skip: no Voxtral-4B-TTS checkpoint");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("voxtral-tts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    let text = "The quick brown fox jumps over the lazy dog.";
    let voice = "neutral_female";
    let frames = 24;
    let mut runner = VoxtralTtsRunner::open(&dir).unwrap();
    let tok = SpeechTokenizer::from_model_dir(&dir).unwrap();
    let prompt = tok.encode_speech(text, voice).unwrap();
    let cfg = GenerationConfig {
        max_frames: frames,
        ..GenerationConfig::default()
    };
    let want = runner
        .synthesize_native_codes(&prompt, voice, &cfg)
        .unwrap();
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "text": text, "voice": voice, "max_frames": frames}),
        )
        .unwrap();
    let codes: Vec<u32> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    let same = codes.iter().zip(&want).take_while(|(a, b)| a == b).count();
    eprintln!(
        "{} of {} frames the runner's ({} from the package)",
        same / 37,
        want.len() / 37,
        codes.len() / 37
    );
    // Until the runner's f32 and the package's graphs tip a near tie (the
    // first semantic code is one: 10.82 against 10.78), the first frames.
    assert!(same >= 37 * 4, "{} frames the same", same / 37);

    // The codec on the runner's frames.
    let n = want.len() / 37;
    let samples = runner.decode_codes_to_pcm(&want, n).unwrap();
    let rows: Vec<Vec<u32>> = want.chunks(37).map(<[u32]>::to_vec).collect();
    let audio = m.synthesize(text, &json!({"frames": rows})).unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    let cos = cosine(&audio.samples, &samples);
    eprintln!(
        "{} samples ({} from the package), cosine {cos}",
        samples.len(),
        audio.samples.len()
    );
    assert_eq!(audio.samples.len(), samples.len());
    assert!(cos > 0.9999, "cosine {cos}");
}
