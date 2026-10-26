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

//! An exported Voxtral Mini 3B (`weights/asr/voxtral-mini-3b`), opened by
//! rlx alone, transcribes as the crate's runner does: its transcript is the
//! runner's, on two clips.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use rlx_voxtral::export::export_rlxp;
use rlx_voxtral::{VoxtralRunner, decode_token_ids, pcm_to_mel, transcription_prompt_ids};
use serde_json::json;

/// A clip at 16 kHz (the assets are 24 kHz).
fn speech(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/jfk")
        .join(name);
    let w = rlx_nemotron_asr::wav::parse(&std::fs::read(path).unwrap()).unwrap();
    rlx_nemotron_asr::wav::resample(&w.samples, w.sample_rate, 16_000)
}

#[test]
fn an_exported_voxtral_transcribes_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/asr/voxtral-mini-3b");
    if !dir.join("tekken.json").is_file() {
        eprintln!("skip: no Voxtral checkpoint (`just fetch-voxtral`)");
        return;
    }
    let clips = ["jfk_voice_clone.wav", "jfk_rust_speech.wav"].map(speech);
    // The export first: it fails fast, the runner takes its time.
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("voxtral.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let want: Vec<String> = {
        let runner = VoxtralRunner::builder()
            .weights(&dir)
            .device(Device::Cpu)
            .build()
            .unwrap();
        let cfg = runner.config().clone();
        clips
            .iter()
            .map(|pcm| {
                let mel = pcm_to_mel(&cfg.audio_config, pcm).unwrap();
                let n = cfg.audio_config.audio_token_count(mel.n_frames);
                let prompt = transcription_prompt_ids(&cfg, n, Some("en"), Some(&dir)).unwrap();
                let ids = runner.generate(&prompt, &mel).unwrap();
                let text = decode_token_ids(Some(&dir), &ids[prompt.len()..]).unwrap();
                eprintln!("runner: {}", text.trim());
                text.trim().to_string()
            })
            .collect()
    };

    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (pcm, want) in clips.iter().zip(&want) {
        let text = m.transcribe(pcm, 16_000, &json!({})).unwrap();
        eprintln!("runner:  {want}\npackage: {text}");
        assert_eq!(&text, want);
    }
}
