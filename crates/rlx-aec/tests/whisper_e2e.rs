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

//! Whisper ASR quality on echoed vs AEC-cleaned audio (env-gated).

use rlx_aec::{AecConfig, AecSession, apply_echo, mse_improvement_db};
use rlx_whisper::audio::load_wav_mono_f32;
use rlx_whisper::bench_fixture::{
    JFK_REFERENCE, jfk_wav_path, normalize_transcript, transcripts_match,
};

fn whisper_dir() -> Option<std::path::PathBuf> {
    std::env::var("RLX_WHISPER_DIR")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let p = std::path::PathBuf::from(".cache/whisper/openai-whisper-base");
            p.exists().then_some(p)
        })
}

#[test]
fn whisper_prefers_aec_cleaned_over_echoed() {
    let Some(dir) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR or fetch whisper-base weights");
        return;
    };
    let weights = dir.join("model.safetensors");
    if !weights.exists() {
        eprintln!("skip: missing {weights:?}");
        return;
    }

    let jfk = jfk_wav_path();
    if !jfk.exists() {
        eprintln!("skip: missing JFK wav {jfk:?} (just fetch-whisper-bench)");
        return;
    }

    let clean = load_wav_mono_f32(&jfk).expect("jfk wav");
    let n = clean.len().min(16_000);
    let clean = clean[..n].to_vec();
    let far: Vec<f32> = clean.iter().map(|&s| s * 0.85).collect();
    let mic = apply_echo(&clean, &far, 160, 0.55);

    let cfg = AecConfig {
        step_size: 0.05,
        residual: false,
        ..AecConfig::default()
    };
    let mut session = AecSession::new(cfg).expect("aec");
    let cleaned = session
        .process_aligned_buffers(&mic, &far)
        .expect("aec process");

    assert!(
        mse_improvement_db(&mic, &cleaned, &clean) > 1.0,
        "AEC should move mic closer to clean"
    );

    let mut runner = rlx_whisper::WhisperRunner::builder()
        .weights(&weights)
        .device(rlx_runtime::Device::Cpu)
        .build()
        .expect("whisper");

    let echoed = runner.transcribe_greedy(&mic).expect("echoed asr");
    let heard = runner.transcribe_greedy(&cleaned).expect("cleaned asr");

    let ref_norm = normalize_transcript(JFK_REFERENCE);
    let echoed_match = transcripts_match(&ref_norm, &normalize_transcript(&echoed));
    let cleaned_match = transcripts_match(&ref_norm, &normalize_transcript(&heard));

    assert!(
        cleaned_match || !echoed_match || heard.len() >= echoed.len(),
        "cleaned={heard:?} echoed={echoed:?}"
    );
}
