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

//! Parity guard for the native (Rust/RLX) Voxtral audio frontend.
//!
//! The reference JSON is a one-time dump of the HF `VoxtralProcessor`
//! (`{n_mels, n_frames, mel: [...], tokens: [...]}`). The native frontend must
//! reproduce it without any Python at test time. Skips when the dump/wav are
//! absent so CI without fixtures stays green.
//!
//! Refresh the dump (needs the venv built in `just`-land, one-off):
//!   python crates/rlx-voxtral/scripts-ref/mel_preprocess.py \
//!     --model-dir .cache/voxtral/Voxtral-Mini-3B-2507 \
//!     --wav .cache/whisper-bench/jfk_16k.wav --json > .cache/voxtral_ref_jfk.json

use rlx_voxtral::{VoxtralAudioConfig, VoxtralConfig, pcm_to_mel, transcription_prompt_ids};
use std::path::PathBuf;

fn ref_path() -> PathBuf {
    std::env::var("RLX_VOXTRAL_REF")
        .unwrap_or_else(|_| ".cache/voxtral_ref_jfk.json".into())
        .into()
}

fn wav_path() -> PathBuf {
    std::env::var("RLX_VOXTRAL_WAV")
        .unwrap_or_else(|_| ".cache/whisper-bench/jfk_16k.wav".into())
        .into()
}

#[test]
fn native_mel_and_prompt_match_hf_reference() {
    let (ref_file, wav) = (ref_path(), wav_path());
    if !ref_file.is_file() || !wav.is_file() {
        eprintln!(
            "skip: missing reference {ref_file:?} or wav {wav:?} (set RLX_VOXTRAL_REF / RLX_VOXTRAL_WAV)"
        );
        return;
    }

    let payload: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_file).expect("read ref json"))
            .expect("parse ref json");
    let ref_n_mels = payload["n_mels"].as_u64().unwrap() as usize;
    let ref_n_frames = payload["n_frames"].as_u64().unwrap() as usize;
    let ref_mel: Vec<f32> = payload["mel"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let ref_tokens: Vec<u32> = payload["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    // ---- mel parity (native Whisper log-mel vs HF WhisperFeatureExtractor) ----
    let audio_cfg = VoxtralAudioConfig::mini_3b();
    let pcm = rlx_voxtral::audio::load_wav_mono_f32(&wav).expect("load wav");
    let mel = pcm_to_mel(&audio_cfg, &pcm).expect("native mel");
    assert_eq!(mel.n_mels, ref_n_mels, "n_mels mismatch");
    assert_eq!(mel.n_frames, ref_n_frames, "n_frames mismatch");
    assert_eq!(mel.data.len(), ref_mel.len(), "mel len mismatch");

    let (mut max_abs, mut sum_abs) = (0f32, 0f64);
    for (a, b) in mel.data.iter().zip(ref_mel.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / mel.data.len() as f64;
    eprintln!(
        "mel parity: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} over {} values",
        mel.data.len()
    );
    // Whisper STFT (rustfft) vs HF (torch) differ only by float rounding; observed
    // max_abs ~3e-5, mean_abs ~5e-8. Thresholds leave ~30x headroom for FFT backends.
    assert!(max_abs < 1e-3, "mel max abs diff too high: {max_abs}");
    assert!(mean_abs < 1e-5, "mel mean abs diff too high: {mean_abs}");

    // ---- transcription prompt parity (language = None, as in the dump) ----
    let cfg = VoxtralConfig::tiny_synthetic();
    let n_audio = audio_cfg.audio_token_count(mel.n_frames);
    let prompt = transcription_prompt_ids(&cfg, n_audio, None, None).expect("native prompt");
    assert_eq!(prompt, ref_tokens, "transcription prompt token mismatch");
    eprintln!("prompt parity: {} tokens, n_audio={n_audio}", prompt.len());
}
