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

//! Native KittenTTS → Whisper ASR intelligibility check.

#![cfg(feature = "native")]

mod support;

use std::sync::Mutex;

use rlx_kittentts::{Device, KittenTTS, SAMPLE_RATE as TTS_RATE, assets};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};
use support::{
    LONG_IPA, assert_audible, resample_linear, style_for, transcript_covers_reference,
    whisper_asr_dir,
};

/// Process-global mel/wave caps race when two native compiles run in parallel.
static NATIVE_WHISPER_LOCK: Mutex<()> = Mutex::new(());

const HELLO_IPA: &str = "həˈloʊ";
const HELLO_REF: &str = "hello";

const LONG_REF: &str = "kitten text to speech system rust";

fn voices_npz() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("KITTEN_VOICES_NPZ") {
        let p = std::path::PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    assets::default_model_dir()
        .ok()
        .and_then(|dir| assets::ModelLayout::resolve(&dir).ok())
        .map(|l| l.voices)
        .filter(|p| p.is_file())
}

fn whisper_runner(dir: &std::path::Path) -> WhisperRunner {
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper runner")
}

fn transcribe_tts_pcm(pcm_24k: &[f32], whisper_dir: &std::path::Path) -> String {
    let pcm_16k = resample_linear(pcm_24k, TTS_RATE, WHISPER_RATE as u32);
    assert!(
        pcm_16k.len() >= WHISPER_RATE / 2,
        "resampled audio too short for Whisper"
    );
    let mut whisper = whisper_runner(whisper_dir);
    whisper
        .transcribe_greedy(&pcm_16k)
        .expect("whisper transcribe")
}

fn load_tts(seq: usize, max_wave: usize) -> Option<KittenTTS> {
    let weights = assets::default_native_weights_dir()?;
    let voices = voices_npz()?;
    support::setup_native_smoke_env();
    Some(
        KittenTTS::load_native(
            &weights,
            &voices,
            Default::default(),
            Default::default(),
            Device::Cpu,
            seq,
            max_wave,
        )
        .expect("load_native"),
    )
}

#[test]
fn native_hello_via_whisper() {
    let _guard = NATIVE_WHISPER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(whisper_dir) = whisper_asr_dir() else {
        eprintln!("skip: run `just fetch-whisper-base`");
        return;
    };
    // Size compile to the IPA chunk (same as CLI auto opts), not the legacy (128, 48k) floor.
    let token_len = rlx_kittentts::ipa_to_ids(HELLO_IPA).len().max(1);
    let (seq, max_wave) = rlx_kittentts::recommended_native_compile_opts(token_len);
    let Some(tts) = load_tts(seq, max_wave) else {
        eprintln!("skip: need kitten native weights + voices.npz");
        return;
    };
    eprintln!("whisper weights: {}", whisper_dir.display());
    eprintln!("native compile opts: seq={seq} max_wave={max_wave}");

    let voice = tts
        .voice_names()
        .iter()
        .find(|v| v.as_str() == "Jasper")
        .cloned()
        .or_else(|| tts.voice_names().first().cloned())
        .expect("voice");
    let audio = tts
        .generate_from_ipa(HELLO_IPA, &voice, 1.0, 6)
        .expect("infer hello");
    assert_audible(&audio, 500);

    let transcript = transcribe_tts_pcm(&audio, &whisper_dir);
    eprintln!("reference: {HELLO_REF}");
    eprintln!("whisper:   {transcript}");

    let lower = transcript.to_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("halo") || lower.contains("helo"),
        "expected hello-like transcript, got: {transcript}"
    );
}

#[test]
fn native_long_ipa_via_whisper() {
    let _guard = NATIVE_WHISPER_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(whisper_dir) = whisper_asr_dir() else {
        eprintln!("skip: run `just fetch-whisper-base`");
        return;
    };
    let token_len = rlx_kittentts::ipa_to_ids(LONG_IPA).len().max(1);
    let (seq, max_wave) = rlx_kittentts::recommended_native_compile_opts(token_len);
    let Some(tts) = load_tts(seq, max_wave) else {
        eprintln!("skip: need kitten native weights + voices.npz");
        return;
    };
    eprintln!("whisper weights: {}", whisper_dir.display());
    eprintln!("native compile opts: seq={seq} max_wave={max_wave} token_len={token_len}");

    let voice = tts
        .voice_names()
        .iter()
        .find(|v| v.as_str() == "Jasper")
        .cloned()
        .or_else(|| tts.voice_names().first().cloned())
        .expect("voice");
    let style = style_for(LONG_IPA);
    let audio = tts
        .generate_from_ipa(LONG_IPA, &voice, 1.0, style)
        .expect("infer long");
    assert_audible(&audio, 40_000);

    let transcript = transcribe_tts_pcm(&audio, &whisper_dir);
    eprintln!("reference: {LONG_REF}");
    eprintln!("whisper:   {transcript}");

    assert!(
        transcript_covers_reference(LONG_REF, &transcript, 0.45),
        "Whisper transcript missed IPA synthesis content.\nref: {LONG_REF}\ngot: {transcript}"
    );
    let lower = transcript.to_lowercase();
    assert!(
        lower.contains("speech") || lower.contains("text") || lower.contains("system"),
        "expected intelligible content in transcript, got: {transcript}"
    );
}
