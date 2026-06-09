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

//! KittenTTS synthesis → resample → Whisper ASR round-trip.

#![cfg(all(feature = "onnx", feature = "espeak"))]

mod support;

use rlx_kittentts::SAMPLE_RATE as TTS_RATE;
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};
use support::{
    LONG_IPA, assert_audible, load_model_on, model_dir, resample_linear,
    transcript_covers_reference, whisper_asr_dir,
};

const TEXT: &str =
    "This is a longer sentence for testing the kitten text to speech system in Rust.";

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

#[test]
fn roundtrip_text_via_whisper() {
    let Some(kitten_dir) = model_dir() else {
        eprintln!("skip: run `just fetch-kittentts`");
        return;
    };
    let Some(whisper_dir) = whisper_asr_dir() else {
        eprintln!("skip: run `just fetch-whisper-base` (or `just fetch-whisper`)");
        return;
    };
    eprintln!("whisper weights: {}", whisper_dir.display());

    let tts = load_model_on(&kitten_dir, Device::Cpu).expect("kitten load");
    let audio = tts
        .generate_from_text(TEXT, "Jasper", 1.0, "en")
        .expect("tts from text");
    assert_audible(&audio, 20_000);

    let transcript = transcribe_tts_pcm(&audio, &whisper_dir);
    eprintln!("reference: {TEXT}");
    eprintln!("whisper:   {transcript}");

    assert!(
        transcript_covers_reference(TEXT, &transcript, 0.45),
        "Whisper transcript missed too much of the reference.\nref: {TEXT}\ngot: {transcript}"
    );
    let lower = transcript.to_lowercase();
    assert!(
        lower.contains("kitten"),
        "expected 'kitten' in transcript, got: {transcript}"
    );
}

#[test]
fn roundtrip_ipa_via_whisper() {
    let Some(kitten_dir) = model_dir() else {
        eprintln!("skip: run `just fetch-kittentts`");
        return;
    };
    let Some(whisper_dir) = whisper_asr_dir() else {
        eprintln!("skip: run `just fetch-whisper-base` (or `just fetch-whisper`)");
        return;
    };
    eprintln!("whisper weights: {}", whisper_dir.display());

    let tts = load_model_on(&kitten_dir, Device::Cpu).expect("kitten load");
    let style = rlx_kittentts::ipa_style_index(LONG_IPA);
    let audio = tts
        .generate_from_ipa(LONG_IPA, "Jasper", 1.0, style)
        .expect("tts from ipa");
    assert_audible(&audio, 40_000);

    let transcript = transcribe_tts_pcm(&audio, &whisper_dir);
    eprintln!("ipa reference (approx): kitten text to speech system");
    eprintln!("whisper: {transcript}");

    let reference = "kitten text to speech system rust";
    assert!(
        transcript_covers_reference(reference, &transcript, 0.5),
        "Whisper transcript missed IPA synthesis content.\nref: {reference}\ngot: {transcript}"
    );
    // IPA synthesis can sound like "key-ton" to ASR even with larger Whisper models.
    let lower = transcript.to_lowercase();
    assert!(
        lower.contains("speech") || lower.contains("text") || lower.contains("system"),
        "expected intelligible content in transcript, got: {transcript}"
    );
}
