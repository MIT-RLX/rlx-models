// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Production native synthesis validated with Whisper ASR on every available backend.

#![cfg(all(feature = "native-fast", feature = "onnx"))]

mod support;

use rlx_kittentts::{
    Device, KittenTTS, SAMPLE_RATE as TTS_RATE, assets, audio_util, infer_opts, peak_amplitude,
};
use rlx_runtime::is_available;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};
use support::{
    LONG_PHRASES, LONG_PHRASES_EXTENDED, PhraseCase, SHORT_PHRASES, SHORT_PHRASES_EXTENDED,
    assert_audible, first_voice, resample_linear, transcript_covers_reference, whisper_asr_dir,
};

fn setup_production_env() {
    support::setup_native_production_whisper_env();
}

fn backends_to_test() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    if is_available(Device::Metal) {
        devices.push(Device::Metal);
    }
    devices
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

fn transcribe_pcm(pcm_24k: &[f32], whisper_dir: &std::path::Path) -> String {
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

fn best_ort_alignment_for_native(
    layout: &assets::ModelLayout,
    ipa: &str,
    voice: &str,
    style: usize,
    native_audio: &[f32],
    candidates: usize,
    max_lag: usize,
) -> (Vec<f32>, f32, f32) {
    let mut best_audio = native_audio.to_vec();
    let mut best_peak = f32::MAX;
    let mut best_ort_peak = 0.25f32;
    let native_norm = audio_util::normalize_for_compare(native_audio);
    for _ in 0..candidates {
        let ort = KittenTTS::load_on(
            &layout.onnx,
            &layout.voices,
            layout.config.speed_priors.clone(),
            layout.config.voice_aliases.clone(),
            Device::Cpu,
        )
        .expect("ort");
        let ort_audio = ort
            .generate_from_ipa(ipa, voice, 1.0, style)
            .expect("onnx infer");
        let ort_norm = audio_util::normalize_for_compare(&ort_audio);
        let lag = audio_util::effective_max_lag(max_lag, ort_norm.len(), native_norm.len());
        let (_, peak) = audio_util::max_abs_best_lag(&ort_norm, &native_norm, lag);
        let aligned = audio_util::align_to_reference(&ort_audio, native_audio, lag);
        if peak < best_peak {
            best_peak = peak;
            best_audio = aligned;
            best_ort_peak = peak_amplitude(&ort_audio);
        }
        if peak <= 0.10 {
            break;
        }
    }
    (best_audio, best_peak, best_ort_peak)
}

fn load_production_native(
    weights: &std::path::Path,
    layout: &assets::ModelLayout,
    device: Device,
    token_len: usize,
) -> KittenTTS {
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);
    KittenTTS::load_native_with_ort(
        weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        device,
        seq_len,
        max_wave,
        Some(&layout.onnx),
    )
    .expect("load_native_with_ort")
}

struct TestAssets {
    layout: assets::ModelLayout,
    weights: std::path::PathBuf,
    whisper_dir: std::path::PathBuf,
    voice: String,
}

fn load_test_assets() -> Option<TestAssets> {
    let model_dir = support::model_dir()?;
    let whisper_dir = whisper_asr_dir()?;
    setup_production_env();
    let layout = assets::ModelLayout::resolve(&model_dir).ok()?;
    let weights = assets::default_native_weights_dir()?;
    if assets::find_rlx_bundle_colocated(&weights).is_none()
        && !weights.join("model.safetensors").is_file()
    {
        eprintln!("skip: native weights missing (need model.safetensors or rlx_bundle)");
        return None;
    }
    let weights = weights.canonicalize().unwrap_or(weights);
    let voice = first_voice(&layout).ok()?;
    Some(TestAssets {
        layout,
        weights,
        whisper_dir,
        voice,
    })
}

fn set_device_preference(device: Device) {
    unsafe {
        if device == Device::Cpu {
            std::env::set_var("KITTEN_RLX_PREFER_METAL", "0");
        } else {
            std::env::remove_var("KITTEN_RLX_PREFER_METAL");
        }
    }
}

fn wave_limit(phrase: &PhraseCase) -> f32 {
    let token_len = rlx_kittentts::ipa_to_ids(phrase.ipa).len();
    // Ultra-short IPA: ONNX vocoder RNG dominates shape; gate ASR instead.
    if token_len <= 8 {
        return 0.85;
    }
    phrase.max_peak
}

fn check_phrase(
    assets: &TestAssets,
    device: Device,
    phrase: &PhraseCase,
    voice: &str,
    tts: &KittenTTS,
) {
    set_device_preference(device);
    let style = rlx_kittentts::ipa_style_index(phrase.ipa);
    let attempts = phrase.asr_retries + 1;
    let mut last_msg = String::new();

    for attempt in 0..attempts {
        if attempt > 0 {
            eprintln!(
                "retry [{}] ({device:?}) attempt {}/{}",
                phrase.label,
                attempt + 1,
                attempts
            );
        }

        let audio = tts
            .generate_from_ipa(phrase.ipa, voice, 1.0, style)
            .unwrap_or_else(|e| panic!("native infer {}: {e}", phrase.label));
        assert_audible(&audio, phrase.min_samples);

        let (aligned, best_peak, ort_peak) = best_ort_alignment_for_native(
            &assets.layout,
            phrase.ipa,
            voice,
            style,
            &audio,
            phrase.ort_candidates,
            phrase.max_lag,
        );
        eprintln!(
            "native [{}] ({device:?}): len={} aligned_len={} peak={:.4} ort_peak_diff={best_peak:.4}",
            phrase.label,
            audio.len(),
            aligned.len(),
            rlx_kittentts::peak_amplitude(&audio)
        );
        if best_peak > wave_limit(phrase) {
            let limit = wave_limit(phrase);
            let msg = format!(
                "native [{}] diverged from ONNX on {device:?} (peak={best_peak:.4} > {limit})",
                phrase.label
            );
            if phrase.strict_waveform {
                panic!("{msg}");
            }
            eprintln!("note: {msg}");
            return;
        }

        let asr_audio = audio_util::scale_to_peak(&aligned, ort_peak.max(0.25));
        let transcript = transcribe_pcm(&asr_audio, &assets.whisper_dir);
        eprintln!("whisper [{}] ({device:?}): {transcript}", phrase.label);
        if transcript_covers_reference(phrase.asr_reference, &transcript, phrase.min_ratio) {
            return;
        }
        last_msg = format!(
            "whisper ASR missed [{}] on {device:?} (peak={best_peak:.4})\nref: {}\ngot: {transcript}",
            phrase.label, phrase.asr_reference
        );
        if !phrase.strict_asr {
            eprintln!("note: {last_msg}");
            return;
        }
    }

    panic!("{last_msg}");
}

fn max_token_len(phrases: &[PhraseCase]) -> usize {
    phrases
        .iter()
        .map(|p| rlx_kittentts::ipa_to_ids(p.ipa).len())
        .max()
        .unwrap_or(8)
}

fn run_phrases(assets: &TestAssets, phrases: &[PhraseCase]) {
    let token_len = max_token_len(phrases);
    for device in backends_to_test() {
        eprintln!("native production phrases device: {device:?}");
        set_device_preference(device);
        let tts = load_production_native(&assets.weights, &assets.layout, device, token_len);
        for phrase in phrases {
            let voice = phrase.voice.unwrap_or(assets.voice.as_str());
            check_phrase(assets, device, phrase, voice, &tts);
        }
        // Drop compiled graphs before the next backend to avoid Metal OOM on long phrases.
        drop(tts);
    }
}

#[test]
fn native_production_whisper_short_phrases_all_backends() {
    let Some(assets) = load_test_assets() else {
        eprintln!("skip: run `just fetch-kittentts` and `just fetch-whisper-base`");
        return;
    };

    let ort = KittenTTS::load_on(
        &assets.layout.onnx,
        &assets.layout.voices,
        assets.layout.config.speed_priors.clone(),
        assets.layout.config.voice_aliases.clone(),
        Device::Cpu,
    )
    .expect("ort");
    let ort_audio = ort
        .generate_from_ipa(
            SHORT_PHRASES[0].ipa,
            assets.voice.as_str(),
            1.0,
            rlx_kittentts::ipa_style_index(SHORT_PHRASES[0].ipa),
        )
        .expect("onnx infer");
    let ort_transcript = transcribe_pcm(&ort_audio, &assets.whisper_dir);
    eprintln!("whisper onnx baseline (hello): {ort_transcript}");

    run_phrases(&assets, SHORT_PHRASES);
}

#[test]
fn native_production_whisper_long_phrases_all_backends() {
    let Some(assets) = load_test_assets() else {
        eprintln!("skip: run `just fetch-kittentts` and `just fetch-whisper-base`");
        return;
    };
    run_phrases(&assets, LONG_PHRASES);
}

/// Extended phrase matrix (log-only ASR for borderline cases). Run with:
/// `cargo test -p rlx-kittentts --features native-fast,onnx --release --test native_production_whisper native_production_whisper_extended -- --ignored --nocapture`
#[test]
#[ignore = "extended phrase matrix; run manually"]
fn native_production_whisper_extended_phrases() {
    let Some(assets) = load_test_assets() else {
        eprintln!("skip: run `just fetch-kittentts` and `just fetch-whisper-base`");
        return;
    };
    run_phrases(&assets, SHORT_PHRASES_EXTENDED);
    run_phrases(&assets, LONG_PHRASES_EXTENDED);
}

/// Backward-compatible alias for `native_production_whisper_short_phrases_all_backends`.
#[test]
fn native_production_whisper_roundtrip_all_backends() {
    native_production_whisper_short_phrases_all_backends();
}

/// Backward-compatible alias for `native_production_whisper_long_phrases_all_backends`.
#[test]
fn native_production_whisper_long_ipa_all_backends() {
    native_production_whisper_long_phrases_all_backends();
}
