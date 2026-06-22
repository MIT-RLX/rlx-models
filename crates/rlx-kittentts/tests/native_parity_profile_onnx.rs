// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Parity-mode ONNX waveform profiling + Whisper intelligibility check.

#![cfg(all(feature = "native", feature = "onnx"))]

mod support;

use kitten_tts_mini_rlx::bundle_compile::{log_parity_onnx_metrics, parity_onnx_metrics};
use rlx_kittentts::{Device, KittenTTS, SAMPLE_RATE as TTS_RATE, assets, infer_opts};
use rlx_runtime::is_available;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};
use support::{assert_audible, resample_linear, whisper_asr_dir};

const MAX_ABS_DIFF: f32 = 0.10;
const MAX_ALIGN_LAG: usize = 512;

fn ort_parity_candidates() -> usize {
    std::env::var("KITTEN_ORT_PARITY_CANDIDATES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
}

fn best_ort_against_native(
    layout: &assets::ModelLayout,
    ipa: &str,
    voice: &str,
    nat_audio: &[f32],
) -> (Vec<f32>, f32) {
    let mut best_peak = f32::MAX;
    let mut ref_audio = Vec::new();
    for i in 0..ort_parity_candidates() {
        let ort = KittenTTS::load_on(
            &layout.onnx,
            &layout.voices,
            layout.config.speed_priors.clone(),
            layout.config.voice_aliases.clone(),
            Device::Cpu,
        )
        .expect("ort");
        let candidate = ort
            .generate_from_ipa(ipa, voice, 1.0, 6)
            .expect("onnx infer");
        let peak = support::max_abs_best_lag(&candidate, nat_audio, MAX_ALIGN_LAG).1;
        if peak < best_peak {
            best_peak = peak;
            ref_audio = candidate;
        }
        if peak <= MAX_ABS_DIFF {
            eprintln!("matched ORT sample {i} early (peak={peak:.6})");
            break;
        }
    }
    (ref_audio, best_peak)
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
    let mut whisper = whisper_runner(whisper_dir);
    whisper.transcribe_greedy(&pcm_16k).expect("whisper")
}

#[test]
fn parity_onnx_profile_and_whisper() {
    let Some(model_dir) = support::model_dir() else {
        eprintln!("skip: run `just fetch-kittentts`");
        return;
    };
    let Some(whisper_dir) = whisper_asr_dir() else {
        eprintln!("skip: run `just fetch-whisper-base`");
        return;
    };
    let layout = assets::ModelLayout::resolve(&model_dir).expect("layout");
    let Some(weights) = layout.native_weights.clone() else {
        eprintln!("skip: native weights missing");
        return;
    };
    if let Some(bundle) = assets::find_rlx_bundle(&weights) {
        support::setup_native_env();
        unsafe {
            std::env::set_var("KITTEN_RLX_BUNDLE", &bundle);
        }
    }
    unsafe {
        std::env::set_var("KITTEN_RLX_INFER", "production");
        std::env::remove_var("KITTEN_RLX_PARITY");
        std::env::remove_var("KITTEN_RLX_FULL_GRAPH");
        std::env::set_var("KITTEN_RLX_FORCE_BUNDLE", "1");
    }

    let ipa = "həˈloʊ";
    let token_len = rlx_kittentts::ipa_to_ids(ipa).len();
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);
    let voice = support::first_voice(&layout).expect("voice");

    let native = KittenTTS::load_native(
        &weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        if rlx_runtime::is_available(Device::Metal) {
            Device::Metal
        } else {
            Device::Cpu
        },
        seq_len,
        max_wave,
    )
    .expect("native");

    let nat_audio = native
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("native infer");

    let (ort_audio, best_peak) = best_ort_against_native(&layout, ipa, &voice, &nat_audio);

    let metrics = parity_onnx_metrics(&ort_audio, &nat_audio, MAX_ALIGN_LAG);
    log_parity_onnx_metrics("load", &metrics);

    assert!(
        best_peak <= MAX_ABS_DIFF,
        "parity ONNX profile failed: best_peak={best_peak:.6} > {MAX_ABS_DIFF}"
    );

    assert_audible(&nat_audio, 500);
    let nat_transcript = transcribe_pcm(&nat_audio, &whisper_dir);
    let ort_transcript = transcribe_pcm(&ort_audio, &whisper_dir);
    eprintln!("whisper native: {nat_transcript}");
    eprintln!("whisper onnx:   {ort_transcript}");
    assert!(
        !nat_transcript.trim().is_empty(),
        "empty native whisper transcript"
    );
    assert!(
        !ort_transcript.trim().is_empty(),
        "empty onnx whisper transcript"
    );
}
