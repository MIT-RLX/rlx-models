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

//! Native RLX graph vs ONNX Runtime waveform parity (when model assets exist).

#![cfg(all(feature = "native", feature = "onnx"))]

mod support;

use std::sync::Mutex;

use rlx_kittentts::{Device, KittenTTS, assets, infer_opts};

static PARITY_TEST_LOCK: Mutex<()> = Mutex::new(());

const MAX_ABS_DIFF: f32 = 0.10;
const MIN_SAMPLES: usize = 100;
const MAX_ALIGN_LAG: usize = 512;
fn ort_parity_candidates() -> usize {
    std::env::var("KITTEN_ORT_PARITY_CANDIDATES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
}

/// Stochastic ONNX vocoder: take the best aligned ORT sample vs native.
fn best_ort_peak_against_native(
    layout: &assets::ModelLayout,
    ipa: &str,
    voice: &str,
    nat_audio: &[f32],
) -> (Vec<f32>, f32) {
    let mut best_peak = f32::MAX;
    let mut ref_audio = Vec::new();

    let candidates = ort_parity_candidates();
    for i in 0..candidates {
        let ort = KittenTTS::load_on(
            &layout.onnx,
            &layout.voices,
            layout.config.speed_priors.clone(),
            layout.config.voice_aliases.clone(),
            Device::Cpu,
        )
        .expect("ort");
        if i == 0 {
            eprintln!("ort backend: {}", ort.ort_ep());
        }
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

#[test]
fn native_matches_onnx_cpu() {
    let _guard = PARITY_TEST_LOCK.lock().expect("parity test lock");
    let Some(model_dir) = support::model_dir() else {
        eprintln!("skip: run `just fetch-kittentts`");
        return;
    };
    support::setup_native_parity_env();
    unsafe {
        std::env::set_var("KITTEN_RLX_FULL_GRAPH", "1");
        std::env::remove_var("KITTEN_RLX_ENABLE_NARROW_WAVEFORM_SLICE");
    }
    let layout = assets::ModelLayout::resolve(&model_dir).expect("layout");
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip: native weights missing");
        return;
    };
    if assets::find_rlx_bundle_colocated(&weights).is_none()
        && !weights.join("model.safetensors").is_file()
    {
        eprintln!(
            "skip: native weights missing (need model.safetensors or rlx_bundle under {})",
            weights.display()
        );
        return;
    }
    let weights = weights.canonicalize().unwrap_or(weights);
    let voice = support::first_voice(&layout).expect("voice");
    let ipa = "həˈloʊ";
    let token_len = rlx_kittentts::ipa_to_ids(ipa).len();
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);

    let native = KittenTTS::load_native_with_ort(
        &weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        Device::Cpu,
        seq_len,
        max_wave,
        Some(&layout.onnx),
    )
    .expect("native");

    let nat_audio = native
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("native infer");

    let (ref_audio, best_peak) = best_ort_peak_against_native(&layout, ipa, &voice, &nat_audio);

    assert!(
        ref_audio.len() >= MIN_SAMPLES && nat_audio.len() >= MIN_SAMPLES,
        "too few samples: onnx={} native={}",
        ref_audio.len(),
        nat_audio.len()
    );

    let (align_lag, max_diff_aligned) =
        support::max_abs_best_lag(&ref_audio, &nat_audio, MAX_ALIGN_LAG);
    let len_ratio = nat_audio.len() as f64 / ref_audio.len().max(1) as f64;

    eprintln!(
        "native vs onnx: len onnx={} native={} aligned_lag={align_lag} \
         max_abs_aligned={max_diff_aligned:.6} best_peak={best_peak:.6} len_ratio={len_ratio:.3}",
        ref_audio.len(),
        nat_audio.len(),
    );

    assert!(
        best_peak <= MAX_ABS_DIFF,
        "native diverged from ONNX (best_peak={best_peak:.6} aligned={max_diff_aligned:.6} \
         lag={align_lag} > {MAX_ABS_DIFF})"
    );
}

#[test]
fn native_pure_matches_onnx_cpu() {
    let _guard = PARITY_TEST_LOCK.lock().expect("parity test lock");
    let Some(model_dir) = support::model_dir() else {
        eprintln!("skip: run `just fetch-kittentts`");
        return;
    };
    support::setup_pure_native_parity_env();
    let layout = assets::ModelLayout::resolve(&model_dir).expect("layout");
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip: native weights missing");
        return;
    };
    if assets::find_rlx_bundle_colocated(&weights).is_none()
        && !weights.join("model.safetensors").is_file()
    {
        eprintln!("skip: native weights missing");
        return;
    }
    let weights = weights.canonicalize().unwrap_or(weights);
    let voice = support::first_voice(&layout).expect("voice");
    let ipa = "həˈloʊ";
    let token_len = rlx_kittentts::ipa_to_ids(ipa).len();
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);

    let native = KittenTTS::load_native_with_ort(
        &weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        Device::Cpu,
        seq_len,
        max_wave,
        Some(&layout.onnx),
    )
    .expect("native");

    let nat_audio = native
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("pure native infer");

    let (ref_audio, best_peak) = best_ort_peak_against_native(&layout, ipa, &voice, &nat_audio);

    eprintln!(
        "pure native vs onnx: len onnx={} native={} best_peak={best_peak:.6}",
        ref_audio.len(),
        nat_audio.len(),
    );

    assert!(
        best_peak <= MAX_ABS_DIFF,
        "pure native diverged from ONNX (best_peak={best_peak:.6} > {MAX_ABS_DIFF})"
    );
}
