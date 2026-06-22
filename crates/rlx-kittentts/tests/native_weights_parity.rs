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

//! Native weights-only graph vs RLX bundle reference (no ONNX Runtime).

#![cfg(feature = "native")]

mod support;

use rlx_kittentts::{Device, KittenTTS, assets, infer_opts};

const MAX_ABS_DIFF: f32 = 0.10;
const MIN_SAMPLES: usize = 100;
const MAX_ALIGN_LAG: usize = 512;

fn setup_parity() {
    unsafe {
        std::env::set_var("KITTENTTS_ORT_INTRA_THREADS", "1");
        std::env::set_var("KITTEN_RLX_PARITY", "1");
        std::env::remove_var("KITTEN_RLX_RNG_SEED");
        let aot = std::env::temp_dir().join(format!("kitten_native_parity_{}", std::process::id()));
        std::env::set_var("KITTEN_RLX_AOT_CACHE", &aot);
    }
}

#[test]
fn native_weights_matches_bundle_reference() {
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip: no decomposed weights under kitten_tts_mini_rlx/weights");
        return;
    };
    if !weights.join("model.safetensors").is_file() {
        eprintln!("skip: model.safetensors missing");
        return;
    }
    let bundle = weights.join("rlx_bundle");
    if !bundle.join("graph.json").is_file() {
        eprintln!("skip: rlx_bundle reference missing");
        return;
    }
    let Some(model_dir) = support::model_dir() else {
        eprintln!("skip: run `just fetch-kittentts` for voices.npz");
        return;
    };
    let layout = assets::ModelLayout::resolve(&model_dir).expect("layout");
    setup_parity();

    let voice = layout
        .voice_names()
        .ok()
        .and_then(|mut v| {
            v.sort();
            v.into_iter().next()
        })
        .unwrap_or_else(|| "default".to_string());
    let ipa = "həˈloʊ";
    let token_len = rlx_kittentts::ipa_to_ids(ipa).len();
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);

    unsafe {
        std::env::set_var("KITTEN_RLX_FORCE_BUNDLE", "1");
        std::env::set_var("KITTEN_RLX_BUNDLE", &bundle);
    }
    let reference = KittenTTS::load_native(
        &weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        Device::Cpu,
        seq_len,
        max_wave,
    )
    .expect("bundle reference");
    let ref_audio = reference
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("bundle infer");

    unsafe {
        std::env::remove_var("KITTEN_RLX_FORCE_BUNDLE");
        std::env::remove_var("KITTEN_RLX_BUNDLE");
    }
    let native = KittenTTS::load_native(
        &weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        Device::Cpu,
        seq_len,
        max_wave,
    )
    .expect("weights native");
    let nat_audio = native
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("weights infer");

    assert!(
        ref_audio.len() >= MIN_SAMPLES && nat_audio.len() >= MIN_SAMPLES,
        "too few samples: bundle={} weights={}",
        ref_audio.len(),
        nat_audio.len()
    );

    let (align_lag, max_diff) = support::max_abs_best_lag(&ref_audio, &nat_audio, MAX_ALIGN_LAG);
    let len_ratio = nat_audio.len() as f64 / ref_audio.len().max(1) as f64;

    eprintln!(
        "weights vs bundle: len bundle={} weights={} aligned_lag={align_lag} \
         max_abs_aligned={max_diff:.6} len_ratio={len_ratio:.3}",
        ref_audio.len(),
        nat_audio.len(),
    );

    assert!(
        max_diff <= MAX_ABS_DIFF,
        "weights-only native diverged from bundle reference \
         (max_abs={max_diff:.6} lag={align_lag} > {MAX_ABS_DIFF})"
    );
}
