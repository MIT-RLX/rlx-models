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

//! Production warm-infer latency gate (`native-fast` / `KITTEN_RLX_INFER=production`).

#![cfg(feature = "native-fast")]

mod support;

use std::time::Instant;

use rlx_kittentts::{Device, KittenTTS, assets, infer_opts};

#[cfg(feature = "onnx")]
use rlx_kittentts::assets::ModelLayout;

fn max_warm_infer_secs() -> f64 {
    std::env::var("KITTEN_RLX_SPEED_TEST_MAX_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0)
}

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

fn setup_production_speed_env() {
    unsafe {
        std::env::set_var("KITTEN_RLX_INFER", "production");
        std::env::remove_var("KITTEN_RLX_PARITY");
        std::env::remove_var("KITTEN_RLX_FULL_GRAPH");
        std::env::set_var("KITTEN_RLX_SKIP_PREWARM", "1");
        let aot = std::env::temp_dir().join(format!("kitten_speed_aot_{}", std::process::id()));
        std::env::set_var("KITTEN_RLX_AOT_CACHE", &aot);
    }
}

#[test]
fn native_production_warm_infer_speed() {
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip: no native weights");
        return;
    };
    let Some(voices) = voices_npz() else {
        eprintln!("skip: run `just fetch-kittentts` for voices.npz");
        return;
    };
    setup_production_speed_env();

    let ipa = "həˈloʊ";
    let token_len = rlx_kittentts::ipa_to_ids(ipa).len();
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);

    let tts = KittenTTS::load_native(
        &weights,
        &voices,
        Default::default(),
        Default::default(),
        Device::Cpu,
        seq_len,
        max_wave,
    )
    .expect("load_native");

    let voice = tts.voice_names().first().expect("voice").clone();

    // Cold compile + first infer (populate AOT cache).
    let _ = tts
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("warmup infer");

    let t0 = Instant::now();
    let audio = tts
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("warm infer");
    let elapsed = t0.elapsed().as_secs_f64();

    support::assert_audible(&audio, 500);

    let limit = max_warm_infer_secs();
    eprintln!(
        "native production warm infer: {:.3}s (limit {limit:.1}s, {} samples)",
        elapsed,
        audio.len()
    );

    assert!(
        elapsed <= limit,
        "production warm infer too slow: {elapsed:.3}s > {limit:.1}s \
         (override with KITTEN_RLX_SPEED_TEST_MAX_SECS)"
    );
}

#[cfg(feature = "onnx")]
#[test]
fn native_production_faster_than_onnx_warm_cpu() {
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip: no native weights");
        return;
    };
    let Some(model_dir) = assets::default_model_dir().ok() else {
        eprintln!("skip: run `just fetch-kittentts`");
        return;
    };
    let layout = ModelLayout::resolve(&model_dir).expect("layout");
    setup_production_speed_env();

    let ipa = "həˈloʊ";
    let token_len = rlx_kittentts::ipa_to_ids(ipa).len();
    let (seq_len, max_wave) = infer_opts::recommended_native_compile_opts(token_len);

    let native = KittenTTS::load_native(
        &weights,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        Device::Cpu,
        seq_len,
        max_wave,
    )
    .expect("load_native");
    let ort = KittenTTS::load_on(
        &layout.onnx,
        &layout.voices,
        layout.config.speed_priors.clone(),
        layout.config.voice_aliases.clone(),
        Device::Cpu,
    )
    .expect("ort");

    let voice = native.voice_names().first().expect("voice").clone();

    let _ = native
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("native warmup");
    let _ = ort
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("ort warmup");

    let t0 = Instant::now();
    let nat_audio = native
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("native warm");
    let native_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let ort_audio = ort
        .generate_from_ipa(ipa, &voice, 1.0, 6)
        .expect("ort warm");
    let ort_secs = t1.elapsed().as_secs_f64();

    support::assert_audible(&nat_audio, 500);
    support::assert_audible(&ort_audio, 500);

    eprintln!(
        "warm infer cpu: native={native_secs:.3}s onnx={ort_secs:.3}s ratio={:.2}",
        native_secs / ort_secs.max(1e-9)
    );

    assert!(
        native_secs < ort_secs,
        "native warm infer ({native_secs:.3}s) should be faster than ONNX ({ort_secs:.3}s) on CPU"
    );
}
