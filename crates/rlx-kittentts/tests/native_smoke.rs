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

//! Smoke test: load decomposed weights and run one native forward pass.

#![cfg(feature = "native")]

mod support;

use support::{LONG_IPA, assert_audible, style_for};

use rlx_kittentts::{Device, KittenTTS, assets};

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

#[test]
fn native_infer_smoke() {
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip native_infer_smoke: no decomposed weights (kitten_tts_mini_rlx/weights)");
        return;
    };
    let Some(voices) = voices_npz() else {
        eprintln!("skip native_infer_smoke: run `just fetch-kittentts` for voices.npz");
        return;
    };
    support::setup_native_smoke_env();

    let tts = KittenTTS::load_native(
        &weights,
        &voices,
        Default::default(),
        Default::default(),
        Device::Cpu,
        128,
        48_000,
    )
    .expect("load_native");

    let voice = tts.voice_names().first().expect("voice").clone();
    let audio = tts
        .generate_from_ipa("həˈloʊ", &voice, 1.0, 6)
        .expect("infer");
    assert_audible(&audio, 500);
    eprintln!("native smoke: {} samples", audio.len());
}

#[test]
fn native_long_sentence_smoke() {
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip native_long_sentence_smoke: no decomposed weights");
        return;
    };
    let Some(voices) = voices_npz() else {
        eprintln!("skip native_long_sentence_smoke: no voices.npz");
        return;
    };
    support::setup_native_smoke_env();

    #[cfg(feature = "onnx")]
    let tts = if let Ok(dir) = assets::default_model_dir() {
        let layout = assets::ModelLayout::resolve(&dir).expect("layout");
        if layout.onnx.is_file() {
            KittenTTS::load_native_from_dir(&dir, Device::Cpu, 256, 200_000)
                .expect("load_native_from_dir")
        } else {
            KittenTTS::load_native(
                &weights,
                &voices,
                Default::default(),
                Default::default(),
                Device::Cpu,
                256,
                200_000,
            )
            .expect("load_native")
        }
    } else {
        KittenTTS::load_native(
            &weights,
            &voices,
            Default::default(),
            Default::default(),
            Device::Cpu,
            256,
            200_000,
        )
        .expect("load_native")
    };

    #[cfg(not(feature = "onnx"))]
    let tts = KittenTTS::load_native(
        &weights,
        &voices,
        Default::default(),
        Default::default(),
        Device::Cpu,
        256,
        200_000,
    )
    .expect("load_native");

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

    #[cfg(feature = "onnx")]
    if let Ok(dir) = assets::default_model_dir() {
        let ort = KittenTTS::load_from_dir(&dir, Device::Cpu).expect("ort");
        let ort_audio = ort
            .generate_from_ipa(LONG_IPA, &voice, 1.0, style)
            .expect("ort long");
        let ids = rlx_kittentts::ipa_to_ids(LONG_IPA);
        let layout = assets::ModelLayout::resolve(&dir).expect("layout");
        let ort_sess = ort::session::Session::builder()
            .expect("ort builder")
            .with_intra_threads(1)
            .expect("ort threads")
            .commit_from_file(&layout.onnx)
            .expect("ort model");
        let voices = layout.voices;
        let style_vec = {
            use rlx_kittentts::npz::load_npz;
            let z = load_npz(&voices).expect("voices npz");
            let arr = z.get("expr-voice-2-m").expect("style matrix");
            arr.row(style).to_vec()
        };
        let ort_dur = rlx_kittentts::ort_duration::fetch_ort_duration(
            &std::sync::Mutex::new(ort_sess),
            &ids,
            &style_vec,
            1.0,
        )
        .expect("ort duration");
        let target = rlx_kittentts::infer_opts::waveform_samples_from_duration(&ort_dur, ids.len())
            .expect("ort duration target");
        eprintln!(
            "native long smoke: native={} target={} ort_wave={}",
            audio.len(),
            target,
            ort_audio.len()
        );
        assert!(
            audio.len() >= target.saturating_sub(600),
            "native underrun (native={} target={})",
            audio.len(),
            target
        );
        assert!(
            audio.len() <= ort_audio.len().saturating_add(600),
            "native overrun vs ORT (native={} ort={} duration_target={})",
            audio.len(),
            ort_audio.len(),
            target
        );
        assert_audible(&audio, target.saturating_sub(12_000).max(8_000));
    }

    #[cfg(not(feature = "onnx"))]
    assert_audible(&audio, 80_000);
}

#[test]
#[cfg(feature = "onnx")]
fn native_long_sentence_pure_smoke() {
    let Some(weights) = assets::default_native_weights_dir() else {
        eprintln!("skip native_long_sentence_pure_smoke: no decomposed weights");
        return;
    };
    let Some(dir) = assets::default_model_dir().ok() else {
        eprintln!("skip native_long_sentence_pure_smoke: no model dir");
        return;
    };
    if !assets::ModelLayout::resolve(&dir)
        .map(|l| l.onnx.is_file())
        .unwrap_or(false)
    {
        eprintln!("skip native_long_sentence_pure_smoke: ONNX missing");
        return;
    }
    support::setup_native_smoke_env();
    unsafe {
        std::env::remove_var("KITTEN_RLX_NO_ORT_DURATION");
        std::env::set_var("KITTEN_RLX_NO_ORT_WAVEFORM_FALLBACK", "1");
    }

    let tts = KittenTTS::load_native_from_dir(&dir, Device::Cpu, 256, 200_000).expect("load");
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
        .expect("pure long infer");

    let layout = assets::ModelLayout::resolve(&dir).expect("layout");
    let ort = KittenTTS::load_from_dir(&dir, Device::Cpu).expect("ort");
    let ort_audio = ort
        .generate_from_ipa(LONG_IPA, &voice, 1.0, style)
        .expect("ort long");
    eprintln!(
        "pure long smoke: native={} ort={}",
        audio.len(),
        ort_audio.len()
    );
    assert_audible(&audio, 80_000);
    assert!(
        audio.len() >= ort_audio.len().saturating_sub(24_000),
        "pure native long underrun (native={} ort={})",
        audio.len(),
        ort_audio.len()
    );
}
