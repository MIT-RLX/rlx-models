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

    assert_audible(&audio, 80_000);
}
