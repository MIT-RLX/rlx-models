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

//! Plain text → espeak → WAV (requires `espeak` + `onnx`).

#![cfg(all(feature = "onnx", feature = "espeak"))]

mod support;

use support::{assert_audible, duration_secs, load_model_on, model_dir};

const LONG_TEXT: &str =
    "This is a longer sentence for testing the kitten text to speech system in Rust.";

#[test]
fn synthesize_text_to_wav() {
    let Some(dir) = model_dir() else {
        eprintln!("skip e2e_text: run `just fetch-kittentts` first");
        return;
    };
    assert!(
        rlx_kittentts::is_espeak_available(),
        "espeak-ng should initialise with bundled-data-en"
    );
    let tts = load_model_on(&dir, rlx_kittentts::Device::Cpu).expect("load");
    let audio = tts
        .generate_from_text(LONG_TEXT, "Jasper", 1.0, "en")
        .expect("generate_from_text");

    assert_audible(&audio, 40_000);

    let out = std::env::temp_dir().join("rlx_kittentts_e2e_text.wav");
    tts.write_wav(&audio, &out).expect("write wav");
    eprintln!(
        "e2e text: {} samples ({:.2}s) -> {}",
        audio.len(),
        duration_secs(audio.len()),
        out.display()
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
fn phonemize_english() {
    let ipa = rlx_kittentts::phonemize("Hello").expect("phonemize");
    assert!(!ipa.is_empty());
    assert!(rlx_kittentts::ipa_content_len(&ipa) > 0);
}
