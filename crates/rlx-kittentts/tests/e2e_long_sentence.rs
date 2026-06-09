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

//! Long IPA sentence → WAV with amplitude and duration checks.

#![cfg(feature = "onnx")]

mod support;

use support::{LONG_IPA, assert_audible, duration_secs, load_model_on, model_dir, style_for};

#[test]
fn synthesize_long_ipa_onnx() {
    let Some(dir) = model_dir() else {
        eprintln!("skip e2e_long_sentence: run `just fetch-kittentts` first");
        return;
    };
    let tts = load_model_on(&dir, rlx_kittentts::Device::Cpu).expect("load");
    let voice = "Jasper";
    let style = style_for(LONG_IPA);
    let audio = tts
        .generate_from_ipa(LONG_IPA, voice, 1.0, style)
        .expect("infer long sentence");

    // ONNX: ~116k samples (~4.8 s) at peak ~0.5
    assert_audible(&audio, 80_000);

    let out = std::env::temp_dir().join("rlx_kittentts_e2e_long.wav");
    tts.write_wav(&audio, &out).expect("write wav");
    let meta = std::fs::metadata(&out).expect("wav metadata");
    assert!(
        meta.len() > 80_000,
        "wav file too small: {} bytes",
        meta.len()
    );
    eprintln!(
        "e2e long onnx: {} samples ({:.2}s) style={style} -> {}",
        audio.len(),
        duration_secs(audio.len()),
        out.display()
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
fn rejects_silent_tokenization() {
    let Some(dir) = model_dir() else {
        return;
    };
    let tts = load_model_on(&dir, rlx_kittentts::Device::Cpu).expect("load");
    let err = tts
        .generate_from_ipa("你好世界", "Jasper", 1.0, 1)
        .expect_err("CJK should not synthesize");
    assert!(
        err.to_string().contains("no phoneme"),
        "unexpected error: {err}"
    );
}
