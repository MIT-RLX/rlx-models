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

//! End-to-end ONNX synthesis when weights are present.

#![cfg(feature = "onnx")]

mod support;

use std::path::PathBuf;

use support::{assert_audible, duration_secs};

use rlx_kittentts::{Device, KittenTTS, assets};

#[test]
fn synthesize_ipa_to_wav() {
    let Ok(model_dir) = assets::default_model_dir() else {
        eprintln!("skip e2e_onnx: run `just fetch-kittentts` first");
        return;
    };
    let layout = assets::ModelLayout::resolve(&model_dir).expect("layout");
    let tts = KittenTTS::load_from_dir(&model_dir, Device::Cpu).expect("load");
    let voice = layout
        .voice_names()
        .expect("voices")
        .into_iter()
        .find(|v| v == "Jasper")
        .expect("Jasper voice");
    let audio = tts
        .generate_from_ipa("həˈloʊ", &voice, 1.0, 6)
        .expect("infer");
    assert_audible(&audio, 1_000);

    let out = std::env::temp_dir().join("rlx_kittentts_e2e_onnx.wav");
    tts.write_wav(&audio, &out).expect("write wav");
    let meta = std::fs::metadata(&out).expect("wav metadata");
    assert!(meta.len() > 1000, "wav file too small");
    eprintln!(
        "e2e onnx: {} samples ({:.2}s) -> {}",
        audio.len(),
        duration_secs(audio.len()),
        out.display()
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
fn cli_layout_matches_load_from_dir() {
    let Ok(model_dir) = assets::default_model_dir() else {
        return;
    };
    let layout = assets::ModelLayout::resolve(&model_dir).expect("layout");
    assert!(layout.onnx.is_file());
    assert!(layout.voices.is_file());
    let _: PathBuf = model_dir;
}
