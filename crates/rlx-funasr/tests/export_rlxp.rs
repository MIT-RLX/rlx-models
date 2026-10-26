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

//! An exported SenseVoiceSmall (`weights/asr/sensevoice-small`), opened by
//! rlx alone, transcribes as the crate's runner does: English (the JFK Rust
//! speech) and the model's own Chinese example, with the language given
//! and detected.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_funasr::export::export_rlxp;
use rlx_funasr::sensevoice::SenseVoice;
use rlx_runtime::Device;
use serde_json::json;

#[test]
fn an_exported_sensevoice_transcribes_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/asr/sensevoice-small");
    if !dir.join("model.pt").is_file() {
        eprintln!("skip: no SenseVoiceSmall checkpoint");
        return;
    }
    let clips = [
        ("../../assets/jfk/jfk_rust_speech.wav", "auto"),
        ("../../assets/jfk/jfk_rust_speech.wav", "en"),
        ("../../weights/asr/sensevoice-small/example/zh.mp3", "auto"),
    ];
    let pcm: Vec<Vec<f32>> = clips
        .iter()
        .map(|(p, _)| {
            rlx_funasr::audio::load_mono(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(p), 16_000)
                .unwrap()
        })
        .collect();
    let want: Vec<String> = {
        let sv = SenseVoice::open(&dir, Device::Cpu).unwrap();
        clips
            .iter()
            .zip(&pcm)
            .map(|((_, lang), a)| {
                let r = sv.transcribe(a, lang, false).unwrap();
                format!("{} {}", r.tags.join(""), r.text)
            })
            .collect()
    };

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("sensevoice.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (((_, lang), a), want) in clips.iter().zip(&pcm).zip(&want) {
        let text = m.transcribe(a, 16_000, &json!({"language": lang})).unwrap();
        let want_text = want.split_once(' ').map_or("", |(_, t)| t);
        eprintln!("runner:  {want}\npackage: {text}");
        assert_eq!(text, want_text);
    }
}

/// An exported Paraformer (`weights/asr/paraformer-zh`) transcribes as the
/// runner does: its own example and SenseVoice's Chinese one.
#[test]
fn an_exported_paraformer_transcribes_as_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("weights/asr/paraformer-zh");
    if !dir.join("model.pt").is_file() {
        eprintln!("skip: no Paraformer checkpoint");
        return;
    }
    let clips = [
        "weights/asr/paraformer-zh/example/asr_example.wav",
        "weights/asr/sensevoice-small/example/zh.mp3",
    ];
    let pcm: Vec<Vec<f32>> = clips
        .iter()
        .filter(|p| root.join(p).is_file())
        .map(|p| rlx_funasr::audio::load_mono(&root.join(p), 16_000).unwrap())
        .collect();
    let want: Vec<String> = {
        let pf = rlx_funasr::paraformer::Paraformer::open(&dir, Device::Cpu).unwrap();
        pcm.iter().map(|a| pf.transcribe(a).unwrap()).collect()
    };
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("paraformer.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    for (a, want) in pcm.iter().zip(&want) {
        let text = m.transcribe(a, 16_000, &json!({})).unwrap();
        eprintln!("runner:  {want}\npackage: {text}");
        assert_eq!(&text, want);
    }
}
