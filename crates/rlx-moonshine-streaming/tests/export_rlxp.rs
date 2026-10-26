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

//! An exported Moonshine v2 streaming package, opened by rlx alone,
//! transcribes as `MoonshineStreaming::transcribe` does, from the real
//! weights (`RLX_MOONSHINE_STREAMING_DIR`, or the workspace's
//! `weights/asr/moonshine-streaming-tiny`): the JFK clip, and a cut of it
//! that is not a whole number of frames.
#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use rlx_core::export::ExportOptions;
use rlx_moonshine_streaming::MoonshineStreaming;
use rlx_moonshine_streaming::export::export_rlxp;
use rlx_runtime::Device;
use serde_json::json;

fn model_dir() -> Option<PathBuf> {
    let dir = rlx_ir::env::var_os("RLX_MOONSHINE_STREAMING_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/asr/moonshine-streaming-tiny")
        });
    dir.join("model.safetensors").is_file().then_some(dir)
}

/// 16-bit PCM WAV samples in `[-1, 1]`.
fn wav(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    let mut at = 12;
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let n = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]) as usize;
        if id == b"data" {
            return b[at + 8..at + 8 + n]
                .chunks_exact(2)
                .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                .collect();
        }
        at += 8 + n + (n & 1);
    }
    panic!("{}: no data chunk", path.display());
}

#[test]
fn an_exported_moonshine_transcribes_what_the_runner_does() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: no Moonshine streaming weights");
        return;
    };
    let clip = wav(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/harness/jfk_16k.wav"));
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("moonshine.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = MoonshineStreaming::open(&dir, Device::Cpu).unwrap();
    for audio in [&clip[..], &clip[..53_017]] {
        let want = runner.transcribe(audio, None).unwrap();
        let got = m.transcribe(audio, 16_000, &json!({})).unwrap();
        eprintln!("{} samples: {got:?}", audio.len());
        assert_eq!(got, want.trim(), "{} samples", audio.len());
    }
}
