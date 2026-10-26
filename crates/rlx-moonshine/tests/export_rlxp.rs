// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A Moonshine checkpoint exported as a package transcribes what
//! `MoonshineRunner` does: the real tiny model (`RLX_MOONSHINE_DIR`, or
//! `/Users/Shared/translator/models/moonshine-tiny`; skipped where absent),
//! the JFK clip and a cut of it that is no whole number of frames.

#![cfg(all(feature = "export", feature = "tokenizer"))]

use std::path::PathBuf;

use rlx_moonshine::MoonshineRunner;
use rlx_runtime::Device;
use serde_json::json;

fn dir() -> Option<PathBuf> {
    let d = rlx_ir::env::var("RLX_MOONSHINE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/Shared/translator/models/moonshine-tiny"));
    d.join("model.safetensors").is_file().then_some(d)
}

fn wav(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    // PCM16 mono: the data chunk's samples.
    let at = b.windows(4).position(|w| w == b"data").unwrap() + 8;
    b[at..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

#[test]
fn an_exported_moonshine_transcribes_what_the_runner_does() {
    let Some(dir) = dir() else {
        eprintln!("skipped: no Moonshine checkpoint");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pcm = wav(&root.join("assets/harness/jfk_16k.wav"));
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("moonshine.rlxp");
    let report = rlx_moonshine::export::export_rlxp(&dir, &out, &Default::default()).unwrap();
    eprintln!("{report}");
    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = MoonshineRunner::builder()
        .weights(dir.join("model.safetensors"))
        .device(Device::Cpu)
        .build()
        .unwrap();
    for n in [pcm.len(), 61_111] {
        let clip = &pcm[..n];
        let want = runner.transcribe(clip).unwrap();
        let got = m.transcribe(clip, 16_000, &json!({})).unwrap();
        eprintln!("{n} samples:\n  runner:  {want}\n  package: {got}");
        assert_eq!(got.trim(), want.trim(), "{n} samples");
    }
}
