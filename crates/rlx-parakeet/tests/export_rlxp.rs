// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! An exported Parakeet-TDT (`weights/asr/parakeet-tdt-0.6b-v3`), opened by
//! rlx alone, transcribes as the crate's runner does.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_nemotron_asr::wav;
use rlx_parakeet::Parakeet;
use rlx_parakeet::export::export_rlxp;
use rlx_runtime::Device;
use serde_json::json;

#[test]
fn an_exported_parakeet_transcribes_as_the_runner_does() {
    let nemo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../weights/asr/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo");
    if !nemo.is_file() {
        eprintln!("skip: no Parakeet-TDT checkpoint");
        return;
    }
    let pk = Parakeet::open(&nemo, Device::Cpu).unwrap();
    let rate = pk.config().sample_rate as u32;
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/jfk/jfk_rust_speech.wav");
    let w = wav::parse(&std::fs::read(path).unwrap()).unwrap();
    let pcm = wav::resample(&w.samples, w.sample_rate, rate);
    let want = pk.transcribe(&pcm).unwrap();
    drop(pk);

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("parakeet.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&nemo, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let got = m.transcribe(&pcm, rate, &json!({})).unwrap();
    eprintln!("runner:  {want}\npackage: {got}");
    assert_eq!(got, want);
}
