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

//! An exported openWakeWord, opened by rlx alone, scores each 80 ms push
//! as `OpenWakeWordEngine` does.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_openwakeword::export::export_rlxp;
use rlx_openwakeword::{OpenWakeWordEngine, OpenWakeWordWeights, WakeConfig, WakeEngine};
use rlx_wake::{OWW_CHUNK_SAMPLES, parse_wav_mono_f32, resample_linear};
use serde_json::json;

#[test]
fn an_exported_openwakeword_scores_as_the_engine_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (rate, pcm) =
        parse_wav_mono_f32(&std::fs::read(root.join("assets/jfk/jfk_rust_speech.wav")).unwrap())
            .unwrap();
    let pcm = resample_linear(&pcm, rate, 16_000);
    let pcm = &pcm[..pcm.len() - 700];

    // The stub's shapes, its weights scaled so the scores move.
    let mut w = OpenWakeWordWeights::stub("hey rlx");
    for v in [
        &mut w.embed.conv1_w,
        &mut w.embed.conv2_w,
        &mut w.embed.conv3_w,
        &mut w.embed.fc_w,
        &mut w.phrase.fc1_w,
        &mut w.phrase.fc2_w,
    ] {
        v.iter_mut().for_each(|x| *x *= 30.0);
    }
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("oww");
    w.save_dir(&dir).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"keyword": "hey rlx"}"#).unwrap();

    let mut engine = OpenWakeWordEngine::new(
        OpenWakeWordWeights::load_dir(&dir, "hey rlx").unwrap(),
        WakeConfig::default(),
    );
    let want: Vec<f32> = pcm
        .chunks(OWW_CHUNK_SAMPLES)
        .map(|c| engine.push_pcm(c).unwrap()[0].score)
        .collect();

    let out = tmp.path().join("oww.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let audio = rlx_runtime::model::Tensor::f32("audio", vec![pcm.len()], pcm);
    let got = m
        .run(&[audio], &json!({"entry": "detect", "sample_rate": 16_000}))
        .unwrap();
    let scores: Vec<f32> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(scores.len(), want.len());
    let diff = scores
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let live = &want[want.len() - 6..];
    eprintln!(
        "{} pushes, max |d| {diff}; last scores {live:?}",
        want.len()
    );
    // The engine's DFT is naive f32 (each angle `ang0 · t` rounded in f32);
    // the package's bases are computed in f64. The 30× weights amplify that.
    assert!(diff < 1e-3, "max |d| {diff}");
}
