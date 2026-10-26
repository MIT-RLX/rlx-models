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

//! An exported wake CNN (nanowakeword's `student_lite` layout), opened by rlx
//! alone, scores each 80 ms push as the engines' streaming does: the mel
//! frontend and the CNN fed `OWW_CHUNK_SAMPLES` at a time.

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_wake::export::export_family;
use rlx_wake::{
    MelConfig, MelFrontend, OWW_CHUNK_SAMPLES, WakeCnn, WakeCnnWeights, parse_wav_mono_f32,
    resample_linear,
};
use serde_json::json;

#[test]
fn an_exported_wake_cnn_scores_as_the_engine_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    // `student_lite`'s shapes with weights of both signs: as shipped they
    // are the stub's, all negative, so every ReLU is dead and every push
    // scores sigmoid(fc2.bias).
    let tmp = tempfile::tempdir().unwrap();
    let mut w = WakeCnnWeights::load(
        &root.join("crates/rlx-nanowakeword/weights/student_lite.safetensors"),
    )
    .unwrap();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for v in [
        &mut w.conv1_w,
        &mut w.conv1_b,
        &mut w.conv2_w,
        &mut w.conv2_b,
        &mut w.conv3_w,
        &mut w.conv3_b,
        &mut w.fc1_w,
        &mut w.fc1_b,
        &mut w.fc2_w,
    ] {
        for x in v.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *x = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6;
        }
    }
    let weights = tmp.path().join("student_lite.safetensors");
    w.save(&weights).unwrap();
    let (rate, pcm) =
        parse_wav_mono_f32(&std::fs::read(root.join("assets/jfk/jfk_voice_clone.wav")).unwrap())
            .unwrap();
    let pcm = resample_linear(&pcm, rate, 16_000);
    // Plus a partial last push.
    let pcm = &pcm[..pcm.len() - 700];

    let mut mel = MelFrontend::new(MelConfig::default());
    let mut cnn = WakeCnn::new(WakeCnnWeights::load(&weights).unwrap());
    let want: Vec<f32> = pcm
        .chunks(OWW_CHUNK_SAMPLES)
        .map(|c| cnn.push_mel_frames(&mel.push(c)))
        .collect();

    let out = tmp.path().join("wake.rlxp");
    eprintln!(
        "{}",
        export_family("nanowakeword", &weights, &out, &ExportOptions::default()).unwrap()
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
    eprintln!(
        "{} pushes, max |d| {diff}; scores {:?}",
        want.len(),
        &want[..12]
    );
    assert!(diff < 1e-4, "max |d| {diff}");
}
