// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! An exported Gepard (`weights/tts/gepard`), opened by rlx alone, speaks as
//! `GepardSynthesizer`'s eager loop does: its frames, greedy and sampled,
//! until a near tie between the runner's eager f32 and the package's
//! graphs; the decoder's speech on the runner's frames.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_gepard::export::export_rlxp;
use rlx_gepard::{GepardSynthesizer, InferOpts};
use rlx_runtime::Device;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn an_exported_gepard_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/gepard");
    if !dir.join("model.safetensors").is_file()
        || !dir.join("nano_dec_1.89kbps.safetensors").is_file()
    {
        eprintln!("skip: no Gepard checkpoint");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("gepard.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    let text = "The quick brown fox jumps over the lazy dog.";
    let frames = 40;
    for greedy in [true, false] {
        let synth = GepardSynthesizer::open_with_compiled(&dir, Device::Cpu, false)
            .unwrap()
            .with_opts(InferOpts {
                max_frames: frames,
                greedy,
                ..InferOpts::default()
            });
        let want = synth.generate_frames(text).unwrap();
        let got = m
            .run(
                &[],
                &json!({"entry": "speak", "text": text, "max_frames": frames,
                        "greedy": greedy}),
            )
            .unwrap();
        let codes: Vec<u32> = got[0]
            .data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
            .collect();
        let want_codes: Vec<u32> = want.iter().flatten().copied().collect();
        let same = codes
            .iter()
            .zip(&want_codes)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "greedy {greedy}: {} of {} frames the runner's ({} from the package)",
            same / 32,
            want.len(),
            codes.len() / 32
        );
        // Until the runner's eager f32 and the package's graphs tip a near
        // tie: greedy, 24 frames at least; sampled, the first 4.
        assert!(
            same >= 32 * if greedy { 24 } else { 4 },
            "{} frames the same",
            same / 32
        );

        // The decoder on the runner's frames.
        let samples = synth.decode_frames(&want).unwrap();
        let audio = m.synthesize(text, &json!({"frames": want})).unwrap();
        assert_eq!(audio.sample_rate, 22_050);
        assert_eq!(audio.samples.len(), samples.len());
        let cos = cosine(&audio.samples, &samples);
        eprintln!("greedy {greedy}: {} samples, cosine {cos}", samples.len());
        assert!(cos > 0.9999, "cosine {cos}");
    }
}
