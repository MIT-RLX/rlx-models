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

//! An exported TADA (`weights/tts/tada`, its voice `harness.tadaprompt`),
//! opened by rlx alone, speaks as the crate's runner does: the generated
//! latents and frame gaps, and the speech.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_runtime::Device;
use rlx_tada::export::export_rlxp;
use rlx_tada::head::SolveOptions;
use rlx_tada::model::TadaModel;
use rlx_tada::prompt::VoicePrompt;
use rlx_tada::synth::SynthOptions;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn floats(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn an_exported_tada_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/tada");
    let voice = dir.join("harness.tadaprompt");
    if !voice.is_file() || !dir.join("tada-codec/decoder/model.safetensors").is_file() {
        eprintln!("skip: no TADA checkpoint and voice (`just fetch-tada`)");
        return;
    }
    rlx_text::install_model_tokenizer();
    let text = "The quick brown fox jumps over the lazy dog.";
    let (want_latents, want_gaps, want_audio) = {
        let mut synth = TadaModel::open(&dir, Device::Cpu, &SolveOptions::default()).unwrap();
        let prompt = VoicePrompt::load(&voice).unwrap();
        let opts = SynthOptions::default();
        let (l, g) = synth.synthesize_latents(&prompt, text, &opts).unwrap();
        let audio = synth.synthesize(&prompt, text, &opts).unwrap();
        (l.iter().copied().collect::<Vec<f32>>(), g, audio)
    };

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("tada.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let got = m
        .run(&[], &json!({"entry": "speak", "text": text}))
        .unwrap();
    let latents = floats(&got[0].data);
    let gaps: Vec<u32> = got[1]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    let same = gaps
        .iter()
        .zip(&want_gaps)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "gaps: {same} of {} the runner's ({} from the package); latents cosine {}",
        want_gaps.len(),
        gaps.len(),
        cosine(&latents, &want_latents)
    );
    assert_eq!(gaps, want_gaps, "the frame gaps");
    assert_eq!(latents.len(), want_latents.len());
    assert!(cosine(&latents, &want_latents) > 0.9999);

    let audio = m.synthesize(text, &json!({})).unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    let cos = cosine(&audio.samples, &want_audio);
    eprintln!(
        "{} samples ({} from the package), cosine {cos}",
        want_audio.len(),
        audio.samples.len()
    );
    assert_eq!(audio.samples.len(), want_audio.len());
    assert!(cos > 0.999, "cosine {cos}");
}
