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

//! `config.json` of the dual-AR checkpoints: S2-Pro and Audio8.
//! An exported Audio8-TTS-Realistic-Air (`weights/tts/audio8-realistic-air`),
//! opened by rlx alone, speaks as the crate's runner does: its codes, with and
//! without a reference voice, until a near tie between the runner's graphs
//! and the package's, and the codec's speech on the runner's codes.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_fish::audio8::{SynthOptions, synthesize_codes};
use rlx_fish::codec::FishCodec;
use rlx_fish::export::export_audio8_rlxp;
use rlx_fish::s2::{Reference, S2Model};
use rlx_fish::{S2Options, audio::read_wav_44k};
use rlx_runtime::Device;
use serde_json::json;

const JFK: &str = "And so my fellow Americans, ask not what your country can do for you, \
                   ask what you can do for your country.";

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

/// Frames from the first on where all ten codes agree.
fn same_frames(got: &[u8], want: &[Vec<u32>]) -> (usize, usize) {
    let flat: Vec<u32> = got
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b) as u32)
        .collect();
    let f = flat.len() / want.len();
    let same = (0..want[0].len().min(f))
        .take_while(|&t| (0..want.len()).all(|i| flat[i * f + t] == want[i][t]))
        .count();
    (same, f)
}

#[test]
fn an_exported_audio8_speaks_as_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("weights/tts/audio8-realistic-air");
    if !dir.join("codec.pth").is_file() {
        eprintln!("skip: no Audio8 checkpoint");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("audio8.rlxp");
    eprintln!(
        "{}",
        export_audio8_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    let text = "The quick brown fox  jumps over\nthe lazy dog.";
    let frames = 24;
    let mut codec = FishCodec::open(&dir, Device::Cpu).unwrap();
    let voice = Reference {
        text: JFK.into(),
        codes: codec
            .encode(&read_wav_44k(&root.join("assets/jfk/jfk_voice_clone.wav")).unwrap())
            .unwrap(),
    };
    codec.release();
    let (plain, cloned) = {
        let mut model = S2Model::open(&dir, Device::Cpu, S2Options::default()).unwrap();
        let enc = rlx_fish::prompt::load_tokenizer(&dir).unwrap();
        let opts = SynthOptions {
            max_new_tokens: frames,
            ..SynthOptions::from_dir(&dir).unwrap()
        };
        (
            synthesize_codes(&mut model, &enc, text, None, &opts).unwrap(),
            synthesize_codes(&mut model, &enc, text, Some(&voice), &opts).unwrap(),
        )
    };

    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "text": text, "max_new_tokens": frames}),
        )
        .unwrap();
    let (same, f) = same_frames(&got[0].data, &plain.codes);
    eprintln!(
        "{same} of {} frames the runner's ({f} from the package)",
        plain.codes[0].len()
    );
    assert!(same >= 4, "{same} frames the same");

    let got = m
        .run(
            &[],
            &json!({
                "entry": "speak", "text": text, "max_new_tokens": frames,
                "reference_text": voice.text, "reference_codes": voice.codes,
            }),
        )
        .unwrap();
    let (same, f) = same_frames(&got[0].data, &cloned.codes);
    eprintln!(
        "cloned: {same} of {} frames the runner's ({f} from the package)",
        cloned.codes[0].len()
    );
    assert!(same >= 4, "{same} cloned frames the same");

    // The codec on the runner's codes.
    let samples = codec.decode(&plain.codes).unwrap();
    let audio = m.synthesize(text, &json!({"codes": plain.codes})).unwrap();
    assert_eq!(audio.sample_rate, 44_100);
    assert_eq!(audio.samples.len(), samples.len());
    let cos = cosine(&audio.samples, &samples);
    eprintln!("{} samples, cosine {cos}", samples.len());
    assert!(cos > 0.9999, "cosine {cos}");
}
