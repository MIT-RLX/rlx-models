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

//! An exported MiraTTS (`weights/tts/miratts`), opened by rlx alone, as
//! `MiraTts` runs: a reference clip's global tokens, the LM's sampled
//! acoustic codes, BiCodec's audio on the runner's codes.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_miratts::codec::MiraCodec;
use rlx_miratts::export::export_rlxp;
use rlx_miratts::lm::MiraLm;
use rlx_miratts::{MiraConfig, MiraTts, tokens};
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn ints(t: &Tensor) -> Vec<u32> {
    t.data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect()
}

#[test]
fn an_exported_miratts_speaks_as_the_runner_does() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/miratts");
    let voice = PathBuf::from("/Users/Shared/neutts-rs/samples/jo.wav");
    if !dir.join("model.safetensors").is_file() || !voice.is_file() {
        eprintln!("skip: no MiraTTS checkpoint or reference voice");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("miratts.rlxp");
    let opts = ExportOptions {
        uppers: vec![512],
        ..ExportOptions::default()
    };
    eprintln!("{}", export_rlxp(&dir, &out, &opts).unwrap());
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    // A reference at 16 kHz (the sample's 22.05 kHz, linearly resampled).
    let mut r = hound::WavReader::open(&voice).unwrap();
    let sr = r.spec().sample_rate as f64;
    let raw: Vec<f32> = r
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect();
    let n = (raw.len() as f64 * 16_000.0 / sr) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let s = i as f64 * sr / 16_000.0;
            let j = s.floor() as usize;
            let f = (s - j as f64) as f32;
            let (a, b) = (raw[j.min(raw.len() - 1)], raw[(j + 1).min(raw.len() - 1)]);
            a + (b - a) * f
        })
        .collect();

    // Its global tokens.
    let mut tts = MiraTts::load(&dir, Device::Cpu).unwrap();
    let want_ctx = tts.encode_ref(&pcm).unwrap();
    let got = m
        .run(
            &[Tensor::f32("reference", vec![pcm.len()], &pcm)],
            &json!({"entry": "speak"}),
        )
        .unwrap();
    let got_ctx = ints(&got[0]);
    let same = got_ctx
        .iter()
        .zip(&want_ctx)
        .filter(|(a, b)| a == b)
        .count();
    eprintln!("global tokens: {same} of {} the runner's", want_ctx.len());
    assert!(same >= 30, "{same} global tokens the same");

    // The LM's acoustic codes, sampled.
    let text = "Good morning, the tea is ready.";
    let (seed, new) = (7u64, 40usize);
    let prompt = tokens::build_tts_prompt(&tts.encode_text(text).unwrap(), &want_ctx);
    let cfg = MiraConfig::load(&dir).unwrap_or_default();
    let want_codes = MiraLm::load(&dir, &cfg, Device::Cpu)
        .unwrap()
        .generate_speech_codes(&prompt, new, seed)
        .unwrap();
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "text": text, "context": want_ctx, "seed": seed,
                    "max_new_tokens": new}),
        )
        .unwrap();
    let got_codes = ints(&got[0]);
    let same = got_codes
        .iter()
        .zip(&want_codes)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "acoustic codes: {same} of {} the runner's ({} from the package)",
        want_codes.len(),
        got_codes.len()
    );
    assert!(same >= 10, "{same} codes the same");

    // BiCodec on the runner's codes.
    let want = MiraCodec::load(&dir.join("decoders"), Device::Cpu)
        .unwrap()
        .decode(&want_codes, &want_ctx)
        .unwrap();
    let got = m
        .synthesize(text, &json!({"codes": want_codes, "context": want_ctx}))
        .unwrap();
    assert_eq!(got.sample_rate, 16_000);
    assert_eq!(got.samples.len(), want.len());
    let cos = cosine(&got.samples, &want);
    eprintln!("{} samples, cosine {cos}", want.len());
    assert!(cos > 0.9999, "cosine {cos}");
}
