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

//! An exported MioTTS, opened by rlx alone, speaks as `MioSession` does
//! (`weights/tts/miotts`, MioCodec beside it): the language model a
//! `generate` entry its program runs, sampled as the runner samples; the
//! codec rlx graphs.
#![cfg(feature = "export")]

use rlx_core::export::ExportOptions;
use rlx_miotts::export::export_rlxp;
use rlx_miotts::{GenerateOpts, MioSession, SPEECH_BASE, default_model_dir};
use rlx_runtime::Device;
use serde_json::json;

#[test]
fn an_exported_miotts_speaks_as_the_runner_does() {
    let dir = default_model_dir();
    if !dir.join("model.safetensors").is_file() {
        eprintln!("skip: no MioTTS checkpoint");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("miotts.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = MioSession::open_defaults(Device::Cpu).unwrap();
    let text = "Good morning, the tea is ready.";
    let want = runner.synthesize(text, &GenerateOpts::default()).unwrap();

    // The language model: the package's entry samples the runner's codes —
    // its streams, its settings — until the two LMs' float rounding tips a
    // near tie (a sampled decoder's ordinary divergence).
    let prompt =
        "<|im_start|>user\nGood morning, the tea is ready.<|im_end|>\n<|im_start|>assistant\n";
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let ids: Vec<i32> = tok
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&i| i as i32)
        .collect();
    let input = rlx_runtime::model::Tensor::new(
        "input_ids",
        rlx_ir::DType::I32,
        vec![1, ids.len()],
        ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    let opts = json!({"entry": "generate", "max_tokens": 16, "temperature": 0.8, "top_p": 1.0,
                      "seed": GenerateOpts::default().seed, "seed_per_step": true});
    let tokens: Vec<u32> = m.run(&[input], &opts).unwrap()[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    let codes: Vec<u32> = tokens
        .iter()
        .filter(|&&t| (SPEECH_BASE..SPEECH_BASE + 12_800).contains(&t))
        .map(|t| t - SPEECH_BASE)
        .collect();
    let n = codes.len().min(want.content_codes.len());
    assert!(n >= 12, "{n} codes compared");
    assert_eq!(
        codes[..n],
        want.content_codes[..n],
        "the first sampled codes"
    );

    // The codec: the runner's codes, through the package's graphs.
    let got = m
        .synthesize(text, &json!({"codes": want.content_codes}))
        .unwrap();
    assert_eq!(got.sample_rate, want.sample_rate);
    assert_eq!(got.samples.len(), want.samples.len());
    let dot: f64 = got
        .samples
        .iter()
        .zip(&want.samples)
        .map(|(x, y)| f64::from(x * y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    let cos = dot / (norm(&got.samples) * norm(&want.samples));
    let worst = got
        .samples
        .iter()
        .zip(&want.samples)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    eprintln!(
        "{n} sampled codes the same; {} samples ({} codes), off by {worst}, cosine {cos}",
        want.samples.len(),
        want.content_codes.len()
    );
    assert!(cos > 0.9999, "cosine {cos}");
}
