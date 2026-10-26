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

//! An exported NeuTTS Air (`weights/tts/neutts`: its GGUF, NeuCodec's
//! decoder, the reference voices in `samples/`), opened by rlx alone:
//! NeuCodec as transformers decodes (`scripts/neucodec_reference.py`'s
//! `neucodec_reference.json`) — and the crate's own decoder too — the
//! backbone's codes the runner's, and text through the program's phonemizer
//! to speech.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_neutts::BackboneModel;
use rlx_neutts::export::export_rlxp;
use rlx_runtime::Device;
use serde_json::{Value, json};

fn weights() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/neutts")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn ints(v: &Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_i64().unwrap() as i32)
        .collect()
}

#[test]
fn an_exported_neutts_speaks_as_the_reference_does() {
    let gguf = weights().join("neutts-air-Q4_0.gguf");
    let decoder = weights().join("neucodec_decoder.safetensors");
    let fixture = weights().join("neucodec_reference.json");
    let voice = weights().join("samples/jo.npy");
    if !gguf.is_file() || !decoder.is_file() || !fixture.is_file() || !voice.is_file() {
        eprintln!("skip: no NeuTTS GGUF, NeuCodec decoder, reference or voices");
        return;
    }
    rlx_text::install_model_tokenizer();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("neutts.rlxp");
    let opts = ExportOptions {
        uppers: vec![1024],
        ..ExportOptions::default()
    };
    eprintln!("{}", export_rlxp(&gguf, &out, &opts).unwrap());
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    // NeuCodec: the reference's codes, through the package's graphs.
    let want: Value = serde_json::from_slice(&std::fs::read(&fixture).unwrap()).unwrap();
    let codes = ints(&want["codes"]);
    let audio: Vec<f32> = want["audio"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_f64().unwrap() as f32)
        .collect();
    let got = m.synthesize("-", &json!({"codes": codes})).unwrap();
    assert_eq!(got.sample_rate, 24_000);
    assert_eq!(got.samples.len(), audio.len());
    let cos = cosine(&got.samples, &audio);
    let max = got
        .samples
        .iter()
        .zip(&audio)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let eager = rlx_neutts::NeuCodecDecoder::from_file(&decoder)
        .unwrap()
        .decode(&codes)
        .unwrap();
    let eager_cos = cosine(&eager, &audio);
    eprintln!(
        "NeuCodec: cosine {cos}, max |Δ| {max:.3e} (the crate's eager decoder: cosine {eager_cos})"
    );
    assert!(cos > 0.99999, "cosine {cos}");
    assert!(eager_cos > 0.99999, "the eager decoder: cosine {eager_cos}");

    // The backbone: the runner's codes for the same prompt and seed, until
    // the two LMs' float rounding tips a near tie.
    let ref_codes = rlx_neutts::export::read_npy_codes(&voice).unwrap();
    let ref_ipa = "soʊ aɪ dʒʌst tɹaɪd njuːfɑːnɪk ænd aɪm dʒɛnjuːɪnli ɪmpɹɛst.";
    let ipa = "ɡʊd mˈɔːɹnɪŋ, ðə tˈiː ɪz ɹˈɛdi.";
    let (seed, new) = (11u32, 28u32);
    let mut backbone = BackboneModel::load_greedy_parity_on(&gguf, 2048, Device::Cpu).unwrap();
    backbone.seed = Some(seed);
    let prompt = rlx_neutts::build_prompt(ref_ipa, ipa, &ref_codes);
    let said = backbone.generate(&prompt, new).unwrap();
    let want = rlx_neutts::extract_ids(&said);
    if want.is_empty() {
        eprintln!("the runner said {said:?}");
    }
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "ipa": ipa, "ref_codes": ref_codes, "ref_ipa": ref_ipa,
                    "seed": seed, "max_new_tokens": new}),
        )
        .unwrap();
    let got: Vec<i32> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
    eprintln!(
        "codes: {same} of {} the runner's ({} from the package)",
        want.len(),
        got.len()
    );
    assert!(same >= 14, "{same} codes the same");

    // Text: the program's phonemizer, the default voice, to speech.
    let speech = m
        .synthesize(
            "Good morning, the tea is ready.",
            &json!({"seed": seed, "max_new_tokens": new}),
        )
        .unwrap();
    assert_eq!(speech.sample_rate, 24_000);
    assert!(!speech.samples.is_empty(), "no speech");
    assert_eq!(speech.samples.len() % 480, 0);
    eprintln!("text: {} samples", speech.samples.len());
}
