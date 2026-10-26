// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! An exported MetaVoice-1B (`weights/tts/metavoice`, EnCodec from
//! `weights/tts/encodec24`), opened by rlx alone, speaks as `MetaVoice`
//! does: the bundled voice's embedding; the first stage's tokens, greedy and
//! sampled, until a near tie between the runner's eager f32 and the
//! package's graphs; the second stage's codes and the speech on the
//! runner's tokens.
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_metavoice::export::export_rlxp;
use rlx_metavoice::{DEFAULT_ENCODEC_PATH, Device, InferOpts, MetaVoice};
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn ints(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect()
}

#[test]
fn an_exported_metavoice_speaks_as_the_runner_does() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("weights/tts/metavoice");
    let enc = root.join(DEFAULT_ENCODEC_PATH);
    if !dir.join("first_stage.safetensors").is_file() || !enc.is_file() {
        eprintln!("skip: no MetaVoice checkpoint or EnCodec");
        return;
    }
    rlx_ir::env::set("RLX_ENCODEC_PATH", enc.to_str().unwrap());
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("metavoice.rlxp");
    eprintln!(
        "{}",
        export_rlxp(&dir, &out, &ExportOptions::default()).unwrap()
    );
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mv = MetaVoice::open_with_encodec(&dir, &enc, Device::Cpu).unwrap();

    // The bundled voice.
    let spk = mv.resolve_speaker_emb(None).unwrap();
    let got = m
        .run(&[], &json!({"entry": "speak", "stage": "speaker"}))
        .unwrap();
    let got: Vec<f32> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let cos = cosine(&got, &spk);
    eprintln!("speaker: cosine {cos}");
    assert!(cos > 0.9999, "speaker cosine {cos}");

    let text = "The quick brown fox jumps over the lazy dog.";
    let new = 40;
    for greedy in [true, false] {
        let opts = InferOpts {
            max_new_tokens: new,
            greedy,
            ..InferOpts::default()
        };
        let want = mv.generate_tokens(text, &spk, &opts).unwrap();
        let got = m
            .run(
                &[],
                &json!({"entry": "speak", "text": text, "speaker": spk,
                        "max_new_tokens": new, "greedy": greedy}),
            )
            .unwrap();
        let tokens = ints(&got[0].data);
        let same = tokens.iter().zip(&want).take_while(|(a, b)| a == b).count();
        let prompt = mv.encode_text(text).unwrap().len();
        eprintln!(
            "greedy {greedy}: {} of {} generated tokens the runner's ({} from the package)",
            same.saturating_sub(prompt),
            want.len() - prompt,
            tokens.len() - prompt
        );
        // Until the runner's eager f32 and the package's graphs tip a near
        // tie: greedy, 24 tokens at least; sampled, the first 8.
        assert!(
            same >= prompt + if greedy { 24 } else { 8 },
            "{same} tokens the same"
        );
    }

    // The second stage's codes, and the speech, on the runner's tokens.
    let opts = InferOpts {
        max_new_tokens: new,
        ..InferOpts::default()
    };
    let tokens = mv.generate_tokens(text, &spk, &opts).unwrap();
    let want = mv.tokens_to_codes(text, &tokens, &spk).unwrap();
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "text": text, "speaker": spk, "tokens": tokens}),
        )
        .unwrap();
    let codes = ints(&got[1].data);
    let want_flat: Vec<u32> = want.iter().flatten().copied().collect();
    assert_eq!(codes.len(), want_flat.len());
    let same = codes.iter().zip(&want_flat).filter(|(a, b)| a == b).count();
    eprintln!("codes: {same} of {} the runner's", want_flat.len());
    assert!(same * 100 >= want_flat.len() * 99, "{same} codes the same");

    let samples = mv.decode_codes(&want).unwrap();
    let audio = m
        .synthesize(text, &json!({"speaker": spk, "tokens": tokens}))
        .unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    let cos = cosine(&audio.samples, &samples);
    eprintln!(
        "{} samples ({} from the package), cosine {cos}",
        samples.len(),
        audio.samples.len()
    );
    assert_eq!(audio.samples.len(), samples.len());
    assert!(cos > 0.999, "cosine {cos}");
}
