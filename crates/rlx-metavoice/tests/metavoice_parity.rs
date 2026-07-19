// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

use std::path::PathBuf;

use rlx_metavoice::{
    DEFAULT_ENCODEC_PATH, DEFAULT_LOCAL_DIR, FirstStageArgs, MetaTokenizer, MetaVoice,
    MetaVoiceConfig,
};
use rlx_runtime::Device;

#[test]
fn config_defaults_point_at_centralized_tts_weights() {
    let cfg = MetaVoiceConfig::default();
    assert_eq!(cfg.model_dir, DEFAULT_LOCAL_DIR);
}

#[test]
fn first_stage_defaults_match_hf_1b() {
    let a = FirstStageArgs::default();
    assert_eq!(a.n_layer, 24);
    assert_eq!(a.n_embd, 2048);
    assert_eq!(a.vocab_sizes, vec![2562]);
}

#[test]
fn tokenize_applies_offset() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/metavoice");
    if !dir.join("tokenizer_metavoice.json").is_file() {
        eprintln!("skip: no tokenizer");
        return;
    }
    let tok = MetaTokenizer::load(dir.join("tokenizer_metavoice.json")).expect("tok");
    let raw = tok.encode("Hi.").expect("encode");
    assert!(!raw.is_empty());
    assert!(raw.iter().all(|&i| i < 512));
    let off = tok.offset;
    assert_eq!(off, 2049);
}

#[test]
fn load_and_synth_short_when_weights_present() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/metavoice");
    let enc = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(DEFAULT_ENCODEC_PATH);
    if !dir.join("first_stage.safetensors").is_file() || !enc.is_file() {
        eprintln!("skip: missing MetaVoice / EnCodec weights");
        return;
    }
    // Keep this cheap: load is ~35s; short greedy AR is the gate.
    if std::env::var_os("RLX_METAVOICE_E2E").is_none() {
        eprintln!("skip: set RLX_METAVOICE_E2E=1 for first-stage+EnCodec e2e");
        return;
    }
    let mv = MetaVoice::open_with_encodec(&dir, &enc, Device::Cpu).expect("open");
    let opts = rlx_metavoice::InferOpts {
        max_new_tokens: 32,
        greedy: true,
        ..Default::default()
    };
    let pcm = mv.synthesize("Hi.", None, &opts).expect("synth");
    assert!(!pcm.is_empty(), "expected PCM");
    let peak = pcm.iter().fold(0f32, |m, &v| m.max(v.abs()));
    eprintln!("metavoice e2e: {} samples, peak {peak:.4}", pcm.len());
}
