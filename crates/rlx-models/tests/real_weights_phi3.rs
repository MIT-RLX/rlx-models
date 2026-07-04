// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Real-weight tests for Phi-3-mini (`phi3` arch).
//
// ```sh
// just fetch-phi3-mini
// just test-phi3-real
// RLX_PHI3_RUN_INFERENCE=1 just test-phi3-real-inference
// ```

use rlx_models::phi::PhiRunner;
use rlx_models::run::{CompatibilityStatus, check_path};
use rlx_runtime::Device;
use std::path::PathBuf;

fn weights_path() -> Option<PathBuf> {
    std::env::var("RLX_PHI3_GGUF").ok().map(PathBuf::from)
}

#[test]
fn config_from_real_phi3_gguf() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_PHI3_GGUF");
        return;
    };
    let cfg = rlx_llama_base::LlamaBaseConfig::from_gguf_path(&path).expect("parse GGUF");
    assert_eq!(cfg.arch, "phi3");
    assert!(cfg.num_hidden_layers >= 16);
    assert!(cfg.hidden_size >= 256);
    eprintln!(
        "Phi-3 config: layers={} hidden={} heads={} kv={} ctx={} vocab={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.max_position_embeddings,
        cfg.vocab_size,
    );
}

#[test]
fn compat_check_reports_phi3_supported() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_PHI3_GGUF");
        return;
    };
    let report = check_path(&path).expect("check_path should succeed");
    match &report.status {
        CompatibilityStatus::Supported { runner, .. } => {
            assert_eq!(*runner, "phi");
            eprintln!("Phi-3: supported → runner `{runner}`");
        }
        other => panic!("expected Supported(phi), got {other:?}\n{report}"),
    }
}

#[test]
fn forward_inference_real_weights() {
    if std::env::var("RLX_PHI3_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_PHI3_RUN_INFERENCE=1");
        return;
    }
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_PHI3_GGUF");
        return;
    };

    let mut runner = PhiRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(64)
        .build()
        .expect("PhiRunner::build");

    let prompt_ids = [1u32, 42, 314];
    let n_new = 4;
    let generated = runner
        .generate_packed(&prompt_ids, n_new, |_| {})
        .expect("generate_packed");
    assert_eq!(generated.len(), n_new);
    eprintln!("Phi-3 inference: {prompt_ids:?} → {generated:?}");
}
