// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Real-weight tests for Gemma 3 270M Instruct (`gemma3` arch).
//
// ```sh
// just fetch-gemma3-270m
// just test-gemma3-real
// RLX_GEMMA3_RUN_INFERENCE=1 just test-gemma3-real-inference
// ```

use rlx_gemma::{GemmaRunner, gemma_cfg_from_gguf};
use rlx_models::run::{ChatTemplate, CompatibilityStatus, check_path};
use rlx_runtime::Device;
use std::path::PathBuf;

fn weights_path() -> Option<PathBuf> {
    std::env::var("RLX_GEMMA3_GGUF").ok().map(PathBuf::from)
}

#[test]
fn config_from_real_gemma3_gguf() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    let raw = rlx_gguf::GgufFile::from_path(&path).expect("open GGUF");
    let cfg = gemma_cfg_from_gguf(&raw).expect("gemma_cfg_from_gguf");
    assert_eq!(cfg.arch, rlx_gemma::GemmaArch::Gemma3);
    assert!(cfg.num_hidden_layers >= 8, "block_count too low: {cfg:?}");
    assert!(cfg.hidden_size >= 64);
    assert!(
        cfg.sliding_window.is_some(),
        "Gemma 3 270M should ship sliding_window in GGUF"
    );
    eprintln!(
        "Gemma 3 270M config: arch={:?} layers={} hidden={} sliding_window={:?} vocab={}",
        cfg.arch, cfg.num_hidden_layers, cfg.hidden_size, cfg.sliding_window, cfg.vocab_size,
    );
}

#[test]
fn compat_check_reports_gemma3_supported() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    let report = check_path(&path).expect("check_path should succeed");
    match &report.status {
        CompatibilityStatus::Supported { runner, .. } => {
            assert_eq!(*runner, "gemma");
            eprintln!("Gemma 3 270M: supported → runner `{runner}`");
        }
        other => panic!("expected Supported(gemma), got {other:?}\n{report}"),
    }
}

#[test]
fn chat_template_on_real_gemma3() {
    let Some(path) = weights_path() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };
    let Ok(template) = ChatTemplate::from_gguf(&path) else {
        eprintln!("skip: GGUF has no tokenizer.chat_template");
        return;
    };
    let msgs = vec![rlx_models::run::ChatMessage::user("hi")];
    let rendered = template
        .render(&msgs, true)
        .expect("Gemma 3 chat template should render with minijinja");
    assert!(rendered.contains("hi"), "missing user content: {rendered}");
    eprintln!(
        "Gemma 3 rendered prompt ({} bytes):\n{}",
        rendered.len(),
        rendered
    );
}

#[test]
fn forward_inference_real_weights() {
    if std::env::var("RLX_GEMMA3_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_GEMMA3_RUN_INFERENCE=1");
        return;
    }
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_GEMMA3_GGUF");
        return;
    };

    let mut runner = GemmaRunner::builder()
        .weights(&weights)
        .packed_weights(true)
        .device(Device::Cpu)
        .max_seq(64)
        .build()
        .expect("GemmaRunner::build");

    let prompt_ids = [1u32, 42, 314];
    let n_new = 4;
    let generated = runner
        .generate_packed(&prompt_ids, n_new, |_| {})
        .expect("generate_packed");
    assert_eq!(generated.len(), n_new);
    eprintln!("Gemma 3 270M inference: {prompt_ids:?} → {generated:?}");
}
