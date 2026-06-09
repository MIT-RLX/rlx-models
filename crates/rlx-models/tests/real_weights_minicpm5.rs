// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Real-weight tests for openbmb/MiniCPM5-1B (HF safetensors).
//
// ```sh
// just fetch-minicpm5   # Hugging Face Hub (hf-download)
// just test-minicpm5-real
// RLX_MINICPM5_RUN_INFERENCE=1 just test-minicpm5-real-inference
// ```

use rlx_minicpm5::{MiniCpm5Runner, llama_config_from_hf, minicpm5_1b_preset};
use rlx_models::run::{CompatibilityStatus, check_path};
use std::path::PathBuf;

fn weights_path() -> Option<PathBuf> {
    std::env::var("RLX_MINICPM5_WEIGHTS")
        .ok()
        .map(PathBuf::from)
}

fn tokenizer_path() -> Option<PathBuf> {
    std::env::var("RLX_MINICPM5_TOKENIZER")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            weights_path().and_then(|p| {
                let t = p.parent()?.join("tokenizer.json");
                t.is_file().then_some(t)
            })
        })
}

#[test]
fn hf_config_matches_preset() {
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_MINICPM5_WEIGHTS");
        return;
    };
    let cfg = llama_config_from_hf(&weights).expect("parse config.json");
    let preset = minicpm5_1b_preset();
    assert_eq!(cfg.hidden_size, preset.hidden_size);
    assert_eq!(cfg.num_hidden_layers, preset.num_hidden_layers);
    assert_eq!(cfg.vocab_size, preset.vocab_size);
    assert_eq!(cfg.head_dim(), preset.head_dim());
    eprintln!(
        "MiniCPM5-1B: layers={} hidden={} vocab={} rope_theta={}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.rope_theta
    );
}

#[test]
fn runner_builds_on_safetensors() {
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_MINICPM5_WEIGHTS");
        return;
    };
    let runner = MiniCpm5Runner::builder()
        .weights(&weights)
        .max_seq(64)
        .build()
        .expect("MiniCpm5Runner::build");
    assert_eq!(runner.base_config().arch, "llama");
    assert_eq!(runner.llama_config().hidden_size, 1536);
}

#[test]
fn compat_check_safetensors_file() {
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_MINICPM5_WEIGHTS");
        return;
    };
    let report = check_path(&weights).expect("check_path");
    match &report.status {
        CompatibilityStatus::Supported { runner } => {
            assert!(
                *runner == "llama32" || *runner == "minicpm5",
                "expected llama32/minicpm5 runner, got {runner}"
            );
        }
        other => eprintln!("compat note (may need llama32 tag): {other:?}"),
    }
}

#[test]
fn forward_inference_minicpm5_1b() {
    if std::env::var("RLX_MINICPM5_RUN_INFERENCE").ok().as_deref() != Some("1") {
        eprintln!("skip: set RLX_MINICPM5_RUN_INFERENCE=1");
        return;
    }
    let Some(weights) = weights_path() else {
        eprintln!("skip: set RLX_MINICPM5_WEIGHTS");
        return;
    };
    let Some(tokenizer) = tokenizer_path() else {
        eprintln!("skip: tokenizer.json not found");
        return;
    };

    let mut runner = MiniCpm5Runner::builder()
        .weights(&weights)
        .max_seq(128)
        .build()
        .expect("build");

    let prompt_ids = rlx_models::llama32::encode_prompt_auto(
        &weights,
        Some(&tokenizer),
        "The capital of France is",
    )
    .expect("encode");

    let n_new = 1;
    let mut emitted = Vec::new();
    let generated = runner
        .generate_packed(&prompt_ids, n_new, |t| emitted.push(t))
        .expect("generate");

    assert_eq!(generated.len(), n_new);
    assert!(
        generated.iter().all(|t| (*t as usize) < 130_560),
        "token out of vocab: {generated:?}"
    );
    eprintln!(
        "MiniCPM5-1B inference: prompt_len={} new={generated:?}",
        prompt_ids.len()
    );
}
