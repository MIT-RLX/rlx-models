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

//! Stage-wise + end-to-end parity vs Hugging Face `baidu/Unlimited-OCR`.
//!
//! Env:
//! - `RLX_UNLIMITED_OCR_DIR` — checkpoint snapshot (required)
//! - `RLX_UNLIMITED_OCR_PYTHON` — python with torch + transformers (optional; e2e)
//! - `RLX_UNLIMITED_OCR_IMAGE` — probe image (optional; default fixtures/sample.jpg)
//!
//! ```bash
//! just fetch-unlimited-ocr
//! just test-unlimited-ocr-parity
//! ```

use anyhow::{Context, Result, ensure};
use rlx_runtime::Device;
use rlx_unlimited_ocr::{
    DOWNSAMPLE_RATIO, IMAGE_TOKEN_ID, ImageMode, InferenceOptions, PATCH_SIZE, SampleOpts,
    UnlimitedOcrConfig, UnlimitedOcrRunner, UnlimitedOcrSession, UnlimitedOcrWeightStore,
    base_image_tokens, num_queries, preprocess_path, require_model_dir, require_probe_image,
    sample_image_path,
};
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

fn model_dir() -> PathBuf {
    require_model_dir().unwrap_or_else(|| {
        panic!(
            "set RLX_UNLIMITED_OCR_DIR or run `just fetch-unlimited-ocr` \
             (need config.json + tokenizer.json)"
        )
    })
}

fn probe_image() -> PathBuf {
    require_probe_image().unwrap_or_else(sample_image_path)
}

fn python_bin() -> Option<String> {
    std::env::var("RLX_UNLIMITED_OCR_PYTHON")
        .ok()
        .or_else(|| std::env::var("PYTHON").ok())
        .or_else(|| Some("python3".into()))
}

#[test]
fn checkpoint_config_and_weight_inventory() {
    let dir = model_dir();
    let cfg = UnlimitedOcrConfig::from_model_dir(&dir).expect("config");
    cfg.validate().expect("validate");
    assert_eq!(cfg.hidden_size, 1280);
    assert_eq!(cfg.num_hidden_layers, 12);
    assert_eq!(cfg.n_routed_experts, 64);
    assert_eq!(cfg.n_shared_experts, 2);
    assert_eq!(cfg.num_experts_per_tok, 6);
    assert_eq!(cfg.first_k_dense_replace, 1);
    assert_eq!(cfg.sliding_window, 128);
    assert!(!cfg.use_mla);
    assert_eq!(cfg.vocab_size, 129_280);
    assert_eq!(cfg.image_token_id, IMAGE_TOKEN_ID);

    let store = UnlimitedOcrWeightStore::open(&dir).expect("weights");
    assert!(
        store.keys().len() > 2000,
        "expected full shard, got {}",
        store.keys().len()
    );
    assert!(store.contains("model.embed_tokens.weight"));
    assert!(store.contains("lm_head.weight"));
    assert!(store.contains("model.image_newline"));
    assert!(store.contains("model.view_seperator"));
    assert!(store.contains("model.sam_model.patch_embed.proj.weight"));
    assert!(store.contains("model.vision_model.embeddings.class_embedding"));
    assert!(store.contains("model.projector.layers.weight"));
    assert!(store.is_moe_layer(1));
    assert!(!store.is_moe_layer(0));
    assert_eq!(store.count_experts(1), 64);
}

#[test]
fn prompt_placeholder_counts_match_hf_formulas() {
    let img = probe_image();
    if !img.is_file() {
        eprintln!("skip: no probe image at {}", img.display());
        return;
    }
    let base = preprocess_path(&img, ImageMode::Base { size: 1024 }).expect("base");
    let q = num_queries(1024);
    assert_eq!(q, 16);
    assert_eq!(
        base.token_count(PATCH_SIZE, DOWNSAMPLE_RATIO),
        base_image_tokens(q)
    );
    let ids = base.image_token_ids(IMAGE_TOKEN_ID, PATCH_SIZE, DOWNSAMPLE_RATIO);
    assert_eq!(ids.len(), base_image_tokens(q));
    assert!(ids.iter().all(|&t| t == IMAGE_TOKEN_ID));

    let gundam = preprocess_path(
        &img,
        ImageMode::Gundam {
            base: 1024,
            tile: 640,
        },
    )
    .expect("gundam");
    let ids_g = gundam.image_token_ids(IMAGE_TOKEN_ID, PATCH_SIZE, DOWNSAMPLE_RATIO);
    assert_eq!(
        ids_g.len(),
        gundam.token_count(PATCH_SIZE, DOWNSAMPLE_RATIO)
    );
    // Global block first in the prompt, then tiles (HF infer()).
    assert_eq!(
        &ids_g[..base_image_tokens(16)],
        &vec![IMAGE_TOKEN_ID; base_image_tokens(16)][..]
    );
}

#[test]
fn tokenizer_prompt_assembly_prepends_bos() -> Result<()> {
    let dir = model_dir();
    let img = probe_image();
    if !img.is_file() {
        eprintln!("skip: no probe image");
        return Ok(());
    }
    let pre = preprocess_path(&img, ImageMode::Base { size: 1024 })?;
    let runner = UnlimitedOcrRunner::open(&dir, Device::Cpu)?;
    let ids = runner.build_prompt_ids("<image>document parsing.", &[pre])?;
    ensure!(!ids.is_empty(), "empty prompt");
    ensure!(ids[0] == 0, "BOS must be 0, got {}", ids[0]);
    ensure!(
        ids.contains(&IMAGE_TOKEN_ID),
        "missing image token {IMAGE_TOKEN_ID}"
    );
    let n_img = ids.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    ensure!(
        n_img == base_image_tokens(16),
        "expected {} image placeholders, got {n_img}",
        base_image_tokens(16)
    );
    Ok(())
}

#[test]
fn load_weights_and_emit_first_logits() -> Result<()> {
    let dir = model_dir();
    let img = probe_image();
    if !img.is_file() {
        eprintln!("skip: no probe image");
        return Ok(());
    }
    let mut runner = UnlimitedOcrRunner::open(&dir, Device::Cpu)?;
    runner.load_weights()?;
    let pre = preprocess_path(&img, ImageMode::Base { size: 1024 })?;
    let opts = SampleOpts {
        max_new_tokens: 1,
        no_repeat_ngram_size: 0,
        ..Default::default()
    };
    let (text, ids, prompt_len) = runner.generate("<image>document parsing.", &[pre], &opts)?;
    ensure!(ids.len() == prompt_len + 1, "expected one new token");
    ensure!(!text.is_empty() || ids[prompt_len] == 1, "empty decode");
    eprintln!("[hf_parity] first token={} text={text:?}", ids[prompt_len]);
    Ok(())
}

#[test]
fn e2e_base_matches_hf_when_python_available() -> Result<()> {
    let dir = model_dir();
    let img = probe_image();
    if !img.is_file() {
        eprintln!("skip e2e: no probe image");
        return Ok(());
    }
    let Some(py) = python_bin() else {
        eprintln!("skip e2e: no python");
        return Ok(());
    };
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/hf_reference_unlimited_ocr.py");
    if !script.is_file() {
        eprintln!("skip e2e: missing {}", script.display());
        return Ok(());
    }
    let out = std::env::temp_dir().join("rlx_unlimited_ocr_hf_base.json");
    let status = Command::new(&py)
        .args([
            script.to_str().unwrap(),
            "--model-dir",
            dir.to_str().unwrap(),
            "--image",
            img.to_str().unwrap(),
            "--mode",
            "base",
            "--max-new-tokens",
            "16",
            "--out",
            out.to_str().unwrap(),
            "--device",
            "cpu",
        ])
        .status();
    let Ok(status) = status else {
        eprintln!("skip e2e: failed to spawn {py}");
        return Ok(());
    };
    if !status.success() {
        eprintln!(
            "skip e2e: HF reference exited {status} (install torch+transformers+trust_remote_code deps)"
        );
        return Ok(());
    }
    let ref_json: Value = serde_json::from_str(&std::fs::read_to_string(&out)?)
        .with_context(|| format!("parse {}", out.display()))?;
    if ref_json.get("error").is_some() {
        eprintln!("skip e2e: {}", ref_json["error"]);
        return Ok(());
    }
    let hf_text = ref_json["text"].as_str().unwrap_or("").trim().to_string();

    let mut options = InferenceOptions::for_ocr()
        .device(Device::Cpu)
        .mode(ImageMode::Base { size: 1024 })
        .max_new_tokens(16);
    options.sample.no_repeat_ngram_size = 35;
    options.sample.ngram_window = 128;
    let mut session = UnlimitedOcrSession::open(&dir, options)?;
    let result = session.run_single(&img)?;
    let rlx_text = result.text.trim().to_string();

    if let Some(ids) = ref_json["token_ids"].as_array() {
        let hf_ids: Vec<u32> = ids
            .iter()
            .filter_map(|v| v.as_u64().map(|u| u as u32))
            .collect();
        if !hf_ids.is_empty() {
            let rlx_new = &result.token_ids[result.prompt_len..];
            ensure!(
                rlx_new == hf_ids.as_slice(),
                "token mismatch:\n  rlx={rlx_new:?}\n  hf ={hf_ids:?}"
            );
        }
    }

    ensure!(
        rlx_text == hf_text,
        "text mismatch:\n  rlx={rlx_text:?}\n  hf ={hf_text:?}"
    );
    Ok(())
}

#[test]
fn e2e_gundam_greedy_smoke() -> Result<()> {
    let dir = model_dir();
    let img = probe_image();
    if !img.is_file() {
        eprintln!("skip: no probe image");
        return Ok(());
    }
    let options = InferenceOptions::for_ocr()
        .device(Device::Cpu)
        .mode(ImageMode::default())
        .max_new_tokens(8);
    let mut session = UnlimitedOcrSession::open(&dir, options)?;
    let result = session.run_single(&img)?;
    ensure!(result.new_tokens > 0, "expected generated tokens");
    ensure!(!result.token_ids.is_empty());
    eprintln!(
        "[hf_parity] gundam smoke new_tokens={} text_len={}",
        result.new_tokens,
        result.text.len()
    );
    Ok(())
}
