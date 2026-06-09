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

//! Env-gated: compile vision / speech runners from real GGUF (CPU).
//!
//! ```text
//! DINOV2_GGUF_PATH=/path/dinov2.gguf \
//!   cargo test -p rlx-models --test vision_gguf_compile dinov2 --release -- --nocapture
//!
//! SAM3_GGUF_PATH=/path/sam3.gguf VISION_GGUF_COMPILE=1 \
//!   cargo test -p rlx-models --test vision_gguf_compile sam3 --release -- --nocapture
//!
//! W2V_BERT_GGUF_PATH=/path/model.gguf RLX_W2V_BERT_DIR=/path/w2v-bert-2.0 \
//!   cargo test -p rlx-models --test vision_gguf_compile w2v --release -- --nocapture
//!
//! FLUX_GGUF_PATH=/path/flux.gguf \
//!   cargo test -p rlx-models --test vision_gguf_compile flux --release -- --nocapture
//! ```

#[path = "vision_gguf/support.rs"]
mod support;

use rlx_models::run::DinoV2Runner;
use rlx_models::{Sam3, Sam3Config, Wav2Vec2BertRunner};
use rlx_runtime::Device;

use support::{compile_gate, env_gguf_path, env_path, weights_dir_for_gguf};

fn dinov2_gguf_config() -> rlx_models::dinov2::DinoV2Config {
    use rlx_models::dinov2::DinoV2Config;
    if let Some(p) = env_path("DINOV2_GGUF_CONFIG") {
        return DinoV2Config::from_file(&p).expect("DINOV2_GGUF_CONFIG");
    }
    let img_size = std::env::var("DINOV2_GGUF_IMG_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(518);
    let mut cfg = match std::env::var("DINOV2_GGUF_VARIANT")
        .unwrap_or_else(|_| "small".into())
        .to_lowercase()
        .as_str()
    {
        "base" => DinoV2Config::vit_base(img_size),
        "large" => DinoV2Config::vit_large(img_size),
        _ => DinoV2Config::vit_small(img_size),
    };
    // Community GGUF is usually encoder-only (no ImageNet head).
    cfg.num_classes = 0;
    cfg
}

#[test]
fn dinov2_gguf_compile_and_forward() {
    let Some(path) = env_gguf_path("DINOV2_GGUF_PATH") else {
        eprintln!("skip: set DINOV2_GGUF_PATH");
        return;
    };
    let cfg = dinov2_gguf_config();
    let mut runner = DinoV2Runner::builder()
        .weights(&path)
        .config(cfg)
        .device(Device::Cpu)
        .batch(1)
        .build()
        .unwrap_or_else(|e| panic!("DinoV2Runner::build {path:?}: {e:#}"));
    let out = runner
        .predict_image(&[128, 128, 128], 1, 1)
        .unwrap_or_else(|e| panic!("dinov2 forward: {e:#}"));
    match out {
        rlx_models::dinov2::DinoV2Output::Tokens {
            per_batch,
            seq,
            hidden,
        } => {
            assert_eq!(per_batch.len(), 1);
            assert_eq!(per_batch[0].len(), seq * hidden);
            eprintln!("dinov2 gguf ok: seq={seq} hidden={hidden}");
        }
        rlx_models::dinov2::DinoV2Output::Logits {
            per_batch,
            num_classes,
        } => {
            assert_eq!(per_batch[0].len(), num_classes);
            eprintln!("dinov2 gguf ok: logits dim={num_classes}");
        }
    }
}

#[test]
fn sam3_gguf_compile() {
    if !compile_gate() {
        eprintln!("skip sam3_gguf_compile: set VISION_GGUF_COMPILE=1 (loads full SAM3 stack)");
        return;
    }
    let Some(path) = env_gguf_path("SAM3_GGUF_PATH") else {
        eprintln!("skip: set SAM3_GGUF_PATH");
        return;
    };
    let path_str = path.to_str().expect("utf-8 path");
    let _sam = Sam3::from_checkpoint_on(path_str, Sam3Config::base(), Device::Cpu)
        .unwrap_or_else(|e| panic!("Sam3::from_checkpoint_on {path:?}: {e:#}"));
    eprintln!("sam3 gguf compile ok: {path:?}");
}

#[test]
fn w2v_bert_gguf_compile_and_encode() {
    let Some(gguf) = env_gguf_path("W2V_BERT_GGUF_PATH") else {
        eprintln!("skip: set W2V_BERT_GGUF_PATH");
        return;
    };
    let dir = weights_dir_for_gguf(&gguf, "RLX_W2V_BERT_DIR");
    let cfg_path = dir.join("config.json");
    if !cfg_path.is_file() {
        eprintln!("skip: need {cfg_path:?} (set RLX_W2V_BERT_DIR to HF snapshot with config.json)");
        return;
    }
    let mut runner = Wav2Vec2BertRunner::builder()
        .weights(&gguf)
        .config_path(&cfg_path)
        .device(Device::Cpu)
        .batch(1)
        .seq(64)
        .build()
        .unwrap_or_else(|e| panic!("Wav2Vec2BertRunner::build: {e:#}"));
    let feat_dim = runner.config().feature_projection_input_dim;
    let n = 64 * feat_dim;
    let features = vec![0.0f32; n];
    let hidden = runner
        .encode_features(&features, None)
        .unwrap_or_else(|e| panic!("encode_features: {e:#}"));
    let h = runner.config().hidden_size;
    assert_eq!(hidden.len(), 64 * h, "hidden len");
    eprintln!(
        "w2v-bert gguf compile ok: hidden[0..4]={:?}",
        &hidden[..4.min(hidden.len())]
    );
}

#[test]
fn flux_gguf_denoiser_extract() {
    let Some(path) = env_gguf_path("FLUX_GGUF_PATH") else {
        eprintln!("skip: set FLUX_GGUF_PATH");
        return;
    };
    use rlx_flux2::{
        Flux2Config, extract_flux2_weights, load_flux2_weight_map, prepare_weight_map,
    };
    use rlx_gguf::GgufFile;

    let raw = GgufFile::from_path(&path).expect("open flux gguf");
    let cfg = Flux2Config::from_gguf(&raw).expect("flux config from gguf");
    let wm = load_flux2_weight_map(&path).expect("load flux gguf");
    let weights = extract_flux2_weights(prepare_weight_map(wm), &cfg)
        .unwrap_or_else(|e| panic!("extract_flux2_weights: {e:#}"));
    assert!(
        !weights.transformer_blocks.is_empty() || !weights.single_transformer_blocks.is_empty(),
        "expected transformer blocks in extracted weights"
    );
    eprintln!(
        "flux gguf extract ok: double={} single={}",
        weights.transformer_blocks.len(),
        weights.single_transformer_blocks.len()
    );
}

#[test]
fn flux_gguf_minimal_compiled_forward() {
    let Some(path) = env_gguf_path("FLUX_GGUF_PATH") else {
        eprintln!("skip: set FLUX_GGUF_PATH");
        return;
    };
    use rlx_gguf::GgufFile;
    use rlx_models::flux2::{
        Flux2Config, compile_flux2_minimal, extract_flux2_weights, load_flux2_weight_map,
        prepare_weight_map,
    };

    let raw = GgufFile::from_path(&path).expect("open flux gguf");
    let cfg = Flux2Config::from_gguf(&raw).expect("flux config from gguf");
    let wm = load_flux2_weight_map(&path).expect("load flux gguf");
    let weights = extract_flux2_weights(prepare_weight_map(wm), &cfg)
        .unwrap_or_else(|e| panic!("extract_flux2_weights: {e:#}"));
    let img_seq = 4usize;
    let batch = 1usize;
    let (mut compiled, _) =
        compile_flux2_minimal(&cfg, &weights, batch, img_seq).expect("compile_flux2_minimal");
    let hidden_len = batch * img_seq * cfg.in_channels;
    let hidden = vec![0.0f32; hidden_len];
    let out = compiled.run(&[("hidden", hidden.as_slice())]);
    assert_eq!(out[0].len(), batch * img_seq * cfg.proj_out_dim());
    eprintln!(
        "flux gguf minimal forward ok: in_ch={} img_seq={} proj_out={}",
        cfg.in_channels,
        img_seq,
        cfg.proj_out_dim()
    );
}
