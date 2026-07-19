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

//! DINOv3 numeric parity against HuggingFace `transformers` (`DINOv3ViTModel`).
//!
//! This test is env-gated: set `DINOV3_FIXTURES` to a directory containing
//! `model.safetensors` (HF keys), `meta.json`, `px.bin`, `last_hidden.bin`
//! and `pooled.bin` (little-endian f32), produced by
//! `scratchpad/dv3/make_fixtures.py`. Without the env var the test is a
//! no-op so CI stays hermetic (the pretrained weights are gated).
//!
//! Run:
//! ```bash
//! DINOV3_FIXTURES=scratchpad/dv3/fixtures \
//!   cargo test -p rlx-dinov3 --test dinov3_parity -- --nocapture
//! ```

use rlx_dinov3::{DinoV3Config, DinoV3Runner};
use std::path::PathBuf;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(
        bytes.len().is_multiple_of(4),
        "{path:?} not a multiple of 4 bytes"
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-20)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn dinov3_cpu_parity_vs_hf() {
    let Ok(dir) = std::env::var("DINOV3_FIXTURES") else {
        eprintln!("DINOV3_FIXTURES unset — skipping DINOv3 HF parity test");
        return;
    };
    let dir = PathBuf::from(dir);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();

    let img_size = meta["image_size"].as_u64().unwrap() as usize;
    let cfg = DinoV3Config {
        hidden_size: meta["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: meta["intermediate_size"].as_u64().unwrap() as usize,
        num_hidden_layers: meta["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: meta["num_attention_heads"].as_u64().unwrap() as usize,
        image_size: img_size,
        patch_size: meta["patch_size"].as_u64().unwrap() as usize,
        num_channels: meta["num_channels"].as_u64().unwrap() as usize,
        hidden_act: meta["hidden_act"].as_str().unwrap().to_string(),
        layer_norm_eps: meta["layer_norm_eps"].as_f64().unwrap(),
        rope_theta: meta["rope_theta"].as_f64().unwrap(),
        query_bias: meta["query_bias"].as_bool().unwrap(),
        key_bias: meta["key_bias"].as_bool().unwrap(),
        value_bias: meta["value_bias"].as_bool().unwrap(),
        proj_bias: meta["proj_bias"].as_bool().unwrap(),
        mlp_bias: meta["mlp_bias"].as_bool().unwrap(),
        layerscale_value: meta["layerscale_value"].as_f64().unwrap(),
        use_gated_mlp: meta["use_gated_mlp"].as_bool().unwrap(),
        num_register_tokens: meta["num_register_tokens"].as_u64().unwrap() as usize,
        final_layer_norm_affine: true,
    };

    let px = read_f32(&dir.join("px.bin")); // [C, H, W] = NCHW batch 1
    let ref_last = read_f32(&dir.join("last_hidden.bin")); // [seq, hidden]
    let ref_pooled = read_f32(&dir.join("pooled.bin")); // [hidden]
    let weights = dir.join("model.safetensors");

    // Devices to check (comma-separated). Requires the matching cargo
    // feature to be enabled at build time (e.g. --features metal). Defaults
    // to CPU so the check is hermetic.
    let dev_list = std::env::var("DINOV3_DEVICES").unwrap_or_else(|_| "cpu".into());

    let mut failures = Vec::new();
    for name in dev_list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let device =
            rlx_cli::parse_standard_device("dinov3", name).expect("parse DINOV3_DEVICES entry");

        let mut runner = DinoV3Runner::builder()
            .weights(&weights)
            .device(device)
            .img_size(img_size)
            .config(cfg.clone())
            .build()
            .unwrap_or_else(|e| panic!("build dinov3 runner on {name}: {e:#}"));

        let out = runner
            .forward_nchw(&px)
            .unwrap_or_else(|e| panic!("forward on {name}: {e:#}"));
        let got_last = &out.tokens[0];
        let got_pooled = &out.embeddings[0];
        assert_eq!(got_last.len(), ref_last.len(), "{name}: last_hidden len");
        assert_eq!(got_pooled.len(), ref_pooled.len(), "{name}: pooled len");

        let cos_last = cosine(got_last, &ref_last);
        let mad_last = max_abs(got_last, &ref_last);
        let cos_pool = cosine(got_pooled, &ref_pooled);
        let mad_pool = max_abs(got_pooled, &ref_pooled);
        eprintln!(
            "DINOv3 [{name:>6}] vs HF: last_hidden cos={cos_last:.8} max_abs={mad_last:.3e} | \
             pooled cos={cos_pool:.8} max_abs={mad_pool:.3e}"
        );

        // All backends (CPU / Metal / MLX / wgpu) are bit-exact vs HF on this
        // graph (~6e-8). Cosine is the parity metric; max_abs is a tight
        // gross-error guard.
        if cos_last <= 0.9999 || cos_pool <= 0.9999 || mad_last >= 1e-4 || mad_pool >= 1e-4 {
            failures.push(format!(
                "{name}: last cos={cos_last:.6} mad={mad_last:.2e} pooled cos={cos_pool:.6} mad={mad_pool:.2e}"
            ));
        }
    }
    assert!(failures.is_empty(), "backend parity failures: {failures:?}");
}
