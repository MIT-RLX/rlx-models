// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported DINOv2 package, opened by rlx alone, gives for an RGB image
//! what `DinoV2Runner::predict_image` does: a two-block model with random
//! weights, written as a timm checkpoint, with and without a classifier
//! head and with register tokens.
#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::ExportOptions;
use rlx_core::export::check::write_checkpoint;
use rlx_dinov2::export::export_rlxp;
use rlx_dinov2::{DinoV2Config, DinoV2Output, DinoV2Runner};
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.4
        })
        .collect()
}

fn checkpoint(cfg: &DinoV2Config) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let h = cfg.hidden_size;
    let inner = cfg.intermediate_size();
    let p = cfg.patch_size;
    let mut t = HashMap::new();
    let mut seed = 11u64;
    let mut put = |k: String, shape: Vec<usize>| {
        seed += 13;
        let n = shape.iter().product();
        t.insert(k, (fill(n, seed), shape));
    };
    put("patch_embed.proj.weight".into(), vec![h, 3, p, p]);
    put("patch_embed.proj.bias".into(), vec![h]);
    put("cls_token".into(), vec![1, 1, h]);
    put("pos_embed".into(), vec![1, cfg.seq_len(), h]);
    if cfg.num_register_tokens > 0 {
        put(
            "register_tokens".into(),
            vec![1, cfg.num_register_tokens, h],
        );
    }
    for i in 0..cfg.num_hidden_layers {
        let l = format!("blocks.{i}");
        for n in ["norm1", "norm2"] {
            put(format!("{l}.{n}.weight"), vec![h]);
            put(format!("{l}.{n}.bias"), vec![h]);
        }
        put(format!("{l}.attn.qkv.weight"), vec![3 * h, h]);
        put(format!("{l}.attn.qkv.bias"), vec![3 * h]);
        put(format!("{l}.attn.proj.weight"), vec![h, h]);
        put(format!("{l}.attn.proj.bias"), vec![h]);
        put(format!("{l}.ls1.gamma"), vec![h]);
        put(format!("{l}.ls2.gamma"), vec![h]);
        put(format!("{l}.mlp.fc1.weight"), vec![inner, h]);
        put(format!("{l}.mlp.fc1.bias"), vec![inner]);
        put(format!("{l}.mlp.fc2.weight"), vec![h, inner]);
        put(format!("{l}.mlp.fc2.bias"), vec![h]);
    }
    put("norm.weight".into(), vec![h]);
    put("norm.bias".into(), vec![h]);
    if cfg.num_classes > 0 {
        put("head.weight".into(), vec![cfg.num_classes, 2 * h]);
        put("head.bias".into(), vec![cfg.num_classes]);
    }
    t
}

fn image(h: usize, w: usize) -> Vec<u8> {
    (0..h * w * 3)
        .map(|i| ((i * 29 + i / 5) % 253) as u8)
        .collect()
}

fn check(classes: usize, registers: usize) {
    let mut cfg = DinoV2Config::new(16, 2, 128, 2, registers);
    cfg.patch_size = 4;
    cfg.num_classes = classes;
    let t = checkpoint(&cfg);
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("dinov2");
    // A timm config, which says nothing the exporter reads: the tensors do.
    write_checkpoint(&dir, &json!({"architecture": "vit_dinov2"}), &t, 8).unwrap();
    let out = tmp.path().join("dinov2.rlxp");
    let report = export_rlxp(&dir, &out, &ExportOptions::default()).unwrap();
    assert_eq!(report.referenced, t.len(), "{report}");

    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = DinoV2Runner::builder()
        .weights(dir.join("model.safetensors"))
        .config(cfg.clone())
        .device(Device::Cpu)
        .build()
        .unwrap();
    for (h, w) in [(16, 16), (40, 23)] {
        let rgb = image(h, w);
        let input = Tensor::new("image", rlx_ir::DType::U8, vec![h, w, 3], rgb.clone());
        let outs = m.run(&[input], &json!({})).unwrap();
        let got = outs[0].to_f32().unwrap();
        let want = match runner.predict_image(&rgb, h, w).unwrap() {
            DinoV2Output::Logits { per_batch, .. } => per_batch[0].clone(),
            DinoV2Output::Tokens { per_batch, .. } => {
                // The class embedding is the first row.
                let e = outs[1].to_f32().unwrap();
                let first = &per_batch[0][..cfg.hidden_size];
                let off = e
                    .iter()
                    .zip(first)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                assert!(off < 1e-4, "class embedding off by {off}");
                per_batch[0].clone()
            }
        };
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let scale = want.iter().fold(0f32, |a, x| a.max(x.abs()));
        assert!(
            worst <= 1e-4 * scale.max(1.0),
            "{h}x{w}: off by {worst} (of {scale})"
        );
    }
}

#[test]
fn an_exported_dinov2_encoder_gives_the_runners_tokens() {
    check(0, 0);
}

#[test]
fn an_exported_dinov2_classifier_with_registers_gives_the_runners_logits() {
    check(10, 2);
}
