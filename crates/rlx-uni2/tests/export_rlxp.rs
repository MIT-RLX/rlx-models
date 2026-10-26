// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported UNI2 package, opened by rlx alone, gives for an RGB image
//! what `Uni2Runner::predict_image` does — the token sequence and the
//! `[CLS]` embedding — for a two-block model with random weights, register
//! tokens and the packed SwiGLU MLP, written as a timm checkpoint.
#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::ExportOptions;
use rlx_core::export::check::write_checkpoint;
use rlx_runtime::model::Tensor;
use rlx_uni2::export::export_rlxp;
use rlx_uni2::{Uni2Config, Uni2Runner};
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

#[test]
fn an_exported_uni2_gives_the_runners_tokens_and_embedding() {
    let cfg = Uni2Config {
        hidden_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        img_size: 20,
        patch_size: 4,
        mlp_hidden_dim: 192,
        layer_norm_eps: 1e-6,
        num_register_tokens: 3,
    };
    let (h, full, inner) = (cfg.hidden_size, cfg.mlp_hidden_dim, cfg.swiglu_inner());
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 5u64;
    let mut put = |k: String, shape: Vec<usize>| {
        seed += 13;
        let n = shape.iter().product();
        t.insert(k, (fill(n, seed), shape));
    };
    put("patch_embed.proj.weight".into(), vec![h, 3, 4, 4]);
    put("patch_embed.proj.bias".into(), vec![h]);
    put("cls_token".into(), vec![1, 1, h]);
    put("reg_token".into(), vec![1, cfg.num_register_tokens, h]);
    put("pos_embed".into(), vec![1, cfg.num_patches(), h]);
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
        put(format!("{l}.mlp.fc1.weight"), vec![full, h]);
        put(format!("{l}.mlp.fc1.bias"), vec![full]);
        put(format!("{l}.mlp.fc2.weight"), vec![h, inner]);
        put(format!("{l}.mlp.fc2.bias"), vec![h]);
    }
    put("norm.weight".into(), vec![h]);
    put("norm.bias".into(), vec![h]);

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("uni2");
    // timm's nested config, which says nothing the exporter reads.
    write_checkpoint(
        &dir,
        &json!({"architecture": "vit_giant_patch14_224"}),
        &t,
        8,
    )
    .unwrap();
    let out = tmp.path().join("uni2.rlxp");
    let report = export_rlxp(&dir, &out, &ExportOptions::default()).unwrap();
    // Every checkpoint tensor is referenced but each block's packed SwiGLU
    // `fc1` weight and bias, which the graph splits into gate and up halves
    // (held).
    assert_eq!(
        report.referenced,
        t.len() - 2 * cfg.num_hidden_layers,
        "{report}"
    );

    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = Uni2Runner::builder()
        .weights(dir.join("model.safetensors"))
        .config(cfg.clone())
        .build()
        .unwrap();
    for (hh, ww) in [(20, 20), (33, 51)] {
        let rgb: Vec<u8> = (0..hh * ww * 3)
            .map(|i| ((i * 31 + i / 3) % 251) as u8)
            .collect();
        let input = Tensor::new("image", rlx_ir::DType::U8, vec![hh, ww, 3], rgb.clone());
        let outs = m.run(&[input], &json!({})).unwrap();
        let want = runner.predict_image(&rgb, hh, ww).unwrap();
        for (got, want, what) in [
            (outs[0].to_f32().unwrap(), &want.tokens[0], "tokens"),
            (outs[1].to_f32().unwrap(), &want.embeddings[0], "embedding"),
        ] {
            assert_eq!(got.len(), want.len(), "{what}");
            let worst = got
                .iter()
                .zip(want)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            assert!(worst < 1e-4, "{hh}x{ww} {what}: off by {worst}");
        }
    }
}
