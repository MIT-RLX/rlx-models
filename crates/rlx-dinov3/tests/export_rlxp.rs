// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported DINOv3 package, opened by rlx alone, gives for an RGB image
//! what `DinoV3Runner::predict_image` does — the token sequence and the
//! `[CLS]` embedding — for a two-layer model with random weights, register
//! tokens and RoPE, with the plain and the gated MLP.
#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::ExportOptions;
use rlx_core::export::check::write_checkpoint;
use rlx_dinov3::DinoV3Config;
use rlx_dinov3::export::export_rlxp;
use rlx_dinov3::runner::DinoV3Runner;
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

fn check(gated: bool) {
    let config = json!({
        "model_type": "dinov3_vit", "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 32,
        "patch_size": 8, "num_register_tokens": 2, "use_gated_mlp": gated,
    });
    let cfg: DinoV3Config = serde_json::from_value(config.clone()).unwrap();
    let (h, inter, p) = (cfg.hidden_size, cfg.intermediate_size, cfg.patch_size);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 9u64;
    let mut put = |k: String, shape: Vec<usize>| {
        seed += 13;
        let n = shape.iter().product();
        t.insert(k, (fill(n, seed), shape));
    };
    put(
        "embeddings.patch_embeddings.weight".into(),
        vec![h, 3, p, p],
    );
    put("embeddings.patch_embeddings.bias".into(), vec![h]);
    put("embeddings.cls_token".into(), vec![1, 1, h]);
    put("embeddings.mask_token".into(), vec![1, 1, h]);
    put("embeddings.register_tokens".into(), vec![1, 2, h]);
    for i in 0..cfg.num_hidden_layers {
        let l = format!("layer.{i}");
        for n in ["norm1", "norm2"] {
            put(format!("{l}.{n}.weight"), vec![h]);
            put(format!("{l}.{n}.bias"), vec![h]);
        }
        for q in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            put(format!("{l}.attention.{q}.weight"), vec![h, h]);
        }
        for q in ["q_proj", "v_proj", "o_proj"] {
            put(format!("{l}.attention.{q}.bias"), vec![h]);
        }
        put(format!("{l}.layer_scale1.lambda1"), vec![h]);
        put(format!("{l}.layer_scale2.lambda1"), vec![h]);
        if gated {
            put(format!("{l}.mlp.gate_proj.weight"), vec![inter, h]);
            put(format!("{l}.mlp.gate_proj.bias"), vec![inter]);
        }
        put(format!("{l}.mlp.up_proj.weight"), vec![inter, h]);
        put(format!("{l}.mlp.up_proj.bias"), vec![inter]);
        put(format!("{l}.mlp.down_proj.weight"), vec![h, inter]);
        put(format!("{l}.mlp.down_proj.bias"), vec![h]);
    }
    put("norm.weight".into(), vec![h]);
    put("norm.bias".into(), vec![h]);

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("dinov3");
    write_checkpoint(&dir, &config, &t, 8).unwrap();
    let out = tmp.path().join("dinov3.rlxp");
    let report = export_rlxp(&dir, &out, &ExportOptions::default()).unwrap();
    // All but the pretraining-only mask token.
    assert_eq!(report.referenced, t.len() - 1, "{report}");

    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut runner = DinoV3Runner::builder()
        .weights(dir.join("model.safetensors"))
        .config(cfg.clone())
        .build()
        .unwrap();
    for (hh, ww) in [(32, 32), (45, 27)] {
        let rgb: Vec<u8> = (0..hh * ww * 3)
            .map(|i| ((i * 23 + i / 9) % 249) as u8)
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

#[test]
fn an_exported_dinov3_gives_the_runners_tokens_and_embedding() {
    check(false);
}

#[test]
fn an_exported_dinov3_with_a_gated_mlp_does_too() {
    check(true);
}
