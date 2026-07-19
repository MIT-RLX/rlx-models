// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// End-to-end smoke test: write a tiny UNI2-shaped checkpoint as
// safetensors, load it through `Uni2Runner`, and run a forward pass.
//
// No numeric reference is available (the real weights are gated), so this
// exercises the full pipeline — safetensors load, `reg_token` key,
// host patchify + `no_embed_class` assembly, compile, the packed-SwiGLU
// plugin, attention, LayerScale — and asserts the output is well-formed
// (correct shape, all finite, not a degenerate constant).

use rlx_uni2::{Uni2Config, Uni2Runner};
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeMap;

fn tiny_cfg() -> Uni2Config {
    Uni2Config {
        hidden_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        img_size: 28,
        patch_size: 14,
        mlp_hidden_dim: 32, // SwiGLU inner = 16
        layer_norm_eps: 1e-6,
        num_register_tokens: 8,
    }
}

/// Small deterministic weights so real numbers flow through every op
/// (a seeded sine pattern in a tight range).
fn fill(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((seed * 131 + i * 17) % 97) as f32 / 97.0 - 0.5) * 0.05)
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn uni2_runner_forward_on_synthetic_checkpoint() {
    let cfg = tiny_cfg();
    let h = cfg.hidden_size;
    let full = cfg.mlp_hidden_dim;
    let inner = cfg.swiglu_inner();
    let np = cfg.num_patches();

    // (name, shape, data) — build owned buffers first.
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut seed = 1usize;
    let push = |tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
                name: &str,
                shape: Vec<usize>,
                seed: &mut usize| {
        let n: usize = shape.iter().product();
        tensors.push((name.to_string(), shape, fill(*seed, n)));
        *seed += 1;
    };

    push(
        &mut tensors,
        "patch_embed.proj.weight",
        vec![h, 3, cfg.patch_size, cfg.patch_size],
        &mut seed,
    );
    push(&mut tensors, "patch_embed.proj.bias", vec![h], &mut seed);
    push(&mut tensors, "cls_token", vec![1, 1, h], &mut seed);
    push(
        &mut tensors,
        "reg_token",
        vec![1, cfg.num_register_tokens, h],
        &mut seed,
    );
    push(&mut tensors, "pos_embed", vec![1, np, h], &mut seed); // no_embed_class → patches only
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("blocks.{i}");
        push(
            &mut tensors,
            &format!("{lp}.norm1.weight"),
            vec![h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.norm1.bias"),
            vec![h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.norm2.weight"),
            vec![h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.norm2.bias"),
            vec![h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.attn.qkv.weight"),
            vec![3 * h, h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.attn.qkv.bias"),
            vec![3 * h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.attn.proj.weight"),
            vec![h, h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.attn.proj.bias"),
            vec![h],
            &mut seed,
        );
        push(&mut tensors, &format!("{lp}.ls1.gamma"), vec![h], &mut seed);
        push(&mut tensors, &format!("{lp}.ls2.gamma"), vec![h], &mut seed);
        push(
            &mut tensors,
            &format!("{lp}.mlp.fc1.weight"),
            vec![full, h],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.mlp.fc1.bias"),
            vec![full],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.mlp.fc2.weight"),
            vec![h, inner],
            &mut seed,
        );
        push(
            &mut tensors,
            &format!("{lp}.mlp.fc2.bias"),
            vec![h],
            &mut seed,
        );
    }
    // norm.weight ~1 so the final LayerNorm doesn't collapse the signal.
    tensors.push(("norm.weight".into(), vec![h], vec![1.0f32; h]));
    push(&mut tensors, "norm.bias", vec![h], &mut seed);

    // Serialize to safetensors (needs owned byte buffers borrowed as views).
    let owned: BTreeMap<String, (Vec<usize>, Vec<u8>)> = tensors
        .into_iter()
        .map(|(n, s, d)| (n, (s, f32_bytes(&d))))
        .collect();
    let views: Vec<(String, TensorView)> = owned
        .iter()
        .map(|(n, (s, b))| {
            (
                n.clone(),
                TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
            )
        })
        .collect();
    let bytes = safetensors::serialize(views, None).unwrap();

    let path =
        std::env::temp_dir().join(format!("rlx_uni2_e2e_{}.safetensors", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();

    let result = std::panic::catch_unwind(|| {
        let mut runner = Uni2Runner::builder()
            .weights(&path)
            .config(cfg.clone())
            .build()
            .expect("runner build");

        // Synthetic gradient tile at native resolution.
        let (hgt, wid) = (cfg.img_size, cfg.img_size);
        let mut rgb = vec![0u8; hgt * wid * 3];
        for y in 0..hgt {
            for x in 0..wid {
                let b = (y * wid + x) * 3;
                rgb[b] = (x * 255 / wid) as u8;
                rgb[b + 1] = (y * 255 / hgt) as u8;
                rgb[b + 2] = ((x + y) * 127 / (hgt + wid)) as u8;
            }
        }

        let out = runner.predict_image(&rgb, hgt, wid).expect("forward");
        (out, cfg)
    });

    let _ = std::fs::remove_file(&path);
    let (out, cfg) = result.expect("forward pass panicked");

    assert_eq!(out.embeddings.len(), 1);
    assert_eq!(out.hidden, cfg.hidden_size);
    assert_eq!(out.seq, cfg.seq_len());
    let emb = &out.embeddings[0];
    assert_eq!(emb.len(), cfg.hidden_size);
    assert!(
        emb.iter().all(|v| v.is_finite()),
        "embedding has non-finite values"
    );
    // Tokens should be the full [seq · hidden] post-norm sequence.
    assert_eq!(out.tokens[0].len(), cfg.seq_len() * cfg.hidden_size);
    assert!(out.tokens[0].iter().all(|v| v.is_finite()));
    // Not a degenerate all-identical vector (the SwiGLU/attention path ran).
    let spread =
        emb.iter().cloned().fold(f32::MIN, f32::max) - emb.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 1e-6,
        "CLS embedding is a flat constant: spread={spread}"
    );
}
