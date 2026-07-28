// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the **lazy (mmap) MLX loader**: `MlxLoader::open_lazy` must return
//! byte-identical tensors to the eager `MlxLoader::open`, while materializing
//! each tensor only on demand (so a worker can load its shard of a checkpoint
//! larger than its RAM). Writes a synthetic mlx-community safetensors dir, opens
//! it both ways, and compares every tensor.
//!
//!   cargo run --release -p rlx-models-core --example mlx_lazy_loader_probe

use anyhow::Result;
use rlx_models_core::weight_loader::{MlxLoader, WeightLoader};
use std::collections::HashMap;

fn main() -> Result<()> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("mlx_lazy_probe");
    std::fs::create_dir_all(&dir)?;

    // ── Synthetic mlx dir: a few f32 tensors of assorted shapes + config.json ──
    let tensors: Vec<(&str, Vec<usize>)> = vec![
        ("model.embed_tokens.weight", vec![6, 8]),
        ("model.layers.0.mlp.gate_proj.weight", vec![10, 8]),
        ("model.norm.weight", vec![8]),
        ("lm_head.weight", vec![6, 8]),
    ];
    let data: HashMap<String, Vec<f32>> = tensors
        .iter()
        .map(|(n, sh)| {
            let len: usize = sh.iter().product();
            let v: Vec<f32> = (0..len)
                .map(|i| {
                    let x =
                        ((i as f64 + 1.0) * (n.len() as f64 + 1.3) * 12.9898).sin() * 43758.5453;
                    (x - x.floor()) as f32 - 0.5
                })
                .collect();
            (n.to_string(), v)
        })
        .collect();
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(n, _)| data[*n].iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = tensors
        .iter()
        .zip(&byte_bufs)
        .map(|((n, sh), bytes)| {
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, sh.clone(), bytes)
                    .unwrap();
            (n.to_string(), view)
        })
        .collect();
    let meta: Option<HashMap<String, String>> = None;
    safetensors::serialize_to_file(views, meta, &dir.join("model.safetensors"))
        .map_err(|e| anyhow::anyhow!("write safetensors: {e}"))?;
    std::fs::write(dir.join("config.json"), br#"{"model_type":"llama"}"#)?;

    let dir_s = dir.to_str().unwrap();

    // ── Compare eager vs lazy: every logical key, byte-identical ──
    let mut eager = MlxLoader::open(dir_s)?;
    let mut lazy = MlxLoader::open_lazy(dir_s)?;
    let mut keys = lazy.remaining_keys();
    keys.sort();
    assert!(!keys.is_empty(), "no keys");

    let mut max_err = 0f32;
    for k in &keys {
        let (e, es) = eager.take(k)?;
        let (l, ls) = lazy.take(k)?;
        assert_eq!(es, ls, "{k}: shape mismatch eager {es:?} vs lazy {ls:?}");
        assert_eq!(e.len(), l.len(), "{k}: len mismatch");
        let err = e
            .iter()
            .zip(&l)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        max_err = max_err.max(err);
        assert_eq!(&data[k], &l, "{k}: lazy data != original");
    }

    println!("── lazy (mmap) MLX loader vs eager ──");
    println!(
        "compared {} tensors: max|eager-lazy| = {max_err:.3e}",
        keys.len()
    );
    if max_err == 0.0 {
        println!("✅ lazy loader is byte-identical to eager (and materializes on demand)");
        Ok(())
    } else {
        Err(anyhow::anyhow!("lazy loader diverged: max|err| {max_err}"))
    }
}
