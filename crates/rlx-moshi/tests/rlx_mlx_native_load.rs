// Native (candle-free) MLX safetensors loader: build a tiny synthetic MLX-format
// file (Q4-packed weight + bf16 tensors + 1-D norm alpha), load via
// `load_eager_weight_map`, and check dequant values, key remapping, and the
// alpha `[d] → [1,1,d]` reshape. Validates the file→map plumbing without the
// 28 GB real model (the affine dequant itself is unit-tested in mlx_dequant).

use rlx_moshi::checkpoint::MoshiCheckpoint;
use rlx_moshi::config::LmConfig;
use rlx_moshi::mlx_dequant::f32_to_bf16;
use rlx_moshi::mlx_weights::load_eager_weight_map;
use safetensors::Dtype;
use safetensors::serialize;
use safetensors::tensor::TensorView;

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
        .collect()
}
fn u32_bytes(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

#[test]
fn native_mlx_q4_loads() {
    // Q4 (bits=4, group=32): one [1,4] u32 weight → [1,32] dequant.
    // 0x7654_3210 packs codes 0..7 LSB-first; scale=2, bias=-1 → 2*code-1.
    let packed = u32_bytes(&[0x7654_3210u32; 4]);
    let scales = bf16_bytes(&[2.0]);
    let biases = bf16_bytes(&[-1.0]);
    let emb = bf16_bytes(&[0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]); // [2,4]
    let alpha = bf16_bytes(&[1.0, 1.1, 0.9, 1.2]); // [4] norm

    let tensors: Vec<(String, TensorView)> = vec![
        (
            "transformer.layers.0.self_attn.in_proj.weight".to_string(),
            TensorView::new(Dtype::U32, vec![1, 4], &packed).unwrap(),
        ),
        (
            "transformer.layers.0.self_attn.in_proj.scales".to_string(),
            TensorView::new(Dtype::BF16, vec![1, 1], &scales).unwrap(),
        ),
        (
            "transformer.layers.0.self_attn.in_proj.biases".to_string(),
            TensorView::new(Dtype::BF16, vec![1, 1], &biases).unwrap(),
        ),
        (
            "text_emb.weight".to_string(),
            TensorView::new(Dtype::BF16, vec![2, 4], &emb).unwrap(),
        ),
        (
            "out_norm.weight".to_string(),
            TensorView::new(Dtype::BF16, vec![4], &alpha).unwrap(),
        ),
    ];
    let bytes = serialize(tensors, &None).unwrap();
    let dir = std::env::temp_dir().join(format!("rlx_mlx_native_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.q4.safetensors");
    std::fs::write(&path, &bytes).unwrap();

    let cfg = LmConfig::v0_1();
    let map =
        load_eager_weight_map(&path, MoshiCheckpoint::Q4MlxSafetensors, &cfg).expect("load mlx");

    // Quantized weight: key remapped `.in_proj.weight` → `.in_proj_weight`,
    // shape [1,32], dequant = 2*code-1 with code = (col % 8).
    let (w, ws) = &map["transformer.layers.0.self_attn.in_proj_weight"];
    assert_eq!(ws, &vec![1, 32], "dequant shape");
    let expect: Vec<f32> = (0..32).map(|i| 2.0 * (i % 8) as f32 - 1.0).collect();
    let max_err = w
        .iter()
        .zip(expect.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-3, "dequant values off: {max_err}");

    // bf16 tensor passes through.
    let (e, es) = &map["text_emb.weight"];
    assert_eq!(es, &vec![2, 4]);
    assert!((e[0] - 0.1).abs() < 1e-2 && (e[4] + 0.1).abs() < 1e-2);

    // 1-D norm `out_norm.weight` → `out_norm.alpha`, reshaped [d] → [1,1,d].
    let (al, als) = &map["out_norm.alpha"];
    assert_eq!(als, &vec![1, 1, 4], "alpha reshape");
    assert_eq!(al.len(), 4);
    assert!((al[0] - 1.0).abs() < 1e-2);

    std::fs::remove_dir_all(&dir).ok();
}
