//! MLX Q4/Q8 safetensors key mapping (env `RLX_MOSHI_MLX_Q4`).

use rlx_moshi::{MoshiCheckpoint, mlx_to_candle_key};

#[test]
fn mlx_key_mapping_samples() {
    assert_eq!(
        mlx_to_candle_key("audio_embs.0.weight").as_deref(),
        Some("emb.0.weight")
    );
    assert_eq!(
        mlx_to_candle_key("depformer.slices.0.norm1.weight").as_deref(),
        Some("depformer.0.norm1.alpha")
    );
    assert_eq!(
        mlx_to_candle_key("transformer.layers.0.self_attn.in_proj.weight").as_deref(),
        Some("transformer.layers.0.self_attn.in_proj_weight")
    );
    assert_eq!(
        mlx_to_candle_key("out_norm.weight").as_deref(),
        Some("out_norm.alpha")
    );
    assert_eq!(mlx_to_candle_key("text_linear.scales"), None);
    let _ = MoshiCheckpoint::Q4MlxSafetensors;
}

#[test]
fn mlx_dequant_tensor_count() {
    let path = match std::env::var("RLX_MOSHI_MLX_Q4") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("skip mlx_dequant_tensor_count (set RLX_MOSHI_MLX_Q4)");
            return;
        }
    };
    if !std::path::Path::new(&path).is_file() {
        eprintln!("skip: {path} not found");
        return;
    }
    let names = rlx_moshi::mlx_weights::MlxSafetensorsFile::open(path.as_ref())
        .expect("open")
        .tensor_names()
        .expect("names");
    assert!(names.len() > 500);
    assert!(names.iter().any(|n| n.ends_with(".scales")));
}
