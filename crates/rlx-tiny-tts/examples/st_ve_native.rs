// Native run of supertonic vector_estimator at DISTINCT lengths (text≠latent),
// reading the exact inputs the python generator wrote, so the output can be
// compared bit-for-bit against onnxruntime. Proves the multi-length import path
// end-to-end on the hardest (cross-attention CFM) graph.
//
// Usage: cargo run -p rlx-tiny-tts --example st_ve_native
use rlx_runtime::DType;
use rlx_tiny_tts::model::TinyModel;
use rlx_tiny_tts::{BundleConfig, Device};
use std::path::PathBuf;

fn read_f32(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))
}

fn main() {
    // Work dir (parity fixtures) and onnx dir are machine-independent: the work
    // dir defaults under the OS temp dir, the onnx dir is resolved relative to
    // this crate (`CARGO_MANIFEST_DIR/../../weights/...`). Both overridable by env.
    let d = std::env::var("RLX_ST_WORK")
        .unwrap_or_else(|_| format!("{}/st_ve_parity", std::env::temp_dir().display()));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{d}/meta.json")).unwrap()).unwrap();
    let t = meta["t"].as_u64().unwrap() as usize;
    let l = meta["l"].as_u64().unwrap() as usize;

    let onnx_dir = PathBuf::from(std::env::var("RLX_ST_ONNX_DIR").unwrap_or_else(|_| {
        format!(
            "{}/../../weights/tts/supertonic-3/onnx",
            env!("CARGO_MANIFEST_DIR")
        )
    }));
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 44100,
        add_blank: true,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        length_scale: 1.0,
        inter_channels: 80,
        gin_channels: 80,
    };
    let m = TinyModel::new(onnx_dir, cfg);
    let device = match std::env::var("RLX_DEV").as_deref() {
        Ok("mlx") => Device::Mlx,
        Ok("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    eprintln!("[st_ve] device={device:?} t={t} l={l}");
    // Bind the two distinct dynamic lengths. `length` is a fallback for any other
    // dim; text_length / latent_length are overridden explicitly.
    let mut g = m
        .compile_named(
            "vector_estimator",
            device,
            l,
            &[("text_length", t), ("latent_length", l)],
        )
        .expect("compile vector_estimator (named lengths)");

    let nl = read_f32(&format!("{d}/noisy_latent"));
    let te = read_f32(&format!("{d}/text_emb"));
    let st = read_f32(&format!("{d}/style_ttl"));
    let lm = read_f32(&format!("{d}/latent_mask"));
    let tm = read_f32(&format!("{d}/text_mask"));
    let cs = read_f32(&format!("{d}/current_step"));
    let ts = read_f32(&format!("{d}/total_step"));
    let inputs: Vec<(&str, &[u8], DType)> = vec![
        ("noisy_latent", &nl, DType::F32),
        ("text_emb", &te, DType::F32),
        ("style_ttl", &st, DType::F32),
        ("latent_mask", &lm, DType::F32),
        ("text_mask", &tm, DType::F32),
        ("current_step", &cs, DType::F32),
        ("total_step", &ts, DType::F32),
    ];
    let out = g.run_typed(&inputs);
    // Report finiteness of every output (main + RLX_ONNX_TAP taps) to localize
    // where nan/inf is born. Tap order matches RLX_ONNX_TAP, appended after out[0].
    let tap_names: Vec<String> = std::env::var("RLX_ONNX_TAP")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    for (i, (bytes, dt)) in out.iter().enumerate() {
        let f: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let nan = f.iter().filter(|v| v.is_nan()).count();
        let inf = f.iter().filter(|v| v.is_infinite()).count();
        let label = if i == 0 {
            "OUT"
        } else {
            tap_names.get(i - 1).map(String::as_str).unwrap_or("?")
        };
        let amax = f
            .iter()
            .filter(|v| v.is_finite())
            .fold(0f32, |a, v| a.max(v.abs()));
        let n = f.len();
        println!("[{i}] {label}: {n} f32 dtype={dt:?} nan={nan} inf={inf} amax={amax:.3e}");
        std::fs::write(format!("{d}/rlx_tap_{i}.f32"), bytes).unwrap();
    }
    std::fs::write(format!("{d}/rlx_out.f32"), &out[0].0).unwrap();
}
