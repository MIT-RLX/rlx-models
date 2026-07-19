use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        if x.is_finite() && y.is_finite() {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from("weights/tts/miocodec");
    let fx = dir.join("fixtures");
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 24000,
        add_blank: false,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        length_scale: 1.0,
        inter_channels: 0,
        gin_channels: 0,
    };
    let model = TinyModel::new(dir, cfg);
    let mut g = model.compile_named("decoder_body", Device::Cpu, 100, &[])?;
    let tokens: Vec<u32> = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        fx.join("fox_tokens.json"),
    )?)?["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut t = tokens;
    t.resize(100, 0);
    let idx: Vec<i64> = t.iter().map(|&x| x as i64).collect();
    let emb: Vec<f32> = std::fs::read(fx.join("en_female.f32"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let eb: Vec<u8> = emb.iter().flat_map(|x| x.to_le_bytes()).collect();
    let outs = g.run_typed(&[
        ("content_token_indices", &ib, DType::I64),
        ("global_embedding", &eb, DType::F32),
    ]);
    let mag: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let phase: Vec<f32> = outs[1]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mag_ref: Vec<f32> = std::fs::read(fx.join("fox_mag_ort.f32"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let phase_ref: Vec<f32> = std::fs::read(fx.join("fox_phase_ort.f32"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!(
        "mag cos={:.4} n={} rlx[:4]={:?} ref[:4]={:?}",
        cos(&mag, &mag_ref),
        mag.len(),
        &mag[..4],
        &mag_ref[..4]
    );
    println!(
        "phase cos={:.4} n={} rlx[:4]={:?} ref[:4]={:?}",
        cos(&phase, &phase_ref),
        phase.len(),
        &phase[..4],
        &phase_ref[..4]
    );
    Ok(())
}
