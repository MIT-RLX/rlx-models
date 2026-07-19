use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from("weights/tts/miocodec/fixtures");
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
    let model = TinyModel::new(dir.clone(), cfg);
    let mut g = model.compile_named("prenet_only", Device::Cpu, 100, &[])?;
    let tokens: Vec<u32> = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        "weights/tts/miocodec/fixtures/fox_tokens.json",
    )?)?["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut t = tokens;
    t.resize(100, 0);
    t.truncate(100);
    let idx: Vec<i64> = t.iter().map(|&x| x as i64).collect();
    let emb: Vec<f32> = std::fs::read(dir.join("en_female.f32"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let eb: Vec<u8> = emb.iter().flat_map(|x| x.to_le_bytes()).collect();
    let outs = g.run_typed(&[
        ("content_token_indices", &ib, DType::I64),
        ("global_embedding", &eb, DType::F32),
    ]);
    let v: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let reference: Vec<f32> = std::fs::read(dir.join("prenet_ref.f32"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = v.len().min(reference.len());
    let maxd = (0..n)
        .map(|i| (v[i] - reference[i]).abs())
        .fold(0.0f32, f32::max);
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let dot: f64 = (0..n).map(|i| v[i] as f64 * reference[i] as f64).sum();
    let na: f64 = v[..n]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = reference[..n]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb + 1e-12);
    println!(
        "elems={} maxd={maxd:.4e} cos={cos:.4} nans={nan} rlx0={} ref0={}",
        v.len(),
        v[0],
        reference[0]
    );
    Ok(())
}
