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
    let mut g = model.compile_named("in_only", Device::Cpu, 200, &[])?;
    let x = std::fs::read(dir.join("stage_tap_3.f32"))?;
    let outs = g.run_typed(&[("/Resize_output_0", &x, DType::F32)]);
    let labels = ["rs", "inn"];
    let refs = ["in_only_rs.f32", "in_only_in.f32"];
    for (i, (name, rf)) in labels.iter().zip(refs).enumerate() {
        let v: Vec<f32> = outs[i]
            .0
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let reference: Vec<f32> = std::fs::read(dir.join(rf))?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!(
            "[{name}] cos={:.4} n={} rlx[:4]={:?} ref[:4]={:?}",
            cos(&v, &reference),
            v.len(),
            &v[..4],
            &reference[..4]
        );
    }
    Ok(())
}
