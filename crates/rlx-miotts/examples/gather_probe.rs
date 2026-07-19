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
    let mut g = model.compile_named("gather_only", Device::Cpu, 4, &[])?;
    let idx: Vec<i64> = vec![5051, 11221, 0, 1];
    let bytes: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let outs = g.run_typed(&[("idx", &bytes, DType::I64)]);
    let v: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let reference: Vec<f32> = std::fs::read(dir.join("gather_ref.f32"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("rlx elems={} ref={}", v.len(), reference.len());
    let n = v.len().min(reference.len());
    let maxd = (0..n)
        .map(|i| (v[i] - reference[i]).abs())
        .fold(0.0f32, f32::max);
    println!(
        "max|d|={maxd:.6e} rlx_head={:?} ref_head={:?}",
        &v[..4],
        &reference[..4]
    );
    Ok(())
}
