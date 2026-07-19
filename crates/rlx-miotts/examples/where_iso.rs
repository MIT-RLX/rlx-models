use rlx_runtime::Device;
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
    let mut g = model.compile_named("where_bool_isolated", Device::Cpu, 100, &[])?;
    let x = std::fs::read(dir.join("where_in_mask.f32"))?;
    let outs = g.run_typed(&[("x", &x, rlx_runtime::DType::F32)]);
    let y: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let reference: Vec<f32> = std::fs::read(dir.join("where_bool_isolated_ref.f32"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut mism = 0;
    for i in 0..y.len().min(reference.len()) {
        let a = y[i];
        let b = reference[i];
        let ok = (a == b)
            || (a.is_infinite() && b.is_infinite() && a.is_sign_negative() == b.is_sign_negative());
        if !ok {
            mism += 1;
            if mism <= 8 {
                println!("mis[{i}] rlx={a} ref={b}");
            }
        }
    }
    println!(
        "rlx[:8]={:?} ref[:8]={:?} mism={mism}/{}",
        &y[..8],
        &reference[..8],
        y.len()
    );
    Ok(())
}
