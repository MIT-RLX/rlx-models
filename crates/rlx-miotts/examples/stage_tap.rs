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
    let mut g = model.compile_named("decoder_stages", Device::Cpu, 100, &[])?;
    let tokens: Vec<u32> = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        dir.join("fox_tokens.json"),
    )?)?["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut t = tokens;
    t.resize(100, 0);
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
    // mag, phase, then stage_tap_0..5
    let labels = [
        "mag",
        "phase",
        "proj",
        "transpose",
        "convT",
        "resize",
        "prior",
        "dec0",
    ];
    for (i, name) in labels.iter().enumerate() {
        let v: Vec<f32> = outs[i]
            .0
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let rfile = if i < 2 {
            format!("{name}.f32")
        } else {
            format!("stage_tap_{}.f32", i - 2)
        };
        // mag/phase refs from earlier body_tap run
        let rpath = if i == 0 {
            dir.join("mag.f32")
        } else if i == 1 {
            dir.join("phase.f32")
        } else {
            dir.join(&rfile)
        };
        let reference: Vec<f32> = std::fs::read(&rpath)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!(
            "[{name}] cos={:.4} n_rlx={} n_ref={} rlx[:3]={:?} ref[:3]={:?}",
            cos(&v, &reference),
            v.len(),
            reference.len(),
            &v[..3.min(v.len())],
            &reference[..3.min(reference.len())]
        );
    }
    Ok(())
}
