use rlx_runtime::Device;
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

fn cos_finite(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            let x = a[i] as f64;
            let y = b[i] as f64;
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
    let mut g = model.compile_named("prenet_mask_tapped", Device::Cpu, 100, &[])?;
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
    let ib: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let outs = g.run_typed(&[("content_token_indices", &ib, rlx_runtime::DType::I64)]);
    let labels = ["h", "trilu", "expand0", "expand1", "where2"];
    let refs = [
        "prenet_ref.f32",
        "mask_tap_0.f32",
        "mask_tap_1.f32",
        "mask_tap_2.f32",
        "mask_tap_3.f32",
    ];
    for (i, ((name, rfile), (bytes, _))) in labels.iter().zip(refs).zip(outs.iter()).enumerate() {
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let reference: Vec<f32> = std::fs::read(dir.join(rfile))?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let c = cos_finite(&v, &reference);
        println!(
            "[{i}] {name}: cos={c:.4} n={} rlx[:8]={:?} ref[:8]={:?}",
            v.len().min(reference.len()),
            &v[..8.min(v.len())],
            &reference[..8.min(reference.len())]
        );
    }
    Ok(())
}
