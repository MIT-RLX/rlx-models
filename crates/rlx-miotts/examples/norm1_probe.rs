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
    // Input name is /Resize_output_0 — TinyModel may need sanitized names
    let mut g = model.compile_named("norm1_only", Device::Cpu, 200, &[])?;
    let x = std::fs::read(dir.join("stage_tap_3.f32"))?;
    // Try common input names
    let input_names = ["/Resize_output_0", "Resize_output_0", "input", "x"];
    let mut outs = None;
    for name in input_names {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.run_typed(&[(name, &x, DType::F32)])
        }));
        if let Ok(o) = r {
            println!("input name ok: {name}");
            outs = Some(o);
            break;
        }
    }
    let outs = outs.expect("no input name worked");
    for (i, (bytes, _)) in outs.iter().enumerate() {
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("out[{i}] n={} [:4]={:?}", v.len(), &v[..4.min(v.len())]);
    }
    let y: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ref_files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("norm1_"))
        .collect();
    println!("ref files: {ref_files:?}");
    for rf in &ref_files {
        if rf.contains("Add") || rf.contains("tap") {
            let reference: Vec<f32> = std::fs::read(dir.join(rf))?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            println!(
                "vs {rf}: cos={:.4} rlx[:3]={:?} ref[:3]={:?}",
                cos(&y, &reference),
                &y[..3],
                &reference[..3.min(reference.len())]
            );
        }
    }
    Ok(())
}
