// Vocoder per-op bisection: feed a FIXED latent to the native vocoder subgraph
// with RLX_ONNX_TAP intermediates exposed, print amax per tap. Run on cpu and
// wgpu (RLX_DEV=cpu|gpu) and diff to find the first op that diverges.
//   RLX_ONNX_TAP="a,b,c" RLX_DEV=gpu cargo run --release -p rlx-supertonic \
//     --no-default-features --features gpu --example voc_tap
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../weights/tts/supertonic-3/onnx");
    let dev = match std::env::var("RLX_DEV").as_deref() {
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        Ok("gpu") => Device::Gpu,
        Ok("ane") | Ok("coreml") => Device::Ane,
        _ => Device::Cpu,
    };
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 44100,
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
    let l = 22usize;
    let ch = 144usize;
    // Deterministic fixed latent [1, 144, L]: a smooth pattern (identical on all backends).
    let latent: Vec<f32> = (0..ch * l)
        .map(|i| ((i % 37) as f32 / 37.0 - 0.5) * 1.5)
        .collect();
    let lat_b: Vec<u8> = latent.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut g = model
        .compile_named("vocoder", dev, l, &[("latent_length", l)])
        .map_err(|e| anyhow::anyhow!("compile vocoder: {e:#}"))?;
    let out = g.run_typed(&[("latent", &lat_b, DType::F32)]);
    let taps: Vec<String> = std::env::var("RLX_ONNX_TAP")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    eprintln!(
        "[voc_tap {dev:?}] {} outputs ({} real + {} taps)",
        out.len(),
        out.len().saturating_sub(taps.len()),
        taps.len()
    );
    for (i, (bytes, _dt)) in out.iter().enumerate() {
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let rms = (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len().max(1) as f64)
            .sqrt();
        let label = if i == 0 {
            "wav_tts(out)".to_string()
        } else {
            taps.get(i - 1).cloned().unwrap_or_default()
        };
        eprintln!(
            "[voc_tap {dev:?}]  [{i:2}] amax={amax:9.4} rms={rms:8.4}  n={:7}  {label}",
            v.len()
        );
    }
    Ok(())
}
