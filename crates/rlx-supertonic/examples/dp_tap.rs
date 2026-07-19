// duration_predictor per-op bisection with FIXED inputs. Run on cpu and wgpu
// (RLX_DEV) and diff per-tap amax to find the first op that diverges.
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
    let t = 20usize;
    let text_ids: Vec<i64> = (0..t).map(|i| (i % 50 + 1) as i64).collect();
    let style_dp: Vec<f32> = (0..8 * 16).map(|i| (i % 13) as f32 / 13.0 - 0.5).collect();
    let text_mask: Vec<f32> = vec![1.0; t];
    let ib: Vec<u8> = text_ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let sb: Vec<u8> = style_dp.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mb: Vec<u8> = text_mask.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut g = model
        .compile_named("duration_predictor", dev, t, &[("text_length", t)])
        .map_err(|e| anyhow::anyhow!("compile dp: {e:#}"))?;
    let out = g.run_typed(&[
        ("text_ids", &ib, DType::I64),
        ("style_dp", &sb, DType::F32),
        ("text_mask", &mb, DType::F32),
    ]);
    let taps: Vec<String> = std::env::var("RLX_ONNX_TAP")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    for (i, (bytes, _)) in out.iter().enumerate() {
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let rms = (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len().max(1) as f64)
            .sqrt();
        let label = if i == 0 {
            "duration(out)".to_string()
        } else {
            taps.get(i - 1).cloned().unwrap_or_default()
        };
        eprintln!(
            "[dp_tap {dev:?}] [{i:2}] amax={amax:10.4} rms={rms:9.4} n={:6}  {label}",
            v.len()
        );
    }
    Ok(())
}
