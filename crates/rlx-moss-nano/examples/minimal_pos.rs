//! Isolate the RoPE-positions op bug: run the minimal `Cast→Cast→CumSum→Sub`
//! chain in rlx and compare to onnxruntime. Env: RLX_MOSS_DIR.
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 48000,
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
    let seq = 8usize;
    let mask_b: Vec<u8> = (0..seq).flat_map(|_| 1i32.to_le_bytes()).collect();
    let mut g = model
        .compile_named(
            "minimal_pos",
            Device::Cpu,
            seq,
            &[("batch", 1), ("dim0", 1)],
        )
        .map_err(|e| anyhow::anyhow!("compile: {e:#}"))?;
    let out = g.run_typed(&[("mask", &mask_b, DType::I32)]);
    let names = ["cb(bool)", "ci(i64)", "cs(i64)", "sub(i64)"];
    for (i, (bytes, dt)) in out.iter().enumerate() {
        let s = match dt {
            DType::I64 => format!(
                "i64 {:?}",
                bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .take(8)
                    .collect::<Vec<_>>()
            ),
            DType::I32 => format!(
                "i32 {:?}",
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .take(8)
                    .collect::<Vec<_>>()
            ),
            DType::Bool | DType::U8 => format!("u8 {:?}", &bytes[..8.min(bytes.len())]),
            DType::F32 => format!(
                "f32 {:?}",
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .take(8)
                    .collect::<Vec<_>>()
            ),
            _ => format!("{:?} len={}", dt, bytes.len()),
        };
        eprintln!("  out[{i}] {} = {s}", names.get(i).unwrap_or(&"?"));
    }
    Ok(())
}
