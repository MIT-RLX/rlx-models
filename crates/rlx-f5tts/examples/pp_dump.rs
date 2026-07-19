//! Dump F5 preprocess intermediates (`RLX_F5_DBG=1`, `RLX_F5_DUMP_PP=…`,
//! `RLX_F5_STOP_AFTER_PRE=1`) for CPU↔CUDA parity.
//!
//! ```bash
//! RLX_F5_DBG=1 RLX_F5_STOP_AFTER_PRE=1 DEV=cpu cargo run -p rlx-f5tts --release --example pp_dump
//! RLX_F5_DBG=1 RLX_F5_STOP_AFTER_PRE=1 DEV=cuda cargo run -p rlx-f5tts --release --example pp_dump --features cuda
//! ```

use std::path::PathBuf;

use rlx_f5tts::{DEFAULT_LOCAL_DIR, F5Native, InferOpts};
use rlx_runtime::Device;

fn main() -> anyhow::Result<()> {
    let model = std::env::var("RLX_F5TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOCAL_DIR));
    let ref_path = std::env::var("RLX_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prompt.wav")
        });
    let (reference, _) = read_wav(&ref_path)?;
    let text = std::env::var("RLX_TEXT")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog.".into());
    let ref_text = std::env::var("RLX_REF_TEXT").unwrap_or_else(|_| {
        "Hello from Kokoro. This is a test of speech synthesis in Rust.".into()
    });
    let opts = InferOpts { nfe: 2, speed: 1.0 };
    let label = std::env::var("DEV").unwrap_or_else(|_| "cpu".into());
    let dev = match label.as_str() {
        "cuda" => Device::Cuda,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        _ => Device::Cpu,
    };
    eprintln!("device={dev:?} model={}", model.display());
    let tts = F5Native::load_on(&model, dev)?;
    let _ = tts.synthesize(&text, &reference, &ref_text, &opts)?;
    Ok(())
}

fn read_wav(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    Ok((samples, spec.sample_rate))
}
