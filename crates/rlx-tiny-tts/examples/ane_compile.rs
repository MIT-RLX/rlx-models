//! Compile one TinyModel ONNX component on CoreML/ANE.
//!   cargo run -p rlx-tiny-tts --release --features coreml --example ane_compile -- \
//!     <onnx_dir> <component> [length]

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::Device;
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::args().nth(1).expect("onnx_dir"));
    let component = std::env::args().nth(2).expect("component");
    let len: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 24_000,
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
    eprintln!("compile {component} len={len} on Ane…");
    let t0 = Instant::now();
    match model.compile_named(&component, Device::Ane, len, &[]) {
        Ok(_) => eprintln!("OK in {:.1}s", t0.elapsed().as_secs_f64()),
        Err(e) => eprintln!("ERR after {:.1}s: {e:#}", t0.elapsed().as_secs_f64()),
    }
    Ok(())
}
