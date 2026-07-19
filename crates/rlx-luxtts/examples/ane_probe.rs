//! Timed CoreML/ANE compile probe for LuxTTS subgraphs.
//!
//! ```bash
//! cargo run -p rlx-luxtts --release --features coreml --example ane_probe -- \
//!   weights/tts/luxtts encoder_body 64
//! ```

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::Device;
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "weights/tts/luxtts".into()),
    );
    let component = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "encoder_body".into());
    let len: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

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
    let model = TinyModel::new(dir, cfg);
    let named: Vec<(&str, usize)> = match component.as_str() {
        "encoder_body" => vec![("S", len)],
        "fm_decoder" => vec![("T", len)],
        "vocoder_spec" => vec![("l", len)],
        _ => vec![],
    };
    eprintln!("compile {component} len={len} named={named:?} on Ane…");
    let t0 = Instant::now();
    match model.compile_named(&component, Device::Ane, len, &named) {
        Ok(_g) => eprintln!("OK in {:.1}s (graph ready)", t0.elapsed().as_secs_f64()),
        Err(e) => eprintln!("ERR after {:.1}s: {e:#}", t0.elapsed().as_secs_f64()),
    }
    Ok(())
}
