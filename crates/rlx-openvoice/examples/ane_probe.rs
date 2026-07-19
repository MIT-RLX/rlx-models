//! Compile OpenVoice tone graphs on CoreML/ANE.
//!   cargo run -p rlx-openvoice --release --features coreml --example ane_probe

use std::path::PathBuf;
use std::time::Instant;

use rlx_runtime::Device;
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "weights/tts/openvoice".into()),
    );
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 22_050,
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
    for (comp, len, named) in [
        (
            "tone_extract",
            64usize,
            vec![("source_audio_len", 64usize)] as Vec<(&str, usize)>,
        ),
        ("tone_color", 64, vec![("target_audio_len", 64)]),
    ] {
        eprintln!("compile {comp} on Ane…");
        let t0 = Instant::now();
        match model.compile_named(comp, Device::Ane, len, &named) {
            Ok(_) => eprintln!("OK in {:.1}s", t0.elapsed().as_secs_f64()),
            Err(e) => eprintln!("ERR after {:.1}s: {e:#}", t0.elapsed().as_secs_f64()),
        }
    }
    Ok(())
}
