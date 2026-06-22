//! Synthesize speech from text with Inflect-Nano-v1.
//!
//! cargo run -p rlx-inflect-nano --example synthesize --release -- \
//!     --data weights/inflect-nano-rlx --text "Hello, world!" --out out.wav

use std::path::PathBuf;

use anyhow::Result;
use rlx_inflect_nano::{InferOpts, InflectNano};

fn arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() -> Result<()> {
    let data = arg("--data", "weights/inflect-nano-rlx");
    let text = arg(
        "--text",
        "The weather is nice today, and I feel very relaxed.",
    );
    let out = arg("--out", "inflect_nano_out.wav");
    let opts = InferOpts {
        length_scale: arg("--length-scale", "1.0").parse().unwrap_or(1.0),
        pitch_scale: arg("--pitch-scale", "1.0").parse().unwrap_or(1.0),
        energy_scale: arg("--energy-scale", "1.0").parse().unwrap_or(1.0),
        ..Default::default()
    };

    let model = InflectNano::load_from_dir(&PathBuf::from(&data))?;
    let t0 = std::time::Instant::now();
    let wav = model.synthesize(&text, &opts)?;
    let secs = t0.elapsed().as_secs_f32();
    let audio_secs = wav.samples.len() as f32 / wav.sample_rate as f32;
    rlx_inflect_nano::audio::write_wav(&PathBuf::from(&out), &wav.samples, wav.sample_rate)?;
    println!(
        "wrote {out} ({audio_secs:.2}s audio in {secs:.3}s, RTF {:.2}x)",
        audio_secs / secs
    );
    Ok(())
}
