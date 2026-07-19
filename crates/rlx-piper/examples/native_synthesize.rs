// Native (ort-free) piper synthesis over RLX.
// Usage:
//   cargo run -p rlx-piper --no-default-features --features native,espeak \
//     --example native_synthesize -- --dir weights/tts/piper \
//     --text "Hello from native piper." --out out.wav [--device cpu|metal|mlx|gpu]
use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_piper::NativeVits;
use rlx_runtime::parse_device;

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn main() -> Result<()> {
    let dir = PathBuf::from(arg("--dir", "weights/tts/piper"));
    let text = arg("--text", "Hello from native piper.");
    let out = PathBuf::from(arg("--out", "native_piper.wav"));
    let device = parse_device(&arg("--device", "cpu")).context("device")?;

    let tts = NativeVits::load(&dir, device).context("load native piper")?;
    let t0 = std::time::Instant::now();
    let audio = tts.synthesize(&text, None).context("synthesize")?;
    let dt = t0.elapsed().as_secs_f32();
    let secs = audio.len() as f32 / tts.sample_rate() as f32;
    let peak = audio.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    tts.write_wav(&audio, &out)?;
    println!(
        "wrote {} ({:.2}s audio, peak {:.3}) in {:.2}s → RTF {:.1}× on {:?}",
        out.display(),
        secs,
        peak,
        dt,
        secs / dt,
        device
    );
    Ok(())
}
