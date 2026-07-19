// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Native (ort-free) Kokoro synthesis: text → 24 kHz WAV through the graph-split
//! RLX path. Requires the split bundle under `<model>/onnx/rlx-split/` (produce
//! it with `scripts/split_kokoro.py`).
//!
//! ```bash
//! cargo run -p rlx-kokoro --no-default-features --features native,espeak \
//!   --example native_synthesize -- \
//!   --model weights/tts/kokoro-82m --voice af_heart \
//!   --text "Hello from native Kokoro." --out out.wav
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use rlx_kokoro::{Device, NativeKokoro, write_wav};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let opt = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };

    let model = PathBuf::from(opt("--model").unwrap_or_else(|| "weights/tts/kokoro-82m".into()));
    let voice = opt("--voice").unwrap_or_else(|| "af_heart".into());
    let text = opt("--text").unwrap_or_else(|| "Hello from native Kokoro.".into());
    let out = PathBuf::from(opt("--out").unwrap_or_else(|| "kokoro_native.wav".into()));
    let speed: f32 = opt("--speed").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let device = match opt("--device").as_deref() {
        Some("metal") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("ane") | Some("coreml") => Device::Ane,
        _ => Device::Cpu,
    };

    let t = std::time::Instant::now();
    let tts = NativeKokoro::load(&model, device).context("load native Kokoro")?;
    eprintln!(
        "[native] loaded {} voices on {device:?} in {:?}",
        tts.voice_names().len(),
        t.elapsed()
    );

    let t = std::time::Instant::now();
    let audio = tts.generate_from_text(&text, &voice, speed)?;
    let synth = t.elapsed();
    let dur_s = audio.len() as f32 / tts.sample_rate() as f32;
    eprintln!(
        "[native] {device:?} '{text}' → {} samples ({dur_s:.2}s audio, {:.2}s synth, {:.1}× RT)",
        audio.len(),
        synth.as_secs_f32(),
        dur_s / synth.as_secs_f32().max(1e-6)
    );

    write_wav(&audio, &out)?;
    eprintln!("[native] wrote {}", out.display());
    Ok(())
}
