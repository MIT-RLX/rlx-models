// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native Parler-TTS synthesis: text → PCM → wav.
//!
//! ```text
//! cargo run -p rlx-parlertts --example synthesize --release -- "Hello world." out.wav
//! ```

use std::path::Path;

use anyhow::Result;
use rlx_parlertts::{
    DEFAULT_DAC_DIR, DEFAULT_DESCRIPTION, DEFAULT_LOCAL_DIR, InferOpts, NativeParler,
    peak_amplitude,
};
use rlx_runtime::{Device, parse_device};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let text = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".into());
    let out = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "parler_out.wav".into());
    let description =
        std::env::var("RLX_PARLER_VOICE").unwrap_or_else(|_| DEFAULT_DESCRIPTION.into());

    let dir = std::env::var("RLX_PARLER_DIR").unwrap_or_else(|_| DEFAULT_LOCAL_DIR.into());
    let dac = std::env::var("RLX_DAC_DIR").unwrap_or_else(|_| DEFAULT_DAC_DIR.into());
    let device = std::env::var("RLX_DEVICE")
        .ok()
        .and_then(|s| parse_device(&s).ok())
        .unwrap_or(Device::Cpu);
    let max_steps = std::env::var("RLX_PARLER_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128usize);
    let greedy = std::env::var_os("RLX_PARLER_GREEDY").is_some();

    let t0 = std::time::Instant::now();
    let p = NativeParler::open(&dir, &dac, device)?;
    eprintln!("[load {:?}] device={device:?}", t0.elapsed());

    let opts = InferOpts {
        max_steps,
        greedy,
        ..Default::default()
    };
    let t1 = std::time::Instant::now();
    let pcm = p.synthesize(&text, &description, &opts)?;
    let secs = pcm.len() as f32 / p.sample_rate() as f32;
    eprintln!(
        "[synth {:?}] {} samples = {:.2}s @ {}Hz, peak {:.3}, RTF {:.1}×",
        t1.elapsed(),
        pcm.len(),
        secs,
        p.sample_rate(),
        peak_amplitude(&pcm),
        secs / t1.elapsed().as_secs_f32().max(1e-6)
    );

    p.write_wav(&pcm, Path::new(&out))?;
    eprintln!("wrote {out}");
    Ok(())
}
