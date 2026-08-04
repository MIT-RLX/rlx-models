// Parakeet-TDT end-to-end transcription.
//   RLX_PARAKEET_NEMO=<parakeet.nemo> [RLX_DEV=metal] \
//     cargo run -p rlx-parakeet --example transcribe -- <audio.wav>
use std::path::Path;

use anyhow::{Context, Result};
use rlx_nemotron_asr::wav;
use rlx_parakeet::Parakeet;
use rlx_runtime::Device;

fn main() -> Result<()> {
    let nemo = std::env::var("RLX_PARAKEET_NEMO")
        .context("set RLX_PARAKEET_NEMO to a Parakeet-TDT .nemo checkpoint")?;
    let wav_path = std::env::args()
        .nth(1)
        .context("usage: transcribe <audio.wav>")?;
    let dev = match std::env::var("RLX_DEV").as_deref() {
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        _ => Device::Cpu,
    };

    let pk = Parakeet::open(Path::new(&nemo), dev)?;
    let target_sr = pk.config().sample_rate as u32;

    let bytes = std::fs::read(&wav_path).with_context(|| format!("read {wav_path}"))?;
    let w = wav::parse(&bytes)?;
    let pcm = if w.sample_rate != target_sr {
        wav::resample(&w.samples, w.sample_rate, target_sr)
    } else {
        w.samples
    };
    eprintln!(
        "[parakeet] device={dev:?} durations={:?} samples={} sr={target_sr}",
        pk.durations(),
        pcm.len()
    );
    let text = pk.transcribe(&pcm)?;
    println!("{text}");
    Ok(())
}
