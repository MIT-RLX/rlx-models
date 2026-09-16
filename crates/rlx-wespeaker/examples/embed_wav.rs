//! Embed a 16 kHz mono WAV with native WeSpeaker.
//!
//! ```bash
//! cargo run -p rlx-wespeaker --release --example embed_wav --features "native,metal" -- \
//!   clip.wav weights/wespeaker-voxceleb-resnet34-LM metal
//! ```

use std::path::PathBuf;

use anyhow::Result;
use rlx_runtime::Device;
use rlx_wespeaker::WeSpeaker;

fn parse_device(s: &str) -> Device {
    match s.to_ascii_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "ane" | "coreml" => Device::Ane,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let wav = PathBuf::from(
        args.next()
            .expect("usage: embed_wav <wav> [model_dir] [device]"),
    );
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "weights/wespeaker-voxceleb-resnet34-LM".into()),
    );
    let device = parse_device(&args.next().unwrap_or_else(|| "cpu".into()));

    let mut reader = hound::WavReader::open(&wav)?;
    let spec = reader.spec();
    anyhow::ensure!(spec.channels == 1, "mono required");
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;

    let mut spk = WeSpeaker::open_on(&dir, device)?;
    let t0 = std::time::Instant::now();
    let emb = spk.embed_pcm(&pcm)?;
    println!(
        "device={:?} dim={} norm={:.4} elapsed={:.3}s first={:?}",
        spk.device(),
        emb.len(),
        emb.iter().map(|x| x * x).sum::<f32>().sqrt(),
        t0.elapsed().as_secs_f32(),
        &emb[..4]
    );
    Ok(())
}
