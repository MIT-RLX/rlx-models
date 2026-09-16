//! Diarize a 16 kHz mono WAV (native WeSpeaker on RLX).
//!
//! ```bash
//! cargo run -p rlx-diarize --release --example diarize_wav --features "wespeaker,metal" -- \
//!   clip.wav [model_dir] [device]
//! ```

use std::path::PathBuf;

use anyhow::Result;
use rlx_diarize::{DiarizeConfig, DiarizeSession};
use rlx_runtime::Device;

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
            .expect("usage: diarize_wav <wav> [model_dir] [device]"),
    );
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/wespeaker-voxceleb-resnet34-LM"));
    let device = parse_device(&args.next().unwrap_or_else(|| "cpu".into()));

    let mut reader = hound::WavReader::open(&wav)?;
    let spec = reader.spec();
    anyhow::ensure!(spec.channels == 1, "mono required");
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    eprintln!(
        "pcm {:.1}s @ {}",
        pcm.len() as f32 / spec.sample_rate as f32,
        spec.sample_rate
    );

    let cfg = DiarizeConfig {
        wespeaker_dir: Some(model_dir),
        device,
        ..DiarizeConfig::default()
    };
    let mut session = DiarizeSession::new(cfg)?;
    eprintln!(
        "backend={}",
        if session.using_wespeaker() {
            "WeSpeaker/RLX"
        } else {
            "mel-stat"
        }
    );
    let t0 = std::time::Instant::now();
    let turns = session.diarize(&pcm)?;
    eprintln!(
        "turns={} in {:.2}s",
        turns.len(),
        t0.elapsed().as_secs_f32()
    );
    for t in &turns {
        println!(
            "{{\"speaker_id\":{},\"start\":{:.3},\"end\":{:.3}}}",
            t.speaker_id, t.start, t.end
        );
    }
    Ok(())
}
