//! CLI for Sesame CSM-1B.

use anyhow::Result;
use clap::Parser;
use rlx_cli::parse_device;
use rlx_runtime::Device;

use crate::generate::GenerateOpts;
use crate::session::{SesameSession, load_wav_mono_24k, write_wav};
use crate::tokenize::{default_mimi_dir, default_model_dir};

#[derive(Parser, Debug)]
#[command(
    name = "rlx-sesame",
    about = "Sesame CSM-1B TTS (Llama backbone + Mimi)"
)]
pub struct Args {
    /// Model directory (`config.json`, `model.safetensors`, tokenizer).
    #[arg(long, default_value_os_t = default_model_dir())]
    pub model_dir: std::path::PathBuf,

    /// Mimi codec directory.
    #[arg(long, default_value_os_t = default_mimi_dir())]
    pub mimi_dir: std::path::PathBuf,

    /// Text to synthesize.
    #[arg(long)]
    pub text: String,

    /// Output WAV path.
    #[arg(long, default_value = "/tmp/sesame.wav")]
    pub output: std::path::PathBuf,

    /// Device for Mimi (LM is eager CPU in this arc).
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Speaker id (default 0).
    #[arg(long, default_value_t = 0)]
    pub speaker: u32,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.9)]
    pub temperature: f32,

    /// Top-k sampling.
    #[arg(long, default_value_t = 50)]
    pub topk: usize,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Greedy (argmax) instead of top-k.
    #[arg(long, default_value_t = false)]
    pub greedy: bool,

    /// Max audio frames (~80 ms each @ 12.5 Hz).
    #[arg(long, default_value_t = 250)]
    pub max_frames: usize,

    /// Optional context WAV (24 kHz preferred) for continuity.
    #[arg(long)]
    pub context: Option<std::path::PathBuf>,
}

pub fn run(args: Args) -> Result<()> {
    let device = resolve_device(&args.device)?;
    eprintln!(
        "sesame: loading model={} mimi={} device={device:?}",
        args.model_dir.display(),
        args.mimi_dir.display()
    );
    let mut session = SesameSession::open_on(&args.model_dir, &args.mimi_dir, device)?;
    let opts = GenerateOpts {
        speaker: args.speaker,
        max_audio_frames: args.max_frames,
        temperature: args.temperature,
        topk: args.topk,
        seed: args.seed,
        greedy: args.greedy,
    };
    let context_pcm = match &args.context {
        Some(p) => Some(load_wav_mono_24k(p)?),
        None => None,
    };
    let result = session.synthesize_with_context(&args.text, context_pcm.as_deref(), &opts)?;
    write_wav(&args.output, &result.samples, result.sample_rate)?;
    eprintln!(
        "sesame: wrote {} ({} frames, {:.2}s @ {} Hz)",
        args.output.display(),
        result.audio_frames.len(),
        result.samples.len() as f32 / result.sample_rate as f32,
        result.sample_rate
    );
    Ok(())
}

/// Resolve `auto` → preferred device for this host.
pub fn resolve_device(s: &str) -> Result<Device> {
    if s.eq_ignore_ascii_case("auto") {
        #[cfg(feature = "metal")]
        {
            return Ok(Device::Metal);
        }
        #[allow(unreachable_code)]
        Ok(Device::Cpu)
    } else {
        parse_device(s)
    }
}
