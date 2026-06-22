use crate::checkpoint::MoshiCheckpoint;
use crate::config::MoshiVariant;
use crate::device::parse_moshi_device;
use crate::download::{
    default_mimi_dir, default_moshi_dir_for, ensure_weights_checkpoint, fetch_moshi_checkpoint,
};
use crate::session::{GenerationConfig, MoshiSession};
use anyhow::{Context, Result};
use rlx_mimi::audio::write_wav_mono;
use rlx_runtime::Device;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fetch,
    OneWay,
    Duplex,
}

pub fn run(args: &[String]) -> Result<()> {
    let mut mode = Mode::OneWay;
    let mut moshi_dir: Option<PathBuf> = None;
    let mut mimi_dir: Option<PathBuf> = None;
    let mut prompt = String::from("Hello, I'm Moshi.");
    let mut in_wav: Option<PathBuf> = None;
    let mut out_wav: Option<PathBuf> = None;
    let mut max_steps = 25usize;
    let mut variant = MoshiVariant::MoshikoOneWay;
    let mut device = Device::Cpu;
    let mut checkpoint = MoshiCheckpoint::from_env_or_default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--fetch" => mode = Mode::Fetch,
            "--checkpoint" => {
                i += 1;
                let name = args.get(i).context("--checkpoint NAME")?;
                checkpoint = MoshiCheckpoint::parse(name).with_context(|| {
                    format!("unknown checkpoint {name} (bf16, q8, q4, q8-mlx, mlx-bf16)")
                })?;
            }
            "--variant" => {
                i += 1;
                let name = args.get(i).context("--variant NAME")?;
                variant = MoshiVariant::parse(name).with_context(|| {
                    format!("unknown variant {name} (moshiko-one-way, moshiko, moshika, …)")
                })?;
                if variant.is_duplex() {
                    mode = Mode::Duplex;
                }
            }
            "--duplex" => {
                mode = Mode::Duplex;
                variant = match variant {
                    MoshiVariant::Moshika | MoshiVariant::MoshikaOneWay => MoshiVariant::Moshika,
                    _ => MoshiVariant::Moshiko,
                };
            }
            "--model-dir" => {
                i += 1;
                moshi_dir = Some(PathBuf::from(args.get(i).context("--model-dir path")?));
            }
            "--mimi-dir" => {
                i += 1;
                mimi_dir = Some(PathBuf::from(args.get(i).context("--mimi-dir path")?));
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).context("--prompt text")?.clone();
            }
            "--in-wav" => {
                i += 1;
                in_wav = Some(PathBuf::from(args.get(i).context("--in-wav path")?));
                mode = Mode::Duplex;
                variant = match variant {
                    MoshiVariant::Moshika | MoshiVariant::MoshikaOneWay => MoshiVariant::Moshika,
                    _ => MoshiVariant::Moshiko,
                };
            }
            "--out-wav" => {
                i += 1;
                out_wav = Some(PathBuf::from(args.get(i).context("--out-wav path")?));
            }
            "--max-steps" => {
                i += 1;
                max_steps = args.get(i).context("--max-steps N")?.parse()?;
            }
            "--device" => {
                i += 1;
                device = parse_moshi_device(args.get(i).context("--device NAME")?)?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other} (try --help)"),
        }
        i += 1;
    }

    let moshi_dir = moshi_dir.unwrap_or_else(|| default_moshi_dir_for(variant, checkpoint));
    let mimi_dir = mimi_dir.unwrap_or_else(default_mimi_dir);

    if mode == Mode::Fetch {
        fetch_moshi_checkpoint(variant, checkpoint, &moshi_dir)?;
        rlx_mimi::fetch_mimi(&mimi_dir)?;
        return Ok(());
    }

    ensure_weights_checkpoint(&moshi_dir, variant, checkpoint)?;
    rlx_mimi::ensure_weights(&mimi_dir)?;

    let out = out_wav.unwrap_or_else(|| PathBuf::from("/tmp/moshi-out.wav"));
    let cfg = GenerationConfig {
        max_steps,
        ..GenerationConfig::default()
    };

    eprintln!("moshi: {}", moshi_dir.display());
    eprintln!("mimi:  {}", mimi_dir.display());
    eprintln!(
        "variant: {variant:?}, voice: {:?}, checkpoint: {checkpoint:?}, device: {device:?}, max_steps: {max_steps}",
        variant.voice()
    );

    let mut session =
        MoshiSession::open_with_checkpoint(&moshi_dir, &mimi_dir, variant, device, checkpoint)?;
    let result = match mode {
        Mode::Duplex => {
            let wav = in_wav.context("--in-wav required for duplex")?;
            session.generate_duplex(&wav, &cfg)?
        }
        Mode::OneWay | Mode::Fetch => session.generate_one_way(&prompt, &cfg)?,
    };

    write_wav_mono(&out, &result.samples, result.sample_rate)?;
    eprintln!(
        "wrote {} ({} samples, {} frames, transcript: {:?})",
        out.display(),
        result.samples.len(),
        result.audio_frames.len(),
        result.transcript
    );
    Ok(())
}

fn print_help() {
    eprintln!(
        "rlx-moshi — Kyutai Moshi speech-to-speech (native Rust)

Usage:
  rlx-moshi --prompt \"Hello.\" --out-wav /tmp/out.wav
  rlx-moshi --duplex --in-wav user.wav --out-wav /tmp/reply.wav
  rlx-moshi --variant moshika --prompt \"Hi.\" --out-wav /tmp/moshika.wav
  rlx-moshi --fetch

Options:
  --model-dir DIR   Moshi LM dir (default: .cache/<voice>-<checkpoint>)
  --mimi-dir DIR    Mimi codec dir (default: RLX_MIMI_DIR or .cache/mimi)
  --prompt TEXT     One-way text prompt
  --in-wav PATH     User audio for full-duplex
  --out-wav PATH    Output WAV (default: /tmp/moshi-out.wav)
  --max-steps N     Codec frames to generate (default: 25 ≈ 2 s)
  --device NAME     Inference device (cpu, metal, cuda, auto, …)
  --checkpoint NAME Weight preset: bf16, q8, q4, q8-mlx, mlx-bf16
  --variant NAME    Voice + mode: moshiko-one-way (default), moshiko, moshika-one-way, moshika
  --duplex          Full-duplex Moshiko (or Moshika if --variant moshika*)
  --fetch           Download checkpoint weights + mimi codec

Env:
  RLX_MOSHI_DIR         Override model dir (skips voice/checkpoint cache naming)
  RLX_MOSHI_CHECKPOINT  Default checkpoint preset (bf16, q8, q4, q8-mlx, mlx-bf16)
  RLX_MOSHI_VOICE       Default voice when variant unset (moshiko, moshika)
"
    );
    let _ = env::var("RLX_MOSHI_DIR");
}
