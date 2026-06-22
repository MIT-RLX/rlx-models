//! Manual CLI parser for `rlx-kyutai-tts` — mirrors the layout of `rlx-moshi`.

use crate::checkpoint::{KyutaiTtsCheckpoint, KyutaiTtsVoice};
use crate::config::KyutaiTtsConfig;
use crate::device::parse_kyutai_tts_device;
use crate::download::{
    default_kyutai_tts_dir, default_mimi_dir, ensure_weights_checkpoint,
    fetch_kyutai_tts_checkpoint,
};
use crate::session::{GenerationConfig, KyutaiTtsSession};
use anyhow::{Context, Result};
use rlx_mimi::audio::write_wav_mono;
use rlx_runtime::Device;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fetch,
    Info,
    Synthesize,
}

pub fn run(args: &[String]) -> Result<()> {
    let mut mode = Mode::Synthesize;
    let mut model_dir: Option<PathBuf> = None;
    let mut mimi_dir: Option<PathBuf> = None;
    let mut prompt = String::from("Hello from Kyutai TTS.");
    let mut out_wav: Option<PathBuf> = None;
    let mut max_steps = 100usize;
    let mut device = Device::Cpu;
    let mut checkpoint = KyutaiTtsCheckpoint::from_env_or_default();
    let mut voice = KyutaiTtsVoice::unconditional();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--fetch" => mode = Mode::Fetch,
            "--info" => mode = Mode::Info,
            "--checkpoint" => {
                i += 1;
                let name = args.get(i).context("--checkpoint NAME")?;
                checkpoint = KyutaiTtsCheckpoint::parse(name)
                    .with_context(|| format!("unknown checkpoint {name} (1.6b-en_fr)"))?;
            }
            "--voice" => {
                i += 1;
                let name = args.get(i).context("--voice NAME")?;
                voice = KyutaiTtsVoice::new(name.clone());
            }
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(args.get(i).context("--model-dir path")?));
            }
            "--mimi-dir" => {
                i += 1;
                mimi_dir = Some(PathBuf::from(args.get(i).context("--mimi-dir path")?));
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).context("--prompt text")?.clone();
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
                device = parse_kyutai_tts_device(args.get(i).context("--device NAME")?)?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other} (try --help)"),
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(default_kyutai_tts_dir);
    let mimi_dir = mimi_dir.unwrap_or_else(default_mimi_dir);

    eprintln!("kyutai-tts: {}", model_dir.display());
    eprintln!("mimi:       {}", mimi_dir.display());
    eprintln!(
        "checkpoint: {checkpoint:?}, device: {device:?}, max_steps: {max_steps}, voice: {:?}",
        if voice.is_unconditional() {
            "<unconditional>"
        } else {
            voice.name.as_str()
        }
    );

    match mode {
        Mode::Fetch => {
            fetch_kyutai_tts_checkpoint(checkpoint, &model_dir)?;
            rlx_mimi::fetch_mimi(&mimi_dir)?;
        }
        Mode::Info => {
            ensure_weights_checkpoint(&model_dir, checkpoint)?;
            let cfg = KyutaiTtsConfig::v1_6b_en_fr();
            print_info(&cfg);
        }
        Mode::Synthesize => {
            ensure_weights_checkpoint(&model_dir, checkpoint)?;
            rlx_mimi::ensure_weights(&mimi_dir)?;

            let mut session =
                KyutaiTtsSession::open_with_checkpoint(&model_dir, &mimi_dir, device, checkpoint)?;
            session.set_voice(voice);
            let gen_cfg = GenerationConfig {
                max_steps,
                ..GenerationConfig::default()
            };
            let result = session.generate(&prompt, &gen_cfg)?;
            let out = out_wav.unwrap_or_else(|| PathBuf::from("/tmp/kyutai-tts-out.wav"));
            write_wav_mono(&out, &result.samples, result.sample_rate)?;
            eprintln!(
                "wrote {} ({} samples, {} frames)",
                out.display(),
                result.samples.len(),
                result.audio_frames.len()
            );
        }
    }

    Ok(())
}

fn print_info(cfg: &KyutaiTtsConfig) {
    eprintln!("Kyutai TTS — architecture preset:");
    eprintln!(
        "  backbone:  {} layers × {} heads, d_model={}, context={}",
        cfg.num_layers, cfg.num_heads, cfg.dim, cfg.context
    );
    eprintln!(
        "  hidden_scale: {}, norm: {}, gating: {}",
        cfg.hidden_scale, cfg.norm, cfg.gating
    );
    eprintln!(
        "  depformer: {} layers × {} heads, d_model={}, ff={}, low_rank={}, weights_per_step={}",
        cfg.depformer.num_layers,
        cfg.depformer.num_heads,
        cfg.depformer.dim,
        cfg.depformer.dim_feedforward,
        cfg.depformer.low_rank_embeddings,
        cfg.depformer.weights_per_step,
    );
    eprintln!(
        "  codebooks: n_q={}, dep_q={}, card={}, text_card={}",
        cfg.n_q, cfg.dep_q, cfg.card, cfg.text_card
    );
    eprintln!(
        "  streams:   demux_second_stream={}, cross_attention={}, audio_delay={} s ({} frames)",
        cfg.demux_second_stream,
        cfg.cross_attention,
        cfg.tts_config.audio_delay,
        cfg.audio_delay_frames(),
    );
    let conds: Vec<String> = cfg.conditioners.keys().cloned().collect();
    eprintln!("  conditioners: {}", conds.join(", "));
    eprintln!(
        "  fuser:     sum={:?}, cross={:?}",
        cfg.fuser.sum, cfg.fuser.cross
    );
}

fn print_help() {
    eprintln!(
        "rlx-kyutai-tts — Kyutai TTS (1.6B en/fr) inference scaffold

Usage:
  rlx-kyutai-tts --fetch
  rlx-kyutai-tts --info
  rlx-kyutai-tts --prompt \"Hello.\" --out-wav /tmp/out.wav --device metal

Options:
  --model-dir DIR   Kyutai TTS dir (default: .cache/kyutai-tts-1.6b-en_fr)
  --mimi-dir DIR    Mimi codec dir (default: RLX_MIMI_DIR or .cache/mimi)
  --prompt TEXT     Text to synthesise
  --out-wav PATH    Output WAV (default: /tmp/kyutai-tts-out.wav)
  --max-steps N     Frames to generate (default: 100 ≈ 8 s @ 12.5 Hz)
  --voice NAME      Voice embedding from kyutai/tts-voices (default: unconditional)
  --device NAME     Inference device (cpu, metal, cuda, auto, …)
  --checkpoint NAME Weight preset: 1.6b-en_fr (default)
  --fetch           Download checkpoint + Mimi codec
  --info            Print resolved architecture config

Env:
  RLX_KYUTAI_TTS_DIR         Override model dir (skips default cache naming)
  RLX_KYUTAI_TTS_CHECKPOINT  Default checkpoint preset (1.6b-en_fr)
"
    );
    let _ = env::var("RLX_KYUTAI_TTS_DIR");
}
