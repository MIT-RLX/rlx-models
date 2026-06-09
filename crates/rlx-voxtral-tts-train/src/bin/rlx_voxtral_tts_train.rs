// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Voxtral voice-clone training CLI.

use anyhow::{Context, Result, bail};
use rlx_core::STANDARD_DEVICE_NAMES;
use rlx_voxtral_tts_train::{
    EncoderTrainConfig, LoraTrainConfig, default_train_all, inject_weights, train_all,
    train_encoder, train_lora,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_help();
        bail!("missing subcommand");
    }
    match args[1].as_str() {
        "encoder" => run_encoder(&args[2..]),
        "lora" => run_lora(&args[2..]),
        "all" => run_all(&args[2..]),
        "manifest" => run_manifest(&args[2..]),
        "inject" => run_inject(&args[2..]),
        "--help" | "-h" | "help" => {
            print_help();
            Ok(())
        }
        other => bail!("unknown subcommand {other}"),
    }
}

fn run_encoder(args: &[String]) -> Result<()> {
    let model_dir = parse_path(args, "--model-dir").context("--model-dir")?;
    let wav_dir = parse_path(args, "--wav-dir").context("--wav-dir")?;
    let out_dir = parse_path(args, "--out-dir")
        .unwrap_or_else(|| PathBuf::from(".cache/voxtral/train/encoder"));
    let mut cfg = EncoderTrainConfig::from_cli(model_dir, wav_dir, out_dir);
    cfg.manifest = parse_path(args, "--manifest");
    cfg.resume_weights = parse_path(args, "--resume-weights");
    cfg.resume_step = parse_usize(args, "--resume-step").unwrap_or(cfg.resume_step);
    cfg.device = parse_string(args, "--device").or(cfg.device);
    if let Some(epochs) = parse_usize(args, "--epochs") {
        cfg.epochs = epochs;
    }
    if let Some(steps) = parse_usize(args, "--steps-per-epoch") {
        cfg.steps_per_epoch = steps;
    }
    if let Some(n) = parse_usize(args, "--checkpoint-every-epoch") {
        cfg.checkpoint_every_epoch = n;
    }
    cfg.report_path = parse_path(args, "--report").or(cfg.report_path);
    cfg.eval_wav = parse_path(args, "--eval-wav").or(cfg.eval_wav);
    if let Some(n) = parse_usize(args, "--early-stop-patience") {
        cfg.early_stop_patience = n;
    }
    if let Some(d) = parse_f64(args, "--early-stop-min-delta") {
        cfg.early_stop_min_delta = d;
    }
    let result = train_encoder(&cfg)?;
    eprintln!(
        "encoder done — best_recon_l1={:.6} best_step={} epochs={}/{} early_stop={} steps/s={:.2} ms/step={:.1} report={}",
        result.best_recon_l1,
        result.report.best_step,
        result.report.epochs_completed,
        result.report.epochs,
        result.report.early_stopped,
        result.report.steps_per_sec,
        result.report.ms_per_step,
        cfg.report_path
            .unwrap_or_else(|| cfg.out_dir.join("train_report.json"))
            .display()
    );
    Ok(())
}

fn run_lora(args: &[String]) -> Result<()> {
    let model_dir = parse_path(args, "--model-dir").context("--model-dir")?;
    let reference = parse_path(args, "--reference-wav-dir").context("--reference-wav-dir")?;
    let out_dir =
        parse_path(args, "--out-dir").unwrap_or_else(|| PathBuf::from(".cache/voxtral/train/lora"));
    let mut cfg = LoraTrainConfig::from_cli(model_dir, reference, out_dir);
    cfg.encoder_weights = parse_path(args, "--encoder-weights");
    cfg.manifest = parse_path(args, "--manifest");
    cfg.resume_weights = parse_path(args, "--resume-weights");
    cfg.resume_step = parse_usize(args, "--resume-step").unwrap_or(cfg.resume_step);
    cfg.device = parse_string(args, "--device").or(cfg.device);
    if let Some(epochs) = parse_usize(args, "--epochs") {
        cfg.epochs = epochs;
    }
    let result = train_lora(&cfg)?;
    eprintln!(
        "lora done — best_loss={:.6}, adapters in {}",
        result.best_loss,
        cfg.out_dir.join("lora_adapters.safetensors").display()
    );
    Ok(())
}

fn run_all(args: &[String]) -> Result<()> {
    let model_dir = parse_path(args, "--model-dir").context("--model-dir")?;
    let wav_dir = parse_path(args, "--wav-dir").context("--wav-dir")?;
    let out_root =
        parse_path(args, "--out-dir").unwrap_or_else(|| PathBuf::from(".cache/voxtral/train"));
    let manifest = parse_path(args, "--manifest");
    let mut cfg = default_train_all(&model_dir, &wav_dir, &out_root, manifest.clone());
    cfg.encoder.manifest = manifest.clone();
    cfg.lora.manifest = manifest;
    let resume = parse_path(args, "--resume-weights");
    let resume_step = parse_usize(args, "--resume-step");
    if let Some(path) = resume.clone() {
        cfg.encoder.resume_weights = Some(path.clone());
        cfg.lora.resume_weights = Some(path);
    }
    if let Some(step) = resume_step {
        cfg.encoder.resume_step = step;
        cfg.lora.resume_step = step;
    }
    cfg.encoder.device = parse_string(args, "--device").or(cfg.encoder.device);
    cfg.lora.device = cfg.encoder.device.clone();
    if parse_flag(args, "--no-inject") {
        cfg.inject = false;
    }
    let result = train_all(&cfg)?;
    eprintln!(
        "train-all done — encoder_loss={:.6} lora_loss={:.6}",
        result.encoder_loss, result.lora_loss
    );
    if let Some(path) = result.consolidated {
        eprintln!("merged checkpoint: {}", path.display());
    }
    Ok(())
}

fn run_manifest(args: &[String]) -> Result<()> {
    let wav_dir = parse_path(args, "--wav-dir").context("--wav-dir")?;
    let out = parse_path(args, "--out").unwrap_or_else(|| PathBuf::from("manifest.json"));
    let sample_rate = parse_usize(args, "--sample-rate").unwrap_or(24_000) as u32;
    rlx_voxtral_tts_train::dataset::build_manifest_from_dir(&wav_dir, &out, sample_rate)?;
    eprintln!("wrote manifest {}", out.display());
    Ok(())
}

fn run_inject(args: &[String]) -> Result<()> {
    let model_dir = parse_path(args, "--model-dir").context("--model-dir")?;
    let encoder = parse_path(args, "--encoder-weights");
    let lora = parse_path(args, "--lora-weights");
    let out = inject_weights(&model_dir, encoder.as_deref(), lora.as_deref())?;
    eprintln!("merged weights written to {}", out.display());
    Ok(())
}

fn parse_path(args: &[String], name: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}

fn parse_string(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_usize(args: &[String], name: &str) -> Option<usize> {
    parse_string(args, name).and_then(|s| s.parse().ok())
}

fn parse_f64(args: &[String], name: &str) -> Option<f64> {
    parse_string(args, name).and_then(|s| s.parse().ok())
}

fn parse_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn print_help() {
    eprintln!(
        "rlx-voxtral-tts-train — native RLX voice clone training

Subcommands:
  encoder  Train codec encoder (Phase 1)
  lora     Train LoRA adapters on LM (Phase 2)
  all      Phase 1 + Phase 2 + inject merged weights
  manifest Build JSON manifest from a WAV directory
  inject   Merge encoder and/or LoRA weights into consolidated.safetensors

Options (encoder / lora / all):
  --manifest PATH       JSON manifest with optional per-file transcript
  --resume-weights PATH Continue from encoder_step_*.safetensors or lora_step_*.safetensors
  --resume-step N       Global step offset for LR schedule (default 0; env RESUME_STEP)
  --device DEVICE       auto (default) | {STANDARD_DEVICE_NAMES}
                        auto picks the first available GPU backend, else CPU.
                        Override with env RLX_DEVICE.

Env (training):
  RLX_DEVICE                          same as --device when flag omitted
  RESUME_WEIGHTS / RESUME_STEP        resume checkpoint path and step offset
  CHECKPOINT_EVERY=N                  write encoder_step_N safetensors periodically
  CHECKPOINT_EVERY_EPOCH=N            write encoder_epoch_NNNN safetensors every N epochs (ablation)
  TRAIN_REPORT=PATH                   JSON bench report (default: out_dir/train_report.json)
  EVAL_WAV=PATH                       fixed clip for per-epoch recon metrics
  EARLY_STOP_PATIENCE=N               stop after N epochs without eval improvement (0=off)
  EARLY_STOP_MIN_DELTA=F              min eval L1 drop to reset patience (default 1e-7)
  RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU  1|true — force CPU backward (hybrid forward GPU)
  RLX_COMPILE_OUTPUT_CAP              max backward outputs before CPU fallback (default 1024; MLX enforces)
  RLX_VOXTRAL_TTS_TRAIN_NATIVE_BACKWARD  1|true — skip output-cap CPU fallback on GPU backends
  RLX_VOXTRAL_TTS_TRAIN_GPU_STEP      1 — run one GPU backward step in CI (tests)
  LOW_VRAM                            shorter clips, no GAN, smaller graph
  PRODUCTION=1                        production defaults (rank 16, longer LoRA)
  USE_WHISPER_ASR=1                   Whisper CER in encoder ASR loss (needs WHISPER_MODEL_DIR)
  WHISPER_MODEL_DIR                   openai/whisper-tiny layout (model.safetensors + config.json)
  PRECOMPUTE_DISTILL=1                  precompute teacher batches once (large upfront, fast steps)
  LORA_TIMING=1                         log per-step wall time
  LORA_N_LAYERS=N                       train first N LM layers (default: full stack)
  LORA_METAL_BACKWARD=1                 try GPU backward on full graph (may OOM / diverge)
  GRAD_ACCUM=N                          micro-batch gradient accumulation (default from profile)
  DISTILL_TEXT / DISTILL_VOICE        LoRA teacher prompt defaults

Examples:
  cargo run -p rlx-voxtral-tts-train --features metal -- encoder \\
    --model-dir .cache/voxtral/Voxtral-4B-TTS-2603 \\
    --wav-dir ./wavs --out-dir ./out/encoder --device metal

  just features=all-backends voxtral-tts-train-encoder -- \\
    --model-dir $RLX_VOXTRAL_TTS_DIR --wav-dir ./wavs --device auto

  LOW_VRAM=1 cargo run -p rlx-voxtral-tts-train -- encoder ...

  cargo run -p rlx-voxtral-tts-train -- inject \\
    --model-dir .cache/voxtral/Voxtral-4B-TTS-2603 \\
    --encoder-weights ./out/encoder/best_encoder.safetensors
"
    );
}
