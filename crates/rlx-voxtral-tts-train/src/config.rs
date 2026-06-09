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

//! Training hyperparameters and LOW_VRAM profile.

use rlx_voxtral_tts::config::CodecArgs;
use std::env;
use std::path::PathBuf;

pub const PREFIX_CODEC: &str = "audio_tokenizer.";

#[derive(Debug, Clone)]
pub struct TrainProfile {
    pub max_audio_sec: f32,
    pub batch_size: usize,
    pub grad_accum: usize,
    pub use_discriminator: bool,
    pub use_asr: bool,
    pub whisper_on_cpu: bool,
}

impl TrainProfile {
    pub fn from_env() -> Self {
        let low_vram = env_flag("LOW_VRAM");
        let production = env_flag("PRODUCTION");
        if low_vram {
            Self {
                max_audio_sec: env_f32("MAX_AUDIO_SEC", 4.0),
                batch_size: 1,
                grad_accum: env_usize("GRAD_ACCUM", 4),
                use_discriminator: env_flag_default("USE_DISCRIMINATOR", false),
                use_asr: env_flag_default("USE_ASR", true),
                whisper_on_cpu: true,
            }
        } else if production {
            Self {
                max_audio_sec: env_f32("MAX_AUDIO_SEC", 10.0),
                batch_size: env_usize("BATCH_SIZE", 1),
                grad_accum: env_usize("GRAD_ACCUM", 1),
                use_discriminator: env_flag_default("USE_DISCRIMINATOR", true),
                use_asr: env_flag_default("USE_ASR", true),
                whisper_on_cpu: env_flag_default("WHISPER_CPU", true),
            }
        } else {
            Self {
                max_audio_sec: env_f32("MAX_AUDIO_SEC", 10.0),
                batch_size: env_usize("BATCH_SIZE", 1),
                grad_accum: env_usize("GRAD_ACCUM", 1),
                use_discriminator: env_flag_default("USE_DISCRIMINATOR", true),
                use_asr: env_flag_default("USE_ASR", true),
                whisper_on_cpu: env_flag_default("WHISPER_CPU", false),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncoderTrainConfig {
    pub model_dir: PathBuf,
    pub wav_dir: PathBuf,
    pub manifest: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub lr: f64,
    pub lr_min: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub commitment_delta: f32,
    pub diversity_weight: f32,
    pub mel_weight: f32,
    pub stft_weight: f32,
    pub gan_weight: f32,
    pub asr_weight: f32,
    pub gan_warmup_steps: usize,
    pub rec_decay: f32,
    pub profile: TrainProfile,
    pub verbose: bool,
    pub resume_weights: Option<PathBuf>,
    pub resume_step: usize,
    /// CLI `--device` (`auto`, `cpu`, `metal`, …). Falls back to `RLX_DEVICE`.
    pub device: Option<String>,
    /// Save `encoder_epoch_N.safetensors` every N epochs (0 = off). Env: `CHECKPOINT_EVERY_EPOCH`.
    pub checkpoint_every_epoch: usize,
    /// JSON bench report path. Env: `TRAIN_REPORT`; default `{out_dir}/train_report.json`.
    pub report_path: Option<PathBuf>,
    /// Fixed WAV for per-epoch recon metrics. Env: `EVAL_WAV`.
    pub eval_wav: Option<PathBuf>,
    /// Stop after N epochs without eval (or graph) improvement. 0 = off. Env: `EARLY_STOP_PATIENCE`.
    pub early_stop_patience: usize,
    /// Minimum metric drop to count as improvement. Env: `EARLY_STOP_MIN_DELTA`.
    pub early_stop_min_delta: f64,
}

impl EncoderTrainConfig {
    pub fn from_cli(model_dir: PathBuf, wav_dir: PathBuf, out_dir: PathBuf) -> Self {
        Self {
            model_dir,
            wav_dir,
            manifest: None,
            out_dir,
            epochs: env_usize("EPOCHS", 100),
            steps_per_epoch: env_usize("STEPS_PER_EPOCH", 500),
            lr: env_f64("LR", 1e-4),
            lr_min: env_f64("LR_MIN", 1e-5),
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: env_f32(
                "WEIGHT_DECAY",
                if env_flag("LOW_VRAM") { 1e-4 } else { 1e-2 },
            ),
            grad_clip: 1.0,
            commitment_delta: 0.1,
            diversity_weight: 0.1,
            mel_weight: env_f32("MEL_WEIGHT", if env_flag("LOW_VRAM") { 0.0 } else { 45.0 }),
            stft_weight: env_f32("STFT_WEIGHT", if env_flag("LOW_VRAM") { 0.0 } else { 2.0 }),
            gan_weight: 1.0,
            asr_weight: 0.5,
            gan_warmup_steps: 2000,
            rec_decay: 0.9999,
            profile: TrainProfile::from_env(),
            verbose: true,
            resume_weights: env::var("RESUME_WEIGHTS").ok().map(PathBuf::from),
            resume_step: env_usize("RESUME_STEP", 0),
            device: env::var("RLX_DEVICE").ok(),
            checkpoint_every_epoch: env_usize("CHECKPOINT_EVERY_EPOCH", 0),
            report_path: env::var("TRAIN_REPORT").ok().map(PathBuf::from),
            eval_wav: env::var("EVAL_WAV").ok().map(PathBuf::from),
            early_stop_patience: env_usize("EARLY_STOP_PATIENCE", 0),
            early_stop_min_delta: env_f64("EARLY_STOP_MIN_DELTA", 1e-7),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoraTrainConfig {
    pub model_dir: PathBuf,
    pub encoder_weights: Option<PathBuf>,
    pub reference_wav_dir: PathBuf,
    pub manifest: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub rank: usize,
    pub alpha: f32,
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub max_seq_tokens: usize,
    pub lr: f64,
    pub grad_clip: f32,
    pub checkpoint_layers: bool,
    pub profile: TrainProfile,
    pub verbose: bool,
    pub resume_weights: Option<PathBuf>,
    pub resume_step: usize,
    /// CLI `--device` (`auto`, `cpu`, `metal`, …). Falls back to `RLX_DEVICE`.
    pub device: Option<String>,
}

impl LoraTrainConfig {
    pub fn from_cli(model_dir: PathBuf, reference_wav_dir: PathBuf, out_dir: PathBuf) -> Self {
        let production = env_flag("PRODUCTION");
        Self {
            model_dir,
            encoder_weights: None,
            reference_wav_dir,
            manifest: None,
            out_dir,
            rank: env_usize("LORA_RANK", if production { 16 } else { 8 }),
            alpha: env_f32("LORA_ALPHA", if production { 32.0 } else { 16.0 }),
            epochs: env_usize("EPOCHS", if production { 40 } else { 20 }),
            steps_per_epoch: env_usize("STEPS_PER_EPOCH", if production { 500 } else { 200 }),
            max_seq_tokens: env_usize("MAX_SEQ_TOKENS", 512),
            lr: env_f64("LR", 5e-5),
            grad_clip: 1.0,
            checkpoint_layers: env_flag_default("LORA_CHECKPOINT", true),
            profile: TrainProfile::from_env(),
            verbose: true,
            resume_weights: env::var("RESUME_WEIGHTS").ok().map(PathBuf::from),
            resume_step: env_usize("RESUME_STEP", 0),
            device: env::var("RLX_DEVICE").ok(),
        }
    }
}

/// Layer count for LoRA distillation. `LOW_VRAM=1` → 1 layer; else `LORA_N_LAYERS` or full stack.
pub fn lora_distill_layers(
    _cfg: &LoraTrainConfig,
    text: &rlx_voxtral_tts::config::TextConfig,
) -> usize {
    if env_flag("LOW_VRAM") {
        1
    } else if let Some(n) = env::var("LORA_N_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        n.min(text.num_hidden_layers).max(1)
    } else {
        text.num_hidden_layers
    }
}

/// Patch count after mono PCM is padded to whole patches.
pub fn patch_count(cfg: &CodecArgs, max_audio_sec: f32) -> usize {
    let samples = (cfg.sampling_rate as f32 * max_audio_sec).ceil() as usize;
    let patch = cfg.pretransform_patch_size;
    samples.div_ceil(patch)
}

/// Latent frame count after encoder conv strides (matches eager layout).
pub fn latent_frames(cfg: &CodecArgs, n_patches: usize) -> usize {
    use crate::codec_graph::conv1d_output_time;
    let input_k = cfg.patch_proj_kernel_size;
    let input_pad = input_k.saturating_sub(1);
    let mut t = conv1d_output_time(n_patches, input_k, 1, input_pad);
    let kernels = cfg.encoder_convs_kernels();
    let strides = cfg.encoder_convs_strides();
    let lens = cfg.encoder_transformer_lengths();
    for (stage, _) in lens.iter().enumerate() {
        let k = kernels[stage];
        let st = strides[stage];
        if k != 1 || st != 1 || stage + 1 == lens.len() {
            let pad_left = k.saturating_sub(st);
            t = conv1d_output_time(t, k, st, pad_left);
        }
    }
    t.max(1)
}

pub fn cosine_lr(step: usize, total: usize, lr_max: f64, lr_min: f64) -> f64 {
    if total == 0 {
        return lr_max;
    }
    let t = step as f64 / total as f64;
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
}

pub fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn env_flag_default(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

pub fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_frames_downsamples_with_strides() {
        let cfg = CodecArgs {
            channels: 1,
            sampling_rate: 24000,
            pretransform_patch_size: 240,
            patch_proj_kernel_size: 7,
            semantic_codebook_size: 8192,
            semantic_dim: 256,
            acoustic_codebook_size: 21,
            acoustic_dim: 36,
            dim: 1024,
            hidden_dim: 4096,
            head_dim: 128,
            n_heads: 8,
            n_kv_heads: 8,
            attn_sliding_window_size: 16,
            encoder_transformer_lengths_str: "2,2,2,2".into(),
            encoder_convs_kernels_str: "4,4,4,3".into(),
            encoder_convs_strides_str: "2,2,2,1".into(),
            decoder_transformer_lengths_str: "2,2,2,2".into(),
            decoder_convs_kernels_str: "3,4,4,4".into(),
            decoder_convs_strides_str: "1,2,2,2".into(),
        };
        let n = patch_count(&cfg, 4.0);
        assert_eq!(n, 400);
        assert_eq!(latent_frames(&cfg, n), 50);
    }
}
