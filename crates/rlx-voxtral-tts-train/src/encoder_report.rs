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

//! Training bench report — timing, loss curves, epoch checkpoints for ablation.

use crate::asr_loss::whisper_mel_mse;
use crate::audio_metrics::mel_similarity;
use anyhow::{Context, Result};
use rlx_runtime::CompiledGraph;
use rlx_voxtral_tts::codec::encoder::load_mono_wav;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct EncoderTrainReport {
    pub started_at_unix: u64,
    pub device: String,
    pub backward_device: String,
    pub model_dir: String,
    pub wav_dir: String,
    pub out_dir: String,
    pub hyperparams: TrainHyperparams,
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub total_steps: usize,
    pub compile_ms: f64,
    pub train_ms: f64,
    pub steps_per_sec: f64,
    pub ms_per_step: f64,
    pub best_recon_l1: SerdeF64,
    pub best_step: usize,
    pub best_encoder: String,
    pub eval_wav: Option<String>,
    pub checkpoint_every_epoch: usize,
    pub epoch_checkpoints: Vec<String>,
    pub epochs_metrics: Vec<EpochMetrics>,
    pub early_stop_patience: usize,
    pub early_stop_min_delta: f64,
    pub early_stopped: bool,
    pub stopped_epoch: Option<usize>,
    pub stop_reason: Option<String>,
    pub epochs_completed: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainHyperparams {
    pub lr: f64,
    pub lr_min: f64,
    pub weight_decay: f32,
    pub mel_weight: f32,
    pub stft_weight: f32,
    pub commitment_delta: f32,
    pub diversity_weight: f32,
    pub max_audio_sec: f32,
    pub use_discriminator: bool,
    pub use_asr: bool,
    pub early_stop_patience: usize,
    pub early_stop_min_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub steps: usize,
    pub graph_loss_mean: SerdeF64,
    pub graph_loss_min: SerdeF64,
    pub recon_l1_mean: SerdeF64,
    pub recon_l1_min: SerdeF64,
    pub mel_mse_mean: SerdeF64,
    pub mel_sim_mean: SerdeF64,
    pub mel_sim_max: SerdeF32,
    pub grad_norm_mean: SerdeF64,
    pub grad_norm_max: SerdeF64,
    pub lr_end: SerdeF64,
    pub eval_recon_l1: Option<SerdeF64>,
    pub eval_mel_mse: Option<SerdeF32>,
    pub eval_mel_similarity: Option<SerdeF32>,
    pub checkpoint: Option<String>,
    pub epoch_wall_ms: f64,
    pub step_ms_mean: f64,
}

/// JSON-friendly float: NaN/Inf serialize as null.
#[derive(Debug, Clone, Copy)]
pub struct SerdeF64(pub f64);

#[derive(Debug, Clone, Copy)]
pub struct SerdeF32(pub f32);

impl Serialize for SerdeF64 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            serializer.serialize_f64(self.0)
        } else {
            serializer.serialize_none()
        }
    }
}

impl Serialize for SerdeF32 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            serializer.serialize_f32(self.0)
        } else {
            serializer.serialize_none()
        }
    }
}

impl From<f64> for SerdeF64 {
    fn from(v: f64) -> Self {
        Self(v)
    }
}

impl From<f32> for SerdeF32 {
    fn from(v: f32) -> Self {
        Self(v)
    }
}

pub struct EpochAccumulator {
    pub epoch: usize,
    pub steps: usize,
    graph_sum: f64,
    graph_min: f64,
    grad_sum: f64,
    grad_max: f32,
    step_ms_sum: f64,
    lr_end: f64,
    started: Instant,
}

impl EpochAccumulator {
    pub fn new(epoch: usize) -> Self {
        Self {
            epoch,
            steps: 0,
            graph_sum: 0.0,
            graph_min: f64::INFINITY,
            grad_sum: 0.0,
            grad_max: 0.0,
            step_ms_sum: 0.0,
            lr_end: 0.0,
            started: Instant::now(),
        }
    }

    pub fn record_step(&mut self, graph_loss: f64, grad_norm: f32, lr: f64, step_ms: f64) {
        self.steps += 1;
        if graph_loss.is_finite() {
            self.graph_sum += graph_loss;
            self.graph_min = self.graph_min.min(graph_loss);
        }
        if grad_norm.is_finite() {
            self.grad_sum += grad_norm as f64;
            self.grad_max = self.grad_max.max(grad_norm);
        }
        self.step_ms_sum += step_ms;
        self.lr_end = lr;
    }

    pub fn graph_mean(&self) -> f64 {
        self.graph_sum / self.steps.max(1) as f64
    }

    pub fn finish(
        self,
        eval_recon_l1: Option<f64>,
        eval_mel_mse: Option<f32>,
        eval_mel_similarity: Option<f32>,
        checkpoint: Option<PathBuf>,
    ) -> EpochMetrics {
        let n = self.steps.max(1) as f64;
        let eval_l1 = eval_recon_l1.unwrap_or(f64::NAN);
        let eval_mse = eval_mel_mse.unwrap_or(f32::NAN) as f64;
        let eval_sim = eval_mel_similarity.unwrap_or(f32::NAN) as f64;
        EpochMetrics {
            epoch: self.epoch,
            steps: self.steps,
            graph_loss_mean: SerdeF64(self.graph_sum / n),
            graph_loss_min: SerdeF64(finite_or_nan(self.graph_min)),
            recon_l1_mean: SerdeF64(eval_l1),
            recon_l1_min: SerdeF64(eval_l1),
            mel_mse_mean: SerdeF64(eval_mse),
            mel_sim_mean: SerdeF64(eval_sim),
            mel_sim_max: SerdeF32(eval_mel_similarity.unwrap_or(f32::NAN)),
            grad_norm_mean: SerdeF64(self.grad_sum / n),
            grad_norm_max: SerdeF64(self.grad_max as f64),
            lr_end: SerdeF64(self.lr_end),
            eval_recon_l1: eval_recon_l1.map(SerdeF64),
            eval_mel_mse: eval_mel_mse.map(SerdeF32),
            eval_mel_similarity: eval_mel_similarity.map(SerdeF32),
            checkpoint: checkpoint.map(|p| p.display().to_string()),
            epoch_wall_ms: self.started.elapsed().as_secs_f64() * 1000.0,
            step_ms_mean: self.step_ms_sum / n,
        }
    }
}

pub struct EvalMetrics {
    pub recon_l1: f64,
    pub mel_mse: f32,
    pub mel_similarity: f32,
}

pub fn prepare_pcm_for_layout(pcm: &[f32], patch_size: usize, n_patches: usize) -> Vec<f32> {
    let want_samples = n_patches * patch_size;
    let mut out = if pcm.len() >= want_samples {
        pcm[..want_samples].to_vec()
    } else {
        let mut v = pcm.to_vec();
        v.resize(want_samples, 0.0);
        v
    };
    let rem = out.len() % patch_size;
    if rem != 0 {
        out.extend(std::iter::repeat_n(0f32, patch_size - rem));
    }
    out
}

pub fn eval_reconstruction(
    recon: &mut CompiledGraph,
    eval_pcm: &[f32],
    patch_size: usize,
    layout_wav_t: usize,
    layout_n_patches: usize,
) -> Result<EvalMetrics> {
    let pcm = prepare_pcm_for_layout(eval_pcm, patch_size, layout_n_patches);
    let audio = crate::dataset::WavDataset::patches_to_ncl(&pcm, patch_size);
    debug_assert_eq!(audio.len(), patch_size * layout_n_patches);
    let target = pad_ncl_target(&pcm, patch_size, layout_wav_t);
    let outs = recon.run(&[("audio", audio.as_slice())]);
    let recon_ncl = outs.first().cloned().unwrap_or_else(|| target.clone());
    let recon_pcm = crate::dataset::WavDataset::ncl_to_pcm(&recon_ncl, patch_size);
    let ncl_len = recon_ncl.len().min(target.len());
    let recon_l1 = host_l1(&recon_ncl[..ncl_len], &target[..ncl_len]);
    let cmp_len = eval_pcm.len().min(recon_pcm.len());
    let mel_mse = whisper_mel_mse(&recon_pcm[..cmp_len], &eval_pcm[..cmp_len]);
    let mel_sim = mel_similarity(&eval_pcm[..cmp_len], &recon_pcm[..cmp_len]);
    Ok(EvalMetrics {
        recon_l1,
        mel_mse,
        mel_similarity: mel_sim,
    })
}

pub fn load_eval_pcm(path: &Path, sample_rate: u32, max_sec: f32) -> Result<Vec<f32>> {
    let mut pcm =
        load_mono_wav(path, sample_rate).with_context(|| format!("eval wav {}", path.display()))?;
    let max_samples = (max_sec * sample_rate as f32) as usize;
    if pcm.len() > max_samples {
        pcm.truncate(max_samples);
    }
    Ok(pcm)
}

pub fn write_report(path: &Path, report: &EncoderTrainReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("write report {}", path.display()))?;
    Ok(())
}

pub fn print_report_summary(report: &EncoderTrainReport) {
    eprintln!("\n[encoder-bench] ── summary ──");
    eprintln!(
        "  device={} backward={} compile={:.1}ms train={:.1}s ({:.2} steps/s, {:.1} ms/step)",
        report.device,
        report.backward_device,
        report.compile_ms,
        report.train_ms / 1000.0,
        report.steps_per_sec,
        report.ms_per_step
    );
    eprintln!(
        "  best_recon_l1={:.6} step={} checkpoints={} epochs={}/{}",
        opt_f64(report.best_recon_l1),
        report.best_step,
        report.epoch_checkpoints.len(),
        report.epochs_completed,
        report.epochs
    );
    if report.early_stopped {
        eprintln!(
            "  early_stop: epoch {} — {}",
            report.stopped_epoch.unwrap_or(0),
            report.stop_reason.as_deref().unwrap_or("?")
        );
    }
    eprintln!(
        "  lr={:.2e} wd={} mel_w={} stft_w={} max_audio={}s",
        report.hyperparams.lr,
        report.hyperparams.weight_decay,
        report.hyperparams.mel_weight,
        report.hyperparams.stft_weight,
        report.hyperparams.max_audio_sec
    );
    eprintln!("  epoch  graph_l1  recon_l1  recon_min  mel_sim  eval_l1  eval_mel");
    for e in &report.epochs_metrics {
        eprintln!(
            "  {:>5}  {:>8.4}  {:>8.4}  {:>9.4}  {:>7.4}  {:>7.4}  {:>7.4}",
            e.epoch,
            opt_f64(e.graph_loss_mean),
            opt_f64(e.recon_l1_mean),
            opt_f64(e.recon_l1_min),
            opt_f64(e.mel_sim_mean),
            e.eval_recon_l1.map(opt_f64).unwrap_or(f64::NAN),
            e.eval_mel_similarity
                .map(|v| v.0 as f64)
                .unwrap_or(f64::NAN)
        );
    }
}

fn opt_f64(v: SerdeF64) -> f64 {
    if v.0.is_finite() { v.0 } else { f64::NAN }
}

fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() { v } else { f64::NAN }
}

fn pad_ncl_target(pcm: &[f32], patch_size: usize, wav_t: usize) -> Vec<f32> {
    let ncl = crate::dataset::WavDataset::patches_to_ncl(pcm, patch_size);
    let mut out = vec![0f32; patch_size * wav_t];
    let copy = out.len().min(ncl.len());
    out[..copy].copy_from_slice(&ncl[..copy]);
    out
}

fn host_l1(recon: &[f32], target: &[f32]) -> f64 {
    let n = recon.len().min(target.len()).max(1);
    recon
        .iter()
        .zip(target.iter())
        .take(n)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() as f64
        / n as f64
}

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}
