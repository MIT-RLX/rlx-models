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

//! Phase 1 codec encoder training loop.

use anyhow::{Context, Result};
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::adam::AdamState;
use crate::asr_loss::AsrLoss;
use crate::checkpoint::{export_encoder_weights, load_encoder_weights};
use crate::codec_graph::{CodecGraphLayout, build_codec_recon_graph};
use crate::compile::compile_train_backward;
use crate::config::{EncoderTrainConfig, cosine_lr, patch_count};
use crate::dataset::WavDataset;
use crate::device::resolve_train_device;
use crate::discriminator::DiscriminatorBank;
use crate::early_stop::EarlyStopState;
use crate::encoder_loss::build_encoder_train_graph;
use crate::encoder_report::{
    EncoderTrainReport, EpochAccumulator, SerdeF64, TrainHyperparams as ReportHyperparams,
    eval_reconstruction, load_eval_pcm, ms, print_report_summary, write_report,
};
use crate::weights::{WeightStore, fit_params_to_graph, load_codec_weights};
use rlx_runtime::Session;

pub struct EncoderTrainResult {
    pub best_weights: WeightStore,
    pub best_recon_l1: f64,
    pub report: EncoderTrainReport,
}

pub fn train_encoder(cfg: &EncoderTrainConfig) -> Result<EncoderTrainResult> {
    let train_started = Instant::now();
    let device = resolve_train_device(cfg.device.as_deref())?;
    if cfg.verbose {
        eprintln!("[encoder] device={device:?}");
    }
    let model_cfg = VoxtralTtsConfig::from_model_dir(&cfg.model_dir)?;
    let codec = &model_cfg.audio_config.codec_args;
    fs::create_dir_all(&cfg.out_dir)?;

    let dataset = if let Some(m) = &cfg.manifest {
        WavDataset::from_manifest(m, codec, cfg.profile.max_audio_sec)?
    } else {
        WavDataset::from_dir(&cfg.wav_dir, codec, cfg.profile.max_audio_sec)?
    };

    let eval_wav = resolve_eval_wav(cfg)?;
    let eval_pcm = if let Some(ref path) = eval_wav {
        Some(load_eval_pcm(
            path,
            codec.sampling_rate as u32,
            cfg.profile.max_audio_sec,
        )?)
    } else {
        None
    };

    if cfg.verbose {
        if let Some(ref p) = eval_wav {
            eprintln!(
                "[encoder] eval clip {} ({} samples)",
                p.display(),
                eval_pcm.as_ref().map(|v| v.len()).unwrap_or(0)
            );
        }
        eprintln!(
            "[encoder] lr={:.2e} wd={} mel_w={} stft_w={} max_audio={}s",
            cfg.lr, cfg.weight_decay, cfg.mel_weight, cfg.stft_weight, cfg.profile.max_audio_sec
        );
    }

    let n_patches = patch_count(codec, cfg.profile.max_audio_sec);
    let layout = CodecGraphLayout::new(codec, n_patches);
    let train_meta = build_encoder_train_graph(
        codec,
        &layout,
        cfg.mel_weight,
        cfg.stft_weight,
        cfg.commitment_delta,
        cfg.diversity_weight,
        if cfg.profile.use_discriminator {
            cfg.gan_weight
        } else {
            0.0
        },
        if cfg.profile.use_asr {
            cfg.asr_weight
        } else {
            0.0
        },
    );

    let compile_started = Instant::now();
    let compiled = compile_train_backward(device, train_meta.backward.clone(), "encoder")?;
    let compile_ms = ms(compile_started.elapsed());
    let (backward_device, mut backward) = compiled;
    if cfg.verbose && backward_device != device {
        eprintln!("[encoder] active backward={backward_device:?}");
    }

    let trainable_names: Vec<String> = train_meta.params.iter().map(|s| s.name.clone()).collect();

    let (mut enc_weights, dec_weights) = load_codec_weights(&cfg.model_dir, true, codec)?;
    if let Some(path) = &cfg.resume_weights {
        let loaded = load_encoder_weights(path)?;
        enc_weights.merge(&loaded);
        if cfg.verbose {
            eprintln!(
                "[encoder] resumed weights from {} (step {})",
                path.display(),
                cfg.resume_step
            );
        }
    }
    fit_params_to_graph(&mut enc_weights, &train_meta.params)?;
    let mut weights = WeightStore::default();
    weights.merge(&enc_weights);
    weights.merge(&dec_weights);
    fit_params_to_graph(&mut weights, &train_meta.fwd.params)?;

    let mut adam = AdamState::new_for_names(&trainable_names, &enc_weights);
    let mut best_recon_l1 = f64::INFINITY;
    let mut best_step = 0usize;
    let mut best = enc_weights.clone();
    let total_steps = cfg.epochs * cfg.steps_per_epoch;
    let disc = if cfg.profile.use_discriminator {
        Some(DiscriminatorBank::new(8, layout.wav_t.max(n_patches)))
    } else {
        None
    };
    let mut disc_exec = disc.as_ref().map(|bank| bank.compile(device));
    let mut recon_forward = {
        let mut fwd = build_codec_recon_graph(codec, &layout)?;
        fwd.graph.set_outputs(vec![fwd.recon_wav]);
        Session::new(device).compile(fwd.graph)
    };
    let mut recon_metrics = {
        let mut fwd = build_codec_recon_graph(codec, &layout)?;
        fwd.graph.set_outputs(vec![fwd.recon_wav]);
        Session::new(rlx_runtime::Device::Cpu).compile(fwd.graph)
    };

    let proj_cols = layout.wav_t.max(n_patches);
    let mel_basis = banded_projection_basis(64, proj_cols);
    let stft_basis = banded_projection_basis(128, proj_cols);
    let asr = AsrLoss::from_env();

    for (name, data) in &weights.0 {
        backward.set_param(name, data);
        recon_forward.set_param(name, data);
        recon_metrics.set_param(name, data);
    }

    let sync_trainable = |backward: &mut rlx_runtime::CompiledGraph,
                          recon: &mut rlx_runtime::CompiledGraph,
                          metrics: &mut rlx_runtime::CompiledGraph,
                          enc: &WeightStore,
                          names: &[String]| {
        for name in names {
            if let Some(data) = enc.get(name) {
                backward.set_param(name, data);
                recon.set_param(name, data);
                metrics.set_param(name, data);
            }
        }
    };

    let sync_all_weights = |recon: &mut rlx_runtime::CompiledGraph,
                            metrics: &mut rlx_runtime::CompiledGraph,
                            store: &WeightStore| {
        for (name, data) in &store.0 {
            recon.set_param(name, data);
            metrics.set_param(name, data);
        }
    };

    let mut epoch_checkpoints: Vec<String> = Vec::new();
    let mut epochs_metrics = Vec::new();
    let report_path = cfg
        .report_path
        .clone()
        .unwrap_or_else(|| cfg.out_dir.join("train_report.json"));

    let hyperparams = ReportHyperparams {
        lr: cfg.lr,
        lr_min: cfg.lr_min,
        weight_decay: cfg.weight_decay,
        mel_weight: cfg.mel_weight,
        stft_weight: cfg.stft_weight,
        commitment_delta: cfg.commitment_delta,
        diversity_weight: cfg.diversity_weight,
        max_audio_sec: cfg.profile.max_audio_sec,
        use_discriminator: cfg.profile.use_discriminator,
        use_asr: cfg.profile.use_asr,
        early_stop_patience: cfg.early_stop_patience,
        early_stop_min_delta: cfg.early_stop_min_delta,
    };

    let mut early_stop = EarlyStopState::new(cfg.early_stop_patience, cfg.early_stop_min_delta);
    if cfg.verbose && early_stop.enabled() {
        eprintln!(
            "[encoder] early_stop patience={} min_delta={:.2e}",
            cfg.early_stop_patience, cfg.early_stop_min_delta
        );
    }

    for epoch in 0..cfg.epochs {
        let mut acc = EpochAccumulator::new(epoch);
        for step in 0..cfg.steps_per_epoch {
            let global = epoch * cfg.steps_per_epoch + step;
            if global < cfg.resume_step {
                continue;
            }
            let step_started = Instant::now();
            let lr = cosine_lr(global, total_steps, cfg.lr, cfg.lr_min);
            let batch = dataset.sample_batch()?;
            let audio = WavDataset::patches_to_ncl(&batch.pcm, codec.pretransform_patch_size);
            let target = pad_ncl_target(&batch.pcm, codec.pretransform_patch_size, layout.wav_t);

            let rec_w = cfg.rec_decay.powi(global as i32);

            let recon_out = if cfg.profile.use_discriminator {
                recon_forward.run(&[("audio", audio.as_slice())])
            } else {
                vec![target.clone()]
            };
            let recon_flat = recon_out.first().cloned().unwrap_or_else(|| target.clone());

            let gan_scale = if global >= cfg.gan_warmup_steps {
                1.0f32
            } else {
                global as f32 / cfg.gan_warmup_steps.max(1) as f32
            };
            let d_fake = if let Some(ref mut execs) = disc_exec {
                DiscriminatorBank::generator_hinge_loss(execs, &target, &recon_flat)
                    * rec_w
                    * gan_scale
            } else {
                0.0
            };
            let asr_mse = if cfg.profile.use_asr {
                asr.loss(&recon_flat, &target, batch.transcript.as_deref())
            } else {
                0.0
            };

            let d_fake_v = [d_fake];
            let asr_v = [asr_mse];
            let outs = backward.run(&[
                ("audio", audio.as_slice()),
                ("target_wav", target.as_slice()),
                ("mel_basis", mel_basis.as_slice()),
                ("stft_basis", stft_basis.as_slice()),
                ("d_fake", d_fake_v.as_slice()),
                ("asr_mse", asr_v.as_slice()),
                ("d_output", &[1.0f32]),
            ]);

            let graph_loss = outs.first().and_then(|v| v.first()).copied().unwrap_or(0.0) as f64;
            let mut grads = WeightStore::default();
            for (slot, gout) in train_meta.params.iter().zip(outs.iter().skip(1)) {
                grads.0.insert(slot.name.clone(), gout.clone());
            }
            let grad_norm = global_grad_norm(&grads);

            adam.step(
                &mut enc_weights,
                &grads,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.weight_decay,
                1e-8,
                cfg.grad_clip,
            );
            weights.merge(&enc_weights);
            sync_trainable(
                &mut backward,
                &mut recon_forward,
                &mut recon_metrics,
                &enc_weights,
                &trainable_names,
            );

            let checkpoint_every = std::env::var("CHECKPOINT_EVERY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if checkpoint_every > 0 && global > 0 && global.is_multiple_of(checkpoint_every) {
                let ckpt = cfg
                    .out_dir
                    .join(format!("encoder_step_{global}.safetensors"));
                export_encoder_weights(&enc_weights, &ckpt, codec).ok();
            }

            let step_ms = ms(step_started.elapsed());
            acc.record_step(graph_loss, grad_norm, lr, step_ms);

            if cfg.verbose && (global.is_multiple_of(10) || global + 1 == total_steps) {
                eprintln!(
                    "[encoder] step {global}/{total_steps} graph={graph_loss:.4} grad={grad_norm:.2e} lr={lr:.2e} ms={step_ms:.0}"
                );
            }
        }

        let (eval_l1, eval_mel_mse, eval_mel) = if let Some(ref pcm) = eval_pcm {
            sync_all_weights(&mut recon_forward, &mut recon_metrics, &weights);
            match eval_reconstruction(
                &mut recon_metrics,
                pcm,
                codec.pretransform_patch_size,
                layout.wav_t,
                layout.n_patches,
            ) {
                Ok(m) => {
                    if m.recon_l1.is_finite() && m.recon_l1 < best_recon_l1 {
                        best_recon_l1 = m.recon_l1;
                        best_step = (epoch + 1) * cfg.steps_per_epoch - 1;
                        best = enc_weights.clone();
                    }
                    if cfg.verbose {
                        eprintln!(
                            "[encoder] epoch {} eval l1={:.6} mel_mse={:.6} mel_sim={:.4}",
                            epoch + 1,
                            m.recon_l1,
                            m.mel_mse,
                            m.mel_similarity
                        );
                    }
                    (Some(m.recon_l1), Some(m.mel_mse), Some(m.mel_similarity))
                }
                Err(e) => {
                    eprintln!("[encoder] epoch {} eval failed: {e:#}", epoch + 1);
                    (None, None, None)
                }
            }
        } else {
            (None, None, None)
        };

        let epoch_ckpt = if cfg.checkpoint_every_epoch > 0
            && (epoch + 1).is_multiple_of(cfg.checkpoint_every_epoch)
        {
            let path = cfg
                .out_dir
                .join(format!("encoder_epoch_{epoch:04}.safetensors"));
            export_encoder_weights(&enc_weights, &path, codec)
                .with_context(|| format!("epoch checkpoint epoch={epoch}"))?;
            epoch_checkpoints.push(path.display().to_string());
            Some(path)
        } else {
            None
        };

        if cfg.verbose {
            eprintln!(
                "[encoder] epoch {}/{} graph_mean={:.4} eval_l1={:.6} mel_sim={:.4}",
                epoch + 1,
                cfg.epochs,
                acc.graph_mean(),
                eval_l1.unwrap_or(f64::NAN),
                eval_mel.unwrap_or(f32::NAN) as f64
            );
        }

        let graph_mean = acc.graph_mean();
        epochs_metrics.push(acc.finish(eval_l1, eval_mel_mse, eval_mel, epoch_ckpt));

        let stop_metric = eval_l1.or_else(|| {
            if eval_pcm.is_some() {
                None
            } else {
                Some(graph_mean)
            }
        });
        if early_stop.observe(epoch + 1, stop_metric) && cfg.verbose {
            eprintln!(
                "[encoder] early stop at epoch {} — {}",
                epoch + 1,
                early_stop.stop_reason.as_deref().unwrap_or("?")
            );
        }

        let epochs_completed = epochs_metrics.len();
        let completed_steps = epochs_completed * cfg.steps_per_epoch;

        let partial = build_report(
            cfg,
            &hyperparams,
            device,
            backward_device,
            compile_ms,
            ms(train_started.elapsed()),
            completed_steps,
            best_recon_l1,
            best_step,
            &eval_wav,
            &epoch_checkpoints,
            &epochs_metrics,
            &early_stop,
        );
        write_report(&report_path, &partial).ok();

        if early_stop.stopped {
            break;
        }
    }

    let epochs_completed = epochs_metrics.len();
    let completed_steps = epochs_completed * cfg.steps_per_epoch;

    let best_path = cfg.out_dir.join("best_encoder.safetensors");
    export_encoder_weights(&best, &best_path, codec)?;

    let train_ms = ms(train_started.elapsed());

    let report = build_report(
        cfg,
        &hyperparams,
        device,
        backward_device,
        compile_ms,
        train_ms,
        completed_steps,
        best_recon_l1,
        best_step,
        &eval_wav,
        &epoch_checkpoints,
        &epochs_metrics,
        &early_stop,
    );

    write_report(&report_path, &report)?;
    print_report_summary(&report);
    eprintln!("[encoder-bench] wrote {}", report_path.display());

    Ok(EncoderTrainResult {
        best_weights: best,
        best_recon_l1,
        report,
    })
}

fn build_report(
    cfg: &EncoderTrainConfig,
    hyperparams: &ReportHyperparams,
    device: rlx_runtime::Device,
    backward_device: rlx_runtime::Device,
    compile_ms: f64,
    train_ms: f64,
    completed_steps: usize,
    best_recon_l1: f64,
    best_step: usize,
    eval_wav: &Option<PathBuf>,
    epoch_checkpoints: &[String],
    epochs_metrics: &[crate::encoder_report::EpochMetrics],
    early_stop: &EarlyStopState,
) -> EncoderTrainReport {
    EncoderTrainReport {
        started_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        device: format!("{device:?}"),
        backward_device: format!("{backward_device:?}"),
        model_dir: cfg.model_dir.display().to_string(),
        wav_dir: cfg.wav_dir.display().to_string(),
        out_dir: cfg.out_dir.display().to_string(),
        hyperparams: hyperparams.clone(),
        epochs: cfg.epochs,
        steps_per_epoch: cfg.steps_per_epoch,
        total_steps: cfg.epochs * cfg.steps_per_epoch,
        compile_ms,
        train_ms,
        steps_per_sec: if train_ms > 0.0 && completed_steps > 0 {
            completed_steps as f64 / (train_ms / 1000.0)
        } else {
            0.0
        },
        ms_per_step: if completed_steps > 0 {
            train_ms / completed_steps as f64
        } else {
            0.0
        },
        best_recon_l1: SerdeF64(best_recon_l1),
        best_step,
        best_encoder: cfg
            .out_dir
            .join("best_encoder.safetensors")
            .display()
            .to_string(),
        eval_wav: eval_wav.as_ref().map(|p| p.display().to_string()),
        checkpoint_every_epoch: cfg.checkpoint_every_epoch,
        epoch_checkpoints: epoch_checkpoints.to_vec(),
        epochs_metrics: epochs_metrics.to_vec(),
        early_stop_patience: cfg.early_stop_patience,
        early_stop_min_delta: cfg.early_stop_min_delta,
        early_stopped: early_stop.stopped,
        stopped_epoch: early_stop.stopped_epoch,
        stop_reason: early_stop.stop_reason.clone(),
        epochs_completed: epochs_metrics.len(),
    }
}

fn resolve_eval_wav(cfg: &EncoderTrainConfig) -> Result<Option<PathBuf>> {
    if let Some(path) = &cfg.eval_wav {
        if path.is_file() {
            return Ok(Some(path.clone()));
        }
        anyhow::bail!("eval wav not found: {}", path.display());
    }
    if cfg.wav_dir.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(&cfg.wav_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "wav"))
            .collect();
        entries.sort();
        if let Some(first) = entries.into_iter().next() {
            return Ok(Some(first));
        }
    }
    Ok(None)
}

fn pad_ncl_target(pcm: &[f32], patch_size: usize, wav_t: usize) -> Vec<f32> {
    let ncl = WavDataset::patches_to_ncl(pcm, patch_size);
    let mut out = vec![0f32; patch_size * wav_t];
    let copy = out.len().min(ncl.len());
    out[..copy].copy_from_slice(&ncl[..copy]);
    out
}

fn global_grad_norm(grads: &WeightStore) -> f32 {
    let mut norm_sq = 0.0f32;
    for g in grads.0.values() {
        for gi in g {
            if gi.is_finite() {
                norm_sq += gi * gi;
            }
        }
    }
    norm_sq.sqrt()
}

/// Deterministic band-averaging projection (replaces random mel/STFT bases).
fn banded_projection_basis(n_rows: usize, n_cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_rows * n_cols];
    let band = (n_cols / n_rows.max(1)).max(1);
    for r in 0..n_rows {
        let start = r * band;
        let end = (start + band).min(n_cols);
        if start >= end {
            continue;
        }
        let scale = 1.0 / (end - start) as f32;
        for c in start..end {
            out[r * n_cols + c] = scale;
        }
    }
    out
}
