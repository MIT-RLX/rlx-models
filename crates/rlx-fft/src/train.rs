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

//! Training loop for butterfly twiddle factors.

use crate::butterfly::{
    butterfly_encdec_roundtrip_batch, butterfly_forward_real_batch,
    butterfly_inverse_complex_batch, butterfly_train_step_dir, butterfly_train_step_encdec,
};
use crate::config::{EncDecTrainConfig, FftLearnConfig, TrainConfig, TransformDir};
use crate::reference::{fft_real_batch, ifft_complex_batch, max_abs_error, mse, roundtrip_scale};
use crate::twiddle::exact_twiddles;
use crate::weights::{EncDecWeights, WeightStore, export_safetensors};
use anyhow::Result;
use rand::prelude::*;
use std::path::Path;
use std::time::Instant;

pub struct TrainResult {
    pub final_mse: f32,
    pub max_error: f32,
    pub weights: WeightStore,
    pub steps: usize,
    pub elapsed_ms: f64,
    pub direction: TransformDir,
}

pub struct EncDecTrainResult {
    pub reconstruction_mse: f32,
    pub spectrum_mse: f32,
    pub roundtrip_max_error: f32,
    pub weights: EncDecWeights,
    pub steps: usize,
    pub elapsed_ms: f64,
}

pub fn random_batch(rng: &mut impl Rng, batch: usize, n_fft: usize) -> Vec<f32> {
    (0..batch * n_fft)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

pub fn random_complex_batch(rng: &mut impl Rng, batch: usize, n_fft: usize) -> Vec<f32> {
    (0..batch * n_fft * 2)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

pub fn train_butterfly(cfg: &TrainConfig) -> Result<TrainResult> {
    train_butterfly_dir(cfg, cfg.direction)
}

pub fn eager_train_from_env() -> bool {
    std::env::var("RLX_FFT_EAGER_TRAIN")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
}

pub fn train_butterfly_dir(cfg: &TrainConfig, dir: TransformDir) -> Result<TrainResult> {
    if eager_train_from_env() {
        train_butterfly_dir_eager(cfg, dir)
    } else {
        crate::train_rlx::train_butterfly_rlx(cfg, dir)
    }
}

pub fn train_butterfly_eager(cfg: &TrainConfig, dir: TransformDir) -> Result<TrainResult> {
    train_butterfly_dir_eager(cfg, dir)
}

fn train_butterfly_dir_eager(cfg: &TrainConfig, dir: TransformDir) -> Result<TrainResult> {
    cfg.model.validate()?;
    let started = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut twiddles = exact_twiddles(&cfg.model);
    let n = cfg.model.n_fft;
    let batch = cfg.model.batch;

    let mut last_mse;
    for step in 0..cfg.steps {
        let input = if dir.is_forward() {
            random_batch(&mut rng, batch, n)
        } else {
            random_complex_batch(&mut rng, batch, n)
        };
        last_mse = butterfly_train_step_dir(&input, &mut twiddles, batch, n, cfg.lr as f32, dir)?;

        if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
            eprintln!("[train {dir:?}] step {} mse={last_mse:.6e}", step + 1);
        }
    }

    let (final_mse, max_err) = evaluate_twiddles(&twiddles, &cfg.model, dir, 8, &mut rng)?;

    let weights = WeightStore::from_twiddles(&twiddles, n);
    if let Some(dir_path) = &cfg.out_dir {
        std::fs::create_dir_all(dir_path)?;
        let fname = "twiddles.safetensors";
        export_safetensors(&dir_path.join(fname), &weights)?;
        let meta = serde_json::json!({
            "n_fft": n,
            "batch": batch,
            "steps": cfg.steps,
            "direction": format!("{dir:?}"),
            "final_mse": final_mse,
            "max_error": max_err,
        });
        std::fs::write(
            dir_path.join("train_report.json"),
            serde_json::to_vec_pretty(&meta)?,
        )?;
    }

    Ok(TrainResult {
        final_mse,
        max_error: max_err,
        weights,
        steps: cfg.steps,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        direction: dir,
    })
}

pub fn train_encdec(cfg: &EncDecTrainConfig) -> Result<EncDecTrainResult> {
    if eager_train_from_env() {
        train_encdec_eager(cfg)
    } else {
        crate::train_rlx::train_encdec_rlx(cfg)
    }
}

pub fn train_encdec_eager(cfg: &EncDecTrainConfig) -> Result<EncDecTrainResult> {
    train_encdec_eager_impl(cfg)
}

fn train_encdec_eager_impl(cfg: &EncDecTrainConfig) -> Result<EncDecTrainResult> {
    cfg.model.validate()?;
    let started = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut encoder_tw = exact_twiddles(&cfg.model);
    let mut decoder_tw = exact_twiddles(&cfg.model);
    let n = cfg.model.n_fft;
    let batch = cfg.model.batch;

    let mut last_recon;
    let mut last_spec;
    for step in 0..cfg.steps {
        let signal = random_batch(&mut rng, batch, n);
        let losses = butterfly_train_step_encdec(
            &signal,
            &mut encoder_tw,
            &mut decoder_tw,
            batch,
            n,
            cfg.lr as f32,
            cfg.spectrum_weight,
        )?;
        last_recon = losses.reconstruction;
        last_spec = losses.spectrum;

        if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
            eprintln!(
                "[train encdec] step {} recon_mse={last_recon:.6e} spectrum_mse={last_spec:.6e}",
                step + 1
            );
        }
    }

    let (recon_mse, spec_mse, max_err) =
        evaluate_encdec(&encoder_tw, &decoder_tw, &cfg.model, 8, &mut rng)?;

    let weights = EncDecWeights::from_twiddles(&encoder_tw, &decoder_tw, n);
    if let Some(dir_path) = &cfg.out_dir {
        std::fs::create_dir_all(dir_path)?;
        export_safetensors(&dir_path.join("encdec.safetensors"), &weights.merged())?;
        let meta = serde_json::json!({
            "n_fft": n,
            "batch": batch,
            "steps": cfg.steps,
            "spectrum_weight": cfg.spectrum_weight,
            "reconstruction_mse": recon_mse,
            "spectrum_mse": spec_mse,
            "roundtrip_max_error": max_err,
        });
        std::fs::write(
            dir_path.join("train_report.json"),
            serde_json::to_vec_pretty(&meta)?,
        )?;
    }

    Ok(EncDecTrainResult {
        reconstruction_mse: recon_mse,
        spectrum_mse: spec_mse,
        roundtrip_max_error: max_err,
        weights,
        steps: cfg.steps,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn evaluate_encdec(
    encoder_tw: &[f32],
    decoder_tw: &[f32],
    cfg: &FftLearnConfig,
    batches: usize,
    rng: &mut impl Rng,
) -> Result<(f32, f32, f32)> {
    let scale = roundtrip_scale(cfg.n_fft);
    let mut recon_acc = 0f32;
    let mut spec_acc = 0f32;
    let mut max_acc = 0f32;
    for _ in 0..batches {
        let signal = random_batch(rng, cfg.batch, cfg.n_fft);
        let spectrum = butterfly_forward_real_batch(&signal, encoder_tw, cfg.batch, cfg.n_fft)?;
        let target_spec = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        spec_acc += mse(&spectrum, &target_spec);

        let recovered = butterfly_encdec_roundtrip_batch(
            &signal, encoder_tw, decoder_tw, cfg.batch, cfg.n_fft,
        )?;
        let mut target = vec![0f32; cfg.batch * cfg.n_fft * 2];
        for b in 0..cfg.batch {
            for i in 0..cfg.n_fft {
                let base = b * cfg.n_fft * 2 + i * 2;
                target[base] = signal[b * cfg.n_fft + i] * scale;
            }
        }
        recon_acc += mse(&recovered, &target);
        max_acc = max_acc.max(max_abs_error(&recovered, &target));
    }
    Ok((
        recon_acc / batches as f32,
        spec_acc / batches as f32,
        max_acc,
    ))
}

pub fn evaluate_encdec_weights(
    weights: &EncDecWeights,
    cfg: &FftLearnConfig,
    batches: usize,
) -> Result<(f32, f32, f32)> {
    cfg.validate()?;
    let encoder = weights.encoder_twiddles(cfg.n_fft)?;
    let decoder = weights.decoder_twiddles(cfg.n_fft)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    evaluate_encdec(&encoder, &decoder, cfg, batches, &mut rng)
}

fn evaluate_twiddles(
    twiddles: &[f32],
    cfg: &FftLearnConfig,
    dir: TransformDir,
    batches: usize,
    rng: &mut impl Rng,
) -> Result<(f32, f32)> {
    let mut mse_acc = 0f32;
    let mut max_acc = 0f32;
    for _ in 0..batches {
        let (pred, target) = if dir.is_forward() {
            let signal = random_batch(rng, cfg.batch, cfg.n_fft);
            let pred = butterfly_forward_real_batch(&signal, twiddles, cfg.batch, cfg.n_fft)?;
            let target = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
            (pred, target)
        } else {
            let spectrum = random_complex_batch(rng, cfg.batch, cfg.n_fft);
            let pred = butterfly_inverse_complex_batch(&spectrum, twiddles, cfg.batch, cfg.n_fft)?;
            let target = ifft_complex_batch(&spectrum, cfg.batch, cfg.n_fft)?;
            (pred, target)
        };
        mse_acc += mse(&pred, &target);
        max_acc = max_acc.max(max_abs_error(&pred, &target));
    }
    Ok((mse_acc / batches as f32, max_acc))
}

pub fn evaluate_weights(
    weights: &WeightStore,
    cfg: &FftLearnConfig,
    batches: usize,
) -> Result<(f32, f32)> {
    evaluate_weights_dir(weights, cfg, batches, TransformDir::Forward)
}

pub fn evaluate_weights_dir(
    weights: &WeightStore,
    cfg: &FftLearnConfig,
    batches: usize,
    dir: TransformDir,
) -> Result<(f32, f32)> {
    cfg.validate()?;
    let twiddles = weights.to_twiddles(cfg.n_fft)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    evaluate_twiddles(&twiddles, cfg, dir, batches, &mut rng)
}

pub fn load_weights(path: &Path) -> Result<WeightStore> {
    crate::weights::load_safetensors(path)
}
