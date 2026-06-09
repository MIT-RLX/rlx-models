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

//! Three-phase encoder → decoder → joint training.

use crate::butterfly::{butterfly_train_step_dir, butterfly_train_step_encdec};
use crate::config::{FftLearnConfig, PhasedTrainConfig, TransformDir};
use crate::reference::{fft_real_batch, ifft_complex_batch, max_abs_error, mse, roundtrip_scale};
use crate::train::{random_batch, random_complex_batch};
use crate::twiddle::exact_twiddles;
use crate::weights::{EncDecWeights, export_safetensors};
use anyhow::Result;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    pub name: String,
    pub steps: usize,
    pub elapsed_ms: f64,
    pub encoder_spectrum_mse: f32,
    pub encoder_spectrum_max_err: f32,
    pub decoder_time_mse: f32,
    pub decoder_time_max_err: f32,
    pub roundtrip_mse: f32,
    pub roundtrip_max_err: f32,
    pub checkpoint: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PhasedTrainResult {
    pub phases: Vec<PhaseMetrics>,
    pub final_weights: EncDecWeights,
    pub total_elapsed_ms: f64,
}

pub fn train_phased_encdec(cfg: &PhasedTrainConfig) -> Result<PhasedTrainResult> {
    cfg.model.validate()?;
    let out = cfg
        .out_dir
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("train-phased requires --out DIR"))?;
    std::fs::create_dir_all(out)?;

    let n = cfg.model.n_fft;
    let batch = cfg.model.batch;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut encoder_tw = exact_twiddles(&cfg.model);
    let mut decoder_tw = exact_twiddles(&cfg.model);
    let total_started = Instant::now();
    let mut phases = Vec::new();

    if cfg.encoder_steps > 0 {
        let started = Instant::now();
        for step in 0..cfg.encoder_steps {
            let signal = random_batch(&mut rng, batch, n);
            butterfly_train_step_dir(
                &signal,
                &mut encoder_tw,
                batch,
                n,
                cfg.lr as f32,
                TransformDir::Forward,
            )?;
            if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
                eprintln!("[phase1 encoder] step {}/{}", step + 1, cfg.encoder_steps);
            }
        }
        let checkpoint = save_phase(out, "phase1_encoder", &encoder_tw, &decoder_tw, n)?;
        phases.push(measure_phase(
            "phase1_encoder",
            cfg.encoder_steps,
            started.elapsed().as_secs_f64() * 1000.0,
            &encoder_tw,
            &decoder_tw,
            &cfg.model,
            &mut rng,
            checkpoint,
        )?);
    }

    if cfg.decoder_steps > 0 {
        let started = Instant::now();
        for step in 0..cfg.decoder_steps {
            let spectrum = random_complex_batch(&mut rng, batch, n);
            butterfly_train_step_dir(
                &spectrum,
                &mut decoder_tw,
                batch,
                n,
                cfg.lr as f32,
                TransformDir::Inverse,
            )?;
            if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
                eprintln!("[phase2 decoder] step {}/{}", step + 1, cfg.decoder_steps);
            }
        }
        let checkpoint = save_phase(out, "phase2_decoder", &encoder_tw, &decoder_tw, n)?;
        phases.push(measure_phase(
            "phase2_decoder",
            cfg.decoder_steps,
            started.elapsed().as_secs_f64() * 1000.0,
            &encoder_tw,
            &decoder_tw,
            &cfg.model,
            &mut rng,
            checkpoint,
        )?);
    }

    if cfg.joint_steps > 0 {
        let started = Instant::now();
        for step in 0..cfg.joint_steps {
            let signal = random_batch(&mut rng, batch, n);
            butterfly_train_step_encdec(
                &signal,
                &mut encoder_tw,
                &mut decoder_tw,
                batch,
                n,
                cfg.lr as f32,
                cfg.spectrum_weight,
            )?;
            if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
                eprintln!("[phase3 joint] step {}/{}", step + 1, cfg.joint_steps);
            }
        }
        let checkpoint = save_phase(out, "phase3_joint", &encoder_tw, &decoder_tw, n)?;
        phases.push(measure_phase(
            "phase3_joint",
            cfg.joint_steps,
            started.elapsed().as_secs_f64() * 1000.0,
            &encoder_tw,
            &decoder_tw,
            &cfg.model,
            &mut rng,
            checkpoint,
        )?);
    }

    let final_weights = EncDecWeights::from_twiddles(&encoder_tw, &decoder_tw, n);
    let report = PhasedTrainResult {
        phases,
        final_weights,
        total_elapsed_ms: total_started.elapsed().as_secs_f64() * 1000.0,
    };
    std::fs::write(
        out.join("phased_train_report.json"),
        serde_json::to_vec_pretty(&report.phases)?,
    )?;
    Ok(report)
}

fn save_phase(
    out: &Path,
    name: &str,
    encoder_tw: &[f32],
    decoder_tw: &[f32],
    n_fft: usize,
) -> Result<PathBuf> {
    let dir = out.join(name);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("encdec.safetensors");
    let weights = EncDecWeights::from_twiddles(encoder_tw, decoder_tw, n_fft);
    export_safetensors(&path, &weights.merged())?;
    Ok(path)
}

fn measure_phase(
    name: &str,
    steps: usize,
    elapsed_ms: f64,
    encoder_tw: &[f32],
    decoder_tw: &[f32],
    cfg: &FftLearnConfig,
    rng: &mut impl Rng,
    checkpoint: PathBuf,
) -> Result<PhaseMetrics> {
    let (enc_mse, enc_max, dec_mse, dec_max, rt_mse, rt_max) =
        precision_encdec(encoder_tw, decoder_tw, cfg, 8, rng)?;
    Ok(PhaseMetrics {
        name: name.to_string(),
        steps,
        elapsed_ms,
        encoder_spectrum_mse: enc_mse,
        encoder_spectrum_max_err: enc_max,
        decoder_time_mse: dec_mse,
        decoder_time_max_err: dec_max,
        roundtrip_mse: rt_mse,
        roundtrip_max_err: rt_max,
        checkpoint,
    })
}

pub fn precision_encdec(
    encoder_tw: &[f32],
    decoder_tw: &[f32],
    cfg: &FftLearnConfig,
    batches: usize,
    rng: &mut impl Rng,
) -> Result<(f32, f32, f32, f32, f32, f32)> {
    use crate::butterfly::{
        butterfly_encdec_roundtrip_batch, butterfly_forward_real_batch,
        butterfly_inverse_complex_batch,
    };

    let mut enc_mse = 0f32;
    let mut enc_max = 0f32;
    let mut dec_mse = 0f32;
    let mut dec_max = 0f32;
    let mut rt_mse = 0f32;
    let mut rt_max = 0f32;
    let scale = roundtrip_scale(cfg.n_fft);

    for _ in 0..batches {
        let signal = random_batch(rng, cfg.batch, cfg.n_fft);
        let pred_spec = butterfly_forward_real_batch(&signal, encoder_tw, cfg.batch, cfg.n_fft)?;
        let ref_spec = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        enc_mse += mse(&pred_spec, &ref_spec);
        enc_max = enc_max.max(max_abs_error(&pred_spec, &ref_spec));

        let pred_time =
            butterfly_inverse_complex_batch(&ref_spec, decoder_tw, cfg.batch, cfg.n_fft)?;
        let ref_time = ifft_complex_batch(&ref_spec, cfg.batch, cfg.n_fft)?;
        dec_mse += mse(&pred_time, &ref_time);
        dec_max = dec_max.max(max_abs_error(&pred_time, &ref_time));

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
        rt_mse += mse(&recovered, &target);
        rt_max = rt_max.max(max_abs_error(&recovered, &target));
    }
    let n = batches as f32;
    Ok((
        enc_mse / n,
        enc_max,
        dec_mse / n,
        dec_max,
        rt_mse / n,
        rt_max,
    ))
}
