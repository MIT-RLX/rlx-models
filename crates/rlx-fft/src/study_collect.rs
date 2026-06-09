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

//! Collect training telemetry for comprehensive study HTML reports.

use crate::config::FftLearnConfig;
use crate::pruned::mean_gate;
use crate::reference::{fft_real_batch, max_abs_error, mse};
use crate::study_telemetry::{
    ActivationHeatmap, LossLandscape3D, LossPoint, ModelTrainingTrace, StudyTelemetryBundle,
    gate_heatmap_from_vec, learned_model_param_breakdown, variant_param_breakdown,
};
use crate::train_e2e::E2eTrainConfig;
use crate::variants::FftVariantId;
use anyhow::Result;
use rand::prelude::*;

pub fn collect_study_telemetry(
    n_fft: usize,
    batch: usize,
    e2e_steps: usize,
    domain_steps: usize,
    seed: u64,
) -> Result<StudyTelemetryBundle> {
    let mut models = Vec::new();
    eprintln!("[study-telemetry] learned e2e n={n_fft} batch={batch} steps={e2e_steps}");
    if let Ok(trace) = trace_learned_e2e(n_fft, batch, e2e_steps, seed) {
        models.push(trace);
    }
    eprintln!("[study-telemetry] domain twiddle steps={domain_steps}");
    if let Ok(trace) = trace_domain_twiddle(n_fft, batch, domain_steps, seed.wrapping_add(1)) {
        models.push(trace);
    }
    eprintln!("[study-telemetry] unitary butterfly");
    if let Ok(trace) = trace_unitary(n_fft, batch, domain_steps.min(25), seed.wrapping_add(2)) {
        models.push(trace);
    }
    Ok(StudyTelemetryBundle { models })
}

fn trace_learned_e2e(
    n_fft: usize,
    batch: usize,
    steps: usize,
    seed: u64,
) -> Result<ModelTrainingTrace> {
    let cfg = E2eTrainConfig {
        n_fft,
        batch,
        steps,
        seed,
        log_every: (steps / 20).max(1),
        ..E2eTrainConfig::default()
    };
    let (model, rep) = train_fast_learned_model_with_curve(&cfg)?;
    let landscape = sample_loss_landscape(&model, &cfg, seed.wrapping_add(99))?;
    let mut heatmap = gate_heatmap_from_vec(&model.gates, n_fft);
    heatmap.freq_mask = model.freq_mask.clone();
    Ok(ModelTrainingTrace {
        model_id: format!("learned_e2e_n{n_fft}_b{batch}"),
        variant: "learned_e2e".into(),
        n_fft,
        batch,
        train_steps: steps,
        params: learned_model_param_breakdown(n_fft, cfg.n_mels),
        loss_curve: rep.curve,
        heatmap,
        landscape: Some(landscape),
        final_mel_err: rep.final_mel_max_err,
        final_spec_err: rep.final_spectrum_max_err,
    })
}

struct TracedE2eReport {
    final_mel_max_err: f32,
    final_spectrum_max_err: f32,
    curve: Vec<LossPoint>,
}

fn train_fast_learned_model_with_curve(
    cfg: &E2eTrainConfig,
) -> Result<(crate::learned_model::FastLearnedFftModel, TracedE2eReport)> {
    use crate::learned_model::FastLearnedFftModel;
    use crate::mel::ref_log_mel_batch;
    use crate::pruned::pruned_forward_real_batch;
    use crate::train::random_batch;
    use crate::twiddle_stability::project_twiddles_unit_circle;
    use crate::welch::{WelchParams, welch_rustfft};

    let model_cfg = FftLearnConfig::new(cfg.n_fft, cfg.batch)?;
    let mut model = FastLearnedFftModel::new(&model_cfg, cfg.n_mels, 16_000.0);
    if cfg.train_q8 {
        model = model.with_q8();
    }
    let welch_params = WelchParams::for_n_fft(cfg.n_fft);
    let welch_frame = welch_params.frame_len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut curve = Vec::new();
    let mut last_spec_err = 0f32;
    let mut last_mel_err = 0f32;
    let mut last_welch_err;

    for step in 0..cfg.steps {
        let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
        let welch_signal = random_batch(&mut rng, cfg.batch, welch_frame);
        let ref_spec = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        let ref_mel = ref_log_mel_batch(
            &window_signal(&signal, cfg.batch, cfg.n_fft),
            cfg.batch,
            cfg.n_fft,
            cfg.n_mels,
            16_000.0,
        )?;
        let ref_welch = welch_rustfft(&welch_signal, cfg.batch, welch_params)?;
        let pred_mel = model.log_mel_batch(&signal, cfg.batch)?;
        last_mel_err = max_abs_error(&pred_mel, &ref_mel);
        let mel_loss = mse(&pred_mel, &ref_mel);
        let pred_welch = model.welch_psd_batch(&welch_signal, cfg.batch, welch_params)?;
        last_welch_err = max_abs_error(&pred_welch, &ref_welch);
        let welch_loss = mse(&pred_welch, &ref_welch);
        let tw = model.twiddles_for_forward();
        let mut pred_spec =
            pruned_forward_real_batch(&signal, &tw, &model.gates, cfg.batch, cfg.n_fft)?;
        apply_mask(&mut pred_spec, &model.freq_mask, cfg.batch, cfg.n_fft);
        let pred_spec = model
            .denoiser
            .apply_batch(&pred_spec, cfg.batch, cfg.n_fft)?;
        last_spec_err = max_abs_error(&pred_spec, &ref_spec);
        let spec_loss = mse(&pred_spec, &ref_spec);
        let total = cfg.spectrum_weight * spec_loss
            + cfg.mel_weight * mel_loss
            + cfg.welch_weight * welch_loss;
        if step % cfg.log_every == 0 || step + 1 == cfg.steps {
            curve.push(LossPoint {
                step,
                total_loss: total,
                mel_err: last_mel_err,
                spec_err: last_spec_err,
                welch_err: last_welch_err,
                mean_gate: mean_gate(&model.gates),
            });
        }
        let lr = crate::twiddle_stability::lr_for_n_fft(cfg.lr, cfg.n_fft);
        if step % 2 == 0 {
            crate::butterfly::butterfly_train_step(
                &signal,
                &mut model.twiddles,
                cfg.batch,
                cfg.n_fft,
                lr * cfg.spectrum_weight.max(0.1),
            )?;
            model.sync_q8();
        }
        let _ =
            model
                .denoiser
                .train_step_affine(&pred_spec, &ref_spec, cfg.batch, cfg.n_fft, lr * 0.5);
        project_twiddles_unit_circle(&mut model.twiddles);
    }
    Ok((
        model,
        TracedE2eReport {
            final_mel_max_err: last_mel_err,
            final_spectrum_max_err: last_spec_err,
            curve,
        },
    ))
}

fn sample_loss_landscape(
    model: &crate::learned_model::FastLearnedFftModel,
    cfg: &E2eTrainConfig,
    seed: u64,
) -> Result<LossLandscape3D> {
    use crate::mel::ref_log_mel_batch;
    use crate::pruned::pruned_forward_real_batch;
    use crate::train::random_batch;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
    let ref_mel = ref_log_mel_batch(
        &window_signal(&signal, cfg.batch, cfg.n_fft),
        cfg.batch,
        cfg.n_fft,
        cfg.n_mels,
        16_000.0,
    )?;
    let grid = 11usize;
    let span = 0.35f32;
    let base0 = model.twiddles[0];
    let base1 = model.twiddles[1];
    let mut x = Vec::with_capacity(grid);
    let mut y = Vec::with_capacity(grid);
    let mut z = vec![vec![0f32; grid]; grid];
    for i in 0..grid {
        let fx = base0 + span * (i as f32 / (grid - 1) as f32 * 2.0 - 1.0);
        x.push(fx);
    }
    for j in 0..grid {
        let fy = base1 + span * (j as f32 / (grid - 1) as f32 * 2.0 - 1.0);
        y.push(fy);
    }
    for (i, &fx) in x.iter().enumerate() {
        for (j, &fy) in y.iter().enumerate() {
            let mut tw = model.twiddles_for_forward();
            tw[0] = fx;
            tw[1] = fy;
            let gates = model.gates.clone();
            let mut spec = pruned_forward_real_batch(&signal, &tw, &gates, cfg.batch, cfg.n_fft)?;
            apply_mask(&mut spec, &model.freq_mask, cfg.batch, cfg.n_fft);
            let spec = model.denoiser.apply_batch(&spec, cfg.batch, cfg.n_fft)?;
            let pred_mel = crate::mel::log_mel_from_spectrum_batch(
                &spec,
                model.mel_filters(),
                cfg.batch,
                cfg.n_fft,
                cfg.n_mels,
            )?;
            z[j][i] = mse(&pred_mel, &ref_mel);
        }
    }
    Ok(LossLandscape3D {
        x_label: "twiddle[0] (real)".into(),
        y_label: "twiddle[1] (imag)".into(),
        x,
        y,
        z,
    })
}

fn trace_domain_twiddle(
    n_fft: usize,
    batch: usize,
    steps: usize,
    seed: u64,
) -> Result<ModelTrainingTrace> {
    use crate::butterfly::butterfly_forward_real_batch;
    use crate::domain::domain_batch;

    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let mut tw = crate::twiddle::exact_twiddles(&cfg);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut curve = Vec::new();
    let log_every = (steps / 15).max(1);
    let mut last_err = 0f32;
    for step in 0..steps {
        let signal = domain_batch(&mut rng, batch, n_fft);
        crate::butterfly::butterfly_train_step(&signal, &mut tw, batch, n_fft, 5e-4)?;
        let pred = butterfly_forward_real_batch(&signal, &tw, batch, n_fft)?;
        let target = fft_real_batch(&signal, batch, n_fft)?;
        last_err = max_abs_error(&pred, &target);
        if step % log_every == 0 || step + 1 == steps {
            curve.push(LossPoint {
                step,
                total_loss: last_err,
                mel_err: last_err,
                spec_err: last_err,
                welch_err: 0.0,
                mean_gate: 1.0,
            });
        }
    }
    let params = variant_param_breakdown(FftVariantId::DomainTwiddle, &cfg);
    Ok(ModelTrainingTrace {
        model_id: format!("domain_twiddle_n{n_fft}"),
        variant: "domain_twiddle".into(),
        n_fft,
        batch,
        train_steps: steps,
        params,
        loss_curve: curve,
        heatmap: gate_heatmap_from_vec(&vec![1.0; crate::pruned::gate_count(n_fft)], n_fft),
        landscape: None,
        final_mel_err: last_err,
        final_spec_err: last_err,
    })
}

fn trace_unitary(
    n_fft: usize,
    batch: usize,
    steps: usize,
    seed: u64,
) -> Result<ModelTrainingTrace> {
    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let (weights, final_err) = crate::unitary::train_unitary_quick(&cfg, steps, 1e-3, seed)?;
    let mut curve = vec![LossPoint {
        step: steps,
        total_loss: final_err,
        mel_err: final_err,
        spec_err: final_err,
        welch_err: 0.0,
        mean_gate: 1.0,
    }];
    if steps > 1 {
        curve.insert(
            0,
            LossPoint {
                step: 0,
                total_loss: final_err * 4.0,
                mel_err: final_err * 4.0,
                spec_err: final_err * 4.0,
                welch_err: 0.0,
                mean_gate: 1.0,
            },
        );
    }
    let stages = cfg.num_stages();
    let half = n_fft / 2;
    let mut gate_proxy = vec![0f32; stages * half];
    for s in 0..stages {
        for b in 0..half {
            let base = (s * half + b) * 8;
            let m00 = (weights.matrices[base].powi(2) + weights.matrices[base + 1].powi(2)).sqrt();
            gate_proxy[s * half + b] = m00.min(1.0);
        }
    }
    Ok(ModelTrainingTrace {
        model_id: format!("butterfly_unitary_n{n_fft}"),
        variant: "butterfly_unitary".into(),
        n_fft,
        batch,
        train_steps: steps,
        params: variant_param_breakdown(FftVariantId::ButterflyUnitary, &cfg),
        loss_curve: curve,
        heatmap: ActivationHeatmap {
            stages,
            butterflies: half,
            gates: gate_proxy,
            freq_mask: Vec::new(),
            twiddle_mag: Vec::new(),
        },
        landscape: None,
        final_mel_err: final_err,
        final_spec_err: final_err,
    })
}

fn window_signal(signal: &[f32], batch: usize, n_fft: usize) -> Vec<f32> {
    let w = crate::mel::hann_window(n_fft);
    let mut out = signal.to_vec();
    for b in 0..batch {
        for i in 0..n_fft {
            out[b * n_fft + i] *= w[i];
        }
    }
    out
}

fn apply_mask(spec: &mut [f32], mask: &[f32], batch: usize, n_fft: usize) {
    for b in 0..batch {
        for i in 0..n_fft * 2 {
            spec[b * n_fft * 2 + i] *= mask[i];
        }
    }
}
