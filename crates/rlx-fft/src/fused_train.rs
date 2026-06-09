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

//! Fused encoder–decoder training — single forward/backward pass, batched reference FFT.

use crate::butterfly::{
    EncDecStepLoss, apply_conj, backward_butterfly_twiddles, forward_butterfly_traced,
};
use crate::reference::fft_real_batch;
use crate::second_order::TwiddleOptState;
use crate::train::random_batch;
use crate::twiddle_stability::{apply_twiddle_update, lr_for_n_fft};
use anyhow::{Result, ensure};
use rand::prelude::*;

/// Fused enc–dec step: one batched reference spectrum, shared backward path, stability hooks.
pub fn fused_encdec_train_step(
    signal: &[f32],
    encoder_tw: &mut [f32],
    decoder_tw: &mut [f32],
    batch: usize,
    n_fft: usize,
    base_lr: f64,
    spectrum_weight: f32,
    grad_clip: f32,
    project_twiddles: bool,
    opt: Option<&mut TwiddleOptState>,
) -> Result<EncDecStepLoss> {
    ensure!(signal.len() == batch * n_fft);
    let half = n_fft / 2;
    let stages = n_fft.trailing_zeros() as usize;
    let scale = n_fft as f32;
    let norm = (batch * n_fft * 2) as f32;
    let lr = lr_for_n_fft(base_lr, n_fft);

    let ref_spec = fft_real_batch(signal, batch, n_fft)?;

    let mut enc_tw_grad = vec![0f32; stages * half * 2];
    let mut dec_tw_grad = vec![0f32; stages * half * 2];
    let mut recon_loss = 0f32;
    let mut spectrum_loss = 0f32;

    for b in 0..batch {
        let x = &signal[b * n_fft..(b + 1) * n_fft];
        let mut enc_in = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            enc_in[i * 2] = x[i];
        }

        let enc_trace = forward_butterfly_traced(enc_in, encoder_tw, n_fft, true)?;
        let spectrum = enc_trace.output.clone();

        let mut dec_in = spectrum.clone();
        apply_conj(&mut dec_in, n_fft);
        let dec_trace = forward_butterfly_traced(dec_in, decoder_tw, n_fft, true)?;
        let mut recovered = dec_trace.output.clone();
        apply_conj(&mut recovered, n_fft);

        for i in 0..n_fft {
            let d_re = recovered[i * 2] - x[i] * scale;
            let d_im = recovered[i * 2 + 1];
            recon_loss += d_re * d_re + d_im * d_im;
        }

        let ref_b = &ref_spec[b * n_fft * 2..(b + 1) * n_fft * 2];
        if spectrum_weight > 0.0 {
            for i in 0..n_fft * 2 {
                let d = spectrum[i] - ref_b[i];
                spectrum_loss += d * d;
            }
        }

        let mut grad_rec = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            grad_rec[i * 2] = 2.0 * (recovered[i * 2] - x[i] * scale) / norm;
            grad_rec[i * 2 + 1] = 2.0 * recovered[i * 2 + 1] / norm;
        }

        let mut grad_spec = backward_butterfly_twiddles(
            grad_rec,
            &dec_trace,
            decoder_tw,
            n_fft,
            &mut dec_tw_grad,
            true,
        );
        apply_conj(&mut grad_spec, n_fft);

        if spectrum_weight > 0.0 {
            for i in 0..n_fft * 2 {
                grad_spec[i] += 2.0 * spectrum_weight * (spectrum[i] - ref_b[i]) / norm;
            }
        }

        let _ = backward_butterfly_twiddles(
            grad_spec,
            &enc_trace,
            encoder_tw,
            n_fft,
            &mut enc_tw_grad,
            false,
        );
    }

    if let Some(state) = opt {
        state.step_pair(
            encoder_tw,
            decoder_tw,
            &enc_tw_grad,
            &dec_tw_grad,
            lr,
            grad_clip,
            project_twiddles,
        );
    } else {
        apply_twiddle_update(encoder_tw, &enc_tw_grad, lr, grad_clip, project_twiddles);
        apply_twiddle_update(decoder_tw, &dec_tw_grad, lr, grad_clip, project_twiddles);
    }

    Ok(EncDecStepLoss {
        reconstruction: recon_loss / (batch * n_fft) as f32,
        spectrum: if spectrum_weight > 0.0 {
            spectrum_loss / (batch * n_fft * 2) as f32
        } else {
            0.0
        },
    })
}

/// Run fused training loop until convergence or max steps.
pub fn fused_encdec_train_until_converged(
    encoder_tw: &mut [f32],
    decoder_tw: &mut [f32],
    batch: usize,
    n_fft: usize,
    base_lr: f64,
    spectrum_weight: f32,
    max_steps: usize,
    min_steps: usize,
    converge_every: usize,
    converge_patience: usize,
    converge_delta: f32,
    grad_clip: f32,
    project_twiddles: bool,
    optimizer: crate::second_order::TwiddleOptimizer,
    rng: &mut impl Rng,
    label: &str,
) -> Result<(usize, bool, f32)> {
    use crate::config::FftLearnConfig;
    use crate::train_phased::precision_encdec;

    let stages = n_fft.trailing_zeros() as usize;
    let half = n_fft / 2;
    let tw_len = stages * half * 2;
    let mut opt = TwiddleOptState::new(optimizer, tw_len, tw_len);

    let mut best = f32::INFINITY;
    let mut stale = 0usize;
    let mut step = 0usize;
    let mut converged = false;
    let cfg = FftLearnConfig::new(n_fft, batch)?;

    while step < max_steps {
        let signal = random_batch(rng, batch, n_fft);
        fused_encdec_train_step(
            &signal,
            encoder_tw,
            decoder_tw,
            batch,
            n_fft,
            base_lr,
            spectrum_weight,
            grad_clip,
            project_twiddles,
            Some(&mut opt),
        )?;

        step += 1;
        if step >= min_steps && step.is_multiple_of(converge_every) {
            let (_, _, _, _, rt_mse, _) = precision_encdec(encoder_tw, decoder_tw, &cfg, 4, rng)?;
            if rt_mse.is_finite() {
                eprintln!("  [{label}] step {step} holdout_mse={rt_mse:.6e} best={best:.6e}");
                let improved = !best.is_finite()
                    || (best - rt_mse) / best.max(1e-12) > converge_delta
                    || (best - rt_mse) > converge_delta * 1e-4;
                if improved {
                    best = rt_mse;
                    stale = 0;
                } else {
                    stale += 1;
                }
                if stale >= converge_patience {
                    converged = true;
                    break;
                }
            }
        }
    }

    Ok((step, converged, best))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FftLearnConfig;
    use crate::second_order::{TwiddleOptState, TwiddleOptimizer};
    use crate::twiddle::exact_twiddles;
    use crate::twiddle_stability::max_twiddle_magnitude;
    use rand::SeedableRng;

    #[test]
    fn fused_1024_no_nan_over_steps() {
        let n = 1024usize;
        let batch = 4;
        let model = FftLearnConfig::new(n, batch).unwrap();
        let mut enc = exact_twiddles(&model);
        let mut dec = exact_twiddles(&model);
        let stages = n.trailing_zeros() as usize;
        let tw_len = stages * (n / 2) * 2;
        let mut opt = TwiddleOptState::new(TwiddleOptimizer::Adam, tw_len, tw_len);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for step in 0..40 {
            let signal = random_batch(&mut rng, batch, n);
            let loss = fused_encdec_train_step(
                &signal,
                &mut enc,
                &mut dec,
                batch,
                n,
                1e-4,
                1.0,
                1.0,
                true,
                Some(&mut opt),
            )
            .unwrap();
            assert!(loss.reconstruction.is_finite(), "step {step} recon NaN");
            assert!(max_twiddle_magnitude(&enc) < 1.01);
        }
    }
}
