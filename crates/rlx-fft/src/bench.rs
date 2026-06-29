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

//! Benchmark learned butterfly FFT vs `rustfft` and native RLX `Op::Fft`.

use crate::butterfly::{butterfly_forward_real_batch, butterfly_inverse_complex_batch};
use crate::config::{FftLearnConfig, TransformDir};
use crate::device::resolve_train_device;
use crate::reference::{fft_real_batch, ifft_complex_batch, max_abs_error};
use crate::runner::FftLearnRunner;
use crate::train::{random_batch, random_complex_batch};
use crate::twiddle::exact_twiddles;
use crate::weights::{EncDecWeights, WeightStore, load_safetensors};
use anyhow::{Result, ensure};
use rand::prelude::*;
use rlx_runtime::Device;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BenchReport {
    pub direction: TransformDir,
    pub n_fft: usize,
    pub batch: usize,
    pub iters: usize,
    pub device: Device,
    /// `exact twiddles` or `learned checkpoint`.
    pub butterfly_weights: String,
    pub rustfft_ms: f64,
    pub rlx_fft_ms: f64,
    pub butterfly_eager_ms: f64,
    pub butterfly_compiled_ms: f64,
    pub rlx_fft_err: f32,
    pub butterfly_eager_err: f32,
    pub butterfly_compiled_err: f32,
}

pub fn bench_all_dir(
    n_fft: usize,
    batch: usize,
    iters: usize,
    dir: TransformDir,
    device: Device,
    with_butterfly_compiled: bool,
    weights_path: Option<&Path>,
) -> Result<BenchReport> {
    ensure!(iters >= 1);
    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let (twiddles, butterfly_weights) = resolve_butterfly_weights(&cfg, dir, weights_path)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);

    let (signal, spectrum_interleaved, rlx_input, rlx_input_name) = if dir.is_forward() {
        let signal = random_batch(&mut rng, batch, n_fft);
        (signal, Vec::new(), None, "")
    } else {
        let spectrum = random_complex_batch(&mut rng, batch, n_fft);
        let block = crate::rlx_fft::interleaved_to_block(&spectrum, batch, n_fft);
        (Vec::new(), spectrum, Some(block), "spectrum")
    };

    let rustfft_ms = time_iters(iters, || {
        if dir.is_forward() {
            let _ = fft_real_batch(&signal, batch, n_fft)?;
        } else {
            let _ = ifft_complex_batch(&spectrum_interleaved, batch, n_fft)?;
        }
        Ok(())
    })?;

    eprintln!("[bench] compiling native RLX Op::Fft on {device:?}…");
    let mut rlx_exec = crate::rlx_fft::compile_rlx_fft(&cfg, dir, device)?;
    // Warm the pipeline: the first run() lazily builds the GPU PSO / first-dispatch
    // shader on Metal/wgpu, which would otherwise pollute the small-batch cells.
    if dir.is_forward() {
        rlx_exec.run(&[("signal", &signal)]);
    } else {
        rlx_exec.run(&[(rlx_input_name, rlx_input.as_ref().expect("ifft block"))]);
    }
    let rlx_fft_ms = time_iters(iters, || {
        if dir.is_forward() {
            rlx_exec.run(&[("signal", &signal)]);
        } else {
            let block = rlx_input.as_ref().expect("ifft block");
            rlx_exec.run(&[(rlx_input_name, block)]);
        }
        Ok(())
    })?;

    let target = if dir.is_forward() {
        fft_real_batch(&signal, batch, n_fft)?
    } else {
        ifft_complex_batch(&spectrum_interleaved, batch, n_fft)?
    };

    let rlx_out = if dir.is_forward() {
        rlx_exec.run(&[("signal", &signal)])
    } else {
        rlx_exec.run(&[(rlx_input_name, rlx_input.as_ref().unwrap())])
    };
    let rlx_pred = crate::reference::block_to_interleaved(&rlx_out[0], batch, n_fft);
    let rlx_fft_err = max_abs_error(&rlx_pred, &target);

    let butterfly_eager_ms = time_iters(iters, || {
        if dir.is_forward() {
            let _ = butterfly_forward_real_batch(&signal, &twiddles, batch, n_fft)?;
        } else {
            let _ =
                butterfly_inverse_complex_batch(&spectrum_interleaved, &twiddles, batch, n_fft)?;
        }
        Ok(())
    })?;

    let compiled_input = if dir.is_forward() {
        signal.clone()
    } else {
        spectrum_interleaved.clone()
    };

    let eager_pred = if dir.is_forward() {
        butterfly_forward_real_batch(&signal, &twiddles, batch, n_fft)?
    } else {
        butterfly_inverse_complex_batch(&spectrum_interleaved, &twiddles, batch, n_fft)?
    };
    let butterfly_eager_err = max_abs_error(&eager_pred, &target);

    let (butterfly_compiled_ms, butterfly_compiled_err) = if with_butterfly_compiled {
        eprintln!("[bench] compiling learned butterfly graph on {device:?}…");
        match bench_butterfly_compiled(
            &cfg,
            dir,
            device,
            &compiled_input,
            &target,
            iters,
            &twiddles,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[bench] butterfly compiled skipped: {e:#}");
                (f64::NAN, f32::NAN)
            }
        }
    } else {
        (f64::NAN, f32::NAN)
    };

    Ok(BenchReport {
        direction: dir,
        n_fft,
        batch,
        iters,
        device,
        butterfly_weights,
        rustfft_ms,
        rlx_fft_ms,
        butterfly_eager_ms,
        butterfly_compiled_ms,
        rlx_fft_err,
        butterfly_eager_err,
        butterfly_compiled_err,
    })
}

pub fn bench_all(
    n_fft: usize,
    batch: usize,
    iters: usize,
    dir: TransformDir,
    device_name: &str,
    with_butterfly_compiled: bool,
    weights_path: Option<&Path>,
) -> Result<BenchReport> {
    let device = resolve_train_device(Some(device_name))?;
    bench_all_dir(
        n_fft,
        batch,
        iters,
        dir,
        device,
        with_butterfly_compiled,
        weights_path,
    )
}

/// Legacy API: rustfft vs butterfly eager only.
pub fn bench_reference_vs_learned_dir(
    n_fft: usize,
    batch: usize,
    iters: usize,
    dir: TransformDir,
) -> Result<(f64, f64, f32)> {
    let report = bench_all_dir(n_fft, batch, iters, dir, Device::Cpu, false, None)?;
    Ok((
        report.rustfft_ms,
        report.butterfly_eager_ms,
        report.butterfly_eager_err,
    ))
}

pub fn bench_reference_vs_learned(
    n_fft: usize,
    batch: usize,
    iters: usize,
) -> Result<(f64, f64, f32)> {
    bench_reference_vs_learned_dir(n_fft, batch, iters, TransformDir::Forward)
}

fn bench_butterfly_compiled(
    cfg: &FftLearnConfig,
    dir: TransformDir,
    device: Device,
    input: &[f32],
    target: &[f32],
    iters: usize,
    twiddles: &[f32],
) -> Result<(f64, f32)> {
    let store = WeightStore::from_twiddles(twiddles, cfg.n_fft);
    let mut runner = FftLearnRunner::with_weights_dir(cfg.clone(), &store, dir)?;
    runner.load_compiled(device)?;
    let _ = runner.forward(input)?;
    let ms = time_iters(iters, || {
        let _ = runner.forward(input)?;
        Ok(())
    })?;
    let pred = runner.forward(input)?;
    Ok((ms, max_abs_error(&pred, target)))
}

fn resolve_butterfly_weights(
    cfg: &FftLearnConfig,
    dir: TransformDir,
    weights_path: Option<&Path>,
) -> Result<(Vec<f32>, String)> {
    let Some(path) = weights_path else {
        return Ok((exact_twiddles(cfg), "exact twiddles".to_string()));
    };

    let store = load_safetensors(path)?;
    if let Ok(tw) = store.to_twiddles(cfg.n_fft) {
        return Ok((tw, format!("learned ({})", path.display())));
    }

    let encdec = EncDecWeights::from_merged(&store, cfg.n_fft)?;
    let tw = if dir.is_forward() {
        encdec.encoder_twiddles(cfg.n_fft)?
    } else {
        encdec.decoder_twiddles(cfg.n_fft)?
    };
    Ok((
        tw,
        format!(
            "learned encdec {} ({})",
            if dir.is_forward() {
                "encoder"
            } else {
                "decoder"
            },
            path.display()
        ),
    ))
}

fn time_iters(iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FftLearnConfig;

    #[test]
    fn rlx_fft_graph_builds() {
        use crate::rlx_fft::{build_rlx_fft_forward_graph, build_rlx_fft_inverse_graph};
        let cfg = FftLearnConfig::new(64, 2).unwrap();
        let _ = build_rlx_fft_forward_graph(&cfg);
        let _ = build_rlx_fft_inverse_graph(&cfg);
    }

    #[test]
    #[ignore = "slow compile; run with `cargo test -p rlx-fft bench_cpu_forward_smoke -- --ignored`"]
    fn bench_cpu_forward_smoke() {
        let report = bench_all_dir(64, 4, 3, TransformDir::Forward, Device::Cpu, false, None)
            .expect("bench");
        assert!(report.rustfft_ms >= 0.0);
        assert!(report.rlx_fft_ms >= 0.0);
        assert!(report.rlx_fft_err < 1e-3);
    }
}
