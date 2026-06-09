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

//! Encoder–decoder speed and precision benchmarks.

use crate::butterfly::{
    butterfly_encdec_roundtrip_batch, butterfly_forward_real_batch, butterfly_inverse_complex_batch,
};
use crate::config::FftLearnConfig;
use crate::device::resolve_train_device;
use crate::reference::{fft_real_batch, ifft_complex_batch};
use crate::runner::FftLearnRunner;
use crate::train::random_batch;
use crate::train_phased::precision_encdec;
use crate::twiddle::exact_twiddles;
use crate::weights::{EncDecWeights, WeightStore, load_safetensors};
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rlx_runtime::Device;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncDecBenchRow {
    pub label: String,
    pub checkpoint: Option<PathBuf>,
    pub device: String,
    pub iters: usize,
    pub encoder_ms: f64,
    pub decoder_ms: f64,
    pub roundtrip_ms: f64,
    pub rustfft_ms: f64,
    pub encoder_spectrum_mse: f32,
    pub encoder_spectrum_max_err: f32,
    pub decoder_time_mse: f32,
    pub decoder_time_max_err: f32,
    pub roundtrip_mse: f32,
    pub roundtrip_max_err: f32,
    pub roundtrip_compiled_ms: Option<f64>,
}

pub fn bench_encdec_weights(
    weights: &EncDecWeights,
    cfg: &FftLearnConfig,
    iters: usize,
    device_name: &str,
    with_compiled: bool,
    label: &str,
    checkpoint: Option<PathBuf>,
) -> Result<EncDecBenchRow> {
    ensure!(iters >= 1);
    let device = resolve_train_device(Some(device_name))?;
    let encoder_tw = weights.encoder_twiddles(cfg.n_fft)?;
    let decoder_tw = weights.decoder_twiddles(cfg.n_fft)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);

    let signal = random_batch(&mut rng, cfg.batch, cfg.n_fft);
    let spectrum = butterfly_forward_real_batch(&signal, &encoder_tw, cfg.batch, cfg.n_fft)?;

    let encoder_ms = time_iters(iters, || {
        let _ = butterfly_forward_real_batch(&signal, &encoder_tw, cfg.batch, cfg.n_fft)?;
        Ok(())
    })?;

    let decoder_ms = time_iters(iters, || {
        let _ = butterfly_inverse_complex_batch(&spectrum, &decoder_tw, cfg.batch, cfg.n_fft)?;
        Ok(())
    })?;

    let roundtrip_ms = time_iters(iters, || {
        let _ = butterfly_encdec_roundtrip_batch(
            &signal,
            &encoder_tw,
            &decoder_tw,
            cfg.batch,
            cfg.n_fft,
        )?;
        Ok(())
    })?;

    let rustfft_ms = time_iters(iters, || {
        let spec = fft_real_batch(&signal, cfg.batch, cfg.n_fft)?;
        let _ = ifft_complex_batch(&spec, cfg.batch, cfg.n_fft)?;
        Ok(())
    })?;

    let (enc_mse, enc_max, dec_mse, dec_max, rt_mse, rt_max) =
        precision_encdec(&encoder_tw, &decoder_tw, cfg, 16, &mut rng)?;

    let roundtrip_compiled_ms = if with_compiled {
        Some(bench_roundtrip_compiled(
            cfg,
            &encoder_tw,
            &decoder_tw,
            &signal,
            device,
            iters,
        )?)
    } else {
        None
    };

    Ok(EncDecBenchRow {
        label: label.to_string(),
        checkpoint,
        device: format!("{device:?}"),
        iters,
        encoder_ms,
        decoder_ms,
        roundtrip_ms,
        rustfft_ms,
        encoder_spectrum_mse: enc_mse,
        encoder_spectrum_max_err: enc_max,
        decoder_time_mse: dec_mse,
        decoder_time_max_err: dec_max,
        roundtrip_mse: rt_mse,
        roundtrip_max_err: rt_max,
        roundtrip_compiled_ms,
    })
}

pub fn bench_phased_dir(
    phased_dir: &Path,
    cfg: &FftLearnConfig,
    iters: usize,
    device_name: &str,
    with_compiled: bool,
) -> Result<Vec<EncDecBenchRow>> {
    let mut rows = Vec::new();
    for phase in ["phase1_encoder", "phase2_decoder", "phase3_joint"] {
        let path = phased_dir.join(phase).join("encdec.safetensors");
        if !path.is_file() {
            continue;
        }
        let store = load_safetensors(&path)?;
        let weights = EncDecWeights::from_merged(&store, cfg.n_fft)?;
        rows.push(bench_encdec_weights(
            &weights,
            cfg,
            iters,
            device_name,
            with_compiled,
            phase,
            Some(path),
        )?);
    }
    if rows.is_empty() {
        anyhow::bail!(
            "no phase checkpoints under {} (expected phase1_encoder/ …)",
            phased_dir.display()
        );
    }
    Ok(rows)
}

pub fn bench_exact_baseline(
    cfg: &FftLearnConfig,
    iters: usize,
    device_name: &str,
) -> Result<EncDecBenchRow> {
    let tw = exact_twiddles(cfg);
    let weights = EncDecWeights::from_twiddles(&tw, &tw, cfg.n_fft);
    bench_encdec_weights(
        &weights,
        cfg,
        iters,
        device_name,
        false,
        "exact_twiddles",
        None,
    )
}

fn bench_roundtrip_compiled(
    cfg: &FftLearnConfig,
    encoder_tw: &[f32],
    decoder_tw: &[f32],
    signal: &[f32],
    device: Device,
    iters: usize,
) -> Result<f64> {
    let enc_store = WeightStore::from_twiddles(encoder_tw, cfg.n_fft);
    let dec_store = WeightStore::from_twiddles(decoder_tw, cfg.n_fft);
    let mut enc = FftLearnRunner::with_weights(cfg.clone(), &enc_store)?;
    let mut dec = FftLearnRunner::with_weights_ifft(cfg.clone(), &dec_store)?;
    enc.load_compiled(device)?;
    dec.load_compiled(device)?;
    let _ = enc.forward(signal)?;
    time_iters(iters, || {
        let spec = enc.forward(signal)?;
        let _ = dec.forward(&spec)?;
        Ok(())
    })
}

fn time_iters(iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

pub fn print_encdec_bench_table(rows: &[EncDecBenchRow]) {
    eprintln!(
        "\n{:<16} {:>10} {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}",
        "phase", "enc ms", "dec ms", "rt ms", "rust ms", "enc max", "dec max", "rt max"
    );
    for r in rows {
        eprintln!(
            "{:<16} {:>10.4} {:>10.4} {:>10.4} {:>10.4} | {:>10.3e} {:>10.3e} {:>10.3e}",
            r.label,
            r.encoder_ms,
            r.decoder_ms,
            r.roundtrip_ms,
            r.rustfft_ms,
            r.encoder_spectrum_max_err,
            r.decoder_time_max_err,
            r.roundtrip_max_err,
        );
        if let Some(ms) = r.roundtrip_compiled_ms {
            eprintln!("  └ roundtrip compiled: {ms:.4} ms");
        }
    }
}

pub fn write_encdec_bench_json(path: &Path, rows: &[EncDecBenchRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(rows)?)
        .with_context(|| format!("write {}", path.display()))
}
