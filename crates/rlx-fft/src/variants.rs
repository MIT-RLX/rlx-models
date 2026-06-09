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

//! FFT implementation variants for ablation (Tiers A/B/C + baselines).

use crate::butterfly::{build_butterfly_forward_graph, butterfly_forward_real_batch};
use crate::config::FftLearnConfig;
use crate::domain::train_domain_twiddles;
use crate::fused::{build_fused_spectral_graph, fused_spectral_eager, unit_mask};
use crate::q8::Q8Twiddles;
use crate::reference::{fft_real_batch, max_abs_error};
use crate::stockham::{build_stockham_forward_graph, stockham_forward_real_batch};
use crate::train::random_batch;
use crate::twiddle::exact_twiddles;
use crate::unitary::{UnitaryWeights, train_unitary_quick};
use crate::welch::{
    WelchParams, compile_welch_rlx_fft, welch_butterfly, welch_rlx_op_fft, welch_rustfft,
};
use anyhow::{Result, bail};
use rand::prelude::*;
use rlx_runtime::{CompiledGraph, Device};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FftVariantId {
    Rustfft,
    RlxOpFft,
    RlxOpIfft,
    ButterflyEager,
    ButterflyCompiled,
    StockhamEager,
    StockhamCompiled,
    FusedSpectralEager,
    FusedSpectralCompiled,
    ButterflyQ8,
    ButterflyUnitary,
    DomainTwiddle,
    WelchRustfft,
    WelchRlxOpFft,
    WelchButterflyEager,
    WelchButterflyCompiled,
}

impl FftVariantId {
    pub fn all() -> &'static [Self] {
        Self::all_with_compiled(true)
    }

    pub fn all_eager() -> &'static [Self] {
        Self::all_with_compiled(false)
    }

    fn all_with_compiled(compiled: bool) -> &'static [Self] {
        if compiled {
            &[
                Self::Rustfft,
                Self::RlxOpFft,
                Self::RlxOpIfft,
                Self::ButterflyEager,
                Self::ButterflyCompiled,
                Self::StockhamEager,
                Self::StockhamCompiled,
                Self::FusedSpectralEager,
                Self::FusedSpectralCompiled,
                Self::ButterflyQ8,
                Self::ButterflyUnitary,
                Self::DomainTwiddle,
            ]
        } else {
            &[
                Self::Rustfft,
                Self::RlxOpFft,
                Self::RlxOpIfft,
                Self::ButterflyEager,
                Self::StockhamEager,
                Self::FusedSpectralEager,
                Self::ButterflyQ8,
                Self::ButterflyUnitary,
                Self::DomainTwiddle,
            ]
        }
    }

    pub fn welch_variants(with_compiled: bool) -> &'static [Self] {
        if with_compiled {
            &[
                Self::WelchRustfft,
                Self::WelchRlxOpFft,
                Self::WelchButterflyEager,
                Self::WelchButterflyCompiled,
            ]
        } else {
            &[
                Self::WelchRustfft,
                Self::WelchRlxOpFft,
                Self::WelchButterflyEager,
            ]
        }
    }

    pub fn is_welch(self) -> bool {
        matches!(
            self,
            Self::WelchRustfft
                | Self::WelchRlxOpFft
                | Self::WelchButterflyEager
                | Self::WelchButterflyCompiled
        )
    }

    pub fn tier(self) -> &'static str {
        match self {
            Self::Rustfft
            | Self::RlxOpFft
            | Self::RlxOpIfft
            | Self::ButterflyEager
            | Self::WelchRustfft
            | Self::WelchRlxOpFft
            | Self::WelchButterflyEager => "baseline",
            Self::StockhamEager
            | Self::StockhamCompiled
            | Self::FusedSpectralEager
            | Self::FusedSpectralCompiled => "A",
            Self::ButterflyCompiled | Self::ButterflyQ8 | Self::WelchButterflyCompiled => "B",
            Self::ButterflyUnitary | Self::DomainTwiddle => "C",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Rustfft => "rustfft",
            Self::RlxOpFft => "rlx_op_fft",
            Self::RlxOpIfft => "rlx_op_ifft",
            Self::ButterflyEager => "butterfly_eager",
            Self::ButterflyCompiled => "butterfly_compiled",
            Self::StockhamEager => "stockham_eager",
            Self::StockhamCompiled => "stockham_compiled",
            Self::FusedSpectralEager => "fused_spectral_eager",
            Self::FusedSpectralCompiled => "fused_spectral_compiled",
            Self::ButterflyQ8 => "butterfly_q8",
            Self::ButterflyUnitary => "butterfly_unitary",
            Self::DomainTwiddle => "domain_twiddle",
            Self::WelchRustfft => "welch_rustfft",
            Self::WelchRlxOpFft => "welch_rlx_op_fft",
            Self::WelchButterflyEager => "welch_butterfly_eager",
            Self::WelchButterflyCompiled => "welch_butterfly_compiled",
        }
    }

    pub fn supports_inverse(self) -> bool {
        matches!(self, Self::Rustfft | Self::RlxOpIfft | Self::ButterflyEager)
    }

    pub fn needs_training(self) -> bool {
        matches!(self, Self::ButterflyUnitary | Self::DomainTwiddle)
    }
}

pub struct VariantState {
    pub twiddles: Vec<f32>,
    pub q8: Option<Q8Twiddles>,
    pub unitary: Option<UnitaryWeights>,
    pub mask: Vec<f32>,
    compiled_butterfly: Option<CompiledGraph>,
    compiled_stockham: Option<CompiledGraph>,
    compiled_fused: Option<CompiledGraph>,
    compiled_rlx: Option<CompiledGraph>,
    compiled_rlx_inv: Option<CompiledGraph>,
    compiled_welch_rlx: Option<CompiledGraph>,
    compiled_welch_butterfly: Option<CompiledGraph>,
    spectrum_block: Vec<f32>,
    inverse_spectrum: Vec<f32>,
}

impl VariantState {
    pub fn new(cfg: &FftLearnConfig) -> Self {
        Self {
            twiddles: exact_twiddles(cfg),
            q8: None,
            unitary: None,
            mask: unit_mask(cfg.n_fft),
            compiled_butterfly: None,
            compiled_stockham: None,
            compiled_fused: None,
            compiled_rlx: None,
            compiled_rlx_inv: None,
            compiled_welch_rlx: None,
            compiled_welch_butterfly: None,
            spectrum_block: Vec::new(),
            inverse_spectrum: Vec::new(),
        }
    }

    pub fn set_inverse_input_block(&mut self, block: Vec<f32>) {
        self.spectrum_block = block;
    }

    pub fn set_inverse_spectrum(&mut self, spectrum: Vec<f32>) {
        self.inverse_spectrum = spectrum;
    }

    pub fn prepare(
        &mut self,
        variant: FftVariantId,
        cfg: &FftLearnConfig,
        device: Device,
        train_steps: usize,
        seed: u64,
    ) -> Result<()> {
        match variant {
            FftVariantId::ButterflyQ8 => {
                self.q8 = Some(Q8Twiddles::from_f32(&self.twiddles));
            }
            FftVariantId::ButterflyUnitary => {
                if cfg.n_fft <= 64 && cfg.batch <= 8 && train_steps > 0 {
                    let (w, _) = train_unitary_quick(cfg, train_steps.min(25), 1e-3, seed)?;
                    self.unitary = Some(w);
                } else {
                    self.unitary = Some(UnitaryWeights::exact_init(cfg));
                }
            }
            FftVariantId::DomainTwiddle => {
                let (tw, _) = train_domain_twiddles(cfg, train_steps, 5e-4, seed)?;
                self.twiddles = tw;
            }
            FftVariantId::ButterflyCompiled if self.compiled_butterfly.is_none() => {
                self.compiled_butterfly = Some(compile_butterfly(cfg, device, &self.twiddles)?);
            }
            FftVariantId::StockhamCompiled if self.compiled_stockham.is_none() => {
                self.compiled_stockham = Some(compile_stockham(cfg, device, &self.twiddles)?);
            }
            FftVariantId::FusedSpectralCompiled if self.compiled_fused.is_none() => {
                self.compiled_fused = Some(compile_fused(cfg, device, &self.mask)?);
            }
            FftVariantId::RlxOpFft if self.compiled_rlx.is_none() => {
                self.compiled_rlx = Some(crate::rlx_fft::compile_rlx_fft(
                    cfg,
                    crate::config::TransformDir::Forward,
                    device,
                )?);
            }
            FftVariantId::RlxOpIfft if self.compiled_rlx_inv.is_none() => {
                self.compiled_rlx_inv = Some(crate::rlx_fft::compile_rlx_fft(
                    cfg,
                    crate::config::TransformDir::Inverse,
                    device,
                )?);
            }
            FftVariantId::WelchRlxOpFft if self.compiled_welch_rlx.is_none() => {
                let params = WelchParams::for_n_fft(cfg.n_fft);
                self.compiled_welch_rlx = Some(compile_welch_rlx_fft(cfg.batch, params, device)?);
            }
            FftVariantId::WelchButterflyCompiled if self.compiled_welch_butterfly.is_none() => {
                let params = WelchParams::for_n_fft(cfg.n_fft);
                let welch_cfg = FftLearnConfig::new(cfg.n_fft, cfg.batch * params.n_segments)?;
                self.compiled_welch_butterfly =
                    Some(compile_butterfly(&welch_cfg, device, &self.twiddles)?);
            }
            _ => {}
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        variant: FftVariantId,
        signal: &[f32],
        cfg: &FftLearnConfig,
    ) -> Result<Vec<f32>> {
        let n = cfg.n_fft;
        let batch = cfg.batch;
        match variant {
            FftVariantId::Rustfft => fft_real_batch(signal, batch, n),
            FftVariantId::RlxOpFft => {
                let exec = self.compiled_rlx.as_mut().expect("rlx compiled");
                Ok(crate::rlx_fft::rlx_fft_forward(exec, signal, batch, n))
            }
            FftVariantId::ButterflyEager | FftVariantId::DomainTwiddle => {
                butterfly_forward_real_batch(signal, &self.twiddles, batch, n)
            }
            FftVariantId::ButterflyCompiled => {
                let exec = self
                    .compiled_butterfly
                    .as_mut()
                    .expect("butterfly compiled");
                Ok(exec.run(&[("signal", signal)]).remove(0))
            }
            FftVariantId::StockhamEager => {
                stockham_forward_real_batch(signal, &self.twiddles, batch, n)
            }
            FftVariantId::StockhamCompiled => {
                let exec = self.compiled_stockham.as_mut().expect("stockham compiled");
                Ok(exec.run(&[("signal", signal)]).remove(0))
            }
            FftVariantId::FusedSpectralEager => {
                fused_spectral_eager(signal, &self.twiddles, &self.mask, batch, n)
            }
            FftVariantId::FusedSpectralCompiled => {
                let exec = self.compiled_fused.as_mut().expect("fused compiled");
                Ok(exec.run(&[("signal", signal)]).remove(0))
            }
            FftVariantId::ButterflyQ8 => self
                .q8
                .as_ref()
                .expect("q8")
                .forward_real_batch(signal, batch, n),
            FftVariantId::ButterflyUnitary => self
                .unitary
                .as_ref()
                .expect("unitary")
                .forward_real_batch(signal, batch, n),
            FftVariantId::RlxOpIfft => bail!("rlx_op_ifft is inverse-only; call inverse()"),
            FftVariantId::WelchRustfft
            | FftVariantId::WelchRlxOpFft
            | FftVariantId::WelchButterflyEager
            | FftVariantId::WelchButterflyCompiled => {
                bail!("{} is welch-only; call welch()", variant.label())
            }
        }
    }

    pub fn welch(
        &mut self,
        variant: FftVariantId,
        signal: &[f32],
        cfg: &FftLearnConfig,
    ) -> Result<Vec<f32>> {
        let params = WelchParams::for_n_fft(cfg.n_fft);
        let batch = cfg.batch;
        match variant {
            FftVariantId::WelchRustfft => welch_rustfft(signal, batch, params),
            FftVariantId::WelchRlxOpFft => {
                let exec = self
                    .compiled_welch_rlx
                    .as_mut()
                    .expect("welch rlx compiled");
                welch_rlx_op_fft(exec, signal, batch, params)
            }
            FftVariantId::WelchButterflyEager => {
                welch_butterfly(signal, &self.twiddles, batch, params)
            }
            FftVariantId::WelchButterflyCompiled => {
                let window = crate::welch::hann_window(params.n_fft);
                let segs = crate::welch::welch_windowed_segments(signal, batch, params, &window)?;
                let exec = self
                    .compiled_welch_butterfly
                    .as_mut()
                    .expect("welch butterfly compiled");
                let spec = exec.run(&[("signal", &segs)]).remove(0);
                Ok(crate::welch::average_welch_psd(
                    &spec,
                    batch,
                    params.n_segments,
                    params.n_fft,
                ))
            }
            other => bail!("variant {} has no welch path", other.label()),
        }
    }

    pub fn inverse(&mut self, variant: FftVariantId, cfg: &FftLearnConfig) -> Result<Vec<f32>> {
        let n = cfg.n_fft;
        let batch = cfg.batch;
        match variant {
            FftVariantId::Rustfft => {
                crate::reference::ifft_complex_batch(&self.inverse_spectrum, batch, n)
            }
            FftVariantId::RlxOpIfft => {
                let exec = self.compiled_rlx_inv.as_mut().expect("rlx inv compiled");
                Ok(crate::rlx_fft::rlx_fft_inverse_block(
                    exec,
                    &self.spectrum_block,
                    batch,
                    n,
                ))
            }
            FftVariantId::ButterflyEager => crate::butterfly::butterfly_inverse_complex_batch(
                &self.inverse_spectrum,
                &self.twiddles,
                batch,
                n,
            ),
            other => bail!("variant {} has no inverse path", other.label()),
        }
    }
}

fn compile_butterfly(
    cfg: &FftLearnConfig,
    device: Device,
    twiddles: &[f32],
) -> Result<CompiledGraph> {
    use crate::weights::WeightStore;
    let built = build_butterfly_forward_graph(cfg)?;
    let store = WeightStore::from_twiddles(twiddles, cfg.n_fft);
    let mut compiled = crate::compile::try_compile_graph(device, built.graph)?;
    store.apply_butterfly(&mut compiled, cfg.batch, cfg.n_fft);
    Ok(compiled)
}

fn compile_stockham(
    cfg: &FftLearnConfig,
    device: Device,
    twiddles: &[f32],
) -> Result<CompiledGraph> {
    use crate::weights::WeightStore;
    let (graph, _names) = build_stockham_forward_graph(cfg)?;
    let store = WeightStore::from_twiddles(twiddles, cfg.n_fft);
    let mut compiled = crate::compile::try_compile_graph(device, graph)?;
    store.apply_butterfly(&mut compiled, cfg.batch, cfg.n_fft);
    Ok(compiled)
}

fn compile_fused(cfg: &FftLearnConfig, device: Device, mask: &[f32]) -> Result<CompiledGraph> {
    let (graph, names) = build_fused_spectral_graph(cfg)?;
    let mut compiled = crate::compile::try_compile_graph(device, graph)?;
    for (i, name) in names.iter().enumerate() {
        compiled.set_param(name, &[mask[i]]);
    }
    Ok(compiled)
}

pub fn variants_for_direction(with_compiled: bool, forward: bool) -> Vec<FftVariantId> {
    let all = if with_compiled {
        FftVariantId::all()
    } else {
        FftVariantId::all_eager()
    };
    all.iter()
        .copied()
        .filter(|v| {
            if v.is_welch() {
                return false;
            }
            if forward {
                !matches!(v, FftVariantId::RlxOpIfft)
            } else {
                v.supports_inverse()
            }
        })
        .collect()
}

pub fn bench_variant_ms(
    state: &mut VariantState,
    variant: FftVariantId,
    cfg: &FftLearnConfig,
    signal: &[f32],
    iters: usize,
) -> Result<f64> {
    let _ = state.forward(variant, signal, cfg)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        state.forward(variant, signal, cfg)?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

pub fn variants_for_welch(with_compiled: bool) -> Vec<FftVariantId> {
    FftVariantId::welch_variants(with_compiled).to_vec()
}

pub fn bench_variant_ms_welch(
    state: &mut VariantState,
    variant: FftVariantId,
    cfg: &FftLearnConfig,
    signal: &[f32],
    iters: usize,
) -> Result<f64> {
    let _ = state.welch(variant, signal, cfg)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        state.welch(variant, signal, cfg)?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

pub fn variant_welch_error(
    state: &mut VariantState,
    variant: FftVariantId,
    cfg: &FftLearnConfig,
    signal: &[f32],
) -> Result<f32> {
    if matches!(variant, FftVariantId::WelchRustfft) {
        return Ok(0.0);
    }
    let pred = state.welch(variant, signal, cfg)?;
    let params = WelchParams::for_n_fft(cfg.n_fft);
    let target = welch_rustfft(signal, cfg.batch, params)?;
    Ok(max_abs_error(&pred, &target))
}

pub fn fixed_ablation_welch_signal(seed: u64, batch: usize, n_fft: usize) -> Vec<f32> {
    let params = WelchParams::for_n_fft(n_fft);
    let frame = params.frame_len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    random_batch(&mut rng, batch, frame)
}

pub fn bench_variant_ms_inverse(
    state: &mut VariantState,
    variant: FftVariantId,
    cfg: &FftLearnConfig,
    iters: usize,
) -> Result<f64> {
    let _ = state.inverse(variant, cfg)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        state.inverse(variant, cfg)?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

pub fn variant_spectrum_error(
    state: &mut VariantState,
    variant: FftVariantId,
    cfg: &FftLearnConfig,
    signal: &[f32],
) -> Result<f32> {
    if matches!(
        variant,
        FftVariantId::FusedSpectralEager | FftVariantId::FusedSpectralCompiled
    ) {
        return Ok(0.0);
    }
    let pred = state.forward(variant, signal, cfg)?;
    let target = fft_real_batch(signal, cfg.batch, cfg.n_fft)?;
    Ok(max_abs_error(&pred, &target))
}

pub fn variant_inverse_error(
    state: &mut VariantState,
    variant: FftVariantId,
    cfg: &FftLearnConfig,
) -> Result<f32> {
    let pred = state.inverse(variant, cfg)?;
    let target =
        crate::reference::ifft_complex_batch(&state.inverse_spectrum, cfg.batch, cfg.n_fft)?;
    Ok(max_abs_error(&pred, &target))
}

pub fn fixed_ablation_signal(seed: u64, batch: usize, n_fft: usize) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    random_batch(&mut rng, batch, n_fft)
}

pub fn fixed_ablation_spectrum(seed: u64, batch: usize, n_fft: usize) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    crate::train::random_complex_batch(&mut rng, batch, n_fft)
}

pub fn ensure_variant_ready(variant: FftVariantId, device: Device) -> Result<()> {
    if matches!(
        variant,
        FftVariantId::ButterflyCompiled
            | FftVariantId::StockhamCompiled
            | FftVariantId::FusedSpectralCompiled
            | FftVariantId::RlxOpFft
            | FftVariantId::RlxOpIfft
            | FftVariantId::WelchRlxOpFft
            | FftVariantId::WelchButterflyCompiled
    ) && device == Device::Cpu
    {
        return Ok(());
    }
    if matches!(variant, FftVariantId::Rustfft) {
        return Ok(());
    }
    Ok(())
}

pub fn skip_on_device(variant: FftVariantId, device: Device) -> bool {
    let _ = (variant, device);
    false
}

pub fn validate_variant_output(
    variant: FftVariantId,
    pred_len: usize,
    cfg: &FftLearnConfig,
) -> Result<()> {
    let expected = cfg.batch * cfg.n_fft * 2;
    if pred_len != expected {
        bail!(
            "variant {} output len {pred_len} != expected {expected}",
            variant.label()
        );
    }
    Ok(())
}
