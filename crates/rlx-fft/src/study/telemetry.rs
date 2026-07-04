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

//! Training telemetry for study reports: loss curves, param counts, activation heatmaps, loss landscape.

use crate::model::config::FftLearnConfig;
use crate::model::pruned::init_gates;
use crate::model::twiddle::exact_twiddles;
use crate::model::unitary::UnitaryWeights;
use crate::model::variants::FftVariantId;
use serde::{Deserialize, Serialize};

pub const F32_BYTES: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamBreakdown {
    pub twiddles: usize,
    pub gates: usize,
    pub freq_mask: usize,
    pub denoiser: usize,
    pub unitary: usize,
    pub mel_filters: usize,
    pub q8_packed: usize,
    pub total_params: usize,
    pub memory_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossPoint {
    pub step: usize,
    pub total_loss: f32,
    pub mel_err: f32,
    pub spec_err: f32,
    pub welch_err: f32,
    pub mean_gate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossLandscape3D {
    /// Grid over twiddle[0] (real) vs twiddle[1] (imag) with other weights fixed.
    pub x_label: String,
    pub y_label: String,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationHeatmap {
    /// Row = FFT stage, col = butterfly index; values in [0,1] (gate activation).
    pub stages: usize,
    pub butterflies: usize,
    pub gates: Vec<f32>,
    /// Optional per-bin frequency mask (n_fft complex bins).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub freq_mask: Vec<f32>,
    /// Twiddle magnitude per stage×butterfly (|w|).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub twiddle_mag: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingTrace {
    pub model_id: String,
    pub variant: String,
    pub n_fft: usize,
    pub batch: usize,
    pub train_steps: usize,
    pub params: ParamBreakdown,
    pub loss_curve: Vec<LossPoint>,
    pub heatmap: ActivationHeatmap,
    pub landscape: Option<LossLandscape3D>,
    pub final_mel_err: f32,
    pub final_spec_err: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StudyTelemetryBundle {
    pub models: Vec<ModelTrainingTrace>,
}

pub fn variant_param_breakdown(variant: FftVariantId, cfg: &FftLearnConfig) -> ParamBreakdown {
    let stages = cfg.num_stages();
    let half = cfg.n_fft / 2;
    let tw = stages * half * 2;
    let (unitary, gates, freq_mask, denoiser, mel_filters, q8) = match variant {
        FftVariantId::Rustfft | FftVariantId::RlxOpFft | FftVariantId::RlxOpIfft => {
            (0, 0, 0, 0, 0, 0)
        }
        FftVariantId::ButterflyUnitary => (UnitaryWeights::param_count(cfg.n_fft), 0, 0, 0, 0, 0),
        FftVariantId::ButterflyQ8 => (tw, 0, 0, 0, 0, tw / 2),
        FftVariantId::WelchRustfft
        | FftVariantId::WelchRlxOpFft
        | FftVariantId::WelchButterflyEager
        | FftVariantId::WelchButterflyCompiled => (tw, 0, 0, 0, 0, 0),
        _ => (tw, 0, 0, 0, 0, 0),
    };
    let total = tw + unitary + gates + freq_mask + denoiser + mel_filters + q8;
    ParamBreakdown {
        twiddles: tw,
        gates,
        freq_mask,
        denoiser,
        unitary,
        mel_filters,
        q8_packed: q8,
        total_params: total,
        memory_bytes: total * F32_BYTES,
    }
}

pub fn learned_model_param_breakdown(n_fft: usize, n_mels: usize) -> ParamBreakdown {
    let cfg = FftLearnConfig::new(n_fft, 1).expect("n_fft");
    let tw = exact_twiddles(&cfg).len();
    let gates = init_gates(n_fft).len();
    let fm = n_fft * 2;
    let dn = n_fft * 2 * 2;
    let mf = n_mels * (n_fft / 2 + 1);
    let total = tw + gates + fm + dn + mf;
    ParamBreakdown {
        twiddles: tw,
        gates,
        freq_mask: fm,
        denoiser: dn,
        unitary: 0,
        mel_filters: mf,
        q8_packed: 0,
        total_params: total,
        memory_bytes: total * F32_BYTES,
    }
}

pub fn gate_heatmap_from_vec(gates: &[f32], n_fft: usize) -> ActivationHeatmap {
    let stages = crate::model::butterfly::num_stages(n_fft);
    let half = n_fft / 2;
    let mut tw_mag = vec![0f32; stages * half];
    let tw = exact_twiddles(&FftLearnConfig::new(n_fft, 1).unwrap());
    for s in 0..stages {
        for b in 0..half {
            let w_base = crate::model::twiddle::twiddle_index(s, b, half, 0);
            let re = tw[w_base];
            let im = tw[w_base + 1];
            tw_mag[s * half + b] = (re * re + im * im).sqrt();
        }
    }
    ActivationHeatmap {
        stages,
        butterflies: half,
        gates: gates.to_vec(),
        freq_mask: Vec::new(),
        twiddle_mag: tw_mag,
    }
}
