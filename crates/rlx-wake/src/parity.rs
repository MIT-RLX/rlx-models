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

//! Exact cross-backend score parity (RLX backends only — no ONNX).

use anyhow::{Result, bail};
use rlx_runtime::Device;

use crate::device::{available_devices, bench_device_label, ensure_backend_ready};
use crate::{WakeEngine, WakeStep, score_wav};

/// Exact equality of score trajectories (bit-identical f32).
pub fn scores_exact_match(a: &[WakeStep], b: &[WakeStep]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x.score.to_bits() == y.score.to_bits() && x.fired == y.fired)
}

/// Max |Δscore| between two trajectories (len mismatch → +inf).
pub fn max_abs_score_delta(a: &[WakeStep], b: &[WakeStep]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.score - y.score).abs())
        .fold(0.0_f32, f32::max)
}

/// Fraction of matching scores (1.0 = 100% parity).
pub fn score_parity_fraction(a: &[WakeStep], b: &[WakeStep]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let matched = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.score.to_bits() == y.score.to_bits())
        .count();
    matched as f32 / a.len() as f32
}

#[derive(Debug, Clone)]
pub struct BackendParityRow {
    pub device: &'static str,
    pub steps: usize,
    pub parity: f32,
    pub max_abs: f32,
    pub exact: bool,
}

/// Score `pcm` on CPU, then on every available RLX backend using `make_engine`.
///
/// Engines must use the shared `rlx-cpu` BLAS numerical path so results are
/// bit-identical across device slots (wake micro-batches; same policy as VAD).
pub fn run_backend_parity<E, F>(pcm: &[f32], mut make_engine: F) -> Result<Vec<BackendParityRow>>
where
    E: WakeEngine,
    F: FnMut(Device) -> Result<E>,
{
    ensure_backend_ready(Device::Cpu)?;
    let mut cpu_eng = make_engine(Device::Cpu)?;
    let cpu_steps = score_wav(&mut cpu_eng, pcm)?;

    let mut rows = Vec::new();
    for device in available_devices() {
        ensure_backend_ready(device)?;
        let mut eng = make_engine(device)?;
        let steps = score_wav(&mut eng, pcm)?;
        let exact = scores_exact_match(&cpu_steps, &steps);
        let parity = score_parity_fraction(&cpu_steps, &steps);
        let max_abs = max_abs_score_delta(&cpu_steps, &steps);
        rows.push(BackendParityRow {
            device: bench_device_label(device),
            steps: steps.len(),
            parity,
            max_abs,
            exact,
        });
    }
    Ok(rows)
}

/// Require every backend row to be exact (100% parity).
pub fn assert_100_percent_parity(rows: &[BackendParityRow]) -> Result<()> {
    let mut bad = Vec::new();
    for r in rows {
        if !r.exact || r.parity < 1.0 {
            bad.push(format!(
                "{}: parity={:.4} max_abs={:.3e} exact={}",
                r.device, r.parity, r.max_abs, r.exact
            ));
        }
    }
    if !bad.is_empty() {
        bail!("wake backend parity below 100%:\n  {}", bad.join("\n  "));
    }
    Ok(())
}
