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

//! Parity against the upstream `timestral` package on released weights.
//!
//! Needs a checkpoint and a reference file from `scripts/timestral_reference_dump.py`:
//!
//! ```bash
//! RLX_TIMESTRAL_DIR=/path/Timestral-5M RLX_TIMESTRAL_REF=/path/ref-5M.safetensors \
//!   cargo test -p rlx-timestral --release --test reference_parity -- --nocapture
//! ```
//!
//! `RLX_TIMESTRAL_DEVICE` picks the backend (default `cpu`). With
//! `RLX_TIMESTRAL_REF64` (a `--dtype float64` reference) the f64 number format is
//! checked against upstream run in float64 as well.

use std::path::PathBuf;

use ndarray::{Array3, ArrayD, IxDyn};
use rlx_core::weight_map::WeightMap;
use rlx_timestral::{ForecastOptions, TimestralForecaster};

/// Relative to the largest reference magnitude of each tensor.
const TOL: f32 = 2e-3;
/// Relative to the reference's range. One f32 step at a level of 1e6 is already ~1.5e-3
/// of the `level` case's range, so this cannot be tighter.
const RANGE_TOL: f32 = 1e-2;

struct Reference(WeightMap);

impl Reference {
    fn get(&self, key: &str) -> Option<ArrayD<f32>> {
        let (data, shape) = self.0.get(key)?;
        Some(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
    }

    fn get3(&self, key: &str) -> Option<Array3<f32>> {
        self.get(key).map(|a| a.into_dimensionality().unwrap())
    }
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(
            0f32,
            |a, d| if d.is_nan() { f32::INFINITY } else { a.max(d) },
        )
        / scale
}

/// Relative to the reference's range, so a large constant level cannot hide an error.
fn spread_err(got: &[f32], want: &[f32]) -> f32 {
    let lo = want.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = want.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    rel_err(got, want) * scale / (hi - lo).max(1e-6)
}

fn setup() -> Option<(PathBuf, Reference)> {
    let dir = rlx_ir::env::var_os("RLX_TIMESTRAL_DIR")?;
    let refs = rlx_ir::env::var_os("RLX_TIMESTRAL_REF")?;
    let map = WeightMap::from_file(refs.to_str().unwrap()).expect("load reference file");
    Some((dir.into(), Reference(map)))
}

fn device() -> rlx_runtime::Device {
    let name = rlx_ir::env::var_result("RLX_TIMESTRAL_DEVICE").unwrap_or_else(|_| "cpu".into());
    rlx_cli::parse_device(&name).expect("RLX_TIMESTRAL_DEVICE")
}

#[test]
fn forecasts_match_upstream() {
    let Some((dir, refs)) = setup() else {
        eprintln!("skip: set RLX_TIMESTRAL_DIR and RLX_TIMESTRAL_REF");
        return;
    };
    let mut fc = TimestralForecaster::load(&dir, device()).expect("load checkpoint");
    let mut failures = Vec::new();
    eprintln!(
        "{:<8} {:>12} {:>12} {:>12}",
        "case", "raw rel", "full rel", "full/range"
    );
    for case in ["uni", "long", "multi", "cov", "level", "nonneg"] {
        let context = refs.get3(&format!("{case}.context")).unwrap();
        let want_raw = refs.get(&format!("{case}.raw")).unwrap();
        let want_full = refs.get(&format!("{case}.full")).unwrap();
        let horizon = want_raw.shape()[2];

        let opts = ForecastOptions {
            past_covariates: refs.get3(&format!("{case}.past_cov")),
            known_covariates: refs.get3(&format!("{case}.known_cov")),
            categorical_known_covariates: refs.get3(&format!("{case}.cat_known")),
            ..ForecastOptions::default()
        };
        let raw_opts = ForecastOptions {
            symmetric_averaging: false,
            clip_nonnegative: false,
            ..opts.clone()
        };

        let raw = fc.forecast(context.view(), horizon, &raw_opts).unwrap();
        let full = fc.forecast(context.view(), horizon, &opts).unwrap();
        assert_eq!(raw.quantiles.shape(), want_raw.shape(), "{case}");
        let e_raw = rel_err(
            raw.quantiles.as_slice().unwrap(),
            want_raw.as_slice().unwrap(),
        );
        let e_full = rel_err(
            full.quantiles.as_slice().unwrap(),
            want_full.as_slice().unwrap(),
        );
        let e_range = spread_err(
            full.quantiles.as_slice().unwrap(),
            want_full.as_slice().unwrap(),
        );
        eprintln!("{case:<8} {e_raw:>12.3e} {e_full:>12.3e} {e_range:>12.3e}");
        if !(e_raw < TOL && e_full < TOL && e_range < RANGE_TOL) {
            failures.push(format!(
                "{case}: raw {e_raw:.3e}, full {e_full:.3e}, full/range {e_range:.3e}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "differs from upstream:\n{}",
        failures.join("\n")
    );
}

/// The transformer output of the first block, with bucket padding off so the
/// sequence matches upstream token for token.
#[test]
fn hidden_state_matches_upstream() {
    let Some((dir, refs)) = setup() else {
        eprintln!("skip: set RLX_TIMESTRAL_DIR and RLX_TIMESTRAL_REF");
        return;
    };
    let mut fc = TimestralForecaster::load(&dir, device()).expect("load checkpoint");
    fc.model_mut().bucketing = false;
    fc.model_mut().capture_hidden = true;
    let context = refs.get3("uni.context").unwrap();
    let want = refs.get("uni.hidden").unwrap();
    let horizon = refs.get("uni.raw").unwrap().shape()[2];
    fc.forecast(context.view(), horizon, &ForecastOptions::raw())
        .unwrap();
    let got = fc.model().last_hidden().expect("hidden captured");
    let e = rel_err(got, want.as_slice().unwrap());
    eprintln!("uni hidden: rel {e:.3e} over {} values", got.len());
    assert!(e < TOL, "hidden state differs from upstream by {e:.3e}");
}

/// Forecasts in f32 and under `--number-format f64` against upstream run in
/// float64 (`scripts/timestral_reference_dump.py --dtype float64`, path in
/// `RLX_TIMESTRAL_REF64`), on `RLX_TIMESTRAL_DEVICE` (GPUs run f64 in
/// double-single).
///
/// The f64 run keeps host math and the network in f64; what is left is f32
/// rounding where the network's inputs and outputs and the [`Forecast`] API
/// are f32 (3e-8 to 1e-7, 7e-7 on the 1500-step horizon, whose fed-back
/// medians pass that boundary every block). f32 is at 5e-7 to 1.2e-5.
///
/// [`Forecast`]: rlx_timestral::Forecast
#[test]
fn f64_forecasts_match_upstream_float64() {
    const F64_TOL: f64 = 1e-6;
    let (Some(dir), Some(path)) = (
        rlx_ir::env::var_os("RLX_TIMESTRAL_DIR"),
        rlx_ir::env::var_os("RLX_TIMESTRAL_REF64"),
    ) else {
        eprintln!("skip: set RLX_TIMESTRAL_DIR and RLX_TIMESTRAL_REF64");
        return;
    };
    let refs = Reference(WeightMap::from_file(path.to_str().unwrap()).expect("reference"));
    let dir = PathBuf::from(dir);
    let mut f32_fc = TimestralForecaster::load(&dir, device()).unwrap();
    let mut f64_fc = TimestralForecaster::load(&dir, device()).unwrap();
    f64_fc
        .model_mut()
        .set_number_format(Some(rlx_runtime::NumberFormat::F64))
        .unwrap();
    let mut failures = Vec::new();
    eprintln!(
        "{:<8} {:>8} {:>12} {:>12}",
        "case", "", "f32 rel", "f64 rel"
    );
    for case in ["uni", "long", "multi", "cov", "level", "nonneg"] {
        let context = refs.get3(&format!("{case}.context")).unwrap();
        let opts = ForecastOptions {
            past_covariates: refs.get3(&format!("{case}.past_cov")),
            known_covariates: refs.get3(&format!("{case}.known_cov")),
            categorical_known_covariates: refs.get3(&format!("{case}.cat_known")),
            ..ForecastOptions::default()
        };
        let raw_opts = ForecastOptions {
            symmetric_averaging: false,
            clip_nonnegative: false,
            ..opts.clone()
        };
        for (kind, o) in [("raw", &raw_opts), ("full", &opts)] {
            let hi = refs.get(&format!("{case}.{kind}")).unwrap();
            let lo = refs
                .get(&format!("{case}.{kind}.lo"))
                .expect("a --dtype float64 reference");
            let want: Vec<f64> = hi
                .iter()
                .zip(lo.iter())
                .map(|(&h, &l)| h as f64 + l as f64)
                .collect();
            let horizon = hi.shape()[2];
            let mut errs = [0f64; 2];
            for (e, fc) in errs.iter_mut().zip([&mut f32_fc, &mut f64_fc]) {
                let got = fc.forecast(context.view(), horizon, o).unwrap().quantiles;
                *e = rel_err_f64(got.as_slice().unwrap(), &want);
            }
            eprintln!("{case:<8} {kind:>8} {:>12.3e} {:>12.3e}", errs[0], errs[1]);
            if !(errs[1] < F64_TOL && errs[1] <= errs[0]) {
                failures.push(format!(
                    "{case} {kind}: f32 {:.3e}, f64 {:.3e}",
                    errs[0], errs[1]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "f64 not within {F64_TOL:e} of upstream float64, or worse than f32:\n{}",
        failures.join("\n")
    );
}

fn rel_err_f64(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-12);
    got.iter()
        .zip(want)
        .map(|(&a, b)| (a as f64 - b).abs())
        .fold(
            0f64,
            |a, d| if d.is_nan() { f64::INFINITY } else { a.max(d) },
        )
        / scale
}
