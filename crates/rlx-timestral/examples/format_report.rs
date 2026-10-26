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

//! How far each number format moves the forecast from f32.
//!
//! Forecasts the cases of a `scripts/timestral_reference_dump.py` file (only its
//! inputs are used) in f32 and in each format, and reports per case the median
//! and largest |forecast − f32 forecast| in units of the f32 forecast's 5–95%
//! band width at that step, over all steps and quantiles.
//!
//! ```text
//! cargo run --release -p rlx-timestral --example format_report -- \
//!     .cache/timestral/5M .cache/timestral/ref-5M.safetensors bf16 f16 f8 f6 f4
//! ```
//!
//! With a CSV instead of a reference file, it reports accuracy against the data:
//! the weighted quantile loss (`rlx_timestral::train::rolling_wql`) of forecasting
//! `HORIZON` (default 96) steps from up to `CONTEXT` (default 512) at `WINDOWS`
//! (default 20) origins per column, in f32 and each format.
//!
//! ```text
//! cargo run --release -p rlx-timestral --example format_report -- \
//!     .cache/timestral/5M ETTh1-values.csv bf16 f8 f4
//! ```
//!
//! `RLX_NUMBER_FORMAT_KEEP` applies, so op kinds can be bisected
//! (`RLX_NUMBER_FORMAT_KEEP='!MatMul'` rounds matmul outputs only).

use anyhow::{Context, Result};
use ndarray::{Array3, Array4, Axis};
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{Device, NumberFormat};
use rlx_timestral::{ForecastOptions, TimestralForecaster};

const CASES: [&str; 6] = ["uni", "long", "multi", "cov", "level", "nonneg"];

fn get3(refs: &WeightMap, key: &str) -> Option<Array3<f32>> {
    let (data, shape) = refs.get(key)?;
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data.to_vec()).ok()
}

fn forecasts(fc: &mut TimestralForecaster, refs: &WeightMap) -> Result<Vec<Array4<f32>>> {
    CASES
        .iter()
        .map(|case| {
            let context = get3(refs, &format!("{case}.context")).context("context")?;
            let horizon = refs.get(&format!("{case}.raw")).context("raw")?.1[2];
            let opts = ForecastOptions {
                past_covariates: get3(refs, &format!("{case}.past_cov")),
                known_covariates: get3(refs, &format!("{case}.known_cov")),
                categorical_known_covariates: get3(refs, &format!("{case}.cat_known")),
                ..ForecastOptions::raw()
            };
            Ok(fc.forecast(context.view(), horizon, &opts)?.quantiles)
        })
        .collect()
}

/// (median, max) of |got − base| / band, the band being base's q95 − q05 per step.
fn band_err(got: &Array4<f32>, base: &Array4<f32>, lo: usize, hi: usize) -> (f64, f64) {
    let mut errs = Vec::new();
    for (g, b) in got.lanes(Axis(3)).into_iter().zip(base.lanes(Axis(3))) {
        let band = ((b[hi] - b[lo]) as f64).abs().max(1e-12);
        errs.extend(
            g.iter()
                .zip(b)
                .map(|(x, y)| (*x as f64 - *y as f64).abs() / band),
        );
    }
    errs.sort_by(f64::total_cmp);
    (errs[errs.len() / 2], *errs.last().unwrap_or(&0.0))
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    rlx_ir::env::read(name).map_or(Ok(default), |v| v.parse().context(name.to_string()))
}

/// WQL against the data, in f32 and each format.
fn wql_report(fc: &mut TimestralForecaster, csv: &str, formats: &[String]) -> Result<()> {
    let series = rlx_timestral::cli::load_columns(&csv.into())?;
    let (context, horizon) = (env_usize("CONTEXT", 512)?, env_usize("HORIZON", 96)?);
    let windows = env_usize("WINDOWS", 20)?;
    let score = |fc: &mut TimestralForecaster| {
        rlx_timestral::train::rolling_wql(fc, &series, context, horizon, windows)
    };
    let base = score(fc)?;
    println!(
        "{} series x {windows} windows, horizon {horizon}",
        series.len()
    );
    println!("{:<8} {:>8} {:>9}", "format", "WQL", "vs f32");
    println!("{:<8} {base:>8.4} {:>9}", "f32", "");
    for name in formats {
        let format: NumberFormat = name.parse().map_err(anyhow::Error::msg)?;
        fc.model_mut().set_number_format(Some(format))?;
        let w = score(fc)?;
        println!("{name:<8} {w:>8.4} {:>+8.1}%", 100.0 * (w / base - 1.0));
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(args.first().context("checkpoint dir")?);
    let data = args.get(1).context("reference file or CSV")?;
    let device = rlx_ir::env::var_result("RLX_TIMESTRAL_DEVICE")
        .map(|d| rlx_cli::parse_device(&d))
        .unwrap_or(Ok(Device::Cpu))?;
    let mut fc = TimestralForecaster::load(dir, device)?;
    if data.ends_with(".csv") {
        return wql_report(&mut fc, data, &args[2..]);
    }
    let refs = WeightMap::from_file(data)?;
    let levels = fc.quantile_levels().to_vec();
    let at = |q: f32| levels.iter().position(|&l| (l - q).abs() < 1e-6);
    let (lo, hi) = (at(0.05).unwrap_or(0), at(0.95).unwrap_or(levels.len() - 1));
    let base = forecasts(&mut fc, &refs)?;

    print!("{:<8}", "format");
    for case in CASES {
        print!(" {case:>15}");
    }
    println!("   (median / max, in 5-95% band widths)");
    for name in &args[2..] {
        let format: NumberFormat = name.parse().map_err(anyhow::Error::msg)?;
        fc.model_mut().set_number_format(Some(format))?;
        let got = forecasts(&mut fc, &refs)?;
        print!("{name:<8}");
        for (g, b) in got.iter().zip(&base) {
            let (med, max) = band_err(g, b, lo, hi);
            print!(" {:>15}", format!("{med:.1e}/{max:.1e}"));
        }
        println!();
    }
    Ok(())
}
