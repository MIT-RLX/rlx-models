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

//! Warm forecast latency: best and median of `RUNS` (default 10) calls with the
//! default options after one warm-up, for the README's speed table.
//!
//! ```text
//! cargo run --release -p rlx-timestral --example bench -- DIR series.csv [HORIZON]
//! ```
//!
//! `series.csv` is one value per line. `RLX_TIMESTRAL_DEVICE` picks the backend.

use anyhow::{Context, Result};
use ndarray::Array3;
use rlx_runtime::Device;
use rlx_timestral::{ForecastOptions, TimestralForecaster};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(args.first().context("checkpoint dir")?);
    let series: Vec<f32> = std::fs::read_to_string(args.get(1).context("series.csv")?)?
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let horizon: usize = args.get(2).map_or(Ok(96), |h| h.parse())?;
    let runs: usize = std::env::var("RUNS").map_or(Ok(10), |r| r.parse())?;
    let device = rlx_ir::env::var_result("RLX_TIMESTRAL_DEVICE")
        .map(|d| rlx_cli::parse_device(&d))
        .unwrap_or(Ok(Device::Cpu))?;
    let mut fc = TimestralForecaster::load(dir, device)?;
    let ctx = Array3::from_shape_vec((1, 1, series.len()), series.clone())?;
    let opts = ForecastOptions::default();
    let t = std::time::Instant::now();
    fc.forecast(ctx.view(), horizon, &opts)?;
    let first = t.elapsed().as_secs_f64() * 1e3;
    let mut ms: Vec<f64> = (0..runs)
        .map(|_| {
            let t = std::time::Instant::now();
            fc.forecast(ctx.view(), horizon, &opts)
                .map(|_| t.elapsed().as_secs_f64() * 1e3)
        })
        .collect::<Result<_>>()?;
    ms.sort_by(f64::total_cmp);
    println!(
        "{:?}: context {}, horizon {horizon}: first {first:.1} ms, warm best {:.1} ms, median {:.1} ms",
        device,
        series.len(),
        ms[0],
        ms[ms.len() / 2]
    );
    Ok(())
}
