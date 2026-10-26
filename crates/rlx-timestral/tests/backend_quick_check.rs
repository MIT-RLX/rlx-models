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

//! Synthetic-weight checks: bucket padding is exact, and every backend forecasts the
//! same numbers as CPU (not merely finite ones).

use ndarray::Array3;
use rlx_runtime::Device;
use rlx_timestral::{
    ForecastOptions, TimestralConfig, TimestralForecaster, TimestralModel, available_devices,
    device_label,
};

fn forecaster(device: Device, bucketing: bool) -> TimestralForecaster {
    let mut model = TimestralModel::synth(TimestralConfig::synth_tiny(), 11, device);
    model.bucketing = bucketing;
    TimestralForecaster::new(model)
}

/// Two series × two variates with gaps and a late start, a past and a known
/// covariate, and a horizon spanning several decode blocks.
fn inputs() -> (Array3<f32>, ForecastOptions, usize) {
    let t = 77;
    let horizon = 45;
    let ctx = Array3::from_shape_fn((2, 2, t), |(b, v, i)| {
        if (b == 0 && v == 1 && i < 20) || (b == 1 && (30..36).contains(&i)) {
            f32::NAN
        } else {
            ((i as f32 + 3.0 * b as f32) * 0.3 + v as f32).sin() * (2.0 + b as f32)
                + 0.05 * i as f32
        }
    });
    let opts = ForecastOptions {
        past_covariates: Some(Array3::from_shape_fn((2, 1, t), |(b, _, i)| {
            (i as f32 * 0.7 + b as f32).cos()
        })),
        known_covariates: Some(Array3::from_shape_fn((2, 1, t + horizon), |(_, _, i)| {
            (i % 7 >= 5) as u8 as f32
        })),
        ..ForecastOptions::default()
    };
    (ctx, opts, horizon)
}

fn run(fc: &mut TimestralForecaster, covariates: bool) -> Vec<f32> {
    let (ctx, opts, horizon) = inputs();
    let opts = if covariates {
        opts
    } else {
        ForecastOptions::default()
    };
    let out = fc.forecast(ctx.view(), horizon, &opts).expect("forecast");
    assert_eq!(out.quantiles.shape(), &[2, 2, horizon, 5]);
    let q = out.quantiles.as_slice().unwrap().to_vec();
    assert!(q.iter().all(|v| v.is_finite()), "non-finite forecast");
    for row in q.chunks(5) {
        assert!(
            row.windows(2).all(|w| w[0] <= w[1]),
            "quantiles cross: {row:?}"
        );
    }
    q
}

#[test]
fn bucket_padding_is_exact() {
    for covariates in [false, true] {
        let a = run(&mut forecaster(Device::Cpu, false), covariates);
        let b = run(&mut forecaster(Device::Cpu, true), covariates);
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            diff <= 1e-5 * scale,
            "covariates={covariates}: padded forecast differs by {diff} (scale {scale})"
        );
    }
}

#[test]
fn all_available_backends_forecast_like_cpu() {
    for covariates in [false, true] {
        let cpu = run(&mut forecaster(Device::Cpu, true), covariates);
        assert!(
            cpu.iter().any(|v| v.abs() > 1e-6),
            "CPU forecast is all zero"
        );
        let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs()));
        let mut bad = Vec::new();
        for dev in available_devices()
            .into_iter()
            .filter(|&d| d != Device::Cpu)
        {
            let got = run(&mut forecaster(dev, true), covariates);
            let rel = got
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
                / scale;
            eprintln!(
                "covariates={covariates} {}: rel {rel:.3e}",
                device_label(dev)
            );
            if !rel.is_finite() || rel >= 2e-3 {
                bad.push(format!("{}: {rel:.3e}", device_label(dev)));
            }
        }
        assert!(
            bad.is_empty(),
            "covariates={covariates}, differs from CPU: {bad:?}"
        );
    }
}

#[test]
fn number_format_option_changes_compute() {
    let run = |fmt: Option<&str>| {
        let mut fc = forecaster(Device::Cpu, true);
        fc.model_mut()
            .set_number_format(fmt.map(|f| f.parse().unwrap()))
            .unwrap();
        run(&mut fc, false)
    };
    let default = run(None);
    assert_eq!(
        run(Some("f32")),
        default,
        "explicit f32 must equal the default"
    );
    let f8 = run(Some("f8"));
    assert_ne!(f8, default, "f8 must change the forecast");
    // f64 runs natively on the CPU and moves the forecast by f32 rounding only.
    let f64 = run(Some("f64"));
    let worst = f64
        .iter()
        .zip(&default)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0f32, f32::max);
    assert!(worst < 1e-4, "f64 moved the forecast by {worst:.3e}");
    // f64x2 (double-single on the CPU too) lands on the same forecast.
    let df = run(Some("f64x2"));
    let gap = df
        .iter()
        .zip(&f64)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0f32, f32::max);
    assert!(gap < 1e-6, "f64x2 differs from f64 by {gap:.3e}");
}
