// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A Timestral package forecasts what `TimestralForecaster` does: one series,
//! several series with gaps and a late start, covariates past and known, a
//! horizon of several decode blocks — synthetic weights, and the 5M
//! checkpoint where it is on disk (`.cache/timestral/5M`, or
//! `RLX_TIMESTRAL_DIR`).

#![cfg(feature = "export")]

use std::path::{Path, PathBuf};

use ndarray::Array3;
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use rlx_timestral::export::ExportOptions;
use rlx_timestral::{
    ForecastOptions, TimestralConfig, TimestralForecaster, TimestralModel, TimestralWeights,
};
use serde_json::json;

struct Case {
    name: &'static str,
    context: Array3<f32>,
    horizon: usize,
    opts: ForecastOptions,
    symmetric: bool,
}

fn series(b: usize, v: usize, t: usize, gaps: bool) -> Array3<f32> {
    Array3::from_shape_fn((b, v, t), |(b, v, i)| {
        if gaps && ((b == 0 && v == 1 && i < 20) || (b == 1 && (30..36).contains(&i))) {
            f32::NAN
        } else {
            ((i as f32 + 3.0 * b as f32) * 0.3 + v as f32).sin() * (2.0 + b as f32)
                + 0.05 * i as f32
                + 4.0
        }
    })
}

fn cases(ps: usize, block: usize) -> Vec<Case> {
    let t = 7 * ps / 2 + 5;
    // Past several decode blocks.
    let long = (2 * block + 1) * ps + 3;
    let covariates = |t: usize, horizon: usize| ForecastOptions {
        past_covariates: Some(Array3::from_shape_fn((2, 1, t), |(b, _, i)| {
            (i as f32 * 0.7 + b as f32).cos()
        })),
        known_covariates: Some(Array3::from_shape_fn((2, 1, t + horizon), |(_, _, i)| {
            (i % 7 >= 5) as u8 as f32
        })),
        ..ForecastOptions::default()
    };
    vec![
        Case {
            name: "one series",
            context: series(1, 1, t, false),
            horizon: ps + 7,
            opts: ForecastOptions::default(),
            symmetric: true,
        },
        Case {
            name: "raw, past several blocks",
            context: series(1, 1, t, false),
            horizon: long,
            opts: ForecastOptions::raw(),
            symmetric: false,
        },
        Case {
            name: "two series, two variates, gaps",
            context: series(2, 2, t, true),
            horizon: 3 * ps,
            opts: ForecastOptions::default(),
            symmetric: true,
        },
        Case {
            name: "covariates",
            context: series(2, 2, t, true),
            horizon: 2 * ps + 1,
            opts: covariates(t, 2 * ps + 1),
            symmetric: true,
        },
    ]
}

fn tensor(name: &str, a: &Array3<f32>) -> Tensor {
    let v: Vec<f32> = a.iter().copied().collect();
    Tensor::f32(name, a.shape().to_vec(), &v)
}

fn check(cfg: TimestralConfig, weights: TimestralWeights, what: &str) {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("timestral.rlxp");
    let report =
        rlx_timestral::export::export_model(&cfg, &weights, &out, &ExportOptions::default())
            .unwrap_or_else(|e| panic!("{what}: export: {e:#}"));
    eprintln!("{what}: {report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let (ps, block) = (cfg.patch_size, cfg.block_size);
    let mut fc = TimestralForecaster::new(TimestralModel::new(cfg, &weights, Device::Cpu));
    for case in cases(ps, block) {
        let want = fc
            .forecast(case.context.view(), case.horizon, &case.opts)
            .unwrap();
        let mut inputs = vec![tensor("context", &case.context)];
        for (name, a) in [
            ("past_covariates", &case.opts.past_covariates),
            ("known_covariates", &case.opts.known_covariates),
        ] {
            if let Some(a) = a {
                inputs.push(tensor(name, a));
            }
        }
        let options = json!({
            "horizon": case.horizon,
            "symmetric_averaging": case.symmetric,
            "clip_nonnegative": case.opts.clip_nonnegative,
        });
        let outs = m
            .run(&inputs, &options)
            .unwrap_or_else(|e| panic!("{what}, {}: {e:#}", case.name));
        let got = outs.iter().find(|t| t.name == "quantiles").unwrap();
        assert_eq!(got.shape, want.quantiles.shape(), "{what}, {}", case.name);
        let got = got.to_f32().unwrap();
        let (mut worst, mut scale) = (0f32, 0f32);
        for (a, b) in got.iter().zip(want.quantiles.iter()) {
            worst = worst.max((a - b).abs());
            scale = scale.max(b.abs());
        }
        eprintln!(
            "{what}, {}: {} values, largest difference {worst:.3e} (values up to {scale:.2})",
            case.name,
            got.len()
        );
        assert!(
            worst <= 1e-4 * scale.max(1.0),
            "{what}, {}: off by {worst}",
            case.name
        );
    }
}

#[test]
fn a_package_forecasts_what_the_forecaster_does() {
    let cfg = TimestralConfig::synth_tiny();
    let weights = TimestralWeights::synth(&cfg, 11);
    check(cfg, weights, "synthetic");
}

#[test]
fn the_5m_checkpoint_exports() {
    let dir = rlx_ir::env::var("RLX_TIMESTRAL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/timestral/5M"));
    if !dir.join("model.safetensors").exists() {
        eprintln!("no checkpoint at {}; skipped", dir.display());
        return;
    }
    let cfg = TimestralConfig::from_file(&dir.join("config.json")).unwrap();
    let weights = TimestralWeights::load(&dir, &cfg).unwrap();
    check(cfg, weights, "5M");
}
