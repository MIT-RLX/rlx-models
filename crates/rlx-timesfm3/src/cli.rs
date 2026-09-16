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

//! `rlx-timesfm3` CLI.

use crate::config::TimesFM3Config;
use crate::device::{available_device_labels, parse_device_list, resolve_device};
use crate::forecaster::TimesFM3ForecasterBuilder;
use crate::host::TimesFM3Model;
use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use std::fs;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut context_path: Option<PathBuf> = None;
    let mut horizon: usize = 96;
    let mut return_quantiles = false;
    let mut synth = false;
    let mut device: Option<String> = None;
    let mut devices: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model-dir" => weights = Some(rlx_cli::req(args, &mut i)?.into()),
            "--config" => config = Some(rlx_cli::req(args, &mut i)?.into()),
            "--context" | "--context-csv" => {
                context_path = Some(rlx_cli::req(args, &mut i)?.into())
            }
            "--horizon" => horizon = rlx_cli::req(args, &mut i)?.parse().context("horizon")?,
            "--quantiles" => return_quantiles = true,
            "--synth" => synth = true,
            "--device" => device = Some(rlx_cli::req(args, &mut i)?),
            "--devices" => devices = Some(rlx_cli::req(args, &mut i)?),
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    if synth {
        return run_synth(horizon, return_quantiles, device.as_deref());
    }

    let weights = weights.ok_or_else(|| anyhow::anyhow!("--weights required (or --synth)"))?;

    if let Some(list) = devices {
        return compare_devices(
            &weights,
            config.as_deref(),
            &list,
            horizon,
            context_path.as_ref(),
        );
    }

    let mut builder = TimesFM3ForecasterBuilder::new().weights(&weights);
    if let Some(c) = config {
        builder = builder.config(TimesFM3Config::from_file(&c)?);
    }
    if let Some(name) = device {
        builder = builder.device_name(&name)?;
    }
    let mut forecaster = builder.build()?;

    let context = load_context(context_path.as_ref())?;

    let out = forecaster.predict(&context, horizon, return_quantiles)?;
    print_forecast(&out, return_quantiles, horizon);
    Ok(())
}

fn run_synth(horizon: usize, return_quantiles: bool, device: Option<&str>) -> Result<()> {
    let dev = match device {
        Some(name) => resolve_device(name)?,
        None => Device::Cpu,
    };
    if dev == Device::Cpu && std::env::var("RLX_TIMESFM3_COMPILED_CORE").is_err() {
        let cfg = TimesFM3Config::synth_tiny();
        let model = TimesFM3Model::synth(cfg, 42);
        let ctx: Vec<f32> = (0..128).map(|t| (t as f32 * 0.1).sin()).collect();
        let ctx2 = ndarray::Array3::from_shape_vec((1, 1, ctx.len()), ctx)?;
        let out = model.decode(ctx2.view(), horizon, None, None, None);
        let median = model.cfg.median_quantile_index();
        if return_quantiles {
            println!("synth logits shape {:?}", out.shape());
        } else {
            for t in 0..horizon.min(8) {
                println!("{t}\t{}", out[[0, 0, t, median]]);
            }
        }
        return Ok(());
    }
    let cfg = TimesFM3Config::synth_tiny();
    let model = TimesFM3Model::synth(cfg.clone(), 42);
    let mut forecaster = crate::forecaster::TimesFM3Forecaster::from_model(model, dev);
    let ctx: Vec<f32> = (0..128).map(|t| (t as f32 * 0.1).sin()).collect();
    let out = forecaster.predict(&ctx, horizon, return_quantiles)?;
    print_forecast(&out, return_quantiles, horizon);
    Ok(())
}

fn compare_devices(
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    list: &str,
    horizon: usize,
    context_path: Option<&PathBuf>,
) -> Result<()> {
    let devs = parse_device_list(list)?;
    let context = load_context(context_path)?;
    for dev in devs {
        let mut builder = TimesFM3ForecasterBuilder::new()
            .weights(weights)
            .device(dev);
        if let Some(c) = config {
            builder = builder.config(TimesFM3Config::from_file(c)?);
        }
        let mut forecaster = builder.build()?;
        let out = forecaster.predict(&context, horizon, false)?;
        println!(
            "{}: first={:.6}",
            crate::device::device_label(dev),
            out.forecast.first().copied().unwrap_or(f32::NAN)
        );
    }
    Ok(())
}

fn load_context(context_path: Option<&PathBuf>) -> Result<Vec<f32>> {
    if let Some(p) = context_path {
        load_context_csv(p)
    } else {
        Ok((0..512).map(|t| (t as f32 * 0.05).sin() + 1.0).collect())
    }
}

fn load_context_csv(path: &PathBuf) -> Result<Vec<f32>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut vals = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for part in line.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            vals.push(part.parse().with_context(|| format!("parse {part}"))?);
        }
    }
    if vals.is_empty() {
        bail!("no values in context file");
    }
    Ok(vals)
}

fn print_forecast(out: &crate::forecaster::ForecastOutput, return_quantiles: bool, horizon: usize) {
    if return_quantiles {
        println!(
            "median forecast (first 8): {:?}",
            &out.forecast[..8.min(out.forecast.len())]
        );
        if let Some(q) = &out.quantiles {
            println!("quantiles flat len {}", q.len());
        }
    } else {
        for (i, v) in out.forecast.iter().take(horizon).enumerate() {
            println!("{i}\t{v}");
        }
    }
}

fn print_help() {
    let devs = available_device_labels().join("|");
    println!(
        "rlx-timesfm3 — TimesFM-3 zero-shot time-series forecasting\n\
         \n\
         Usage:\n\
           rlx-timesfm3 --weights DIR|model.safetensors [--device {devs}] [--context series.csv] [--horizon N]\n\
           rlx-timesfm3 --synth [--device {devs}] [--horizon N]\n\
           rlx-timesfm3 --weights DIR --devices all|cpu,metal\n\
         \n\
         Weights: google/timesfm-3.0-pytorch (config.json + model.safetensors)\n\
         License: TimesFM Non-Commercial License v1.0 for official 3.0 weights\n\
         Env: RLX_TIMESFM3_COMPILED_CORE=1 runs compiled core on cpu (parity debugging)"
    );
}
