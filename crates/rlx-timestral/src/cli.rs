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

//! `rlx-timestral` CLI.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use ndarray::Array3;
use rlx_runtime::Device;

use crate::config::TimestralConfig;
use crate::forecaster::{ForecastOptions, TimestralForecaster};
use crate::model::TimestralModel;
use crate::{available_devices, device_label};

pub fn run(args: &[String]) -> Result<()> {
    if args.first().map(String::as_str) == Some("qat") {
        return run_qat(&args[1..]);
    }
    let mut args = args.to_vec();
    let number_format = take_number_format_arg(&mut args)?;
    let args = &args[..];
    let mut weights: Option<String> = None;
    let mut context_path: Option<PathBuf> = None;
    let mut horizon = 96usize;
    let mut device = "cpu".to_string();
    let mut devices: Option<String> = None;
    let mut synth = false;
    let mut opts = ForecastOptions::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model-dir" | "--repo" => weights = Some(rlx_cli::req(args, &mut i)?),
            "--context" => context_path = Some(rlx_cli::req(args, &mut i)?.into()),
            "--horizon" => horizon = rlx_cli::req(args, &mut i)?.parse().context("--horizon")?,
            "--device" => device = rlx_cli::req(args, &mut i)?,
            "--devices" => devices = Some(rlx_cli::req(args, &mut i)?),
            "--synth" => {
                synth = true;
                i += 1;
            }
            "--no-symmetric" => {
                opts.symmetric_averaging = false;
                i += 1;
            }
            "--no-clip" => {
                opts.clip_nonnegative = false;
                i += 1;
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let context = load_context(context_path.as_ref())?;
    let devs: Vec<Device> = match devices.as_deref() {
        Some("all") => available_devices(),
        Some(list) => list
            .split(',')
            .map(|d| rlx_cli::parse_standard_device("rlx-timestral", d.trim()))
            .collect::<Result<_>>()?,
        None => vec![rlx_cli::parse_standard_device("rlx-timestral", &device)?],
    };

    for dev in devs {
        let mut fc = if synth {
            TimestralForecaster::new(TimestralModel::synth(
                TimestralConfig::synth_tiny(),
                42,
                dev,
            ))
        } else {
            let w = weights.as_deref().context(
                "--weights DIR (or a Hub repo id with --features hf-download), or --synth",
            )?;
            TimestralForecaster::from_pretrained(w, dev)?
        };
        fc.model_mut().set_number_format(number_format)?;
        let ctx = Array3::from_shape_vec((1, 1, context.len()), context.clone())?;
        let t0 = Instant::now();
        let out = fc.forecast(ctx.view(), horizon, &opts)?;
        let cold = t0.elapsed();
        let t1 = Instant::now();
        let out2 = fc.forecast(ctx.view(), horizon, &opts)?;
        let warm = t1.elapsed();
        debug_assert_eq!(out.quantiles, out2.quantiles);

        let levels = &out.quantile_levels;
        let lo = levels[1];
        let hi = levels[levels.len() - 2];
        let med = out.median();
        let q_lo = out.quantile(lo)?;
        let q_hi = out.quantile(hi)?;
        println!(
            "# {} — context {} steps, horizon {horizon}; first call {:.1} ms, warm {:.1} ms",
            device_label(dev),
            context.len(),
            cold.as_secs_f64() * 1e3,
            warm.as_secs_f64() * 1e3
        );
        println!("step\tq{lo}\tmedian\tq{hi}");
        for t in 0..horizon {
            println!(
                "{t}\t{:.6}\t{:.6}\t{:.6}",
                q_lo[[0, 0, t]],
                med[[0, 0, t]],
                q_hi[[0, 0, t]]
            );
        }
    }
    Ok(())
}

/// `--number-format FMT`, removed from `args`.
fn take_number_format_arg(args: &mut Vec<String>) -> Result<Option<rlx_runtime::NumberFormat>> {
    let Some(i) = args.iter().position(|a| a == "--number-format") else {
        return Ok(None);
    };
    let spec = args
        .get(i + 1)
        .cloned()
        .context("--number-format needs a value")?;
    args.drain(i..i + 2);
    spec.parse()
        .map(Some)
        .map_err(|e: String| anyhow::anyhow!("--number-format: {e}"))
}

/// One value per line or comma-separated; empty fields and `nan` mark missing values.
fn load_context(path: Option<&PathBuf>) -> Result<Vec<f32>> {
    let Some(path) = path else {
        return Ok((0..512)
            .map(|t| {
                let t = t as f32;
                10.0 * (2.0 * std::f32::consts::PI * t / 24.0).sin() + 0.02 * t + 50.0
            })
            .collect());
    };
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut vals = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        for part in line.split(',').map(str::trim) {
            if part.is_empty() || part.eq_ignore_ascii_case("nan") {
                vals.push(f32::NAN);
            } else {
                vals.push(part.parse().with_context(|| format!("parse {part:?}"))?);
            }
        }
    }
    if vals.is_empty() {
        bail!("no values in {}", path.display());
    }
    Ok(vals)
}

fn print_help() {
    println!(
        "rlx-timestral — Mistral Timestral zero-shot time-series forecasting\n\
         \n\
         Usage:\n\
           rlx-timestral --weights DIR [--device cpu|metal|…] [--context series.csv] [--horizon N]\n\
           rlx-timestral --weights DIR --devices all|cpu,metal\n\
           rlx-timestral --synth [--device …]\n\
           rlx-timestral qat --weights DIR --data series.csv --out DIR [--number-format f4]\n\
               [--steps N] [--lr F] [--batch N] [--context-patches N] [--horizon-patches N]\n\
               [--holdout N] [--device …]   fine-tune / QAT on CSV columns, report held-out WQL\n\
         \n\
         Options:\n\
           --weights DIR     checkpoint directory (config.json + model.safetensors);\n\
                             a Hub repo id like mistralai/Timestral-209M with --features hf-download\n\
           --context FILE    one value per line or comma-separated; empty/nan = missing\n\
           --no-symmetric    skip symmetric averaging (one pass instead of two)\n\
           --no-clip         do not clip non-negative series at 0\n\
           --number-format F run in f16, bf16, f8, f6, f4, fNeXmY, … (default f32)\n\
         \n\
         Prints the median and the 5% / 95% quantiles per step."
    );
}

/// `rlx-timestral qat`: fine-tune (with `--number-format`, quantization-aware)
/// on the series of a CSV, report held-out WQL before and after, export.
fn run_qat(args: &[String]) -> Result<()> {
    use crate::train::{TrainOptions, export, holdout_wql, train};
    let mut args = args.to_vec();
    let format = take_number_format_arg(&mut args)?;
    let (mut weights, mut data, mut out) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let mut device = "cpu".to_string();
    let mut holdout = 64usize;
    let mut o = TrainOptions {
        format,
        ..TrainOptions::default()
    };
    let mut i = 0;
    while i < args.len() {
        let num = |i: &mut usize| -> Result<String> { rlx_cli::req(&args, i) };
        match args[i].as_str() {
            "--weights" => weights = Some(num(&mut i)?.into()),
            "--data" => data = Some(num(&mut i)?.into()),
            "--out" => out = Some(num(&mut i)?.into()),
            "--device" => device = num(&mut i)?,
            "--steps" => o.steps = num(&mut i)?.parse().context("--steps")?,
            "--lr" => o.lr = num(&mut i)?.parse().context("--lr")?,
            "--batch" => o.batch = num(&mut i)?.parse().context("--batch")?,
            "--context-patches" => {
                o.context_patches = num(&mut i)?.parse().context("--context-patches")?
            }
            "--horizon-patches" => {
                o.horizon_patches = num(&mut i)?.parse().context("--horizon-patches")?
            }
            "--holdout" => holdout = num(&mut i)?.parse().context("--holdout")?,
            "--seed" => o.seed = num(&mut i)?.parse().context("--seed")?,
            other => bail!("qat: unknown argument {other} (see --help)"),
        }
    }
    let weights = weights.context("qat: --weights DIR required")?;
    let data = data.context("qat: --data series.csv required")?;
    let out = out.context("qat: --out DIR required")?;
    o.device = rlx_cli::parse_standard_device("rlx-timestral", &device)?;

    let cfg = TimestralConfig::from_file(&weights.join("config.json"))?;
    let w = crate::TimestralWeights::load(&weights, &cfg)?;
    let series = load_columns(&data)?;
    let train_part: Vec<Vec<f32>> = series
        .iter()
        .map(|s| s[..s.len().saturating_sub(holdout)].to_vec())
        .collect();
    let ctx = o.context_patches * cfg.patch_size;
    let score = |dir: &std::path::Path| -> Result<f32> {
        let mut fc = TimestralForecaster::load(dir, o.device)?;
        fc.model_mut().set_number_format(format)?;
        holdout_wql(&mut fc, &series, ctx, holdout)
    };
    let before = score(&weights)?;
    let fmt = format.map_or("f32".to_string(), |f| f.to_string());
    println!("held-out WQL in {fmt} before: {before:.5}");
    let trained = train(&cfg, &w, &train_part, &o)?;
    let n = trained.losses.len();
    let avg = |l: &[f32]| l.iter().sum::<f32>() / l.len().max(1) as f32;
    println!(
        "pinball loss: first {} steps {:.5} -> last {} steps {:.5}",
        n.min(10),
        avg(&trained.losses[..n.min(10)]),
        n.min(10),
        avg(&trained.losses[n.saturating_sub(10)..])
    );
    export(
        &weights.join("config.json"),
        &cfg,
        &w,
        &trained.params,
        &out,
    )?;
    let after = score(&out)?;
    println!(
        "held-out WQL in {fmt} after:  {after:.5}  (exported to {})",
        out.display()
    );
    Ok(())
}

/// CSV columns as series (header row skipped if not numeric); empty / `nan` = missing.
pub fn load_columns(path: &PathBuf) -> Result<Vec<Vec<f32>>> {
    let text = fs_read(path)?;
    let mut cols: Vec<Vec<f32>> = Vec::new();
    for line in text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let parsed: Vec<Option<f32>> = cells
            .iter()
            .map(|c| {
                if c.is_empty() || c.eq_ignore_ascii_case("nan") {
                    Some(f32::NAN)
                } else {
                    c.parse().ok()
                }
            })
            .collect();
        if parsed.iter().any(Option::is_none) {
            ensure!(
                cols.is_empty(),
                "non-numeric row after the header: {line:?}"
            );
            continue; // header
        }
        if cols.is_empty() {
            cols = vec![Vec::new(); parsed.len()];
        }
        ensure!(parsed.len() == cols.len(), "ragged CSV row: {line:?}");
        for (c, v) in cols.iter_mut().zip(parsed) {
            c.push(v.unwrap());
        }
    }
    ensure!(!cols.is_empty(), "no data in {}", path.display());
    Ok(cols)
}

fn fs_read(path: &PathBuf) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}
