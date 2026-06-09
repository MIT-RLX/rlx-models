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

//! Decompose an ONNX file into a generated RLX Rust crate + weights.

use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_onnx_decompose::{DecomposeOptions, WeightsFormat, decompose, decompose_bundle};

fn usage() -> &'static str {
    "usage: rlx-onnx-decompose <model.onnx> -o <out_dir> [--bundle <bundle_dir>] [--weights safetensors|gguf] [--crate-name NAME] [--seq-len N] [--rlx-root PATH]"
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut onnx = None;
    let mut bundle_dir = None;
    let mut out_dir = None;
    let mut weights_format = WeightsFormat::Safetensors;
    let mut opts = DecomposeOptions::default();

    while let Some(a) = args.next() {
        match a.as_str() {
            arg if arg.ends_with(".onnx") => {
                onnx = Some(PathBuf::from(arg));
            }
            "--bundle" => {
                bundle_dir = Some(PathBuf::from(args.next().context("--bundle dir")?));
            }
            "--rlx-root" => {
                opts.rlx_root = Some(PathBuf::from(args.next().context("--rlx-root")?));
            }
            "-o" | "--out" => {
                out_dir = Some(PathBuf::from(args.next().context("-o dir")?));
            }
            "--weights" => {
                let s = args.next().context("--weights")?;
                weights_format = match s.as_str() {
                    "safetensors" | "st" => WeightsFormat::Safetensors,
                    "gguf" => WeightsFormat::Gguf,
                    other => bail!("unknown weights format {other}"),
                };
            }
            "--crate-name" => {
                opts.crate_name = Some(args.next().context("--crate-name")?);
            }
            "--seq-len" => {
                opts.sequence_length = args
                    .next()
                    .context("--seq-len")?
                    .parse()
                    .context("--seq-len")?;
            }
            "--max-samples" => {
                opts.max_waveform_samples = args
                    .next()
                    .context("--max-samples")?
                    .parse()
                    .context("--max-samples")?;
            }
            "--help" | "-h" => {
                println!("{}", usage());
                println!();
                println!("Writes:");
                println!("  <out_dir>/src/{{lib,graph,weights}}.rs");
                println!("  <out_dir>/weights/model.safetensors");
                println!("  <out_dir>/decompose_report.json");
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let out_dir = out_dir.context("-o <out_dir> is required")?;
    opts.weights_format = weights_format;

    let plan = if let Some(bundle_dir) = bundle_dir {
        if opts.crate_name.is_none() {
            let stem = bundle_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("bundle_model");
            opts.crate_name = Some(rlx_onnx_decompose::sanitize_crate_name(stem));
        }
        decompose_bundle(&bundle_dir, &out_dir, &opts)?
    } else {
        let onnx = onnx.context("first argument must be model.onnx or use --bundle <dir>")?;
        if opts.crate_name.is_none() {
            let stem = onnx
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("onnx_model");
            opts.crate_name = Some(rlx_onnx_decompose::sanitize_crate_name(stem));
        }
        decompose(&onnx, &out_dir, &opts)?
    };

    println!("decomposed: {}", out_dir.display());
    println!("  crate: {}", plan.crate_name);
    let wf = match opts.weights_format {
        WeightsFormat::Safetensors => "safetensors",
        WeightsFormat::Gguf => "gguf",
    };
    println!("  weights: {} ({} tensors)", wf, plan.params.len());
    println!(
        "  coverage: lowered={} skipped={} unsupported={:?}",
        plan.import_report.lowered, plan.import_report.skipped, plan.import_report.unsupported
    );
    Ok(())
}
