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

//! Command-line entry — `rlx-clinicalbert --weights … --device cpu [--token-ids 101,…]`.

use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use std::path::PathBuf;

use crate::runner::{ClinicalBertRunner, Pooling};

const HELP: &str = "\
rlx-clinicalbert — ClinicalBERT encoder forward + pooled embedding.

USAGE:
  rlx-clinicalbert --weights <PATH> [options]

OPTIONS:
  --weights PATH          Safetensors file or model directory (required).
  --config PATH           Override path to config.json (default: next to weights).
  --variant NAME          One of: huang, bio_clinical, bio_discharge.
                          Overrides config.json when set (uses built-in preset).
  --device DEV            cpu | metal | mlx | cuda | rocm | wgpu | vulkan (default: cpu).
  --batch N               Compile batch size (default 1).
  --max-seq N             Compile sequence length (default min(512, max_pos)).
  --token-ids IDS         Comma-separated WordPiece IDs (padded/truncated to --max-seq).
                          Default: a [CLS] + [SEP] dummy of length 2.
  --pooling POOL          cls | mean | none (default: cls).
  --print N               Print first N output floats (default: 8).
  -h, --help              Show this help.
";

#[derive(Default)]
struct Args {
    weights: Option<PathBuf>,
    config: Option<PathBuf>,
    variant: Option<String>,
    device: Option<String>,
    batch: Option<usize>,
    max_seq: Option<usize>,
    token_ids: Option<String>,
    pooling: Option<String>,
    print: Option<usize>,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut out = Args::default();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "--weights" => {
                out.weights = Some(PathBuf::from(args.get(i + 1).context("--weights PATH")?));
                i += 2;
            }
            "--config" => {
                out.config = Some(PathBuf::from(args.get(i + 1).context("--config PATH")?));
                i += 2;
            }
            "--variant" => {
                out.variant = Some(args.get(i + 1).context("--variant NAME")?.clone());
                i += 2;
            }
            "--device" => {
                out.device = Some(args.get(i + 1).context("--device DEV")?.clone());
                i += 2;
            }
            "--batch" => {
                out.batch = Some(args.get(i + 1).context("--batch N")?.parse()?);
                i += 2;
            }
            "--max-seq" => {
                out.max_seq = Some(args.get(i + 1).context("--max-seq N")?.parse()?);
                i += 2;
            }
            "--token-ids" => {
                out.token_ids = Some(args.get(i + 1).context("--token-ids IDS")?.clone());
                i += 2;
            }
            "--pooling" => {
                out.pooling = Some(args.get(i + 1).context("--pooling POOL")?.clone());
                i += 2;
            }
            "--print" => {
                out.print = Some(args.get(i + 1).context("--print N")?.parse()?);
                i += 2;
            }
            other => bail!("rlx-clinicalbert: unknown flag {other}"),
        }
    }
    Ok(out)
}

fn parse_device(name: &str) -> Result<Device> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "metal" | "mps" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" | "hip" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => bail!("rlx-clinicalbert: unknown device {other}"),
    })
}

fn parse_variant(name: &str) -> Result<crate::config::ClinicalBertVariant> {
    use crate::config::ClinicalBertVariant::*;
    Ok(match name.to_ascii_lowercase().as_str() {
        "huang" | "clinicalbert" | "medicalai" => Huang,
        "bio_clinical" | "bio-clinical" | "bioclinical" => BioClinical,
        "bio_discharge" | "bio-discharge" | "biodischarge" | "discharge" => BioDischarge,
        other => bail!("rlx-clinicalbert: unknown variant {other}"),
    })
}

/// Library entry point used by both the binary and downstream registrations.
pub fn run(args: &[String]) -> Result<()> {
    let args = parse(args)?;
    let weights = args
        .weights
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--weights PATH is required (try --help)"))?;
    let device = match args.device.as_deref() {
        Some(d) => parse_device(d)?,
        None => Device::Cpu,
    };

    let mut builder = ClinicalBertRunner::builder()
        .weights(&weights)
        .device(device);
    if let Some(b) = args.batch {
        builder = builder.batch(b);
    }
    if let Some(s) = args.max_seq {
        builder = builder.max_seq(s);
    }
    if let Some(v) = args.variant.as_deref() {
        builder = builder.variant(parse_variant(v)?);
    }
    if let Some(p) = args.config.as_ref() {
        builder = builder.config_path(p);
    }
    if let Some(p) = args.pooling.as_deref() {
        let pool = Pooling::from_str_opt(p)
            .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: unknown --pooling {p}"))?;
        builder = builder.pooling(pool);
    }

    let mut runner = builder.build()?;
    let (b, s) = runner.compiled_shape();

    let token_ids: Vec<f32> = match args.token_ids.as_deref() {
        Some(csv) => {
            let mut v: Vec<f32> = csv
                .split(',')
                .map(|t| t.trim().parse::<u32>().map(|x| x as f32))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("parsing --token-ids")?;
            if v.len() > b * s {
                v.truncate(b * s);
            }
            while v.len() < b * s {
                v.push(0.0);
            }
            v
        }
        None => {
            // [CLS] = 101, [SEP] = 102, rest padded with 0 ([PAD]).
            let mut v = vec![0f32; b * s];
            for bi in 0..b {
                v[bi * s] = 101.0;
                if s >= 2 {
                    v[bi * s + 1] = 102.0;
                }
            }
            v
        }
    };

    let mut attention_mask = vec![0f32; b * s];
    for bi in 0..b {
        for si in 0..s {
            attention_mask[bi * s + si] = if token_ids[bi * s + si] > 0.0 {
                1.0
            } else {
                0.0
            };
        }
    }
    let token_type_ids = vec![0f32; b * s];
    let mut position_ids = vec![0f32; b * s];
    for bi in 0..b {
        for si in 0..s {
            position_ids[bi * s + si] = si as f32;
        }
    }

    let out = runner.embed(&token_ids, &attention_mask, &token_type_ids, &position_ids)?;
    let n = args.print.unwrap_or(8).min(out.len());
    println!(
        "rlx-clinicalbert: device={device:?} variant={:?} pooling={:?} batch={} seq={} hidden={} out_len={}",
        runner.config().variant,
        runner.pooling(),
        b,
        s,
        runner.hidden_size(),
        out.len()
    );
    print!("first {n} floats: [");
    for (i, x) in out.iter().take(n).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{x:.6}");
    }
    println!("]");

    Ok(())
}
