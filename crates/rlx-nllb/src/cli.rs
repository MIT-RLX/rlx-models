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

//! `rlx-nllb` command-line interface.

use crate::{GenerateConfig, NllbConfig, NllbRunner};
use anyhow::{Result, anyhow, bail};
use rlx_cli::parse_standard_device;
use std::path::PathBuf;

struct Args {
    weights: PathBuf,
    text: String,
    src: String,
    tgt: String,
    device: rlx_runtime::Device,
    max_new_tokens: usize,
    beams: usize,
}

fn parse_args() -> Result<Args> {
    let mut weights = None;
    let mut text = None;
    let mut src = "eng_Latn".to_string();
    let mut tgt = "fra_Latn".to_string();
    let mut device_s = "cpu".to_string();
    let mut max_new_tokens = 128usize;
    let mut beams = 1usize;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" | "--model-dir" => weights = it.next().map(PathBuf::from),
            "--text" => text = it.next(),
            "--src" | "--src-lang" => {
                src = it.next().ok_or_else(|| anyhow!("--src needs a value"))?
            }
            "--tgt" | "--tgt-lang" => {
                tgt = it.next().ok_or_else(|| anyhow!("--tgt needs a value"))?
            }
            "--device" => {
                device_s = it.next().ok_or_else(|| anyhow!("--device needs a value"))?;
            }
            "--max-new-tokens" => {
                max_new_tokens = it
                    .next()
                    .ok_or_else(|| anyhow!("--max-new-tokens needs a value"))?
                    .parse()?;
            }
            "--beams" | "--num-beams" => {
                beams = it
                    .next()
                    .ok_or_else(|| anyhow!("--beams needs a value"))?
                    .parse()?;
            }
            "-h" | "--help" => {
                println!(
                    "rlx-nllb --weights DIR --text \"...\" [--src eng_Latn] [--tgt fra_Latn] \
                     [--device cpu|metal|mlx|cuda|rocm|gpu|vulkan] [--max-new-tokens N] [--beams N]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(Args {
        weights: weights.ok_or_else(|| anyhow!("--weights DIR required"))?,
        text: text.ok_or_else(|| anyhow!("--text required"))?,
        src,
        tgt,
        device: parse_standard_device("nllb", &device_s)?,
        max_new_tokens,
        beams,
    })
}

pub fn run() -> Result<()> {
    let args = parse_args()?;
    let cfg = if args.weights.join("config.json").is_file() {
        NllbConfig::from_hf_config_json(&args.weights.join("config.json"))?
    } else {
        NllbConfig::distilled_600m()
    };
    let mut runner = NllbRunner::builder()
        .weights(&args.weights)
        .device(args.device)
        .config(cfg)
        .build()?;
    let opts =
        GenerateConfig::from_config(runner.config(), args.max_new_tokens).with_beams(args.beams);
    let out = runner.translate(&args.text, &args.src, &args.tgt, &opts)?;
    println!("{out}");
    Ok(())
}
