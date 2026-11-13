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

//! `rlx-kraken --model MODEL.mlmodel --image LINE.png [--image …]`

use crate::KrakenRunner;
use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

const HELP: &str = "rlx-kraken — Kraken text recognition (one text line per image)\n\
flags: --model <recognition .mlmodel> --image <line> [--image …]\n\
       [--device cpu|metal|mlx|wgpu|…] [--timing]";

/// Parse args and run the `rlx-kraken` command.
pub fn run(args: &[String]) -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut device = "cpu".to_string();
    let mut timing = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "--weights" => model = Some(req(args, &mut i)?.into()),
            "--image" => images.push(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--timing" => {
                timing = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("{HELP}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}\n{HELP}"),
        }
    }
    let model = model.ok_or_else(|| anyhow!("--model is required\n{HELP}"))?;
    if images.is_empty() {
        bail!("at least one --image is required\n{HELP}");
    }
    let device = parse_standard_device("kraken", &device)?;
    let mut runner = KrakenRunner::open(&model, device)?;
    for path in &images {
        let t0 = std::time::Instant::now();
        let r = runner.recognize_path(path)?;
        if timing {
            eprintln!(
                "[rlx-kraken] {} — {} frames in {:?}",
                path.display(),
                r.frames,
                t0.elapsed()
            );
        }
        println!("{}", r.text);
    }
    Ok(())
}
