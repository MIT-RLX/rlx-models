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

//! `rlx-trocr --weights DIR --image LINE.png [--image …] [--max-new-tokens N]`

use crate::{DEFAULT_MAX_TOKENS, TrOcrRunner};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

const HELP: &str = "rlx-trocr — TrOCR text recognition (one text line per image)\n\
flags: --weights <HF model dir> --image <line> [--image …]\n\
       [--max-new-tokens N] (default 64) [--device cpu|metal|mlx|wgpu|…] [--timing]";

/// Parse args and run the `rlx-trocr` command.
pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut device = "cpu".to_string();
    let mut max_new = 64usize;
    let mut timing = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--image" => images.push(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--max-new-tokens" => {
                max_new = req(args, &mut i)?.parse().context("--max-new-tokens")?;
            }
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
    let weights = weights.ok_or_else(|| anyhow!("--weights is required\n{HELP}"))?;
    if images.is_empty() {
        bail!("at least one --image is required\n{HELP}");
    }
    let device = parse_standard_device("trocr", &device)?;
    let t0 = std::time::Instant::now();
    let mut runner =
        TrOcrRunner::open_with(&weights, device, (max_new + 1).max(DEFAULT_MAX_TOKENS))?;
    if timing {
        eprintln!("[rlx-trocr] loaded + compiled in {:?}", t0.elapsed());
    }
    for path in &images {
        let t0 = std::time::Instant::now();
        let r = runner.recognize_path(path, max_new)?;
        if timing {
            eprintln!(
                "[rlx-trocr] {} — {} tokens in {:?}",
                path.display(),
                r.ids.len() - 1,
                t0.elapsed()
            );
        }
        println!("{}", r.text);
    }
    Ok(())
}
