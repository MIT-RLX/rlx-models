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

//! `rlx-dan --weights dan_rimes_page.pt --image PAGE.png [--layout]`

use crate::{DEFAULT_MAX_CHARS, DanRunner};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

const HELP: &str = "rlx-dan — DAN full-page handwritten text recognition\n\
flags: --weights <dan_*.pt> --image <page> [--image …]\n\
       [--layout]  keep DAN's layout tokens (Ⓑ … ⓑ) in the output\n\
       [--max-chars N] (default 3000) [--device cpu|metal|mlx|wgpu|…] [--timing]\n\
pages: the RIMES model was trained on 150 dpi scans (half the 300 dpi originals)";

/// Parse args and run the `rlx-dan` command.
pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut device = "cpu".to_string();
    let mut max_chars = DEFAULT_MAX_CHARS;
    let (mut layout, mut timing) = (false, false);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--image" => images.push(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--max-chars" => max_chars = req(args, &mut i)?.parse().context("--max-chars")?,
            "--layout" => {
                layout = true;
                i += 1;
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
    let device = parse_standard_device("dan", &device)?;
    let mut runner = DanRunner::open(&weights, device)?;
    for path in &images {
        let t0 = std::time::Instant::now();
        let r = runner.recognize_path(path, max_chars)?;
        if timing {
            eprintln!(
                "[rlx-dan] {} — {} tokens in {:?}",
                path.display(),
                r.ids.len(),
                t0.elapsed()
            );
        }
        println!("{}", if layout { r.raw.clone() } else { r.text() });
    }
    Ok(())
}
