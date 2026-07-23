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

use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

use crate::config::Tier;
use crate::runner::PpOcrV6Runner;

pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut tier = Tier::Tiny;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => model_dir = Some(req(args, &mut i)?.into()),
            "--image" => image = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--tier" => {
                tier = req(args, &mut i)?.parse()?;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-ppocrv6 — PP-OCRv6 tiny/small (native RLX HIR + safetensors)\n\
                     Flags:\n\
                       --model-dir DIR   cache dir with {{det,rec}}/model.safetensors\n\
                       --tier tiny|small\n\
                       --image PATH\n\
                       [--device cpu|metal|…] [--dry]"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let device = parse_standard_device("ppocrv6", &device).or_else(|e| {
        // CoreML / ANE is opt-in via `coreml` feature (LM-style device set).
        if device.eq_ignore_ascii_case("coreml") || device.eq_ignore_ascii_case("ane") {
            rlx_cli::parse_lm_device("ppocrv6", &device)
        } else {
            Err(e)
        }
    })?;
    let model_dir = model_dir.ok_or_else(|| anyhow!("--model-dir is required"))?;

    eprintln!("[rlx-ppocrv6] tier={} device={device:?}", tier.as_str());
    let runner = PpOcrV6Runner::builder()
        .tier(tier)
        .model_dir(model_dir)
        .device(device)
        .build()?;

    if dry {
        eprintln!("[rlx-ppocrv6] --dry set; skipping inference");
        return Ok(());
    }

    let image = image.ok_or_else(|| anyhow!("--image is required unless --dry"))?;
    let t0 = std::time::Instant::now();
    let out = runner.predict_path(&image)?;
    eprintln!(
        "[rlx-ppocrv6] {} lines in {:?}",
        out.lines.len(),
        t0.elapsed()
    );
    if !out.text.is_empty() {
        println!("{}", out.text);
    }
    Ok(())
}
