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

// RLX CLI for SAM 3
use crate::{Sam3, Sam3Config};
use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_sam_device, req};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut text_tokens: Vec<u32> = Vec::new();
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--text-tokens" => {
                text_tokens = req(args, &mut i)?
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-sam3 — flags: --weights --device (cpu|metal|mlx|cuda|rocm|gpu|vulkan) --text-tokens --dry"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_sam_device("sam3", &device)?;
    let path = weights.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?;
    if text_tokens.is_empty() {
        text_tokens = (0..32u32).collect();
        eprintln!("[rlx-sam3] using placeholder text tokens 0..32");
    }
    if dry {
        return Ok(());
    }
    let h_in = 1024usize;
    let w_in = 1024usize;
    let rgb = vec![128u8; h_in * w_in * 3];
    let mut sam = Sam3::from_checkpoint_on(path, Sam3Config::base(), device)?;
    let pred = sam.predict_image_text(&rgb, h_in, w_in, &text_tokens)?;
    eprintln!(
        "[rlx-sam3] instances={} mask_shape={:?} scores[..5]={:?}",
        pred.num_instances,
        pred.mask_shape,
        &pred.scores[..pred.scores.len().min(5)]
    );
    Ok(())
}
