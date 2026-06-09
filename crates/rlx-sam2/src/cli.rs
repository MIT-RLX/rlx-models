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

// RLX CLI for SAM 2
use crate::{Sam2, Sam2Config};
use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_sam_device, req};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut point: Option<(f32, f32)> = None;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--point" => {
                let v = req(args, &mut i)?;
                let parts: Vec<&str> = v.split(',').collect();
                if parts.len() != 2 {
                    bail!("--point expects X,Y");
                }
                point = Some((parts[0].trim().parse()?, parts[1].trim().parse()?));
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-sam2 — flags: --weights --device (cpu|metal|mlx|cuda|rocm|gpu|vulkan) --point --dry"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_sam_device("sam2", &device)?;
    let path = weights.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?;
    let variant = rlx_ir::env::var("RLX_SAM2_VARIANT").unwrap_or_else(|| "tiny".to_string());
    let cfg = match variant.as_str() {
        "tiny" => Sam2Config::hiera_tiny(),
        "small" => Sam2Config::hiera_small(),
        "base_plus" => Sam2Config::hiera_base_plus(),
        "large" => Sam2Config::hiera_large(),
        other => bail!("RLX_SAM2_VARIANT must be tiny|small|base_plus|large, got {other}"),
    };
    if dry {
        return Ok(());
    }
    let h_in = 1024usize;
    let w_in = 1024usize;
    let rgb = vec![128u8; h_in * w_in * 3];
    let (cx, cy) = point.unwrap_or((512.0, 512.0));
    let mut sam = Sam2::from_safetensors_on(path, cfg, device)?;
    let pred = sam.predict_image(
        &rgb,
        h_in,
        w_in,
        Some((&[cx, cy], &[1.0f32])),
        None,
        None,
        true,
    )?;
    eprintln!(
        "[rlx-sam2] masks={} out={}x{} iou={:?}",
        pred.num_masks,
        pred.h_out,
        pred.w_out,
        &pred.iou_pred[..pred.iou_pred.len().min(pred.num_masks)]
    );
    Ok(())
}
