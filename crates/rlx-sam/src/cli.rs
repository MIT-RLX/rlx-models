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

// RLX CLI for SAM 1
use crate::{Sam, SamConfig};
use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_sam_device, req};
use std::path::PathBuf;

pub fn run_sam1(args: &[String]) -> Result<()> {
    run(args)
}

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
                    bail!("--point expects X,Y, got {v}");
                }
                point = Some((
                    parts[0].trim().parse().map_err(|_| anyhow!("--point X"))?,
                    parts[1].trim().parse().map_err(|_| anyhow!("--point Y"))?,
                ));
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-sam1 — SAM v1; flags: --weights --device (cpu|metal|mlx|cuda|rocm|gpu|vulkan|tpu) --point --dry"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_sam_device("sam", &device)?;
    let weights_str = weights.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?;
    let variant = rlx_ir::env::var("RLX_SAM_VARIANT").unwrap_or_else(|| "vit_b".to_string());
    let cfg = match variant.as_str() {
        "vit_b" => SamConfig::vit_b(),
        "vit_l" => SamConfig::vit_l(),
        "vit_h" => SamConfig::vit_h(),
        other => bail!("RLX_SAM_VARIANT must be vit_b|vit_l|vit_h, got {other}"),
    };
    eprintln!("[rlx-sam1] weights={weights:?} device={device:?}");
    if dry {
        return Ok(());
    }
    let h_in = 1024usize;
    let w_in = 1024usize;
    let mut rgb = vec![0u8; h_in * w_in * 3];
    for y in 0..h_in {
        for x in 0..w_in {
            let base = (y * w_in + x) * 3;
            rgb[base] = (x * 255 / w_in) as u8;
            rgb[base + 1] = (y * 255 / h_in) as u8;
            rgb[base + 2] = ((x + y) * 127 / (h_in + w_in)) as u8;
        }
    }
    let (cx, cy) = point.unwrap_or((w_in as f32 / 2.0, h_in as f32 / 2.0));
    let mut sam = Sam::from_safetensors_on(weights_str, cfg, device)?;
    let (pred, _) = sam.forward(
        &rgb,
        h_in,
        w_in,
        Some((&[cx, cy], &[1.0f32])),
        None,
        None,
        true,
    )?;
    eprintln!(
        "[rlx-sam1] masks={} side={} iou={:?}",
        pred.num_masks,
        pred.mask_side,
        &pred.iou_pred[..pred.iou_pred.len().min(pred.num_masks)]
    );
    Ok(())
}

// Stubs for multiplexer — use `rlx-sam2` / `rlx-sam3` binaries.
pub fn run_sam2(_args: &[String]) -> Result<()> {
    bail!("use `rlx-sam2` (or `rlx-run sam2`)")
}
pub fn run_sam3(_args: &[String]) -> Result<()> {
    bail!("use `rlx-sam3` (or `rlx-run sam3`)")
}
