// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Unlabeled image loading for SSL calibration / continual pre-training.

use std::path::Path;

use anyhow::{Result, bail};

use crate::snapvit::CalibImage;

/// Load up to `max` decodable images (jpeg/png) from `dir` as [`CalibImage`]s.
pub fn load_images(dir: &Path, max: usize) -> Result<Vec<CalibImage>> {
    if !dir.is_dir() {
        bail!("data path is not a directory: {}", dir.display());
    }
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for path in entries {
        if out.len() >= max {
            break;
        }
        let ok_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
            .unwrap_or(false);
        if !ok_ext {
            continue;
        }
        if let Ok(img) = image::open(&path) {
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            out.push(CalibImage {
                rgb: rgb.into_raw(),
                h: h as usize,
                w: w as usize,
            });
        }
    }
    if out.is_empty() {
        bail!("no decodable jpg/png images found in {}", dir.display());
    }
    Ok(out)
}

/// Deterministic synthetic images (fallback when no `--data` directory is given).
pub fn synthetic_images(n: usize, side: usize) -> Vec<CalibImage> {
    (0..n)
        .map(|i| {
            let mut s = (i as u32).wrapping_mul(2654435761).wrapping_add(1);
            let rgb = (0..side * side * 3)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (s >> 24) as u8
                })
                .collect();
            CalibImage {
                rgb,
                h: side,
                w: side,
            }
        })
        .collect()
}
