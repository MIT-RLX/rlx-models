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

//! End-to-end VLASH action prediction example (config + preprocessing demo).

use anyhow::Result;
use rlx_vlash::{VlashConfig, preprocess::resize_with_pad_normalize};

fn main() -> Result<()> {
    let cfg = VlashConfig::pi05();
    println!(
        "VLASH {} — prefix width {}, suffix width {}, {} joint layers, chunk {}",
        cfg.variant.as_str(),
        cfg.prefix_width(),
        cfg.suffix_width(),
        cfg.vlm.layers,
        cfg.chunk_size
    );

    // Demo the image preprocessing on a synthetic 3×120×160 image.
    let (h, w) = (120usize, 160usize);
    let chw: Vec<f32> = (0..3 * h * w).map(|i| (i % 251) as f32 / 251.0).collect();
    let nchw = resize_with_pad_normalize(&chw, h, w, cfg.image_size);
    println!(
        "preprocessed image → {} values in [{:.3}, {:.3}]",
        nchw.len(),
        nchw.iter().cloned().fold(f32::INFINITY, f32::min),
        nchw.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    Ok(())
}
