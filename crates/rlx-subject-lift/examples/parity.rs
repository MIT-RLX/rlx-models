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

//! Parity against the Vision framework: for `image.png` + `image-vision-mask.png`
//! (Vision's `generateScaledMaskForImage`, written by a small Swift tool), run
//! every pixel-encoding / channel-order / upsample variant and report how
//! closely its saliency matches the vendor's mask.
//!
//! `cargo run -p rlx-subject-lift --release --example parity -- [--device D] a.png a-vision-mask.png …`

use rlx_runtime::Device;
use rlx_subject_lift::{InputNorm, LiftOptions, SubjectLift, Upsample};
use std::str::FromStr;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut device = Device::Cpu;
    if args.first().map(String::as_str) == Some("--device") {
        device = Device::from_str(&args[1]).map_err(|e| anyhow::anyhow!("{e}"))?;
        args.drain(..2);
    }
    let pairs: Vec<(image::RgbImage, Vec<f32>)> = args
        .chunks(2)
        .map(|p| {
            let img = image::open(&p[0]).unwrap().to_rgb8();
            let m = image::open(&p[1]).unwrap().to_luma8();
            let m = image::imageops::resize(
                &m,
                img.width(),
                img.height(),
                image::imageops::FilterType::Triangle,
            );
            (img, m.pixels().map(|v| v.0[0] as f32 / 255.0).collect())
        })
        .collect();
    let only: Option<String> = std::env::var("VARIANT").ok();
    for up in [Upsample::Bilinear, Upsample::Nearest] {
        for norm in [
            InputNorm::Unit,
            InputNorm::Byte,
            InputNorm::Signed,
            InputNorm::ImageNet,
        ] {
            for bgr in [false, true] {
                let tag = format!("{up:?}/{norm:?}/{}", if bgr { "BGR" } else { "RGB" });
                if only.as_ref().is_some_and(|o| !tag.contains(o.as_str())) {
                    continue;
                }
                let mut lift = SubjectLift::open(
                    None,
                    device,
                    LiftOptions {
                        norm,
                        bgr,
                        upsample: up,
                        ..Default::default()
                    },
                )?;
                let (mut mae, mut corr) = (0.0f64, 0.0f64);
                let mut gates = Vec::new();
                for (img, reference) in &pairs {
                    let l = lift.run(img)?;
                    gates.push(l.gating);
                    let ours = l.alpha(img.width(), img.height());
                    let n = ours.len() as f64;
                    let (ma, mb) = (
                        ours.iter().map(|&v| v as f64).sum::<f64>() / n,
                        reference.iter().map(|&v| v as f64).sum::<f64>() / n,
                    );
                    let (mut sab, mut saa, mut sbb, mut e) = (0.0, 0.0, 0.0, 0.0);
                    for (&a, &b) in ours.iter().zip(reference) {
                        let (da, db) = (a as f64 - ma, b as f64 - mb);
                        sab += da * db;
                        saa += da * da;
                        sbb += db * db;
                        e += (a as f64 - b as f64).abs();
                    }
                    corr += sab / (saa * sbb).sqrt().max(1e-12);
                    mae += e / n;
                }
                let k = pairs.len() as f64;
                println!(
                    "{tag:<28} corr {:.4}  MAE {:.4}  gating {:?}",
                    corr / k,
                    mae / k,
                    gates
                        .iter()
                        .map(|g| (g * 100.0).round() / 100.0)
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    Ok(())
}
