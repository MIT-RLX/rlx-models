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

//! Background-removal models compared against the vendor's Vision framework.
//!
//! For each `image.png` with `image-vision-mask.png` (Vision's
//! `generateScaledMaskForImage`, the reference), every matting backend's alpha
//! is scored: mean |Δα|, IoU of the α>0.5 masks, and boundary F-score (edge
//! pixels matched within 2 px). Writes side-by-side sheets with `--sheet DIR`,
//! every alpha with `--dump DIR`; `--feather S,S,…` adds the subject-lift port at
//! other feather widths.
//!
//! `cargo run -p rlx-genmoji --release --features metal --example matte_compare -- \
//!    --device metal [--sheet DIR] [--dump DIR] [--feather S,…] a.png a-vision-mask.png …`

use anyhow::Context;
use image::{GrayImage, RgbImage};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::str::FromStr;

fn edges(m: &[bool], w: usize, h: usize) -> Vec<bool> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            m[i] && [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ]
            .iter()
            .any(|&(a, b)| a >= w || b >= h || !m[b * w + a])
        })
        .collect()
}

fn near(e: &[bool], w: usize, h: usize, r: isize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for (i, &v) in e.iter().enumerate() {
        if !v {
            continue;
        }
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for dy in -r..=r {
            for dx in -r..=r {
                let (a, b) = (x + dx, y + dy);
                if a >= 0 && b >= 0 && (a as usize) < w && (b as usize) < h {
                    out[b as usize * w + a as usize] = true;
                }
            }
        }
    }
    out
}

/// (MAE, IoU, boundary F) of `a` against reference `r`.
fn score(a: &[f32], r: &[f32], w: usize, h: usize) -> (f64, f64, f64) {
    let mae = a
        .iter()
        .zip(r)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>()
        / a.len() as f64;
    let (ma, mr): (Vec<bool>, Vec<bool>) = (
        a.iter().map(|&v| v > 0.5).collect(),
        r.iter().map(|&v| v > 0.5).collect(),
    );
    let inter = ma.iter().zip(&mr).filter(|(x, y)| **x && **y).count() as f64;
    let uni = ma.iter().zip(&mr).filter(|(x, y)| **x || **y).count() as f64;
    let (ea, er) = (edges(&ma, w, h), edges(&mr, w, h));
    let (na, nr) = (near(&ea, w, h, 2), near(&er, w, h, 2));
    let tp_p = ea.iter().zip(&nr).filter(|(e, n)| **e && **n).count() as f64;
    let tp_r = er.iter().zip(&na).filter(|(e, n)| **e && **n).count() as f64;
    let (p, rc) = (
        tp_p / ea.iter().filter(|&&e| e).count().max(1) as f64,
        tp_r / er.iter().filter(|&&e| e).count().max(1) as f64,
    );
    (
        mae,
        inter / uni.max(1.0),
        if p + rc > 0.0 {
            2.0 * p * rc / (p + rc)
        } else {
            0.0
        },
    )
}

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut device = Device::Cpu;
    let mut sheet: Option<PathBuf> = None;
    let mut feathers: Vec<f32> = Vec::new();
    let mut dump: Option<PathBuf> = None;
    while args.first().is_some_and(|a| a.starts_with("--")) {
        match args[0].as_str() {
            "--device" => {
                device = Device::from_str(&args[1]).map_err(|e| anyhow::anyhow!("{e}"))?
            }
            "--sheet" => sheet = Some(args[1].clone().into()),
            // Every backend's alpha as `DIR/<image>.<backend>.png`.
            "--dump" => dump = Some(args[1].clone().into()),
            // Extra subject-lift feather widths to compare, `S,S,…`.
            "--feather" => {
                for f in args[1].split(',') {
                    feathers.push(f.parse().context("--feather")?);
                }
            }
            o => anyhow::bail!("unknown flag {o}"),
        }
        args.drain(..2);
    }
    let items: Vec<(String, RgbImage, Vec<f32>)> = args
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
            let name = PathBuf::from(&p[0])
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            (
                name,
                img,
                m.pixels().map(|v| v.0[0] as f32 / 255.0).collect(),
            )
        })
        .collect();

    type Backend = Box<dyn FnMut(&RgbImage) -> anyhow::Result<Vec<f32>>>;
    let mut backends: Vec<(&str, Backend)> = Vec::new();
    let mut lift = rlx_subject_lift::SubjectLift::open(None, device, Default::default())?;
    backends.push((
        "vision-port",
        Box::new(move |img| Ok(lift.run(img)?.alpha(img.width(), img.height()))),
    ));
    for feather in feathers {
        let opts = rlx_subject_lift::LiftOptions {
            feather,
            ..Default::default()
        };
        let mut lift = rlx_subject_lift::SubjectLift::open(None, device, opts)?;
        let name: &'static str = Box::leak(format!("vision-port~{feather}").into_boxed_str());
        backends.push((
            name,
            Box::new(move |img| Ok(lift.run(img)?.alpha(img.width(), img.height()))),
        ));
    }
    let lite = PathBuf::from("weights/image/birefnet/BiRefNet_lite");
    let full = PathBuf::from("weights/image/birefnet/BiRefNet");
    for (name, dir, side) in [
        ("birefnet-lite@512", &lite, 512),
        ("birefnet-lite@1024", &lite, 1024),
        ("birefnet@1024", &full, 1024),
    ] {
        if dir.join("model.safetensors").is_file() {
            let t0 = std::time::Instant::now();
            let mut m = rlx_birefnet::BiRefNet::open(dir, side, device)?;
            eprintln!("{name}: compiled in {:.1}s", t0.elapsed().as_secs_f32());
            backends.push((name, Box::new(move |img| m.alpha(img))));
        }
    }
    let cut = rlx_genmoji::Cutout::default();
    backends.push((
        "classic",
        Box::new(move |img| {
            let m = rlx_genmoji::remove_white_background(img, cut);
            Ok(m.pixels().map(|p| p.0[3] as f32 / 255.0).collect())
        }),
    ));

    println!(
        "{:<20} {:>8} {:>8} {:>10} {:>9}",
        "backend", "MAE", "IoU", "boundaryF", "ms/img"
    );
    let mut alphas: Vec<Vec<Vec<f32>>> = vec![Vec::new(); backends.len()];
    for (bi, (name, f)) in backends.iter_mut().enumerate() {
        let (mut mae, mut iou, mut bf, mut ms) = (0.0, 0.0, 0.0, 0.0);
        for (_, img, r) in &items {
            let t0 = std::time::Instant::now();
            let a = f(img)?;
            ms += t0.elapsed().as_secs_f64() * 1e3;
            let (m, i, b) = score(&a, r, img.width() as usize, img.height() as usize);
            mae += m;
            iou += i;
            bf += b;
            alphas[bi].push(a);
        }
        let n = items.len() as f64;
        println!(
            "{name:<20} {:>8.4} {:>8.4} {:>10.4} {:>9.0}",
            mae / n,
            iou / n,
            bf / n,
            ms / n
        );
    }

    if let Some(dir) = sheet {
        std::fs::create_dir_all(&dir)?;
        for (k, (name, img, r)) in items.iter().enumerate() {
            let (w, h) = img.dimensions();
            let cols = alphas.len() + 1;
            let mut out = RgbImage::from_pixel(w * cols as u32, h, image::Rgb([40, 40, 60]));
            let mut put = |col: usize, a: &[f32]| {
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) as usize;
                        let p = img.get_pixel(x, y).0;
                        let bg = [40.0, 40.0, 60.0];
                        let c: [u8; 3] = std::array::from_fn(|c| {
                            (p[c] as f32 * a[i] + bg[c] * (1.0 - a[i])) as u8
                        });
                        out.put_pixel(col as u32 * w + x, y, image::Rgb(c));
                    }
                }
            };
            put(0, r);
            for (bi, a) in alphas.iter().enumerate() {
                put(bi + 1, &a[k]);
            }
            out.save(dir.join(format!("{name}.png")))?;
        }
        let names: Vec<&str> = std::iter::once("vision")
            .chain(backends.iter().map(|(n, _)| *n))
            .collect();
        eprintln!("sheets: columns = {names:?}");
    }
    if let Some(dir) = dump {
        std::fs::create_dir_all(&dir)?;
        for (bi, (bname, _)) in backends.iter().enumerate() {
            for (k, (name, img, _)) in items.iter().enumerate() {
                let (w, h) = img.dimensions();
                let a = &alphas[bi][k];
                GrayImage::from_fn(w, h, |x, y| {
                    image::Luma([(a[(y * w + x) as usize] * 255.0).round() as u8])
                })
                .save(dir.join(format!("{name}.{bname}.png")))?;
            }
        }
    }
    Ok(())
}
