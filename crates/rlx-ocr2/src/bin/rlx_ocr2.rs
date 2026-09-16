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

//! `rlx-ocr2` CLI — run the native OCR pipeline on an image (or a single text line).
//!
//!   rlx-ocr2 pack  <assets-dir> <out.rlxp> [--container flat|zip|dir]
//!   rlx-ocr2 image <pack.rlxp> <image>
//!   rlx-ocr2 line  <pack.rlxp> <line-image>
//!   rlx-ocr2 image <recipe.json> <det.safetensors> <rec.safetensors> <codemap.txt> <image> [ngram.bin] [lexicon.tsv]
//!   rlx-ocr2 line  <rec.safetensors> <codemap.txt> <line-image>
//!
//! A `.rlxp` carries every asset in one file; the loose-file forms take them separately,
//! where `ngram.bin` and/or `lexicon.tsv` enable beam-search correction.
//! Backend + timing are controlled by the `OCR2_*` environment knobs.

use anyhow::Result;
use rlx_ocr2::{Ocr2, Recognizer, preprocess};
use rlx_runtime::Device;
use std::path::Path;

fn usage() -> ! {
    eprintln!("usage:");
    #[cfg(feature = "rlxp")]
    {
        eprintln!("  rlx-ocr2 pack  <assets-dir> <out.rlxp> [--container flat|zip|dir]");
        eprintln!("  rlx-ocr2 image <pack.rlxp> <image>");
        eprintln!("  rlx-ocr2 line  <pack.rlxp> <line-image>");
    }
    eprintln!(
        "  rlx-ocr2 image <recipe.json> <det.safetensors> <rec.safetensors> <codemap.txt> <image> [ngram.bin] [lexicon.tsv]"
    );
    eprintln!("  rlx-ocr2 line  <rec.safetensors> <codemap.txt> <line-image>");
    eprintln!();
    eprintln!("env: OCR2_DEVICE=cpu|metal|mlx|cuda|gpu|vulkan|coreml   OCR2_TIMING=1");
    std::process::exit(2);
}

/// Print one line per detected text region.
fn print_lines(lines: &[rlx_ocr2::OcrLine]) {
    for line in lines {
        let (x0, y0, x1, y1) = line.bbox;
        println!("[{x0},{y0} {x1},{y1}]  {}", line.text);
    }
}

/// `OCR2_REPEAT=N` re-runs in-process (iter 1 = cold compile, 2+ = warm/cached).
fn repeats() -> usize {
    std::env::var("OCR2_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

#[cfg(feature = "rlxp")]
fn cmd_pack(a: &[String]) -> Result<()> {
    use rlx_pkg::ContainerKind;
    let container = match a.iter().position(|s| s == "--container") {
        Some(i) => match a.get(i + 1).map(String::as_str) {
            Some("flat") | None => ContainerKind::Flat,
            Some("zip") => ContainerKind::Zip,
            Some("dir") => ContainerKind::Dir,
            Some(other) => anyhow::bail!("unknown container {other}"),
        },
        None => ContainerKind::Flat,
    };
    rlx_ocr2::write_pack(Path::new(&a[2]), Path::new(&a[3]), container)?;
    eprintln!("wrote {}", a[3]);
    Ok(())
}

fn device() -> Device {
    match std::env::var("OCR2_DEVICE").as_deref() {
        Ok("metal") => Device::Metal,
        Ok("mlx") => Device::Mlx,
        Ok("gpu") | Ok("wgpu") => Device::Gpu,
        Ok("coreml") | Ok("ane") => Device::Ane,
        Ok("vulkan") => Device::Vulkan,
        Ok("cuda") => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        #[cfg(feature = "rlxp")]
        Some("pack") if a.len() >= 4 => cmd_pack(&a)?,
        #[cfg(feature = "rlxp")]
        Some("line") if a.len() == 4 => {
            let rec = Recognizer::load_pack(Path::new(&a[2]), device())?;
            let (luma, width) = preprocess::luma_line(Path::new(&a[3]))?;
            println!("{}", rec.recognize(&luma, width)?);
        }
        #[cfg(feature = "rlxp")]
        Some("image") if a.len() == 4 => {
            let pack = rlx_ocr2::Ocr2Pack::open(Path::new(&a[2]))?;
            eprintln!("[pack: correction={}]", pack.has_rescorer());
            let ocr = pack.into_pipeline(device())?;
            let mut lines = Vec::new();
            for _ in 0..repeats() {
                lines = ocr.recognize_image(Path::new(&a[3]))?;
            }
            print_lines(&lines);
        }
        Some("line") if a.len() == 5 => {
            let rec = Recognizer::load(Path::new(&a[2]), Path::new(&a[3]), device())?;
            let (luma, width) = preprocess::luma_line(Path::new(&a[4]))?;
            println!("{}", rec.recognize(&luma, width)?);
        }
        Some("image") if a.len() >= 7 => {
            let mut ocr = Ocr2::load(
                Path::new(&a[2]),
                Path::new(&a[3]),
                Path::new(&a[4]),
                Path::new(&a[5]),
                device(),
            )?;
            // optional scoring stack: a[7]=ngram.bin  a[8]=lexicon.tsv
            let ngram = a.get(7).map(Path::new).filter(|p| p.is_file());
            let lexicon = a.get(8).map(Path::new).filter(|p| p.is_file());
            if ngram.is_some() || lexicon.is_some() {
                ocr = ocr.with_rescorer(rlx_ocr2::Rescorer::load_en(ngram, lexicon)?);
                eprintln!(
                    "[correction: ngram={} lexicon={}]",
                    ngram.is_some(),
                    lexicon.is_some()
                );
            }
            let mut lines = Vec::new();
            for _ in 0..repeats() {
                lines = ocr.recognize_image(Path::new(&a[6]))?;
            }
            print_lines(&lines);
        }
        _ => usage(),
    }
    Ok(())
}
