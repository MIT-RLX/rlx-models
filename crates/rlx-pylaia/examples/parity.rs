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

//! Parity against `scripts/pylaia_reference.py` dumps.
//!
//! ```text
//! cargo run -p rlx-pylaia --release --example parity -- MODEL_DIR REF_DIR [DEVICE]
//! ```
//!
//! Checks, in order: the preprocessed input (from the image file the
//! reference read), the logits on the reference's exact input tensor, and
//! the greedy transcription of the image read here.
//!
//! The input check allows one gray level: the `image` crate's JPEG decoder
//! and Pillow's libjpeg round a fraction of a percent of pixels apart. The
//! network is then fed the reference tensor so that difference does not
//! leak into the logits comparison.

use anyhow::{Context, Result, bail};
use rlx_core::parity::Deviation;
use rlx_pylaia::{PylaiaRunner, preprocess};
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: parity MODEL_DIR REF_DIR [DEVICE]");
    }
    let model = PathBuf::from(&args[0]);
    let refd = PathBuf::from(&args[1]);
    let device = rlx_cli::parse_standard_device("pylaia", args.get(2).map_or("cpu", |s| s))?;

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refd.join("meta.json"))?)?;
    let image = meta["image"].as_str().context("meta.image")?;
    let want_text = meta["text"].as_str().context("meta.text")?;

    let mut runner = PylaiaRunner::open(&model, device)?;
    let gray = preprocess::to_gray(&image::open(image)?);
    let min_w = runner.config().min_width(gray.height() as usize);
    let (x, h, w) = preprocess::to_input(&gray, min_w);
    let want_x = read_f32(&refd.join("input.bin"))?;
    let dx = Deviation::between(&x, &want_x);
    println!("input  [{h}x{w}]  {dx}");

    let t0 = std::time::Instant::now();
    let logits = runner.logits(&want_x, h, w)?;
    let first = t0.elapsed();
    let t1 = std::time::Instant::now();
    let _ = runner.logits(&want_x, h, w)?;
    let warm = t1.elapsed();
    let want = read_f32(&refd.join("logits.bin"))?;
    let dl = Deviation::between(&logits, &want);
    println!("logits {dl}");

    let rec = runner.recognize_gray(&gray)?;
    println!("text   rlx: {:?}", rec.text);
    println!("       ref: {want_text:?}");
    println!("time   first {first:?} (compile + run), warm {warm:?}");

    let ok = dx.max_abs <= 1.0 / 255.0 + 1e-6 && dl.is_within(1e-4) && rec.text == want_text;
    if !ok {
        bail!("parity FAILED");
    }
    println!("parity OK on {device:?}");
    Ok(())
}
