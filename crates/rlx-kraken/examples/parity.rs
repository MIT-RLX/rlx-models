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

//! Parity against `scripts/kraken_reference.py` dumps.
//!
//! ```text
//! cargo run -p rlx-kraken --release --example parity -- MODEL.mlmodel REF_DIR [DEVICE]
//! ```
//!
//! Checks the preprocessed line against Kraken's (Pillow-exact Lanczos, so
//! within one gray level — a JPEG decoder may round a pixel apart), the
//! logits on the reference's exact input tensor, and the transcription
//! produced from the image file through this crate's own preprocessing.

use anyhow::{Context, Result, bail};
use rlx_core::parity::Deviation;
use rlx_kraken::{KrakenRunner, preprocess};
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
        bail!("usage: parity MODEL.mlmodel REF_DIR [DEVICE]");
    }
    let model = PathBuf::from(&args[0]);
    let refd = PathBuf::from(&args[1]);
    let device = rlx_cli::parse_standard_device("kraken", args.get(2).map_or("cpu", |s| s))?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refd.join("meta.json"))?)?;
    let image = meta["image"].as_str().context("meta.image")?;
    let want_text = meta["text"].as_str().context("meta.text")?;
    let shape: Vec<usize> = meta["input"]
        .as_array()
        .context("meta.input")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let (h, w) = (shape[1], shape[2]);

    let mut runner = KrakenRunner::open(&model, device)?;
    let want_x = read_f32(&refd.join("input.bin"))?;
    let img = image::open(image)?;
    let inp = runner.spec().input;
    let (x, xh, xw) = preprocess::line_tensor(&img, inp.channels, inp.height, runner.padding());
    if (xh, xw) != (h, w) {
        bail!("input size {xh}x{xw}, reference {h}x{w}");
    }
    let dx = Deviation::between(&x, &want_x);
    println!("input  [{h}x{w}]  {dx}");

    let t0 = std::time::Instant::now();
    let logits = runner.logits(&want_x, h, w)?;
    let first = t0.elapsed();
    let t1 = std::time::Instant::now();
    let _ = runner.logits(&want_x, h, w)?;
    let warm = t1.elapsed();
    // The reference stores [classes, frames]; the graph emits [frames, classes].
    let want = read_f32(&refd.join("logits.bin"))?;
    let classes = meta["logits"][0].as_u64().context("meta.logits")? as usize;
    let frames = want.len() / classes;
    let mut want_fc = vec![0f32; want.len()];
    for c in 0..classes {
        for t in 0..frames {
            want_fc[t * classes + c] = want[c * frames + t];
        }
    }
    let dl = Deviation::between(&logits, &want_fc);
    println!("logits {dl}");

    let rec = runner.recognize(&img)?;
    println!("text   rlx: {:?}", rec.text);
    println!("       ref: {want_text:?}");
    println!("time   first {first:?} (compile + run), warm {warm:?}");
    if !(dx.max_abs <= 1.0 / 255.0 + 1e-6 && dl.is_within(1e-4) && rec.text == want_text) {
        bail!("parity FAILED");
    }
    println!("parity OK on {device:?}");
    Ok(())
}
