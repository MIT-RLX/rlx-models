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

//! Parity against `scripts/dan_reference.py` dumps.
//!
//! ```text
//! cargo run -p rlx-dan --release --example parity -- dan_rimes_page.pt REF_DIR [DEVICE]
//! ```
//!
//! Checks the network input, the encoder features, the decision-layer
//! logits at every step teacher-forced on the reference's greedy output,
//! and the greedy ids and text.

use anyhow::{Context, Result, bail};
use rlx_core::parity::Deviation;
use rlx_dan::DanRunner;
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
        bail!("usage: parity CHECKPOINT REF_DIR [DEVICE]");
    }
    let model = PathBuf::from(&args[0]);
    let refd = PathBuf::from(&args[1]);
    let device = rlx_cli::parse_standard_device("dan", args.get(2).map_or("cpu", |s| s))?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refd.join("meta.json"))?)?;
    let image = meta["image"].as_str().context("meta.image")?;
    let want_text = meta["text"].as_str().context("meta.text")?;
    let want_ids: Vec<usize> = meta["ids"]
        .as_array()
        .context("meta.ids")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let num = |k: &str| -> Option<[f32; 3]> {
        let a = meta[k].as_array()?;
        Some([0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
    };

    let mut r = DanRunner::open(&model, device)?;
    if let (Some(m), Some(s)) = (num("mean"), num("std")) {
        r = r.with_normalization(m, s);
    }
    let img = image::open(image)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let x = r.input(&img);
    let dx = Deviation::between(&x, &read_f32(&refd.join("input.bin"))?);
    println!("input  [{h}x{w}] {dx}");

    let t0 = std::time::Instant::now();
    let feats = r.encode(&x, h, w)?;
    let t_enc = t0.elapsed();
    let df = Deviation::between(&feats, &read_f32(&refd.join("features.bin"))?);
    println!("feats  {df}");

    let want_logits = read_f32(&refd.join("logits.bin"))?;
    let classes = r.charset().len() + 1;
    let start = r.charset().len() + 1;
    let mut worst = Deviation::between(&[0.0], &[0.0]);
    let mut prev = start;
    let t0 = std::time::Instant::now();
    for (pos, &tok) in want_ids.iter().enumerate() {
        let got = r.step(prev, pos)?;
        let d = Deviation::between(&got, &want_logits[pos * classes..(pos + 1) * classes]);
        if d.rel >= worst.rel {
            worst = d;
        }
        prev = tok;
    }
    let t_dec = t0.elapsed();
    println!("logits worst of {} steps: {worst}", want_ids.len());

    // Same cap as the reference: it may stop on the limit, not the end token.
    let rec = r.recognize(&img, want_ids.len())?;
    let same_ids = rec.ids == want_ids;
    println!(
        "ids    {} of {} identical{}",
        rec.ids
            .iter()
            .zip(&want_ids)
            .take_while(|(a, b)| a == b)
            .count(),
        want_ids.len(),
        if same_ids { "" } else { " (diverged)" }
    );
    println!("text   rlx: {:?}", rec.raw);
    println!("       ref: {want_text:?}");
    println!(
        "time   encode {t_enc:?} (incl. compile), {} steps {t_dec:?}",
        want_ids.len()
    );
    if !(dx.is_within(1e-6)
        && df.is_within(1e-3)
        && worst.is_within(1e-3)
        && same_ids
        && rec.raw == want_text)
    {
        bail!("parity FAILED");
    }
    println!("parity OK on {device:?}");
    Ok(())
}
