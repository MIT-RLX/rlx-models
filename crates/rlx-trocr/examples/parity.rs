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

//! Parity against `scripts/trocr_reference.py` dumps.
//!
//! ```text
//! cargo run -p rlx-trocr --release --example parity -- MODEL_DIR REF_DIR [DEVICE]
//! ```
//!
//! Checks the processor output (`pixel_values`), the encoder states, the
//! decoder logits at every position teacher-forced on the reference's
//! greedy output, the greedy ids, and the decoded text.
//!
//! The encoder tolerance is looser than the logits': on a near-blank patch
//! the final LayerNorm (eps 1e-12) divides float noise by a tiny standard
//! deviation, so one or two token rows move by ~1e-3 relative while the
//! rest agree to ~1e-5 — and the decoder, which attends over all of them,
//! still matches to ~1e-5.

use anyhow::{Context, Result, bail};
use rlx_core::parity::Deviation;
use rlx_trocr::TrOcrRunner;
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
    let device = rlx_cli::parse_standard_device("trocr", args.get(2).map_or("cpu", |s| s))?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refd.join("meta.json"))?)?;
    let image = meta["image"].as_str().context("meta.image")?;
    let want_text = meta["text"].as_str().context("meta.text")?;
    let want_ids: Vec<u32> = meta["ids"]
        .as_array()
        .context("meta.ids")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();

    let t0 = std::time::Instant::now();
    let mut r = TrOcrRunner::open(&model, device)?;
    println!("load   {:?} (weights + compile)", t0.elapsed());

    let img = image::open(image)?.to_rgb8();
    let px = r.pixels(&img);
    let want_px = read_f32(&refd.join("pixels.bin"))?;
    let dp = Deviation::between(&px, &want_px);
    println!("pixels {dp}");

    let t0 = std::time::Instant::now();
    let enc = r.encode(&want_px)?;
    let t_enc = t0.elapsed();
    let de = Deviation::between(&enc, &read_f32(&refd.join("encoder.bin"))?);
    println!("enc    {de}");

    let want_logits = read_f32(&refd.join("logits.bin"))?;
    let vocab = r.config().decoder.vocab;
    let mut worst = Deviation::between(&[0.0], &[0.0]);
    for (pos, &tok) in want_ids[..want_ids.len() - 1].iter().enumerate() {
        let got = r.step(tok, pos)?;
        let d = Deviation::between(&got, &want_logits[pos * vocab..(pos + 1) * vocab]);
        if d.rel >= worst.rel {
            worst = d;
        }
    }
    println!("logits worst position: {worst}");

    r.encode(&want_px)?;
    let t0 = std::time::Instant::now();
    let ids = r.generate(64)?;
    let t_gen = t0.elapsed();
    println!("ids    rlx {ids:?}");
    println!("       ref {want_ids:?}");
    let rec = r.recognize(&img, 64)?;
    println!("text   rlx: {:?}", rec.text);
    println!("       ref: {want_text:?}");
    println!(
        "time   encode {t_enc:?}, generate {} tokens {t_gen:?}",
        ids.len() - 1
    );
    let ok = dp.max_abs <= 2.0 / 255.0 / 0.5 + 1e-5
        && de.is_within(5e-3)
        && worst.is_within(1e-3)
        && ids == want_ids
        && rec.text == want_text;
    if !ok {
        bail!("parity FAILED");
    }
    println!("parity OK on {device:?}");
    Ok(())
}
