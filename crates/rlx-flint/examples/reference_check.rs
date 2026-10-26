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

//! Compare the paged runner with a `scripts/flint_reference.py` dump, at every
//! position of its prompt.
//!
//! ```text
//! python3 scripts/flint_reference.py --model-dir weights/lm/flint \
//!     --text "Write a Python function that reverses a string." --chat \
//!     --out /tmp/flint-real-ref.safetensors [--layers N]
//! cargo run -p rlx-flint --release --example reference_check [--features metal] -- \
//!     weights/lm/flint /tmp/flint-real-ref.safetensors [--device metal] \
//!     [--layers N] [--chunk N]
//! ```

use anyhow::{Context, Result, bail};
use rlx_flint::{FlintRunner, RunnerOptions};
use rlx_runtime::Device;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!(
            "usage: reference_check MODEL_DIR REF.safetensors [--device D] [--layers N] [--chunk N]"
        );
    }
    let (dir, refp) = (&args[0], &args[1]);
    let mut opts = RunnerOptions::new(Device::Cpu);
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        let v = it.next().context("flag needs a value")?;
        match a.as_str() {
            "--device" => opts.device = rlx_cli::parse_device(v)?,
            "--layers" => opts.layers = Some(v.parse()?),
            "--chunk" => opts.prefill_chunk = v.parse()?,
            other => bail!("unknown flag {other}"),
        }
    }
    let bytes = std::fs::read(refp).with_context(|| refp.clone())?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let ids: Vec<u32> = st
        .tensor("ids")?
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect();
    let want: Vec<f32> = st
        .tensor("logits")?
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();

    opts.max_context = ids.len().next_power_of_two().max(64);
    let t0 = std::time::Instant::now();
    let mut run = FlintRunner::open(dir, opts)?;
    eprintln!("compiled in {:.1}s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let got = run.all_logits(&ids)?;
    eprintln!(
        "{} positions in {:.1}s",
        ids.len(),
        t1.elapsed().as_secs_f64()
    );

    let v = run.config().vocab_size;
    let rows = want.len() / v;
    let off = (ids.len() - rows) * v; // a --last-only reference holds the tail
    let mut worst = 0.0f32;
    let mut agree = 0;
    for r in 0..rows {
        let a = &got[off + r * v..off + (r + 1) * v];
        let b = &want[r * v..(r + 1) * v];
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let (dot, na, nb) = a
            .iter()
            .zip(b)
            .fold((0.0f64, 0.0f64, 0.0f64), |(d, x, y), (p, q)| {
                (
                    d + (*p as f64) * (*q as f64),
                    x + (*p as f64).powi(2),
                    y + (*q as f64).powi(2),
                )
            });
        let (ta, tb) = (
            rlx_runtime::search::argmax_u32(a),
            rlx_runtime::search::argmax_u32(b),
        );
        agree += (ta == tb) as usize;
        worst = worst.max(d);
        println!(
            "pos {:3}: max|Δ| {d:.3e} (|logit|max {scale:.1})  cos {:.7}  top1 rlx {ta} ref {tb}",
            ids.len() - rows + r,
            dot / (na.sqrt() * nb.sqrt())
        );
    }
    println!("worst max|Δ| {worst:.3e}; top-1 agrees at {agree}/{rows}");
    let t = run.timings();
    eprintln!(
        "layers {:.2}s  experts {:.2}s ({} matrices)  head {:.2}s",
        t.layers.as_secs_f64(),
        t.experts.as_secs_f64(),
        t.expert_reads,
        t.head.as_secs_f64()
    );
    Ok(())
}
