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

//! Per-utterance latency for a long-lived [`S1Runner`] — the shape a dictation
//! app actually has: build once, normalize many.
//!
//! Reports first-call vs steady-state separately, because the first call through
//! any given prompt length pays a graph compile that later calls of that length
//! do not.
//!
//! ```text
//! cargo run --release -p rlx-s1 --features metal --example s1_bench -- \
//!     --weights /Volumes/FOUR/weights/lm/s1-mini --device metal --repeats 3
//! ```

use anyhow::{Result, anyhow};
use rlx_s1::{Controls, S1Runner};
use std::time::Instant;

const UTTERANCES: &[&str] = &[
    "so um i need to like send the the report by uh friday no wait make that thursday",
    "i think the answer is forty two no sorry forty three",
    "let's meet at half past two tomorrow uh actually make it three fifteen p m",
    "send it to support at superwhisper dot com",
];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights = String::new();
    let mut device = "cpu".to_string();
    let mut repeats = 3usize;
    let mut bucket: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |i: &mut usize| -> Result<String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--weights" => weights = need(&mut i)?,
            "--device" => device = need(&mut i)?,
            "--repeats" => repeats = need(&mut i)?.parse()?,
            "--prefill-bucket" => bucket = Some(need(&mut i)?.parse()?),
            other => return Err(anyhow!("unknown arg {other:?}")),
        }
        i += 1;
    }
    if weights.is_empty() {
        return Err(anyhow!("--weights <PATH> is required"));
    }

    let device = rlx_cli::parse_standard_device("s1", &device)?;
    let t_build = Instant::now();
    let mut b = S1Runner::builder().weights(&weights).device(device);
    if let Some(step) = bucket {
        b = b.prefill_bucket(step);
    }
    let mut runner = b.build()?;
    println!(
        "build: {:.2}s ({device:?})",
        t_build.elapsed().as_secs_f64()
    );

    for (u, text) in UTTERANCES.iter().enumerate() {
        let prompt_len = runner.encode_prompt(text, Controls::new())?.len();
        for r in 0..repeats {
            let mut n = 0usize;
            let t = Instant::now();
            let out = runner.normalize_pass(text, Controls::new(), |_| n += 1)?;
            let dt = t.elapsed().as_secs_f64();
            let tag = if r == 0 { "first" } else { "warm " };
            println!(
                "u{u} {tag} prompt={prompt_len:3}  out={n:3} tok  {dt:6.2}s  \
                 {:5.1} tok/s   {out:?}",
                n as f64 / dt.max(1e-9)
            );
        }
    }
    Ok(())
}
