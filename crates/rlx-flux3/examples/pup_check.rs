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

//! Does the upsampler's fast decode pick the reference's tokens?
//!
//! The presence penalty is subtracted from logit *values*, so a decode path
//! whose logits are off by a scale factor would pick different tokens than
//! the reference even though plain greedy (an argmax) agrees. This runs the
//! fast path ([`Upsampler::generate_ids`]: prefill, then bucketed decode with
//! the penalty) and a slow one that re-prefills the whole context every
//! token and applies the penalty to those (correctly scaled) logits, and
//! reports where they part.
//!
//! `cargo run -p rlx-flux3 --release --features metal --example pup_check -- \
//!     --pup weights/lm/flux3-klein-pup --prompt "a red fox" --tokens 24 --device metal`

use anyhow::{Context, Result, anyhow};
use rlx_flux3::pup::{T2I_PRESENCE_PENALTY, Upsampler, UpsamplerPaths};
use rlx_qwen35::Qwen35Runner;
use std::str::FromStr;

fn main() -> Result<()> {
    let mut pup = None;
    let mut prompt = "a red fox asleep in the snow".to_string();
    let mut tokens = 24usize;
    let mut device = "cpu".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().ok_or_else(|| anyhow!("{a} needs a value"));
        match a.as_str() {
            "--pup" => pup = Some(v()?),
            "--prompt" => prompt = v()?,
            "--tokens" => tokens = v()?.parse()?,
            "--device" => device = v()?,
            other => return Err(anyhow!("unknown argument {other}")),
        }
    }
    let paths = UpsamplerPaths::discover(pup.context("--pup is required")?.as_ref())?;
    let dev = rlx_runtime::Device::from_str(&device).map_err(|e| anyhow!("{e}"))?;

    let mut up = Upsampler::open(&paths, dev)?;
    up.max_new_tokens = tokens;
    let ids = up.prompt_ids("t2i", &prompt)?;
    println!("prompt: {} tokens", ids.len());
    let t0 = std::time::Instant::now();
    let fast = up.generate_ids(&ids, T2I_PRESENCE_PENALTY)?;
    println!(
        "fast: {} tokens in {:.1}s",
        fast.len(),
        t0.elapsed().as_secs_f32()
    );
    drop(up);

    let mut slow_runner = Qwen35Runner::builder()
        .weights(&paths.weights)
        .device(dev)
        .max_seq(ids.len() + tokens + 8)
        .skip_auto_mmproj(true)
        .packed_weights(paths.weights.extension().is_some_and(|e| e == "gguf"))
        .build()?;
    let mut context = ids.clone();
    let mut slow: Vec<u32> = Vec::new();
    let mut margins = Vec::new();
    for _ in 0..fast.len() {
        let mut l = slow_runner.predict_logits(&context)?.logits;
        for &t in &slow {
            l[t as usize] -= T2I_PRESENCE_PENALTY;
        }
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
        margins.push(l[idx[0]] - l[idx[1]]);
        let t = idx[0] as u32;
        slow.push(t);
        context.push(t);
    }
    let agree = fast.iter().zip(&slow).take_while(|(a, b)| a == b).count();
    println!("fast: {fast:?}\nslow: {slow:?}");
    println!(
        "agree on the first {agree} of {} tokens; smallest top-2 margin {:.3}",
        fast.len(),
        margins.iter().copied().fold(f32::INFINITY, f32::min)
    );
    Ok(())
}
