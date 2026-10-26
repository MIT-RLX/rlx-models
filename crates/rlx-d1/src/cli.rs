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

//! `rlx-d1` — one System One decision from the command line.
//!
//! ```text
//! rlx-d1 --weights weights/lm/d1-3b [--device cpu|metal|mlx|…]
//!        [--prefer Q4_K_M] [--mmproj mmproj-d1-3B-Q8_0.gguf]
//!        (--state TEXT | --state-json JSON | --state-file PATH)
//!        (--questions JSON | --questions-file PATH)
//!        [--image PATH]… [--timing]
//! ```
//!
//! `--questions` is an object of named Decision Index questions. The
//! response is d1's `system_one` JSON on stdout. `--weights` is the HF
//! directory, a quantized text GGUF, or a directory of GGUFs (`--prefer`
//! picks one; see [`crate::D1::open_with`]).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use rlx_runtime::Device;

use crate::{D1, Json, OpenOptions, RgbImage};

const USAGE: &str = "usage: rlx-d1 --weights DIR|FILE.gguf [--device DEV] \
[--prefer QUANT] [--mmproj FILE.gguf] [--weight-storage auto|f32|f16|bf16] \
[--dequant-cache off|unbounded|BYTES] \
(--state TEXT | --state-json JSON | --state-file PATH) \
(--questions JSON | --questions-file PATH) [--image PATH]... [--timing]";

pub fn run(argv: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut state = Json::Null;
    let mut questions: Option<Json> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut timing = false;
    let mut opts = OpenOptions::default();
    let mut it = argv.iter();
    let value = |it: &mut std::slice::Iter<'_, String>, flag: &str| -> Result<String> {
        it.next()
            .cloned()
            .ok_or_else(|| anyhow!("{flag} needs a value\n{USAGE}"))
    };
    let parse =
        |text: &str, what: &str| Json::parse(text).with_context(|| format!("parsing {what}"));
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--weights" => weights = Some(value(&mut it, arg)?.into()),
            "--device" => device = rlx_cli::parse_device(&value(&mut it, arg)?)?,
            "--state" => state = Json::Str(value(&mut it, arg)?),
            "--state-json" => state = parse(&value(&mut it, arg)?, "--state-json")?,
            "--state-file" => {
                let p = value(&mut it, arg)?;
                state = parse(&std::fs::read_to_string(&p).with_context(|| p.clone())?, &p)?;
            }
            "--questions" => questions = Some(parse(&value(&mut it, arg)?, "--questions")?),
            "--questions-file" => {
                let p = value(&mut it, arg)?;
                questions = Some(parse(
                    &std::fs::read_to_string(&p).with_context(|| p.clone())?,
                    &p,
                )?);
            }
            "--image" => images.push(value(&mut it, arg)?.into()),
            "--prefer" => opts.prefer = Some(value(&mut it, arg)?),
            "--mmproj" => opts.mmproj = Some(value(&mut it, arg)?.into()),
            "--weight-storage" => opts.weight_storage = value(&mut it, arg)?.parse()?,
            "--dequant-cache" => {
                // rlx-cpu's dequant cache is process-wide; this binary owns the
                // process, so it may set it (a library must not).
                let v = value(&mut it, arg)?;
                match v.as_str() {
                    "off" => rlx_ir::env::set("RLX_DEQUANT_CACHE", "0"),
                    "unbounded" => rlx_ir::env::set("RLX_DEQUANT_CACHE_MAX_BYTES", "unbounded"),
                    bytes => {
                        let n = parse_bytes(bytes)
                            .with_context(|| format!("--dequant-cache {bytes}"))?;
                        rlx_ir::env::set("RLX_DEQUANT_CACHE_MAX_BYTES", n.to_string());
                    }
                }
            }
            "--timing" => timing = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown argument `{other}`\n{USAGE}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required\n{USAGE}"))?;
    let questions = questions.ok_or_else(|| anyhow!("--questions is required\n{USAGE}"))?;
    let images: Vec<RgbImage> = images
        .iter()
        .map(|p| RgbImage::open(p))
        .collect::<Result<_>>()?;

    let t0 = Instant::now();
    let mut d1 = D1::open_with(&weights, device, &opts)?;
    let t_load = t0.elapsed();
    let t1 = Instant::now();
    let out = d1.system_one(&state, &questions, &images)?;
    let t_first = t1.elapsed();
    println!("{}", out.to_python_json(Some(2)));
    if timing {
        let t2 = Instant::now();
        d1.system_one(&state, &questions, &images)?;
        eprintln!(
            "[rlx-d1] load {:.2}s, first call {:.2}s (compiles), warm call {:.3}s",
            t_load.as_secs_f64(),
            t_first.as_secs_f64(),
            t2.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// `123`, `512M`, `8G` (binary multiples).
fn parse_bytes(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('k' | 'K') => (&s[..s.len() - 1], 1u64 << 10),
        Some('m' | 'M') => (&s[..s.len() - 1], 1 << 20),
        Some('g' | 'G') => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    Ok(num.parse::<u64>()? * mult)
}
