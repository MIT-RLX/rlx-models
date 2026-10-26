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

//! Quality and speed of quantized expert stores against the checkpoint's own
//! bf16 experts, teacher-forced over one text.
//!
//! ```text
//! cargo run -p rlx-flint --release --example expert_formats -- \
//!     weights/lm/flint [--text-file F] [--chunk N] [--device cpu] \
//!     --experts STORE_DIR [--experts STORE_DIR ...]
//! ```
//!
//! For every position of the text: perplexity of the text itself, and against
//! the bf16 run, top-1 agreement and the mean KL(bf16 ‖ store).

use anyhow::{Result, bail};
use rlx_flint::chat::FlintTokenizer;
use rlx_flint::{FlintRunner, RunnerOptions};
use std::time::Instant;

const DEFAULT_TEXT: &str = "A hash map stores key-value pairs in an array of buckets. \
To insert a key, the map hashes it, reduces the hash modulo the number of buckets, \
and places the pair in that bucket. Two keys can land in the same bucket; this is a \
collision. Separate chaining keeps a small list per bucket, while open addressing \
probes other slots until it finds a free one. When the load factor grows past a \
threshold, the map allocates a larger array and rehashes every entry.\n\n\
fn fib(n: u64) -> u64 {\n    let (mut a, mut b) = (0, 1);\n    for _ in 0..n {\n        \
let t = a + b;\n        a = b;\n        b = t;\n    }\n    a\n}\n";

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let lse = row.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(weights) = args.first().cloned() else {
        bail!(
            "usage: expert_formats WEIGHTS [--text-file F] [--chunk N] [--device D] --experts DIR ..."
        );
    };
    let (mut text, mut chunk, mut device, mut stores) = (
        DEFAULT_TEXT.to_string(),
        16usize,
        rlx_runtime::Device::Cpu,
        Vec::new(),
    );
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().cloned().unwrap_or_default();
        match a.as_str() {
            "--text-file" => text = std::fs::read_to_string(v())?,
            "--chunk" => chunk = v().parse()?,
            "--device" => device = rlx_cli::parse_device(&v())?,
            "--experts" => stores.push(v()),
            other => bail!("unknown argument {other}"),
        }
    }
    let ids = FlintTokenizer::open(&weights)?.encode(&text)?;
    eprintln!("{} tokens, prefill chunk {chunk}", ids.len());

    let run = |store: Option<&str>| -> Result<(Vec<f32>, f64, usize)> {
        let mut o = RunnerOptions::new(device)
            .max_context(ids.len().next_power_of_two().max(64))
            .prefill_chunk(chunk);
        if let Some(s) = store {
            o = o.expert_store(s);
        }
        let mut r = FlintRunner::open(&weights, o)?;
        let t = Instant::now();
        let logits = r.all_logits(&ids)?;
        Ok((logits, t.elapsed().as_secs_f64(), r.config().vocab_size))
    };

    let (base, t_base, v) = run(None)?;
    let n = ids.len();
    let ppl = |logits: &[f32]| -> f64 {
        let nll: f64 = (0..n - 1)
            .map(|p| -log_softmax(&logits[p * v..(p + 1) * v])[ids[p + 1] as usize])
            .sum();
        (nll / (n - 1) as f64).exp()
    };
    println!(
        "{:<44} {:>8} {:>8} {:>10} {:>9}",
        "experts", "time s", "ppl", "top-1 agr", "mean KL"
    );
    println!(
        "{:<44} {:>8.1} {:>8.3} {:>10} {:>9}",
        "bf16 (checkpoint)",
        t_base,
        ppl(&base),
        "-",
        "-"
    );
    for s in &stores {
        let (got, t, _) = run(Some(s))?;
        let (mut agree, mut kl) = (0usize, 0f64);
        for p in 0..n {
            let (a, b) = (&base[p * v..(p + 1) * v], &got[p * v..(p + 1) * v]);
            agree += (argmax(a) == argmax(b)) as usize;
            let (la, lb) = (log_softmax(a), log_softmax(b));
            kl += la
                .iter()
                .zip(&lb)
                .map(|(x, y)| x.exp() * (x - y))
                .sum::<f64>();
        }
        println!(
            "{:<44} {:>8.1} {:>8.3} {:>9.1}% {:>9.5}",
            s,
            t,
            ppl(&got),
            100.0 * agree as f64 / n as f64,
            kl / n as f64
        );
    }
    Ok(())
}
