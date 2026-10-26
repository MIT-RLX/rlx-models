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

//! `rlx-flint` — generate from the command line.
//!
//! ```text
//! rlx-flint --weights weights/lm/flint [--device cpu|metal|mlx]
//!           (--prompt TEXT | --raw TEXT | --prompt-ids 1,2,3) [--system TEXT] [--tools FILE]
//!           [--max-tokens N] [--max-context N] [--prefill-chunk N] [--layers N]
//!           [--sample [--temperature T --top-p P --top-k K --seed S]] [--timing]
//!           [--prefetch] [--experts STORE_DIR]
//! rlx-flint quantize --weights DIR --scheme q8_0|q4_0 --out STORE_DIR
//! ```
//!
//! `--prompt` wraps the text as one user turn with the chat template;
//! `--raw` encodes it as is. Greedy by default; `--sample` uses the model
//! card's settings (temperature 1.0, top-p 0.95, top-k 20) unless overridden.
//! `quantize` writes the routed experts block-quantized ([`crate::quant`]);
//! `--experts` then reads them from that store instead of the checkpoint.
//! `--tools` reads a JSON array of tool declarations (OpenAI's `tools`) into
//! the prompt; any calls in the reply are printed to stderr as JSON, one per
//! line.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use rlx_ir::Philox4x32;
use rlx_runtime::Device;
use rlx_runtime::samplers::{SamplerChain, SamplerState, Temperature, TopK, TopP};

use crate::chat::{FlintTokenizer, Message, Role, render_with_tools};
use crate::runner::{FlintRunner, RunnerOptions};
use crate::tools::{Json, Tool, parse_reply};

const USAGE: &str = "usage: rlx-flint --weights DIR [--device DEV] \
(--prompt TEXT | --raw TEXT | --prompt-ids IDS) [--system TEXT] [--tools FILE] [--max-tokens N] \
[--max-context N] [--prefill-chunk N] [--layers N] \
[--sample [--temperature T] [--top-p P] [--top-k K] [--seed S]] [--timing] [--prefetch] [--experts STORE_DIR]\n       \
rlx-flint quantize --weights DIR --scheme q8_0|q4_0 --out STORE_DIR";

pub fn run(argv: &[String]) -> Result<()> {
    if argv.first().map(String::as_str) == Some("quantize") {
        return quantize(&argv[1..]);
    }
    let mut weights: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let (mut prompt, mut raw, mut ids_arg, mut system) = (None, None, None, None);
    let mut max_tokens = 256usize;
    let mut opts = RunnerOptions::new(Device::Cpu);
    let (mut sample, mut temperature, mut top_p, mut top_k, mut seed) =
        (false, 1.0f32, 0.95f32, 20usize, 0u64);
    let mut timing = false;
    let mut tools_file: Option<PathBuf> = None;
    let mut it = argv.iter();
    let value = |it: &mut std::slice::Iter<'_, String>, flag: &str| -> Result<String> {
        it.next()
            .cloned()
            .ok_or_else(|| anyhow!("{flag} needs a value\n{USAGE}"))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--weights" => weights = Some(value(&mut it, arg)?.into()),
            "--device" => device = rlx_cli::parse_device(&value(&mut it, arg)?)?,
            "--prompt" => prompt = Some(value(&mut it, arg)?),
            "--raw" => raw = Some(value(&mut it, arg)?),
            "--prompt-ids" => ids_arg = Some(value(&mut it, arg)?),
            "--system" => system = Some(value(&mut it, arg)?),
            "--tools" => tools_file = Some(value(&mut it, arg)?.into()),
            "--max-tokens" => max_tokens = value(&mut it, arg)?.parse()?,
            "--max-context" => opts.max_context = value(&mut it, arg)?.parse()?,
            "--prefill-chunk" => opts.prefill_chunk = value(&mut it, arg)?.parse::<usize>()?.max(1),
            "--layers" => opts.layers = Some(value(&mut it, arg)?.parse()?),
            "--sample" => sample = true,
            "--temperature" => temperature = value(&mut it, arg)?.parse()?,
            "--top-p" => top_p = value(&mut it, arg)?.parse()?,
            "--top-k" => top_k = value(&mut it, arg)?.parse()?,
            "--seed" => seed = value(&mut it, arg)?.parse()?,
            "--timing" => timing = true,
            "--prefetch" => opts.prefetch_experts = true,
            "--experts" => opts.expert_store = Some(value(&mut it, arg)?.into()),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown argument `{other}`\n{USAGE}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required\n{USAGE}"))?;
    opts.device = device;
    let tools = match &tools_file {
        Some(f) => Tool::list_from_json(&std::fs::read_to_string(f)?)
            .map_err(|e| anyhow!("--tools {f:?}: {e}"))?,
        None => Vec::new(),
    };
    if !tools.is_empty() && prompt.is_none() {
        bail!("--tools needs --prompt (the chat template writes the declarations)");
    }

    let tok = FlintTokenizer::open(&weights).ok();
    let ids: Vec<u32> = match (prompt, raw, ids_arg) {
        (Some(p), None, None) => {
            let mut msgs = Vec::new();
            if let Some(s) = system {
                msgs.push(Message::new(Role::System, s));
            }
            msgs.push(Message::new(Role::User, p));
            let tok = tok
                .as_ref()
                .ok_or_else(|| anyhow!("--prompt needs tokenizer.json in {weights:?}"))?;
            tok.encode(&render_with_tools(&msgs, &tools, true))?
        }
        (None, Some(r), None) => {
            let tok = tok
                .as_ref()
                .ok_or_else(|| anyhow!("--raw needs tokenizer.json in {weights:?}"))?;
            tok.encode(&r)?
        }
        (None, None, Some(s)) => s
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<_, _>>()?,
        _ => bail!("give exactly one of --prompt / --raw / --prompt-ids\n{USAGE}"),
    };
    if ids.is_empty() {
        bail!("the prompt encoded to no tokens");
    }

    let t0 = Instant::now();
    let mut run = FlintRunner::open(&weights, opts)?;
    let t_load = t0.elapsed();
    eprintln!(
        "[rlx-flint] {} layers compiled on {:?} in {:.1}s; prompt {} tokens",
        run.config().num_hidden_layers,
        device,
        t_load.as_secs_f64(),
        ids.len()
    );

    let chain = SamplerChain::builder()
        .push(Temperature { t: temperature })
        .push(TopK { k: top_k })
        .push(TopP {
            p: top_p,
            min_keep: 1,
        })
        .build();
    let mut state = SamplerState::new();
    let mut rng = Philox4x32::new(seed);

    let t1 = Instant::now();
    let mut logits = run.extend(&ids)?;
    let t_prefill = t1.elapsed();
    let t2 = Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let stdout = std::io::stdout();
    for _ in 0..max_tokens {
        let next = if sample {
            chain.sample(&mut logits, &out, &mut state, &mut rng)
        } else {
            rlx_runtime::search::argmax_u32(&logits)
        };
        if run.config().is_eos(next) {
            break;
        }
        out.push(next);
        match tok.as_ref() {
            // Print the newly completed text; a multi-byte character may span
            // tokens, so decode the whole tail and emit what is new.
            Some(t) => {
                let s = t.decode(&out)?;
                if s.len() > printed && s.is_char_boundary(printed) {
                    let mut lock = stdout.lock();
                    write!(lock, "{}", &s[printed..])?;
                    lock.flush()?;
                    printed = s.len();
                }
            }
            None => print!("{next} "),
        }
        if run.position() >= run.capacity() {
            break;
        }
        logits = run.extend(&[next])?;
    }
    println!();
    if let (false, Some(t)) = (tools.is_empty(), tok.as_ref()) {
        for call in parse_reply(&t.decode(&out)?, &tools).tool_calls {
            let line = Json::Object(vec![
                ("name".into(), Json::String(call.name.clone())),
                ("arguments".into(), call.arguments),
            ]);
            eprintln!("[rlx-flint] tool call: {}", line.to_python());
        }
    }
    if timing {
        let t = run.timings();
        let dt = t2.elapsed().as_secs_f64();
        eprintln!(
            "[rlx-flint] prefill {:.2}s ({:.1} tok/s); decode {} tokens in {:.2}s ({:.2} tok/s)",
            t_prefill.as_secs_f64(),
            ids.len() as f64 / t_prefill.as_secs_f64(),
            out.len(),
            dt,
            out.len() as f64 / dt.max(1e-9)
        );
        eprintln!(
            "[rlx-flint] layer graphs {:.2}s, routed experts {:.2}s ({} matrices), head {:.2}s",
            t.layers.as_secs_f64(),
            t.experts.as_secs_f64(),
            t.expert_reads,
            t.head.as_secs_f64()
        );
        if t.prefetched > 0 {
            eprintln!(
                "[rlx-flint] read ahead {} experts, {} of them fired ({:.0}%)",
                t.prefetched,
                t.prefetch_used,
                100.0 * t.prefetch_used as f64 / t.prefetched as f64
            );
        }
    }
    Ok(())
}

/// `rlx-flint quantize`: write a block-quantized expert store.
fn quantize(argv: &[String]) -> Result<()> {
    let (mut weights, mut scheme, mut out) = (None, None, None);
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        let mut value = || {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow!("{arg} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--weights" => weights = Some(PathBuf::from(value()?)),
            "--scheme" => scheme = Some(crate::quant::ExpertScheme::parse(&value()?)?),
            "--out" => out = Some(PathBuf::from(value()?)),
            other => bail!("unknown argument `{other}`\n{USAGE}"),
        }
    }
    let (Some(weights), Some(scheme), Some(out)) = (weights, scheme, out) else {
        bail!("quantize needs --weights, --scheme and --out\n{USAGE}");
    };
    let cfg = crate::config::FlintConfig::from_path(&weights)?;
    let ckpt = crate::checkpoint::Checkpoint::open(&weights)?;
    let t = Instant::now();
    let bytes = crate::quant::write_store(&ckpt, &cfg, scheme, &out)?;
    eprintln!(
        "[rlx-flint] wrote {:.1} GB of {} experts to {out:?} in {:.0}s",
        bytes as f64 / 1e9,
        scheme.name(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
