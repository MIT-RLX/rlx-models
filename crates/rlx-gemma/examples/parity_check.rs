// Compare rlx-gemma's predict_logits and llama.cpp's last_token_logits
// for the same GGUF + prompt. Run one backend at a time to keep compile
// and peak memory lower:
//
//   cargo run -p rlx-gemma --release --features "tokenizer parity-llama" \
//     --example parity_check -- <gguf> llama
//
//   cargo run -p rlx-gemma --release --features "tokenizer apple-silicon" \
//     --example parity_check -- <gguf> rlx
//
// Or both in one process:
//
//   cargo run -p rlx-gemma --release --features "tokenizer apple-silicon parity-llama" \
//     --example parity_check -- <gguf> both

use anyhow::{Context, Result, bail};
use rlx_gemma::{GemmaConfigSource, GemmaRunnerBuilder, encode_chat_prompt_auto};
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

#[cfg(feature = "parity-llama")]
use rlx_gemma::llama_reference;

const PROMPT: &str = "Write a Python function that returns True when a string is a palindrome.";

const GREEDY_STEPS: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParityStep {
    Llama,
    Rlx,
    Both,
    GreedyLlama,
    GreedyRlx,
    GreedyBoth,
    RlxCpu,
    RlxCpuPacked,
    HiddenLlama,
    HiddenRlx,
    HiddenBoth,
}

impl ParityStep {
    fn parse(raw: &str) -> Result<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "llama" | "llama.cpp" | "llama-cpp" => Ok(Self::Llama),
            "rlx" | "rlx-gemma" => Ok(Self::Rlx),
            "rlx-cpu" | "rlx_cpu" | "rlx-f32" => Ok(Self::RlxCpu),
            "rlx-cpu-packed" | "rlx_cpu_packed" => Ok(Self::RlxCpuPacked),
            "hidden-llama" | "hidden_llama" => Ok(Self::HiddenLlama),
            "hidden-rlx" | "hidden_rlx" => Ok(Self::HiddenRlx),
            "hidden" | "hidden-both" | "hidden_both" => Ok(Self::HiddenBoth),
            "both" | "all" => Ok(Self::Both),
            "greedy-llama" | "greedy_llama" => Ok(Self::GreedyLlama),
            "greedy-rlx" | "greedy_rlx" => Ok(Self::GreedyRlx),
            "greedy" | "greedy-both" | "greedy_both" => Ok(Self::GreedyBoth),
            other => bail!(
                "unknown step {other:?}; use llama | rlx | both | greedy | greedy-llama | greedy-rlx"
            ),
        }
    }

    fn is_greedy(self) -> bool {
        matches!(self, Self::GreedyLlama | Self::GreedyRlx | Self::GreedyBoth)
    }
}

fn logit_stats(label: &str, logits: &[f32]) -> Option<(usize, f32)> {
    let mut n_nan = 0usize;
    let mut max = f32::NEG_INFINITY;
    let mut max_idx = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            n_nan += 1;
            continue;
        }
        if v > max {
            max = v;
            max_idx = i;
        }
    }
    println!(
        "{label}: n={} nan={n_nan} max={max:.4e} argmax={max_idx}",
        logits.len()
    );
    if n_nan == 0 {
        let mut ranked: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top = ranked.len().min(5);
        print!("{label} top-{top}:");
        for (idx, val) in ranked.iter().take(top) {
            print!(" {idx}={val:.3e}");
        }
        println!();
        // llama.cpp reference argmax for the coder parity prompt (greedy step 0).
        const LLAMA_REF: u32 = 8291;
        if (LLAMA_REF as usize) < logits.len() {
            println!(
                "{label} logit[{LLAMA_REF}] = {:.4e}",
                logits[LLAMA_REF as usize]
            );
        }
    }
    if n_nan > 0 {
        None
    } else {
        Some((max_idx, max))
    }
}

fn prompt_ids(gguf: &Path) -> Result<Vec<u32>> {
    if let Ok(raw) = std::env::var("RLX_PARITY_IDS") {
        let ids: Result<Vec<u32>> = raw
            .split(',')
            .map(|s| s.trim().parse().context("RLX_PARITY_IDS parse"))
            .collect();
        return ids;
    }
    let ids = encode_chat_prompt_auto(gguf, None, None, PROMPT, true)?;
    Ok(ids)
}

fn run_llama(gguf: &Path, prompt_ids: &[u32]) -> Result<Option<(usize, f32)>> {
    #[cfg(feature = "parity-llama")]
    {
        println!("\n— llama.cpp reference —");
        let llama_logits = llama_reference::last_token_logits(gguf, prompt_ids)
            .context("llama_reference::last_token_logits")?;
        Ok(logit_stats("llama.cpp", &llama_logits))
    }
    #[cfg(not(feature = "parity-llama"))]
    {
        let _ = (gguf, prompt_ids);
        bail!("llama step requires --features parity-llama");
    }
}

fn run_rlx(gguf: &Path, prompt_ids: &[u32]) -> Result<Option<(usize, f32)>> {
    run_rlx_on(gguf, prompt_ids, Device::Metal, true)
}

fn run_rlx_cpu_f32(gguf: &Path, prompt_ids: &[u32]) -> Result<Option<(usize, f32)>> {
    println!("\n— rlx-gemma CPU F32 dequant path —");
    run_rlx_on(gguf, prompt_ids, Device::Cpu, false)
}

fn compare_hidden(label: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    if n == 0 {
        println!("{label}: empty");
        return;
    }
    let mut l2 = 0f64;
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        l2 += (x - y).powi(2);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    l2 = l2.sqrt();
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        f64::NAN
    };
    println!("{label}: len={n} L2={l2:.3e} cosine={cos:.6}");
}

fn run_llama_hidden(gguf: &Path, prompt_ids: &[u32]) -> Result<Vec<f32>> {
    #[cfg(feature = "parity-llama")]
    {
        println!("\n— llama.cpp last-token hidden —");
        llama_reference::last_token_hidden(gguf, prompt_ids)
            .context("llama_reference::last_token_hidden")
    }
    #[cfg(not(feature = "parity-llama"))]
    {
        let _ = (gguf, prompt_ids);
        bail!("hidden llama step requires --features parity-llama");
    }
}

fn run_rlx_hidden(gguf: &Path, prompt_ids: &[u32], device: Device) -> Result<Vec<f32>> {
    println!("\n— rlx-gemma last-token hidden (device={device:?}) —");
    let mut runner = GemmaRunnerBuilder::default()
        .weights(gguf.to_str().context("gguf path utf8")?)
        .device(device)
        .config(GemmaConfigSource::Embedded)
        .packed_weights(true)
        .max_seq(512)
        .build()
        .context("build runner")?;
    runner
        .predict_last_hidden(prompt_ids)
        .context("predict_last_hidden")
}

fn run_rlx_on(
    gguf: &Path,
    prompt_ids: &[u32],
    device: Device,
    packed: bool,
) -> Result<Option<(usize, f32)>> {
    if packed {
        println!("\n— rlx-gemma packed Q4 path —");
    }
    let mut runner = GemmaRunnerBuilder::default()
        .weights(gguf.to_str().context("gguf path utf8")?)
        .device(device)
        .config(GemmaConfigSource::Embedded)
        .packed_weights(packed)
        .max_seq(512)
        .build()
        .context("build runner")?;
    let rlx_logits = runner
        .predict_logits(prompt_ids)
        .context("predict_logits")?;
    Ok(logit_stats(
        if packed {
            "rlx-gemma"
        } else {
            "rlx-gemma-cpu-f32"
        },
        &rlx_logits,
    ))
}

fn run_llama_greedy(gguf: &Path, prompt_ids: &[u32]) -> Result<Vec<u32>> {
    #[cfg(feature = "parity-llama")]
    {
        println!("\n— llama.cpp greedy (temp 0 / argmax) —");
        let toks = llama_reference::greedy_generation_ids(gguf, prompt_ids, GREEDY_STEPS, 512)
            .context("llama_reference::greedy_generation_ids")?;
        println!("llama.cpp greedy ids ({}) = {toks:?}", toks.len());
        Ok(toks)
    }
    #[cfg(not(feature = "parity-llama"))]
    {
        let _ = (gguf, prompt_ids);
        bail!("greedy llama step requires --features parity-llama");
    }
}

fn run_rlx_greedy(gguf: &Path, prompt_ids: &[u32]) -> Result<Vec<u32>> {
    println!("\n— rlx-gemma greedy packed (SampleOpts::greedy / temp 0) —");
    let mut runner = GemmaRunnerBuilder::default()
        .weights(gguf.to_str().context("gguf path utf8")?)
        .device(Device::Cpu)
        .config(GemmaConfigSource::Embedded)
        .packed_weights(true)
        .max_seq(512)
        .sample(SampleOpts {
            temperature: 0.0,
            ..SampleOpts::greedy()
        })
        .build()
        .context("build runner")?;
    let toks = runner.generate(prompt_ids, GREEDY_STEPS as usize, |_| {})?;
    println!("rlx-gemma greedy ids ({}) = {toks:?}", toks.len());
    Ok(toks)
}

fn compare_greedy(llama: &[u32], rlx: &[u32]) {
    println!("\n— greedy comparison —");
    let n = llama.len().min(rlx.len());
    let mut first_mismatch = None;
    for i in 0..n {
        if llama[i] != rlx[i] {
            first_mismatch = Some(i);
            break;
        }
    }
    if llama == rlx {
        println!("GREEDY MATCH ✓ ({n} tokens identical)");
    } else if let Some(i) = first_mismatch {
        println!(
            "GREEDY MISMATCH at step {i}: llama={} rlx={}",
            llama[i], rlx[i]
        );
        println!("  llama prefix = {:?}", &llama[..n.min(i + 4)]);
        println!("  rlx   prefix = {:?}", &rlx[..n.min(i + 4)]);
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf: PathBuf = args
        .next()
        .context("usage: parity_check <gguf> [llama|rlx|both|greedy|greedy-llama|greedy-rlx]")?
        .into();
    let step = match args.next() {
        Some(s) => ParityStep::parse(&s)?,
        None => ParityStep::Both,
    };

    if !gguf.is_file() {
        bail!("GGUF not found: {gguf:?}");
    }

    let ids = prompt_ids(&gguf).context("encode chat prompt")?;
    println!("prompt = {PROMPT:?}");
    println!("prompt_ids = {ids:?} (len={})", ids.len());

    if matches!(
        step,
        ParityStep::HiddenLlama | ParityStep::HiddenRlx | ParityStep::HiddenBoth
    ) {
        let llama_h = match step {
            ParityStep::HiddenLlama | ParityStep::HiddenBoth => {
                Some(run_llama_hidden(&gguf, &ids)?)
            }
            _ => None,
        };
        let rlx_h = match step {
            ParityStep::HiddenRlx | ParityStep::HiddenBoth => {
                Some(run_rlx_hidden(&gguf, &ids, Device::Cpu)?)
            }
            _ => None,
        };
        if matches!(step, ParityStep::HiddenBoth) {
            compare_hidden(
                "llama vs rlx hidden",
                llama_h.as_ref().unwrap(),
                rlx_h.as_ref().unwrap(),
            );
        }
        return Ok(());
    }

    if step.is_greedy() {
        let llama_toks = match step {
            ParityStep::GreedyLlama | ParityStep::GreedyBoth => {
                Some(run_llama_greedy(&gguf, &ids)?)
            }
            _ => None,
        };
        let rlx_toks = match step {
            ParityStep::GreedyRlx | ParityStep::GreedyBoth => Some(run_rlx_greedy(&gguf, &ids)?),
            _ => None,
        };
        if matches!(step, ParityStep::GreedyBoth) {
            compare_greedy(llama_toks.as_ref().unwrap(), rlx_toks.as_ref().unwrap());
        }
        return Ok(());
    }

    let llama_top = match step {
        ParityStep::Llama | ParityStep::Both => Some(run_llama(&gguf, &ids)?),
        ParityStep::Rlx
        | ParityStep::GreedyLlama
        | ParityStep::GreedyRlx
        | ParityStep::GreedyBoth
        | ParityStep::RlxCpu
        | ParityStep::RlxCpuPacked
        | ParityStep::HiddenLlama
        | ParityStep::HiddenRlx
        | ParityStep::HiddenBoth => None,
    };
    let rlx_top = match step {
        ParityStep::Rlx | ParityStep::Both => Some(run_rlx(&gguf, &ids)?),
        ParityStep::RlxCpu => Some(run_rlx_cpu_f32(&gguf, &ids)?),
        ParityStep::RlxCpuPacked => Some(run_rlx_on(&gguf, &ids, Device::Cpu, true)?),
        ParityStep::Llama
        | ParityStep::GreedyLlama
        | ParityStep::GreedyRlx
        | ParityStep::GreedyBoth
        | ParityStep::HiddenLlama
        | ParityStep::HiddenRlx
        | ParityStep::HiddenBoth => None,
    };

    if matches!(step, ParityStep::Both) {
        println!("\n— comparison —");
        match (llama_top, rlx_top) {
            (Some(Some((l_idx, l_val))), Some(Some((r_idx, r_val)))) => {
                println!("llama argmax = {l_idx} (val {l_val:.4e})");
                println!("rlx   argmax = {r_idx} (val {r_val:.4e})");
                if l_idx == r_idx {
                    println!("TOP-1 MATCH ✓");
                } else {
                    println!("TOP-1 MISMATCH");
                }
            }
            (Some(Some((l_idx, l_val))), Some(None)) => {
                println!("llama top-1 = {l_idx} (val {l_val:.4e}); rlx logits are NaN.");
            }
            (Some(None), _) => {
                println!("llama_reference returned NaN — investigate llama-cpp-2 bridge.");
            }
            _ => {}
        }
    }

    Ok(())
}
