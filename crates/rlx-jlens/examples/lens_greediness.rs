//! Does the lens's story depend on how greedily you decode it?
//!
//! Every readout so far reports `argmax`. That is the greediest possible
//! decode, and it is worth knowing which conclusions survive a softer one —
//! a "the answer forms at layer 18" claim is less interesting if it is an
//! artifact of only ever looking at the mode.
//!
//! Temperature is applied to the *logits* on both sides: the layer's
//! `unembed(J_l · h_l)` and the model's own final logits. Three numbers per
//! layer and temperature, averaged over held-out prompts:
//!
//! * **top-1** — does the lens's argmax equal the model's? Reported once,
//!   because scaling logits cannot reorder them: argmax, and the rank of any
//!   token, are temperature-invariant. Only the *probabilities* move.
//! * **match** — `Σ_v p_lens(v) · p_final(v)`, the chance two independent
//!   samples agree. This is what actually changes with greediness, and at
//!   `T → 0` it converges to the top-1 column.
//! * **KL** — `KL(lens ‖ final)` in bits at that temperature.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen35,qwen35-tokenizer,metal --release \
//!     --example lens_greediness -- --device metal --lens qwen35.lens.safetensors \
//!     --temp 0.25 --temp 0.5 --temp 1.0 --temp 2.0
//! ```

#![cfg(feature = "qwen35")]

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::GgufLoader;
use rlx_jlens::models::qwen35::Qwen35LensModel;
use rlx_jlens::{JacobianLens, LensModel, Readout};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{Device, Session};

/// Held out from `--corpus crates`, which is what the lens is fitted on.
const HELD_OUT: &[&str] = &[
    "Fact: The capital of Japan is Tokyo. Fact: The capital city of France is",
    "The Eiffel Tower is located in the city of",
    "Fact: The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "The author of the play Romeo and Juliet is William",
    "Fact: The largest ocean on Earth is the",
    "The currency used in Japan is the",
    "Fact: The tallest mountain in the world is Mount",
];

fn softmax_t(logits: &[f32], t: f32) -> Vec<f32> {
    let inv = 1.0 / t.max(1e-6);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|l| ((l - max) * inv).exp()).collect();
    let sum: f32 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    p
}

fn kl_bits(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q)
        .filter(|(a, b)| **a > 0.0 && **b > 0.0)
        .map(|(a, b)| a * (a / b).log2())
        .sum()
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0)
}

fn main() -> Result<()> {
    let mut lens_path = String::new();
    let mut device = Device::Cpu;
    let mut temps: Vec<f32> = Vec::new();
    let mut weights: Option<String> = None;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .with_context(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--lens" => {
                lens_path = need(i)?;
                i += 2;
            }
            "--device" => {
                device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            "--temp" => {
                temps.push(need(i)?.parse()?);
                i += 2;
            }
            "--weights" => {
                weights = Some(need(i)?);
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }
    if lens_path.is_empty() {
        bail!("pass --lens");
    }
    if temps.is_empty() {
        temps = vec![0.25, 0.5, 1.0, 2.0];
    }

    let path: std::path::PathBuf = match weights {
        Some(p) => p.into(),
        None => match std::env::var("RLX_JLENS_QWEN35_GGUF") {
            Ok(p) => p.into(),
            Err(_) => {
                let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../weights/Qwen3.5-0.8B-gguf");
                let mut found: Vec<_> = std::fs::read_dir(&dir)
                    .with_context(|| format!("no weights directory at {}", dir.display()))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
                    .collect();
                found.sort();
                found.into_iter().next_back().context("no .gguf found")?
            }
        },
    };

    let mut loader = GgufLoader::from_file(path.to_str().context("non-utf8 path")?)?;
    let cfg = Qwen35Config::from_gguf(loader.file())?;
    let w = Qwen35Weights::from_loader(&mut loader, &cfg)?;
    let model = Qwen35LensModel::new(cfg, w).with_name("qwen35");
    let d = model.d_model();

    let lens = JacobianLens::load(&lens_path)?;
    let layers = lens.layers();
    let target = lens.target_layer;
    eprintln!(
        "lens {lens_path} ({} prompts fitted), {} layers, target {target}, {:?}",
        lens.n_prompts,
        layers.len(),
        device
    );

    let mut readout = Readout::new(&model, 1, device)?;
    let vocab = readout.vocab();
    let n_l = layers.len();
    let n_t = temps.len();

    let mut top1 = vec![0usize; n_l];
    let mut matchp = vec![vec![0.0f64; n_t]; n_l];
    let mut kls = vec![vec![0.0f64; n_t]; n_l];
    let mut n = 0usize;

    for text in HELD_OUT {
        let ids = rlx_qwen35::encode_prompt_from_gguf(&path, text)?;
        let seq = ids.len();
        if seq < 3 {
            continue;
        }
        let tokens: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
        let read_pos = seq - 1;

        let stack = model.stack(&layers, target, 1, seq)?;
        let mut fwd = Session::new(device).compile(stack.tapped.graph().clone());
        for (name, data) in &stack.params {
            fwd.set_param(name, data);
        }
        let outs = fwd.run(&[(stack.token_input.as_str(), &tokens[..])]);
        let row = |o: &[f32]| -> Vec<f32> { o[read_pos * d..(read_pos + 1) * d].to_vec() };

        let final_logits = readout.logits(&row(&outs[0]))?;
        let answer = argmax(&final_logits);
        n += 1;

        for (slot, &layer) in layers.iter().enumerate() {
            let j = lens.get(layer).context("layer")?;
            let logits = readout.logits(&j.transport(&row(&outs[slot + 1])))?;
            if argmax(&logits) == answer {
                top1[slot] += 1;
            }
            for (ti, &t) in temps.iter().enumerate() {
                let pl = softmax_t(&logits, t);
                let pf = softmax_t(&final_logits, t);
                matchp[slot][ti] += pl.iter().zip(&pf).map(|(a, b)| (a * b) as f64).sum::<f64>();
                kls[slot][ti] += kl_bits(&pl, &pf) as f64;
            }
        }
        let _ = vocab;
    }
    if n == 0 {
        bail!("no usable held-out prompt");
    }

    println!("\n{n} held-out prompts. `top-1` is temperature-invariant — scaling logits");
    println!("cannot reorder them, so argmax and rank are fixed; only the mass moves.\n");
    print!("{:<7} {:<8}", "layer", "top-1");
    for t in &temps {
        print!("  match@{t:<5}");
    }
    for t in &temps {
        print!("  KL@{t:<8}");
    }
    println!();
    println!("{}", "─".repeat(7 + 8 + temps.len() * 26));
    for (slot, &layer) in layers.iter().enumerate() {
        print!("{:<7} {:<8}", layer, format!("{}/{n}", top1[slot]));
        for ti in 0..n_t {
            print!("  {:<11.4}", matchp[slot][ti] / n as f64);
        }
        for ti in 0..n_t {
            print!("  {:<10.3}", kls[slot][ti] / n as f64);
        }
        println!();
    }
    Ok(())
}
