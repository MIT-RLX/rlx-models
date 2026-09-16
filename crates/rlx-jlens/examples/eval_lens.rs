//! How good is a fitted lens, and how many prompts did it need?
//!
//! `fit_corpus` reports `mean shift`, which says the running estimate has
//! stopped *moving*. That is convergence, not quality — an estimator can settle
//! onto a poor answer. This measures quality directly, on prompts the lens was
//! not fitted on, against the only ground truth that needs no labels: what the
//! model itself goes on to say.
//!
//! Three numbers per layer, averaged over the held-out set:
//!
//! * **agreement** — fraction of prompts where the lens's top-1 *is* the model's
//!   own next token. The headline number.
//! * **rank** — median rank of that token under the lens. Degrades gracefully
//!   where agreement is all-or-nothing.
//! * **KL** — `KL(lens ‖ final)` in bits, how far the whole distribution still is.
//!
//! Pass `--lens` once per snapshot from `fit_corpus --snapshots` to get quality
//! as a function of corpus size:
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen35,qwen35-tokenizer,metal --release \
//!     --example eval_lens -- --device metal \
//!     --lens fit.lens.safetensors.n1.safetensors \
//!     --lens fit.lens.safetensors.n8.safetensors \
//!     --lens fit.lens.safetensors.n32.safetensors
//! ```

#![cfg(feature = "qwen35")]

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::GgufLoader;
use rlx_jlens::models::qwen35::Qwen35LensModel;
use rlx_jlens::{JacobianLens, LensModel, Readout, rank_of};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{Device, Session};

/// Held out by construction: none of this is in the repository's markdown, which
/// is what `fit_corpus --corpus docs` fits on.
const HELD_OUT: &[&str] = &[
    "Fact: The capital of Japan is Tokyo. Fact: The capital city of France is",
    "The Eiffel Tower is located in the city of",
    "Fact: The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "Water freezes at a temperature of zero degrees",
    "The author of the play Romeo and Juliet is William",
    "Fact: The largest ocean on Earth is the",
    "In the sentence, the opposite of hot is",
    "The currency used in Japan is the",
    "Fact: The tallest mountain in the world is Mount",
];

struct Args {
    lenses: Vec<String>,
    device: Device,
    weights: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        lenses: Vec::new(),
        device: Device::Cpu,
        weights: None,
    };
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
                a.lenses.push(need(i)?);
                i += 2;
            }
            "--device" => {
                a.device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            "--weights" => {
                a.weights = Some(need(i)?);
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }
    if a.lenses.is_empty() {
        bail!("pass at least one --lens");
    }
    Ok(a)
}

fn weights_path(explicit: Option<String>) -> Result<std::path::PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.into());
    }
    if let Ok(p) = std::env::var("RLX_JLENS_QWEN35_GGUF") {
        return Ok(p.into());
    }
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/Qwen3.5-0.8B-gguf");
    let mut found: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("no weights directory at {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    found.sort();
    found
        .into_iter()
        .next_back()
        .context("no .gguf found; pass --weights or set RLX_JLENS_QWEN35_GGUF")
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
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

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let path = weights_path(args.weights)?;

    eprintln!("loading {} on {:?}", path.display(), args.device);
    let mut loader = GgufLoader::from_file(path.to_str().context("non-utf8 path")?)?;
    let cfg = Qwen35Config::from_gguf(loader.file())?;
    let weights = Qwen35Weights::from_loader(&mut loader, &cfg)?;
    let model = Qwen35LensModel::new(cfg, weights).with_name("qwen35");
    let d = model.d_model();

    let lenses: Vec<(String, JacobianLens)> = args
        .lenses
        .iter()
        .map(|p| JacobianLens::load(p).map(|l| (p.clone(), l)))
        .collect::<Result<_>>()?;
    let layers = lenses[0].1.layers();
    let target = lenses[0].1.target_layer;
    for (p, l) in &lenses {
        if l.layers() != layers || l.target_layer != target {
            bail!("{p} taps different layers than the first lens");
        }
    }

    // [lens][layer] accumulators over the held-out set.
    let n_l = lenses.len();
    let n_layers = layers.len();
    let mut agree = vec![vec![0usize; n_layers]; n_l];
    let mut ranks: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); n_layers]; n_l];
    let mut kls = vec![vec![0.0f64; n_layers]; n_l];
    let mut n_prompts = 0usize;

    // One `Readout` row: only the position that predicts the next token matters.
    let mut readout = Readout::new(&model, 1, args.device)?;
    let vocab = readout.vocab();

    for text in HELD_OUT {
        let ids = rlx_qwen35::encode_prompt_from_gguf(&path, text)?;
        let seq = ids.len();
        if seq < 3 {
            eprintln!("skipping {text:?}: too short");
            continue;
        }
        let tokens: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
        let read_pos = seq - 1;

        // One forward per prompt, shared by every lens under test.
        let stack = model.stack(&layers, target, 1, seq)?;
        let mut fwd = Session::new(args.device).compile(stack.tapped.graph().clone());
        for (name, data) in &stack.params {
            fwd.set_param(name, data);
        }
        let outs = fwd.run(&[(stack.token_input.as_str(), &tokens[..])]);
        let row = |o: &[f32]| -> Vec<f32> { o[read_pos * d..(read_pos + 1) * d].to_vec() };

        let final_logits = readout.logits(&row(&outs[0]))?;
        let final_p = softmax(&final_logits);
        let answer = final_p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .context("empty vocabulary")?;
        n_prompts += 1;

        for (slot, _layer) in layers.iter().enumerate() {
            let resid = row(&outs[slot + 1]);
            for (li, (_, lens)) in lenses.iter().enumerate() {
                let j = lens.get(layers[slot]).context("layer present")?;
                let logits = readout.logits(&j.transport(&resid))?;
                let r = rank_of(&logits, answer);
                if r == 0 {
                    agree[li][slot] += 1;
                }
                ranks[li][slot].push(r as f32);
                kls[li][slot] += kl_bits(&softmax(&logits[..vocab]), &final_p) as f64;
            }
        }
    }

    if n_prompts == 0 {
        bail!("no held-out prompt was usable");
    }
    println!("\n{n_prompts} held-out prompts, target layer {target}\n");
    for (li, (name, lens)) in lenses.iter().enumerate() {
        let short = std::path::Path::new(name)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| name.clone());
        println!(
            "── {short}  ({} prompts fitted) ─────────────",
            lens.n_prompts
        );
        println!(
            "{:<7} {:<12} {:<14} mean KL (bits)",
            "layer", "agreement", "median rank"
        );
        let mut best: Option<usize> = None;
        for (slot, &layer) in layers.iter().enumerate() {
            let a = agree[li][slot] as f32 / n_prompts as f32;
            if a >= 0.5 && best.is_none() {
                best = Some(layer);
            }
            println!(
                "{:<7} {:<12} {:<14} {:.2}",
                layer,
                format!("{}/{n_prompts}", agree[li][slot]),
                median(&mut ranks[li][slot].clone()),
                kls[li][slot] / n_prompts as f64,
            );
        }
        let deepest = n_layers - 1;
        println!(
            "  first layer with >=50% agreement: {}   |   deepest-layer agreement {}/{n_prompts}, KL {:.2}\n",
            best.map(|l| l.to_string()).unwrap_or_else(|| "none".into()),
            agree[li][deepest],
            kls[li][deepest] / n_prompts as f64,
        );
    }
    Ok(())
}
