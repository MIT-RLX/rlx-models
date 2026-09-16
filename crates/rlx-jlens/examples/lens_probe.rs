//! Look at what every layer is disposed to say, at every position.
//!
//! `lens_readout` answers "what does the lens read at the last position?".
//! This answers the mechanistic questions behind it:
//!
//! * **When does the answer form?** Rank and probability of the model's own
//!   final token, read out at each layer.
//! * **Where does it form?** A layer × position grid — the answer usually
//!   appears at the subject token before it appears at the position that
//!   actually predicts it, and the grid shows that migration.
//! * **How decided is the model?** Entropy of the lens distribution per layer,
//!   and its KL to the model's final distribution.
//! * **What does the transport buy?** Every number is printed for the Jacobian
//!   lens and the plain logit lens side by side.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen35,qwen35-tokenizer,metal --release \
//!     --example lens_probe -- --device metal --lens qwen35.lens.safetensors \
//!     --prompt "Fact: The capital of Japan is Tokyo. Fact: The capital city of France is"
//! ```
//!
//! Without `--lens` it fits `J` on the prompt being probed, which is circular
//! for claims about generalization but fine for looking at one prompt.

#![cfg(feature = "qwen35")]

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::GgufLoader;
use rlx_jlens::models::qwen35::Qwen35LensModel;
use rlx_jlens::{FitConfig, Jacobian, JacobianLens, LensModel, Readout, StackLens, rank_of};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{Device, Session};

struct Args {
    prompt: String,
    lens: Option<String>,
    device: Device,
    every: usize,
    dim_batch: usize,
    positions: usize,
    weights: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        prompt: "Fact: The capital of Japan is Tokyo. \
                 Fact: The capital city of France is"
            .to_string(),
        lens: None,
        device: Device::Cpu,
        every: 2,
        dim_batch: 8,
        positions: 14,
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
            "--prompt" => {
                a.prompt = need(i)?;
                i += 2;
            }
            "--lens" => {
                a.lens = Some(need(i)?);
                i += 2;
            }
            "--device" => {
                a.device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            "--every" => {
                a.every = need(i)?.parse::<usize>()?.max(1);
                i += 2;
            }
            "--dim-batch" => {
                a.dim_batch = need(i)?.parse::<usize>()?.max(1);
                i += 2;
            }
            "--positions" => {
                a.positions = need(i)?.parse()?;
                i += 2;
            }
            "--weights" => {
                a.weights = Some(need(i)?);
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
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

/// Softmax in a numerically safe way, returning the distribution.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    p
}

/// Shannon entropy in bits. Low means the layer has already committed.
fn entropy_bits(p: &[f32]) -> f32 {
    -p.iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| v * v.log2())
        .sum::<f32>()
}

/// `KL(lens ‖ final)` in bits — how far this layer's readout still is from what
/// the model actually ends up saying.
fn kl_bits(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q)
        .filter(|(a, b)| **a > 0.0 && **b > 0.0)
        .map(|(a, b)| a * (a / b).log2())
        .sum()
}

/// Trim a decoded token to `w` columns, making whitespace visible.
fn cell(s: &str, w: usize) -> String {
    let t = s
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace(' ', "·");
    let t: String = t.chars().take(w).collect();
    format!("{t:<w$}")
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let path = weights_path(args.weights)?;

    eprintln!("loading {} on {:?}", path.display(), args.device);
    let mut loader = GgufLoader::from_file(path.to_str().context("non-utf8 path")?)?;
    let cfg = Qwen35Config::from_gguf(loader.file())?;
    let weights = Qwen35Weights::from_loader(&mut loader, &cfg)?;
    let model = Qwen35LensModel::new(cfg, weights).with_name("qwen35");

    let ids = rlx_qwen35::encode_prompt_from_gguf(&path, &args.prompt)?;
    let seq = ids.len();
    if seq < 3 {
        bail!("prompt tokenizes to {seq} token(s); need at least 3");
    }
    let tokens: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    let decode_one = |id: u32| -> String {
        rlx_qwen35::decode_ids_from_gguf(&path, &[id], false).unwrap_or_default()
    };

    let fitted = args.lens.as_deref().map(JacobianLens::load).transpose()?;
    let (layers, target): (Vec<usize>, usize) = match &fitted {
        Some(l) => (l.layers(), l.target_layer),
        None => {
            let t = model.n_layers() - 1;
            ((0..=t).step_by(args.every).collect(), t)
        }
    };
    let read_pos = seq - 1;
    let cfg_fit = FitConfig {
        dim_batch: args.dim_batch,
        skip_first: 1.min(seq.saturating_sub(2)),
        device: args.device,
    };

    let (jacobians, fwd_batch): (Vec<Jacobian>, usize) = match &fitted {
        Some(l) => {
            if l.d_model != model.d_model() {
                bail!(
                    "lens is {}-wide but the model is {}-wide",
                    l.d_model,
                    model.d_model()
                );
            }
            eprintln!(
                "lens {} — fitted over {} prompts, {} layers, target {target}",
                args.lens.as_deref().unwrap_or("?"),
                l.n_prompts,
                layers.len()
            );
            let js = layers
                .iter()
                .map(|ly| {
                    l.get(*ly)
                        .cloned()
                        .with_context(|| format!("lens has no layer {ly}"))
                })
                .collect::<Result<Vec<_>>>()?;
            (js, 1)
        }
        None => {
            eprintln!(
                "no --lens: fitting J on this prompt ({} layers) …",
                layers.len()
            );
            let mut lens = StackLens::new(&model, &layers, target, seq, cfg_fit)?;
            let batched = lens.replicate_tokens(&tokens)?;
            let js = lens.jacobians(&batched)?;
            (js, cfg_fit.dim_batch)
        }
    };
    let batched: Vec<f32> = tokens.repeat(fwd_batch);

    // Residuals at every tapped layer and every position, from the model's own
    // forward. `outs = [h_target, tap_0, tap_1, …]`.
    let stack = model.stack(&layers, target, fwd_batch, seq)?;
    let mut fwd = Session::new(args.device).compile(stack.tapped.graph().clone());
    for (name, data) in &stack.params {
        fwd.set_param(name, data);
    }
    let outs = fwd.run(&[(stack.token_input.as_str(), &batched[..])]);

    let d = model.d_model();
    // Decoding is `rows × vocab`, and the vocabulary is 152k wide — at long
    // context that dominates everything else, so decode only the positions this
    // run actually reports rather than the whole sequence. At seq 48 the
    // difference is nothing; at seq 500 it is 9 MB of logits per call instead
    // of 300 MB.
    let shown = args.positions.min(seq);
    let mut probe: Vec<usize> = (seq - shown..seq).collect();
    if !probe.contains(&read_pos) {
        probe.push(read_pos);
    }
    let read_slot = probe
        .iter()
        .position(|&p| p == read_pos)
        .expect("read position probed");
    let mut readout = Readout::new(&model, probe.len(), args.device)?;
    let vocab = readout.vocab();

    // Batch element 0 only (every replica carries the same prompt), gathered
    // down to the probed positions.
    let take = |o: &[f32]| -> Vec<f32> {
        let mut rows = Vec::with_capacity(probe.len() * d);
        for &p in &probe {
            rows.extend_from_slice(&o[p * d..(p + 1) * d]);
        }
        rows
    };

    // What the model itself ends up saying — the row every layer is anticipating.
    let final_logits = readout.logits(&take(&outs[0]))?;
    let final_row = &final_logits[read_slot * vocab..(read_slot + 1) * vocab];
    let final_p = softmax(final_row);
    let answer = final_p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .context("empty vocabulary")?;

    println!(
        "\nprompt: {:?}\n{seq} tokens, reading position {read_pos} ({:?})",
        args.prompt,
        decode_one(ids[read_pos])
    );
    println!(
        "model's own answer: {:?}  p = {:.3}\n",
        decode_one(answer),
        final_p[answer as usize]
    );

    // ── when does the answer form? ──
    println!("{}", "─".repeat(96));
    println!(
        "{:<6} {:<24} {:<22} {:<9} {:<9} entropy  KL→final",
        "layer", "jacobian lens top-1", "logit lens top-1", "p(answer)", "rank"
    );
    println!("{}", "─".repeat(96));

    let mut grid: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut first_top1: Option<usize> = None;

    for (slot, &layer) in layers.iter().enumerate() {
        let resid = take(&outs[slot + 1]);
        let lensed = readout.logits(&jacobians[slot].transport(&resid))?;
        let plain = readout.logits(&resid)?;

        let lrow = &lensed[read_slot * vocab..(read_slot + 1) * vocab];
        let prow = &plain[read_slot * vocab..(read_slot + 1) * vocab];
        let lp = softmax(lrow);
        let rank = rank_of(lrow, answer);
        if rank == 0 && first_top1.is_none() {
            first_top1 = Some(layer);
        }

        let top_of = |row: &[f32]| -> u32 {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        println!(
            "{:<6} {:<24} {:<22} {:<9.4} {:<9} {:>7.2}  {:>7.2}",
            layer,
            format!("{:?}", decode_one(top_of(lrow))),
            format!("{:?}", decode_one(top_of(prow))),
            lp[answer as usize],
            rank,
            entropy_bits(&lp),
            kl_bits(&lp, &final_p),
        );

        // Top-1 per position, for the migration grid.
        grid.push((
            layer,
            (0..probe.len())
                .map(|r| top_of(&lensed[r * vocab..(r + 1) * vocab]))
                .collect(),
        ));
    }
    println!("{}", "─".repeat(96));
    match first_top1 {
        Some(l) => println!("the answer becomes top-1 at layer {l}"),
        None => println!("the answer never becomes top-1 under the lens"),
    }

    // ── where does it form? ──
    println!("\nlayer × position — top-1 token the lens reads (· = space)");
    println!("last {shown} of {seq} positions\n");

    let w = 9;
    print!("{:<12}", "position");
    for (layer, _) in &grid {
        print!("{:<w$}", format!("L{layer}"));
    }
    println!();
    println!("{}", "─".repeat(12 + grid.len() * w));
    for (slot, &pos) in probe.iter().enumerate().take(shown) {
        print!("{:<12}", cell(&decode_one(ids[pos]), 11));
        for (_, tops) in &grid {
            print!("{}", cell(&decode_one(tops[slot]), w));
        }
        println!();
    }
    println!("{}", "─".repeat(12 + grid.len() * w));
    Ok(())
}
