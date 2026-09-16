//! Run a prompt through the Jacobian lens and print the top word per layer.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen35 --release --example lens_readout
//! cargo run -p rlx-jlens --features qwen35,metal --release --example lens_readout -- \
//!     --device metal --prompt "The capital of France is"
//! ```
//!
//! For each tapped layer this fits `J_l = ∂h_target/∂h_l`, transports the
//! residual at the chosen position through it, and decodes with the model's own
//! unembedding. The plain logit lens — the same readout with no transport — is
//! printed alongside, because the difference between the two columns *is* what
//! the Jacobian buys.
//!
//! With no `--lens`, `J_l` is fitted on the prompt being read — the same
//! machinery, and enough to show the readout working. `--lens <path>` instead
//! loads a lens fitted over a corpus by `fit_corpus`, which is the claim worth
//! demonstrating: the same `J_l` reads out a prompt it was never fitted on.

#![cfg(feature = "qwen35")]

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::GgufLoader;
use rlx_jlens::models::qwen35::Qwen35LensModel;
use rlx_jlens::{FitConfig, Jacobian, JacobianLens, LensModel, Readout, StackLens};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{Device, Session};

struct Args {
    prompt: String,
    lens: Option<String>,
    device: Device,
    top_k: usize,
    every: usize,
    dim_batch: usize,
    weights: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        prompt: "The capital of France is".to_string(),
        lens: None,
        device: Device::Cpu,
        top_k: 5,
        every: 4,
        dim_batch: 8,
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
            "--top-k" => {
                a.top_k = need(i)?.parse()?;
                i += 2;
            }
            "--dim-batch" => {
                a.dim_batch = need(i)?.parse::<usize>()?.max(1);
                i += 2;
            }
            "--every" => {
                a.every = need(i)?.parse::<usize>()?.max(1);
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

/// Trim a decoded token for table display, making whitespace visible.
fn show(s: &str) -> String {
    let t = s.replace('\n', "\\n").replace('\t', "\\t");
    let t = if t.len() > 14 {
        format!("{}…", &t[..13])
    } else {
        t
    };
    format!("{t:<14}")
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

    eprintln!("prompt {:?} -> {seq} tokens", args.prompt);
    eprint!("  ");
    for &t in &ids {
        eprint!("{:?} ", decode_one(t));
    }
    eprintln!();

    // A loaded lens dictates which layers are read and what they transport into;
    // without one we tap every `--every`-th layer and fit on this prompt.
    let fitted = args.lens.as_deref().map(JacobianLens::load).transpose()?;
    let (layers, target): (Vec<usize>, usize) = match &fitted {
        Some(l) => (l.layers(), l.target_layer),
        None => {
            let t = model.n_layers() - 1;
            ((0..=t).step_by(args.every).collect(), t)
        }
    };
    // The last position is what the model would predict from, and the lens skips
    // leading sink positions when averaging.
    let read_pos = seq - 1;
    let cfg_fit = FitConfig {
        dim_batch: args.dim_batch,
        skip_first: 1.min(seq.saturating_sub(2)),
        device: args.device,
    };

    // Reading through a fitted lens needs no backward pass at all, so the
    // forward runs at batch 1 rather than at the width the estimator needs.
    let (jacobians, fwd_batch): (Vec<Jacobian>, usize) = match &fitted {
        Some(l) => {
            if l.d_model != model.d_model() {
                bail!(
                    "lens is {}-wide but {} is {}-wide",
                    l.d_model,
                    model.name(),
                    model.d_model()
                );
            }
            eprintln!(
                "lens {} — fitted over {} prompts, {} layers, target {}",
                args.lens.as_deref().unwrap_or("?"),
                l.n_prompts,
                layers.len(),
                target
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
            eprintln!("fitting J for {} layers (target {target}) …", layers.len());
            let start = std::time::Instant::now();
            let mut lens = StackLens::new(&model, &layers, target, seq, cfg_fit)?;
            let batched = lens.replicate_tokens(&tokens)?;
            let js = lens.jacobians(&batched)?;
            eprintln!(
                "  fitted in {:.1}s | {}",
                start.elapsed().as_secs_f64(),
                lens.timing().summary()
            );
            (js, cfg_fit.dim_batch)
        }
    };
    let batched: Vec<f32> = tokens.repeat(fwd_batch);

    // Residuals at each tapped layer, from the model's own forward.
    let stack = model.stack(&layers, target, fwd_batch, seq)?;
    let mut fwd = Session::new(args.device).compile(stack.tapped.graph().clone());
    for (name, data) in &stack.params {
        fwd.set_param(name, data);
    }
    let outs = fwd.run(&[(stack.token_input.as_str(), &batched[..])]);

    let d = model.d_model();
    let mut readout = Readout::new(&model, 1, args.device)?;

    // What the model itself predicts, as the reference row.
    let h_final = &outs[0];
    let final_row = &h_final[read_pos * d..(read_pos + 1) * d];
    let model_top = readout.read(final_row, None, args.top_k)?.remove(0);

    println!(
        "\nprompt: {:?}   reading position {read_pos} ({:?})\n",
        args.prompt,
        decode_one(ids[read_pos])
    );
    println!(
        "{:<7} {:<16} {:<16} top-k (jacobian lens)",
        "layer", "jacobian lens", "logit lens"
    );
    println!("{}", "─".repeat(88));

    for (slot, &layer) in layers.iter().enumerate() {
        // outputs = [h_target, tap_0, tap_1, …]
        let tap = &outs[slot + 1];
        let row = &tap[read_pos * d..(read_pos + 1) * d];

        let lensed = readout
            .read(row, Some(&jacobians[slot]), args.top_k)?
            .remove(0);
        let plain = readout.read(row, None, args.top_k)?.remove(0);

        let top_list: Vec<String> = lensed
            .iter()
            .map(|t| format!("{:?}", decode_one(t.token_id)))
            .collect();
        println!(
            "{:<7} {} {} {}",
            layer,
            show(&decode_one(lensed[0].token_id)),
            show(&decode_one(plain[0].token_id)),
            top_list.join(" ")
        );
    }

    println!("{}", "─".repeat(88));
    println!(
        "{:<7} {} {:<16} model's own output",
        format!("{target}*"),
        show(&decode_one(model_top[0].token_id)),
        ""
    );
    println!(
        "\n* the target layer's own logits — the row every lens column is trying to anticipate"
    );
    Ok(())
}
