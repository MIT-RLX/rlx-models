//! Fit a Jacobian lens over a corpus and save it.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen35,qwen35-tokenizer,metal --release \
//!     --example fit_corpus -- --device metal --corpus ./docs --prompts 32 \
//!     --out /tmp/qwen35.lens.safetensors
//! ```
//!
//! Fitting is what makes the lens prompt-independent: `J_l` is an expectation
//! over prompts, positions and targets, so a lens fitted here applies to any
//! prompt. `--resume` picks up a killed run from its checkpoint.
//!
//! The corpus is whatever text you point it at. The reference uses WikiText-103;
//! anything prose-like works, and the printed convergence numbers tell you
//! whether you have fed it enough.

#![cfg(feature = "qwen35")]

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::GgufLoader;
use rlx_jlens::models::qwen35::Qwen35LensModel;
use rlx_jlens::{CorpusFit, FitConfig, LensModel, StackLens, paragraphs, to_prompts};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::Device;

struct Args {
    corpus: String,
    out: String,
    checkpoint: Option<String>,
    resume: bool,
    device: Device,
    seq: usize,
    prompts: usize,
    every: usize,
    dim_batch: usize,
    min_chars: usize,
    snapshots: bool,
    weights: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        corpus: "docs".to_string(),
        out: "lens.safetensors".to_string(),
        checkpoint: None,
        resume: false,
        device: Device::Cpu,
        seq: 48,
        prompts: 16,
        every: 4,
        dim_batch: 8,
        min_chars: 400,
        snapshots: false,
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
            "--corpus" => {
                a.corpus = need(i)?;
                i += 2;
            }
            "--out" => {
                a.out = need(i)?;
                i += 2;
            }
            "--checkpoint" => {
                a.checkpoint = Some(need(i)?);
                i += 2;
            }
            "--resume" => {
                a.resume = true;
                i += 1;
            }
            "--device" => {
                a.device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            "--seq" => {
                a.seq = need(i)?.parse()?;
                i += 2;
            }
            "--prompts" => {
                a.prompts = need(i)?.parse()?;
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
            "--min-chars" => {
                a.min_chars = need(i)?.parse()?;
                i += 2;
            }
            "--snapshots" => {
                a.snapshots = true;
                i += 1;
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

/// Concatenate every text-ish file under `root` (or `root` itself if a file).
fn read_corpus(root: &str) -> Result<String> {
    let path = std::path::Path::new(root);
    if path.is_file() {
        return std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()));
    }
    let mut out = String::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .flatten()
        {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .extension()
                .is_some_and(|e| e == "md" || e == "txt" || e == "rst")
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                out.push_str(&text);
                out.push_str("\n\n");
            }
        }
    }
    if out.is_empty() {
        bail!("no .md/.txt/.rst files under {root}");
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let path = weights_path(args.weights)?;

    eprintln!("loading {} on {:?}", path.display(), args.device);
    let mut loader = GgufLoader::from_file(path.to_str().context("non-utf8 path")?)?;
    let cfg = Qwen35Config::from_gguf(loader.file())?;
    let weights = Qwen35Weights::from_loader(&mut loader, &cfg)?;
    let model = Qwen35LensModel::new(cfg, weights).with_name("qwen35");

    let text = read_corpus(&args.corpus)?;
    let chunks = paragraphs(&text, args.min_chars);
    eprintln!(
        "corpus {}: {:.1} MB -> {} chunks of >= {} chars",
        args.corpus,
        text.len() as f64 / 1e6,
        chunks.len(),
        args.min_chars
    );
    let prompts = to_prompts(&chunks, args.seq, args.prompts, |s| {
        rlx_qwen35::encode_prompt_from_gguf(&path, s)
    })?;
    if prompts.is_empty() {
        bail!(
            "no chunk tokenized to >= {} tokens; lower --seq or --min-chars",
            args.seq
        );
    }
    eprintln!("{} prompts of {} tokens", prompts.len(), args.seq);

    let target = model.n_layers() - 1;
    let layers: Vec<usize> = (0..=target).step_by(args.every).collect();
    let fit_cfg = FitConfig {
        dim_batch: args.dim_batch,
        skip_first: rlx_jlens::SKIP_FIRST_N_POSITIONS.min(args.seq.saturating_sub(2)),
        device: args.device,
    };

    let mut fit = match (&args.checkpoint, args.resume) {
        (Some(p), true) if std::path::Path::new(p).exists() => {
            let f = CorpusFit::load(p)?;
            // A checkpoint carries its own layer set, and `observe` only checks
            // the *count*. Resuming with a different --every that happens to
            // tap the same number of layers would fold layer 3's Jacobian into
            // layer 2's running sum and silently produce a wrong lens, so
            // compare the sets themselves.
            if f.layers() != layers {
                bail!(
                    "checkpoint {p} was fitted on layers {:?}, but this run taps {:?}; \
                     re-run with matching --every or start a fresh checkpoint",
                    f.layers(),
                    layers
                );
            }
            if f.d_model() != model.d_model() || f.target_layer() != target {
                bail!(
                    "checkpoint {p} is {}-wide targeting layer {}, but this run is {}-wide \
                     targeting layer {target}",
                    f.d_model(),
                    f.target_layer(),
                    model.d_model()
                );
            }
            eprintln!(
                "resuming from {p}: {} prompts already folded in",
                f.n_done()
            );
            f
        }
        _ => CorpusFit::new(&layers, model.d_model(), target)?,
    };

    eprintln!(
        "fitting {} layers (target {target}), skipping the first {} positions …",
        layers.len(),
        fit_cfg.skip_first
    );
    // Bigger is not faster here, and the total work does not change with it —
    // every setting fills the same d rows of J. What changes is how many n×n
    // delta-net state tiles are live at once (`dim_batch · heads · n² · 4` B);
    // past ~8 MB they stop fitting in cache and the scan slows down. On Metal a
    // wider batch also multiplies the reduction-order noise that the whole-stack
    // transport compounds. Measured on an M4 Pro, one prompt at seq 48:
    //
    //   dim_batch     4      8     16     32     64
    //   time       44.6s  44.1s  49.9s  57.3s  ~79s
    //   run-vs-run    —   0.003  0.005  —      0.154
    if args.dim_batch > 8 && args.device != Device::Cpu {
        eprintln!(
            "warning: --dim-batch {} is both slower and noisier than 8 on GPU; \
             see tests/dim_batch_invariance.rs",
            args.dim_batch
        );
    }
    let mut lens = StackLens::new(&model, &layers, target, args.seq, fit_cfg)?;
    let start = std::time::Instant::now();

    for (idx, prompt) in prompts.iter().enumerate() {
        if idx < fit.next_idx() {
            continue; // already folded in by an earlier run
        }
        let t0 = std::time::Instant::now();
        let batched = lens.replicate_tokens(prompt)?;
        let js = lens.jacobians(&batched)?;
        let p = fit.observe(&js)?;
        // The first prompt has no running mean to move, so its shift is NaN by
        // construction — printing that verbatim reads like a numerical failure.
        let shift = if p.mean_rel_change.is_nan() {
            "     —".to_string()
        } else {
            format!("{:.2e}", p.mean_rel_change)
        };
        eprintln!(
            "  prompt {}/{}  {:.1}s  max||J||/sqrt(d) = {:.3}  mean shift = {shift}",
            idx + 1,
            prompts.len(),
            t0.elapsed().as_secs_f64(),
            p.scaled_norm,
        );
        if let Some(ckpt) = &args.checkpoint {
            fit.save(ckpt)?;
        }
        // Snapshot the lens at 1, 2, 4, 8 … prompts so the fit's *quality* can be
        // plotted against corpus size afterwards. `mean shift` says the estimate
        // has stopped moving, which is not the same as saying it is any good.
        let n = p.n_done;
        if args.snapshots && (n.is_power_of_two() || idx + 1 == prompts.len()) {
            let path = format!("{}.n{n}.safetensors", args.out);
            fit.clone().finish()?.save(&path)?;
            eprintln!("    snapshot -> {path}");
        }
    }

    let fitted = fit.finish()?;
    fitted.save(&args.out)?;
    eprintln!(
        "\nfitted over {} prompts in {:.1}s -> {}",
        fitted.n_prompts,
        start.elapsed().as_secs_f64(),
        args.out
    );
    for (layer, j) in fitted.iter() {
        eprintln!("  layer {layer:2}: ||J||/sqrt(d) = {:.4}", j.scaled_norm());
    }
    Ok(())
}
