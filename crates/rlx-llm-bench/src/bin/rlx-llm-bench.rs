// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! `rlx-llm-bench` — one CLI over the speed / quality / parity dimensions.
//!
//! ```text
//! rlx-llm-bench <task> [flags]
//!   task: speed | mmlu | gsm8k | parity | all | fetch
//!
//!   --model <kind>        model family (default: qwen3)
//!   --weights <path>      checkpoint (file or mlx-community dir)
//!   --device <dev>        cpu|metal|mlx|coreml|cuda|rocm|wgpu|vulkan (default cpu)
//!   --tokenizer <path>    tokenizer.json (required for mmlu/gsm8k)
//!   --max-seq <n>         prefill/decode bucket hint (default 2048)
//!   --eos <id,id>         stop ids (default: family default)
//!   --name <str>          leaderboard name (default: weights stem)
//!
//!   --data <path.jsonl>   dataset for mmlu/gsm8k (default: built-in synthetic)
//!   --fetch               download the real set (MMLU cais/mmlu, GSM8K openai/gsm8k)
//!   --cache-dir <dir>     dataset cache (default $RLX_LLM_BENCH_CACHE or .cache/llm-bench)
//!   --refetch             re-download even if cached
//!   --dataset <d>         for the `fetch` task: mmlu|gsm8k|both
//!   --limit <n>           cap scored/fetched docs
//!   --mmlu-mode <m>       cloze|letter (default cloze)
//!   --prompt-len <n>      speed synthetic prompt length (default 64)
//!   --decode-tokens <n>   speed tokens to generate (default 64)
//!   --ref <path.json>     parity reference dump
//!
//!   --report <path.md>    also write a markdown leaderboard
//!   --dry-run             use a weightless mock runner (no checkpoint)
//!   --mock-vocab <n>      mock vocab size for --dry-run (default 256)
//! ```

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};

use rlx_llm_bench::adapters::{self, BuildSpec};
use rlx_llm_bench::mock::MockRunner;
use rlx_llm_bench::model::BenchModel;
use rlx_llm_bench::parity::{self, ReferenceDump};
use rlx_llm_bench::quality::datasets::{
    GenDoc, McDoc, load_gen_jsonl, load_mc_jsonl, synthetic_gen, synthetic_mc,
};
use rlx_llm_bench::quality::fetch::{self, FetchConfig};
use rlx_llm_bench::quality::{self, Gsm8kOptions, MmluMode, MmluOptions};
use rlx_llm_bench::report::Report;
use rlx_llm_bench::speed::{self, SpeedConfig};

#[derive(Debug)]
struct Args {
    task: String,
    model: String,
    weights: Option<PathBuf>,
    device: String,
    tokenizer: Option<PathBuf>,
    max_seq: usize,
    eos_ids: Vec<u32>,
    name: Option<String>,
    data: Option<PathBuf>,
    /// Download the standard dataset for a task when `--data` is absent.
    fetch: bool,
    /// Cache dir for fetched datasets.
    cache_dir: PathBuf,
    /// Force re-download even if a cached copy exists.
    refetch: bool,
    /// Dataset selector for the `fetch` task (`mmlu`|`gsm8k`|`both`).
    dataset: Option<String>,
    limit: Option<usize>,
    mmlu_mode: MmluMode,
    /// Explicit `--mmlu-mode`, so the `mc` task can fall back to a dataset's
    /// natural default when unset.
    mode_override: Option<MmluMode>,
    prompt_len: usize,
    decode_tokens: usize,
    /// GSM8K generation cap.
    max_new: usize,
    /// Override the F32-vs-packed choice: `Some(true)` = force F32 (fast cached
    /// decode + logit scoring), `Some(false)` = force packed, `None` = per-task
    /// default (quality tasks pick F32).
    f32_override: Option<bool>,
    reference: Option<PathBuf>,
    /// Write THIS backend's logit dump for later cross-backend parity (run on
    /// CPU with `--save-ref`, then on CUDA/ROCm/… with `--ref`).
    save_ref: Option<PathBuf>,
    report: Option<PathBuf>,
    /// Write per-document predictions as JSONL (for cross-harness diffing).
    dump: Option<PathBuf>,
    dry_run: bool,
    mock_vocab: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        task: String::new(),
        model: "qwen3".into(),
        weights: None,
        device: "cpu".into(),
        tokenizer: None,
        max_seq: 2048,
        eos_ids: Vec::new(),
        name: None,
        data: None,
        fetch: false,
        cache_dir: fetch::default_cache_dir(),
        refetch: false,
        dataset: None,
        limit: None,
        mmlu_mode: MmluMode::Cloze,
        mode_override: None,
        prompt_len: 64,
        decode_tokens: 64,
        max_new: 256,
        f32_override: None,
        reference: None,
        save_ref: None,
        report: None,
        dump: None,
        dry_run: false,
        mock_vocab: 256,
    };
    let mut it = std::env::args().skip(1);
    let mut positional: Option<String> = None;
    while let Some(arg) = it.next() {
        let mut next = || {
            it.next()
                .ok_or_else(|| anyhow!("flag {arg} expects a value"))
        };
        match arg.as_str() {
            "-h" | "--help" | "help" => {
                print_help();
                std::process::exit(0);
            }
            "--model" => a.model = next()?,
            "--weights" => a.weights = Some(PathBuf::from(next()?)),
            "--device" => a.device = next()?,
            "--tokenizer" => a.tokenizer = Some(PathBuf::from(next()?)),
            "--max-seq" => a.max_seq = next()?.parse()?,
            "--eos" => {
                a.eos_ids = next()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u32>())
                    .collect::<std::result::Result<_, _>>()?
            }
            "--name" => a.name = Some(next()?),
            "--data" => a.data = Some(PathBuf::from(next()?)),
            "--fetch" => a.fetch = true,
            "--cache-dir" => a.cache_dir = PathBuf::from(next()?),
            "--refetch" => {
                a.refetch = true;
                a.fetch = true;
            }
            "--dataset" => a.dataset = Some(next()?),
            "--limit" => a.limit = Some(next()?.parse()?),
            "--mmlu-mode" => {
                let m = match next()?.as_str() {
                    "cloze" => MmluMode::Cloze,
                    "letter" => MmluMode::Letter,
                    "raw" => MmluMode::Raw,
                    other => bail!("--mmlu-mode must be cloze|letter|raw, got {other:?}"),
                };
                a.mmlu_mode = m;
                a.mode_override = Some(m);
            }
            "--prompt-len" => a.prompt_len = next()?.parse()?,
            "--decode-tokens" => a.decode_tokens = next()?.parse()?,
            "--max-new" => a.max_new = next()?.parse()?,
            "--force-f32" => a.f32_override = Some(true),
            "--packed" => a.f32_override = Some(false),
            "--ref" => a.reference = Some(PathBuf::from(next()?)),
            "--save-ref" => a.save_ref = Some(PathBuf::from(next()?)),
            "--report" => a.report = Some(PathBuf::from(next()?)),
            "--dump" => a.dump = Some(PathBuf::from(next()?)),
            "--dry-run" => a.dry_run = true,
            "--mock-vocab" => a.mock_vocab = next()?.parse()?,
            other if other.starts_with('-') => bail!("unknown flag {other}"),
            other => {
                if positional.is_some() {
                    bail!("unexpected extra argument {other:?}");
                }
                positional = Some(other.to_string());
            }
        }
    }
    a.task =
        positional.ok_or_else(|| anyhow!("missing task (speed|mmlu|gsm8k|parity|all|fetch)"))?;
    Ok(a)
}

fn print_help() {
    eprintln!("rlx-llm-bench <speed|mmlu|mc|gsm8k|parity|all|fetch> [flags]");
    eprintln!();
    eprintln!("  model:   --model <kind> --weights <path> --device <dev> --tokenizer <json>");
    eprintln!("  data:    --data <jsonl> | --fetch [--cache-dir <dir>] [--refetch]  --limit <n>");
    eprintln!("  mc:      mc --dataset <name> [--mmlu-mode letter|cloze|raw]");
    eprintln!(
        "           benches: {}",
        rlx_llm_bench::quality::fetch::MC_SOURCES
            .iter()
            .map(|s| s.name)
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "  quality: --mmlu-mode letter|cloze|raw  --max-new <n> (gsm8k)  --force-f32 | --packed"
    );
    eprintln!("  speed:   --prompt-len <n> --decode-tokens <n>      parity: --ref <json>");
    eprintln!("  fetch:   fetch --dataset gsm8k|both|<mc-bench> [--limit <n>] [--cache-dir <dir>]");
    eprintln!("  output:  --report <md>   --dump <jsonl>   --dry-run   --mock-vocab <n>");
    eprintln!();
    eprintln!("  compiled adapters: {}", adapters::compiled_adapters());
}

fn build_model(args: &Args, force_f32: bool) -> Result<BenchModel> {
    if args.dry_run {
        let tokenizer = match &args.tokenizer {
            Some(p) => Some(
                tokenizers::Tokenizer::from_file(p)
                    .map_err(|e| anyhow!("loading tokenizer {}: {e}", p.display()))?,
            ),
            None => None,
        };
        return Ok(BenchModel::new(
            args.name.clone().unwrap_or_else(|| "mock".into()),
            "mock",
            Box::new(MockRunner::new(args.mock_vocab)),
            tokenizer,
            args.eos_ids.clone(),
        ));
    }
    let weights = args
        .weights
        .clone()
        .ok_or_else(|| anyhow!("--weights is required (or use --dry-run)"))?;
    let spec = BuildSpec {
        model_kind: args.model.clone(),
        weights,
        device: adapters::parse_device(&args.device)?,
        max_seq: args.max_seq,
        tokenizer: args.tokenizer.clone(),
        eos_ids: args.eos_ids.clone(),
        force_f32,
        name: args.name.clone(),
    };
    adapters::build_model(&spec)
}

fn run_speed(model: &mut BenchModel, args: &Args, report: &mut Report) -> Result<()> {
    let cfg = SpeedConfig {
        prompt_ids: Vec::new(),
        prompt_len: args.prompt_len,
        decode_tokens: args.decode_tokens,
        warmup: true,
    };
    let r = speed::run_speed(model, &cfg)?;
    println!("{}", r.bench_line(&model.name, &model.device));
    println!(
        "  prefill {:.1} tok/s | decode {:.1} tok/s | ttft {:.1} ms | rss {} MB",
        r.prefill_toks_s, r.decode_toks_s, r.ttft_ms, r.peak_rss_mb
    );
    report.add_speed(&model.name, &model.device, &r);
    Ok(())
}

fn fetch_config(args: &Args) -> FetchConfig {
    FetchConfig {
        cache_dir: args.cache_dir.clone(),
        limit: args.limit,
        force: args.refetch,
    }
}

/// Resolve MMLU docs: explicit `--data`, else `--fetch` the real set, else the
/// built-in synthetic smoke set.
fn resolve_mmlu_docs(args: &Args) -> Result<Vec<McDoc>> {
    if let Some(p) = &args.data {
        return load_mc_jsonl(p);
    }
    if args.fetch {
        let path = fetch::fetch_mmlu(&fetch_config(args))?;
        return load_mc_jsonl(&path);
    }
    eprintln!(
        "[mmlu] no --data/--fetch: using built-in synthetic docs (add --fetch for real MMLU)"
    );
    Ok(synthetic_mc())
}

/// Resolve GSM8K docs: explicit `--data`, else `--fetch`, else synthetic.
fn resolve_gsm8k_docs(args: &Args) -> Result<Vec<GenDoc>> {
    if let Some(p) = &args.data {
        return load_gen_jsonl(p);
    }
    if args.fetch {
        let path = fetch::fetch_gsm8k(&fetch_config(args))?;
        return load_gen_jsonl(&path);
    }
    eprintln!(
        "[gsm8k] no --data/--fetch: using built-in synthetic docs (add --fetch for real GSM8K)"
    );
    Ok(synthetic_gen())
}

fn run_mmlu(model: &mut BenchModel, args: &Args, report: &mut Report) -> Result<()> {
    let docs = resolve_mmlu_docs(args)?;
    let opts = MmluOptions {
        mode: args.mmlu_mode,
        max_docs: args.limit,
    };
    let r = quality::run_mmlu(model, &docs, &opts)?;
    println!("{}", r.bench_line(&model.name, &model.device));
    println!(
        "  MMLU n={} acc={:.4} acc_norm={:.4} (headline {:.4})",
        r.n,
        r.acc,
        r.acc_norm,
        r.headline()
    );
    report.add_mmlu(&model.name, &model.device, &r);
    if let Some(p) = &args.dump {
        let mut s = String::new();
        for (i, pr) in r.preds.iter().enumerate() {
            s.push_str(&format!(
                "{{\"i\":{},\"gold\":{},\"best\":{},\"best_norm\":{}}}\n",
                i, pr.gold, pr.best, pr.best_norm
            ));
        }
        std::fs::write(p, s)?;
        eprintln!(
            "wrote {} mmlu predictions -> {}",
            r.preds.len(),
            p.display()
        );
    }
    Ok(())
}

/// Generic multiple-choice bench (ARC, HellaSwag, OpenBookQA, WinoGrande, …),
/// selected by `--dataset`. Reuses the MMLU scorer, so `Letter`-mode benches
/// ride the fast bucketed packed path.
fn run_mc(model: &mut BenchModel, args: &Args, report: &mut Report, mode: MmluMode) -> Result<()> {
    let name = args
        .dataset
        .clone()
        .ok_or_else(|| anyhow!("mc task needs --dataset <name> (arc_challenge, arc_easy, hellaswag, openbookqa, winogrande, mmlu)"))?;
    let docs = match &args.data {
        Some(p) => load_mc_jsonl(p)?,
        None => {
            let path = fetch::fetch_mc(&name, &fetch_config(args))?;
            load_mc_jsonl(&path)?
        }
    };
    let opts = MmluOptions {
        mode,
        max_docs: args.limit,
    };
    let r = quality::run_mmlu(model, &docs, &opts)?;
    println!(
        "LLMBENCH kind=mc dataset={name} model={} device={} n={} acc={:.4} acc_norm={:.4} mode={:?}",
        model.name, model.device, r.n, r.acc, r.acc_norm, r.mode
    );
    println!(
        "  {name} n={} headline={:.4} (acc={:.4} acc_norm={:.4})",
        r.n,
        r.headline(),
        r.acc,
        r.acc_norm
    );
    report.push(
        &model.name,
        &model.device,
        &format!("{name}/headline"),
        format!("{:.4}", r.headline()),
    );
    if let Some(p) = &args.dump {
        let mut s = String::new();
        for (i, pr) in r.preds.iter().enumerate() {
            s.push_str(&format!(
                "{{\"i\":{},\"gold\":{},\"best\":{},\"best_norm\":{}}}\n",
                i, pr.gold, pr.best, pr.best_norm
            ));
        }
        std::fs::write(p, s)?;
        eprintln!(
            "wrote {} {name} predictions -> {}",
            r.preds.len(),
            p.display()
        );
    }
    Ok(())
}

fn run_gsm8k(model: &mut BenchModel, args: &Args, report: &mut Report) -> Result<()> {
    let docs = resolve_gsm8k_docs(args)?;
    let opts = Gsm8kOptions {
        max_docs: args.limit,
        max_new_tokens: args.max_new,
        ..Gsm8kOptions::default()
    };
    let r = quality::run_gsm8k(model, &docs, &opts)?;
    println!("{}", r.bench_line(&model.name, &model.device));
    println!("  GSM8K n={} acc={:.4}", r.n, r.acc);
    report.add_gsm8k(&model.name, &model.device, &r);
    if let Some(p) = &args.dump {
        let mut s = String::new();
        for (i, pr) in r.preds.iter().enumerate() {
            let line = serde_json::json!({
                "i": i, "gold": pr.gold, "pred": pr.pred, "correct": pr.correct,
            });
            s.push_str(&serde_json::to_string(&line)?);
            s.push('\n');
        }
        std::fs::write(p, s)?;
        eprintln!(
            "wrote {} gsm8k predictions -> {}",
            r.preds.len(),
            p.display()
        );
    }
    Ok(())
}

fn run_parity(model: &mut BenchModel, args: &Args, report: &mut Report) -> Result<()> {
    // Deterministic synthetic prompt — same ids on every backend, so a dump
    // saved on one backend is directly comparable on another.
    let synth_prompt = |vocab: usize, len: usize| -> Vec<u32> {
        (0..len.max(1))
            .map(|i| ((i % vocab.max(1)) as u32).max(1))
            .collect()
    };

    // Produce this backend's reference dump for later cross-backend comparison.
    if let Some(save) = &args.save_ref {
        let prompt = synth_prompt(model.vocab_size(), args.prompt_len);
        let logits = model.context_last_logits(&prompt)?;
        ReferenceDump::from_logits(prompt, logits).save(save)?;
        eprintln!(
            "[parity] saved {} reference logits -> {}",
            model.device,
            save.display()
        );
        // Pure producer mode: nothing to compare against yet.
        if args.reference.is_none() {
            return Ok(());
        }
    }

    let dump = match &args.reference {
        Some(p) => ReferenceDump::load(p)?,
        None => {
            // Self-parity: dump our own logits, then compare — proves the path
            // and yields cosine 1.0 / argmax match on a healthy runner.
            let prompt = synth_prompt(model.vocab_size(), args.prompt_len);
            let logits = model.context_last_logits(&prompt)?;
            eprintln!("[parity] no --ref given; using self-dump as reference");
            ReferenceDump::from_logits(prompt, logits)
        }
    };
    let r = parity::run_parity(model, &dump)?;
    println!("{}", r.bench_line(&model.name, &model.device));
    report.add_parity(&model.name, &model.device, &r);
    Ok(())
}

fn run_fetch(args: &Args) -> Result<()> {
    let cfg = fetch_config(args);
    match args.dataset.as_deref().unwrap_or("both") {
        "mmlu" => {
            let p = fetch::fetch_mmlu(&cfg)?;
            println!("mmlu -> {}", p.display());
        }
        "gsm8k" => {
            let p = fetch::fetch_gsm8k(&cfg)?;
            println!("gsm8k -> {}", p.display());
        }
        "both" => {
            let m = fetch::fetch_mmlu(&cfg)?;
            println!("mmlu -> {}", m.display());
            let g = fetch::fetch_gsm8k(&cfg)?;
            println!("gsm8k -> {}", g.display());
        }
        name if fetch::mc_source(name).is_some() => {
            let p = fetch::fetch_mc(name, &cfg)?;
            println!("{name} -> {}", p.display());
        }
        other => bail!(
            "--dataset must be gsm8k|both or an MC bench ({}), got {other:?}",
            fetch::MC_SOURCES
                .iter()
                .map(|s| s.name)
                .collect::<Vec<_>>()
                .join("|")
        ),
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // `fetch` just downloads datasets — no model needed.
    if args.task == "fetch" {
        return run_fetch(&args);
    }

    let mut report = Report::new();

    // Resolve the mc task's effective scoring mode up front — it drives the
    // F32-vs-packed choice below.
    let mc_mode = if args.task == "mc" {
        let default = args
            .dataset
            .as_deref()
            .and_then(fetch::mc_source)
            .map(|s| s.default_mode)
            .unwrap_or(MmluMode::Letter);
        args.mode_override.unwrap_or(default)
    } else {
        MmluMode::Letter
    };

    // Which tasks need the F32 host-driven decode path (prefill_logits +
    // decode_logits)? Generation (gsm8k) and multi-token MC (cloze/raw) do. But
    // single-token MC (letter mode) needs only the context's last-position
    // logits, which the packed path serves from ONE bucketed graph — far faster
    // on compiled backends (Metal/CUDA/ROCm/wgpu/Vulkan) than recompiling the
    // F32 prefill per context length. So letter-mode MC stays on packed unless
    // the user forces otherwise.
    let needs_f32 = args.f32_override.unwrap_or(match args.task.as_str() {
        "mmlu" => matches!(args.mmlu_mode, MmluMode::Cloze),
        "mc" => !matches!(mc_mode, MmluMode::Letter),
        "gsm8k" | "all" => true,
        _ => false,
    });
    let mut model = build_model(&args, needs_f32)?;

    match args.task.as_str() {
        "speed" => run_speed(&mut model, &args, &mut report)?,
        "mmlu" => run_mmlu(&mut model, &args, &mut report)?,
        "mc" => run_mc(&mut model, &args, &mut report, mc_mode)?,
        "gsm8k" => run_gsm8k(&mut model, &args, &mut report)?,
        "parity" => run_parity(&mut model, &args, &mut report)?,
        "all" => {
            run_speed(&mut model, &args, &mut report)?;
            // Text tasks need a tokenizer; skip with a note rather than erroring
            // the whole run.
            if model.tokenizer.is_some() {
                run_mmlu(&mut model, &args, &mut report)?;
                run_gsm8k(&mut model, &args, &mut report)?;
            } else {
                eprintln!("[all] skipping mmlu/gsm8k: no --tokenizer supplied");
            }
            run_parity(&mut model, &args, &mut report)?;
        }
        other => bail!("unknown task {other:?}; expected speed|mmlu|mc|gsm8k|parity|all|fetch"),
    }

    if let Some(p) = &args.report {
        report.write(p)?;
        eprintln!("wrote report {}", p.display());
    }
    Ok(())
}
