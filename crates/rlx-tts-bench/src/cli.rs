//! Clap CLI for `rlx-tts-bench`.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::devices::{filter_available, parse_device_list};
use crate::isolate::{is_worker, run_isolated};
use crate::phrases::{DEFAULT_LONG, DEFAULT_SHORT};
use crate::report::{write_html, write_markdown, write_results_jsonl, write_summary_json};
use crate::stress::{StressConfig, run_stress};
use crate::suite::{RunConfig, gate_failed, list_adapters, run_suite, select_models};

#[derive(Parser, Debug)]
#[command(
    name = "rlx-tts-bench",
    about = "Unified TTS bench across RLX model crates"
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Discover adapters and weight resolution
    List,
    /// Run the bench matrix
    Run(RunArgs),
    /// Large synthetic corpus (≥1000) → optional ref TTS → rlx-tts validate (+ Whisper)
    Stress(StressArgs),
}

#[derive(Parser, Debug, Clone)]
pub struct RunArgs {
    /// Comma list of model ids, or `all`
    #[arg(short = 'm', long = "models", default_value = "fake")]
    pub models: String,

    /// `auto` or comma list: cpu,metal,mlx,gpu,cuda,ane
    #[arg(short = 'd', long = "devices", default_value = "cpu")]
    pub devices: String,

    /// `short`, `long`, or both
    #[arg(long = "phrases", default_value = "short")]
    pub phrases: String,

    #[arg(long = "text-short")]
    pub text_short: Option<String>,

    #[arg(long = "text-long")]
    pub text_long: Option<String>,

    #[arg(long = "whisper", default_value_t = false)]
    pub whisper: bool,

    #[arg(long = "spectral", default_value_t = false)]
    pub spectral: bool,

    #[arg(long = "noise", default_value_t = false)]
    pub noise: bool,

    #[arg(long = "clone", default_value_t = false)]
    pub clone: bool,

    #[arg(long = "clone-ref")]
    pub clone_ref: Option<PathBuf>,

    #[arg(long = "clone-ref-text")]
    pub clone_ref_text: Option<String>,

    #[arg(long = "iters")]
    pub iters: Option<usize>,

    #[arg(long = "warmup", default_value_t = 0)]
    pub warmup: usize,

    #[arg(long = "seed", default_value_t = 0)]
    pub seed: u64,

    #[arg(long = "out-dir", default_value = "/tmp/tts-bench")]
    pub out_dir: PathBuf,

    #[arg(long = "html", default_value = "report.html")]
    pub html: PathBuf,

    /// Markdown backend matrices (RTF / cos / Whisper).
    #[arg(long = "md", default_value = "BACKENDS.md")]
    pub md: PathBuf,

    #[arg(long = "json", default_value = "results.jsonl")]
    pub json: PathBuf,

    /// Exit non-zero if short/plain fox hits fall below N
    #[arg(long = "fail-under-fox")]
    pub fail_under_fox: Option<usize>,

    /// Run each (model, device) in a child process (default). Survives abort/OOM/hang.
    #[arg(long = "no-isolate", default_value_t = false)]
    pub no_isolate: bool,

    /// Skip cells already present in `--json` under `--out-dir`.
    #[arg(long = "resume", default_value_t = false)]
    pub resume: bool,

    /// Kill a hung (model, device) worker after this many seconds (default 2400 = 40m).
    #[arg(long = "timeout-secs", default_value_t = 2400)]
    pub timeout_secs: u64,
}

#[derive(Parser, Debug, Clone)]
pub struct StressArgs {
    /// How many synthetic phrases to validate (default 1000).
    #[arg(long = "n", default_value_t = 1000)]
    pub n: usize,

    /// Skip the first N generated phrases (paging / resume windows).
    #[arg(long = "offset", default_value_t = 0)]
    pub offset: usize,

    /// Deterministic corpus seed.
    #[arg(long = "seed", default_value_t = 42)]
    pub seed: u64,

    #[arg(long = "target", default_value = "rlx-tts")]
    pub target: String,

    /// Optional reference TTS for spectral compare (e.g. piper, kittentts, fake).
    #[arg(long = "ref-model")]
    pub ref_model: Option<String>,

    #[arg(long = "device", default_value = "cpu")]
    pub device: String,

    /// Whisper greedy ASR coverage + CER on target audio.
    #[arg(long = "whisper", default_value_t = true)]
    pub whisper: bool,

    /// Disable Whisper even if weights are present.
    #[arg(long = "no-whisper", default_value_t = false)]
    pub no_whisper: bool,

    /// Spectral cosine vs `--ref-model` when set.
    #[arg(long = "spectral", default_value_t = false)]
    pub spectral: bool,

    /// Write every target WAV under `out-dir/wav/`.
    #[arg(long = "save-wav", default_value_t = false)]
    pub save_wav: bool,

    /// Also write every Nth WAV (0 = off) even without `--save-wav`.
    #[arg(long = "save-wav-every", default_value_t = 50)]
    pub save_wav_every: usize,

    /// Dump the planned corpus JSONL before running.
    #[arg(long = "write-corpus", default_value_t = true)]
    pub write_corpus: bool,

    /// Optional plain-text / JSONL corpus file (overrides synthetic generator).
    #[arg(long = "corpus-file")]
    pub corpus_file: Option<PathBuf>,

    #[arg(long = "out-dir", default_value = "/tmp/rlx-tts-stress")]
    pub out_dir: PathBuf,

    /// Skip ids already present in `stress_results.jsonl`.
    #[arg(long = "resume", default_value_t = false)]
    pub resume: bool,

    /// Exit non-zero if median Whisper coverage falls below this.
    #[arg(long = "fail-under-coverage")]
    pub fail_under_coverage: Option<f64>,

    /// Exit non-zero if ok/(ok+err) falls below this (e.g. 0.95).
    #[arg(long = "fail-under-ok-rate")]
    pub fail_under_ok_rate: Option<f64>,
}

pub fn entry(cli: Cli) -> Result<()> {
    match cli.cmd {
        Command::List => {
            list_adapters();
            Ok(())
        }
        Command::Run(args) => run(args),
        Command::Stress(args) => stress(args),
    }
}

fn stress(args: StressArgs) -> Result<()> {
    let devices = filter_available(&parse_device_list(&args.device)?);
    let device = *devices
        .first()
        .ok_or_else(|| anyhow::anyhow!("no runnable device from --device {}", args.device))?;
    let whisper = args.whisper && !args.no_whisper;
    let spectral = args.spectral || args.ref_model.is_some();
    let cfg = StressConfig {
        n: args.n,
        seed: args.seed,
        target_model: args.target,
        ref_model: args.ref_model,
        device,
        whisper,
        spectral,
        save_wav: args.save_wav,
        save_wav_every: args.save_wav_every,
        out_dir: args.out_dir,
        resume: args.resume,
        offset: args.offset,
        fail_under_coverage: args.fail_under_coverage,
        fail_under_ok_rate: args.fail_under_ok_rate,
        corpus_file: args.corpus_file,
        write_corpus: args.write_corpus,
    };
    let _ = run_stress(&cfg)?;
    Ok(())
}

fn run(args: RunArgs) -> Result<()> {
    let models = select_models(&args.models);
    let devices = filter_available(&parse_device_list(&args.devices)?);
    if devices.is_empty() {
        anyhow::bail!("no runnable devices from --devices {}", args.devices);
    }

    // Parent process: isolate by default (workers pass --no-isolate).
    if !args.no_isolate && !is_worker() {
        let rows = run_isolated(&args, &models, &devices)?;
        if gate_failed(&rows, args.fail_under_fox) {
            anyhow::bail!("fail-under-fox gate triggered");
        }
        return Ok(());
    }

    let short = args
        .text_short
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SHORT.to_string());
    let long = args
        .text_long
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_LONG.to_string());
    let mut phrases = Vec::new();
    for p in args.phrases.split(',') {
        match p.trim() {
            "short" => phrases.push(("short".into(), short.clone())),
            "long" => phrases.push(("long".into(), long.clone())),
            "" => {}
            other => anyhow::bail!("unknown phrase id '{other}' (use short,long)"),
        }
    }
    if phrases.is_empty() {
        phrases.push(("short".into(), short));
    }

    let iters = args
        .iters
        .or_else(|| std::env::var("RLX_ITERS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(1);

    let json_path = if args.json.is_absolute() {
        args.json.clone()
    } else {
        args.out_dir.join(&args.json)
    };
    let html_path = if args.html.is_absolute() {
        args.html.clone()
    } else {
        args.out_dir.join(&args.html)
    };
    let md_path = if args.md.is_absolute() {
        args.md.clone()
    } else {
        args.out_dir.join(&args.md)
    };
    let summary_path = args.out_dir.join("summary.json");

    // Truncate incremental target so a re-run of this worker slice is clean.
    if let Some(parent) = json_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_results_jsonl(&json_path, &[])?;

    let cfg = RunConfig {
        models,
        devices,
        phrases,
        whisper: args.whisper,
        spectral: args.spectral,
        noise: args.noise,
        clone: args.clone,
        iters,
        warmup: args.warmup,
        seed: args.seed,
        out_dir: args.out_dir.clone(),
        clone_ref: args.clone_ref,
        clone_ref_text: args.clone_ref_text,
        fail_under_fox: args.fail_under_fox,
        incremental_json: Some(json_path.clone()),
    };

    eprintln!(
        "rlx-tts-bench: {} model(s) × {} device(s) → {}",
        cfg.models.len(),
        cfg.devices.len(),
        cfg.out_dir.display()
    );

    let rows = run_suite(&cfg)?;
    // Rewrite clean (incremental already has rows; this de-dupes if any).
    write_results_jsonl(&json_path, &rows)?;

    // Workers only need the slice jsonl; parent owns the final HTML/summary.
    if is_worker() {
        eprintln!(
            "wrote {}  (worker slice, {} row(s))",
            json_path.display(),
            rows.len()
        );
        if gate_failed(&rows, args.fail_under_fox) {
            anyhow::bail!("fail-under-fox gate triggered");
        }
        return Ok(());
    }

    let summary = write_summary_json(&summary_path, &rows)?;
    write_html(&html_path, &rows, &summary)?;
    write_markdown(
        &md_path,
        &rows,
        &format!(
            "Host: local `rlx-tts-bench` run. Devices: {}.\n",
            args.devices
        ),
    )?;

    eprintln!(
        "wrote {}  {}  {}  {}",
        json_path.display(),
        summary_path.display(),
        html_path.display(),
        md_path.display()
    );
    eprintln!(
        "summary: ok={} skipped={} failed={}",
        summary.n_ok, summary.n_skipped, summary.n_failed
    );

    if gate_failed(&rows, args.fail_under_fox) {
        anyhow::bail!("fail-under-fox gate triggered");
    }
    Ok(())
}
