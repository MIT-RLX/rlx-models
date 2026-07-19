//! Clap CLI for `rlx-tts-bench`.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::devices::{filter_available, parse_device_list};
use crate::isolate::{is_worker, run_isolated};
use crate::phrases::{DEFAULT_LONG, DEFAULT_SHORT};
use crate::report::{write_html, write_markdown, write_results_jsonl, write_summary_json};
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

pub fn entry(cli: Cli) -> Result<()> {
    match cli.cmd {
        Command::List => {
            list_adapters();
            Ok(())
        }
        Command::Run(args) => run(args),
    }
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

    let short = args.text_short.unwrap_or_else(|| DEFAULT_SHORT.to_string());
    let long = args.text_long.unwrap_or_else(|| DEFAULT_LONG.to_string());
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
