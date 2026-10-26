// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// `rlx-export`: a checkpoint of any family enabled in this build as an
// `.rlxp` package rlx runs alone, and (`--check`, with the `runner`
// feature) proof that the package decodes what the family's runner does.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use rlx_models::export::{ExportOptions, export_checkpoint, exporters, register_builtin_exporters};

const USAGE: &str = "\
usage: rlx-export <checkpoint> -o <out.rlxp> [options]
       rlx-export --list

<checkpoint> is a Hugging Face directory (config.json, safetensors,
tokenizer.json) or a GGUF; its family is picked as its runner is.

options:
  --family <name>       use this exporter (see --list) instead of detecting one
  --context <n>         longest context the package runs (decoders)
  --uppers <a,b,..>     cache capacities to build the step at (decoders)
  --seqs <a,b,..>       sequence lengths (encoders)
  --weights <dtype>     matmul weights as f32 (default), f16 or bf16
  --embed-weights       copy every weight into the package (default: reference
                        the checkpoint's files beside it)
  --no-prefill          no prompt function: prompts go through the step
  --no-batch            the step is not symbolic in its batch
  --speculate <n>       draft tokens a check function takes (0: none)
  --frontend <path>     write a shared front end (log-mel, resize) there
  --variant <name>      which of the checkpoint's variants (Jina v5: its task)
  --version <v>         the model's version in its model block
  --check               decode with the package and with the family's runner
                        and compare (language models; needs `runner`)
  --check-only          check the package already at -o, without exporting
  --device <d>          device for --check (default cpu)
  --prompt <text>       a --check prompt (repeatable; default: three), as a
                        user's message in the package's chat template
  --raw                 --check prompts as they are, with no chat template
  --tokens <n>          tokens --check decodes after each prompt (default 16)";

fn list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse()
                .with_context(|| format!("not a number: {v}"))
        })
        .collect()
}

struct Args {
    checkpoint: PathBuf,
    out: PathBuf,
    family: Option<String>,
    opts: ExportOptions,
    check: bool,
    check_only: bool,
    device: String,
    prompts: Vec<String>,
    raw: bool,
    tokens: usize,
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Option<Args>> {
    let mut a = Args {
        checkpoint: PathBuf::new(),
        out: PathBuf::new(),
        family: None,
        opts: ExportOptions::default(),
        check: false,
        check_only: false,
        device: "cpu".into(),
        prompts: Vec::new(),
        raw: false,
        tokens: 16,
    };
    let mut positional = Vec::new();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} takes a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(None);
            }
            "--list" => {
                register_builtin_exporters();
                for e in exporters() {
                    println!("{:<10} {:<26} {}", e.name, e.task, e.claims.join(", "));
                }
                return Ok(None);
            }
            "-o" | "--out" => a.out = value("-o")?.into(),
            "--family" => a.family = Some(value("--family")?),
            "--context" => a.opts.max_context = Some(value("--context")?.parse()?),
            "--uppers" => a.opts.uppers = list(&value("--uppers")?)?,
            "--seqs" => a.opts.seqs = list(&value("--seqs")?)?,
            "--weights" => {
                a.opts.weight_dtype = Some(match value("--weights")?.as_str() {
                    "f32" => rlx_ir::DType::F32,
                    "f16" => rlx_ir::DType::F16,
                    "bf16" => rlx_ir::DType::BF16,
                    other => bail!("--weights {other}: f32, f16 or bf16"),
                })
            }
            "--embed-weights" => a.opts.embed_weights = true,
            "--no-prefill" => a.opts.prefill.clear(),
            "--no-batch" => a.opts.batch = false,
            "--speculate" => a.opts.speculate = value("--speculate")?.parse()?,
            "--frontend" => a.opts.frontend_package = Some(value("--frontend")?.into()),
            "--variant" => a.opts.variant = Some(value("--variant")?),
            "--version" => a.opts.version = value("--version")?,
            "--check" => a.check = true,
            "--check-only" => (a.check, a.check_only) = (true, true),
            "--device" => a.device = value("--device")?,
            "--prompt" => a.prompts.push(value("--prompt")?),
            "--raw" => a.raw = true,
            "--tokens" => a.tokens = value("--tokens")?.parse()?,
            s if s.starts_with('-') => bail!("unknown option {s}\n\n{USAGE}"),
            _ => positional.push(arg),
        }
    }
    match positional.as_slice() {
        [c] => a.checkpoint = c.into(),
        _ => bail!("one checkpoint, please\n\n{USAGE}"),
    }
    if a.out.as_os_str().is_empty() {
        bail!("-o <out.rlxp> is required\n\n{USAGE}");
    }
    Ok(Some(a))
}

/// `prompts` as ids, by the package's own tokenizer (its `tokenizer.json`
/// sidecar: what the package decodes with, a GGUF's built from its
/// metadata).
fn tokenize(package: &std::path::Path, prompts: &[String]) -> Result<Vec<Vec<u32>>> {
    let bytes = rlx_runtime::Package::open(package)
        .and_then(|p| p.sidecar("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{}: its tokenizer.json: {e}", package.display()))?;
    let tok = tokenizers::Tokenizer::from_bytes(bytes)
        .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    prompts
        .iter()
        .map(|p| {
            tok.encode(p.as_str(), false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!("tokenize {p:?}: {e}"))
        })
        .collect()
}

fn run() -> Result<bool> {
    let Some(a) = parse(std::env::args().skip(1))? else {
        return Ok(true);
    };
    if !a.check_only {
        let t = std::time::Instant::now();
        let (exporter, report) =
            export_checkpoint(&a.checkpoint, &a.out, &a.opts, a.family.as_deref())?;
        let size = std::fs::metadata(&a.out).map(|m| m.len()).unwrap_or(0);
        println!(
            "{} ({}) -> {} ({:.1} MB, {:.1} s)\n{report}",
            exporter.name,
            exporter.task,
            a.out.display(),
            size as f64 / 1e6,
            t.elapsed().as_secs_f64()
        );
    }
    if !a.check {
        return Ok(true);
    }
    #[cfg(feature = "runner")]
    {
        let prompts = if a.prompts.is_empty() {
            vec![
                "The capital of France is".to_string(),
                "def fibonacci(n):".to_string(),
                "Once upon a time, in a small village by the sea,".to_string(),
            ]
        } else {
            a.prompts.clone()
        };
        let prompts = if a.raw {
            prompts
        } else {
            prompts
                .iter()
                .map(|p| rlx_models::export::as_chat(&a.out, p))
                .collect::<Result<Vec<_>>>()?
        };
        let ids = tokenize(&a.out, &prompts)?;
        let c =
            rlx_models::export::check_package(&a.checkpoint, &a.out, &ids, a.tokens, &a.device)?;
        println!("check on {}: {c}", a.device);
        Ok(c.passed())
    }
    #[cfg(not(feature = "runner"))]
    {
        let _ = tokenize;
        bail!("--check needs the family's runner: build with `--features runner`")
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("rlx-export: {e:#}");
            ExitCode::FAILURE
        }
    }
}
