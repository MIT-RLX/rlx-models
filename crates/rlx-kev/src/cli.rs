// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `rlx-kev` — answer a System One request from the command line.
//!
//! ```text
//! rlx-kev --run runs/kev-4b --base models/Qwen3.5-4B-Base --request ticket.json
//! rlx-kev --run runs/kev-4b --encode-only --request ticket.json
//! rlx-kev --run runs/kev-4b --info
//! ```
//!
//! `--encode-only` needs no base model and no weights: it dumps the packed
//! encoding as JSON, which is the artifact to diff against `kev.model.encode`
//! when checking the token protocol.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

/// Parsed command line.
#[derive(Debug, Clone)]
pub struct Args {
    pub run: PathBuf,
    pub base: Option<PathBuf>,
    pub request: Option<String>,
    pub device: String,
    pub temperature: Option<f32>,
    pub fixed: Option<(usize, usize)>,
    pub max_rows: usize,
    pub max_seq: usize,
    pub encode_only: bool,
    pub info: bool,
    pub serve: bool,
    pub port: u16,
    /// `Some(capacity)` enables state-prefix caching.
    pub prefix_cache: Option<usize>,
    pub prefix_min_tokens: usize,
    pub train: bool,
    pub data: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub epochs: usize,
    pub accum: usize,
    pub records_per_step: usize,
    pub train_rows: usize,
    pub lr: f32,
    pub head_lr: f32,
    pub lora_rank: usize,
    /// Export the adapter + head every N optimizer steps (0 disables).
    ///
    /// A full-suite run is 8-14 hours on one machine, and the final export
    /// only happens after the last step — so a run that is killed, OOMs, or
    /// hits a thermal stall at hour 13 leaves nothing behind. One already was.
    pub checkpoint_every: usize,
    /// Frozen suite directory — enables the released recipe.
    pub suite: Option<PathBuf>,
    pub eval: Option<String>,
    pub limit: Option<usize>,
    /// Delta mode: warm-start the adapter + head from this checkpoint.
    pub init_from: Option<PathBuf>,
    /// Fit the calibration temperature on a split and write it into `--run`.
    pub calibrate: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            run: PathBuf::new(),
            base: None,
            request: None,
            device: "cpu".into(),
            temperature: None,
            fixed: None,
            max_rows: 32,
            max_seq: 2048,
            encode_only: false,
            info: false,
            serve: false,
            port: 8009,
            prefix_cache: None,
            prefix_min_tokens: 384,
            train: false,
            data: None,
            out: None,
            tokenizer: None,
            epochs: 1,
            accum: 1,
            records_per_step: 1,
            train_rows: 8,
            lr: 2e-5,
            head_lr: 0.0,
            lora_rank: 16,
            checkpoint_every: 100,
            suite: None,
            eval: None,
            limit: None,
            init_from: None,
            calibrate: false,
        }
    }
}

const USAGE: &str = "\
rlx-kev — Jev-style decision models (Qwen3.5 + LoRA + pointer head)

USAGE:
  rlx-kev --run <DIR> [--base <DIR>] [--request <FILE|->] [OPTIONS]

REQUIRED:
  --run <DIR>          kev checkpoint: head.pt, adapter, tokenizer.json

OPTIONS:
  --base <DIR>         Qwen3.5-*-Base snapshot (config.json + safetensors).
                       Required unless --encode-only / --info.
  --request <FILE|->   POST /v1/systemone body; `-` reads stdin.
  --device <NAME>      cpu (default), metal, mlx, cuda, rocm, gpu, vulkan
  --temperature <T>    override the checkpoint calibration; 1.0 = raw logits
  --fixed <R,S>        pin the compiled shape to R rows x S tokens and
                       release the base weights after building
  --max-rows <N>       bucketed-policy ceiling on questions (default 32)
  --max-seq <N>        bucketed-policy ceiling on tokens (default 2048)
  --encode-only        print the packed encoding as JSON and exit
  --info               print checkpoint metadata and exit
  --prefix-cache [N]   cache the encoded state across requests (N states,
                       default 4). Encodes a repeated state once instead of
                       once per question; costs extra compiled arenas.
  --prefix-min <N>     minimum state tokens to cache (default 384)
  --train              fit a LoRA adapter + pointer head (needs `train`)
  --data <FILE>        labelled JSONL for --train (see src/train.rs)
  --out <DIR>          write the trained checkpoint here
  --checkpoint-every <N>  export the adapter + head every N optimizer steps
                       (default 100, 0 disables). A full-suite run is 8-14h and
                       the final export only lands after the last step.
  --tokenizer <FILE>   tokenizer.json for --train (default: <run>/tokenizer.json)
  --epochs <N>         passes over the data (default 1)
  --accum <N>          micro-batches per optimizer step (default 1)
  --records <N>        records per micro-batch (default 1)
  --train-rows <N>     question rows the training graph compiles for (default 8)
  --lr <F>             learning rate (default 2e-5)
  --head-lr <F>        pointer-head learning rate (0 = same as --lr)
  --lora-rank <N>      LoRA rank (default 16)
  --suite <DIR>        frozen suite dir (e.g. kev-suites/v7/decision-v7);
                       with --train this runs the RELEASED recipe
  --eval <SPLIT>       score a checkpoint on a suite split
                       (development | test | calibration | train)
  --limit <N>          use only N records (strided, so all sources appear)
  --calibrate          fit the temperature on a split (default: calibration)
                       and write it into <run>/head.json
  --init-from <DIR>    delta training: warm-start from this checkpoint's
                       adapter instead of the bare base model
  --serve              run the TypeSafe-compatible HTTP sidecar
                       (POST /v1/systemone; needs the `serve` feature)
  --port <N>           port for --serve (default 8009)
  -h, --help           this text
";

/// Parse `argv` (without the program name).
pub fn parse(argv: &[String]) -> Result<Option<Args>> {
    let mut a = Args::default();
    let mut it = argv.iter().peekable();
    let mut saw_run = false;

    let need = |it: &mut std::iter::Peekable<std::slice::Iter<'_, String>>,
                flag: &str|
     -> Result<String> {
        it.next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
    };

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--run" => {
                a.run = PathBuf::from(need(&mut it, "--run")?);
                saw_run = true;
            }
            "--base" => a.base = Some(PathBuf::from(need(&mut it, "--base")?)),
            "--request" => a.request = Some(need(&mut it, "--request")?),
            "--device" => a.device = need(&mut it, "--device")?,
            "--temperature" => {
                a.temperature = Some(
                    need(&mut it, "--temperature")?
                        .parse()
                        .context("--temperature")?,
                )
            }
            "--fixed" => {
                let v = need(&mut it, "--fixed")?;
                let (r, s) = v
                    .split_once(',')
                    .ok_or_else(|| anyhow::anyhow!("--fixed wants R,S (got {v:?})"))?;
                a.fixed = Some((
                    r.trim().parse().context("--fixed rows")?,
                    s.trim().parse().context("--fixed seq")?,
                ));
            }
            "--max-rows" => {
                a.max_rows = need(&mut it, "--max-rows")?.parse().context("--max-rows")?
            }
            "--max-seq" => a.max_seq = need(&mut it, "--max-seq")?.parse().context("--max-seq")?,
            "--encode-only" => a.encode_only = true,
            "--info" => a.info = true,
            "--prefix-cache" => {
                // Optional value: `--prefix-cache` or `--prefix-cache 8`.
                let n = it
                    .peek()
                    .and_then(|v| v.parse::<usize>().ok())
                    .inspect(|_| {
                        it.next();
                    })
                    .unwrap_or(4);
                a.prefix_cache = Some(n);
            }
            "--prefix-min" => {
                a.prefix_min_tokens = need(&mut it, "--prefix-min")?
                    .parse()
                    .context("--prefix-min")?
            }
            "--train" => a.train = true,
            "--data" => a.data = Some(PathBuf::from(need(&mut it, "--data")?)),
            "--out" => a.out = Some(PathBuf::from(need(&mut it, "--out")?)),
            "--tokenizer" => a.tokenizer = Some(PathBuf::from(need(&mut it, "--tokenizer")?)),
            "--epochs" => a.epochs = need(&mut it, "--epochs")?.parse().context("--epochs")?,
            "--accum" => a.accum = need(&mut it, "--accum")?.parse().context("--accum")?,
            "--records" => {
                a.records_per_step = need(&mut it, "--records")?.parse().context("--records")?
            }
            "--train-rows" => {
                a.train_rows = need(&mut it, "--train-rows")?
                    .parse()
                    .context("--train-rows")?
            }
            "--lr" => a.lr = need(&mut it, "--lr")?.parse().context("--lr")?,
            "--head-lr" => a.head_lr = need(&mut it, "--head-lr")?.parse().context("--head-lr")?,
            "--lora-rank" => {
                a.lora_rank = need(&mut it, "--lora-rank")?
                    .parse()
                    .context("--lora-rank")?
            }
            "--checkpoint-every" => {
                a.checkpoint_every = need(&mut it, "--checkpoint-every")?
                    .parse()
                    .context("--checkpoint-every")?
            }
            "--calibrate" => a.calibrate = true,
            "--suite" => a.suite = Some(PathBuf::from(need(&mut it, "--suite")?)),
            "--eval" => a.eval = Some(need(&mut it, "--eval")?),
            "--limit" => a.limit = Some(need(&mut it, "--limit")?.parse().context("--limit")?),
            "--init-from" => a.init_from = Some(PathBuf::from(need(&mut it, "--init-from")?)),
            "--serve" => a.serve = true,
            "--port" => a.port = need(&mut it, "--port")?.parse().context("--port")?,
            other => bail!("unknown argument {other:?}\n\n{USAGE}"),
        }
    }
    if !saw_run {
        bail!("--run is required\n\n{USAGE}");
    }
    Ok(Some(a))
}

#[cfg(feature = "tokenizer")]
fn read_request(spec: &str) -> Result<crate::api::SystemOneRequest> {
    use std::io::Read;
    let text = if spec == "-" {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .context("reading the request from stdin")?;
        s
    } else {
        std::fs::read_to_string(spec).with_context(|| format!("reading {spec}"))?
    };
    crate::api::SystemOneRequest::parse(&text).context("parsing the System One request")
}

/// Entry point for the `rlx-kev` binary.
pub fn main(argv: &[String]) -> Result<()> {
    let Some(args) = parse(argv)? else {
        return Ok(());
    };
    run(&args)
}

#[cfg(all(feature = "tokenizer", not(feature = "train")))]
fn train(_args: &Args) -> Result<()> {
    bail!("--train needs the `train` feature: cargo build -p rlx-kev --features train")
}

/// Fit a LoRA adapter + pointer head on labelled JSONL.
#[cfg(all(feature = "tokenizer", feature = "train"))]
fn train(args: &Args) -> Result<()> {
    use crate::encode::{EncodeOpts, Encoder, HfTokenizer};
    use crate::train::{
        LoraTargets, TrainConfig, Trainer, build_feeds, build_train_graph, export_adapter,
        export_head, load_jsonl, shape_for,
    };

    if let Some(suite) = args.suite.clone() {
        return run_recipe(args, &suite);
    }
    let data = args
        .data
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--train needs --data or --suite"))?;
    let out = args
        .out
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--train needs --out"))?;
    let base = args
        .base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--train needs --base"))?;
    let tok_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| args.run.join("tokenizer.json"));
    if out.exists() {
        bail!("refusing to overwrite {}", out.display());
    }

    let device = rlx_runtime::parse_device(&args.device)
        .map_err(|e| anyhow::anyhow!("--device {}: {e}", args.device))?;
    let encoder = Encoder::new(HfTokenizer::from_file(&tok_path)?)?;
    let examples = load_jsonl(data)?;
    let encode_opts = EncodeOpts {
        max_state: crate::encode::MAX_STATE,
        max_branch: crate::encode::MAX_BRANCH,
        strict: false,
        option_isolation: false,
    };
    let shape = shape_for(&encoder, &examples, encode_opts, args.train_rows)?;
    eprintln!(
        "{} records | shape {}x{} x{} options | device {}",
        examples.len(),
        shape.rows,
        shape.seq,
        shape.max_options,
        args.device
    );

    let tcfg = TrainConfig {
        lora_rank: args.lora_rank,
        lora_alpha: 2.0 * args.lora_rank as f32,
        targets: LoraTargets::All,
        lr: args.lr,
        head_lr: args.head_lr,
        ..TrainConfig::default()
    };

    let adapter = crate::LoraAdapter::open(&args.run).ok();
    let merged = match adapter {
        // Warm start: continue from an existing checkpoint's adapter.
        Some(ad) => crate::weights::load_merged(base, ad)?,
        None => crate::weights::load_merged_base_only(base)?,
    };
    let cfg = merged.cfg.clone();
    let graph = build_train_graph(&cfg, merged.weights, shape, &tcfg)?;
    eprintln!(
        "{} trainable tensors ({} adapter + 4 head)",
        graph.wrt.len(),
        graph.adapter_names.len()
    );
    let mut trainer = Trainer::new(graph, device, &tcfg)?;

    let t0 = std::time::Instant::now();
    let mut step = 0usize;
    for epoch in 0..args.epochs {
        for (mb, chunk) in examples.chunks(args.records_per_step).enumerate() {
            let feeds = build_feeds(&encoder, &cfg, shape, chunk, encode_opts)?;
            let loss = trainer.accumulate(&feeds)?;
            if (mb + 1) % args.accum == 0 {
                let norm = trainer.optimizer_step()?;
                step += 1;
                if step.is_multiple_of(10) || step == 1 {
                    eprintln!(
                        "epoch {epoch} step {step}: loss {loss:.4} grad_norm {norm:.3} \
                         ({:.2}s)",
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
        }
        // Flush a partial accumulation group at the end of the epoch.
        if trainer.pending() > 0 {
            trainer.optimizer_step()?;
            step += 1;
        }
    }

    let meta = crate::checkpoint::Meta {
        base: base.display().to_string(),
        lora: tcfg.lora_rank,
        head_dim: tcfg.head_dim,
        // Calibration is fitted after training, on held-out data; 1.0 means raw
        // logits, which is the honest value for a fresh run.
        temperature: 1.0,
        ..crate::checkpoint::Meta::default()
    };
    export_adapter(&trainer, tcfg.lora_rank, tcfg.lora_alpha, out)?;
    export_head(&trainer, &meta, out)?;
    std::fs::copy(&tok_path, out.join("tokenizer.json"))?;
    eprintln!(
        "wrote {} ({} steps, {:.1}s). Temperature is 1.0 — fit it on held-out data.",
        out.display(),
        step,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(not(feature = "tokenizer"))]
fn run(_args: &Args) -> Result<()> {
    bail!("rlx-kev was built without the `tokenizer` feature; the CLI needs it")
}

#[cfg(feature = "tokenizer")]
fn run(args: &Args) -> Result<()> {
    use crate::api::{answers_json, to_record};
    use crate::checkpoint::Checkpoint;
    use crate::encode::{EncodeOpts, Encoder, HfTokenizer};
    use crate::model::{INFER_MAX_BRANCH, INFER_MAX_STATE, ShapePolicy};

    let mut ckpt = Checkpoint::open(&args.run)?;
    if let Some(t) = args.temperature {
        ckpt.set_temperature(t)?;
    }

    if args.info {
        let m = ckpt.meta();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "run": args.run.display().to_string(),
                "base": m.base,
                "base_revision": m.base_revision,
                "lora": m.lora,
                "head_dim": m.head_dim,
                "option_isolation": m.option_isolation,
                "special_embeddings": m.special_embeddings,
                "weights_dtype": m.weights_dtype,
                "temperature": m.temperature,
            }))?
        );
        return Ok(());
    }

    if args.calibrate {
        return calibrate(args);
    }
    if args.eval.is_some() {
        return evaluate(args);
    }
    if args.train {
        return train(args);
    }
    if args.serve {
        return serve(args, ckpt);
    }

    let spec = args
        .request
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--request is required (or use --info or --serve)"))?;
    let req = read_request(spec)?;

    if args.encode_only {
        let tok = HfTokenizer::from_file(&ckpt.tokenizer_path())?;
        let encoder = Encoder::new(tok)?;
        let (rec, _meta) = to_record(&req)?;
        let enc = encoder.encode(
            &rec,
            EncodeOpts {
                max_state: INFER_MAX_STATE,
                max_branch: INFER_MAX_BRANCH,
                strict: false,
                option_isolation: ckpt.meta().option_isolation,
            },
        )?;
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "ids": enc.ids,
                "seg": enc.seg,
                "pos": enc.pos,
                "opt": enc.opt,
                "decide_idx": enc.decide_idx,
                "opt_idx": enc.opt_idx,
                "state_truncated": enc.state_truncated,
            }))?
        );
        return Ok(());
    }

    let base = args
        .base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--base is required to run the model"))?;
    let device = rlx_runtime::parse_device(&args.device)
        .map_err(|e| anyhow::anyhow!("--device {}: {e}", args.device))?;
    let policy = match args.fixed {
        Some((rows, seq)) => ShapePolicy::Fixed { rows, seq },
        None => ShapePolicy::Bucketed {
            max_rows: args.max_rows,
            max_seq: args.max_seq,
        },
    };

    let started = std::time::Instant::now();
    let mut model = apply_prefix_cache(crate::load(&args.run, base, device, policy)?, args);
    let load_ms = started.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let (answers, usage) = model.answer(&req)?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1e3;

    let body = serde_json::json!({
        "model": req.model,
        "answers": answers_json(&answers),
        "usage": {
            "input_tokens": usage.input_tokens,
            "state_tokens": usage.state_tokens,
        },
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
        "load_ms": (load_ms * 10.0).round() / 10.0,
        "shape": model.compiled_shape().map(|(r, s)| vec![r, s]),
    });
    println!("{}", serde_json::to_string_pretty(&body)?);
    Ok(())
}

#[cfg(all(feature = "tokenizer", not(feature = "serve")))]
fn serve(_args: &Args, _ckpt: crate::checkpoint::Checkpoint) -> Result<()> {
    bail!("--serve needs the `serve` feature: cargo build -p rlx-kev --features serve")
}

#[cfg(all(feature = "tokenizer", feature = "serve"))]
fn serve(args: &Args, ckpt: crate::checkpoint::Checkpoint) -> Result<()> {
    use std::sync::Arc;

    use crate::model::ShapePolicy;
    use crate::serve::{ServerState, serve_blocking};

    let base = args
        .base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--base is required to serve"))?;
    let device = rlx_runtime::parse_device(&args.device)
        .map_err(|e| anyhow::anyhow!("--device {}: {e}", args.device))?;
    let policy = match args.fixed {
        Some((rows, seq)) => ShapePolicy::Fixed { rows, seq },
        None => ShapePolicy::Bucketed {
            max_rows: args.max_rows,
            max_seq: args.max_seq,
        },
    };
    let model = apply_prefix_cache(crate::load(&args.run, base, device, policy)?, args);
    let state = Arc::new(ServerState::new(
        model,
        args.run.display().to_string(),
        base.display().to_string(),
        args.device.clone(),
    ));
    drop(ckpt);
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], args.port));
    serve_blocking(state, addr)
}

/// Install the prefix cache when `--prefix-cache` was given.
#[cfg(all(feature = "tokenizer", feature = "prefix-cache"))]
fn apply_prefix_cache(
    model: crate::model::DecisionModel<crate::encode::HfTokenizer>,
    args: &Args,
) -> crate::model::DecisionModel<crate::encode::HfTokenizer> {
    match args.prefix_cache {
        Some(capacity) => model.with_prefix_cache(crate::prefix::PrefixCacheConfig {
            capacity,
            min_state_tokens: args.prefix_min_tokens,
        }),
        None => model,
    }
}

#[cfg(all(feature = "tokenizer", not(feature = "prefix-cache")))]
fn apply_prefix_cache(
    model: crate::model::DecisionModel<crate::encode::HfTokenizer>,
    _args: &Args,
) -> crate::model::DecisionModel<crate::encode::HfTokenizer> {
    model
}

#[cfg(all(feature = "tokenizer", not(feature = "train")))]
fn calibrate(_args: &Args) -> Result<()> {
    bail!("--calibrate needs the `train` feature (it shares the suite loader)")
}

/// Fit the calibration temperature on a split and write it into the checkpoint.
///
/// Temperature scaling (Guo et al.): one scalar dividing the pointer logits,
/// chosen to minimise NLL. It **cannot** change accuracy — a positive
/// temperature is monotone per row — which is exactly why an uncalibrated
/// checkpoint looks fine on accuracy while its confident-error rate is roughly
/// double what it should be. A freshly trained run carries 1.0.
///
/// The fit is done on already-scored rows, since `ln p` is the logit vector up
/// to a constant that the softmax cancels — so this is one forward pass over
/// the split and then pure host arithmetic, and the result composes with
/// whatever temperature the checkpoint already had.
#[cfg(all(feature = "tokenizer", feature = "train"))]
fn calibrate(args: &Args) -> Result<()> {
    use crate::model::ShapePolicy;

    let suite = args
        .suite
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--calibrate needs --suite"))?;
    let base = args
        .base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--calibrate needs --base"))?;
    let split = args.eval.as_deref().unwrap_or("calibration");
    let records = crate::recipe::load_split(&suite.join(format!("{split}.jsonl")))?;
    let device = rlx_runtime::parse_device(&args.device)
        .map_err(|e| anyhow::anyhow!("--device {}: {e}", args.device))?;
    let policy = match args.fixed {
        Some((rows, seq)) => ShapePolicy::Fixed { rows, seq },
        None => ShapePolicy::Bucketed {
            max_rows: args.max_rows,
            max_seq: args.max_seq,
        },
    };

    // Whatever the checkpoint already divides by: the fitted factor composes
    // with it, so the value written is absolute rather than relative.
    let head_json = args.run.join("head.json");
    let mut ckpt = crate::checkpoint::Checkpoint::open(&args.run)?;
    if let Some(t) = args.temperature {
        ckpt.set_temperature(t)?;
    }
    let t_current = ckpt.meta().temperature;

    let mut model = apply_prefix_cache(crate::load(&args.run, base, device, policy)?, args);
    let t0 = std::time::Instant::now();
    let rows = crate::eval::score(&mut model, &records, args.limit, |i, n| {
        if i.is_multiple_of(25) || i == n {
            eprintln!(
                "  scored {i}/{n} records ({:.0}s)",
                t0.elapsed().as_secs_f64()
            );
        }
    })?;
    let factor = crate::eval::fit_temperature(&rows);
    let fitted = t_current * factor;
    let before = crate::eval::metrics(&rows);
    let after = crate::eval::metrics(&crate::eval::rescaled(&rows, factor));
    println!("{split} at T={t_current}: {before}");
    println!("{split} at T={fitted}: {after}");
    println!(
        "  accuracy is unchanged by construction ({:.4} -> {:.4})",
        before.accuracy, after.accuracy
    );
    // The search is bracketed on [0.05, 20]. Landing on an endpoint means the
    // optimum is outside it — the model is far more (or less) confident than
    // temperature scaling can express, which usually says something about the
    // checkpoint rather than about the calibration.
    if factor <= 0.0501 || factor >= 19.99 {
        println!(
            "  WARNING: the fit pinned at a search-interval endpoint (factor {factor}), so \
             this is a lower bound on the right temperature, not the optimum. Expect that \
             from a checkpoint that is drastically over- or under-confident."
        );
    }

    if !head_json.exists() {
        bail!(
            "fitted temperature {fitted}, but {} has no head.json to update (a torch \
             `head.pt` is not rewritten here) — pass --temperature {fitted} instead",
            args.run.display()
        );
    }
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&head_json)?)?;
    v["temperature"] = serde_json::json!(fitted);
    std::fs::write(&head_json, serde_json::to_string_pretty(&v)?)?;
    println!(
        "wrote temperature {fitted} to {} (was {t_current})",
        head_json.display()
    );
    Ok(())
}

#[cfg(all(feature = "tokenizer", not(feature = "train")))]
fn evaluate(_args: &Args) -> Result<()> {
    bail!("--eval needs the `train` feature (it shares the suite loader)")
}

/// Score a checkpoint on a frozen suite split, reporting kev's own metrics.
#[cfg(all(feature = "tokenizer", feature = "train"))]
fn evaluate(args: &Args) -> Result<()> {
    use crate::model::ShapePolicy;

    let split = args.eval.as_deref().expect("checked by the caller");
    let suite = args
        .suite
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--eval needs --suite"))?;
    let base = args
        .base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--eval needs --base"))?;
    let path = suite.join(format!("{split}.jsonl"));
    let records = crate::recipe::load_split(&path)?;
    let device = rlx_runtime::parse_device(&args.device)
        .map_err(|e| anyhow::anyhow!("--device {}: {e}", args.device))?;

    let policy = match args.fixed {
        Some((rows, seq)) => ShapePolicy::Fixed { rows, seq },
        None => ShapePolicy::Bucketed {
            max_rows: args.max_rows,
            max_seq: args.max_seq,
        },
    };
    let mut model = apply_prefix_cache(crate::load(&args.run, base, device, policy)?, args);

    let t0 = std::time::Instant::now();
    let rows = crate::eval::score(&mut model, &records, args.limit, |i, n| {
        if i.is_multiple_of(25) || i == n {
            eprintln!(
                "  scored {i}/{n} records ({:.0}s)",
                t0.elapsed().as_secs_f64()
            );
        }
    })?;
    let overall = crate::eval::metrics(&rows);
    println!("{split}: {overall}");
    println!(
        "  accuracy standard error {:.4} — a difference smaller than ~2x this is noise",
        crate::eval::accuracy_se(&overall)
    );
    println!("  per source:");
    for (src, m) in crate::eval::by_source(&rows) {
        println!("    {src:24} {m}");
    }
    Ok(())
}

#[cfg(all(feature = "tokenizer", feature = "train"))]
fn run_recipe(args: &Args, suite: &std::path::Path) -> Result<()> {
    use crate::encode::{EncodeOpts, Encoder, HfTokenizer};
    use crate::train::{
        LoraTargets, RecipeConfig, TrainConfig, Trainer, build_feeds, build_train_graph,
        epoch_examples, export_adapter, export_head, one_cycle_lr, with_synthetic_repeat,
    };

    let out = args
        .out
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--train --suite needs --out"))?;
    let base = args
        .base
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--train needs --base"))?;
    if out.exists() {
        bail!("refusing to overwrite {}", out.display());
    }
    let tok_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| args.run.join("tokenizer.json"));
    let device = rlx_runtime::parse_device(&args.device)
        .map_err(|e| anyhow::anyhow!("--device {}: {e}", args.device))?;

    let records = crate::recipe::load_split(&suite.join("train.jsonl"))?;
    crate::recipe::check_trainable(&records, &[])?;
    let cfg = RecipeConfig {
        train: TrainConfig {
            lora_rank: args.lora_rank,
            lora_alpha: 2.0 * args.lora_rank as f32,
            targets: LoraTargets::All,
            lr: args.lr,
            head_lr: args.head_lr,
            ..TrainConfig::default()
        },
        epochs: args.epochs,
        records_per_step: args.records_per_step,
        accum: args.accum,
        rows: args.train_rows,
        ..RecipeConfig::default()
    };
    let records = with_synthetic_repeat(&records, cfg.synthetic_repeat);
    let mut limited = records;
    if let Some(n) = args.limit {
        limited.truncate(n);
    }
    eprintln!(
        "recipe: {} records, {} epoch(s), batch {} x accum {}, lr {:.1e} (OneCycle pct_start {})",
        limited.len(),
        cfg.epochs,
        cfg.records_per_step,
        cfg.accum,
        cfg.train.lr,
        cfg.one_cycle_pct_start
    );

    let encoder = Encoder::new(HfTokenizer::from_file(&tok_path)?)?;
    let encode_opts = EncodeOpts {
        max_state: crate::encode::MAX_STATE,
        max_branch: crate::encode::MAX_BRANCH,
        strict: false,
        option_isolation: false,
    };

    // Materialize every epoch up front: OneCycle needs `total_steps`, and the
    // augmentation RNG must be driven exactly once per record per epoch.
    let mut epochs: Vec<Vec<crate::train::TrainExample>> = Vec::with_capacity(cfg.epochs);
    for e in 0..cfg.epochs {
        epochs.push(epoch_examples(&limited, &cfg, e)?);
    }
    // Token budget per step. Activations are f32 across 24 layers with
    // 6144-wide DeltaNet intermediates, and the backward roughly triples that,
    // so `rows x seq` is the quantity that decides whether the arena fits.
    // 2048 measured comfortable for Kev-0.8B in ~64 GB; the reference used
    // gradient checkpointing instead, which rlx does not have.
    //
    // It is NOT a speed knob, despite looking like one. `examples/pad_audit`
    // runs the real packer over the whole corpus on CPU: across budgets
    // 1024..8192 x rows 8/16 the total computed positions move by under 3%
    // (4.10M worst, 3.91M best). Raising it trades unfilled rows for sequence
    // padding almost one-for-one. Set it for arena headroom only.
    let token_budget: usize = std::env::var("KEV_TRAIN_TOKEN_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);

    let meta = crate::checkpoint::Meta {
        base: base.display().to_string(),
        lora: cfg.train.lora_rank,
        head_dim: cfg.train.head_dim,
        temperature: 1.0,
        ..crate::checkpoint::Meta::default()
    };
    // Copy the tokenizer up front rather than after the last step, so a
    // checkpoint written mid-run is a complete, loadable directory.
    std::fs::create_dir_all(out)?;
    std::fs::copy(&tok_path, out.join("tokenizer.json"))?;

    let mut batched: Vec<Vec<(crate::train::TrainShape, Vec<usize>)>> =
        Vec::with_capacity(cfg.epochs);
    for (e, ex) in epochs.iter().enumerate() {
        let b = crate::train::shape_batches(&encoder, ex, encode_opts, cfg.rows, token_budget)?;
        // Same-shape batches stay contiguous (rebuilds are expensive) but the
        // group order is randomized, so difficulty is spread across the LR
        // schedule instead of all landing at the annealed tail.
        let mut rng =
            crate::pyrandom::PyRandom::from_source_seed(cfg.seed, &format!("batch-order:{e}"));
        batched.push(crate::train::shuffle_shape_groups(b, &mut rng));
    }
    let total_steps: usize = batched.iter().map(|b| b.len().div_ceil(cfg.accum)).sum();
    let mut shapes: Vec<crate::train::TrainShape> =
        batched.iter().flatten().map(|(s, _)| *s).collect();
    shapes.sort_by_key(|s| (s.seq, s.max_options, s.rows));
    shapes.dedup();
    eprintln!(
        "  {} examples/epoch after augmentation | {} batches, {} distinct shapes \
         (token budget {token_budget}) | {} optimizer steps",
        epochs[0].len(),
        batched[0].len(),
        shapes.len(),
        total_steps
    );

    // `--run` supplies only the tokenizer for training. Warm-starting is
    // explicit (`--init-from`), matching kev's own `--init_from`: silently
    // folding `--run`'s adapter in would make every run a delta run.
    let init = args.init_from.clone();
    let merged = match &init {
        Some(dir) => crate::weights::load_merged(base, crate::LoraAdapter::open(dir)?)?,
        None => crate::weights::load_merged_base_only(base)?,
    };
    eprintln!(
        "  start: {}",
        init.as_ref().map_or_else(
            || "bare base model (cold start)".to_string(),
            |d| format!("warm from {}", d.display())
        )
    );
    let qcfg = merged.cfg.clone();
    // One trainer per shape would duplicate the parameter arena, so the
    // trainer is rebuilt when the shape changes. Batches are shape-sorted, so
    // that happens once per distinct shape rather than per step — but the
    // adapter/head weights and Adam state must carry across, which is what
    // `Trainer::rebuild_at` does.
    let mut trainer: Option<crate::train::Trainer> = None;
    let mut carried: Option<std::collections::HashMap<String, Vec<f32>>> = None;
    let mut carried_opt: Option<(rlx_tune::AdamState, rlx_tune::AdamState)> = None;
    // Rebuild accounting. One graph is compiled per shape bucket and a rebuild
    // costs far more than a step (measured: ~2-4 s to load the base, 3-21 s to
    // build the graph, 10-21 s for Metal to compile it), so the bucketing is worth
    // reporting — it is tunable via `KEV_TRAIN_TOKEN_BUDGET` and its cost is
    // otherwise invisible next to the per-step loss lines.
    let mut rebuilds = 0usize;
    let mut rebuild_s = 0.0f64;
    let weights_src = std::sync::Arc::new((base.to_path_buf(), init.clone()));

    let head_max = if cfg.train.head_lr > 0.0 {
        cfg.train.head_lr
    } else {
        cfg.train.lr
    };
    let t0 = std::time::Instant::now();
    let mut step = 0usize;
    let mut running = 0.0f32;
    let mut running_chance = 0.0f32;
    let mut real_tok = 0usize;
    let mut slot_tok = 0usize;
    let mut idle_row_tok = 0usize;
    let mut pad_tok = 0usize;
    let mut seen = 0usize;
    let mut cur_shape: Option<crate::train::TrainShape> = None;
    let mut pending = 0usize;

    for (e, (examples, batches)) in epochs.iter().zip(&batched).enumerate() {
        for (shape, idx) in batches {
            if cur_shape != Some(*shape) {
                // Flush before swapping graphs so no gradient is lost.
                if pending > 0
                    && let Some(t) = trainer.as_mut()
                {
                    t.optimizer_step()?;
                    step += 1;
                    pending = 0;
                }
                if let Some(mut t) = trainer.take() {
                    carried = Some(t.params().clone());
                    // Carry the Adam moments too, not just the weights: a fresh
                    // optimizer per shape restarts the bias correction, and with
                    // 28 shape changes in an epoch that was enough to diverge the
                    // run during the OneCycle ramp.
                    carried_opt = Some(t.take_optimizer_state());
                }
                let t_load = std::time::Instant::now();
                // The base stays bare even for a warm start: the checkpoint's
                // adapter is restored into the *trainable* parameters below
                // instead of being folded into the weights. Folding it would
                // train fine and then export only the new delta, so loading the
                // result as `base + adapter` would drop everything the
                // checkpoint knew (measured: total ||B|| 86.85 -> 0.35).
                let m = crate::weights::load_merged_base_only(&weights_src.0)?;
                let load_s = t_load.elapsed().as_secs_f64();
                let t_build = std::time::Instant::now();
                let mut g = build_train_graph(&qcfg, m.weights, *shape, &cfg.train)?;
                let build_s = t_build.elapsed().as_secs_f64();
                if let Some(prev) = &carried {
                    for (k, v) in prev {
                        if let Some(slot) = g.params.get_mut(k)
                            && slot.len() == v.len()
                        {
                            slot.copy_from_slice(v);
                        }
                    }
                } else if let Some(d) = &weights_src.1 {
                    // First build of a warm start: carry BOTH halves of the
                    // checkpoint into the trainable parameters. The adapter has
                    // to continue rather than be merged away, or this run's own
                    // export is not self-contained; the head has to be restored
                    // because `build_train_graph` always seeds it
                    // pseudo-randomly, and a random readout scores at chance no
                    // matter how good the backbone is. Later rebuilds take the
                    // `carried` branch above.
                    crate::train::restore_adapter(&mut g, &crate::LoraAdapter::open(d)?)?;
                    let head = crate::Checkpoint::open(d)
                        .with_context(|| format!("opening {} for --init-from", d.display()))?
                        .pointer_head()?;
                    crate::train::restore_head(&mut g, &head)?;
                    eprintln!("  restored the adapter + pointer head from {}", d.display());
                }
                let t_tr = std::time::Instant::now();
                let mut fresh = Trainer::new(g, device, &cfg.train)?;
                let compile_s = t_tr.elapsed().as_secs_f64();
                rebuilds += 1;
                rebuild_s += load_s + build_s + compile_s;
                eprintln!(
                    "  [shape {}x{} x{} options] rebuild #{rebuilds} at {:.0}s: \
                     load {load_s:.1}s + graph {build_s:.1}s + compile {compile_s:.1}s",
                    shape.rows,
                    shape.seq,
                    shape.max_options,
                    t0.elapsed().as_secs_f64()
                );
                if let Some(state) = carried_opt.take() {
                    fresh.restore_optimizer_state(state);
                }
                trainer = Some(fresh);
                cur_shape = Some(*shape);
            }
            let chunk: Vec<crate::train::TrainExample> =
                idx.iter().map(|&i| examples[i].clone()).collect();
            let feeds = build_feeds(&encoder, &qcfg, *shape, &chunk, encode_opts)?;
            let t = trainer.as_mut().expect("built above");
            // Shape utilisation: the step computes `rows × seq` positions but only
            // `real_tokens` of them carry data. Everything else is padding the
            // bucketing introduced, and it costs full forward+backward.
            real_tok += feeds.real_tokens;
            slot_tok += shape.rows * shape.seq;
            // Split the waste: rows the batch never filled, vs tokens padded
            // inside the rows it did. They have different fixes — the first is
            // packing, the second is length grouping — so reporting one number
            // for both says nothing about which to attack.
            idle_row_tok += (shape.rows - feeds.active_rows) * shape.seq;
            pad_tok += feeds.active_rows * shape.seq - feeds.real_tokens;
            running += t.accumulate(&feeds)?;
            // The chance floor for THIS batch: `cross_entropy_masked` averages
            // over all `shape.rows`, so a uniform predictor scores
            // `Σ ln(n_options) / rows` — masked rows contribute 0 to both.
            running_chance += chunk
                .iter()
                .flat_map(|ex| ex.record.questions.iter())
                .map(|q| (q.options.len().max(1) as f32).ln())
                .sum::<f32>()
                / shape.rows as f32;
            seen += 1;
            pending += 1;
            if pending >= cfg.accum {
                t.set_lr(
                    one_cycle_lr(cfg.train.lr, step, total_steps, cfg.one_cycle_pct_start),
                    one_cycle_lr(head_max, step, total_steps, cfg.one_cycle_pct_start),
                );
                let norm = t.optimizer_step()?;
                step += 1;
                pending = 0;
                if step.is_multiple_of(25) || step == 1 {
                    // Print the loss against the chance floor of the batches it
                    // came from. The floor moves with the shape mix — `ln 2` on a
                    // yes/no bucket, `ln 128` on banking77 — so a bare loss is not
                    // readable: 0.709 looks excellent and is exactly chance at two
                    // options. Not reporting this let a 1529-step run finish,
                    // report a falling loss, and score at chance.
                    let mean_loss = running / seen as f32;
                    let mean_chance = running_chance / seen as f32;
                    eprintln!(
                        "  epoch {e} step {step}/{total_steps}: loss {mean_loss:.4} \
                         (chance {mean_chance:.4}{}) grad_norm {norm:.2} lr {:.2e} ({:.0}s)",
                        if mean_loss >= mean_chance {
                            ", AT CHANCE"
                        } else {
                            ""
                        },
                        one_cycle_lr(cfg.train.lr, step, total_steps, cfg.one_cycle_pct_start),
                        t0.elapsed().as_secs_f64()
                    );
                    running_chance = 0.0;
                    running = 0.0;
                    seen = 0;
                }
                if args.checkpoint_every > 0 && step.is_multiple_of(args.checkpoint_every) {
                    let w = std::time::Instant::now();
                    crate::train::checkpoint_atomically(
                        t,
                        cfg.train.lora_rank,
                        cfg.train.lora_alpha,
                        &meta,
                        out,
                    )?;
                    eprintln!(
                        "  [checkpoint at step {step} -> {} in {:.1}s]",
                        out.display(),
                        w.elapsed().as_secs_f64()
                    );
                }
            }
        }
    }
    let mut trainer = trainer.ok_or_else(|| anyhow::anyhow!("no batches were built"))?;
    if pending > 0 {
        trainer.optimizer_step()?;
        step += 1;
    }

    export_adapter(&trainer, cfg.train.lora_rank, cfg.train.lora_alpha, out)?;
    export_head(&trainer, &meta, out)?;
    eprintln!(
        "  {rebuilds} graph rebuilds cost {rebuild_s:.0}s of {:.0}s total ({:.0}%); \
         shape utilisation {:.0}% ({real_tok} real of {slot_tok} computed positions; \
         waste: {:.0}% unfilled rows, {:.0}% seq padding)",
        t0.elapsed().as_secs_f64(),
        100.0 * rebuild_s / t0.elapsed().as_secs_f64().max(1.0),
        100.0 * real_tok as f64 / slot_tok.max(1) as f64,
        100.0 * idle_row_tok as f64 / slot_tok.max(1) as f64,
        100.0 * pad_tok as f64 / slot_tok.max(1) as f64
    );
    eprintln!(
        "wrote {} ({step} steps, {:.0}s). Temperature is 1.0 — fit it on the \
         calibration split before reporting calibrated numbers.",
        out.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
