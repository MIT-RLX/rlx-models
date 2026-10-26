// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `rlx-kimi-k3` — inspect, run, train and distribute Kimi-K3 (Moonshot AI).
//!
//! `rlx-kimi-k3 <model-dir>` still prints the config summary; everything else
//! is a subcommand (`rlx-kimi-k3 help`).

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::dist::{
    Cluster, DistOptions, PipelineOptions, PipelineTrainer, ServeOptions, StageSpec, generate,
    parse_stages, plan_stages, serve,
};
use rlx_kimi_k3::export::{export_full, load_graph_weight, save_lora_adapter};
use rlx_kimi_k3::inspect::{
    TraceOptions, dump_weights, inventory, layer_graph_text, logical_names, selected, summarize,
    trace, weight_stats,
};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::source::{CheckpointSource, WeightSource, load_flow_weights};
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_kimi_k3::train::{
    Batcher, TrainSpec, WholeTrainer, build_train_graph, load_corpus, merge_lora, scheduled_lr,
    split_trained,
};
use rlx_kimi_k3::train_stream::{Params, StreamOptions, StreamTrainer};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "rlx-kimi-k3", about = "Kimi-K3 (Moonshot AI) on RLX")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the model config summary.
    Info { model: PathBuf },
    /// Write a small synthetic checkpoint with the real tensor layout.
    Synth(SynthArgs),
    /// List, summarize, take statistics of, or dump checkpoint weights.
    Weights(WeightsArgs),
    /// Run a prompt layer by layer and record every named activation.
    Trace(TraceArgs),
    /// Print one layer's op graph.
    Graph(GraphArgs),
    /// Fine-tune (full or LoRA) on one machine; `--nnodes N` for data-parallel.
    Train(TrainArgs),
    /// Serve a layer range of the pipeline (inference and training).
    Worker(WorkerArgs),
    /// Check pipeline workers are up and agree on the model.
    Ping(PingArgs),
    /// Greedy generation, locally or over pipeline workers; `--state-log` records
    /// every KDA state (feature `state-log`).
    Generate(GenerateArgs),
    /// Rebuild, dump or verify KDA states from a `--state-log` directory
    /// (feature `state-log`).
    KdaState(KdaStateArgs),
    /// Estimate the nodes (RAM / VRAM / storage / IO / network / compute) a
    /// workload needs, and its speed.
    Size(SizeArgs),
    /// Pipeline-parallel training over workers.
    TrainDist(TrainDistArgs),
}

#[derive(Args)]
struct SynthArgs {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 4)]
    layers: usize,
    #[arg(long, default_value_t = 32)]
    hidden: usize,
    #[arg(long, default_value_t = 64)]
    vocab: usize,
    #[arg(long, default_value_t = 8)]
    experts: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

/// Layer range `a..b`, `a-b` or a single count `n` (= `0..n`).
fn parse_range(s: &str) -> Result<Range<usize>> {
    if let Some((a, b)) = s.split_once("..").or_else(|| s.split_once('-')) {
        return Ok(a.trim().parse()?..b.trim().parse()?);
    }
    Ok(0..s.trim().parse()?)
}

fn parse_ids(s: &str) -> Result<Vec<u32>> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse()
                .with_context(|| format!("{t:?} is not a token id"))
        })
        .collect()
}

/// Byte count in GB / MB / kB.
fn size(b: usize) -> String {
    match b {
        b if b >= 1 << 30 => format!("{:.2} GB", b as f64 / 1e9),
        b if b >= 1 << 20 => format!("{:.2} MB", b as f64 / 1e6),
        b => format!("{:.1} kB", b as f64 / 1e3),
    }
}

fn device(s: &str) -> Result<Device> {
    rlx_cli::parse_device(s)
}

#[derive(Args)]
struct WeightsArgs {
    #[arg(long)]
    model_dir: PathBuf,
    /// Keep tensors whose name contains any of these.
    #[arg(long = "match")]
    include: Vec<String>,
    /// Layer range (`10..12`); top-level tensors are excluded when given.
    #[arg(long)]
    layers: Option<String>,
    /// Per-tensor statistics (reads every selected tensor, one at a time).
    #[arg(long)]
    stats: bool,
    /// Write the selected tensors as f32 safetensors into this directory.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Refuse dumps larger than this many GB of f32.
    #[arg(long, default_value_t = 8.0)]
    max_gb: f64,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct TraceArgs {
    #[arg(long)]
    model_dir: PathBuf,
    /// Prompt token ids.
    #[arg(long)]
    tokens: String,
    /// Next-token labels for `--grads` (default: the prompt shifted left).
    #[arg(long)]
    labels: Option<String>,
    /// Stop after this many layers (`n` or `0..n`; default all, with the head).
    #[arg(long)]
    layers: Option<String>,
    /// Keep activations whose name contains any of these.
    #[arg(long = "match")]
    include: Vec<String>,
    /// Write tensors and the manifest here.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Only statistics; no tensors written.
    #[arg(long)]
    stats_only: bool,
    /// Also record each layer's input cotangent of the next-token loss.
    #[arg(long)]
    grads: bool,
    #[arg(long, default_value = "cpu")]
    device: String,
}

#[derive(Args)]
struct GraphArgs {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: usize,
    #[arg(long, default_value_t = 4)]
    seq: usize,
    /// Every node instead of the op histogram.
    #[arg(long)]
    full: bool,
}

#[derive(Args, Clone)]
struct TrainCommon {
    #[arg(long)]
    model_dir: PathBuf,
    /// Token-id corpus: one sequence per line.
    #[arg(long, conflicts_with = "text")]
    tokens_file: Option<PathBuf>,
    /// Text corpus (blank-line separated documents), tokenized with `--tokenizer`
    /// (default `<model-dir>/tokenizer.json`).
    #[arg(long)]
    text: Option<PathBuf>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// `lora` or `full`.
    #[arg(long, default_value = "lora")]
    mode: String,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    #[arg(long, default_value_t = 32.0)]
    alpha: f32,
    /// LoRA target modules (substrings); default: every attention / FFN projection.
    #[arg(long = "target")]
    targets: Vec<String>,
    /// Never train params whose name contains any of these.
    #[arg(long)]
    freeze: Vec<String>,
    /// Only train params whose name contains any of these (e.g. `l92.`).
    #[arg(long)]
    only: Vec<String>,
    /// Full mode: also train the embedding table.
    #[arg(long)]
    train_embed: bool,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 64)]
    seq: usize,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    /// Micro-batches per optimizer step.
    #[arg(long, default_value_t = 1)]
    accum: usize,
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// Global gradient-norm clip (0 disables).
    #[arg(long, default_value_t = 1.0)]
    clip: f32,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    /// Cosine-decay the LR to this fraction of `--lr` (unset: constant).
    #[arg(long)]
    cosine: Option<f32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Keep layer weights in RAM between steps (when the model fits).
    #[arg(long)]
    cache_layers: bool,
    /// Output directory: `adapter/` (LoRA), `checkpoint/` (full or `--merge`),
    /// `state/` (resumable).
    #[arg(long)]
    out: PathBuf,
    /// Save resumable state every N steps (0: only at the end).
    #[arg(long, default_value_t = 0)]
    save_every: usize,
    /// Resume from a previous `--out` directory's `state/`.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// LoRA: also write a checkpoint with the adapter folded in.
    #[arg(long)]
    merge: bool,
    /// Write exported weights as bf16 (default f32).
    #[arg(long)]
    bf16: bool,
}

impl TrainCommon {
    fn spec(&self) -> Result<TrainSpec> {
        let mut s = match self.mode.as_str() {
            "lora" => TrainSpec::lora(self.rank, self.alpha, &self.targets),
            "full" => TrainSpec::full(),
            m => bail!("--mode must be lora or full, not {m:?}"),
        };
        s.train_embed = self.train_embed;
        s.freeze = self.freeze.clone();
        s.only = self.only.clone();
        Ok(s)
    }

    fn corpus(&self) -> Result<Vec<Vec<u32>>> {
        match (&self.tokens_file, &self.text) {
            (Some(f), _) => load_corpus(f, None),
            (None, Some(t)) => {
                let tk = self
                    .tokenizer
                    .clone()
                    .unwrap_or_else(|| self.model_dir.join("tokenizer.json"));
                load_corpus(t, Some(&tk))
            }
            (None, None) => bail!("give --tokens-file or --text"),
        }
    }

    fn lr_at(&self, step: usize) -> f32 {
        scheduled_lr(
            self.lr,
            step,
            self.steps,
            self.warmup,
            self.cosine.is_some(),
            self.cosine.unwrap_or(0.0),
        )
    }

    fn clip(&self) -> Option<f32> {
        (self.clip > 0.0).then_some(self.clip)
    }
}

#[derive(Args)]
struct TrainArgs {
    #[command(flatten)]
    c: TrainCommon,
    /// `stream` (one layer resident — the real checkpoint) or `whole` (one graph;
    /// only for models that fit).
    #[arg(long, default_value = "stream")]
    engine: String,
    /// Truncate to the first N layers (+ head): quick checks on big checkpoints.
    #[arg(long)]
    layers: Option<usize>,
    /// Data-parallel ranks on this machine (spawned automatically); across
    /// machines set RANK / WORLD / PEERS instead.
    #[arg(long)]
    nnodes: Option<u32>,
}

#[derive(Args)]
struct WorkerArgs {
    #[arg(long)]
    addr: String,
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Dense weights held as f32, bf16 (half the bytes, exact for a bf16
    /// checkpoint) or int8 (a quarter, lossy).
    #[arg(long, default_value = "f32")]
    dense: String,
}

#[derive(Args, Clone)]
struct StagesArgs {
    /// `host:port:start:end,…` — explicit layer ranges.
    #[arg(long, conflicts_with = "workers")]
    stages: Option<String>,
    /// `host:port,…` — ranges planned with `--split`.
    #[arg(long)]
    workers: Option<String>,
    /// Layer counts (`30,33,30`) or relative weights (`w:64,24,24`); default even.
    #[arg(long)]
    split: Option<String>,
    /// Seconds a worker may stay silent before it is declared unresponsive.
    #[arg(long, default_value_t = 60)]
    timeout: u64,
}

impl StagesArgs {
    fn resolve(&self, n_layers: usize) -> Result<Vec<StageSpec>> {
        match (&self.stages, &self.workers) {
            (Some(s), _) => parse_stages(s),
            (None, Some(w)) => {
                let addrs: Vec<String> = w.split(',').map(|a| a.trim().to_string()).collect();
                plan_stages(&addrs, n_layers, self.split.as_deref())
            }
            (None, None) => bail!("give --stages or --workers"),
        }
    }

    fn dist(&self) -> DistOptions {
        DistOptions {
            timeout: Duration::from_secs(self.timeout.max(1)),
            ..DistOptions::default()
        }
    }
}

#[derive(Args)]
struct PingArgs {
    #[arg(long)]
    model_dir: PathBuf,
    #[command(flatten)]
    s: StagesArgs,
}

#[derive(Args)]
struct GenerateArgs {
    #[arg(long)]
    model_dir: PathBuf,
    #[command(flatten)]
    s: StagesArgs,
    /// Prompt token ids (or use `--prompts-file`).
    #[arg(long, default_value = "")]
    tokens: String,
    /// One prompt of token ids per line; over workers they are pipelined so
    /// every stage stays busy.
    #[arg(long)]
    prompts_file: Option<PathBuf>,
    /// Dense weights held as f32, bf16 or int8 (see `worker --dense`).
    #[arg(long, default_value = "f32")]
    dense: String,
    /// Tokens to generate.
    #[arg(long, alias = "gen", default_value_t = 8)]
    max_tokens: usize,
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Write a KDA state log here (single-node only; feature `state-log`).
    #[arg(long)]
    state_log: Option<PathBuf>,
    /// State log: full snapshot every N tokens.
    #[arg(long, default_value_t = 64)]
    snapshot_every: usize,
    /// State log: skip the raw projections (conv windows then only at snapshots).
    #[arg(long)]
    no_conv: bool,
    /// State log: prefill one token per step so every prompt position's state is
    /// the run's own.
    #[arg(long)]
    stepwise_prefill: bool,
}

#[derive(Args)]
struct KdaStateArgs {
    /// A `generate --state-log` directory.
    #[arg(long)]
    log: PathBuf,
    /// Position (tokens consumed) to rebuild; default the last.
    #[arg(long)]
    position: Option<usize>,
    /// Write the rebuilt state as safetensors.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Rebuild every snapshot from the previous one and compare bit for bit.
    #[arg(long)]
    verify: bool,
    /// Device to rebuild on (default: the one recorded in the log).
    #[arg(long)]
    device: Option<String>,
}

#[derive(Args)]
struct SizeArgs {
    /// Checkpoint or config directory (exact bytes when it has an index); omit
    /// for the built-in Kimi-K3 reference dims.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Node preset (`--list-nodes`).
    #[arg(long, default_value = "mac-mini-m4pro-64")]
    node: String,
    #[arg(long)]
    list_nodes: bool,
    /// Size on every preset and print one row each.
    #[arg(long)]
    compare: bool,
    #[arg(long)]
    ram_gb: Option<f64>,
    #[arg(long)]
    ram_bw: Option<f64>,
    #[arg(long)]
    vram_gb: Option<f64>,
    #[arg(long)]
    vram_bw: Option<f64>,
    #[arg(long)]
    disk_gb: Option<f64>,
    /// Disk read GB/s.
    #[arg(long)]
    disk_gbs: Option<f64>,
    #[arg(long)]
    net_gbps: Option<f64>,
    /// Accelerator (or CPU, when no GPU) TFLOP/s.
    #[arg(long)]
    tflops: Option<f64>,
    #[arg(long)]
    reserve_gb: Option<f64>,
    /// `infer`, `lora` or `full`.
    #[arg(long, default_value = "infer")]
    mode: String,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    /// `paged` (on disk, read per step) or `resident` (in memory).
    #[arg(long, default_value = "paged")]
    experts: String,
    /// Paged experts: memory (GB, cluster total) caching recently used ones.
    #[arg(long, default_value_t = 0.0)]
    expert_cache_gb: f64,
    /// `pipeline` or `expert` (expert-parallel).
    #[arg(long, default_value = "pipeline")]
    parallel: String,
    /// Dense-weight residency: bf16, int8 or f32.
    #[arg(long, default_value = "bf16")]
    dense: String,
    /// KV context (inference) / sequence length (training).
    #[arg(long, default_value_t = 4096)]
    context: usize,
    /// Concurrent sequences (inference) / sequences per micro-batch (training).
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Fraction of nominal FLOP/s and bandwidth achieved.
    #[arg(long, default_value_t = 0.5)]
    efficiency: f64,
    /// Add nodes until this many tokens/s (if more nodes help).
    #[arg(long)]
    target_tps: Option<f64>,
    #[arg(long, default_value_t = 1024)]
    max_nodes: usize,
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct TrainDistArgs {
    #[command(flatten)]
    c: TrainCommon,
    #[command(flatten)]
    s: StagesArgs,
}

fn main() -> Result<()> {
    // `rlx-kimi-k3 <model-dir-or-config.json>` keeps working.
    let mut argv: Vec<String> = std::env::args().collect();
    if argv.len() == 2 && Path::new(&argv[1]).exists() {
        argv.insert(1, "info".into());
    }
    match Cli::parse_from(argv).cmd {
        Cmd::Info { model } => info(&model),
        Cmd::Synth(a) => synth(a),
        Cmd::Weights(a) => weights(a),
        Cmd::Trace(a) => trace_cmd(a),
        Cmd::Graph(a) => graph(a),
        Cmd::Train(a) => train(a),
        Cmd::Worker(a) => {
            rlx_kimi_k3::common::set_dense_residency(Some(a.dense.parse()?));
            serve(
                &a.addr,
                &a.model_dir.to_string_lossy(),
                ServeOptions {
                    device: device(&a.device)?,
                    ..ServeOptions::default()
                },
            )
        }
        Cmd::Ping(a) => ping(a),
        Cmd::Generate(a) => generate_cmd(a),
        Cmd::KdaState(a) => kda_state(a),
        Cmd::Size(a) => size_cmd(a),
        Cmd::TrainDist(a) => train_dist(a),
    }
}

fn info(model: &Path) -> Result<()> {
    let cfg = KimiK3Config::load(model)?;
    let t = &cfg.text_config;
    let kda = (0..t.num_hidden_layers)
        .filter(|&i| t.is_kda_layer(i))
        .count();
    println!(
        "Kimi-K3: {} layers ({kda} KDA + {} MLA), hidden {}, {} experts ({} active + {} shared), vocab {}",
        t.num_hidden_layers,
        t.num_hidden_layers - kda,
        t.hidden_size,
        t.num_experts.unwrap_or(0),
        t.num_experts_per_token,
        t.num_shared_experts,
        t.vocab_size,
    );
    println!(
        "  vision tower: {}",
        if cfg.vision_config.is_some() {
            "yes"
        } else {
            "no"
        }
    );
    Ok(())
}

fn synth(a: SynthArgs) -> Result<()> {
    let d = SynthSpec::default();
    let spec = SynthSpec {
        layers: a.layers,
        hidden: a.hidden,
        vocab: a.vocab,
        num_experts: a.experts,
        top_k: d.top_k.min(a.experts),
        // Every third layer (1-indexed) is MLA, the rest KDA — like the real mix.
        kda_layers: (1..=a.layers).filter(|l| l % 3 != 0).collect(),
        seed: a.seed,
        ..d
    };
    write_synthetic_checkpoint(&a.out, &spec)?;
    println!(
        "wrote a synthetic Kimi-K3 checkpoint to {}",
        a.out.display()
    );
    info(&a.out)
}

fn weights(a: WeightsArgs) -> Result<()> {
    let mut ck = CheckpointLoader::open(&a.model_dir)?;
    let inv = inventory(&mut ck)?;
    let layers = a.layers.as_deref().map(parse_range).transpose()?;
    let names: Vec<String> = logical_names(&inv)
        .into_iter()
        .filter(|n| selected(n, &a.include, layers.as_ref()))
        .collect();
    if a.dump.is_none() && !a.stats {
        let picked: Vec<_> = inv
            .iter()
            .filter(|e| {
                let logical = e
                    .name
                    .strip_suffix("_packed")
                    .or_else(|| e.name.strip_suffix("_scale"))
                    .unwrap_or(&e.name);
                selected(logical, &a.include, layers.as_ref())
            })
            .cloned()
            .collect();
        let s = summarize(&picked);
        if a.json {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({ "summary": s, "tensors": picked })
                )?
            );
            return Ok(());
        }
        for e in &picked {
            println!(
                "{:<80} {:>5} {:?} {}",
                e.name,
                e.dtype,
                e.shape,
                size(e.bytes)
            );
        }
        println!("\n{} tensors, {}", s.tensors, size(s.bytes));
        for (k, (n, b)) in &s.by_component {
            println!("  {k:<22} {n:>7} tensors {:>12}", size(*b));
        }
        for (k, b) in &s.by_dtype {
            println!("  dtype {k:<16} {:>12}", size(*b));
        }
        return Ok(());
    }
    if a.stats {
        let rows = weight_stats(&mut ck, &names, |n, shape, st| {
            if !a.json {
                println!("{n:<80} {shape:?}  {st}");
            }
        })?;
        if a.json {
            let v: Vec<_> = rows
                .iter()
                .map(|(n, s, st)| serde_json::json!({ "name": n, "shape": s, "stats": st }))
                .collect();
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
    }
    if let Some(out) = &a.dump {
        dump_weights(&mut ck, &names, out, Some((a.max_gb * 1e9) as usize))?;
        println!("dumped {} tensors to {}", names.len(), out.display());
    }
    Ok(())
}

fn trace_cmd(a: TraceArgs) -> Result<()> {
    let mut src = CheckpointSource::open(&a.model_dir)?;
    let tokens = parse_ids(&a.tokens)?;
    ensure!(!tokens.is_empty(), "no tokens");
    let n = src.num_layers();
    let layers = match &a.layers {
        Some(s) => parse_range(s)?,
        None => 0..n,
    };
    let labels = match &a.labels {
        Some(l) => parse_ids(l)?,
        None => {
            let mut l = tokens[1..].to_vec();
            l.push(*tokens.last().expect("non-empty"));
            l
        }
    };
    let cfg = src.text_config().flow_config(1, tokens.len());
    let opts = TraceOptions {
        layers,
        include: a.include,
        tensors: !a.stats_only,
        grads: a.grads,
        device: device(&a.device)?,
    };
    let s = trace(
        &mut src,
        &cfg,
        &tokens,
        Some(&labels),
        &opts,
        a.out.as_deref(),
    )?;
    for e in &s.entries {
        println!("{:<44} {:?}  {}", e.name, e.shape, e.stats);
    }
    for (l, ex) in &s.experts {
        println!("layer {l}: experts {ex:?}");
    }
    if !s.top5.is_empty() {
        println!("next-token top-5: {:?}", s.top5);
    }
    if let Some(l) = s.loss {
        println!("next-token loss: {l:.6}");
    }
    if let Some(o) = &a.out {
        println!("wrote {}", o.display());
    }
    Ok(())
}

fn graph(a: GraphArgs) -> Result<()> {
    let mut src = CheckpointSource::open(&a.model_dir)?;
    let cfg = src.text_config().flow_config(1, a.seq);
    print!("{}", layer_graph_text(&mut src, &cfg, a.layer, !a.full)?);
    Ok(())
}

/// Write the outputs of a run: adapter / checkpoint per mode.
fn export_outputs(
    c: &TrainCommon,
    spec: &TrainSpec,
    trained: &Params,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<()> {
    let (weights, adapters) = split_trained(trained);
    if let Some(lora) = spec.lora_spec() {
        let dir = c.out.join("adapter");
        save_lora_adapter(
            &dir,
            &adapters,
            shapes,
            lora.rank,
            lora.alpha,
            &lora.target_modules,
        )?;
        println!("LoRA adapter → {}", dir.display());
        if c.merge {
            let mut ck = CheckpointLoader::open(&c.model_dir)?;
            let mut base = Params::new();
            for name in adapters.keys() {
                if let Some(w) = name
                    .strip_prefix("lora.")
                    .and_then(|n| n.strip_suffix(".a"))
                {
                    base.insert(w.to_string(), load_graph_weight(&mut ck, w)?);
                }
            }
            merge_lora(&mut base, &adapters, shapes, lora.scale())?;
            let mut wshapes = HashMap::new();
            for (w, v) in &base {
                let k = shapes[&format!("lora.{w}.a")][0];
                wshapes.insert(w.clone(), vec![k, v.len() / k]);
            }
            let dir = c.out.join("checkpoint");
            export_full(&c.model_dir, &dir, &base, &wshapes, c.bf16)?;
            println!("merged checkpoint → {}", dir.display());
        }
    } else {
        let dir = c.out.join("checkpoint");
        export_full(&c.model_dir, &dir, &weights, shapes, c.bf16)?;
        println!(
            "checkpoint ({} updated tensors) → {}",
            weights.len(),
            dir.display()
        );
    }
    Ok(())
}

fn train(a: TrainArgs) -> Result<()> {
    use rlx_tune::cluster::{Role, launch_or_join};
    let (rank, world, comm) = match launch_or_join()? {
        Role::Launcher => return Ok(()),
        Role::Worker { rank, world, comm } => (rank, world, comm),
    };
    let _ = a.nnodes; // consumed by launch_or_join from argv
    let c = &a.c;
    let spec = c.spec()?;
    let dev = device(&c.device)?;
    let corpus = c.corpus()?;
    // Each data-parallel rank draws its own batches.
    let mut batcher = Batcher::new(corpus, c.seq, c.seed.wrapping_add(rank as u64 * 1_000_003))?;
    let lead = rank == 0;
    if lead {
        println!(
            "training Kimi-K3 ({} engine, {} mode) on {} rank(s): batch {} × seq {}, {} steps × {} micro-batches",
            a.engine, c.mode, world, c.batch, c.seq, c.steps, c.accum
        );
    }
    let state_dir = c.out.join("state");
    match a.engine.as_str() {
        "stream" => {
            let src = CheckpointSource::open(&c.model_dir)?;
            let cfg = src.text_config().flow_config(c.batch, c.seq);
            let opts = StreamOptions {
                lr: c.lr,
                weight_decay: c.weight_decay,
                clip: c.clip(),
                device: dev,
                seed: c.seed,
                cache_layers: c.cache_layers,
            };
            let mut st = StreamTrainer::new(Box::new(src), cfg, spec.clone(), opts, a.layers);
            if let Some(cm) = comm {
                st = st.with_comm(cm);
            }
            if let Some(r) = &c.resume {
                st.load_state(&r.join("state"))?;
                if lead {
                    println!("resumed at step {}", st.step);
                }
            }
            while st.step < c.steps {
                st.stage.opt.lr = c.lr_at(st.step);
                let micro: Vec<_> = (0..c.accum.max(1))
                    .map(|_| batcher.next(c.batch, c.seq))
                    .collect();
                let s = st.train_step(&micro)?;
                if lead {
                    println!("{s} | lr {:.3e}", st.stage.opt.lr);
                    if c.save_every > 0 && s.step % c.save_every == 0 {
                        st.save_state(&state_dir)?;
                    }
                }
            }
            if lead {
                st.save_state(&state_dir)?;
                export_outputs(c, &spec, st.trained(), &st.stage.shapes)?;
            }
        }
        "whole" => {
            ensure!(c.resume.is_none(), "--resume is for the stream engine");
            let mut src = CheckpointSource::open(&c.model_dir)?;
            let cfg = src.text_config().flow_config(c.batch, c.seq);
            let (w, embed) = load_flow_weights(&mut src, &cfg)?;
            let tg = build_train_graph(&w, &embed, &cfg, &spec, &HashMap::new(), c.seed)?;
            let mut dp = rlx_tune::DpConfig::new(c.lr)
                .device(dev)
                .weight_decay(c.weight_decay)
                .grad_accum(c.accum.max(1))
                .warmup(c.warmup);
            if let Some(m) = c.cosine {
                dp = dp.cosine(m);
            }
            if let Some(cl) = c.clip() {
                dp = dp.clip(cl);
            }
            let mut wt = WholeTrainer::new(tg, c.steps, comm.as_deref(), &dp)?;
            for _ in 0..c.steps {
                let m = wt.step_with(&mut || batcher.next(c.batch, c.seq))?;
                if lead {
                    println!("{m}");
                }
            }
            if lead {
                export_outputs(c, &spec, &wt.trained(), &wt.shapes)?;
            }
        }
        e => bail!("--engine must be stream or whole, not {e:?}"),
    }
    Ok(())
}

fn ping(a: PingArgs) -> Result<()> {
    let src = CheckpointSource::open(&a.model_dir)?;
    let tc = src.text_config().clone();
    let stages = a.s.resolve(tc.num_hidden_layers)?;
    let mut cluster = Cluster::connect(&stages, Some(&tc), &a.s.dist())?;
    for (s, c) in stages.iter().zip(&mut cluster.conns) {
        println!("{:<24} layers {:?}  {}", s.addr, s.layers, c.ping()?);
    }
    println!(
        "{} workers agree on the model and cover all {} layers",
        stages.len(),
        tc.num_hidden_layers
    );
    Ok(())
}

fn generate_cmd(a: GenerateArgs) -> Result<()> {
    let src = CheckpointSource::open(&a.model_dir)?;
    let tc = src.text_config().clone();
    rlx_kimi_k3::common::set_dense_residency(Some(a.dense.parse()?));
    let dev = device(&a.device)?;
    let mut ck = CheckpointLoader::open(&a.model_dir)?;
    let make = |seq: usize| tc.flow_config(1, seq);
    let distributed = a.s.stages.is_some() || a.s.workers.is_some();
    if let Some(f) = &a.prompts_file {
        ensure!(a.state_log.is_none(), "--state-log takes a single prompt");
        let prompts: Vec<Vec<u32>> = std::fs::read_to_string(f)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(parse_ids)
            .collect::<Result<_>>()?;
        let t = std::time::Instant::now();
        let outs = if distributed {
            let stages = a.s.resolve(tc.num_hidden_layers)?;
            let mut cluster = Cluster::connect(&stages, Some(&tc), &a.s.dist())?;
            rlx_kimi_k3::dist::generate_batch(
                &mut cluster,
                &mut ck,
                make,
                &prompts,
                a.max_tokens,
                dev,
            )?
        } else {
            prompts
                .iter()
                .map(|p| {
                    rlx_kimi_k3::runner::run_generate(
                        &mut ck,
                        &tc,
                        make,
                        p,
                        a.max_tokens,
                        tc.num_hidden_layers,
                        dev,
                    )
                })
                .collect::<Result<_>>()?
        };
        for o in &outs {
            println!("{o:?}");
        }
        let n = (outs.len() * a.max_tokens) as f64;
        println!(
            "{n} tokens in {:.2}s ({:.2} tokens/s)",
            t.elapsed().as_secs_f64(),
            n / t.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    let prompt = parse_ids(&a.tokens)?;
    ensure!(!prompt.is_empty(), "give --tokens or --prompts-file");
    let t = std::time::Instant::now();
    let out = if distributed {
        ensure!(a.state_log.is_none(), "--state-log is single-node only");
        let stages = a.s.resolve(tc.num_hidden_layers)?;
        let mut cluster = Cluster::connect(&stages, Some(&tc), &a.s.dist())?;
        generate(&mut cluster, &mut ck, make, &prompt, a.max_tokens, dev)?
    } else if let Some(dir) = &a.state_log {
        state_logged_generate(&a, &tc, &mut ck, &prompt, dir, dev)?
    } else {
        rlx_kimi_k3::runner::run_generate(
            &mut ck,
            &tc,
            make,
            &prompt,
            a.max_tokens,
            tc.num_hidden_layers,
            dev,
        )?
    };
    println!(
        "{out:?}  ({:.2}s, {:.2}s/token)",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() / a.max_tokens.max(1) as f64
    );
    Ok(())
}

#[cfg(feature = "state-log")]
fn state_logged_generate(
    a: &GenerateArgs,
    tc: &rlx_kimi_k3::config::KimiLinearConfig,
    ck: &mut CheckpointLoader,
    prompt: &[u32],
    dir: &Path,
    dev: Device,
) -> Result<Vec<u32>> {
    use rlx_kimi_k3::state_log::{StateLog, StateLogOptions, generate_logged};
    let out = generate_logged(
        ck,
        tc,
        |seq| tc.flow_config(1, seq),
        prompt,
        a.max_tokens,
        tc.num_hidden_layers,
        dev,
        &StateLogOptions {
            dir: dir.to_path_buf(),
            snapshot_every: a.snapshot_every,
            conv: !a.no_conv,
            stepwise_prefill: a.stepwise_prefill,
        },
    )?;
    let m = StateLog::open(dir)?.manifest;
    println!(
        "state log → {}: {} positions, {} KDA layers, {}/token + {} per snapshot ({} snapshots)",
        dir.display(),
        m.positions,
        m.kda_layers.len(),
        size(m.record_bytes()),
        size(m.snapshot_bytes()),
        m.snapshots.len()
    );
    Ok(out)
}

#[cfg(not(feature = "state-log"))]
fn state_logged_generate(
    _: &GenerateArgs,
    _: &rlx_kimi_k3::config::KimiLinearConfig,
    _: &mut CheckpointLoader,
    _: &[u32],
    _: &Path,
    _: Device,
) -> Result<Vec<u32>> {
    bail!("--state-log needs a build with `--features state-log`")
}

#[cfg(feature = "state-log")]
fn kda_state(a: KdaStateArgs) -> Result<()> {
    use rlx_kimi_k3::state_log::StateLog;
    let log = StateLog::open(&a.log)?;
    let m = &log.manifest;
    let dev = match &a.device {
        Some(d) => device(d)?,
        None => device(&m.device.to_lowercase())?,
    };
    println!(
        "{}: {} positions ({} prompt), KDA layers {:?}, snapshots {:?}",
        a.log.display(),
        m.positions,
        m.prompt_len,
        m.kda_layers,
        m.snapshots
    );
    if a.verify {
        let r = log.verify(dev)?;
        println!(
            "verify: {} spans, max |diff| {:e}, {}",
            r.spans.len(),
            r.max_abs_diff,
            if r.exact {
                "bit-exact"
            } else {
                "NOT bit-exact"
            }
        );
        ensure!(r.exact, "rebuilt states differ from the stored snapshots");
    }
    let p = a.position.unwrap_or(m.positions);
    let st = log.state_at(p, dev)?;
    let mut tensors = Vec::new();
    for l in &st {
        let s = rlx_kimi_k3::inspect::Stats::of(&l.scan);
        println!("layer {:>3} scan  {s}", l.layer);
        let (cw, proj) = (m.conv_kernel - 1, m.heads * m.head_dim);
        tensors.push((
            format!("l{}.scan", l.layer),
            vec![m.heads, m.head_dim, m.head_dim],
            l.scan.clone(),
        ));
        for (n, v) in [
            ("conv_q", &l.conv_q),
            ("conv_k", &l.conv_k),
            ("conv_v", &l.conv_v),
        ] {
            if !v.is_empty() {
                tensors.push((format!("l{}.{n}", l.layer), vec![cw, proj], v.clone()));
            }
        }
    }
    if let Some(out) = &a.dump {
        rlx_kimi_k3::inspect::write_f32_safetensors(out, &tensors)?;
        println!("state after {p} tokens → {}", out.display());
    }
    Ok(())
}

#[cfg(not(feature = "state-log"))]
fn kda_state(_: KdaStateArgs) -> Result<()> {
    bail!("kda-state needs a build with `--features state-log`")
}

fn train_dist(a: TrainDistArgs) -> Result<()> {
    let c = &a.c;
    let spec = c.spec()?;
    let src = CheckpointSource::open(&c.model_dir)?;
    let tc = src.text_config().clone();
    drop(src);
    let stages = a.s.resolve(tc.num_hidden_layers)?;
    let mut batcher = Batcher::new(c.corpus()?, c.seq, c.seed)?;
    let state_dir = c.out.join("state");
    let opts = PipelineOptions {
        spec: spec.clone(),
        batch: c.batch,
        seq: c.seq,
        lr: c.lr,
        weight_decay: c.weight_decay,
        clip: c.clip(),
        seed: c.seed,
        cache_layers: c.cache_layers,
        resume: c
            .resume
            .as_ref()
            .map(|r| r.join("state").to_string_lossy().to_string()),
        device: device(&c.device)?,
        dist: a.s.dist(),
    };
    let mut pt = PipelineTrainer::new(&c.model_dir, &stages, &opts)?;
    println!(
        "pipeline training over {} stages {:?}",
        stages.len(),
        stages.iter().map(|s| s.layers.clone()).collect::<Vec<_>>()
    );
    for step in 0..c.steps {
        pt.set_lr(c.lr_at(step))?;
        let micro: Vec<_> = (0..c.accum.max(1))
            .map(|_| batcher.next(c.batch, c.seq))
            .collect();
        let s = pt.train_step(&micro)?;
        println!("{s}");
        if c.save_every > 0 && s.step % c.save_every == 0 {
            pt.save(&state_dir.to_string_lossy())?;
        }
    }
    pt.save(&state_dir.to_string_lossy())?;
    let (trained, shapes) = pt.fetch_trained()?;
    export_outputs(c, &spec, &trained, &shapes)
}

fn size_cmd(a: SizeArgs) -> Result<()> {
    use rlx_kimi_k3::sizing::{
        Experts, Footprint, Mode, Parallel, Workload, estimate, kimi_k3_reference,
        nodes_for_throughput, preset, presets,
    };
    if a.list_nodes {
        for p in presets() {
            println!(
                "{:<24} RAM {:>5.0} GB @ {:>4.0} GB/s  VRAM {:>4.0} GB @ {:>5.0} GB/s  disk {:>5.0} GB @ {:>4.1} GB/s  net {:>3.0} Gb/s  {:>5.0} TFLOP/s{}",
                p.name,
                p.ram_gb,
                p.ram_bw_gbs,
                p.vram_gb,
                p.vram_bw_gbs,
                p.disk_gb,
                p.disk_read_gbs,
                p.net_gbps,
                if p.gpu_tflops > 0.0 {
                    p.gpu_tflops
                } else {
                    p.cpu_tflops
                },
                if p.unified { "  unified" } else { "" }
            );
        }
        return Ok(());
    }
    let (fp, source) = match &a.model_dir {
        Some(d) => {
            let tc = KimiK3Config::load(d)?.text_config;
            let fp = Footprint::from_config(&tc);
            if d.join("model.safetensors.index.json").exists() {
                let mut ck = CheckpointLoader::open(d)?;
                (
                    fp.with_checkpoint(&mut ck)?,
                    format!("{} (checkpoint bytes measured)", d.display()),
                )
            } else {
                (fp, format!("{} (config)", d.display()))
            }
        }
        None => (
            Footprint::from_config(&kimi_k3_reference()),
            "built-in Kimi-K3 reference dims".to_string(),
        ),
    };
    let w = Workload {
        mode: match a.mode.as_str() {
            "infer" => Mode::Infer,
            "lora" => Mode::TrainLora { rank: a.rank },
            "full" => Mode::TrainFull,
            m => bail!("--mode must be infer, lora or full, not {m:?}"),
        },
        experts: match a.experts.as_str() {
            "paged" => Experts::Paged {
                cache_gb: a.expert_cache_gb,
            },
            "resident" => Experts::Resident,
            e => bail!("--experts must be paged or resident, not {e:?}"),
        },
        parallel: match a.parallel.as_str() {
            "pipeline" => Parallel::Pipeline,
            "expert" => Parallel::ExpertParallel,
            p => bail!("--parallel must be pipeline or expert, not {p:?}"),
        },
        dense_bytes: match a.dense.as_str() {
            "bf16" => 2.0,
            "int8" => 1.0,
            "f32" => 4.0,
            d => bail!("--dense must be bf16, int8 or f32, not {d:?}"),
        },
        context: a.context,
        batch: a.batch,
        efficiency: a.efficiency,
    };
    let apply = |mut n: rlx_kimi_k3::sizing::NodeSpec| {
        if let Some(v) = a.ram_gb {
            n.ram_gb = v;
        }
        if let Some(v) = a.ram_bw {
            n.ram_bw_gbs = v;
        }
        if let Some(v) = a.vram_gb {
            n.vram_gb = v;
        }
        if let Some(v) = a.vram_bw {
            n.vram_bw_gbs = v;
        }
        if let Some(v) = a.disk_gb {
            n.disk_gb = v;
        }
        if let Some(v) = a.disk_gbs {
            n.disk_read_gbs = v;
        }
        if let Some(v) = a.net_gbps {
            n.net_gbps = v;
        }
        if let Some(v) = a.tflops {
            if n.gpu_tflops > 0.0 {
                n.gpu_tflops = v;
            } else {
                n.cpu_tflops = v;
            }
        }
        if let Some(v) = a.reserve_gb {
            n.reserve_gb = v;
        }
        n
    };
    let run = |node: &rlx_kimi_k3::sizing::NodeSpec| match a.target_tps {
        Some(t) => nodes_for_throughput(&fp, &w, node, t, a.max_nodes),
        None => Some(estimate(&fp, &w, node)),
    };
    if a.compare {
        println!("model: {source}");
        println!(
            "{:<24} {:>6} {:<18} {:>11} {:>11}  bottleneck",
            "node", "nodes", "set by", "tokens/s", "s/token"
        );
        for p in presets() {
            match run(&apply(p.clone())) {
                Some(e) => println!(
                    "{:<24} {:>6} {:<18} {:>11.3} {:>11.3}  {}",
                    p.name, e.nodes, e.limiting, e.tokens_per_s, e.latency_s, e.step.bottleneck
                ),
                None => println!(
                    "{:<24}   target not reachable within {} nodes",
                    p.name, a.max_nodes
                ),
            }
        }
        return Ok(());
    }
    let node = apply(preset(&a.node)?);
    let Some(e) = run(&node) else {
        bail!(
            "{} tokens/s is not reachable with up to {} {} nodes in this configuration{}",
            a.target_tps.unwrap_or_default(),
            a.max_nodes,
            node.name,
            if w.parallel == Parallel::Pipeline {
                " (a pipeline does not speed up one sequence: raise --batch, or try --parallel expert)"
            } else {
                ""
            }
        )
    };
    if a.json {
        println!("{}", serde_json::to_string_pretty(&e)?);
    } else {
        println!("source: {source}");
        print!("{e}");
        println!(
            "\n(roofline estimate from nominal specs × efficiency, uniform routing; not a measurement)"
        );
    }
    Ok(())
}
