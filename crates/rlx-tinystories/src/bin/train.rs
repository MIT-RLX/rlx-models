//! Train a nanoGPT-style LLM from scratch on TinyStories.
//!
//! ```text
//! # auto-download the full train split and train on Metal:
//! cargo run --release -p rlx-tinystories --bin rlx-tinystories-train
//! # or point at a local text file, quick CPU run:
//! cargo run --release -p rlx-tinystories --bin rlx-tinystories-train -- \
//!     --data corpus.txt --device cpu --steps 500 --smoke
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use rlx_tensor::{DType, Device, Func, LrSchedule, is_available};

use rlx_tinystories::config::GptConfig;
use rlx_tinystories::data::{Batcher, Corpus};
use rlx_tinystories::optim::HybridOptimizer;
use rlx_tinystories::progress::Progress;
use rlx_tinystories::sample::{GenOptions, generate};
use rlx_tinystories::{checkpoint, model};

struct Args {
    data: Option<PathBuf>,
    split: String,
    steps: usize,
    lr: f32,
    muon_lr: f32,
    grad_clip: f32,
    precision: DType,
    /// Emulated low-precision QAT: `(spec, exp_bits, man_bits, max_normal)`.
    fake_quant: Option<(String, u32, u32, f32)>,
    /// One-shot: report per-parameter gradient finiteness, then exit.
    diag_grads: bool,
    device: Option<String>,
    out: PathBuf,
    eval_every: usize,
    sample_every: usize,
    seed: u64,
    smoke: bool,
    max_bytes: usize,
    /// Target BPE vocab size (0 = byte-level, 256 tokens). Denser tokens ⇒ more
    /// text per fixed-length sequence ⇒ fewer steps to a given bits/byte.
    bpe: usize,
    /// RAM budget (GB) for the training arena. When the estimated arena for the
    /// requested batch exceeds it, the batch is auto-capped so training won't
    /// OOM. `None` ⇒ `RLX_MAX_RAM_BYTES`, else 70% of physical RAM.
    max_ram_gb: Option<f64>,
    cfg_overrides: Overrides,
}

#[derive(Default)]
struct Overrides {
    batch: Option<usize>,
    seq: Option<usize>,
    layers: Option<usize>,
    embd: Option<usize>,
    heads: Option<usize>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        data: None,
        split: "train".into(),
        steps: 2000,
        lr: 3e-4,
        muon_lr: 2e-2,
        grad_clip: 1.0,
        precision: DType::F32,
        fake_quant: None,
        diag_grads: false,
        device: None,
        out: PathBuf::from("weights/tinystories/tinystories.rlxts"),
        eval_every: 200,
        sample_every: 500,
        seed: 1337,
        smoke: false,
        max_bytes: usize::MAX,
        bpe: 0,
        max_ram_gb: None,
        cfg_overrides: Overrides::default(),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |i: &mut usize, argv: &[String], flag: &str| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--data" => a.data = Some(PathBuf::from(next(&mut i, &argv, "--data")?)),
            "--split" => a.split = next(&mut i, &argv, "--split")?,
            "--steps" => a.steps = next(&mut i, &argv, "--steps")?.parse()?,
            "--lr" => a.lr = next(&mut i, &argv, "--lr")?.parse()?,
            "--muon-lr" => a.muon_lr = next(&mut i, &argv, "--muon-lr")?.parse()?,
            "--grad-clip" => a.grad_clip = next(&mut i, &argv, "--grad-clip")?.parse()?,
            "--precision" => {
                let name = next(&mut i, &argv, "--precision")?;
                a.precision = rlx_tinystories::precision::parse(&name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown --precision {name:?} (expected {})",
                        rlx_tinystories::precision::names()
                    )
                })?;
            }
            "--fake-quant" => {
                let spec = next(&mut i, &argv, "--fake-quant")?;
                let (e, m, max) = rlx_tensor::lowp::parse_format(&spec).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown --fake-quant {spec:?} (try nvf4|f8e4m3|bf8|f16 or generic fXmYeZ e.g. f8m3e4)"
                    )
                })?;
                a.fake_quant = Some((spec, e, m, max));
            }
            "--device" => a.device = Some(next(&mut i, &argv, "--device")?),
            "--out" => a.out = PathBuf::from(next(&mut i, &argv, "--out")?),
            "--eval-every" => a.eval_every = next(&mut i, &argv, "--eval-every")?.parse()?,
            "--sample-every" => a.sample_every = next(&mut i, &argv, "--sample-every")?.parse()?,
            "--seed" => a.seed = next(&mut i, &argv, "--seed")?.parse()?,
            "--max-bytes" => a.max_bytes = next(&mut i, &argv, "--max-bytes")?.parse()?,
            "--bpe" => a.bpe = next(&mut i, &argv, "--bpe")?.parse()?,
            "--max-ram-gb" => a.max_ram_gb = Some(next(&mut i, &argv, "--max-ram-gb")?.parse()?),
            "--batch" => a.cfg_overrides.batch = Some(next(&mut i, &argv, "--batch")?.parse()?),
            "--seq" => a.cfg_overrides.seq = Some(next(&mut i, &argv, "--seq")?.parse()?),
            "--layers" => a.cfg_overrides.layers = Some(next(&mut i, &argv, "--layers")?.parse()?),
            "--embd" => a.cfg_overrides.embd = Some(next(&mut i, &argv, "--embd")?.parse()?),
            "--heads" => a.cfg_overrides.heads = Some(next(&mut i, &argv, "--heads")?.parse()?),
            "--smoke" => a.smoke = true,
            "--diag-grads" => a.diag_grads = true,
            "-h" | "--help" => {
                println!(
                    "rlx-tinystories-train [--data FILE] [--split train|valid] [--steps N]\n\
                     [--lr F (AdamW)] [--muon-lr F] [--grad-clip F] [--precision f32|f16|bf16]\n\
                     [--fake-quant nvf4|f8e4m3|bf8|f16|fXmYeZ] [--device cpu|metal] [--out FILE]\n\
                     [--seed N] [--smoke] [--batch N] [--seq N] [--layers N] [--embd N] [--heads N]\n\
                     [--eval-every N] [--sample-every N] [--max-bytes N] [--max-ram-gb G]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?} (try --help)"),
        }
        i += 1;
    }
    Ok(a)
}

fn resolve_config(a: &Args) -> Result<GptConfig> {
    let mut cfg = if a.smoke {
        GptConfig::smoke()
    } else {
        GptConfig::default_metal()
    };
    let o = &a.cfg_overrides;
    if let Some(x) = o.batch {
        cfg.batch = x;
    }
    if let Some(x) = o.seq {
        cfg.block_size = x;
    }
    if let Some(x) = o.layers {
        cfg.n_layer = x;
    }
    if let Some(x) = o.embd {
        cfg.n_embd = x;
    }
    if let Some(x) = o.heads {
        cfg.n_head = x;
    }
    // Auto-cap the batch to a RAM budget so training won't OOM at execution
    // time. Budget: --max-ram-gb, else RLX_MAX_RAM_BYTES, else 70% of physical
    // RAM. The estimate is linear in batch (accurate at large batch where OOM
    // bites, conservative at small batch), so capping only ever shrinks toward
    // safety.
    if let Some(budget) = ram_budget_bytes(a) {
        let want = cfg.batch;
        let est = estimate_arena_bytes(&cfg, want);
        if est > budget {
            let per1 = estimate_arena_bytes(&cfg, 1).max(1);
            let max_b = (budget / per1).max(1);
            eprintln!(
                "auto-batch: est arena {:.1} GB @ batch={want} > budget {:.1} GB \
                 → capping batch {want} → {max_b} (set --max-ram-gb to override)",
                est as f64 / 1e9,
                budget as f64 / 1e9,
            );
            cfg.batch = max_b;
        }
    }
    cfg.check().map_err(|e| anyhow::anyhow!(e))?;
    Ok(cfg)
}

/// RAM budget for the training arena, in bytes: `--max-ram-gb`, else
/// `RLX_MAX_RAM_BYTES`, else 70% of physical RAM (`sysctl hw.memsize`).
fn ram_budget_bytes(a: &Args) -> Option<usize> {
    if let Some(g) = a.max_ram_gb {
        return Some((g * 1e9) as usize);
    }
    if let Ok(v) = std::env::var("RLX_MAX_RAM_BYTES") {
        if let Ok(b) = v.parse::<usize>() {
            return Some(b);
        }
    }
    physical_ram_bytes().map(|r| r * 7 / 10)
}

/// Physical RAM in bytes via `sysctl -n hw.memsize` (macOS). `None` if it can't
/// be determined — the caller then applies no cap.
fn physical_ram_bytes() -> Option<usize> {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Estimated peak training-arena bytes for `batch`. Linear model calibrated to
/// the measured ~85 MB/batch at seq=256, 6L, embd=256, heads=8 (ff=1024): the
/// `n_head*seq` term captures the O(seq²) attention scores; the 4.2 factor
/// covers the fwd+bwd + gradient + optimizer live set. Over-predicts at small
/// batch (safe) and is accurate at large batch (where OOM actually occurs).
fn estimate_arena_bytes(cfg: &GptConfig, batch: usize) -> usize {
    let ff = cfg.ffn();
    let per_token = cfg.n_embd + ff + cfg.n_head * cfg.block_size;
    let per_batch = 4.2 * (cfg.n_layer * cfg.block_size * per_token * 4) as f64;
    (per_batch * batch as f64) as usize
}

fn resolve_corpus(a: &Args) -> Result<Corpus> {
    if let Some(path) = &a.data {
        return Corpus::open(path);
    }
    #[cfg(feature = "download")]
    {
        eprintln!(
            "no --data given: downloading TinyStories '{}' split from the Hub (first run only)…",
            a.split
        );
        let path = rlx_tinystories::data::download(&a.split)?;
        eprintln!("corpus: {}", path.display());
        Corpus::open(&path)
    }
    #[cfg(not(feature = "download"))]
    {
        let _ = a;
        bail!("no --data given and the `download` feature is disabled; pass --data FILE")
    }
}

fn pick_device(arg: &Option<String>) -> Device {
    match arg.as_deref() {
        Some("cpu") => Device::Cpu,
        Some("metal") => Device::Metal,
        Some("cuda") => Device::Cuda,
        _ => {
            if is_available(Device::Metal) {
                Device::Metal
            } else {
                Device::Cpu
            }
        }
    }
}

/// Snapshot the current weights (name → data) for checkpointing / generation.
fn params_of(model: &Func) -> Vec<(String, Vec<f32>)> {
    model
        .param_names()
        .into_iter()
        .map(|n| {
            let d = model.param_binding(&n).unwrap().to_vec();
            (n, d)
        })
        .collect()
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut cfg = resolve_config(&args)?;
    let dev = pick_device(&args.device);

    let corpus = resolve_corpus(&args)?;
    let used = corpus.len().min(args.max_bytes);
    let data = &corpus.bytes()[..used];

    // Tokenizer: byte-level (id == byte, 256 tokens) by default, or a
    // from-scratch BPE trained on this corpus. BPE packs more text per token, so
    // a fixed-length sequence carries more context ⇒ fewer steps to a target
    // bits/byte. The gather embedding makes the larger vocab affordable (ids on
    // the bus, not a one-hot). `bytes_per_tok` renormalizes the per-token loss
    // to true bits/byte, so byte-level and BPE runs compare on equal footing.
    let bpe = (args.bpe > 0).then(|| {
        println!(
            "training BPE (target vocab {}) on {:.1} MB…",
            args.bpe,
            data.len() as f64 / 1e6
        );
        rlx_tinystories::bpe::Bpe::train(data, args.bpe)
    });
    let ids: Vec<u32> = bpe.as_ref().map(|b| b.encode(data)).unwrap_or_default();
    let tokens = match &bpe {
        Some(_) => rlx_tinystories::data::Tokens::Ids(&ids),
        None => rlx_tinystories::data::Tokens::Bytes(data),
    };
    if let Some(b) = &bpe {
        cfg.vocab = b.vocab_size();
    }
    let bytes_per_tok = if tokens.is_empty() {
        1.0
    } else {
        data.len() as f32 / tokens.len() as f32
    };

    // 99% train / 1% held-out tail for the eval-loss estimate.
    let split = (tokens.len() * 99 / 100)
        .max(cfg.block_size + 1)
        .min(tokens.len().saturating_sub(cfg.block_size + 1));
    let train_data = tokens.range(0..split);
    let val_data = tokens.range(split..tokens.len());

    println!(
        "rlx-tinystories: device={dev:?} params≈{} cfg={{layers:{}, embd:{}, heads:{}, ctx:{}, batch:{}}}",
        cfg.n_params(),
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.block_size,
        cfg.batch,
    );
    println!(
        "corpus: {} bytes ({:.1} MB), vocab={} tokens={} ({:.2} bytes/token), train={} val={}",
        corpus.len(),
        corpus.len() as f64 / 1e6,
        cfg.vocab,
        tokens.len(),
        bytes_per_tok,
        train_data.len(),
        val_data.len(),
    );

    // Build + initialize the GPT (rlx! DSL graph), then train.
    let mut m = model::init(
        model::build(&cfg, cfg.batch, true, args.precision),
        &cfg,
        args.seed,
    );
    let batcher = Batcher::new(&cfg);
    let mut rng = rlx_tinystories::Rng::new(args.seed ^ 0xD1B5);

    // Backward-kernel benchmark: warm the compile cache, then time forward-only
    // (`run_on` of the loss) vs fwd+bwd (`value_and_grad_all().run_on`). The
    // difference is the backward. `RLX_BENCH_BWD=1` → print + exit.
    if std::env::var("RLX_BENCH_BWD").is_ok() {
        let (tok, tgt) = batcher.sample(&train_data, &mut rng);
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        let vg = m.value_and_grad_all();
        for _ in 0..3 {
            let _ = m.run_on(dev, feed);
            let _ = vg.run_on(dev, feed);
        }
        let n = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = m.run_on(dev, feed);
        }
        let fwd = t0.elapsed().as_secs_f64() / n as f64 * 1e3;
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            let _ = vg.run_on(dev, feed);
        }
        let fwdbwd = t1.elapsed().as_secs_f64() / n as f64 * 1e3;
        println!(
            "BWD_BENCH forward={fwd:.1}ms fwd+bwd={fwdbwd:.1}ms backward={:.1}ms ratio={:.2}x",
            fwdbwd - fwd,
            (fwdbwd - fwd) / fwd
        );
        return Ok(());
    }

    // One-shot gradient diagnostic: which parameter's grad goes non-finite?
    if args.diag_grads {
        let (tok, tgt) = batcher.sample(&train_data, &mut rng);
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        let names = m.param_names();
        let out = m.value_and_grad_all().run_on(dev, feed);
        println!("diag: loss = {}", out[0][0]);
        let mut bad = 0;
        for (i, n) in names.iter().enumerate() {
            let g = &out[i + 1];
            let finite = g.iter().all(|x| x.is_finite());
            let maxabs = g.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let flag = if !finite { " <-- NON-FINITE" } else { "" };
            println!("  {n:10}  finite={finite}  maxabs={maxabs:.3e}{flag}");
            if !finite {
                bad += 1;
            }
        }
        println!("diag: {bad}/{} params with non-finite grads", names.len());
        rlx_tensor::clear_cache();
        return Ok(());
    }

    // Muon (2-D weight matrices) + AdamW (embeddings, biases, norms).
    let mut opt = HybridOptimizer::new(args.lr, args.muon_lr, 0.1);
    let sched = LrSchedule::WarmupCosine {
        base: args.lr,
        min: args.lr * 0.1,
        warmup: (args.steps / 20).max(1),
        total: args.steps,
    };
    println!(
        "optimizer: Muon(lr={:.0e}) + AdamW(lr={:.0e}), warmup-cosine; compute={:?}",
        args.muon_lr, args.lr, args.precision
    );
    if let Some((spec, e, m, mx)) = &args.fake_quant {
        println!(
            "  fake-quant (QAT): {spec} = E{e}M{m}, max {mx} — weights emulated at this precision"
        );
    }

    let gen_opts = GenOptions {
        max_new_tokens: 200,
        temperature: 0.8,
        top_k: 40,
        seed: 7,
    };

    println!("grad-clip: {} (global L2 norm)", args.grad_clip);

    // Optimizer-only microbench: isolates the CPU-side Muon/AdamW step (the
    // GPU-idle bubble) from the fwd+bwd graph, on the real model's parameter
    // shapes. `RLX_BENCH_OPT=1` times `step_batch` best-of-N for serial
    // (Muon then AdamW) vs `RLX_TS_OPT_OVERLAP=1` (Muon ∥ AdamW), then exits.
    if std::env::var("RLX_BENCH_OPT").is_ok() {
        use rlx_tensor::{OptItem, Optimizer};
        opt.set_lr(args.lr);
        let names: Vec<String> = m.param_names();
        let shapes: Vec<Vec<usize>> = names.iter().map(|n| m.param_shape_of(n).unwrap()).collect();
        let mut datas: Vec<Vec<f32>> = names
            .iter()
            .map(|n| m.param_binding(n).unwrap().to_vec())
            .collect();
        // Deterministic, nonzero pseudo-gradients (identical for both modes) so
        // Newton–Schulz does real matmul work; scaled small so trust-region is
        // representative.
        let grads: Vec<Vec<f32>> = datas
            .iter()
            .enumerate()
            .map(|(i, d)| {
                (0..d.len())
                    .map(|k| {
                        let h = (i
                            .wrapping_mul(2654435761)
                            .wrapping_add(k.wrapping_mul(40503)))
                            & 0xffff;
                        (h as f32 / 65535.0 - 0.5) * 0.02
                    })
                    .collect()
            })
            .collect();
        let n_muon = names
            .iter()
            .zip(&shapes)
            .filter(|(n, s)| s.len() == 2 && !n.starts_with("wte") && !n.starts_with("wpe"))
            .count();
        println!(
            "BENCH_OPT: {} params ({} Muon 2-D / {} AdamW), {} scalars",
            names.len(),
            n_muon,
            names.len() - n_muon,
            datas.iter().map(|d| d.len()).sum::<usize>()
        );
        let mut time_mode = |overlap: &str, iters: usize| -> f64 {
            unsafe {
                std::env::set_var("RLX_TS_OPT_OVERLAP", overlap);
            }
            let mut best = f64::INFINITY;
            for it in 0..iters {
                let mut items: Vec<OptItem> = datas
                    .iter_mut()
                    .enumerate()
                    .map(|(i, d)| OptItem {
                        name: &names[i],
                        shape: &shapes[i],
                        param: d.as_mut_slice(),
                        grad: &grads[i],
                    })
                    .collect();
                let t = Instant::now();
                opt.step_batch(&mut items);
                let el = t.elapsed().as_secs_f64() * 1e3;
                opt.end_iteration();
                if it >= 10 && el < best {
                    best = el; // skip warmup iters
                }
            }
            best
        };
        // Interleave to share thermal state; best-of-many each.
        let iters = 260;
        let mut s = f64::INFINITY;
        let mut o = f64::INFINITY;
        for _ in 0..4 {
            s = s.min(time_mode("0", iters));
            o = o.min(time_mode("1", iters));
        }
        println!(
            "BENCH_OPT: serial={s:.3}ms  overlap={o:.3}ms  speedup={:.2}x  saved={:.3}ms ({:.1}%)",
            s / o,
            s - o,
            (s - o) / s * 100.0
        );
        rlx_tensor::clear_cache();
        return Ok(());
    }

    // Emulated low-precision (QAT) params, if requested.
    let fq: Option<(u32, u32, f32)> = args.fake_quant.as_ref().map(|(_, e, m, mx)| (*e, *m, *mx));

    // Full training-step benchmark: warm the compile cache + optimizer state,
    // then time the real step (fwd + bwd + grad-clip + Muon/AdamW update).
    // `RLX_BENCH_TRAIN=1` → print tokens/s + per-step ms and exit.
    if std::env::var("RLX_BENCH_TRAIN").is_ok() {
        let bt = (cfg.batch * cfg.block_size) as f64;
        let mut step_of = |m: Func, step: usize| -> Func {
            let (tok, tgt) = batcher.sample(&train_data, &mut rng);
            let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
            m.train_step_all_at_on_clipped(dev, &mut opt, &sched, step, args.grad_clip, feed)
                .0
        };
        for step in 0..3 {
            m = step_of(m, step);
        }
        let n = 10;
        let t = Instant::now();
        for step in 3..3 + n {
            m = step_of(m, step);
        }
        let ms = t.elapsed().as_secs_f64() / n as f64 * 1e3;
        println!(
            "TRAIN_BENCH device={dev:?} batch={} seq={} BT={} step={ms:.1}ms tok/s={:.0}",
            cfg.batch,
            cfg.block_size,
            cfg.batch * cfg.block_size,
            bt / (ms / 1e3)
        );
        rlx_tensor::clear_cache();
        return Ok(());
    }

    // Gradient recording: `RLX_RECORD_GRADS=path` → for each of `--steps` steps,
    // dump per-parameter RAW gradient statistics (L2 norm, mean, std, |max|) plus
    // the global grad norm and loss to JSONL, then take the real optimizer step.
    // Grads come from `value_and_grad_all` (pre-clip); the identical feed then
    // drives `train_step` so the recorded grads match the ones that trained.
    if let Ok(path) = std::env::var("RLX_RECORD_GRADS") {
        use std::io::Write;
        let names = m.param_names();
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for step in 0..args.steps {
            let (tok, tgt) = batcher.sample(&train_data, &mut rng);
            let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
            // Time forward+backward (value_and_grad) separately from the optimizer
            // step so the JSONL doubles as a per-step throughput/latency bench.
            let t_vg = Instant::now();
            let out = m.value_and_grad_all().run_on(dev, feed);
            let vg_ms = t_vg.elapsed().as_secs_f64() * 1e3;
            let loss = out[0][0];
            let mut global_sq = 0f64;
            let mut line =
                format!("{{\"step\":{step},\"loss\":{loss:.6},\"vg_ms\":{vg_ms:.3},\"grads\":{{");
            for (i, name) in names.iter().enumerate() {
                let g = &out[i + 1];
                let n = g.len().max(1) as f64;
                let sum: f64 = g.iter().map(|&x| x as f64).sum();
                let sq: f64 = g.iter().map(|&x| (x as f64) * (x as f64)).sum();
                let mean = sum / n;
                let std = (sq / n - mean * mean).max(0.0).sqrt();
                let absmax = g.iter().fold(0f32, |a, &x| a.max(x.abs()));
                global_sq += sq;
                if i > 0 {
                    line.push(',');
                }
                line.push_str(&format!(
                    "\"{name}\":{{\"norm\":{:.6e},\"mean\":{mean:.6e},\"std\":{std:.6e},\"absmax\":{absmax:.6e}}}",
                    sq.sqrt()
                ));
            }
            // Take the real step so params progress (same feed ⇒ same grads).
            let t_step = Instant::now();
            m = m
                .train_step_all_at_on_clipped(dev, &mut opt, &sched, step, args.grad_clip, feed)
                .0;
            let step_ms = t_step.elapsed().as_secs_f64() * 1e3;
            line.push_str(&format!(
                "}},\"global_norm\":{:.6e},\"step_ms\":{step_ms:.3}}}\n",
                global_sq.sqrt()
            ));
            f.write_all(line.as_bytes())?;
            if (step + 1) % 50 == 0 {
                eprintln!(
                    "  recorded step {}/{}  loss {loss:.3}  vg {vg_ms:.0}ms step {step_ms:.0}ms",
                    step + 1,
                    args.steps
                );
            }
        }
        f.flush()?;
        println!("recorded {} steps of gradient stats → {path}", args.steps);
        rlx_tensor::clear_cache();
        return Ok(());
    }

    let t0 = Instant::now();
    let mut ema = f32::NAN;
    let mut best_val = f32::INFINITY;
    let mut saved_any = false;
    let mut progress = Progress::new(args.steps);
    for step in 0..args.steps {
        let (tok, tgt) = batcher.sample(&train_data, &mut rng);
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        let (next, loss) = if let Some((e, mb, mx)) = fq {
            // Quantization-aware step: weights → fXmYeZ grid on the forward,
            // straight-through gradient to the f32 masters.
            m.train_step_all_at_on_qat(
                dev,
                &mut opt,
                &sched,
                step,
                args.grad_clip,
                |w| rlx_tensor::lowp::quantize_slice_scaled(w, e, mb, mx),
                feed,
            )
        } else {
            m.train_step_all_at_on_clipped(dev, &mut opt, &sched, step, args.grad_clip, feed)
        };
        m = next;
        let l = loss[0];
        ema = if ema.is_nan() {
            l
        } else {
            0.95 * ema + 0.05 * l
        };
        let is_last = step + 1 == args.steps;
        progress.tick(step + 1, ema, sched.lr_at(step));

        // Diverged? Stop — the best checkpoint on disk is retained.
        if !l.is_finite() {
            progress.note(&format!(
                "  !! train loss became {l} at step {} — stopping (best checkpoint kept)",
                step + 1
            ));
            break;
        }

        let do_eval = args.eval_every > 0
            && ((step + 1) % args.eval_every == 0 || is_last)
            && val_data.len() > cfg.block_size + 1;
        if do_eval {
            let (vtok, vtgt) = batcher.sample(&val_data, &mut rng);
            let vfeed: &[(&str, &[f32])] = &[("tok_ids", &vtok), ("tgt_ids", &vtgt)];
            let vloss = m.run_on(dev, vfeed)[0][0];
            // Keep-best: only persist when the held-out loss improves, so a
            // later divergence can never overwrite the good weights.
            let mark = if vloss.is_finite() && vloss < best_val {
                best_val = vloss;
                match checkpoint::save(&args.out, &cfg, &params_of(&m), bpe.as_ref()) {
                    Ok(()) => {
                        saved_any = true;
                        " ★ best (saved)"
                    }
                    Err(e) => {
                        progress.note(&format!("  !! checkpoint save failed: {e}"));
                        ""
                    }
                }
            } else {
                ""
            };
            // Per-token loss → true bits/byte (÷ bytes-per-token), so byte-level
            // and BPE runs are on the same axis: fewer bits/byte = better model.
            progress.note(&format!(
                "  ├─ step {}  val loss {vloss:.4}  (bits/byte {:.3}){mark}",
                step + 1,
                vloss / std::f32::consts::LN_2 / bytes_per_tok
            ));
        }

        if args.sample_every > 0 && ((step + 1) % args.sample_every == 0 || is_last) {
            let story = generate(
                &cfg,
                &params_of(&m),
                "Once upon a time",
                dev,
                &gen_opts,
                bpe.as_ref(),
            );
            progress.note(&format!("  └─ sample: {}", story.replace('\n', "⏎")));
        }
    }
    progress.finish();

    // Fallback save if keep-best never fired (e.g. --eval-every 0 or no val data).
    if !saved_any {
        checkpoint::save(&args.out, &cfg, &params_of(&m), bpe.as_ref())?;
    }
    println!(
        "done in {:.1}s → best val loss {:.4}, checkpoint {}",
        t0.elapsed().as_secs_f64(),
        best_val,
        args.out.display()
    );
    rlx_tensor::clear_cache();
    Ok(())
}
