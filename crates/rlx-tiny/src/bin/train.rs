//! Train a nanoGPT-style LLM from scratch on TinyStories.
//!
//! ```text
//! # auto-download the full train split and train on Metal:
//! cargo run --release -p rlx-tiny --bin rlx-tiny-train
//! # or point at a local text file, quick CPU run:
//! cargo run --release -p rlx-tiny --bin rlx-tiny-train -- \
//!     --data corpus.txt --device cpu --steps 500 --smoke
//! ```

use std::path::PathBuf;

use anyhow::{Result, bail};
use rlx_tensor::{DType, Device, Func, LrSchedule, is_available};

use rlx_tiny::config::GptConfig;
use rlx_tiny::data::Corpus;
use rlx_tiny::model;
use rlx_tiny::optim::HybridOptimizer;
use rlx_tiny::sample::GenOptions;
use rlx_tiny::{TrainOpts, Trainer};

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
    /// PQ-init the codebooks + u8 indices from a trained DENSE `rlx-tinystories`
    /// checkpoint (product quantization) instead of random init. Same tensors,
    /// data-derived values — starts far closer to the dense reference.
    init_from: Option<PathBuf>,
    /// Distill from a trained dense teacher: add `alpha ×` soft-CE against the
    /// teacher's per-token distribution to the training loss (no new model params).
    distill: Option<PathBuf>,
    cfg_overrides: Overrides,
}

/// Distillation mixing weight (soft-CE against the teacher). Fixed — no new knob.
const DISTILL_ALPHA: f32 = 0.5;

/// Default tokenizer vocab. BPE (`>256`) packs ~4 bytes/token on TinyStories, so
/// each fixed-length window carries ~4× more context ⇒ **far** lower bits/byte per
/// step than byte-level (validated: 2.94 → 1.36 bits/byte at matched steps). Opt
/// back to raw bytes with `--bpe 0`.
const DEFAULT_BPE_VOCAB: usize = 2048;
/// When BPE is on and no `--max-bytes` is given, cap the corpus here: BPE `encode`
/// materializes the whole split as a `Vec<u32>` (byte-level streams from the mmap
/// and needs no cap), so an uncapped 2.2 GB corpus would need ~9 GB of RAM. 64 MB
/// ⇒ ~25 M tokens, ample for the default step budget.
const BPE_DEFAULT_CORPUS_BYTES: usize = 64 << 20;

#[derive(Default)]
struct Overrides {
    batch: Option<usize>,
    seq: Option<usize>,
    layers: Option<usize>,
    embd: Option<usize>,
    heads: Option<usize>,
    synth_stages: Option<usize>,
    lora_rank: Option<usize>,
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
        out: PathBuf::from("weights/tiny/tinystories.rlxts"),
        eval_every: 200,
        sample_every: 500,
        seed: 1337,
        smoke: false,
        max_bytes: usize::MAX,
        bpe: DEFAULT_BPE_VOCAB,
        init_from: None,
        distill: None,
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
                a.precision = rlx_tiny::precision::parse(&name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown --precision {name:?} (expected {})",
                        rlx_tiny::precision::names()
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
            "--init-from" => a.init_from = Some(PathBuf::from(next(&mut i, &argv, "--init-from")?)),
            "--distill" => a.distill = Some(PathBuf::from(next(&mut i, &argv, "--distill")?)),
            "--batch" => a.cfg_overrides.batch = Some(next(&mut i, &argv, "--batch")?.parse()?),
            "--seq" => a.cfg_overrides.seq = Some(next(&mut i, &argv, "--seq")?.parse()?),
            "--layers" => a.cfg_overrides.layers = Some(next(&mut i, &argv, "--layers")?.parse()?),
            "--embd" => a.cfg_overrides.embd = Some(next(&mut i, &argv, "--embd")?.parse()?),
            "--heads" => a.cfg_overrides.heads = Some(next(&mut i, &argv, "--heads")?.parse()?),
            "--synth-stages" => {
                a.cfg_overrides.synth_stages = Some(next(&mut i, &argv, "--synth-stages")?.parse()?)
            }
            "--lora-rank" => {
                a.cfg_overrides.lora_rank = Some(next(&mut i, &argv, "--lora-rank")?.parse()?)
            }
            "--smoke" => a.smoke = true,
            "--diag-grads" => a.diag_grads = true,
            "-h" | "--help" => {
                println!(
                    "rlx-tiny-train [--data FILE] [--split train|valid] [--steps N]\n\
                     [--lr F (AdamW)] [--muon-lr F] [--grad-clip F] [--precision f32|f16|bf16]\n\
                     [--fake-quant nvf4|f8e4m3|bf8|f16|fXmYeZ] [--device cpu|metal] [--out FILE]\n\
                     [--seed N] [--smoke] [--batch N] [--seq N] [--layers N] [--embd N] [--heads N]\n\
                     [--synth-stages N (residual-VQ stages)] [--lora-rank N (0 disables)]\n\
                     [--init-from DENSE.rlxts (PQ codebook/index init from a dense checkpoint)]\n\
                     [--distill DENSE.rlxts (soft-CE distillation from a dense teacher)]\n\
                     [--eval-every N] [--sample-every N] [--max-bytes N]"
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
    if let Some(x) = o.synth_stages {
        cfg.synth_stages = x;
    }
    if let Some(x) = o.lora_rank {
        cfg.lora_rank = x;
    }
    cfg.check().map_err(|e| anyhow::anyhow!(e))?;
    Ok(cfg)
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
        let path = rlx_tiny::data::download(&a.split)?;
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
        _ => {
            if is_available(Device::Metal) {
                Device::Metal
            } else {
                Device::Cpu
            }
        }
    }
}

/// Load a dense `rlx-tinystories` checkpoint as `(architecture cfg, params)`.
fn load_dense(
    path: &std::path::Path,
) -> Result<(rlx_tinystories::config::GptConfig, Vec<(String, Vec<f32>)>)> {
    let (cfg, params, _bpe) = rlx_tinystories::checkpoint::load(path)?;
    Ok((cfg, params))
}

/// Force the synth model's architecture to match the dense reference (so every
/// weight / embedding / norm shape lines up 1:1 for PQ-init and distillation).
fn apply_dense_arch(cfg: &mut GptConfig, d: &rlx_tinystories::config::GptConfig) {
    cfg.vocab = d.vocab;
    cfg.block_size = d.block_size;
    cfg.n_layer = d.n_layer;
    cfg.n_head = d.n_head;
    cfg.n_embd = d.n_embd;
}

fn main() -> Result<()> {
    let mut args = parse_args()?;
    let mut cfg = resolve_config(&args)?;
    let dev = pick_device(&args.device);

    // ── Dense-init (--init-from) / distillation (--distill) setup ────────────
    // Both derive from a trained DENSE rlx-tinystories checkpoint and require the
    // synth model to share its architecture (shapes must line up). BPE is
    // unsupported here (the embeddings/vocab are copied from the dense byte-level
    // model), so guard against the combo.
    if (args.init_from.is_some() || args.distill.is_some()) && args.bpe > 0 {
        eprintln!(
            "note: --init-from/--distill copy byte-level embeddings from the dense model; disabling BPE for this run"
        );
        args.bpe = 0;
    }
    // PQ codebook/index init from the dense weights.
    let synth_init = if let Some(path) = &args.init_from {
        let (dcfg, dparams) = load_dense(path)?;
        apply_dense_arch(&mut cfg, &dcfg);
        cfg.check().map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "init-from: PQ-quantizing dense {} → codebooks (NE=256, ED=4), {} stages, lora_rank {}",
            path.display(),
            cfg.synth_stages,
            cfg.lora_rank
        );
        let si = model::SynthInit::from_dense(&cfg, &dparams);
        // Reconstruction fidelity — how faithfully the codebooks encode each weight.
        let report = si.reconstruction_report(&cfg, &dparams);
        let mut s1 = 0f32;
        let mut sf = 0f32;
        for (name, e1, ef) in &report {
            println!("  {name:9}  stage1 rel-err {e1:.4}  full(+lora) rel-err {ef:.4}");
            s1 += *e1;
            sf += *ef;
        }
        if !report.is_empty() {
            println!(
                "  mean reconstruction rel-err: stage1 {:.4}  full {:.4}  ({} weights)",
                s1 / report.len() as f32,
                sf / report.len() as f32,
                report.len()
            );
        }
        Some(si)
    } else {
        None
    };
    // Distillation teacher (dense forward → logits fed as a soft target).
    let teacher: Option<Func> = if let Some(path) = &args.distill {
        let (dcfg, dparams) = load_dense(path)?;
        apply_dense_arch(&mut cfg, &dcfg);
        cfg.check().map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "distill: teacher {} (soft-CE alpha {DISTILL_ALPHA})",
            path.display()
        );
        let mut t = rlx_tinystories::model::build(&dcfg, cfg.batch, false, DType::F32);
        for (n, d) in &dparams {
            t = t.with_param(n.clone(), d.clone());
        }
        Some(t)
    } else {
        None
    };

    let corpus = resolve_corpus(&args)?;
    // BPE `encode` materializes the whole corpus as `Vec<u32>`, so cap it for the
    // BPE path unless the user set --max-bytes explicitly. Byte-level streams from
    // the mmap and stays uncapped.
    let bpe_cap = if args.bpe > 0 && args.max_bytes == usize::MAX {
        BPE_DEFAULT_CORPUS_BYTES
    } else {
        usize::MAX
    };
    let used = corpus.len().min(args.max_bytes).min(bpe_cap);
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
        rlx_tiny::bpe::Bpe::train(data, args.bpe)
    });
    let ids: Vec<u32> = bpe.as_ref().map(|b| b.encode(data)).unwrap_or_default();
    let tokens = match &bpe {
        Some(_) => rlx_tiny::data::Tokens::Ids(&ids),
        None => rlx_tiny::data::Tokens::Bytes(data),
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
        "rlx-tiny: device={dev:?} params≈{} cfg={{layers:{}, embd:{}, heads:{}, ctx:{}, batch:{}}}",
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

    // Build + initialize the GPT (rlx! DSL graph), then train. The graph is the
    // same shape in every case — only the codebook/index *values* (PQ vs random)
    // and the training *loss* (distill adds a soft-CE term) differ.
    let distilling = teacher.is_some();
    let m = match (&synth_init, distilling) {
        (Some(si), false) => model::init_dense(
            model::build_dense_init(&cfg, cfg.batch, true, si),
            &cfg,
            si,
            args.seed,
        ),
        (Some(si), true) => model::init_dense(
            model::build_distill(&cfg, cfg.batch, Some(si), DISTILL_ALPHA),
            &cfg,
            si,
            args.seed,
        ),
        (None, true) => model::init(
            model::build_distill(&cfg, cfg.batch, None, DISTILL_ALPHA),
            &cfg,
            args.seed,
        ),
        (None, false) => model::init(model::build(&cfg, cfg.batch, true), &cfg, args.seed),
    };
    // The ACTUAL trainable-scalar count (codebooks + LoRA + embeddings + norms +
    // biases + KAN coeffs) — far fewer than the dense-equivalent capacity the
    // banner's `params≈` reports, since every weight is a synthesized codebook.
    let n_trainable: usize = m
        .param_names()
        .iter()
        .filter_map(|n| m.param_shape_of(n))
        .map(|dims| dims.iter().product::<usize>())
        .sum();
    println!(
        "trainable: {n_trainable} scalars across {} tensors (dense-equivalent capacity ≈{})",
        m.param_names().len(),
        cfg.n_params(),
    );
    let mut rng = rlx_tiny::Rng::new(args.seed ^ 0xD1B5);

    // Muon (2-D weight matrices) + AdamW (embeddings, biases, norms).
    let opt = HybridOptimizer::new(args.lr, args.muon_lr, 0.1);
    let sched = LrSchedule::WarmupCosine {
        base: args.lr,
        min: args.lr * 0.1,
        warmup: (args.steps / 20).max(1),
        total: args.steps,
    };

    // Assemble the trainer: model + optimizer + schedule + loop policy.
    let opts = TrainOpts {
        steps: args.steps,
        grad_clip: args.grad_clip,
        eval_every: args.eval_every,
        sample_every: args.sample_every,
        out: args.out.clone(),
        bpe,
        bytes_per_tok,
        gen_opts: GenOptions {
            max_new_tokens: 200,
            temperature: 0.8,
            top_k: 40,
            seed: 7,
        },
        // Emulated low-precision (QAT) params, if requested.
        fake_quant: args.fake_quant.as_ref().map(|(_, e, m, mx)| (*e, *m, *mx)),
    };
    let mut trainer = Trainer::new(cfg, dev, m, opt, sched, teacher, opts);

    // One-shot diagnostics run on the freshly-built model, then exit (before the
    // training banner, matching the original ordering).
    if std::env::var("RLX_BENCH_BWD").is_ok() {
        trainer.bench_backward(train_data, &mut rng);
        return Ok(());
    }
    if args.diag_grads {
        trainer.diagnose_grads(train_data, &mut rng);
        return Ok(());
    }

    println!(
        "optimizer: Muon(lr={:.0e}) + AdamW(lr={:.0e}), warmup-cosine; compute={:?}",
        args.muon_lr, args.lr, args.precision
    );
    if let Some((spec, e, m, mx)) = &args.fake_quant {
        println!(
            "  fake-quant (QAT): {spec} = E{e}M{m}, max {mx} — weights emulated at this precision"
        );
    }
    println!("grad-clip: {} (global L2 norm)", args.grad_clip);

    trainer.run(train_data, val_data, &mut rng)
}
