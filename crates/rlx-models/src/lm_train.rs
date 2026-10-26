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

//! Fine-tuning and quantization-aware training for the LM families that can
//! produce a full-sequence training graph.
//!
//! A family registers a constructor ([`register_lm_trainer`]) that builds its
//! [`TrainableLm`] — the causal forward from token ids to logits with every
//! weight as an f32 param — from a checkpoint directory; [`qat_lm`] does the
//! rest through `rlx_tune::lm`. The checkpoint must be dense safetensors
//! (`model.safetensors`); packed GGUF weights cannot be trained.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use anyhow::{Context, Result, bail, ensure};
use rlx_runtime::NumberFormat;
pub use rlx_tune::lm::{LmTrainOptions, TrainableLm, TrainedLm};

/// Build a family's training forward for `batch × seq` tokens from a
/// checkpoint directory.
pub type LmTrainerCtor = fn(dir: &Path, batch: usize, seq: usize) -> Result<TrainableLm>;

fn registry() -> &'static RwLock<HashMap<&'static str, LmTrainerCtor>> {
    static R: OnceLock<RwLock<HashMap<&'static str, LmTrainerCtor>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register `ctor` as the trainer of runner family `family` (the name
/// `auto_runner` sniffs, e.g. `"qwen3"`).
pub fn register_lm_trainer(family: &'static str, ctor: LmTrainerCtor) {
    registry()
        .write()
        .expect("lm trainer registry")
        .insert(family, ctor);
}

/// Families with a registered trainer.
pub fn registered_lm_trainers() -> Vec<&'static str> {
    register_builtin_lm_trainers();
    let mut v: Vec<_> = registry()
        .read()
        .expect("lm trainer registry")
        .keys()
        .copied()
        .collect();
    v.sort_unstable();
    v
}

fn register_builtin_lm_trainers() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        #[cfg(feature = "qwen3")]
        register_lm_trainer("qwen3", qwen3_trainable);
        #[cfg(feature = "llama32")]
        register_lm_trainer("llama32", llama32_trainable);
        #[cfg(feature = "gemma")]
        {
            register_lm_trainer("gemma", gemma_trainable);
            // TranslateGemma checkpoints are Gemma 3 with a translation prompt.
            register_lm_trainer("translategemma", gemma_trainable);
        }
        // Dense Mistral checkpoints are Llama-shaped (`MistralRunner` wraps
        // the llama32 runner); Qwen3-VL's text tower is plain Qwen3.
        #[cfg(feature = "llama32")]
        register_lm_trainer("mistral", llama32_trainable);
        #[cfg(feature = "qwen3")]
        register_lm_trainer("qwen3-vl", qwen3_trainable);
        #[cfg(feature = "minimax")]
        register_lm_trainer("minimax-m3", minimax_m3_trainable);
        register_lm_trainer("qwen4exp", |_, _, _| {
            bail!(
                "qwen4exp cannot be trained yet: its MoE router (`group_limited_gate`) has no \
                 gradient, so routing would stay frozen, and its per-layer embeddings need a \
                 token-dependent host input the training loop does not build"
            )
        });
        register_lm_trainer("lfm", |_, _, _| {
            bail!(
                "lfm cannot be trained yet: it only has a single-step decode graph, and its SSM \
                 step op has no gradient"
            )
        });
        #[cfg(feature = "qwen35")]
        register_lm_trainer("qwen35", qwen35_trainable);
    });
}

/// The checkpoint directory's safetensors weights file: `model.safetensors`,
/// or the only `.safetensors` file (a one-shard checkpoint).
fn dense_weights(dir: &Path) -> Result<PathBuf> {
    let file = dir.join("model.safetensors");
    if file.is_file() {
        return Ok(file);
    }
    let shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    match shards.as_slice() {
        [one] => Ok(one.clone()),
        [] => bail!(
            "{}: training needs dense safetensors weights (packed GGUF cannot be trained)",
            dir.display()
        ),
        _ => bail!(
            "{}: sharded checkpoints are not supported for training yet",
            dir.display()
        ),
    }
}

#[cfg(feature = "qwen3")]
fn qwen3_trainable(dir: &Path, batch: usize, seq: usize) -> Result<TrainableLm> {
    let cfg = rlx_qwen3::Qwen3Config::from_file(&dir.join("config.json"))?;
    let file = dense_weights(dir)?;
    let mut wm =
        rlx_core::weight_map::WeightMap::from_file(file.to_str().context("non-UTF8 path")?)?;
    let (graph, params) =
        rlx_qwen3::build_qwen3_graph_sized(&cfg, &mut wm, batch, seq, true, false)?;
    Ok(TrainableLm {
        token_input: token_input(&graph)?,
        graph,
        params,
        batch,
        seq,
        extra_inputs: Vec::new(),
    })
}

#[cfg(feature = "llama32")]
fn llama32_trainable(dir: &Path, batch: usize, seq: usize) -> Result<TrainableLm> {
    let cfg = rlx_llama32::Llama32Config::from_file(&dir.join("config.json"))?;
    let file = dense_weights(dir)?;
    let mut wm =
        rlx_core::weight_map::WeightMap::from_file(file.to_str().context("non-UTF8 path")?)?;
    let (graph, params) =
        rlx_llama32::build_llama32_graph_sized(&cfg, &mut wm, batch, seq, true, false)?;
    Ok(TrainableLm {
        token_input: token_input(&graph)?,
        graph,
        params,
        batch,
        seq,
        extra_inputs: Vec::new(),
    })
}

/// The graph's token-id input: its one integer-typed input, else the input
/// named `input_ids` (Gemma declares token ids as F32).
fn token_input(graph: &rlx_ir::Graph) -> Result<String> {
    let inputs: Vec<(&str, bool)> = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Input { name } => Some((name.as_str(), n.shape.dtype().is_int())),
            _ => None,
        })
        .collect();
    let ints: Vec<&str> = inputs.iter().filter(|i| i.1).map(|i| i.0).collect();
    match ints.as_slice() {
        [one] => Ok(one.to_string()),
        [] if inputs.iter().any(|i| i.0 == "input_ids") => Ok("input_ids".into()),
        other => bail!("expected one integer token input, found {other:?}"),
    }
}

#[cfg(feature = "gemma")]
fn gemma_trainable(dir: &Path, batch: usize, seq: usize) -> Result<TrainableLm> {
    let cfg = rlx_gemma::GemmaConfig::from_file(&dir.join("config.json"))?;
    let file = dense_weights(dir)?;
    let mut wm =
        rlx_core::weight_map::WeightMap::from_file(file.to_str().context("non-UTF8 path")?)?;
    let (graph, params) =
        rlx_gemma::build_gemma_graph_sized(&cfg, &mut wm, batch, seq, true, false)?;
    Ok(TrainableLm {
        token_input: token_input(&graph)?,
        graph,
        params,
        batch,
        seq,
        extra_inputs: Vec::new(),
    })
}

#[cfg(feature = "qwen35")]
fn qwen35_trainable(dir: &Path, batch: usize, seq: usize) -> Result<TrainableLm> {
    use rlx_qwen35::hf::{HfQwen35Fixups, hf_decoder_layer_count};
    let mut cfg = rlx_qwen35::Qwen35Config::from_hf_config_json(&dir.join("config.json"))?;
    let base = rlx_core::safetensors_checkpoint::SafetensorsMmapLoader::open(dir)
        .with_context(|| format!("opening safetensors in {}", dir.display()))?;
    // The HF checkpoint keeps MTP as a separate module, so every decoder
    // layer on disk is a trunk layer (see `hf_decoder_layer_count`).
    if let Some(found) = hf_decoder_layer_count(base.checkpoint().keys()) {
        cfg.num_hidden_layers = found;
        cfg.nextn_predict_layers = 0;
    }
    // `A_log` arrives as `-exp(A_log)`, the form the GGUF-shaped builder uses.
    let mut loader = HfQwen35Fixups::new(rlx_core::HfTranslatingLoader::new(base));
    let weights = rlx_qwen35::Qwen35Weights::from_loader(&mut loader, &cfg)?;
    // Runtime MRoPE: the cos/sin rows are inputs, not a baked 262k-row table.
    let (graph, params, packed) = rlx_qwen35::build_qwen35_graph_sized_ext(
        &cfg, weights, batch, seq, true, false, false, false, None, true, false, false, false,
    )?;
    ensure!(
        packed.is_empty(),
        "qwen35: packed weights cannot be trained"
    );
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&cfg, seq, None, cfg.rope_dim_count / 2);
    Ok(TrainableLm {
        token_input: token_input(&graph)?,
        graph,
        params,
        batch,
        seq,
        extra_inputs: vec![("rope_cos".into(), cos), ("rope_sin".into(), sin)],
    })
}

#[cfg(feature = "minimax")]
fn minimax_m3_trainable(dir: &Path, batch: usize, seq: usize) -> Result<TrainableLm> {
    use rlx_minimax::m3::{
        MiniMaxM3Config, build_m3_text_flow, rope_tables, weights::load_m3_text_snapshot,
    };
    ensure!(
        batch == 1,
        "minimax-m3 trains at --batch 1 (its graph is batch 1)"
    );
    let cfg = MiniMaxM3Config::from_hf_config_json(&dir.join("config.json"))?;
    let file = dense_weights(dir)?;
    let snapshot = load_m3_text_snapshot(&cfg, file.to_str().context("non-UTF8 path")?)?;
    let mut wm = rlx_core::weight_map::WeightMap::from_tensors(snapshot);
    let (graph, params) =
        rlx_core::flow_util::graph_from_built(build_m3_text_flow(&cfg, &mut wm, seq, true)?)?;
    let (cos, sin) = rope_tables(seq, cfg.n_rot(), cfg.rope_theta);
    Ok(TrainableLm {
        graph,
        params,
        token_input: "input_ids".into(),
        batch,
        seq,
        extra_inputs: vec![("rope_cos".into(), cos), ("rope_sin".into(), sin)],
    })
}

/// Build the training forward for the checkpoint in `dir`, picking the family
/// the way `auto_runner` does.
pub fn trainable_lm(dir: &Path, batch: usize, seq: usize) -> Result<TrainableLm> {
    register_builtin_lm_trainers();
    let family = rlx_cli::auto_sniff(dir)?.runner_name;
    let ctor = registry()
        .read()
        .expect("lm trainer registry")
        .get(family)
        .copied();
    let Some(ctor) = ctor else {
        bail!(
            "`{family}` has no training graph yet (trainable: {}) — it needs a \
             full-sequence logits builder registered with `register_lm_trainer`",
            registered_lm_trainers().join(", ")
        );
    };
    ctor(dir, batch, seq)
}

/// What [`qat_lm`] measured.
pub struct QatReport {
    /// Fraction of weight elements trained, and large untrained params.
    pub coverage: (f64, Vec<(String, usize)>),
    /// Held-out next-token loss in the target format, before and after.
    pub before: f32,
    pub after: f32,
    pub losses: Vec<f32>,
}

/// Fine-tune (with `opts.format`, quantization-aware) the checkpoint in `dir`
/// on `train` token sequences, export to `out`, and score held-out loss on
/// `holdout` sequences in the same format before and after.
pub fn qat_lm(
    dir: &Path,
    train: &[Vec<u32>],
    holdout: &[Vec<u32>],
    batch: usize,
    seq: usize,
    opts: &LmTrainOptions,
    out: &Path,
) -> Result<QatReport> {
    let lm = trainable_lm(dir, batch, seq)?;
    let map = rlx_tune::lm::map_checkpoint(&lm, &dense_weights(dir)?)?;
    let eval = |params: &HashMap<String, Vec<f32>>| {
        rlx_tune::lm::eval_lm_loss(
            &lm,
            &map,
            params,
            holdout,
            opts.format,
            8,
            opts.device,
            opts.seed,
        )
    };
    let before = eval(&HashMap::new())?;
    let trained = rlx_tune::lm::train_lm(&lm, &map, train, opts)?;
    rlx_tune::lm::export_lm(&trained.params, &map, out)?;
    let after = eval(&trained.params)?;
    Ok(QatReport {
        coverage: map.coverage(),
        before,
        after,
        losses: trained.losses,
    })
}

/// Parse `--number-format`-style text, for CLIs.
pub fn parse_format(spec: &str) -> Result<NumberFormat> {
    spec.parse().map_err(|e: String| anyhow::anyhow!(e))
}

/// `rlx-run qat`: fine-tune / QAT an LM checkpoint on a text file.
///
/// ```text
/// rlx-run qat --weights DIR --text corpus.txt --out DIR [--number-format f8]
///   [--tokenizer DIR/tokenizer.json] [--steps 100] [--lr 1e-5] [--batch 2]
///   [--seq 128] [--holdout 0.1] [--device cpu] [--seed 0]
/// ```
pub fn cli(args: &[String]) -> Result<()> {
    let (mut weights, mut text, mut out, mut tokenizer) = (
        None::<PathBuf>,
        None::<PathBuf>,
        None::<PathBuf>,
        None::<PathBuf>,
    );
    let (mut batch, mut seq, mut holdout) = (2usize, 128usize, 0.1f32);
    let mut device = "cpu".to_string();
    let mut opts = LmTrainOptions::default();
    let mut i = 0;
    while i < args.len() {
        let v = |i: &mut usize| rlx_cli::req(args, i);
        match args[i].as_str() {
            "--weights" => weights = Some(v(&mut i)?.into()),
            "--text" => text = Some(v(&mut i)?.into()),
            "--out" => out = Some(v(&mut i)?.into()),
            "--tokenizer" => tokenizer = Some(v(&mut i)?.into()),
            "--number-format" => opts.format = Some(parse_format(&v(&mut i)?)?),
            "--steps" => opts.steps = v(&mut i)?.parse().context("--steps")?,
            "--lr" => opts.lr = v(&mut i)?.parse().context("--lr")?,
            "--batch" => batch = v(&mut i)?.parse().context("--batch")?,
            "--seq" => seq = v(&mut i)?.parse().context("--seq")?,
            "--holdout" => holdout = v(&mut i)?.parse().context("--holdout")?,
            "--device" => device = v(&mut i)?,
            "--seed" => opts.seed = v(&mut i)?.parse().context("--seed")?,
            "-h" | "--help" => {
                println!(
                    "rlx-run qat --weights DIR --text corpus.txt --out DIR [--number-format f8] \
                     [--tokenizer FILE] [--steps N] [--lr F] [--batch N] [--seq N] [--holdout F] \
                     [--device NAME] [--seed N]\n\nTrainable families: {}",
                    registered_lm_trainers().join(", ")
                );
                return Ok(());
            }
            other => bail!("qat: unknown argument {other} (see --help)"),
        }
    }
    // `rlx-run` takes `--number-format` before dispatching and publishes it
    // process-wide; recover it here — the trainer itself ignores that setting.
    if opts.format.is_none()
        && let Some(spec) = rlx_runtime::env::var("RLX_NUMBER_FORMAT")
    {
        opts.format = Some(parse_format(&spec)?);
    }
    let weights = weights.context("qat: --weights DIR required")?;
    let text = text.context("qat: --text FILE required")?;
    let out = out.context("qat: --out DIR required")?;
    opts.device = rlx_cli::parse_standard_device("rlx-run qat", &device)?;
    let tok_path = tokenizer.unwrap_or_else(|| weights.join("tokenizer.json"));
    let tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", tok_path.display()))?;
    let corpus =
        std::fs::read_to_string(&text).with_context(|| format!("read {}", text.display()))?;
    let ids: Vec<u32> = tok
        .encode(corpus.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    let split = ((ids.len() as f32) * (1.0 - holdout)) as usize;
    ensure!(
        split > seq + 1 && ids.len() - split > seq + 1,
        "corpus of {} tokens is too short for --seq {seq} with --holdout {holdout}",
        ids.len()
    );
    let (train, held) = (vec![ids[..split].to_vec()], vec![ids[split..].to_vec()]);
    let fmt = opts.format.map_or("f32".to_string(), |f| f.to_string());
    println!(
        "{} tokens ({} train / {} held out); training in {fmt}",
        ids.len(),
        split,
        ids.len() - split
    );
    let r = qat_lm(&weights, &train, &held, batch, seq, &opts, &out)?;
    println!("trained {:.1}% of weight elements", 100.0 * r.coverage.0);
    for (name, n) in &r.coverage.1 {
        println!("  not trained (no checkpoint tensor): {name} ({n} values)");
    }
    let avg = |l: &[f32]| l.iter().sum::<f32>() / l.len().max(1) as f32;
    let n = r.losses.len();
    println!(
        "train loss: first {} steps {:.4} -> last {} steps {:.4}",
        n.min(10),
        avg(&r.losses[..n.min(10)]),
        n.min(10),
        avg(&r.losses[n.saturating_sub(10)..])
    );
    println!(
        "held-out loss in {fmt}: {:.4} -> {:.4} (perplexity {:.2} -> {:.2}); exported to {}",
        r.before,
        r.after,
        r.before.exp(),
        r.after.exp(),
        out.display()
    );
    Ok(())
}
