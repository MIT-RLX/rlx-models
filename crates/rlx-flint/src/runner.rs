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

//! Tokens in, logits out, with the routed experts paged off disk.
//!
//! Flint is 39.8 B parameters of which 37 B are routed experts — 47 layers ×
//! 256 × 3 bf16 matrices, 74 GB — and a token fires 6 per layer. Holding them
//! is the whole problem, so this runner never does: every per-layer graph is
//! split at the one point where the next step depends on data.
//!
//! ```text
//!   host embed ─► for each layer:
//!                  layer graph  : h += attn(norm(h));           dense layer: h += mlp(norm(h))
//!                                 x = norm(h); ids, w = router(x); partial = h + shared(x)
//!                  host         : h = partial + Σ w · expert_ids(x), each fired expert read
//!                                 from the checkpoint and computed in its stored dtype
//!            ─► head graph: lm_head(norm(h_last))
//! ```
//!
//! * **The router's scores come out of the layer graph; the pick is made on
//!   the host** ([`crate::moe::select_experts`]), with the CPU body of the
//!   gate kernel the resident flow puts in its graph. The host needs the ids
//!   anyway to page the experts in, and no backend has to implement the
//!   custom op, so the runner lowers wherever the dense ops do.
//! * **The routed experts run on the host** ([`crate::experts`]), straight
//!   from the bytes read: no decode to f32, no transpose, no copy into a graph.
//!   A token's experts are 1.8 GB of bf16 that are used once — an LRU of 48
//!   experts per layer (13 GB) would still miss more than half the time — so
//!   the work is the read, and a graph would only add copies around it.
//! * **Each fired expert is read once per step**, however many rows of a
//!   prefill chunk chose it.
//! * **Experts are read with `pread`** ([`crate::checkpoint`]), so the OS
//!   page cache is the expert cache. Nothing here sizes it: on a machine with
//!   RAM to spare the whole checkpoint ends up cached; on one without, the
//!   kernel evicts what was used least recently.
//!
//! The key/value cache lives on the host and is an input to each layer graph:
//! full layers hold `max_context` positions in order, sliding layers a ring of
//! `sliding_window - 1` (see [`crate::mask`]). That keeps the graphs portable —
//! no backend has to honour an in-place cache write — at the price of feeding
//! the cache in every step.

use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlx_core::flow_util::compile_built;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, WeightStorage};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, LmRunner};

use crate::attention::AttnDims;
use crate::checkpoint::{Checkpoint, CheckpointRef};
use crate::common::{linear, rms_norm};
use crate::config::FlintConfig;
use crate::experts::{ExpertWeights, add_routed};
use crate::flow::{
    EMBED_KEY, LM_HEAD_KEY, NORM_KEY, StepTensors, emit_attn_sublayer, emit_ffn_pre_route,
    emit_ffn_resident,
};
use crate::mask::{SlotLedger, rope_tables, step_bias};
use crate::moe::{load_expert_bias, select_experts};
use crate::prefetch::ExpertPrefetcher;
use crate::quant::QuantStore;

/// How to build a [`FlintRunner`].
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    pub device: Device,
    /// Positions the full-attention layers can hold — prompt plus output.
    pub max_context: usize,
    /// Prompt positions per prefill step. `1` reuses the decode graphs; above 1
    /// compiles a second set, which on backends without parameter sharing
    /// (CPU) holds the attention weights twice.
    pub prefill_chunk: usize,
    /// Run only the first `n` layers — for a truncated checkpoint subset.
    pub layers: Option<usize>,
    /// Guess each layer's experts one layer early and ask the OS to read them
    /// ahead ([`crate::prefetch`]). Changes timing only, never the output.
    /// Off by default: the guesses are good, but a speedup has not been shown.
    pub prefetch_experts: bool,
    /// Read the routed experts from a block-quantized store written by
    /// [`crate::quant::write_store`] instead of the checkpoint.
    pub expert_store: Option<std::path::PathBuf>,
}

impl RunnerOptions {
    pub fn new(device: Device) -> Self {
        RunnerOptions {
            device,
            max_context: 4096,
            prefill_chunk: 16,
            layers: None,
            prefetch_experts: false,
            expert_store: None,
        }
    }

    pub fn max_context(mut self, n: usize) -> Self {
        self.max_context = n;
        self
    }

    pub fn prefill_chunk(mut self, n: usize) -> Self {
        self.prefill_chunk = n.max(1);
        self
    }

    pub fn layers(mut self, n: usize) -> Self {
        self.layers = Some(n);
        self
    }

    pub fn expert_store(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.expert_store = Some(dir.into());
        self
    }

    pub fn prefetch_experts(mut self, on: bool) -> Self {
        self.prefetch_experts = on;
        self
    }
}

/// Where a run spent its time.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// Layer graphs: attention, router, shared expert, dense MLPs.
    pub layers: Duration,
    /// Selecting, reading and computing the routed experts.
    pub experts: Duration,
    /// Final norm + LM head.
    pub head: Duration,
    /// Expert matrices read (3 per fired expert per layer).
    pub expert_reads: u64,
    /// Experts read ahead of their layer, and how many it then fired.
    pub prefetched: u64,
    pub prefetch_used: u64,
    pub steps: u64,
}

struct LayerGraph {
    graph: CompiledGraph,
    cap: usize,
    sliding: bool,
    rope: bool,
    dense: bool,
    /// The selection bias of a sparse layer, if the checkpoint has one.
    expert_bias: Option<Vec<f32>>,
}

/// Every graph one step size needs.
struct StepSet {
    chunk: usize,
    layers: Vec<LayerGraph>,
}

/// A compiled Flint, ready to be driven token by token.
pub struct FlintRunner {
    cfg: FlintConfig,
    ckpt: Arc<Checkpoint>,
    device: Device,
    decode: StepSet,
    prefill: Option<StepSet>,
    head: CompiledGraph,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    full: SlotLedger,
    sliding: SlotLedger,
    pos: usize,
    prefetch: Option<ExpertPrefetcher>,
    quant: Option<QuantStore>,
    timings: Timings,
}

impl FlintRunner {
    /// Open a checkpoint directory (HF layout: `config.json` + safetensors).
    pub fn open(dir: impl AsRef<Path>, opts: RunnerOptions) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = FlintConfig::from_path(dir)?;
        let ckpt = Checkpoint::open(dir)?;
        Self::from_checkpoint(cfg, Arc::new(ckpt), opts)
    }

    /// Build over an already-open checkpoint with an explicit config.
    pub fn from_checkpoint(
        cfg: FlintConfig,
        ckpt: Arc<Checkpoint>,
        opts: RunnerOptions,
    ) -> Result<Self> {
        let cfg = match opts.layers {
            Some(n) => cfg.truncated(n)?,
            None => cfg,
        };
        if opts.max_context == 0 {
            bail!("flint: max_context must be >= 1");
        }
        let full_cap = opts.max_context;
        let swa_cap = cfg
            .sliding_history()
            .unwrap_or(full_cap)
            .min(full_cap)
            .max(1);

        let decode = build_step_set(&cfg, &ckpt, opts.device, 1, full_cap, swa_cap, None)?;
        let prefill = if opts.prefill_chunk > 1 {
            Some(build_step_set(
                &cfg,
                &ckpt,
                opts.device,
                opts.prefill_chunk,
                full_cap,
                swa_cap,
                Some(&decode),
            )?)
        } else {
            None
        };
        let head = compile_built(build_head_flow(&cfg, &ckpt)?, opts.device)?;

        let kv = cfg.kv_dim();
        let (k_cache, v_cache) = (0..cfg.num_hidden_layers)
            .map(|l| {
                let cap = decode.layers[l].cap;
                (vec![0.0f32; cap * kv], vec![0.0f32; cap * kv])
            })
            .unzip();
        let quant = opts
            .expert_store
            .as_ref()
            .map(QuantStore::open)
            .transpose()?;
        let prefetch = if opts.prefetch_experts {
            Some(ExpertPrefetcher::new(&ckpt, &cfg)?)
        } else {
            None
        };
        Ok(FlintRunner {
            cfg,
            ckpt,
            device: opts.device,
            decode,
            prefill,
            head,
            k_cache,
            v_cache,
            full: SlotLedger::linear(full_cap),
            sliding: SlotLedger::ring(swa_cap),
            pos: 0,
            prefetch,
            quant,
            timings: Timings::default(),
        })
    }

    pub fn config(&self) -> &FlintConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Positions already in the cache.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Positions the full-attention cache can hold.
    pub fn capacity(&self) -> usize {
        self.full.cap()
    }

    pub fn timings(&self) -> Timings {
        let mut t = self.timings;
        if let Some(p) = &self.prefetch {
            t.prefetched = p.issued;
            t.prefetch_used = p.used;
        }
        t
    }

    /// Forget the sequence. The graphs stay compiled.
    pub fn reset(&mut self) {
        self.full.clear();
        self.sliding.clear();
        self.pos = 0;
        for c in self.k_cache.iter_mut().chain(self.v_cache.iter_mut()) {
            c.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Append `ids` to the sequence and return the logits after the last one.
    pub fn extend(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            bail!("flint: nothing to extend with");
        }
        let chunk = self.prefill.as_ref().map_or(1, |p| p.chunk);
        let mut last = Vec::new();
        for piece in ids.chunks(chunk) {
            let hidden = self.step(piece, piece.len() > 1)?;
            last = self.head_logits(&hidden[(piece.len() - 1) * self.cfg.hidden_size..])?;
        }
        Ok(last)
    }

    /// Logits at **every** position of `ids`, from a fresh sequence. For
    /// checking against a reference; generation only needs the last row.
    pub fn all_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.reset();
        let chunk = self.prefill.as_ref().map_or(1, |p| p.chunk);
        let hsz = self.cfg.hidden_size;
        let mut out = Vec::with_capacity(ids.len() * self.cfg.vocab_size);
        for piece in ids.chunks(chunk) {
            let hidden = self.step(piece, piece.len() > 1)?;
            for r in 0..piece.len() {
                out.extend(self.head_logits(&hidden[r * hsz..(r + 1) * hsz])?);
            }
        }
        Ok(out)
    }

    /// Greedy generation after `prompt`, stopping at an EOS id, at `n_new`, at
    /// capacity, or when `on_token` returns `false`.
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.reset();
        let mut logits = self.extend(prompt)?;
        let mut out = Vec::new();
        for _ in 0..n_new {
            let next = rlx_runtime::search::argmax_u32(&logits);
            out.push(next);
            if self.cfg.is_eos(next) || !on_token(next) || self.pos >= self.capacity() {
                break;
            }
            logits = self.extend(&[next])?;
        }
        Ok(out)
    }

    fn head_logits(&mut self, h_row: &[f32]) -> Result<Vec<f32>> {
        let t = Instant::now();
        let h = &h_row[..self.cfg.hidden_size];
        let out = self
            .head
            .run(&[("h", h)])
            .pop()
            .ok_or_else(|| anyhow!("flint: the head graph produced no output"))?;
        self.timings.head += t.elapsed();
        Ok(out)
    }

    /// Run `tokens` through every layer at positions `pos..`, commit their keys
    /// and values, and return the final hidden state `[chunk, hidden]` (rows
    /// past `tokens.len()` are padding).
    fn step(&mut self, tokens: &[u32], use_prefill: bool) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let set = match (&mut self.prefill, use_prefill) {
            (Some(p), true) => p,
            _ => &mut self.decode,
        };
        let (c, n) = (set.chunk, tokens.len());
        if n == 0 || n > c {
            bail!("flint: {n} tokens do not fit a step of {c}");
        }
        if self.pos + n > self.full.cap() {
            bail!(
                "flint: {} + {n} positions exceed max_context {}",
                self.pos,
                self.full.cap()
            );
        }
        let (hsz, kv) = (cfg.hidden_size, cfg.kv_dim());

        // Padding rows are zero vectors: RMSNorm keeps them at zero, the
        // causal mask keeps them out of every real row, and they are never
        // committed to the cache.
        let mut h = self.ckpt.rows_f32(EMBED_KEY, tokens)?;
        h.resize(c * hsz, 0.0);
        let positions: Vec<usize> = (self.pos..self.pos + c).collect();
        let (cos, sin) = rope_tables(&positions, cfg.head_dim, cfg.rope_theta);
        let bias_full = step_bias(self.full.positions(), &positions, None);
        let bias_swa = step_bias(self.sliding.positions(), &positions, cfg.sliding_history());
        let full_slots: Vec<usize> = positions[..n]
            .iter()
            .map(|&p| self.full.slot_for(p).expect("capacity checked above"))
            .collect();
        let swa_slots: Vec<usize> = positions[..n]
            .iter()
            .map(|&p| self.sliding.slot_for(p).expect("a ring always has a slot"))
            .collect();

        for (l, lg) in set.layers.iter_mut().enumerate() {
            let t = Instant::now();
            let bias = if lg.sliding { &bias_swa } else { &bias_full };
            let mut inputs: Vec<(&str, &[f32])> = vec![
                ("h", &h),
                ("bias", bias),
                ("k_cache", &self.k_cache[l]),
                ("v_cache", &self.v_cache[l]),
            ];
            if lg.rope {
                inputs.push(("rope_cos", &cos));
                inputs.push(("rope_sin", &sin));
            }
            let out = lg
                .graph
                .run(&inputs)
                .pop()
                .ok_or_else(|| anyhow!("flint: layer {l} graph produced no output"))?;
            self.timings.layers += t.elapsed();

            // Packed: [h | (x | scores)? | k | v].
            let mut at = 0usize;
            let mut take = |len: usize| {
                let s = &out[at..at + len];
                at += len;
                s
            };
            let h_next = take(c * hsz);
            let moe = (!lg.dense).then(|| (take(c * hsz), take(c * cfg.num_experts)));
            let (k_rows, v_rows) = (take(c * kv), take(c * kv));
            let slots = if lg.sliding { &swa_slots } else { &full_slots };
            for (i, &s) in slots.iter().enumerate() {
                store_kv(
                    &mut self.k_cache[l],
                    &mut self.v_cache[l],
                    &k_rows[i * kv..(i + 1) * kv],
                    &v_rows[i * kv..(i + 1) * kv],
                    s,
                    lg.cap,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                );
            }

            h = match moe {
                None => h_next.to_vec(),
                Some((x, scores)) => {
                    let t = Instant::now();
                    // Real rows only; padding rows keep expert 0 at weight 0.
                    let (ids, w) = select_experts(
                        cfg,
                        &scores[..n * cfg.num_experts],
                        lg.expert_bias.as_deref(),
                    )?;
                    let src = match &self.quant {
                        Some(q) => ExpertWeights::Quantized(q),
                        None => ExpertWeights::Checkpoint(&self.ckpt),
                    };
                    if let Some(p) = self.prefetch.as_mut() {
                        p.observe(l, &ids);
                        p.after_route(src, cfg, l, x, n);
                    }
                    let mut y = h_next.to_vec();
                    let fired = add_routed(src, cfg, l, x, &ids, &w, n, &mut y)?;
                    self.timings.expert_reads += 3 * fired as u64;
                    self.timings.experts += t.elapsed();
                    y
                }
            };
        }

        for &p in &positions[..n] {
            self.full.commit(p);
            self.sliding.commit(p);
        }
        self.pos += n;
        self.timings.steps += 1;
        Ok(h)
    }
}

/// Write one position's key and value rows (`[kv_heads · head_dim]` each)
/// into cache slot `slot`, in the layouts the layer graph reads: keys as
/// `[kv_heads, head_dim, cap]`, values as `[kv_heads, cap, head_dim]`.
#[allow(clippy::too_many_arguments)]
fn store_kv(
    kt: &mut [f32],
    v: &mut [f32],
    k_row: &[f32],
    v_row: &[f32],
    slot: usize,
    cap: usize,
    nkv: usize,
    hd: usize,
) {
    for h in 0..nkv {
        for d in 0..hd {
            kt[(h * hd + d) * cap + slot] = k_row[h * hd + d];
        }
        v[(h * cap + slot) * hd..(h * cap + slot + 1) * hd]
            .copy_from_slice(&v_row[h * hd..(h + 1) * hd]);
    }
}

/// Input names shared by the layer graphs.
fn layer_flow(cfg: &FlintConfig, layer: usize, chunk: usize, cap: usize) -> ModelFlow {
    let f = DType::F32;
    let spec = cfg.layer(layer);
    let nkv = cfg.num_key_value_heads;
    let mut flow = ModelFlow::new(format!("flint.layer{layer}"))
        .with_profile(CompileProfile::llama32_prefill())
        .with_weight_storage(WeightStorage::Auto)
        .input("h", Shape::new(&[1, chunk, cfg.hidden_size], f))
        .input("bias", Shape::new(&[1, 1, chunk, cap + chunk], f))
        .input("k_cache", Shape::new(&[nkv, cfg.head_dim, cap], f))
        .input("v_cache", Shape::new(&[nkv, cap, cfg.head_dim], f));
    if spec.rope {
        let t = Shape::new(&[chunk, cfg.head_dim / 2], f);
        flow = flow.input("rope_cos", t.clone()).input("rope_sin", t);
    }
    flow
}

/// One layer's graph for steps of `chunk` positions over a cache of `cap`.
pub fn build_layer_flow(
    cfg: &FlintConfig,
    ckpt: &Checkpoint,
    layer: usize,
    chunk: usize,
    cap: usize,
) -> Result<BuiltModel> {
    let spec = cfg.layer(layer);
    let total = if spec.dense {
        chunk * (cfg.hidden_size + 2 * cfg.kv_dim())
    } else {
        chunk * (2 * cfg.hidden_size + cfg.num_experts + 2 * cfg.kv_dim())
    };
    let cfg2 = cfg.clone();
    layer_flow(cfg, layer, chunk, cap)
        .plugin_named("layer", move |emit, _| {
            let cfg = &cfg2;
            let h = emit.flow_input("h")?.hir_id();
            let rope = if spec.rope {
                Some((
                    emit.flow_input("rope_cos")?.hir_id(),
                    emit.flow_input("rope_sin")?.hir_id(),
                ))
            } else {
                None
            };
            let bias = emit.flow_input("bias")?.hir_id();
            let cache = (
                emit.flow_input("k_cache")?.hir_id(),
                emit.flow_input("v_cache")?.hir_id(),
            );
            let step = StepTensors {
                rope,
                bias_full: bias,
                bias_sliding: bias,
            };
            let dims = AttnDims::from_config(cfg, chunk, cap);
            let (h, k, v) = emit_attn_sublayer(emit, cfg, layer, h, dims, step, Some(cache))?;
            let mut parts: Vec<HirNodeId> = Vec::with_capacity(6);
            if spec.dense {
                parts.push(emit_ffn_resident(emit, cfg, layer, h)?);
            } else {
                let pre = emit_ffn_pre_route(emit, cfg, layer, h)?;
                parts.extend([pre.partial, pre.x2d, pre.scores]);
            }
            parts.extend([k, v]);
            let mut gb = HirMut::new(emit.hir());
            let flat: Vec<HirNodeId> = parts
                .into_iter()
                .map(|p| {
                    let n: usize = gb
                        .shape(p)
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    gb.reshape_(p, vec![1, n as i64])
                })
                .collect();
            let out = gb.concat_(flat, 1);
            Ok(Some(emit.wrap(out, Shape::new(&[1, total], DType::F32))))
        })
        .output("packed")
        .build_with(&mut CheckpointRef(ckpt), None)
}

/// Final norm and LM head over one row: `h [1, 1, hidden]` → `[1, vocab]`.
pub fn build_head_flow(cfg: &FlintConfig, ckpt: &Checkpoint) -> Result<BuiltModel> {
    let f = DType::F32;
    let (hsz, vocab, eps) = (cfg.hidden_size, cfg.vocab_size, cfg.rms_norm_eps);
    let head_key = if cfg.tie_word_embeddings {
        EMBED_KEY
    } else {
        LM_HEAD_KEY
    };
    ModelFlow::new("flint.head")
        .with_profile(CompileProfile::llama32_prefill())
        .with_weight_storage(WeightStorage::Auto)
        .input("h", Shape::new(&[1, 1, hsz], f))
        .plugin_named("head", move |emit, _| {
            let h = emit.flow_input("h")?.hir_id();
            let x = rms_norm(emit, NORM_KEY, h, hsz, eps)?;
            let x = {
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(x, vec![1, hsz as i64])
            };
            let y = linear(emit, head_key, x)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, vocab], f))))
        })
        .output("logits")
        .build_with(&mut CheckpointRef(ckpt), None)
}

fn build_step_set(
    cfg: &FlintConfig,
    ckpt: &Checkpoint,
    device: Device,
    chunk: usize,
    full_cap: usize,
    swa_cap: usize,
    share_with: Option<&StepSet>,
) -> Result<StepSet> {
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let spec = cfg.layer(l);
        let cap = if spec.sliding { swa_cap } else { full_cap };
        let built = build_layer_flow(cfg, ckpt, l, chunk, cap)
            .with_context(|| format!("flint: building layer {l}"))?;
        let mut graph = compile_built(built, device)?;
        if let Some(other) = share_with {
            // Same weights, different step size: on backends that support it
            // the second graph adopts the first's parameter buffer.
            let _ = graph.share_params_from(&other.layers[l].graph);
        }
        layers.push(LayerGraph {
            graph,
            cap,
            sliding: spec.sliding,
            rope: spec.rope,
            dense: spec.dense,
            expert_bias: if spec.dense {
                None
            } else {
                load_expert_bias(ckpt, l)?
            },
        });
    }
    Ok(StepSet { chunk, layers })
}

impl LmRunner for FlintRunner {
    fn family(&self) -> &'static str {
        "flint"
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.reset();
        self.extend(prompt_ids)
    }

    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.reset();
        self.extend(prompt_ids)
    }

    fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.extend(&[token])
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_greedy(prompt_ids, n_new, on_token)
    }
}
