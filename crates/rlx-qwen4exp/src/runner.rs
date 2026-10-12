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

//! Tokens in, logits out — the [`rlx_runtime::LmRunner`] side of this crate.
//!
//! Everything else here hands back a [`BuiltModel`](rlx_flow::BuiltModel) and
//! leaves the driving to the caller: compile it, derive the M-RoPE tables,
//! gather the n-gram rows against the right history, thread 36 layers of
//! recurrent state, pad the chunk, re-read the exported state. That is the
//! right seam for a port under test and the wrong one for running the model,
//! so this module closes it.
//!
//! ```no_run
//! use rlx_qwen4exp::runner::{Qwen4ExpRunner, RunnerOptions};
//! use rlx_runtime::Device;
//!
//! let mut run = Qwen4ExpRunner::open("model.gguf", RunnerOptions::new(Device::Metal))?;
//! let out = run.generate_greedy(&[9707, 11], 32, |t| {
//!     print!("{t} ");
//!     true
//! })?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## What makes it fast
//!
//! * **One compile, many steps.** The step graph is compiled once in
//!   [`Qwen4ExpRunner::open`] and reused for every token. Compiling per step is
//!   the single easiest way to lose an order of magnitude here, and it is not a
//!   hypothetical — a sibling port did exactly that and paid 50×.
//! * **Prefill in chunks, decode one at a time.** With
//!   [`RunnerOptions::prefill_chunk`] above 1 the prompt goes through a second
//!   graph that processes `chunk` positions per `run()`, right-padding the last
//!   one — the arrangement `tests/decode_equivalence.rs` pins against a
//!   single-shot prefill. The trade is explicit: a second compiled graph means
//!   the weights are resident **twice**, so the default is 1 (share the decode
//!   graph, no extra memory).
//! * **The scan state can stay on the device.** 96% of this model's per-token
//!   host traffic is the GatedDeltaNet scan — `48 · 128²` f32 per linear layer,
//!   113 MB across 36 layers. [`ScanState::InPlace`] keeps it in a persistent
//!   param instead, but it only survives to the next `run()` on CPU, Metal and
//!   wgpu, and the param cannot be re-zeroed without rebuilding the graph — so
//!   one compiled graph then serves exactly one sequence. [`scan_state_for`]
//!   encodes which devices qualify; [`RunnerOptions::single_sequence`] opts in.
//!
//! ## What it does not do
//!
//! * **No tokenizer.** [`LmRunner`] is defined over token ids and so is this;
//!   the GGUF's `qwen35`-pre BPE vocabulary and chat template are not wired.
//! * **The n-gram table is the caller's.** One layer of this model reads a
//!   51-billion-parameter hashed embedding table (~28 GB packed). It is never a
//!   graph parameter — rows are gathered on the host through
//!   [`PleRowSource`] — so a runner for the real checkpoint needs a row source
//!   that reads them lazily. [`Qwen4ExpRunner::with_ple_rows`] takes one, and
//!   [`Qwen4ExpRunner::open`] refuses rather than pretending the rows are zero.
//! * **Context is capped at the QSA budget** (2048 positions), for the reason
//!   [`crate::decode`] gives: past it the indexer genuinely selects.

use anyhow::{Context, Result, bail};
use std::path::Path;

use rlx_core::flow_util::compile_built;
use rlx_flow::WeightSource;
use rlx_runtime::{CompiledGraph, Device, LmRunner};

use crate::config::{PleHash, Qwen4ExpConfig};
use crate::decode::{
    DecodeNames, DecodeSession, ScanState, build_qwen4exp_step_flow, build_qwen4exp_step_flow_with,
};
use crate::flow::{OutputKind, ROPE_COS_INPUT, ROPE_SIN_INPUT};
use crate::ple::{GgufPleRows, PLE_TABLE_TENSOR, PleRowSource, ngram_embeddings, ple_embed_input};
use crate::rope::{mrope_tables, text_section_pos};

/// Where the scan state can live on `device`.
///
/// [`ScanState::InPlace`] needs the in-place `Op::GatedDeltaNet` carry write to
/// still be there on the next `run()`. CPU, Metal and wgpu keep it; MLX and
/// CoreML substitute the new state into a per-evaluation environment, where it
/// silently freezes at zero — which is a wrong answer, not a slow one, so
/// everything else gets [`ScanState::Portable`].
pub fn scan_state_for(device: Device) -> ScanState {
    match device {
        Device::Cpu | Device::Metal | Device::Gpu => ScanState::InPlace,
        _ => ScanState::Portable,
    }
}

/// How to build the runner.
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    pub device: Device,
    /// Decode capacity in positions. `0` takes the config's QSA budget, which
    /// is the most this model can decode without a position-ordered indexer
    /// key cache.
    pub cap: usize,
    /// Positions per prefill `run()`. `1` shares the decode graph; above 1
    /// compiles a second graph and holds the weights twice.
    pub prefill_chunk: usize,
    /// Read quantized weights packed (fused `DequantMatMul`, no f32 weight
    /// materialized). Ignored for an already-f32 checkpoint.
    pub packed: bool,
    /// `None` keeps the scan state portable, so the runner can be [`reset`] and
    /// re-driven. `Some` is taken verbatim — see [`scan_state_for`].
    ///
    /// [`reset`]: Qwen4ExpRunner::reset
    pub scan: Option<ScanState>,
    /// Where the n-gram table lives, when it is not in the file being opened.
    ///
    /// [`Qwen4ExpRunner::open`] attaches a [`GgufPleRows`] over the model file
    /// itself when it carries `per_layer_token_embd.weight`. The published
    /// checkpoint is split across shards, so point this at whichever one holds
    /// the table. An explicit path that cannot be opened is an error, not a
    /// silent fallback.
    pub ple_table: Option<std::path::PathBuf>,
}

impl RunnerOptions {
    /// Defaults: the config's QSA budget, unchunked prefill, packed weights,
    /// portable scan state.
    pub fn new(device: Device) -> Self {
        RunnerOptions {
            device,
            cap: 0,
            prefill_chunk: 1,
            packed: true,
            scan: None,
            ple_table: None,
        }
    }

    /// Read the n-gram table from `path` rather than from the model file.
    pub fn ple_table(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.ple_table = Some(path.into());
        self
    }

    /// Chunk the prompt `chunk` positions at a time. Costs a second compiled
    /// graph, and with it a second resident copy of the weights.
    ///
    /// ```no_run
    /// use rlx_qwen4exp::runner::{Qwen4ExpRunner, RunnerOptions};
    /// use rlx_runtime::Device;
    ///
    /// let opts = RunnerOptions::new(Device::Metal).prefill_chunk(64);
    /// let mut run = Qwen4ExpRunner::open("model.gguf", opts)?;
    /// let logits = run.prefill_logits(&[9707, 11, 1879])?;
    /// let next = run.decode_logits(0)?;
    /// # let _ = (logits, next);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn prefill_chunk(mut self, chunk: usize) -> Self {
        self.prefill_chunk = chunk.max(1);
        self
    }

    /// Keep the scan state on the device where that is sound, trading the
    /// ability to [`Qwen4ExpRunner::reset`] for ~96% of the per-token host
    /// traffic. One such runner serves one sequence.
    pub fn single_sequence(mut self) -> Self {
        self.scan = Some(scan_state_for(self.device));
        self
    }
}

struct Graph {
    compiled: CompiledGraph,
    chunk: usize,
}

/// A compiled Qwen3.8-Flash-Next ready to be driven token by token.
pub struct Qwen4ExpRunner {
    cfg: Qwen4ExpConfig,
    device: Device,
    cap: usize,
    names: DecodeNames,
    session: DecodeSession,
    decode: Graph,
    prefill: Option<Graph>,
    ple: Option<Box<dyn PleRowSource + Send>>,
    /// `ple_layer_index` for each PLE module, in layer order.
    ple_modules: Vec<usize>,
    consumed: usize,
    scan: ScanState,
    /// Kept so [`Qwen4ExpRunner::reset`] can rebuild the session without
    /// rebuilding the graph, and so the session can be re-pointed between the
    /// chunked prefill graph and the one-token decode graph.
    layout: crate::decode::DecodeLayout,
    prefill_layout: Option<crate::decode::DecodeLayout>,
}

impl Qwen4ExpRunner {
    /// Open a GGUF (single file or the first shard of a split) and compile it.
    pub fn open(path: impl AsRef<Path>, opts: RunnerOptions) -> Result<Self> {
        let path = path.as_ref();
        let cfg = Qwen4ExpConfig::from_gguf_path(path)
            .with_context(|| format!("qwen4exp: reading metadata from {path:?}"))?;
        Self::open_with_config(path, cfg, opts)
    }

    /// [`Self::open`] with the config supplied — for a checkpoint whose
    /// metadata needs a correction the loader cannot make for itself.
    pub fn open_with_config(
        path: impl AsRef<Path>,
        cfg: Qwen4ExpConfig,
        opts: RunnerOptions,
    ) -> Result<Self> {
        let path = path.as_ref();
        let cap = Self::resolve_cap(&cfg, &opts)?;
        let scan = opts.scan.unwrap_or(ScanState::Portable);

        let mut me = {
            let mut loader = rlx_core::weight_loader::load_from_path(path_str(path)?)
                .with_context(|| format!("qwen4exp: opening {path:?}"))?;
            let built = Self::build(&cfg, loader.as_mut(), opts.packed, cap, 1, scan)?;
            Self::assemble(cfg, &opts, cap, built, 1)?
        };

        if opts.prefill_chunk > 1 {
            // A fresh loader: building a graph consumes the weights out of its
            // source, and these are two independent graphs.
            let mut loader = rlx_core::weight_loader::load_from_path(path_str(path)?)
                .with_context(|| format!("qwen4exp: re-opening {path:?} for prefill"))?;
            let chunk = opts.prefill_chunk;
            let (model, layout) =
                Self::build(&me.cfg, loader.as_mut(), opts.packed, cap, chunk, scan)?;
            me.prefill = Some(Graph {
                compiled: compile_built(model, opts.device)?,
                chunk,
            });
            me.prefill_layout = Some(layout);
        }

        // The table is 28 GB and nine of its rows are wanted per step, so it is
        // never a graph parameter — attach a reader over it instead.
        if !me.ple_modules.is_empty() {
            let hd = me.cfg.ple_head_dim;
            match opts.ple_table.as_deref() {
                Some(explicit) => {
                    me.ple = Some(Box::new(GgufPleRows::open(explicit, hd).with_context(
                        || format!("qwen4exp: opening the n-gram table at {explicit:?}"),
                    )?));
                }
                // Not in this shard is the normal case for a split checkpoint,
                // so stay quiet here and let the first step name the fix.
                None => {
                    if let Ok(rows) = GgufPleRows::open(path, hd) {
                        me.ple = Some(Box::new(rows));
                    }
                }
            }
        }
        Ok(me)
    }

    /// Build from a caller-held [`WeightSource`] — one graph, so
    /// [`RunnerOptions::prefill_chunk`] must be 1. For synthetic weights and
    /// for callers that already hold a loader.
    pub fn from_source(
        cfg: Qwen4ExpConfig,
        weights: &mut dyn WeightSource,
        opts: RunnerOptions,
    ) -> Result<Self> {
        if opts.prefill_chunk > 1 {
            bail!(
                "qwen4exp: prefill_chunk = {} needs a second graph, and building \
                 one consumes a second copy of the weights; use `open`, which \
                 can re-read the file",
                opts.prefill_chunk
            );
        }
        let cap = Self::resolve_cap(&cfg, &opts)?;
        let scan = opts.scan.unwrap_or(ScanState::Portable);
        let built = build_qwen4exp_step_flow(&cfg, weights, cap, 1, OutputKind::Logits, scan)?;
        Self::assemble(cfg, &opts, cap, built, 1)
    }

    /// Like [`Self::from_source`], with a second source for the chunked
    /// prefill graph.
    ///
    /// Building a graph consumes the weights out of its source, so a caller
    /// that wants chunked prefill has to provide them twice — and pays for a
    /// second resident copy. [`Self::open`] does this for you by re-reading the
    /// file.
    pub fn from_sources(
        cfg: Qwen4ExpConfig,
        decode_weights: &mut dyn WeightSource,
        prefill_weights: &mut dyn WeightSource,
        opts: RunnerOptions,
    ) -> Result<Self> {
        let chunk = opts.prefill_chunk;
        if chunk < 2 {
            bail!("qwen4exp: from_sources is for prefill_chunk >= 2; use from_source");
        }
        let cap = Self::resolve_cap(&cfg, &opts)?;
        let scan = opts.scan.unwrap_or(ScanState::Portable);
        let decode =
            build_qwen4exp_step_flow(&cfg, decode_weights, cap, 1, OutputKind::Logits, scan)?;
        let (pmodel, plarge) =
            build_qwen4exp_step_flow(&cfg, prefill_weights, cap, chunk, OutputKind::Logits, scan)?;
        let mut me = Self::assemble(cfg, &opts, cap, decode, 1)?;
        me.prefill = Some(Graph {
            compiled: compile_built(pmodel, opts.device)?,
            chunk,
        });
        me.prefill_layout = Some(plarge);
        Ok(me)
    }

    /// Supply the n-gram row source this model's PLE layer reads.
    pub fn with_ple_rows(mut self, rows: Box<dyn PleRowSource + Send>) -> Self {
        self.ple = Some(rows);
        self
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Positions already in the cache.
    pub fn position(&self) -> usize {
        self.consumed
    }

    /// How many positions this runner can hold.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Per-token host traffic for the recurrent state, in bytes. `0` once the
    /// scan is [`ScanState::InPlace`].
    pub fn state_bytes(&self) -> usize {
        self.session.state_bytes()
    }

    /// Clear the cache and the n-gram history.
    ///
    /// Fails under [`ScanState::InPlace`]: the scan param is zeroed at compile
    /// time and cannot be re-zeroed, so the only honest reset is a rebuild.
    pub fn reset(&mut self) -> Result<()> {
        if self.scan == ScanState::InPlace {
            bail!(
                "qwen4exp: this runner keeps the scan state on the device \
                 (RunnerOptions::single_sequence), which cannot be re-zeroed \
                 without rebuilding the graph — open a new runner per sequence"
            );
        }
        self.session = DecodeSession::new(&self.cfg, self.layout.clone(), self.cap);
        self.consumed = 0;
        Ok(())
    }

    /// Prefill `ids` and return the logits at the last position.
    pub fn prefill_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            bail!("qwen4exp: prefill needs at least one token");
        }
        let chunk = self.prefill.as_ref().map(|g| g.chunk).unwrap_or(1);
        if chunk > 1 {
            let layout = self
                .prefill_layout
                .clone()
                .expect("a chunked prefill graph always carries its layout");
            self.session.set_chunk(layout, chunk)?;
        }
        let out = self.prefill_chunks(ids, chunk);
        if chunk > 1 {
            // Back onto the one-token layout even if a chunk failed, so the
            // session is never left parsing the wrong graph's output.
            self.session.set_chunk(self.layout.clone(), 1)?;
        }
        out
    }

    fn prefill_chunks(&mut self, ids: &[u32], chunk: usize) -> Result<Vec<f32>> {
        let mut last = Vec::new();
        let mut at = 0usize;
        while at < ids.len() {
            let real = (ids.len() - at).min(chunk);
            let logits = self.step(&ids[at..at + real], chunk)?;
            // The chunk's own last real position.
            let v = self.cfg.vocab_size;
            last = logits[(real - 1) * v..real * v].to_vec();
            at += real;
        }
        Ok(last)
    }

    /// Feed one token and return the next-position logits.
    pub fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.step(&[token], 1)
    }

    /// Greedy generation after `prompt_ids`.
    ///
    /// Stops when `on_token` returns `false`, at `n_new`, or at capacity. There
    /// is deliberately no built-in EOS rule: the only end-of-text id this crate
    /// reads is `qwen4exp.ple.eos_token_id`, which is the token the n-gram hash
    /// resets on, and inferring a sampling policy from it would be a guess.
    pub fn generate_greedy<F: FnMut(u32) -> bool>(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: F,
    ) -> Result<Vec<u32>> {
        let mut logits = self.prefill_logits(prompt_ids)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax(&logits);
            out.push(next);
            if !on_token(next) {
                break;
            }
            if self.consumed >= self.cap {
                break;
            }
            logits = self.decode_logits(next)?;
        }
        Ok(out)
    }

    // ───────────────────────────── internals ─────────────────────────────

    /// Both graphs are built at the same cache capacity — one session serves
    /// them, and its buffers are sized by `cap`. A step of `chunk` positions
    /// reaches `cap + chunk`, which is what has to stay inside the QSA budget.
    fn resolve_cap(cfg: &Qwen4ExpConfig, opts: &RunnerOptions) -> Result<usize> {
        let chunk = opts.prefill_chunk.max(1);
        // `qsa_is_dense` compares *blocks*, so one step of `chunk` positions
        // past the budget is free up to the compression ratio; the default is
        // the largest cap that still leaves room for a step.
        let cap = if opts.cap == 0 {
            cfg.indexer_budget.saturating_sub(chunk - 1)
        } else {
            opts.cap
        };
        if cap == 0 {
            bail!("qwen4exp: capacity must be > 0");
        }
        if !cfg.qsa_is_dense(cap + chunk) {
            bail!(
                "qwen4exp: a capacity of {cap} with a prefill chunk of {chunk} \
                 reaches {} positions, past indexer_budget = {}",
                cap + chunk,
                cfg.indexer_budget
            );
        }
        Ok(cap)
    }

    fn build(
        cfg: &Qwen4ExpConfig,
        loader: &mut dyn rlx_core::weight_loader::WeightLoader,
        packed: bool,
        cap: usize,
        chunk: usize,
        scan: ScanState,
    ) -> Result<(rlx_flow::BuiltModel, crate::decode::DecodeLayout)> {
        if packed {
            let mut src = rlx_core::flow_bridge::PackedWeightLoaderSource(loader);
            build_qwen4exp_step_flow_with(
                cfg,
                &mut src,
                cap,
                chunk,
                OutputKind::Logits,
                scan,
                false,
            )
        } else {
            let mut map = rlx_core::weight_map::WeightMap::from_weight_loader_dequant_all(loader)?;
            let mut src = rlx_core::flow_util::WeightMapSource(&mut map);
            build_qwen4exp_step_flow_with(
                cfg,
                &mut src,
                cap,
                chunk,
                OutputKind::Logits,
                scan,
                false,
            )
        }
    }

    fn assemble(
        cfg: Qwen4ExpConfig,
        opts: &RunnerOptions,
        cap: usize,
        built: (rlx_flow::BuiltModel, crate::decode::DecodeLayout),
        chunk: usize,
    ) -> Result<Self> {
        let (model, layout) = built;
        let compiled = compile_built(model, opts.device)?;
        let scan = opts.scan.unwrap_or(ScanState::Portable);
        let names = DecodeNames::new(&cfg);
        let session = if chunk == 1 {
            DecodeSession::new(&cfg, layout.clone(), cap)
        } else {
            DecodeSession::with_chunk(&cfg, layout.clone(), cap, chunk)
        };
        let ple_modules = (0..cfg.num_hidden_layers)
            .filter_map(|l| cfg.ple_layer_index(l))
            .collect();
        Ok(Qwen4ExpRunner {
            cfg,
            device: opts.device,
            cap,
            names,
            session,
            decode: Graph { compiled, chunk },
            prefill: None,
            ple: None,
            ple_modules,
            consumed: 0,
            scan,
            layout,
            prefill_layout: None,
        })
    }

    /// One `run()`: `real` tokens right-padded into a `chunk`-wide step.
    fn step(&mut self, tokens: &[u32], chunk: usize) -> Result<Vec<f32>> {
        let real = tokens.len();
        if real == 0 || real > chunk {
            bail!("qwen4exp: {real} tokens do not fit a chunk of {chunk}");
        }
        if self.consumed + real > self.cap {
            bail!(
                "qwen4exp: {} + {real} positions exceed the capacity of {} — \
                 past the QSA budget the indexer genuinely selects, which needs \
                 a position-ordered indexer key cache",
                self.consumed,
                self.cap
            );
        }
        if !self.ple_modules.is_empty() && self.ple.is_none() {
            bail!(
                "qwen4exp: this config has {} PLE module(s) whose n-gram rows \
                 are gathered on the host, and {PLE_TABLE_TENSOR} was not in \
                 the file opened — point `RunnerOptions::ple_table` at the \
                 shard that holds it, or supply any `with_ple_rows` source",
                self.ple_modules.len()
            );
        }

        // Right-pad with the EOS the reference substitutes at masked PLE
        // positions, so the padding is the same no-op the tests pin.
        let mut ids: Vec<f32> = Vec::with_capacity(chunk);
        let mut toks: Vec<u32> = Vec::with_capacity(chunk);
        for i in 0..chunk {
            let t = if i < real {
                tokens[i]
            } else {
                self.cfg.ple_eos_token_id
            };
            toks.push(t);
            ids.push(t as f32);
        }

        let positions: Vec<[usize; 4]> = (0..chunk)
            .map(|i| text_section_pos(self.consumed + i))
            .collect();
        let (cos, sin) = mrope_tables(&self.cfg, chunk, Some(&positions));

        // n-gram rows, hashed against the history as it stands *before* this
        // step's tokens join it.
        let mut ple_bufs: Vec<(String, Vec<f32>)> = Vec::with_capacity(self.ple_modules.len());
        if let Some(rows) = self.ple.as_mut() {
            let history = self.session.ngram_history().to_vec();
            for &idx in &self.ple_modules {
                let hash = PleHash::derive(&self.cfg, idx);
                let emb = ngram_embeddings(&hash, &toks, &history, self.cfg.ple_head_dim, |r| {
                    rows.row(r)
                })?;
                ple_bufs.push((ple_embed_input(idx), emb));
            }
        }

        self.session.set_real_len(real)?;
        let graph = if chunk == 1 {
            &mut self.decode
        } else {
            self.prefill.as_mut().expect("chunked step without a graph")
        };

        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", ids.as_slice()),
            (ROPE_COS_INPUT, cos.as_slice()),
            (ROPE_SIN_INPUT, sin.as_slice()),
        ];
        for (name, buf) in &ple_bufs {
            inputs.push((name.as_str(), buf.as_slice()));
        }
        inputs.extend(self.session.inputs(&self.cfg, &self.names));

        let packed = graph
            .compiled
            .run(&inputs)
            .pop()
            .ok_or_else(|| anyhow::anyhow!("qwen4exp: the step graph produced no output"))?;
        let logits = self.session.commit(&packed)?.to_vec();

        for &t in &toks[..real] {
            self.session.push_token(t);
        }
        self.consumed += real;
        Ok(logits)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

impl LmRunner for Qwen4ExpRunner {
    fn family(&self) -> &'static str {
        "qwen4exp"
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Qwen4ExpRunner::prefill_logits(self, prompt_ids)
    }

    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Qwen4ExpRunner::prefill_logits(self, prompt_ids)
    }

    fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        Qwen4ExpRunner::decode_logits(self, token)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        Qwen4ExpRunner::generate_greedy(self, prompt_ids, n_new, on_token)
    }
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| anyhow::anyhow!("qwen4exp: {p:?} is not valid UTF-8"))
}
