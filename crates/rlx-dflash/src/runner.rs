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

//! The DFlash propose/verify loop.
//!
//! ## Who owns what
//!
//! [`DflashDrafter`] owns the three compiled graphs and the drafter's KV cache
//! and knows nothing about any particular target. [`DflashTarget`] is the small
//! surface a target model must expose: run one verification pass and hand back
//! per-slot distributions *and* the residual taps those slots produced.
//! [`DflashLoop`] drives the two.
//!
//! Keeping the taps in the same return value as the distributions is not
//! incidental — the target's verification forward is exactly the forward that
//! produces the taps for the tokens it just committed, so asking for them
//! separately would mean running the target twice.
//!
//! ## One round
//!
//! ```text
//!   drafter: [anchor, MASK, …] --decode--> draft tokens (+ slate dists)
//!   target:  [anchor, draft…]  --verify--> per-slot dists + per-slot taps
//!   accept:  maximal coupling over the slates -> accepted prefix + 1 bonus
//!   inject:  taps for slots 0..=accepted  ->  drafter KV cache
//!   next anchor = the bonus token (its taps arrive with the NEXT verify)
//! ```
//!
//! Slot 0 of the verify pass is the *previous* anchor: it was in the noise
//! block, so its KV never came from the injection path and it is still missing
//! from the drafter's cache. Injecting `0..=accepted` closes that gap exactly.
//!
//! ## Shapes are fixed up front
//!
//! Every graph is compiled in [`DflashDrafter::new`], never mid-generation:
//! the encoder and injection run at a fixed chunk width (short batches are
//! zero-padded, which is safe because injection is per-position), and the
//! decoder gets one graph per power-of-two cache bucket with the slack hidden
//! by `attn_mask`. A graph keyed on the exact `past_seq` would recompile once
//! per accepted token.
//!
//! ## Two limits that come out of that choice
//!
//! **Sliding-window attention is not supported past the window.** Padding the
//! cache breaks the identity `key index == absolute position` (real keys sit at
//! `0..n_past`, the block at `cap..cap+block`), and rlx derives a positional
//! mask's origin from `k_seq - q_seq`. So `MaskKind::SlidingWindow` would
//! measure the window from the wrong place, and a `[batch, key_len]` value mask
//! cannot express a per-query window at all. Inside the window the padding mask
//! is exact and the two agree, so [`DflashDrafter::new`] refuses a
//! `max_context` larger than the checkpoint's window rather than quietly
//! attending outside it.
//!
//! **Shared params are held once per bucket.** `token_embd.weight` and
//! `output.weight` come from the target and are uploaded into every decoder
//! graph, so N buckets cost N copies. On a 248k-vocab target that is ~5 GB
//! each; set `min_bucket == max_context` for a single bucket when memory-bound.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::{DType, Philox4x32};
use rlx_runtime::spec_decode::{AcceptDecision, SparseDist, speculative_accept_sparse};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::builder::{
    DecoderOutputs, build_decoder_graph, build_encoder_graph, build_kv_inject_graph, rope_tables,
};
use crate::config::DflashConfig;
use crate::selector::{SelectorLattice, SelectorSampling, walk_lattice};
use crate::speculate::SpecStats;

/// One verification pass from the target.
#[derive(Debug, Clone, Default)]
pub struct TargetStep {
    /// Post-sampler distribution at each of the `draft.len() + 1` slots:
    /// slot 0 conditions on the anchor, slot `i` on `draft[..i]`. Sparse
    /// because a target with top-k/top-p applied is sparse anyway.
    pub dists: Vec<SparseDist>,
    /// Residual taps for the same slots, row-major
    /// `[(draft.len() + 1), n_taps * hidden]`, layers concatenated in
    /// `cfg.target_layers` order.
    pub taps: Vec<f32>,
}

/// What a target model must expose to be drafted for.
pub trait DflashTarget {
    /// `target_layers.len() * hidden` — the width of one tap row.
    fn tap_dim(&self) -> usize;

    /// Run the prompt through the target.
    ///
    /// Returns taps for **every** prompt token (`prompt.len()` rows) and at
    /// least one distribution, the last of which is the target's choice for
    /// the token that follows the prompt. That token becomes the first anchor;
    /// its own taps arrive with the first verification pass, exactly as they do
    /// for every later anchor.
    fn prefill(&mut self, prompt: &[u32]) -> Result<TargetStep>;

    /// Run `[anchor] ++ draft` through the target in one forward.
    ///
    /// `draft` may be empty, in which case this is a plain single-token step
    /// and the result has exactly one slot.
    fn step(&mut self, anchor: u32, draft: &[u32]) -> Result<TargetStep>;

    /// Drop everything the target speculatively appended past `keep` of the
    /// drafted tokens. Called after every round, since the target ran over the
    /// whole draft but only a prefix was accepted.
    fn rollback(&mut self, keep: usize) -> Result<()>;
}

/// A [`DflashTarget`] assembled from closures.
///
/// The drafter deliberately does not depend on any model crate: a target is
/// whatever can run tokens and hand back taps. This lets a caller who already
/// drives, say, an rlx-qwen3 decode loop plug it in without writing an impl,
/// and keeps the model wiring on the model's side of the fence.
///
/// The two closures must agree with [`DflashTarget`]'s contract — `prefill`
/// returns one tap row per prompt token, `step` returns `draft.len() + 1` slots
/// of both taps and distributions. [`DflashLoop`] checks the widths and refuses
/// rather than fusing mismatched taps through `fc`.
pub struct CallbackTarget<P, S, R> {
    tap_dim: usize,
    prefill: P,
    step: S,
    rollback: R,
}

impl<P, S, R> CallbackTarget<P, S, R>
where
    P: FnMut(&[u32]) -> Result<TargetStep>,
    S: FnMut(u32, &[u32]) -> Result<TargetStep>,
    R: FnMut(usize) -> Result<()>,
{
    pub fn new(tap_dim: usize, prefill: P, step: S, rollback: R) -> Self {
        Self {
            tap_dim,
            prefill,
            step,
            rollback,
        }
    }
}

impl<P, S, R> DflashTarget for CallbackTarget<P, S, R>
where
    P: FnMut(&[u32]) -> Result<TargetStep>,
    S: FnMut(u32, &[u32]) -> Result<TargetStep>,
    R: FnMut(usize) -> Result<()>,
{
    fn tap_dim(&self) -> usize {
        self.tap_dim
    }
    fn prefill(&mut self, prompt: &[u32]) -> Result<TargetStep> {
        (self.prefill)(prompt)
    }
    fn step(&mut self, anchor: u32, draft: &[u32]) -> Result<TargetStep> {
        (self.step)(anchor, draft)
    }
    fn rollback(&mut self, keep: usize) -> Result<()> {
        (self.rollback)(keep)
    }
}

/// Knobs that change what gets compiled, so they are fixed at construction.
#[derive(Debug, Clone)]
pub struct DrafterOptions {
    pub device: Device,
    /// Largest committed context the drafter will ever hold. Decides the
    /// bucket ladder; exceeding it is an error rather than a silent truncation.
    pub max_context: usize,
    /// Smallest cache bucket. Rounds below this still pad up to it.
    pub min_bucket: usize,
    /// Greedy, or sample the selector path at this temperature. Sampling is
    /// what makes the sparse-coupling verify meaningful; greedy drafts are
    /// verified by prefix agreement instead.
    pub sampling: SelectorSampling,
    /// Drop a draft shorter than this rather than spend a target forward on it.
    pub n_min: usize,
    pub seed: u64,
}

impl Default for DrafterOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            max_context: 4096,
            min_bucket: 128,
            sampling: SelectorSampling::Greedy,
            n_min: 1,
            seed: 0x5eed,
        }
    }
}

/// Memoizing loader shim — public because *any* caller building more than one
/// DFlash graph from one checkpoint needs it.
///
/// Building N graphs from one checkpoint means reading each weight N times,
/// but `WeightLoader::take` is destructive by design (so callers can spot
/// weights they never used). This remembers what it handed out.
///
/// The packed K-quant path is *not* memoized and does not need to be —
/// `packed_meta` / `tensor_bytes_borrowed` are non-destructive borrows, so on a
/// K-quant checkpoint the big tensors cost nothing extra here. Only dense F32
/// tensors are held, which on an all-F32 checkpoint is one full copy.
pub struct ReplayLoader<'a> {
    inner: &'a mut dyn WeightLoader,
    plain: HashMap<String, (Vec<f32>, Vec<usize>)>,
    transposed: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl<'a> ReplayLoader<'a> {
    pub fn new(inner: &'a mut dyn WeightLoader) -> Self {
        Self {
            inner,
            plain: HashMap::new(),
            transposed: HashMap::new(),
        }
    }
}

impl WeightLoader for ReplayLoader<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Some(hit) = self.plain.get(key) {
            return Ok(hit.clone());
        }
        let got = self.inner.take(key)?;
        self.plain.insert(key.to_string(), got.clone());
        Ok(got)
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Some(hit) = self.transposed.get(key) {
            return Ok(hit.clone());
        }
        let got = self.inner.take_transposed(key)?;
        self.transposed.insert(key.to_string(), got.clone());
        Ok(got)
    }
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        self.inner.tensor_bytes_borrowed(key)
    }
    fn packed_meta(&self, key: &str) -> Option<(rlx_ir::quant::QuantScheme, Vec<usize>)> {
        self.inner.packed_meta(key)
    }
    fn arch_hint(&self) -> Option<&str> {
        self.inner.arch_hint()
    }
}

fn bind(
    compiled: &mut CompiledGraph,
    params: HashMap<String, Vec<f32>>,
    packed: &HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    weights: &dyn WeightLoader,
) -> Result<()> {
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    for name in packed.keys() {
        let bytes = weights
            .tensor_bytes_borrowed(name)
            .with_context(|| format!("dflash: packed bytes for {name}"))?;
        compiled.set_param_typed(name, bytes, DType::U8);
    }
    Ok(())
}

/// Smallest power-of-two bucket at or above `n`, floored at `min`.
fn bucket_for(n: usize, min: usize) -> usize {
    let mut b = min.max(1);
    while b < n {
        b *= 2;
    }
    b
}

/// The drafter: three graphs, a KV cache, and the block-proposal step.
pub struct DflashDrafter {
    cfg: DflashConfig,
    opts: DrafterOptions,
    /// Positions injected per encoder/inject call. Short batches zero-pad.
    chunk: usize,
    encoder: CompiledGraph,
    inject: CompiledGraph,
    /// Decoder per cache bucket, ascending.
    decoders: Vec<(usize, CompiledGraph, DecoderOutputs)>,
    /// `2 * n_layers` tensors, each `n_past * kv_proj_dim`.
    cache: Vec<Vec<f32>>,
    n_past: usize,
    rng: Philox4x32,
    /// Shared params still waiting on the target's copy.
    missing_shared: Vec<String>,
}

impl DflashDrafter {
    /// Compile every graph the drafter will use.
    pub fn new(
        cfg: DflashConfig,
        weights: &mut dyn WeightLoader,
        opts: DrafterOptions,
    ) -> Result<Self> {
        if cfg.block_size < 2 {
            bail!(
                "dflash: block_size {} leaves no draft slots after the anchor",
                cfg.block_size
            );
        }
        // Beyond the window the padded-bucket layout cannot reproduce the
        // checkpoint's sliding-window attention (see the module docs), and
        // attending outside it would silently diverge from the reference.
        if let Some(w) = cfg.sliding_window
            && opts.max_context > w
        {
            bail!(
                "dflash: max_context {} exceeds the checkpoint's sliding window {w}; \
                 bucketed padding cannot express a per-query window, so set \
                 max_context <= {w}",
                opts.max_context
            );
        }
        let mut w = ReplayLoader::new(weights);
        let device = opts.device;
        let chunk = cfg.block_size + 1;

        let mut packed = HashMap::new();
        let (g, p) = build_encoder_graph(&cfg, &mut w, 1, chunk, &mut packed)?;
        let mut encoder = Session::new(device).compile(g);
        bind(&mut encoder, p, &packed, &w)?;

        let mut packed = HashMap::new();
        let (g, p) = build_kv_inject_graph(&cfg, &mut w, 1, chunk, &mut packed)?;
        let mut inject = Session::new(device).compile(g);
        bind(&mut inject, p, &packed, &w)?;

        // Bucket ladder: one decoder per power-of-two cache capacity.
        let mut decoders = Vec::new();
        let mut missing_shared: Vec<String> = Vec::new();
        let mut cap = opts.min_bucket.max(1);
        loop {
            let mut packed = HashMap::new();
            let (g, p, meta) =
                build_decoder_graph(&cfg, &mut w, 1, cfg.block_size, cap, &mut packed)?;
            let mut c = Session::new(device).compile(g);
            bind(&mut c, p, &packed, &w)?;
            if decoders.is_empty() {
                missing_shared = meta.shared_params.clone();
            }
            decoders.push((cap, c, meta));
            if cap >= opts.max_context {
                break;
            }
            cap *= 2;
        }

        let n_layers = cfg.num_hidden_layers;
        Ok(Self {
            rng: Philox4x32::new(opts.seed),
            cfg,
            opts,
            chunk,
            encoder,
            inject,
            decoders,
            cache: vec![Vec::new(); n_layers * 2],
            n_past: 0,
            missing_shared,
        })
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }

    pub fn config(&self) -> &DflashConfig {
        &self.cfg
    }

    /// Tensors the drafter shares with the target and does not carry itself —
    /// `token_embd.weight`, `output.weight`. Every released DFlash checkpoint
    /// leaves both out, since an Eagle-style head has neither.
    ///
    /// Feed each to [`Self::set_shared_param`] before drafting. Until then
    /// [`Self::propose`] refuses: an unset param does not fail, it drafts
    /// noise, and noise is indistinguishable from a badly-trained drafter.
    pub fn missing_shared(&self) -> &[String] {
        &self.missing_shared
    }

    /// Upload the target's copy of a shared tensor into every graph that
    /// declared it.
    ///
    /// `token_embd.weight` is `[vocab, hidden]` (row-indexed by `Gather`);
    /// `output.weight` is `[hidden, vocab]` (right-multiplied). Getting those
    /// the wrong way round is a shape error, not silent, so it is safe to try.
    pub fn set_shared_param(&mut self, name: &str, data: &[f32]) -> Result<()> {
        if !self.missing_shared.iter().any(|n| n == name) {
            bail!(
                "dflash: {name} is not a shared param of this checkpoint (needs {:?})",
                self.missing_shared
            );
        }
        for (_, dec, _) in &mut self.decoders {
            dec.set_param(name, data);
        }
        self.missing_shared.retain(|n| n != name);
        Ok(())
    }

    /// Forget everything; the next `inject` starts a fresh sequence.
    pub fn reset(&mut self) {
        for c in &mut self.cache {
            c.clear();
        }
        self.n_past = 0;
    }

    /// Fuse `taps` and append the resulting K/V to the cache.
    ///
    /// `taps` is `[n_rows, tap_dim]` for the tokens at positions
    /// `[n_past, n_past + n_rows)`. Rows are processed in fixed-width chunks
    /// and zero-padded; injection is per-position, so the padding cannot leak
    /// into a real row.
    pub fn inject(&mut self, taps: &[f32]) -> Result<()> {
        let tap_dim = self.cfg.fused_input_dim();
        if !taps.len().is_multiple_of(tap_dim) {
            bail!(
                "dflash: taps length {} is not a multiple of tap width {tap_dim}",
                taps.len()
            );
        }
        let rows = taps.len() / tap_dim;
        let kv_dim = self.cfg.kv_proj_dim();

        for start in (0..rows).step_by(self.chunk) {
            let n = (rows - start).min(self.chunk);
            let mut padded = vec![0f32; self.chunk * tap_dim];
            padded[..n * tap_dim].copy_from_slice(&taps[start * tap_dim..(start + n) * tap_dim]);

            let fused = self
                .encoder
                .run(&[("dflash_taps", padded.as_slice())])
                .remove(0);

            let positions: Vec<usize> = (0..self.chunk).map(|i| self.n_past + start + i).collect();
            let (cos, sin) = rope_tables(&positions, self.cfg.head_dim, self.cfg.rope_theta);
            let out = self.inject.run(&[
                ("dflash_fused", fused.as_slice()),
                ("rope_cos", cos.as_slice()),
                ("rope_sin", sin.as_slice()),
            ]);

            for (slot, tensor) in out.iter().enumerate() {
                self.cache[slot].extend_from_slice(&tensor[..n * kv_dim]);
            }
        }

        self.n_past += rows;
        if self.n_past > self.opts.max_context {
            bail!(
                "dflash: cache holds {} tokens, past max_context {}",
                self.n_past,
                self.opts.max_context
            );
        }
        Ok(())
    }

    /// Draft one block continuing from `anchor`.
    ///
    /// Returns the proposed tokens and, when sampling, the slate distribution
    /// behind each one. An empty result means the drafter declined (shorter
    /// than `n_min`) and the caller should decode normally this step.
    pub fn propose(&mut self, anchor: u32) -> Result<(Vec<u32>, Vec<SparseDist>)> {
        let block = self.cfg.block_size;
        let mask_id = self
            .cfg
            .mask_token_id
            .context("dflash: checkpoint has no mask token id — cannot build a noise block")?;

        if !self.missing_shared.is_empty() {
            bail!(
                "dflash: drafting without the target's {:?} would emit noise — \
                 upload them with set_shared_param first",
                self.missing_shared
            );
        }
        let cap = bucket_for(self.n_past, self.opts.min_bucket);
        let idx = self
            .decoders
            .iter()
            .position(|(c, _, _)| *c == cap)
            .with_context(|| format!("dflash: no decoder compiled for cache bucket {cap}"))?;
        let kv_dim = self.cfg.kv_proj_dim();

        // Pad the cache up to the bucket and mask the slack.
        let mut past: Vec<Vec<f32>> = Vec::with_capacity(self.cache.len());
        for c in &self.cache {
            let mut p = vec![0f32; cap * kv_dim];
            p[..c.len()].copy_from_slice(c);
            past.push(p);
        }
        let mut attn = vec![0f32; cap + block];
        attn[..self.n_past].fill(1.0);
        attn[cap..].fill(1.0);

        let mut tokens = vec![mask_id as f32; block];
        tokens[0] = anchor as f32;
        let anchor_in = [anchor as f32];
        let positions: Vec<usize> = (self.n_past..self.n_past + block).collect();
        let (cos, sin) = rope_tables(&positions, self.cfg.head_dim, self.cfg.rope_theta);

        let names: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("noise_tokens", tokens.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
            ("attn_mask", attn.as_slice()),
        ];
        if self.cfg.dflash2.is_some() {
            inputs.push(("anchor_ids", &anchor_in));
        }
        for (i, n) in names.iter().enumerate() {
            inputs.push((n.as_str(), past[i].as_slice()));
        }

        let (_, decoder, meta) = &mut self.decoders[idx];
        let out = decoder.run(&inputs);

        match (meta.selector, meta.logits) {
            (Some((ci, si)), _) => {
                let d2 = self.cfg.dflash2.expect("selector implies dflash2");
                let lattice = SelectorLattice {
                    batch: 1,
                    block,
                    top_k: d2.selector_top_k,
                    cand_ids: out[ci].iter().map(|v| *v as u32).collect(),
                    scores: out[si].clone(),
                };
                let mut blocks =
                    walk_lattice(&lattice, self.opts.sampling, self.opts.n_min, &mut self.rng);
                let b = blocks.remove(0);
                let dists = b
                    .dists
                    .into_iter()
                    .map(|d| SparseDist {
                        ids: d.ids,
                        probs: d.probs,
                    })
                    .collect();
                Ok((b.tokens, dists))
            }
            (None, Some(li)) => {
                // DFlash v1: no lattice, so each slot is its own argmax and
                // there is no proposal distribution to verify against.
                let v = self.cfg.vocab_size;
                let mut tokens = Vec::with_capacity(block - 1);
                for pos in 1..block {
                    let row = &out[li][pos * v..(pos + 1) * v];
                    let best = row
                        .iter()
                        .enumerate()
                        .fold((0usize, f32::NEG_INFINITY), |acc, (i, x)| {
                            if *x > acc.1 { (i, *x) } else { acc }
                        })
                        .0;
                    tokens.push(best as u32);
                }
                if tokens.len() < self.opts.n_min {
                    tokens.clear();
                }
                Ok((tokens, Vec::new()))
            }
            _ => bail!("dflash: decoder produced neither logits nor a lattice"),
        }
    }
}

/// Drives a [`DflashDrafter`] against a [`DflashTarget`].
pub struct DflashLoop<T: DflashTarget> {
    pub drafter: DflashDrafter,
    pub target: T,
    pub stats: SpecStats,
    rng: Philox4x32,
}

impl<T: DflashTarget> DflashLoop<T> {
    pub fn new(drafter: DflashDrafter, target: T, seed: u64) -> Result<Self> {
        let want = drafter.config().fused_input_dim();
        let got = target.tap_dim();
        if want != got {
            bail!(
                "dflash: target emits {got}-wide taps but this drafter fuses {want} \
                 ({} layers x {} hidden) — target_layers do not match",
                drafter.config().target_layers.len(),
                drafter.config().hidden_size
            );
        }
        Ok(Self {
            drafter,
            target,
            stats: SpecStats::default(),
            rng: Philox4x32::new(seed ^ 0x85eb_ca6b),
        })
    }

    /// Prefill the prompt and return the first anchor.
    ///
    /// Every prompt token is injected; the returned anchor is *not*, because it
    /// occupies slot 0 of the first noise block. Getting that off by one leaves
    /// the drafter attending to a cache that disagrees with the target's by one
    /// position for the whole generation.
    pub fn seed(&mut self, prompt: &[u32]) -> Result<u32> {
        if prompt.is_empty() {
            bail!("dflash: cannot seed from an empty prompt");
        }
        self.drafter.reset();
        let step = self.target.prefill(prompt)?;

        let tap_dim = self.target.tap_dim();
        let want = prompt.len() * tap_dim;
        if step.taps.len() != want {
            bail!(
                "dflash: prefill returned {} tap values for a {}-token prompt (want {want})",
                step.taps.len(),
                prompt.len()
            );
        }
        let last = step
            .dists
            .last()
            .context("dflash: prefill returned no distribution for the first token")?;
        let anchor = self
            .pick(last)
            .context("dflash: prefill's final distribution is empty")?;

        self.drafter.inject(&step.taps)?;
        Ok(anchor)
    }

    /// Seed from `prompt`, then run rounds until `max_new` tokens are produced
    /// or `stop` accepts one.
    ///
    /// Returns the generated tokens, not including the prompt. Because every
    /// round commits at least one token, this always terminates.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        mut stop: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let mut anchor = self.seed(prompt)?;
        let mut out = vec![anchor];
        if stop(anchor) || out.len() >= max_new {
            out.truncate(max_new);
            return Ok(out);
        }
        while out.len() < max_new {
            let committed = self.step(anchor)?;
            anchor = *committed
                .last()
                .context("dflash: a round committed nothing — cannot advance")?;
            for t in committed {
                out.push(t);
                if stop(t) || out.len() >= max_new {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    /// Resolve one token from a target distribution, under the same rule the
    /// accept path uses: argmax when the drafter is greedy, else a draw.
    fn pick(&mut self, d: &SparseDist) -> Option<u32> {
        match self.drafter.opts.sampling {
            SelectorSampling::Greedy => argmax_id(d),
            SelectorSampling::Temperature(_) => sample_sparse(d, &mut self.rng),
        }
    }

    /// One speculative round. Returns the tokens committed this step —
    /// always at least one, so speculation never loses ground to plain
    /// decoding.
    pub fn step(&mut self, anchor: u32) -> Result<Vec<u32>> {
        let (draft, dists) = self.drafter.propose(anchor)?;
        let step = self.target.step(anchor, &draft)?;

        if step.dists.len() != draft.len() + 1 {
            bail!(
                "dflash: target returned {} slots for a {}-token draft (want {})",
                step.dists.len(),
                draft.len(),
                draft.len() + 1
            );
        }

        // Sampled drafts carry their slates, so verify by maximal coupling and
        // stay exactly on the target's distribution. A greedy draft has no
        // slate to couple against, so fall back to prefix agreement.
        let decision = if dists.len() == draft.len() && !draft.is_empty() {
            speculative_accept_sparse(&draft, &dists, &step.dists, &mut self.rng)
        } else {
            greedy_accept(&draft, &step.dists)
        };

        let accepted = decision.accepted.len();
        self.target.rollback(accepted)?;

        // Slot 0 is the previous anchor, which sat in the noise block and so
        // never went through injection; slots 1..=accepted are the accepted
        // draft tokens. The bonus becomes the next anchor and its taps arrive
        // with the next verify.
        let tap_dim = self.target.tap_dim();
        let inject_rows = accepted + 1;
        if step.taps.len() < inject_rows * tap_dim {
            bail!(
                "dflash: target returned {} tap values, need {} for {inject_rows} slots",
                step.taps.len(),
                inject_rows * tap_dim
            );
        }
        self.drafter.inject(&step.taps[..inject_rows * tap_dim])?;

        let mut out = decision.accepted;
        if let Some(bonus) = decision.corrected {
            out.push(bonus);
        }
        self.stats.record(draft.len(), accepted, out.len());
        Ok(out)
    }
}

/// Prefix agreement against the target's argmax, plus the bonus token.
///
/// This is the exact rule for a greedy draft: `draft[i]` is kept iff the target
/// would have emitted it, so the output is identical to plain greedy decoding.
fn greedy_accept(draft: &[u32], target: &[SparseDist]) -> AcceptDecision {
    let mut accepted = Vec::with_capacity(draft.len());
    for (i, d) in draft.iter().enumerate() {
        if argmax_id(&target[i]) == Some(*d) {
            accepted.push(*d);
        } else {
            return AcceptDecision {
                accepted,
                corrected: argmax_id(&target[i]),
            };
        }
    }
    AcceptDecision {
        corrected: argmax_id(&target[draft.len()]),
        accepted,
    }
}

fn sample_sparse(d: &SparseDist, rng: &mut Philox4x32) -> Option<u32> {
    let sum: f32 = d.probs.iter().sum();
    if d.ids.is_empty() {
        return None;
    }
    if sum <= f32::MIN_POSITIVE {
        return Some(d.ids[0]);
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0f32;
    for (id, p) in d.ids.iter().zip(&d.probs) {
        acc += *p;
        if r <= acc {
            return Some(*id);
        }
    }
    d.ids.last().copied()
}

fn argmax_id(d: &SparseDist) -> Option<u32> {
    d.ids
        .iter()
        .zip(&d.probs)
        .fold(None::<(u32, f32)>, |acc, (id, p)| match acc {
            Some((_, best)) if best >= *p => acc,
            _ => Some((*id, *p)),
        })
        .map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Dflash2Config;

    fn dist(ids: &[u32], probs: &[f32]) -> SparseDist {
        SparseDist {
            ids: ids.to_vec(),
            probs: probs.to_vec(),
        }
    }

    #[test]
    fn buckets_are_powers_of_two_above_the_floor() {
        assert_eq!(bucket_for(0, 128), 128);
        assert_eq!(bucket_for(128, 128), 128);
        assert_eq!(bucket_for(129, 128), 256);
        assert_eq!(bucket_for(1000, 128), 1024);
    }

    /// A target that wants the sequence `1, 2, 3, …`.
    ///
    /// It advances on `rollback`, which is the only signal telling it how much
    /// of the speculative batch survived — so it exercises the same state
    /// machine a real target has to run.
    struct CountingTarget {
        tap_dim: usize,
        /// Tokens the target considers committed so far.
        emitted: u32,
        /// Every `(anchor, draft)` the loop asked about.
        calls: Vec<(u32, Vec<u32>)>,
        rollbacks: Vec<usize>,
    }

    impl CountingTarget {
        fn new(tap_dim: usize) -> Self {
            Self {
                tap_dim,
                emitted: 0,
                calls: Vec::new(),
                rollbacks: Vec::new(),
            }
        }
    }

    impl DflashTarget for CountingTarget {
        fn tap_dim(&self) -> usize {
            self.tap_dim
        }
        fn prefill(&mut self, prompt: &[u32]) -> Result<TargetStep> {
            self.emitted = 1;
            Ok(TargetStep {
                dists: vec![dist(&[1, 101], &[0.9, 0.1])],
                taps: vec![0.05f32; prompt.len() * self.tap_dim],
            })
        }
        fn step(&mut self, anchor: u32, draft: &[u32]) -> Result<TargetStep> {
            self.calls.push((anchor, draft.to_vec()));
            let slots = draft.len() + 1;
            let mut dists = Vec::with_capacity(slots);
            for i in 0..slots {
                // Slot i conditions on `draft[..i]`, so it wants the token
                // that follows the anchor plus i.
                let want = self.emitted + 1 + i as u32;
                dists.push(dist(&[want, want + 100], &[0.9, 0.1]));
            }
            Ok(TargetStep {
                dists,
                taps: vec![0.25f32; slots * self.tap_dim],
            })
        }
        fn rollback(&mut self, keep: usize) -> Result<()> {
            self.rollbacks.push(keep);
            self.emitted += keep as u32 + 1;
            Ok(())
        }
    }

    /// The loop must reject a target whose taps are the wrong width rather
    /// than fuse garbage through `fc`.
    #[test]
    fn mismatched_tap_width_is_refused() {
        let cfg = crate::builder::tests::cfg(None);
        let mut w = crate::builder::tests::synth(&cfg);
        let drafter = DflashDrafter::new(
            cfg,
            &mut w,
            DrafterOptions {
                max_context: 128,
                min_bucket: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let target = CountingTarget::new(3);
        let err = match DflashLoop::new(drafter, target, 1) {
            Err(e) => e,
            Ok(_) => panic!("a 3-wide tap must not be accepted by this drafter"),
        };
        assert!(format!("{err}").contains("target_layers"), "got: {err}");
    }

    fn loop_over(dflash2: Option<Dflash2Config>) -> DflashLoop<CountingTarget> {
        loop_with(dflash2, SelectorSampling::Greedy)
    }

    fn loop_with(
        dflash2: Option<Dflash2Config>,
        sampling: SelectorSampling,
    ) -> DflashLoop<CountingTarget> {
        let cfg = crate::builder::tests::cfg(dflash2);
        let mut w = crate::builder::tests::synth(&cfg);
        let tap_dim = cfg.fused_input_dim();
        let drafter = DflashDrafter::new(
            cfg,
            &mut w,
            DrafterOptions {
                max_context: 128,
                min_bucket: 32,
                sampling,
                ..Default::default()
            },
        )
        .unwrap();
        DflashLoop::new(drafter, CountingTarget::new(tap_dim), 7).unwrap()
    }

    /// Under `Temperature` the drafter carries its slates, so the loop must
    /// route through `speculative_accept_sparse` rather than prefix agreement
    /// — that is the only path that keeps a *sampled* draft on the target's
    /// exact distribution. Every committed token must still be one the target
    /// itself put mass on.
    #[test]
    fn sampled_drafts_verify_against_the_target_distribution() {
        let d2 = Dflash2Config {
            conv_kernel_size: 2,
            conv_group_size: 2,
            selector_rank: 3,
            selector_top_k: 2,
        };
        let mut l = loop_with(Some(d2), SelectorSampling::Temperature(1.0));
        let mut anchor = l.seed(&[10, 11, 12]).unwrap();

        let (draft, dists) = l.drafter.propose(anchor).unwrap();
        assert_eq!(
            dists.len(),
            draft.len(),
            "a sampled walk must record one slate per proposed token"
        );
        assert!(
            dists.iter().all(|d| d.ids.len() == d2.selector_top_k),
            "each slate is the position's top-k"
        );
        for (t, d) in draft.iter().zip(&dists) {
            assert!(d.ids.contains(t), "token {t} is not in its own slate");
        }

        for _ in 0..3 {
            let out = l.step(anchor).unwrap();
            assert!(!out.is_empty());
            // The mock target only ever puts mass on `next + i` and `+100`;
            // anything else would mean the verifier leaked a draft token.
            let allowed: Vec<u32> = (1..40u32).flat_map(|i| [i, 100 + i]).collect();
            for t in &out {
                assert!(allowed.contains(t), "committed {t} outside target support");
            }
            anchor = *out.last().unwrap();
        }
    }

    /// The floor that makes speculation safe: however wrong the drafter is,
    /// a round still commits at least one token and the cache still advances.
    #[test]
    fn every_round_commits_at_least_one_token() {
        let mut l = loop_over(None);
        let mut anchor = l.seed(&[10, 11, 12]).unwrap();

        for round in 0..4 {
            let before = l.drafter.n_past();
            let out = l.step(anchor).unwrap();
            assert!(!out.is_empty(), "round {round} committed nothing");
            assert!(
                l.drafter.n_past() > before,
                "round {round} did not grow the cache"
            );
            anchor = *out.last().unwrap();
        }
        assert_eq!(l.stats.steps, 4);
        assert!(l.stats.tokens_per_target_forward() >= 1.0);
    }

    /// Injection must cover the previous anchor plus the accepted prefix —
    /// `accepted + 1` rows. Dropping the anchor row leaves a permanent hole in
    /// the cache that no later round repairs.
    #[test]
    fn cache_grows_by_accepted_plus_the_previous_anchor() {
        let mut l = loop_over(None);
        let anchor = l.seed(&[10, 11, 12]).unwrap();

        let before = l.drafter.n_past();
        let out = l.step(anchor).unwrap();
        // out = accepted ++ [bonus]; the bonus is NOT injected this round.
        let accepted = out.len() - 1;
        assert_eq!(
            l.drafter.n_past() - before,
            accepted + 1,
            "expected anchor + {accepted} accepted"
        );
        assert_eq!(l.target.rollbacks, vec![accepted]);
    }

    /// A DFlash2 checkpoint drafts through the selector and still round-trips
    /// the loop's bookkeeping.
    #[test]
    fn dflash2_round_trips_the_loop() {
        let d2 = Dflash2Config {
            conv_kernel_size: 2,
            conv_group_size: 2,
            selector_rank: 3,
            selector_top_k: 2,
        };
        let mut l = loop_over(Some(d2));
        let seeded = l.seed(&[10, 11, 12]).unwrap();

        let out = l.step(seeded).unwrap();
        assert!(!out.is_empty());
        // The target was asked to verify exactly what the drafter proposed.
        let (anchor, draft) = &l.target.calls[0];
        assert_eq!(*anchor, seeded);
        assert!(draft.len() < l.drafter.config().block_size);
    }

    /// The prompt is injected in full; the first anchor is NOT, because it
    /// sits in slot 0 of the first noise block. An off-by-one here desyncs the
    /// drafter's cache from the target's for the whole generation.
    #[test]
    fn seed_injects_the_prompt_but_not_the_anchor() {
        let mut l = loop_over(None);
        let prompt = [10u32, 11, 12, 13, 14];
        let anchor = l.seed(&prompt).unwrap();
        assert_eq!(
            l.drafter.n_past(),
            prompt.len(),
            "cache must cover exactly the prompt"
        );
        // The anchor is the target's own next-token choice, unseen by the cache.
        assert_eq!(anchor, 1, "mock target starts its sequence at 1");

        // The block therefore starts right after the prompt.
        let (_draft, _) = l.drafter.propose(anchor).unwrap();
        assert_eq!(l.drafter.n_past(), prompt.len(), "propose must not inject");
    }

    /// A target whose prefill taps do not cover every prompt token would leave
    /// a silent hole in the cache; refuse instead.
    #[test]
    fn short_prefill_taps_are_refused() {
        let mut l = loop_over(None);
        l.target.tap_dim = l.drafter.config().fused_input_dim();
        // Ask for 5 tokens but the mock returns rows for what it was given —
        // so lie about the prompt length by seeding through a truncated view.
        let err = {
            struct ShortPrefill(usize);
            impl DflashTarget for ShortPrefill {
                fn tap_dim(&self) -> usize {
                    self.0
                }
                fn prefill(&mut self, _p: &[u32]) -> Result<TargetStep> {
                    Ok(TargetStep {
                        dists: vec![dist(&[1], &[1.0])],
                        taps: vec![0.0; self.0], // one row, not five
                    })
                }
                fn step(&mut self, _a: u32, _d: &[u32]) -> Result<TargetStep> {
                    unreachable!()
                }
                fn rollback(&mut self, _k: usize) -> Result<()> {
                    Ok(())
                }
            }
            let cfg = crate::builder::tests::cfg(None);
            let mut w = crate::builder::tests::synth(&cfg);
            let tap_dim = cfg.fused_input_dim();
            let drafter = DflashDrafter::new(
                cfg,
                &mut w,
                DrafterOptions {
                    max_context: 128,
                    min_bucket: 32,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut l2 = DflashLoop::new(drafter, ShortPrefill(tap_dim), 1).unwrap();
            l2.seed(&[1, 2, 3, 4, 5]).unwrap_err()
        };
        assert!(format!("{err}").contains("tap values"), "got: {err}");
    }

    /// `generate` must terminate and honour both bounds. Every round commits at
    /// least one token, so a bounded loop cannot hang.
    #[test]
    fn generate_respects_max_new_and_stop() {
        let mut l = loop_over(None);
        let out = l.generate(&[10, 11, 12], 7, |_| false).unwrap();
        assert_eq!(out.len(), 7, "must stop exactly at max_new");

        let mut l = loop_over(None);
        let out = l.generate(&[10, 11, 12], 100, |t| t == 3).unwrap();
        assert_eq!(*out.last().unwrap(), 3, "stop token must be included");
        assert!(out.len() < 100);
    }

    /// **The correctness property of greedy speculation**: the output is
    /// exactly what the target would have produced on its own, no matter how
    /// wrong the drafter is. The drafter here runs on synthetic weights, so its
    /// proposals are noise — and the result must still be the target's own
    /// sequence, token for token.
    #[test]
    fn greedy_speculation_reproduces_plain_decoding_exactly() {
        let mut l = loop_over(None);
        let out = l.generate(&[10, 11, 12], 12, |_| false).unwrap();
        let want: Vec<u32> = (1..=12).collect();
        assert_eq!(out, want, "speculation changed the target's output");

        // And it did cost real draft attempts, so the property is not vacuous.
        assert!(l.stats.drafted > 0, "no drafting happened");
    }

    /// The closure adapter is the documented way to plug a real decode loop in
    /// without this crate depending on a model.
    #[test]
    fn callback_target_drives_the_loop() {
        let cfg = crate::builder::tests::cfg(None);
        let mut w = crate::builder::tests::synth(&cfg);
        let tap_dim = cfg.fused_input_dim();
        let drafter = DflashDrafter::new(
            cfg,
            &mut w,
            DrafterOptions {
                max_context: 128,
                min_bucket: 32,
                ..Default::default()
            },
        )
        .unwrap();

        let target = CallbackTarget::new(
            tap_dim,
            move |p: &[u32]| {
                Ok(TargetStep {
                    dists: vec![dist(&[42], &[1.0])],
                    taps: vec![0.1; p.len() * tap_dim],
                })
            },
            move |_a: u32, d: &[u32]| {
                let slots = d.len() + 1;
                Ok(TargetStep {
                    dists: vec![dist(&[42], &[1.0]); slots],
                    taps: vec![0.1; slots * tap_dim],
                })
            },
            |_k: usize| Ok(()),
        );

        let mut l = DflashLoop::new(drafter, target, 3).unwrap();
        let out = l.generate(&[1, 2, 3], 5, |_| false).unwrap();
        assert_eq!(out, vec![42; 5], "this target only ever wants 42");
        assert!(l.stats.steps > 0);
    }

    /// Greedy acceptance is exact: keep the prefix the target's argmax agrees
    /// with, then emit the target's own next token.
    #[test]
    fn greedy_accept_keeps_the_agreeing_prefix() {
        let target = vec![
            dist(&[5, 9], &[0.8, 0.2]),
            dist(&[6, 9], &[0.8, 0.2]),
            dist(&[7, 9], &[0.8, 0.2]),
        ];
        let d = greedy_accept(&[5, 99], &target);
        assert_eq!(d.accepted, vec![5]);
        assert_eq!(d.corrected, Some(6), "must emit the target's own choice");

        let d = greedy_accept(&[5, 6], &target);
        assert_eq!(d.accepted, vec![5, 6]);
        assert_eq!(d.corrected, Some(7), "bonus from the extra slot");
    }
}
