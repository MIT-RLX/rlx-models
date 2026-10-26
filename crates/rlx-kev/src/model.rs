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

//! The decision model: pack a request, run one prefill, read out
//! probabilities. No token is ever generated.
//!
//! # Why the row form
//!
//! Qwen3.5 interleaves attention with **Gated DeltaNet** layers, which are
//! recurrent and therefore ignore attention masks entirely. The packed
//! block-causal mask that isolates question branches on an attention-only
//! backbone is a no-op on those layers, so each question instead runs as its
//! own row: `state ++ branch_k`. Isolation is then exact by construction —
//! the rows are independent computations.
//!
//! That form has a property worth naming, because it is what lets this run on
//! a stock causal prefill graph with no custom mask at all: within a row the
//! state occupies positions `0..Ls` and the branch continues at `Ls..`, so
//! **position `i` is just `i`**. The branch-restart bookkeeping in
//! [`crate::encode`] only bites in the packed form.
//!
//! Positions are still supplied at runtime rather than read from a baked
//! table. Qwen3.5 declares `max_position_embeddings = 262144`, so the baked
//! MRoPE cos/sin pair would be ~268 MB of parameters for a model whose
//! records never exceed a couple of thousand tokens.
//!
//! # Why right-padding is safe
//!
//! Rows are right-padded to a common length. Every layer here is causal —
//! attention masks keys above the query index, and the DeltaNet recurrence
//! only ever reads its own past — so a pad at position `p` cannot influence
//! any output at a position below `p`. All readouts are below every pad.
//! (This is the same reasoning kev states and measured as exact; it is *not*
//! true of a prefill that exports recurrent state, which is why we build the
//! plain prefill graph rather than the cache-exporting one.)

use anyhow::{Context, Result, bail};
#[cfg(feature = "prefix-cache")]
use rlx_qwen35::Qwen35DecodeCache;
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::api::{Answer, QuestionMeta, Record, SystemOneRequest, to_answers, to_record};
use crate::encode::{EncodeOpts, Encoder, TokenizerLike, rows_of};
use crate::pointer::PointerHead;
#[cfg(feature = "prefix-cache")]
use crate::prefix::{PrefixCache, PrefixCacheConfig, PrefixEntry, cache_feeds};

/// Serving limits. Training used 384 / 1024; the per-branch cap mirrors
/// Jev's ~32k, bounded in practice by the base model's window.
pub const INFER_MAX_STATE: usize = 8192;
pub const INFER_MAX_BRANCH: usize = 8192;

/// Sequence lengths are rounded up to a multiple of this so a steady
/// workload stops recompiling. Matches kev's `KEV_SHAPE_BUCKET`.
pub const SHAPE_BUCKET: usize = 64;

/// How the compiled shape is chosen.
///
/// RLX compiles per shape and each compiled graph owns its own parameter
/// arena, so a resident variant costs a full copy of the weights. That is why
/// there is a single compiled slot rather than a cache: two live variants of a
/// 4B f32 backbone is 32 GB of duplicated weights.
#[derive(Debug, Clone, Copy)]
pub enum ShapePolicy {
    /// Round up to the next power-of-two row count and `SHAPE_BUCKET`
    /// multiple, recompiling when the bucket changes. Keeps the base weights
    /// resident so a rebuild is possible.
    Bucketed { max_rows: usize, max_seq: usize },
    /// Always run at exactly this shape. Compiles once, then releases the
    /// base weights — the lowest-memory option, and the right one for a
    /// server with a predictable workload.
    Fixed { rows: usize, seq: usize },
}

impl Default for ShapePolicy {
    fn default() -> Self {
        Self::Bucketed {
            max_rows: 32,
            max_seq: 2048,
        }
    }
}

/// Where one question's readout sits inside its row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowReadout {
    /// Index of this question's row in the batch.
    pub row: usize,
    /// Position of `<decide>` within the row.
    pub decide: usize,
    /// Position of each option's `</opt>` within the row.
    pub opts: Vec<usize>,
}

/// Final-norm hidden states for one record's question rows.
///
/// Exposed because it is the natural bisect point when probabilities
/// disagree with a reference: it separates "the trunk is wrong" from "the
/// readout is wrong", which the probabilities alone cannot.
#[derive(Debug, Clone)]
pub struct RowHidden {
    pub rows: usize,
    pub seq: usize,
    pub hidden_size: usize,
    /// `[rows, seq, hidden_size]`, row-major.
    pub hidden: Vec<f32>,
    pub readouts: Vec<RowReadout>,
}

impl RowHidden {
    /// The hidden vector at `(row, position)`.
    pub fn at(&self, row: usize, pos: usize) -> &[f32] {
        let off = (row * self.seq + pos) * self.hidden_size;
        &self.hidden[off..off + self.hidden_size]
    }
}

/// Rebuilds the weight bundle from disk.
///
/// The bundle is ~3.5 GB in f32 for Kev-0.8B and is only needed while a graph
/// is being built — afterwards the compiled arena owns its own copy. Holding
/// it for the lifetime of the model just to allow a future reshape costs that
/// much resident memory permanently. Keeping a recipe instead means the
/// bundle can be dropped after every build and re-read on the rare occasion
/// a new shape is needed (~1.5 s, versus 3.5 GB held forever).
pub type WeightsReloader = Box<dyn Fn() -> Result<Qwen35Weights> + Send>;

/// One loaded checkpoint, ready to answer requests.
pub struct DecisionModel<T: TokenizerLike> {
    encoder: Encoder<T>,
    head: PointerHead,
    cfg: Qwen35Config,
    /// Dropped after each graph build; re-read via [`Self::reloader`] when a
    /// new shape is needed.
    weights: Option<Qwen35Weights>,
    reloader: Option<WeightsReloader>,
    device: Device,
    policy: ShapePolicy,
    /// `(rows, seq, graph)` — one slot, see [`ShapePolicy`].
    compiled: Option<(usize, usize, CompiledGraph)>,
    /// Optional state-prefix path: `(bucket, graph)` for the state pass and
    /// `(past, n_new, graph)` for the branch continuation, plus the LRU.
    #[cfg(feature = "prefix-cache")]
    prefix: Option<PrefixCache>,
    #[cfg(feature = "prefix-cache")]
    state_graph: Option<(usize, CompiledGraph)>,
    #[cfg(feature = "prefix-cache")]
    cont_graph: Option<(usize, usize, CompiledGraph)>,
    /// `output_norm` gamma, kept because the continuation graph emits the
    /// hidden *before* the final RMS norm — see [`Self::apply_output_norm`].
    #[cfg(feature = "prefix-cache")]
    output_norm: Vec<f32>,
    encode_opts: EncodeOpts,
}

impl<T: TokenizerLike> DecisionModel<T> {
    /// Assemble a model from its parts.
    ///
    /// Errors when the pointer head and the backbone disagree about the
    /// hidden size — a mismatch there reads as a shape error only if you are
    /// lucky, and as plausible garbage if the sizes happen to divide.
    pub fn new(
        encoder: Encoder<T>,
        head: PointerHead,
        cfg: Qwen35Config,
        weights: Qwen35Weights,
        device: Device,
        policy: ShapePolicy,
    ) -> Result<Self> {
        if head.hidden != cfg.hidden_size {
            bail!(
                "pointer head expects hidden={} but the backbone is {}",
                head.hidden,
                cfg.hidden_size
            );
        }
        // CoreML cannot run a Gated DeltaNet trunk: the GDN prologue narrows a
        // fused projection and feeds the slices through `sigmoid`/`softplus`
        // into a reshape, and MPSGraph miscompiles exactly that shape (it
        // rewrites the reshape onto the narrow's parent and drops the offset).
        // `rlx-coreml` now refuses instead of emitting a module that aborts the
        // process, but the refusal lands mid-compile — after the base model has
        // been read. Check here so the caller finds out before paying for that.
        if device == Device::Ane && has_linear_attention(&cfg) {
            bail!(
                "CoreML/ANE cannot run this model: Qwen3.5 interleaves Gated \
                 DeltaNet layers, and MPSGraph miscompiles the narrow-then-reshape \
                 shape their prologue emits. Use cpu, metal or mlx."
            );
        }
        if rlx_ir::env::parse_or::<usize>("RLX_QWEN35_SWA_WINDOW", 0) > 0 {
            bail!(
                "RLX_QWEN35_SWA_WINDOW is set; sliding-window prefill drops middle \
                 context and would change every probability this model returns"
            );
        }
        #[cfg(feature = "prefix-cache")]
        let output_norm = weights.output_norm.clone();
        Ok(Self {
            encoder,
            head,
            cfg,
            weights: Some(weights),
            reloader: None,
            #[cfg(feature = "prefix-cache")]
            prefix: None,
            #[cfg(feature = "prefix-cache")]
            state_graph: None,
            #[cfg(feature = "prefix-cache")]
            cont_graph: None,
            #[cfg(feature = "prefix-cache")]
            output_norm,
            device,
            policy,
            compiled: None,
            encode_opts: EncodeOpts {
                max_state: INFER_MAX_STATE,
                max_branch: INFER_MAX_BRANCH,
                strict: false,
                option_isolation: false,
            },
        })
    }

    /// Enable state-prefix caching.
    ///
    /// **Experimental and currently incorrect** — see [`crate::prefix`].
    /// Behind the off-by-default `prefix-cache` feature.
    #[cfg(feature = "prefix-cache")]
    pub fn with_prefix_cache(mut self, cfg: PrefixCacheConfig) -> Self {
        self.prefix = Some(PrefixCache::new(cfg));
        self
    }

    /// `(hits, misses, cached_states)`, or `None` when caching is off.
    #[cfg(feature = "prefix-cache")]
    pub fn prefix_stats(&self) -> Option<(usize, usize, usize)> {
        self.prefix
            .as_ref()
            .map(|p| (p.hits(), p.misses(), p.len()))
    }

    /// Supply a way to re-read the weight bundle, so it can be released
    /// after each build instead of held for the model's lifetime.
    pub fn with_reloader(mut self, reload: WeightsReloader) -> Self {
        self.reloader = Some(reload);
        self
    }

    /// Whether the weight bundle is currently resident.
    pub fn weights_resident(&self) -> bool {
        self.weights.is_some()
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    pub fn head(&self) -> &PointerHead {
        &self.head
    }

    pub fn encoder(&self) -> &Encoder<T> {
        &self.encoder
    }

    /// The compiled shape currently resident, if any.
    pub fn compiled_shape(&self) -> Option<(usize, usize)> {
        self.compiled.as_ref().map(|(r, s, _)| (*r, *s))
    }

    /// Per-question probability vectors for one record.
    pub fn probs(&mut self, rec: &Record) -> Result<Vec<Vec<f32>>> {
        let h = self.row_hidden(rec)?;
        let d = h.hidden_size;
        let mut out = Vec::with_capacity(h.readouts.len());
        for ro in &h.readouts {
            let h_decide = h.at(ro.row, ro.decide);
            let mut h_opts = Vec::with_capacity(ro.opts.len() * d);
            for o in &ro.opts {
                h_opts.extend_from_slice(h.at(ro.row, *o));
            }
            out.push(self.head.probs(h_decide, &h_opts)?);
        }
        Ok(out)
    }

    /// Run the trunk and return the per-row hidden states plus readout
    /// offsets, without applying the pointer head.
    pub fn row_hidden(&mut self, rec: &Record) -> Result<RowHidden> {
        let enc = self.encoder.encode(rec, self.encode_opts)?;
        let (state_ids, _state_pos, rows) = rows_of(&enc)?;
        if rows.is_empty() {
            bail!("record has no questions");
        }
        let ls = state_ids.len();

        // Each question becomes one causal row: the state, then its branch.
        let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        let mut readouts: Vec<RowReadout> = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            let mut ids = Vec::with_capacity(ls + r.ids.len());
            ids.extend_from_slice(&state_ids);
            ids.extend_from_slice(&r.ids);
            seqs.push(ids);
            readouts.push(RowReadout {
                row: i,
                decide: ls + r.decide,
                opts: r.opts.iter().map(|o| ls + o).collect(),
            });
        }

        // Prefix path: encode the state once, replay each branch against it.
        #[cfg(feature = "prefix-cache")]
        if self.prefix.as_ref().is_some_and(|p| p.eligible(ls)) {
            return self.row_hidden_cached(&state_ids, &rows, readouts);
        }

        let longest = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let (rows_n, seq_n) = self.resolve_shape(seqs.len(), longest)?;

        let hidden = self.forward(&seqs, rows_n, seq_n)?;
        Ok(RowHidden {
            rows: rows_n,
            seq: seq_n,
            hidden_size: self.cfg.hidden_size,
            hidden,
            readouts,
        })
    }

    /// Answer a `POST /v1/systemone` request.
    pub fn answer(&mut self, req: &SystemOneRequest) -> Result<(Vec<(String, Answer)>, Usage)> {
        let (rec, meta) = to_record(req)?;
        let probs = self.probs(&rec)?;
        let answers = to_answers(&probs, &meta)?;
        let usage = self.usage(&rec, &meta)?;
        Ok((answers, usage))
    }

    fn usage(&self, rec: &Record, _meta: &[QuestionMeta]) -> Result<Usage> {
        let enc = self.encoder.encode(rec, self.encode_opts)?;
        Ok(Usage {
            input_tokens: enc.ids.len(),
            state_tokens: enc.state_len(),
        })
    }

    /// Pick (and validate) the compiled shape for a request.
    /// The compiled shape this record would need, without running it.
    ///
    /// One graph is compiled per `(rows, seq)` bucket and swapping between them
    /// is expensive, so a caller scoring many records can group by this and pay
    /// one compile per bucket instead of one per record. Encoding is cheap next
    /// to a forward pass, so computing it twice is a good trade.
    pub fn shape_hint(&self, rec: &Record) -> Result<(usize, usize)> {
        let enc = self.encoder.encode(rec, self.encode_opts)?;
        let (state_ids, _pos, rows) = rows_of(&enc)?;
        let longest = rows
            .iter()
            .map(|r| state_ids.len() + r.ids.len())
            .max()
            .unwrap_or(0);
        self.resolve_shape(rows.len(), longest)
    }

    fn resolve_shape(&self, rows: usize, longest: usize) -> Result<(usize, usize)> {
        match self.policy {
            ShapePolicy::Fixed { rows: r, seq: s } => {
                if rows > r || longest > s {
                    bail!(
                        "request needs {rows}×{longest} but the model is pinned to \
                         {r}×{s}; raise the fixed shape or use a bucketed policy"
                    );
                }
                Ok((r, s))
            }
            ShapePolicy::Bucketed { max_rows, max_seq } => {
                let r = rows.next_power_of_two().min(max_rows).max(rows);
                let s = longest.div_ceil(SHAPE_BUCKET) * SHAPE_BUCKET;
                if r > max_rows || s > max_seq {
                    bail!(
                        "request needs {r}×{s} which exceeds the {max_rows}×{max_seq} \
                         limit ({rows} questions, {longest} tokens in the longest branch)"
                    );
                }
                Ok((r, s.max(SHAPE_BUCKET)))
            }
        }
    }

    /// Run the backbone, returning the final-norm hidden states
    /// `[rows, seq, hidden]`.
    fn forward(&mut self, seqs: &[Vec<u32>], rows: usize, seq: usize) -> Result<Vec<f32>> {
        self.ensure_compiled(rows, seq)?;

        // Pad id 0 rather than the tokenizer's: pads sit after every real
        // token and nothing causal can read forward, so the value is
        // unobservable — and 0 is in range for every vocabulary.
        let mut input_ids = vec![0f32; rows * seq];
        for (i, s) in seqs.iter().enumerate() {
            for (j, id) in s.iter().enumerate() {
                input_ids[i * seq + j] = *id as f32;
            }
        }

        // Row positions are 0..seq-1 and identical across rows, so one
        // `[seq, rot_half]` MRoPE table serves the whole batch.
        //
        // `rot_half` is `rope_dim_count / 2`, **not** `key_length / 2`.
        // Qwen3.5 uses partial rotary (64 of 256 head dims here), so the two
        // differ by 4x; feeding the wider stride silently mis-reads every
        // position and the model still returns a plausible distribution.
        let rot_half = self.cfg.rope_dim_count / 2;
        let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&self.cfg, seq, None, rot_half);
        debug_assert_eq!(cos.len(), seq * rot_half);

        let graph = self
            .compiled
            .as_mut()
            .map(|(_, _, g)| g)
            .expect("compiled above");
        let out = graph.run(&[
            ("input_ids", &input_ids[..]),
            ("rope_cos", &cos[..]),
            ("rope_sin", &sin[..]),
        ]);
        let want = rows * seq * self.cfg.hidden_size;
        let hidden = out
            .first()
            .ok_or_else(|| anyhow::anyhow!("prefill graph produced no outputs"))?;
        if hidden.len() != want {
            bail!(
                "expected [{rows}, {seq}, {}] = {want} hidden values, got {}",
                self.cfg.hidden_size,
                hidden.len()
            );
        }
        Ok(hidden.clone())
    }

    fn ensure_compiled(&mut self, rows: usize, seq: usize) -> Result<()> {
        if matches!(self.compiled, Some((r, s, _)) if r == rows && s == seq) {
            return Ok(());
        }
        if self.weights.is_none() {
            let reload = self.reloader.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "the weight bundle was released after building {:?} and this model \
                     has no reloader, so shape {rows}x{seq} cannot be compiled; pin the \
                     shape with a Fixed policy or construct the model via `load`",
                    self.compiled_shape()
                )
            })?;
            // Drop the old graph FIRST: reloading while the previous arena is
            // still alive is the one case that doubles peak memory.
            self.compiled = None;
            self.weights = Some(reload().context("re-reading the base weights")?);
        }
        let weights = self.weights.as_ref().expect("resident above");

        // Drop the previous variant: its arena holds a full copy of the
        // weights, and holding two at once doubles resident memory.
        self.compiled = None;

        let (hir, params, packed) = rlx_qwen35::build_qwen35_prefill_flow_ext(
            &self.cfg, weights, rows, seq, /*with_lm_head*/ false,
            /*last_logits_only*/ false, /*enable_mtp_head*/ false,
            /*runtime_mrope*/ true, /*fast_mtp*/ false,
            /*export_normed_hidden*/ true,
        )
        .with_context(|| format!("building the qwen35 prefill graph at {rows}×{seq}"))?;
        if !packed.is_empty() {
            bail!(
                "this path expects dequantized f32 weights, but {} tensor(s) came \
                 back packed",
                packed.len()
            );
        }

        // Release the source weights BEFORE compiling. `params` already holds
        // its own f32 copy of everything the graph needs, so keeping the
        // bundle alive across compile + upload doubles the resident cost at
        // exactly the peak — measured at ~3.5 GB on Kev-0.8B. A later reshape
        // re-reads it through the reloader.
        let pinned = matches!(self.policy, ShapePolicy::Fixed { .. });
        if pinned || self.reloader.is_some() {
            self.weights = None;
        }

        let mut compiled = Session::new(self.device)
            .compile_hir(hir)
            .with_context(|| format!("compiling the prefill graph at {rows}×{seq}"))?;
        // Drain rather than iterate: each uploaded tensor is freed as we go,
        // so the peak is params + arena + one tensor, not two full copies.
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }

        self.compiled = Some((rows, seq, compiled));
        Ok(())
    }
}

/// Whether any trunk layer is a Gated DeltaNet ("linear attention") block.
///
/// `full_attention_interval` is the stride at which full-attention layers are
/// placed; a stride of 1 (or a single layer) means every layer is attention
/// and there is no recurrent block.
fn has_linear_attention(cfg: &Qwen35Config) -> bool {
    let trunk = cfg
        .num_hidden_layers
        .saturating_sub(cfg.nextn_predict_layers);
    cfg.full_attention_interval > 1 && trunk > 0
}

/// Token accounting for the response body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: usize,
    pub state_tokens: usize,
}

/// The state-prefix path.
///
/// **Experimental: this does not currently reproduce the uncached answers.**
/// See [`crate::prefix`] for the investigation so far. Behind the
/// off-by-default `prefix-cache` feature so no default build can reach it.
#[cfg(feature = "prefix-cache")]
impl<T: TokenizerLike> DecisionModel<T> {
    /// Bucket a length up to a `SHAPE_BUCKET` multiple.
    fn bucket(n: usize) -> usize {
        n.div_ceil(SHAPE_BUCKET).max(1) * SHAPE_BUCKET
    }

    /// The prefix-cached forward: one state pass, then one continuation per
    /// question row.
    ///
    /// Readout offsets returned here are **branch-local** — the state no
    /// longer occupies positions in the row — so they are rewritten from the
    /// packed offsets the caller computed.
    fn row_hidden_cached(
        &mut self,
        state_ids: &[u32],
        rows: &[crate::encode::Row],
        packed_readouts: Vec<RowReadout>,
    ) -> Result<RowHidden> {
        let ls = state_ids.len();
        let ls_bucket = Self::bucket(ls);
        let longest_branch = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let new_bucket = Self::bucket(longest_branch);
        let d = self.cfg.hidden_size;

        if let ShapePolicy::Bucketed { max_seq, .. } = self.policy
            && ls_bucket + new_bucket > max_seq
        {
            bail!(
                "prefix path needs {ls_bucket}+{new_bucket} tokens which exceeds max_seq {max_seq}"
            );
        }

        // ── state pass (cached) ──────────────────────────────────────
        let have = self
            .prefix
            .as_mut()
            .and_then(|p| p.get(state_ids))
            .map(|e| (e.cache.clone(), e.bucket));
        let (cache, past_bucket) = match have {
            Some(hit) => hit,
            None => {
                let cache = self.run_state_pass(state_ids, ls_bucket)?;
                if let Some(p) = self.prefix.as_mut() {
                    p.insert(PrefixEntry {
                        state_ids: state_ids.to_vec(),
                        cache: cache.clone(),
                        bucket: ls_bucket,
                    });
                }
                (cache, ls_bucket)
            }
        };

        // ── continuation, one row at a time ──────────────────────────
        self.ensure_cont_graph(past_bucket, new_bucket)?;
        let n_head = self.cfg.num_attention_heads;
        let rot_half = self.cfg.rope_dim_count / 2;

        let mut hidden = vec![0f32; rows.len() * new_bucket * d];
        for (i, row) in rows.iter().enumerate() {
            let mut ids = vec![0f32; new_bucket];
            for (j, t) in row.ids.iter().enumerate() {
                ids[j] = *t as f32;
            }
            // Branch positions continue the state: Ls, Ls+1, …
            let sections: Vec<[usize; 4]> = (0..new_bucket)
                .map(|j| rlx_qwen35::text_section_pos(ls + j))
                .collect();
            let (cos, sin) =
                rlx_qwen35::mrope_prefill_feeds(&self.cfg, new_bucket, Some(&sections), rot_half);
            // `past_seq` = the REAL state length, `bucket_upper` = the padded
            // extent, so the pad slots between them are masked out.
            let mask = rlx_qwen35::build_verify_bias_mask(1, n_head, ls, new_bucket, past_bucket);

            let owned = cache_feeds(&cache);
            let graph = self
                .cont_graph
                .as_mut()
                .map(|(_, _, g)| g)
                .expect("continuation graph built above");
            let mut feeds: Vec<(&str, &[f32])> = vec![
                ("input_ids", &ids[..]),
                ("rope_cos", &cos[..]),
                ("rope_sin", &sin[..]),
                ("mask", &mask[..]),
            ];
            for (n, v) in &owned {
                feeds.push((n.as_str(), v.as_slice()));
            }
            let out = graph.run(&feeds);
            let h = out
                .first()
                .ok_or_else(|| anyhow::anyhow!("continuation graph produced no outputs"))?;
            let want = new_bucket * d;
            if h.len() < want {
                bail!(
                    "continuation returned {} values, expected at least {want}",
                    h.len()
                );
            }
            let dst = &mut hidden[i * want..(i + 1) * want];
            dst.copy_from_slice(&h[..want]);
            self.apply_output_norm(dst);
        }

        // Rewrite the readouts: rows are now branch-only.
        let readouts = packed_readouts
            .into_iter()
            .enumerate()
            .map(|(i, r)| RowReadout {
                row: i,
                decide: r.decide - ls,
                opts: r.opts.iter().map(|o| o - ls).collect(),
            })
            .collect();

        Ok(RowHidden {
            rows: rows.len(),
            seq: new_bucket,
            hidden_size: d,
            hidden,
            readouts,
        })
    }

    /// Apply the trunk's final RMS norm, in place, to `[rows, hidden]`.
    ///
    /// The prefill flow's `export_normed_hidden` returns the hidden *after*
    /// `output_norm`; the decode flow's hidden output is the value *before* it
    /// (`primary = logits.unwrap_or(h_lm)`, and the norm lives on the logits
    /// path inside `emit_qwen35_prefill_tail`). The two are not the same
    /// quantity, and the difference is easy to miss: the RMS is nearly
    /// identical, so only a handful of large channels disagree — element 0 was
    /// -50 where it should have been -6.3, which read as "close but wrong".
    ///
    /// Qwen3.5 uses the `(1 + weight)` convention, matching
    /// `Qwen35Config::rms_norm_offset`.
    fn apply_output_norm(&self, rows: &mut [f32]) {
        let d = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let offset = if self.cfg.rms_norm_offset { 1.0 } else { 0.0 };
        for row in rows.chunks_exact_mut(d) {
            let ms = row.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (i, x) in row.iter_mut().enumerate() {
                *x = *x * inv * (self.output_norm[i] + offset);
            }
        }
    }

    /// Run the state alone through a prefill graph that exports the trunk's
    /// recurrent + KV state.
    fn run_state_pass(&mut self, state_ids: &[u32], bucket: usize) -> Result<Qwen35DecodeCache> {
        self.ensure_state_graph(bucket)?;
        let ls = state_ids.len();
        let rot_half = self.cfg.rope_dim_count / 2;
        let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&self.cfg, bucket, None, rot_half);

        let mut ids = vec![0f32; bucket];
        for (j, t) in state_ids.iter().enumerate() {
            ids[j] = *t as f32;
        }
        // The GDN scan and conv window must not advance over the padding.
        let pad = rlx_qwen35::prompt_pad_mask_feeds(&self.cfg, 1, bucket, &[ls]);
        let zeros = rlx_qwen35::zero_recurrent_inputs(&self.cfg, 1);
        let last = [(ls.saturating_sub(1)) as f32];

        let graph = self
            .state_graph
            .as_mut()
            .map(|(_, g)| g)
            .expect("state graph built above");
        let mut feeds: Vec<(&str, &[f32])> = vec![
            ("input_ids", &ids[..]),
            ("rope_cos", &cos[..]),
            ("rope_sin", &sin[..]),
            ("last_token_idx", &last[..]),
        ];
        for (n, v) in pad.iter().chain(zeros.iter()) {
            feeds.push((n.as_str(), v.as_slice()));
        }
        let outputs = graph.run(&feeds);

        let (_trunk, cache, _mtp) = rlx_qwen35::seed_cache_from_outputs(
            &self.cfg,
            1,
            bucket,
            &[ls],
            outputs,
            false,
            /*trunk_is_hidden*/ true,
        )?;
        Ok(cache)
    }

    fn ensure_state_graph(&mut self, bucket: usize) -> Result<()> {
        if matches!(self.state_graph, Some((b, _)) if b == bucket) {
            return Ok(());
        }
        let weights = self.resident_weights()?;
        self.state_graph = None;
        let (g, params, packed) = rlx_qwen35::build_qwen35_graph_sized_ext(
            &self.cfg, weights, 1, bucket, /*with_lm_head*/ false,
            /*last_logits_only*/ false, /*enable_mtp_head*/ false,
            /*export_recurrent_state*/ true, /*decode_past_seq*/ None,
            /*runtime_mrope*/ true, /*fast_mtp*/ false,
            /*export_normed_hidden*/ true, /*export_trunk_layer_hiddens*/ false,
        )
        .with_context(|| format!("building the state-pass graph at 1x{bucket}"))?;
        anyhow::ensure!(packed.is_empty(), "state pass: unexpected packed weights");
        let mut c = Session::new(self.device).compile(g);
        for (n, v) in params {
            c.set_param(&n, &v);
        }
        self.state_graph = Some((bucket, c));
        Ok(())
    }

    fn ensure_cont_graph(&mut self, past: usize, n_new: usize) -> Result<()> {
        if matches!(self.cont_graph, Some((p, n, _)) if p == past && n == n_new) {
            return Ok(());
        }
        let weights = self.resident_weights()?;
        self.cont_graph = None;
        let opts = rlx_qwen35::Qwen35DecodeOpts {
            batch: 1,
            past_seq: past,
            seq: n_new,
            // Emit hidden for EVERY new position: the readouts sit at the
            // `</opt>` and `<decide>` tokens, not just the last one.
            verify_all: true,
            dynamic_past: false,
            use_custom_mask: true,
            bias_mask: true,
            force_host_embed: false,
            enable_mtp_head: false,
            fast_mtp: false,
            // `with_lm_head = !fast_greedy_lm_head` — this is how the decode
            // flow is asked for the normed hidden instead of logits.
            fast_greedy_lm_head: true,
            profile: None,
            residual_out: false,
            greedy_token: false,
            multi_row: false,
        };
        let (hir, params, packed) = rlx_qwen35::build_qwen35_decode_model_flow(
            &self.cfg,
            std::sync::Arc::new(weights),
            &opts,
        )
        .with_context(|| format!("building the continuation graph at 1x{n_new} past={past}"))?;
        anyhow::ensure!(packed.is_empty(), "continuation: unexpected packed weights");
        let mut c = Session::new(self.device)
            .compile_hir(hir)
            .with_context(|| format!("compiling the continuation graph at 1x{n_new}"))?;
        for (n, v) in params {
            c.set_param(&n, &v);
        }
        self.cont_graph = Some((past, n_new, c));
        Ok(())
    }

    /// The weight bundle, re-reading it if it was released after a build.
    fn resident_weights(&mut self) -> Result<Qwen35Weights> {
        if self.weights.is_none() {
            let reload = self.reloader.as_ref().ok_or_else(|| {
                anyhow::anyhow!("the weight bundle was released and no reloader is installed")
            })?;
            self.weights = Some(reload().context("re-reading the base weights")?);
        }
        Ok(self.weights.clone().expect("resident above"))
    }
}
