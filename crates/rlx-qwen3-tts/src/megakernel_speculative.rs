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

//! Talker speculative-decoding AR loop — `cfg(feature = "speculative-decode")`.
//!
//! Wraps [`Qwen3TtsMegakernel`] with a [`synthesize_codec_ar_speculative`]
//! variant of [`crate::megakernel::Qwen3TtsMegakernel::synthesize_codec_ar`]
//! that uses a [`DraftModel`] to propose `K` future g0 codec tokens,
//! verifies them in a single batched talker forward, and commits the
//! longest matching prefix (plus one "free" verifier token at the mismatch
//! position).
//!
//! # First-cut scope and the trivial-draft approximation
//!
//! This module implements the speculative formulation that's easiest to ship
//! correctly: a small draft model proposes only g0 tokens; the K verifier
//! input embeddings are all set to the codec_emb produced from the most
//! recent committed (g0, hidden) pair. That is — the verifier sees
//! `codec_emb(t)` repeated K+1 times at positions `[t, t+1, ..., t+K]`,
//! and produces K+1 hidden rows whose lm_head argmax is compared to the
//! K drafts + 1 "free" verifier output beyond the drafts.
//!
//! Strict autoregressive equivalence would require each row of the verifier
//! batch to be `codec_emb(t+i) = CP(true_hidden(t+i), draft[i])` — but
//! `true_hidden(t+i)` is exactly what the verifier is producing, so without
//! the draft owning its own (cheap) hidden states there is no way to obtain
//! the correct embedding cheaply. The approximation "codec_emb stays the
//! same" is exact for silences and sustained vowels, and inexact (but
//! often still high-acceptance) for stretches where g0 is in fact constant.
//! Audio quality is validated end-to-end via Whisper round-trip.
//!
//! A learned draft (a tiny Qwen3-shaped decoder that exposes its own hidden
//! state) can lift this approximation: row `i` of the verifier batch becomes
//! `CP(draft_hidden(t+i), draft_g0(t+i))`. The acceptance loop here is
//! unchanged for such a draft — just swap the `DraftModel` impl and let
//! `propose_with_inputs` (a v2 extension) plumb the per-row inputs through.

use crate::config::TalkerConfig;
use crate::megakernel::Qwen3TtsMegakernel;
use crate::talker::eager::DraftKvCache;
use crate::talker::learned_draft::LearnedDraft;
use crate::talker::math::{
    apply_repetition_penalty, linear_logits_flat_into, sample_greedy_talker_codec,
};
use crate::talker::speculative::{
    AcceptancePolicy, DEFAULT_DRAFT_LEN, DraftModel, SpecRunStats, SpecStepStats,
};
use anyhow::{Context, Result, ensure};
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::time::Instant;

/// Configuration for one speculative-decoding run.
pub struct SpecConfig<'a, D: DraftModel> {
    /// Drafter — proposes `draft_len` g0 tokens per verify batch.
    pub draft: &'a mut D,
    /// How many tokens the draft proposes per step (verifier batch size = K + 1).
    pub draft_len: usize,
    /// Acceptance policy for matching draft vs verifier outputs.
    pub policy: AcceptancePolicy,
    /// Repetition penalty (same semantics as the non-spec loop).
    pub rep_penalty: f32,
    /// Pad embed for CP `predict_groups_fill_emb`.
    pub tts_pad_embed: &'a [f32],
    /// Minimum committed frames before honouring an EOS (matches the plain loop).
    pub min_frames: usize,
    /// Maximum total committed codec frames (hard ceiling).
    pub max_frames: usize,
    /// Anti-cascade guard. When the most recent `cascade_guard_window`
    /// committed g0 tokens are all identical, the megakernel skips
    /// speculation for that step and does plain single-token AR instead.
    /// This breaks the runaway-acceptance pathology that affects any draft
    /// whose proposals don't vary across the K slots (e.g. [`TrivialDraft`]).
    /// Set to `0` to disable. Default is 3.
    pub cascade_guard_window: usize,
    /// When `Some(n)`, use self-speculative early-exit drafting: the
    /// talker's first `n` transformer layers act as the draft model.
    /// Bypasses the [`DraftModel`] dispatch entirely — drafts and
    /// per-position codec_embs are produced inline by the megakernel via
    /// [`TalkerEagerModel::early_exit_decode_step`].
    ///
    /// `n` should be roughly `talker_layers / 6..4` (e.g. 4–7 layers for a
    /// 28-layer talker). Smaller → cheaper draft but lower acceptance;
    /// larger → higher acceptance but draft cost approaches verifier cost.
    pub early_exit_layers: Option<usize>,
    /// When `Some(&mut LearnedDraft)`, use the supplied independent
    /// (separately-loaded) Qwen3-shaped sidecar as the draft. Takes
    /// precedence over [`Self::early_exit_layers`] and the [`DraftModel`]
    /// trait dispatch. Expects the draft to use the *same* hidden size,
    /// head count, head_dim, and `codec_head`/`codec_embedding` tables
    /// as the verifier talker (the v1 constraint — see
    /// [`crate::talker::learned_draft`] docs).
    pub learned_draft: Option<&'a mut LearnedDraft>,
}

impl<'a, D: DraftModel> SpecConfig<'a, D> {
    pub fn new(
        draft: &'a mut D,
        rep_penalty: f32,
        tts_pad_embed: &'a [f32],
        min_frames: usize,
        max_frames: usize,
    ) -> Self {
        Self {
            draft,
            draft_len: DEFAULT_DRAFT_LEN,
            policy: AcceptancePolicy::default(),
            rep_penalty,
            tts_pad_embed,
            min_frames,
            max_frames,
            cascade_guard_window: 3,
            early_exit_layers: None,
            learned_draft: None,
        }
    }

    pub fn with_cascade_guard_window(mut self, n: usize) -> Self {
        self.cascade_guard_window = n;
        self
    }

    pub fn with_early_exit_layers(mut self, n: usize) -> Self {
        self.early_exit_layers = Some(n);
        self
    }

    pub fn with_learned_draft(mut self, draft: &'a mut LearnedDraft) -> Self {
        self.learned_draft = Some(draft);
        self
    }

    pub fn with_draft_len(mut self, k: usize) -> Self {
        self.draft_len = k;
        self
    }
}

/// Result of a speculative AR run.
pub struct SpecRunResult {
    pub codec_frames: Vec<Vec<u32>>,
    pub stats: SpecRunStats,
    pub prefill_secs: f64,
    pub talker_secs: f64,
    pub cp_secs: f64,
    pub total_secs: f64,
}

/// Stepwise speculative codec-AR session — held across calls to
/// [`Qwen3TtsMegakernel::codec_ar_speculative_step`].
pub struct SpecCodecArState {
    pub(crate) hidden: Vec<f32>,
    pub(crate) logits_scratch: Vec<f32>,
    pub(crate) codec_emb_scratch: Vec<f32>,
    pub(crate) cp_input_scratch: Vec<f32>,
    pub(crate) verify_in: Array2<f32>,
    pub(crate) codec_table: Vec<f32>,
    pub(crate) past_g0: Vec<u32>,
    /// Frames emitted so far (in order). Index into this for partial decoding.
    pub codec_frames: Vec<Vec<u32>>,
    pub stats: SpecRunStats,
    pub talker_secs: f64,
    pub cp_secs: f64,
    pub prefill_secs: f64,
    pub(crate) max_frames: usize,
    pub(crate) draft_kv: Option<DraftKvCache>,
    pub(crate) draft_buf: Vec<f32>,
    pub(crate) draft_emb_buf: Vec<f32>,
    pub(crate) prior_codec_embs: Vec<Vec<f32>>,
    pub(crate) done: bool,
    pub(crate) hidden_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) eos: u32,
    pub(crate) n_groups: usize,
    pub(crate) k_draft: usize,
}

impl SpecCodecArState {
    /// True once the AR has hit its terminal state (EOS or max_frames).
    pub fn is_done(&self) -> bool {
        self.done || self.codec_frames.len() >= self.max_frames
    }

    /// How many frames have been emitted so far.
    pub fn frames_emitted(&self) -> usize {
        self.codec_frames.len()
    }

    /// Finalize and return the produced frames + accumulated timings.
    pub fn finish(self) -> (Vec<Vec<u32>>, SpecRunStats, f64, f64, f64) {
        (
            self.codec_frames,
            self.stats,
            self.prefill_secs,
            self.talker_secs,
            self.cp_secs,
        )
    }
}

/// Outcome of one speculative AR step.
#[derive(Debug, Clone)]
pub struct SpecStepOutcome {
    /// Indices of newly committed frames in [`SpecCodecArState::codec_frames`].
    pub new_frame_indices: Vec<usize>,
    /// True when the session has reached a terminal state.
    pub done: bool,
}

impl Qwen3TtsMegakernel {
    /// Speculative-decoded codec AR. Same input/output contract as the plain
    /// [`Self::synthesize_codec_ar`] but with verification batched K+1 at a
    /// time. Requires:
    /// - The eager talker backend (rolling back KV on the GPU path is not yet
    ///   implemented; this errors clearly if called on a non-eager backend).
    /// - The non-fused CP path (`fused = None`, `cp = Some(_)`); the fused
    ///   path bypasses talker.codec_head_flat and would need its own loop.
    /// Begin a stepwise speculative codec-AR session. Drive it with
    /// [`Self::codec_ar_speculative_step`] to interleave AR with downstream
    /// work (partial decode, network send).
    pub fn begin_codec_ar_speculative<D: DraftModel>(
        &mut self,
        prefill_embeds: ArrayView2<f32>,
        talker_cfg: &TalkerConfig,
        cfg: &mut SpecConfig<'_, D>,
    ) -> Result<SpecCodecArState> {
        ensure!(
            self.talker_is_eager(),
            "speculative AR requires the eager talker backend"
        );
        ensure!(
            cfg.draft_len >= 1,
            "draft_len must be >= 1 (got {})",
            cfg.draft_len
        );

        let horizon = prefill_embeds.nrows().saturating_add(cfg.max_frames);
        let t_prefill = Instant::now();
        self.clear_prefill_cache_for_stepwise();
        self.talker_prefill_core(prefill_embeds)?;
        self.talker_warm_eager_decode_rope_if_eager()?;
        self.talker_prepare_decode_pipeline(horizon)?;
        let prefill_secs = t_prefill.elapsed().as_secs_f64();
        cfg.draft.reset();

        let hidden_dim = talker_cfg.hidden_size;
        let vocab_size = talker_cfg.vocab_size;
        let eos = talker_cfg.codec_eos_token_id;
        let k_draft = cfg.draft_len;
        let n_groups = talker_cfg.num_code_groups;

        let mut hidden = vec![0f32; hidden_dim];
        hidden.copy_from_slice(self.talker_hidden_row().as_slice().unwrap());
        let logits_scratch = vec![0f32; vocab_size];
        let codec_emb_scratch = vec![0f32; hidden_dim];
        let cp_input_scratch = vec![0f32; hidden_dim];
        let verify_in = Array2::<f32>::zeros((k_draft + 1, hidden_dim));
        let (codec_table, codec_table_hidden) = {
            let cp = self
                .cp_engine_mut()
                .context("speculative AR requires non-fused CP engine")?;
            let (table, h) = cp.talker_codec_flat();
            (table.to_vec(), h)
        };
        ensure!(
            codec_table_hidden == hidden_dim,
            "codec_table hidden {} != talker hidden {}",
            codec_table_hidden,
            hidden_dim
        );

        let early_exit_n = cfg.early_exit_layers;
        if let Some(ld) = cfg.learned_draft.as_deref_mut() {
            ld.reset_kv();
            ld.prefill_sync(prefill_embeds)?;
        }
        let draft_kv: Option<DraftKvCache> = if let Some(n) = early_exit_n {
            let max_n = self.talker_engine_ref().num_layers_eager()?;
            ensure!(
                n >= 1 && n <= max_n,
                "early_exit_layers {} out of [1, {}]",
                n,
                max_n
            );
            let mut kv = DraftKvCache::new(n);
            let prefill_rows = prefill_embeds.nrows();
            for r in 0..prefill_rows {
                let row = prefill_embeds.row(r);
                let row_slice = row.as_slice().expect("prefill row contiguous");
                let _ = self
                    .talker_engine_mut()
                    .early_exit_decode_step(row_slice, &mut kv, r)?;
            }
            Some(kv)
        } else {
            None
        };

        Ok(SpecCodecArState {
            hidden,
            logits_scratch,
            codec_emb_scratch,
            cp_input_scratch,
            verify_in,
            codec_table,
            past_g0: Vec::new(),
            codec_frames: Vec::new(),
            stats: SpecRunStats::default(),
            talker_secs: 0.0,
            cp_secs: 0.0,
            prefill_secs,
            max_frames: cfg.max_frames,
            draft_kv,
            draft_buf: vec![0f32; hidden_dim],
            draft_emb_buf: vec![0f32; hidden_dim],
            prior_codec_embs: Vec::new(),
            done: false,
            hidden_dim,
            vocab_size,
            eos,
            n_groups,
            k_draft,
        })
    }

    /// Advance a speculative AR session by one verify step. May commit 1..=K+1
    /// codec frames per call. Returns indices of newly committed frames.
    pub fn codec_ar_speculative_step<D: DraftModel>(
        &mut self,
        state: &mut SpecCodecArState,
        _talker_cfg: &TalkerConfig,
        cfg: &mut SpecConfig<'_, D>,
    ) -> Result<SpecStepOutcome> {
        if state.is_done() {
            state.done = true;
            return Ok(SpecStepOutcome {
                new_frame_indices: Vec::new(),
                done: true,
            });
        }

        let frame_start = state.codec_frames.len();
        let hidden_dim = state.hidden_dim;
        let vocab_size = state.vocab_size;
        let eos = state.eos;
        let k_draft = state.k_draft;
        let n_groups = state.n_groups;
        let early_exit_n = cfg.early_exit_layers;
        let use_learned = cfg.learned_draft.is_some();
        const PRIOR_EMB_CAP: usize = 256;

        // ---- 1. Sample g0(t) from current hidden ----
        let g0_t = self.sample_g0_from_hidden(
            &state.hidden,
            &state.past_g0,
            &mut state.logits_scratch,
            cfg.rep_penalty,
            vocab_size,
            eos,
        )?;
        if g0_t == eos {
            if state.codec_frames.len() >= cfg.min_frames {
                state.done = true;
            }
            return Ok(step_outcome(state, frame_start));
        }

        // CP for g0(t) → codec_emb(t) and full groups.
        let t_cp = Instant::now();
        let groups_t = self.cp_predict_groups_fill_emb(
            &state.hidden,
            g0_t,
            cfg.tts_pad_embed,
            &mut state.codec_emb_scratch,
        )?;
        state.cp_secs += t_cp.elapsed().as_secs_f64();
        state.past_g0.push(g0_t);
        state.codec_frames.push(groups_t);
        push_capped(
            &mut state.prior_codec_embs,
            state.codec_emb_scratch.clone(),
            PRIOR_EMB_CAP,
        );

        if state.codec_frames.len() >= cfg.max_frames {
            state.done = true;
            return Ok(step_outcome(state, frame_start));
        }

        // ---- Anti-cascade guard ----
        let in_cascade =
            cfg.cascade_guard_window > 0 && state.past_g0.len() >= cfg.cascade_guard_window && {
                let tail = &state.past_g0[state.past_g0.len() - cfg.cascade_guard_window..];
                tail.iter().all(|&x| x == g0_t)
            };
        if in_cascade {
            let t_step = Instant::now();
            self.talker_engine_mut().decode_hidden_into(
                ArrayView1::from(&state.codec_emb_scratch),
                &mut state.hidden,
            )?;
            state.talker_secs += t_step.elapsed().as_secs_f64();
            if let Some(kv) = state.draft_kv.as_mut() {
                let dim = self.talker_engine_ref().kv_dim_eager()?;
                let pos = kv.past_len(dim);
                let _ = self.talker_engine_mut().early_exit_decode_step(
                    &state.codec_emb_scratch,
                    kv,
                    pos,
                )?;
            }
            if let Some(ld) = cfg.learned_draft.as_deref_mut() {
                let pos = ld.past_len();
                let _ = ld.decode_step(&state.codec_emb_scratch, pos)?;
            }
            state.stats.record(SpecStepStats {
                drafted: 0,
                accepted: 0,
                used_free_token: false,
            });
            return Ok(step_outcome(state, frame_start));
        }

        self.fill_codec_logits(
            &state.hidden,
            &state.past_g0[..state.past_g0.len() - 1],
            &mut state.logits_scratch,
            cfg.rep_penalty,
        )?;
        cfg.draft.set_step_context(
            g0_t,
            &state.logits_scratch,
            &state.hidden,
            &state.codec_emb_scratch,
        );

        // ---- 2. Draft K (g0, codec_emb) pairs ----
        let (draft_pairs, draft_provides_inputs) = if use_learned {
            let pairs = self.learned_draft_propose(
                cfg.learned_draft
                    .as_deref_mut()
                    .context("learned_draft set but missing")?,
                &state.past_g0,
                &state.codec_emb_scratch,
                cfg.tts_pad_embed,
                cfg.rep_penalty,
                vocab_size,
                eos,
                k_draft,
                &mut state.draft_emb_buf,
                &mut state.logits_scratch,
                &mut state.cp_secs,
            )?;
            (pairs, true)
        } else if early_exit_n.is_some() {
            let kv = state
                .draft_kv
                .as_mut()
                .context("early_exit_layers set but draft_kv is None")?;
            let pairs = self.early_exit_propose(
                kv,
                &state.past_g0,
                &state.codec_emb_scratch,
                g0_t,
                cfg.tts_pad_embed,
                cfg.rep_penalty,
                vocab_size,
                eos,
                k_draft,
                &mut state.draft_buf,
                &mut state.draft_emb_buf,
                &mut state.logits_scratch,
                &mut state.cp_secs,
            )?;
            (pairs, true)
        } else {
            let pairs = cfg.draft.propose_inputs(
                &state.past_g0,
                &state.codec_emb_scratch,
                &state.prior_codec_embs,
                k_draft,
            )?;
            (pairs, cfg.draft.provides_own_inputs())
        };
        ensure!(
            draft_pairs.len() == k_draft,
            "draft returned {} pairs, expected {}",
            draft_pairs.len(),
            k_draft
        );
        let drafts: Vec<u32> = draft_pairs.iter().map(|(g, _)| *g).collect();

        // ---- 3. Build verifier batch [K+1, hidden] via SwapG0 ----
        // Row 0 = real codec_emb(t) at position N+t. Rows 1..K+1 = "what
        // codec_emb(t+i) would look like if the previous g0 had been
        // drafts[i-1]." We approximate by starting from codec_emb(t) and
        // swapping ONLY the g0 group-embedding component:
        //
        //   row_{i+1} = codec_emb(t) - group_embed_0[g0(t)] + group_embed_0[drafts[i]]
        //
        // The 15 CP-conditional groups (g1..g15) are unchanged. They're
        // strictly wrong for the drafted token, but they're in-distribution
        // (real codec_embs the model produced for SOME frame), so the
        // verifier doesn't see garbage. This breaks the uniformity
        // pathology where all K+1 inputs being identical biases the
        // verifier toward repeat tokens.
        //
        // The propose_inputs return is mostly ignored here — we keep only
        // the g0 tokens for the swap. A learned draft that overrides
        // propose_inputs with its own per-position codec_embs gets used
        // verbatim via the `pair.1.len() == hidden_dim` check (the if-
        // branch below).
        let row0_codec = state.codec_emb_scratch.clone();
        {
            let mut row0 = state.verify_in.row_mut(0);
            row0.as_slice_mut().unwrap().copy_from_slice(&row0_codec);
        }
        let g0_t_row =
            &state.codec_table[(g0_t as usize) * hidden_dim..((g0_t as usize) + 1) * hidden_dim];
        for (i, (draft_g0, draft_emb)) in draft_pairs.iter().enumerate() {
            let mut row = state.verify_in.row_mut(i + 1);
            let row_slice = row.as_slice_mut().unwrap();
            if draft_provides_inputs {
                ensure!(
                    draft_emb.len() == hidden_dim,
                    "draft codec_emb row {} len {} != hidden {}",
                    i,
                    draft_emb.len(),
                    hidden_dim
                );
                row_slice.copy_from_slice(draft_emb);
            } else {
                row_slice.copy_from_slice(&row0_codec);
                let dg = *draft_g0 as usize;
                let draft_row = &state.codec_table[dg * hidden_dim..(dg + 1) * hidden_dim];
                for j in 0..hidden_dim {
                    row_slice[j] = row_slice[j] - g0_t_row[j] + draft_row[j];
                }
            }
        }

        // ---- 4. Verifier batched forward. past_len += K+1. ----
        let t_verify = Instant::now();
        let hiddens = self.talker_decode_batched(state.verify_in.view())?;
        state.talker_secs += t_verify.elapsed().as_secs_f64();
        ensure!(
            hiddens.dim() == (k_draft + 1, hidden_dim),
            "verifier returned bad shape {:?}",
            hiddens.dim()
        );

        // ---- 5. Score verifier g0 for each row + walk acceptance. ----
        let mut effective_past = state.past_g0.clone();
        let mut n_accept = 0usize;
        for i in 0..k_draft {
            let row = hiddens.row(i);
            let row_slice = row.as_slice().expect("hiddens row contiguous");
            let g0_verifier = self.sample_g0_from_hidden(
                row_slice,
                &effective_past,
                &mut state.logits_scratch,
                cfg.rep_penalty,
                vocab_size,
                eos,
            )?;
            let accept_this = match cfg.policy {
                AcceptancePolicy::GreedyArgmax => g0_verifier == drafts[i],
            };
            if !accept_this {
                break;
            }
            effective_past.push(drafts[i]);
            n_accept += 1;
        }

        // ---- 6. Commit accepted drafts ----
        for i in 0..n_accept {
            let g0_i = drafts[i];
            state
                .cp_input_scratch
                .copy_from_slice(hiddens.row(i).as_slice().expect("hiddens row contiguous"));
            let t_cp = Instant::now();
            let groups_i = self.cp_predict_groups_fill_emb(
                &state.cp_input_scratch,
                g0_i,
                cfg.tts_pad_embed,
                &mut state.codec_emb_scratch,
            )?;
            state.cp_secs += t_cp.elapsed().as_secs_f64();
            ensure!(groups_i.len() == n_groups, "cp groups len mismatch");
            state.past_g0.push(g0_i);
            state.codec_frames.push(groups_i);
            push_capped(
                &mut state.prior_codec_embs,
                state.codec_emb_scratch.clone(),
                PRIOR_EMB_CAP,
            );
            if state.codec_frames.len() >= cfg.max_frames {
                state.done = true;
                break;
            }
        }

        // ---- 7. Update hidden for next iteration ----
        state
            .hidden
            .copy_from_slice(hiddens.row(n_accept).as_slice().unwrap());

        // ---- 8. Rollback unused verifier KV rows ----
        let rollback = k_draft - n_accept;
        if rollback > 0 {
            self.talker_rollback_kv(rollback)?;
        }
        if let Some(kv) = state.draft_kv.as_mut() {
            let dim = self.talker_engine_ref().kv_dim_eager()?;
            kv.rollback(rollback, dim);
        }
        if let Some(ld) = cfg.learned_draft.as_deref_mut() {
            ld.rollback_kv(rollback);
        }

        // ---- 9. Telemetry ----
        state.stats.record(SpecStepStats {
            drafted: k_draft,
            accepted: n_accept,
            used_free_token: false,
        });
        cfg.draft.on_commit(n_accept);

        Ok(step_outcome(state, frame_start))
    }

    pub fn synthesize_codec_ar_speculative<D: DraftModel>(
        &mut self,
        prefill_embeds: ArrayView2<f32>,
        talker_cfg: &TalkerConfig,
        mut cfg: SpecConfig<'_, D>,
    ) -> Result<SpecRunResult> {
        let t_start = Instant::now();
        let mut state = self.begin_codec_ar_speculative(prefill_embeds, talker_cfg, &mut cfg)?;
        while !state.is_done() {
            let outcome = self.codec_ar_speculative_step(&mut state, talker_cfg, &mut cfg)?;
            if outcome.done {
                break;
            }
        }
        let (codec_frames, stats, prefill_secs, talker_secs, cp_secs) = state.finish();
        Ok(SpecRunResult {
            codec_frames,
            stats,
            prefill_secs,
            talker_secs,
            cp_secs,
            total_secs: t_start.elapsed().as_secs_f64(),
        })
    }

    fn talker_warm_eager_decode_rope_if_eager(&mut self) -> Result<()> {
        if self.talker_engine_mut().is_eager() {
            self.talker_engine_mut().warm_eager_decode_rope()?;
        }
        Ok(())
    }

    fn talker_decode_batched(&mut self, embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        self.talker_engine_mut().decode_batched(embeds)
    }

    fn talker_rollback_kv(&mut self, n: usize) -> Result<()> {
        self.talker_engine_mut().rollback_kv(n)
    }

    fn talker_is_eager(&self) -> bool {
        self.talker_engine_ref().is_eager()
    }

    fn sample_g0_from_hidden(
        &self,
        hidden: &[f32],
        past_g0: &[u32],
        logits_scratch: &mut [f32],
        rep_penalty: f32,
        vocab_size: usize,
        eos: u32,
    ) -> Result<u32> {
        let (head, vocab, hdim) = self.talker_engine_ref().codec_head_flat();
        linear_logits_flat_into(hidden, head, vocab, hdim, logits_scratch)?;
        apply_repetition_penalty(logits_scratch, past_g0, rep_penalty);
        Ok(sample_greedy_talker_codec(logits_scratch, vocab_size, eos))
    }

    fn cp_predict_groups_fill_emb(
        &mut self,
        hidden: &[f32],
        g0: u32,
        pad: &[f32],
        codec_emb: &mut [f32],
    ) -> Result<Vec<u32>> {
        let cp = self
            .cp_engine_mut()
            .context("speculative AR requires non-fused CP engine")?;
        cp.predict_groups_fill_emb(hidden, g0, pad, codec_emb)
    }

    fn fill_codec_logits(
        &self,
        hidden: &[f32],
        past_g0: &[u32],
        logits_out: &mut [f32],
        rep_penalty: f32,
    ) -> Result<()> {
        let (head, vocab, hdim) = self.talker_engine_ref().codec_head_flat();
        linear_logits_flat_into(hidden, head, vocab, hdim, logits_out)?;
        apply_repetition_penalty(logits_out, past_g0, rep_penalty);
        Ok(())
    }

    /// Drafting variant that uses a standalone [`LearnedDraft`] sidecar
    /// rather than the talker's own first N layers. Mirrors
    /// [`Self::early_exit_propose`] but the K+1 forwards run through the
    /// learned draft's independent layer stack + KV cache.
    #[allow(clippy::too_many_arguments)]
    fn learned_draft_propose(
        &mut self,
        draft: &mut LearnedDraft,
        past_g0: &[u32],
        codec_emb_t: &[f32],
        pad: &[f32],
        rep_penalty: f32,
        vocab_size: usize,
        eos: u32,
        k: usize,
        draft_emb_buf: &mut [f32],
        logits_scratch: &mut [f32],
        cp_secs: &mut f64,
    ) -> Result<Vec<(u32, Vec<f32>)>> {
        // Step 0: process codec_emb(t) at the draft's current position.
        let pos0 = draft.past_len();
        let mut draft_hidden = draft.decode_step(codec_emb_t, pos0)?;
        let mut pairs: Vec<(u32, Vec<f32>)> = Vec::with_capacity(k);
        let mut effective_past: Vec<u32> = past_g0.to_vec();
        for _ in 0..k {
            // Sample next g0 from the draft's hidden via the verifier's codec_head.
            let (head, vocab, hdim) = self.talker_engine_ref().codec_head_flat();
            linear_logits_flat_into(&draft_hidden, head, vocab, hdim, logits_scratch)?;
            apply_repetition_penalty(logits_scratch, &effective_past, rep_penalty);
            let g_draft = sample_greedy_talker_codec(logits_scratch, vocab_size, eos);
            // Build codec_emb for the drafted g0 via the verifier's CP path.
            let t_cp = Instant::now();
            let cp = self
                .cp_engine_mut()
                .context("learned_draft_propose requires non-fused CP engine")?;
            let _ = cp.predict_groups_fill_emb(&draft_hidden, g_draft, pad, draft_emb_buf)?;
            *cp_secs += t_cp.elapsed().as_secs_f64();
            pairs.push((g_draft, draft_emb_buf.to_vec()));
            // Advance draft by feeding its proposed codec_emb at next position.
            let pos = draft.past_len();
            draft_hidden = draft.decode_step(draft_emb_buf, pos)?;
            effective_past.push(g_draft);
        }
        Ok(pairs)
    }

    /// Run early-exit drafting for `k` future g0 tokens, advancing the
    /// draft KV cache by `k + 1` rows (one for codec_emb(t) at position
    /// `past_len(verifier)`, then `k` for the drafted continuations).
    ///
    /// Returns `(g0, codec_emb)` pairs for each drafted future position.
    /// The codec_embs are produced via the full CP path so the verifier
    /// sees in-distribution inputs.
    #[allow(clippy::too_many_arguments)]
    fn early_exit_propose(
        &mut self,
        kv: &mut DraftKvCache,
        past_g0: &[u32],
        codec_emb_t: &[f32],
        g0_t: u32,
        pad: &[f32],
        rep_penalty: f32,
        vocab_size: usize,
        eos: u32,
        k: usize,
        draft_hidden_buf: &mut [f32],
        draft_emb_buf: &mut [f32],
        logits_scratch: &mut [f32],
        cp_secs: &mut f64,
    ) -> Result<Vec<(u32, Vec<f32>)>> {
        let dim = self.talker_engine_ref().kv_dim_eager()?;
        // Step 0: process codec_emb(t) at the current draft position. The
        // draft KV grows by 1 row to mirror the talker's just-sampled token.
        let pos0 = kv.past_len(dim);
        let h0 = self
            .talker_engine_mut()
            .early_exit_decode_step(codec_emb_t, kv, pos0)?;
        draft_hidden_buf.copy_from_slice(&h0);

        let mut pairs: Vec<(u32, Vec<f32>)> = Vec::with_capacity(k);
        // Effective past for repetition penalty: starts with g0_t already
        // committed.
        let mut effective_past: Vec<u32> = past_g0.to_vec();
        // past_g0 already includes g0_t when this is called; sample-time
        // history mirrors that.
        let _ = g0_t;
        for _ in 0..k {
            // Sample next g0 from the draft hidden via the talker's
            // codec head.
            let (head, vocab, hdim) = self.talker_engine_ref().codec_head_flat();
            linear_logits_flat_into(draft_hidden_buf, head, vocab, hdim, logits_scratch)?;
            apply_repetition_penalty(logits_scratch, &effective_past, rep_penalty);
            let g_draft = sample_greedy_talker_codec(logits_scratch, vocab_size, eos);
            // Build codec_emb for the drafted token via the full CP path.
            let t_cp = Instant::now();
            let cp = self
                .cp_engine_mut()
                .context("early_exit_propose requires non-fused CP engine")?;
            let _groups =
                cp.predict_groups_fill_emb(draft_hidden_buf, g_draft, pad, draft_emb_buf)?;
            *cp_secs += t_cp.elapsed().as_secs_f64();
            pairs.push((g_draft, draft_emb_buf.to_vec()));
            // Advance the draft KV by feeding the new codec_emb at the
            // next position.
            let pos = kv.past_len(dim);
            let h_next = self
                .talker_engine_mut()
                .early_exit_decode_step(draft_emb_buf, kv, pos)?;
            draft_hidden_buf.copy_from_slice(&h_next);
            effective_past.push(g_draft);
        }
        Ok(pairs)
    }
}

fn push_capped<T>(v: &mut Vec<T>, item: T, cap: usize) {
    if cap == 0 {
        return;
    }
    if v.len() == cap {
        v.remove(0);
    }
    v.push(item);
}

fn step_outcome(state: &SpecCodecArState, frame_start: usize) -> SpecStepOutcome {
    SpecStepOutcome {
        new_frame_indices: (frame_start..state.codec_frames.len()).collect(),
        done: state.is_done(),
    }
}
