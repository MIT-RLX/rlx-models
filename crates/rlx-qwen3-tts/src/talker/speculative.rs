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

//! Talker speculative decoding — `cfg(feature = "speculative-decode")`.
//!
//! # Overview
//!
//! Per-step talker AR decode is the dominant cost in Qwen3-TTS synthesis.
//! Speculative decoding amortises that cost by:
//!
//! 1. A small **draft** model proposes `K` future talker tokens cheaply.
//! 2. The big talker **verifies** those `K` proposals in a single batched
//!    forward pass over `K + 1` token positions (one for the just-sampled
//!    current token + `K` for the drafted future).
//! 3. We commit the just-sampled token + the longest prefix of drafts where
//!    the verifier's argmax matches the draft (under [`AcceptancePolicy`]).
//! 4. The verifier's hidden at row `n_accept` becomes the *next* step's
//!    `state.hidden`. On mismatch, we roll the big talker's KV cache back to
//!    `past_len_before_step + 1 + n_accept` (i.e. drop the `K - n_accept`
//!    unused verifier rows).
//!
//! On a long accepted run, one big-talker forward commits `1 + n_accept`
//! tokens at roughly the AMX batched-N=K+1 cost — which is `~K+1` matvecs'
//! worth of flops but well under `K+1` matvecs of wall-clock thanks to AMX.
//!
//! # First cut (this module, v1)
//!
//! - [`TrivialDraft`] — predict-same-as-last-frame g0. Calibration baseline;
//!   acceptance is bursty and content-dependent but non-zero on stretches of
//!   sustained vowels/silence. Replaces cleanly with a learned draft later.
//! - Batched verify is delegated to `TalkerEngine::forward_batched_decode`
//!   (added alongside this module, also gated).
//! - KV rollback is delegated to `TalkerEagerModel::rollback_kv` (added
//!   alongside this module, also gated).
//! - Acceptance is **greedy argmax-match** in v1. Sampling-policy-aware
//!   acceptance (multinomial draw equality, top-k overlap) is a v2 extension.
//!
//! # Not in scope (v1)
//!
//! - CP (code-predictor) speculation — talker frame is `(g0, g1..g15)`; CP
//!   tokens g1..g15 are still produced one-at-a-time by the CP head from
//!   talker hidden state, unchanged.
//! - Learned draft model loading / training — first cut ships [`TrivialDraft`].
//! - Tree-structured speculation (Medusa/EAGLE-style multi-branch). v1 is
//!   linear: one draft chain of length `K`.

use anyhow::Result;

/// How many tokens a draft proposes per verification step.
///
/// `K=4` is a reasonable starting point: AMX matvecs at batch-N=5 are
/// ~3-4x faster per row than 5 separate N=1 matvecs on our talker shape,
/// so even ~30% acceptance is net-positive end-to-end. Higher `K` widens
/// the verify batch (slightly cheaper per accepted token) but raises the
/// rollback waste on misses; tune per draft.
pub const DEFAULT_DRAFT_LEN: usize = 4;

/// Acceptance policy for matching a drafted token against the big talker's
/// verifier output.
///
/// The talker uses sampled decoding (`temp=0.9, top_k=50` in the shipped
/// configs), so strict greedy-argmax acceptance is an under-approximation —
/// it'll only accept draft tokens that happen to be the big talker's argmax,
/// which is a fraction of the legitimately "the big talker would have
/// produced this" set.
///
/// `GreedyArgmax` is what v1 ships; richer policies arrive when we have
/// the sampling RNG plumbed through the verify call.
#[derive(Debug, Clone, Copy)]
pub enum AcceptancePolicy {
    /// Accept iff `draft_token == argmax(verifier_logits[i])`. Deterministic;
    /// safe baseline; under-accepts vs. sampling.
    GreedyArgmax,
}

impl Default for AcceptancePolicy {
    fn default() -> Self {
        Self::GreedyArgmax
    }
}

/// Source of draft tokens. Implementations propose `k` future g0 codec
/// tokens given the recent generation history.
///
/// Draft cost should be a small fraction of one big-talker step — otherwise
/// the verifier can't outrun the draft and speculation loses. For
/// [`TrivialDraft`] cost is ~zero; for a learned 4-6-layer Qwen3-shaped
/// draft, cost is ~`(draft_layers / talker_layers)` of one talker step.
///
/// # Per-position verifier inputs
///
/// In the v1 acceptance loop, the verifier processes `K + 1` input rows at
/// positions `[t, t+1, ..., t+K]` and produces `K + 1` hidden states. The
/// *correct* input at position `t + i` is `codec_emb(t + i) = CP(true_hidden(t+i), g0(t+i))`,
/// but `true_hidden(t+i)` is what the verifier itself is producing — so
/// some approximation is required.
///
/// The default approximation (used by [`TrivialDraft`]) is "copy
/// `codec_emb(t)` to every row." That's exact for sustained vowels and
/// silences but biases the verifier into producing K+1 near-identical
/// hidden states, which in turn over-accepts repeat-token drafts.
///
/// Drafts that can produce per-position input embeddings should override
/// [`Self::propose_inputs`] to supply them; the megakernel passes those
/// rows into the verifier batch instead of repeating `codec_emb(t)`.
pub trait DraftModel {
    /// Propose `k` g0 codec tokens.
    ///
    /// `history` is the committed talker g0 token sequence (oldest first).
    /// Length of returned vec is `k` — the trait promises a fixed-size
    /// proposal so the verifier batch shape is known up front.
    fn propose(&mut self, history: &[u32], k: usize) -> Result<Vec<u32>>;

    /// Propose `k` (g0_token, codec_emb) pairs.
    ///
    /// `codec_emb_t` is `codec_emb` of the just-committed `g0(t)` —
    /// suitable as a per-position input when the draft has nothing
    /// better. `prior_codec_embs` is the most recent committed
    /// `codec_embs` (oldest first, length up to ~256 typical).
    ///
    /// **Returns**: `Vec<(g0_token, codec_emb_row)>` of length `k`.
    /// Row `i` is used at verifier batch position `t + i + 1` (after the
    /// `codec_emb_t` row at position `t`).
    ///
    /// Default implementation delegates to [`Self::propose`] and uses
    /// `codec_emb_t` for every row — the trivial approximation that
    /// biases toward repetition.
    fn propose_inputs(
        &mut self,
        history: &[u32],
        codec_emb_t: &[f32],
        _prior_codec_embs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<(u32, Vec<f32>)>> {
        let drafts = self.propose(history, k)?;
        Ok(drafts
            .into_iter()
            .map(|g| (g, codec_emb_t.to_vec()))
            .collect())
    }

    /// Hand the draft the per-step context it might need: the just-sampled
    /// `g0(t)`, the lm_head logits used to sample it (so a draft can pick
    /// top-K from them), the current `hidden(t)`, and `codec_emb(t)`.
    ///
    /// Called by the megakernel **before each** `propose_inputs` call.
    /// Stateless drafts (e.g. [`TrivialDraft`]) ignore this; [`TopKDraft`]
    /// captures `logits` here for use in [`Self::propose`].
    fn set_step_context(
        &mut self,
        _g0_t: u32,
        _logits: &[f32],
        _hidden_t: &[f32],
        _codec_emb_t: &[f32],
    ) {
    }

    /// `true` iff this draft's [`Self::propose_inputs`] returns codec_embs
    /// that should be fed to the verifier *verbatim*. When `false` (default),
    /// the megakernel ignores the codec_emb in each pair and instead
    /// synthesises verifier inputs via cheap **SwapG0 substitution**:
    ///
    ///   `verifier_in[i+1] = codec_emb(t) - group_embed_0[g0(t)] + group_embed_0[drafts[i]]`
    ///
    /// SwapG0 is the right default for any non-learned draft — it gives
    /// varied, in-distribution verifier inputs at zero compute cost.
    /// Learned drafts that produce real per-position codec_embs (e.g. a
    /// tiny Qwen3-shaped sidecar) should override this to return `true`.
    fn provides_own_inputs(&self) -> bool {
        false
    }

    /// Called after the verifier commits `n_accepted` of the previous
    /// proposal. Lets stateful drafts trim their own KV cache to match.
    /// Default is a no-op (stateless drafts).
    fn on_commit(&mut self, _n_accepted: usize) {}

    /// Reset draft state at utterance boundary.
    fn reset(&mut self) {}
}

/// Top-K draft: proposes the K most-likely g0 tokens from the verifier's
/// own `lm_head(hidden(t))` logits.
///
/// **Combined with SwapG0 input synthesis**, this is the strongest cheap
/// non-learned heuristic. Slot 0 (`drafts[0] = top-1 = g0(t)`) gets
/// accepted on sustained-vowel / repeat-token frames. Slots 1..K
/// (`top-2..top-K`) are best-effort — they're predictions for the *current*
/// position's alternates, not for future positions, so they only happen to
/// match when the verifier's prediction at row `i` (with input
/// `SwapG0(codec_emb(t), g0(t), drafts[i])`) coincides with `drafts[i+1]`.
/// That's rare for non-sustained speech, but the slot-0 accept-rate alone
/// already moves throughput in the right direction.
///
/// The draft is **stateless w.r.t. logits**: the megakernel passes them in
/// via the trait's [`DraftModel::set_current_logits`] hook before each
/// `propose` call. If the megakernel hasn't set them yet (utterance start
/// or after `reset`), `propose` falls back to repeating the history's
/// last g0 — same as [`TrivialDraft`].
#[derive(Debug, Default, Clone)]
pub struct TopKDraft {
    /// Most recent logits handed in by the megakernel.
    last_logits: Option<Vec<f32>>,
}

impl DraftModel for TopKDraft {
    fn propose(&mut self, history: &[u32], k: usize) -> Result<Vec<u32>> {
        if let Some(logits) = self.last_logits.as_ref() {
            Ok(top_k_indices(logits, k))
        } else {
            let last = history.last().copied().unwrap_or(0);
            Ok(vec![last; k])
        }
    }

    fn set_step_context(
        &mut self,
        _g0_t: u32,
        logits: &[f32],
        _hidden_t: &[f32],
        _codec_emb_t: &[f32],
    ) {
        match &mut self.last_logits {
            Some(buf) => {
                if buf.len() != logits.len() {
                    buf.resize(logits.len(), 0.0);
                }
                buf.copy_from_slice(logits);
            }
            None => self.last_logits = Some(logits.to_vec()),
        }
    }

    fn reset(&mut self) {
        self.last_logits = None;
    }
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<u32> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let k = k.min(pairs.len());
    pairs.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(k);
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().map(|(i, _)| i as u32).collect()
}

/// Trivial draft: predict-same-as-last-frame g0.
///
/// Implements [`DraftModel::propose`] as "the last committed g0 token,
/// repeated `k` times." Acceptance is bursty and content-dependent (high on
/// sustained vowels / silence, zero on rapid phoneme transitions) but this
/// is the calibration baseline — it costs nothing and lower-bounds the
/// speculation framework's overhead.
///
/// Once a real (tiny Qwen3-shaped, e.g. 4-layer / 256-dim) draft is trained,
/// it slots in behind the same [`DraftModel`] trait and the acceptance loop
/// is unchanged.
#[derive(Debug, Default, Clone)]
pub struct TrivialDraft;

impl DraftModel for TrivialDraft {
    fn propose(&mut self, history: &[u32], k: usize) -> Result<Vec<u32>> {
        let last = history.last().copied().unwrap_or(0);
        Ok(vec![last; k])
    }
}

/// Shifted-history draft: proposes the last `K` committed g0 tokens as
/// the next `K` (oldest-of-window first), and supplies the matching
/// committed `codec_embs` as per-position verifier inputs.
///
/// The premise: speech locally repeats — the previous `K` frames are a
/// decent (varied) prior on the next `K`. Where the trivial draft biases
/// toward sustained tokens, this draft biases toward *cycle* tokens —
/// no better in raw accuracy but it breaks the verifier-input-uniformity
/// pathology, so the verifier produces varied hiddens at each row and
/// stops over-accepting repeats.
///
/// Like [`TrivialDraft`], this is a calibration baseline rather than a
/// production-quality draft — it's how we measure the *upper bound* of
/// what varied-input speculation can deliver without training.
#[derive(Debug, Default, Clone)]
pub struct ShiftedHistoryDraft;

impl DraftModel for ShiftedHistoryDraft {
    fn propose(&mut self, history: &[u32], k: usize) -> Result<Vec<u32>> {
        let n = history.len();
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            // Window of the last `k` history tokens, oldest first.
            // If history is shorter than `k`, pad with the oldest available.
            let idx = if n >= k {
                n - k + i
            } else {
                i.min(n.saturating_sub(1))
            };
            out.push(history.get(idx).copied().unwrap_or(0));
        }
        Ok(out)
    }

    fn propose_inputs(
        &mut self,
        history: &[u32],
        codec_emb_t: &[f32],
        prior_codec_embs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<(u32, Vec<f32>)>> {
        let drafts = self.propose(history, k)?;
        let n = prior_codec_embs.len();
        let out = drafts
            .into_iter()
            .enumerate()
            .map(|(i, g)| {
                let emb = if n >= k {
                    prior_codec_embs[n - k + i].clone()
                } else if n > 0 {
                    prior_codec_embs[i.min(n - 1)].clone()
                } else {
                    codec_emb_t.to_vec()
                };
                (g, emb)
            })
            .collect();
        Ok(out)
    }
}

/// Per-step stats — useful for tuning `K`, picking acceptance policies, and
/// reporting in benches.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStepStats {
    /// Number of drafted tokens this step (`= k`).
    pub drafted: usize,
    /// Number of drafted tokens the verifier accepted before the first
    /// mismatch. In `[0, drafted]`.
    pub accepted: usize,
    /// Reserved — set to `false` in the v1 (no-free-token) formulation.
    /// Kept on the struct so a future v2 (sampling-policy-aware acceptance
    /// with a "free" verifier-corrected token at the mismatch position) can
    /// plug in without breaking the telemetry shape.
    pub used_free_token: bool,
}

impl SpecStepStats {
    /// Total tokens committed this step. In the v1 formulation each spec
    /// step always commits `1 + accepted` tokens (the sequentially-sampled
    /// g0(t) plus each accepted draft). The verifier's hidden at row
    /// `accepted` carries forward as the next iter's `state.hidden` rather
    /// than being consumed as a free commit.
    pub fn committed(&self) -> usize {
        1 + self.accepted + if self.used_free_token { 1 } else { 0 }
    }
}

/// Running stats across an utterance.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecRunStats {
    pub steps: usize,
    pub total_drafted: usize,
    pub total_accepted: usize,
    pub total_committed: usize,
}

impl SpecRunStats {
    pub fn record(&mut self, step: SpecStepStats) {
        self.steps += 1;
        self.total_drafted += step.drafted;
        self.total_accepted += step.accepted;
        self.total_committed += step.committed();
    }

    /// Acceptance rate — drafted tokens accepted, in `[0, 1]`. Useful for
    /// "is this draft worth the verify-batch overhead?" decisions.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted == 0 {
            0.0
        } else {
            (self.total_accepted as f32) / (self.total_drafted as f32)
        }
    }

    /// Mean committed tokens per big-talker forward pass. `1.0` means
    /// speculation paid for itself exactly (no speedup, no slowdown);
    /// `>1.0` is real wall-clock win.
    pub fn tokens_per_verify(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            (self.total_committed as f32) / (self.steps as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_draft_repeats_last_token() {
        let mut d = TrivialDraft;
        let p = d.propose(&[7, 42, 9], 4).unwrap();
        assert_eq!(p, vec![9, 9, 9, 9]);
    }

    #[test]
    fn trivial_draft_empty_history_yields_zero() {
        let mut d = TrivialDraft;
        let p = d.propose(&[], 3).unwrap();
        assert_eq!(p, vec![0, 0, 0]);
    }

    #[test]
    fn run_stats_acceptance_rate() {
        let mut run = SpecRunStats::default();
        run.record(SpecStepStats {
            drafted: 4,
            accepted: 2,
            used_free_token: false,
        });
        run.record(SpecStepStats {
            drafted: 4,
            accepted: 4,
            used_free_token: false,
        });
        assert_eq!(run.total_drafted, 8);
        assert_eq!(run.total_accepted, 6);
        // step 1: 1 + 2 = 3 committed. step 2: 1 + 4 = 5 committed. total = 8.
        assert_eq!(run.total_committed, 8);
        assert!((run.acceptance_rate() - 0.75).abs() < 1e-6);
        // 8 tokens over 2 verify batches = 4 tokens-per-verify.
        assert!((run.tokens_per_verify() - 4.0).abs() < 1e-6);
    }
}
