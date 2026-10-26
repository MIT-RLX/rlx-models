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

//! Beam search for the `PDecTranslatorBlock` stage.
//!
//! The search is generic over [`NmtDecoder`], so it is exercised by unit tests
//! against a scripted decoder and will drive the NMT model unchanged once
//! [`crate::netfile`] can load one.
//!
//! # What is and is not pinned down
//!
//! These come straight from the shipped config and are implemented exactly:
//! beam width, `nbest`, the length budget ([`crate::pdec::PDecParams::length_budget`]),
//! length normalisation (`norm-costs`), suppressed tokens, and the
//! `finished_score` stopping rule.
//!
//! `rs-beam`, `lm-weight` and `veto-factor` are carried through and applied by
//! explicitly named policies, but the OS's exact formulas are **not** confirmed
//! — nothing in the config or the framework strings defines them. Both default
//! to inert ([`PruningPolicy::Disabled`], zero bias) so the search cannot
//! silently diverge from the reference in a way that looks like a bug in the
//! model. Once a reference decode is available to diff against, pin the policy
//! down here and flip the default.

use crate::pdec::PDecParams;
use anyhow::{Result, ensure};
use rlx_search::{BeamConfig, BeamSearch, Constraint, FinishRule, LiveAtEnd, Scores};
use std::collections::BTreeSet;

/// The model side of beam search.
///
/// `prefix` always begins with the target control tokens, so an implementation
/// that caches state can key on it directly.
pub trait NmtDecoder {
    /// Number of target vocabulary entries.
    fn vocab_size(&self) -> usize;

    /// Id that terminates a hypothesis.
    fn eos_id(&self) -> u32;

    /// Natural-log probabilities for the next token given `prefix`.
    /// Must return exactly [`NmtDecoder::vocab_size`] values.
    fn log_probs(&mut self, prefix: &[u32]) -> Result<Vec<f32>>;

    /// Decoded text of `tokens`, if the decoder can produce it.
    ///
    /// Only needed for [`SearchOptions::no_repeat_char_ngram`]: some
    /// repetitions are invisible at the token level, because a rare word gets
    /// spelled out of generic fragments and the same letters reappear under a
    /// different tiling. Returning `None` disables that check.
    fn surface(&self, _tokens: &[u32]) -> Option<String> {
        None
    }
}

/// How live hypotheses are pruned each step, beyond keeping the top `beam`.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PruningPolicy {
    /// Keep the top `beam` and nothing else. Default: the OS's `rs-beam`
    /// semantics are unverified, and guessing them would corrupt output in a
    /// way that is hard to attribute.
    #[default]
    Disabled,
    /// Additionally drop hypotheses scoring more than `factor × |best|` below
    /// the best live hypothesis. Provided so `rs-beam` can be wired up and
    /// A/B-tested once a reference decode exists; not enabled by default.
    RelativeToBest { factor: f64 },
}

/// One decoded hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    /// Generated ids, excluding the priming control tokens and the EOS.
    pub tokens: Vec<u32>,
    /// Sum of log probabilities.
    pub score: f64,
    /// `score` divided by token count when `norm-costs` is set, else `score`.
    pub normalized_score: f64,
}

/// Tunables that are not part of the config's own vocabulary.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub pruning: PruningPolicy,
    /// Token ids removed from consideration entirely — the resolved
    /// `shortlist-suppress-tokens`.
    pub suppressed: BTreeSet<u32>,
    /// Refuse to emit a target n-gram this hypothesis already contains.
    ///
    /// **Off by default (0), because the OS's decoder does not do this.** Its
    /// config sets `veto-factor: 0` and names no n-gram constraint, and the
    /// shipped framework loops in exactly the way this would prevent — it
    /// renders `Avocado platypus they are my followers` as
    /// `Sie sind meine Anhänger Anhänger Anhänger Anhänger Anhänger`. Turning
    /// this on produces *better* output but no longer reproduces the OS, which
    /// is the thing this port is for. Set it when you want quality over
    /// fidelity; leave it at 0 for parity work.
    ///
    /// A value of 3 is the usual choice. 2 is aggressive enough to block
    /// legitimate repeats (`très très`).
    pub no_repeat_ngram: usize,
    /// Refuse to emit a run of this many characters the hypothesis already
    /// contains. 0 disables it.
    ///
    /// Catches what [`SearchOptions::no_repeat_ngram`] cannot: a word absent
    /// from the vocabulary is spelled from generic fragments, so a stutter like
    /// `ornithornithaque` for `ornithorynque` repeats *letters* without
    /// repeating any token n-gram. Needs [`NmtDecoder::surface`].
    ///
    /// Short values reject ordinary language — common function words share long
    /// substrings — so this wants measuring, not guessing.
    pub no_repeat_char_ngram: usize,
    /// Exponent in `score / len^alpha`, overriding `norm-costs`.
    ///
    /// The config's `norm-costs` is a boolean, which is the two endpoints of
    /// this: true is `alpha = 1`, false is `alpha = 0`. Both were measured, and
    /// both are worse than they could be — 1.0 promotes a longer malformed
    /// hypothesis over a shorter clean one, 0.0 loses five matches against
    /// the OS. `rlx-nllb` spells the same idea as a tunable exponent
    /// (`GenerateConfig::length_penalty`), so this follows its convention and
    /// lets the middle be searched.
    ///
    /// `None` keeps the config's own boolean.
    pub length_penalty: Option<f64>,
}

/// Runs beam search.
///
/// `priming` is the target-side control token ids (see
/// [`crate::pdec::PDecParams::target_tokens`]); they seed every hypothesis and
/// are stripped from the results. `source_len` sets the length budget.
///
/// The engine is [`rlx_search::BeamSearch`]: each hypothesis proposes its top
/// `beam`, every EOS candidate finishes, and the search stops once
/// `finished_score` hypotheses are complete and no live one can overtake them.
pub fn search(
    decoder: &mut dyn NmtDecoder,
    params: &PDecParams,
    priming: &[u32],
    source_len: usize,
    options: &SearchOptions,
) -> Result<Vec<Hypothesis>> {
    let vocab = decoder.vocab_size();
    ensure!(vocab > 0, "decoder reports an empty vocabulary");
    let beam = params.beam.max(1);
    let want = params.nbest.max(1);
    let alpha = options
        .length_penalty
        .unwrap_or(if params.norm_costs { 1.0 } else { 0.0 });

    let mut cfg = BeamConfig::new(beam, params.length_budget(source_len))
        .stop([decoder.eos_id()])
        .length_penalty(alpha);
    // Top `beam` continuations of each hypothesis is enough: no hypothesis
    // outside its own top `beam` can enter the global top `beam` once every
    // parent contributes that many.
    cfg.proposals = beam;
    cfg.scores = Scores::LogProbs;
    cfg.finish = FinishRule::Any;
    // `finished_score` stops once this many hypotheses are complete.
    cfg.pool = params.stop_mode_finished_score_beam.max(want);
    cfg.stop_when_unbeatable = true;
    // Budget exhausted with hypotheses still running: keep them, since the OS
    // reports a truncated translation rather than nothing (`maxTokensReached`).
    cfg.live_at_end = LiveAtEnd::IfNoneFinished;
    cfg.nbest = want;
    cfg.suppress = options.suppressed.iter().copied().collect();
    cfg.no_repeat_ngram = options.no_repeat_ngram;
    cfg.prune_relative = match options.pruning {
        PruningPolicy::Disabled => None,
        PruningPolicy::RelativeToBest { factor } => Some(factor),
    };

    let checked = |lp: Vec<f32>| -> Result<Vec<f32>> {
        ensure!(
            lp.len() == vocab,
            "decoder returned {} log-probs for a vocabulary of {vocab}",
            lp.len()
        );
        Ok(lp)
    };
    let mut rows = vec![checked(decoder.log_probs(priming)?)?];
    let mut bs = BeamSearch::new(cfg, priming);
    while !bs.is_done() {
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let veto = &mut CharRunVeto {
            decoder: &*decoder,
            n: options.no_repeat_char_ngram,
        };
        let Some(step) = bs.advance(&refs, veto) else {
            break;
        };
        rows = (0..step.len())
            .map(|i| checked(decoder.log_probs(step.sequence(i))?))
            .collect::<Result<_>>()?;
    }
    Ok(bs
        .finish()
        .into_iter()
        .map(|h| Hypothesis {
            tokens: h.tokens,
            score: h.score,
            normalized_score: h.normalized,
        })
        .collect())
}

/// [`SearchOptions::no_repeat_char_ngram`]: needs the decoded text, so it runs
/// per proposal rather than as a vocabulary mask.
struct CharRunVeto<'a> {
    decoder: &'a dyn NmtDecoder,
    n: usize,
}

impl Constraint for CharRunVeto<'_> {
    fn reject(&mut self, sequence: &[u32], token: u32) -> bool {
        if self.n == 0 {
            return false;
        }
        let mut next = sequence.to_vec();
        next.push(token);
        self.decoder
            .surface(&next)
            .is_some_and(|t| repeats_char_run(&t, self.n))
    }
}

/// Does `text` end with a run of `n` characters that appears earlier in it?
///
/// Only the *final* run is tested, because every earlier one was already
/// rejected when it was the final one — so this stays O(len) per candidate
/// rather than rescanning the whole hypothesis.
fn repeats_char_run(text: &str, n: usize) -> bool {
    let c: Vec<char> = text.chars().collect();
    if n == 0 || c.len() <= n {
        return false;
    }
    let tail = &c[c.len() - n..];
    c[..c.len() - 1].windows(n).any(|w| w == tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decoder whose distribution depends only on the length of the prefix, so
    /// expected output can be reasoned about exactly.
    struct Scripted {
        /// `steps[i]` are the log-probs emitted at generation step `i`.
        steps: Vec<Vec<f32>>,
        eos: u32,
        priming: usize,
    }

    impl NmtDecoder for Scripted {
        fn vocab_size(&self) -> usize {
            self.steps[0].len()
        }
        fn eos_id(&self) -> u32 {
            self.eos
        }
        fn log_probs(&mut self, prefix: &[u32]) -> Result<Vec<f32>> {
            let step = prefix.len() - self.priming;
            Ok(self
                .steps
                .get(step)
                .cloned()
                .unwrap_or_else(|| self.steps.last().cloned().expect("non-empty script")))
        }
    }

    fn params(beam: usize, nbest: usize) -> PDecParams {
        PDecParams {
            beam,
            nbest,
            max_seq_length: 8,
            max_seq_length_floor: 8,
            max_seq_length_relative: 2.0,
            stop_mode_finished_score_beam: 1,
            ..PDecParams::default()
        }
    }

    #[test]
    fn greedy_path_is_found_and_priming_is_stripped() {
        // vocab 3, id 2 = EOS. Step 0 favours token 1, step 1 favours EOS.
        let mut d = Scripted {
            steps: vec![vec![-2.0, -0.1, -3.0], vec![-2.0, -2.0, -0.1]],
            eos: 2,
            priming: 2,
        };
        let out = search(
            &mut d,
            &params(1, 1),
            &[100, 101],
            4,
            &SearchOptions::default(),
        )
        .expect("search runs");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tokens, vec![1], "priming and EOS must not appear");
    }

    #[test]
    fn beam_beats_greedy_when_the_first_step_is_a_trap() {
        // Step 0: token 0 slightly better than token 1.
        // Step 1: continuing from anything, EOS costs -0.1.
        // Give token 1 a much better second step by making step 1 depend on
        // length only — so instead we make step 0 nearly tied and check that a
        // width-2 beam keeps both, returning 2 distinct nbest results.
        let mut d = Scripted {
            steps: vec![vec![-0.10, -0.11, -9.0], vec![-9.0, -9.0, -0.1]],
            eos: 2,
            priming: 0,
        };
        let out =
            search(&mut d, &params(2, 2), &[], 4, &SearchOptions::default()).expect("search runs");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].tokens, vec![0]);
        assert_eq!(out[1].tokens, vec![1]);
        assert!(out[0].score > out[1].score);
    }

    #[test]
    fn suppressed_tokens_are_never_emitted() {
        // Token 1 is the most likely at step 0 but is suppressed.
        let mut d = Scripted {
            steps: vec![vec![-2.0, -0.1, -9.0], vec![-9.0, -9.0, -0.1]],
            eos: 2,
            priming: 0,
        };
        let opts = SearchOptions {
            suppressed: [1u32].into_iter().collect(),
            ..SearchOptions::default()
        };
        let out = search(&mut d, &params(2, 1), &[], 4, &opts).expect("search runs");
        assert!(!out[0].tokens.contains(&1));
        assert_eq!(out[0].tokens, vec![0]);
    }

    /// The search as it was before it moved onto `rlx_search`, verbatim.
    mod previous {
        use super::super::{Hypothesis, NmtDecoder, PruningPolicy, SearchOptions};
        use crate::pdec::PDecParams;
        use anyhow::{Result, ensure};

        #[derive(Clone)]
        struct Live {
            /// Control tokens followed by generated ids.
            prefix: Vec<u32>,
            score: f64,
            generated: usize,
        }

        /// Runs beam search.
        ///
        /// `priming` is the target-side control token ids (see
        /// [`crate::pdec::PDecParams::target_tokens`]); they seed every hypothesis and
        /// are stripped from the results. `source_len` sets the length budget.
        pub fn old_search(
            decoder: &mut dyn NmtDecoder,
            params: &PDecParams,
            priming: &[u32],
            source_len: usize,
            options: &SearchOptions,
        ) -> Result<Vec<Hypothesis>> {
            let vocab = decoder.vocab_size();
            ensure!(vocab > 0, "decoder reports an empty vocabulary");
            let eos = decoder.eos_id();
            let budget = params.length_budget(source_len);
            let beam = params.beam.max(1);
            let want = params.nbest.max(1);
            // `finished_score` stops once this many hypotheses are complete.
            let finished_target = params.stop_mode_finished_score_beam.max(want);

            let mut live = vec![Live {
                prefix: priming.to_vec(),
                score: 0.0,
                generated: 0,
            }];
            let mut finished: Vec<Hypothesis> = Vec::new();

            for _ in 0..budget {
                if live.is_empty() {
                    break;
                }

                let mut candidates: Vec<Live> = Vec::with_capacity(live.len() * beam);
                for hyp in &live {
                    let lp = decoder.log_probs(&hyp.prefix)?;
                    ensure!(
                        lp.len() == vocab,
                        "decoder returned {} log-probs for a vocabulary of {vocab}",
                        lp.len()
                    );
                    // Top `beam` continuations of this hypothesis is enough: no
                    // hypothesis outside its own top `beam` can enter the global top
                    // `beam` once every parent contributes that many.
                    let mut idx: Vec<u32> = (0..vocab as u32)
                        .filter(|i| !options.suppressed.contains(i))
                        .collect();
                    idx.sort_unstable_by(|a, b| {
                        lp[*b as usize]
                            .partial_cmp(&lp[*a as usize])
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.cmp(b))
                    });
                    for &tok in idx.iter().take(beam) {
                        let s = lp[tok as usize];
                        if !s.is_finite() {
                            continue;
                        }
                        if repeats_ngram(&hyp.prefix, tok, options.no_repeat_ngram) {
                            continue;
                        }
                        if options.no_repeat_char_ngram > 0 {
                            let mut next = hyp.prefix.clone();
                            next.push(tok);
                            if decoder
                                .surface(&next)
                                .is_some_and(|t| repeats_char_run(&t, options.no_repeat_char_ngram))
                            {
                                continue;
                            }
                        }
                        let mut prefix = hyp.prefix.clone();
                        prefix.push(tok);
                        candidates.push(Live {
                            prefix,
                            score: hyp.score + s as f64,
                            generated: hyp.generated + 1,
                        });
                    }
                }

                if candidates.is_empty() {
                    break;
                }

                // Rank, split off completions, keep the top `beam` live.
                candidates.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut next: Vec<Live> = Vec::with_capacity(beam);
                for cand in candidates {
                    let last = *cand.prefix.last().expect("candidate has a token");
                    if last == eos {
                        let tokens: Vec<u32> =
                            cand.prefix[priming.len()..cand.prefix.len() - 1].to_vec();
                        finished.push(finish(
                            tokens,
                            cand.score,
                            params.norm_costs,
                            options.length_penalty,
                        ));
                    } else if next.len() < beam {
                        next.push(cand);
                    }
                }

                if let PruningPolicy::RelativeToBest { factor } = options.pruning
                    && let Some(best) = next.first().map(|h| h.score)
                {
                    let cutoff = best - factor * best.abs();
                    next.retain(|h| h.score >= cutoff);
                }

                live = next;

                // `finished_score`: stop when enough hypotheses are complete and no
                // live one can still overtake the worst kept completion. Scores are
                // log-probabilities, so a live hypothesis only ever decreases.
                if finished.len() >= finished_target {
                    finished.sort_by(|a, b| {
                        b.normalized_score
                            .partial_cmp(&a.normalized_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let worst_kept = finished[finished_target - 1].score;
                    if live.iter().all(|h| h.score <= worst_kept) {
                        break;
                    }
                }
            }

            // Budget exhausted with hypotheses still running: keep them, since the OS
            // reports a truncated translation rather than nothing (`maxTokensReached`).
            if finished.is_empty() {
                for hyp in live {
                    let tokens = hyp.prefix[priming.len()..].to_vec();
                    finished.push(finish(
                        tokens,
                        hyp.score,
                        params.norm_costs,
                        options.length_penalty,
                    ));
                }
            }

            finished.sort_by(|a, b| {
                b.normalized_score
                    .partial_cmp(&a.normalized_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            finished.truncate(want);
            Ok(finished)
        }

        /// Would appending `tok` repeat an `n`-gram the prefix already contains?
        ///
        /// Subword loops are what this catches: `ornithorynque` decoded as
        /// `ornitho`+`rnith`+`aque` repeats a bigram mid-word, and length
        /// normalization then *rewards* the longer broken hypothesis — measured, the
        /// malformed output scored -5.142 raw against -4.333 for the clean one, yet
        /// won on normalized score.
        fn repeats_ngram(prefix: &[u32], tok: u32, n: usize) -> bool {
            if n == 0 || prefix.len() + 1 < n {
                return false;
            }
            let tail = &prefix[prefix.len() + 1 - n..];
            // The candidate n-gram is the last n-1 emitted tokens plus `tok`.
            prefix
                .windows(n)
                .any(|w| w[..n - 1] == *tail && w[n - 1] == tok)
        }

        /// Does `text` end with a run of `n` characters that appears earlier in it?
        ///
        /// Only the *final* run is tested, because every earlier one was already
        /// rejected when it was the final one — so this stays O(len) per candidate
        /// rather than rescanning the whole hypothesis.
        fn repeats_char_run(text: &str, n: usize) -> bool {
            let c: Vec<char> = text.chars().collect();
            if n == 0 || c.len() <= n {
                return false;
            }
            let tail = &c[c.len() - n..];
            c[..c.len() - 1].windows(n).any(|w| w == tail)
        }

        fn finish(
            tokens: Vec<u32>,
            score: f64,
            norm_costs: bool,
            length_penalty: Option<f64>,
        ) -> Hypothesis {
            let alpha = length_penalty.unwrap_or(if norm_costs { 1.0 } else { 0.0 });
            let normalized_score = if alpha != 0.0 && !tokens.is_empty() {
                score / (tokens.len() as f64).powf(alpha)
            } else {
                score
            };
            Hypothesis {
                tokens,
                score,
                normalized_score,
            }
        }
    }

    /// Log-probs that depend on the whole prefix, so hypotheses differ.
    struct Hashed {
        vocab: usize,
        eos: u32,
        seed: u64,
    }

    impl NmtDecoder for Hashed {
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn eos_id(&self) -> u32 {
            self.eos
        }
        fn log_probs(&mut self, prefix: &[u32]) -> Result<Vec<f32>> {
            let mut h = 0xcbf2_9ce4_8422_2325u64 ^ self.seed;
            for &t in prefix {
                h = (h ^ t as u64).wrapping_mul(0x0100_0000_01b3);
            }
            let mut row: Vec<f32> = (0..self.vocab)
                .map(|i| {
                    let mut x = h ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    x ^= x >> 31;
                    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    (x >> 40) as f32 / (1u64 << 24) as f32 * 6.0
                })
                .collect();
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            row.iter_mut().for_each(|v| *v -= lse);
            Ok(row)
        }
        fn surface(&self, tokens: &[u32]) -> Option<String> {
            Some(
                tokens
                    .iter()
                    .map(|&t| (b'a' + (t % 4) as u8) as char)
                    .collect(),
            )
        }
    }

    #[test]
    fn matches_the_previous_search() {
        for seed in 0..150u64 {
            for (beam, nbest, norm, lp, finished_beam) in [
                (1, 1, true, None, 1),
                (3, 2, true, None, 3),
                (4, 3, false, None, 2),
                (3, 3, true, Some(0.6), 4),
            ] {
                let mut p = params(beam, nbest);
                p.norm_costs = norm;
                p.stop_mode_finished_score_beam = finished_beam;
                p.max_seq_length = 6;
                p.max_seq_length_floor = 6;
                let variants = [
                    SearchOptions {
                        length_penalty: lp,
                        ..SearchOptions::default()
                    },
                    SearchOptions {
                        suppressed: [3u32, 5].into_iter().collect(),
                        pruning: PruningPolicy::RelativeToBest { factor: 0.8 },
                        no_repeat_char_ngram: 3,
                        length_penalty: lp,
                        ..SearchOptions::default()
                    },
                ];
                for opts in &variants {
                    let mk = || Hashed {
                        vocab: 7,
                        eos: 2,
                        seed,
                    };
                    let want = previous::old_search(&mut mk(), &p, &[6, 6], 4, opts).unwrap();
                    let got = search(&mut mk(), &p, &[6, 6], 4, opts).unwrap();
                    assert_eq!(got.len(), want.len(), "seed {seed} beam {beam}");
                    for (g, w) in got.iter().zip(&want) {
                        assert_eq!(g.tokens, w.tokens, "seed {seed} beam {beam}");
                        assert!((g.score - w.score).abs() < 1e-9);
                        assert!((g.normalized_score - w.normalized_score).abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn budget_exhaustion_still_returns_a_truncated_hypothesis() {
        // EOS is never attractive, so nothing ever finishes.
        let mut d = Scripted {
            steps: vec![vec![-0.1, -2.0, -50.0]],
            eos: 2,
            priming: 0,
        };
        let mut p = params(1, 1);
        p.max_seq_length = 3;
        p.max_seq_length_floor = 3;
        let out = search(&mut d, &p, &[], 1, &SearchOptions::default()).expect("search runs");
        assert_eq!(
            out[0].tokens.len(),
            3,
            "should hit the budget and report it"
        );
    }

    #[test]
    fn relative_pruning_drops_far_behind_branches() {
        // Step 0: token 0 at -0.1, token 1 far behind at -5.0. Step 1: EOS.
        let script = || Scripted {
            steps: vec![vec![-0.1, -5.0, -9.0], vec![-9.0, -9.0, -0.1]],
            eos: 2,
            priming: 0,
        };
        let starts_with_one = |out: &[Hypothesis]| out.iter().any(|h| h.tokens.first() == Some(&1));

        // Unpruned, a width-2 beam explores the weak branch and returns it.
        let baseline = search(
            &mut script(),
            &params(2, 2),
            &[],
            4,
            &SearchOptions::default(),
        )
        .expect("search runs");
        assert!(
            starts_with_one(&baseline),
            "baseline should keep the weak branch"
        );

        // Pruned, the weak branch is cut and never appears in any result.
        let pruned = search(
            &mut script(),
            &params(2, 2),
            &[],
            4,
            &SearchOptions {
                pruning: PruningPolicy::RelativeToBest { factor: 0.5 },
                ..SearchOptions::default()
            },
        )
        .expect("search runs");
        assert!(
            !starts_with_one(&pruned),
            "pruning must cut the weak branch"
        );
        assert_eq!(pruned[0].tokens, vec![0]);
    }

    #[test]
    fn mismatched_vocab_size_is_an_error() {
        struct Bad;
        impl NmtDecoder for Bad {
            fn vocab_size(&self) -> usize {
                4
            }
            fn eos_id(&self) -> u32 {
                0
            }
            fn log_probs(&mut self, _: &[u32]) -> Result<Vec<f32>> {
                Ok(vec![-1.0; 3])
            }
        }
        let err = search(&mut Bad, &params(1, 1), &[], 1, &SearchOptions::default())
            .expect_err("must reject a short distribution");
        assert!(err.to_string().contains("log-probs"), "{err}");
    }
}
