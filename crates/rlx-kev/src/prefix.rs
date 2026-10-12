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

//! State-prefix caching: encode the state once, replay only the branches.
//!
//! In the row form every question's row is `state ++ branch_k`, so a request
//! with `Q` questions recomputes the state `Q` times, and a repeated state
//! recomputes it again on every request. With a 772-token state and 50-token
//! branches that is 94% waste.
//!
//! The fix is two graphs instead of one:
//!
//! ```text
//! state pass    batch 1, seq Ls        -> hidden + recurrent/KV state
//! continuation  batch 1, past Ls,      -> hidden for the branch positions
//!               seq n_new, bias mask
//! ```
//!
//! The state pass runs once per distinct state (cached, LRU); the
//! continuation runs once per question, resuming from the cached state. Both
//! are bucketed so a steady workload stops recompiling.
//!
//! # The trap this hit
//!
//! The prefill flow's `export_normed_hidden` returns the hidden **after**
//! `output_norm`. The decode flow's hidden output is the value **before** it
//! (`primary = logits.unwrap_or(h_lm)`, and the final norm lives on the
//! logits path inside `emit_qwen35_prefill_tail`). They are not the same
//! quantity, and the difference hides well: the RMS agreed to 1.763 against
//! an expected 1.759 — noise, by eye — while element 0 was -50 where it
//! should have been -6.3, and every probability came out ~0.27 wrong.
//!
//! `DecisionModel::apply_output_norm` closes that gap, and
//! `tests/continuation_equivalence.rs` pins it on raw token ids so a
//! regression cannot be blamed on the encoder or the readout.
//!
//! # Why this is opt-in
//!
//! Each compiled shape owns a full parameter arena. Caching therefore costs a
//! *second* arena — ~3.5 GB for Kev-0.8B in f32 — to save recomputation. That
//! is a good trade for long repeated states and a bad one for short states,
//! so it is off by default and gated on a minimum state length.
//!
//! # Why one row at a time
//!
//! The continuation graph is `rlx-qwen35`'s multi-token decode flow, which
//! mis-reshapes the Gated DeltaNet conv state at `batch > 1` with `seq > 1`
//! (`reshape [4,4,3,6144] -> [4,3,6144]` at build time — it panics rather
//! than miscomputing, at least). Questions are therefore replayed one row at
//! a time against the shared cached state. That still turns `Q x (Ls + br)`
//! token-forwards into `Ls + Q x br`, which is where nearly all of the win
//! is; batching the rows would only amortize weight reads.

use std::collections::VecDeque;

use rlx_qwen35::Qwen35DecodeCache;

/// How the prefix cache behaves.
#[derive(Debug, Clone, Copy)]
pub struct PrefixCacheConfig {
    /// Distinct states kept. `0` disables caching.
    pub capacity: usize,
    /// Below this many state tokens the split is not worth a second arena
    /// (and, for short states, not worth the extra dispatch either).
    pub min_state_tokens: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        // kev's defaults: 4 states, 384 tokens.
        Self {
            capacity: 4,
            min_state_tokens: 384,
        }
    }
}

/// One cached state: the tokens it was built from and the resulting trunk
/// state, plus the bucket the graph was compiled at.
pub struct PrefixEntry {
    pub state_ids: Vec<u32>,
    pub cache: Qwen35DecodeCache,
    /// Padded length the state graph ran at; `past_seq` for the continuation.
    pub bucket: usize,
}

/// An LRU of cached state prefixes.
pub struct PrefixCache {
    cfg: PrefixCacheConfig,
    /// Most-recently-used last.
    entries: VecDeque<PrefixEntry>,
    hits: usize,
    misses: usize,
}

impl PrefixCache {
    pub fn new(cfg: PrefixCacheConfig) -> Self {
        Self {
            cfg,
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn config(&self) -> PrefixCacheConfig {
        self.cfg
    }

    pub fn hits(&self) -> usize {
        self.hits
    }

    pub fn misses(&self) -> usize {
        self.misses
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether a state of this length is worth caching.
    pub fn eligible(&self, state_len: usize) -> bool {
        self.cfg.capacity > 0 && state_len >= self.cfg.min_state_tokens
    }

    /// Look up a state by its **exact** token vector.
    ///
    /// Exact rather than longest-prefix: a shorter shared prefix is not
    /// reusable here, because the branch rows resume at `Ls` and the trunk
    /// state is only valid for the whole state it was built from.
    pub fn get(&mut self, state_ids: &[u32]) -> Option<&PrefixEntry> {
        let idx = self.entries.iter().position(|e| e.state_ids == state_ids)?;
        // Move to the back (most recent).
        let e = self.entries.remove(idx).expect("index from position");
        self.entries.push_back(e);
        self.hits += 1;
        self.entries.back()
    }

    /// Record a freshly computed state, evicting the least-recently-used.
    pub fn insert(&mut self, entry: PrefixEntry) {
        self.misses += 1;
        if self.cfg.capacity == 0 {
            return;
        }
        if let Some(idx) = self
            .entries
            .iter()
            .position(|e| e.state_ids == entry.state_ids)
        {
            self.entries.remove(idx);
        }
        self.entries.push_back(entry);
        while self.entries.len() > self.cfg.capacity {
            self.entries.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Replicate a batch-1 cache across `batch` rows.
///
/// Unused by the current one-row-at-a-time replay, but the shape the batched
/// path would need; kept next to the cache so the two stay consistent.
pub fn broadcast(cache: &Qwen35DecodeCache, batch: usize) -> Qwen35DecodeCache {
    use rlx_qwen35::Qwen35LayerState;
    let rep = |v: &Vec<f32>| -> Vec<f32> {
        let mut out = Vec::with_capacity(v.len() * batch);
        for _ in 0..batch {
            out.extend_from_slice(v);
        }
        out
    };
    Qwen35DecodeCache {
        batch,
        past_seq: cache.past_seq,
        abs_pos: cache.abs_pos,
        prompt_lens: vec![cache.prompt_lens.first().copied().unwrap_or(cache.past_seq); batch],
        layers: cache
            .layers
            .iter()
            .map(|l| match l {
                Qwen35LayerState::Linear {
                    conv_state,
                    ssm_state,
                } => Qwen35LayerState::Linear {
                    conv_state: rep(conv_state),
                    ssm_state: rep(ssm_state),
                },
                Qwen35LayerState::FullAttn {
                    past_k,
                    past_v,
                    packed,
                } => Qwen35LayerState::FullAttn {
                    past_k: rep(past_k),
                    past_v: rep(past_v),
                    packed: packed.clone(),
                },
            })
            .collect(),
    }
}

/// Graph feeds naming the cached state, for the continuation graph.
///
/// Mirrors the input names the decode flow declares: `past_k_l{il}` /
/// `past_v_l{il}` for full-attention layers, `conv_state_l{il}` /
/// `ssm_state_l{il}` for Gated DeltaNet layers.
pub fn cache_feeds(cache: &Qwen35DecodeCache) -> Vec<(String, Vec<f32>)> {
    use rlx_qwen35::Qwen35LayerState;
    let mut out = Vec::with_capacity(cache.layers.len() * 2);
    for (il, layer) in cache.layers.iter().enumerate() {
        match layer {
            Qwen35LayerState::Linear {
                conv_state,
                ssm_state,
            } => {
                out.push((format!("conv_state_l{il}"), conv_state.clone()));
                out.push((format!("ssm_state_l{il}"), ssm_state.clone()));
            }
            Qwen35LayerState::FullAttn { past_k, past_v, .. } => {
                out.push((format!("past_k_l{il}"), past_k.clone()));
                out.push((format!("past_v_l{il}"), past_v.clone()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(ids: &[u32]) -> PrefixEntry {
        PrefixEntry {
            state_ids: ids.to_vec(),
            cache: Qwen35DecodeCache {
                batch: 1,
                past_seq: ids.len(),
                abs_pos: ids.len(),
                prompt_lens: vec![ids.len()],
                layers: Vec::new(),
            },
            bucket: ids.len(),
        }
    }

    #[test]
    fn lookup_is_exact_not_prefix() {
        let mut c = PrefixCache::new(PrefixCacheConfig {
            capacity: 4,
            min_state_tokens: 0,
        });
        c.insert(entry(&[1, 2, 3]));
        assert!(c.get(&[1, 2]).is_none(), "a shorter prefix must not hit");
        assert!(
            c.get(&[1, 2, 3, 4]).is_none(),
            "a longer state must not hit"
        );
        assert!(c.get(&[1, 2, 3]).is_some());
        assert_eq!(c.hits(), 1);
    }

    #[test]
    fn eviction_is_least_recently_used() {
        let mut c = PrefixCache::new(PrefixCacheConfig {
            capacity: 2,
            min_state_tokens: 0,
        });
        c.insert(entry(&[1]));
        c.insert(entry(&[2]));
        // Touch 1 so 2 becomes the least recent.
        assert!(c.get(&[1]).is_some());
        c.insert(entry(&[3]));
        assert_eq!(c.len(), 2);
        assert!(c.get(&[1]).is_some(), "recently used must survive");
        assert!(c.get(&[2]).is_none(), "least recently used must be evicted");
    }

    #[test]
    fn reinserting_the_same_state_does_not_duplicate() {
        let mut c = PrefixCache::new(PrefixCacheConfig {
            capacity: 4,
            min_state_tokens: 0,
        });
        c.insert(entry(&[7, 7]));
        c.insert(entry(&[7, 7]));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn eligibility_respects_the_floor_and_a_zero_capacity() {
        let c = PrefixCache::new(PrefixCacheConfig {
            capacity: 4,
            min_state_tokens: 384,
        });
        assert!(!c.eligible(383));
        assert!(c.eligible(384));
        let off = PrefixCache::new(PrefixCacheConfig {
            capacity: 0,
            min_state_tokens: 0,
        });
        assert!(!off.eligible(10_000), "capacity 0 disables caching");
    }
}
