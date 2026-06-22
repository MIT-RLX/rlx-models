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

//! Draft-to-target vocabulary mapping.
//!
//! The EAGLE3 draft model has its own smaller vocab (e.g. 32K for the
//! RedHatAI/Gemma 4 checkpoint) while the verifier uses the full
//! target vocab (262144 for Gemma 4).
//!
//! ## Semantics: d2t is an OFFSET, not an absolute id
//!
//! The on-disk `d2t` buffer stores the **delta** between the target
//! id and the draft id at each draft index. vLLM-speculators' source
//! does the mapping as:
//!
//! ```python
//! input_ids = input_ids + self.d2t[input_ids]
//! ```
//!
//! So `target_id = draft_id + d2t[draft_id]`. Earlier revisions of
//! this code treated `d2t[i]` as the absolute target id and were
//! wrong by exactly `draft_id` everywhere — caught by reading the
//! reference, not by any synthetic test, since both the synthetic
//! test data and the buggy implementation agreed on
//! `target_id == d2t[draft_id]`.
//!
//! The `t2d` buffer is the inverse mask used during draft training
//! (1 if the target token is in the draft vocab, else 0). We don't
//! use it at inference.

use anyhow::{Context, Result, bail};

/// Draft-vocab → target-vocab map. Backed by a flat `Vec<u32>` of
/// offsets such that `target_id = draft_id + d2t[draft_id]`.
#[derive(Debug, Clone)]
pub struct D2tMap {
    /// Per-draft-id offset. `target_id = draft_id + d2t[draft_id]`.
    /// Mirrors vLLM's `input_ids = input_ids + self.d2t[input_ids]`.
    d2t_offsets: Vec<u32>,
    target_vocab_size: usize,
}

impl D2tMap {
    /// Build directly from a draft-vocab-sized `u32` offset table.
    /// `offsets[i]` is the delta to add to draft id `i` to obtain
    /// the target id.
    pub fn new(offsets: Vec<u32>, target_vocab_size: usize) -> Result<Self> {
        if offsets.is_empty() {
            bail!("d2t: empty mapping");
        }
        if target_vocab_size == 0 {
            bail!("d2t: target_vocab_size must be > 0");
        }
        let target_u32 =
            u32::try_from(target_vocab_size).context("d2t: target_vocab_size must fit in u32")?;
        // Validate every draft id lands inside the target vocab.
        for (draft_id, &off) in offsets.iter().enumerate() {
            let draft_u32 = draft_id as u32;
            let target = match draft_u32.checked_add(off) {
                Some(t) => t,
                None => bail!("d2t: draft id {} + offset {} overflows u32", draft_id, off),
            };
            if target >= target_u32 {
                bail!(
                    "d2t: draft id {} + offset {} = target id {} >= \
                     target_vocab_size {}",
                    draft_id,
                    off,
                    target,
                    target_vocab_size,
                );
            }
        }
        Ok(Self {
            d2t_offsets: offsets,
            target_vocab_size,
        })
    }

    pub fn draft_vocab_size(&self) -> usize {
        self.d2t_offsets.len()
    }

    pub fn target_vocab_size(&self) -> usize {
        self.target_vocab_size
    }

    /// Raw offsets buffer (length = draft_vocab_size).
    pub fn offsets(&self) -> &[u32] {
        &self.d2t_offsets
    }

    /// Map a single draft-token id to its target id.
    /// `target_id = draft_id + d2t[draft_id]`.
    /// Panics if `draft_id >= draft_vocab_size` — caller controls
    /// the source of `draft_id` (its own sampler / argmax).
    pub fn map_token(&self, draft_id: u32) -> u32 {
        draft_id + self.d2t_offsets[draft_id as usize]
    }

    /// Scatter draft-vocab logits into a target-vocab vector.
    ///
    /// `draft_logits.len()` must equal `draft_vocab_size`. The output
    /// vector has length `target_vocab_size`; entries not covered by
    /// any draft id stay at `f32::NEG_INFINITY` so a downstream
    /// softmax assigns them zero probability.
    pub fn scatter_logits(&self, draft_logits: &[f32]) -> Vec<f32> {
        assert_eq!(
            draft_logits.len(),
            self.d2t_offsets.len(),
            "scatter_logits: expected {} draft logits, got {}",
            self.d2t_offsets.len(),
            draft_logits.len()
        );
        let mut out = vec![f32::NEG_INFINITY; self.target_vocab_size];
        for (draft_id, &logit) in draft_logits.iter().enumerate() {
            let target_id = (draft_id as u32 + self.d2t_offsets[draft_id]) as usize;
            // Aliasing across draft ids shouldn't happen for a real
            // (draft_id + offset) mapping (it's strictly injective
            // when each offset is unique per draft_id), but keep
            // max-on-collision for defensive correctness on
            // pathological inputs.
            if logit > out[target_id] {
                out[target_id] = logit;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `target_id = draft_id + d2t[draft_id]`. Offsets of 10, 11,
    /// 12, 13 map draft ids {0,1,2,3} to target ids {10, 12, 14, 16}.
    #[test]
    fn map_token_applies_offset() {
        let m = D2tMap::new(vec![10, 11, 12, 13], 100).unwrap();
        assert_eq!(m.draft_vocab_size(), 4);
        assert_eq!(m.target_vocab_size(), 100);
        assert_eq!(m.map_token(0), 10); // 0 + 10
        assert_eq!(m.map_token(1), 12); // 1 + 11
        assert_eq!(m.map_token(2), 14); // 2 + 12
        assert_eq!(m.map_token(3), 16); // 3 + 13
    }

    #[test]
    fn map_token_offset_zero_is_identity() {
        let m = D2tMap::new(vec![0, 0, 0], 8).unwrap();
        assert_eq!(m.map_token(0), 0);
        assert_eq!(m.map_token(1), 1);
        assert_eq!(m.map_token(2), 2);
    }

    #[test]
    fn rejects_empty_or_overflowing() {
        assert!(D2tMap::new(vec![], 100).is_err());
        // draft id 1 + offset 99 = 100, target_vocab_size = 100 ⇒ OOR
        assert!(D2tMap::new(vec![0, 99], 100).is_err());
        assert!(D2tMap::new(vec![0], 0).is_err());
    }

    #[test]
    fn scatter_fills_unmapped_with_neg_inf() {
        // Draft vocab = 3, target vocab = 6.
        // Offsets {1, 2, 3}: draft 0→1, 1→3, 2→5.
        let m = D2tMap::new(vec![1, 2, 3], 6).unwrap();
        let draft = vec![0.1, 0.2, 0.3];
        let scattered = m.scatter_logits(&draft);
        assert_eq!(scattered.len(), 6);
        assert_eq!(scattered[0], f32::NEG_INFINITY);
        assert!((scattered[1] - 0.1).abs() < 1e-9);
        assert_eq!(scattered[2], f32::NEG_INFINITY);
        assert!((scattered[3] - 0.2).abs() < 1e-9);
        assert_eq!(scattered[4], f32::NEG_INFINITY);
        assert!((scattered[5] - 0.3).abs() < 1e-9);
    }

    #[test]
    fn scatter_keeps_max_on_collision() {
        // Force two draft ids to land on the same target id:
        // draft 0 (offset 2) → target 2
        // draft 1 (offset 1) → target 2
        // draft 2 (offset 3) → target 5
        let m = D2tMap::new(vec![2, 1, 3], 6).unwrap();
        let draft = vec![-1.0, 7.5, 0.0];
        let scattered = m.scatter_logits(&draft);
        assert!((scattered[2] - 7.5).abs() < 1e-9);
        assert!((scattered[5] - 0.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "expected 3 draft logits")]
    fn scatter_panics_on_wrong_len() {
        let m = D2tMap::new(vec![1, 2, 3], 6).unwrap();
        let _ = m.scatter_logits(&[0.0; 2]);
    }
}
