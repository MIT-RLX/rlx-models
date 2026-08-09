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

//! [`Eagle3Speculator`] — adapts an EAGLE3 draft model into the
//! workspace's `rlx_runtime::spec_decode::Speculator` trait.
//!
//! ## What this scaffolds
//!
//! - Holds the [`Eagle3Config`], a [`D2tMap`], the loaded
//!   [`Eagle3DraftWeights`], and a handle to a
//!   [`VerifierHiddenSource`] that the speculator calls to obtain
//!   the verifier's last K hidden states.
//! - Implements `Speculator::propose` end-to-end *if* a compiled
//!   draft graph is provided. The draft graph builder lives in
//!   [`crate::draft`].
//! - `verify()` is left as a no-op — verification probabilities come
//!   from the verifier's own forward pass; the consumer wires the
//!   verifier into a separate `Speculator` impl (mirroring the
//!   MTP precedent in `crates/rlx-qwen35/src/spec.rs`).
//!
//! ## What is *not* yet wired
//!
//! - `propose()` returns an error until the draft graph from
//!   [`crate::draft`] is built. The function signature is final.
//!
//! ## Speculator round
//!
//! 1. `propose(context, n)` calls `verifier_hidden.read()` for the
//!    last `n + 1` aux hidden-state windows the verifier produced
//!    for `context`.
//! 2. For each step `i ∈ 0..n`, the draft graph runs once with
//!    `(h_aux_i, prev_token_id, kv_cache)` and returns
//!    `(draft_logits_i, kv_cache_next)`.
//! 3. The argmax over `draft_logits_i` is the proposed draft token.
//!    [`D2tMap::scatter_logits`] expands it to target-vocab space.
//! 4. Both draft KV cache and verifier KV cache are checkpointed
//!    before the round and restored if `commit()` rolls back —
//!    mirrors the GDN cache pattern in `Qwen35MtpDraft`.

use anyhow::Result;
use rlx_runtime::Device;
use rlx_runtime::spec_decode::{DraftProposal, Speculator, VerifyResult};

use crate::config::Eagle3Config;
use crate::d2t::D2tMap;
use crate::draft::{DraftGeom, DraftWeightRefs, Eagle3DraftReference};
use crate::hir_runner::{DraftKvCache, HirDraftRunner};
use crate::reference::softmax_in_place;
use crate::weights::Eagle3DraftWeights;

/// Source of verifier hidden states. Implemented by the verifier-side
/// runner (e.g. `rlx_gemma::runner::GemmaDecodeRunner`).
///
/// The expected return is `Vec<Vec<f32>>` of length
/// `aux_layer_ids.len()`, each row of length
/// `batch * 1 * target_hidden_size`. Order matches the layer-id
/// order passed to the verifier's
/// `with_aux_hidden_outputs(...)` call.
pub trait VerifierHiddenSource {
    /// Read the aux hidden states corresponding to the most recent
    /// verifier decode step. Returns one tensor per layer id, in
    /// `aux_layer_ids` order.
    fn aux_hidden_states(&self) -> Result<Vec<Vec<f32>>>;

    /// `target_hidden_size` — used to validate tensor shapes.
    fn target_hidden_size(&self) -> usize;

    /// Number of aux layer ids.
    fn num_aux_layers(&self) -> usize;
}

/// EAGLE3 draft speculator. Generic over the verifier hidden source
/// to keep the crate decoupled from any specific verifier.
pub struct Eagle3Speculator<H: VerifierHiddenSource> {
    cfg: Eagle3Config,
    d2t: D2tMap,
    weights: Eagle3DraftWeights,
    verifier_hidden: H,
    /// Last sampled draft token, used as next-step embedding input.
    /// Wired across rounds in `propose()`.
    prev_token: Option<u32>,
    /// Compiled HIR draft runner. When `Some`, `propose` uses it
    /// (10× faster than the scalar reference on Metal). When `None`,
    /// `propose` falls back to the pure-Rust scalar forward.
    hir: Option<HirDraftRunner>,
}

impl<H: VerifierHiddenSource> Eagle3Speculator<H> {
    pub fn new(cfg: Eagle3Config, weights: Eagle3DraftWeights, verifier_hidden: H) -> Result<Self> {
        let d2t = D2tMap::new(weights.d2t().to_vec(), cfg.target_vocab_size())?;
        // Aux-source / config sanity checks.
        if verifier_hidden.target_hidden_size() != cfg.target_hidden_size() {
            anyhow::bail!(
                "Eagle3Speculator: verifier target_hidden_size={} != config target_hidden_size={}",
                verifier_hidden.target_hidden_size(),
                cfg.target_hidden_size(),
            );
        }
        let aux_n = cfg
            .eagle_aux_hidden_state_layer_ids
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(3);
        if verifier_hidden.num_aux_layers() != aux_n {
            anyhow::bail!(
                "Eagle3Speculator: verifier exposes {} aux layers, config wants {}",
                verifier_hidden.num_aux_layers(),
                aux_n,
            );
        }
        Ok(Self {
            cfg,
            d2t,
            weights,
            verifier_hidden,
            prev_token: None,
            hir: None,
        })
    }

    /// Compile the HIR draft graph on `device` and use it from
    /// `propose()` instead of the scalar reference. `n_max` should
    /// equal `cfg.speculative_tokens` (the largest `n` you'll
    /// ever pass).
    pub fn with_hir_runner(mut self, device: Device, n_max: usize) -> Result<Self> {
        let geom: DraftGeom = DraftGeom::from_cfg(&self.cfg);
        let runner = HirDraftRunner::new(&self.weights, geom, n_max, device)?;
        self.hir = Some(runner);
        Ok(self)
    }

    /// True if `propose` will route through the compiled HIR graph.
    pub fn uses_hir(&self) -> bool {
        self.hir.is_some()
    }

    pub fn config(&self) -> &Eagle3Config {
        &self.cfg
    }

    pub fn d2t(&self) -> &D2tMap {
        &self.d2t
    }

    pub fn weights(&self) -> &Eagle3DraftWeights {
        &self.weights
    }

    pub fn verifier_hidden(&self) -> &H {
        &self.verifier_hidden
    }

    /// Drop the cached `prev_token` (call when the verifier rejects
    /// the round so the next propose() reads fresh context).
    pub fn reset_prev_token(&mut self) {
        self.prev_token = None;
    }
}

impl<H: VerifierHiddenSource> Eagle3Speculator<H> {
    /// Core propose — fallible variant. The trait method
    /// [`Speculator::propose`] just unwraps this.
    ///
    /// Routes through the compiled HIR runner if one was attached
    /// via `with_hir_runner`; otherwise falls back to the scalar
    /// reference. Both produce numerically identical greedy
    /// proposals (parity pinned in `tests/hir_parity.rs`).
    pub fn propose_inner(&mut self, context: &[u32], n: usize) -> Result<DraftProposal> {
        if n == 0 {
            return Ok(DraftProposal {
                tokens: Vec::new(),
                probs: Vec::new(),
            });
        }
        // Read the verifier's aux hidden states ONCE per round —
        // they're only consumed by the fc fusion on step 0.
        let aux = self.verifier_hidden.aux_hidden_states()?;
        anyhow::ensure!(
            !aux.is_empty(),
            "Eagle3Speculator: verifier returned 0 aux hidden states",
        );

        // First step's "prev token" is the last token of the
        // verifier context — or the cached prev_token if the
        // previous round committed (so commit-then-propose chains
        // see the just-accepted token).
        let prev_token0 = match self.prev_token {
            Some(t) => t,
            None => *context
                .last()
                .ok_or_else(|| anyhow::anyhow!("propose: context is empty"))?,
        };

        // The scalar reference is needed for `init_hidden` (the fc
        // fusion + optional input_norm) regardless of which backend
        // runs the per-step decoder. fc fusion is one matmul on
        // `[3*H]` → `[H]` — cheap enough that doing it on host is
        // fine; it's not currently in the HIR graph.
        let refs = DraftWeightRefs::from_weights(&self.weights, &self.cfg)?;
        let scalar_draft = Eagle3DraftReference::new(&self.cfg, refs);
        let mut hidden = scalar_draft.init_hidden(&aux);
        drop(scalar_draft);

        if let Some(hir) = &self.hir {
            anyhow::ensure!(
                n <= hir.n_max(),
                "n={n} exceeds HIR runner's n_max={}",
                hir.n_max(),
            );
        }

        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        let mut probs: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut prev_token = prev_token0;

        if let Some(runner) = self.hir.as_mut() {
            let mut cache = DraftKvCache::default();
            for step in 0..n {
                let (draft_logits, new_hidden) =
                    runner.step(step, prev_token, &hidden, &mut cache)?;
                let draft_id = crate::reference::argmax(&draft_logits);
                let target_id = self.d2t.map_token(draft_id);
                tokens.push(target_id);
                let mut row = self.d2t.scatter_logits(&draft_logits);
                softmax_in_place(&mut row);
                probs.push(row);
                hidden = new_hidden;
                prev_token = target_id;
            }
        } else {
            // Scalar reference fallback. Rebuild the draft because
            // we dropped it above to release the borrow on weights.
            let refs = DraftWeightRefs::from_weights(&self.weights, &self.cfg)?;
            let mut draft = Eagle3DraftReference::new(&self.cfg, refs);
            for _ in 0..n {
                let (draft_logits, new_hidden) = draft.step(&hidden, prev_token)?;
                let draft_id = crate::reference::argmax(&draft_logits);
                let target_id = self.d2t.map_token(draft_id);
                tokens.push(target_id);
                let mut row = self.d2t.scatter_logits(&draft_logits);
                softmax_in_place(&mut row);
                probs.push(row);
                hidden = new_hidden;
                prev_token = target_id;
            }
        }

        Ok(DraftProposal { tokens, probs })
    }
}

impl<H: VerifierHiddenSource> Speculator for Eagle3Speculator<H> {
    fn propose(&mut self, context: &[u32], n: usize) -> DraftProposal {
        self.propose_inner(context, n)
            .expect("Eagle3Speculator::propose failed; use propose_inner for Result")
    }

    fn verify(&mut self, _context: &[u32], _proposed: &[u32]) -> VerifyResult {
        // Verifier wires its own Speculator impl — see the MTP
        // pattern in `crates/rlx-qwen35/src/spec.rs`.
        VerifyResult { probs: Vec::new() }
    }

    fn commit(&mut self, _context: &[u32], accepted: &[u32]) {
        if let Some(&last) = accepted.last() {
            self.prev_token = Some(last);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Eagle3Config;

    /// A canned verifier hidden source used to drive shape / config
    /// validation tests. Returns `aux_layer_ids.len()` zero-filled
    /// tensors of length `target_hidden_size`.
    struct CannedHidden {
        target_hidden: usize,
        layers: usize,
    }
    impl VerifierHiddenSource for CannedHidden {
        fn aux_hidden_states(&self) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0; self.target_hidden]; self.layers])
        }
        fn target_hidden_size(&self) -> usize {
            self.target_hidden
        }
        fn num_aux_layers(&self) -> usize {
            self.layers
        }
    }

    fn tiny_cfg() -> Eagle3Config {
        // Mirror the RedHatAI layout but small enough for unit tests.
        // 3-layer aux extraction, 16-dim hidden, 8 draft vocab, 32
        // target vocab.
        let json = r#"{
            "draft_vocab_size": 8,
            "norm_before_residual": true,
            "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 4,
                "vocab_size": 32
            }
        }"#;
        Eagle3Config::from_bytes(json.as_bytes()).unwrap()
    }

    fn synth_weights() -> Eagle3DraftWeights {
        // Build a synthetic safetensors blob carrying just `d2t`.
        // In-memory `serialize` — no temp files, so tests stay
        // parallel-safe.
        use safetensors::serialize;
        use safetensors::tensor::{Dtype as StDtype, TensorView};
        use std::collections::HashMap;
        let d2t: Vec<u32> = vec![3, 5, 7, 11, 13, 17, 19, 23];
        let d2t_bytes: Vec<u8> = bytemuck::cast_slice(&d2t).to_vec();
        let d2t_view = TensorView::new(StDtype::U32, vec![8], &d2t_bytes).unwrap();
        let mut map: HashMap<&str, TensorView<'_>> = HashMap::new();
        map.insert("d2t", d2t_view);
        let bytes = serialize(&map, None).unwrap();
        Eagle3DraftWeights::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn rejects_target_hidden_mismatch() {
        let cfg = tiny_cfg();
        let weights = synth_weights();
        let bad = CannedHidden {
            target_hidden: 999,
            layers: 3,
        };
        let res = Eagle3Speculator::new(cfg, weights, bad);
        let err = match res {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("target_hidden_size"));
    }

    #[test]
    fn rejects_aux_layer_count_mismatch() {
        let cfg = tiny_cfg();
        let weights = synth_weights();
        let bad = CannedHidden {
            target_hidden: 16,
            layers: 1, // config wants 3
        };
        let res = Eagle3Speculator::new(cfg, weights, bad);
        let err = match res {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("aux layers"));
    }

    #[test]
    fn commit_caches_last_accepted_token() {
        let cfg = tiny_cfg();
        let weights = synth_weights();
        let canned = CannedHidden {
            target_hidden: 16,
            layers: 3,
        };
        let mut spec = Eagle3Speculator::new(cfg, weights, canned).unwrap();
        assert!(spec.prev_token.is_none());
        spec.commit(&[1, 2, 3], &[42]);
        assert_eq!(spec.prev_token, Some(42));
        spec.commit(&[1, 2, 3, 42], &[]); // no tokens — no update
        assert_eq!(spec.prev_token, Some(42));
        spec.reset_prev_token();
        assert!(spec.prev_token.is_none());
    }

    #[test]
    fn verify_returns_empty_probs_by_design() {
        let cfg = tiny_cfg();
        let weights = synth_weights();
        let canned = CannedHidden {
            target_hidden: 16,
            layers: 3,
        };
        let mut spec = Eagle3Speculator::new(cfg, weights, canned).unwrap();
        let v = spec.verify(&[1, 2], &[3, 4]);
        assert!(v.probs.is_empty());
    }
}
