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

//! Self-speculative decoding wrapper around [`Qwen3Generator`].
//!
//! Implements [`rlx_runtime::spec_decode::Speculator`] so a single
//! Qwen3 instance can be plugged in as either the draft or the target
//! side of [`rlx_runtime::spec_decode::SpecDecoder`].
//!
//! **Important caveat:** when the same model is used as both draft and
//! target, the proposal and verification distributions are identical
//! by construction → every proposed token accepts with probability 1
//! → the spec-decoding round always emits `n` tokens, identical to
//! plain greedy decoding. This delivers no wall-clock speedup; it is
//! structurally a no-op that proves the pipeline works.
//!
//! Real self-speculative speedup requires asymmetry between draft and
//! target — either:
//!   1. Two different Qwen3 sizes (e.g. 0.6B draft + 32B target), or
//!   2. A draft head (MTP / Eagle-style) inside the same model that
//!      runs faster than the full forward.
//!
//! The MTP-head case is the path matching the user's stated intent;
//! when those heads are added to the Qwen3 builder, the corresponding
//! `Qwen3MtpSpeculator` slots in as the `draft` argument to
//! `SpecDecoder::new(draft, target, n, seed)`. The plumbing — context
//! reset, RoPE-offset bookkeeping, KV-cache lifecycle, prob
//! collection — is all this module.

use crate::config::Qwen3Config;
use crate::generator::Qwen3Generator;
use crate::sampling::softmax_logits;
use anyhow::Result;
use rlx_runtime::spec_decode::{DraftProposal, Speculator, VerifyResult};

/// `Speculator` adapter wrapping a `Qwen3Generator`. Each
/// `propose`/`verify` call resets the wrapped generator's internal
/// state and re-seeds the KV cache from `context` — necessary because
/// the `Speculator` trait gives no acceptance feedback, so the
/// speculator cannot incrementally advance its cache to track the
/// SpecDecoder's chosen tokens.
///
/// **Cost:** each call pays one full prefill over `context`, then
/// `n - 1` decode steps. A spec-decode round (one `propose` + one
/// `verify`) is therefore 2 × prefill + 2(n-1) decodes — strictly
/// slower than plain cached decoding. Acceptable for correctness
/// validation and as a structural foundation; the optimization (cache
/// sharing, batched verify) is a separate slice.
pub struct Qwen3Speculator {
    inner: Qwen3Generator,
}

impl Qwen3Speculator {
    pub fn new(gn: Qwen3Generator) -> Self {
        Self { inner: gn }
    }

    pub fn config(&self) -> &Qwen3Config {
        self.inner.config()
    }

    fn argmax(xs: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in xs.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }

    fn propose_inner(&mut self, context: &[u32], n: usize) -> Result<DraftProposal> {
        if n == 0 {
            return Ok(DraftProposal {
                tokens: vec![],
                probs: vec![],
            });
        }
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        let mut probs: Vec<Vec<f32>> = Vec::with_capacity(n);

        let mut logits = self.inner.prefill_get_last_logits(context)?;
        for i in 0..n {
            let p = softmax_logits(&logits);
            let tok = Self::argmax(&logits);
            tokens.push(tok);
            probs.push(p);
            if i + 1 < n {
                logits = self.inner.decode_get_logits(tok)?;
            }
        }
        Ok(DraftProposal { tokens, probs })
    }

    fn verify_inner(&mut self, context: &[u32], proposed: &[u32]) -> Result<VerifyResult> {
        let n = proposed.len();
        if n == 0 {
            return Ok(VerifyResult { probs: vec![] });
        }
        let mut probs: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut logits = self.inner.prefill_get_last_logits(context)?;
        for i in 0..n {
            probs.push(softmax_logits(&logits));
            if i + 1 < n {
                logits = self.inner.decode_get_logits(proposed[i])?;
            }
        }
        Ok(VerifyResult { probs })
    }
}

impl Speculator for Qwen3Speculator {
    fn propose(&mut self, context: &[u32], n: usize) -> DraftProposal {
        self.propose_inner(context, n)
            .expect("Qwen3Speculator::propose failed")
    }

    fn verify(&mut self, context: &[u32], proposed: &[u32]) -> VerifyResult {
        self.verify_inner(context, proposed)
            .expect("Qwen3Speculator::verify failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Qwen3Config;
    use crate::sampling::SampleOpts;
    use rlx_core::weight_map::WeightMap;
    use rlx_runtime::Device;
    use rlx_runtime::spec_decode::SpecDecoder;
    use std::collections::HashMap;

    fn tiny_cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 16,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            qk_norm: true,
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    /// Deterministic weight pattern — same as in generator.rs tests so
    /// numerical results compose across modules.
    fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
        let dh = cfg.head_dim;
        let pat = |n: usize, salt: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
                    (x as f32 / (1u32 << 24) as f32) - 0.5
                })
                .collect()
        };
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        t.insert(
            "model.embed_tokens.weight".into(),
            (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
        );
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            t.insert(
                format!("{lp}.input_layernorm.weight"),
                (pat(h, 100 + i as u32), vec![h]),
            );
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (pat(h, 200 + i as u32), vec![h]),
            );
            t.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                (pat(q_dim * h, 300 + i as u32), vec![q_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                (pat(kv_dim * h, 400 + i as u32), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                (pat(kv_dim * h, 500 + i as u32), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                (pat(h * q_dim, 600 + i as u32), vec![h, q_dim]),
            );
            t.insert(
                format!("{lp}.self_attn.q_norm.weight"),
                (pat(dh, 700 + i as u32), vec![dh]),
            );
            t.insert(
                format!("{lp}.self_attn.k_norm.weight"),
                (pat(dh, 800 + i as u32), vec![dh]),
            );
            t.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                (pat(int_dim * h, 900 + i as u32), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (pat(int_dim * h, 1000 + i as u32), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (pat(h * int_dim, 1100 + i as u32), vec![h, int_dim]),
            );
        }
        t.insert("model.norm.weight".into(), (pat(h, 2000), vec![h]));
        t.insert(
            "lm_head.weight".into(),
            (pat(cfg.vocab_size * h, 3000), vec![cfg.vocab_size, h]),
        );
        WeightMap::from_tensors(t)
    }

    fn make_speculator(cfg: &Qwen3Config) -> Qwen3Speculator {
        let mut wm = synthetic_weights(cfg);
        let gn = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        Qwen3Speculator::new(gn)
    }

    /// Self-spec with identical draft and target: every proposed token
    /// must accept (q == p elementwise → q/p == 1 → r < 1 always).
    /// Output should match plain greedy cached decoding.
    #[test]
    fn self_spec_matches_plain_greedy() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let rounds = 2;
        let n = 3;

        // Reference: plain cached greedy decoding.
        let mut wm = synthetic_weights(&cfg);
        let mut gn_ref = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn_ref.prefill(&prompt);
        let ref_tokens = gn_ref
            .generate_cached(rounds * n, SampleOpts::greedy())
            .unwrap();

        // SpecDecoder with two Qwen3Speculators on the same weights.
        let draft = make_speculator(&cfg);
        let target = make_speculator(&cfg);
        let mut dec = SpecDecoder::new(draft, target, n, /*seed*/ 0xC0FFEE);

        let mut context: Vec<u32> = prompt.clone();
        let mut spec_tokens: Vec<u32> = Vec::with_capacity(rounds * n);
        for _ in 0..rounds {
            let new_tokens = dec.step(&context);
            // With identical distributions, every round must emit
            // exactly `n` tokens (no rejection → no `corrected`).
            assert_eq!(
                new_tokens.len(),
                n,
                "self-spec with identical draft/target must emit n tokens/round"
            );
            context.extend_from_slice(&new_tokens);
            spec_tokens.extend_from_slice(&new_tokens);
        }

        // Spec-decoded sequence must equal plain greedy. (Tolerance: if
        // softmax+sample-from rounding makes the corrected path pick a
        // different token vs greedy argmax, this fails — but in the
        // all-accept case, no corrected token is ever drawn.)
        assert_eq!(
            spec_tokens, ref_tokens,
            "self-spec output diverged from plain greedy — pipeline bug"
        );
    }

    #[test]
    fn propose_returns_n_tokens_with_valid_probs() {
        let cfg = tiny_cfg();
        let mut spec = make_speculator(&cfg);
        let ctx = vec![1u32, 2, 3];
        let n = 4;
        let prop = spec.propose(&ctx, n);
        assert_eq!(prop.tokens.len(), n);
        assert_eq!(prop.probs.len(), n);
        for (i, row) in prop.probs.iter().enumerate() {
            assert_eq!(row.len(), cfg.vocab_size, "row {i}");
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {i} sum {sum}");
            // Greedy: propose's chosen token at i must be argmax of row i.
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            assert_eq!(prop.tokens[i], argmax, "row {i} not argmax");
        }
    }

    #[test]
    fn verify_returns_matching_n_with_valid_probs() {
        let cfg = tiny_cfg();
        let mut spec = make_speculator(&cfg);
        let ctx = vec![1u32, 2, 3];
        let proposed = vec![5u32, 7, 2];
        let v = spec.verify(&ctx, &proposed);
        assert_eq!(v.probs.len(), proposed.len());
        for (i, row) in v.probs.iter().enumerate() {
            assert_eq!(row.len(), cfg.vocab_size);
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {i} sum {sum}");
        }
    }
}
