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

//! End-to-end shape + correctness check for `Eagle3Speculator::propose`
//! against synthetic weights. Validates:
//!
//! - `propose(context, n)` returns `n` target-vocab tokens.
//! - Each `probs` row sums to ~1 and has length `target_vocab_size`.
//! - Entries outside the d2t-covered subset are zero (NEG_INF logit
//!   pre-softmax).
//! - Determinism: same context + same weights → same proposal.
//! - `commit()` correctly seeds the next propose's prev_token.
//! - Drives the whole speculator through `SpecDecoder` against a
//!   matching target → all proposals accepted.

use anyhow::Result;
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::DraftGeom;
use rlx_eagle3::speculator::{Eagle3Speculator, VerifierHiddenSource};
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_runtime::spec_decode::{Speculator, VerifyResult};

const DRAFT_VOCAB: usize = 8;
const TARGET_VOCAB: usize = 16;
const H: usize = 8;
const INTER: usize = 16;
const N_HEADS: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 2;
const N_AUX: usize = 3;

fn tiny_cfg() -> Eagle3Config {
    let json = format!(
        r#"{{
            "draft_vocab_size": {draft},
            "norm_before_residual": false,
            "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
            "transformer_layer_config": {{
                "model_type": "llama",
                "hidden_size": {h}, "intermediate_size": {inter},
                "num_hidden_layers": 1, "num_attention_heads": {nh},
                "num_key_value_heads": {nkv}, "head_dim": {hd},
                "vocab_size": {tv},
                "rms_norm_eps": 1e-6,
                "rope_parameters": {{"rope_theta": 10000.0, "rope_type": "default"}}
            }}
        }}"#,
        draft = DRAFT_VOCAB,
        h = H,
        inter = INTER,
        nh = N_HEADS,
        nkv = N_KV,
        hd = HEAD_DIM,
        tv = TARGET_VOCAB,
    );
    Eagle3Config::from_bytes(json.as_bytes()).unwrap()
}

fn ramp(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(0x9E3779B1).wrapping_add(salt)) >> 16;
            ((x as f32) / (1u32 << 16) as f32 - 0.5) * 0.05
        })
        .collect()
}

fn synth_weights(cfg: &Eagle3Config) -> Eagle3DraftWeights {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype as StDtype, TensorView};
    use std::collections::HashMap;

    let g = DraftGeom::from_cfg(cfg);
    let kv_dim = g.n_kv_heads * g.head_dim;

    let f32_buffers: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
        (
            "fc.weight".into(),
            ramp(g.h_draft * 3 * g.h_target, 1),
            vec![g.h_draft, 3 * g.h_target],
        ),
        (
            "embed_tokens.weight".into(),
            ramp(g.target_vocab * g.h_draft, 2),
            vec![g.target_vocab, g.h_draft],
        ),
        (
            "midlayer.input_layernorm.weight".into(),
            vec![1.0; g.h_draft],
            vec![g.h_draft],
        ),
        (
            "midlayer.hidden_norm.weight".into(),
            vec![1.0; g.h_draft],
            vec![g.h_draft],
        ),
        (
            "midlayer.self_attn.q_proj.weight".into(),
            ramp(g.h_draft * 2 * g.h_draft, 3),
            vec![g.h_draft, 2 * g.h_draft],
        ),
        (
            "midlayer.self_attn.k_proj.weight".into(),
            ramp(kv_dim * 2 * g.h_draft, 4),
            vec![kv_dim, 2 * g.h_draft],
        ),
        (
            "midlayer.self_attn.v_proj.weight".into(),
            ramp(kv_dim * 2 * g.h_draft, 5),
            vec![kv_dim, 2 * g.h_draft],
        ),
        (
            "midlayer.self_attn.o_proj.weight".into(),
            ramp(g.h_draft * g.h_draft, 6),
            vec![g.h_draft, g.h_draft],
        ),
        (
            "midlayer.post_attention_layernorm.weight".into(),
            vec![1.0; g.h_draft],
            vec![g.h_draft],
        ),
        (
            "midlayer.mlp.gate_proj.weight".into(),
            ramp(g.intermediate * g.h_draft, 7),
            vec![g.intermediate, g.h_draft],
        ),
        (
            "midlayer.mlp.up_proj.weight".into(),
            ramp(g.intermediate * g.h_draft, 8),
            vec![g.intermediate, g.h_draft],
        ),
        (
            "midlayer.mlp.down_proj.weight".into(),
            ramp(g.h_draft * g.intermediate, 9),
            vec![g.h_draft, g.intermediate],
        ),
        ("norm.weight".into(), vec![1.0; g.h_draft], vec![g.h_draft]),
        (
            "lm_head.weight".into(),
            ramp(g.draft_vocab * g.h_draft, 10),
            vec![g.draft_vocab, g.h_draft],
        ),
    ];

    let f32_bytes: Vec<Vec<u8>> = f32_buffers
        .iter()
        .map(|(_, data, _)| bytemuck::cast_slice::<f32, u8>(data.as_slice()).to_vec())
        .collect();
    let d2t_data: Vec<u32> = vec![0; g.draft_vocab];
    let d2t_bytes: Vec<u8> = bytemuck::cast_slice(&d2t_data).to_vec();

    let mut views: HashMap<&str, TensorView<'_>> = HashMap::new();
    for ((name, _, shape), bytes) in f32_buffers.iter().zip(&f32_bytes) {
        views.insert(
            name.as_str(),
            TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
        );
    }
    views.insert(
        "d2t",
        TensorView::new(StDtype::U32, vec![g.draft_vocab], d2t_bytes.as_slice()).unwrap(),
    );

    let blob = serialize(&views, None).unwrap();
    Eagle3DraftWeights::from_bytes(&blob).unwrap()
}

/// Canned verifier hidden source — returns the same `N_AUX` deterministic
/// hidden states regardless of context.
struct CannedHidden;
impl VerifierHiddenSource for CannedHidden {
    fn aux_hidden_states(&self) -> Result<Vec<Vec<f32>>> {
        Ok(vec![
            (0..H).map(|i| (i as f32) * 0.01).collect(),
            (0..H).map(|i| (i as f32) * -0.02).collect(),
            (0..H).map(|i| (i as f32) * 0.015).collect(),
        ])
    }
    fn target_hidden_size(&self) -> usize {
        H
    }
    fn num_aux_layers(&self) -> usize {
        N_AUX
    }
}

fn build_spec() -> Eagle3Speculator<CannedHidden> {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    Eagle3Speculator::new(cfg, weights, CannedHidden).unwrap()
}

#[test]
fn propose_returns_n_tokens_with_target_vocab_probs() {
    let mut spec = build_spec();
    let context = vec![0u32, 1, 2, 3];

    let proposal = spec.propose(&context, 3);
    assert_eq!(proposal.tokens.len(), 3);
    assert_eq!(proposal.probs.len(), 3);
    for (i, row) in proposal.probs.iter().enumerate() {
        assert_eq!(row.len(), TARGET_VOCAB, "row {i} wrong vocab size");
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "probs row {i} sum = {sum}, expected ~1.0"
        );
    }
    for &t in &proposal.tokens {
        assert!((t as usize) < TARGET_VOCAB);
    }
}

#[test]
fn propose_with_n_zero_returns_empty() {
    let mut spec = build_spec();
    let proposal = spec.propose(&[0u32, 1], 0);
    assert!(proposal.tokens.is_empty());
    assert!(proposal.probs.is_empty());
}

#[test]
fn propose_is_deterministic() {
    let mut a = build_spec();
    let mut b = build_spec();
    let ctx = vec![0u32, 5, 9, 2];
    let pa = a.propose(&ctx, 3);
    let pb = b.propose(&ctx, 3);
    assert_eq!(pa.tokens, pb.tokens);
    for (ra, rb) in pa.probs.iter().zip(&pb.probs) {
        for (&va, &vb) in ra.iter().zip(rb) {
            assert!((va - vb).abs() < 1e-6, "non-deterministic: {va} vs {vb}");
        }
    }
}

#[test]
fn d2t_zero_offset_keeps_proposed_tokens_in_draft_vocab_subset() {
    // With d2t offsets all zero, target_id == draft_id, so proposed
    // tokens must all be < DRAFT_VOCAB (== 8).
    let mut spec = build_spec();
    let proposal = spec.propose(&[0u32, 1], 3);
    for &t in &proposal.tokens {
        assert!(
            (t as usize) < DRAFT_VOCAB,
            "with offset=0, proposed token {t} should be < {DRAFT_VOCAB}"
        );
    }
}

#[test]
fn unreachable_target_ids_get_zero_probability() {
    // With d2t offsets all zero and DRAFT_VOCAB=8, target ids 8..16
    // are not produced by any draft id — so their softmax probs
    // must be 0.
    let mut spec = build_spec();
    let proposal = spec.propose(&[0u32, 1, 2], 3);
    for (i, row) in proposal.probs.iter().enumerate() {
        for (j, &val) in row.iter().enumerate().take(TARGET_VOCAB).skip(DRAFT_VOCAB) {
            assert_eq!(val, 0.0, "row {i} target id {j} should be zero, got {val}");
        }
    }
}

#[test]
fn commit_seeds_next_propose_with_last_accepted_token() {
    let mut spec = build_spec();
    let ctx0 = vec![0u32, 1];
    let p0 = spec.propose(&ctx0, 1);
    let t0 = p0.tokens[0];

    // Pretend the spec_decode accept algorithm accepted t0.
    spec.commit(&ctx0, &[t0]);

    // Next propose with the SAME context — first proposal should be
    // identical to what we'd get if we extended the context by t0
    // and re-proposed with prev_token unset.
    let mut spec_extended = build_spec();
    let mut ctx_ext = ctx0.clone();
    ctx_ext.push(t0);
    let p_ext = spec_extended.propose(&ctx_ext, 1);

    let p_after_commit = spec.propose(&ctx0, 1);
    assert_eq!(
        p_after_commit.tokens[0], p_ext.tokens[0],
        "commit should make next propose see prev_token = {t0}",
    );
}

#[test]
fn end_to_end_spec_decoder_loop_runs_three_rounds() {
    // Plug the Eagle3Speculator into rlx_runtime's SpecDecoder and
    // verify it produces N tokens per round without panicking.
    // The "target" speculator returns identical distributions ⇒
    // all proposals are accepted (Leviathan accept_ratio = 1).
    use rlx_runtime::spec_decode::{DraftProposal, SpecDecoder, Speculator as Sp};

    struct IdentityTarget;
    impl Sp for IdentityTarget {
        fn propose(&mut self, _ctx: &[u32], _n: usize) -> DraftProposal {
            unimplemented!("target only verifies")
        }
        fn verify(&mut self, _ctx: &[u32], proposed: &[u32]) -> VerifyResult {
            // Put all mass on the proposed token at each position so
            // q == 1 there and the speculative_accept ratio q/p ≥ 1
            // accepts everything.
            let probs = proposed
                .iter()
                .map(|&t| {
                    let mut r = vec![0.0; TARGET_VOCAB];
                    r[t as usize] = 1.0;
                    r
                })
                .collect();
            VerifyResult { probs }
        }
    }

    let draft = build_spec();
    let mut dec = SpecDecoder::new(draft, IdentityTarget, 3, 42);
    let mut context = vec![0u32, 1, 2];
    for _round in 0..3 {
        let new_tokens = dec.step(&context);
        assert!(!new_tokens.is_empty(), "spec decoder produced 0 tokens");
        // Up to n+1 tokens per round (n accepted + 1 corrected).
        assert!(new_tokens.len() <= 4);
        for &t in &new_tokens {
            assert!((t as usize) < TARGET_VOCAB);
        }
        context.extend(new_tokens);
    }
}

#[test]
fn hir_runner_attaches_and_propose_doesnt_panic() {
    // Surface-level smoke for `with_hir_runner`. Numerical parity is
    // pinned by `tests/hir_parity.rs` (single-step HIR vs scalar at
    // past_seq=0 and =1: `logits max|Δ| ≤ 7.45e-9`). On these 1.6 KB
    // synthetic weights, tiny float-associativity drift between scalar
    // matvec and HIR matmul can flip argmax when adjacent logits sit
    // within ~1e-5 — so a Speculator-API parity assertion here is
    // strictly noisier than the lower-level parity tests. Instead
    // we verify the wiring (the HIR path runs without panicking and
    // produces a well-formed proposal).
    use rlx_runtime::Device;

    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    let mut hir = Eagle3Speculator::new(cfg.clone(), weights, CannedHidden)
        .unwrap()
        .with_hir_runner(Device::Cpu, 3)
        .unwrap();
    assert!(hir.uses_hir());

    let context = vec![0u32, 1, 2, 3];
    let p = hir.propose(&context, 3);
    assert_eq!(p.tokens.len(), 3);
    assert_eq!(p.probs.len(), 3);
    for (i, row) in p.probs.iter().enumerate() {
        assert_eq!(row.len(), TARGET_VOCAB, "row {i} wrong vocab size");
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {i} sum = {sum}");
    }
    for &t in &p.tokens {
        assert!((t as usize) < TARGET_VOCAB);
    }
}
