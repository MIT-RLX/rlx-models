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

//! Live wiring test: a real `GemmaGenerator` decodes with the EAGLE3
//! aux tap on, writes per-step aux states to an `AuxStateBuffer`,
//! and the `Eagle3Speculator` (HIR runner on CPU) consumes them to
//! produce a `DraftProposal`.
//!
//! Uses synthetic weights end-to-end so it runs in <1s. The geometries
//! line up exactly the way they would for a real Gemma 4 + RedHatAI
//! EAGLE3 deployment — same field semantics, same aux-id flow, same
//! `target_hidden_size` constraint, same d2t-offset application.
//!
//! Together with `tests/hir_parity.rs` (single-step numerical parity)
//! and `tests/propose_e2e.rs` (Speculator API smoke), this pins the
//! whole stack from `GemmaRunner::generate_with_aux` through the
//! bridge into `Eagle3Speculator::propose`.

#![cfg(feature = "gemma")]

use anyhow::Result;
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::DraftGeom;
use rlx_eagle3::gemma_bridge::AuxStateBuffer;
use rlx_eagle3::speculator::Eagle3Speculator;
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_gemma::config::{GemmaArch, GemmaConfig, GemmaRopeMap};
use rlx_gemma::generator::GemmaGenerator;
use rlx_qwen3::sampling::SampleOpts;
use rlx_runtime::Device;
use rlx_runtime::spec_decode::Speculator;
use std::collections::HashMap;

/// Tiny Gemma config — matches `crate::config::GemmaConfig::tiny_test`
/// in shape but pub-accessible from this integration test.
fn tiny_gemma_cfg() -> GemmaConfig {
    GemmaConfig {
        arch: GemmaArch::Gemma,
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        head_dim: Some(8),
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        sliding_window: None,
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        layer_types: Vec::new(),
        rope_parameters: GemmaRopeMap::default(),
        global_head_dim: None,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
        use_bidirectional_attention: None,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        use_double_wide_mlp: false,
        enable_moe_block: false,
    }
}

/// Synthetic Gemma weights — deterministic non-zero pattern so the
/// generator doesn't see all-zero logits.
fn synthetic_gemma_weights(cfg: &GemmaConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let h = cfg.hidden_size;
    let dh = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let kv = cfg.num_key_value_heads;
    let q_dim = nh * dh;
    let kv_dim = kv * dh;
    let i_dim = cfg.intermediate_size;
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
        let s = i as u32;
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (pat(h, 100 + s), vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (pat(h, 200 + s), vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (pat(q_dim * h, 300 + s), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (pat(kv_dim * h, 400 + s), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (pat(kv_dim * h, 500 + s), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (pat(h * q_dim, 600 + s), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (pat(i_dim * h, 700 + s), vec![i_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (pat(i_dim * h, 800 + s), vec![i_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (pat(h * i_dim, 900 + s), vec![h, i_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (pat(h, 99), vec![h]));
    t
}

/// Eagle3 draft config whose `target_hidden_size` matches the Gemma
/// verifier's `hidden_size`.
fn tiny_eagle3_cfg(target_hidden: usize, aux_layer_ids: &[usize]) -> Eagle3Config {
    let ids_json = format!(
        "[{}]",
        aux_layer_ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let json = format!(
        r#"{{
            "draft_vocab_size": 16,
            "target_hidden_size": {th},
            "norm_before_residual": true,
            "eagle_aux_hidden_state_layer_ids": {ids},
            "transformer_layer_config": {{
                "model_type": "llama",
                "hidden_size": {th}, "intermediate_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 4,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {{"rope_theta": 10000.0, "rope_type": "default"}}
            }}
        }}"#,
        th = target_hidden,
        ids = ids_json,
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

fn synth_eagle3_weights(cfg: &Eagle3Config) -> Eagle3DraftWeights {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype as StDtype, TensorView};

    let geom = DraftGeom::from_cfg(cfg);
    let kv_dim = geom.n_kv_heads * geom.head_dim;
    let q_dim = geom.n_heads * geom.head_dim;
    let aux_n = cfg
        .eagle_aux_hidden_state_layer_ids
        .as_ref()
        .map(|v| v.len())
        .unwrap_or(3);

    let buffers: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
        (
            "fc.weight".into(),
            ramp(geom.h_draft * aux_n * geom.h_target, 1),
            vec![geom.h_draft, aux_n * geom.h_target],
        ),
        (
            "embed_tokens.weight".into(),
            ramp(geom.target_vocab * geom.h_draft, 2),
            vec![geom.target_vocab, geom.h_draft],
        ),
        (
            "midlayer.input_layernorm.weight".into(),
            vec![1.0; geom.h_draft],
            vec![geom.h_draft],
        ),
        (
            "midlayer.hidden_norm.weight".into(),
            vec![1.0; geom.h_draft],
            vec![geom.h_draft],
        ),
        (
            "midlayer.self_attn.q_proj.weight".into(),
            ramp(q_dim * 2 * geom.h_draft, 3),
            vec![q_dim, 2 * geom.h_draft],
        ),
        (
            "midlayer.self_attn.k_proj.weight".into(),
            ramp(kv_dim * 2 * geom.h_draft, 4),
            vec![kv_dim, 2 * geom.h_draft],
        ),
        (
            "midlayer.self_attn.v_proj.weight".into(),
            ramp(kv_dim * 2 * geom.h_draft, 5),
            vec![kv_dim, 2 * geom.h_draft],
        ),
        (
            "midlayer.self_attn.o_proj.weight".into(),
            ramp(geom.h_draft * q_dim, 6),
            vec![geom.h_draft, q_dim],
        ),
        (
            "midlayer.post_attention_layernorm.weight".into(),
            vec![1.0; geom.h_draft],
            vec![geom.h_draft],
        ),
        (
            "midlayer.mlp.gate_proj.weight".into(),
            ramp(geom.intermediate * geom.h_draft, 7),
            vec![geom.intermediate, geom.h_draft],
        ),
        (
            "midlayer.mlp.up_proj.weight".into(),
            ramp(geom.intermediate * geom.h_draft, 8),
            vec![geom.intermediate, geom.h_draft],
        ),
        (
            "midlayer.mlp.down_proj.weight".into(),
            ramp(geom.h_draft * geom.intermediate, 9),
            vec![geom.h_draft, geom.intermediate],
        ),
        (
            "norm.weight".into(),
            vec![1.0; geom.h_draft],
            vec![geom.h_draft],
        ),
        (
            "lm_head.weight".into(),
            ramp(geom.draft_vocab * geom.h_draft, 10),
            vec![geom.draft_vocab, geom.h_draft],
        ),
    ];

    let bytes_per: Vec<Vec<u8>> = buffers
        .iter()
        .map(|(_, d, _)| bytemuck::cast_slice::<f32, u8>(d.as_slice()).to_vec())
        .collect();
    let d2t: Vec<u32> = vec![0; geom.draft_vocab];
    let d2t_bytes: Vec<u8> = bytemuck::cast_slice(&d2t).to_vec();
    let mut views: HashMap<&str, TensorView<'_>> = HashMap::new();
    for ((name, _, shape), bytes) in buffers.iter().zip(&bytes_per) {
        views.insert(
            name.as_str(),
            TensorView::new(StDtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
        );
    }
    views.insert(
        "d2t",
        TensorView::new(StDtype::U32, vec![geom.draft_vocab], d2t_bytes.as_slice()).unwrap(),
    );
    let blob = serialize(&views, None).unwrap();
    Eagle3DraftWeights::from_bytes(&blob).unwrap()
}

#[test]
fn gemma_generates_aux_speculator_proposes() -> Result<()> {
    // ── Geometries: Gemma's hidden_size MUST equal Eagle3's
    //   target_hidden_size. Both are 16 here.
    let gemma_cfg = tiny_gemma_cfg();
    assert_eq!(gemma_cfg.hidden_size, 16);

    let aux_ids = vec![0, 2]; // low + high of Gemma's 4 layers
    let eagle_cfg = tiny_eagle3_cfg(gemma_cfg.hidden_size, &aux_ids);
    let eagle_weights = synth_eagle3_weights(&eagle_cfg);

    // ── Build the Gemma generator (on CPU).
    let mut gemma_wm =
        rlx_core::weight_map::WeightMap::from_tensors(synthetic_gemma_weights(&gemma_cfg));
    let mut gemma = GemmaGenerator::from_loader(gemma_cfg.clone(), &mut gemma_wm, Device::Cpu)?;
    gemma.set_aux_hidden_layer_ids(aux_ids.clone());
    assert!(gemma.aux_enabled());

    // ── Build the bridge + speculator.
    let aux_buffer = AuxStateBuffer::new();
    let writer = aux_buffer.clone();
    let hidden_source = aux_buffer.into_hidden_source(gemma_cfg.hidden_size, aux_ids.len());

    let n = eagle_cfg.speculative_tokens;
    let mut speculator = Eagle3Speculator::new(eagle_cfg.clone(), eagle_weights, hidden_source)?
        .with_hir_runner(Device::Cpu, n)?;
    assert!(speculator.uses_hir());

    // ── Drive one EAGLE3 round end-to-end ─────────────────────────
    // 1. Gemma prefill + first decode step (the prefill seed yields
    //    no aux; the next decode step yields aux for the current
    //    position).
    gemma.prefill(&[1u32, 2, 3]);
    let _seed_tok = gemma.step_cached(SampleOpts::greedy())?; // no aux (prefill seed)
    assert!(gemma.take_last_aux().is_none());

    let _decode_tok = gemma.step_cached(SampleOpts::greedy())?;
    let aux = gemma.take_last_aux().expect("decode step produces aux");
    assert_eq!(aux.len(), aux_ids.len());
    for row in &aux {
        assert_eq!(row.len(), gemma_cfg.hidden_size);
        assert!(row.iter().all(|v| v.is_finite()));
    }

    // 2. Hand aux to the bridge; speculator proposes.
    writer.write(aux);
    let ctx = vec![1u32, 2, 3, _seed_tok, _decode_tok];
    let proposal = speculator.propose(&ctx, n);

    assert_eq!(proposal.tokens.len(), n, "n target-vocab tokens proposed");
    assert_eq!(proposal.probs.len(), n);
    for (i, row) in proposal.probs.iter().enumerate() {
        assert_eq!(
            row.len(),
            eagle_cfg.target_vocab_size(),
            "row {i} sized to target vocab"
        );
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {i} sums to {sum}");
    }
    for &t in &proposal.tokens {
        assert!((t as usize) < eagle_cfg.target_vocab_size());
    }

    // 3. Second round — Gemma steps again, writes new aux, speculator
    //    proposes again. Proves the buffer round-trips on each round.
    let _ = gemma.step_cached(SampleOpts::greedy())?;
    let aux2 = gemma.take_last_aux().expect("second-step aux");
    writer.write(aux2);
    let proposal2 = speculator.propose(&ctx, n);
    assert_eq!(proposal2.tokens.len(), n);

    Ok(())
}
