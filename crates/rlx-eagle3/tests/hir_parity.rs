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

//! Numerical parity: the HIR-built draft step (compiled on CPU)
//! must produce the same `(logits, new_hidden)` as the pure-Rust
//! scalar `Eagle3DraftReference` on identical synthetic inputs.
//!
//! This is the gating correctness check for the HIR port. The bench
//! numbers in `BENCH_BACKENDS.md` only matter if the HIR forward
//! actually computes the right answer.

use anyhow::Result;
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::{DraftGeom, DraftWeightRefs, Eagle3DraftReference};
use rlx_eagle3::hir_draft::{build_draft_step_graph, input_names as I, tensor_names as T};
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_runtime::{Device, Session};

// ── Tiny config (mirrors `crate::draft::tests::tiny_cfg`) ──────────
const H: usize = 16;
const I_DIM: usize = 32;
const N_HEADS: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 4;
const V_TARGET: usize = 32;
const V_DRAFT: usize = 8;

fn tiny_cfg() -> Eagle3Config {
    let json = format!(
        r#"{{
            "draft_vocab_size": {V_DRAFT},
            "norm_before_residual": true,
            "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
            "transformer_layer_config": {{
                "model_type": "llama",
                "hidden_size": {H}, "intermediate_size": {I_DIM},
                "num_hidden_layers": 1, "num_attention_heads": {N_HEADS},
                "num_key_value_heads": {N_KV}, "head_dim": {HEAD_DIM},
                "vocab_size": {V_TARGET},
                "rms_norm_eps": 1e-6,
                "rope_parameters": {{"rope_theta": 10000.0, "rope_type": "default"}}
            }}
        }}"#,
    );
    Eagle3Config::from_bytes(json.as_bytes()).unwrap()
}

/// Deterministic ramp; salt distinguishes weight tensors.
fn ramp(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(0x9E3779B1).wrapping_add(salt)) >> 16;
            ((x as f32) / (1u32 << 16) as f32 - 0.5) * 0.05
        })
        .collect()
}

fn synth_weights(geom: DraftGeom) -> Eagle3DraftWeights {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype as StDtype, TensorView};
    use std::collections::HashMap;

    let kv_dim = geom.n_kv_heads * geom.head_dim;
    let q_dim = geom.n_heads * geom.head_dim;

    let f32_buffers: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
        (
            "fc.weight".into(),
            ramp(geom.h_draft * 3 * geom.h_target, 1),
            vec![geom.h_draft, 3 * geom.h_target],
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
        // q/k/v_proj: disk shape [n_heads*head_dim, 2*H] or [kv_heads*head_dim, 2*H].
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
        // o_proj: disk shape [H, q_dim].
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

    let f32_bytes: Vec<Vec<u8>> = f32_buffers
        .iter()
        .map(|(_, data, _)| bytemuck::cast_slice::<f32, u8>(data.as_slice()).to_vec())
        .collect();
    let d2t_data: Vec<u32> = vec![0; geom.draft_vocab];
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
        TensorView::new(StDtype::U32, vec![geom.draft_vocab], d2t_bytes.as_slice()).unwrap(),
    );

    let blob = serialize(&views, None).unwrap();
    Eagle3DraftWeights::from_bytes(&blob).unwrap()
}

/// Transpose `[rows, cols]` → `[cols, rows]` row-major.
fn transpose_rc(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn hir_step_matches_reference_at_past_seq_0() -> Result<()> {
    let cfg = tiny_cfg();
    let geom = DraftGeom::from_cfg(&cfg);
    let weights = synth_weights(geom);

    // ── Reference forward ────────────────────────────────────────
    let refs = DraftWeightRefs::from_weights(&weights, &cfg)?;
    let mut draft = Eagle3DraftReference::new(&cfg, refs);
    let aux: Vec<Vec<f32>> = vec![
        (0..H).map(|i| (i as f32) * 0.01).collect(),
        (0..H).map(|i| (i as f32) * -0.02).collect(),
        (0..H).map(|i| (i as f32) * 0.015).collect(),
    ];
    let h0 = draft.init_hidden(&aux);
    let tok: u32 = 1;
    let (logits_ref, new_hidden_ref) = draft.step(&h0, tok)?;

    // ── HIR forward via Session::compile(Device::Cpu) ────────────
    let graph = build_draft_step_graph(geom, 0);
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(graph);

    let get = |name: &str| -> &[f32] { weights.get(name).map(|t| t.data.as_slice()).unwrap() };
    let q_dim = geom.n_heads * geom.head_dim;
    let kv_dim = geom.n_kv_heads * geom.head_dim;
    let two_h = 2 * geom.h_draft;

    compiled.set_param(T::INPUT_LAYERNORM, get("decoder.input_layernorm.weight"));
    compiled.set_param(T::HIDDEN_NORM, get("decoder.hidden_norm.weight"));
    compiled.set_param(
        T::POST_ATTN_LN,
        get("decoder.post_attention_layernorm.weight"),
    );
    compiled.set_param(T::NORM, get("norm.weight"));
    let q_t = transpose_rc(get("decoder.self_attn.q_proj.weight"), q_dim, two_h);
    let k_t = transpose_rc(get("decoder.self_attn.k_proj.weight"), kv_dim, two_h);
    let v_t = transpose_rc(get("decoder.self_attn.v_proj.weight"), kv_dim, two_h);
    compiled.set_param(T::Q_PROJ, &q_t);
    compiled.set_param(T::K_PROJ, &k_t);
    compiled.set_param(T::V_PROJ, &v_t);
    let o_t = transpose_rc(get("decoder.self_attn.o_proj.weight"), geom.h_draft, q_dim);
    compiled.set_param(T::O_PROJ, &o_t);
    let gate_t = transpose_rc(
        get("decoder.mlp.gate_proj.weight"),
        geom.intermediate,
        geom.h_draft,
    );
    let up_t = transpose_rc(
        get("decoder.mlp.up_proj.weight"),
        geom.intermediate,
        geom.h_draft,
    );
    let down_t = transpose_rc(
        get("decoder.mlp.down_proj.weight"),
        geom.h_draft,
        geom.intermediate,
    );
    compiled.set_param(T::GATE_PROJ, &gate_t);
    compiled.set_param(T::UP_PROJ, &up_t);
    compiled.set_param(T::DOWN_PROJ, &down_t);
    let lm_t = transpose_rc(get("lm_head.weight"), geom.draft_vocab, geom.h_draft);
    compiled.set_param(T::LM_HEAD, &lm_t);
    let zero_beta = vec![0.0f32; geom.h_draft];
    compiled.set_param(T::ZERO_BETA, &zero_beta);

    // Host-side embed lookup (same as bench).
    let embed_tokens = get("embed_tokens.weight");
    let prev_embed: Vec<f32> = embed_tokens[(tok as usize) * H..(tok as usize + 1) * H].to_vec();
    let past_k: Vec<f32> = Vec::new();
    let past_v: Vec<f32> = Vec::new();
    let half = HEAD_DIM / 2;
    let rope_cos: Vec<f32> = vec![1.0; half]; // position 0 ⇒ cos=1
    let rope_sin: Vec<f32> = vec![0.0; half]; // position 0 ⇒ sin=0

    let outs = compiled.run(&[
        (I::PREV_EMBED, prev_embed.as_slice()),
        (I::PREV_HIDDEN, h0.as_slice()),
        (I::PAST_K, past_k.as_slice()),
        (I::PAST_V, past_v.as_slice()),
        (I::ROPE_COS, rope_cos.as_slice()),
        (I::ROPE_SIN, rope_sin.as_slice()),
    ]);
    let logits_hir = outs[0].clone();
    let new_hidden_hir = outs[1].clone();

    // ── Compare ──────────────────────────────────────────────────
    let logits_diff = max_abs_diff(&logits_ref, &logits_hir);
    let hidden_diff = max_abs_diff(&new_hidden_ref, &new_hidden_hir);
    println!(
        "HIR vs reference: logits max|Δ| = {logits_diff:.2e}, hidden max|Δ| = {hidden_diff:.2e}"
    );

    // Tolerance: scalar f32 forward vs HIR-compiled f32 forward
    // should agree to a few ULPs across ~10 ops. 1e-4 is generous;
    // 1e-5 would also typically hold for this tiny graph.
    let tol = 1e-4;
    assert!(
        logits_diff < tol,
        "logits diverged: max|Δ| = {logits_diff:.2e} >= {tol}"
    );
    assert!(
        hidden_diff < tol,
        "new_hidden diverged: max|Δ| = {hidden_diff:.2e} >= {tol}"
    );

    Ok(())
}

#[test]
fn hir_step_matches_reference_at_past_seq_1() -> Result<()> {
    let cfg = tiny_cfg();
    let geom = DraftGeom::from_cfg(&cfg);
    let weights = synth_weights(geom);

    // Run scalar TWO steps to populate KV cache + advance hidden.
    let refs = DraftWeightRefs::from_weights(&weights, &cfg)?;
    let mut draft = Eagle3DraftReference::new(&cfg, refs);
    let aux: Vec<Vec<f32>> = vec![
        (0..H).map(|i| (i as f32) * 0.01).collect(),
        (0..H).map(|i| (i as f32) * -0.02).collect(),
        (0..H).map(|i| (i as f32) * 0.015).collect(),
    ];
    let h0 = draft.init_hidden(&aux);
    let (_, h1_ref) = draft.step(&h0, 1)?;
    let past_k_after_step0 = draft.past_k().to_vec();
    let past_v_after_step0 = draft.past_v().to_vec();
    // Reset and re-run two steps; track second logits.
    let refs2 = DraftWeightRefs::from_weights(&weights, &cfg)?;
    let mut draft2 = Eagle3DraftReference::new(&cfg, refs2);
    let h0b = draft2.init_hidden(&aux);
    let (_, h1) = draft2.step(&h0b, 1)?;
    let (logits_ref_step2, hidden_ref_step2) = draft2.step(&h1, 5)?;

    // HIR: compile past_seq=1 graph. Provide past_k/past_v from scalar's
    // post-step-0 state so the inputs match.
    let graph = build_draft_step_graph(geom, 1);
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(graph);

    let get = |name: &str| -> &[f32] { weights.get(name).map(|t| t.data.as_slice()).unwrap() };
    let q_dim = geom.n_heads * geom.head_dim;
    let kv_dim = geom.n_kv_heads * geom.head_dim;
    let two_h = 2 * geom.h_draft;

    compiled.set_param(T::INPUT_LAYERNORM, get("decoder.input_layernorm.weight"));
    compiled.set_param(T::HIDDEN_NORM, get("decoder.hidden_norm.weight"));
    compiled.set_param(
        T::POST_ATTN_LN,
        get("decoder.post_attention_layernorm.weight"),
    );
    compiled.set_param(T::NORM, get("norm.weight"));
    let q_t = transpose_rc(get("decoder.self_attn.q_proj.weight"), q_dim, two_h);
    let k_t = transpose_rc(get("decoder.self_attn.k_proj.weight"), kv_dim, two_h);
    let v_t = transpose_rc(get("decoder.self_attn.v_proj.weight"), kv_dim, two_h);
    compiled.set_param(T::Q_PROJ, &q_t);
    compiled.set_param(T::K_PROJ, &k_t);
    compiled.set_param(T::V_PROJ, &v_t);
    let o_t = transpose_rc(get("decoder.self_attn.o_proj.weight"), geom.h_draft, q_dim);
    compiled.set_param(T::O_PROJ, &o_t);
    let gate_t = transpose_rc(
        get("decoder.mlp.gate_proj.weight"),
        geom.intermediate,
        geom.h_draft,
    );
    let up_t = transpose_rc(
        get("decoder.mlp.up_proj.weight"),
        geom.intermediate,
        geom.h_draft,
    );
    let down_t = transpose_rc(
        get("decoder.mlp.down_proj.weight"),
        geom.h_draft,
        geom.intermediate,
    );
    compiled.set_param(T::GATE_PROJ, &gate_t);
    compiled.set_param(T::UP_PROJ, &up_t);
    compiled.set_param(T::DOWN_PROJ, &down_t);
    let lm_t = transpose_rc(get("lm_head.weight"), geom.draft_vocab, geom.h_draft);
    compiled.set_param(T::LM_HEAD, &lm_t);
    let zero_beta = vec![0.0f32; geom.h_draft];
    compiled.set_param(T::ZERO_BETA, &zero_beta);

    // Inputs: same prev_hidden (h1) and prev_token (5) as scalar's
    // 2nd step. RoPE at position 1.
    let embed_tokens = get("embed_tokens.weight");
    let tok = 5usize;
    let prev_embed: Vec<f32> = embed_tokens[tok * H..(tok + 1) * H].to_vec();
    // Compute rope row at position 1.
    let half = HEAD_DIM / 2;
    let theta = 10_000.0f32;
    let (rope_cos, rope_sin): (Vec<f32>, Vec<f32>) = {
        let mut c = vec![0.0; half];
        let mut s = vec![0.0; half];
        for k in 0..half {
            let exp = -(2.0 * k as f64) / (HEAD_DIM as f64);
            let freq = (theta as f64).powf(exp);
            let angle = 1.0 * freq; // position = 1
            c[k] = angle.cos() as f32;
            s[k] = angle.sin() as f32;
        }
        (c, s)
    };

    let outs = compiled.run(&[
        (I::PREV_EMBED, prev_embed.as_slice()),
        (I::PREV_HIDDEN, h1.as_slice()),
        (I::PAST_K, past_k_after_step0.as_slice()),
        (I::PAST_V, past_v_after_step0.as_slice()),
        (I::ROPE_COS, rope_cos.as_slice()),
        (I::ROPE_SIN, rope_sin.as_slice()),
    ]);
    let logits_hir = outs[0].clone();
    let hidden_hir = outs[1].clone();

    let logits_diff = max_abs_diff(&logits_ref_step2, &logits_hir);
    let hidden_diff = max_abs_diff(&hidden_ref_step2, &hidden_hir);
    println!(
        "step-2 parity (HIR vs reference): logits max|Δ| = {logits_diff:.2e}, hidden max|Δ| = {hidden_diff:.2e}"
    );
    let _ = h1_ref;

    let tol = 1e-3;
    assert!(
        logits_diff < tol,
        "step-2 logits diverged: {logits_diff:.2e}"
    );
    assert!(
        hidden_diff < tol,
        "step-2 hidden diverged: {hidden_diff:.2e}"
    );
    Ok(())
}
