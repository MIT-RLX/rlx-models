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

//! HIR draft graph for `Eagle3Speculator::propose`.
//!
//! Builds one `Graph` per speculation-step `past_seq` and lets the
//! Session compile it for whichever backend the caller picked. This
//! is the multi-op submission path — the per-call MLX floor pays
//! once per step instead of once per matmul.
//!
//! See [`crate::reference`] for the scalar Rust forward this graph
//! mirrors. Architecture is pinned to vLLM-speculators
//! `Eagle3FirstLayerMixin.forward` — split → norm → recat → q/k/v →
//! GQA → o + MLP → final_norm + lm_head.

use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, HirGraphExt, Shape, hir_to_graph};

use crate::draft::DraftGeom;

/// Outputs that the compiled draft step emits, in declaration order
/// (so the caller can grab them by index after `compiled.run(...)`).
#[derive(Debug, Clone, Copy)]
pub enum DraftStepOutput {
    Logits = 0,
    NewHidden = 1,
    NewK = 2,
    NewV = 3,
}

/// Named tensor keys used for parameters of the compiled draft step.
/// Match what [`crate::weights::Eagle3DraftWeights`] surfaces after
/// canonicalization (see `crate::weights` module docs).
pub mod tensor_names {
    pub const EMBED_TOKENS: &str = "embed_tokens.weight";
    pub const INPUT_LAYERNORM: &str = "decoder.input_layernorm.weight";
    pub const HIDDEN_NORM: &str = "decoder.hidden_norm.weight";
    pub const Q_PROJ: &str = "decoder.self_attn.q_proj.weight";
    pub const K_PROJ: &str = "decoder.self_attn.k_proj.weight";
    pub const V_PROJ: &str = "decoder.self_attn.v_proj.weight";
    pub const O_PROJ: &str = "decoder.self_attn.o_proj.weight";
    pub const POST_ATTN_LN: &str = "decoder.post_attention_layernorm.weight";
    pub const GATE_PROJ: &str = "decoder.mlp.gate_proj.weight";
    pub const UP_PROJ: &str = "decoder.mlp.up_proj.weight";
    pub const DOWN_PROJ: &str = "decoder.mlp.down_proj.weight";
    pub const NORM: &str = "norm.weight";
    pub const LM_HEAD: &str = "lm_head.weight";
    /// Zero buffer for RMSNorm beta (length = h_draft, all zeros).
    pub const ZERO_BETA: &str = "zero_beta";
}

/// Input keys (graph-input ports) the draft step expects each call.
///
/// **`prev_embed` is host-gathered**: the caller reads the
/// `prev_target_token` row from `embed_tokens.weight` (a 1.4 GB
/// table for the Gemma 4 draft) and passes the 21 KB row in here.
/// This avoids: (1) keeping 1.4 GB resident on every backend,
/// (2) MLX's `mlx_indices_i64` host-eval that crashes Compiled
/// mode, (3) Metal's MPSNDArray nil-device errors on small
/// gather ops.
pub mod input_names {
    pub const PREV_EMBED: &str = "prev_embed";
    pub const PREV_HIDDEN: &str = "prev_hidden";
    pub const PAST_K: &str = "past_k";
    pub const PAST_V: &str = "past_v";
    pub const ROPE_COS: &str = "rope_cos";
    pub const ROPE_SIN: &str = "rope_sin";
}

/// Repeat KV heads for GQA. Reimplements the same pattern as
/// `rlx_flow::blocks::self_attn::repeat_kv`, which is pub(crate).
fn repeat_kv(
    g: &mut HirMut,
    x: rlx_ir::HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> rlx_ir::HirNodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

/// Build a single-step draft graph for a fixed `past_seq`.
///
/// For `past_seq == 0` the KV cache inputs are dummy `[1, 0, kv_dim]`
/// tensors that just feed the concat (a zero-length axis is a no-op
/// for concat). Subsequent steps recompile with the new past_seq.
///
/// Inputs (declaration order — graph-input names listed in [`input_names`]):
/// - `prev_hidden`: `[1, 1, h_draft]` — previous step's decoder out
///   (or `fc(aux)` on step 0).
/// - `prev_token`: `[1, 1]` — target-vocab token id, f32 (the
///   runtime gather op consumes it as an index).
/// - `past_k`: `[1, past_seq, kv_dim]`.
/// - `past_v`: `[1, past_seq, kv_dim]`.
/// - `rope_cos`, `rope_sin`: `[1, head_dim/2]` each — the rotation
///   row for `position = past_seq`.
///
/// Outputs (in [`DraftStepOutput`] order):
/// 1. `logits`: `[1, 1, draft_vocab]`
/// 2. `new_hidden`: `[1, 1, h_draft]`
/// 3. `new_k`: `[1, past_seq+1, kv_dim]`
/// 4. `new_v`: `[1, past_seq+1, kv_dim]`
pub fn build_draft_step_graph(geom: DraftGeom, past_seq: usize) -> Graph {
    let f = DType::F32;
    let h = geom.h_draft;
    let two_h = 2 * h;
    let q_dim = geom.n_heads * geom.head_dim;
    let kv_dim = geom.n_kv_heads * geom.head_dim;
    let half = geom.head_dim / 2;
    let group = geom.n_heads / geom.n_kv_heads;
    let cur_seq = past_seq + 1;

    let mut hir = HirModule::new("eagle3_draft_step");
    let mut gb = HirMut::new(&mut hir);

    // ── Inputs ───────────────────────────────────────────────────
    // prev_embed is host-gathered (see input_names::PREV_EMBED docs).
    let prev_embed = gb.input(input_names::PREV_EMBED, Shape::new(&[1, 1, h], f));
    let prev_hidden = gb.input(input_names::PREV_HIDDEN, Shape::new(&[1, 1, h], f));
    let past_k = gb.input(input_names::PAST_K, Shape::new(&[1, past_seq, kv_dim], f));
    let past_v = gb.input(input_names::PAST_V, Shape::new(&[1, past_seq, kv_dim], f));
    let rope_cos = gb.input(input_names::ROPE_COS, Shape::new(&[1, half], f));
    let rope_sin = gb.input(input_names::ROPE_SIN, Shape::new(&[1, half], f));

    // ── Params (transposed where the mm convention requires it) ──
    // Mirror Gemma's flow: weights stored on disk as [out, in] get
    // loaded transposed so `mm(x, w)` reads `[..., K] @ [K, N]`.
    let input_layernorm = gb.param(tensor_names::INPUT_LAYERNORM, Shape::new(&[h], f));
    let hidden_norm = gb.param(tensor_names::HIDDEN_NORM, Shape::new(&[h], f));
    // q/k/v_proj stored on disk as [out, 2*H]; load transposed → [2*H, out].
    let q_w = gb.param(tensor_names::Q_PROJ, Shape::new(&[two_h, q_dim], f));
    let k_w = gb.param(tensor_names::K_PROJ, Shape::new(&[two_h, kv_dim], f));
    let v_w = gb.param(tensor_names::V_PROJ, Shape::new(&[two_h, kv_dim], f));
    // o_proj stored as [H, q_dim]; load transposed → [q_dim, H].
    let o_w = gb.param(tensor_names::O_PROJ, Shape::new(&[q_dim, h], f));
    let post_attn_ln = gb.param(tensor_names::POST_ATTN_LN, Shape::new(&[h], f));
    // gate/up_proj as [I, H]; load transposed → [H, I].
    let gate_w = gb.param(
        tensor_names::GATE_PROJ,
        Shape::new(&[h, geom.intermediate], f),
    );
    let up_w = gb.param(
        tensor_names::UP_PROJ,
        Shape::new(&[h, geom.intermediate], f),
    );
    // down_proj as [H, I]; load transposed → [I, H].
    let down_w = gb.param(
        tensor_names::DOWN_PROJ,
        Shape::new(&[geom.intermediate, h], f),
    );
    let norm_w = gb.param(tensor_names::NORM, Shape::new(&[h], f));
    // lm_head as [V_draft, H]; load transposed → [H, V_draft].
    let lm_head_w = gb.param(tensor_names::LM_HEAD, Shape::new(&[h, geom.draft_vocab], f));
    // RMSNorm needs a beta arg; set to zeros at upload time.
    let zero_beta = gb.param(tensor_names::ZERO_BETA, Shape::new(&[h], f));

    // ── Split-and-norm modified first layer ──────────────────────
    // (embed already gathered on host — see prev_embed input.)
    let eps = geom.rms_eps;
    let embed_normed = gb.rms_norm(prev_embed, input_layernorm, zero_beta, eps);
    let hidden_normed = gb.rms_norm(prev_hidden, hidden_norm, zero_beta, eps);
    let residual = if geom.norm_before_residual {
        hidden_normed
    } else {
        prev_hidden
    };

    // Concat last axis: [1, 1, H] + [1, 1, H] → [1, 1, 2H].
    let x = gb.concat_(vec![embed_normed, hidden_normed], 2);

    let q = gb.mm(x, q_w);
    let k = gb.mm(x, k_w);
    let v = gb.mm(x, v_w);

    let q_rope = gb.rope(q, rope_cos, rope_sin, geom.head_dim);
    let k_rope = gb.rope(k, rope_cos, rope_sin, geom.head_dim);

    // Append k_rope, v to past KV along seq axis. Metal's MPS
    // crashes on zero-length tensor descriptors, so when
    // past_seq=0 we skip the concat — the new k/v IS the full
    // KV cache after this step. The past_k/past_v graph inputs
    // are still declared (callers always pass empty slices for
    // them) so the input-name table stays consistent.
    let (new_k, new_v) = if past_seq == 0 {
        (k_rope, v)
    } else {
        (
            gb.concat_(vec![past_k, k_rope], 1),
            gb.concat_(vec![past_v, v], 1),
        )
    };

    let k_rep = repeat_kv(&mut gb, new_k, geom.n_kv_heads, geom.head_dim, group);
    let v_rep = repeat_kv(&mut gb, new_v, geom.n_kv_heads, geom.head_dim, group);

    // Attention: q [1, 1, q_dim] · k_rep [1, cur_seq, q_dim] →
    // [1, 1, q_dim]. Causal mask is trivially satisfied — single
    // query attends to all past + current entries.
    let attn = gb.attention_kind(
        q_rope,
        k_rep,
        v_rep,
        geom.n_heads,
        geom.head_dim,
        MaskKind::Causal,
        Shape::new(&[1, 1, q_dim], f),
    );

    let attn_out = gb.mm(attn, o_w);
    let h_attn = gb.add(residual, attn_out);

    let mlp_in = gb.rms_norm(h_attn, post_attn_ln, zero_beta, eps);
    let gate = gb.mm(mlp_in, gate_w);
    let up = gb.mm(mlp_in, up_w);
    let gate_act = gb.silu(gate);
    let swiglu = gb.mul(gate_act, up);
    let mlp_out = gb.mm(swiglu, down_w);

    let new_hidden = gb.add(h_attn, mlp_out);
    let final_normed = gb.rms_norm(new_hidden, norm_w, zero_beta, eps);
    let logits = gb.mm(final_normed, lm_head_w);

    // Outputs in DraftStepOutput order.
    gb.set_outputs(vec![logits, new_hidden, new_k, new_v]);

    // Silence "unused" warnings on cur_seq (it's a documentation
    // anchor + future shape-check site).
    let _ = cur_seq;

    hir_to_graph(hir).expect("eagle3 draft HIR lowers cleanly")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Eagle3Config;

    fn tiny_geom() -> DraftGeom {
        let json = r#"{
            "draft_vocab_size": 8,
            "norm_before_residual": true,
            "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 4,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }"#;
        let cfg = Eagle3Config::from_bytes(json.as_bytes()).unwrap();
        DraftGeom::from_cfg(&cfg)
    }

    #[test]
    fn build_lowers_past_seq_zero() {
        let geom = tiny_geom();
        // Just verify the graph builds + lowers without panicking.
        let _g = build_draft_step_graph(geom, 0);
    }

    #[test]
    fn build_lowers_past_seq_two() {
        let geom = tiny_geom();
        let _g = build_draft_step_graph(geom, 2);
    }

    #[test]
    fn build_produces_four_outputs() {
        let geom = tiny_geom();
        let g = build_draft_step_graph(geom, 1);
        assert_eq!(g.outputs.len(), 4, "logits + hidden + K + V");
    }
}
