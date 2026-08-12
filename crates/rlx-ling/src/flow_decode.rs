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

//! Whole-model incremental decode: one token per `run()`, O(1) in prefix length.
//!
//! Every piece of recurrent state is an explicit graph **input** that comes back
//! out in a single packed **output**, and [`DecodeSession`] feeds it forward.
//! Deliberately not params-with-in-place-update: `Op::GatedDeltaNet{carry_state}`
//! documents that, and CPU/Metal/wgpu honour it, but MLX substitutes the new
//! state into its evaluation env where it does not survive to the next `run()` —
//! so a param-bound state silently freezes at its initial value there. See
//! [`crate::kda::KdaState`].
//!
//! Layout of the packed output (all f32, concatenated on one axis):
//!
//! ```text
//!   [ logits (vocab)                                   ]   (or hidden, if no lm_head)
//!   [ per KDA layer: cq, ck, cv, scan                  ]
//!   [ per MLA layer: k_new (h·qk), v_new (h·vd)        ]
//! ```
//!
//! The KDA scan state dominates that traffic — `num_heads · head_dim²` floats per
//! layer, ~19 MB/token for Ling-3.0-tiny across its 18 KDA layers, round-tripped
//! each step. That is the price of portability; see [`DecodeSession::state_bytes`].

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::moe::{DeepseekMoeDims, emit_deepseek_moe};
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{AttnKind, LingConfig};
use crate::flow::EMBED_KEY;
use crate::kda::{KdaDims, KdaState, emit_kda_decode};
use crate::mla::{MlaCache, MlaDims, ROPE_COS, ROPE_SIN, emit_mla_decode};

/// Where the KDA scan state lives between decode steps.
///
/// The scan state is by far the largest piece of decode state —
/// `num_heads · head_dim²` floats per KDA layer, 18.9 MB/token for Ling-3.0-tiny —
/// and round-tripping it through graph I/O costs a measurable slice of decode
/// (`Concat` was 15.1 ms of a 75.9 ms CPU token, most of it this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanState {
    /// Threaded through graph I/O. Correct on every backend.
    Portable,
    /// Kept in a persistent param that `Op::GatedDeltaNet { carry_state }` updates
    /// in place, so it never crosses the host boundary.
    ///
    /// Only valid where that in-place update actually survives to the next
    /// `run()` — CPU, Metal and wgpu do; **MLX and CoreML do not** (they
    /// substitute the new state into a per-evaluation env), and there the state
    /// silently freezes at zero. `decode_equivalence` covers both modes, so a
    /// backend that cannot do this fails loudly rather than drifting.
    ///
    /// Also note the param starts zeroed at compile time and cannot be reset
    /// without rebuilding — one compiled graph serves one sequence.
    InPlace,
}

/// Input name for a KDA layer's conv/scan state.
pub fn kda_state_input(layer: usize, which: &str) -> String {
    format!("state.kda{layer}.{which}")
}
/// Input name for an MLA layer's key/value cache.
pub fn mla_cache_input(layer: usize, which: &str) -> String {
    format!("state.mla{layer}.{which}")
}
/// Shared key-validity mask across all MLA layers.
pub const MLA_MASK_INPUT: &str = "state.mla.mask";

/// Byte offsets and widths of everything packed into the decode output.
#[derive(Debug, Clone)]
pub struct DecodeLayout {
    /// Width of the leading logits (or hidden) block.
    pub head_width: usize,
    /// Per KDA layer, in layer order: `(conv_width, scan_width)`.
    pub kda: Vec<(usize, usize, usize)>,
    /// Per MLA layer, in layer order: `(k_width, v_width)`.
    pub mla: Vec<(usize, usize, usize)>,
    /// Total packed width.
    pub total: usize,
}

impl DecodeLayout {
    fn build(cfg: &LingConfig, head_width: usize, scan_mode: ScanState) -> Self {
        let conv_w = (cfg.short_conv_kernel_size - 1) * cfg.kda_proj_dim();
        let scan_w = cfg.num_attention_heads * cfg.head_dim * cfg.head_dim;
        let k_w = cfg.num_attention_heads * cfg.qk_head_dim();
        let v_w = cfg.num_attention_heads * cfg.v_head_dim;
        let mut off = head_width;
        let (mut kda, mut mla) = (Vec::new(), Vec::new());
        for i in 0..cfg.num_hidden_layers {
            match cfg.attn_kind(i) {
                AttnKind::Kda => {
                    let sw = if scan_mode == ScanState::InPlace {
                        0
                    } else {
                        scan_w
                    };
                    kda.push((off, conv_w, sw));
                    off += 3 * conv_w + sw;
                }
                AttnKind::Mla => {
                    mla.push((off, k_w, v_w));
                    off += k_w + v_w;
                }
            }
        }
        Self {
            head_width,
            kda,
            mla,
            total: off,
        }
    }
}

/// `x @ W` with the weight held as **F16** instead of F32.
///
/// Decode is memory-bound, so halving a weight's bytes buys close to half its
/// time. The activation stays f32 and accumulation stays f32 — only the stored
/// weight narrows (Metal dispatches this to `sgemm_f16w`, CPU to
/// `Thunk::SgemmF16` / `sgemm_f16_rhs`).
///
/// **Measured on Ling-3.0-tiny and NOT worth it for `lm_head`** (which is the
/// obvious candidate at 157184 x 1536 = 0.97 GB/token, a third of decode's
/// traffic). Halving those bytes bought only ~2% on Metal and *regressed* CPU by
/// ~16% (Accelerate's f32 sgemm beats the f16-widen path), while visibly degrading
/// greedy output: f32 continues "The capital of Germany is Berlin…", f16 drifts to
/// "Is the capital of France the same capital as…". f16 carries ~5e-4 relative
/// error, and with 157k near-tied logits that flips argmax often enough to derail
/// generation.
///
/// Kept, defaulted off, because the *mechanism* — declaring a typed F16 param from
/// a plugin via `emit.state.typed_params` — is the reusable part, and the next
/// person trying narrow weights (bf16 experts, where the payoff is 1.73 GB/token)
/// should not have to rediscover it.
fn linear_f16(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let key = format!("{prefix}.weight");
    let (data, shape) = emit.weights.take(&key, true)?;
    anyhow::ensure!(
        shape.len() == 2,
        "{key}: expected a 2-D weight, got {shape:?}"
    );
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    drop(data);
    let id = emit
        .hir()
        .param(&key, Shape::new(&[shape[0], shape[1]], DType::F16));
    emit.state.typed_params.push((key, bytes, DType::F16));
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, id))
}

use crate::quant::{Quant, QuantPlan, linear};

fn rmsnorm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{key}.weight"), false)?;
    let zb = emit.synth_param(
        &format!("{key}.dzb"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

fn dense_mlp(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, q: Quant) -> Result<HirNodeId> {
    let gate = linear(emit, &format!("{prefix}.gate_proj"), x, q)?;
    let up = linear(emit, &format!("{prefix}.up_proj"), x, q)?;
    let swiglu = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(gate);
        gb.mul(a, up)
    };
    linear(emit, &format!("{prefix}.down_proj"), swiglu, q)
}

/// Build the single-token decode graph for a cache capacity of `cap` positions.
///
/// `weights` must already have been through [`crate::prepare_checkpoint`].
/// Returns the built model plus the layout describing how to slice its output.
pub fn build_ling_decode_flow(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_ling_decode_flow_with(cfg, weights, cap, with_lm_head, ScanState::Portable)
}

/// As [`build_ling_decode_flow`], choosing where the KDA scan state lives.
pub fn build_ling_decode_flow_with(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
    scan_mode: ScanState,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_ling_decode_flow_opts(cfg, weights, cap, with_lm_head, scan_mode, false)
}

/// As [`build_ling_decode_flow_with`], at a chosen weight precision. See
/// [`crate::quant`]; with [`Quant::MXFP4`] the expert banks are declared by name
/// and streamed in after compile.
pub fn build_ling_decode_flow_quant(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
    scan_mode: ScanState,
    quant: Quant,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_ling_decode_flow_full(
        cfg,
        weights,
        cap,
        with_lm_head,
        scan_mode,
        false,
        quant.into(),
    )
}

/// As [`build_ling_decode_flow_with`], plus `lm_head_f16`: store the LM head as
/// F16. See [`linear_f16`].
pub fn build_ling_decode_flow_opts(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
    scan_mode: ScanState,
    lm_head_f16: bool,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_ling_decode_flow_full(
        cfg,
        weights,
        cap,
        with_lm_head,
        scan_mode,
        lm_head_f16,
        QuantPlan::F32,
    )
}

/// Every decode-graph knob at once. `lm_head_f16` and a non-F32 `quant` are
/// mutually exclusive for the LM head — `quant` wins (MXFP4 is both smaller and,
/// unlike f16, group-scaled, so it does not carry f16's flat 5e-4 relative error
/// across 157k near-tied logits).
#[allow(clippy::too_many_arguments)]
pub fn build_ling_decode_flow_full(
    cfg: &LingConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
    scan_mode: ScanState,
    lm_head_f16: bool,
    plan: QuantPlan,
) -> Result<(BuiltModel, DecodeLayout)> {
    cfg.validate()?;
    let quant = plan.proj;
    let lm_head_f16 = lm_head_f16 && plan.lm_head == Quant::F32;
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let half = cfg.qk_rope_head_dim / 2;
    let head_width = if with_lm_head { cfg.vocab_size } else { hidden };
    let layout = DecodeLayout::build(cfg, head_width, scan_mode);

    let mla_dims = MlaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        q_lora_rank: cfg.q_lora_rank,
        kv_lora_rank: cfg.kv_lora_rank,
        qk_nope_head_dim: cfg.qk_nope_head_dim,
        qk_rope_head_dim: cfg.qk_rope_head_dim,
        v_head_dim: cfg.v_head_dim,
        gate: cfg.attn_gate(),
        eps,
        seq: 1,
        quant,
    };
    let kda_dims = KdaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim,
        conv_kernel: cfg.short_conv_kernel_size,
        no_lora: cfg.no_kda_lora,
        lower_bound: cfg.kda_lower_bound,
        eps,
        seq: 1,
        quant,
    };
    let moe_dims = DeepseekMoeDims {
        hidden,
        moe_inter: cfg.moe_intermediate_size,
        n_routed: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        routed_scaling: cfg.routed_scaling_factor,
        shared_inter: cfg.shared_intermediate_size(),
        seq: 1,
        experts_pretransposed: true,
        mxfp4_group: quant.group_size().map(|g| g as u32),
    };

    let conv_w = (cfg.short_conv_kernel_size - 1) * cfg.kda_proj_dim();
    let (h, hd) = (cfg.num_attention_heads, cfg.head_dim);
    let k_w = h * cfg.qk_head_dim();
    let v_w = h * cfg.v_head_dim;

    let mut flow = ModelFlow::new("ling3_decode")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, 1], f))
        .input(ROPE_COS, Shape::new(&[1, half], f))
        .input(ROPE_SIN, Shape::new(&[1, half], f))
        .zero_beta_named("ling.dec.zero_beta.hidden", hidden);
    for i in 0..cfg.num_hidden_layers {
        match cfg.attn_kind(i) {
            AttnKind::Kda => {
                for w in ["cq", "ck", "cv"] {
                    flow = flow.input(
                        kda_state_input(i, w),
                        Shape::new(&[1, cfg.short_conv_kernel_size - 1, cfg.kda_proj_dim()], f),
                    );
                }
                if scan_mode == ScanState::Portable {
                    flow = flow.input(kda_state_input(i, "scan"), Shape::new(&[1, h, hd, hd], f));
                }
            }
            AttnKind::Mla => {
                flow = flow
                    .input(mla_cache_input(i, "k"), Shape::new(&[1, cap, k_w], f))
                    .input(mla_cache_input(i, "v"), Shape::new(&[1, cap, v_w], f));
            }
        }
    }
    flow = flow
        .input(MLA_MASK_INPUT, Shape::new(&[1, cap + 1], f))
        .embed(EMBED_KEY);

    // One plugin builds the whole stack: the per-layer state outputs have to be
    // collected in one place to be packed into a single graph output.
    let cfg_layers: Vec<AttnKind> = (0..cfg.num_hidden_layers)
        .map(|i| cfg.attn_kind(i))
        .collect();
    let is_moe: Vec<bool> = (0..cfg.num_hidden_layers)
        .map(|i| cfg.is_moe_layer(i))
        .collect();
    let vocab = cfg.vocab_size;
    let total = layout.total;
    flow = flow.plugin_named("decode", move |emit, prev| {
        let mut x = prev
            .ok_or_else(|| anyhow!("decode flow needs the embedding output"))?
            .hir_id();
        let mut packed: Vec<HirNodeId> = Vec::new();
        let mask = emit.flow_input(MLA_MASK_INPUT)?.hir_id();

        for (i, kind) in cfg_layers.iter().enumerate() {
            let prefix = format!("model.layers.{i}");
            let normed = rmsnorm(emit, &format!("{prefix}.input_layernorm"), x, hidden, eps)?;
            let attn_prefix = format!("{prefix}.attention");
            let attn = match kind {
                AttnKind::Kda => {
                    let st = KdaState {
                        conv_q: emit.flow_input(&kda_state_input(i, "cq"))?.hir_id(),
                        conv_k: emit.flow_input(&kda_state_input(i, "ck"))?.hir_id(),
                        conv_v: emit.flow_input(&kda_state_input(i, "cv"))?.hir_id(),
                        scan: match scan_mode {
                            ScanState::Portable => {
                                emit.flow_input(&kda_state_input(i, "scan"))?.hir_id()
                            }
                            ScanState::InPlace => emit.synth_param(
                                &kda_state_input(i, "scan"),
                                vec![0.0; h * hd * hd],
                                Shape::new(&[1, h, hd, hd], f),
                            ),
                        },
                    };
                    let (out, next) = emit_kda_decode(emit, &attn_prefix, normed, st, kda_dims)?;
                    let mut gb = HirMut::new(emit.hir());
                    for n in [next.conv_q, next.conv_k, next.conv_v] {
                        let flat = gb.reshape_(n, vec![1, conv_w as i64]);
                        packed.push(flat);
                    }
                    if scan_mode == ScanState::Portable {
                        // `st.scan` after the op is the updated state on every backend.
                        let scan_flat = gb.reshape_(st.scan, vec![1, (h * hd * hd) as i64]);
                        packed.push(scan_flat);
                    }
                    out
                }
                AttnKind::Mla => {
                    let cache = MlaCache {
                        k: emit.flow_input(&mla_cache_input(i, "k"))?.hir_id(),
                        v: emit.flow_input(&mla_cache_input(i, "v"))?.hir_id(),
                        mask,
                        cap,
                    };
                    let (out, k_new, v_new) =
                        emit_mla_decode(emit, &attn_prefix, normed, cache, mla_dims)?;
                    let mut gb = HirMut::new(emit.hir());
                    let kf = gb.reshape_(k_new, vec![1, k_w as i64]);
                    let vf = gb.reshape_(v_new, vec![1, v_w as i64]);
                    packed.push(kf);
                    packed.push(vf);
                    out
                }
            };
            let h1 = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, attn)
            };
            let normed2 = rmsnorm(
                emit,
                &format!("{prefix}.post_attention_layernorm"),
                h1,
                hidden,
                eps,
            )?;
            let ffn = if is_moe[i] {
                emit_deepseek_moe(emit, &format!("{prefix}.mlp"), normed2, moe_dims)?
            } else {
                dense_mlp(emit, &format!("{prefix}.mlp"), normed2, quant)?
            };
            x = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(h1, ffn)
            };
        }

        let x = rmsnorm(emit, "model.norm", x, hidden, eps)?;
        let head = if with_lm_head {
            let y = if lm_head_f16 {
                linear_f16(emit, "lm_head", x)?
            } else {
                linear(emit, "lm_head", x, plan.lm_head)?
            };
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(y, vec![1, vocab as i64])
        } else {
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(x, vec![1, hidden as i64])
        };

        let mut gb = HirMut::new(emit.hir());
        let mut all = vec![head];
        all.extend(packed);
        let out = gb.concat_(all, 1);
        Ok(Some(emit.wrap(out, Shape::new(&[1, total], f))))
    });

    let built = flow
        .output("packed")
        .build_with(&mut WeightMapSource(weights), None)?;
    Ok((built, layout))
}

/// Host-side decode state: what gets fed in and read back each step.
pub struct DecodeSession {
    pub layout: DecodeLayout,
    cap: usize,
    pos: usize,
    kda: Vec<[Vec<f32>; 4]>, // cq, ck, cv, scan
    mla: Vec<(Vec<f32>, Vec<f32>)>,
    mask: Vec<f32>,
}

impl DecodeSession {
    pub fn new(cfg: &LingConfig, layout: DecodeLayout, cap: usize) -> Self {
        let conv_w = (cfg.short_conv_kernel_size - 1) * cfg.kda_proj_dim();
        // Zero-width when the graph keeps the scan state in a param.
        let scan_w = layout.kda.first().map(|&(_, _, s)| s).unwrap_or(0);
        let k_w = cfg.num_attention_heads * cfg.qk_head_dim();
        let v_w = cfg.num_attention_heads * cfg.v_head_dim;
        let n_kda = layout.kda.len();
        let n_mla = layout.mla.len();
        let mut mask = vec![0f32; cap + 1];
        mask[cap] = 1.0; // the current token is always visible to itself
        Self {
            layout,
            cap,
            pos: 0,
            kda: (0..n_kda)
                .map(|_| {
                    [
                        vec![0f32; conv_w],
                        vec![0f32; conv_w],
                        vec![0f32; conv_w],
                        vec![0f32; scan_w],
                    ]
                })
                .collect(),
            mla: (0..n_mla)
                .map(|_| (vec![0f32; cap * k_w], vec![0f32; cap * v_w]))
                .collect(),
            mask,
        }
    }

    /// Bytes of state moved in+out per token — the portability tax of threading
    /// state through I/O instead of mutating it in place.
    pub fn state_bytes(&self) -> usize {
        let kda: usize = self.kda.iter().flatten().map(|v| v.len()).sum();
        let mla: usize = self.mla.iter().map(|(k, v)| k.len() + v.len()).sum();
        (kda + mla + self.mask.len()) * 4
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Named input bindings for the next `run()`, in the order the graph expects.
    pub fn inputs<'a>(
        &'a self,
        cfg: &LingConfig,
        names: &'a DecodeNames,
    ) -> Vec<(&'a str, &'a [f32])> {
        let mut v: Vec<(&str, &[f32])> = Vec::new();
        let (mut ki, mut mi) = (0usize, 0usize);
        for i in 0..cfg.num_hidden_layers {
            match cfg.attn_kind(i) {
                AttnKind::Kda => {
                    let s = &self.kda[ki];
                    let n_state = if s[3].is_empty() { 3 } else { 4 };
                    for j in 0..n_state {
                        v.push((names.kda[ki][j].as_str(), s[j].as_slice()));
                    }
                    ki += 1;
                }
                AttnKind::Mla => {
                    v.push((names.mla[mi].0.as_str(), self.mla[mi].0.as_slice()));
                    v.push((names.mla[mi].1.as_str(), self.mla[mi].1.as_slice()));
                    mi += 1;
                }
            }
        }
        v.push((MLA_MASK_INPUT, self.mask.as_slice()));
        v
    }

    /// Consume a step's packed output: store the new states and return the
    /// leading logits (or hidden) slice.
    pub fn commit<'a>(&mut self, packed: &'a [f32]) -> Result<&'a [f32]> {
        if packed.len() < self.layout.total {
            anyhow::bail!(
                "decode output is {} wide, expected at least {}",
                packed.len(),
                self.layout.total
            );
        }
        for (li, &(off, conv_w, scan_w)) in self.layout.kda.iter().enumerate() {
            let s = &mut self.kda[li];
            for j in 0..3 {
                s[j].copy_from_slice(&packed[off + j * conv_w..off + (j + 1) * conv_w]);
            }
            if scan_w > 0 {
                let so = off + 3 * conv_w;
                s[3].copy_from_slice(&packed[so..so + scan_w]);
            }
        }
        if self.pos < self.cap {
            for (li, &(off, k_w, v_w)) in self.layout.mla.iter().enumerate() {
                let (kc, vc) = &mut self.mla[li];
                kc[self.pos * k_w..(self.pos + 1) * k_w].copy_from_slice(&packed[off..off + k_w]);
                vc[self.pos * v_w..(self.pos + 1) * v_w]
                    .copy_from_slice(&packed[off + k_w..off + k_w + v_w]);
            }
            self.mask[self.pos] = 1.0;
        }
        self.pos += 1;
        Ok(&packed[..self.layout.head_width])
    }
}

/// Pre-formatted input names, so the hot loop does no string formatting.
pub struct DecodeNames {
    pub kda: Vec<[String; 4]>,
    pub mla: Vec<(String, String)>,
}

impl DecodeNames {
    pub fn new(cfg: &LingConfig) -> Self {
        let mut kda = Vec::new();
        let mut mla = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            match cfg.attn_kind(i) {
                AttnKind::Kda => kda.push([
                    kda_state_input(i, "cq"),
                    kda_state_input(i, "ck"),
                    kda_state_input(i, "cv"),
                    kda_state_input(i, "scan"),
                ]),
                AttnKind::Mla => {
                    mla.push((mla_cache_input(i, "k"), mla_cache_input(i, "v")));
                }
            }
        }
        Self { kda, mla }
    }
}
