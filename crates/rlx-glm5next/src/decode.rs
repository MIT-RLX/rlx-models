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
//! out in one packed **output**, and [`DecodeSession`] feeds it forward.
//! Deliberately not params-with-in-place-update: `Op::GatedDeltaNet` documents
//! that CPU/Metal/wgpu honour a carried state buffer, but MLX substitutes the
//! new state into its evaluation env where it does not survive to the next
//! `run()`, so a param-bound state silently freezes at its initial value there.
//! See [`ScanState`] and [`crate::kda::KdaState`].
//!
//! Layout of the packed output (all f32, concatenated on one axis):
//!
//! ```text
//!   [ logits (vocab)                              ]   (or hidden, if no lm_head)
//!   [ per KDA layer: cq, ck, cv, scan             ]
//!   [ per MLA layer: latent (kv_lora)             ]
//! ```
//!
//! ## The two state budgets are wildly different
//!
//! The MLA cache is tiny because it stores the **latent**, not expanded keys and
//! values — 512 floats per token per layer, 46 MB across 11 layers at
//! `cap = 2048`. See [`crate::mla::MlaCache`].
//!
//! The KDA scan state is the opposite: `num_heads · head_dim² = 64 · 128²` is
//! 1 M floats — **4 MB per KDA layer per token**, 143 MB round-tripped through
//! graph I/O every step across 34 layers. That is the price of portability;
//! [`ScanState::InPlace`] avoids it where the backend supports it, and
//! [`DecodeSession::state_bytes`] reports it.
//!
//! ## Why decode stops at `index_topk`
//!
//! DSA selection is the identity only while the context fits the top-k budget
//! (see [`crate::indexer`]). Past 2048 tokens the indexer genuinely selects, and
//! reproducing that at decode needs a per-layer cache of indexer keys and gate
//! scores plus the reference's dynamic `first_key` walk — neither is emitted
//! here. [`build_glm5next_decode_flow`] therefore *rejects* a capacity beyond
//! the budget rather than quietly running dense attention, which would be a
//! different model from the one the weights were trained for.

use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::rms_norm;
use crate::config::{AttnKind, Glm5NextConfig};
use crate::flow::EMBED_KEY;
use crate::kda::{KdaDims, KdaState, emit_kda_decode};
use crate::mhc::{MhcDims, emit_mhc_expand, emit_mhc_gates, emit_mhc_head, emit_mhc_split};
use crate::mla::{MlaCache, MlaDims, emit_mla_decode};
use crate::moe::{MoeDims, emit_dense_mlp, emit_glm5next_moe};

/// Where the KDA scan state lives between decode steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanState {
    /// Threaded through graph I/O. Correct on every backend, and 4 MB per KDA
    /// layer per token of host traffic.
    Portable,
    /// Kept in a persistent param that `Op::GatedDeltaNet { carry_state }`
    /// updates in place, so it never crosses the host boundary.
    ///
    /// Only valid where that in-place update survives to the next `run()` —
    /// CPU, Metal and wgpu do; **MLX and CoreML do not** (they substitute the
    /// new state into a per-evaluation env), and there the state silently
    /// freezes at zero. The param also starts zeroed at compile time and cannot
    /// be reset without rebuilding, so one compiled graph serves one sequence.
    InPlace,
}

/// Input name for a KDA layer's conv/scan state.
pub fn kda_state_input(layer: usize, which: &str) -> String {
    format!("state.kda{layer}.{which}")
}
/// Input name for an MLA layer's latent cache.
pub fn mla_cache_input(layer: usize) -> String {
    format!("state.mla{layer}.latent")
}
/// Shared key-validity mask across all MLA layers.
pub const MLA_MASK_INPUT: &str = "state.mla.mask";

/// Byte offsets and widths of everything packed into the decode output.
#[derive(Debug, Clone)]
pub struct DecodeLayout {
    /// Width of the leading logits (or hidden) block.
    pub head_width: usize,
    /// Per KDA layer, in layer order: `(offset, conv_width, scan_width)`.
    /// `scan_width` is 0 under [`ScanState::InPlace`].
    pub kda: Vec<(usize, usize, usize)>,
    /// Per MLA layer, in layer order: `(offset, latent_width)`.
    pub mla: Vec<(usize, usize)>,
    /// Total packed width.
    pub total: usize,
}

impl DecodeLayout {
    fn build(cfg: &Glm5NextConfig, head_width: usize, scan_mode: ScanState) -> Self {
        let conv_w = (cfg.linear_conv_kernel_dim - 1) * cfg.kda_proj();
        let scan_w = cfg.linear_num_heads * cfg.linear_head_dim * cfg.linear_head_dim;
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
                AttnKind::MlaDsa => {
                    mla.push((off, cfg.kv_lora_rank));
                    off += cfg.kv_lora_rank;
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

/// Build the single-token decode graph for a cache capacity of `cap` positions.
pub fn build_glm5next_decode_flow(
    cfg: &Glm5NextConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_glm5next_decode_flow_with(cfg, weights, cap, with_lm_head, ScanState::Portable)
}

/// As [`build_glm5next_decode_flow`], choosing where the KDA scan state lives.
pub fn build_glm5next_decode_flow_with(
    cfg: &Glm5NextConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
    scan_mode: ScanState,
) -> Result<(BuiltModel, DecodeLayout)> {
    cfg.validate()?;
    if cfg.with_mtp {
        bail!("glm5next: the MTP block is not emitted by the decode flow yet");
    }
    // See the module docs: past the top-k budget the DSA indexer genuinely
    // selects, and running dense attention instead would be a different model.
    if !cfg.dsa_is_dense(cap + 1) {
        bail!(
            "glm5next: decode capacity {cap} (+1 for the current token) exceeds \
             index_topk = {}, where the DSA indexer stops being the identity. \
             Decoding past that needs the cached-indexer path, which is not \
             emitted yet — running dense attention here would silently be a \
             different model.",
            cfg.index_topk
        );
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let head_width = if with_lm_head { cfg.vocab_size } else { hidden };
    let layout = DecodeLayout::build(cfg, head_width, scan_mode);

    let mhc = MhcDims {
        hidden,
        streams: cfg.hc_mult,
        sinkhorn_iters: cfg.hc_sinkhorn_iters,
        eps: cfg.hc_eps,
        norm_eps: eps,
        seq: 1,
    };
    let kda_dims = KdaDims {
        hidden,
        num_heads: cfg.linear_num_heads,
        head_dim: cfg.linear_head_dim,
        conv_kernel: cfg.linear_conv_kernel_dim,
        lower_bound: cfg.linear_lower_bound,
        eps,
        seq: 1,
    };
    let mla_dims = MlaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        q_lora_rank: cfg.q_lora_rank,
        kv_lora_rank: cfg.kv_lora_rank,
        qk_nope_head_dim: cfg.qk_nope_head_dim,
        v_head_dim: cfg.v_head_dim,
        eps,
        seq: 1,
    };
    let moe_dims = MoeDims {
        paged: false,
        hidden,
        moe_inter: cfg.moe_intermediate_size,
        n_routed: cfg.n_routed_experts,
        top_k: cfg.num_experts_per_tok,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        routed_scaling: cfg.routed_scaling_factor,
        swiglu_limit: Some(cfg.swiglu_limit),
        seq: 1,
    };
    let swiglu = Some(cfg.swiglu_limit);

    let conv_w = (cfg.linear_conv_kernel_dim - 1) * cfg.kda_proj();
    let (kh, khd) = (cfg.linear_num_heads, cfg.linear_head_dim);
    let scan_w = kh * khd * khd;

    let mut flow = ModelFlow::new("glm5next_decode")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, 1], f))
        .zero_beta_named("glm5next.dec.zero_beta.hidden", hidden);
    for i in 0..cfg.num_hidden_layers {
        match cfg.attn_kind(i) {
            AttnKind::Kda => {
                for w in ["cq", "ck", "cv"] {
                    flow = flow.input(
                        kda_state_input(i, w),
                        Shape::new(&[1, cfg.linear_conv_kernel_dim - 1, cfg.kda_proj()], f),
                    );
                }
                if scan_mode == ScanState::Portable {
                    flow = flow.input(
                        kda_state_input(i, "scan"),
                        Shape::new(&[1, kh, khd, khd], f),
                    );
                }
            }
            AttnKind::MlaDsa => {
                flow = flow.input(
                    mla_cache_input(i),
                    Shape::new(&[1, cap, cfg.kv_lora_rank], f),
                );
            }
        }
    }
    flow = flow
        .input(MLA_MASK_INPUT, Shape::new(&[1, cap + 1], f))
        .embed(EMBED_KEY);

    // One plugin builds the whole stack: the per-layer state outputs have to be
    // collected in one place to be packed into a single graph output.
    let kinds: Vec<AttnKind> = (0..cfg.num_hidden_layers)
        .map(|i| cfg.attn_kind(i))
        .collect();
    let is_moe: Vec<bool> = (0..cfg.num_hidden_layers)
        .map(|i| cfg.is_moe_layer(i))
        .collect();
    let total = layout.total;
    let vocab = cfg.vocab_size;
    let kv_lora = cfg.kv_lora_rank;
    flow = flow.plugin_named("decode", move |emit, prev| {
        let embed = prev
            .ok_or_else(|| anyhow!("decode flow needs the embedding output"))?
            .hir_id();
        let mut packed: Vec<HirNodeId> = Vec::new();
        let mask = emit.flow_input(MLA_MASK_INPUT)?.hir_id();

        // Broadcast the embedding into `hc_mult` residual streams, exactly as
        // prefill does — mHC carries no state across tokens, only across depth.
        let mut x = emit_mhc_split(emit, embed, mhc);

        for (i, kind) in kinds.iter().enumerate() {
            let prefix = format!("blk.{i}");

            // ── attention site ──
            let gates = emit_mhc_gates(emit, &format!("{prefix}.hc_attn"), x, mhc)?;
            let normed = rms_norm(
                emit,
                &format!("{prefix}.attn_norm"),
                gates.collapsed,
                hidden,
                eps,
            )?;
            let branch = match kind {
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
                                vec![0.0; scan_w],
                                Shape::new(&[1, kh, khd, khd], f),
                            ),
                        },
                    };
                    let (out, next) = emit_kda_decode(emit, &prefix, normed, st, kda_dims)?;
                    let mut gb = HirMut::new(emit.hir());
                    for n in [next.conv_q, next.conv_k, next.conv_v] {
                        let flat = gb.reshape_(n, vec![1, conv_w as i64]);
                        packed.push(flat);
                    }
                    if scan_mode == ScanState::Portable {
                        // After the op, `st.scan` *is* the updated state on
                        // every backend.
                        let flat = gb.reshape_(st.scan, vec![1, scan_w as i64]);
                        packed.push(flat);
                    }
                    out
                }
                AttnKind::MlaDsa => {
                    let cache = MlaCache {
                        latent: emit.flow_input(&mla_cache_input(i))?.hir_id(),
                        mask,
                        cap,
                    };
                    let (out, latent_new) =
                        emit_mla_decode(emit, &prefix, normed, cache, mla_dims)?;
                    let mut gb = HirMut::new(emit.hir());
                    let flat = gb.reshape_(latent_new, vec![1, kv_lora as i64]);
                    packed.push(flat);
                    out
                }
            };
            x = emit_mhc_expand(emit, branch, x, gates, mhc);

            // ── FFN site ──
            let gates = emit_mhc_gates(emit, &format!("{prefix}.hc_ffn"), x, mhc)?;
            let normed = rms_norm(
                emit,
                &format!("{prefix}.ffn_norm"),
                gates.collapsed,
                hidden,
                eps,
            )?;
            let branch = if is_moe[i] {
                emit_glm5next_moe(emit, &prefix, normed, moe_dims)?
            } else {
                emit_dense_mlp(emit, &prefix, normed, 1, hidden, swiglu)?
            };
            x = emit_mhc_expand(emit, branch, x, gates, mhc);
        }

        let x = emit_mhc_head(emit, x, mhc);
        let x = rms_norm(emit, "output_norm", x, hidden, eps)?;
        let x2d = {
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(x, vec![1, hidden as i64])
        };
        let head = if with_lm_head {
            let y = crate::common::linear(emit, "output.weight", x2d)?;
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(y, vec![1, vocab as i64])
        } else {
            x2d
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
    mla: Vec<Vec<f32>>,      // latent cache, [cap * kv_lora]
    mask: Vec<f32>,
}

impl DecodeSession {
    pub fn new(cfg: &Glm5NextConfig, layout: DecodeLayout, cap: usize) -> Self {
        let conv_w = (cfg.linear_conv_kernel_dim - 1) * cfg.kda_proj();
        // Zero-width when the graph keeps the scan state in a param.
        let scan_w = layout.kda.first().map(|&(_, _, s)| s).unwrap_or(0);
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
                .map(|_| vec![0f32; cap * cfg.kv_lora_rank])
                .collect(),
            mask,
        }
    }

    /// Bytes of state moved in+out per token — the portability tax of threading
    /// state through I/O instead of mutating it in place.
    pub fn state_bytes(&self) -> usize {
        let kda: usize = self.kda.iter().flatten().map(|v| v.len()).sum();
        let mla: usize = self.mla.iter().map(|v| v.len()).sum();
        (kda + mla + self.mask.len()) * 4
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Named input bindings for the next `run()`, in the order the graph expects.
    pub fn inputs<'a>(
        &'a self,
        cfg: &Glm5NextConfig,
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
                AttnKind::MlaDsa => {
                    v.push((names.mla[mi].as_str(), self.mla[mi].as_slice()));
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
            bail!(
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
            for (li, &(off, w)) in self.layout.mla.iter().enumerate() {
                self.mla[li][self.pos * w..(self.pos + 1) * w]
                    .copy_from_slice(&packed[off..off + w]);
            }
            self.mask[self.pos] = 1.0;
        } else {
            bail!(
                "decode ran past the compiled cache capacity of {} positions",
                self.cap
            );
        }
        self.pos += 1;
        Ok(&packed[..self.layout.head_width])
    }
}

/// Pre-formatted input names, so the hot loop does no string formatting.
pub struct DecodeNames {
    pub kda: Vec<[String; 4]>,
    pub mla: Vec<String>,
}

impl DecodeNames {
    pub fn new(cfg: &Glm5NextConfig) -> Self {
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
                AttnKind::MlaDsa => mla.push(mla_cache_input(i)),
            }
        }
        Self { kda, mla }
    }
}
