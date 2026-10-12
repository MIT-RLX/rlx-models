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
//! new state into its per-evaluation environment where it does not survive to
//! the next `run()`, so a param-bound state silently freezes at its initial
//! value there. [`ScanState::InPlace`] is the opt-in for backends that do.
//!
//! Layout of the packed output (all f32, concatenated on one axis):
//!
//! ```text
//!   [ logits (vocab)                               ]   (or hidden, or the hc residual)
//!   [ per linear layer: conv, scan                 ]
//!   [ per attention layer: k row, v row            ]
//!   [ the PLE layer's conv history                 ]
//! ```
//!
//! ## This model's state budget is dominated by one thing
//!
//! | piece | per layer | × layers | per token |
//! |---|--:|--:|--:|
//! | GatedDeltaNet scan (`48 · 128²`) | 786 432 f32 | 36 | **113 MB** |
//! | GatedDeltaNet conv (`3 · 10240`) | 30 720 f32 | 36 | 4.4 MB |
//! | attention k+v row (`2 · 512`) | 1 024 f32 | 12 | 49 KB |
//! | PLE conv (`9 · 10240`) | 92 160 f32 | 1 | 369 KB |
//!
//! The scan state is 96% of it, and that is the price of routing state through
//! graph I/O rather than mutating it in place. [`DecodeSession::state_bytes`]
//! reports the total; [`ScanState::InPlace`] removes the scan half of it where
//! the backend supports that.
//!
//! ## Three pieces of state, not two
//!
//! A PLE layer keeps **three** cache rows, which is what
//! `Qwen4ExpTextConfig.number_of_conv_states = 3` counts: the GatedDeltaNet
//! convolution window, the PLE convolution window, and the trailing
//! `ngram_size − 1` **token ids** the hash needs. The third is not a tensor —
//! it lives on the host in [`DecodeSession`], because the hash runs there (see
//! [`crate::ple`]). Forgetting it makes every n-gram at a chunk boundary hash
//! as if the sequence had just started.
//!
//! ## Why decode stops at the QSA budget
//!
//! QSA selection is the identity only while the context fits the block budget
//! (see [`crate::indexer`]). Past 2048 tokens the indexer genuinely selects, and
//! reproducing that at decode needs a per-layer cache of indexer keys kept in
//! *position* order — which the append-at-the-end KV cache deliberately is not
//! (see [`crate::attention::AttnCache`]). [`build_qwen4exp_decode_flow`]
//! therefore *rejects* a capacity beyond the budget rather than quietly running
//! dense attention, which would be a different model from the one the weights
//! were trained for.

use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, WeightSource};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::{AttnCache, AttnDims, emit_attention_with_cache};
use crate::common::ChunkInputs;
use crate::config::{LayerKind, PleHash, Qwen4ExpConfig};
use crate::flow::{EMBED_KEY, OutputKind, ROPE_COS_INPUT, ROPE_SIN_INPUT};
use crate::gdn::{GdnDims, GdnState, emit_gdn_with_state};
use crate::hc::{HcDims, emit_hc_combine, emit_hc_mix, emit_hc_split};
use crate::indexer::QsaDims;
use crate::moe::{MoeDims, emit_moe};
use crate::ple::{PleDims, emit_ple_with_state, ple_embed_input};

/// Where the GatedDeltaNet scan state lives between decode steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanState {
    /// Threaded through graph I/O. Correct on every backend, and 3.1 MB per
    /// linear layer per token of host traffic.
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

/// Input name for a linear-attention layer's conv or scan state.
pub fn gdn_state_input(layer: usize, which: &str) -> String {
    format!("state.gdn{layer}.{which}")
}
/// Input name for an attention layer's key or value cache.
pub fn attn_cache_input(layer: usize, which: &str) -> String {
    format!("state.attn{layer}.{which}")
}
/// Input name for a PLE layer's convolution history.
pub fn ple_state_input(layer: usize) -> String {
    format!("state.ple{layer}.conv")
}
/// Shared key-validity mask across all attention layers, `[1, cap + chunk]`.
pub const ATTN_MASK_INPUT: &str = "state.attn.mask";
/// `[1, chunk]` — `1.0` real, `0.0` right-padding. See
/// [`crate::common::ChunkInputs`].
pub const PAD_MASK_INPUT: &str = "pad_mask";
/// `[1, 1]` — `real_len - chunk`, so `<= 0`. See
/// [`crate::common::ChunkInputs`].
pub const CONV_SHIFT_INPUT: &str = "conv_shift";

/// Byte offsets and widths of everything packed into the decode output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeLayout {
    /// Width of the leading logits / hidden / residual block.
    pub head_width: usize,
    /// Per linear layer, in layer order: `(offset, conv_width, scan_width)`.
    /// `scan_width` is 0 under [`ScanState::InPlace`].
    pub gdn: Vec<(usize, usize, usize)>,
    /// Per attention layer, in layer order: `(offset, kv_width)` — the `k` row
    /// is at `offset`, the `v` row at `offset + kv_width`.
    pub attn: Vec<(usize, usize)>,
    /// Per PLE layer, in `ple_layers` order: `(offset, conv_width)`.
    pub ple: Vec<(usize, usize)>,
    /// Total packed width.
    pub total: usize,
}

impl DecodeLayout {
    fn build(
        cfg: &Qwen4ExpConfig,
        per_token_head: usize,
        chunk: usize,
        scan_mode: ScanState,
    ) -> Self {
        let head_width = per_token_head * chunk;
        let conv_w = (cfg.ssm_conv_kernel - 1) * cfg.gdn_conv_dim();
        let scan_w = cfg.n_v_heads() * cfg.ssm_state_size * cfg.ssm_state_size;
        // One row per token in the chunk — a chunk of N appends N cache cells.
        let kv_w = cfg.num_key_value_heads * cfg.head_dim * chunk;
        let ple_w = cfg.ple_conv_state_len() * cfg.hc_dim();
        let mut off = head_width;
        let (mut gdn, mut attn, mut ple) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..cfg.num_hidden_layers {
            if cfg.ple_layer_index(i).is_some() {
                ple.push((off, ple_w));
                off += ple_w;
            }
            match cfg.layer_kind(i) {
                LayerKind::Linear => {
                    let sw = if scan_mode == ScanState::InPlace {
                        0
                    } else {
                        scan_w
                    };
                    gdn.push((off, conv_w, sw));
                    off += conv_w + sw;
                }
                LayerKind::Indexed => {
                    attn.push((off, kv_w));
                    off += 2 * kv_w;
                }
            }
        }
        Self {
            head_width,
            gdn,
            attn,
            ple,
            total: off,
        }
    }
}

/// Build the single-token decode graph for a cache capacity of `cap` positions.
pub fn build_qwen4exp_decode_flow(
    cfg: &Qwen4ExpConfig,
    weights: &mut WeightMap,
    cap: usize,
    with_lm_head: bool,
) -> Result<(BuiltModel, DecodeLayout)> {
    let out = if with_lm_head {
        OutputKind::Logits
    } else {
        OutputKind::Hidden
    };
    build_qwen4exp_decode_flow_with(
        cfg,
        &mut WeightMapSource(weights),
        cap,
        out,
        ScanState::Portable,
    )
}

/// As [`build_qwen4exp_decode_flow`], over any [`WeightSource`] and choosing
/// where the scan state lives and what the head produces.
pub fn build_qwen4exp_decode_flow_with(
    cfg: &Qwen4ExpConfig,
    weights: &mut dyn WeightSource,
    cap: usize,
    output: OutputKind,
    scan_mode: ScanState,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_qwen4exp_step_flow(cfg, weights, cap, 1, output, scan_mode)
}

/// Build a graph that advances the model by `chunk` tokens against a cache of
/// `cap` already-seen positions.
///
/// `chunk = 1` is decode. `chunk > 1` is **chunked prefill**: a long prompt is
/// consumed `chunk` tokens at a time, which bounds the activation memory (and
/// the `[chunk, cap+chunk]` attention bias) instead of scaling it with the
/// whole prompt. The two are one builder on purpose — every stateful piece has
/// a different spelling from the stateless prefill, and sharing the code means
/// `tests/decode_equivalence.rs` covers both at once.
///
/// A short final chunk is handled by padding on the **right** and binding
/// [`PAD_MASK_INPUT`] / [`CONV_SHIFT_INPUT`]; see [`crate::common::ChunkInputs`]
/// for why only the exported state needs them.
///
/// [`OutputKind::HcResidual`] is what a speculative loop wants: the MTP block
/// reads the trunk's hyper-connection residual, *not* the collapsed hidden
/// state (see [`crate::mtp`]).
pub fn build_qwen4exp_step_flow(
    cfg: &Qwen4ExpConfig,
    weights: &mut dyn WeightSource,
    cap: usize,
    chunk: usize,
    output: OutputKind,
    scan_mode: ScanState,
) -> Result<(BuiltModel, DecodeLayout)> {
    build_qwen4exp_step_flow_with(cfg, weights, cap, chunk, output, scan_mode, false)
}

/// [`build_qwen4exp_step_flow`], optionally taking `inputs_embeds` instead of
/// gathering the embedding table in the graph — the multimodal seam, and the
/// way to decode without a 1.2 GB resident `token_embd`. See
/// [`crate::FlowOptions::host_embed`].
#[allow(clippy::too_many_arguments)]
pub fn build_qwen4exp_step_flow_with(
    cfg: &Qwen4ExpConfig,
    weights: &mut dyn WeightSource,
    cap: usize,
    chunk: usize,
    output: OutputKind,
    scan_mode: ScanState,
    host_embed: bool,
) -> Result<(BuiltModel, DecodeLayout)> {
    cfg.validate()?;
    if chunk == 0 {
        bail!("qwen4exp: chunk must be >= 1");
    }
    if cfg.with_mtp {
        bail!(
            "qwen4exp: the MTP block is built by `mtp::build_qwen4exp_mtp_flow`, \
             not by the trunk decode flow"
        );
    }
    // See the module docs: past the budget the QSA indexer genuinely selects,
    // and running dense attention instead would be a different model.
    if !cfg.qsa_is_dense(cap + chunk) {
        bail!(
            "qwen4exp: capacity {cap} (+{chunk} for this chunk) exceeds \
             indexer_budget = {}, where the QSA indexer stops being the \
             identity. Decoding past that needs a position-ordered indexer key \
             cache, which is not emitted yet — running dense attention here \
             would silently be a different model.",
            cfg.indexer_budget
        );
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let rope_width = crate::rope::table_width(cfg);
    let per_token_head = match output {
        OutputKind::Logits => cfg.vocab_size,
        OutputKind::Hidden => hidden,
        OutputKind::HcResidual => cfg.hc_dim(),
    };
    let layout = DecodeLayout::build(cfg, per_token_head, chunk, scan_mode);

    let hc = HcDims::from_config(cfg, chunk);
    let gdn = GdnDims::from_config(cfg, chunk);
    let attn = AttnDims::from_config(cfg, chunk);
    // Passed so the layer can assert the selection really is the identity here;
    // `emit_attention_with_cache` refuses a cache otherwise.
    // The indexer sees the whole cache plus this step, not just the step.
    let qsa = QsaDims::from_config(cfg, cap + chunk);
    let moe = MoeDims::from_config(cfg, chunk);
    let ple = PleDims::from_config(cfg, chunk);

    let conv_w = (cfg.ssm_conv_kernel - 1) * cfg.gdn_conv_dim();
    let scan_w = cfg.n_v_heads() * cfg.ssm_state_size * cfg.ssm_state_size;
    let kv_row = cfg.num_key_value_heads * cfg.head_dim;
    let kv_w = kv_row * chunk;
    let ple_w = cfg.ple_conv_state_len() * cfg.hc_dim();

    let mut flow = ModelFlow::new("qwen4exp_step")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, chunk], f))
        // A continued run's positions are never `0..seq`, so the M-RoPE tables
        // are always inputs here.
        .input(ROPE_COS_INPUT, Shape::new(&[chunk, rope_width], f))
        .input(ROPE_SIN_INPUT, Shape::new(&[chunk, rope_width], f))
        .input(PAD_MASK_INPUT, Shape::new(&[1, chunk], f))
        .input(CONV_SHIFT_INPUT, Shape::new(&[1, 1], f));
    for idx in 0..cfg.ple_layers.len() {
        flow = flow.input(
            ple_embed_input(idx),
            Shape::new(&[chunk, cfg.ple_embed_dim()], f),
        );
    }
    for i in 0..cfg.num_hidden_layers {
        if cfg.ple_layer_index(i).is_some() {
            flow = flow.input(
                ple_state_input(i),
                Shape::new(&[1, cfg.ple_conv_state_len(), cfg.hc_dim()], f),
            );
        }
        match cfg.layer_kind(i) {
            LayerKind::Linear => {
                flow = flow.input(
                    gdn_state_input(i, "conv"),
                    Shape::new(&[1, cfg.ssm_conv_kernel - 1, cfg.gdn_conv_dim()], f),
                );
                if scan_mode == ScanState::Portable {
                    flow = flow.input(
                        gdn_state_input(i, "scan"),
                        Shape::new(
                            &[1, cfg.n_v_heads(), cfg.ssm_state_size, cfg.ssm_state_size],
                            f,
                        ),
                    );
                }
            }
            LayerKind::Indexed => {
                for which in ["k", "v"] {
                    flow = flow.input(attn_cache_input(i, which), Shape::new(&[1, cap, kv_row], f));
                }
            }
        }
    }
    flow = flow.input(ATTN_MASK_INPUT, Shape::new(&[1, cap + chunk], f));
    flow = if host_embed {
        flow.embed_host(EMBED_KEY, hidden)
    } else {
        flow.embed(EMBED_KEY)
    };

    // One plugin builds the whole stack: the per-layer state outputs have to be
    // collected in one place to be packed into a single graph output.
    let kinds: Vec<LayerKind> = (0..cfg.num_hidden_layers)
        .map(|i| cfg.layer_kind(i))
        .collect();
    let ple_idx: Vec<Option<usize>> = (0..cfg.num_hidden_layers)
        .map(|i| cfg.ple_layer_index(i))
        .collect();
    let total = layout.total;
    let vocab = cfg.vocab_size;
    let hc_dim = cfg.hc_dim();
    let (nv, st) = (cfg.n_v_heads(), cfg.ssm_state_size);

    flow = flow.plugin_named("decode", move |emit, prev| {
        let embed = prev
            .ok_or_else(|| anyhow!("decode flow needs the embedding output"))?
            .hir_id();
        let mut packed: Vec<HirNodeId> = Vec::new();
        let mask = emit.flow_input(ATTN_MASK_INPUT)?.hir_id();
        let cos = emit.flow_input(ROPE_COS_INPUT)?.hir_id();
        let sin = emit.flow_input(ROPE_SIN_INPUT)?.hir_id();
        let chunk_in = ChunkInputs {
            pad_mask: Some(emit.flow_input(PAD_MASK_INPUT)?.hir_id()),
            conv_shift: Some(emit.flow_input(CONV_SHIFT_INPUT)?.hir_id()),
        };

        // Hyper-connections carry no state across tokens, only across depth, so
        // the split is exactly prefill's.
        let mut x = emit_hc_split(emit, embed, hc);

        for (i, kind) in kinds.iter().enumerate() {
            let prefix = format!("blk.{i}");

            if let Some(pi) = ple_idx[i] {
                let emb = emit.flow_input(&ple_embed_input(pi))?.hir_id();
                let st_in = emit.flow_input(&ple_state_input(i))?.hir_id();
                let (out, next) =
                    emit_ple_with_state(emit, &prefix, x, emb, ple, Some(st_in), chunk_in)?;
                x = out;
                let next = next.expect("a PLE state in gives a state out");
                let mut gb = HirMut::new(emit.hir());
                packed.push(gb.reshape_(next, vec![1, ple_w as i64]));
            }

            // ── token mixer ──
            let gates = emit_hc_mix(emit, &format!("{prefix}.hc_attn"), x, hc, true)?;
            let branch = match kind {
                LayerKind::Linear => {
                    let state = GdnState {
                        conv: emit.flow_input(&gdn_state_input(i, "conv"))?.hir_id(),
                        scan: match scan_mode {
                            ScanState::Portable => {
                                emit.flow_input(&gdn_state_input(i, "scan"))?.hir_id()
                            }
                            ScanState::InPlace => emit.synth_param(
                                &gdn_state_input(i, "scan"),
                                vec![0.0; scan_w],
                                Shape::new(&[1, nv, st, st], f),
                            ),
                        },
                    };
                    let (out, next) = emit_gdn_with_state(
                        emit,
                        &prefix,
                        gates.mixed,
                        gdn,
                        Some(state),
                        chunk_in,
                    )?;
                    let next = next.expect("a GDN state in gives a state out");
                    let mut gb = HirMut::new(emit.hir());
                    packed.push(gb.reshape_(next.conv, vec![1, conv_w as i64]));
                    if scan_mode == ScanState::Portable {
                        packed.push(gb.reshape_(next.scan, vec![1, scan_w as i64]));
                    }
                    out
                }
                LayerKind::Indexed => {
                    let cache = AttnCache {
                        k: emit.flow_input(&attn_cache_input(i, "k"))?.hir_id(),
                        v: emit.flow_input(&attn_cache_input(i, "v"))?.hir_id(),
                        mask,
                        cap,
                    };
                    let (out, rows) = emit_attention_with_cache(
                        emit,
                        &prefix,
                        gates.mixed,
                        cos,
                        sin,
                        attn,
                        Some(qsa),
                        Some(cache),
                    )?;
                    let rows = rows.expect("a cache in gives cache rows out");
                    let mut gb = HirMut::new(emit.hir());
                    packed.push(gb.reshape_(rows.k, vec![1, kv_w as i64]));
                    packed.push(gb.reshape_(rows.v, vec![1, kv_w as i64]));
                    out
                }
            };
            let x_mid = emit_hc_combine(emit, branch, x, gates, hc);

            // ── MoE ──
            let gates = emit_hc_mix(emit, &format!("{prefix}.hc_ffn"), x_mid, hc, true)?;
            let branch = emit_moe(emit, &prefix, gates.mixed, moe)?;
            x = emit_hc_combine(emit, branch, x_mid, gates, hc);
        }

        // The final mixer collapses the streams AND is the output norm.
        let head = match output {
            OutputKind::HcResidual => {
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(x, vec![1, (chunk * hc_dim) as i64])
            }
            OutputKind::Hidden => {
                let gates = emit_hc_mix(emit, "output_hc", x, hc, false)?;
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(gates.mixed, vec![1, (chunk * hidden) as i64])
            }
            OutputKind::Logits => {
                let gates = emit_hc_mix(emit, "output_hc", x, hc, false)?;
                let x2d = {
                    let mut gb = HirMut::new(emit.hir());
                    gb.reshape_(gates.mixed, vec![chunk as i64, hidden as i64])
                };
                let y = crate::common::linear(emit, "output.weight", x2d)?;
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(y, vec![1, (chunk * vocab) as i64])
            }
        };

        let mut gb = HirMut::new(emit.hir());
        let mut all = vec![head];
        all.extend(packed);
        let out = gb.concat_(all, 1);
        Ok(Some(emit.wrap(out, Shape::new(&[1, total], f))))
    });

    let built = flow.output("packed").build_with(weights, None)?;
    Ok((built, layout))
}

/// Host-side decode state: what gets fed in and read back each step.
pub struct DecodeSession {
    pub layout: DecodeLayout,
    cap: usize,
    chunk: usize,
    pos: usize,
    /// `[1, chunk]` validity row and the `[1, 1]` conv shift for this step.
    pad: Vec<f32>,
    shift: [f32; 1],
    /// `(conv, scan)` per linear layer.
    gdn: Vec<(Vec<f32>, Vec<f32>)>,
    /// `(k, v)` per attention layer, each `[cap * kv_width]`.
    attn: Vec<(Vec<f32>, Vec<f32>)>,
    /// Convolution history per PLE layer.
    ple: Vec<Vec<f32>>,
    mask: Vec<f32>,
    /// The trailing `ngram_size - 1` token ids, oldest first — the third cache
    /// row a PLE layer needs, and the only one that is not a tensor.
    history: Vec<u32>,
    ngram_context: usize,
}

impl DecodeSession {
    /// A single-token decode session.
    pub fn new(cfg: &Qwen4ExpConfig, layout: DecodeLayout, cap: usize) -> Self {
        Self::with_chunk(cfg, layout, cap, 1)
    }

    /// A session for a graph built with [`build_qwen4exp_step_flow`]'s `chunk`.
    pub fn with_chunk(
        cfg: &Qwen4ExpConfig,
        layout: DecodeLayout,
        cap: usize,
        chunk: usize,
    ) -> Self {
        let conv_w = (cfg.ssm_conv_kernel - 1) * cfg.gdn_conv_dim();
        // Zero-width when the graph keeps the scan state in a param.
        let scan_w = layout.gdn.first().map(|&(_, _, s)| s).unwrap_or(0);
        let kv_row = cfg.num_key_value_heads * cfg.head_dim;
        let ple_w = cfg.ple_conv_state_len() * cfg.hc_dim();
        let mut mask = vec![0f32; cap + chunk];
        // This step's own tokens are always live; causality among them is the
        // graph's `[chunk, cap+chunk]` bias, not this row.
        for m in mask[cap..].iter_mut() {
            *m = 1.0;
        }
        Self {
            cap,
            chunk,
            pos: 0,
            pad: vec![1.0; chunk],
            shift: [0.0],
            gdn: (0..layout.gdn.len())
                .map(|_| (vec![0f32; conv_w], vec![0f32; scan_w]))
                .collect(),
            attn: (0..layout.attn.len())
                .map(|_| (vec![0f32; cap * kv_row], vec![0f32; cap * kv_row]))
                .collect(),
            ple: (0..layout.ple.len()).map(|_| vec![0f32; ple_w]).collect(),
            mask,
            history: Vec::new(),
            ngram_context: cfg.ple_ngram_size.saturating_sub(1),
            layout,
        }
    }

    /// Bytes of state moved in+out per token — the portability tax of threading
    /// state through I/O instead of mutating it in place.
    pub fn state_bytes(&self) -> usize {
        let gdn: usize = self.gdn.iter().map(|(c, s)| c.len() + s.len()).sum();
        let attn: usize = self.attn.iter().map(|(k, v)| k.len() + v.len()).sum();
        let ple: usize = self.ple.iter().map(|v| v.len()).sum();
        (gdn + attn + ple + self.mask.len()) * 4
    }

    /// Tokens this session's graph advances per `run()`.
    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// Re-point this session at a graph with a different `chunk`, keeping the
    /// state it has accumulated.
    ///
    /// This is what lets one session serve a chunked prefill and then a
    /// one-token-at-a-time decode: the recurrent state (GatedDeltaNet conv and
    /// scan, the attention cache, the PLE convolution window, the n-gram token
    /// history) does not depend on how many positions a step advances. What
    /// does is the *packed output layout* and the two chunk-shaped inputs, so
    /// only those are rebuilt. Pass the [`DecodeLayout`] that
    /// [`build_qwen4exp_step_flow`] returned for `chunk`.
    pub fn set_chunk(&mut self, layout: DecodeLayout, chunk: usize) -> Result<()> {
        if chunk == 0 {
            bail!("qwen4exp: chunk must be >= 1");
        }
        // A mismatched layout would parse the next `commit` at the wrong
        // offsets and quietly corrupt the cache, so check it against `chunk`
        // rather than trusting the pairing.
        if let Some(&(_, kv_w)) = layout.attn.first()
            && !kv_w.is_multiple_of(chunk)
        {
            bail!(
                "qwen4exp: a layout whose kv width is {kv_w} does not belong to \
                 a chunk of {chunk}"
            );
        }
        if !layout.head_width.is_multiple_of(chunk) {
            bail!(
                "qwen4exp: a layout whose head width is {} does not belong to a \
                 chunk of {chunk}",
                layout.head_width
            );
        }
        let mut mask = vec![0f32; self.cap + chunk];
        mask[..self.cap].copy_from_slice(&self.mask[..self.cap]);
        for m in mask[self.cap..].iter_mut() {
            *m = 1.0;
        }
        self.mask = mask;
        self.pad = vec![1.0; chunk];
        self.shift = [0.0];
        self.chunk = chunk;
        self.layout = layout;
        Ok(())
    }

    /// Declare how many of the next step's `chunk` tokens are real.
    ///
    /// Only a short final chunk needs this. It drives
    /// [`PAD_MASK_INPUT`] and [`CONV_SHIFT_INPUT`], which exist solely to keep
    /// the exported state from advancing over padding — see
    /// [`crate::common::ChunkInputs`].
    pub fn set_real_len(&mut self, real: usize) -> Result<()> {
        if real == 0 || real > self.chunk {
            bail!(
                "qwen4exp: real_len {real} must be in 1..={} for this graph",
                self.chunk
            );
        }
        for (i, p) in self.pad.iter_mut().enumerate() {
            *p = if i < real { 1.0 } else { 0.0 };
        }
        self.shift[0] = real as f32 - self.chunk as f32;
        // A padded key must not be attended by a later chunk either.
        for (i, m) in self.mask[self.cap..].iter_mut().enumerate() {
            *m = if i < real { 1.0 } else { 0.0 };
        }
        Ok(())
    }

    /// How many real tokens the next step will consume.
    pub fn real_len(&self) -> usize {
        self.pad.iter().filter(|p| **p > 0.5).count()
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// The trailing token window the PLE hash needs, oldest first.
    ///
    /// Pass this to [`crate::ple::ngram_rows_with_history`] (or
    /// [`Self::ple_embeddings`]) when hashing the next token: without it, every
    /// decoded token hashes as if it began a fresh sequence.
    pub fn ngram_history(&self) -> &[u32] {
        &self.history
    }

    /// Gather the `ple_embeddings.{i}` input for `token` at this position.
    ///
    /// Uses [`Self::ngram_history`], so stepping a decode reproduces the row
    /// indices a single-shot prefill of the same tokens would compute.
    pub fn ple_embeddings(
        &self,
        cfg: &Qwen4ExpConfig,
        ple_layer_index: usize,
        token: u32,
        table: &mut dyn crate::ple::PleRowSource,
    ) -> Result<Vec<f32>> {
        let hash = PleHash::derive(cfg, ple_layer_index);
        crate::ple::ngram_embeddings(&hash, &[token], &self.history, cfg.ple_head_dim, |r| {
            table.row(r)
        })
    }

    /// Record the token being decoded, so the next step's hash sees it.
    ///
    /// Call this once per step, before or after `run()` but exactly once; the
    /// embeddings for *this* step must be gathered with the history as it was
    /// **before** the call.
    pub fn push_token(&mut self, token: u32) {
        if self.ngram_context == 0 {
            return;
        }
        self.history.push(token);
        let excess = self.history.len().saturating_sub(self.ngram_context);
        if excess > 0 {
            self.history.drain(..excess);
        }
    }

    /// Named input bindings for the next `run()`.
    pub fn inputs<'a>(
        &'a self,
        cfg: &Qwen4ExpConfig,
        names: &'a DecodeNames,
    ) -> Vec<(&'a str, &'a [f32])> {
        let mut v: Vec<(&str, &[f32])> = Vec::new();
        let (mut gi, mut ai, mut pi) = (0usize, 0usize, 0usize);
        for i in 0..cfg.num_hidden_layers {
            if cfg.ple_layer_index(i).is_some() {
                v.push((names.ple[pi].as_str(), self.ple[pi].as_slice()));
                pi += 1;
            }
            match cfg.layer_kind(i) {
                LayerKind::Linear => {
                    let (c, s) = &self.gdn[gi];
                    v.push((names.gdn[gi][0].as_str(), c.as_slice()));
                    if !s.is_empty() {
                        v.push((names.gdn[gi][1].as_str(), s.as_slice()));
                    }
                    gi += 1;
                }
                LayerKind::Indexed => {
                    let (k, val) = &self.attn[ai];
                    v.push((names.attn[ai][0].as_str(), k.as_slice()));
                    v.push((names.attn[ai][1].as_str(), val.as_slice()));
                    ai += 1;
                }
            }
        }
        v.push((ATTN_MASK_INPUT, self.mask.as_slice()));
        v.push((PAD_MASK_INPUT, self.pad.as_slice()));
        v.push((CONV_SHIFT_INPUT, self.shift.as_slice()));
        v
    }

    /// Consume a step's packed output: store the new states and return the
    /// leading logits / hidden / residual slice.
    pub fn commit<'a>(&mut self, packed: &'a [f32]) -> Result<&'a [f32]> {
        if packed.len() < self.layout.total {
            bail!(
                "decode output is {} wide, expected at least {}",
                packed.len(),
                self.layout.total
            );
        }
        let real = self.real_len();
        if self.pos + real > self.cap {
            bail!(
                "decode ran past the compiled cache capacity of {} positions \
                 ({} stored + {real} new)",
                self.cap,
                self.pos
            );
        }
        for (li, &(off, conv_w, scan_w)) in self.layout.gdn.iter().enumerate() {
            let (c, s) = &mut self.gdn[li];
            c.copy_from_slice(&packed[off..off + conv_w]);
            if scan_w > 0 {
                s.copy_from_slice(&packed[off + conv_w..off + conv_w + scan_w]);
            }
        }
        for (li, &(off, w)) in self.layout.ple.iter().enumerate() {
            self.ple[li].copy_from_slice(&packed[off..off + w]);
        }
        // The cache rows for this step's tokens go to slots `pos..pos+real`,
        // and the mask turns them on. Storing at `pos` rather than appending at
        // the end is what keeps the cache in position order for a future
        // indexer, even though attention itself does not need it to be.
        for (li, &(off, w)) in self.layout.attn.iter().enumerate() {
            let row = w / self.chunk;
            let (k, v) = &mut self.attn[li];
            let dst = self.pos * row;
            k[dst..dst + real * row].copy_from_slice(&packed[off..off + real * row]);
            v[dst..dst + real * row].copy_from_slice(&packed[off + w..off + w + real * row]);
        }
        for t in 0..real {
            self.mask[self.pos + t] = 1.0;
        }
        self.pos += real;
        Ok(&packed[..self.layout.head_width])
    }
}

/// Pre-formatted input names, so the hot loop does no string formatting.
pub struct DecodeNames {
    /// `[conv, scan]` per linear layer.
    pub gdn: Vec<[String; 2]>,
    /// `[k, v]` per attention layer.
    pub attn: Vec<[String; 2]>,
    /// One per PLE layer.
    pub ple: Vec<String>,
}

impl DecodeNames {
    pub fn new(cfg: &Qwen4ExpConfig) -> Self {
        let mut gdn = Vec::new();
        let mut attn = Vec::new();
        let mut ple = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            if cfg.ple_layer_index(i).is_some() {
                ple.push(ple_state_input(i));
            }
            match cfg.layer_kind(i) {
                LayerKind::Linear => {
                    gdn.push([gdn_state_input(i, "conv"), gdn_state_input(i, "scan")])
                }
                LayerKind::Indexed => {
                    attn.push([attn_cache_input(i, "k"), attn_cache_input(i, "v")])
                }
            }
        }
        Self { gdn, attn, ple }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::tiny_config;

    #[test]
    fn the_layout_accounts_for_every_byte() {
        let cfg = tiny_config(2048);
        let layout = DecodeLayout::build(&cfg, cfg.vocab_size, 1, ScanState::Portable);
        let conv_w = (cfg.ssm_conv_kernel - 1) * cfg.gdn_conv_dim();
        let scan_w = cfg.n_v_heads() * cfg.ssm_state_size * cfg.ssm_state_size;
        let kv_w = cfg.num_key_value_heads * cfg.head_dim;
        let ple_w = cfg.ple_conv_state_len() * cfg.hc_dim();
        let want = cfg.vocab_size
            + layout.gdn.len() * (conv_w + scan_w)
            + layout.attn.len() * 2 * kv_w
            + layout.ple.len() * ple_w;
        assert_eq!(layout.total, want);
        assert_eq!(layout.gdn.len(), 3);
        assert_eq!(layout.attn.len(), 1);
        assert_eq!(layout.ple.len(), 1);
    }

    /// Nothing may overlap: two layers writing the same slice of the packed
    /// output would make one of them silently read the other's state.
    #[test]
    fn packed_slices_are_disjoint_and_contiguous() {
        let cfg = tiny_config(2048);
        let layout = DecodeLayout::build(&cfg, 7, 1, ScanState::Portable);
        let mut spans: Vec<(usize, usize)> = vec![(0, 7)];
        for &(off, c, s) in &layout.gdn {
            spans.push((off, c + s));
        }
        for &(off, w) in &layout.attn {
            spans.push((off, 2 * w));
        }
        for &(off, w) in &layout.ple {
            spans.push((off, w));
        }
        spans.sort();
        let mut cursor = 0;
        for (off, len) in spans {
            assert_eq!(off, cursor, "gap or overlap at offset {off}");
            cursor += len;
        }
        assert_eq!(cursor, layout.total);
    }

    #[test]
    fn in_place_scan_drops_the_scan_from_the_packed_output() {
        let cfg = tiny_config(2048);
        let portable = DecodeLayout::build(&cfg, 1, 1, ScanState::Portable);
        let in_place = DecodeLayout::build(&cfg, 1, 1, ScanState::InPlace);
        let scan_w = cfg.n_v_heads() * cfg.ssm_state_size * cfg.ssm_state_size;
        assert_eq!(portable.total - in_place.total, portable.gdn.len() * scan_w);
        assert!(in_place.gdn.iter().all(|&(_, _, s)| s == 0));
    }

    /// The hash history is a sliding window of exactly `ngram_size - 1` tokens,
    /// oldest first — the order [`crate::ple::ngram_rows_with_history`] expects.
    /// A chunk of N appends N cache rows and produces N head rows; everything
    /// else (the convolution windows, the scan matrix) is a window and does not
    /// scale with the chunk.
    #[test]
    fn a_chunked_layout_scales_only_the_per_token_parts() {
        let cfg = tiny_config(2048);
        let one = DecodeLayout::build(&cfg, 5, 1, ScanState::Portable);
        let four = DecodeLayout::build(&cfg, 5, 4, ScanState::Portable);
        assert_eq!(four.head_width, 4 * one.head_width);
        for (a, b) in one.gdn.iter().zip(&four.gdn) {
            assert_eq!((a.1, a.2), (b.1, b.2), "conv/scan must not scale");
        }
        for (a, b) in one.attn.iter().zip(&four.attn) {
            assert_eq!(4 * a.1, b.1, "kv rows scale with the chunk");
        }
        for (a, b) in one.ple.iter().zip(&four.ple) {
            assert_eq!(a.1, b.1, "the PLE window must not scale");
        }
    }

    #[test]
    fn the_token_history_slides() {
        let cfg = tiny_config(2048);
        let layout = DecodeLayout::build(&cfg, 1, 1, ScanState::Portable);
        let mut s = DecodeSession::new(&cfg, layout, 8);
        assert_eq!(s.ngram_history(), &[] as &[u32]);
        s.push_token(5);
        assert_eq!(s.ngram_history(), &[5]);
        s.push_token(6);
        assert_eq!(s.ngram_history(), &[5, 6]);
        s.push_token(7);
        assert_eq!(s.ngram_history(), &[6, 7], "window is ngram_size - 1 = 2");
    }

    #[test]
    fn decode_past_the_capacity_is_an_error_not_a_wrap() {
        let cfg = tiny_config(2048);
        let layout = DecodeLayout::build(&cfg, 1, 1, ScanState::Portable);
        let total = layout.total;
        let mut s = DecodeSession::new(&cfg, layout, 2);
        let packed = vec![0.0f32; total];
        s.commit(&packed).unwrap();
        s.commit(&packed).unwrap();
        assert!(s.commit(&packed).is_err());
    }
}
