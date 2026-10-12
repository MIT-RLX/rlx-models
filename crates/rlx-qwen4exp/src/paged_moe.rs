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

//! Driving one MoE layer against experts paged off disk.
//!
//! This is the last thing between the port and the full checkpoint. The model
//! is `512 × 56B`: every one of its 48 layers carries 512 experts across three
//! banks, and a token fires **10**. Holding them is the whole problem; reading
//! only what fires is the whole solution.
//!
//! ```text
//!   x ──┬─► router graph ── ids[rows][top_k] ──► pager.expert_at (3 banks)
//!       │                                              │
//!       │                                    set_param_range per slot
//!       ▼                                              ▼
//!       └──────────────────────────────► paged graph ──► y
//! ```
//!
//! Three things make it work, and each is a trap if done the obvious way.
//!
//! * **The router runs first, as its own graph.** Chicken and egg: you cannot
//!   gather the experts a token fires until you know which they are, and the
//!   graph that would tell you is the graph you are trying to feed. So the
//!   router — a `[rows, hidden] × [hidden, 512]` matmul, orders of magnitude
//!   smaller than the expert GEMMs it selects — is compiled separately and run
//!   first. It is [`crate::moe::emit_qwen4exp_router`], the *same* emitter the
//!   layer uses, so the ids are the layer's by construction. Re-deriving top-k
//!   on the host instead would rest on the host and the graph accumulating a
//!   512-wide dot product identically, which a BLAS threshold or a device
//!   change moves silently — and a flipped pick weights the token by another
//!   expert's matrix and looks entirely normal.
//! * **Slot order is the contract.** Slot `r * top_k + k` holds the expert that
//!   row `r` fired at position `k`, which is what lets the graph's expert index
//!   be a constant ([`MoeDims::paged`]). Flattening the router's ids row-major
//!   produces exactly that; any other order pairs tokens with other tokens'
//!   gates while still producing finite, plausible output.
//! * **Experts go straight into their slot in the arena.** Gathering into a
//!   `Vec` and uploading it copies every active byte twice, and those bytes are
//!   the dominant cost of the whole path.
//!
//! ## What it costs
//!
//! Per token, per layer: `top_k` expert slices instead of `num_experts` — 10 of
//! 512. At this model's widths (`hidden = 2560`, `expert_ffn = 640`) that is
//! 10 × 3 × 1.64M f32 ≈ **49 MB** against the 2.5 GB the resident banks of one
//! layer would need, or 120 GB across the stack.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::sync::Arc;

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{BankIx, ExpertPager};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_runtime::{CompiledGraph, Device};

use crate::moe::{MoeDims, emit_moe, emit_qwen4exp_router};

/// The three routed banks, in the order [`PagedMoeLayer`] pages them.
pub const BANKS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

/// Where one paged forward pass spent its time.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimes {
    pub route: std::time::Duration,
    pub page: std::time::Duration,
    pub compute: std::time::Duration,
}

/// How a bank's bytes on disk become the graph's f32 slots.
///
/// The published checkpoint's banks are **quantized, and not uniformly** —
/// `blk.0`'s gate and up are IQ1_S while `blk.1`'s are IQ2_XXS, and every
/// `ffn_down_exps` is IQ4_NL. So the dtype is read per bank per layer rather
/// than assumed, and a paged read of the real model has to dequantize each
/// expert slice before it lands in its slot.
#[derive(Debug, Clone, Copy)]
enum SlotEncoding {
    /// Already f32 on disk: the bytes go straight into the slot.
    F32,
    /// Dequantize `elements` of `dtype` into the slot.
    Quantized {
        dtype: rlx_gguf::GgmlType,
        elements: usize,
    },
}

/// One MoE layer whose routed experts live on disk.
pub struct PagedMoeLayer {
    prefix: String,
    layer: usize,
    dims: MoeDims,
    pager: Arc<ExpertPager>,
    router: CompiledGraph,
    compiled: CompiledGraph,
    /// f32 elements per expert, per bank.
    per_expert: [usize; 3],
    bank_ix: [BankIx; 3],
    encoding: [SlotEncoding; 3],
    /// Reused across tokens so dequantizing allocates once, not per expert.
    scratch: Vec<u8>,
    whole_uploads: usize,
}

impl PagedMoeLayer {
    /// Build the layer.
    ///
    /// `tensors` supplies everything except the routed banks — the router, the
    /// shared expert and its gate. The banks are declared to the graph at their
    /// **slot** width (`seq * top_k`), so nothing here ever holds all 512; that
    /// is the point.
    pub fn new(
        prefix: impl Into<String>,
        layer: usize,
        dims: MoeDims,
        device: Device,
        tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
        pager: Arc<ExpertPager>,
    ) -> Result<Self> {
        Self::build(prefix, layer, dims, device, tensors, pager, None)
    }

    /// Page a **quantized** checkpoint: the banks keep their GGUF dtype on
    /// disk and each fired expert is dequantized on its way into its slot.
    ///
    /// This is the only form the published model exists in, and the dtype is
    /// not uniform — `blk.0`'s gate and up banks are IQ1_S where `blk.1`'s are
    /// IQ2_XXS, and every `ffn_down_exps` is IQ4_NL — so it is read per bank
    /// per layer out of the header rather than assumed.
    ///
    /// The disk read shrinks with the quantization (an IQ2_XXS expert is ~8x
    /// smaller than f32) and the dequantization lands in the same f32 slots the
    /// resident path uses. Feeding the quantized bytes to the graph *directly*,
    /// as `DequantGroupedMatMul` over packed slot banks, would avoid both the
    /// host dequantization and the f32 arena — that is the optimization this
    /// leaves on the table, not a correctness gap.
    pub fn from_gguf(
        prefix: impl Into<String>,
        layer: usize,
        dims: MoeDims,
        device: Device,
        tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
        pager: Arc<ExpertPager>,
        gguf: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let prefix = prefix.into();
        let path = gguf.as_ref();
        let header = rlx_gguf::GgufFile::header_from_path(path)
            .map_err(|e| anyhow!("qwen4exp: reading {path:?} header: {e}"))?;
        let mut dtypes = [rlx_gguf::GgmlType::F32; 3];
        for (i, bank) in BANKS.iter().enumerate() {
            let name = format!("{prefix}.{bank}.weight");
            let t = header
                .get(&name)
                .ok_or_else(|| anyhow!("qwen4exp: {path:?} has no tensor {name}"))?;
            dtypes[i] = t.dtype;
        }
        Self::build(prefix, layer, dims, device, tensors, pager, Some(dtypes))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        prefix: impl Into<String>,
        layer: usize,
        mut dims: MoeDims,
        device: Device,
        mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
        pager: Arc<ExpertPager>,
        dtypes: Option<[rlx_gguf::GgmlType; 3]>,
    ) -> Result<Self> {
        let prefix = prefix.into();
        if dims.top_k == 0 || dims.seq == 0 {
            bail!("qwen4exp paged MoE: top_k and seq must both be >= 1");
        }
        dims.paged = true;
        let slots = dims.seq * dims.top_k;

        // Fail here rather than inside the first forward pass.
        for key in [
            "ffn_gate_inp.weight",
            "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ] {
            if !tensors.contains_key(&format!("{prefix}.{key}")) {
                bail!("{prefix}: no `{key}` — cannot run the layer");
            }
        }

        // Declare the banks at slot width. The contents are placeholders —
        // every forward overwrites them with that token's gather — but the
        // shape fixes the graph, and it must be `slots`, not `num_experts`.
        let per_expert = [
            dims.expert_ffn * dims.hidden,
            dims.expert_ffn * dims.hidden,
            dims.hidden * dims.expert_ffn,
        ];
        let shapes = [
            vec![slots, dims.expert_ffn, dims.hidden],
            vec![slots, dims.expert_ffn, dims.hidden],
            vec![slots, dims.hidden, dims.expert_ffn],
        ];
        for (i, bank) in BANKS.iter().enumerate() {
            tensors.insert(
                format!("{prefix}.{bank}.weight"),
                (vec![0.0f32; slots * per_expert[i]], shapes[i].clone()),
            );
        }

        let shape = Shape::new(&[1, dims.seq, dims.hidden], DType::F32);

        // The router graph, from the same tensors.
        let mut rwm = WeightMap::from_tensors(tensors.clone());
        let rp = prefix.clone();
        let router_built = ModelFlow::new("qwen4exp_router")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", shape.clone())
            .plugin_named("router", move |emit, _prev| {
                let x = emit.flow_input("x")?.hir_id();
                let idx = emit_qwen4exp_router(emit, &rp, x, dims)?;
                Ok(Some(emit.wrap(
                    idx,
                    Shape::new(&[dims.seq, dims.top_k], DType::F32),
                )))
            })
            .output("top_idx")
            .build_with(&mut WeightMapSource(&mut rwm), None)?;
        let router = compile_built(router_built, device)?;

        let mut wm = WeightMap::from_tensors(tensors);
        let p = prefix.clone();
        let s = shape.clone();
        let built = ModelFlow::new("qwen4exp_paged_moe")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", shape.clone())
            .plugin_named("moe", move |emit, _prev| {
                let x = emit.flow_input("x")?.hir_id();
                let out = emit_moe(emit, &p, x, dims)?;
                Ok(Some(emit.wrap(out, s.clone())))
            })
            .output("y")
            .build_with(&mut WeightMapSource(&mut wm), None)?;
        let compiled = compile_built(built, device)?;

        // Resolve the banks once, and check their geometry here rather than on
        // the first token: a bank whose experts are a different size than the
        // graph's slots would otherwise be found mid-forward, or — on the
        // partial-write path — not at all.
        let mut bank_ix = [BankIx::default(); 3];
        let mut encoding = [SlotEncoding::F32; 3];
        for (i, bank) in BANKS.iter().enumerate() {
            let ix = pager.bank_ix(layer, bank).with_context(|| {
                format!("layer {layer} bank `{bank}` is not registered with the pager")
            })?;
            let enc = match dtypes {
                // An f32 bank's expert is its elements; a quantized one's is
                // whatever that dtype packs them into.
                None => SlotEncoding::F32,
                Some(d) => SlotEncoding::Quantized {
                    dtype: d[i],
                    elements: per_expert[i],
                },
            };
            let want = match enc {
                SlotEncoding::F32 => per_expert[i] * std::mem::size_of::<f32>(),
                SlotEncoding::Quantized { dtype, elements } => {
                    rlx_gguf::bytes_for_public(dtype, elements).ok_or_else(|| {
                        anyhow!(
                            "layer {layer} bank `{bank}`: {elements} elements are not \
                             block-aligned for {dtype:?}"
                        )
                    })?
                }
            };
            let got = pager.location(ix).bytes_per_expert;
            if got != want {
                bail!(
                    "layer {layer} bank `{bank}`: the pager holds {got}-byte experts \
                     but the graph's slots want {want} bytes ({} elements as {}) — \
                     one of the two has the wrong shape",
                    per_expert[i],
                    match enc {
                        SlotEncoding::F32 => "f32".to_string(),
                        SlotEncoding::Quantized { dtype, .. } => format!("{dtype:?}"),
                    }
                );
            }
            encoding[i] = enc;
            bank_ix[i] = ix;
        }

        Ok(Self {
            encoding,
            scratch: Vec::new(),
            prefix,
            layer,
            dims,
            pager,
            router,
            compiled,
            per_expert,
            bank_ix,
            whole_uploads: 0,
        })
    }

    /// Which experts each row fires, in the order the layer will weight them.
    pub fn route(&mut self, x: &[f32]) -> Result<Vec<Vec<usize>>> {
        let (rows, h) = (self.dims.seq, self.dims.hidden);
        if x.len() != rows * h {
            bail!(
                "qwen4exp paged MoE: input is {} elements, expected {}",
                x.len(),
                rows * h
            );
        }
        let idx = self
            .router
            .run(&[("x", x)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("the router graph produced no output"))?;
        if idx.len() != rows * self.dims.top_k {
            bail!(
                "the router returned {} ids, expected seq*top_k = {}",
                idx.len(),
                rows * self.dims.top_k
            );
        }
        let n = self.dims.num_experts;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(self.dims.top_k);
            for k in 0..self.dims.top_k {
                // Round, not truncate: ids are integers carried in f32, and
                // `as usize` on a 510.9999 would silently pick expert 510.
                // Exact for every id below 2^24, so rounding costs nothing.
                let v = idx[r * self.dims.top_k + k];
                if !v.is_finite() || v < -0.5 || v > n as f32 - 0.5 {
                    bail!("the router returned expert id {v}, outside 0..{n}");
                }
                let id = v.round() as usize;
                if id >= n {
                    bail!("the router returned expert id {id} >= {n}");
                }
                row.push(id);
            }
            out.push(row);
        }
        Ok(out)
    }

    /// Run the layer over `x` (`[seq * hidden]`), paging in what it fires.
    pub fn forward(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let (y, _) = self.forward_timed(x)?;
        Ok(y)
    }

    /// [`Self::forward`], also reporting where the time went.
    ///
    /// Which of the three phases dominates decides what is worth optimizing,
    /// and it is not stable across models — a wide MoE on a fast disk is
    /// compute-bound where a narrow one on a slow disk is not.
    pub fn forward_timed(&mut self, x: &[f32]) -> Result<(Vec<f32>, PhaseTimes)> {
        use std::time::Instant;
        let t = Instant::now();
        let fired = self.route(x)?;
        let route = t.elapsed();

        let t = Instant::now();
        self.load_fired(&fired)?;
        let page = t.elapsed();

        let t = Instant::now();
        let y = self
            .compiled
            .run(&[("x", x)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("the paged MoE produced no output"))?;
        let compute = t.elapsed();
        Ok((
            y,
            PhaseTimes {
                route,
                page,
                compute,
            },
        ))
    }

    /// Page in the fired experts and bind them to the graph's bank params.
    fn load_fired(&mut self, fired: &[Vec<usize>]) -> Result<()> {
        let ids: Vec<usize> = fired.iter().flatten().copied().collect();
        debug_assert_eq!(ids.len(), self.dims.seq * self.dims.top_k);

        for (i, bank) in BANKS.iter().enumerate() {
            let name = format!("{}.{bank}.weight", self.prefix);
            let per_bytes = self.per_expert[i] * std::mem::size_of::<f32>();

            // Straight into the slot: gathering into a Vec and uploading that
            // copies every active byte twice, and those bytes dominate.
            let mut partial = true;
            for (slot, &id) in ids.iter().enumerate() {
                let raw = self.pager.expert_at(self.bank_ix[i], id)?;
                let bytes: &[u8] = match self.encoding[i] {
                    SlotEncoding::F32 => &raw,
                    SlotEncoding::Quantized { dtype, elements } => {
                        decode_into(&mut self.scratch, dtype, &raw, elements)?
                    }
                };
                if !self
                    .compiled
                    .set_param_range(&name, slot * per_bytes, bytes)
                {
                    partial = false;
                    break;
                }
            }
            if !partial {
                // This backend cannot write a sub-range of the param; assemble
                // the whole bank instead. Correct, just the slower path.
                self.whole_uploads += 1;
                match self.encoding[i] {
                    SlotEncoding::F32 => {
                        let bytes = self.pager.gather_at(self.bank_ix[i], &ids)?;
                        self.compiled.set_param_typed(&name, &bytes, DType::F32);
                    }
                    SlotEncoding::Quantized { dtype, elements } => {
                        let mut whole = Vec::with_capacity(ids.len() * per_bytes);
                        for &id in &ids {
                            let raw = self.pager.expert_at(self.bank_ix[i], id)?;
                            whole.extend_from_slice(decode_into(
                                &mut self.scratch,
                                dtype,
                                &raw,
                                elements,
                            )?);
                        }
                        self.compiled.set_param_typed(&name, &whole, DType::F32);
                    }
                }
            }
        }
        Ok(())
    }

    /// How many bank uploads fell back to a whole-buffer write. Zero means
    /// every expert went straight into its slot — one copy, not two.
    pub fn whole_bank_uploads(&self) -> usize {
        self.whole_uploads
    }

    /// Which block of the stack this layer is.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Paging statistics so far — hit rate, bytes off disk.
    pub fn stats(&self) -> rlx_distributed::PagerStats {
        self.pager.stats()
    }

    /// Bytes this layer reads per token, against what holding the banks costs.
    pub fn bytes_per_token(&self) -> usize {
        let per: usize = self.per_expert.iter().sum();
        self.dims.seq * self.dims.top_k * per * std::mem::size_of::<f32>()
    }

    /// Bytes the resident banks of this layer would need.
    pub fn resident_bytes(&self) -> usize {
        let per: usize = self.per_expert.iter().sum();
        self.dims.num_experts * per * std::mem::size_of::<f32>()
    }
}

/// Dequantize one expert slice into `scratch` and return it as f32 bytes.
///
/// `scratch` is the layer's, reused across every expert of every token, so the
/// dequantization allocates once rather than per read.
fn decode_into<'a>(
    scratch: &'a mut Vec<u8>,
    dtype: rlx_gguf::GgmlType,
    raw: &[u8],
    elements: usize,
) -> Result<&'a [u8]> {
    let vals = rlx_gguf::dequant_typed(dtype, raw, elements, "paged expert")
        .map_err(|e| anyhow!("dequantizing a paged {dtype:?} expert: {e}"))?;
    scratch.clear();
    scratch.reserve(elements * 4);
    for v in &vals {
        scratch.extend_from_slice(&v.to_ne_bytes());
    }
    Ok(scratch.as_slice())
}
