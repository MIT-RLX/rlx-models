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
//! This is the loop that joins the two halves: [`rlx_distributed::ExpertPager`]
//! reads the fired experts, [`MoeDims::paged`] gives a graph that indexes them
//! by slot, and the code here runs them in order.
//!
//! ```text
//!   x ──┬─► host router  ── ids[rows][top_k] ──► pager.gather (3 banks)
//!       │                                              │
//!       │                                    set_param per bank
//!       ▼                                              ▼
//!       └──────────────────────────────► paged graph ──► y
//! ```
//!
//! ## Why the router runs on the host
//!
//! Chicken and egg: you cannot gather the experts a token fires until you know
//! which they are, and the graph that would tell you is the graph you are trying
//! to feed. So the router — a `[rows, hidden] x [hidden, n_routed]` matmul, four
//! orders of magnitude smaller than the expert GEMMs it selects — runs first, on
//! the host.
//!
//! It has to agree with the graph's router *exactly*, not approximately: a
//! single differing pick silently pairs a token with another expert's weights.
//! [`PagedMoeLayer::route`] therefore calls the same `group_limited_topk` the in-graph
//! `llada2.group_limited_gate` op calls, rather than reimplementing top-k.
//!
//! ## What this costs
//!
//! Per token, per layer: `top_k` expert slices instead of `n_routed`. For
//! GLM-5.3-Flash that is 8 of 288 — 0.05 GB against 1.88 GB — which is the
//! `per_layer_expert_active_bytes` the cluster planner has been pricing all
//! along.

use crate::moe::{MoeDims, emit_glm5next_moe, emit_glm5next_router};
use anyhow::{Context, Result, anyhow, bail};
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_distributed::{BankIx, ExpertPager};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::sync::Arc;

/// Where one paged forward pass spent its time.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimes {
    /// Host router: which experts this token fires.
    pub route: std::time::Duration,
    /// Paging the fired experts in and binding them to the graph.
    pub page: std::time::Duration,
    /// Running the layer.
    pub compute: std::time::Duration,
}

/// The three routed banks, in the order this module always uses them.
pub const BANKS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

/// One MoE layer whose routed experts live on disk.
pub struct PagedMoeLayer {
    prefix: String,
    /// Which block this is — only needed for diagnostics now that the bank
    /// handles are resolved, but a paged layer that cannot say which layer it
    /// is makes every error message ambiguous across 42 of them.
    layer: usize,
    dims: MoeDims,
    pager: Arc<ExpertPager>,
    /// Router-only graph: the same emitters, weights and kernels the layer
    /// graph routes with, so the two cannot disagree. See
    /// [`crate::moe::emit_glm5next_router`].
    router: CompiledGraph,
    /// Graph built over `dims.paged == true`, so its banks are slot-indexed.
    compiled: CompiledGraph,
    /// f32 elements in one expert of each bank, in `BANKS` order.
    per_expert: [usize; 3],
    /// Bank handles, resolved once at construction — so a missing bank is a
    /// build-time error rather than a first-token one.
    bank_ix: [BankIx; 3],
    /// Times a bank had to be uploaded whole because the backend refused a
    /// sub-range write.
    ///
    /// Worth counting rather than assuming: the fallback is *correct*, so a
    /// backend that never supports partial writes produces identical output
    /// and quietly doubles the bytes moved per token. Nothing about the answer
    /// reveals which path ran.
    whole_uploads: usize,
}

impl PagedMoeLayer {
    /// Build the layer.
    ///
    /// `tensors` supplies everything except the routed banks — the router, the
    /// bias, the shared expert. The banks come from `pager` and are declared to
    /// the graph at their **slot** width (`seq * top_k`), so nothing here ever
    /// holds the full `n_routed` banks; that is the point.
    pub fn new(
        prefix: impl Into<String>,
        layer: usize,
        mut dims: MoeDims,
        device: Device,
        mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
        pager: Arc<ExpertPager>,
    ) -> Result<Self> {
        let prefix = prefix.into();
        if dims.top_k == 0 || dims.seq == 0 {
            bail!("paged MoE: top_k and seq must both be >= 1");
        }
        dims.paged = true;
        let slots = dims.seq * dims.top_k;

        for key in ["ffn_gate_inp.weight", "exp_probs_b.bias"] {
            if !tensors.contains_key(&format!("{prefix}.{key}")) {
                bail!("{prefix}: no `{key}` — cannot route");
            }
        }

        // Declare the banks at slot width. The contents are placeholders — every
        // forward pass overwrites them with that token's gather — but the shape
        // is what fixes the graph, and it must be `slots`, not `n_routed`.
        let per_expert = [
            dims.moe_inter * dims.hidden,
            dims.moe_inter * dims.hidden,
            dims.hidden * dims.moe_inter,
        ];
        let shapes = [
            vec![slots, dims.moe_inter, dims.hidden],
            vec![slots, dims.moe_inter, dims.hidden],
            vec![slots, dims.hidden, dims.moe_inter],
        ];
        for (i, bank) in BANKS.iter().enumerate() {
            tensors.insert(
                format!("{prefix}.{bank}.weight"),
                (vec![0.0f32; slots * per_expert[i]], shapes[i].clone()),
            );
        }

        let shape = Shape::new(&[1, dims.seq, dims.hidden], DType::F32);

        // Router graph first, from the same tensors.
        let mut rwm = WeightMap::from_tensors(tensors.clone());
        let rs = shape.clone();
        let rp = prefix.clone();
        let router_built = ModelFlow::new("glm5next_router")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", shape.clone())
            .plugin_named("router", move |emit, _prev| {
                let x = emit.flow_input("x")?.hir_id();
                let idx = emit_glm5next_router(emit, &rp, x, dims)?;
                Ok(Some(emit.wrap(
                    idx,
                    Shape::new(&[dims.seq, dims.top_k], DType::F32),
                )))
            })
            .output("top_idx")
            .build_with(&mut WeightMapSource(&mut rwm), None)?;
        let _ = rs;
        let router = compile_built(router_built, device)?;

        let mut wm = WeightMap::from_tensors(tensors);
        let p = prefix.clone();
        let s = shape.clone();
        let built = ModelFlow::new("glm5next_paged_moe")
            .with_profile(CompileProfile::llama32_prefill())
            .input("x", shape.clone())
            .plugin_named("moe", move |emit, _prev| {
                let x = emit.flow_input("x")?.hir_id();
                let out = emit_glm5next_moe(emit, &p, x, dims)?;
                Ok(Some(emit.wrap(out, s.clone())))
            })
            .output("y")
            .build_with(&mut WeightMapSource(&mut wm), None)?;
        let compiled = compile_built(built, device)?;

        // Resolve the banks once, and check their geometry here rather than on
        // the first token: a bank whose experts are a different size than the
        // graph's slots would otherwise be discovered mid-forward, or — worse,
        // on the partial-write path — not at all.
        let mut bank_ix = [BankIx::default(); 3];
        for (i, bank) in BANKS.iter().enumerate() {
            let ix = pager.bank_ix(layer, bank).with_context(|| {
                format!("layer {layer} bank `{bank}` is not registered with the pager")
            })?;
            let want = per_expert[i] * std::mem::size_of::<f32>();
            let got = pager.location(ix).bytes_per_expert;
            if got != want {
                bail!(
                    "layer {layer} bank `{bank}`: the pager holds {got}-byte experts \
                     but the graph's slots are {want} bytes ({} f32) — one of the \
                     two has the wrong shape",
                    per_expert[i]
                );
            }
            bank_ix[i] = ix;
        }

        Ok(Self {
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

    /// Which experts each row fires, in the order the layer graph will weight
    /// them.
    ///
    /// Runs [`crate::moe::emit_glm5next_router`] as a compiled graph rather than
    /// recomputing the routing on the host, so the ids are the layer graph's by
    /// construction rather than by agreement.
    ///
    /// The hazard this closes is latent, not observed: a host loop matched this
    /// router on 400/400 inputs at GLM-5.3-Flash's shape. But top-k is a
    /// comparison, the 8th-to-9th score gap gets as tight as 4.6e-5, and two
    /// f32 accumulation orders of the same dot product differ by up to 2.7e-4 —
    /// so the agreement rests on the host and the graph happening to accumulate
    /// alike, which a BLAS threshold or a device change moves silently. A
    /// flipped pick weights the token by another expert's matrix and looks
    /// entirely normal.
    pub fn route(&mut self, x: &[f32]) -> Result<Vec<Vec<usize>>> {
        let (rows, h) = (self.dims.seq, self.dims.hidden);
        if x.len() != rows * h {
            bail!(
                "paged MoE: input is {} elements, expected {}",
                x.len(),
                rows * h
            );
        }
        let idx = self
            .router
            .run(&[("x", x)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("router graph produced no output"))?;
        if idx.len() != rows * self.dims.top_k {
            bail!(
                "router returned {} ids, expected seq*top_k = {}",
                idx.len(),
                rows * self.dims.top_k
            );
        }
        // Ids come back f32-encoded, as `Op::Custom` produces them.
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(self.dims.top_k);
            for k in 0..self.dims.top_k {
                // Round, not truncate: the ids are integers carried in f32, and
                // `as usize` on a 286.9999 would silently pick expert 286. Exact
                // for every id below 2^24, so rounding is free insurance rather
                // than a fudge.
                let v = idx[r * self.dims.top_k + k];
                if !v.is_finite() || v < -0.5 || v >= self.dims.n_routed as f32 - 0.5 + 1.0 {
                    bail!("router returned expert id {v} out of range");
                }
                let id = v.round() as usize;
                if id >= self.dims.n_routed {
                    bail!("router returned expert id {id} >= {}", self.dims.n_routed);
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
    /// Worth having as API rather than as a one-off measurement: which of the
    /// three phases dominates decides what is worth optimizing, and it is not
    /// stable across models — a wide MoE on a fast disk is compute-bound where
    /// a narrow one on a slow disk is not.
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
            .ok_or_else(|| anyhow!("paged MoE produced no output"))?;
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
    ///
    /// Slot order is the contract with the graph: slot `r * top_k + k` holds the
    /// expert row `r` fired at position `k`, which is what makes the graph's
    /// expert index a constant. Flattening `fired` row-major produces exactly
    /// that, and doing it any other way pairs tokens with other tokens' gates
    /// while still producing finite, plausible output.
    fn load_fired(&mut self, fired: &[Vec<usize>]) -> Result<()> {
        let ids: Vec<usize> = fired.iter().flatten().copied().collect();
        debug_assert_eq!(ids.len(), self.dims.seq * self.dims.top_k);

        for (i, bank) in BANKS.iter().enumerate() {
            let name = format!("{}.{bank}.weight", self.prefix);
            let per_bytes = self.per_expert[i] * std::mem::size_of::<f32>();

            // Write each expert straight into its slot in the graph's arena.
            //
            // The obvious version — gather into a Vec, then upload the Vec —
            // copies every active byte twice, and those bytes are the dominant
            // cost of the whole paged path: GLM-5.3-Flash moves ~2.1 GB per
            // token across 42 layers, so the redundant copy is ~2 GB of memcpy
            // per token at memory bandwidth. Measured on the gather alone,
            // 4 MB/token already ran at 20 GB/s, i.e. bandwidth-bound.
            let mut partial = true;
            for (slot, &id) in ids.iter().enumerate() {
                let bytes = self.pager.expert_at(self.bank_ix[i], id)?;
                if !self
                    .compiled
                    .set_param_range(&name, slot * per_bytes, &bytes)
                {
                    partial = false;
                    break;
                }
            }
            if !partial {
                // The backend cannot write a sub-range of this param; fall back
                // to assembling the whole bank. Correct, just the slower path.
                self.whole_uploads += 1;
                let bytes = self.pager.gather_at(self.bank_ix[i], &ids)?;
                self.compiled.set_param_typed(&name, &bytes, DType::F32);
            }
        }
        Ok(())
    }

    /// How many bank uploads fell back to a whole-buffer write.
    ///
    /// Zero means every expert went straight into its slot in the arena — one
    /// copy of the active bytes per token instead of two.
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
    ///
    /// The ratio the cluster planner uses to decide whether a stage can page.
    pub fn bytes_per_token(&self) -> usize {
        let per: usize = self.per_expert.iter().sum();
        self.dims.seq * self.dims.top_k * per * std::mem::size_of::<f32>()
    }
}
