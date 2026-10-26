// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! One decoder layer at a time: route it, run it (optionally exposing every
//! named intermediate), and differentiate it.
//!
//! A layer is the function `(h, snapshots_in) → h_out` (the Attention-Residual
//! snapshot list only ever grows by the incoming `h` at a block boundary, so it
//! is threaded on the host and the graph needs no snapshot outputs). For a MoE
//! layer the expert *selection* is made first, by a small graph running the
//! layer up to the FFN input plus the model's own gate op; the layer graph then
//! takes the selection as inputs and the picked experts as a compact bank, so
//! the routed experts never have to be resident and the selection is exactly the
//! one the full in-graph router makes.
//!
//! Every graph here is built under [`CanonicalParams`], so param names are the
//! checkpoint's ([`crate::export::hf_name`]) and LoRA targets real modules.
//!
//! The backward of a layer is a vector–Jacobian product: the layer graph's
//! output is differentiated with `d_output` seeded by the cotangent arriving
//! from the layer above ([`rlx_autodiff::grad_with_loss_wrt`] takes a
//! non-scalar output), giving the cotangents of `h` and every snapshot plus the
//! parameter gradients. Chaining these from the head down is exact backprop
//! with one layer's activations live at a time.

use crate::common::{CanonicalParams, Tap, taps_begin, taps_take};
use crate::flow::{FfnWeights, FlowConfig, LayerWeights, build_head, build_layer};
use crate::moe::{FixedRouting, MoeWeights, build_moe_route};
use crate::source::{HeadWeights, WeightSource};
use anyhow::{Context, Result, bail, ensure};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

pub type Params = HashMap<String, Vec<f32>>;

/// Which experts each `(row, k)` slot of a MoE layer picked.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRouting {
    /// `[rows·k]` global expert ids (f32, as the graph takes them).
    pub idx: Vec<f32>,
    /// Distinct experts picked, ascending — the compact bank's order.
    pub uniq: Vec<usize>,
    /// `[rows·k]` positions into `uniq`.
    pub compact: Vec<f32>,
}

impl LayerRouting {
    pub fn from_idx(idx: Vec<f32>) -> Self {
        let ids: Vec<usize> = idx.iter().map(|&x| x.round().max(0.0) as usize).collect();
        let mut uniq = ids.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let pos: HashMap<usize, usize> = uniq.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        let compact = ids.iter().map(|e| pos[e] as f32).collect();
        Self { idx, uniq, compact }
    }

    /// Compact-bank size: the picked experts rounded up to a power of two, so the
    /// layer graph's shape — and anything keyed on it — repeats across batches.
    /// The padding experts are zero and never indexed.
    pub fn bank_size(&self) -> usize {
        self.uniq.len().next_power_of_two().max(1)
    }
}

/// What a layer graph is being built for.
pub struct LayerSpec<'a> {
    pub cfg: &'a FlowConfig,
    pub device: Device,
    /// LoRA on the matching linear weights (adapter values from `overrides`).
    pub lora: Option<&'a rlx_tune::LoraSpec>,
    /// Values that replace the source's: trained weights and LoRA adapters, by
    /// canonical graph name.
    pub overrides: &'a Params,
    /// Seed for LoRA adapters `overrides` does not hold yet
    /// ([`crate::train::init_adapter`] — deterministic per name, so a stage that
    /// first sees an adapter in its backward builds the same value).
    pub seed: u64,
    /// Reuse compiled graphs across calls (see [`GraphCache`]); `None` builds and
    /// compiles every time.
    pub cache: Option<&'a std::cell::RefCell<GraphCache>>,
}

/// Compiled layer graphs kept across steps, keyed by (kind, layer, snapshot
/// count, compact-bank size). A hit skips graph building, autodiff and
/// compilation; only what can change is re-uploaded — the `overrides` (trained
/// weights, adapters) and the compact expert bank. The compiled graphs hold the
/// layer's weights, so this is for stages whose layers fit in memory.
#[derive(Default)]
pub struct GraphCache {
    graphs: HashMap<(u8, usize, usize, usize), Cached>,
}

impl GraphCache {
    pub fn len(&self) -> usize {
        self.graphs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }
    pub fn clear(&mut self) {
        self.graphs.clear();
    }
}

struct Cached {
    compiled: CompiledGraph,
    /// Every param name in the graph (overrides outside it are skipped).
    params: std::collections::HashSet<String>,
    /// Params a backward graph differentiates, in output order.
    wrt: Vec<String>,
    /// Their starting values and shapes, from the build.
    wrt_values: Params,
    wrt_shapes: HashMap<String, Vec<usize>>,
}

/// A graph ready to compile, as the cached runner needs it.
struct Built {
    graph: Graph,
    params: Params,
    wrt: Vec<String>,
    shapes: HashMap<String, Vec<usize>>,
}

const KIND_ROUTE: u8 = 0;
const KIND_FWD: u8 = 1;
const KIND_BWD: u8 = 2;
const KIND_HEAD_BWD: u8 = 3;

/// Outputs of a (possibly cached) run plus the backward bookkeeping.
struct Ran {
    outs: Vec<Vec<f32>>,
    wrt: Vec<String>,
    wrt_values: Params,
    wrt_shapes: HashMap<String, Vec<usize>>,
}

/// Run a graph through `spec.cache` when there is one. `extra` are params whose
/// value changes per call independently of `overrides` (the compact bank).
fn run_graph(
    spec: &LayerSpec<'_>,
    key: (u8, usize, usize, usize),
    build: impl FnOnce() -> Result<Built>,
    extra: &[(String, &[f32])],
    feeds: &[(&str, &[f32])],
) -> Result<Ran> {
    let make = |b: Built| -> Cached {
        let params = b.params.keys().cloned().collect();
        let wrt_values = b
            .wrt
            .iter()
            .filter_map(|n| b.params.get(n).map(|v| (n.clone(), v.clone())))
            .collect();
        let wrt_shapes = b
            .wrt
            .iter()
            .filter_map(|n| b.shapes.get(n).map(|v| (n.clone(), v.clone())))
            .collect();
        Cached {
            compiled: compile(b.graph, &b.params, spec.device),
            params,
            wrt: b.wrt,
            wrt_values,
            wrt_shapes,
        }
    };
    let go = |c: &mut Cached, refresh: bool| -> Ran {
        if refresh {
            for (n, v) in spec.overrides {
                if c.params.contains(n) {
                    c.compiled.set_param(n, v);
                }
            }
            for (n, v) in extra {
                c.compiled.set_param(n, v);
            }
        }
        Ran {
            outs: c.compiled.run(feeds),
            wrt: c.wrt.clone(),
            wrt_values: c.wrt_values.clone(),
            wrt_shapes: c.wrt_shapes.clone(),
        }
    };
    match spec.cache {
        Some(cell) => {
            let mut cache = cell.borrow_mut();
            let hit = cache.graphs.contains_key(&key);
            if !hit {
                let c = make(build()?);
                cache.graphs.insert(key, c);
            }
            let c = cache.graphs.get_mut(&key).expect("inserted above");
            Ok(go(c, hit))
        }
        None => {
            let mut c = make(build()?);
            Ok(go(&mut c, false))
        }
    }
}

/// The compact bank padded to [`LayerRouting::bank_size`] experts.
fn padded_bank(
    r: &LayerRouting,
    mut gate_up: Vec<f32>,
    mut down: Vec<f32>,
    cfg: &FlowConfig,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (l, mi, n) = (cfg.moe.latent, cfg.moe.moe_inter, r.bank_size());
    ensure!(
        gate_up.len() == r.uniq.len() * l * 2 * mi && down.len() == r.uniq.len() * mi * l,
        "compact bank does not match {} experts",
        r.uniq.len()
    );
    gate_up.resize(n * l * 2 * mi, 0.0);
    down.resize(n * mi * l, 0.0);
    Ok((gate_up, down))
}

fn bank_names(i: usize) -> (String, String) {
    (
        format!("l{i}.block_sparse_moe.experts.gate_up"),
        format!("l{i}.block_sparse_moe.experts.down"),
    )
}

/// A built (not yet compiled) layer graph.
pub struct LayerGraph {
    pub graph: Graph,
    /// Every param's value (source weights with `overrides` applied).
    pub params: Params,
    /// Every param's shape, by name (adapters included).
    pub shapes: HashMap<String, Vec<usize>>,
    /// LoRA params injected into this graph, `lora.<weight>.a/.b`.
    pub adapters: Vec<rlx_tune::AdapterParam>,
    /// Named intermediates exposed as outputs `1..`, in order.
    pub taps: Vec<Tap>,
    pub n_snaps_in: usize,
}

fn input_name_snap(j: usize) -> String {
    format!("snap{j}")
}

/// Lower an HIR layer to a [`Graph`], inject LoRA, and resolve param values.
fn finish(
    hir: HirModule,
    params: Params,
    spec: &LayerSpec<'_>,
) -> Result<(
    Graph,
    Params,
    HashMap<String, Vec<usize>>,
    Vec<rlx_tune::AdapterParam>,
)> {
    let (mut graph, mut params) = rlx_core::flow_util::graph_from_hir(hir, params)?;
    let mut adapters = Vec::new();
    if let Some(lora) = spec.lora {
        let (g2, ad) = rlx_tune::inject_lora(&graph, lora, rlx_tune::FuseMode::Unfused);
        graph = g2;
        adapters = ad;
    }
    let mut shapes = HashMap::new();
    for n in graph.nodes() {
        if let Op::Param { name } = &n.op {
            let dims: Vec<usize> = n.shape.dims().iter().map(|d| d.unwrap_static()).collect();
            shapes.insert(name.clone(), dims);
        }
    }
    for (name, v) in spec.overrides {
        if let Some(s) = shapes.get(name) {
            ensure!(
                s.iter().product::<usize>() == v.len(),
                "override {name}: {} values for shape {s:?}",
                v.len()
            );
            params.insert(name.clone(), v.clone());
        }
    }
    for a in &adapters {
        params
            .entry(a.name.clone())
            .or_insert_with(|| crate::train::init_adapter(&a.name, &a.shape, spec.seed));
    }
    Ok((graph, params, shapes, adapters))
}

/// Compile `graph` on `device` and upload `params`.
pub fn compile(graph: Graph, params: &Params, device: Device) -> CompiledGraph {
    let mut c = Session::new(device).compile(graph);
    for (n, v) in params {
        c.set_param(n, v);
    }
    c
}

/// Is layer `i` an Attention-Residual block boundary (snapshots its input)?
pub fn is_boundary(i: usize, cfg: &FlowConfig) -> bool {
    i.is_multiple_of(cfg.attn_res_block_size)
}

fn layer_inputs(
    g: &mut HirMut,
    cfg: &FlowConfig,
    n_snaps: usize,
) -> (rlx_ir::hir::HirNodeId, Vec<rlx_ir::hir::HirNodeId>) {
    let shape = Shape::new(&[cfg.batch, cfg.seq, cfg.hidden], DType::F32);
    let h = g.input("h", shape.clone());
    let snaps = (0..n_snaps)
        .map(|j| g.input(input_name_snap(j), shape.clone()))
        .collect();
    (h, snaps)
}

fn feeds<'a>(h: &'a [f32], snaps: &'a [Vec<f32>]) -> Vec<(String, &'a [f32])> {
    let mut v = vec![("h".to_string(), h)];
    for (j, s) in snaps.iter().enumerate() {
        v.push((input_name_snap(j), s.as_slice()));
    }
    v
}

/// The expert selection layer `i` makes on `(h, snaps)` — `None` for a dense
/// layer. Runs the layer up to its FFN input plus the model's gate op, with the
/// same params (trained weights, adapters) the layer itself will use.
pub fn route_layer(
    i: usize,
    lw: &LayerWeights,
    h: &[f32],
    snaps: &[Vec<f32>],
    spec: &LayerSpec<'_>,
) -> Result<Option<LayerRouting>> {
    let FfnWeights::Moe(mw) = &lw.ffn else {
        return Ok(None);
    };
    let cfg = spec.cfg;
    let build = || -> Result<Built> {
        let _canon = CanonicalParams::new();
        let mut hir = HirModule::new(format!("kimi_route_l{i}"));
        let mut g = HirMut::new(&mut hir);
        let (hn, sn) = layer_inputs(&mut g, cfg, snaps.len());
        let mut params = Params::new();
        let (mn, _, _) = crate::flow::build_layer_pre_ffn(&mut g, &mut params, i, hn, sn, lw, cfg)?;
        let (_, top_idx, _) = build_moe_route(
            &mut g,
            &mut params,
            &format!("l{i}.block_sparse_moe"),
            mn,
            &mw.router,
            &mw.e_score_bias,
            &mw.down_latent,
            cfg.moe,
        )?;
        g.set_outputs(vec![top_idx]);
        let (graph, params, shapes, _) = finish(hir, params, spec)?;
        Ok(Built {
            graph,
            params,
            wrt: Vec::new(),
            shapes,
        })
    };
    let f = feeds(h, snaps);
    let refs: Vec<(&str, &[f32])> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let ran = run_graph(spec, (KIND_ROUTE, i, snaps.len(), 0), build, &[], &refs)?;
    let idx = ran.outs.into_iter().next().context("router output")?;
    Ok(Some(LayerRouting::from_idx(idx)))
}

/// Build layer `i`'s graph `(h, snap0…) → h_out` (+ taps). `routing` with its
/// compact bank — padded to [`LayerRouting::bank_size`] experts — is required for
/// a MoE layer whose weights carry no expert banks.
pub fn build_layer_graph(
    i: usize,
    lw: &LayerWeights,
    n_snaps_in: usize,
    routing: Option<(&LayerRouting, Vec<f32>, Vec<f32>)>,
    spec: &LayerSpec<'_>,
    with_taps: bool,
) -> Result<LayerGraph> {
    let cfg = spec.cfg;
    let _canon = CanonicalParams::new();
    let mut hir = HirModule::new(format!("kimi_layer_l{i}"));
    let mut g = HirMut::new(&mut hir);
    let (hn, sn) = layer_inputs(&mut g, cfg, n_snaps_in);
    let mut params = Params::new();
    let rows = cfg.batch * cfg.seq;
    let k = cfg.moe.top_k;
    let fixed = match (&lw.ffn, routing) {
        (FfnWeights::Moe(_), Some((r, gate_up, down))) => {
            let n = r.bank_size();
            let (l, mi) = (cfg.moe.latent, cfg.moe.moe_inter);
            ensure!(
                gate_up.len() == n * l * 2 * mi && down.len() == n * mi * l,
                "layer {i}: compact bank is not padded to {n} experts"
            );
            let idx = g.input("route_idx", Shape::new(&[rows, k], DType::F32));
            let cidx = g.input("route_cidx", Shape::new(&[rows, k], DType::F32));
            Some(FixedRouting {
                idx,
                compact_idx: cidx,
                gate_up,
                down,
                n,
            })
        }
        (FfnWeights::Moe(m), None) if m.experts_gate_up.is_empty() => {
            bail!("layer {i}: MoE weights without expert banks need a routing")
        }
        _ => None,
    };
    if with_taps {
        taps_begin();
    }
    let sn_ids = sn.clone();
    let built = build_layer(&mut g, &mut params, i, hn, sn, lw, cfg, fixed.as_ref());
    let taps = if with_taps { taps_take() } else { Vec::new() };
    let (h_out, _snaps_out) = built?;
    // Taps that are graph inputs would be outputs aliasing inputs; drop them
    // (the caller already holds those values).
    let mut inputs: Vec<rlx_ir::hir::HirNodeId> = vec![hn];
    inputs.extend(&sn_ids);
    if let Some(f) = &fixed {
        inputs.extend([f.idx, f.compact_idx]);
    }
    let taps: Vec<Tap> = taps
        .into_iter()
        .filter(|t| !inputs.contains(&t.node))
        .collect();
    let mut outs = vec![h_out];
    outs.extend(taps.iter().map(|t| t.node));
    g.set_outputs(outs);
    let (graph, params, shapes, adapters) = finish(hir, params, spec)?;
    Ok(LayerGraph {
        graph,
        params,
        shapes,
        adapters,
        taps,
        n_snaps_in,
    })
}

fn routing_feeds(r: Option<&LayerRouting>) -> Vec<(String, Vec<f32>)> {
    match r {
        Some(r) => vec![
            ("route_idx".into(), r.idx.clone()),
            ("route_cidx".into(), r.compact.clone()),
        ],
        None => Vec::new(),
    }
}

/// The snapshot list leaving layer `i`.
pub fn snaps_out(i: usize, cfg: &FlowConfig, h_in: &[f32], snaps_in: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut s = snaps_in.to_vec();
    if is_boundary(i, cfg) {
        s.push(h_in.to_vec());
    }
    s
}

/// Result of [`forward_layer`].
pub struct LayerForward {
    pub h_out: Vec<f32>,
    pub snaps_out: Vec<Vec<f32>>,
    pub routing: Option<LayerRouting>,
    /// `(tap, values)` when taps were requested.
    pub taps: Vec<(Tap, Vec<f32>)>,
}

/// Load layer `i` from `src`, route it, and run it forward.
pub fn forward_layer(
    src: &mut dyn WeightSource,
    i: usize,
    h: &[f32],
    snaps: &[Vec<f32>],
    spec: &LayerSpec<'_>,
    with_taps: bool,
) -> Result<LayerForward> {
    let lw = src.layer(i, spec.cfg)?;
    forward_layer_weights(src, i, &lw, h, snaps, spec, with_taps)
}

/// [`forward_layer`] on already-loaded layer weights.
pub fn forward_layer_weights(
    src: &mut dyn WeightSource,
    i: usize,
    lw: &LayerWeights,
    h: &[f32],
    snaps: &[Vec<f32>],
    spec: &LayerSpec<'_>,
    with_taps: bool,
) -> Result<LayerForward> {
    let routing = route_layer(i, lw, h, snaps, spec)?;
    let bank = match &routing {
        Some(r) => {
            let (gu, dn) = src.experts(i, &r.uniq, spec.cfg)?;
            Some(padded_bank(r, gu, dn, spec.cfg)?)
        }
        None => None,
    };
    let mut taps = Vec::new();
    let build = || -> Result<Built> {
        let lg = build_layer_graph(
            i,
            lw,
            snaps.len(),
            routing
                .as_ref()
                .zip(bank.clone())
                .map(|(r, (gu, dn))| (r, gu, dn)),
            spec,
            with_taps,
        )?;
        taps = lg.taps;
        Ok(Built {
            graph: lg.graph,
            params: lg.params,
            wrt: Vec::new(),
            shapes: lg.shapes,
        })
    };
    let (gn, dn) = bank_names(i);
    let extra: Vec<(String, &[f32])> = match &bank {
        Some((gu, d)) => vec![(gn, gu.as_slice()), (dn, d.as_slice())],
        None => Vec::new(),
    };
    let rf = routing_feeds(routing.as_ref());
    let mut f = feeds(h, snaps);
    f.extend(rf.iter().map(|(n, v)| (n.clone(), v.as_slice())));
    let refs: Vec<(&str, &[f32])> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    // Tapped graphs have extra outputs; never cache them under the plain key.
    let ran = if with_taps {
        let plain = LayerSpec {
            cache: None,
            ..*spec
        };
        run_graph(&plain, (KIND_FWD, i, snaps.len(), 0), build, &extra, &refs)?
    } else {
        let n = routing.as_ref().map_or(0, LayerRouting::bank_size);
        run_graph(spec, (KIND_FWD, i, snaps.len(), n), build, &extra, &refs)?
    };
    let mut outs = ran.outs.into_iter();
    let h_out = outs.next().context("layer output")?;
    let taps = taps.into_iter().zip(outs).collect();
    Ok(LayerForward {
        snaps_out: snaps_out(i, spec.cfg, h, snaps),
        h_out,
        routing,
        taps,
    })
}

/// Result of [`backward_layer`].
pub struct LayerBackward {
    /// Cotangent of the layer's input `h` (snapshot pass-through included).
    pub d_h: Vec<f32>,
    /// Cotangents of the incoming snapshots.
    pub d_snaps: Vec<Vec<f32>>,
    /// Gradients of the requested params.
    pub grads: Params,
    /// The values those params had (what an optimizer step starts from).
    pub values: Params,
    /// Their graph shapes (what export needs to lay them out).
    pub shapes: HashMap<String, Vec<usize>>,
}

/// VJP of layer `i` at `(h, snaps)` against `ct_h` (cotangent of `h_out`) and
/// `ct_snaps_out` (cotangents of the snapshot list leaving the layer). `wrt`
/// names the params to take gradients of. `routing` must be the selection the
/// forward made (recomputing it is what the forward did; passing it keeps the
/// two identical even when the weights moved in between).
#[allow(clippy::too_many_arguments)]
pub fn backward_layer(
    src: &mut dyn WeightSource,
    i: usize,
    lw: &LayerWeights,
    h: &[f32],
    snaps: &[Vec<f32>],
    routing: Option<&LayerRouting>,
    ct_h: &[f32],
    ct_snaps_out: &[Vec<f32>],
    wrt: &dyn Fn(&str) -> bool,
    spec: &LayerSpec<'_>,
) -> Result<LayerBackward> {
    let cfg = spec.cfg;
    let bank = match routing {
        Some(r) => {
            let (gu, dn) = src.experts(i, &r.uniq, cfg)?;
            Some(padded_bank(r, gu, dn, cfg)?)
        }
        None => None,
    };
    let n_snaps = snaps.len();
    let build = || -> Result<Built> {
        let lg = build_layer_graph(
            i,
            lw,
            n_snaps,
            routing.zip(bank.clone()).map(|(r, (gu, dn))| (r, gu, dn)),
            spec,
            false,
        )?;
        let mut params_wrt: Vec<String> = lg.shapes.keys().filter(|n| wrt(n)).cloned().collect();
        params_wrt.sort();
        let mut targets: Vec<rlx_autodiff::Wrt> = vec![rlx_autodiff::Wrt::Leaf("h".into())];
        targets.extend((0..n_snaps).map(|j| rlx_autodiff::Wrt::Leaf(input_name_snap(j))));
        targets.extend(params_wrt.iter().cloned().map(rlx_autodiff::Wrt::Leaf));
        let bwd = rlx_autodiff::grad_with_loss_wrt(
            &lg.graph,
            &targets,
            rlx_autodiff::GradWithLossOptions::TRAINING.with_aux(false),
        );
        Ok(Built {
            graph: bwd,
            params: lg.params,
            wrt: params_wrt,
            shapes: lg.shapes,
        })
    };
    let (gn, dn) = bank_names(i);
    let extra: Vec<(String, &[f32])> = match &bank {
        Some((gu, d)) => vec![(gn, gu.as_slice()), (dn, d.as_slice())],
        None => Vec::new(),
    };
    let rf = routing_feeds(routing);
    let mut f = feeds(h, snaps);
    f.extend(rf.iter().map(|(n, v)| (n.clone(), v.as_slice())));
    f.push(("d_output".into(), ct_h));
    let refs: Vec<(&str, &[f32])> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let n_bank = routing.map_or(0, LayerRouting::bank_size);
    let ran = run_graph(spec, (KIND_BWD, i, n_snaps, n_bank), build, &extra, &refs)?;
    let mut outs = ran.outs.into_iter();
    let _h_out = outs.next().context("backward: forward value")?;
    let mut d_h = outs.next().context("backward: d_h")?;
    let mut d_snaps: Vec<Vec<f32>> = (0..n_snaps)
        .map(|_| outs.next().context("backward: d_snap"))
        .collect::<Result<_>>()?;
    let mut grads = Params::new();
    for name in &ran.wrt {
        grads.insert(name.clone(), outs.next().context("backward: param grad")?);
    }
    let (values, shapes) = (ran.wrt_values, ran.wrt_shapes);
    // Host-threaded snapshot pass-through: snaps_out = snaps_in ++ [h if boundary].
    let m = snaps.len();
    ensure!(
        ct_snaps_out.len() == m + usize::from(is_boundary(i, cfg)),
        "layer {i}: {} snapshot cotangents for {} snapshots out",
        ct_snaps_out.len(),
        m + usize::from(is_boundary(i, cfg))
    );
    for (d, ct) in d_snaps.iter_mut().zip(ct_snaps_out) {
        add_into(d, ct);
    }
    if is_boundary(i, cfg) {
        add_into(&mut d_h, &ct_snaps_out[m]);
    }
    Ok(LayerBackward {
        d_h,
        d_snaps,
        grads,
        values,
        shapes,
    })
}

pub(crate) fn add_into(dst: &mut [f32], src: &[f32]) {
    for (a, b) in dst.iter_mut().zip(src) {
        *a += *b;
    }
}

/// Build the head `(h, snaps) → logits [b, s, vocab]` (+ taps).
pub fn build_head_graph(
    head: &HeadWeights,
    n_snaps: usize,
    spec: &LayerSpec<'_>,
    with_taps: bool,
) -> Result<LayerGraph> {
    let cfg = spec.cfg;
    let _canon = CanonicalParams::new();
    let mut hir = HirModule::new("kimi_head");
    let mut g = HirMut::new(&mut hir);
    let (hn, sn) = layer_inputs(&mut g, cfg, n_snaps);
    let mut params = Params::new();
    if with_taps {
        taps_begin();
    }
    let logits = build_head(
        &mut g,
        &mut params,
        hn,
        &sn,
        &head.final_norm,
        &head.out_res_norm,
        &head.out_res_proj,
        &head.lm_head,
        false,
        cfg,
    );
    let taps = if with_taps { taps_take() } else { Vec::new() };
    let mut outs = vec![logits];
    outs.extend(taps.iter().map(|t| t.node));
    g.set_outputs(outs);
    let (graph, params, shapes, adapters) = finish(hir, params, spec)?;
    Ok(LayerGraph {
        graph,
        params,
        shapes,
        adapters,
        taps,
        n_snaps_in: n_snaps,
    })
}

/// Next-token loss at the head and its cotangents.
pub struct HeadBackward {
    pub loss: f32,
    pub d_h: Vec<f32>,
    pub d_snaps: Vec<Vec<f32>>,
    pub grads: Params,
    pub values: Params,
    pub shapes: HashMap<String, Vec<usize>>,
}

/// Masked mean next-token cross-entropy of the head on `(h, snaps)` against
/// `labels` (`[b·s]` token ids as f32, `mask` 1/0 per position), with the
/// cotangents of `h` and every snapshot and the gradients of `wrt` params.
#[allow(clippy::too_many_arguments)]
pub fn head_loss_backward(
    head: &HeadWeights,
    h: &[f32],
    snaps: &[Vec<f32>],
    labels: &[f32],
    mask: &[f32],
    wrt: &dyn Fn(&str) -> bool,
    spec: &LayerSpec<'_>,
) -> Result<HeadBackward> {
    let cfg = spec.cfg;
    let n_snaps = snaps.len();
    let build = || -> Result<Built> {
        let lg = build_head_graph(head, n_snaps, spec, false)?;
        let mut graph = lg.graph;
        let rows = cfg.batch * cfg.seq;
        let logits = graph.outputs[0];
        let flat = graph.add_node(
            Op::Reshape {
                new_shape: vec![rows as i64, cfg.vocab as i64],
            },
            vec![logits],
            Shape::new(&[rows, cfg.vocab], DType::F32),
        );
        let lab = graph.input("labels", Shape::new(&[rows], DType::F32));
        let msk = graph.input("label_mask", Shape::new(&[rows], DType::F32));
        let loss = rlx_tune::cross_entropy_masked(&mut graph, flat, lab, msk);
        graph.set_outputs(vec![loss]);
        let mut params_wrt: Vec<String> = lg.shapes.keys().filter(|n| wrt(n)).cloned().collect();
        params_wrt.sort();
        let mut targets: Vec<rlx_autodiff::Wrt> = vec![rlx_autodiff::Wrt::Leaf("h".into())];
        targets.extend((0..n_snaps).map(|j| rlx_autodiff::Wrt::Leaf(input_name_snap(j))));
        targets.extend(params_wrt.iter().cloned().map(rlx_autodiff::Wrt::Leaf));
        let bwd = rlx_autodiff::grad_with_loss_wrt(
            &graph,
            &targets,
            rlx_autodiff::GradWithLossOptions::TRAINING.with_aux(false),
        );
        Ok(Built {
            graph: bwd,
            params: lg.params,
            wrt: params_wrt,
            shapes: lg.shapes,
        })
    };
    let one = [1.0f32];
    let mut f = feeds(h, snaps);
    f.push(("labels".into(), labels));
    f.push(("label_mask".into(), mask));
    f.push(("d_output".into(), &one));
    let refs: Vec<(&str, &[f32])> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let ran = run_graph(
        spec,
        (KIND_HEAD_BWD, usize::MAX, n_snaps, 0),
        build,
        &[],
        &refs,
    )?;
    let mut outs = ran.outs.into_iter();
    let loss = outs.next().context("head loss")?[0];
    let d_h = outs.next().context("head d_h")?;
    let d_snaps = (0..n_snaps)
        .map(|_| outs.next().context("head d_snap"))
        .collect::<Result<_>>()?;
    let mut grads = Params::new();
    for name in &ran.wrt {
        grads.insert(name.clone(), outs.next().context("head param grad")?);
    }
    Ok(HeadBackward {
        loss,
        d_h,
        d_snaps,
        grads,
        values: ran.wrt_values,
        shapes: ran.wrt_shapes,
    })
}

/// Run the head forward: logits `[b·s·vocab]` and taps.
pub fn forward_head(
    head: &HeadWeights,
    h: &[f32],
    snaps: &[Vec<f32>],
    spec: &LayerSpec<'_>,
    with_taps: bool,
) -> Result<(Vec<f32>, Vec<(Tap, Vec<f32>)>)> {
    let lg = build_head_graph(head, snaps.len(), spec, with_taps)?;
    let taps = lg.taps.clone();
    let mut c = compile(lg.graph, &lg.params, spec.device);
    let f = feeds(h, snaps);
    let refs: Vec<(&str, &[f32])> = f.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let mut outs = c.run(&refs).into_iter();
    let logits = outs.next().context("head logits")?;
    Ok((logits, taps.into_iter().zip(outs).collect()))
}

/// The weights of `lw`'s MoE FFN, if any.
pub fn moe_weights(lw: &LayerWeights) -> Option<&MoeWeights> {
    match &lw.ffn {
        FfnWeights::Moe(m) => Some(m),
        FfnWeights::Dense(_) => None,
    }
}
