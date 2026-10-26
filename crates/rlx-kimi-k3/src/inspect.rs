// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Look inside a Kimi-K3 checkpoint and a forward pass.
//!
//! * **Weights** — [`inventory`] lists every tensor (dtype, shape, bytes, shard)
//!   grouped by layer and component without reading tensor data;
//!   [`weight_stats`] reduces tensors one at a time to summary statistics;
//!   [`dump_weights`] writes selected tensors as f32 safetensors (MXFP4 experts
//!   dequantized under their logical `….weight` name).
//! * **Activations** — [`trace`] runs a forward one layer at a time and captures
//!   every named intermediate ([`crate::common::tap`]: the layer input, both
//!   Attention-Residual mixes, attention internals, router scores and picks,
//!   expert latents, shared experts, FFN output, …) plus each MoE layer's expert
//!   selection; with `grads` it also runs the backward of the next-token loss
//!   and records each layer's input cotangent. Results stream to a directory
//!   (one safetensors file per layer + `manifest.json` with statistics) so the
//!   whole model never has to be in memory.
//! * **Structure** — [`layer_graph_text`] prints one layer's op graph.

use crate::common::{CanonicalParams, Tap};
use crate::flow::{FlowConfig, build_layer};
use crate::layer_exec::{
    LayerSpec, backward_layer, forward_head, forward_layer_weights, head_loss_backward,
};
use crate::loader::CheckpointLoader;
use crate::source::WeightSource;
use anyhow::{Context, Result, ensure};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use safetensors::{Dtype, tensor::TensorView};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Summary statistics of a tensor (f64 accumulation).
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq)]
pub struct Stats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f32,
    pub max: f32,
    pub absmax: f32,
    pub rms: f64,
    /// Fraction of exact zeros.
    pub zeros: f64,
    pub nan: usize,
    pub inf: usize,
}

impl Stats {
    pub fn of(v: &[f32]) -> Self {
        let (mut s, mut s2) = (0f64, 0f64);
        let (mut min, mut max, mut absmax) = (f32::INFINITY, f32::NEG_INFINITY, 0f32);
        let (mut zeros, mut nan, mut inf, mut n) = (0usize, 0usize, 0usize, 0usize);
        for &x in v {
            if x.is_nan() {
                nan += 1;
                continue;
            }
            if x.is_infinite() {
                inf += 1;
                continue;
            }
            n += 1;
            s += x as f64;
            s2 += (x as f64) * (x as f64);
            min = min.min(x);
            max = max.max(x);
            absmax = absmax.max(x.abs());
            if x == 0.0 {
                zeros += 1;
            }
        }
        let nf = n.max(1) as f64;
        let mean = s / nf;
        Self {
            count: v.len(),
            mean,
            std: (s2 / nf - mean * mean).max(0.0).sqrt(),
            min: if n == 0 { 0.0 } else { min },
            max: if n == 0 { 0.0 } else { max },
            absmax,
            rms: (s2 / nf).sqrt(),
            zeros: zeros as f64 / nf,
            nan,
            inf,
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean {:+.4e} std {:.4e} min {:+.4e} max {:+.4e} |max| {:.4e} rms {:.4e} zeros {:.1}%",
            self.mean,
            self.std,
            self.min,
            self.max,
            self.absmax,
            self.rms,
            self.zeros * 100.0
        )?;
        if self.nan + self.inf > 0 {
            write!(f, " NaN {} Inf {}", self.nan, self.inf)?;
        }
        Ok(())
    }
}

/// One checkpoint tensor as stored.
#[derive(Debug, Clone, Serialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub shard: String,
    /// `embed`, `head`, `layer`, `vision`, `projector` or `other`.
    pub group: String,
    pub layer: Option<usize>,
    /// Within a layer: `attention`, `norms`, `attn_res`, `dense_mlp`,
    /// `moe_router`, `moe_dense`, `experts`; else the group.
    pub component: String,
}

/// Classify a checkpoint tensor name.
pub fn classify(name: &str) -> (String, Option<usize>, String) {
    if let Some(rest) = name.strip_prefix("language_model.model.layers.") {
        let (i, local) = rest.split_once('.').unwrap_or((rest, ""));
        let comp = if local.starts_with("self_attn.") {
            "attention"
        } else if local.contains("_res_") {
            "attn_res"
        } else if local.starts_with("mlp.") {
            "dense_mlp"
        } else if local.contains(".experts.") {
            "experts"
        } else if local.contains(".gate.") {
            "moe_router"
        } else if local.starts_with("block_sparse_moe.") {
            "moe_dense"
        } else if local.contains("layernorm") {
            "norms"
        } else {
            "other"
        };
        return ("layer".into(), i.parse().ok(), comp.into());
    }
    let g = if name.contains("embed_tokens") {
        "embed"
    } else if name.starts_with("language_model.lm_head")
        || name.starts_with("language_model.model.")
    {
        "head"
    } else if name.starts_with("vision_tower.") {
        "vision"
    } else if name.starts_with("mm_projector.") {
        "projector"
    } else {
        "other"
    };
    (g.into(), None, g.into())
}

/// Every tensor in the checkpoint, header-only.
pub fn inventory(ck: &mut CheckpointLoader) -> Result<Vec<TensorEntry>> {
    let mut out = Vec::new();
    for name in ck.tensor_names() {
        let (dtype, shape, bytes, shard) = ck.tensor_meta(&name)?;
        let (group, layer, component) = classify(&name);
        out.push(TensorEntry {
            name,
            dtype,
            shape,
            bytes,
            shard,
            group,
            layer,
            component,
        });
    }
    Ok(out)
}

/// Bytes and tensor counts per `(group, layer, component)`, plus totals per dtype.
#[derive(Debug, Default, Serialize)]
pub struct InventorySummary {
    pub tensors: usize,
    pub bytes: usize,
    pub by_dtype: BTreeMap<String, usize>,
    /// `group` or `layer.component` → (tensors, bytes)
    pub by_component: BTreeMap<String, (usize, usize)>,
    /// layer → bytes (routed experts included)
    pub by_layer: BTreeMap<usize, usize>,
}

pub fn summarize(entries: &[TensorEntry]) -> InventorySummary {
    let mut s = InventorySummary::default();
    for e in entries {
        s.tensors += 1;
        s.bytes += e.bytes;
        *s.by_dtype.entry(e.dtype.clone()).or_default() += e.bytes;
        let key = match e.layer {
            Some(_) => format!("layer.{}", e.component),
            None => e.group.clone(),
        };
        let c = s.by_component.entry(key).or_default();
        c.0 += 1;
        c.1 += e.bytes;
        if let Some(l) = e.layer {
            *s.by_layer.entry(l).or_default() += e.bytes;
        }
    }
    s
}

/// Logical tensors: MXFP4 `…_packed` / `…_scale` pairs fold into one `….weight`.
pub fn logical_names(entries: &[TensorEntry]) -> Vec<String> {
    let mut v: Vec<String> = entries
        .iter()
        .filter_map(|e| {
            if let Some(base) = e.name.strip_suffix("_packed") {
                Some(base.to_string())
            } else if e.name.ends_with("_scale") && e.dtype == "U8" {
                None
            } else {
                Some(e.name.clone())
            }
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Does `name` pass a filter: every `include` substring present (any when empty)
/// and the layer (if any) inside `layers`.
pub fn selected(name: &str, include: &[String], layers: Option<&std::ops::Range<usize>>) -> bool {
    if !include.is_empty() && !include.iter().any(|p| name.contains(p.as_str())) {
        return false;
    }
    match (layers, classify(name).1) {
        (Some(r), Some(l)) => r.contains(&l),
        (Some(_), None) => false,
        _ => true,
    }
}

/// Statistics of each selected logical tensor, read one at a time.
pub fn weight_stats(
    ck: &mut CheckpointLoader,
    names: &[String],
    mut on_each: impl FnMut(&str, &[usize], &Stats),
) -> Result<Vec<(String, Vec<usize>, Stats)>> {
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        let (shape, v) = ck.tensor_f32_any(n)?;
        let st = Stats::of(&v);
        on_each(n, &shape, &st);
        out.push((n.clone(), shape, st));
    }
    Ok(out)
}

/// Write named f32 tensors with shapes to one safetensors file.
pub fn write_f32_safetensors(
    path: &Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<()> {
    let bufs: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(n, s, v)| {
            (
                n.clone(),
                s.clone(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = bufs
        .iter()
        .map(|(n, s, b)| {
            TensorView::new(Dtype::F32, s.clone(), b)
                .map(|v| (n.clone(), v))
                .map_err(|e| anyhow::anyhow!("{n}: {e:?}"))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, None, path)
        .map_err(|e| anyhow::anyhow!("writing {path:?}: {e:?}"))
}

/// Dump selected logical tensors as f32 into `out_dir`: one
/// `weights-<group>.safetensors` per layer (`layer-NNN`) or top-level group,
/// plus `manifest.json` (shapes, source dtypes, statistics). Refuses to write
/// more than `max_bytes` of f32 unless it is `None`.
pub fn dump_weights(
    ck: &mut CheckpointLoader,
    names: &[String],
    out_dir: &Path,
    max_bytes: Option<usize>,
) -> Result<serde_json::Value> {
    std::fs::create_dir_all(out_dir)?;
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total = 0usize;
    for n in names {
        let elems: usize = match ck.tensor_meta(n) {
            Ok((_, s, _, _)) => s.iter().product(),
            Err(_) => {
                let (_, s, _, _) = ck.tensor_meta(&format!("{n}_packed"))?;
                s.iter().product::<usize>() * 2
            }
        };
        total += elems * 4;
        let (g, l, _) = classify(n);
        let key = match l {
            Some(l) => format!("layer-{l:03}"),
            None => g,
        };
        groups.entry(key).or_default().push(n.clone());
    }
    if let Some(cap) = max_bytes {
        ensure!(
            total <= cap,
            "the selection is {:.2} GB as f32 (limit {:.2} GB); narrow it with \
             --match / --layers or raise --max-gb",
            total as f64 / 1e9,
            cap as f64 / 1e9
        );
    }
    let mut manifest = serde_json::Map::new();
    for (key, ns) in groups {
        let file = format!("weights-{key}.safetensors");
        let mut tensors = Vec::with_capacity(ns.len());
        for n in &ns {
            let (shape, v) = ck.tensor_f32_any(n)?;
            let stored = ck
                .tensor_meta(n)
                .map(|m| m.0)
                .unwrap_or_else(|_| "MXFP4".into());
            manifest.insert(
                n.clone(),
                serde_json::json!({
                    "file": file,
                    "shape": shape,
                    "stored_dtype": stored,
                    "stats": Stats::of(&v),
                }),
            );
            tensors.push((n.clone(), shape, v));
        }
        write_f32_safetensors(&out_dir.join(&file), &tensors)?;
    }
    let m = serde_json::Value::Object(manifest);
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&m)?,
    )?;
    Ok(m)
}

/// What [`trace`] records.
#[derive(Debug, Clone)]
pub struct TraceOptions {
    /// Layers to run; the forward stops after the last (the head runs only when
    /// the range reaches the final layer).
    pub layers: std::ops::Range<usize>,
    /// Tap-name substrings to keep (all when empty). `l3.`, `.router`, `ffn_out`…
    pub include: Vec<String>,
    /// Write tensors; when false only statistics go to the manifest.
    pub tensors: bool,
    /// Also run the backward of the next-token loss on `labels` and record each
    /// layer's input cotangent `d l{i}.in` (and the head's input cotangent).
    pub grads: bool,
    pub device: rlx_runtime::Device,
}

/// One recorded activation.
#[derive(Debug, Clone, Serialize)]
pub struct TraceEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub stats: Stats,
    pub file: Option<String>,
}

/// Summary of a trace: every entry, each MoE layer's experts, and the head's
/// top-5 next tokens at the last position (when the head ran).
#[derive(Debug, Clone, Serialize, Default)]
pub struct TraceSummary {
    pub entries: Vec<TraceEntry>,
    /// layer → distinct experts picked
    pub experts: BTreeMap<usize, Vec<usize>>,
    pub top5: Vec<(u32, f32)>,
    pub loss: Option<f32>,
}

fn keep(name: &str, include: &[String]) -> bool {
    include.is_empty() || include.iter().any(|p| name.contains(p.as_str()))
}

/// Run `tokens` through the model one layer at a time and record every named
/// intermediate (and, with `opts.grads`, the per-layer input cotangents of the
/// next-token loss against `labels`). Writes `layer-NNN.safetensors`,
/// `head.safetensors` and `manifest.json` into `out_dir` when given.
pub fn trace(
    src: &mut dyn WeightSource,
    cfg: &FlowConfig,
    tokens: &[u32],
    labels: Option<&[u32]>,
    opts: &TraceOptions,
    out_dir: Option<&Path>,
) -> Result<TraceSummary> {
    ensure!(
        tokens.len() == cfg.batch * cfg.seq,
        "{} tokens for a batch·seq = {} config",
        tokens.len(),
        cfg.batch * cfg.seq
    );
    ensure!(
        opts.layers.end <= src.num_layers() && opts.layers.start == 0,
        "trace layers must start at 0 and end within {} (got {:?})",
        src.num_layers(),
        opts.layers
    );
    if let Some(d) = out_dir {
        std::fs::create_dir_all(d)?;
    }
    let store = HashMap::new();
    let spec = LayerSpec {
        cfg,
        device: opts.device,
        lora: None,
        overrides: &store,
        seed: 0,
        cache: None,
    };
    let mut summary = TraceSummary::default();
    let shape3 = vec![cfg.batch, cfg.seq, cfg.hidden];
    let record = |group: &str,
                  items: Vec<(String, Vec<usize>, Vec<f32>)>,
                  summary: &mut TraceSummary|
     -> Result<()> {
        let file = format!("{group}.safetensors");
        let mut tensors = Vec::new();
        for (name, shape, v) in items {
            if !keep(&name, &opts.include) {
                continue;
            }
            summary.entries.push(TraceEntry {
                stats: Stats::of(&v),
                file: (opts.tensors && out_dir.is_some()).then(|| file.clone()),
                name: name.clone(),
                shape: shape.clone(),
            });
            if opts.tensors {
                tensors.push((name, shape, v));
            }
        }
        if let (Some(d), false) = (out_dir, tensors.is_empty()) {
            write_f32_safetensors(&d.join(&file), &tensors)?;
        }
        Ok(())
    };
    let tap_items = |taps: Vec<(Tap, Vec<f32>)>| -> Vec<(String, Vec<usize>, Vec<f32>)> {
        taps.into_iter()
            .map(|(t, v)| (t.name, t.shape, v))
            .collect()
    };

    let mut h = src.embed(tokens, cfg.hidden)?;
    let mut snaps: Vec<Vec<f32>> = Vec::new();
    // Stash for the optional backward: (h_in, snapshot count, routing).
    let mut stash = Vec::new();
    record(
        "embed",
        vec![("embed".into(), shape3.clone(), h.clone())],
        &mut summary,
    )?;
    for i in opts.layers.clone() {
        let lw = src.layer(i, cfg)?;
        let out = forward_layer_weights(src, i, &lw, &h, &snaps, &spec, true)?;
        if let Some(r) = &out.routing {
            summary.experts.insert(i, r.uniq.clone());
        }
        let mut items = vec![(format!("l{i}.in"), shape3.clone(), h.clone())];
        items.extend(tap_items(out.taps));
        if let Some(r) = &out.routing {
            // The selection is a graph input under fixed routing, so it is not a
            // tap; record it here.
            items.push((
                format!("l{i}.block_sparse_moe.top_idx"),
                vec![cfg.batch * cfg.seq, cfg.moe.top_k],
                r.idx.clone(),
            ));
        }
        record(&format!("layer-{i:03}"), items, &mut summary)?;
        if opts.grads {
            stash.push((h.clone(), snaps.len(), out.routing.clone()));
        }
        h = out.h_out;
        snaps = out.snaps_out;
    }
    if opts.layers.end == src.num_layers() {
        let head = src.head(cfg)?;
        let (logits, taps) = forward_head(&head, &h, &snaps, &spec, true)?;
        let v = cfg.vocab;
        let last = &logits[(cfg.batch * cfg.seq - 1) * v..];
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|a, b| last[*b].total_cmp(&last[*a]));
        summary.top5 = idx.iter().take(5).map(|&t| (t as u32, last[t])).collect();
        let mut items = tap_items(taps);
        items.push(("logits".into(), vec![cfg.batch, cfg.seq, cfg.vocab], logits));
        if opts.grads {
            let labels = labels.context("activation gradients need labels")?;
            let lab: Vec<f32> = labels.iter().map(|&t| t as f32).collect();
            let mask = vec![1.0; lab.len()];
            let none = |_: &str| false;
            let hb = head_loss_backward(&head, &h, &snaps, &lab, &mask, &none, &spec)?;
            summary.loss = Some(hb.loss);
            items.push(("d head.in".into(), shape3.clone(), hb.d_h.clone()));
            record("head", items, &mut summary)?;
            let (mut ct_h, mut ct_snaps) = (hb.d_h, hb.d_snaps);
            for (li, i) in opts.layers.clone().enumerate().rev() {
                let (h_in, m, routing) = &stash[li];
                let lw = src.layer(i, cfg)?;
                let m_out = m + usize::from(crate::layer_exec::is_boundary(i, cfg));
                let b = backward_layer(
                    src,
                    i,
                    &lw,
                    h_in,
                    &snaps[..*m],
                    routing.as_ref(),
                    &ct_h,
                    &ct_snaps[..m_out],
                    &none,
                    &spec,
                )?;
                ct_h = b.d_h;
                ct_snaps.truncate(*m);
                for (dst, d) in ct_snaps.iter_mut().zip(b.d_snaps) {
                    *dst = d;
                }
                record(
                    &format!("grad-layer-{i:03}"),
                    vec![(format!("d l{i}.in"), shape3.clone(), ct_h.clone())],
                    &mut summary,
                )?;
            }
        } else {
            record("head", items, &mut summary)?;
        }
    } else {
        ensure!(
            !opts.grads,
            "activation gradients need the full layer range"
        );
    }
    if let Some(d) = out_dir {
        std::fs::write(
            d.join("manifest.json"),
            serde_json::to_vec_pretty(&summary)?,
        )?;
    }
    Ok(summary)
}

/// One layer's op graph (canonical params) as text: `stats` = op histogram,
/// else the full node listing.
pub fn layer_graph_text(
    src: &mut dyn WeightSource,
    cfg: &FlowConfig,
    i: usize,
    stats: bool,
) -> Result<String> {
    let lw = src.layer(i, cfg)?;
    let _canon = CanonicalParams::new();
    let mut hir = HirModule::new(format!("kimi_layer_l{i}"));
    let mut g = HirMut::new(&mut hir);
    let shape = Shape::new(&[cfg.batch, cfg.seq, cfg.hidden], DType::F32);
    let h = g.input("h", shape.clone());
    // The snapshot count a full forward would carry into this layer.
    let n_snaps = (0..i)
        .filter(|j| j.is_multiple_of(cfg.attn_res_block_size))
        .count();
    let snaps = (0..n_snaps)
        .map(|j| g.input(format!("snap{j}"), shape.clone()))
        .collect();
    let mut params = HashMap::new();
    let routing = match crate::layer_exec::moe_weights(&lw) {
        Some(_) => {
            let rows = cfg.batch * cfg.seq;
            let k = cfg.moe.top_k;
            let (l, mi) = (cfg.moe.latent, cfg.moe.moe_inter);
            let n = k.next_power_of_two();
            Some(crate::moe::FixedRouting {
                idx: g.input("route_idx", Shape::new(&[rows, k], DType::F32)),
                compact_idx: g.input("route_cidx", Shape::new(&[rows, k], DType::F32)),
                gate_up: vec![0.0; n * l * 2 * mi],
                down: vec![0.0; n * mi * l],
                n,
            })
        }
        None => None,
    };
    let (out, _) = build_layer(&mut g, &mut params, i, h, snaps, &lw, cfg, routing.as_ref())?;
    g.set_outputs(vec![out]);
    Ok(if stats {
        rlx_ir::inspect_hir_stats(&hir)
    } else {
        rlx_ir::inspect_hir(&hir)
    })
}
