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

//! The `cr_td` (CRAFT-style) text detector as a native rlx-ir graph.
//!
//! The detector is a 265-op graph (grouped-conv encoder + SE blocks + FPN + 7 heads).
//! Rather than hand-code it, we interpret a compact recipe (validated head-by-head to
//! cos≈1.0 against a numeric fixture). Input is host-side imagenet-normalized RGB
//! `[1,3,480,480]`; outputs are the 7 heatmap heads.

use crate::graph::OcrGraphBuilder;
use anyhow::{Result, anyhow, bail};
use rlx_core::vision_ops_ir::{avg_pool2d, conv2d_bias, conv2d_bias_groups, max_pool2d_2x2};
use rlx_core::weight_map::WeightMap;
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::Activation;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Deserialize)]
struct Recipe {
    input_shape: Vec<usize>,
    ops: Vec<OpDef>,
    outputs: Vec<String>,
}

#[derive(Deserialize)]
struct OpDef {
    op: String,
    name: String,
    #[serde(rename = "in")]
    ins: Vec<String>,
    out: String,
    shape: Vec<usize>, // [n,k,h,w]
    #[serde(default)]
    w: Option<String>,
    #[serde(default)]
    b: Option<String>,
    #[serde(default)]
    kernel: Option<Vec<usize>>,
    #[serde(default)]
    groups: Option<usize>,
    #[serde(default)]
    stride: Option<Vec<usize>>,
    #[serde(default)]
    pad: Option<Vec<usize>>, // [pt,pb,pl,pr] (symmetric in this model)
    #[serde(default)]
    relu: Option<bool>,
    #[serde(default)]
    max: Option<f32>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    global: Option<bool>,
    #[serde(default)]
    start: Option<usize>,
    #[serde(default)]
    end: Option<usize>,
}

/// Build the detector graph from its recipe JSON. Output order matches `recipe.outputs`.
pub fn build_detector_graph(
    recipe_json: &str,
    wm: &mut WeightMap,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    build_detector_graph_heads(recipe_json, wm, None)
}

/// Like [`build_detector_graph`] but restricts the outputs to `want` (a subset of the recipe's
/// head names). Only the ops feeding those heads are built — the rest are pruned by backward
/// reachability. Passing `None` keeps all 7 heads (parity path). The pipeline only consumes
/// `region_score` + `link_score_horizontal`, so pruning the other 5 heads drops ~28% of ops
/// (incl. every upsample/softmax feeding them) with zero effect on the retained outputs.
pub fn build_detector_graph_heads(
    recipe_json: &str,
    wm: &mut WeightMap,
    want: Option<&[String]>,
) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let recipe: Recipe = serde_json::from_str(recipe_json)?;
    let outputs: Vec<String> = match want {
        Some(w) => w.to_vec(),
        None => recipe.outputs.clone(),
    };

    // Backward reachability: which op outputs are needed to produce `outputs`. Recipe ops are in
    // topological order, so a single reverse pass propagates need from consumers to producers.
    let mut needed: std::collections::HashSet<String> = outputs.iter().cloned().collect();
    for o in recipe.ops.iter().rev() {
        if needed.contains(&o.out) {
            for i in &o.ins {
                needed.insert(i.clone());
            }
        }
    }

    let mut b = OcrGraphBuilder::new("ocr2_detector");
    let batch = recipe.input_shape[0];
    let mut node: HashMap<String, HirNodeId> = HashMap::new();
    let mut shp: HashMap<String, Vec<usize>> = HashMap::new();

    let x = b.m().input("x", Shape::new(&recipe.input_shape, DType::F32));
    node.insert("x".into(), x);
    shp.insert("x".into(), recipe.input_shape.clone());

    for o in &recipe.ops {
        if !needed.contains(&o.out) {
            continue; // op feeds only pruned heads
        }
        let s = &o.shape;
        let (oc, oh, ow) = (s[1], s[2], s[3]);
        let get = |m: &HashMap<String, HirNodeId>, k: &str| -> Result<HirNodeId> {
            m.get(k).copied().ok_or_else(|| anyhow!("missing node {k} for op {}", o.name))
        };
        let id = match o.op.as_str() {
            "conv" => {
                let xin = get(&node, &o.ins[0])?;
                let weight = b.load_param(wm, o.w.as_ref().unwrap())?;
                let bias = match &o.b {
                    Some(k) => b.load_param(wm, k)?,
                    None => b.zeros(&format!("{}.zb", o.name), &[oc]),
                };
                let k = o.kernel.clone().unwrap();
                let st = o.stride.clone().unwrap();
                let pad = o.pad.clone().unwrap();
                let (ph, pw) = (pad[0], pad[2]);
                let groups = o.groups.unwrap_or(1);
                let y = if groups == 1 {
                    conv2d_bias(&mut b.m(), xin, weight, bias, batch, oc, k[0], k[1],
                                [st[0], st[1]], [ph, pw], oh, ow)
                } else {
                    conv2d_bias_groups(&mut b.m(), xin, weight, bias, batch, oc, k[0], k[1],
                                       [st[0], st[1]], [ph, pw], groups, oh, ow)
                };
                if o.relu.unwrap_or(false) { b.m().relu(y) } else { y }
            }
            "relu" => { let x = get(&node, &o.ins[0])?; b.m().relu(x) }
            "sigmoid" => {
                let x = get(&node, &o.ins[0])?;
                b.m().activation(Activation::Sigmoid, x, Shape::new(s, DType::F32))
            }
            "add" => {
                let y = b.m().add(get(&node, &o.ins[0])?, get(&node, &o.ins[1])?);
                if o.relu.unwrap_or(false) { b.m().relu(y) } else { y }
            }
            "mul" => b.m().mul(get(&node, &o.ins[0])?, get(&node, &o.ins[1])?),
            "clamp" => {
                // Inputs are ReLU'd (>=0); clamp_max to beta = beta - relu(beta - x).
                let x = get(&node, &o.ins[0])?;
                let beta = o.max.unwrap_or(f32::INFINITY);
                let bc = b.const_full(&format!("{}.beta", o.name), &[1, 1, 1, 1], beta);
                let t = b.m().sub(bc, x);
                let t = b.m().relu(t);
                b.m().sub(bc, t)
            }
            "pool" => {
                let x = get(&node, &o.ins[0])?;
                let ish = shp.get(&o.ins[0]).unwrap();
                let (ih, iw) = (ish[2], ish[3]);
                if o.global.unwrap_or(false) {
                    b.m().mean(x, vec![2, 3], true) // SE squeeze -> [1,C,1,1]
                } else if o.kind.as_deref() == Some("max") {
                    max_pool2d_2x2(&mut b.m(), x, batch, oc, ih, iw)
                } else {
                    avg_pool2d(&mut b.m(), x, [2, 2], [2, 2], batch, oc, ih, iw)
                }
            }
            "upsample" => { let x = get(&node, &o.ins[0])?; b.m().resize_bilinear2d(x, oh, ow, false) }
            "concat" => {
                let ids: Result<Vec<_>> = o.ins.iter().map(|n| get(&node, n)).collect();
                b.m().concat_(ids?, 1)
            }
            "slice" => {
                let x = get(&node, &o.ins[0])?;
                let (st, en) = (o.start.unwrap(), o.end.unwrap());
                b.m().narrow_(x, 1, st, en - st)
            }
            "softmax" => {
                // Channel-axis softmax from primitives (rlx `sm` axis handling is unreliable on
                // 4D NCHW). Mean-center first for numerical stability (softmax is shift-invariant).
                let x = get(&node, &o.ins[0])?;
                let mu = b.m().mean(x, vec![1], true); // [1,1,H,W]
                let xc = b.m().sub(x, mu);
                let e = b.m().activation(Activation::Exp, xc, Shape::new(s, DType::F32));
                let denom = b.m().sum(e, vec![1], true);
                b.m().div(e, denom)
            }
            "copy" => get(&node, &o.ins[0])?,
            other => bail!("unknown detector op {other}"),
        };
        node.insert(o.out.clone(), id);
        shp.insert(o.out.clone(), s.clone());
    }

    let outs: Result<Vec<_>> = outputs.iter().map(|h| get_out(&node, h)).collect();
    b.m().set_outputs(outs?);
    b.finish()
}

fn get_out(m: &HashMap<String, HirNodeId>, k: &str) -> Result<HirNodeId> {
    m.get(k).copied().ok_or_else(|| anyhow!("missing output head {k}"))
}

/// Detector runner: loads recipe + weights, compiles, runs a normalized image.
pub struct Detector {
    recipe_json: String,
    weights_path: PathBuf,
    heads: Vec<String>,
    device: Device,
    compiled: Mutex<Option<CompiledGraph>>, // compiled once, reused across calls
}

impl Detector {
    /// Full 7-head detector (all heads; used by parity tests + the CLI).
    pub fn load(recipe: &Path, weights: &Path, device: Device) -> Result<Self> {
        let recipe_json = std::fs::read_to_string(recipe)?;
        let r: Recipe = serde_json::from_str(&recipe_json)?;
        Self::from_json(recipe_json, weights, device, r.outputs)
    }

    /// Detector restricted to `heads` (a subset of the recipe's outputs). Ops feeding only the
    /// other heads are pruned at build time — the end-to-end pipeline uses this to compute just
    /// the two heads grouping needs (`region_score`, `link_score_horizontal`).
    pub fn load_heads(
        recipe: &Path,
        weights: &Path,
        device: Device,
        heads: Vec<String>,
    ) -> Result<Self> {
        let recipe_json = std::fs::read_to_string(recipe)?;
        Self::from_json(recipe_json, weights, device, heads)
    }

    fn from_json(
        recipe_json: String,
        weights: &Path,
        device: Device,
        heads: Vec<String>,
    ) -> Result<Self> {
        Ok(Self {
            recipe_json,
            weights_path: weights.to_path_buf(),
            heads,
            device,
            compiled: Mutex::new(None),
        })
    }

    pub fn head_names(&self) -> &[String] {
        &self.heads
    }

    /// `input` = host-normalized RGB `[1,3,480,480]` row-major. Returns `(head, data)` per head.
    /// The graph is built + compiled on the first call and cached; later calls only run it.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<(String, Vec<f32>)>> {
        let mut guard = self.compiled.lock().map_err(|_| anyhow!("lock poisoned"))?;
        if guard.is_none() {
            let path = self.weights_path.to_str().ok_or_else(|| anyhow!("weights path not UTF-8"))?;
            let mut wm = WeightMap::from_file(path)?;
            let (graph, params) =
                build_detector_graph_heads(&self.recipe_json, &mut wm, Some(&self.heads))?;
            *guard = Some(crate::compile::compile_encoder(
                graph,
                params,
                self.device,
                crate::env::no_fusion(),
            ));
        }
        let outs = guard.as_mut().unwrap().run(&[("x", input)]);
        if outs.len() != self.heads.len() {
            bail!("detector produced {} outputs, expected {}", outs.len(), self.heads.len());
        }
        Ok(self.heads.iter().cloned().zip(outs).collect())
    }
}
