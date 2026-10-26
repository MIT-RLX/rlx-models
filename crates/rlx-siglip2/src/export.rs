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

//! Export a fixed-resolution SigLIP 2 checkpoint's image tower as an
//! `.rlxp` package that rlx runs with no rlx-models code: a `pipeline`
//! entry whose stages are the image preprocessing (an RGB image of any
//! size → normalized pixels, as a graph), the patch embedding and the
//! vision encoder with its pooling head. The preprocessing can be its own
//! package, shared by every model that takes the same pixels, and runs
//! alone as one.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::{Checkpoint, Package, package_dir, param_dims, relative_path};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::model::Tensor;
use serde_json::{Value, json};

use crate::config::Variant;
use crate::{Siglip2Config, extract_pooling_weights};

/// What both packages take: an RGB image of any size, 8 bits a channel.
static IMAGE_CONTRACT: std::sync::LazyLock<Value> = std::sync::LazyLock::new(
    || json!({"name": "image", "kind": "image", "dtype": "u8", "shape": [null, null, 3], "color": "rgb", "layout": "hwc", "range": [0, 255]}),
);

/// The longest image side the preprocessing takes (its index table).
pub const MAX_SIDE: usize = 8192;

pub use rlx_core::export::{ExportOptions, ExportReport};

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// Pillow's fixed point: coefficients are integers scaled by `2^22`, a
/// sum starts at `2^21` (round half up) and is shifted down by 22.
const PRECISION_BITS: i32 = 22;
/// Coefficients split `a·2^7 + b` so every sum of a pass is an integer
/// below `2^24`, exact in f32 in any order: `Σ p·a ≤ 255·2^15`, and
/// `Σ p·b ≤ 255·127·taps` up to ~450 taps (8192 px shrunk to 224 is 75).
const SPLIT_BITS: i32 = 7;

/// Pillow's antialiased bilinear resample of `n` samples to `out`, as an
/// `[out, n]` matrix of its integer coefficients: output `i` centered at
/// `(i + 0.5)·n/out`, the triangle widened by `max(n/out, 1)` when
/// shrinking, normalized, then `round(k·2^22)`. `n_value` is `[1, 1]`
/// holding `n`, so the matrix follows the input's size when the graph is
/// symbolic in it.
///
/// The triangle is taken in integers, `max(0, 2·max(n, out) − |(2j + 1)·out
/// − (2i + 1)·n|)`, exact in f32 (all below `2^24`), so a coefficient
/// rounds once, in the division that normalizes it. In f32 the distance
/// `j + 0.5 − center` cancels: off by up to 73 in `2^22`, enough to move
/// one output in a hundred by a step.
fn resample_matrix(g: &mut Graph, iota: NodeId, n: usize, n_value: NodeId, out: usize) -> NodeId {
    let one = g.constant(1.0, DType::F32);
    let two = g.constant(2.0, DType::F32);
    let out_value = g.constant(out as f64, DType::F32);
    // (2i + 1)·n, [out, 1].
    let i = g.narrow_(iota, 0, 0, out);
    let i = g.reshape_(i, vec![out as i64, 1]);
    let i = g.mul(i, two);
    let i = g.add(i, one);
    let center = g.mul(i, n_value);
    // (2j + 1)·out, [1, n].
    let j = g.narrow_(iota, 0, 0, n);
    let j = g.reshape_(j, vec![1, n as i64]);
    let j = g.mul(j, two);
    let j = g.add(j, one);
    let j = g.mul(j, out_value);
    let d = g.sub(j, center);
    let d = g.abs(d);
    let width = g.binary(BinaryOp::Max, n_value, out_value, f32s(&[1, 1]));
    let width = g.mul(width, two);
    let w = g.sub(width, d);
    let w = g.relu(w);
    let total = g.reduce(w, ReduceOp::Sum, vec![1], true, f32s(&[out, 1]));
    let w = g.div(w, total);
    let fixed = g.constant(f64::from(1 << PRECISION_BITS), DType::F32);
    let w = g.mul(w, fixed);
    let half = g.constant(0.5, DType::F32);
    let w = g.add(w, half);
    floor(g, w)
}

fn floor(g: &mut Graph, x: NodeId) -> NodeId {
    let shape = g.node(x).shape.clone();
    g.activation(Activation::Floor, x, shape)
}

/// `x [rows, n]` (8-bit values) through Pillow's pass with the integer
/// coefficients `k [out, n]`: `clamp((2^21 + x·kᵀ) >> 22, 0, 255)`, exact.
fn resample_pass(g: &mut Graph, x: NodeId, k: NodeId) -> NodeId {
    let step = g.constant(f64::from(1 << SPLIT_BITS), DType::F32);
    let inv_step = g.constant(1.0 / f64::from(1 << SPLIT_BITS), DType::F32);
    let hi = g.mul(k, inv_step);
    let hi = floor(g, hi);
    let lo = g.mul(hi, step);
    let lo = g.sub(k, lo);
    let hi_t = g.transpose_(hi, vec![1, 0]);
    let lo_t = g.transpose_(lo, vec![1, 0]);
    let a = g.mm(x, hi_t);
    let b = g.mm(x, lo_t);
    // ⌊(a·2^7 + b + 2^21) / 2^22⌋ = ⌊(a + ⌊(b + 2^21) / 2^7⌋) / 2^15⌋.
    let round = g.constant(f64::from(1 << (PRECISION_BITS - 1)), DType::F32);
    let b = g.add(b, round);
    let b = g.mul(b, inv_step);
    let b = floor(g, b);
    let sum = g.add(a, b);
    let down = g.constant(
        1.0 / f64::from(1 << (PRECISION_BITS - SPLIT_BITS)),
        DType::F32,
    );
    let sum = g.mul(sum, down);
    let sum = floor(g, sum);
    g.clamp_(sum, 0.0, 255.0)
}

/// SigLIP's preprocessing at an `h × w` input: `image [h, w, 3]` (RGB,
/// 0–255) and its dims → `pixels [1, 3, size, size]`. Resized exactly to
/// `size × size` as Pillow does, in its fixed point (horizontal pass,
/// 8-bit store, vertical pass, 8-bit store), then `/255`, then
/// `(v - 0.5) / 0.5`.
fn preprocess_graph_at(h: usize, w: usize, size: usize) -> Graph {
    let mut g = Graph::new("preprocess");
    let image = g.input("image", f32s(&[h, w, 3]));
    let dims = g.input("dims", f32s(&[3]));
    let iota = g.param("preprocess.iota", f32s(&[MAX_SIDE]));
    let side = |g: &mut Graph, axis: usize| {
        let v = g.narrow_(dims, 0, axis, 1);
        g.reshape_(v, vec![1, 1])
    };
    let (h_value, w_value) = (side(&mut g, 0), side(&mut g, 1));
    let wx = resample_matrix(&mut g, iota, w, w_value, size);
    let wy = resample_matrix(&mut g, iota, h, h_value, size);
    // [h, w, 3] → rows of [3·h, w] → [3·h, size].
    let x = g.transpose_(image, vec![2, 0, 1]);
    let x = g.reshape_(x, vec![(3 * h) as i64, w as i64]);
    let x = resample_pass(&mut g, x, wx);
    // [3, h, size] → [3, size, h] → rows of [3·size, h] → [3·size, size].
    let x = g.reshape_(x, vec![3, h as i64, size as i64]);
    let x = g.transpose_(x, vec![0, 2, 1]);
    let x = g.reshape_(x, vec![(3 * size) as i64, h as i64]);
    let x = resample_pass(&mut g, x, wy);
    // [3, x, y] → [3, y, x], then the normalization.
    let x = g.reshape_(x, vec![3, size as i64, size as i64]);
    let x = g.transpose_(x, vec![0, 2, 1]);
    let scale = g.constant(2.0 / 255.0, DType::F32);
    let x = g.mul(x, scale);
    let one = g.constant(1.0, DType::F32);
    let x = g.sub(x, one);
    let out = g.reshape_(x, vec![1, 3, size as i64, size as i64]);
    g.set_outputs(vec![out]);
    g
}

/// SigLIP's preprocessing, symbolic in the image's height (`height`) and
/// width (`width`).
pub fn preprocess_graph(size: usize) -> Result<Graph> {
    // Two builds at sizes no other dim has, generalized.
    let (a, b) = (
        preprocess_graph_at(301, 307, size),
        preprocess_graph_at(311, 313, size),
    );
    rlx_ir::dynamic::generalize(&a, &b, &[(0, 301, 311), (1, 307, 313)])
        .map_err(|e| anyhow::anyhow!("the preprocessing is not symbolic in the image size: {e}"))
}

/// The preprocessing's own tensor.
fn preprocess_tensors() -> Vec<Tensor> {
    let iota: Vec<f32> = (0..MAX_SIDE).map(|i| i as f32).collect();
    vec![Tensor::f32("preprocess.iota", vec![MAX_SIDE], &iota)]
}

/// The preprocessing package's model block: it runs alone as a pipeline.
fn preprocess_block(size: usize) -> Value {
    json!({
        "name": format!("siglip-preprocess-{size}"),
        "task": "image-preprocessing",
        "entries": {"pipeline": {
            "task": "image-preprocessing",
            "inputs": [IMAGE_CONTRACT.clone()],
            "stages": [preprocess_stage(None)],
            "outputs": [{"name": "pixels", "kind": "tensor", "dtype": "f32", "shape": [1, 3, size, size], "layout": "nchw"}],
        }},
        "metadata": {
            "input": "RGB [height, width, 3], 0-255",
            "output": format!("pixels [1, 3, {size}, {size}], bilinear (Pillow), mean = std = 0.5"),
        },
    })
}

/// The preprocessing stage. It runs on the CPU, where its sums and its
/// one division are IEEE-exact; a GPU's fast division moves a coefficient
/// (Metal: 2–16 values a step off). Stages pass values through the host,
/// so this costs no copy.
fn preprocess_stage(package: Option<String>) -> Value {
    let mut s = json!({
        "function": "preprocess",
        "device": "cpu",
        "inputs": {"image": "image", "dims": "image#shape"},
        "outputs": ["pixels"],
    });
    if let Some(p) = package {
        s["package"] = json!(p);
    }
    s
}

/// Patches of `pixels [1, 3, size, size]` through the conv-as-matmul, plus
/// its bias and the position embedding → `hidden [1, patches, width]`.
fn patch_embed_graph(cfg: &Siglip2Config) -> Graph {
    let v = &cfg.vision;
    let (size, ps, width) = (v.image_size, v.patch_size, v.width);
    let side = size / ps;
    let np = side * side;
    let pd = 3 * ps * ps;
    let mut g = Graph::new("patch_embed");
    let pixels = g.input("pixels", f32s(&[1, 3, size, size]));
    let conv = g.param(
        "vision_model.embeddings.patch_embedding.weight",
        f32s(&[width, 3, ps, ps]),
    );
    let bias = g.param(
        "vision_model.embeddings.patch_embedding.bias",
        f32s(&[width]),
    );
    let pos = g.param(
        "vision_model.embeddings.position_embedding.weight",
        f32s(&[np, width]),
    );
    // [3, py, ry, px, rx] → [py, px, 3, ry, rx]: one row per patch, in the
    // conv weight's [c, ry, rx] order.
    let p = g.reshape_(
        pixels,
        vec![3, side as i64, ps as i64, side as i64, ps as i64],
    );
    let p = g.transpose_(p, vec![1, 3, 0, 2, 4]);
    let p = g.reshape_(p, vec![np as i64, pd as i64]);
    let w = g.reshape_(conv, vec![width as i64, pd as i64]);
    let w = g.transpose_(w, vec![1, 0]);
    let h = g.mm(p, w);
    let h = g.add(h, bias);
    let h = g.add(h, pos);
    let out = g.reshape_(h, vec![1, np as i64, width as i64]);
    g.set_outputs(vec![out]);
    g
}

/// Export the SigLIP 2 checkpoint in `dir` (fixed resolution) as the
/// package `out`: an RGB image in, its image embedding (`pooler_output`)
/// out.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = Siglip2Config::from_hf_config_json(&dir.join("config.json"))?;
    if cfg.variant != Variant::Fixed {
        bail!("only fixed-resolution SigLIP 2 exports (NaFlex packs patches on the host)");
    }
    let st = dir.join("model.safetensors");
    let st_str = st.to_str().context("weights path is not UTF-8")?;
    let source = WeightMap::from_file(st_str)?;
    let mut wm = WeightMap::from_file(st_str)?;

    // The patch embedding reads the checkpoint's tensors as stored.
    let embed = patch_embed_graph(&cfg);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    for name in param_dims(&embed).keys() {
        let (v, _) = source
            .get(name)
            .with_context(|| format!("the checkpoint has no '{name}'"))?;
        params.insert(name.clone(), v.to_vec());
    }
    let pooling = extract_pooling_weights(&mut wm, &cfg, 1)?;
    let (mut vision, p) =
        graph_from_built(crate::flow::build_vision_flow(&cfg, &mut wm, 1, pooling)?)?;
    vision.name = "vision".into();
    params.extend(p);
    let hidden_in = vision
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            rlx_ir::Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .context("the vision graph has no input")?;
    let mut graphs = vec![embed, vision];

    // The preprocessing: in this package, or its own.
    let size = cfg.vision.image_size;
    let pre = preprocess_graph(size)?;
    let pre_dims = vec![("height".to_string(), 0), ("width".to_string(), 1)];
    let mut pkg = Package::for_checkpoint(
        &Checkpoint::open(dir)?,
        opts,
        "siglip2",
        "image-feature-extraction",
        "rlx-siglip2",
    );
    let package = match &opts.frontend_package {
        Some(path) => {
            let mut fe = Package::new("", &opts.version, "siglip2", "", "rlx-siglip2");
            fe.model = preprocess_block(size);
            fe.function(pre);
            fe.dims = pre_dims;
            fe.held = preprocess_tensors();
            fe.write(path, opts)?;
            let rel = relative_path(
                package_dir(out),
                &path.canonicalize().unwrap_or(path.clone()),
            );
            Some(rel.display().to_string())
        }
        None => {
            graphs.insert(0, pre);
            pkg.held.extend(preprocess_tensors());
            pkg.dims = pre_dims;
            None
        }
    };
    pkg.graphs = graphs;
    pkg.reference_map(
        &params,
        &source,
        "safetensors",
        &st,
        out,
        opts.embed_weights,
    )?;
    drop(source);
    pkg.entry(
        "pipeline",
        json!({
            "task": "image-feature-extraction",
            "inputs": [IMAGE_CONTRACT.clone()],
            "stages": [
                preprocess_stage(package),
                {"function": "patch_embed", "inputs": {"pixels": "pixels"}, "outputs": ["hidden"]},
                {"function": "vision", "inputs": {hidden_in: "hidden"}, "outputs": ["embedding"]},
            ],
            "outputs": [{"name": "embedding", "kind": "embedding", "dtype": "f32", "shape": [1, cfg.embed_dim]}],
        }),
    );
    pkg.metadata("image_size", json!(size));
    pkg.metadata("embedding", json!("pooler_output (not normalized)"));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("siglip2", "image-feature-extraction", export_rlxp)
            .claims(&["siglip", "siglip2"]),
    );
}
