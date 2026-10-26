// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export subject lifting as an `.rlxp` package rlx runs alone: a
//! `pipeline` from an RGB image of any size to its foreground alpha and
//! gating confidence, every step a graph — `imageops::resize` (Triangle) to
//! the network's square and the `[-1, 1]` encoding
//! ([`rlx_core::export::vision::image_resize_normalize`]), the network, the
//! saliency feathered ([`crate::FEATHER`], a separable Gaussian with clamped
//! borders as two matmuls) and resized back to the image
//! ([`rlx_core::export::vision::image_resize_mask`]).
//! `Model.run({image: u8 [h, w, 3]})` returns `alpha` (`f32 [h, w]`, 0–1)
//! and `gating` (`f32 [1]`).
//!
//! The network's weights are the vendor's, read from the copy macOS ships; the
//! package holds them. Only [`crate::LiftOptions::default`]'s settings are
//! exported.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rlx_core::export::graph::{Pipeline, Stage};
use rlx_core::export::vision::{
    CLIP_IOTA, HEIGHT, ImageFilter, WIDTH, clip_iota, image_contract, image_resize_mask,
    image_resize_normalize,
};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_neuralhash::NetFile;

use crate::{FEATHER, InputNorm, LiftOptions, SYSTEM_MODEL, Upsample};

/// The feather as a matrix: `B[i, j]` the weight row `i` takes from `j`
/// (the Gaussian's taps, clamped at the borders, normalized).
fn blur_matrix(side: usize, sigma: f32) -> Vec<f32> {
    let mut b = vec![0f32; side * side];
    if sigma <= 0.0 {
        for i in 0..side {
            b[i * side + i] = 1.0;
        }
        return b;
    }
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r)
        .map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f32 = k.iter().sum();
    for i in 0..side {
        for d in -r..=r {
            let j = (i as isize + d).clamp(0, side as isize - 1) as usize;
            b[i * side + j] += k[(d + r) as usize] / norm;
        }
    }
    b
}

/// The network's output `[side² + 1]` → `mask` (the saliency, feathered:
/// `B · S · Bᵀ`) and `gating`.
fn heads(side: usize) -> Graph {
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let n = side * side;
    let mut g = Graph::new("heads");
    let out = g.input("output", f32s(&[n + 1]));
    let s = g.narrow_(out, 0, 0, n);
    let s = g.reshape_(s, vec![side as i64, side as i64]);
    let b = g.param("feather", f32s(&[side, side]));
    let s = g.mm(b, s);
    let bt = g.transpose_(b, vec![1, 0]);
    let s = g.mm(s, bt);
    let mask = g.reshape_(s, vec![n as i64]);
    g.node_mut(mask).name = Some("mask".into());
    let gating = g.narrow_(out, 0, n, 1);
    g.node_mut(gating).name = Some("gating".into());
    g.set_outputs(vec![mask, gating]);
    g
}

/// Export subject lifting from `path` (an `.espresso.net`, default
/// [`SYSTEM_MODEL`]) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let net_path = if path.as_os_str().is_empty() || path.is_dir() {
        PathBuf::from(SYSTEM_MODEL)
    } else {
        path.to_path_buf()
    };
    let o = LiftOptions::default();
    ensure!(
        matches!(o.norm, InputNorm::Signed) && !o.bgr && o.upsample == Upsample::Bilinear,
        "the exporter writes the default settings only"
    );
    let net =
        NetFile::open(&net_path).with_context(|| format!("reading {}", net_path.display()))?;
    let side = net.shapes.get("input_image").map(|s| s.h).unwrap_or(512);
    let built = crate::graph::build(
        &net,
        "input_image",
        side,
        &["saliency", "gating_confidence"],
        o.upsample,
    )?;
    let (mut network, params) = rlx_core::flow_util::graph_from_built(built)?;
    network.name = "network".into();
    let mut params: HashMap<String, Vec<f32>> = params;
    params.insert(CLIP_IOTA.into(), clip_iota());
    params.insert("feather".into(), blur_matrix(side, FEATHER));
    let stages = vec![
        // `v / 127.5 − 1` is `(v / 255 − 0.5) / 0.5`.
        Stage::new(image_resize_normalize(
            side,
            ImageFilter::Triangle,
            [0.5; 3],
            [0.5; 3],
        )?)
        .input("image", "image")
        .input("dims", "image#shape")
        .outputs(&["pixels"])
        .on("cpu"),
        Stage::new(network)
            .input("input_image", "pixels")
            .outputs(&["output"]),
        Stage::new(heads(side)).outputs(&["mask", "gating"]),
        Stage::new(image_resize_mask(side, ImageFilter::Triangle)?)
            .input("dims", "image#shape")
            .outputs(&["alpha"])
            .on("cpu"),
    ];
    let mut p = Pipeline::new("subject-lift", "image-segmentation", stages);
    p.inputs = vec![image_contract()];
    p.outputs = vec!["alpha".into(), "gating".into()];
    p.dims = vec![("height".into(), HEIGHT), ("width".into(), WIDTH)];
    p.metadata = vec![
        ("image_size".into(), serde_json::json!(side)),
        ("feather".into(), serde_json::json!(FEATHER)),
    ];
    p.exported_by = "rlx-subject-lift".into();
    p.write(params, None, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family subject-lift`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "subject-lift",
        "image-segmentation",
        export_rlxp,
    ));
}
