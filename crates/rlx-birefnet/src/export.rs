// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a BiRefNet checkpoint as an `.rlxp` package rlx runs alone: a
//! `pipeline` from an RGB image of any size to its foreground alpha, every
//! step a graph — `imageops::resize` to the network's side (Triangle, as
//! [`crate::BiRefNet::preprocess`]) and ImageNet normalization
//! ([`rlx_core::export::vision::image_resize_normalize`]), the network, the
//! sigmoid, and the mask resized back to the image
//! ([`rlx_core::export::vision::image_resize_mask`]).
//! `Model.run({image: u8 [h, w, 3]})` returns `alpha`, `f32 [h, w]` in 0–1.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use rlx_core::export::graph::{Pipeline, Stage};
use rlx_core::export::vision::{
    CLIP_IOTA, HEIGHT, ImageFilter, WIDTH, clip_iota, image_contract, image_resize_mask,
    image_resize_normalize,
};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Shape};

use crate::weights::Weights;

/// The side the checkpoints are trained at.
pub const SIDE: usize = 1024;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// `logits` `[side²]` → `mask`: their sigmoid.
fn sigmoid(side: usize) -> Graph {
    let mut g = Graph::new("sigmoid");
    let x = g.input("logits", Shape::new(&[side * side], DType::F32));
    let s = g.sigmoid(x);
    g.node_mut(s).name = Some("mask".into());
    g.set_outputs(vec![s]);
    g
}

/// Export the BiRefNet checkpoint in `dir` (`model.safetensors`) as the
/// package `out`, the network at [`SIDE`].
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_at(dir, SIDE, out, opts)
}

/// [`export_rlxp`] with the network at `side` (a multiple of 32).
pub fn export_at(
    dir: &Path,
    side: usize,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let w = Arc::new(Weights::load(dir)?);
    let built = crate::graph::build(w, side, &["logits"])?;
    let (mut network, mut params) = rlx_core::flow_util::graph_from_built(built)?;
    network.name = "network".into();
    params.insert(CLIP_IOTA.into(), clip_iota());
    let stages = vec![
        Stage::new(image_resize_normalize(
            side,
            ImageFilter::Triangle,
            MEAN,
            STD,
        )?)
        .input("image", "image")
        .input("dims", "image#shape")
        .outputs(&["pixels"])
        .on("cpu"),
        Stage::new(network)
            .input("image", "pixels")
            .outputs(&["logits"]),
        Stage::new(sigmoid(side)).outputs(&["mask"]),
        Stage::new(image_resize_mask(side, ImageFilter::Triangle)?)
            .input("dims", "image#shape")
            .outputs(&["alpha"])
            .on("cpu"),
    ];
    let mut p = Pipeline::new("birefnet", "image-segmentation", stages);
    p.inputs = vec![image_contract()];
    p.outputs = vec!["alpha".into()];
    p.dims = vec![("height".into(), HEIGHT), ("width".into(), WIDTH)];
    p.metadata = vec![("image_size".into(), serde_json::json!(side))];
    p.exported_by = "rlx-birefnet".into();
    let ck = rlx_core::export::Checkpoint::open(dir).ok();
    p.write(params, ck.as_ref(), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family birefnet`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("birefnet", "image-segmentation", export_rlxp));
}
