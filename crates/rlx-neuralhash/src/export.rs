// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export NeuralHash as an `.rlxp` package rlx runs alone: a `pipeline`
//! from an RGB image of any size to its 96-bit hash, every step a graph —
//! Pillow's antialiased bicubic resize to 360 × 360 with `x / 255 · 2 − 1`
//! ([`rlx_core::export::vision::pillow_preprocess`], the runner's own
//! [`crate::preprocess::preprocessor`]), the descriptor network as the
//! runner builds it from the network file, the seed projection and the
//! `score ≥ 0` step. `Model.run({image})` returns `embedding` (128 floats),
//! `scores` (96) and `bits` (96, each 0 or 1, the hash's bits MSB first).
//!
//! The network's weights are the vendor's and are read from the copy macOS ships;
//! the package holds them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rlx_core::export::graph::{Pipeline, Stage};
use rlx_core::export::vision::{
    CLIP_IOTA, HEIGHT, WIDTH, clip_iota, image_contract, pillow_preprocess,
};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};

use crate::seed::{EMBED_DIM, SeedMatrix};
use crate::{HASH_BITS, NeuralHashSpec, netfile::NetFile};

/// The seed file's name beside the net (as macOS ships it).
pub const SEED_FILE: &str = "neuralhash_128x96_seed1.dat";

/// The net and the seed for `path`: an `.espresso.net` with the seed beside
/// it, or a directory holding both.
fn locate(path: &Path) -> Result<(PathBuf, PathBuf)> {
    let net = if path.is_dir() {
        let mut nets: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".espresso.net"))
            .filter(|p| p.to_string_lossy().contains("NeuralHash"))
            .collect();
        nets.sort();
        nets.pop()
            .with_context(|| format!("{}: no NeuralHash .espresso.net", path.display()))?
    } else {
        path.to_path_buf()
    };
    let dir = net.parent().unwrap_or(Path::new("."));
    let seed = dir.join(SEED_FILE);
    ensure!(
        seed.is_file(),
        "{}: no {SEED_FILE} beside the net",
        dir.display()
    );
    Ok((net, seed))
}

/// `embedding` (the descriptor network's output) → `scores` `[96]` (the
/// seed rows' dot products) and `bits` `[96]` (`score ≥ 0` as 0 / 1).
fn projection(embedding: &Shape) -> Graph {
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("hash");
    let e = g.input("embedding", embedding.clone());
    let e = g.reshape_(e, vec![EMBED_DIM as i64, 1]);
    let seed = g.param("seed", f32s(&[HASH_BITS, EMBED_DIM]));
    let scores = g.mm(seed, e);
    let scores = g.reshape_(scores, vec![HASH_BITS as i64]);
    g.node_mut(scores).name = Some("scores".into());
    let zero = g.constant(0.0, DType::F32);
    let ge = g.add_node(
        Op::Compare(rlx_ir::op::CmpOp::Ge),
        vec![scores, zero],
        Shape::new(&[HASH_BITS], DType::Bool),
    );
    let bits = g.add_node(Op::Cast { to: DType::F32 }, vec![ge], f32s(&[HASH_BITS]));
    g.node_mut(bits).name = Some("bits".into());
    g.set_outputs(vec![scores, bits]);
    g
}

/// Export NeuralHash from `path` (the vendor's `NeuralHashv3b*.espresso.net`
/// with its `.shape` / `.weights` siblings and the seed, or their
/// directory) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let (net_path, seed_path) = locate(path)?;
    let net = NetFile::open(&net_path)?;
    let spec = NeuralHashSpec::from_netfile(&net)
        .context("deriving the architecture from the network's layer list")?;
    spec.validate_neuralhash_io()?;
    let mut weights = crate::weights::from_netfile(&net).context("extracting the weights")?;
    let (network, mut params) = crate::flow::build_graph(&spec, &mut weights)?;
    let embedding = network.node(network.outputs[0]).shape.clone();
    let seed = SeedMatrix::open(&seed_path)?;
    params.insert("seed".into(), seed.as_slice().to_vec());
    params.insert(CLIP_IOTA.into(), clip_iota());

    let mut network = network;
    network.name = "descriptor".into();
    let stages = vec![
        Stage::new(pillow_preprocess(&crate::preprocess::preprocessor())?)
            .input("image", "image")
            .input("dims", "image#shape")
            .outputs(&["pixels"])
            .on("cpu"),
        Stage::new(network)
            .input(&spec.input, "pixels")
            .outputs(&["embedding"]),
        Stage::new(projection(&embedding)).outputs(&["scores", "bits"]),
    ];
    let mut p = Pipeline::new("neuralhash", "image-feature-extraction", stages);
    p.inputs = vec![image_contract()];
    p.outputs = vec!["embedding".into(), "scores".into(), "bits".into()];
    p.dims = vec![("height".into(), HEIGHT), ("width".into(), WIDTH)];
    p.metadata = vec![
        ("image_size".into(), serde_json::json!(crate::INPUT_SIZE)),
        ("hash_bits".into(), serde_json::json!(HASH_BITS)),
    ];
    p.exported_by = "rlx-neuralhash".into();
    let params: HashMap<String, Vec<f32>> = params;
    p.write(params, None, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family neuralhash`): an network file names no
/// `model_type`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "neuralhash",
        "image-feature-extraction",
        export_rlxp,
    ));
}
