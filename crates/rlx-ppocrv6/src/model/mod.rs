//! Native HIR graph builders for PP-OCRv6 (offline ONNX → Rust).
//!
//! [`build_detection`] / [`build_recognition`] load safetensors and call the
//! tier-specific builders under [`crate::native`]. Regenerated via
//! `scripts/ppocrv6_emit_native.py`. No runtime ONNX import.

use crate::config::Tier;
use crate::native::{small_det, small_rec, tiny_det, tiny_rec};
use anyhow::{Context, Result};
use rlx_core::flow_util::graph_from_hir;
use rlx_ir::Graph;
use std::collections::HashMap;
use std::path::Path;

pub struct NativeGraph {
    pub graph: Graph,
    pub params: HashMap<String, Vec<f32>>,
    pub input_name: String,
}

pub fn build_detection(tier: Tier, weights_dir: &Path, height: usize, width: usize) -> Result<NativeGraph> {
    match tier {
        Tier::Tiny => {
            let w = tiny_det::load_weights(weights_dir)
                .with_context(|| format!("load tiny det weights from {}", weights_dir.display()))?;
            let opts = tiny_det::GraphOptions {
                height,
                width,
                sequence_length: height.max(width),
                ..Default::default()
            };
            let (hir, params) = tiny_det::build_hir(&w, &opts)?;
            let (graph, params) = graph_from_hir(hir, params)?;
            Ok(NativeGraph {
                graph,
                params,
                input_name: "x".into(),
            })
        }
        Tier::Small => {
            let w = small_det::load_weights(weights_dir)
                .with_context(|| format!("load small det weights from {}", weights_dir.display()))?;
            let opts = small_det::GraphOptions {
                height,
                width,
                sequence_length: height.max(width),
                ..Default::default()
            };
            let (hir, params) = small_det::build_hir(&w, &opts)?;
            let (graph, params) = graph_from_hir(hir, params)?;
            Ok(NativeGraph {
                graph,
                params,
                input_name: "x".into(),
            })
        }
    }
}

pub fn build_recognition(
    tier: Tier,
    weights_dir: &Path,
    height: usize,
    width: usize,
) -> Result<NativeGraph> {
    match tier {
        Tier::Tiny => {
            let w = tiny_rec::load_weights(weights_dir)
                .with_context(|| format!("load tiny rec weights from {}", weights_dir.display()))?;
            let opts = tiny_rec::GraphOptions {
                height,
                width,
                sequence_length: width,
                ..Default::default()
            };
            let (hir, params) = tiny_rec::build_hir(&w, &opts)?;
            let (graph, params) = graph_from_hir(hir, params)?;
            Ok(NativeGraph {
                graph,
                params,
                input_name: "x".into(),
            })
        }
        Tier::Small => {
            let w = small_rec::load_weights(weights_dir)
                .with_context(|| format!("load small rec weights from {}", weights_dir.display()))?;
            let opts = small_rec::GraphOptions {
                height,
                width,
                sequence_length: width,
                ..Default::default()
            };
            let (hir, params) = small_rec::build_hir(&w, &opts)?;
            let (graph, params) = graph_from_hir(hir, params)?;
            Ok(NativeGraph {
                graph,
                params,
                input_name: "x".into(),
            })
        }
    }
}
