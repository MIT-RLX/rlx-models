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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_onnx_import::{
    BundleManifest, BundleNode, ImportOptions, ImportReport, TypedParams, build_hir_from_parts,
    load_bundle, prepare_onnx_file, rewrite,
};

type GraphEmitParts = (
    Vec<BundleNode>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<usize>>,
);

fn graph_for_emit(
    manifest: &BundleManifest,
    import_opts: &ImportOptions,
    typed_params: &TypedParams,
    nodes: Vec<BundleNode>,
    mut params: HashMap<String, Vec<f32>>,
    mut init_shapes: HashMap<String, Vec<usize>>,
) -> Result<GraphEmitParts> {
    let quant_weight_keys: HashSet<String> = typed_params.keys().cloned().collect();
    let rewritten = rewrite::rewrite_graph(
        nodes,
        &params,
        &init_shapes,
        manifest,
        import_opts,
        &quant_weight_keys,
    );
    params.extend(rewritten.extra_params);
    init_shapes.extend(rewritten.extra_shapes);
    Ok((rewritten.nodes, params, init_shapes))
}

/// Default `rlx` repo root relative to `rlx-onnx-decompose` (`rlx-models/crates/rlx-onnx-decompose` → `../../../rlx`).
pub fn default_rlx_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rlx")
}

pub fn resolve_rlx_root(opts: &DecomposeOptions) -> PathBuf {
    opts.rlx_root.clone().unwrap_or_else(default_rlx_root)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsFormat {
    Safetensors,
    Gguf,
}

#[derive(Debug)]
pub struct DecomposePlan {
    pub source_onnx: PathBuf,
    pub manifest: BundleManifest,
    pub nodes: Vec<BundleNode>,
    pub params: HashMap<String, Vec<f32>>,
    pub i64_params: HashMap<String, Vec<i64>>,
    pub init_shapes: HashMap<String, Vec<usize>>,
    pub import_report: ImportReport,
    pub crate_name: String,
    pub module_name: String,
}

#[derive(Debug, Clone)]
pub struct DecomposeOptions {
    pub sequence_length: usize,
    pub max_waveform_samples: usize,
    pub weights_format: WeightsFormat,
    pub crate_name: Option<String>,
    /// Path to the `rlx` repo root (for generated `Cargo.toml` path deps).
    pub rlx_root: Option<PathBuf>,
    /// Path to the `rlx-models` repo root.
    pub rlx_models_root: Option<PathBuf>,
}

impl Default for DecomposeOptions {
    fn default() -> Self {
        Self {
            sequence_length: 128,
            max_waveform_samples: 24_000,
            weights_format: WeightsFormat::Safetensors,
            crate_name: None,
            rlx_root: None,
            rlx_models_root: None,
        }
    }
}

impl DecomposePlan {
    pub fn from_onnx(path: &Path, opts: &DecomposeOptions) -> Result<Self> {
        let import_opts = ImportOptions {
            sequence_length: opts.sequence_length,
            max_waveform_samples: opts.max_waveform_samples,
            ..ImportOptions::default()
        };
        let (manifest, nodes, params, i64_params, init_shapes) =
            prepare_onnx_file(path).context("parse ONNX")?;
        let typed_params = TypedParams::new();
        let (nodes, params, init_shapes) = graph_for_emit(
            &manifest,
            &import_opts,
            &typed_params,
            nodes,
            params,
            init_shapes,
        )
        .context("rewrite graph for emit")?;
        let (_, _, _, import_report) = build_hir_from_parts(
            &manifest,
            nodes.clone(),
            params.clone(),
            typed_params,
            i64_params.clone(),
            &init_shapes,
            import_opts,
        )
        .context("dry-run lower for coverage")?;

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("onnx_model");
        let crate_name = opts
            .crate_name
            .clone()
            .unwrap_or_else(|| sanitize_crate_name(stem));
        let module_name = crate_name.replace('-', "_");

        Ok(Self {
            source_onnx: path.to_path_buf(),
            manifest,
            nodes,
            params,
            i64_params,
            init_shapes,
            import_report,
            crate_name,
            module_name,
        })
    }

    pub fn from_bundle(dir: &Path, opts: &DecomposeOptions) -> Result<Self> {
        let bundle = load_bundle(dir).context("load RLX bundle")?;
        let import_opts = ImportOptions {
            sequence_length: opts.sequence_length,
            max_waveform_samples: opts.max_waveform_samples,
            ..ImportOptions::default()
        };
        let (params, init_shapes) =
            rlx_onnx_import::tensor_data::load_f32_params(&bundle.weight_bytes)?;
        let i64_params =
            rlx_onnx_import::tensor_data::load_i64_params(&bundle.weight_bytes).unwrap_or_default();
        let typed_params = if import_opts.use_quantized_kernels {
            rlx_onnx_import::tensor_data::load_typed_quant_params(&bundle.weight_bytes)?.0
        } else {
            TypedParams::new()
        };
        let (nodes, params, init_shapes) = graph_for_emit(
            &bundle.manifest,
            &import_opts,
            &typed_params,
            bundle.nodes.clone(),
            params,
            init_shapes,
        )
        .context("rewrite graph for emit")?;
        let (_, _, _, import_report) = build_hir_from_parts(
            &bundle.manifest,
            nodes.clone(),
            params.clone(),
            typed_params,
            i64_params.clone(),
            &init_shapes,
            import_opts,
        )
        .context("dry-run lower for coverage")?;

        let stem = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("bundle_model");
        let crate_name = opts
            .crate_name
            .clone()
            .unwrap_or_else(|| sanitize_crate_name(stem));
        let module_name = crate_name.replace('-', "_");

        Ok(Self {
            source_onnx: PathBuf::from(&bundle.manifest.source_onnx),
            manifest: bundle.manifest,
            nodes,
            params,
            i64_params,
            init_shapes,
            import_report,
            crate_name,
            module_name,
        })
    }
}

pub fn decompose(
    onnx_path: &Path,
    out_dir: &Path,
    opts: &DecomposeOptions,
) -> Result<DecomposePlan> {
    let plan = DecomposePlan::from_onnx(onnx_path, opts)?;
    export_plan_artifacts(&plan, None, out_dir, opts)?;
    Ok(plan)
}

pub fn decompose_bundle(
    bundle_dir: &Path,
    out_dir: &Path,
    opts: &DecomposeOptions,
) -> Result<DecomposePlan> {
    let plan = DecomposePlan::from_bundle(bundle_dir, opts)?;
    export_plan_artifacts(&plan, Some(bundle_dir), out_dir, opts)?;
    Ok(plan)
}

fn export_plan_artifacts(
    plan: &DecomposePlan,
    bundle_dir: Option<&Path>,
    out_dir: &Path,
    opts: &DecomposeOptions,
) -> Result<()> {
    let weights_dir = out_dir.join("weights");
    if let Some(bundle_dir) = bundle_dir {
        std::fs::create_dir_all(&weights_dir)?;
        let src = bundle_dir.join("weights.safetensors");
        let dst = weights_dir.join("model.safetensors");
        std::fs::copy(&src, &dst)
            .with_context(|| format!("copy {} → {}", src.display(), dst.display()))?;
    } else {
        let _ = crate::weights::export_weights(
            &plan.params,
            &plan.i64_params,
            &plan.init_shapes,
            &weights_dir,
            opts.weights_format,
        )?;
    }
    crate::emit::write_generated_crate(out_dir, plan, opts.weights_format, opts)?;
    Ok(())
}

pub fn sanitize_crate_name(stem: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for c in stem.chars() {
        let ok = c.is_ascii_alphanumeric() || c == '_';
        if ok && (c.is_ascii_alphabetic() || !out.is_empty()) {
            out.push(if c == '.' {
                '_'
            } else {
                c.to_ascii_lowercase()
            });
            prev_underscore = c == '_';
        } else if !prev_underscore && !out.is_empty() {
            out.push('_');
            prev_underscore = true;
        }
    }
    if out.is_empty() || out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out = format!("onnx_{out}");
    }
    out
}
