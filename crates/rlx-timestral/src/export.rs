// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a Timestral checkpoint as an `.rlxp` package rlx runs alone: the
//! network as graphs symbolic in the packed rows and the context's patches —
//! one per decoder length a forecast block pads to, and the covariate
//! window's (as many decoder tokens as patches) — and a program answering
//! `run` with what [`crate::TimestralForecaster::forecast`] does around them,
//! built from this crate's own `model.rs` and `forecaster.rs`.
//! `Model.run({context: f32 [B, V, T]}, {horizon: H})` returns `quantiles`,
//! f32 `[B, V, H, Q]`, and `levels`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::export::{Package, program};
use rlx_ir::Graph;
use serde_json::json;

use crate::config::TimestralConfig;
use crate::graph::{GraphDims, build_graph, graph_params};
use crate::weights::TimestralWeights;

/// Packed rows (series × variates).
const ROWS: u32 = 0;
/// Context patches.
const PATCHES: u32 = 1;
/// Rows × patches, and rows × the sequence (`patches + decoder`): products
/// a fit of two symbols does not give with an offset, so sized as their own.
const ROW_PATCHES: u32 = 2;
const ROW_SEQ: u32 = 3;

fn lower(cfg: &TimestralConfig, dims: GraphDims) -> Result<Graph> {
    let hir = build_graph(cfg, dims);
    let (g, _) = rlx_core::flow_util::graph_from_hir(hir, HashMap::new())?;
    Ok(g)
}

/// The graph for `n_dec` decoder tokens (`None`: as many as the patches),
/// symbolic in [`ROWS`] and [`PATCHES`]: built at sizes no other dim has,
/// generalized, and checked against a build at another.
fn symbolic(cfg: &TimestralConfig, n_dec: Option<usize>) -> Result<Graph> {
    let sizes = [(5usize, 37usize), (7, 41), (11, 53), (13, 47), (17, 59)];
    let dims = |r: usize, l: usize| GraphDims {
        r,
        l,
        n_dec: n_dec.unwrap_or(l),
        hidden: false,
        train: false,
    };
    let built: Vec<Graph> = sizes
        .iter()
        .map(|&(r, l)| lower(cfg, dims(r, l)))
        .collect::<Result<_>>()?;
    let refs: Vec<&Graph> = built.iter().collect();
    let mut syms = vec![
        (ROWS, sizes.iter().map(|s| s.0).collect::<Vec<_>>()),
        (PATCHES, sizes.iter().map(|s| s.1).collect()),
        (ROW_PATCHES, sizes.iter().map(|s| s.0 * s.1).collect()),
    ];
    if let Some(n) = n_dec {
        syms.push((ROW_SEQ, sizes.iter().map(|s| s.0 * (s.1 + n)).collect()));
    }
    let g = rlx_ir::dynamic::generalize_many(&refs, &syms, Some(64))
        .map_err(|e| anyhow!("the network is not symbolic in its rows and patches: {e}"))?;
    let (r, l) = (19, 43);
    let check = lower(cfg, dims(r, l))?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(ROWS, r);
    bind.set(PATCHES, l);
    bind.set(ROW_PATCHES, r * l);
    if let Some(n) = n_dec {
        bind.set(ROW_SEQ, r * (l + n));
    }
    let (got, want) = (
        format!("{}", rlx_ir::dynamic::specialize(&g, &bind)),
        format!("{check}"),
    );
    if got != want {
        let (a, b) = got
            .lines()
            .zip(want.lines())
            .find(|(a, b)| a != b)
            .unwrap_or_default();
        bail!(
            "the network generalized over five sizes is not the one built at another: `{a}` \
             where it builds `{b}`"
        );
    }
    Ok(g)
}

/// Export the checkpoint in `dir` (`config.json`, `model.safetensors`) as
/// the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = TimestralConfig::from_file(&dir.join("config.json"))?;
    let weights =
        TimestralWeights::load(dir, &cfg).with_context(|| format!("loading {}", dir.display()))?;
    export_model(&cfg, &weights, out, opts)
}

/// [`export_rlxp`] from a loaded model (synthetic weights in tests).
pub fn export_model(
    cfg: &TimestralConfig,
    weights: &TimestralWeights,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let mut pkg = Package::new(
        "timestral",
        &opts.version,
        "timestral",
        "time-series-forecasting",
        "rlx-timestral",
    );
    // A block decodes up to `block_size` patches, padded to a power of two
    // (`TimestralModel::predict_context_only`).
    let mut blocks = Vec::new();
    let mut n = 1;
    while n < 2 * cfg.block_size.max(1) {
        let mut g = symbolic(cfg, Some(n)).with_context(|| format!("{n} decoder tokens"))?;
        g.name = format!("block_{n}");
        blocks.push(json!({"decoder": n, "function": g.name}));
        pkg.function(g);
        n *= 2;
    }
    let mut g = symbolic(cfg, None).context("the covariate window")?;
    g.name = "window".into();
    pkg.function(g);
    pkg.dims = vec![
        ("rows".into(), ROWS),
        ("patches".into(), PATCHES),
        ("row_patches".into(), ROW_PATCHES),
        ("row_seq".into(), ROW_SEQ),
    ];

    let params = graph_params(cfg, weights);
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), d, &params[n]);
    }

    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    let config = json!({
        "model": serde_json::to_value(cfg)?,
        "forecast_query": weights.forecast_query.to_vec(),
        "blocks": blocks,
        "window": "window",
    });
    program::entry(
        &mut pkg,
        "forecast",
        module,
        &["run"],
        "time-series-forecasting",
        config,
    );
    pkg.metadata("quantile_levels", json!(cfg.quantile_knots));
    pkg.metadata("patch_size", json!(cfg.patch_size));
    pkg.metadata("context_length", json!(cfg.context_length));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "timestral",
        "time-series-forecasting",
        export_rlxp,
    ));
}
