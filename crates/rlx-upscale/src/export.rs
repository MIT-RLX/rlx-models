// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a super-resolution checkpoint as an `.rlxp` package rlx runs
//! alone: the network as one graph symbolic in its tile side (a
//! convolutional net) or one graph per tile side (a windowed transformer,
//! whose window counts are no single symbol), and a program answering
//! `run` with what [`crate::Upscaler`] does around it — the tile plan
//! ([`crate::tile`]), the stitch, the 8-bit conversion and the alpha
//! channel. `Model.run({image: u8 [h, w, 3 or 4]})` returns `image`, the
//! upscaled `u8 [h·s, w·s, …]`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::export::{Package, program};
use serde_json::json;

use crate::config::{Arch, ModelConfig};
use crate::weights::{self, Checkpoint};

/// The tile symbol.
const TILE: u32 = 0;

fn load(path: &Path) -> Result<(Checkpoint, ModelConfig)> {
    let mut ck = Checkpoint::open(path).with_context(|| format!("loading {}", path.display()))?;
    let cfg =
        crate::detect::detect(&ck).with_context(|| format!("identifying {}", path.display()))?;
    if cfg.arch == Arch::Span && !ck.conv3xc_prefixes().is_empty() {
        weights::fuse_all_conv3xc(&mut ck)?;
    }
    Ok((ck, cfg))
}

fn build(path: &Path, tile: usize) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
    let (ck, cfg) = load(path)?;
    let built = crate::graph::build(&cfg, ck.into_weight_map(), tile, tile)?;
    ensure!(
        built.unused.is_empty(),
        "{} left {} checkpoint tensors unread",
        cfg.summary(),
        built.unused.len()
    );
    Ok((built.graph, built.params))
}

/// Export the checkpoint at `path` (`.pth`, `.safetensors`, … as
/// [`crate::Upscaler::open`] takes it) as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let (_, cfg) = load(path)?;
    let m = cfg.size_multiple().max(1);
    let overlap = if cfg.arch.is_transformer() { m } else { 16 };
    let tile = cfg.default_tile();
    let round_up = |v: usize| v.div_ceil(m) * m;
    let floor = round_up(2 * overlap + m);
    let want = round_up(tile).max(floor);

    let mut pkg = Package::new(
        cfg.arch.name(),
        &opts.version,
        cfg.arch.name(),
        "image-to-image",
        "rlx-upscale",
    );
    // One graph symbolic in the tile, from two sides no other dim has,
    // checked at a third; else one graph per side the plan can pick.
    let (a, b, c) = (round_up(101), round_up(149), round_up(67));
    let (ga, params) = build(path, a)?;
    let (gb, _) = build(path, b)?;
    let symbolic = rlx_ir::dynamic::generalize(&ga, &gb, &[(TILE, a, b)]).and_then(|g| {
        let (gc, _) = build(path, c).map_err(|e| e.to_string())?;
        let mut bind = rlx_ir::DimBinding::new();
        bind.set(TILE, c);
        if format!("{}", rlx_ir::dynamic::specialize(&g, &bind)) != format!("{gc}") {
            return Err(
                "the network generalized from two tiles is not the one built at a third".into(),
            );
        }
        Ok(g)
    });
    drop((ga, gb));
    let (config_functions, params, symbolic_failure) = match symbolic {
        Ok(mut g) => {
            g.name = "upscale".into();
            pkg.function(g);
            pkg.dims = vec![("tile".into(), TILE)];
            (json!({"function": "upscale"}), params, None)
        }
        Err(e) => {
            let mut builds = Vec::new();
            let mut t = floor;
            while t <= want {
                let (mut g, p) = build(path, t)?;
                g.name = format!("upscale_{t}");
                builds.push((t, g, p));
                t += m;
            }
            let sides: Vec<usize> = builds.iter().map(|(t, _, _)| *t).collect();
            let (graphs, params) = rlx_core::export::graph::per_size(builds);
            for g in graphs {
                pkg.function(g);
            }
            let functions: Vec<_> = sides
                .iter()
                .map(|t| json!({"tile": t, "function": format!("upscale_{t}")}))
                .collect();
            (json!({"functions": functions}), params, Some(e))
        }
    };
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
    let mut config = json!({
        "scale": cfg.scale,
        "in_channels": cfg.in_ch,
        "out_channels": cfg.out_ch,
        "size_multiple": m,
        "overlap": overlap,
        "tile": tile,
    });
    for (k, v) in config_functions.as_object().into_iter().flatten() {
        config[k] = v.clone();
    }
    program::entry(
        &mut pkg,
        "upscale",
        module,
        &["run"],
        "image-to-image",
        config,
    );
    pkg.metadata("architecture", json!(cfg.summary()));
    pkg.metadata("scale", json!(cfg.scale));
    let report = pkg.write(out, opts)?;
    Ok(ExportReport {
        symbolic_failure,
        ..report
    })
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family upscale`): a super-resolution checkpoint names
/// no `model_type`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("upscale", "image-to-image", export_rlxp));
}
