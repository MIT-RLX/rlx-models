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

//! CLI entry points for `rlx-hoct` / `rlx-run hoct`.
//!
//! Subcommands: `track` (full pipeline) and `predict` (random-batch forward).

use crate::config::{GraphConfig, IlpWeights};
use crate::device::HoctDeviceRunner;
use crate::io::OutputFormat;
use crate::runner::HoctRunner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_device, req};
use std::path::PathBuf;

/// Dispatch `track` / `predict` from argv after the binary name.
pub fn run(args: &[String]) -> Result<()> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_help();
        return Ok(());
    }
    match args[0].as_str() {
        "track" => run_track(&args[1..]),
        "predict" => run_predict(&args[1..]),
        other => bail!("unknown subcommand `{other}` (try track|predict)"),
    }
}

fn print_help() {
    eprintln!(
        "rlx-hoct — HOCT cell tracking\n\
         \n\
         track   -m|--weights PATH --labels PATH [-o|--out DIR] [-f ctc|geff]\n\
                 [--window N] [--stride N] [--max-distance F] [--neighbors K]\n\
                 [--max-dt T] [-d|--device NAME]\n\
         predict -m|--weights PATH [--batch B] [--nodes N] [--edges E]\n\
         \n\
         ILP: --appearance F --disappearance F --division F --node F\n\
              --edge-bias F --delta-t-weight F"
    );
}

fn parse_graph_and_ilp(args: &[String], i: &mut usize) -> Result<(GraphConfig, IlpWeights)> {
    let mut graph = GraphConfig::default();
    let mut ilp = IlpWeights::default();
    while *i < args.len() {
        match args[*i].as_str() {
            "--distance-threshold" | "--max-distance" => {
                graph.distance_threshold = req(args, i)?.parse().context("float")?;
            }
            "--n-neighbors" | "--neighbors" => {
                graph.n_neighbors = req(args, i)?.parse().context("usize")?;
            }
            "--max-delta-t" | "--max-dt" => {
                graph.max_delta_t = req(args, i)?.parse().context("i32")?;
            }
            "--appearance" => ilp.appearance = req(args, i)?.parse().context("float")?,
            "--disappearance" => ilp.disappearance = req(args, i)?.parse().context("float")?,
            "--division" => ilp.division = req(args, i)?.parse().context("float")?,
            "--node" => ilp.node = req(args, i)?.parse().context("float")?,
            "--edge-bias" => ilp.edge_bias = req(args, i)?.parse().context("float")?,
            "--delta-t-weight" => ilp.delta_t_weight = req(args, i)?.parse().context("float")?,
            other if other.starts_with('-') => break,
            _ => break,
        }
    }
    Ok((graph, ilp))
}

fn run_track(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut labels: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut format = OutputFormat::Ctc;
    let mut window = 5usize;
    let mut stride = 1usize;
    let mut device = String::from("cpu");
    let mut i = 0;
    let (graph_cfg, ilp_weights) = parse_graph_and_ilp(args, &mut i)?;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "-m" => weights = Some(req(args, &mut i)?.into()),
            "--labels" => labels = Some(req(args, &mut i)?.into()),
            "--out" | "-o" => out = Some(req(args, &mut i)?.into()),
            "--format" | "-f" => format = OutputFormat::parse(&req(args, &mut i)?)?,
            "--window" => window = req(args, &mut i)?.parse().context("usize")?,
            "--stride" => stride = req(args, &mut i)?.parse().context("usize")?,
            "--device" | "-d" => device = req(args, &mut i)?,
            other => bail!("unknown flag: {other}"),
        }
    }
    let device = parse_device(&device).unwrap_or(rlx_runtime::Device::Cpu);
    let weights = weights.ok_or_else(|| anyhow!("--weights / -m required"))?;
    let labels = labels.ok_or_else(|| anyhow!("--labels required"))?;
    if device != rlx_runtime::Device::Cpu {
        let _ = HoctDeviceRunner::from_weights(&weights, device)
            .context("compile HOCT score head on device")?;
        eprintln!("[rlx-hoct] score head compiled on {device:?}");
    }
    let runner = HoctRunner::builder()
        .weights(weights)
        .graph_cfg(graph_cfg)
        .ilp_weights(ilp_weights)
        .window_size(window)
        .stride(stride)
        .build()?;
    let sol = runner.track_path(&labels, out.as_deref(), format)?;
    eprintln!("[rlx-hoct] track done — {} links", sol.links.len());
    Ok(())
}

fn run_predict(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut batch = 1usize;
    let mut nodes = 8usize;
    let mut edges = 12usize;
    let mut seed = 42u64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "-m" => weights = Some(req(args, &mut i)?.into()),
            "--batch" => batch = req(args, &mut i)?.parse().context("usize")?,
            "--nodes" => nodes = req(args, &mut i)?.parse().context("usize")?,
            "--edges" => edges = req(args, &mut i)?.parse().context("usize")?,
            "--seed" => seed = req(args, &mut i)?.parse().context("u64")?,
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights / -m required"))?;
    let runner = HoctRunner::builder().weights(weights).build()?;
    let mut rng = seed;
    let mut rnd = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f32 / u32::MAX as f32
    };
    let d = runner.config().feature_dim;
    let mut node_features = ndarray::Array3::<f32>::zeros((batch, nodes, d));
    let mut node_pos = ndarray::Array3::<f32>::zeros((batch, nodes, 3));
    let mut edge_pos = ndarray::Array3::<f32>::zeros((batch, edges, 3));
    let mut edge_indices = ndarray::Array3::<i64>::zeros((batch, edges, 2));
    for b in 0..batch {
        for n in 0..nodes {
            for k in 0..d {
                node_features[[b, n, k]] = rnd() * 2.0 - 1.0;
            }
            for k in 0..3 {
                node_pos[[b, n, k]] = rnd() * 100.0;
            }
        }
        for e in 0..edges {
            edge_indices[[b, e, 0]] = (rnd() * nodes as f32) as i64 % nodes as i64;
            edge_indices[[b, e, 1]] = (rnd() * nodes as f32) as i64 % nodes as i64;
            for k in 0..3 {
                edge_pos[[b, e, k]] = rnd() * 100.0;
            }
        }
    }
    let node_mask = ndarray::Array2::<bool>::from_elem((batch, nodes), true);
    let edge_mask = ndarray::Array2::<bool>::from_elem((batch, edges), true);
    let out = runner.model.forward(
        &node_features.view(),
        &node_pos.view(),
        &edge_pos.view(),
        &edge_indices,
        &node_mask,
        &edge_mask,
    );
    eprintln!(
        "[rlx-hoct] predict logits shape=({},{},{}), orphan_sum={}",
        out.edge_logits.len_of(ndarray::Axis(0)),
        out.edge_logits.len_of(ndarray::Axis(1)),
        out.edge_logits.len_of(ndarray::Axis(2)),
        out.orphan_logits.sum()
    );
    Ok(())
}
