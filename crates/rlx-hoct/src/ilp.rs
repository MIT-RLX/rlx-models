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

//! Tracking ILP (HOCT Appendix B) via `good_lp` + HiGHS.
//!
//! Variables `x_e, y_i, a_i, b_i, δ_i` with flow constraints. Two-pass
//! tracklet mode bans `|Δt|>1` on pass 1 then solves the full graph.

use crate::config::IlpWeights;
use crate::softmax::{EdgeScore, NodeOrphan};
use anyhow::{Context, Result};
use good_lp::{Expression, Solution, SolverModel, constraint, default_solver, variable, variables};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TrackletLink {
    pub src: usize,
    pub dst: usize,
    pub delta_t: i32,
    pub edge_id: usize,
}

/// Selected nodes/edges after ILP (plus appearance / division markers).
#[derive(Debug, Clone, Default)]
pub struct IlpSolution {
    pub active_nodes: Vec<usize>,
    pub links: Vec<TrackletLink>,
    pub appearances: Vec<usize>,
    pub disappearances: Vec<usize>,
    pub divisions: Vec<usize>,
}

/// `w_e = (-similarity + edge_bias) * exp(-λ (|Δt|-1))` (minimized).
fn edge_weight(e: &EdgeScore, w: &IlpWeights) -> f64 {
    let sim = e.similarity;
    let damp = (-w.delta_t_weight * ((e.delta_t.abs() as f32) - 1.0)).exp();
    ((-sim + w.edge_bias) * damp) as f64
}

/// Two-pass: (1) Δt=1 tracklets only, (2) full candidate graph with flow constraints.
pub fn solve_tracking(
    edges: &[EdgeScore],
    orphans: &[NodeOrphan],
    node_times: &[f32],
    w: &IlpWeights,
    tracklet_solver: bool,
) -> Result<IlpSolution> {
    let orphan_map: HashMap<usize, f32> =
        orphans.iter().map(|o| (o.node_id, o.orphan_prob)).collect();
    if tracklet_solver {
        let _pass1 = solve_ilp(edges, &orphan_map, node_times, w, true)?;
        // Pass 2 uses all edges (tracklet linking); pass-1 solution seeds are not
        // hard-constrained here — matching HOCT's TrackletSolver rebuild.
        solve_ilp(edges, &orphan_map, node_times, w, false)
    } else {
        solve_ilp(edges, &orphan_map, node_times, w, false)
    }
}

fn solve_ilp(
    edges: &[EdgeScore],
    orphan_map: &HashMap<usize, f32>,
    node_times: &[f32],
    w: &IlpWeights,
    ban_long: bool,
) -> Result<IlpSolution> {
    let n_nodes = node_times.len();
    if n_nodes == 0 {
        return Ok(IlpSolution::default());
    }

    let t_min = node_times.iter().copied().fold(f32::INFINITY, f32::min);
    let t_max = node_times.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut vars = variables!();
    let y: Vec<_> = (0..n_nodes)
        .map(|_| vars.add(variable().binary()))
        .collect();
    let a: Vec<_> = (0..n_nodes)
        .map(|_| vars.add(variable().binary()))
        .collect();
    let b: Vec<_> = (0..n_nodes)
        .map(|_| vars.add(variable().binary()))
        .collect();
    let delta: Vec<_> = (0..n_nodes)
        .map(|_| vars.add(variable().binary()))
        .collect();

    let mut x_vars = Vec::new();
    for e in edges {
        if ban_long && e.delta_t.abs() > 1 {
            continue;
        }
        let v = vars.add(variable().binary());
        x_vars.push((e.clone(), v));
    }

    // Minimize sum w_e x_e + sum (w_n y + w_a a + w_b b + w_δ δ)
    let mut objective = Expression::from(0.0);
    for (e, xv) in &x_vars {
        objective += edge_weight(e, w) * *xv;
    }
    for i in 0..n_nodes {
        let orphan = orphan_map.get(&i).copied().unwrap_or(0.0);
        let not_first = if (node_times[i] - t_min).abs() > 1e-6 {
            1.0f32
        } else {
            0.0
        };
        let not_last = if (node_times[i] - t_max).abs() > 1e-6 {
            1.0f32
        } else {
            0.0
        };
        let w_a = (w.appearance * (1.0 - orphan) * not_first) as f64;
        let w_b = (w.disappearance * not_last) as f64;
        let w_d = w.division as f64;
        let w_n = w.node as f64;
        objective += w_n * y[i] + w_a * a[i] + w_b * b[i] + w_d * delta[i];
    }

    let mut model = vars.minimise(objective).using(default_solver);

    for j in 0..n_nodes {
        // y_j = a_j + sum incoming x
        let mut incoming = Expression::from(a[j]);
        for (e, xv) in &x_vars {
            if e.dst == j {
                incoming += *xv;
            }
        }
        model = model.with(constraint!(incoming == y[j]));
    }

    for i in 0..n_nodes {
        // y_i + δ_i = b_i + sum outgoing x
        let mut lhs = Expression::from(y[i]);
        lhs += delta[i];
        let mut rhs = Expression::from(b[i]);
        for (e, xv) in &x_vars {
            if e.src == i {
                rhs += *xv;
            }
        }
        model = model.with(constraint!(lhs == rhs));
        // y_i >= δ_i
        model = model.with(constraint!(y[i] >= delta[i]));
    }

    let solution = model.solve().context("HOCT ILP solve failed")?;

    let mut out = IlpSolution::default();
    for i in 0..n_nodes {
        if solution.value(y[i]) > 0.5 {
            out.active_nodes.push(i);
        }
        if solution.value(a[i]) > 0.5 {
            out.appearances.push(i);
        }
        if solution.value(b[i]) > 0.5 {
            out.disappearances.push(i);
        }
        if solution.value(delta[i]) > 0.5 {
            out.divisions.push(i);
        }
    }
    for (e, xv) in x_vars {
        if solution.value(xv) > 0.5 {
            out.links.push(TrackletLink {
                src: e.src,
                dst: e.dst,
                delta_t: e.delta_t,
                edge_id: e.edge_id,
            });
        }
    }
    Ok(out)
}

/// Backward-compatible alias used by the runner.
pub fn solve_tracklets(
    edges: &[EdgeScore],
    orphans: &[NodeOrphan],
    node_times: &[f32],
    w: &IlpWeights,
) -> Result<IlpSolution> {
    solve_tracking(edges, orphans, node_times, w, true)
}
