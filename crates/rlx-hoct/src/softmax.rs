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

//! Multi-Δt parental softmax (HOCT / Trackastra-style aggregation).
//!
//! Windowed edge logits are median-aggregated in `exp` space, then normalized
//! per `(target, Δt)` with orphan constant `exp(0)=1`. Orphan probabilities are
//! a Δt-weighted average across gaps.

use ndarray::Array3;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EdgeScore {
    pub edge_id: usize,
    pub src: usize,
    pub dst: usize,
    pub delta_t: i32,
    /// Log-space score (post-softmax: `ln(similarity)`).
    pub logit: f32,
    /// Parental probability (or pre-softmax `exp(logit)` during aggregation).
    pub similarity: f32,
}

/// Aggregated orphan probability for one node.
#[derive(Debug, Clone)]
pub struct NodeOrphan {
    pub node_id: usize,
    pub orphan_prob: f32,
}

fn median_f32(xs: &mut [f32]) -> f32 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// Aggregate windowed edge logits (median `exp(logit)`), then parental-normalize.
///
/// For each target `j` and temporal gap `Δt`:
/// `p(e|j,Δt) = exp(ℓ_e) / (Σ_{e'→j,Δt} exp(ℓ_{e'}) + exp(orphan))`
/// with `orphan = 0` ⇒ `exp(orphan) = 1`.
pub fn parental_softmax_aggregate(
    windows: &[Vec<(usize, usize, usize, i32, f32)>],
    orphan_windows: &[Vec<(usize, f32)>],
    delta_t_weight: f32,
) -> (Vec<EdgeScore>, Vec<NodeOrphan>) {
    // edge_id -> list of exp(logit), plus src/dst/dt from first sighting
    let mut edge_exps: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut edge_meta: HashMap<usize, (usize, usize, i32)> = HashMap::new();
    for win in windows {
        for &(eid, src, dst, dt, logit) in win {
            edge_exps
                .entry(eid)
                .or_default()
                .push(logit.clamp(f32::NEG_INFINITY, 20.0).exp());
            edge_meta.entry(eid).or_insert((src, dst, dt));
        }
    }

    let mut node_orph_exps: HashMap<usize, Vec<f32>> = HashMap::new();
    for win in orphan_windows {
        for &(nid, ologit) in win {
            node_orph_exps
                .entry(nid)
                .or_default()
                .push(ologit.clamp(f32::NEG_INFINITY, 20.0).exp());
        }
    }

    let mut edges: Vec<EdgeScore> = Vec::new();
    for (eid, mut exps) in edge_exps {
        let med = median_f32(&mut exps);
        let (src, dst, dt) = edge_meta[&eid];
        edges.push(EdgeScore {
            edge_id: eid,
            src,
            dst,
            delta_t: dt,
            logit: med.ln().max(-80.0),
            similarity: med, // temporarily store sim_exp
        });
    }

    // Parental softmax per (target, delta_t)
    let mut groups: HashMap<(usize, i32), Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        groups.entry((e.dst, e.delta_t)).or_default().push(i);
    }

    let mut orphan_by_node_dt: HashMap<(usize, i32), f32> = HashMap::new();
    for ((dst, dt), idxs) in &groups {
        let orphan_exp = node_orph_exps
            .get(dst)
            .map(|v| {
                let mut c = v.clone();
                median_f32(&mut c)
            })
            .unwrap_or(1.0);
        let sum_sim: f32 = idxs.iter().map(|&i| edges[i].similarity).sum::<f32>() + orphan_exp;
        for &i in idxs {
            edges[i].similarity = if sum_sim > 0.0 {
                edges[i].similarity / sum_sim
            } else {
                0.0
            };
            edges[i].logit = edges[i].similarity.max(1e-12).ln();
        }
        orphan_by_node_dt.insert(
            (*dst, *dt),
            if sum_sim > 0.0 {
                orphan_exp / sum_sim
            } else {
                1.0
            },
        );
    }

    // Aggregate orphan_prob over Δt with temporal weights
    let mut by_node: HashMap<usize, Vec<(f32, f32)>> = HashMap::new();
    for ((nid, dt), p) in orphan_by_node_dt {
        let w = (-delta_t_weight * ((dt.abs() as f32) - 1.0)).exp();
        by_node.entry(nid).or_default().push((p * w, w));
    }
    let orphans: Vec<NodeOrphan> = by_node
        .into_iter()
        .map(|(node_id, parts)| {
            let num: f32 = parts.iter().map(|(a, _)| a).sum();
            let den: f32 = parts.iter().map(|(_, b)| b).sum::<f32>().max(1e-12);
            NodeOrphan {
                node_id,
                orphan_prob: num / den,
            }
        })
        .collect();

    (edges, orphans)
}

pub fn logits_to_window_rows(
    edge_indices: &Array3<i64>,
    edge_logits: &Array3<f32>,
    edge_ids: &[usize],
    nodes_t: &[f32],
) -> Vec<(usize, usize, usize, i32, f32)> {
    let e = edge_indices.len_of(ndarray::Axis(1));
    let mut out = Vec::with_capacity(e);
    for ei in 0..e {
        let i = edge_indices[[0, ei, 0]] as usize;
        let j = edge_indices[[0, ei, 1]] as usize;
        let dt = (nodes_t[j] - nodes_t[i]).round() as i32;
        let eid = edge_ids.get(ei).copied().unwrap_or(ei);
        out.push((eid, i, j, dt, edge_logits[[0, ei, 0]]));
    }
    out
}
