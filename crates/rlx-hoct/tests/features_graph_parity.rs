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

use ndarray::Array3;
use rlx_hoct::config::GraphConfig;
use rlx_hoct::features::{nodes_to_array, regionprops_2d};
use rlx_hoct::graph::CandidateGraph;

#[test]
fn regionprops_finds_two_blobs() {
    let mut labels = Array3::<u32>::zeros((2, 8, 8));
    // t=0: label 1 at (2,2), label 2 at (5,5)
    labels[[0, 2, 2]] = 1;
    labels[[0, 2, 3]] = 1;
    labels[[0, 3, 2]] = 1;
    labels[[0, 5, 5]] = 2;
    // t=1: moved
    labels[[1, 2, 3]] = 1;
    labels[[1, 6, 6]] = 2;

    let mut nodes = Vec::new();
    for t in 0..2 {
        let slice = labels.slice(ndarray::s![t..t + 1, .., ..]).to_owned();
        nodes.extend(regionprops_2d(&slice, t, None, 1.0));
    }
    assert!(nodes.len() >= 4);
    assert!(
        nodes
            .iter()
            .any(|n| n.label == 1 && (n.t - 0.0).abs() < 1e-6)
    );
    assert!(nodes.iter().any(|n| n.label == 2));

    let cfg = GraphConfig {
        distance_threshold: 300.0,
        n_neighbors: 5,
        max_delta_t: 3,
        spatial_scale: 1.0,
    };
    let graph = CandidateGraph::build(nodes, &cfg);
    assert!(!graph.edges.is_empty());
    let feats = nodes_to_array(&graph.nodes);
    assert_eq!(feats.shape()[2], 19);
}

#[test]
fn candidate_edges_forward_in_time() {
    let mut labels = Array3::<u32>::zeros((3, 4, 4));
    labels[[0, 1, 1]] = 1;
    labels[[1, 1, 2]] = 1;
    labels[[2, 2, 2]] = 1;
    let mut nodes = Vec::new();
    for t in 0..3 {
        let slice = labels.slice(ndarray::s![t..t + 1, .., ..]).to_owned();
        nodes.extend(regionprops_2d(&slice, t, None, 1.0));
    }
    let graph = CandidateGraph::build(nodes, &GraphConfig::default());
    for e in &graph.edges {
        assert!(e.delta_t > 0);
        assert!(e.delta_t <= 3);
    }
}
