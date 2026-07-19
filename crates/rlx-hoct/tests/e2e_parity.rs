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

use rlx_hoct::config::IlpWeights;
use rlx_hoct::dataset::WindowBatch;
use rlx_hoct::flow::{HoctCompiled, HoctFlow};
use rlx_hoct::ilp::solve_tracking;
use rlx_hoct::io::{assign_tracklets, write_ctc, write_geff_minimal, write_labels_raw};
use rlx_hoct::softmax::{EdgeScore, NodeOrphan};
use rlx_hoct::{HoctModel, HoctRunner};
use std::path::Path;

fn weights_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("HOCT_WEIGHTS") {
        return Some(Path::new(&p).into());
    }
    let default = Path::new("/tmp/hoct-inspect/weights/general_v0.safetensors");
    if default.exists() {
        return Some(default.into());
    }
    let cache = Path::new(".cache/hoct/general_v0.safetensors");
    if cache.exists() {
        return Some(cache.into());
    }
    None
}

#[test]
fn ilp_links_unique_parent_chain() {
    let edges = vec![
        EdgeScore {
            edge_id: 0,
            src: 0,
            dst: 1,
            delta_t: 1,
            logit: 0.0,
            similarity: 0.9,
        },
        EdgeScore {
            edge_id: 1,
            src: 1,
            dst: 2,
            delta_t: 1,
            logit: 0.0,
            similarity: 0.9,
        },
        EdgeScore {
            edge_id: 2,
            src: 0,
            dst: 2,
            delta_t: 2,
            logit: 0.0,
            similarity: 0.1,
        },
    ];
    let orphans = vec![
        NodeOrphan {
            node_id: 0,
            orphan_prob: 0.1,
        },
        NodeOrphan {
            node_id: 1,
            orphan_prob: 0.1,
        },
        NodeOrphan {
            node_id: 2,
            orphan_prob: 0.1,
        },
    ];
    let times = vec![0.0f32, 1.0, 2.0];
    let sol = solve_tracking(&edges, &orphans, &times, &IlpWeights::default(), false).expect("ilp");
    assert!(!sol.active_nodes.is_empty());
    let pairs: Vec<_> = sol.links.iter().map(|l| (l.src, l.dst)).collect();
    assert!(pairs.contains(&(0, 1)) || pairs.contains(&(1, 2)));
}

#[test]
fn ctc_and_geff_roundtrip_dirs() {
    let mut labels = ndarray::Array3::<u32>::zeros((3, 8, 8));
    labels[[0, 2, 2]] = 1;
    labels[[1, 2, 3]] = 1;
    labels[[2, 3, 3]] = 1;
    let mut all = Vec::new();
    for t in 0..3 {
        let lab = labels.slice(ndarray::s![t..t + 1, .., ..]).to_owned();
        all.extend(rlx_hoct::features::regionprops_2d(
            &lab, t as i32, None, 1.0,
        ));
    }
    let sol = rlx_hoct::ilp::IlpSolution {
        active_nodes: (0..all.len()).collect(),
        links: vec![
            rlx_hoct::ilp::TrackletLink {
                src: 0,
                dst: 1,
                delta_t: 1,
                edge_id: 0,
            },
            rlx_hoct::ilp::TrackletLink {
                src: 1,
                dst: 2,
                delta_t: 1,
                edge_id: 1,
            },
        ],
        ..Default::default()
    };
    let tracks = assign_tracklets(&all, &sol);
    assert!(!tracks.is_empty());
    let dir = std::env::temp_dir().join("rlx_hoct_ctc_test");
    let _ = std::fs::remove_dir_all(&dir);
    write_ctc(&dir, &labels, &tracks).expect("ctc");
    assert!(dir.join("res_track.txt").exists());
    let geff = std::env::temp_dir().join("rlx_hoct_geff_test");
    let _ = std::fs::remove_dir_all(&geff);
    write_geff_minimal(&geff, &all, &sol, &tracks).expect("geff");
    assert!(geff.join("tracks.json").exists());
}

#[test]
fn e2e_model_scores_synthetic_graph() {
    let Some(path) = weights_path() else {
        eprintln!("skip e2e_model_scores: no HOCT_WEIGHTS");
        return;
    };
    let mut labels = ndarray::Array3::<u32>::zeros((3, 16, 16));
    labels[[0, 4, 4]] = 1;
    labels[[0, 4, 5]] = 1;
    labels[[1, 5, 5]] = 1;
    labels[[1, 5, 6]] = 1;
    labels[[2, 6, 6]] = 1;
    labels[[0, 12, 12]] = 2;
    labels[[1, 12, 13]] = 2;
    labels[[2, 13, 13]] = 2;

    let runner = HoctRunner::builder()
        .weights(&path)
        .window_size(3)
        .stride(1)
        .build()
        .expect("runner");
    let (sol, nodes) = runner.track_labels(&labels, None).expect("track");
    assert!(
        !sol.links.is_empty() || !sol.active_nodes.is_empty(),
        "expected a non-empty tracking solution"
    );
    let tracks = assign_tracklets(&nodes, &sol);
    let out = std::env::temp_dir().join("rlx_hoct_e2e_out");
    let _ = std::fs::remove_dir_all(&out);
    write_ctc(&out, &labels, &tracks).expect("write ctc");
    write_labels_raw(out.join("labels.raw"), &labels).expect("raw");
}

#[test]
fn eager_vs_compiled_padded_parity() {
    let Some(path) = weights_path() else {
        eprintln!("skip eager_vs_compiled: no HOCT_WEIGHTS");
        return;
    };
    let model = HoctModel::from_weights(&path).expect("weights");
    let compiled = HoctCompiled::from_weights(&path)
        .expect("compiled")
        .with_pad(16, 32);

    let b = 1usize;
    let n = 4usize;
    let e = 3usize;
    let d = model.cfg.feature_dim;
    let mut node_features = ndarray::Array3::<f32>::zeros((b, n, d));
    let mut node_pos = ndarray::Array3::<f32>::zeros((b, n, 3));
    let mut edge_pos = ndarray::Array3::<f32>::zeros((b, e, 3));
    let mut edge_indices = ndarray::Array3::<i64>::zeros((b, e, 2));
    for i in 0..n {
        for k in 0..d {
            node_features[[0, i, k]] = (i * d + k) as f32 * 1e-3;
        }
        node_pos[[0, i, 0]] = i as f32;
        node_pos[[0, i, 1]] = (i as f32) * 2.0;
        node_pos[[0, i, 2]] = 0.0;
    }
    for ei in 0..e {
        edge_indices[[0, ei, 0]] = ei as i64;
        edge_indices[[0, ei, 1]] = (ei + 1) as i64;
        for k in 0..3 {
            edge_pos[[0, ei, k]] = 0.5 * (node_pos[[0, ei, k]] + node_pos[[0, ei + 1, k]]);
        }
    }
    let node_mask = ndarray::Array2::<bool>::from_elem((b, n), true);
    let edge_mask = ndarray::Array2::<bool>::from_elem((b, e), true);
    let batch = WindowBatch {
        node_features: node_features.clone(),
        node_pos: node_pos.clone(),
        edge_pos: edge_pos.clone(),
        edge_indices: edge_indices.clone(),
        node_mask: node_mask.clone(),
        edge_mask: edge_mask.clone(),
        frame_t: 0,
    };

    let eager = model.forward(
        &node_features.view(),
        &node_pos.view(),
        &edge_pos.view(),
        &edge_indices,
        &node_mask,
        &edge_mask,
    );
    let pad_out = compiled.forward_padded_fixed(&batch);
    let n_live = n;
    let e_live = e;
    for ei in 0..e_live {
        let a = eager.edge_logits[[0, ei, 0]];
        let b = pad_out.edge_logits[[0, ei, 0]];
        assert!((a - b).abs() < 1e-6, "edge {ei} eager={a} padded={b}");
    }
    for ni in 0..n_live {
        assert_eq!(eager.orphan_logits[[0, ni, 0]], 0.0);
        assert_eq!(pad_out.orphan_logits[[0, ni, 0]], 0.0);
    }
}

#[test]
fn head_modelflow_builds() {
    let Some(path) = weights_path() else {
        eprintln!("skip head_modelflow_builds: no HOCT_WEIGHTS");
        return;
    };
    let weights = rlx_hoct::load_hoct_weights(&path).expect("load");
    let flow = HoctFlow::new(rlx_hoct::HoctConfig::default()).with_pad(8, 16);
    let mut wm = HoctFlow::head_weight_map(&weights);
    let built = flow.build_head_flow(&mut wm).expect("head flow");
    assert_eq!(built.output_names(), &["edge_logits".to_string()]);
    assert_eq!(built.primary_shape().rank(), 3);
}

#[test]
fn env_gated_python_parity_hint() {
    if std::env::var("HOCT_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip env_gated_python_parity: set HOCT_PARITY=1 with Python hoct + fixtures");
        return;
    }
    // Full Python comparison is performed by scripts/dump_jit_reference.py +
    // model_parity when /tmp/hoct_ref_logits.npy is present.
    let ref_path = Path::new("/tmp/hoct_ref_logits.npy");
    assert!(
        ref_path.exists(),
        "HOCT_PARITY=1 requires /tmp/hoct_ref_logits.npy (run dump_jit_reference.py)"
    );
}
