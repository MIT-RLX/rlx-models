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

use ndarray::{Array2, Array3};
use rlx_hoct::config::HoctConfig;
use rlx_hoct::geometry::{line_to_line_distances, pairwise_sqdist};
use rlx_hoct::model::HoctModel;
use rlx_hoct::rope3d::{apply_rope_rotation, apply_rope3d};
use std::path::Path;

fn weights_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("HOCT_WEIGHTS") {
        return Some(Path::new(&p).into());
    }
    let default = Path::new("/tmp/hoct-inspect/weights/general_v0.safetensors");
    if default.exists() {
        return Some(default.into());
    }
    None
}

fn fixture_dir() -> Option<std::path::PathBuf> {
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jit_ref");
    if local.join("logits.npy").exists() {
        return Some(local);
    }
    None
}

fn load_ref_inputs() -> Option<(
    Array3<f32>,
    Array3<f32>,
    Array3<f32>,
    Array3<i64>,
    Array2<bool>,
    Array2<bool>,
)> {
    let dir = fixture_dir().or_else(|| {
        let stem = Path::new("/tmp/hoct_ref_logits");
        Path::new(&format!("{}_node_features.npy", stem.display()))
            .exists()
            .then(|| Path::new("/tmp").to_path_buf())
    })?;

    let (nf, npos, epos, eidx, nmask, emask) = if dir.ends_with("jit_ref") {
        (
            dir.join("node_features.npy"),
            dir.join("node_pos.npy"),
            dir.join("edge_pos.npy"),
            dir.join("edge_indices.npy"),
            dir.join("node_mask.npy"),
            dir.join("edge_mask.npy"),
        )
    } else {
        let stem = "/tmp/hoct_ref_logits";
        (
            Path::new(&format!("{stem}_node_features.npy")).to_path_buf(),
            Path::new(&format!("{stem}_node_pos.npy")).to_path_buf(),
            Path::new(&format!("{stem}_edge_pos.npy")).to_path_buf(),
            Path::new(&format!("{stem}_edge_indices.npy")).to_path_buf(),
            Path::new(&format!("{stem}_node_mask.npy")).to_path_buf(),
            Path::new(&format!("{stem}_edge_mask.npy")).to_path_buf(),
        )
    };

    let node_features: Array3<f32> = ndarray_npy::read_npy(&nf).ok()?;
    let node_pos: Array3<f32> = ndarray_npy::read_npy(&npos).ok()?;
    let edge_pos: Array3<f32> = ndarray_npy::read_npy(&epos).ok()?;
    let edge_indices: Array3<i64> = ndarray_npy::read_npy(&eidx).ok()?;
    let node_mask: Array2<bool> = ndarray_npy::read_npy(&nmask).ok()?;
    let edge_mask: Array2<bool> = ndarray_npy::read_npy(&emask).ok()?;
    Some((
        node_features,
        node_pos,
        edge_pos,
        edge_indices,
        node_mask,
        edge_mask,
    ))
}

#[test]
fn geometry_line_to_line_is_finite() {
    let node_pos =
        Array3::from_shape_vec((1, 3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
    let edge_indices = Array3::from_shape_vec((1, 2, 2), vec![0, 1, 1, 2]).unwrap();
    let d = line_to_line_distances(&node_pos, &edge_indices);
    assert!(d.iter().all(|v| v.is_finite()));
}

#[test]
fn rope3d_shape_preserved() {
    let cfg = HoctConfig::default();
    let x = ndarray::Array4::<f32>::from_elem((1, 4, 5, 72), 0.1);
    let pos = Array3::<f32>::from_elem((1, 5, 3), 10.0);
    let log_freq = vec![0.01f32; 48];
    let reflect_vec = vec![1.0f32; 288];
    let mut eye = vec![0.0f32; 72 * 72];
    for i in 0..72 {
        eye[i * 72 + i] = 1.0;
    }
    let y = apply_rope3d(&cfg, &x.view(), &pos, &log_freq, &reflect_vec, &eye);
    assert_eq!(y.shape(), x.shape());
}

#[test]
fn model_orphan_zeros_and_finite_logits() {
    let Some(path) = weights_path() else {
        eprintln!("skip model_orphan_zeros: set HOCT_WEIGHTS or install /tmp/hoct-inspect/weights");
        return;
    };
    let model = HoctModel::from_weights(&path).expect("load weights");

    let (node_features, node_pos, edge_pos, edge_indices, node_mask, edge_mask) =
        if let Some(inputs) = load_ref_inputs() {
            inputs
        } else {
            eprintln!("warning: /tmp/hoct_ref_logits_*.npy missing — using deterministic fallback");
            let b = 1usize;
            let n = 8usize;
            let e = 12usize;
            let d = model.cfg.feature_dim;
            let mut node_features = Array3::<f32>::zeros((b, n, d));
            let mut node_pos = Array3::<f32>::zeros((b, n, 3));
            let edge_pos = Array3::<f32>::zeros((b, e, 3));
            let mut edge_indices = Array3::<i64>::zeros((b, e, 2));
            for i in 0..n {
                for k in 0..d {
                    node_features[[0, i, k]] = (i * d + k) as f32 * 1e-3;
                }
                node_pos[[0, i, 0]] = i as f32;
            }
            for ei in 0..e {
                edge_indices[[0, ei, 0]] = (ei % n) as i64;
                edge_indices[[0, ei, 1]] = ((ei + 1) % n) as i64;
            }
            let node_mask = Array2::<bool>::from_elem((b, n), true);
            let edge_mask = Array2::<bool>::from_elem((b, e), true);
            (
                node_features,
                node_pos,
                edge_pos,
                edge_indices,
                node_mask,
                edge_mask,
            )
        };

    let out = model.forward(
        &node_features.view(),
        &node_pos.view(),
        &edge_pos.view(),
        &edge_indices,
        &node_mask,
        &edge_mask,
    );
    assert_eq!(out.orphan_logits.sum(), 0.0);
    assert!(out.edge_logits.iter().all(|v| v.is_finite()));

    let ref_path = fixture_dir()
        .map(|d| d.join("logits.npy"))
        .unwrap_or_else(|| Path::new("/tmp/hoct_ref_logits.npy").to_path_buf());
    if ref_path.exists() {
        let ref_logits: Array3<f32> = ndarray_npy::read_npy(&ref_path).expect("read ref npy");
        let diff = (&out.edge_logits - &ref_logits).mapv(f32::abs);
        let max = diff.iter().copied().fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "max logits diff {max} (need JIT parity within 1e-4)"
        );
        let ref_node = fixture_dir()
            .map(|d| d.join("node_h.npy"))
            .unwrap_or_else(|| Path::new("/tmp/hoct_ref_node_h.npy").to_path_buf());
        if ref_node.exists() {
            let ref_n: Array3<f32> = ndarray_npy::read_npy(&ref_node).expect("node ref");
            let nd = (&out.node_hidden - &ref_n).mapv(f32::abs);
            let nmax = nd.iter().copied().fold(0.0f32, f32::max);
            let nrel = out
                .node_hidden
                .iter()
                .zip(ref_n.iter())
                .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                .fold(0.0f32, f32::max);
            // Absolute ~1e-4 on O(10–100) activations; relative matches logit quality.
            assert!(
                nmax < 5e-4 || nrel < 1e-5,
                "node hidden max abs {nmax} rel {nrel}"
            );
        }
        let ref_edge = fixture_dir()
            .map(|d| d.join("edge_h.npy"))
            .unwrap_or_else(|| Path::new("/tmp/hoct_ref_edge_h.npy").to_path_buf());
        if ref_edge.exists() {
            let ref_e: Array3<f32> = ndarray_npy::read_npy(&ref_edge).expect("edge ref");
            let ed = (&out.edge_hidden - &ref_e).mapv(f32::abs);
            let emax = ed.iter().copied().fold(0.0f32, f32::max);
            let erel = out
                .edge_hidden
                .iter()
                .zip(ref_e.iter())
                .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                .fold(0.0f32, f32::max);
            // Edge hidden is O(1e3–1e4); fp32 block accumulation → ~1e-4 rel,
            // with occasional abs peaks around ~1.5e-2 on real fixtures.
            assert!(
                emax < 2e-2 && erel < 1e-3,
                "edge hidden max abs {emax} rel {erel}"
            );
        }
    }
}

#[test]
fn rope_matches_python_reference() {
    let path_x = Path::new("/tmp/hoct_rope_x.npy");
    if !path_x.exists() {
        eprintln!(
            "skip rope_matches_python_reference: run crates/rlx-hoct/scripts/compare_rope.py first"
        );
        return;
    }
    let x: ndarray::Array4<f32> = ndarray_npy::read_npy(path_x).expect("x");
    let pos: Array3<f32> = ndarray_npy::read_npy("/tmp/hoct_rope_pos.npy").expect("pos");
    let y_ref: ndarray::Array4<f32> = ndarray_npy::read_npy("/tmp/hoct_rope_ref.npy").expect("ref");
    let model = HoctModel::from_weights(weights_path().expect("weights")).expect("model");
    let w = &model.weights.node_blocks[0].attn;
    let rot = apply_rope_rotation(&model.cfg, &x.view(), &pos, &w.log_freq);
    let rot_ref: ndarray::Array4<f32> =
        ndarray_npy::read_npy("/tmp/hoct_rope_rotated.npy").expect("rot");
    let rot_max = (&rot - &rot_ref)
        .mapv(f32::abs)
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(rot_max < 1e-4, "rope rotation max diff {rot_max}");
    let y = apply_rope3d(
        &model.cfg,
        &x.view(),
        &pos,
        &w.log_freq,
        &w.reflect_vec,
        &w.eye,
    );
    let max = (&y - &y_ref)
        .mapv(f32::abs)
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(max < 1e-4, "rope max diff {max}");
}

#[test]
fn pairwise_sqdist_symmetric() {
    let pos = Array3::from_shape_vec((1, 2, 3), vec![0.0, 0.0, 0.0, 3.0, 4.0, 0.0]).unwrap();
    let d2 = pairwise_sqdist(&pos);
    assert!((d2[[0, 0, 1]] - d2[[0, 1, 0]]).abs() < 1e-5);
}
