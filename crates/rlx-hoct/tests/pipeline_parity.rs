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
use rlx_hoct::features::{border_dist_centroid, regionprops_2d};
use rlx_hoct::softmax::{EdgeScore, parental_softmax_aggregate};
use std::path::Path;

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pipeline")
}

#[test]
fn border_dist_matches_python_fixture() {
    let path = fixture_dir().join("border_dist.npy");
    if !path.exists() {
        eprintln!("skip: run scripts/dump_pipeline_fixtures.py");
        return;
    }
    let ref_bd: ndarray::Array1<f32> = ndarray_npy::read_npy(&path).unwrap();
    // Centroids matching dump script: (z=0,y=4.5,x=5) and (0,12.5,12.5) in shape (1,16,16)
    let a = border_dist_centroid(&[0.0, 4.5, 5.0], &[1, 16, 16], 5.0);
    let b = border_dist_centroid(&[0.0, 12.5, 12.5], &[1, 16, 16], 5.0);
    assert!((a - ref_bd[0]).abs() < 1e-6, "a={a} ref={}", ref_bd[0]);
    assert!((b - ref_bd[1]).abs() < 1e-6, "b={b} ref={}", ref_bd[1]);
}

#[test]
fn parental_softmax_matches_python_fixture() {
    let dir = fixture_dir();
    let sim_path = dir.join("soft_similarity.npy");
    if !sim_path.exists() {
        eprintln!("skip: run scripts/dump_pipeline_fixtures.py");
        return;
    }
    let sim_exp: ndarray::Array1<f32> =
        ndarray_npy::read_npy(dir.join("soft_sim_exp.npy")).unwrap();
    let target: ndarray::Array1<i64> = ndarray_npy::read_npy(dir.join("soft_target.npy")).unwrap();
    let delta_t: ndarray::Array1<i64> =
        ndarray_npy::read_npy(dir.join("soft_delta_t.npy")).unwrap();
    let ref_sim: ndarray::Array1<f32> = ndarray_npy::read_npy(&sim_path).unwrap();

    // Build one "window" of raw logits such that exp(logit)=sim_exp (orphan=0 → exp=1).
    let rows: Vec<(usize, usize, usize, i32, f32)> = (0..sim_exp.len())
        .map(|i| {
            let dst = target[i] as usize;
            let dt = delta_t[i] as i32;
            let logit = sim_exp[i].ln();
            (i, 0usize, dst, dt, logit)
        })
        .collect();
    let orphan_win = vec![(1usize, 0.0f32), (2usize, 0.0f32)];
    let (edges, orphans) = parental_softmax_aggregate(&[rows], &[orphan_win], 0.5);
    let mut by_id: Vec<&EdgeScore> = edges.iter().collect();
    by_id.sort_by_key(|e| e.edge_id);
    for (i, e) in by_id.iter().enumerate() {
        assert!(
            (e.similarity - ref_sim[i]).abs() < 1e-5,
            "edge {i}: got {} ref {}",
            e.similarity,
            ref_sim[i]
        );
    }
    assert!(!orphans.is_empty());
}

#[test]
fn regionprops_border_and_diameter_sane() {
    let mut labels = Array3::<u32>::zeros((1, 16, 16));
    labels[[0, 4, 4]] = 1;
    labels[[0, 4, 5]] = 1;
    labels[[0, 4, 6]] = 1;
    labels[[0, 5, 4]] = 1;
    labels[[0, 5, 5]] = 1;
    labels[[0, 5, 6]] = 1;
    let nodes = regionprops_2d(&labels, 0, None, 1.0);
    assert_eq!(nodes.len(), 1);
    let n = &nodes[0];
    assert!((n.y - 4.5).abs() < 1e-5);
    assert!((n.x - 5.0).abs() < 1e-5);
    // Z=1 pad → 3D diameter (6V/π)^(1/3)
    let expected_d = (6.0f32 * 6.0 / std::f32::consts::PI).powf(1.0 / 3.0);
    assert!((n.diameter - expected_d).abs() < 1e-5);
    // Centroid border vs shape (1,16,16), cutoff 5
    let bd = border_dist_centroid(&[n.z, n.y, n.x], &[1, 16, 16], 5.0);
    assert!((n.border_dist - bd).abs() < 1e-6);
    // Inertia I_zz ≈ 0.9167 for this blob (skimage 3D Z=1)
    assert!(
        (n.inertia[0] - 0.916_666_7).abs() < 1e-4,
        "Izz={}",
        n.inertia[0]
    );
}
