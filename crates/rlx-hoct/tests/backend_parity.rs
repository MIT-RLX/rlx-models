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

//! Backend matrix: compiled HOCT score head on every available RLX device.
//!
//! ```sh
//! cargo test -p rlx-hoct --test backend_parity --features apple-silicon --release
//! just features=all-backends test-hoct-backends
//! ```

use ndarray::{Array2, Array3};
use rlx_hoct::device::{HoctDeviceRunner, device_label, parity_backends};
use rlx_hoct::model::HoctModel;
use rlx_runtime::Device;
use std::path::Path;

fn weights_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("HOCT_WEIGHTS") {
        return Some(Path::new(&p).into());
    }
    for p in [
        "/tmp/hoct-inspect/weights/general_v0.safetensors",
        ".cache/hoct/general_v0.safetensors",
    ] {
        let path = Path::new(p);
        if path.exists() {
            return Some(path.into());
        }
    }
    None
}

fn max_abs(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn fixture_batch(
    model: &HoctModel,
) -> (
    Array3<f32>,
    Array3<f32>,
    Array3<f32>,
    Array3<i64>,
    Array2<bool>,
    Array2<bool>,
) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jit_ref");
    if dir.join("node_features.npy").exists() {
        let nf = ndarray_npy::read_npy(dir.join("node_features.npy")).unwrap();
        let npos = ndarray_npy::read_npy(dir.join("node_pos.npy")).unwrap();
        let epos = ndarray_npy::read_npy(dir.join("edge_pos.npy")).unwrap();
        let eidx = ndarray_npy::read_npy(dir.join("edge_indices.npy")).unwrap();
        let nmask = ndarray_npy::read_npy(dir.join("node_mask.npy")).unwrap();
        let emask = ndarray_npy::read_npy(dir.join("edge_mask.npy")).unwrap();
        return (nf, npos, epos, eidx, nmask, emask);
    }
    let b = 1usize;
    let n = 4usize;
    let e = 3usize;
    let d = model.cfg.feature_dim;
    let mut nf = Array3::<f32>::zeros((b, n, d));
    let mut npos = Array3::<f32>::zeros((b, n, 3));
    let epos = Array3::<f32>::zeros((b, e, 3));
    let mut eidx = Array3::<i64>::zeros((b, e, 2));
    for i in 0..n {
        for k in 0..d {
            nf[[0, i, k]] = (i * d + k) as f32 * 1e-3;
        }
        npos[[0, i, 0]] = i as f32;
    }
    for ei in 0..e {
        eidx[[0, ei, 0]] = ei as i64;
        eidx[[0, ei, 1]] = (ei + 1) as i64;
    }
    (
        nf,
        npos,
        epos,
        eidx,
        Array2::from_elem((b, n), true),
        Array2::from_elem((b, e), true),
    )
}

#[test]
fn score_head_matches_eager_on_all_backends() {
    let Some(path) = weights_path() else {
        eprintln!("skip backend_parity: no HOCT_WEIGHTS");
        return;
    };
    let model = HoctModel::from_weights(&path).expect("weights");
    let (nf, npos, epos, eidx, nmask, emask) = fixture_batch(&model);
    let eager = model.forward(
        &nf.view(),
        &npos.view(),
        &epos.view(),
        &eidx,
        &nmask,
        &emask,
    );
    let e_live = eager.edge_logits.len_of(ndarray::Axis(1));

    for device in parity_backends() {
        let label = device_label(device);
        eprintln!("[hoct] backend {label}…");
        let mut runner = match HoctDeviceRunner::from_parts(
            model.weights.clone(),
            model.cfg,
            device,
            e_live.max(16),
            1,
        ) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("skip {label}: {err}");
                continue;
            }
        };
        let out = runner
            .forward(
                &nf.view(),
                &npos.view(),
                &epos.view(),
                &eidx,
                &nmask,
                &emask,
            )
            .unwrap_or_else(|e| panic!("{label} forward failed: {e}"));
        let diff = max_abs(&out.edge_logits, &eager.edge_logits);
        assert!(
            diff < 1e-4,
            "{label} score-head max abs vs eager CPU {diff}"
        );
        assert_eq!(out.orphan_logits.sum(), 0.0);
        // Body tensors must match bit-for-bit (same eager path).
        if device == Device::Cpu {
            assert_eq!(max_abs(&out.edge_hidden, &eager.edge_hidden), 0.0);
        }
        eprintln!("[hoct] {label} ok (logit max abs {diff:.3e})");
    }
}
