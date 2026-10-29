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

//! Latency of each native graph vs the upstream `.onnx` on ONNX Runtime (the
//! reference), same inputs, iterations interleaved so both see the same
//! machine load. Page-sized shapes; prints median / p90 per network.
//!
//! ```sh
//! INDRA_UPSTREAM_DIR=.cache/indra-upstream INDRA_MODEL_DIR=.cache/indra \
//!   cargo test -p rlx-indra --features parity-ort --release --test indra_bench -- --nocapture
//! ```
//! `INDRA_BENCH_DEVICE` (default `cpu`), `INDRA_BENCH_ITERS` (default 15).

use ort::session::Session;
use ort::value::{DynValue, Tensor};
use rlx_indra::graph::{self, Compiled, Weights};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[(v.len() * 9 / 10).min(v.len() - 1)])
}

fn ink(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (s >> 60) == 0 { 0.9 } else { 0.0 }
        })
        .collect()
}

struct Case {
    name: String,
    net: Compiled,
    key: Vec<usize>,
    native_in: Vec<(&'static str, Vec<f32>)>,
    ort: Session,
    ort_in: Vec<(&'static str, DynValue)>,
}

#[test]
fn bench_networks() {
    let (Some(up), Some(native)) = (
        std::env::var_os("INDRA_UPSTREAM_DIR"),
        std::env::var_os("INDRA_MODEL_DIR"),
    ) else {
        eprintln!("skip: set INDRA_UPSTREAM_DIR and INDRA_MODEL_DIR");
        return;
    };
    let (up, native) = (PathBuf::from(up), PathBuf::from(native));
    let device = std::env::var("INDRA_BENCH_DEVICE")
        .ok()
        .map(|s| rlx_runtime::parse_device(&s).unwrap())
        .unwrap_or(Device::Cpu);
    let iters: usize = std::env::var("INDRA_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let w = |n: &str| Weights::load(&native.join(format!("{n}.safetensors"))).unwrap();
    let sess = |n: &str| {
        Session::builder()
            .unwrap()
            .commit_from_file(up.join(format!("{n}.onnx")))
            .unwrap()
    };
    let t32 = |shape: Vec<usize>, d: Vec<f32>| Tensor::from_array((shape, d)).unwrap().into_dyn();
    let ti64 = |d: Vec<i64>| Tensor::from_array(([d.len()], d)).unwrap().into_dyn();

    let mut cases = Vec::new();
    // Detector: a letter page at the default long side.
    let (h, wd) = (1024, 800);
    let x = ink(h * wd, 1);
    cases.push(Case {
        name: format!("det {h}×{wd}"),
        net: Compiled::new(w("det"), device, |w, k| graph::det::build(w, k[0], k[1])),
        key: vec![h, wd],
        native_in: vec![("x", x.clone())],
        ort: sess("det"),
        ort_in: vec![("x", t32(vec![1, 1, h, wd], x))],
    });
    // Recognizers: a batch of 32 lines, 512 px wide.
    for net in ["rec", "rec_cjk", "rec_world"] {
        let (b, wd) = (32, 512);
        let wt = w(net);
        let rows = graph::rec::RecSpec::from_weights(&wt).unwrap().height;
        let x = ink(b * rows * wd, 2);
        let widths = vec![wd; b];
        let (mask, inv) = graph::rec::line_masks(&widths, wd / graph::rec::STRIDE);
        cases.push(Case {
            name: format!("{net} B={b} W={wd}"),
            net: Compiled::new(wt, device, |w, k| graph::rec::build(w, k[0], k[1])),
            key: vec![b, wd],
            native_in: vec![("x", x.clone()), ("mask", mask), ("inv_count", inv)],
            ort: sess(net),
            ort_in: vec![
                ("x", t32(vec![b, 1, rows, wd], x)),
                ("widths", ti64(vec![wd as i64; b])),
            ],
        });
    }
    // Script router: 64 lines.
    let (b, wd) = (64, 512);
    let x = ink(b * 32 * wd, 3);
    let (mask, inv) =
        graph::classifier::script_masks(&vec![wd; b], wd / graph::classifier::SCRIPT_STRIDE);
    cases.push(Case {
        name: format!("script B={b} W={wd}"),
        net: Compiled::new(w("script"), device, |w, k| {
            graph::classifier::build_script(w, k[0], k[1])
        }),
        key: vec![b, wd],
        native_in: vec![("x", x.clone()), ("mask", mask), ("inv_count", inv)],
        ort: sess("script"),
        ort_in: vec![
            ("x", t32(vec![b, 1, 32, wd], x)),
            ("widths", ti64(vec![wd as i64; b])),
        ],
    });
    // Orientation.
    let x = ink(768 * 768, 4);
    cases.push(Case {
        name: "orient 768".into(),
        net: Compiled::new(w("orient"), device, |w, k| {
            graph::classifier::build_orient(w, 1, k[0])
        }),
        key: vec![768],
        native_in: vec![("x", x.clone())],
        ort: sess("orient"),
        ort_in: vec![("x", t32(vec![1, 1, 768, 768], x))],
    });
    // Reading order: 128 lines.
    let n = 128;
    let feats: Vec<f32> = ink(n * 16, 5)
        .iter()
        .enumerate()
        .map(|(i, v)| v + (i % 16) as f32 * 0.03)
        .collect();
    cases.push(Case {
        name: format!("order N={n}"),
        net: Compiled::new(w("order"), device, |w, k| graph::order::build(w, k[0])),
        key: vec![n],
        native_in: vec![("feats", feats.clone()), ("key_bias", vec![0.0; n])],
        ort: sess("order"),
        ort_in: vec![
            ("feats", t32(vec![1, n, 16], feats)),
            (
                "mask",
                Tensor::from_array(([1usize, n], vec![true; n]))
                    .unwrap()
                    .into_dyn(),
            ),
        ],
    });

    eprintln!("device {device:?}, {iters} iterations, median / p90 ms");
    eprintln!(
        "{:<24} {:>10} {:>16} {:>16} {:>8}",
        "network", "compile", "rlx", "onnxruntime", "speedup"
    );
    for c in &mut cases {
        let Case {
            name,
            net,
            key,
            native_in,
            ort,
            ort_in,
        } = c;
        let ins: Vec<(&str, &[f32])> = native_in.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        let t = Instant::now();
        net.run(key, &ins).unwrap();
        let compile = t.elapsed().as_secs_f64() * 1e3;
        for _ in 0..2 {
            net.run(key, &ins).unwrap();
            ort.run(
                ort_in
                    .iter()
                    .map(|(k, v)| (*k, v.view()))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
        let (mut tn, mut to) = (Vec::new(), Vec::new());
        for _ in 0..iters {
            let t = Instant::now();
            net.run(key, &ins).unwrap();
            tn.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            ort.run(
                ort_in
                    .iter()
                    .map(|(k, v)| (*k, v.view()))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            to.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ((nm, np), (om, op)) = (stats(tn), stats(to));
        eprintln!(
            "{:<24} {:>10.1} {:>7.2} / {:>6.2} {:>7.2} / {:>6.2} {:>7.2}×",
            name,
            compile,
            nm,
            np,
            om,
            op,
            om / nm
        );
    }
}
