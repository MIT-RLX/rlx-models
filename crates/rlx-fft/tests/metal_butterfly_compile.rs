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

//! Metal compile checks for butterfly narrow/concat graphs.

#![cfg(feature = "metal")]

use rlx_fft::butterfly::build_butterfly_forward_graph;
use rlx_fft::compile::try_compile_graph;
use rlx_fft::config::FftLearnConfig;
use rlx_fft::reference::{fft_real_batch, max_abs_error};
use rlx_fft::twiddle::exact_twiddles;
use rlx_fft::weights::WeightStore;
use rlx_ir::Op;
use rlx_runtime::Device;

#[test]
fn butterfly_graph_narrow_count_scales_with_stages() {
    for &n in &[8usize, 16, 32, 64] {
        let cfg = FftLearnConfig::new(n, 1).unwrap();
        let built = build_butterfly_forward_graph(&cfg).unwrap();
        let narrow = built
            .graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, Op::Narrow { .. }))
            .count();
        let gather = built
            .graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, Op::Gather { .. }))
            .count();
        eprintln!(
            "n_fft={n} nodes={} narrow={narrow} gather={gather}",
            built.graph.nodes().len()
        );
        let stages = n.trailing_zeros() as usize;
        assert_eq!(gather, 0, "vectorized butterfly avoids gather");
        assert!(
            narrow <= n + stages * 4 + 4,
            "O(n) bit-reverse + O(stages) stage narrows, got {narrow}"
        );
    }
}

#[test]
#[ignore = "slow Metal thunk compile (~minutes for n=64); run with --ignored"]
fn butterfly_compiles_and_runs_on_metal_with_mpsgraph_guard() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("Metal unavailable — skip");
        return;
    }
    let cfg = FftLearnConfig::new(64, 2).unwrap();
    let built = build_butterfly_forward_graph(&cfg).unwrap();
    let mut compiled = try_compile_graph(Device::Metal, built.graph).expect("metal compile");
    WeightStore::from_twiddles(&exact_twiddles(&cfg), 64).apply_butterfly(&mut compiled, 2, 64);

    let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).sin()).collect();
    let pred = compiled.run(&[("signal", &signal)]).remove(0);
    let target = fft_real_batch(&signal, 2, 64).unwrap();
    let err = max_abs_error(&pred, &target);
    assert!(err < 1e-3, "compiled butterfly vs ref err={err}");
}

#[test]
fn butterfly_compiles_on_metal_n16() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("Metal unavailable — skip");
        return;
    }
    let cfg = FftLearnConfig::new(16, 1).unwrap();
    let built = build_butterfly_forward_graph(&cfg).unwrap();
    let mut compiled = try_compile_graph(Device::Metal, built.graph).expect("metal n=16 compile");
    WeightStore::from_twiddles(&exact_twiddles(&cfg), 16).apply_butterfly(&mut compiled, 1, 16);
    let signal: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
    let pred = compiled.run(&[("signal", &signal)]).remove(0);
    let target = fft_real_batch(&signal, 1, 16).unwrap();
    assert!(max_abs_error(&pred, &target) < 1e-3);
}
