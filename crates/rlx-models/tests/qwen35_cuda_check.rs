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

// basic test: tiny synthetic qwen35 graph on CUDA (skips when no driver).

mod compile_support;

#[cfg(feature = "cuda")]
mod cuda_tests {
    use rlx_models::qwen35::synth;
    use rlx_models::{Qwen35RunnerBuilder, build_qwen35_graph_sized};
    use rlx_runtime::Device;

    #[test]
    fn qwen35_tiny_graph_runs_on_cuda() {
        if !rlx_runtime::is_available(Device::Cuda) {
            eprintln!("skip: CUDA not available");
            return;
        }
        let _ = Qwen35RunnerBuilder::default();
        let cfg = synth::tiny_cfg();
        let weights = synth::synth_weights(&cfg);
        let (graph, params, _packed) =
            build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, true).expect("build");
        let mut compiled =
            super::compile_support::compile_qwen35_prefill(Device::Cuda, graph, params);
        let ids = vec![1.0f32, 2.0, 3.0, 4.0];
        let last_token_idx = vec![3.0f32];
        let outs = compiled.run(&[("input_ids", &ids), ("last_token_idx", &last_token_idx)]);
        assert!(!outs.is_empty());
        assert!(outs[0].iter().all(|v| v.is_finite()));
    }
}
