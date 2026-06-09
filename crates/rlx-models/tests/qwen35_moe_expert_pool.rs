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

//! MoE expert pool wiring for synthetic qwen35moe (TIDE policy, no GGUF).

use rlx_models::qwen35::synth;
use rlx_models::qwen35::{Qwen35Runner, build_moe_offload};
use rlx_runtime::Device;

#[test]
fn synthetic_moe_runner_enables_expert_pool() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let runner = Qwen35Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .max_gpu_experts_per_layer(2)
        .expert_refresh_every_decode_steps(4)
        .build()
        .expect("build moe runner with offload");
    let mo = runner
        .moe_offload()
        .expect("offload when budget < num_experts");
    assert_eq!(mo.pools.len(), 3);
    assert_eq!(mo.pools[0].num_experts(), cfg.num_experts);
    assert_eq!(mo.pools[0].gpu_budget(), 2);
}

#[test]
fn build_moe_offload_refresh_from_synthetic_routing() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let mut mo =
        build_moe_offload(&cfg, &weights, Some(2), None, Some(2), 1.5, false).expect("offload");
    let flat: Vec<u32> = vec![0, 0, 1, 2, 1];
    mo.pools[0].refresh_from_indices(&flat);
    assert!(mo.pools[0].is_gpu_resident(0));
}
