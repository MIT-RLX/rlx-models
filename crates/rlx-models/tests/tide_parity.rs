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

//! Parity with `/Users/Shared/TIDE` (ims-kdks/TIDE reference repo).

use rlx_models::llada2::{GenerateConfig, synth as llada2_synth};
use rlx_models::qwen35::{Qwen35Runner, build_moe_offload, synth};
use rlx_models::tide::{
    LLaDA2MoeConfig, PredictiveOffloadParams, TideOffloadStats, TideRunner,
    enable_predictive_expert_offload, gpu_expert_budget_from_device_memory,
    num_transfer_tokens_schedule, refresh_experts,
};
use rlx_runtime::Device;

#[test]
fn llada2_config_loads_from_tide_json() {
    let json = std::fs::read_to_string("/Users/Shared/TIDE/model/config.json")
        .expect("read /Users/Shared/TIDE/model/config.json");
    let cfg = LLaDA2MoeConfig::from_json_str(&json).expect("parse");
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.num_sparse_moe_layers(), 19);
}

#[test]
fn vram_budget_matches_tide_formula() {
    let free = 80 * 1024 * 1024 * 1024usize;
    let total = 96 * 1024 * 1024 * 1024usize;
    let expert_bytes = 2048 * 512 * 3 * 4;
    let layers = 19usize;
    let (budget, reserve) =
        gpu_expert_budget_from_device_memory(free, total, expert_bytes, layers, 256, 128, 1.5);
    let reserve_gb = (1.5 * 1024f64.powi(3)) as usize;
    let reserve_10 = total / 10;
    assert_eq!(reserve, reserve_gb.max(reserve_10));
    let usable = free.saturating_sub(reserve);
    let expect = (usable / (expert_bytes * layers)).min(128);
    assert_eq!(budget, expect);
}

#[test]
fn refresh_experts_matches_generate_loop() {
    assert!(refresh_experts(true, 2, 2, 2, 5));
    assert!(refresh_experts(true, 2, 2, 2, 0));
    assert!(!refresh_experts(true, 2, 3, 2, 1));
    assert!(refresh_experts(true, 2, 3, 2, 0));
    assert!(refresh_experts(false, 2, 0, 0, 99));
}

#[test]
fn num_transfer_tokens_matches_tide_helper() {
    assert_eq!(num_transfer_tokens_schedule(32, 32), vec![1; 32]);
    assert_eq!(num_transfer_tokens_schedule(10, 3), vec![4, 3, 3]);
}

#[test]
fn enable_predictive_offload_returns_tide_keys() {
    let params = PredictiveOffloadParams::new(128, 256, 19, 2048 * 512 * 3 * 4);
    let (_, info) = enable_predictive_expert_offload(&params).expect("offload");
    assert!(info.enabled);
    assert!(info.gpu_expert_budget_per_layer <= 128);
    assert_eq!(info.num_sparse_moe_layers, 19);
}

#[test]
fn qwen35_runner_exposes_tide_api() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let runner = Qwen35Runner::builder()
        .inline_weights(cfg, weights)
        .device(Device::Cpu)
        .max_seq(8)
        .enable_predictive_expert_offload(2)
        .jump_steps(4)
        .moe_collect_stats(true)
        .build()
        .expect("runner");
    assert!(runner.predictive_offload_enabled());
    assert_eq!(runner.jump_steps(), 4);
    let info = runner.predictive_offload_info().expect("info");
    assert_eq!(info.gpu_expert_budget_per_layer, 2);
    let mo = runner.moe_offload().expect("state");
    assert_eq!(mo.jump_steps, 4);
    assert!(mo.collect_stats);
}

#[test]
fn qwen35_build_moe_offload_jump_steps() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let mo = build_moe_offload(&cfg, &weights, Some(2), None, Some(3), 1.5, false).expect("mo");
    assert!(!mo.should_refresh_forward(1, false));
    assert!(mo.should_refresh_forward(0, false));
    assert!(mo.should_refresh_forward(0, true));
}

#[test]
fn tide_runner_generate_and_offload_api() {
    let cfg = llada2_synth::tiny_cfg();
    let weights = llada2_synth::tiny_weights(&cfg);
    let mut tide = TideRunner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch_seq(1, 8)
        .tide_enable_predictive_expert_offload(2, 1.5, true, 2)
        .build()
        .map(TideRunner::from_llada2)
        .expect("runner");
    assert!(tide.predictive_offload_enabled());
    let info = tide.predictive_offload_info().expect("info");
    assert!(info.enabled);
    assert_eq!(info.jump_steps, 2);
    let dict = tide.get_offload_stats().as_tide_dict();
    for key in ["cpu_tokens", "gpu_tokens", "promotions", "demotions"] {
        assert!(dict.contains_key(key), "missing {key}");
    }
    let gen_cfg = GenerateConfig {
        block_length: 4,
        steps: 2,
        gen_length: 4,
        collect_stats: true,
        ..GenerateConfig::from_model(&cfg)
    };
    assert!((gen_cfg.threshold - 0.9).abs() < f32::EPSILON);
    let (out, stats) = tide.generate(&[1, 2], &gen_cfg).expect("generate");
    assert!(!out.is_empty());
    if !stats.is_empty() {
        let s = &stats[0].offload_stats;
        assert!(s.cpu_tokens > 0 || s.gpu_tokens > 0 || s.promotions > 0);
    }
}

#[test]
fn get_offload_stats_keys_match_pytorch() {
    let stats = TideOffloadStats::default();
    let keys: std::collections::HashSet<_> = stats.as_tide_dict().keys().copied().collect();
    for k in [
        "cpu_tokens",
        "gpu_tokens",
        "cpu_calls",
        "gpu_calls",
        "cpu_compute_time",
        "gpu_compute_time",
        "cpu_tokens_move_time",
        "gpu_tokens_move_time",
        "experts_move_time",
        "promotions",
        "demotions",
    ] {
        assert!(keys.contains(k), "missing {k}");
    }
}
