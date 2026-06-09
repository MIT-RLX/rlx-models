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

//! Synthetic LLaDA2 forward + TIDE offload quick check.

use rlx_models::llada2::{LLaDA2Runner, synth};
use rlx_models::tide::{BlockDenoiseConfig, refresh_experts};
use rlx_runtime::Device;

#[test]
fn llada2_synth_forward_compiles_and_runs() {
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let batch = 1;
    let seq = 8;
    let mut runner = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch_seq(batch, seq)
        .build()
        .expect("runner");

    let ids: Vec<f32> = (0..seq).map(|i| (i % cfg.vocab_size) as f32).collect();
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let mask = vec![0f32; batch * seq * seq];
    let logits = runner.forward_logits(&ids, &pos, &mask).expect("forward");
    assert_eq!(logits.len(), batch * seq * cfg.vocab_size);
}

#[test]
fn llada2_moe_offload_wiring() {
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let runner = LLaDA2Runner::builder()
        .inline_weights(cfg, weights)
        .device(Device::Cpu)
        .batch_seq(1, 8)
        .enable_predictive_expert_offload(2)
        .jump_steps(2)
        .moe_collect_stats(true)
        .build()
        .expect("runner");
    assert!(runner.predictive_offload_enabled());
    assert_eq!(runner.jump_steps(), 2);
}

#[test]
fn llada2_gate_group_limited_topk() {
    use rlx_models::llada2::gate::group_limited_topk;
    let scores = vec![
        0.1, 0.9, 0.2, 0.8, //
        0.5, 0.5, 0.5, 0.5,
    ];
    let (probs, idx) = group_limited_topk(&scores, 2, 4, 2, 1, 2);
    assert_eq!(idx.len(), 4);
    assert_eq!(probs.len(), 4);
    assert!(probs.iter().all(|p| *p > 0.0));
}

#[test]
fn refresh_experts_block_diffusion() {
    assert!(refresh_experts(true, 2, 1, 0, 0));
    assert!(!refresh_experts(true, 2, 1, 0, 1));
}

#[test]
fn block_diffusion_mask_blocks_future_blocks() {
    use rlx_models::llada2::block_diffusion_attention_mask;
    let m = block_diffusion_attention_mask(1, 8, 4);
    assert!(m[5].is_infinite() && m[5] < 0.0);
    assert_eq!(m[32], 0.0);
}

#[test]
fn moe_expert_store_apply_param_names() {
    use rlx_models::llada2::moe_store::{apply_moe_store_to_compiled, build_moe_expert_store};
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let store = build_moe_expert_store(&cfg, &weights).expect("store");
    let (graph, params) =
        rlx_models::llada2::build_llada2_forward_graph(&cfg, &weights, 1, 4).expect("graph");
    let built = rlx_flow::BuiltModel::from_graph(graph, params).expect("built");
    let mut compiled = rlx_models::flow_util::compile_built_cpu(built).expect("compile");
    apply_moe_store_to_compiled(&store, &mut compiled);
    assert!(!store.layers[0].gate.as_slice().is_empty());
}

#[test]
fn llada2_block_denoise_one_step() {
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let mut runner = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch_seq(1, 8)
        .build()
        .expect("runner");
    let mut loop_ = runner.block_denoise_loop(BlockDenoiseConfig {
        block_length: 4,
        steps: 2,
        gen_length: 4,
        mask_id: cfg.mask_token_id,
        eos_id: cfg.eos_token_id,
        ..Default::default()
    });
    let (out, _) = loop_.generate(&[1, 2]).expect("denoise");
    assert!(!out.is_empty());
}
