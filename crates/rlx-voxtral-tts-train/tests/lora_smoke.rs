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

use rlx_runtime::{Device, Session};
use rlx_voxtral_tts::config::TextConfig;
use rlx_voxtral_tts_train::adam::AdamState;
use rlx_voxtral_tts_train::lm_lora_graph::build_lora_train_graph;
use rlx_voxtral_tts_train::weights::WeightStore;

fn tiny_text_config() -> TextConfig {
    TextConfig {
        hidden_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        vocab_size: 1024,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 128,
        rope_theta: 1_000_000.0,
        intermediate_size: Some(128),
    }
}

#[test]
fn lora_graph_q_dim_differs_from_hidden() {
    let text = TextConfig {
        hidden_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 20,
        vocab_size: 1024,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 128,
        rope_theta: 1_000_000.0,
        intermediate_size: Some(128),
    };
    build_lora_train_graph(&text, 16, 4, 1).expect("q_dim != hidden");
}

#[test]
fn lora_one_backward_step_cpu() {
    let text = tiny_text_config();
    let seq = 32;
    let graph = build_lora_train_graph(&text, seq, 4, 1).expect("lora graph");
    let session = Session::new(Device::Cpu);
    let mut backward = session.compile(graph.backward.clone());

    let mut weights = WeightStore::default();
    for slot in &graph.params {
        weights.0.insert(
            slot.name.clone(),
            rlx_voxtral_tts_train::weights::init_lora_param(
                &slot.name,
                4,
                text.hidden_size,
                text.num_attention_heads * text.head_dim,
                text.num_key_value_heads * text.head_dim,
                text.intermediate_size.unwrap_or(text.hidden_size * 3),
            ),
        );
    }
    for (name, data) in &weights.0 {
        backward.set_param(name, data);
    }

    let embed_len = seq * text.hidden_size;
    let inputs = vec![0.01f32; embed_len];
    let target = vec![0.011f32; embed_len];
    let outs = backward.run(&[
        ("inputs_embeds", &inputs),
        ("target_embeds", &target),
        ("d_output", &[1.0f32]),
    ]);
    assert!(!outs.is_empty());

    let mut grads = WeightStore::default();
    for (slot, gout) in graph.params.iter().zip(outs.iter().skip(1)) {
        grads.0.insert(slot.name.clone(), gout.clone());
    }
    let mut adam = AdamState::new_like(&weights);
    adam.step(&mut weights, &grads, 1e-4, 0.9, 0.999, 0.0, 1e-8, 1.0);
}
