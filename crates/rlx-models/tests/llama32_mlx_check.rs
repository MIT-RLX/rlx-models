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

// basic test: tiny synthetic llama32 graph on MLX (macOS only).

mod compile_support;

#[cfg(all(target_os = "macos", feature = "mlx"))]
mod mlx_tests {
    use rlx_models::weight_map::WeightMap;
    use rlx_models::{Llama32Config, build_llama32_graph_sized};
    use rlx_runtime::{Device, Session};
    use std::collections::HashMap;

    fn tiny_cfg() -> Llama32Config {
        Llama32Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: None,
            rope_scaling: None,
            rope_style: rlx_ir::RopeStyle::NeoX,
            gguf_arch: None,
            rope_dim: None,
        }
    }

    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
    }

    fn synthetic_weights(cfg: &Llama32Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

        t.insert(
            "model.embed_tokens.weight".into(),
            (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
        );
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            t.insert(
                format!("{lp}.input_layernorm.weight"),
                (vec![1.0; h], vec![h]),
            );
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (vec![1.0; h], vec![h]),
            );
            t.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                (ramp(q_dim * h, 0.01), vec![q_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                (ramp(h * q_dim, 0.01), vec![h, q_dim]),
            );
            t.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                (ramp(int_dim * h, 0.01), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (ramp(int_dim * h, 0.01), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (ramp(h * int_dim, 0.01), vec![h, int_dim]),
            );
        }
        t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
        t.insert(
            "lm_head.weight".into(),
            (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
        );
        WeightMap::from_tensors(t)
    }

    #[test]
    fn llama32_tiny_graph_runs_on_mlx() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let (graph, params) =
            build_llama32_graph_sized(&cfg, &mut wm, 1, 4, true, false).expect("build");
        let mut compiled =
            compile_support::compile_llama32_prefill(Device::Mlx, graph, params.clone());

        let ids = vec![1.0f32, 2.0, 3.0, 4.0];
        let outs = compiled.run(&[("input_ids", &ids), ("last_token_idx", &[3.0f32])]);
        assert!(!outs.is_empty());
        assert!(outs[0].iter().all(|v| v.is_finite()));
    }
}
