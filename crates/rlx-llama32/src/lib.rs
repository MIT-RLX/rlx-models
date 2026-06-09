// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// LLaMA-3.2 — Meta's small Llama 3.x causal LMs (1B / 3B).
//
// Standard Llama decoder: GQA + RoPE (with Llama 3 scaling) + SwiGLU +
// RMSNorm. No QK-norm. Weight keys follow HuggingFace `LlamaForCausalLM`
// / GGUF `llama` architecture layout.
//
// **Tier-0 API:** [`Llama32Flow`] — fluent builder over `rlx-flow` blocks.

pub mod builder;
pub mod capabilities;
pub mod cli;
pub mod config;
pub mod flow;
pub mod generator;
pub mod rope;
pub mod runner;

pub use builder::{
    build_llama32_decode_graph_sized, build_llama32_decode_graph_sized_ext,
    build_llama32_decode_hir_dynamic_ext, build_llama32_decode_hir_sized,
    build_llama32_decode_hir_sized_ext, build_llama32_graph_sized,
    build_llama32_graph_sized_last_logits, build_llama32_graph_sized_packed,
    build_llama32_prefill_hir_dynamic_ext,
};
pub use capabilities::validate_device;
pub use config::{Llama32Config, Llama32RopeScaling, Llama32RopeType, llama32_cfg_from_gguf};
pub use flow::{
    LLAMA32_PROFILE_FILE, Llama32DecodeOpts, Llama32Flow, Llama32Mode, Llama32PrefillOpts,
    LlamaLayerCtx, build_llama32_decode_built, build_llama32_decode_flow,
    build_llama32_prefill_built, build_llama32_prefill_flow, llama32_profile_near_weights,
};
pub use generator::Llama32Generator;
#[cfg(feature = "tokenizer")]
pub use rlx_qwen35::decode_ids_auto;
pub use rlx_qwen35::{encode_prompt, encode_prompt_auto, resolve_tokenizer_path};
pub use runner::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

#[cfg(feature = "parity-llama")]
pub use rlx_qwen35::llama_oracle;

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn tiny_cfg() -> Llama32Config {
        Llama32Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: None,
            rope_scaling: None,
        }
    }

    fn synthetic_weights(cfg: &Llama32Config) -> WeightMap {
        synthetic_weights_ext(cfg, false)
    }

    fn synthetic_weights_ext(cfg: &Llama32Config, with_lm_head: bool) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];

        t.insert(
            "model.embed_tokens.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            t.insert(format!("{lp}.input_layernorm.weight"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (z(h), vec![h]),
            );
            t.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                (z(q_dim * h), vec![q_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                (z(kv_dim * h), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                (z(kv_dim * h), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                (z(h * q_dim), vec![h, q_dim]),
            );
            t.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                (z(int_dim * h), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (z(int_dim * h), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (z(h * int_dim), vec![h, int_dim]),
            );
        }
        t.insert("model.norm.weight".into(), (z(h), vec![h]));
        if with_lm_head {
            t.insert(
                "lm_head.weight".into(),
                (
                    z(cfg.vocab_size * cfg.hidden_size),
                    vec![cfg.vocab_size, cfg.hidden_size],
                ),
            );
        }
        WeightMap::from_tensors(t)
    }

    #[test]
    fn layer_override_falls_back_to_default() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let built = Llama32Flow::new(&cfg)
            .prefill()
            .batch(1)
            .seq(4)
            .layer(|ctx| ctx.default_stage())
            .build(&mut wm)
            .unwrap();
        let (hir, _params) = built.into_parts().unwrap();
        assert_eq!(hir.outputs.len(), 1);
    }

    #[test]
    fn fluent_prefill_flow_builds() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let built = Llama32Flow::new(&cfg)
            .prefill()
            .batch(1)
            .seq(4)
            .profile_prefill()
            .build(&mut wm)
            .unwrap();
        let (hir, _params) = built.into_parts().unwrap();
        assert_eq!(hir.outputs.len(), 1);
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn fluent_decode_flow_builds_with_kv() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights_ext(&cfg, true);
        let built = Llama32Flow::new(&cfg)
            .decode()
            .batch(1)
            .past(4)
            .profile_decode()
            .build(&mut wm)
            .unwrap();
        let (hir, _params) = built.into_parts().unwrap();
        assert_eq!(hir.outputs.len(), 1 + 2 * cfg.num_hidden_layers);
    }

    #[test]
    fn prefill_graph_builds_and_consumes_every_weight() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let (g, _params) = build_llama32_graph_sized(&cfg, &mut wm, 1, 4, false, false).unwrap();
        assert_eq!(g.outputs.len(), 1);
        let out = g.outputs[0];
        let dims: Vec<usize> = g
            .shape(out)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(dims, vec![1, 4, cfg.hidden_size]);
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn lm_head_produces_logits_shape() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights_ext(&cfg, true);
        let (g, _) = build_llama32_graph_sized(&cfg, &mut wm, 1, 4, true, false).unwrap();
        let out = g.outputs[0];
        let dims: Vec<usize> = g
            .shape(out)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(dims, vec![1, 4, cfg.vocab_size]);
    }

    #[test]
    fn decode_graph_builds_with_kv_outputs() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights_ext(&cfg, true);
        use crate::flow::{Llama32DecodeOpts, build_llama32_decode_flow};
        let opts = Llama32DecodeOpts {
            batch: 1,
            past_seq: 4,
            dynamic_past: false,
            use_custom_mask: false,
            profile: None,
        };
        let (hir, _params) = build_llama32_decode_flow(&cfg, &mut wm, &opts).unwrap();
        assert_eq!(hir.outputs.len(), 1 + 2 * cfg.num_hidden_layers);
    }

    #[test]
    fn profile_toml_loads() {
        use rlx_flow::CompileProfile;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/llama32.rlx.toml");
        let p = CompileProfile::from_toml_path(&path).unwrap();
        assert_eq!(p.fusion.policy, rlx_flow::FusionPolicyKind::Direct);
    }

    #[test]
    fn dynamic_prefill_specializes_per_seq() {
        use rlx_flow::CompileProfile;
        use rlx_ir::DimBinding;
        use rlx_ir::logical_kernel::KernelDispatchConfig;
        use rlx_runtime::Device;
        use rlx_runtime::compile_cache::DynamicDimCompileCache;

        let cfg = tiny_cfg();
        let mut wm = synthetic_weights_ext(&cfg, true);
        let max_seq = 8;
        let mut cache = DynamicDimCompileCache::new(Device::Cpu, 4);

        let (_, template_params) =
            build_llama32_prefill_hir_dynamic_ext(&cfg, &mut wm, 1, max_seq, false)
                .expect("dynamic prefill HIR");

        let mut template_loaded = false;
        for (seq, ids) in [
            (4usize, vec![1.0f32, 2.0, 3.0, 4.0]),
            (6usize, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ] {
            let binding = DimBinding::batch_seq(1, seq);
            let cfg_c = cfg.clone();
            let mut wm_c = synthetic_weights_ext(&cfg_c, true);
            let profile = CompileProfile::llama32_prefill();
            let opts = rlx_core::flow_bridge::compile_options_from_profile(
                &profile,
                Device::Cpu,
                KernelDispatchConfig::default(),
            );
            let compiled = cache
                .get_or_specialize(
                    seq as u64,
                    &binding,
                    || {
                        if template_loaded {
                            panic!("dynamic HIR builder must run only once");
                        }
                        template_loaded = true;
                        build_llama32_prefill_hir_dynamic_ext(&cfg_c, &mut wm_c, 1, max_seq, false)
                            .expect("dynamic prefill HIR")
                            .0
                    },
                    &opts,
                )
                .expect("specialize dynamic prefill");

            for (name, data) in &template_params {
                compiled.set_param(name, data);
            }

            let last_idx = vec![(seq - 1) as f32];
            let outs = compiled.run(&[("input_ids", &ids), ("last_token_idx", &last_idx)]);
            assert_eq!(outs[0].len(), cfg.vocab_size, "seq={seq}");
            for v in &outs[0] {
                assert!(v.is_finite(), "seq={seq} logit={v}");
            }
        }
        assert_eq!(cache.len(), 2);
    }
}
