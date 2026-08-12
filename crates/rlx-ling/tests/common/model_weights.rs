// Shared tiny-model config + synthetic weights (included, not a test target).
#[allow(dead_code)]
fn tiny_model_config() -> LingConfig {
    let mut c = tiny_model_config_inner(4);
    c.num_hidden_layers = 4;
    c
}

/// 4-layer tiny config; `layer_group_size` picks the attention mix.
#[allow(dead_code)]
fn tiny_model_config_inner(layer_group_size: usize) -> LingConfig {
    LingConfig::from_json_str(&format!(
        r#"{{"vocab_size":32,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":4,
            "num_attention_heads":2,"head_dim":8,"rms_norm_eps":1e-6,"rope_theta":600000.0,
            "num_experts":8,"num_experts_per_tok":2,"num_shared_experts":1,
            "moe_intermediate_size":8,"moe_shared_expert_intermediate_size":8,
            "n_group":2,"topk_group":1,"routed_scaling_factor":2.5,"first_k_dense_replace":1,
            "q_lora_rank":12,"kv_lora_rank":10,"qk_nope_head_dim":8,"qk_rope_head_dim":4,
            "v_head_dim":8,"rope_interleave":true,
            "gated_attention_proj_granularity_type":"head_wise",
            "layer_group_size":{layer_group_size},"short_conv_kernel_size":4,
            "no_kda_lora":true,"kda_safe_gate":true,"kda_lower_bound":-5.0,
            "tie_word_embeddings":false}}"#
    ))
    .expect("parse config")
}

fn model_weights(cfg: &LingConfig) -> WeightMap {
    use rlx_ling::config::AttnKind;
    let h = cfg.hidden_size;
    let hh = cfg.num_attention_heads;
    let proj = cfg.kda_proj_dim();
    let (hd, qk) = (cfg.head_dim, cfg.qk_head_dim());
    let ql = cfg.q_lora_rank.unwrap();
    let (kvl, rope, nope, vd) = (
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
    );
    let mi = cfg.moe_intermediate_size;
    let si = cfg.shared_intermediate_size();
    let e = cfg.num_experts;

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    // Norm gammas centre on 1 and `A_log` spans the reference init's `log U(1,16)`.
    // Uniformly tiny weights would shrink every activation toward zero, where any
    // two backends agree trivially and the comparison proves nothing.
    let mut put_scaled = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                          k: String,
                          shape: Vec<usize>,
                          offset: f32,
                          gain: f32| {
        let n: usize = shape.iter().product();
        seed += 3;
        let v = fill(n, seed).iter().map(|x| offset + gain * x).collect();
        t.insert(k, (v, shape));
    };
    macro_rules! put {
        ($t:expr, $k:expr, $s:expr $(,)?) => {
            put_scaled($t, $k, $s, 0.0, 1.0)
        };
    }
    macro_rules! put_norm {
        ($t:expr, $k:expr, $s:expr $(,)?) => {
            put_scaled($t, $k, $s, 1.0, 1.0)
        };
    }
    /// `A_log = log U(1,16)` ⇒ `exp(A_log) ∈ [1,16]`, which saturates the KDA gate.
    macro_rules! put_a_log {
        ($t:expr, $k:expr, $s:expr $(,)?) => {
            put_scaled($t, $k, $s, 2.834, 2.0)
        };
    }

    put!(&mut t, rlx_ling::EMBED_KEY.into(), vec![cfg.vocab_size, h]);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        put_norm!(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put_norm!(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h]
        );
        let at = format!("{lp}.attention");
        match cfg.attn_kind(i) {
            AttnKind::Mla => {
                put!(&mut t, format!("{at}.q_a_proj.weight"), vec![ql, h]);
                put_norm!(&mut t, format!("{at}.q_a_layernorm.weight"), vec![ql]);
                put!(&mut t, format!("{at}.q_b_proj.weight"), vec![hh * qk, ql]);
                put!(
                    &mut t,
                    format!("{at}.kv_a_proj_with_mqa.weight"),
                    vec![kvl + rope, h],
                );
                put_norm!(&mut t, format!("{at}.kv_a_layernorm.weight"), vec![kvl]);
                put!(
                    &mut t,
                    format!("{at}.kv_b_proj.weight"),
                    vec![hh * (nope + vd), kvl],
                );
                put!(&mut t, format!("{at}.g_proj.weight"), vec![hh, h]);
                put!(&mut t, format!("{at}.dense.weight"), vec![h, hh * vd]);
            }
            AttnKind::Kda => {
                for p in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"] {
                    put!(&mut t, format!("{at}.{p}.weight"), vec![proj, h]);
                }
                for c in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    put!(
                        &mut t,
                        format!("{at}.{c}.weight"),
                        vec![proj, 1, cfg.short_conv_kernel_size],
                    );
                }
                put!(&mut t, format!("{at}.b_proj.weight"), vec![hh, h]);
                put_a_log!(&mut t, format!("{at}.A_log"), vec![hh]);
                put!(&mut t, format!("{at}.dt_bias"), vec![proj]);
                put_norm!(&mut t, format!("{at}.o_norm.weight"), vec![hd]);
                put!(&mut t, format!("{at}.o_proj.weight"), vec![h, proj]);
            }
        }
        let mlp = format!("{lp}.mlp");
        if cfg.is_moe_layer(i) {
            put!(&mut t, format!("{mlp}.gate.weight"), vec![e, h]);
            put!(&mut t, format!("{mlp}.gate.expert_bias"), vec![e]);
            for ei in 0..e {
                let b = format!("{mlp}.experts.{ei}");
                put!(&mut t, format!("{b}.gate_proj.weight"), vec![mi, h]);
                put!(&mut t, format!("{b}.up_proj.weight"), vec![mi, h]);
                put!(&mut t, format!("{b}.down_proj.weight"), vec![h, mi]);
            }
            for (p, n) in [("gate_proj", si), ("up_proj", si)] {
                put!(
                    &mut t,
                    format!("{mlp}.shared_experts.{p}.weight"),
                    vec![n, h],
                );
            }
            put!(
                &mut t,
                format!("{mlp}.shared_experts.down_proj.weight"),
                vec![h, si],
            );
        } else {
            let di = cfg.intermediate_size;
            put!(&mut t, format!("{mlp}.gate_proj.weight"), vec![di, h]);
            put!(&mut t, format!("{mlp}.up_proj.weight"), vec![di, h]);
            put!(&mut t, format!("{mlp}.down_proj.weight"), vec![h, di]);
        }
    }
    put_norm!(&mut t, "model.norm.weight".into(), vec![h]);
    put!(&mut t, "lm_head.weight".into(), vec![cfg.vocab_size, h]);
    WeightMap::from_tensors(t)
}
