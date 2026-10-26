// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! An exported Llama 4 package, opened by rlx alone, greedy-decodes what
//! `Llama4Runner` does: a three-layer MoE model (RoPE and NoPE layers) with
//! random weights, written as a checkpoint and exported.
#![cfg(feature = "export")]

use std::collections::HashMap;

use rlx_core::export::check::write_checkpoint;
use rlx_llama4::export::{ExportOptions, export_rlxp};
use rlx_llama4::runner::Llama4Runner;
use rlx_runtime::Device;
use serde_json::json;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.8
        })
        .collect()
}

#[test]
fn an_exported_llama4_decodes_what_the_runner_does() {
    let (vocab, h, inter, layers, heads, kv, hd, experts) = (64, 16, 32, 3, 4, 2, 4, 4);
    let config = json!({
        "model_type": "llama4_text", "vocab_size": vocab, "hidden_size": h,
        "intermediate_size": inter, "intermediate_size_mlp": inter,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": hd, "num_local_experts": experts,
        "num_experts_per_tok": 1, "no_rope_layer_interval": 2,
        "interleave_moe_layer_step": 1, "max_position_embeddings": 4096,
        "eos_token_id": 2,
    });
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put = |k: String, shape: Vec<usize>| {
        seed += 5;
        let n = shape.iter().product();
        t.insert(k, (fill(n, seed), shape));
    };
    put("model.embed_tokens.weight".into(), vec![vocab, h]);
    for i in 0..layers {
        let lp = format!("model.layers.{i}");
        put(format!("{lp}.input_layernorm.weight"), vec![h]);
        put(format!("{lp}.post_attention_layernorm.weight"), vec![h]);
        put(format!("{lp}.self_attn.q_proj.weight"), vec![heads * hd, h]);
        put(format!("{lp}.self_attn.k_proj.weight"), vec![kv * hd, h]);
        put(format!("{lp}.self_attn.v_proj.weight"), vec![kv * hd, h]);
        put(format!("{lp}.self_attn.o_proj.weight"), vec![h, heads * hd]);
        let ff = format!("{lp}.feed_forward");
        put(format!("{ff}.router.weight"), vec![experts, h]);
        put(
            format!("{ff}.experts.gate_up_proj"),
            vec![experts, h, 2 * inter],
        );
        put(format!("{ff}.experts.down_proj"), vec![experts, inter, h]);
        put(
            format!("{ff}.shared_expert.gate_proj.weight"),
            vec![inter, h],
        );
        put(format!("{ff}.shared_expert.up_proj.weight"), vec![inter, h]);
        put(
            format!("{ff}.shared_expert.down_proj.weight"),
            vec![h, inter],
        );
    }
    put("model.norm.weight".into(), vec![h]);
    put("lm_head.weight".into(), vec![vocab, h]);

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("llama4");
    write_checkpoint(&dir, &config, &t, vocab).unwrap();
    let out = tmp.path().join("llama4.rlxp");
    let opts = ExportOptions {
        uppers: vec![16, 32],
        ..ExportOptions::default()
    };
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    assert_eq!(report.functions, ["forward"]);
    // Every checkpoint tensor is read from the checkpoint, not copied.
    assert_eq!(report.referenced, t.len(), "{report}");

    rlx_text::install_model_tokenizer();
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let lm = m.lm().unwrap();
    let mut runner = Llama4Runner::from_checkpoint(&dir, Device::Cpu).unwrap();
    for prompt in [vec![5u32, 9, 13, 17, 21, 7, 3, 11], vec![40, 41]] {
        let n = 32 - prompt.len();
        let want = runner.generate(&prompt, n, Some(2), &mut |_| true).unwrap();
        let got = lm.generate(&prompt, n, &mut |t| t != 2).unwrap();
        assert!(want.len() > 4, "the runner stopped at once: {want:?}");
        assert_eq!(got, want, "prompt {prompt:?}");
    }
    // Past the longest size the package says so.
    let err = lm.predict_logits(&[3; 33]).unwrap_err().to_string();
    assert!(err.contains("the context is full"), "{err}");
}
