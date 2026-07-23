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

//! Synthetic Qwen3.5 VLM quick check: vision tower + hidden-state prefill + decode.

mod compile_support;

use rlx_models::{
    Qwen35LayerFfn,
    qwen35::{
        MEDIA_MARKER, MatWeight, MmProjConfig, MmProjWeights, MultimodalPrompt, Qwen35Config,
        Qwen35FullAttnLayer, Qwen35LinearLayer, Qwen35RunnerBuilder, Qwen35TrunkLayer,
        Qwen35VisionEncoder, Qwen35Weights,
    },
};
use rlx_runtime::Device;

fn mat(data: Vec<f32>) -> MatWeight {
    MatWeight::F32(data)
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn tiny_mmproj_cfg() -> MmProjConfig {
    MmProjConfig {
        patch_size: 2,
        n_embd: 16,
        n_head: 2,
        n_layer: 1,
        image_size: 4,
        image_min_pixels: 16,
        image_max_pixels: 256,
        n_merge: 2,
        eps: 1e-6,
        projector_type: "qwen3vl".into(),
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        spatial_merge_size: 2,
        llm_hidden_size: 16,
        n_ff: 32,
        deepstack_layers: vec![],
    }
}

fn tiny_lm_cfg() -> Qwen35Config {
    Qwen35Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 3,
        nextn_predict_layers: 0,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        key_length: 4,
        value_length: 4,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_dim_count: 4,
        rope_dim_sections: vec![],
        mrope_interleaved: false,
        full_attention_interval: 3,
        ssm_conv_kernel: 4,
        ssm_group_count: 2,
        ssm_inner_size: 8,
        ssm_state_size: 4,
        ssm_time_step_rank: 2,
        tie_word_embeddings: true,

        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

fn linear_layer(cfg: &Qwen35Config) -> Qwen35LinearLayer {
    let n_embd = cfg.hidden_size;
    let n_state = cfg.ssm_state_size;
    let n_k_heads = cfg.ssm_group_count;
    let n_v_heads = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k_heads;
    let value_dim = n_state * n_v_heads;
    let conv_channels = key_dim * 2 + value_dim;
    let n_ff = cfg.intermediate_size;
    let k_conv = cfg.ssm_conv_kernel;
    Qwen35LinearLayer {
        attn_norm: vec![1.0; n_embd],
        attn_post_norm: vec![1.0; n_embd],
        attn_qkv: mat(ramp(n_embd * conv_channels, 0.01)),
        attn_gate: mat(ramp(n_embd * value_dim, 0.01)),
        ssm_conv1d: ramp(k_conv * conv_channels, 0.02),
        ssm_dt_bias: ramp(n_v_heads, 0.05),
        ssm_a: vec![-1.0; n_v_heads],
        ssm_beta: mat(ramp(n_embd * n_v_heads, 0.01)),
        ssm_alpha: mat(ramp(n_embd * n_v_heads, 0.01)),
        ssm_norm: vec![1.0; n_state],
        ssm_out: mat(ramp(value_dim * n_embd, 0.01)),
        ffn: Qwen35LayerFfn::Dense {
            gate: mat(ramp(n_embd * n_ff, 0.01)),
            down: mat(ramp(n_ff * n_embd, 0.01)),
            up: mat(ramp(n_embd * n_ff, 0.01)),
        },
    }
}

fn full_attn_layer(cfg: &Qwen35Config) -> Qwen35FullAttnLayer {
    let n_embd = cfg.hidden_size;
    let n_head = cfg.num_attention_heads;
    let n_kv_head = cfg.num_key_value_heads;
    let head_dim = cfg.key_length;
    let q_gate_cols = n_head * head_dim * 2;
    let kv_cols = n_kv_head * head_dim;
    let n_ff = cfg.intermediate_size;
    Qwen35FullAttnLayer {
        attn_norm: vec![1.0; n_embd],
        attn_post_norm: vec![1.0; n_embd],
        attn_q_gate: mat(ramp(n_embd * q_gate_cols, 0.01)),
        attn_k: mat(ramp(n_embd * kv_cols, 0.01)),
        attn_v: mat(ramp(n_embd * kv_cols, 0.01)),
        attn_output: mat(ramp(n_head * head_dim * n_embd, 0.01)),
        attn_q_norm: vec![1.0; head_dim],
        attn_k_norm: vec![1.0; head_dim],
        ffn: Qwen35LayerFfn::Dense {
            gate: mat(ramp(n_embd * n_ff, 0.01)),
            down: mat(ramp(n_ff * n_embd, 0.01)),
            up: mat(ramp(n_embd * n_ff, 0.01)),
        },
    }
}

fn synth_lm_weights(cfg: &Qwen35Config) -> Qwen35Weights {
    let n_embd = cfg.hidden_size;
    let n_vocab = cfg.vocab_size;
    let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
    let interval = cfg.full_attention_interval.max(1);
    let mut trunk = Vec::new();
    for il in 0..n_main {
        let is_full = ((il + 1) % interval) == 0;
        trunk.push(if is_full {
            Qwen35TrunkLayer::FullAttn(full_attn_layer(cfg))
        } else {
            Qwen35TrunkLayer::Linear(linear_layer(cfg))
        });
    }
    Qwen35Weights {
        token_embd: std::sync::Arc::from(ramp(n_vocab * n_embd, 0.001)),
        output_norm: vec![1.0; n_embd],
        output: None,
        token_embd_lm: None,
        trunk_layers: trunk,
        mtp_layers: vec![],
    }
}

fn fake_tokenizer(text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(text.bytes().map(|b| (b as u32 % 31 + 1).max(1)).collect())
}

#[test]
fn qwen35_vlm_hidden_prefill_and_decode_quick_check() {
    let mmcfg = tiny_mmproj_cfg();
    let mmweights = MmProjWeights::synthetic(&mmcfg);
    let lmcfg = tiny_lm_cfg();
    let lmweights = synth_lm_weights(&lmcfg);

    let mut runner = Qwen35RunnerBuilder::default()
        .inline_weights(lmcfg.clone(), lmweights.clone())
        .inline_mmproj(mmcfg.clone(), mmweights.clone())
        .device(Device::Cpu)
        .max_seq(64)
        .last_logits_only(true)
        .build()
        .expect("vlm runner");

    assert!(runner.has_vision());

    let img_w = 4;
    let img_h = 4;
    let rgb: Vec<u8> = (0..(img_w * img_h * 3)).map(|i| (i % 251) as u8).collect();
    let mut enc =
        Qwen35VisionEncoder::from_parts(mmcfg, mmweights, img_w, img_h).expect("vision encoder");
    let vision = enc.encode_rgb(&rgb, img_w, img_h).expect("encode");

    let prompt = format!("before{MEDIA_MARKER}after");
    let mm = MultimodalPrompt {
        prompt: &prompt,
        vision: &vision,
    };
    let prefill = mm
        .assemble(fake_tokenizer, &lmweights.token_embd, lmcfg.hidden_size, 0)
        .expect("assemble");
    assert!(prefill.mrope_sections.len() == prefill.seq.len());

    let seed = runner
        .prefill_from_assembled(prefill)
        .expect("hidden prefill");
    assert_eq!(seed.trunk_logits.len(), lmcfg.vocab_size);
    assert!(seed.trunk_logits.iter().all(|v| v.is_finite()));

    let step = runner.decode_get_logits(3).expect("decode step");
    assert_eq!(step.len(), lmcfg.vocab_size);
    assert!(step.iter().all(|v| v.is_finite()));
}
