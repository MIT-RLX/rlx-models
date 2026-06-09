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

//! LocateAnything synthetic backend benchmark (no real weights).
//!
//! Measures compile + one warm run for:
//! - MoonViT encoder (tiny config)
//! - projector
//! - LM prefill
//! - LM decode (single step)
//!
//! ```bash
//! cargo run -p rlx-locateanything --example backend_synth_bench --release --features all-backends
//! ```

use anyhow::Result;
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::BuiltModel;
use rlx_locateanything::compile_support::{lm_decode_compile_options, metal_lm_compile_guard};
use rlx_locateanything::config::{LocateAnythingConfig, LocateAnythingTextConfig, MoonVitConfig};
use rlx_locateanything::lm_flow::{
    build_locateanything_decode_built_ext, build_locateanything_prefill_built,
};
use rlx_locateanything::moonvit::MoonVitCache;
use rlx_locateanything::preprocess::preprocess_image;
use rlx_locateanything::projector::build_projector_built;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

fn device_label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        _ => "other",
    }
}

fn all_backend_devices() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(feature = "metal")]
        Device::Metal,
        #[cfg(feature = "mlx")]
        Device::Mlx,
        #[cfg(feature = "cuda")]
        Device::Cuda,
        #[cfg(feature = "rocm")]
        Device::Rocm,
        #[cfg(feature = "gpu")]
        Device::Gpu,
        #[cfg(feature = "vulkan")]
        Device::Vulkan,
    ]
}

fn tiny_cfg() -> LocateAnythingConfig {
    LocateAnythingConfig {
        model_type: "locateanything".into(),
        image_token_index: 99,
        box_start_token_id: 1,
        box_end_token_id: 2,
        coord_start_token_id: 3,
        coord_end_token_id: 4,
        ref_start_token_id: 5,
        ref_end_token_id: 6,
        none_token_id: 7,
        mlp_connector_layers: 2,
        mlp_checkpoint: false,
        text_config: LocateAnythingTextConfig {
            vocab_size: 32,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            block_size: 6,
            causal_attn: true,
            bos_token_id: 0,
            eos_token_id: 1,
            null_token_id: None,
            switch_token_id: None,
            text_mask_token_id: Some(31),
        },
        vision_config: MoonVitConfig {
            model_type: "moonvit".into(),
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_hidden_layers: 1,
            patch_size: 14,
            merge_kernel_size: [2, 2],
            init_pos_emb_height: 4,
            init_pos_emb_width: 4,
        },
        preprocessor: Default::default(),
    }
}

fn z(n: usize) -> Vec<f32> {
    vec![0.02f32; n]
}

fn synthetic_vision_weights(cfg: &LocateAnythingConfig) -> WeightMap {
    let h = cfg.vision_config.hidden_size;
    let ps = cfg.vision_config.patch_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "vision_model.patch_embed.proj.weight".into(),
        (z(h * 3 * ps * ps), vec![h, 3, ps, ps]),
    );
    t.insert("vision_model.patch_embed.proj.bias".into(), (z(h), vec![h]));
    t.insert(
        "vision_model.patch_embed.pos_emb.weight".into(),
        (z(4 * 4 * h), vec![4, 4, h]),
    );
    let lp = "vision_model.encoder.blocks.0";
    t.insert(format!("{lp}.norm0.weight"), (z(h), vec![h]));
    t.insert(format!("{lp}.norm0.bias"), (z(h), vec![h]));
    t.insert(format!("{lp}.wqkv.weight"), (z(h * 3 * h), vec![h * 3, h]));
    t.insert(format!("{lp}.wqkv.bias"), (z(h * 3), vec![h * 3]));
    t.insert(format!("{lp}.wo.weight"), (z(h * h), vec![h, h]));
    t.insert(format!("{lp}.wo.bias"), (z(h), vec![h]));
    t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
    t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
    t.insert(format!("{lp}.mlp.fc0.weight"), (z(32 * h), vec![32, h]));
    t.insert(format!("{lp}.mlp.fc0.bias"), (z(32), vec![32]));
    t.insert(format!("{lp}.mlp.fc1.weight"), (z(h * 32), vec![h, 32]));
    t.insert(format!("{lp}.mlp.fc1.bias"), (z(h), vec![h]));
    t.insert(
        "vision_model.encoder.final_layernorm.weight".into(),
        (z(h), vec![h]),
    );
    t.insert(
        "vision_model.encoder.final_layernorm.bias".into(),
        (z(h), vec![h]),
    );
    WeightMap::from_tensors(t)
}

fn synthetic_projector_weights(cfg: &LocateAnythingConfig) -> WeightMap {
    let in_dim = cfg.projector_input_dim();
    let out_dim = cfg.text_config.hidden_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("mlp1.0.weight".into(), (z(in_dim), vec![in_dim]));
    t.insert("mlp1.0.bias".into(), (z(in_dim), vec![in_dim]));
    t.insert(
        "mlp1.1.weight".into(),
        (z(out_dim * in_dim), vec![out_dim, in_dim]),
    );
    t.insert("mlp1.1.bias".into(), (z(out_dim), vec![out_dim]));
    t.insert(
        "mlp1.3.weight".into(),
        (z(out_dim * out_dim), vec![out_dim, out_dim]),
    );
    t.insert("mlp1.3.bias".into(), (z(out_dim), vec![out_dim]));
    WeightMap::from_tensors(t)
}

fn synthetic_lm_weights(cfg: &LocateAnythingConfig) -> WeightMap {
    let tc = &cfg.text_config;
    let h = tc.hidden_size;
    let q_dim = tc.num_attention_heads * tc.head_dim();
    let kv_dim = tc.num_key_value_heads * tc.head_dim();
    let int_dim = tc.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "language_model.model.embed_tokens.weight".into(),
        (z(tc.vocab_size * h), vec![tc.vocab_size, h]),
    );
    for i in 0..tc.num_hidden_layers {
        let lp = format!("language_model.model.layers.{i}");
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
            (z(q_dim * h), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.bias"),
            (z(q_dim), vec![q_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (z(kv_dim * h), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.bias"),
            (z(kv_dim), vec![kv_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (z(kv_dim * h), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.bias"),
            (z(kv_dim), vec![kv_dim]),
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
        t.insert(
            format!("{lp}.self_attn.q_norm.weight"),
            (vec![1.0; tc.head_dim()], vec![tc.head_dim()]),
        );
        t.insert(
            format!("{lp}.self_attn.k_norm.weight"),
            (vec![1.0; tc.head_dim()], vec![tc.head_dim()]),
        );
    }
    t.insert(
        "language_model.model.norm.weight".into(),
        (vec![1.0; h], vec![h]),
    );
    t.insert(
        "language_model.lm_head.weight".into(),
        (z(tc.vocab_size * h), vec![tc.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

fn compile_decode_built(built: BuiltModel, device: Device) -> Result<rlx_runtime::CompiledGraph> {
    let options = lm_decode_compile_options(device);
    let (graph, params) = graph_from_built(built)?;
    metal_lm_compile_guard(device, || {
        let mut compiled = Session::new(device).compile_with(graph, &options);
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }
        Ok(compiled)
    })
}

fn run_one(device: Device) -> Result<()> {
    let label = device_label(device);
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        println!("  [{label}] skip (unavailable)");
        return Ok(());
    }

    let cfg = tiny_cfg();
    let img = image::RgbImage::new(28, 28);
    let prep = preprocess_image(&image::DynamicImage::ImageRgb8(img), &cfg)?;

    let t0 = Instant::now();
    let mut wm_v = synthetic_vision_weights(&cfg);
    let mut vit_cache = MoonVitCache::default();
    let merged = vit_cache.encode(&cfg.vision_config, Some(&mut wm_v), &prep, device)?;
    let vit_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let mut wm_p = synthetic_projector_weights(&cfg);
    let n_tokens = merged.len() / cfg.projector_input_dim();
    let built_p = build_projector_built(&cfg, &mut wm_p, 1, n_tokens)?;
    let params_p = built_p.model.params().clone();
    let mut proj = compile_built(built_p.model, device)?;
    for (n, d) in &params_p {
        proj.set_param(n, d);
    }
    let proj_out = proj
        .run(&[("vision", merged.as_slice())])
        .into_iter()
        .next()
        .expect("proj out");
    let proj_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let seq = 8usize;
    let t0 = Instant::now();
    let mut wm_lm = synthetic_lm_weights(&cfg);
    let pre_built = build_locateanything_prefill_built(&cfg, &mut wm_lm, 1, seq, true, true)?;
    let params_pre = pre_built.params().clone();
    let mut pre = compile_built(pre_built, device)?;
    for (n, d) in &params_pre {
        pre.set_param(n, d);
    }
    let h = cfg.text_config.hidden_size;
    let inputs_embeds = vec![0.01f32; seq * h];
    let pre_out = pre.run(&[("inputs_embeds", inputs_embeds.as_slice())]);
    let _logits = &pre_out[0];
    let pre_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let mut wm_dec = synthetic_lm_weights(&cfg);
    let dec_built = build_locateanything_decode_built_ext(&cfg, &mut wm_dec, 1, 8, false, false)?;
    let mut dec = compile_decode_built(dec_built, device)?;
    let vocab = cfg.text_config.vocab_size;
    let layers = cfg.text_config.num_hidden_layers;
    let kv_dim = cfg.text_config.num_key_value_heads * cfg.text_config.head_dim();
    // past_len=0
    let token = [1f32];
    let (cos, sin) =
        rlx_locateanything::lm_flow::compute_rope_chunk(&cfg.text_config.to_qwen3_config(), 0, 1);
    let mut past_names = Vec::with_capacity(2 * layers);
    let mut past_bufs = Vec::with_capacity(2 * layers);
    for i in 0..layers {
        past_names.push(format!("past_k_{i}"));
        past_bufs.push(vec![0f32; 8 * kv_dim]);
        past_names.push(format!("past_v_{i}"));
        past_bufs.push(vec![0f32; 8 * kv_dim]);
    }
    let mut run_in: Vec<(&str, &[f32])> = vec![
        ("input_ids", &token),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ];
    for (n, b) in past_names.iter().zip(past_bufs.iter()) {
        run_in.push((n.as_str(), b.as_slice()));
    }
    // NOTE: we only care it runs; avoid heavy allocations by not padding huge.
    let step = dec.run(&run_in);
    assert_eq!(step[0].len(), vocab);
    let dec_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  [{label}] vit={vit_ms:.1}ms proj={proj_ms:.1}ms prefill={pre_ms:.1}ms decode_compile+run={dec_ms:.1}ms (proj_out={}f32)",
        proj_out.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("# locateanything backend_synth_bench");
    let mut failed = Vec::new();
    for device in all_backend_devices() {
        let label = device_label(device).to_string();
        let outcome = catch_unwind(AssertUnwindSafe(|| run_one(device)));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("  [{label}] ERROR: {e:#}");
                failed.push(label);
            }
            Err(_) => {
                eprintln!("  [{label}] ERROR: panic");
                failed.push(label);
            }
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        anyhow::bail!("failed backends: {}", failed.join(", "))
    }
}
