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

//! Phase 2 LoRA distillation training loop.

use anyhow::{Context, Result, ensure};
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use std::fs;
use std::time::Instant;

use crate::adam::AdamState;
use crate::checkpoint::{export_lora_weights, load_lora_weights};
use crate::compile::{backward_cpu_only_from_env, compile_train_backward_opts};
use crate::config::{LoraTrainConfig, cosine_lr, env_flag, lora_distill_layers};
use crate::device::resolve_train_device;
use crate::distill_dataset::DistillDataset;
use crate::lm_lora_graph::build_lora_train_graph;
use crate::weights::{WeightStore, init_lora_param, load_lora_backbone_for_graph};

pub struct LoraTrainResult {
    pub adapters: WeightStore,
    pub best_loss: f64,
}

pub fn train_lora(cfg: &LoraTrainConfig) -> Result<LoraTrainResult> {
    let device = resolve_train_device(cfg.device.as_deref())?;
    let timing = env_flag("LORA_TIMING");
    if cfg.verbose {
        eprintln!("[lora] device={device:?}");
    }
    let model_cfg = VoxtralTtsConfig::from_model_dir(&cfg.model_dir)?;
    fs::create_dir_all(&cfg.out_dir)?;

    let seq = cfg.max_seq_tokens.min(512);
    let n_layers = lora_distill_layers(cfg, &model_cfg.text_config);
    let h = model_cfg.text_config.hidden_size;
    let q_dim = model_cfg.text_config.num_attention_heads * model_cfg.text_config.head_dim;
    let kv_dim = model_cfg.text_config.num_key_value_heads * model_cfg.text_config.head_dim;
    let ffn_dim = model_cfg.text_config.intermediate_size.unwrap_or(h * 3);
    let rank = cfg.rank;
    let grad_accum = cfg.profile.grad_accum.max(1);

    let graph = build_lora_train_graph(&model_cfg.text_config, seq, rank, n_layers)?;
    let force_cpu_backward = device != rlx_runtime::Device::Cpu && backward_cpu_only_from_env();
    if force_cpu_backward && cfg.verbose {
        eprintln!("[lora] RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU=1 — backward on CPU");
    }

    let compile_started = Instant::now();
    let (backward_device, mut backward) =
        compile_train_backward_opts(device, graph.backward.clone(), "lora", force_cpu_backward)?;
    if cfg.verbose {
        eprintln!(
            "[lora] compile {:.1}s backward={backward_device:?} grad_accum={grad_accum}",
            compile_started.elapsed().as_secs_f64()
        );
    }

    let base_names: Vec<String> = graph
        .forward
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Param { name }
                if !name.starts_with("lora.") && name != "__zero" && !name.starts_with("rope.") =>
            {
                Some(name.clone())
            }
            _ => None,
        })
        .collect();
    let base_weights = load_lora_backbone_for_graph(&cfg.model_dir, &base_names)?;
    let mut lora_weights = WeightStore::default();
    for slot in &graph.params {
        lora_weights.0.insert(
            slot.name.clone(),
            init_lora_param(&slot.name, rank, h, q_dim, kv_dim, ffn_dim),
        );
    }
    if let Some(path) = &cfg.resume_weights {
        let loaded = load_lora_weights(path)?;
        lora_weights.merge(&loaded);
        if cfg.verbose {
            eprintln!(
                "[lora] resumed adapters from {} (step {})",
                path.display(),
                cfg.resume_step
            );
        }
    }

    let all_weights = {
        let mut w = base_weights.clone();
        w.merge(&lora_weights);
        w
    };
    for (name, data) in &all_weights.0 {
        backward.set_param(name, data);
    }
    backward.set_param("rope.cos", &graph.rope_cos);
    backward.set_param("rope.sin", &graph.rope_sin);

    let mut distill = DistillDataset::open(
        &cfg.model_dir,
        &cfg.reference_wav_dir,
        cfg.manifest.as_deref(),
        seq,
        cfg.encoder_weights.as_deref(),
        cfg.epochs,
        cfg.steps_per_epoch,
    )?;
    if cfg.verbose {
        eprintln!(
            "[lora] distill samples={} layers={n_layers} rank={rank} seq={seq} encoder_overlay={}",
            distill.len(),
            cfg.encoder_weights
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "none".into())
        );
        if cfg.encoder_weights.is_some() {
            eprintln!("[lora] using trained encoder overlay for voice embeddings");
        }
    }

    let checkpoint_every = checkpoint_every_steps();

    let mut adam = AdamState::new_like(&lora_weights);
    let total = cfg.epochs * cfg.steps_per_epoch;
    let mut best_loss = f64::INFINITY;
    let mut best = lora_weights.clone();
    let embed_len = seq * h;
    let mut inputs = vec![0f32; embed_len];
    let mut target = vec![0f32; embed_len];
    let mut acc_grads = WeightStore::default();

    for epoch in 0..cfg.epochs {
        for step in 0..cfg.steps_per_epoch {
            let global = epoch * cfg.steps_per_epoch + step;
            if global < cfg.resume_step {
                continue;
            }
            let lr = cosine_lr(global, total, cfg.lr, cfg.lr * 0.1);
            let step_started = Instant::now();
            acc_grads.0.clear();
            let mut accum_loss = 0.0f64;
            let mut micros = 0usize;

            for micro in 0..grad_accum {
                let sample_step = global * grad_accum + micro;
                let batch = distill.sample(sample_step)?;
                let active = batch.seq * h;
                ensure!(batch.seq <= seq, "distill seq {} > graph {seq}", batch.seq);
                inputs.fill(0.0);
                target.fill(0.0);
                inputs[..active].copy_from_slice(&batch.inputs[..active]);
                target[..active].copy_from_slice(&batch.targets[..active]);

                let outs = backward.run(&[
                    ("inputs_embeds", &inputs),
                    ("target_embeds", &target),
                    ("d_output", &[1.0f32]),
                ]);
                let loss = outs.first().and_then(|v| v.first()).copied().unwrap_or(0.0) as f64;
                accum_loss += loss;
                micros += 1;

                let scale = 1.0 / grad_accum as f32;
                for (slot, gout) in graph.params.iter().zip(outs.iter().skip(1)) {
                    accumulate_grad(&mut acc_grads, &slot.name, gout, scale);
                }
            }

            let loss = accum_loss / micros as f64;
            adam.step(
                &mut lora_weights,
                &acc_grads,
                lr,
                0.9,
                0.999,
                0.0,
                1e-8,
                cfg.grad_clip,
            );
            for (name, data) in &lora_weights.0 {
                backward.set_param(name, data);
            }
            if loss.is_finite() && loss < best_loss {
                best_loss = loss;
                best = lora_weights.clone();
            }
            if checkpoint_every > 0 && global > 0 && global.is_multiple_of(checkpoint_every) {
                let ckpt = cfg.out_dir.join(format!("lora_step_{global}.safetensors"));
                export_lora_weights(&lora_weights, &ckpt).ok();
            }
            if cfg.verbose && (global.is_multiple_of(10) || timing) {
                eprintln!(
                    "[lora] step {global}/{total} loss={loss:.6} micros={micros} layers={n_layers} {:.1}s/step",
                    step_started.elapsed().as_secs_f64()
                );
            }
        }
    }

    let out = cfg.out_dir.join("lora_adapters.safetensors");
    export_lora_weights(&best, &out).context("write lora safetensors")?;
    Ok(LoraTrainResult {
        adapters: best,
        best_loss,
    })
}

fn accumulate_grad(acc: &mut WeightStore, name: &str, grad: &[f32], scale: f32) {
    match acc.0.get_mut(name) {
        Some(slot) => {
            let n = slot.len().min(grad.len());
            for (a, g) in slot.iter_mut().zip(grad.iter().take(n)) {
                *a += g * scale;
            }
        }
        None => {
            acc.0
                .insert(name.to_string(), grad.iter().map(|g| g * scale).collect());
        }
    }
}

fn checkpoint_every_steps() -> usize {
    std::env::var("CHECKPOINT_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
