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

//! MLX/Metal LoRA distillation on JFK `train_with_codes.jsonl`.

use anyhow::{Result, ensure};
use rlx_qwen3_tts::config::Qwen3TtsConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use std::fs;
use std::time::Instant;

use crate::adam::AdamState;
use crate::compile::compile_train_backward;
use crate::config::JfkLoraConfig;
use crate::dataset::CodesDataset;
use crate::device::resolve_train_device;
use crate::distill_cache::{DistillCache, default_cache_path};
use crate::talker_lora_graph::build_talker_lora_graph;
use crate::weights::{WeightStore, init_lora_param, load_talker_backbone_from_store};

pub fn train_jfk_lora(cfg: &JfkLoraConfig) -> Result<()> {
    let device = resolve_train_device(cfg.device.as_deref())?;
    if cfg.verbose {
        eprintln!(
            "[jfk-lora] device={device:?} speaker={} grad_accum={}",
            cfg.speaker, cfg.grad_accum
        );
    }

    let tts_cfg = Qwen3TtsConfig::from_model_dir(&cfg.model_dir)?;
    let talker = tts_cfg.talker().clone();
    let qwen3 = talker.to_qwen3_config();
    let store = Qwen3TtsWeightStore::open(&cfg.model_dir)?;

    let data = CodesDataset::open(&cfg.train_jsonl)?;
    ensure!(
        !data.records.is_empty(),
        "empty {}",
        cfg.train_jsonl.display()
    );

    let seq = cfg.max_seq.min(256);
    let n_layers = cfg.n_layers.min(talker.num_hidden_layers);
    let graph = build_talker_lora_graph(&qwen3, seq, cfg.rank, n_layers)?;

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

    let base_weights = load_talker_backbone_from_store(&store, &base_names)?;
    let h = talker.hidden_size;
    let q_dim = talker.num_attention_heads * talker.head_dim;
    let kv_dim = talker.num_key_value_heads * talker.head_dim;
    let ffn = talker.intermediate_size;

    let mut lora_weights = WeightStore::default();
    for slot in &graph.params {
        lora_weights.0.insert(
            slot.name.clone(),
            init_lora_param(&slot.name, cfg.rank, h, q_dim, kv_dim, ffn),
        );
    }

    let compile_t0 = Instant::now();
    let (backward_device, mut backward) =
        compile_train_backward(device, graph.backward.clone(), "jfk-lora")?;
    if cfg.verbose {
        eprintln!(
            "[jfk-lora] compile {:.1}s backward={backward_device:?}",
            compile_t0.elapsed().as_secs_f64()
        );
    }

    for (name, data) in &base_weights.0 {
        backward.set_param(name, data);
    }
    for (name, data) in &lora_weights.0 {
        backward.set_param(name, data);
    }
    backward.set_param("rope.cos", &graph.rope_cos);
    backward.set_param("rope.sin", &graph.rope_sin);
    backward.set_param("__zero", &vec![0f32; h]);

    let max_clips = if cfg.max_clips == 0 {
        data.len()
    } else {
        cfg.max_clips.min(data.len())
    };
    let cache_path = cfg
        .cache_path
        .clone()
        .or_else(|| Some(default_cache_path(&cfg.train_jsonl, seq, max_clips, device)));
    let cache = DistillCache::open_or_build(
        &tts_cfg,
        &talker,
        &store,
        &data,
        device,
        seq,
        max_clips,
        cache_path.as_deref(),
        cfg.verbose,
    )?;

    let mut adam = AdamState::new_for_names(lora_weights.0.keys(), &lora_weights);
    fs::create_dir_all(&cfg.out_dir)?;

    let total_steps = cfg.epochs * cfg.steps_per_epoch;
    let grad_accum = cfg.grad_accum.max(1);
    let mut acc_grads = WeightStore::default();
    let d_loss = [1.0f32];
    let train_t0 = Instant::now();

    for step in 0..total_steps {
        if step % grad_accum == 0 {
            acc_grads.0.clear();
        }
        let batch = cache.get(step)?;
        let outs = backward.run(&[
            ("inputs_embeds", batch.inputs.as_slice()),
            ("target_embeds", batch.targets.as_slice()),
            ("d_output", d_loss.as_slice()),
        ]);
        let scale = 1.0 / grad_accum as f32;
        for (slot, gout) in graph.params.iter().zip(outs.iter().skip(1)) {
            let entry = acc_grads
                .0
                .entry(slot.name.clone())
                .or_insert_with(|| vec![0.0; gout.len()]);
            for (a, &g) in entry.iter_mut().zip(gout.iter()) {
                *a += g * scale;
            }
        }

        if (step + 1) % grad_accum == 0 || step + 1 == total_steps {
            let loss = outs
                .first()
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(f32::NAN);
            adam.step(&mut lora_weights, &acc_grads, cfg.lr, 0.9, 0.999, 1e-8, 1.0);
            for (name, data) in &lora_weights.0 {
                backward.set_param(name, data);
            }
            if cfg.verbose && step % 5 == 0 {
                eprintln!("[jfk-lora] step {step}/{total_steps} loss={loss:.6}");
            }
        }
    }

    if cfg.verbose {
        eprintln!(
            "[jfk-lora] train loop {:.1}s",
            train_t0.elapsed().as_secs_f64()
        );
    }

    write_mlx_checkpoint(cfg, &tts_cfg, &cfg.speaker)?;
    if cfg.verbose {
        eprintln!("[jfk-lora] wrote {}", cfg.out_dir.display());
    }
    Ok(())
}

fn write_mlx_checkpoint(
    cfg: &JfkLoraConfig,
    _tts_cfg: &Qwen3TtsConfig,
    speaker: &str,
) -> Result<()> {
    use anyhow::Context;
    let out = &cfg.out_dir;
    fs::create_dir_all(out)?;
    let src_cfg = fs::read_to_string(cfg.model_dir.join("config.json"))?;
    let mut root: serde_json::Value = serde_json::from_str(&src_cfg)?;
    root["tts_model_type"] = serde_json::Value::String("custom_voice".into());
    if let Some(tc) = root
        .get_mut("talker_config")
        .and_then(|v| v.as_object_mut())
    {
        tc.entry("spk_id")
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
            .context("spk_id object")?
            .insert(speaker.to_string(), serde_json::json!(3000));
        tc.entry("spk_is_dialect")
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
            .context("spk_is_dialect object")?
            .insert(speaker.to_string(), serde_json::json!(false));
    }
    fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&root)?,
    )?;
    fs::copy(
        cfg.model_dir.join("model.safetensors"),
        out.join("model.safetensors"),
    )
    .ok();
    for name in [
        "tokenizer.json",
        "generation_config.json",
        "preprocessor_config.json",
    ] {
        let p = cfg.model_dir.join(name);
        if p.is_file() {
            fs::copy(&p, out.join(name)).ok();
        }
    }
    Ok(())
}
