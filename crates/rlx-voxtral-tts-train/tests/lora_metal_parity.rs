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

//! CPU vs Metal LoRA forward loss parity on real Voxtral weights (env-gated).

use rlx_runtime::{Device, Session, is_available};
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use rlx_voxtral_tts_train::LoraTrainConfig;
use rlx_voxtral_tts_train::compile::compile_train_backward_opts;
use rlx_voxtral_tts_train::config::lora_distill_layers;
use rlx_voxtral_tts_train::lm_lora_graph::build_lora_train_graph;
use rlx_voxtral_tts_train::weights::{WeightStore, init_lora_param, load_lora_backbone_for_graph};

fn model_dir() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from("/Users/Shared/rlx-models/.cache/voxtral/Voxtral-4B-TTS-2603");
    if p.join("consolidated.safetensors").is_file() {
        Some(p)
    } else {
        None
    }
}

fn run_loss(device: Device, force_cpu_bwd: bool, forward_only: bool) -> f64 {
    let model_dir = model_dir().expect("weights");
    let model_cfg = VoxtralTtsConfig::from_model_dir(&model_dir).expect("cfg");
    let cfg = LoraTrainConfig {
        model_dir: model_dir.clone(),
        reference_wav_dir: model_dir.clone(),
        out_dir: std::env::temp_dir().join("lora-parity"),
        ..LoraTrainConfig::from_cli(
            model_dir.clone(),
            model_dir.clone(),
            std::env::temp_dir().join("lora-parity"),
        )
    };

    let seq = 128;
    let n_layers = lora_distill_layers(&cfg, &model_cfg.text_config);
    eprintln!(
        "[parity] device={device:?} layers={n_layers} force_cpu_bwd={force_cpu_bwd} forward_only={forward_only}"
    );
    let h = model_cfg.text_config.hidden_size;
    let q_dim = model_cfg.text_config.num_attention_heads * model_cfg.text_config.head_dim;
    let kv_dim = model_cfg.text_config.num_key_value_heads * model_cfg.text_config.head_dim;
    let ffn_dim = model_cfg.text_config.intermediate_size.unwrap_or(h * 3);
    let rank = 16;

    let graph = build_lora_train_graph(&model_cfg.text_config, seq, rank, n_layers).expect("graph");
    let mut runner = if forward_only {
        Session::new(device).compile(graph.forward.clone())
    } else {
        let (_, backward) =
            compile_train_backward_opts(device, graph.backward.clone(), "lora", force_cpu_bwd)
                .expect("compile");
        backward
    };

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
    let base = load_lora_backbone_for_graph(&model_dir, &base_names).expect("base");
    let mut lora = WeightStore::default();
    for slot in &graph.params {
        lora.0.insert(
            slot.name.clone(),
            init_lora_param(&slot.name, rank, h, q_dim, kv_dim, ffn_dim),
        );
    }
    let mut all = base;
    all.merge(&lora);
    for (name, data) in &all.0 {
        runner.set_param(name, data);
    }
    runner.set_param("rope.cos", &graph.rope_cos);
    runner.set_param("rope.sin", &graph.rope_sin);

    let active = 64 * h;
    let mut inputs = vec![0f32; seq * h];
    let mut target = vec![0f32; seq * h];
    for (i, v) in inputs.iter_mut().enumerate().take(active) {
        *v = (i as f32 * 0.0013).sin() * 0.02;
    }
    for (i, v) in target.iter_mut().enumerate().take(active) {
        *v = inputs[i] + 0.001;
    }

    let outs = if forward_only {
        runner.run(&[("inputs_embeds", &inputs), ("target_embeds", &target)])
    } else {
        runner.run(&[
            ("inputs_embeds", &inputs),
            ("target_embeds", &target),
            ("d_output", &[1.0f32]),
        ])
    };
    outs.first()
        .and_then(|v| v.first())
        .copied()
        .unwrap_or(f32::NAN) as f64
}

#[test]
fn lora_26layer_loss_cpu_vs_metal() {
    if std::env::var("RLX_VOXTRAL_TTS_TRAIN_GPU_STEP")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip lora_26layer_loss_cpu_vs_metal (set RLX_VOXTRAL_TTS_TRAIN_GPU_STEP=1)");
        return;
    }
    if model_dir().is_none() {
        eprintln!("skip lora_26layer_loss_cpu_vs_metal (weights missing)");
        return;
    }
    if !cfg!(feature = "metal") || !is_available(Device::Metal) {
        eprintln!("skip lora_26layer_loss_cpu_vs_metal (Metal unavailable)");
        return;
    }
    if std::env::var("PRODUCTION").ok().as_deref() != Some("1") {
        eprintln!("skip lora_26layer_loss_cpu_vs_metal (set PRODUCTION=1)");
        return;
    }

    let cpu = run_loss(Device::Cpu, false, false);
    let metal = run_loss(Device::Metal, false, false);
    eprintln!(
        "lora 26-layer loss cpu={cpu:.6} metal={metal:.6} ratio={}",
        metal / cpu.max(1e-12)
    );
    assert!(cpu.is_finite() && cpu > 0.0, "cpu loss {cpu}");
    assert!(metal.is_finite(), "metal loss {metal}");
    let rel = ((metal - cpu) / cpu).abs();
    assert!(
        rel < 0.05 || (metal - cpu).abs() < 0.5,
        "metal loss {metal} vs cpu {cpu} (rel {rel})"
    );
}

#[test]
fn lora_13layer_forward_cpu_vs_metal() {
    if std::env::var("RLX_VOXTRAL_TTS_TRAIN_GPU_STEP")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skip lora_13layer_forward_cpu_vs_metal (set RLX_VOXTRAL_TTS_TRAIN_GPU_STEP=1)");
        return;
    }
    if model_dir().is_none() {
        eprintln!("skip lora_13layer_forward_cpu_vs_metal (weights missing)");
        return;
    }
    if !cfg!(feature = "metal") || !is_available(Device::Metal) {
        eprintln!("skip lora_13layer_forward_cpu_vs_metal (Metal unavailable)");
        return;
    }
    if std::env::var("LORA_N_LAYERS").ok().as_deref() != Some("13") {
        eprintln!("skip lora_13layer_forward_cpu_vs_metal (set LORA_N_LAYERS=13)");
        return;
    }

    let cpu = run_loss(Device::Cpu, false, true);
    let metal = run_loss(Device::Metal, false, true);
    eprintln!(
        "lora 13-layer forward loss cpu={cpu:.6} metal={metal:.6} ratio={}",
        metal / cpu.max(1e-12)
    );
    assert!(cpu.is_finite() && cpu > 0.0, "cpu loss {cpu}");
    assert!(metal.is_finite(), "metal loss {metal}");
    let rel = ((metal - cpu) / cpu).abs();
    assert!(
        rel < 0.05 || (metal - cpu).abs() < 0.5,
        "metal forward loss {metal} vs cpu {cpu} (rel {rel})"
    );
}
