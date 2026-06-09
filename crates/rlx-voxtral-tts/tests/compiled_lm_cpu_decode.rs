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

//! CPU compile + run for TTS LM decode graph (validates shapes without Metal).

use rlx_core::flow_util::compile_built;
use rlx_runtime::Device;
use rlx_voxtral_tts::lm_flow::build_tts_backbone_decode_built;
use rlx_voxtral_tts::{VoxtralTtsConfig, VoxtralTtsWeightStore};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let env = std::env::var("RLX_VOXTRAL_TTS_DIR").ok().map(PathBuf::from);
    let mut candidates = env.into_iter().chain([
        PathBuf::from(".cache/voxtral/Voxtral-4B-TTS-2603"),
        PathBuf::from("../../.cache/voxtral/Voxtral-4B-TTS-2603"),
    ]);
    candidates.find(|p| p.join("consolidated.safetensors").is_file())
}

#[test]
fn cpu_decode_graph_compiles_and_runs_one_step() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_VOXTRAL_TTS_DIR");
        return;
    };
    let cfg = VoxtralTtsConfig::from_model_dir(&dir).expect("config");
    let store = VoxtralTtsWeightStore::open(&dir).expect("store");
    let hidden = cfg.text_config.hidden_size;
    let past_len = 4usize;
    let kv_dim = cfg.text_config.num_key_value_heads * cfg.text_config.head_dim;
    let llama = cfg.text_config.llama_config();
    let half = llama.head_dim() / 2;
    let n_layers = cfg.text_config.num_hidden_layers;

    let mut wm = store.load_backbone().expect("wm");
    let built = build_tts_backbone_decode_built(&cfg.text_config, &mut wm, 1, past_len)
        .expect("decode built");
    let params = built.params().clone();
    let mut compiled = compile_built(built, Device::Cpu).expect("cpu compile");
    for (name, data) in &params {
        compiled.set_param(name, data);
    }

    let embed = vec![0.01f32; hidden];
    let cos = vec![1.0f32; half];
    let sin = vec![0.0f32; half];
    let past_k = vec![0.0f32; past_len * kv_dim];
    let past_v = vec![0.0f32; past_len * kv_dim];

    let k_names: Vec<String> = (0..n_layers).map(|i| format!("past_k_{i}")).collect();
    let v_names: Vec<String> = (0..n_layers).map(|i| format!("past_v_{i}")).collect();
    let mut input_refs: Vec<(&str, &[f32])> = Vec::with_capacity(3 + 2 * n_layers);
    input_refs.push(("inputs_embeds", &embed));
    input_refs.push(("rope_cos", &cos));
    input_refs.push(("rope_sin", &sin));
    for i in 0..n_layers {
        input_refs.push((&k_names[i], &past_k));
        input_refs.push((&v_names[i], &past_v));
    }

    let outputs = compiled.run(&input_refs);
    assert_eq!(outputs.len(), 1 + 2 * n_layers);
    assert_eq!(outputs[0].len(), hidden);
    assert!(outputs[0].iter().all(|v| v.is_finite()));
    eprintln!(
        "cpu decode graph ran: hidden len {} kv tensors {}",
        outputs[0].len(),
        outputs.len() - 1
    );
}
