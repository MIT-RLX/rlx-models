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

//! Mistral Voxtral — Whisper-style audio encoder + 4× projector + Llama text decoder.
//!
//! Weights: HuggingFace safetensors (`mistralai/Voxtral-Mini-3B-2507`) with
//! `audio_tower.*`, `multi_modal_projector.*`, and `language_model.*` tensors.
//!
//! Audio and text embeddings are fused additively at `[audio_token_id]` placeholders
//! before the Llama trunk runs (see [`embed::fuse_inputs_embeds`]).

pub mod audio;
pub mod cli;
pub mod config;
pub mod embed;
pub mod encoder;
pub mod lm_flow;
pub mod load;
pub mod projector;
pub mod runner;
pub mod weights;

pub use audio::{
    MelSpectrogram, N_FRAMES, N_SAMPLES, SAMPLE_RATE, mel_from_flat, pcm_to_mel,
    pcm_to_mel_and_prompt,
};
pub use config::{VoxtralAudioConfig, VoxtralConfig};
pub use embed::{argmax_token, decode_token_ids, fuse_inputs_embeds, transcription_prompt_ids};
pub use encoder::{
    build_voxtral_encoder_built, build_voxtral_encoder_conv1_built,
    build_voxtral_encoder_conv2_built, build_voxtral_encoder_hir, build_voxtral_encoder_stem_built,
};
pub use lm_flow::{build_voxtral_decode_built, build_voxtral_prefill_built};
pub use load::{
    PREFIX_AUDIO_TOWER, PREFIX_LANGUAGE_MODEL, PREFIX_PROJECTOR, VoxtralWeightStore,
    load_audio_weights, load_language_model_weights, load_projector_weights, load_weight_map_keys,
    load_weight_map_with_prefixes, load_weight_snapshot, resolve_model_dir,
};
pub use projector::{build_voxtral_projector_built, build_voxtral_projector_hir};
pub use runner::{VoxtralRunner, VoxtralRunnerBuilder};
pub use weights::{LanguageModelPrefixLoader, VoxtralWeightPrefix};

pub const FAMILY: &str = "Voxtral";
pub const HF_MODEL_ID_MINI_3B: &str = "mistralai/Voxtral-Mini-3B-2507";

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    fn synth_weights(cfg: &VoxtralConfig) -> WeightMap {
        let ac = &cfg.audio_config;
        let d = ac.d_model;
        let m = ac.num_mel_bins;
        let mlp = ac.intermediate_size;
        let text_h = cfg.text_config.hidden_size;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.01f32; n];

        t.insert(
            VoxtralWeightPrefix::enc_conv1_w().into(),
            (z(d * m * 3), vec![d, m, 3]),
        );
        t.insert(VoxtralWeightPrefix::enc_conv1_b().into(), (z(d), vec![d]));
        t.insert(
            VoxtralWeightPrefix::enc_conv2_w().into(),
            (z(d * d * 3), vec![d, d, 3]),
        );
        t.insert(VoxtralWeightPrefix::enc_conv2_b().into(), (z(d), vec![d]));
        t.insert(
            VoxtralWeightPrefix::enc_embed_positions().into(),
            (
                z(ac.max_source_positions * d),
                vec![ac.max_source_positions, d],
            ),
        );
        t.insert(VoxtralWeightPrefix::enc_ln_post_w().into(), (z(d), vec![d]));
        t.insert(VoxtralWeightPrefix::enc_ln_post_b().into(), (z(d), vec![d]));
        for i in 0..ac.encoder_layers {
            for name in ["self_attn.q_proj", "self_attn.out_proj", "self_attn.v_proj"] {
                t.insert(
                    VoxtralWeightPrefix::enc_layer(i, &format!("{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
                t.insert(
                    VoxtralWeightPrefix::enc_layer(i, &format!("{name}.bias")),
                    (z(d), vec![d]),
                );
            }
            t.insert(
                VoxtralWeightPrefix::enc_layer(i, "self_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(
                VoxtralWeightPrefix::enc_layer(i, "fc1.weight"),
                (z(mlp * d), vec![mlp, d]),
            );
            t.insert(
                VoxtralWeightPrefix::enc_layer(i, "fc1.bias"),
                (z(mlp), vec![mlp]),
            );
            t.insert(
                VoxtralWeightPrefix::enc_layer(i, "fc2.weight"),
                (z(d * mlp), vec![d, mlp]),
            );
            t.insert(
                VoxtralWeightPrefix::enc_layer(i, "fc2.bias"),
                (z(d), vec![d]),
            );
            for n in ["self_attn_layer_norm", "final_layer_norm"] {
                t.insert(
                    VoxtralWeightPrefix::enc_layer(i, &format!("{n}.weight")),
                    (z(d), vec![d]),
                );
                t.insert(
                    VoxtralWeightPrefix::enc_layer(i, &format!("{n}.bias")),
                    (z(d), vec![d]),
                );
            }
        }

        t.insert(
            VoxtralWeightPrefix::projector_linear1().into(),
            (z(text_h * mlp), vec![text_h, mlp]),
        );
        t.insert(
            VoxtralWeightPrefix::projector_linear2().into(),
            (z(text_h * text_h), vec![text_h, text_h]),
        );

        let llama = cfg.llama_config();
        let h = llama.hidden_size;
        let q_dim = llama.q_proj_dim();
        let kv_dim = llama.kv_proj_dim();
        let int_dim = llama.intermediate_size;
        t.insert(
            VoxtralWeightPrefix::lm_embed_tokens().into(),
            (z(llama.vocab_size * h), vec![llama.vocab_size, h]),
        );
        t.insert(VoxtralWeightPrefix::lm_norm().into(), (z(h), vec![h]));
        t.insert(
            VoxtralWeightPrefix::lm_head().into(),
            (z(llama.vocab_size * h), vec![llama.vocab_size, h]),
        );
        for i in 0..llama.num_hidden_layers {
            let lp = format!("language_model.model.layers.{i}");
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

        WeightMap::from_tensors(t)
    }

    #[test]
    fn voxtral_encoder_projector_prefill_runs() {
        let cfg = VoxtralConfig::tiny_synthetic();
        let mel_frames = 8;
        let batch = 1;
        let enc_seq = cfg.audio_config.encoder_seq_len(mel_frames);
        assert!(enc_seq.is_multiple_of(4));

        let mut wm = synth_weights(&cfg);
        let enc =
            build_voxtral_encoder_built(&cfg.audio_config, &mut wm, batch, mel_frames).unwrap();
        let enc_params = enc.params().clone();
        let mut enc_c = rlx_core::flow_util::compile_built(enc, Device::Cpu).unwrap();
        for (n, d) in &enc_params {
            enc_c.set_param(n, d);
        }
        let mel = vec![0.02f32; batch * cfg.audio_config.num_mel_bins * mel_frames];
        let enc_out = enc_c.run(&[("mel", &mel)]).into_iter().next().unwrap();

        let mut wm2 = synth_weights(&cfg);
        let proj = build_voxtral_projector_built(&cfg, &mut wm2, batch, enc_seq).unwrap();
        let proj_params = proj.params().clone();
        let mut proj_c = rlx_core::flow_util::compile_built(proj, Device::Cpu).unwrap();
        for (n, d) in &proj_params {
            proj_c.set_param(n, d);
        }
        let audio_embeds = proj_c
            .run(&[("encoder_hidden", &enc_out)])
            .into_iter()
            .next()
            .unwrap();
        let n_audio = enc_seq / 4;
        assert_eq!(
            audio_embeds.len(),
            batch * n_audio * cfg.text_config.hidden_size
        );

        let prompt: Vec<u32> = vec![cfg.audio_token_id; n_audio];
        let wm_ro = synth_weights(&cfg);
        let fused = fuse_inputs_embeds(&cfg, &wm_ro, &prompt, &audio_embeds).unwrap();

        let mut wm3 = synth_weights(&cfg);
        let prefill =
            build_voxtral_prefill_built(&cfg, &mut wm3, batch, prompt.len(), false, false).unwrap();
        let pre_params = prefill.params().clone();
        let mut pre_c = rlx_core::flow_util::compile_built(prefill, Device::Cpu).unwrap();
        for (n, d) in &pre_params {
            pre_c.set_param(n, d);
        }
        let logits = pre_c
            .run(&[("inputs_embeds", &fused)])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(
            logits.len(),
            batch * prompt.len() * cfg.text_config.vocab_size
        );
    }

    #[test]
    fn voxtral_prefill_last_logits_only_runs() {
        let cfg = VoxtralConfig::tiny_synthetic();
        let batch = 1;
        let mel_frames = 8;
        let enc_seq = cfg.audio_config.encoder_seq_len(mel_frames);
        let n_audio = enc_seq / 4;
        let prompt: Vec<u32> = vec![cfg.audio_token_id; n_audio];
        let mut wm = synth_weights(&cfg);
        let enc =
            build_voxtral_encoder_built(&cfg.audio_config, &mut wm, batch, mel_frames).unwrap();
        let mel = vec![0.02f32; batch * cfg.audio_config.num_mel_bins * mel_frames];
        let mut enc_c = rlx_core::flow_util::compile_built(enc, Device::Cpu).unwrap();
        let enc_out = enc_c.run(&[("mel", &mel)]).into_iter().next().unwrap();
        let mut wm2 = synth_weights(&cfg);
        let proj = build_voxtral_projector_built(&cfg, &mut wm2, batch, enc_seq).unwrap();
        let mut proj_c = rlx_core::flow_util::compile_built(proj, Device::Cpu).unwrap();
        let audio_embeds = proj_c
            .run(&[("encoder_hidden", &enc_out)])
            .into_iter()
            .next()
            .unwrap();
        let wm_ro = synth_weights(&cfg);
        let fused = fuse_inputs_embeds(&cfg, &wm_ro, &prompt, &audio_embeds).unwrap();
        let mut wm3 = synth_weights(&cfg);
        let prefill =
            build_voxtral_prefill_built(&cfg, &mut wm3, batch, prompt.len(), false, true).unwrap();
        let mut pre_c = rlx_core::flow_util::compile_built(prefill, Device::Cpu).unwrap();
        let logits = pre_c
            .run(&[("inputs_embeds", &fused)])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(logits.len(), batch * cfg.text_config.vocab_size);
    }
}
