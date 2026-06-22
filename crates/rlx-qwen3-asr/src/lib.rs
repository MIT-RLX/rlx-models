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

//! Qwen3-ASR — Alibaba's all-in-one speech-recognition model for RLX.
//!
//! A Qwen3-Omni audio encoder (chunked Conv2d stem + windowed transformer +
//! 2-layer adapter) feeds projected audio embeddings into a tied-head Qwen3
//! decoder. Audio and text fuse at `<|audio_pad|>` placeholders before the
//! decoder prefill, then the transcription is generated autoregressively.
//!
//! Weights: HF safetensors (`Qwen/Qwen3-ASR-0.6B`) with `thinker.audio_tower.*`
//! and `thinker.model.*` / `thinker.lm_head.weight` tensors.

pub mod audio;
pub mod cli;
pub mod config;
pub mod embed;
pub mod encoder;
pub mod lm_flow;
pub mod load;
pub mod runner;
pub mod weights;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use audio::{AudioGeometry, MelSpectrogram, pcm_to_log_mel};
pub use config::{AudioEncoderConfig, Qwen3AsrConfig};
pub use embed::{argmax_token, count_audio_placeholders, fuse_inputs_embeds};
pub use encoder::build_encoder_built;
pub use lm_flow::{
    build_asr_decode_built, build_asr_decode_built_opts, build_asr_prefill_built, rope_slice,
    rope_tables,
};
pub use load::{AsrWeightStore, resolve_model_dir};
pub use runner::{AsrRunner, AsrRunnerBuilder, StreamChunk};
pub use weights::{AsrWeightPrefix, LanguageModelPrefixLoader};

#[cfg(feature = "tokenizer")]
pub use tokenizer::AsrTokenizer;

pub const FAMILY: &str = "Qwen3-ASR";
pub const HF_MODEL_ID_0_6B: &str = "Qwen/Qwen3-ASR-0.6B";
pub const HF_MODEL_ID_1_7B: &str = "Qwen/Qwen3-ASR-1.7B";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::AsrWeightPrefix;
    use rlx_core::flow_util::compile_built;
    use rlx_core::weight_map::WeightMap;
    use rlx_qwen3::Qwen3Config;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    fn tiny_audio_cfg() -> AudioEncoderConfig {
        serde_json::from_str(
            r#"{"d_model":8,"num_mel_bins":16,"num_hidden_layers":2,
                "encoder_attention_heads":2,"encoder_ffn_dim":16,
                "downsample_hidden_size":4,"output_dim":8,
                "max_source_positions":64,"n_window":2,"n_window_infer":16}"#,
        )
        .unwrap()
    }

    fn synth_audio_weights(cfg: &AudioEncoderConfig, freq_pc: usize) -> WeightMap {
        let d = cfg.d_model;
        let ds = cfg.downsample_hidden_size;
        let ffn = cfg.encoder_ffn_dim;
        let fan = ds * freq_pc;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.02f32; n];

        t.insert(
            AsrWeightPrefix::CONV2D1_W.into(),
            (z(ds * 9), vec![ds, 1, 3, 3]),
        );
        t.insert(AsrWeightPrefix::CONV2D1_B.into(), (z(ds), vec![ds]));
        t.insert(
            AsrWeightPrefix::CONV2D2_W.into(),
            (z(ds * ds * 9), vec![ds, ds, 3, 3]),
        );
        t.insert(AsrWeightPrefix::CONV2D2_B.into(), (z(ds), vec![ds]));
        t.insert(
            AsrWeightPrefix::CONV2D3_W.into(),
            (z(ds * ds * 9), vec![ds, ds, 3, 3]),
        );
        t.insert(AsrWeightPrefix::CONV2D3_B.into(), (z(ds), vec![ds]));
        t.insert(
            AsrWeightPrefix::CONV_OUT_W.into(),
            (z(d * fan), vec![d, fan]),
        );
        t.insert(AsrWeightPrefix::LN_POST_W.into(), (z(d), vec![d]));
        t.insert(AsrWeightPrefix::LN_POST_B.into(), (z(d), vec![d]));
        t.insert(AsrWeightPrefix::PROJ1_W.into(), (z(d * d), vec![d, d]));
        t.insert(AsrWeightPrefix::PROJ1_B.into(), (z(d), vec![d]));
        t.insert(
            AsrWeightPrefix::PROJ2_W.into(),
            (z(cfg.output_dim * d), vec![cfg.output_dim, d]),
        );
        t.insert(
            AsrWeightPrefix::PROJ2_B.into(),
            (z(cfg.output_dim), vec![cfg.output_dim]),
        );

        for i in 0..cfg.num_hidden_layers {
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                t.insert(
                    AsrWeightPrefix::audio_layer(i, &format!("self_attn.{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
                t.insert(
                    AsrWeightPrefix::audio_layer(i, &format!("self_attn.{name}.bias")),
                    (z(d), vec![d]),
                );
            }
            for n in ["self_attn_layer_norm", "final_layer_norm"] {
                t.insert(
                    AsrWeightPrefix::audio_layer(i, &format!("{n}.weight")),
                    (z(d), vec![d]),
                );
                t.insert(
                    AsrWeightPrefix::audio_layer(i, &format!("{n}.bias")),
                    (z(d), vec![d]),
                );
            }
            t.insert(
                AsrWeightPrefix::audio_layer(i, "fc1.weight"),
                (z(ffn * d), vec![ffn, d]),
            );
            t.insert(
                AsrWeightPrefix::audio_layer(i, "fc1.bias"),
                (z(ffn), vec![ffn]),
            );
            t.insert(
                AsrWeightPrefix::audio_layer(i, "fc2.weight"),
                (z(d * ffn), vec![d, ffn]),
            );
            t.insert(AsrWeightPrefix::audio_layer(i, "fc2.bias"), (z(d), vec![d]));
        }
        WeightMap::from_tensors(t)
    }

    #[test]
    fn encoder_builds_and_runs() {
        let cfg = tiny_audio_cfg();
        let n_frames = 8usize;
        let geom = AudioGeometry::new(&cfg, n_frames).unwrap();
        assert_eq!(geom.num_chunks, 2);
        assert_eq!(geom.num_audio_tokens, 2);

        let mut wm = synth_audio_weights(&cfg, geom.freq_pc);
        let built = build_encoder_built(&cfg, &mut wm, &geom).unwrap();
        let params = built.params().clone();
        let mut c = compile_built(built, Device::Cpu).unwrap();
        for (n, d) in &params {
            c.set_param(n, d);
        }
        let padded = geom.num_chunks * geom.max_chunk_len;
        let mel = vec![0.05f32; cfg.num_mel_bins * padded];
        let out = c
            .run(&[("mel", mel.as_slice())])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(out.len(), geom.num_audio_tokens * cfg.output_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    fn tiny_text_cfg() -> Qwen3Config {
        serde_json::from_str(
            r#"{"vocab_size":32,"hidden_size":16,"intermediate_size":32,
                "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":8,"max_position_embeddings":64,"rope_theta":1000000,
                "rms_norm_eps":1e-6,"tie_word_embeddings":true}"#,
        )
        .unwrap()
    }

    fn synth_text_weights(cfg: &Qwen3Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q = cfg.q_proj_dim();
        let kv = cfg.kv_proj_dim();
        let int = cfg.intermediate_size;
        let dh = cfg.head_dim;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.02f32; n];
        t.insert(
            "thinker.model.embed_tokens.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );
        t.insert(
            "thinker.lm_head.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );
        t.insert("thinker.model.norm.weight".into(), (z(h), vec![h]));
        for i in 0..cfg.num_hidden_layers {
            let p = format!("thinker.model.layers.{i}");
            t.insert(format!("{p}.input_layernorm.weight"), (z(h), vec![h]));
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                (z(h), vec![h]),
            );
            t.insert(
                format!("{p}.self_attn.q_proj.weight"),
                (z(q * h), vec![q, h]),
            );
            t.insert(
                format!("{p}.self_attn.k_proj.weight"),
                (z(kv * h), vec![kv, h]),
            );
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                (z(kv * h), vec![kv, h]),
            );
            t.insert(
                format!("{p}.self_attn.o_proj.weight"),
                (z(h * q), vec![h, q]),
            );
            t.insert(format!("{p}.self_attn.q_norm.weight"), (z(dh), vec![dh]));
            t.insert(format!("{p}.self_attn.k_norm.weight"), (z(dh), vec![dh]));
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                (z(int * h), vec![int, h]),
            );
            t.insert(
                format!("{p}.mlp.up_proj.weight"),
                (z(int * h), vec![int, h]),
            );
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                (z(h * int), vec![h, int]),
            );
        }
        WeightMap::from_tensors(t)
    }

    #[test]
    fn prefill_from_embeds_builds_and_runs() {
        let cfg = tiny_text_cfg();
        let mut wm = synth_text_weights(&cfg);
        let seq = 5usize;
        let built = {
            let mut loader = LanguageModelPrefixLoader::new(&mut wm);
            build_asr_prefill_built(&cfg, &mut loader, 1, seq, false).unwrap()
        };
        let params = built.params().clone();
        let mut c = compile_built(built, Device::Cpu).unwrap();
        for (n, d) in &params {
            c.set_param(n, d);
        }
        let embeds = vec![0.03f32; seq * cfg.hidden_size];
        let outs = c.run(&[("inputs_embeds", embeds.as_slice())]);
        // logits + per-layer K/V
        assert_eq!(outs[0].len(), cfg.vocab_size);
        assert_eq!(outs.len(), 1 + 2 * cfg.num_hidden_layers);
    }
}
