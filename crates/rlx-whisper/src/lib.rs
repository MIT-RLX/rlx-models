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

//! OpenAI Whisper ASR — encoder (mel → audio states) + autoregressive text decoder.
//!
//! Architecture matches [OpenAI Whisper](https://github.com/openai/whisper) /
//! [fast-whisper-burn](https://github.com/AdrianEddy/fast-whisper-burn): two conv1d
//! layers, sinusoidal encoder positions, transformer encoder/decoder with cross-attention.
//!
//! Weights: HuggingFace `model.safetensors` + `config.json` (e.g. `openai/whisper-tiny`).
//!
//! # Subtitles pipeline
//!
//! Enable `timestamps`, `word-dtw`, and optionally `silero-vad` / `diarize`:
//!
//! - [`WhisperPipeline`] — ASR → segment timestamps → word alignment → speaker labels
//! - [`WhisperTranscript`] — structured segments + [`WordTiming`] list
//! - [`subtitles`] — SRT / VTT / TSV / JSON export
//!
//! GPU routing (Metal): mel encoder and DTW align-hidden on GPU; cross / prefill /
//! bucketed decode on CPU until decoder parity lands. See [`backend::whisper_encoder_device`]
//! and [`backend::whisper_decoder_device`].
//!
//! Runbook: [README.md](README.md).

pub mod audio;
pub mod backend;
pub mod batch;
pub mod bench_fixture;
pub mod builder;
pub mod cache;
pub mod cli;
pub mod config;
pub mod decode;
pub mod flow;
pub mod fused;
pub mod mel;
pub mod runner;
pub mod vad;
pub mod weights;

pub mod alignment_heads;
#[cfg(feature = "word-dtw")]
pub mod cross_attn_align;
#[cfg(feature = "diarize")]
pub mod diarize;
#[cfg(feature = "word-dtw")]
pub mod dtw;
#[cfg(feature = "word-w2v")]
pub mod forced_align;
#[cfg(feature = "timestamps")]
pub mod pipeline;
#[cfg(feature = "silero-vad")]
pub mod silero_vad;
#[cfg(feature = "timestamps")]
pub mod subtitles;
#[cfg(feature = "timestamps")]
pub mod timestamp_parse;
#[cfg(feature = "timestamps")]
pub mod transcript;

pub use audio::{
    EnergyVad, MelSpectrogram, N_FRAMES, N_SAMPLES, SAMPLE_RATE, SpeechSegment, load_wav_mono_f32,
    parse_wav_mono_f32, pcm_segments_by_vad, pcm_segments_by_vad_config, pcm_to_mel,
};
pub use backend::WhisperGraphCtx;
pub use batch::{batched_prompt_f32, replicate_encoder_for_beams, stack_kv_caches};
pub use bench_fixture::{
    JFK_REFERENCE, assert_transcript_matches_reference, bench_cache_dir, ensure_jfk_fixture,
    jfk_reference_path, jfk_wav_path, load_jfk_reference, normalize_transcript, transcripts_match,
};
pub use builder::WhisperGraphOpts;
pub use cache::{WhisperCrossCache, WhisperKvCache};
pub use config::WhisperConfig;
pub use decode::SuppressionMask;
pub use decode::{batched_logits_row, batched_logits_row_owned};
pub use flow::{
    WhisperDecoderFlow, WhisperEncoderFlow, build_whisper_cross_kv_built,
    build_whisper_decode_step_built, build_whisper_decode_step_built_ext,
    build_whisper_decoder_built, build_whisper_decoder_graph_sized,
    build_whisper_decoder_prefill_built, build_whisper_decoder_prefill_built_ext,
    build_whisper_encoder_built, build_whisper_encoder_built_opts,
    build_whisper_encoder_graph_sized, default_mel_frames,
};
pub use fused::{FusedDecoderWeights, FusedEncoderWeights};
pub use mel::{mel_geometry_frames_for_pcm, pcm_to_log_mel, stack_mels};
pub use runner::{WhisperBenchReport, WhisperRunner, WhisperRunnerBuilder};
pub use vad::{VadConfig, VadKind, segments_by_vad};
pub use weights::WhisperWeightPrefix;

#[cfg(feature = "diarize")]
pub use diarize::assign_speakers;
#[cfg(feature = "timestamps")]
pub use pipeline::{WhisperPipeline, WhisperPipelineOpts};
#[cfg(feature = "timestamps")]
pub use subtitles::{SubtitleFormat, to_json_pretty, to_srt, to_tsv, to_vtt};
#[cfg(feature = "timestamps")]
pub use transcript::{TranscriptSegment, WhisperTranscript, WordAlignMode, WordTiming};

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    fn cross_from_enc(
        cfg: &WhisperConfig,
        wm: &mut WeightMap,
        pfx: &weights::WhisperWeightPrefix,
        enc: &[f32],
        enc_seq: usize,
    ) -> WhisperCrossCache {
        let cross_built = build_whisper_cross_kv_built(cfg, wm, pfx, 1, enc_seq).unwrap();
        let cross_params = cross_built.params().clone();
        let mut cross_c = rlx_core::flow_util::compile_built(cross_built, Device::Cpu).unwrap();
        for (n, d) in &cross_params {
            cross_c.set_param(n, d);
        }
        let outs = cross_c.run(&[("encoder_hidden", enc)]);
        crate::cache::cross_from_outputs(cfg.decoder_layers, 1, enc_seq, cfg.d_model, &outs)
            .unwrap()
    }

    fn synth_weights(cfg: &WhisperConfig) -> (WeightMap, weights::WhisperWeightPrefix) {
        let pfx = weights::WhisperWeightPrefix {
            encoder: "model.encoder".into(),
            decoder: "model.decoder".into(),
            hf_embed_names: true,
        };
        let d = cfg.d_model;
        let m = cfg.num_mel_bins;
        let v = cfg.vocab_size;
        let mlp = d * 4;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.01f32; n];
        t.insert(pfx.enc_conv1_w(), (z(d * m * 3), vec![d, m, 3]));
        t.insert(pfx.enc_conv1_b(), (z(d), vec![d]));
        t.insert(pfx.enc_conv2_w(), (z(d * d * 3), vec![d, d, 3]));
        t.insert(pfx.enc_conv2_b(), (z(d), vec![d]));
        t.insert(pfx.enc_ln_post_w(), (z(d), vec![d]));
        t.insert(pfx.enc_ln_post_b(), (z(d), vec![d]));
        for i in 0..cfg.encoder_layers {
            for name in ["self_attn.q_proj", "self_attn.out_proj", "self_attn.v_proj"] {
                t.insert(
                    pfx.enc_layer(i, &format!("{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
                t.insert(pfx.enc_layer(i, &format!("{name}.bias")), (z(d), vec![d]));
            }
            t.insert(
                pfx.enc_layer(i, "self_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(pfx.enc_layer(i, "fc1.weight"), (z(mlp * d), vec![mlp, d]));
            t.insert(pfx.enc_layer(i, "fc1.bias"), (z(mlp), vec![mlp]));
            t.insert(pfx.enc_layer(i, "fc2.weight"), (z(d * mlp), vec![d, mlp]));
            t.insert(pfx.enc_layer(i, "fc2.bias"), (z(d), vec![d]));
            for n in ["self_attn_layer_norm", "final_layer_norm"] {
                t.insert(pfx.enc_layer(i, &format!("{n}.weight")), (z(d), vec![d]));
                t.insert(pfx.enc_layer(i, &format!("{n}.bias")), (z(d), vec![d]));
            }
        }
        t.insert(pfx.dec_embed_tokens(), (z(v * d), vec![v, d]));
        t.insert(
            pfx.dec_embed_positions(),
            (
                z(cfg.max_target_positions * d),
                vec![cfg.max_target_positions, d],
            ),
        );
        t.insert(pfx.dec_ln_w(), (z(d), vec![d]));
        t.insert(pfx.dec_ln_b(), (z(d), vec![d]));
        for i in 0..cfg.decoder_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.out_proj",
                "self_attn.v_proj",
                "encoder_attn.q_proj",
                "encoder_attn.out_proj",
                "encoder_attn.v_proj",
            ] {
                t.insert(
                    pfx.dec_layer(i, &format!("{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
                t.insert(pfx.dec_layer(i, &format!("{name}.bias")), (z(d), vec![d]));
            }
            t.insert(
                pfx.dec_layer(i, "self_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(
                pfx.dec_layer(i, "encoder_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(pfx.dec_layer(i, "fc1.weight"), (z(mlp * d), vec![mlp, d]));
            t.insert(pfx.dec_layer(i, "fc1.bias"), (z(mlp), vec![mlp]));
            t.insert(pfx.dec_layer(i, "fc2.weight"), (z(d * mlp), vec![d, mlp]));
            t.insert(pfx.dec_layer(i, "fc2.bias"), (z(d), vec![d]));
            for n in [
                "self_attn_layer_norm",
                "encoder_attn_layer_norm",
                "final_layer_norm",
            ] {
                t.insert(pfx.dec_layer(i, &format!("{n}.weight")), (z(d), vec![d]));
                t.insert(pfx.dec_layer(i, &format!("{n}.bias")), (z(d), vec![d]));
            }
        }
        (WeightMap::from_tensors(t), pfx)
    }

    #[test]
    fn whisper_encoder_runs() {
        let cfg = WhisperConfig::tiny_synthetic();
        let mel_frames = 8;
        let batch = 1;
        let (mut wm, pfx) = synth_weights(&cfg);
        let built = build_whisper_encoder_built(&cfg, &mut wm, &pfx, batch, mel_frames).unwrap();
        let params = built.params().clone();
        let mut compiled = rlx_core::flow_util::compile_built(built, Device::Cpu).unwrap();
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        let mel = vec![0.02f32; batch * cfg.num_mel_bins * mel_frames];
        let out = compiled.run(&[("mel", &mel)]).into_iter().next().unwrap();
        let enc_seq = cfg.encoder_seq_len(mel_frames);
        assert_eq!(out.len(), batch * enc_seq * cfg.d_model);
    }

    #[test]
    fn whisper_kv_decode_step_runs() {
        let cfg = WhisperConfig::tiny_synthetic();
        let mel_frames = 8;
        let batch = 1;
        let enc_seq = cfg.encoder_seq_len(mel_frames);
        let prompt: Vec<f32> = vec![1.0, 2.0, 3.0];
        let (mut wm, pfx) = synth_weights(&cfg);
        let enc_built =
            build_whisper_encoder_built(&cfg, &mut wm, &pfx, batch, mel_frames).unwrap();
        let enc_params = enc_built.params().clone();
        let mut enc_c = rlx_core::flow_util::compile_built(enc_built, Device::Cpu).unwrap();
        for (n, d) in &enc_params {
            enc_c.set_param(n, d);
        }
        let mel = vec![0.02f32; batch * cfg.num_mel_bins * mel_frames];
        let enc = enc_c.run(&[("mel", &mel)]).into_iter().next().unwrap();

        let (mut wm2, _) = synth_weights(&cfg);
        let cross = cross_from_enc(&cfg, &mut wm2, &pfx, &enc, enc_seq);
        let prefill =
            build_whisper_decoder_prefill_built(&cfg, &mut wm2, &pfx, batch, prompt.len(), enc_seq)
                .unwrap();
        let pre_params = prefill.params().clone();
        let mut pre_c = rlx_core::flow_util::compile_built(prefill, Device::Cpu).unwrap();
        for (n, d) in &pre_params {
            pre_c.set_param(n, d);
        }
        let cross_keys: Vec<String> = (0..cfg.decoder_layers)
            .flat_map(|i| [format!("cross_k_{i}"), format!("cross_v_{i}")])
            .collect();
        let mut pre_in: Vec<(&str, &[f32])> = vec![("token_ids", &prompt)];
        for i in 0..cfg.decoder_layers {
            pre_in.push((cross_keys[2 * i].as_str(), cross.layers_k[i].as_slice()));
            pre_in.push((cross_keys[2 * i + 1].as_str(), cross.layers_v[i].as_slice()));
        }
        let outs = pre_c.run(&pre_in);
        let cache = crate::cache::kv_from_prefill_outputs(
            cfg.decoder_layers,
            batch,
            prompt.len(),
            cfg.d_model,
            &outs[1..],
        )
        .unwrap();
        assert_eq!(cache.past_len, prompt.len());

        let (mut wm3, _) = synth_weights(&cfg);
        let step =
            build_whisper_decode_step_built(&cfg, &mut wm3, &pfx, batch, cache.past_len, enc_seq)
                .unwrap();
        let step_params = step.params().clone();
        let mut step_c = rlx_core::flow_util::compile_built(step, Device::Cpu).unwrap();
        for (n, d) in &step_params {
            step_c.set_param(n, d);
        }
        let key_past: Vec<String> = (0..cfg.decoder_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let cross_keys2: Vec<String> = (0..cfg.decoder_layers)
            .flat_map(|i| [format!("cross_k_{i}"), format!("cross_v_{i}")])
            .collect();
        let past = cache.past_len;
        let pos_ix = [past as f32];
        let mut inputs: Vec<(&str, &[f32])> = vec![("token_id", &[4.0f32]), ("pos_ix", &pos_ix)];
        for i in 0..cfg.decoder_layers {
            inputs.push((cross_keys2[2 * i].as_str(), cross.layers_k[i].as_slice()));
            inputs.push((
                cross_keys2[2 * i + 1].as_str(),
                cross.layers_v[i].as_slice(),
            ));
        }
        for i in 0..cfg.decoder_layers {
            inputs.push((key_past[2 * i].as_str(), cache.layers_k[i].as_slice()));
            inputs.push((key_past[2 * i + 1].as_str(), cache.layers_v[i].as_slice()));
        }
        let step_out = step_c.run(&inputs);
        assert_eq!(step_out[0].len(), cfg.vocab_size);
        assert_eq!(step_out.len(), 1 + 2 * cfg.decoder_layers);
    }

    #[test]
    fn whisper_decoder_runs() {
        let cfg = WhisperConfig::tiny_synthetic();
        let mel_frames = 8;
        let batch = 1;
        let enc_seq = cfg.encoder_seq_len(mel_frames);
        let dec_seq = 4;
        let (mut wm, pfx) = synth_weights(&cfg);
        let enc = build_whisper_encoder_built(&cfg, &mut wm, &pfx, batch, mel_frames).unwrap();
        let enc_params = enc.params().clone();
        let mut enc_c = rlx_core::flow_util::compile_built(enc, Device::Cpu).unwrap();
        for (n, d) in &enc_params {
            enc_c.set_param(n, d);
        }
        let mel = vec![0.02f32; batch * cfg.num_mel_bins * mel_frames];
        let hidden = enc_c.run(&[("mel", &mel)]).into_iter().next().unwrap();

        let (mut wm2, pfx2) = synth_weights(&cfg);
        let dec =
            build_whisper_decoder_built(&cfg, &mut wm2, &pfx2, batch, dec_seq, enc_seq).unwrap();
        let dec_params = dec.params().clone();
        let mut dec_c = rlx_core::flow_util::compile_built(dec, Device::Cpu).unwrap();
        for (n, d) in &dec_params {
            dec_c.set_param(n, d);
        }
        let tokens: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let logits = dec_c
            .run(&[("token_ids", &tokens), ("encoder_hidden", &hidden)])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(logits.len(), batch * dec_seq * cfg.vocab_size);
    }
}
