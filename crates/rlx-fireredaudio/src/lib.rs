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

//! # rlx-fireredaudio
//!
//! **FireRedAudio** (FireRedTeam) on RLX — a general-purpose audio language model
//! with a shared **Qwen3.5 (~9B)** backbone and **decoupled continuous
//! representations**: a Whisper-style Audio Encoder for understanding, and a
//! **RedAE → Patch Encoder → DiT** pathway for speech generation. One model does
//! ASR, audio understanding (optional CoT), zero-shot TTS, voice design, and
//! semantic / acoustic speech editing.
//!
//! Native Rust, composing planned rlx pieces:
//!
//! - **Backbone** → Qwen3.5 hybrid linear/full attention (`rlx-qwen35`).
//! - **Understanding** → 16 kHz mel encoder (Whisper-shaped Conv1d tower).
//! - **Generation** → 24 kHz RedAE latents @ 25 Hz + flow-matching DiT.
//!
//! Checkpoint-free protocol plus the understanding path end-to-end: mel →
//! Conv1d audio encoder → Qwen3.5 backbone (`inputs_embeds` scatter) → text.
//! Generation (RedAE + DiT) is exposed at the API layer; graphs are next.

mod acoustic;
mod audio;
mod cli;
mod config;
mod embed;
mod encoder;
mod hf_config;
mod load;
mod prefix;
mod prompt;
mod rates;
mod runner;
#[cfg(feature = "tokenizer")]
mod tokenizer;
mod weights;

pub use acoustic::{AcousticEdit, format_acoustic, parse_acoustic};
pub use audio::{
    AudioGeometry, MelSpectrogram, after_conv2_len, conv_len, mel_frames_for_samples,
    pcm_to_log_mel,
};
pub use cli::cli_run;
pub use config::{
    AudioEncoderConfig, BackboneConfig, DitConfig, FireRedAudioConfig, PatchEncoderConfig,
    RedVaeConfig, SpecialTokens,
};
pub use embed::{argmax_token, count_audio_placeholders, fuse_inputs_embeds};
pub use encoder::build_encoder_built;
pub use hf_config::load_qwen35_backbone;
pub use load::{WeightStore, resolve_model_dir};
pub use prompt::{
    DEFAULT_ASR_PROMPT, EditType, FireRedTask, GENERIC_SYSTEM_PROMPT, THINK_CLOSED, THINK_END,
    THINK_OPEN, UNDERSTAND_SYSTEM_PROMPT, build_asr_prompt, build_edit_prompt, build_tts_prompt,
    build_understand_prompt, build_voice_design_prompt, chatml, split_thinking,
};
pub use rates::{
    GENERATION_SAMPLE_RATE, PATCH_ENCODER_DOWNSAMPLE_RATE, UNDERSTAND_SAMPLE_RATE,
    VAE_DOWNSAMPLE_RATE, audio_encoder_output_len, pad_generation_len, patch_token_frames,
    vae_latent_frames,
};
pub use runner::{FireRedRunner, FireRedRunnerBuilder, GenerationResult, UnderstandResult};
#[cfg(feature = "tokenizer")]
pub use tokenizer::FireRedTokenizer;
pub use weights::{
    AudioWeightPrefix, KEY_EMBED_TOKENS, KEY_LM_HEAD, PREFIX_AUDIO, PREFIX_BACKBONE, PREFIX_DIT,
    PREFIX_PATCH_ENCODER, PREFIX_RED_VAE, audio_layer,
};

pub use rlx_whisper::{SAMPLE_RATE, load_wav_mono_f32, parse_wav_mono_f32};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::AudioWeightPrefix;
    use rlx_core::flow_util::compile_built;
    use rlx_core::weight_map::WeightMap;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    #[test]
    fn config_defaults_match_released_checkpoint() {
        let c = FireRedAudioConfig::default();
        assert_eq!(c.backbone.hidden_size, 4096);
        assert_eq!(c.backbone.num_hidden_layers, 32);
        assert_eq!(c.backbone.vocab_size, 248_320);
        assert_eq!(c.backbone.num_full_attention_layers(), 8);
        assert_eq!(c.audio_encoder.d_model, 1280);
        assert_eq!(c.audio_encoder.num_mel_bins, 128);
        assert_eq!(c.audio_encoder.head_dim(), 64);
        assert_eq!(c.audio_encoder.chunk_frames(), 3000);
        assert_eq!(c.red_vae.out_dim, 64);
        assert_eq!(c.red_vae.audio_sample_rate, 24_000);
        assert_eq!(c.dit.depth, 11);
        assert_eq!(c.tokens.sosp_idx, 248_077);
        assert_eq!(c.tokens.audio_special_token, "<|AUDIO|>");
        assert_eq!(
            c.tokens.audio_special_token_no_latent,
            "<|AUDIO_NO_LATENT|>"
        );
        c.validate().unwrap();
        assert!((c.vae_frame_rate_hz() - 25.0).abs() < 1e-6);
        assert!((c.patch_token_rate_hz() - 6.25).abs() < 1e-6);
    }

    #[test]
    fn understand_prompt_matches_training_template() {
        let p = build_understand_prompt("这个音频中有几个说话人", 1, "<|AUDIO|>", false).unwrap();
        assert!(p.starts_with("<|im_start|>system\nYou are an audio understanding expert."));
        assert!(p.contains("Audio 1: <|sosp|><|AUDIO|><|eosp|>\n这个音频中有几个说话人"));
        assert!(p.contains(THINK_CLOSED));
        assert!(p.ends_with(THINK_CLOSED));
        let open = build_understand_prompt("q", 2, "<|AUDIO|>", true).unwrap();
        assert!(open.ends_with(THINK_OPEN));
        assert!(!open.contains(THINK_CLOSED));
        assert!(open.contains("Audio 2:"));
    }

    #[test]
    fn asr_uses_default_prompt() {
        let p = build_asr_prompt(1, "<|AUDIO|>").unwrap();
        assert!(p.contains(DEFAULT_ASR_PROMPT));
    }

    #[test]
    fn tts_prompt_forces_audio_mode_prefix() {
        let zh = build_tts_prompt("提示", "目标", "zh", "<|AUDIO_NO_LATENT|>").unwrap();
        assert!(zh.contains("Convert text to speech.\n提示目标"));
        assert!(zh.ends_with("<|sosp|><|AUDIO_NO_LATENT|>"));
        let en = build_tts_prompt("hello", "world", "en", "<|AUDIO_NO_LATENT|>").unwrap();
        assert!(en.contains("Convert text to speech.\nhello world"));
    }

    #[test]
    fn edit_prompts_differ_by_type() {
        let sem =
            build_edit_prompt("delete 'x'", EditType::Semantic, "<|AUDIO_NO_LATENT|>").unwrap();
        assert!(sem.contains("Identify the content of the audio. delete 'x'"));
        let ac = build_edit_prompt(
            "shift the pitch by 3 steps",
            EditType::Acoustic,
            "<|AUDIO_NO_LATENT|>",
        )
        .unwrap();
        assert!(
            ac.contains("Audio 1: <|sosp|><|AUDIO_NO_LATENT|><|eosp|>\nshift the pitch by 3 steps")
        );
        assert!(!ac.contains("Identify the content"));
    }

    #[test]
    fn voice_design_keeps_chinese_bridge() {
        let p = build_voice_design_prompt("bright female", "Hello.").unwrap();
        assert!(p.contains("根据上述音色描述，合成以下文本对应的音频：\nHello."));
    }

    #[test]
    fn thinking_split() {
        let (r, a) = split_thinking("reason here</think>\n\nfinal answer");
        assert_eq!(r, Some("reason here"));
        assert_eq!(a, "final answer");
        let (r2, a2) = split_thinking("no cot");
        assert_eq!(r2, None);
        assert_eq!(a2, "no cot");
    }

    #[test]
    fn acoustic_roundtrip_and_rejects() {
        for edit in [
            AcousticEdit::Pitch { steps: 3 },
            AcousticEdit::Pitch { steps: -1 },
            AcousticEdit::Speed { rate: 0.5 },
            AcousticEdit::Speed { rate: 1.0 },
            AcousticEdit::Volume { gain: 1.2 },
        ] {
            let s = format_acoustic(edit).unwrap();
            assert_eq!(parse_acoustic(&s).unwrap(), edit);
        }
        assert_eq!(
            format_acoustic(AcousticEdit::Pitch { steps: 3 }).unwrap(),
            "shift the pitch by 3 steps"
        );
        assert!(format_acoustic(AcousticEdit::Pitch { steps: 0 }).is_err());
        assert!(parse_acoustic("make it louder").is_err());
        assert!(parse_acoustic("shift the pitch by 1 steps").is_err());
    }

    #[test]
    fn rates_and_pad() {
        assert_eq!(UNDERSTAND_SAMPLE_RATE, 16_000);
        assert_eq!(GENERATION_SAMPLE_RATE, 24_000);
        assert_eq!(pad_generation_len(1), PATCH_ENCODER_DOWNSAMPLE_RATE);
        assert_eq!(pad_generation_len(3840), 3840);
        assert_eq!(vae_latent_frames(3840), 4);
        assert_eq!(patch_token_frames(3840), 1);
        assert_eq!(audio_encoder_output_len(3000), 375);
    }

    #[test]
    fn task_flags() {
        assert!(FireRedTask::Asr.is_understanding());
        assert!(FireRedTask::Tts.is_generation());
        assert!(FireRedTask::Understand.supports_thinking());
        assert!(!FireRedTask::Asr.supports_thinking());
    }

    #[test]
    fn thinking_incompatible_with_tts_prefix() {
        assert!(chatml("s", "u", "<|sosp|>x", true).is_err());
    }

    fn tiny_audio_cfg() -> AudioEncoderConfig {
        AudioEncoderConfig {
            d_model: 8,
            encoder_layers: 2,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 16,
            num_mel_bins: 16,
            output_dim: 8,
            max_source_positions: 64,
            n_window: 2, // chunk = 4
        }
    }

    fn synth_audio_weights(cfg: &AudioEncoderConfig) -> WeightMap {
        let d = cfg.d_model;
        let mels = cfg.num_mel_bins;
        let ffn = cfg.encoder_ffn_dim;
        let od = cfg.output_dim;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.02f32; n];

        t.insert(
            AudioWeightPrefix::CONV1_W.into(),
            (z(d * mels * 3), vec![d, mels, 3]),
        );
        t.insert(AudioWeightPrefix::CONV1_B.into(), (z(d), vec![d]));
        t.insert(
            AudioWeightPrefix::CONV2_W.into(),
            (z(d * d * 3), vec![d, d, 3]),
        );
        t.insert(AudioWeightPrefix::CONV2_B.into(), (z(d), vec![d]));

        t.insert(
            AudioWeightPrefix::ADAPTER_CONV3_W.into(),
            (z(d * d * 3), vec![d, d, 3]),
        );
        t.insert(AudioWeightPrefix::ADAPTER_CONV3_B.into(), (z(d), vec![d]));
        t.insert(
            AudioWeightPrefix::ADAPTER_CONV4_W.into(),
            (z(d * d * 3), vec![d, d, 3]),
        );
        t.insert(AudioWeightPrefix::ADAPTER_CONV4_B.into(), (z(d), vec![d]));
        t.insert(AudioWeightPrefix::ADAPTER_LN_W.into(), (z(d), vec![d]));
        t.insert(AudioWeightPrefix::ADAPTER_LN_B.into(), (z(d), vec![d]));
        t.insert(
            AudioWeightPrefix::ADAPTER_LINEAR1_W.into(),
            (z(od * d), vec![od, d]),
        );
        t.insert(
            AudioWeightPrefix::ADAPTER_LINEAR1_B.into(),
            (z(od), vec![od]),
        );
        t.insert(
            AudioWeightPrefix::ADAPTER_LINEAR2_W.into(),
            (z(od * od), vec![od, od]),
        );
        t.insert(
            AudioWeightPrefix::ADAPTER_LINEAR2_B.into(),
            (z(od), vec![od]),
        );

        for i in 0..cfg.encoder_layers {
            for name in ["q_proj", "v_proj", "out_proj"] {
                t.insert(
                    AudioWeightPrefix::audio_layer(i, &format!("self_attn.{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
                t.insert(
                    AudioWeightPrefix::audio_layer(i, &format!("self_attn.{name}.bias")),
                    (z(d), vec![d]),
                );
            }
            t.insert(
                AudioWeightPrefix::audio_layer(i, "self_attn.k_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            for n in ["self_attn_layer_norm", "final_layer_norm"] {
                t.insert(
                    AudioWeightPrefix::audio_layer(i, &format!("{n}.weight")),
                    (z(d), vec![d]),
                );
                t.insert(
                    AudioWeightPrefix::audio_layer(i, &format!("{n}.bias")),
                    (z(d), vec![d]),
                );
            }
            t.insert(
                AudioWeightPrefix::audio_layer(i, "fc1.weight"),
                (z(ffn * d), vec![ffn, d]),
            );
            t.insert(
                AudioWeightPrefix::audio_layer(i, "fc1.bias"),
                (z(ffn), vec![ffn]),
            );
            t.insert(
                AudioWeightPrefix::audio_layer(i, "fc2.weight"),
                (z(d * ffn), vec![d, ffn]),
            );
            t.insert(
                AudioWeightPrefix::audio_layer(i, "fc2.bias"),
                (z(d), vec![d]),
            );
        }
        WeightMap::from_tensors(t)
    }

    fn run_encoder_on(device: Device) {
        if device != Device::Cpu && !rlx_runtime::is_available(device) {
            eprintln!("skip encoder on {device:?} (unavailable)");
            return;
        }
        let cfg = tiny_audio_cfg();
        let n_frames = 8usize;
        let geom = AudioGeometry::new(&cfg, n_frames).unwrap();
        assert_eq!(geom.num_chunks, 2);
        assert!(geom.num_audio_tokens > 0);

        let mut wm = synth_audio_weights(&cfg);
        let built = build_encoder_built(&cfg, &mut wm, &geom).unwrap();
        let params = built.params().clone();
        let mut c = compile_built(built, device).unwrap();
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

    #[test]
    fn encoder_builds_and_runs() {
        run_encoder_on(Device::Cpu);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn encoder_runs_on_metal() {
        run_encoder_on(Device::Metal);
    }

    #[cfg(all(target_os = "macos", feature = "mlx"))]
    #[test]
    fn encoder_runs_on_mlx() {
        run_encoder_on(Device::Mlx);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn encoder_runs_on_cuda() {
        run_encoder_on(Device::Cuda);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn encoder_runs_on_rocm() {
        run_encoder_on(Device::Rocm);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn encoder_runs_on_wgpu() {
        run_encoder_on(Device::Gpu);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn encoder_runs_on_vulkan() {
        run_encoder_on(Device::Vulkan);
    }

    #[test]
    fn fuse_replaces_audio_slots() {
        let cfg = FireRedAudioConfig::default();
        let h = 4usize;
        let mut tiny = cfg.clone();
        tiny.backbone.hidden_size = h;
        tiny.backbone.vocab_size = 8;
        tiny.tokens.audio_special_token_id = 3;
        let embed: Vec<f32> = (0..8 * h).map(|i| i as f32).collect();
        let audio = vec![100.0f32; 2 * h];
        let ids = [1u32, 3, 2, 3];
        let out = fuse_inputs_embeds(&tiny, &embed, &ids, &audio).unwrap();
        assert_eq!(&out[h..2 * h], &audio[..h]);
        assert_eq!(&out[3 * h..4 * h], &audio[h..]);
        assert_eq!(&out[..h], &embed[h..2 * h]);
    }

    #[test]
    fn loads_released_hf_config_json() {
        let path = std::path::Path::new(".cache/fireredaudio/FireRedAudio/config.json");
        if !path.is_file() {
            return;
        }
        let c = FireRedAudioConfig::from_file(path).unwrap();
        assert_eq!(c.backbone.hidden_size, 4096);
        assert_eq!(c.backbone.num_hidden_layers, 32); // MTP stripped
        assert_eq!(c.audio_encoder.output_dim, 4096);
        assert_eq!(c.red_vae.out_dim, 64);
        let q = c.qwen35_config();
        assert_eq!(q.hidden_size, 4096);
        assert_eq!(q.full_attention_interval, 4);
    }
}
