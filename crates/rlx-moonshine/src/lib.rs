// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Useful Sensors Moonshine English ASR for RLX.
//!
//! Encoder–decoder Transformer (`MoonshineForConditionalGeneration`) over raw
//! 16 kHz mono PCM (Wav2Vec2-style conv frontend, not Whisper mel). Weights are
//! HuggingFace safetensors; decoding uses Florence-style bucketed full-prefix
//! decoder graphs with a host-side tied LM head.

pub mod builder;
pub mod cli;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod flow;
pub mod generate;
pub mod runner;
pub mod weight_source;
pub mod weights;

pub use config::{MoonshineConfig, MoonshineVariant, SAMPLE_RATE};
pub use runner::{MoonshineModel, MoonshineRunner, MoonshineRunnerBuilder};
pub use weights::MoonshineWeightPrefix;

use anyhow::Result;
use rlx_runtime::Device;

/// Ensure `device` is linked into this build (feature gates).
pub fn validate_device(device: Device) -> Result<()> {
    rlx_core::validate_standard_device("moonshine", device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_source::CloningWeightSource;
    use rlx_core::flow_util::compile_built;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    fn synth_weights(cfg: &MoonshineConfig) -> (WeightMap, MoonshineWeightPrefix) {
        let pfx = MoonshineWeightPrefix {
            encoder: "model.encoder".into(),
            decoder: "model.decoder".into(),
            proj_out: None,
        };
        let d = cfg.hidden_size;
        let v = cfg.vocab_size;
        let e_ff = cfg.encoder_intermediate_size;
        let d_ff = cfg.decoder_intermediate_size;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.01f32; n];

        // Conv frontend
        t.insert(pfx.enc_conv1_w(), (z(d * 127), vec![d, 1, 127]));
        t.insert(pfx.enc_conv2_w(), (z((2 * d) * d * 7), vec![2 * d, d, 7]));
        t.insert(pfx.enc_conv2_b(), (z(2 * d), vec![2 * d]));
        t.insert(pfx.enc_conv3_w(), (z(d * (2 * d) * 3), vec![d, 2 * d, 3]));
        t.insert(pfx.enc_conv3_b(), (z(d), vec![d]));
        t.insert(pfx.enc_groupnorm_w(), (z(d), vec![d]));
        t.insert(pfx.enc_groupnorm_b(), (z(d), vec![d]));
        t.insert(pfx.enc_ln_w(), (z(d), vec![d]));

        for i in 0..cfg.encoder_num_hidden_layers {
            for name in ["q_proj", "k_proj", "v_proj"] {
                t.insert(
                    pfx.enc_layer(i, &format!("self_attn.{name}.weight")),
                    (z(d * d), vec![d, d]),
                );
            }
            t.insert(
                pfx.enc_layer(i, "self_attn.o_proj.weight"),
                (z(d * d), vec![d, d]),
            );
            t.insert(pfx.enc_layer(i, "input_layernorm.weight"), (z(d), vec![d]));
            t.insert(
                pfx.enc_layer(i, "post_attention_layernorm.weight"),
                (z(d), vec![d]),
            );
            t.insert(
                pfx.enc_layer(i, "mlp.fc1.weight"),
                (z(e_ff * d), vec![e_ff, d]),
            );
            t.insert(pfx.enc_layer(i, "mlp.fc1.bias"), (z(e_ff), vec![e_ff]));
            t.insert(
                pfx.enc_layer(i, "mlp.fc2.weight"),
                (z(d * e_ff), vec![d, e_ff]),
            );
            t.insert(pfx.enc_layer(i, "mlp.fc2.bias"), (z(d), vec![d]));
        }

        t.insert(pfx.dec_embed_tokens(), (z(v * d), vec![v, d]));
        t.insert(pfx.dec_norm_w(), (z(d), vec![d]));

        for i in 0..cfg.decoder_num_hidden_layers {
            for blk in ["self_attn", "encoder_attn"] {
                for name in ["q_proj", "k_proj", "v_proj"] {
                    t.insert(
                        pfx.dec_layer(i, &format!("{blk}.{name}.weight")),
                        (z(d * d), vec![d, d]),
                    );
                }
                t.insert(
                    pfx.dec_layer(i, &format!("{blk}.o_proj.weight")),
                    (z(d * d), vec![d, d]),
                );
            }
            for n in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "final_layernorm.weight",
            ] {
                t.insert(pfx.dec_layer(i, n), (z(d), vec![d]));
            }
            t.insert(
                pfx.dec_layer(i, "mlp.fc1.weight"),
                (z((2 * d_ff) * d), vec![2 * d_ff, d]),
            );
            t.insert(
                pfx.dec_layer(i, "mlp.fc1.bias"),
                (z(2 * d_ff), vec![2 * d_ff]),
            );
            t.insert(
                pfx.dec_layer(i, "mlp.fc2.weight"),
                (z(d * d_ff), vec![d, d_ff]),
            );
            t.insert(pfx.dec_layer(i, "mlp.fc2.bias"), (z(d), vec![d]));
        }

        (WeightMap::from_tensors(t), pfx)
    }

    #[test]
    fn tiny_preset_and_feat_len() {
        let c = MoonshineConfig::tiny();
        assert_eq!(c.hidden_size, 288);
        assert_eq!(MoonshineConfig::feat_extract_output_length(16_000), 40);
    }

    #[test]
    fn synth_encoder_decoder_forward_cpu() {
        let cfg = MoonshineConfig::synth_tiny();
        let (weights, pfx) = synth_weights(&cfg);
        // ≥895 samples → ≥1 encoder frame; use 1024.
        let audio_len = 1024;
        let enc_seq = MoonshineConfig::feat_extract_output_length(audio_len);
        assert!(enc_seq >= 1);

        let mut src = CloningWeightSource(&weights);
        let enc_built =
            flow::build_encoder_built(&cfg, &mut src, &pfx, 1, audio_len).expect("encoder build");
        let mut enc_g = compile_built(enc_built, Device::Cpu).expect("encoder compile");
        let pcm = vec![0.01f32; audio_len];
        let enc_out = enc_g.run(&[("pcm", pcm.as_slice())]);
        let enc_hidden = enc_out.into_iter().next().expect("enc out");
        assert_eq!(enc_hidden.len(), enc_seq * cfg.hidden_size);

        let mut src = CloningWeightSource(&weights);
        let dec_built = flow::build_decoder_hidden_built(&cfg, &mut src, &pfx, 1, 4, enc_seq)
            .expect("decoder build");
        let mut dec_g = compile_built(dec_built, Device::Cpu).expect("decoder compile");
        let embeds = vec![0.01f32; 4 * cfg.hidden_size];
        let dec_out = dec_g.run(&[
            ("decoder_inputs_embeds", embeds.as_slice()),
            ("encoder_hidden", enc_hidden.as_slice()),
        ]);
        let hidden = dec_out.into_iter().next().expect("dec out");
        assert_eq!(hidden.len(), 4 * cfg.hidden_size);
        assert!(hidden.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn model_encode_pcm_synth() {
        let cfg = MoonshineConfig::synth_tiny();
        let (weights, _) = synth_weights(&cfg);
        let mut model = MoonshineModel::from_weight_map(weights, cfg.clone(), Device::Cpu).unwrap();
        let pcm = vec![0.01f32; 1024];
        let enc = model.encode_pcm(&pcm).unwrap();
        let t = MoonshineConfig::feat_extract_output_length(1024);
        assert_eq!(enc.len(), t * cfg.hidden_size);
    }
}
