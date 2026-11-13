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

//! The three Microsoft handwritten checkpoints' configurations (trimmed to
//! the fields the reader looks at) parse to the right architecture, and
//! the decoder step builds for each without weights.

use rlx_trocr::config::{DecoderAct, EncoderKind, TrOcrConfig};

const SMALL: &str = r#"{"encoder": {"model_type": "deit", "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6, "intermediate_size": 1536, "image_size": 384, "patch_size": 16, "layer_norm_eps": 1e-12, "hidden_act": "gelu", "qkv_bias": true, "num_channels": 3}, "decoder": {"model_type": "trocr", "d_model": 256, "decoder_layers": 6, "decoder_attention_heads": 8, "decoder_ffn_dim": 1024, "activation_function": "relu", "vocab_size": 64044, "scale_embedding": true, "layernorm_embedding": true, "use_learned_position_embeddings": true, "max_position_embeddings": 512, "cross_attention_hidden_size": 384, "decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1}, "eos_token_id": 2}"#;
const BASE: &str = r#"{"encoder": {"model_type": "vit", "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072, "image_size": 384, "patch_size": 16, "layer_norm_eps": 1e-12, "hidden_act": "gelu", "qkv_bias": false, "num_channels": 3}, "decoder": {"model_type": "trocr", "d_model": 1024, "decoder_layers": 12, "decoder_attention_heads": 16, "decoder_ffn_dim": 4096, "activation_function": "gelu", "vocab_size": 50265, "scale_embedding": false, "max_position_embeddings": 512, "cross_attention_hidden_size": 768, "decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1}}"#;
const LARGE: &str = r#"{"encoder": {"model_type": "vit", "hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096, "image_size": 384, "patch_size": 16, "layer_norm_eps": 1e-12, "hidden_act": "gelu", "qkv_bias": false, "num_channels": 3}, "decoder": {"model_type": "trocr", "d_model": 1024, "decoder_layers": 12, "decoder_attention_heads": 16, "decoder_ffn_dim": 4096, "activation_function": "gelu", "vocab_size": 50265, "scale_embedding": false, "max_position_embeddings": 512, "cross_attention_hidden_size": 1024, "decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1}}"#;
const VIT_PRE: &str =
    r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "resample": 2, "size": 384}"#;
const DEIT_PRE: &str =
    r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "resample": 3, "size": 384}"#;

fn parse(cfg: &str, pre: &str) -> TrOcrConfig {
    let c: serde_json::Value = serde_json::from_str(cfg).unwrap();
    let p: serde_json::Value = serde_json::from_str(pre).unwrap();
    TrOcrConfig::from_values(&c, Some(&p)).unwrap()
}

#[test]
fn small_is_deit_with_scaled_relu_decoder() {
    let c = parse(SMALL, DEIT_PRE);
    assert_eq!(c.encoder.kind, EncoderKind::Deit);
    assert_eq!(c.encoder.seq(), 578);
    assert!(c.encoder.qkv_bias);
    assert_eq!(c.decoder.act, DecoderAct::Relu);
    assert!(c.decoder.scale_embedding && c.decoder.layernorm_embedding);
    assert_eq!(
        c.preprocess.filter,
        rlx_core::image_preprocess::Filter::Bicubic
    );
}

#[test]
fn base_and_large_are_vit_without_qkv_bias() {
    for (cfg, hidden, layers) in [(BASE, 768, 12), (LARGE, 1024, 24)] {
        let c = parse(cfg, VIT_PRE);
        assert_eq!(c.encoder.kind, EncoderKind::Vit);
        assert_eq!((c.encoder.hidden, c.encoder.layers), (hidden, layers));
        assert_eq!(c.encoder.seq(), 577);
        assert!(!c.encoder.qkv_bias);
        assert_eq!(c.decoder.act, DecoderAct::Gelu);
        // Absent keys take TrOCRConfig's defaults.
        assert!(!c.decoder.scale_embedding && c.decoder.layernorm_embedding);
        assert_eq!(c.decoder.cross_hidden, hidden);
        assert_eq!(
            c.preprocess.filter,
            rlx_core::image_preprocess::Filter::Bilinear
        );
    }
}

#[test]
fn enc_to_dec_projection_is_rejected() {
    let mut v: serde_json::Value = serde_json::from_str(BASE).unwrap();
    v["decoder"]["cross_attention_hidden_size"] = 1024.into();
    let p: serde_json::Value = serde_json::from_str(VIT_PRE).unwrap();
    assert!(TrOcrConfig::from_values(&v, Some(&p)).is_err());
}
