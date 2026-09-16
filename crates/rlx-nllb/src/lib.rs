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

//! Meta NLLB-200 / M2M100 encoder–decoder machine translation for RLX.
//!
//! Text-only BART-style post-norm stack (ReLU FFN, `scale_embedding`, learned
//! positions with offset 2) matching
//! [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M).

mod builder;
pub mod cli;
pub mod config;
pub mod flow;
pub mod generate;
mod language;
pub mod runner;
pub mod tokenizer;
pub mod weight_source;
pub mod weights;

pub use config::{HF_DISTILLED_600M, NllbConfig};
pub use generate::GenerateConfig;
pub use runner::NllbModel;
pub use tokenizer::NllbTokenizer;

use anyhow::{Result, bail};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Map common ISO / English names (and already-canonical FLORES codes) to
/// NLLB FLORES-200 codes.
pub fn flores_code(lang: &str) -> Option<&'static str> {
    let s = lang.trim();
    let lower = s.to_ascii_lowercase();
    // Already a FLORES code present in the table values.
    for &(_, code) in FLORES_ALIASES {
        if code.eq_ignore_ascii_case(s) {
            return Some(code);
        }
    }
    FLORES_ALIASES
        .iter()
        .find(|(k, _)| *k == lower.as_str())
        .map(|(_, v)| *v)
}

/// Alias table used by [`flores_code`] (ISO / English → FLORES-200).
pub const FLORES_ALIASES: &[(&str, &str)] = &[
    ("en", "eng_Latn"),
    ("eng", "eng_Latn"),
    ("english", "eng_Latn"),
    ("fr", "fra_Latn"),
    ("fra", "fra_Latn"),
    ("french", "fra_Latn"),
    ("de", "deu_Latn"),
    ("deu", "deu_Latn"),
    ("german", "deu_Latn"),
    ("es", "spa_Latn"),
    ("spa", "spa_Latn"),
    ("spanish", "spa_Latn"),
    ("zh", "zho_Hans"),
    ("zho", "zho_Hans"),
    ("chinese", "zho_Hans"),
    ("zh-cn", "zho_Hans"),
    ("zh-hans", "zho_Hans"),
    ("zh-tw", "zho_Hant"),
    ("zh-hant", "zho_Hant"),
    ("zho_hant", "zho_Hant"),
    ("ja", "jpn_Jpan"),
    ("jpn", "jpn_Jpan"),
    ("japanese", "jpn_Jpan"),
    ("ko", "kor_Hang"),
    ("kor", "kor_Hang"),
    ("korean", "kor_Hang"),
    ("it", "ita_Latn"),
    ("ita", "ita_Latn"),
    ("italian", "ita_Latn"),
    ("pt", "por_Latn"),
    ("por", "por_Latn"),
    ("portuguese", "por_Latn"),
    ("ru", "rus_Cyrl"),
    ("rus", "rus_Cyrl"),
    ("russian", "rus_Cyrl"),
    ("ar", "arb_Arab"),
    ("ara", "arb_Arab"),
    ("arabic", "arb_Arab"),
    ("hi", "hin_Deva"),
    ("hin", "hin_Deva"),
    ("hindi", "hin_Deva"),
    ("nl", "nld_Latn"),
    ("nld", "nld_Latn"),
    ("dutch", "nld_Latn"),
    ("pl", "pol_Latn"),
    ("pol", "pol_Latn"),
    ("polish", "pol_Latn"),
    ("tr", "tur_Latn"),
    ("tur", "tur_Latn"),
    ("turkish", "tur_Latn"),
    ("vi", "vie_Latn"),
    ("vie", "vie_Latn"),
    ("vietnamese", "vie_Latn"),
    ("th", "tha_Thai"),
    ("tha", "tha_Thai"),
    ("thai", "tha_Thai"),
    ("uk", "ukr_Cyrl"),
    ("ukr", "ukr_Cyrl"),
    ("ukrainian", "ukr_Cyrl"),
    ("sv", "swe_Latn"),
    ("swe", "swe_Latn"),
    ("swedish", "swe_Latn"),
    ("cs", "ces_Latn"),
    ("ces", "ces_Latn"),
    ("czech", "ces_Latn"),
    ("ro", "ron_Latn"),
    ("ron", "ron_Latn"),
    ("romanian", "ron_Latn"),
    ("hu", "hun_Latn"),
    ("hun", "hun_Latn"),
    ("hungarian", "hun_Latn"),
    ("fi", "fin_Latn"),
    ("fin", "fin_Latn"),
    ("finnish", "fin_Latn"),
    ("el", "ell_Grek"),
    ("ell", "ell_Grek"),
    ("greek", "ell_Grek"),
    ("he", "heb_Hebr"),
    ("heb", "heb_Hebr"),
    ("hebrew", "heb_Hebr"),
    ("id", "ind_Latn"),
    ("ind", "ind_Latn"),
    ("indonesian", "ind_Latn"),
    ("ms", "zsm_Latn"),
    ("zsm", "zsm_Latn"),
    ("malay", "zsm_Latn"),
    ("bn", "ben_Beng"),
    ("ben", "ben_Beng"),
    ("bengali", "ben_Beng"),
    ("fa", "pes_Arab"),
    ("pes", "pes_Arab"),
    ("persian", "pes_Arab"),
    ("farsi", "pes_Arab"),
];

/// Resolve a language string to a FLORES-200 code (owned).
pub fn resolve_flores(lang: &str) -> Result<String> {
    if let Some(c) = flores_code(lang) {
        return Ok(c.to_string());
    }
    let s = lang.trim();
    // Accept opaque `xxx_Script` FLORES-looking codes for langs not in the alias table.
    if s.contains('_') && s.len() >= 6 {
        return Ok(s.to_string());
    }
    bail!("unknown language `{lang}` (try FLORES code like eng_Latn)")
}

/// High-level runner: weights + tokenizer + device.
pub struct NllbRunner {
    model: NllbModel,
}

impl NllbRunner {
    pub fn builder() -> NllbRunnerBuilder {
        NllbRunnerBuilder::default()
    }

    pub fn model(&mut self) -> &mut NllbModel {
        &mut self.model
    }

    pub fn config(&self) -> &NllbConfig {
        self.model.config()
    }

    pub fn translate(
        &mut self,
        src_text: &str,
        src_lang: &str,
        tgt_lang: &str,
        opts: &GenerateConfig,
    ) -> Result<String> {
        self.model.translate(src_text, src_lang, tgt_lang, opts)
    }
}

#[derive(Debug, Default)]
pub struct NllbRunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    config: Option<NllbConfig>,
}

impl NllbRunnerBuilder {
    pub fn weights(mut self, path: impl AsRef<Path>) -> Self {
        self.weights = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn model_dir(self, path: impl AsRef<Path>) -> Self {
        self.weights(path)
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn config(mut self, cfg: NllbConfig) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn build(self) -> Result<NllbRunner> {
        let dir = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("nllb: --weights / .weights(path) required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let cfg = if let Some(c) = self.config {
            c
        } else if dir.is_dir() && dir.join("config.json").is_file() {
            NllbConfig::from_hf_config_json(&dir.join("config.json"))?
        } else {
            NllbConfig::distilled_600m()
        };
        let model = NllbModel::load(&dir, cfg, device)?;
        if model.tokenizer().is_none() {
            bail!(
                "nllb: tokenizer.json not found beside weights at {}",
                dir.display()
            );
        }
        Ok(NllbRunner { model })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_source::CloningWeightSource;
    use crate::weights::lang as lk;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    #[test]
    fn flores_en_fr() {
        assert_eq!(flores_code("en"), Some("eng_Latn"));
        assert_eq!(flores_code("fr"), Some("fra_Latn"));
        assert_eq!(flores_code("eng_Latn"), Some("eng_Latn"));
        assert_eq!(flores_code("zho_Hans"), Some("zho_Hans"));
        assert_eq!(flores_code("swedish"), Some("swe_Latn"));
        assert!(flores_code("xx").is_none());
    }

    #[test]
    fn distilled_preset_dims() {
        let cfg = NllbConfig::distilled_600m();
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.encoder_layers, 12);
        assert_eq!(cfg.decoder_layers, 12);
        assert_eq!(cfg.encoder_attention_heads, 16);
        assert_eq!(cfg.encoder_ffn_dim, 4096);
        assert_eq!(cfg.vocab_size, 256_206);
        assert_eq!(cfg.max_position_embeddings, 1024);
        assert!(cfg.scale_embedding);
        assert_eq!(cfg.activation_function, "relu");
        assert_eq!(cfg.decoder_start_token_id, 2);
        assert_eq!(cfg.pad_token_id, 1);
        assert!((cfg.embed_scale() - 32.0).abs() < 1e-5);
    }

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 0.02 - 0.01
            })
            .collect()
    }

    fn put(
        map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
        key: &str,
        shape: &[usize],
        seed: u32,
    ) {
        let n: usize = shape.iter().product();
        map.insert(key.to_string(), (rand_vec(n, seed), shape.to_vec()));
    }

    fn synthetic_tiny_weights(cfg: &NllbConfig) -> WeightMap {
        let mut t = HashMap::new();
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let ffn = cfg.encoder_ffn_dim;
        let pos = cfg.max_position_embeddings + NllbConfig::POS_OFFSET;
        put(&mut t, lk::SHARED, &[v, d], 1);
        put(&mut t, &lk::enc_embed_positions(), &[pos, d], 2);
        put(&mut t, &lk::dec_embed_positions(), &[pos, d], 3);
        put(&mut t, &lk::enc_layernorm_embedding_w(), &[d], 4);
        put(&mut t, &lk::enc_layernorm_embedding_b(), &[d], 5);
        put(&mut t, &lk::dec_layernorm_embedding_w(), &[d], 6);
        put(&mut t, &lk::dec_layernorm_embedding_b(), &[d], 7);
        // Optional final norms present.
        put(&mut t, &lk::enc_final_layer_norm_w(), &[d], 8);
        put(&mut t, &lk::enc_final_layer_norm_b(), &[d], 9);
        put(&mut t, &lk::dec_final_layer_norm_w(), &[d], 10);
        put(&mut t, &lk::dec_final_layer_norm_b(), &[d], 11);

        let mut seed = 100u32;
        for layer in 0..cfg.encoder_layers {
            let p = |s: &str| lk::enc_layer(layer, s);
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
            ] {
                put(&mut t, &p(&format!("{name}.weight")), &[d, d], seed);
                seed += 1;
                put(&mut t, &p(&format!("{name}.bias")), &[d], seed);
                seed += 1;
            }
            put(&mut t, &p("self_attn_layer_norm.weight"), &[d], seed);
            seed += 1;
            put(&mut t, &p("self_attn_layer_norm.bias"), &[d], seed);
            seed += 1;
            put(&mut t, &p("fc1.weight"), &[ffn, d], seed);
            seed += 1;
            put(&mut t, &p("fc1.bias"), &[ffn], seed);
            seed += 1;
            put(&mut t, &p("fc2.weight"), &[d, ffn], seed);
            seed += 1;
            put(&mut t, &p("fc2.bias"), &[d], seed);
            seed += 1;
            put(&mut t, &p("final_layer_norm.weight"), &[d], seed);
            seed += 1;
            put(&mut t, &p("final_layer_norm.bias"), &[d], seed);
            seed += 1;
        }
        for layer in 0..cfg.decoder_layers {
            let p = |s: &str| lk::dec_layer(layer, s);
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "encoder_attn.q_proj",
                "encoder_attn.k_proj",
                "encoder_attn.v_proj",
                "encoder_attn.out_proj",
            ] {
                put(&mut t, &p(&format!("{name}.weight")), &[d, d], seed);
                seed += 1;
                put(&mut t, &p(&format!("{name}.bias")), &[d], seed);
                seed += 1;
            }
            put(&mut t, &p("self_attn_layer_norm.weight"), &[d], seed);
            seed += 1;
            put(&mut t, &p("self_attn_layer_norm.bias"), &[d], seed);
            seed += 1;
            put(&mut t, &p("encoder_attn_layer_norm.weight"), &[d], seed);
            seed += 1;
            put(&mut t, &p("encoder_attn_layer_norm.bias"), &[d], seed);
            seed += 1;
            put(&mut t, &p("fc1.weight"), &[ffn, d], seed);
            seed += 1;
            put(&mut t, &p("fc1.bias"), &[ffn], seed);
            seed += 1;
            put(&mut t, &p("fc2.weight"), &[d, ffn], seed);
            seed += 1;
            put(&mut t, &p("fc2.bias"), &[d], seed);
            seed += 1;
            put(&mut t, &p("final_layer_norm.weight"), &[d], seed);
            seed += 1;
            put(&mut t, &p("final_layer_norm.bias"), &[d], seed);
            seed += 1;
        }
        WeightMap::from_tensors(t)
    }

    #[test]
    fn graph_builders_compile_with_cloning_source() {
        let cfg = NllbConfig::tiny();
        let wm = synthetic_tiny_weights(&cfg);
        let mut src = CloningWeightSource(&wm);
        let enc = flow::build_encoder_built(&cfg, &mut src, 1, 4).expect("encoder build");
        assert!(!enc.params.is_empty());
        let mut src = CloningWeightSource(&wm);
        let dec = flow::build_decoder_hidden_built(&cfg, &mut src, 1, 4, 4).expect("decoder build");
        assert!(!dec.params.is_empty());
    }

    #[test]
    fn synthetic_encode_decode_logits_smoke() {
        let cfg = NllbConfig::tiny();
        let wm = synthetic_tiny_weights(&cfg);
        let mut model =
            NllbModel::from_weight_map(wm, cfg.clone(), Device::Cpu, None).expect("load");
        let ids = vec![3u32, 4, 5, 2];
        let enc = model.encode_tokens(&ids).expect("encode");
        assert_eq!(enc.len(), ids.len() * cfg.d_model);
        assert!(enc.iter().all(|v| v.is_finite()));
        let dec_ids = vec![cfg.decoder_start_token_id, 7];
        let logits = model
            .decode_logits(&dec_ids, &enc, ids.len(), 8)
            .expect("decode");
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn runner_builder_requires_weights() {
        let err = match NllbRunner::builder().build() {
            Ok(_) => panic!("expected missing-weights error"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("weights"), "{msg}");
        assert!(!msg.contains("hy-mt"), "{msg}");
        assert!(!msg.contains("not implemented"), "{msg}");
    }
}
