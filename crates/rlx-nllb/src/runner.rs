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

//! Staged execution: host-side embedding → M2M100 encode → decoder logits → generation.

use crate::config::NllbConfig;
use crate::generate::GenerateConfig;
use crate::tokenizer::NllbTokenizer;
use crate::weight_source::CloningWeightSource;
use crate::weights::lang as lk;
use anyhow::{Result, anyhow, bail};
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
use rayon::prelude::*;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::path::Path;

/// Loaded NLLB / M2M100 model with lazily compiled graphs.
///
/// The decoder is compiled once per length bucket and reused across steps by
/// padding the prefix — the causal mask keeps each real position independent of
/// trailing pad. Token embedding and LM head run host-side from `shared_table`.
pub struct NllbModel {
    cfg: NllbConfig,
    device: Device,
    weights: WeightMap,
    /// Host `model.shared.weight` `[vocab, d]` — embedding + tied LM head.
    shared_table: Vec<f32>,
    /// Host `final_logits_bias` `[vocab]` (zeros if absent).
    final_logits_bias: Vec<f32>,
    encoders: HashMap<usize, CompiledGraph>,
    /// Decoder hidden-state graphs keyed by `(bucket, enc_seq)`.
    decoders: HashMap<usize, CompiledGraph>,
    embed_scratch: Vec<f32>,
    tokenizer: Option<NllbTokenizer>,
}

impl NllbModel {
    pub fn config(&self) -> &NllbConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn tokenizer(&self) -> Option<&NllbTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Load from a safetensors checkpoint directory or file.
    pub fn load(weights_path: &Path, cfg: NllbConfig, device: Device) -> Result<Self> {
        rlx_core::validate_standard_device("nllb", device)?;
        let weights = if weights_path.is_dir() {
            WeightMap::from_safetensors_dir(weights_path)?
        } else {
            WeightMap::from_file(
                weights_path
                    .to_str()
                    .ok_or_else(|| anyhow!("non-UTF8 weights path"))?,
            )?
        };
        let tokenizer = if weights_path.is_dir() {
            let tk_path = weights_path.join("tokenizer.json");
            if tk_path.is_file() {
                Some(NllbTokenizer::from_file(&tk_path)?)
            } else {
                None
            }
        } else {
            None
        };
        Self::from_weight_map(weights, cfg, device, tokenizer)
    }

    /// Construct from an in-memory [`WeightMap`] (tests / synthetic checkpoints).
    pub fn from_weight_map(
        mut weights: WeightMap,
        cfg: NllbConfig,
        device: Device,
        tokenizer: Option<NllbTokenizer>,
    ) -> Result<Self> {
        rlx_core::validate_standard_device("nllb", device)?;
        let shared_key = if weights.has(lk::SHARED) {
            lk::SHARED
        } else if weights.has(lk::ENC_EMBED_TOKENS) {
            lk::ENC_EMBED_TOKENS
        } else {
            bail!(
                "nllb: missing shared embedding (`{}` or `{}`)",
                lk::SHARED,
                lk::ENC_EMBED_TOKENS
            );
        };
        let (shared_table, shape) = weights.take(shared_key)?;
        if shape.len() != 2 || shape[0] != cfg.vocab_size || shape[1] != cfg.d_model {
            bail!(
                "nllb: shared embedding shape {shape:?} != [{}, {}]",
                cfg.vocab_size,
                cfg.d_model
            );
        }
        let final_logits_bias = weights
            .take(lk::FINAL_LOGITS_BIAS)
            .map(|(d, _)| d)
            .unwrap_or_else(|_| vec![0.0; cfg.vocab_size]);
        Ok(Self {
            cfg,
            device,
            weights,
            shared_table,
            final_logits_bias,
            encoders: HashMap::new(),
            decoders: HashMap::new(),
            embed_scratch: Vec::new(),
            tokenizer,
        })
    }

    /// Attach / replace the tokenizer (e.g. after loading weights from a file).
    pub fn set_tokenizer(&mut self, tokenizer: NllbTokenizer) {
        self.tokenizer = Some(tokenizer);
    }

    /// Embed token ids host-side from the shared table (`embed_scale`).
    pub fn embed_text(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let scale = self.cfg.embed_scale();
        let mut out = vec![0f32; token_ids.len() * d];
        for (i, &tok) in token_ids.iter().enumerate() {
            let src = (tok as usize) * d;
            let dst = i * d;
            for j in 0..d {
                out[dst + j] = self.shared_table[src + j] * scale;
            }
        }
        out
    }

    /// Encode token ids → encoder hidden `[seq · d]`.
    pub fn encode_tokens(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let embeds = self.embed_text(token_ids);
        self.encode(&embeds, token_ids.len())
    }

    /// Encoder over `inputs_embeds [seq · d]` → `encoder_hidden [seq · d]`.
    pub fn encode(&mut self, inputs_embeds: &[f32], seq: usize) -> Result<Vec<f32>> {
        if !self.encoders.contains_key(&seq) {
            let mut src = CloningWeightSource(&self.weights);
            let built = crate::flow::build_encoder_built(&self.cfg, &mut src, 1, seq)?;
            self.encoders
                .insert(seq, compile_built(built, self.device)?);
        }
        let g = self.encoders.get_mut(&seq).unwrap();
        let out = g.run(&[("inputs_embeds", inputs_embeds)]);
        out.into_iter()
            .next()
            .ok_or_else(|| anyhow!("encoder graph produced no output"))
    }

    /// Next-token logits for the last position of `token_ids`. Returns `[vocab]`.
    pub fn decode_logits(
        &mut self,
        token_ids: &[u32],
        encoder_hidden: &[f32],
        enc_seq: usize,
        cap: usize,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let cur = token_ids.len();
        debug_assert!(cur >= 1 && cur <= cap);
        let scale = self.cfg.embed_scale();
        let cap = bucket_len(cur).min(cap).max(cur);

        let key = cap * 1_000_000 + enc_seq;
        if !self.decoders.contains_key(&key) {
            let mut src = CloningWeightSource(&self.weights);
            let built =
                crate::flow::build_decoder_hidden_built(&self.cfg, &mut src, 1, cap, enc_seq)?;
            self.decoders
                .insert(key, compile_built(built, self.device)?);
        }

        let pad = self.cfg.pad_token_id as usize;
        self.embed_scratch.resize(cap * d, 0.0);
        for i in 0..cap {
            let tok = if i < cur { token_ids[i] as usize } else { pad };
            let src = tok * d;
            let dst = i * d;
            for j in 0..d {
                self.embed_scratch[dst + j] = self.shared_table[src + j] * scale;
            }
        }

        let hidden = {
            let g = self.decoders.get_mut(&key).unwrap();
            g.run(&[
                ("decoder_inputs_embeds", self.embed_scratch.as_slice()),
                ("encoder_hidden", encoder_hidden),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("decoder graph produced no output"))?
        };
        let row = &hidden[(cur - 1) * d..cur * d];
        Ok(self.lm_head(row))
    }

    /// Drop compiled encoder/decoder graphs (weights stay loaded).
    ///
    /// Long dub jobs compile a fresh graph per sequence length; unbounded cache
    /// can grow to tens of GB. Prefer [`Self::trim_compile_cache`] between cues
    /// so same-length buckets stay warm.
    pub fn evict_compile_cache(&mut self) {
        self.encoders.clear();
        self.decoders.clear();
    }

    /// Cap compile-cache size (default: 8 encoders / 16 decoders).
    ///
    /// Override with `RLX_NLLB_ENC_CACHE` / `RLX_NLLB_DEC_CACHE`. When over the
    /// limit, drop all graphs (weights stay); next cue rebuilds only what it needs.
    pub fn trim_compile_cache(&mut self) {
        let low_mem = std::env::var("RLX_LOW_MEMORY")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);
        let enc_default = if low_mem { 2usize } else { 8usize };
        let dec_default = if low_mem { 4usize } else { 16usize };
        let enc_cap = std::env::var("RLX_NLLB_ENC_CACHE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(enc_default)
            .clamp(1, 64);
        let dec_cap = std::env::var("RLX_NLLB_DEC_CACHE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(dec_default)
            .clamp(2, 128);
        if self.encoders.len() > enc_cap || self.decoders.len() > dec_cap {
            self.evict_compile_cache();
        }
    }

    /// Translate `src_text` from `src_lang` to `tgt_lang` (ISO or FLORES codes).
    pub fn translate(
        &mut self,
        src_text: &str,
        src_lang: &str,
        tgt_lang: &str,
        opts: &GenerateConfig,
    ) -> Result<String> {
        let src_code = crate::resolve_flores(src_lang)?;
        let tgt_code = crate::resolve_flores(tgt_lang)?;
        let (src_ids, tgt_lang_id) = {
            let tk = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| anyhow!("nllb: tokenizer required for translate()"))?;
            let src_ids = tk.encode(src_text, &src_code)?;
            let tgt_lang_id = tk.lang_token_id(&tgt_code)?;
            (src_ids, tgt_lang_id)
        };
        let enc = self.encode_tokens(&src_ids)?;
        let enc_seq = src_ids.len();
        let mut gencfg = opts.clone();
        gencfg.forced_bos_token_id = Some(tgt_lang_id);
        let out_ids = if gencfg.num_beams > 1 {
            self.generate_beam(&enc, enc_seq, &gencfg)?
        } else {
            self.generate_greedy(&enc, enc_seq, &gencfg)?
        };
        // Drop decoder_start + forced lang code; strip trailing EOS.
        let start = if out_ids.len() >= 2 { 2 } else { out_ids.len() };
        let mut body = out_ids[start..].to_vec();
        if body.last() == Some(&self.cfg.eos_token_id) {
            body.pop();
        }
        let out = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("nllb: tokenizer required for translate()"))?
            .decode(&body)?;
        // Keep warm graphs across cues; only drop when over RSS-safe caps.
        self.trim_compile_cache();
        Ok(out)
    }

    /// Tied LM head + `final_logits_bias` for a single hidden row `[d]`.
    fn lm_head(&self, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let vocab = self.cfg.vocab_size;
        debug_assert_eq!(hidden_row.len(), d);
        let mut logits = vec![0f32; vocab];
        // logits = shared_table[vocab,d] @ hidden[d]  (row-major GEMV)
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            accelerate_sgemv_nn(&self.shared_table, hidden_row, &mut logits, vocab, d);
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            let table = &self.shared_table;
            logits.par_iter_mut().enumerate().for_each(|(v, out)| {
                let row = &table[v * d..v * d + d];
                let mut acc = 0f32;
                for j in 0..d {
                    acc += hidden_row[j] * row[j];
                }
                *out = acc;
            });
        }
        let bias = &self.final_logits_bias;
        if bias.len() == vocab {
            for (o, &b) in logits.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }
        logits
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn accelerate_sgemv_nn(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    // CblasRowMajor + NoTrans: y = A·x, A is m×n with lda=n.
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemv(
            layout: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            x: *const f32,
            incx: i32,
            beta: f32,
            y: *mut f32,
            incy: i32,
        );
    }
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), m);
    unsafe {
        cblas_sgemv(
            ROW_MAJOR,
            NO_TRANS,
            m as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            n as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

/// Decode graph length buckets: smallest entry ≥ `n`.
fn bucket_len(n: usize) -> usize {
    const BUCKETS: [usize; 11] = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256];
    for &b in &BUCKETS {
        if b >= n {
            return b;
        }
    }
    n.next_power_of_two()
}
