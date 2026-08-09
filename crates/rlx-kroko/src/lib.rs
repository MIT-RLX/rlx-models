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

//! # rlx-kroko
//!
//! **Kroko** streaming ASR — a k2/sherpa-style **Zipformer2** transducer with a
//! **stateless context-2 predictor** (blank id `0`), 80-dim fbank features, chunked
//! streaming (chunk 269 / shift 256, subsampling factor 4).
//!
//! The greedy decode loop is shared: this crate reuses
//! [`rlx_audio_blocks::decoders::transducer`] (the stateless-predictor greedy
//! search) and contributes the Kroko-specific config + the glue that turns a
//! Zipformer2 encoder + stateless decoder/joint into a
//! [`StatelessTransducerCore`].
//!
//! Status: config + decode wiring, CPU smoke. The Zipformer2 encoder graph
//! (Conv2dSubsampling + streaming Zipformer2 stacks) and the stateless
//! decoder/joint weights are wired end-to-end once a Kroko package is available;
//! the encoder can reuse the streaming Conformer machinery in
//! `rlx-wav2vec2-bert` / `rlx-nemotron-asr`.

use anyhow::{Result, ensure};
pub use rlx_audio_blocks::decoders::transducer::{
    GreedyTransducerResult, StatelessTransducerCore, TransducerStep,
    run_stateless_transducer_greedy,
};

/// Kroko ASR config (ported from audio.cpp `KrokoASRConfig`). The `Vec` fields are
/// the per-stack Zipformer2 metadata, filled in from the package `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct KrokoConfig {
    pub model_type: String,
    pub variant: String,
    pub language: String,
    pub sample_rate: usize,
    pub feature_dim: usize,
    pub chunk_size: usize,
    pub chunk_shift: usize,
    pub subsampling_factor: usize,
    pub vocab_size: usize,
    pub context_size: usize,
    pub blank_id: i32,
    pub unk_id: i32,
    pub encoder_dims: Vec<usize>,
    pub query_head_dims: Vec<usize>,
    pub value_head_dims: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub num_encoder_layers: Vec<usize>,
    pub cnn_module_kernels: Vec<usize>,
    pub left_context_len: Vec<usize>,
    pub downsampling_factors: Vec<usize>,
}

impl Default for KrokoConfig {
    fn default() -> Self {
        Self {
            model_type: "zipformer2".to_string(),
            variant: "Kroko-Community-Streaming".to_string(),
            language: "auto".to_string(),
            sample_rate: 16_000,
            feature_dim: 80,
            chunk_size: 269,
            chunk_shift: 256,
            subsampling_factor: 4,
            vocab_size: 0,
            context_size: 2,
            blank_id: 0,
            unk_id: 2,
            encoder_dims: Vec::new(),
            query_head_dims: Vec::new(),
            value_head_dims: Vec::new(),
            num_heads: Vec::new(),
            num_encoder_layers: Vec::new(),
            cnn_module_kernels: Vec::new(),
            left_context_len: Vec::new(),
            downsampling_factors: Vec::new(),
        }
    }
}

impl KrokoConfig {
    /// Enforce the invariants audio.cpp asserts for Kroko packages.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "zipformer2",
            "Kroko ASR currently supports only Zipformer2 packages, got {}",
            self.model_type
        );
        ensure!(
            self.subsampling_factor == 4,
            "Kroko ASR expects subsampling_factor 4"
        );
        ensure!(
            self.context_size == 2 && self.blank_id == 0,
            "Kroko ASR expects a context-2 stateless predictor with blank id 0"
        );
        Ok(())
    }
}

/// Greedy decoding tuning (ported from audio.cpp `KrokoDecoderOptions`).
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderOptions {
    pub method: DecodingMethod,
    pub max_active_paths: usize,
    pub blank_penalty: f32,
    pub hotwords_score: f32,
    pub hotwords: Vec<Vec<i32>>,
    /// Max non-blank symbols emitted per encoder frame in greedy search.
    pub max_symbols_per_frame: usize,
}

impl Default for DecoderOptions {
    fn default() -> Self {
        Self {
            method: DecodingMethod::GreedySearch,
            max_active_paths: 4,
            blank_penalty: 0.0,
            hotwords_score: 1.5,
            hotwords: Vec::new(),
            max_symbols_per_frame: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodingMethod {
    GreedySearch,
    ModifiedBeamSearch,
}

/// Greedy-decode a Zipformer2 encoder output using the shared stateless-transducer
/// loop, with Kroko's context size and blank id taken from `cfg`.
pub fn greedy_decode<C: StatelessTransducerCore + ?Sized>(
    cfg: &KrokoConfig,
    core: &mut C,
    encoder_output: &[f32],
    hidden_size: usize,
    opts: &DecoderOptions,
) -> Result<GreedyTransducerResult> {
    ensure!(hidden_size > 0, "hidden_size must be positive");
    ensure!(
        encoder_output.len().is_multiple_of(hidden_size),
        "encoder_output length {} not a multiple of hidden_size {hidden_size}",
        encoder_output.len()
    );
    let frames = encoder_output.len() / hidden_size;
    run_stateless_transducer_greedy(
        core,
        encoder_output,
        frames,
        hidden_size,
        cfg.blank_id,
        cfg.context_size,
        opts.max_symbols_per_frame,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_audio_cpp() {
        let c = KrokoConfig::default();
        assert_eq!(c.model_type, "zipformer2");
        assert_eq!(c.sample_rate, 16_000);
        assert_eq!(c.feature_dim, 80);
        assert_eq!(c.chunk_size, 269);
        assert_eq!(c.chunk_shift, 256);
        assert_eq!(c.subsampling_factor, 4);
        assert_eq!(c.context_size, 2);
        assert_eq!(c.blank_id, 0);
        assert_eq!(c.unk_id, 2);
    }

    #[test]
    fn validate_enforces_zipformer2_invariants() {
        KrokoConfig::default().validate().unwrap();
        let bad = KrokoConfig {
            subsampling_factor: 2,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        let bad2 = KrokoConfig {
            blank_id: 1,
            ..Default::default()
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn decoder_option_defaults() {
        let o = DecoderOptions::default();
        assert_eq!(o.method, DecodingMethod::GreedySearch);
        assert_eq!(o.max_active_paths, 4);
        assert_eq!(o.hotwords_score, 1.5);
    }

    /// Minimal stateless core: emits the scripted labels, ignoring encoder/context.
    struct FakeCore {
        labels: Vec<i32>,
        cursor: usize,
    }
    impl StatelessTransducerCore for FakeCore {
        fn step_argmax(&mut self, _f: &[f32], _ctx: &[i32]) -> TransducerStep {
            let l = self.labels.get(self.cursor).copied().unwrap_or(0);
            self.cursor += 1;
            TransducerStep {
                label: l,
                score: 1.0,
            }
        }
    }

    #[test]
    fn greedy_decode_runs_through_shared_loop() {
        let cfg = KrokoConfig::default();
        let opts = DecoderOptions::default();
        // frame0: emit 11 then blank; frame1: emit 22 then blank.
        let mut core = FakeCore {
            labels: vec![11, 0, 22, 0],
            cursor: 0,
        };
        let hidden = 4;
        let enc = vec![0.0f32; 2 * hidden];
        let out = greedy_decode(&cfg, &mut core, &enc, hidden, &opts).unwrap();
        assert_eq!(out.token_ids, vec![11, 22]);
        assert_eq!(out.frame_indices, vec![0, 1]);
    }
}
