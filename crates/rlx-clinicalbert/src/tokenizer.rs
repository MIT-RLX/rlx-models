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

//! WordPiece tokenizer wrapper — drops the HF `tokenizer.json` straight into
//! a typed encoder that emits the four flat F32 buffers expected by
//! [`crate::ClinicalBertRunner::forward`].

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Tokenizer with helpers tailored to the [`ClinicalBertRunner`] input layout.
pub struct ClinicalBertTokenizer {
    inner: Tokenizer,
}

/// Encoded batch — flat row-major `[batch, seq]` buffers as F32.
pub struct EncodedBatch {
    pub input_ids: Vec<f32>,
    pub attention_mask: Vec<f32>,
    pub token_type_ids: Vec<f32>,
    pub position_ids: Vec<f32>,
    pub batch: usize,
    pub seq: usize,
}

impl ClinicalBertTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("rlx-clinicalbert: tokenizer.from_file: {e}"))?;
        Ok(Self { inner })
    }

    /// Load `tokenizer.json` from a directory or next to a weights file.
    pub fn from_dir_or_sibling(path: &Path) -> Result<Self> {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| std::path::PathBuf::from("."))
        };
        #[cfg(feature = "prepare")]
        if !dir.join("tokenizer.json").is_file() {
            crate::prepare::prepare_clinicalbert_dir(&dir)?;
        }
        let tok = dir.join("tokenizer.json");
        Self::from_file(&tok).with_context(|| format!("loading {tok:?}"))
    }

    /// Encode a batch of texts → padded `[batch, seq]` buffers.
    ///
    /// `seq` must match the runner's compiled sequence length. Texts longer
    /// than `seq` are truncated; shorter texts are right-padded with `[PAD]`.
    pub fn encode_batch(&self, texts: &[&str], seq: usize) -> Result<EncodedBatch> {
        let inputs: Vec<tokenizers::EncodeInput> = texts.iter().map(|t| (*t).into()).collect();
        self.encode_inputs(inputs, seq)
    }

    /// Encode a batch of `(premise, hypothesis)` pairs into the standard
    /// `[CLS] A [SEP] B [SEP]` BERT layout. `token_type_ids` are 0 for
    /// premise positions and 1 for hypothesis positions (post-tokenizer
    /// template). Used for NLI / sentence-pair classification.
    pub fn encode_pairs_batch(&self, pairs: &[(&str, &str)], seq: usize) -> Result<EncodedBatch> {
        let inputs: Vec<tokenizers::EncodeInput> = pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()).into())
            .collect();
        self.encode_inputs(inputs, seq)
    }

    fn encode_inputs(
        &self,
        inputs: Vec<tokenizers::EncodeInput>,
        seq: usize,
    ) -> Result<EncodedBatch> {
        let encodings = self
            .inner
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("rlx-clinicalbert: encode_batch: {e}"))?;

        let batch = encodings.len();
        let mut input_ids = vec![0f32; batch * seq];
        let mut attention_mask = vec![0f32; batch * seq];
        let mut token_type_ids = vec![0f32; batch * seq];
        let mut position_ids = vec![0f32; batch * seq];

        for (bi, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let take = ids.len().min(seq);
            for si in 0..take {
                input_ids[bi * seq + si] = ids[si] as f32;
                attention_mask[bi * seq + si] = mask[si] as f32;
                token_type_ids[bi * seq + si] = types[si] as f32;
            }
            for si in 0..seq {
                position_ids[bi * seq + si] = si as f32;
            }
        }

        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            batch,
            seq,
        })
    }
}
