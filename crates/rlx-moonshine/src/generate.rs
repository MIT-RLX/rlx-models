// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Greedy autoregressive decode for Moonshine.

use crate::runner::MoonshineModel;
use anyhow::Result;

impl MoonshineModel {
    /// Greedy decode from `encoder_hidden [enc_seq · d]`.
    pub fn generate_greedy(&mut self, encoder_hidden: &[f32], enc_seq: usize) -> Result<Vec<u32>> {
        self.reset_decode_state();
        let max_len = self.config().max_position_embeddings.max(2);
        let eos = self.config().eos_token_id;
        let mut seq = vec![self.config().decoder_start_token_id];
        while seq.len() < max_len {
            let next = self.decode_next_token(&seq, encoder_hidden, enc_seq, max_len)?;
            seq.push(next);
            if next == eos {
                break;
            }
        }
        Ok(seq)
    }
}
