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

//! End-to-end text embedding helpers (tokenize → forward → pool).

use anyhow::Result;

use super::bert::RlxBertModel;
use super::pooling::{Pooling, pool_embeddings};
use super::tokenizer::BertTokenizer;

/// Embed texts with a compiled BERT model: tokenize, forward, pool, L2-normalize.
pub fn embed_with_rlx(
    model: &mut RlxBertModel,
    tokenizer: &BertTokenizer,
    texts: &[&str],
    pooling: Pooling,
) -> Result<Vec<Vec<f32>>> {
    let batch = tokenizer.encode_batch(texts)?;
    let b = texts.len();
    let s = batch.seq_len;
    let hs = model.hidden_size();

    model.recompile(b, s)?;

    let ids: Vec<f32> = batch
        .input_ids
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as f32))
        .collect();
    let mask: Vec<f32> = batch
        .attention_mask
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as f32))
        .collect();
    let tt: Vec<f32> = batch
        .token_type_ids
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as f32))
        .collect();
    let pos: Vec<f32> = (0..b).flat_map(|_| (0..s).map(|i| i as f32)).collect();

    let hidden = model.forward(&ids, &mask, &tt, &pos);
    let mask_refs: Vec<&[u32]> = batch.attention_mask.iter().map(|r| r.as_slice()).collect();
    Ok(pool_embeddings(&hidden, &mask_refs, b, s, hs, pooling))
}
