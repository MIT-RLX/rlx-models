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

//! Text tokenization and the Grounding DINO phrase self-attention mask.

/// BERT special tokens that delimit phrases (`[CLS]`, `[SEP]`, `.`, `?`).
pub const SPECIAL_TOKENS: [u32; 4] = [101, 102, 1012, 1029];

/// Tokenized prompt plus the derived attention structures.
#[derive(Debug, Clone)]
pub struct TextTokens {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    /// `[L]` padding mask (1 = real token).
    pub attention_mask: Vec<u8>,
    /// `[L, L]` phrase self-attention mask (1 = allowed to attend).
    pub self_attn_mask: Vec<u8>,
    /// `[L]` per-phrase-reset position ids.
    pub position_ids: Vec<u32>,
    pub seq_len: usize,
}

/// Build the phrase block-diagonal self-attention mask and per-phrase position
/// ids, mirroring HF `generate_masks_with_special_tokens_and_transfer_map`.
///
/// Each run of tokens between two special tokens forms a block that attends
/// only within itself (the trailing special token is included in the block).
/// `[CLS]` and the final token attend only to themselves.
pub fn build_self_attention_mask(input_ids: &[u32]) -> (Vec<u8>, Vec<u32>) {
    let n = input_ids.len();
    let mut mask = vec![0u8; n * n];
    // Diagonal is always allowed.
    for i in 0..n {
        mask[i * n + i] = 1;
    }
    let mut position_ids = vec![0u32; n];
    let mut previous_col = 0usize;
    for (col, &tok) in input_ids.iter().enumerate() {
        if !SPECIAL_TOKENS.contains(&tok) {
            continue;
        }
        if col == 0 || col == n - 1 {
            // self-only (already set on the diagonal); position 0.
            position_ids[col] = 0;
        } else {
            // block (previous_col+1 ..= col) attends to itself.
            for r in (previous_col + 1)..=col {
                for c in (previous_col + 1)..=col {
                    mask[r * n + c] = 1;
                }
                position_ids[r] = (r - (previous_col + 1)) as u32;
            }
        }
        previous_col = col;
    }
    (mask, position_ids)
}

/// Assemble [`TextTokens`] from raw `input_ids` (already includes `[CLS]`/`[SEP]`).
pub fn text_tokens_from_ids(input_ids: Vec<u32>) -> TextTokens {
    let seq_len = input_ids.len();
    let (self_attn_mask, position_ids) = build_self_attention_mask(&input_ids);
    TextTokens {
        token_type_ids: vec![0u32; seq_len],
        attention_mask: vec![1u8; seq_len],
        self_attn_mask,
        position_ids,
        input_ids,
        seq_len,
    }
}

/// Tokenize a free-text prompt with a BERT wordpiece tokenizer loaded from
/// `tokenizer.json`. The prompt is lowercased and `.`-terminated to match the
/// HF `GroundingDinoProcessor` convention.
#[cfg(feature = "tokenizer")]
pub fn tokenize_prompt(
    tokenizer_json: &std::path::Path,
    prompt: &str,
) -> anyhow::Result<TextTokens> {
    use anyhow::anyhow;
    let tk = tokenizers::Tokenizer::from_file(tokenizer_json)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let mut caption = prompt.trim().to_lowercase();
    if !caption.ends_with('.') {
        caption.push('.');
    }
    let enc = tk
        .encode(caption, true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let input_ids: Vec<u32> = enc.get_ids().to_vec();
    Ok(text_tokens_from_ids(input_ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_is_block_diagonal_per_phrase() {
        // [CLS] a b . c . [SEP]  → ids with two phrases "a b" and "c".
        // 101=CLS, 1012='.', 102=SEP; word ids are arbitrary non-special.
        let ids = vec![101, 2000, 2001, 1012, 2002, 1012, 102];
        let n = ids.len();
        let (mask, pos) = build_self_attention_mask(&ids);
        let m = |r: usize, c: usize| mask[r * n + c] == 1;

        // CLS attends only to itself.
        assert!(m(0, 0));
        assert!(!m(0, 1));
        // Phrase "a b ." block = cols 1..=3 attend mutually.
        for r in 1..=3 {
            for c in 1..=3 {
                assert!(m(r, c), "block1 ({r},{c})");
            }
        }
        // Token in phrase 1 must NOT attend to phrase 2.
        assert!(!m(1, 4));
        // Phrase "c ." block = cols 4..=5.
        assert!(m(4, 5) && m(5, 4));
        assert!(!m(4, 1));
        // position ids reset within each phrase block.
        assert_eq!(pos[1], 0);
        assert_eq!(pos[2], 1);
        assert_eq!(pos[3], 2);
        assert_eq!(pos[4], 0);
        assert_eq!(pos[5], 1);
    }
}
