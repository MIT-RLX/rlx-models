// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Multimodal wrapper glue — splicing projected vision tokens into the text
//! embedding stream (`KimiK3ForConditionalGeneration._merge_input_ids_with_image_features`).
//!
//! The processor pre-expands each image's `media_placeholder_token_id` to one
//! token per projected patch, so at the model surface the placeholder rows are
//! contiguous and count exactly `vision_tokens.len() / text_hidden`. We embed the
//! token ids normally, then overwrite the placeholder rows in order with the
//! `[M, text_hidden]` vision block.

use anyhow::{Result, ensure};

/// Overwrite the embedding rows at `media_placeholder_token_id` positions (in
/// order) with the projected vision-token rows. `embeds` is `[seq, hidden]`
/// row-major; `vision` is `[M, hidden]`; there must be exactly `M` placeholder
/// tokens in `input_ids`.
pub fn merge_text_and_vision_embds(
    embeds: &mut [f32],
    input_ids: &[i64],
    hidden: usize,
    vision: &[f32],
    media_placeholder_token_id: i64,
) -> Result<()> {
    ensure!(
        embeds.len() == input_ids.len() * hidden,
        "embeds shape mismatch"
    );
    ensure!(
        vision.len().is_multiple_of(hidden),
        "vision not a multiple of hidden"
    );
    let m = vision.len() / hidden;
    let n_ph = input_ids
        .iter()
        .filter(|&&t| t == media_placeholder_token_id)
        .count();
    ensure!(
        n_ph == m,
        "placeholder count {n_ph} != vision tokens {m} — the processor must pre-expand the placeholder"
    );
    let mut vi = 0usize;
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok == media_placeholder_token_id {
            embeds[i * hidden..(i + 1) * hidden]
                .copy_from_slice(&vision[vi * hidden..(vi + 1) * hidden]);
            vi += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splices_vision_rows_at_placeholders() {
        let hidden = 2;
        let ph = 163605;
        // ids: text, PH, PH, text  → 2 placeholder rows.
        let ids = [10i64, ph, ph, 11];
        let mut embeds = vec![
            1.0, 1.0, // text 0
            0.0, 0.0, // placeholder (to be overwritten)
            0.0, 0.0, // placeholder
            2.0, 2.0, // text 3
        ];
        let vision = vec![7.0, 8.0, 9.0, 10.0]; // [2, 2]
        merge_text_and_vision_embds(&mut embeds, &ids, hidden, &vision, ph).unwrap();
        assert_eq!(embeds, vec![1.0, 1.0, 7.0, 8.0, 9.0, 10.0, 2.0, 2.0]);
    }

    #[test]
    fn rejects_placeholder_count_mismatch() {
        let ids = [10i64, 163605, 11];
        let mut embeds = vec![0.0; 3 * 2];
        let vision = vec![1.0; 2 * 2]; // 2 vision tokens but only 1 placeholder
        assert!(merge_text_and_vision_embds(&mut embeds, &ids, 2, &vision, 163605).is_err());
    }
}
