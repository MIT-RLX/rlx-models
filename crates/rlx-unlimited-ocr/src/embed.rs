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

//! Fuse LM token embeddings with projected vision features.

use anyhow::{Result, bail, ensure};

/// Replace `image_token_id` rows in `token_embeds` (row-major `[n_tokens, hidden]`)
/// with rows from `vision_embeds` (row-major `[n_vision, hidden]`), in order.
pub fn fuse_inputs_embeds(
    token_ids: &[u32],
    token_embeds: &mut [f32],
    hidden: usize,
    image_token_id: u32,
    vision_embeds: &[f32],
) -> Result<()> {
    ensure!(
        token_embeds.len() == token_ids.len() * hidden,
        "token_embeds length mismatch: {} vs {}*{}",
        token_embeds.len(),
        token_ids.len(),
        hidden
    );
    ensure!(
        vision_embeds.len().is_multiple_of(hidden),
        "vision_embeds length {} not a multiple of hidden {}",
        vision_embeds.len(),
        hidden
    );
    let n_vision = vision_embeds.len() / hidden;
    let mut used = 0usize;
    for (i, &tok) in token_ids.iter().enumerate() {
        if tok != image_token_id {
            continue;
        }
        if used >= n_vision {
            bail!("more image placeholder tokens than vision features ({n_vision})");
        }
        let dst = &mut token_embeds[i * hidden..(i + 1) * hidden];
        let src = &vision_embeds[used * hidden..(used + 1) * hidden];
        dst.copy_from_slice(src);
        used += 1;
    }
    ensure!(
        used == n_vision,
        "unused vision features: placed {used} of {n_vision}"
    );
    Ok(())
}

/// Index of the largest value in `logits` (ties resolved to the first index).
pub fn argmax_token(logits: &[f32]) -> u32 {
    debug_assert!(!logits.is_empty());
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_replaces_image_rows_in_order() {
        let token_ids = [1u32, 128_815, 128_815, 2];
        let hidden = 2;
        let mut embeds = vec![0f32; token_ids.len() * hidden];
        embeds[0] = 9.0;
        embeds[6] = 9.0;
        let vision = vec![1.0, 1.0, 2.0, 2.0];
        fuse_inputs_embeds(&token_ids, &mut embeds, hidden, 128_815, &vision).unwrap();
        assert_eq!(&embeds[2..4], &[1.0, 1.0]);
        assert_eq!(&embeds[4..6], &[2.0, 2.0]);
        assert_eq!(embeds[0], 9.0);
        assert_eq!(embeds[6], 9.0);
    }

    #[test]
    fn argmax_picks_first_max() {
        assert_eq!(argmax_token(&[0.1, 0.9, 0.9, 0.2]), 1);
    }
}
