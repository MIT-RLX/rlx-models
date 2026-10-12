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

//! Multimodal prompt assembly — ChatML + vision placeholder tokens.

use crate::mrope::{
    ImageGrid, build_multimodal_mrope_sections, build_multimodal_mrope_sections_multi,
};
use crate::vision::VisionEncodeOutput;
use anyhow::{Result, bail, ensure};

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// Single-image placeholder in user prompts (replaced by vision token span).
pub const MEDIA_MARKER: &str = "<__media__>";
/// `<|im_end|>` as an id — the ChatML turn terminator, and `eos_token_id` in
/// every Qwen2.5-VL `config.json`. Held as an id rather than a string because a
/// decode loop has to compare against it without a tokenizer in hand.
pub const IM_END_TOKEN: u32 = 151645;

pub fn merge_text_and_vision_embd(
    token_embd: &[f32],
    vocab: usize,
    n_embd: usize,
    input_ids: &[u32],
    vision_embd: &[f32],
    vision_start_idx: usize,
    n_vision: usize,
) -> Vec<f32> {
    assert_eq!(token_embd.len(), input_ids.len() * n_embd);
    assert_eq!(vision_embd.len(), n_vision * n_embd);
    let _ = vocab;
    let seq = input_ids.len();
    assert!(vision_start_idx + n_vision <= seq);

    let mut out = token_embd.to_vec();
    for t in 0..n_vision {
        let dst = (vision_start_idx + t) * n_embd;
        let src = t * n_embd;
        out[dst..dst + n_embd].copy_from_slice(&vision_embd[src..src + n_embd]);
    }
    out
}

#[derive(Debug, Clone)]
pub struct MultimodalPrefill {
    pub hidden: Vec<f32>,
    pub mrope_sections: Vec<[usize; 4]>,
    pub last_token_idx: usize,
    pub seq: Vec<u32>,
    /// First `<|image_pad|>` index in `seq` (KV cache key index).
    ///
    /// With more than one image this is the first span's start and
    /// `n_vision_tokens` is the total, so together they cover the text between
    /// the images too. Only the AIF probe reads them, and it asks about one
    /// contiguous region by construction; ordinary generation does not consult
    /// them at all.
    pub vision_start_idx: usize,
    pub n_vision_tokens: usize,
}

pub struct MultimodalPrompt<'a> {
    pub prompt: &'a str,
    pub vision: &'a VisionEncodeOutput,
}

impl<'a> MultimodalPrompt<'a> {
    pub fn assemble(
        &self,
        tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
        token_embd_table: &[f32],
        n_embd: usize,
        text_start_pos: usize,
    ) -> Result<MultimodalPrefill> {
        assemble_many(
            self.prompt,
            std::slice::from_ref(self.vision),
            tokenizer,
            token_embd_table,
            n_embd,
            text_start_pos,
        )
    }
}

/// Assemble a prefill for a prompt carrying any number of images.
///
/// The prompt holds one [`MEDIA_MARKER`] per image, in the order the images
/// were attached, and each is replaced by that image's vision span. A count
/// that does not match is refused rather than guessed at: a reply about the
/// first of three pictures, presented as a reply about all three, is worse
/// than an error.
pub fn assemble_many(
    prompt: &str,
    visions: &[VisionEncodeOutput],
    mut tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
    token_embd_table: &[f32],
    n_embd: usize,
    text_start_pos: usize,
) -> Result<MultimodalPrefill> {
    let parts: Vec<&str> = prompt.split(MEDIA_MARKER).collect();
    if parts.len() != visions.len() + 1 {
        bail!(
            "prompt has {} `{MEDIA_MARKER}` marker(s) and {} image(s) were given",
            parts.len().saturating_sub(1),
            visions.len()
        );
    }

    let start_ids = tokenizer(VISION_START)?;
    let end_ids = tokenizer(VISION_END)?;

    // Built in one pass: the token sequence, where each image's embeddings go
    // in it, and the text runs the positions are derived from.
    let mut seq: Vec<u32> = Vec::new();
    let mut spans: Vec<(usize, &VisionEncodeOutput)> = Vec::with_capacity(visions.len());
    let mut text_runs: Vec<usize> = Vec::with_capacity(parts.len());
    let mut grids: Vec<ImageGrid> = Vec::with_capacity(visions.len());

    for (i, part) in parts.iter().enumerate() {
        let text_ids = tokenizer(part)?;
        seq.extend_from_slice(&text_ids);
        let Some(vision) = visions.get(i) else {
            // The last part: text with no image after it.
            text_runs.push(text_ids.len() + if i == 0 { 0 } else { end_ids.len() });
            break;
        };
        // `<|vision_start|>` is text as far as the positions are concerned,
        // and so is the `<|vision_end|>` that closed the previous image.
        text_runs.push(text_ids.len() + start_ids.len() + if i == 0 { 0 } else { end_ids.len() });
        seq.extend_from_slice(&start_ids);
        spans.push((seq.len(), vision));
        seq.extend(std::iter::repeat_n(0u32, vision.n_tokens));
        seq.extend_from_slice(&end_ids);
        grids.push(ImageGrid {
            nx: vision.grid_x,
            ny: vision.grid_y,
        });
    }

    let vocab = token_embd_table.len() / n_embd;
    let mut hidden = Vec::with_capacity(seq.len() * n_embd);
    for &tok in &seq {
        let row = tok as usize;
        ensure!(row < vocab, "token id {tok} out of vocab range {vocab}");
        let off = row * n_embd;
        hidden.extend_from_slice(&token_embd_table[off..off + n_embd]);
    }
    for (start, vision) in &spans {
        hidden = merge_text_and_vision_embd(
            &hidden,
            vocab,
            n_embd,
            &seq,
            &vision.embeddings,
            *start,
            vision.n_tokens,
        );
    }

    let mrope_sections = build_multimodal_mrope_sections_multi(&text_runs, &grids, text_start_pos);
    ensure!(
        mrope_sections.len() == seq.len(),
        "mrope sections {} != seq {}",
        mrope_sections.len(),
        seq.len()
    );

    Ok(MultimodalPrefill {
        last_token_idx: seq.len().saturating_sub(1),
        // The first image's span, which is what the KV cache key refers to.
        vision_start_idx: spans.first().map(|(s, _)| *s).unwrap_or(0),
        n_vision_tokens: spans.iter().map(|(_, v)| v.n_tokens).sum(),
        hidden,
        mrope_sections,
        seq,
    })
}

/// Assemble prefill from a fixed HF token sequence (parity / replay).
pub fn assemble_from_token_ids(
    input_ids: &[u32],
    vision_start_idx: usize,
    n_vision: usize,
    vision: &VisionEncodeOutput,
    token_embd_table: &[f32],
    n_embd: usize,
    text_start_pos: usize,
) -> Result<MultimodalPrefill> {
    let seq = input_ids.to_vec();
    ensure!(
        vision_start_idx + n_vision <= seq.len(),
        "vision span {vision_start_idx}+{n_vision} exceeds seq {}",
        seq.len()
    );
    ensure!(
        vision.n_tokens == n_vision,
        "vision token count {} != reference {n_vision}",
        vision.n_tokens
    );

    let vocab = token_embd_table.len() / n_embd;
    let mut hidden = Vec::with_capacity(seq.len() * n_embd);
    for &tok in &seq {
        let row = tok as usize;
        ensure!(row < vocab, "token id {tok} out of vocab range {vocab}");
        let off = row * n_embd;
        hidden.extend_from_slice(&token_embd_table[off..off + n_embd]);
    }
    hidden = merge_text_and_vision_embd(
        &hidden,
        vocab,
        n_embd,
        &seq,
        &vision.embeddings,
        vision_start_idx,
        n_vision,
    );

    let text_len_before = vision_start_idx;
    let text_len_after = seq.len() - vision_start_idx - n_vision;
    let mrope_sections = build_multimodal_mrope_sections(
        text_len_before,
        vision.grid_x,
        vision.grid_y,
        text_len_after,
        text_start_pos,
    );
    ensure!(
        mrope_sections.len() == seq.len(),
        "mrope sections {} != seq {}",
        mrope_sections.len(),
        seq.len()
    );

    Ok(MultimodalPrefill {
        hidden,
        mrope_sections,
        last_token_idx: seq.len().saturating_sub(1),
        seq,
        vision_start_idx,
        n_vision_tokens: n_vision,
    })
}

#[cfg(test)]
mod multi_tests {
    use super::*;

    /// A tokenizer that gives every word one id, so the arithmetic in
    /// `assemble_many` can be checked without a real vocabulary.
    fn words(text: &str) -> Result<Vec<u32>> {
        Ok(text.split_whitespace().map(|_| 5u32).collect())
    }

    fn vision(n_tokens: usize, nx: usize, ny: usize, n_embd: usize) -> VisionEncodeOutput {
        VisionEncodeOutput {
            embeddings: vec![1.0; n_tokens * n_embd],
            n_tokens,
            grid_x: nx,
            grid_y: ny,
        }
    }

    const N_EMBD: usize = 4;

    fn table() -> Vec<f32> {
        // 16 rows is enough for the single id the tokenizer above produces.
        vec![0.0; 16 * N_EMBD]
    }

    #[test]
    fn two_images_both_reach_the_sequence() {
        let v = [vision(4, 2, 2, N_EMBD), vision(2, 2, 1, N_EMBD)];
        let out = assemble_many(
            &format!("look at {MEDIA_MARKER} and {MEDIA_MARKER} please"),
            &v,
            words,
            &table(),
            N_EMBD,
            0,
        )
        .expect("two images assemble");
        assert_eq!(out.n_vision_tokens, 6, "both images are in the sequence");
        assert_eq!(out.mrope_sections.len(), out.seq.len());
        assert_eq!(out.hidden.len(), out.seq.len() * N_EMBD);
    }

    /// Each image's embeddings land in its own span. Writing both to the same
    /// place would answer about one picture twice.
    #[test]
    fn each_image_lands_in_its_own_span() {
        let mut first = vision(2, 2, 1, N_EMBD);
        first.embeddings = vec![1.0; 2 * N_EMBD];
        let mut second = vision(2, 2, 1, N_EMBD);
        second.embeddings = vec![2.0; 2 * N_EMBD];

        let out = assemble_many(
            &format!("{MEDIA_MARKER} then {MEDIA_MARKER}"),
            &[first, second],
            words,
            &table(),
            N_EMBD,
            0,
        )
        .expect("assembles");
        let ones = out.hidden.iter().filter(|x| **x == 1.0).count();
        let twos = out.hidden.iter().filter(|x| **x == 2.0).count();
        assert_eq!(ones, 2 * N_EMBD, "the first image is spliced in once");
        assert_eq!(twos, 2 * N_EMBD, "and so is the second");
    }

    #[test]
    fn a_marker_count_that_does_not_match_the_images_is_refused() {
        // Guessing would answer about one picture as though it were three.
        let v = [vision(2, 2, 1, N_EMBD), vision(2, 2, 1, N_EMBD)];
        let err = assemble_many(
            &format!("only one {MEDIA_MARKER} here"),
            &v,
            words,
            &table(),
            N_EMBD,
            0,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("1 `") && err.contains("2 image"), "{err}");
    }

    #[test]
    fn one_image_through_the_many_path_matches_the_single_one() {
        let v = vision(4, 2, 2, N_EMBD);
        let prompt = format!("describe {MEDIA_MARKER} now");
        let many = assemble_many(
            &prompt,
            std::slice::from_ref(&v),
            words,
            &table(),
            N_EMBD,
            0,
        )
        .expect("many");
        let one = MultimodalPrompt {
            prompt: &prompt,
            vision: &v,
        }
        .assemble(words, &table(), N_EMBD, 0)
        .expect("one");
        assert_eq!(many.seq, one.seq);
        assert_eq!(many.mrope_sections, one.mrope_sections);
        assert_eq!(many.vision_start_idx, one.vision_start_idx);
    }
}
