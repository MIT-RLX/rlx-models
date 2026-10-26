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

//! Multimodal prompt assembly — MRoPE sections + embedding merge.

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const MEDIA_MARKER: &str = "<__media__>";

/// Decoder MRoPE position for vision token `i`.
///
/// HF `mrope_section` is `[temporal, height, width]`. Tokens are row-major over
/// the output grid (`i / nx` = row/y, `i % nx` = col/x), matching Qwen2/3-VL
/// `get_rope_index` spatial layout (not llama.cpp's swapped sections).
pub fn image_decoder_pos(nx: usize, ny: usize, pos_0: usize, i: usize) -> [usize; 4] {
    let _ = ny;
    [pos_0, pos_0 + (i / nx), pos_0 + (i % nx), 0]
}

/// Positions consumed by one image chunk in the decoder (M-RoPE path).
pub fn image_chunk_n_pos(nx: usize, ny: usize) -> usize {
    nx.max(ny)
}

/// Build per-token MRoPE section positions for a text + vision + text layout.
pub fn build_multimodal_mrope_sections(
    text_len_before: usize,
    nx: usize,
    ny: usize,
    text_len_after: usize,
    text_start_pos: usize,
) -> Vec<[usize; 4]> {
    let mut out = Vec::with_capacity(text_len_before + nx * ny + text_len_after);
    for i in 0..text_len_before {
        let p = text_start_pos + i;
        out.push([p, p, p, 0]);
    }
    let pos_0 = text_start_pos + text_len_before;
    let n_vision = nx * ny;
    for i in 0..n_vision {
        out.push(image_decoder_pos(nx, ny, pos_0, i));
    }
    let after_start = pos_0 + image_chunk_n_pos(nx, ny);
    for i in 0..text_len_after {
        let p = after_start + i;
        out.push([p, p, p, 0]);
    }
    out
}

/// Replace the vision placeholder span in a flat token embedding buffer.
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

/// Assembled multimodal prefill payload for the Qwen3.5 decoder.
#[derive(Debug, Clone)]
pub struct MultimodalPrefill {
    /// Flat `[seq * n_embd]` hidden states (text embeds + vision rows spliced in).
    pub hidden: Vec<f32>,
    /// Per-token MRoPE section positions.
    pub mrope_sections: Vec<[usize; 4]>,
    /// Index of the last token (for last-logits-only decode).
    pub last_token_idx: usize,
    /// Token ids (vision rows may be placeholder ids).
    pub seq: Vec<u32>,
}

/// High-level prompt wrapper: tokenize + splice vision embeddings.
pub struct MultimodalPrompt<'a> {
    pub prompt: &'a str,
    pub vision: &'a super::encoder::VisionEncodeOutput,
}

impl<'a> MultimodalPrompt<'a> {
    /// Tokenize `prompt` (must contain one [`MEDIA_MARKER`]), merge vision
    /// embeddings, and build MRoPE section positions.
    pub fn assemble(
        &self,
        tokenizer: impl FnMut(&str) -> anyhow::Result<Vec<u32>>,
        token_embd_table: &[f32],
        n_embd: usize,
        text_start_pos: usize,
    ) -> anyhow::Result<MultimodalPrefill> {
        MultiImagePrompt {
            prompt: self.prompt,
            images: std::slice::from_ref(self.vision),
        }
        .assemble(tokenizer, token_embd_table, n_embd, text_start_pos)
    }
}

/// [`MultimodalPrompt`] with several images: the `i`-th [`MEDIA_MARKER`] in
/// `prompt` takes `images[i]`.
///
/// Positions follow HF `get_rope_index`: text runs on one counter; an image
/// starting at `p` puts its tokens at `(p, p + row, p + col)` and the text
/// after it resumes at `p + max(rows, cols)`.
pub struct MultiImagePrompt<'a> {
    pub prompt: &'a str,
    pub images: &'a [super::encoder::VisionEncodeOutput],
}

impl MultiImagePrompt<'_> {
    /// Assemble with token embeddings looked up in a dense `[vocab, n_embd]`
    /// table (ids past the table embed as zeros).
    pub fn assemble(
        &self,
        tokenizer: impl FnMut(&str) -> anyhow::Result<Vec<u32>>,
        token_embd_table: &[f32],
        n_embd: usize,
        text_start_pos: usize,
    ) -> anyhow::Result<MultimodalPrefill> {
        let vocab = token_embd_table.len() / n_embd;
        self.assemble_with(
            tokenizer,
            |ids| {
                let mut out = Vec::with_capacity(ids.len() * n_embd);
                for &t in ids {
                    let t = t as usize;
                    if t >= vocab {
                        out.extend(std::iter::repeat_n(0.0f32, n_embd));
                    } else {
                        out.extend_from_slice(&token_embd_table[t * n_embd..(t + 1) * n_embd]);
                    }
                }
                Ok(out)
            },
            n_embd,
            text_start_pos,
        )
    }

    /// Assemble with token embeddings from `embed` (`ids → [ids.len(), n_embd]`),
    /// e.g. [`crate::Qwen35Runner::embed_tokens`] over a packed table that is
    /// never expanded whole.
    pub fn assemble_with(
        &self,
        tokenizer: impl FnMut(&str) -> anyhow::Result<Vec<u32>>,
        embed: impl FnMut(&[u32]) -> anyhow::Result<Vec<f32>>,
        n_embd: usize,
        text_start_pos: usize,
    ) -> anyhow::Result<MultimodalPrefill> {
        Ok(self
            .assemble_with_mask(tokenizer, embed, n_embd, text_start_pos)?
            .0)
    }

    /// [`Self::assemble_with`], and which tokens are an image's rows.
    pub fn assemble_with_mask(
        &self,
        mut tokenizer: impl FnMut(&str) -> anyhow::Result<Vec<u32>>,
        mut embed: impl FnMut(&[u32]) -> anyhow::Result<Vec<f32>>,
        n_embd: usize,
        text_start_pos: usize,
    ) -> anyhow::Result<(MultimodalPrefill, Vec<bool>)> {
        let parts: Vec<&str> = self.prompt.split(MEDIA_MARKER).collect();
        if parts.len() != self.images.len() + 1 {
            return Err(anyhow::anyhow!(
                "prompt has {} `{MEDIA_MARKER}` marker(s) for {} image(s)",
                parts.len() - 1,
                self.images.len()
            ));
        }
        let start_ids = tokenizer(VISION_START)?;
        let end_ids = tokenizer(VISION_END)?;
        let mut seq: Vec<u32> = Vec::new();
        let mut hidden: Vec<f32> = Vec::new();
        let mut sections: Vec<[usize; 4]> = Vec::new();
        let mut pos = text_start_pos;
        let mut is_image: Vec<bool> = Vec::new();
        let mut push_text = |ids: &[u32],
                             seq: &mut Vec<u32>,
                             hidden: &mut Vec<f32>,
                             sections: &mut Vec<[usize; 4]>,
                             pos: &mut usize|
         -> anyhow::Result<()> {
            let e = embed(ids)?;
            anyhow::ensure!(e.len() == ids.len() * n_embd, "embedding width");
            hidden.extend_from_slice(&e);
            for &tid in ids {
                seq.push(tid);
                sections.push([*pos, *pos, *pos, 0]);
                *pos += 1;
            }
            Ok(())
        };
        for (i, part) in parts.iter().enumerate() {
            let ids = tokenizer(part)?;
            push_text(&ids, &mut seq, &mut hidden, &mut sections, &mut pos)?;
            let Some(v) = self.images.get(i) else {
                break;
            };
            push_text(&start_ids, &mut seq, &mut hidden, &mut sections, &mut pos)?;
            anyhow::ensure!(
                v.embeddings.len() == v.n_tokens * n_embd,
                "image {i}: {} embedding values for {} tokens of width {n_embd}",
                v.embeddings.len(),
                v.n_tokens
            );
            is_image.resize(seq.len(), false);
            for t in 0..v.n_tokens {
                seq.push(0);
                is_image.push(true);
                sections.push(image_decoder_pos(v.grid_x, v.grid_y, pos, t));
            }
            hidden.extend_from_slice(&v.embeddings);
            pos += image_chunk_n_pos(v.grid_x, v.grid_y);
            push_text(&end_ids, &mut seq, &mut hidden, &mut sections, &mut pos)?;
        }
        is_image.resize(seq.len(), false);
        Ok((
            MultimodalPrefill {
                hidden,
                mrope_sections: sections,
                last_token_idx: seq.len().saturating_sub(1),
                seq,
            },
            is_image,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_images_resume_text_after_each_grid() {
        use super::super::encoder::VisionEncodeOutput;
        let img = |gx: usize, gy: usize, v: f32| VisionEncodeOutput {
            embeddings: vec![v; gx * gy * 2],
            grid_x: gx,
            grid_y: gy,
            n_tokens: gx * gy,
        };
        let images = [img(2, 1, 7.0), img(1, 3, 9.0)];
        let table: Vec<f32> = (0..20).map(|i| i as f32).collect(); // vocab 10, n_embd 2
        let tok = |s: &str| -> anyhow::Result<Vec<u32>> {
            Ok(match s {
                VISION_START => vec![1],
                VISION_END => vec![2],
                "" => vec![],
                t => vec![t.len() as u32 + 2],
            })
        };
        let p = MultiImagePrompt {
            prompt: "a<__media__>bb<__media__>c",
            images: &images,
        }
        .assemble(tok, &table, 2, 0)
        .unwrap();
        // a, <s>, img0 (2), </s>, bb, <s>, img1 (3), </s>, c
        assert_eq!(p.seq.len(), 1 + 1 + 2 + 1 + 1 + 1 + 3 + 1 + 1);
        let s = &p.mrope_sections;
        assert_eq!(s[1], [1, 1, 1, 0]); // <s>
        assert_eq!(s[3], [2, 2, 3, 0]); // img0 row 0 col 1
        assert_eq!(s[4], [4, 4, 4, 0]); // </s> at 2 + max(1, 2)
        assert_eq!(s[8], [7, 8, 7, 0]); // img1 row 1
        assert_eq!(s[9], [7, 9, 7, 0]); // img1 row 2
        assert_eq!(s[10], [10, 10, 10, 0]); // </s> at 7 + 3
        assert_eq!(&p.hidden[2 * 2..2 * 3], &[7.0, 7.0]);
        // A marker count that does not match is an error, not a silent splice.
        assert!(
            MultiImagePrompt {
                prompt: "x<__media__>",
                images: &images,
            }
            .assemble(tok, &table, 2, 0)
            .is_err()
        );
    }

    #[test]
    fn image_chunk_n_pos_is_max_grid() {
        assert_eq!(image_chunk_n_pos(4, 6), 6);
    }

    #[test]
    fn mrope_sections_length_matches_tokens() {
        let sec = build_multimodal_mrope_sections(2, 2, 2, 1, 0);
        assert_eq!(sec.len(), 2 + 4 + 1);
        // Vision token i=0 at pos_0=2: [t, h=0, w=0]
        assert_eq!(sec[2], [2, 2, 2, 0]);
        // i=1 (row 0, col 1): width section advances
        assert_eq!(sec[3], [2, 2, 3, 0]);
    }
}
