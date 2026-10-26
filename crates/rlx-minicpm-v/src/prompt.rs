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

//! Chat template and image / video placeholder expansion
//! (`chat_template.jinja` + `MiniCPMV4_7Processor.replace_image_token` /
//! `replace_video_token`).
//!
//! The expanded text is tokenized in one piece, as HF does, so BPE merges
//! across a placeholder's edges come out the same.

/// Image placeholder the chat template emits, one per picture.
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// Video placeholder the chat template emits, one per clip.
pub const VIDEO_PAD: &str = "<|video_pad|>";
const IMAGE_START: &str = "<image>";
const IMAGE_END: &str = "</image>";
const SLICE_START: &str = "<slice>";
const SLICE_END: &str = "</slice>";
const IMAGE_ID_START: &str = "<image_id>";
const IMAGE_ID_END: &str = "</image_id>";

/// A picture or a clip in a user turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Media {
    Image,
    Video,
}

impl Media {
    pub fn pad(self) -> &'static str {
        match self {
            Media::Image => IMAGE_PAD,
            Media::Video => VIDEO_PAD,
        }
    }
}

/// One user turn with `n_images` pictures before `question`, ready for the
/// assistant. `thinking` opens a `<think>` block instead of the empty one.
pub fn chat_prompt(question: &str, n_images: usize, thinking: bool) -> String {
    chat_prompt_media(question, &vec![Media::Image; n_images], thinking)
}

/// [`chat_prompt`] with pictures and clips, in `media` order, before
/// `question` (the template joins the parts with `\n`).
pub fn chat_prompt_media(question: &str, media: &[Media], thinking: bool) -> String {
    let mut parts: Vec<&str> = media.iter().map(|m| m.pad()).collect();
    parts.push(question);
    let content = parts.join("\n");
    let mut s = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        content.trim()
    );
    s.push_str(if thinking {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    });
    s
}

/// Placeholder text of one image: `<image>` + one pad per thumbnail token +
/// `</image>`, then the slice rows (`<slice>…</slice>` × cols, rows joined by
/// `\n`), prefixed with `<image_id>N</image_id>` when `image_id` is set.
pub fn image_placeholder(
    image_id: Option<usize>,
    thumb_tokens: usize,
    slice_tokens: usize,
    grid: Option<(usize, usize)>,
) -> String {
    let mut s = String::new();
    if let Some(i) = image_id {
        s.push_str(&format!("{IMAGE_ID_START}{i}{IMAGE_ID_END}"));
    }
    s.push_str(IMAGE_START);
    s.push_str(&IMAGE_PAD.repeat(thumb_tokens));
    s.push_str(IMAGE_END);
    if let Some((rows, cols)) = grid {
        let one = format!("{SLICE_START}{}{SLICE_END}", IMAGE_PAD.repeat(slice_tokens));
        let row = one.repeat(cols);
        s.push_str(&vec![row; rows].join("\n"));
    }
    s
}

/// Placeholder text of one clip: each frame as `<image>` + one
/// [`VIDEO_PAD`] per thumbnail token + `</image>`, then its slice rows,
/// frames back to back with nothing between them and no `<image_id>`.
/// `frames` holds `(thumb_tokens, slice_tokens, grid)` per frame.
pub fn video_placeholder(frames: &[(usize, usize, Option<(usize, usize)>)]) -> String {
    let mut s = String::new();
    for &(thumb, slice, grid) in frames {
        s.push_str(IMAGE_START);
        s.push_str(&VIDEO_PAD.repeat(thumb));
        s.push_str(IMAGE_END);
        if let Some((rows, cols)) = grid {
            let one = format!("{SLICE_START}{}{SLICE_END}", VIDEO_PAD.repeat(slice));
            s.push_str(&vec![one.repeat(cols); rows].join("\n"));
        }
    }
    s
}

/// Replace the pads of `prompt`, in the order they appear, with
/// `placeholders`: each entry names the pad it expects there.
pub fn expand_media(prompt: &str, placeholders: &[(Media, String)]) -> anyhow::Result<String> {
    let mut out = String::new();
    let mut rest = prompt;
    let mut it = placeholders.iter();
    loop {
        let next = [IMAGE_PAD, VIDEO_PAD]
            .iter()
            .filter_map(|p| rest.find(p).map(|i| (i, *p)))
            .min_by_key(|(i, _)| *i);
        let Some((at, pad)) = next else { break };
        let (media, text) = it.next().ok_or_else(|| {
            anyhow::anyhow!("minicpm-v: prompt has more placeholders than images and videos")
        })?;
        anyhow::ensure!(
            media.pad() == pad,
            "minicpm-v: prompt has {pad} where a {media:?} goes"
        );
        out.push_str(&rest[..at]);
        out.push_str(text);
        rest = &rest[at + pad.len()..];
    }
    anyhow::ensure!(
        it.next().is_none(),
        "minicpm-v: prompt has fewer placeholders than images and videos"
    );
    out.push_str(rest);
    Ok(out)
}

/// Replace the `k`-th [`IMAGE_PAD`] of `prompt` with `placeholders[k]`.
pub fn expand_images(prompt: &str, placeholders: &[String]) -> anyhow::Result<String> {
    let pieces: Vec<&str> = prompt.split(IMAGE_PAD).collect();
    anyhow::ensure!(
        pieces.len() == placeholders.len() + 1,
        "minicpm-v: prompt has {} image placeholders for {} images",
        pieces.len() - 1,
        placeholders.len()
    );
    let mut out = String::from(pieces[0]);
    for (p, rest) in placeholders.iter().zip(&pieces[1..]) {
        out.push_str(p);
        out.push_str(rest);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hf_chat_template() {
        assert_eq!(
            chat_prompt("Describe this image in one sentence.", 1, false),
            "<|im_start|>user\n<|image_pad|>\nDescribe this image in one sentence.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn placeholder_layout() {
        let p = image_placeholder(Some(0), 2, 1, Some((2, 2)));
        assert_eq!(
            p,
            "<image_id>0</image_id><image><|image_pad|><|image_pad|></image>\
             <slice><|image_pad|></slice><slice><|image_pad|></slice>\n\
             <slice><|image_pad|></slice><slice><|image_pad|></slice>"
        );
        let e = expand_images("a<|image_pad|>b", std::slice::from_ref(&p)).unwrap();
        assert!(e.starts_with("a<image_id>") && e.ends_with("</slice>b"));
    }

    #[test]
    fn video_layout_and_mixed_expansion() {
        let v = video_placeholder(&[(2, 0, None), (1, 1, Some((1, 2)))]);
        assert_eq!(
            v,
            "<image><|video_pad|><|video_pad|></image><image><|video_pad|></image>\
             <slice><|video_pad|></slice><slice><|video_pad|></slice>"
        );
        let prompt = chat_prompt_media("q", &[Media::Video, Media::Image], false);
        assert!(prompt.starts_with("<|im_start|>user\n<|video_pad|>\n<|image_pad|>\nq"));
        let e = expand_media(
            &prompt,
            &[(Media::Video, "V".into()), (Media::Image, "I".into())],
        )
        .unwrap();
        assert!(e.starts_with("<|im_start|>user\nV\nI\nq"));
        assert!(expand_media(&prompt, &[(Media::Image, "I".into())]).is_err());
    }
}
