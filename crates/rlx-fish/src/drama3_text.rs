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

//! Drama 3 text handling: the prompt template (`reference-bracket-v1`), its
//! input rules, and the long-text chunking of the hosted server.

use anyhow::{Result, bail, ensure};

use crate::prompt::Segment;

const TTS_INSTRUCTION: &str = "convert the provided text to speech";
const REFERENCE_INSTRUCTION: &str =
    "convert the provided text to speech reference to the following:";
const MESSAGE_END: &str = "<|im_end|>\n";
const GENERATION_START: &str = "<|im_start|>assistant\n<|voice|>";
/// Only the prompt builder may write these.
const STRUCTURAL: [&str; 6] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
];
/// The tokenizer has `<|speaker:0|>` … `<|speaker:15|>`.
const MAX_SPEAKER: u32 = 15;

/// Chunk size threshold in UTF-8 bytes (`DRAMA3_CHUNK_BYTES`).
pub const CHUNK_BYTES: usize = 300;
/// Speakers per chunk (`DRAMA3_CHUNK_MAX_SPEAKERS`).
pub const CHUNK_MAX_SPEAKERS: usize = 3;

/// Unicode `White_Space` — what the reference strips (`str.strip()` with no
/// argument would also strip U+001C..U+001F).
pub fn is_ws(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

fn strip(s: &str) -> &str {
    s.trim_matches(is_ws)
}

/// A `<|speaker:N|>` tag at the start of `s`: `(N as written, tag length)`.
fn speaker_tag_at(s: &str) -> Option<(&str, usize)> {
    let rest = s.strip_prefix("<|speaker:")?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    (digits > 0 && rest[digits..].starts_with("|>")).then(|| (&rest[..digits], 10 + digits + 2))
}

/// Reject empty text, structural tokens, and non-canonical speaker tags.
pub fn validate(text: &str, location: &str) -> Result<()> {
    ensure!(
        !strip(text).is_empty(),
        "{location}: text must not be empty"
    );
    for tok in STRUCTURAL {
        ensure!(
            !text.contains(tok),
            "{location}: {tok} is written by the prompt builder"
        );
    }
    for (i, _) in text.match_indices("<|speaker") {
        let ok = speaker_tag_at(&text[i..]).is_some_and(|(d, _)| {
            d.parse::<u32>()
                .is_ok_and(|n| n <= MAX_SPEAKER && n.to_string() == d)
        });
        ensure!(
            ok,
            "{location}: speaker tags must be <|speaker:0|> to <|speaker:{MAX_SPEAKER}|>; got {text:?}"
        );
    }
    Ok(())
}

/// Prefix `<|speaker:0|> ` unless a tag already leads the text (after white space).
pub fn with_speaker_tag(text: &str) -> String {
    if speaker_tag_at(text.trim_start_matches(is_ws)).is_none() {
        format!("<|speaker:0|> {text}")
    } else {
        text.to_string()
    }
}

/// One earlier chunk or reference: text and codes `[num_codebooks][frames]`.
#[derive(Clone, Debug)]
pub struct Turn {
    pub text: String,
    pub codes: Vec<Vec<u32>>,
}

/// Prompt segments ending in the open assistant voice turn.
pub fn prompt_segments(text: &str, references: &[Turn], history: &[Turn]) -> Result<Vec<Segment>> {
    validate(text, "text")?;
    let mut segs = vec![
        Segment::Text("<|im_start|>system\n".into()),
        Segment::Text(
            if references.is_empty() {
                TTS_INSTRUCTION
            } else {
                REFERENCE_INSTRUCTION
            }
            .into(),
        ),
    ];
    for (i, r) in references.iter().enumerate() {
        let loc = format!("reference {}", i + 1);
        validate(&r.text, &loc)?;
        ensure!(
            !r.codes.is_empty() && !r.codes[0].is_empty(),
            "{loc}: no codes"
        );
        // The tag must follow "Text: " directly; leading white space moves behind it.
        let tagged = with_speaker_tag(&r.text);
        let body = tagged.trim_start_matches(is_ws);
        let leading = &tagged[..tagged.len() - body.len()];
        let (_, n) = speaker_tag_at(body).expect("tagged");
        let transcript = format!("{}{leading}{}", &body[..n], &body[n..]);
        segs.push(Segment::Text(format!(
            "\n\n[Reference {}]\nText: {transcript}\nSpeech: ",
            i + 1
        )));
        segs.push(Segment::Codes(r.codes.clone()));
    }
    segs.push(Segment::Text(MESSAGE_END.into()));
    for (i, h) in history.iter().enumerate() {
        validate(&h.text, &format!("history {}", i + 1))?;
        segs.extend([
            Segment::Text("<|im_start|>user\n".into()),
            Segment::Text(with_speaker_tag(&h.text)),
            Segment::Text(MESSAGE_END.into()),
            Segment::Text(GENERATION_START.into()),
            Segment::Codes(h.codes.clone()),
            Segment::Text(MESSAGE_END.into()),
        ]);
    }
    segs.extend([
        Segment::Text("<|im_start|>user\n".into()),
        Segment::Text(with_speaker_tag(text)),
        Segment::Text(MESSAGE_END.into()),
        Segment::Text(GENERATION_START.into()),
    ]);
    Ok(segs)
}

// ── chunking ────────────────────────────────────────────────────────────

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Length of a protected span starting at byte `i` of `s`, if any:
/// `<|…|>` special tokens, paired `<tag …>…</tag>` markup, `[bracketed]` cues.
fn protected_at(s: &str, i: usize) -> Option<usize> {
    let rest = &s[i..];
    if let Some(r) = rest.strip_prefix("<|") {
        let body = r.find(['|', '>']).unwrap_or(r.len());
        if r[body..].starts_with("|>") {
            return Some(2 + body + 2);
        }
    }
    if let Some(r) = rest.strip_prefix('<') {
        let mut chars = r.char_indices();
        if chars.next().is_some_and(|(_, c)| c.is_ascii_alphabetic()) {
            let run = r
                .find(|c: char| !(is_word(c) || c == '-'))
                .unwrap_or(r.len());
            // `[\w-]*` backtracks until `\b` holds after the name.
            let mut end = run;
            while end > 0 {
                let last = r[..end].chars().next_back().expect("name");
                let next = r[end..].chars().next();
                if next.is_none_or(|n| is_word(last) != is_word(n)) && is_word(last) {
                    let name = &r[..end];
                    if let Some(gt) = r[end..].find('>') {
                        let open = 1 + end + gt + 1;
                        let close = format!("</{name}>");
                        if let Some(at) = s[i + open..].find(&close) {
                            return Some(open + at + close.len());
                        }
                    }
                }
                end -= last_len(&r[..end]);
            }
        }
    }
    if let Some(r) = rest.strip_prefix('[') {
        let body = r.find(['[', ']']).unwrap_or(r.len());
        if r[body..].starts_with(']') {
            return Some(1 + body + 1);
        }
    }
    None
}

fn last_len(s: &str) -> usize {
    s.chars().next_back().map_or(1, char::len_utf8)
}

/// `(text, protected)` pieces.
fn protected_pieces(text: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let (mut last, mut i) = (0, 0);
    while i < text.len() {
        if let Some(n) = protected_at(text, i) {
            if i > last {
                out.push((text[last..i].to_string(), false));
            }
            out.push((text[i..i + n].to_string(), true));
            i += n;
            last = i;
        } else {
            i += text[i..].chars().next().map_or(1, char::len_utf8);
        }
    }
    if last < text.len() {
        out.push((text[last..].to_string(), false));
    }
    out
}

const SENTENCE_ENDS: &[char] = &['.', '!', '?', ';', '。', '！', '？', '；', '…', '\n'];
const CLAUSE_MARKS: &[char] = &[',', ':', '，', '：', '、'];
const SPACES: &[char] = &[' '];

/// Split free pieces longer than `length` bytes after each mark (kept),
/// except a `.` between two digits.
fn break_at_marks(
    pieces: Vec<(String, bool)>,
    length: usize,
    marks: &[char],
) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            out.push((text, protected));
            continue;
        }
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut start = 0;
        for (k, &(pos, c)) in chars.iter().enumerate() {
            let decimal = c == '.'
                && k > 0
                && k + 1 < chars.len()
                && chars[k - 1].1.is_numeric()
                && chars[k + 1].1.is_numeric();
            if marks.contains(&c) && !decimal {
                let end = pos + c.len_utf8();
                out.push((text[start..end].to_string(), false));
                start = end;
            }
        }
        if start < text.len() {
            out.push((text[start..].to_string(), false));
        }
    }
    out
}

fn break_by_length(pieces: Vec<(String, bool)>, length: usize) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            out.push((text, protected));
            continue;
        }
        let mut cur = String::new();
        for c in text.chars() {
            if !cur.is_empty() && cur.len() + c.len_utf8() > length {
                out.push((std::mem::take(&mut cur), false));
            }
            cur.push(c);
        }
        out.push((cur, false));
    }
    out
}

/// `split_by_length`: chunks of about `length` bytes, packed sentence by sentence.
pub fn split_by_length(text: &str, length: usize) -> Vec<String> {
    if strip(text).is_empty() {
        return Vec::new();
    }
    if text.len() <= length {
        return vec![text.to_string()];
    }
    let mut pieces = break_at_marks(protected_pieces(text), 0, SENTENCE_ENDS);
    for marks in [CLAUSE_MARKS, SPACES] {
        pieces = break_at_marks(pieces, length, marks);
    }
    let pieces = break_by_length(pieces, length);
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for (piece, _) in pieces {
        if !cur.is_empty() && cur.len() + piece.len() > length {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(&piece);
    }
    chunks.push(cur);
    chunks
        .iter()
        .map(|c| strip(c))
        .filter(|c| !c.is_empty())
        .map(str::to_string)
        .collect()
}

/// `(speaker, body)` per tagged segment; text before the first tag is speaker 0.
fn speaker_segments(text: &str) -> Vec<(String, String)> {
    let mut tags = Vec::new();
    for (i, _) in text.match_indices("<|speaker:") {
        if let Some((d, n)) = speaker_tag_at(&text[i..]) {
            tags.push((i, n, d.to_string()));
        }
    }
    let mut out = Vec::new();
    let head = &text[..tags.first().map_or(text.len(), |t| t.0)];
    if !strip(head).is_empty() {
        out.push(("0".to_string(), head.to_string()));
    }
    for (k, (i, n, d)) in tags.iter().enumerate() {
        let end = tags.get(k + 1).map_or(text.len(), |t| t.0);
        let body = &text[i + n..end];
        if !strip(body).is_empty() {
            out.push((d.clone(), body.to_string()));
        }
    }
    out
}

/// `split_drama3_text`: the chunks to generate, in order.
pub fn split_text(text: &str, max_bytes: usize, max_speakers: usize) -> Vec<String> {
    let tagged = text
        .match_indices("<|speaker:")
        .any(|(i, _)| speaker_tag_at(&text[i..]).is_some());
    if !tagged {
        return split_by_length(text, max_bytes);
    }
    let mut segments = Vec::new();
    for (spk, body) in speaker_segments(text) {
        for piece in split_by_length(&body, max_bytes) {
            segments.push((spk.clone(), format!("<|speaker:{spk}|> {piece}")));
        }
    }
    let mut chunks = Vec::new();
    let (mut batch, mut speakers, mut size): (Vec<String>, Vec<String>, usize) =
        (Vec::new(), Vec::new(), 0);
    for (spk, seg) in segments {
        let mut new_speaker = !speakers.contains(&spk);
        let too_many = new_speaker && speakers.len() >= max_speakers;
        let too_long = !batch.is_empty() && size + seg.len() > max_bytes;
        if too_many || too_long {
            chunks.push(batch.join("\n"));
            batch.clear();
            speakers.clear();
            size = 0;
            new_speaker = true;
        }
        size += seg.len();
        batch.push(seg);
        if new_speaker {
            speakers.push(spk);
        }
    }
    if !batch.is_empty() {
        chunks.push(batch.join("\n"));
    }
    chunks
}

/// Context for the next chunk: the first chunk pinned, plus the latest one.
#[derive(Default)]
pub struct History {
    pinned: Option<Turn>,
    recent: Option<Turn>,
}

impl History {
    pub fn push(&mut self, turn: Turn) {
        if self.pinned.is_none() {
            self.pinned = Some(turn);
        } else {
            self.recent = Some(turn);
        }
    }

    /// Histories to try, longest first, ending with none.
    pub fn candidates(&self) -> Vec<Vec<Turn>> {
        let pinned: Vec<Turn> = self.pinned.iter().cloned().collect();
        let mut out = Vec::new();
        if let Some(r) = &self.recent {
            out.push(pinned.iter().cloned().chain([r.clone()]).collect());
        }
        out.push(pinned.clone());
        if !pinned.is_empty() {
            out.push(Vec::new());
        }
        out
    }
}

/// Fail fast on a template the checkpoint's tokenizer does not match.
pub fn check_tokenizer(
    enc: &dyn crate::prompt::TextEncoder,
    semantic_begin: u32,
    semantic_end: u32,
    eos: u32,
    vocab: usize,
) -> Result<()> {
    for (tok, want) in [
        ("<|semantic:0|>".to_string(), semantic_begin),
        (format!("<|semantic:{}|>", vocab - 1), semantic_end),
        ("<|im_end|>".to_string(), eos),
    ] {
        match enc.special_id(&tok) {
            Some(id) if id == want => {}
            got => bail!("tokenizer/config mismatch for {tok}: {got:?} vs {want}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_tag_rules() {
        assert_eq!(with_speaker_tag("hi"), "<|speaker:0|> hi");
        assert_eq!(with_speaker_tag("  <|speaker:2|>hi"), "  <|speaker:2|>hi");
        assert!(validate("<|speaker:15|> ok", "t").is_ok());
        assert!(validate("<|speaker:16|> no", "t").is_err());
        assert!(validate("<|speaker:01|> no", "t").is_err());
        assert!(validate("<|im_end|>", "t").is_err());
        assert!(validate(" \n", "t").is_err());
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(
            split_text("Hello there.", CHUNK_BYTES, CHUNK_MAX_SPEAKERS),
            vec!["Hello there."]
        );
        assert!(split_text("   ", CHUNK_BYTES, CHUNK_MAX_SPEAKERS).is_empty());
    }

    #[test]
    fn long_text_breaks_at_sentences_and_keeps_spans() {
        let s = "One two three. [laughs loudly] Four five six! Pi is 3.14 exactly. ";
        let chunks = split_by_length(&s.repeat(8), 60);
        assert!(
            chunks.iter().all(|c| c.len() <= 60 || c.contains('[')),
            "{chunks:?}"
        );
        assert!(chunks.iter().any(|c| c.contains("3.14")));
        assert!(
            chunks
                .iter()
                .all(|c| !c.contains("[laughs") || c.contains("[laughs loudly]"))
        );
    }

    #[test]
    fn protected_markup_spans() {
        let p = protected_pieces("a <whisper>quiet. words</whisper> b <|speaker:1|> c [sigh]");
        let prot: Vec<&str> = p.iter().filter(|x| x.1).map(|x| x.0.as_str()).collect();
        assert_eq!(
            prot,
            vec!["<whisper>quiet. words</whisper>", "<|speaker:1|>", "[sigh]"]
        );
    }

    #[test]
    fn speakers_regroup() {
        let t = "<|speaker:0|> Hi. <|speaker:1|> Hey. <|speaker:0|> Bye.";
        let c = split_text(t, CHUNK_BYTES, CHUNK_MAX_SPEAKERS);
        assert_eq!(
            c,
            vec!["<|speaker:0|>  Hi. \n<|speaker:1|>  Hey. \n<|speaker:0|>  Bye."]
        );
    }

    #[test]
    fn history_candidates() {
        let turn = |s: &str| Turn {
            text: s.into(),
            codes: vec![vec![1]],
        };
        let mut h = History::default();
        assert_eq!(h.candidates().len(), 1);
        h.push(turn("a"));
        h.push(turn("b"));
        h.push(turn("c"));
        let c = h.candidates();
        assert_eq!(c.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 1, 0]);
        assert_eq!(c[0][1].text, "c");
    }
}
