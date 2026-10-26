// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Text side of KittenTTS 2: expression markup, light clean-up, chunking,
//! prompt assembly and joining the per-chunk audio.
//!
//! Ported from `kittenml.kittentts2.text` / `prompt`; the written → spoken
//! step it delegates to `kitten_text_processing` is [`crate::normalize`].

use std::sync::LazyLock;

use regex::Regex;

use crate::config::TokenMap;

/// Emotions a leading `[tag]` may set.
pub const EXPRESSION_TAGS: [&str; 10] = [
    "mundane",
    "nervous",
    "tender",
    "angry",
    "excited",
    "stern",
    "sad",
    "contemplative",
    "surprised",
    "joyful",
];

/// Inline `<event>` tags.
pub const VOCAL_EVENT_TAGS: [&str; 10] = [
    "pause", "sigh", "gasp", "laugh", "giggle", "sob", "scoff", "growl", "um", "gulp",
];

/// Upstream defaults (`text.CHUNK_CHARS` …); the repository config overrides
/// the first two.
pub const CHUNK_CHARS: usize = 380;
pub const CHUNK_MIN_CHARS: usize = 130;
pub const CHUNK_GAP_S: f32 = 0.16;

static TAG_SPAN: LazyLock<Regex> = LazyLock::new(|| {
    let alt = |tags: &[&str]| tags.join("|");
    Regex::new(&format!(
        r"(?i)\[(?:{})\]|<(?:{})>",
        alt(&EXPRESSION_TAGS),
        alt(&VOCAL_EVENT_TAGS)
    ))
    .unwrap()
});
static EMPHASIS_SPAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(\(\(([^()\n]{1,80})\)\)\)").unwrap());
static UI_GLYPHS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "[\u{2022}\u{2023}\u{2043}\u{25e6}\u{25b8}\u{21b3}\u{2192}\u{2190}\u{2191}\u{2193}]+",
    )
    .unwrap()
});
static ELLIPSIS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\.\s+){2,}\.|\u{2026}|\.\.\.").unwrap());
static MULTI_BLANK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]{2,}").unwrap());
static VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b([vV])(\d+(?:\.\d+)+)").unwrap());
static NONTAG_BRACKET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[([^\[\]\n]{1,400})\]|<([^<>\n]{1,60})>|\(([^()\n]{1,400})\)").unwrap()
});
static STRAIGHT_QUOTE_PAIR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""([^"\n]*)""#).unwrap());
static COMMA_COLLISION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r",\s*([,.!?;:])").unwrap());
static NEWLINES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*\n+\s*").unwrap());
static DASH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*(?:[\u{2014}\u{2013}]|--)\s*").unwrap());
static DASH_FIX: LazyLock<[(Regex, &'static str); 5]> = LazyLock::new(|| {
    [
        (Regex::new(r",\s*,+").unwrap(), ", "),
        (Regex::new(r"\s+([,.!?;:])").unwrap(), "$1"),
        (Regex::new(r",\s*([.!?])").unwrap(), "$1"),
        (Regex::new(r"\s{2,}").unwrap(), " "),
        (Regex::new(r"^\s*,\s*").unwrap(), ""),
    ]
});
static DETACHED_MARK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+([?!])").unwrap());
static TRAILING_NONTERMINAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[,;:\\-\u{2013}\u{2014}]+\\s*$").unwrap());

const PLACEHOLDER_OPEN: char = '\u{e000}';
const PLACEHOLDER_CLOSE: char = '\u{e001}';
const ALREADY_TERMINATED: &str = ".!?\"\u{2019}\u{201d}')]";

/// Spoken readings for math glyphs, applied before anything else sees them.
const MATH_GLYPH_WORDS: &[(&str, &str)] = &[
    ("\u{3b1}", "alpha"),
    ("\u{3b2}", "beta"),
    ("\u{3b3}", "gamma"),
    ("\u{3b4}", "delta"),
    ("\u{3b5}", "epsilon"),
    ("\u{3b6}", "zeta"),
    ("\u{3b7}", "eta"),
    ("\u{3b8}", "theta"),
    ("\u{3b9}", "iota"),
    ("\u{3ba}", "kappa"),
    ("\u{3bb}", "lambda"),
    ("\u{3bc}", "mu"),
    ("\u{3bd}", "nu"),
    ("\u{3be}", "xi"),
    ("\u{3c0}", "pi"),
    ("\u{3c1}", "rho"),
    ("\u{3c3}", "sigma"),
    ("\u{3c4}", "tau"),
    ("\u{3c5}", "upsilon"),
    ("\u{3c6}", "phi"),
    ("\u{3d5}", "phi"),
    ("\u{3c7}", "chi"),
    ("\u{3c8}", "psi"),
    ("\u{3c9}", "omega"),
    ("\u{393}", "gamma"),
    ("\u{394}", "delta"),
    ("\u{398}", "theta"),
    ("\u{39b}", "lambda"),
    ("\u{39e}", "xi"),
    ("\u{3a0}", "pi"),
    ("\u{3a3}", "sigma"),
    ("\u{3a6}", "phi"),
    ("\u{3a8}", "psi"),
    ("\u{3a9}", "omega"),
    ("=", "equals"),
    ("\u{2260}", "is not equal to"),
    ("\u{2248}", "approximately equals"),
    ("\u{2264}", "is less than or equal to"),
    ("\u{2265}", "is greater than or equal to"),
    ("\u{b1}", "plus or minus"),
    ("\u{d7}", "times"),
    ("\u{f7}", "divided by"),
    ("\u{b7}", "dot"),
    ("\u{2212}", "minus"),
    ("\u{2217}", "star"),
    ("\u{2218}", "composed with"),
    ("\u{22a4}", "transpose"),
    ("\u{223c}", "distributed as"),
    ("\u{221d}", "proportional to"),
    ("\u{2261}", "is equivalent to"),
    ("\u{2208}", "in"),
    ("\u{2209}", "not in"),
    ("\u{2282}", "subset of"),
    ("\u{2286}", "subset of"),
    ("\u{222a}", "union"),
    ("\u{2229}", "intersect"),
    ("\u{2295}", "plus"),
    ("\u{2297}", "times"),
    ("^", "to the"),
    ("\u{2211}", "sum of"),
    ("\u{220f}", "product of"),
    ("\u{222b}", "integral of"),
    ("\u{2202}", "partial"),
    ("\u{2207}", "gradient of"),
    ("\u{221a}", "square root of"),
    ("\u{221e}", "infinity"),
    ("\u{2206}", "delta"),
    ("\u{2200}", "for all"),
    ("\u{2203}", "there exists"),
    ("\u{21d2}", "implies"),
    ("\u{21d4}", "if and only if"),
    ("\u{2225}", "norm"),
    ("\u{2016}", "norm"),
    ("\u{230a}", "floor of"),
    ("\u{230b}", ""),
    ("\u{2308}", "ceiling of"),
    ("\u{2309}", ""),
    ("\u{211d}", "R"),
    ("\u{2115}", "N"),
    ("\u{2124}", "Z"),
    ("\u{211a}", "Q"),
    ("\u{2102}", "C"),
    ("\u{1d53c}", "the expectation of"),
];
const SUPERSCRIPT_DIGITS: &str =
    "\u{2070}\u{b9}\u{b2}\u{b3}\u{2074}\u{2075}\u{2076}\u{2077}\u{2078}\u{2079}";
const SUBSCRIPT_DIGITS: &str =
    "\u{2080}\u{2081}\u{2082}\u{2083}\u{2084}\u{2085}\u{2086}\u{2087}\u{2088}\u{2089}";

/// True when the text uses any expression, vocal-event or emphasis control —
/// which switches on the `{emo: 1}` conditioning prefix.
pub fn has_expression_tag(text: &str) -> bool {
    TAG_SPAN.is_match(text) || EMPHASIS_SPAN.is_match(text)
}

fn placeholder(i: usize) -> String {
    format!("{PLACEHOLDER_OPEN}{i}{PLACEHOLDER_CLOSE}")
}

/// Swap tags and emphasis spans for sentinel-wrapped indices so nothing below
/// can reorder or reword them.
fn protect_tags(text: &str) -> (String, Vec<String>) {
    let mut tags = Vec::new();
    let mut swap = |re: &Regex, s: &str| {
        re.replace_all(s, |c: &regex::Captures<'_>| {
            tags.push(c[0].to_string());
            placeholder(tags.len() - 1)
        })
        .into_owned()
    };
    let t = swap(&TAG_SPAN, text);
    let t = swap(&EMPHASIS_SPAN, &t);
    (t, tags)
}

fn restore_tags(mut text: String, tags: &[String]) -> String {
    for (i, tag) in tags.iter().enumerate() {
        text = text.replace(&placeholder(i), tag);
    }
    text
}

fn speak_math_glyphs(text: &str) -> String {
    let mut t = ELLIPSIS.replace_all(text, ", ").into_owned();
    let mut out = String::with_capacity(t.len());
    let digit = |c: char| {
        SUPERSCRIPT_DIGITS
            .chars()
            .position(|d| d == c)
            .or_else(|| SUBSCRIPT_DIGITS.chars().position(|d| d == c))
    };
    for c in t.chars() {
        let mut buf = [0u8; 4];
        let key: &str = c.encode_utf8(&mut buf);
        if let Some((_, word)) = MATH_GLYPH_WORDS.iter().find(|(g, _)| *g == key) {
            out.push(' ');
            out.push_str(word);
            out.push(' ');
        } else if let Some(d) = digit(c) {
            out.push(' ');
            out.push(char::from(b'0' + d as u8));
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    t = out;
    MULTI_BLANK.replace_all(&t, " ").into_owned()
}

fn reformat_nontag_brackets(text: &str) -> String {
    let mut t = NONTAG_BRACKET
        .replace_all(text, |c: &regex::Captures<'_>| {
            let inner = (1..=3).find_map(|i| c.get(i)).map_or("", |m| m.as_str());
            format!(", {},", inner.trim())
        })
        .into_owned();
    loop {
        let collapsed = COMMA_COLLISION.replace_all(&t, "$1").into_owned();
        if collapsed == t {
            return collapsed;
        }
        t = collapsed;
    }
}

fn dashes_to_commas(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut t = DASH.replace_all(text, ", ").into_owned();
    for (re, rep) in DASH_FIX.iter() {
        t = re.replace_all(&t, *rep).into_owned();
    }
    t.trim().to_string()
}

/// Capitalize the first letter and guarantee sentence-final punctuation.
fn capitalize_and_terminate(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let t = DETACHED_MARK.replace_all(text, "$1").into_owned();
    let mut out = String::with_capacity(t.len());
    let mut done = false;
    for c in t.chars() {
        if !done && c.is_alphabetic() {
            out.extend(c.to_uppercase());
            done = true;
        } else {
            out.push(c);
        }
    }
    let stripped = out.trim_end();
    if stripped.is_empty() {
        return out;
    }
    if stripped.ends_with(|c: char| ALREADY_TERMINATED.contains(c)) {
        return stripped.to_string();
    }
    let s = TRAILING_NONTERMINAL.replace(stripped, "");
    let s = s.trim_end();
    if s.is_empty() {
        String::new()
    } else {
        format!("{s}.")
    }
}

/// `x.Y` → `x. Y` where a lowercase letter meets an uppercase one across a
/// sentence mark (a hard line wrap the newline collapse fused).
fn split_fused_sentences(text: &str) -> String {
    let cs: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 8);
    for (i, &c) in cs.iter().enumerate() {
        out.push(c);
        if ".!?;:".contains(c)
            && i > 0
            && cs[i - 1].is_ascii_lowercase()
            && cs.get(i + 1).is_some_and(char::is_ascii_uppercase)
        {
            out.push(' ');
        }
    }
    out
}

/// Detach `?`/`!` that directly follows a digit (the number grammar would
/// otherwise read it aloud); [`capitalize_and_terminate`] re-attaches it.
fn detach_marks_after_digits(text: &str) -> String {
    let cs: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 4);
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_ascii_digit() {
            let mut j = i + 1;
            while j < cs.len() && cs[j].is_whitespace() {
                j += 1;
            }
            if j < cs.len()
                && (cs[j] == '?' || cs[j] == '!')
                && cs.get(j + 1).is_none_or(|c| c.is_whitespace())
            {
                out.push(cs[i]);
                out.push(' ');
                out.push(cs[j]);
                i = j + 1;
                continue;
            }
        }
        out.push(cs[i]);
        i += 1;
    }
    out
}

/// The text the model hears (`kittenml.kittentts2.text.normalize_text`).
///
/// Markup is protected first and restored last. With `normalize`, written
/// forms are spoken out by [`crate::normalize`] (pass `false` for
/// non-English text: the grammars are English); a normalizer failure falls
/// back to the cleaned raw text, as upstream's does.
pub fn normalize_text(text: &str, normalize: bool) -> String {
    let (protected, tags) = protect_tags(text);
    let mut t = NEWLINES.replace_all(&protected, " ").into_owned();
    t = split_fused_sentences(&t);
    t = speak_math_glyphs(&t);
    t = VERSION
        .replace_all(&t, |c: &regex::Captures<'_>| {
            format!(
                "{} {}",
                &c[1],
                c[2].split('.').collect::<Vec<_>>().join(" point ")
            )
        })
        .into_owned();
    t = detach_marks_after_digits(&t);
    let raw = dashes_to_commas(&UI_GLYPHS.replace_all(&reformat_nontag_brackets(&t), " "));
    if raw.trim().is_empty() {
        return restore_tags(raw, &tags);
    }
    if !normalize {
        return restore_tags(capitalize_and_terminate(&raw), &tags);
    }
    let spoken = crate::normalize::normalize_english(&smart_quote_pairs(&raw))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if spoken.is_empty() {
        return restore_tags(capitalize_and_terminate(&raw), &tags);
    }
    restore_tags(capitalize_and_terminate(&dashes_to_commas(&spoken)), &tags)
}

/// Paired straight double quotes → curly, so the measure grammar does not
/// read a closing quote after a digit as inches.
fn smart_quote_pairs(text: &str) -> String {
    STRAIGHT_QUOTE_PAIR
        .replace_all(text, "\u{201c}${1}\u{201d}")
        .into_owned()
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Split after `.`/`!`/`?` plus any closing quote/bracket run and whitespace —
/// upstream's `(?<=[.!?])["’”')\]]*\s+`, which drops the closers with the gap.
fn split_sentences(text: &str) -> Vec<String> {
    let cs: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < cs.len() {
        if ".!?".contains(cs[i]) {
            let mut j = i + 1;
            while j < cs.len() && "\"\u{2019}\u{201d}')]".contains(cs[j]) {
                j += 1;
            }
            let ws = j;
            while j < cs.len() && cs[j].is_whitespace() {
                j += 1;
            }
            if j > ws {
                out.push(cs[start..=i].iter().collect());
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if start < cs.len() {
        out.push(cs[start..].iter().collect());
    }
    out.into_iter()
        .filter(|s: &String| !s.trim().is_empty())
        .collect()
}

/// Last char index of `pat` that starts before `limit`, like `str.rfind(pat, 0, limit)`.
fn rfind_before(cs: &[char], pat: &str, limit: usize) -> Option<usize> {
    let p: Vec<char> = pat.chars().collect();
    let end = limit.min(cs.len());
    (0..=end.checked_sub(p.len())?)
        .rev()
        .find(|&i| cs[i..i + p.len()] == p[..])
}

/// Sentence-aware greedy packing into chunks of at most `max_chars`.
///
/// Text that already fits comes back as one chunk; undersized chunks are
/// folded into a neighbour, and every chunk gets a closing mark.
pub fn split_for_synthesis(text: &str, max_chars: usize, min_chars: usize) -> Vec<String> {
    let stripped = text.trim();
    if max_chars == 0 || char_len(stripped) <= max_chars {
        return if stripped.is_empty() {
            Vec::new()
        } else {
            vec![stripped.to_string()]
        };
    }
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for sentence in split_sentences(stripped) {
        let mut sentence = sentence.trim().to_string();
        while char_len(&sentence) > max_chars {
            let cs: Vec<char> = sentence.chars().collect();
            let cut = [", ", "; ", " \u{2014} "]
                .iter()
                .filter_map(|p| rfind_before(&cs, p, max_chars))
                .max();
            let cut = match cut {
                Some(c) if c > max_chars / 3 => c + 1,
                _ => max_chars,
            };
            let head: String = cs[..cut].iter().collect();
            sentence = cs[cut..].iter().collect::<String>().trim().to_string();
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.push(head.trim().to_string());
        }
        if current.is_empty() {
            current = sentence;
        } else if char_len(&current) + 1 + char_len(&sentence) <= max_chars {
            current.push(' ');
            current.push_str(&sentence);
        } else {
            out.push(std::mem::replace(&mut current, sentence));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    merge_short(out, max_chars, min_chars)
        .into_iter()
        .map(|c| ensure_terminal_punct(&c))
        .collect()
}

fn merge_short(parts: Vec<String>, max_chars: usize, min_chars: usize) -> Vec<String> {
    if parts.is_empty() || min_chars == 0 {
        return parts;
    }
    let mut out = parts;
    let limit = (max_chars as f64 * 1.4) as usize;
    let mut changed = true;
    while changed && out.len() > 1 {
        changed = false;
        for i in 0..out.len() {
            let len = char_len(&out[i]);
            if len >= min_chars {
                continue;
            }
            let prev_ok = i > 0 && char_len(&out[i - 1]) + 1 + len <= limit;
            let next_ok = i + 1 < out.len() && len + 1 + char_len(&out[i + 1]) <= limit;
            if prev_ok && (!next_ok || char_len(&out[i - 1]) <= char_len(&out[i + 1])) {
                let chunk = out.remove(i);
                out[i - 1] = format!("{} {}", out[i - 1].trim_end(), chunk.trim_start());
            } else if next_ok {
                let next = out.remove(i + 1);
                out[i] = format!("{} {}", out[i].trim_end(), next.trim_start());
            } else {
                continue;
            }
            changed = true;
            break;
        }
    }
    out
}

fn ensure_terminal_punct(chunk: &str) -> String {
    let c = chunk.trim_end();
    if c.is_empty() || c.ends_with(|ch: char| ".!?,;:\"\u{2019}\u{201d}')]".contains(ch)) {
        return c.to_string();
    }
    format!("{c},")
}

/// The in-context reference: transcript plus its audio as codec tokens.
pub fn build_reference_prefix(tm: &TokenMap, transcript_ids: &[u32], audio: &[u32]) -> Vec<u32> {
    let audio_ids = audio.iter().map(|&t| tm.audio_id_base + t);
    let mut out = vec![tm.start_id];
    if tm.uses_reference_role_tokens() {
        out.push(tm.reference_text_start_id);
        out.extend_from_slice(transcript_ids);
        out.push(tm.reference_text_end_id);
        out.push(tm.reference_speech_start_id);
        out.extend(audio_ids);
        out.push(tm.reference_speech_end_id);
    } else {
        out.push(tm.text_start_id);
        out.extend_from_slice(transcript_ids);
        out.push(tm.speech_start_id);
        out.extend(audio_ids);
        out.push(tm.speech_end_id);
    }
    out
}

/// Full prompt for one chunk: reference prefix (or a bare `[START]`), the
/// optional emotion control, the target text, the end-of-sample cue, and the
/// `<speech_start>` the model continues from.
pub fn build_generation_prompt(
    tm: &TokenMap,
    target_ids: &[u32],
    reference_prefix: Option<&[u32]>,
    emotion_ids: Option<&[u32]>,
) -> Vec<u32> {
    let mut out = match reference_prefix {
        Some(p) if !p.is_empty() => p.to_vec(),
        _ => vec![tm.start_id],
    };
    out.extend_from_slice(emotion_ids.unwrap_or_default());
    out.push(tm.text_start_id);
    out.extend_from_slice(target_ids);
    if tm.final_seg_id != 0 {
        out.push(tm.final_seg_id);
    }
    out.push(tm.speech_start_id);
    out
}

/// Strip a segment's lead-in and trail-out silence (10 ms RMS frames below
/// `thresh_db`), keeping `keep_s` of margin.
pub fn trim_silence(wav: &[f32], sample_rate: u32, thresh_db: f32, keep_s: f32) -> Vec<f32> {
    let sr = sample_rate as f32;
    if wav.len() < (sr * 0.05) as usize {
        return wav.to_vec();
    }
    let hop = ((sr * 0.01) as usize).max(1);
    let thresh = 10f32.powf(thresh_db / 20.0);
    let loud: Vec<usize> = wav
        .chunks_exact(hop)
        .enumerate()
        .filter(|(_, f)| {
            (f.iter().map(|x| x * x).sum::<f32>() / hop as f32 + 1e-12).sqrt() > thresh
        })
        .map(|(i, _)| i)
        .collect();
    let (Some(&first), Some(&last)) = (loud.first(), loud.last()) else {
        return wav.to_vec();
    };
    let keep = (sr * keep_s) as usize;
    let start = (first * hop).saturating_sub(keep);
    let end = ((last + 1) * hop + keep).min(wav.len());
    wav[start..end].to_vec()
}

/// Concatenate chunk audio: trim each segment's edge silence, insert one
/// `gap_s` pause, and fade 8 ms at every edge so seams do not click.
pub fn join_chunks(wavs: &[Vec<f32>], sample_rate: u32, gap_s: f32) -> Option<Vec<f32>> {
    let wavs: Vec<&Vec<f32>> = wavs.iter().filter(|w| !w.is_empty()).collect();
    match wavs.len() {
        0 => return None,
        1 => return Some(wavs[0].clone()),
        _ => {}
    }
    let sr = sample_rate as f32;
    let gap = vec![0f32; (sr * gap_s) as usize];
    let fade = ((sr * 0.008) as usize).max(1);
    let ramp = |i: usize| {
        if fade > 1 {
            i as f32 / (fade - 1) as f32
        } else {
            1.0
        }
    };
    let mut out = Vec::new();
    for (i, w) in wavs.iter().enumerate() {
        let mut w = trim_silence(w, sample_rate, -45.0, 0.02);
        if w.len() > 2 * fade {
            let n = w.len();
            for k in 0..fade {
                w[k] *= ramp(k);
                w[n - fade + k] *= 1.0 - ramp(k);
            }
        }
        out.extend_from_slice(&w);
        if i + 1 != wavs.len() {
            out.extend_from_slice(&gap);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markup_survives_clean_up_verbatim() {
        let s = normalize_text(
            "[joyful] we won the grant <laugh> i can (((hardly))) believe it",
            false,
        );
        assert_eq!(
            s,
            "[joyful] We won the grant <laugh> i can (((hardly))) believe it."
        );
        assert!(has_expression_tag(&s));
        assert!(!has_expression_tag("section [3] says x < 5"));
    }

    #[test]
    fn brackets_dashes_and_glyphs_become_speakable() {
        assert_eq!(
            normalize_text("the result (see below) holds", false),
            "The result, see below, holds."
        );
        assert_eq!(normalize_text("wait\u{2014}what", false), "Wait, what.");
        assert_eq!(normalize_text("x = y\u{b2}", false), "X equals y 2.");
        assert_eq!(
            normalize_text("release v0.8 is out!", false),
            "Release v 0 point 8 is out!"
        );
        assert_eq!(
            normalize_text("line one.\nLine two", false),
            "Line one. Line two."
        );
    }

    #[test]
    fn short_text_is_one_chunk_and_long_text_splits_on_sentences() {
        assert_eq!(
            split_for_synthesis("Hello there.", 380, 130),
            ["Hello there."]
        );
        let sentence = "This sentence is exactly long enough to matter here. ";
        let text = sentence.repeat(12);
        let chunks = split_for_synthesis(&text, 380, 130);
        // Upstream `split_for_synthesis` on the same input.
        assert_eq!(
            chunks.iter().map(|c| char_len(c)).collect::<Vec<_>>(),
            [370, 264]
        );
        for c in &chunks {
            assert!(char_len(c) <= (380.0 * 1.4) as usize, "{c}");
            assert!(c.ends_with('.'), "{c}");
        }
        assert_eq!(chunks.join(" "), text.trim());
    }

    #[test]
    fn prompt_layout_matches_upstream() {
        let tm: TokenMap = serde_json::from_value(serde_json::json!({
            "audio_id_base": 100, "num_audio_tokens": 10, "speech_start_id": 1,
            "speech_end_id": 2, "text_start_id": 3, "text_end_id": 4, "start_id": 5,
            "stop_id": 6, "final_seg_id": 7, "reference_speech_start_id": 8,
            "reference_speech_end_id": 9, "reference_text_start_id": 10,
            "reference_text_end_id": 11
        }))
        .unwrap();
        let prefix = build_reference_prefix(&tm, &[40, 41], &[0, 3]);
        assert_eq!(prefix, [5, 10, 40, 41, 11, 8, 100, 103, 9]);
        let prompt = build_generation_prompt(&tm, &[50], Some(&prefix), Some(&[60]));
        assert_eq!(prompt[prefix.len()..], [60, 3, 50, 7, 1]);
    }

    #[test]
    fn joining_inserts_one_gap_between_trimmed_segments() {
        let sr = 1000;
        let tone: Vec<f32> = (0..400).map(|i| (i as f32 * 0.3).sin() * 0.5).collect();
        let mut padded = vec![0.0; 200];
        padded.extend(&tone);
        padded.extend(vec![0.0; 200]);
        let out = join_chunks(&[padded.clone(), padded], sr, 0.1).unwrap();
        // 400 tone + 2×20 ms margin per side, twice, plus the 100-sample gap.
        assert_eq!(out.len(), 2 * (400 + 40) + 100);
    }
}
