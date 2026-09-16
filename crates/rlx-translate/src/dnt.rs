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

//! `DoNotTranslateBlock`: spans that must survive translation unchanged.
//!
//! The config names no parameters for this block — it just receives the
//! original source and the final target — so the rule set was learned by
//! probing the live framework rather than guessed. Across en->fr, en->de and
//! en->ja the OS preserves these verbatim:
//!
//! ```text
//! https://www.example.com   john.smith@example.com   @SomeHandle   #WWDC2026
//! README.md               +1 415 555 0123          AA123           ABC-123-XYZ
//! ```
//!
//! Two probes did *not* survive, and both are correct behaviour rather than
//! counter-examples: `cargo build --release` is ordinary words and gets
//! translated (`la construction de cargaison --libérer`), and `$99.99` becomes
//! `99,99 $`, which is French currency formatting, not a lost span. So the rule
//! is about *identifiers*, not about anything unusual-looking.
//!
//! Product names (`iPad Pro`, `iPhone 15 Pro Max`) also survive, but they are
//! not pattern-matchable — the model simply keeps them, so they are out of
//! scope here.
//!
//! # Detection, not repair
//!
//! This reports which spans went missing. Putting one *back* needs to know
//! where it belongs in the target, which is what `AlignmentProcessorBlock`
//! is for and that block is not implemented — so guessing an insertion point
//! would be worse than reporting the problem.

/// A span of the source that should appear unchanged in the translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span<'a> {
    pub text: &'a str,
    pub kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Url,
    Email,
    Handle,
    Hashtag,
    Filename,
    Phone,
    /// Mixed letters-and-digits identifier: `AA123`, `ABC-123-XYZ`.
    Code,
}

/// Classifies one whitespace-delimited token, ignoring trailing punctuation.
fn classify(tok: &str) -> Option<Kind> {
    // Sentence punctuation clings to the last token; the span itself does not
    // include it, and neither should the comparison.
    let t = tok.trim_end_matches(['.', ',', ';', ':', '!', '?', ')', '"', '\'']);
    if t.len() < 2 {
        return None;
    }
    if t.starts_with("http://") || t.starts_with("https://") || t.starts_with("www.") {
        return Some(Kind::Url);
    }
    if t.starts_with('@') && t[1..].chars().any(char::is_alphanumeric) {
        return Some(Kind::Handle);
    }
    if t.starts_with('#') && t[1..].chars().any(char::is_alphanumeric) {
        return Some(Kind::Hashtag);
    }
    let ats = t.matches('@').count();
    if ats == 1
        && let Some((user, host)) = t.split_once('@')
        && !user.is_empty()
        && host.contains('.')
        && !host.ends_with('.')
    {
        return Some(Kind::Email);
    }
    // A filename is `stem.ext` with a short alphabetic extension. Requiring the
    // extension to be alphabetic keeps ordinary decimals out.
    if let Some((stem, ext)) = t.rsplit_once('.')
        && !stem.is_empty()
        && (2..=4).contains(&ext.len())
        && ext.chars().all(|c| c.is_ascii_alphabetic())
        && stem.chars().any(char::is_alphanumeric)
    {
        return Some(Kind::Filename);
    }
    let digits = t.chars().filter(char::is_ascii_digit).count();
    let letters = t.chars().filter(|c| c.is_alphabetic()).count();
    if t.starts_with('+') && digits >= 7 && letters == 0 {
        return Some(Kind::Phone);
    }
    // `AA123`, `ABC-123-XYZ`: letters *and* digits in one token. A plain number
    // is not a code — it gets localized (`99.99` -> `99,99`) — and a plain word
    // is just a word.
    if digits > 0 && letters > 0 && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Some(Kind::Code);
    }
    None
}

/// Every do-not-translate span in `source`, in order.
///
/// Phone numbers are the one multi-token case: `+1 415 555 0123` is four
/// whitespace-separated pieces and only means anything joined.
pub fn spans(source: &str) -> Vec<Span<'_>> {
    let toks: Vec<&str> = source.split_whitespace().collect();
    let mut out: Vec<Span<'_>> = Vec::new();
    let mut i = 0usize;
    while i < toks.len() {
        if toks[i].starts_with('+') && toks[i][1..].chars().all(|c| c.is_ascii_digit()) {
            // Absorb the following bare-digit groups into one phone span.
            let mut j = i + 1;
            while j < toks.len() && toks[j].chars().all(|c| c.is_ascii_digit()) && j - i < 6 {
                j += 1;
            }
            let digits: usize = toks[i..j]
                .iter()
                .map(|t| t.matches(char::is_numeric).count())
                .sum();
            if j > i + 1 && digits >= 7 {
                let start = toks[i].as_ptr() as usize - source.as_ptr() as usize;
                let last = toks[j - 1];
                let end = last.as_ptr() as usize - source.as_ptr() as usize + last.len();
                out.push(Span {
                    text: &source[start..end],
                    kind: Kind::Phone,
                });
                i = j;
                continue;
            }
        }
        if let Some(kind) = classify(toks[i]) {
            let t = toks[i].trim_end_matches(['.', ',', ';', ':', '!', '?', ')', '"', '\'']);
            out.push(Span { text: t, kind });
        }
        i += 1;
    }
    out
}

/// Spans of `source` that failed to survive into `target`.
pub fn lost<'a>(source: &'a str, target: &str) -> Vec<Span<'a>> {
    // Case-insensitively: the sentence caser runs after this and turns a
    // leading `#hashtag` into `#Hashtag`, which is not a lost span but was
    // reported as one.
    let folded = target.to_lowercase();
    spans(source)
        .into_iter()
        .filter(|s| !folded.contains(&s.text.to_lowercase()))
        .collect()
}

/// The writing system a run of text belongs to, coarsely.
///
/// Only the distinctions this crate's languages need. `Latin` covers ASCII and
/// the accented ranges; `Other` is anything unclassified, which is treated as
/// not worth protecting rather than guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Latin,
    Cyrillic,
    Arabic,
    Devanagari,
    Thai,
    Han,
    Kana,
    Hangul,
    Other,
}

/// The script of `c`, or `None` for anything script-neutral (digits, spaces,
/// punctuation) that should join whichever run it sits in.
pub fn script_of(c: char) -> Option<Script> {
    // A script's own digits belong to it. This matters for repair rather than
    // classification: a corrupted Arabic word can come back containing U+0667,
    // which is not alphabetic, and treating it as neutral splits one run into
    // two so the counts stop matching and nothing is repaired.
    match c as u32 {
        0x0660..=0x0669 | 0x06F0..=0x06F9 => return Some(Script::Arabic),
        0x0966..=0x096F => return Some(Script::Devanagari),
        0x0E50..=0x0E59 => return Some(Script::Thai),
        _ => {}
    }
    if !c.is_alphabetic() {
        return None;
    }
    Some(match c as u32 {
        0x0000..=0x024F | 0x1E00..=0x1EFF => Script::Latin,
        0x0370..=0x03FF => Script::Other, // Greek
        0x0400..=0x052F => Script::Cyrillic,
        0x0590..=0x05FF => Script::Other, // Hebrew
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF => Script::Arabic,
        0x0900..=0x097F => Script::Devanagari,
        0x0E00..=0x0E7F => Script::Thai,
        0x3040..=0x30FF => Script::Kana,
        0xAC00..=0xD7AF | 0x1100..=0x11FF => Script::Hangul,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF => Script::Han,
        _ => Script::Other,
    })
}

/// The script a locale is written in.
pub fn script_for_locale(locale: &str) -> Script {
    match locale.split(['-', '_']).next().unwrap_or(locale) {
        "ru" | "uk" => Script::Cyrillic,
        "ar" => Script::Arabic,
        "hi" => Script::Devanagari,
        "th" => Script::Thai,
        "zh" => Script::Han,
        "ja" => Script::Kana,
        "ko" => Script::Hangul,
        _ => Script::Latin,
    }
}

/// Maximal runs of `want`, as `(byte offset, text)`.
fn runs_of(text: &str, want: Script) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut cur: Option<(usize, String)> = None;
    for (i, c) in text.char_indices() {
        if script_of(c) == Some(want) {
            cur.get_or_insert((i, String::new())).1.push(c);
        } else if let Some(r) = cur.take() {
            out.push(r);
        }
    }
    out.extend(cur);
    out
}

/// Restores runs of a script that neither language uses.
///
/// A model asked to carry text it cannot represent does not drop it, it
/// *alters* it: `en_US-fr_FR` renders `"السلام hello"` as `"Bonjour السلنم"`,
/// one Arabic letter changed, and in the other word order it produces an
/// Arabic-Indic digit. The framework returns the run untouched. Silent
/// corruption of text the user typed is worse than leaving it untranslated.
///
/// Deliberately conservative: it only rewrites when the output holds exactly as
/// many runs of that script as the source did, so a genuine translation into
/// that script is never overwritten.
pub fn restore_foreign_scripts(source: &str, target: &str, src_loc: &str, tgt_loc: &str) -> String {
    let (a, b) = (script_for_locale(src_loc), script_for_locale(tgt_loc));
    let mut out = target.to_string();
    for script in [
        Script::Cyrillic,
        Script::Arabic,
        Script::Devanagari,
        Script::Thai,
        Script::Han,
        Script::Kana,
        Script::Hangul,
    ] {
        // Latin is never foreign — it carries names and codes in every language
        // — and a script either side actually writes is the model's business.
        if script == a || script == b {
            continue;
        }
        let want = runs_of(source, script);
        if want.is_empty() {
            continue;
        }
        let got = runs_of(&out, script);
        if got.len() != want.len() || got.iter().zip(&want).all(|(g, w)| g.1 == w.1) {
            continue;
        }
        // Right to left, so earlier offsets stay valid.
        for (g, w) in got.iter().zip(&want).rev() {
            out.replace_range(g.0..g.0 + g.1.len(), &w.1);
        }
    }
    out
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_lost_span_is_not_lost_merely_by_being_capitalised() {
        // The sentence caser runs after the check, so a leading `#hashtag`
        // arrives as `#Hashtag`. That was reported as a span lost in
        // translation on every sentence starting with one.
        assert!(lost("#hashtag and @handle", "#Hashtag et @handle").is_empty());
        assert!(!lost("visit https://example.com now", "visitez maintenant").is_empty());
    }

    #[test]
    fn foreign_script_runs_are_put_back() {
        // Measured: `en_US-fr_FR` renders "السلام hello" as "Bonjour السلنم",
        // one letter altered, and the other word order yields an Arabic-Indic
        // digit. The framework returns the run untouched.
        let out = restore_foreign_scripts(
            "\u{627}\u{644}\u{633}\u{644}\u{627}\u{645} hello",
            "Bonjour \u{627}\u{644}\u{633}\u{644}\u{646}\u{645}",
            "en_US",
            "fr_FR",
        );
        assert_eq!(out, "Bonjour \u{627}\u{644}\u{633}\u{644}\u{627}\u{645}");
        // A digit inside the corrupted run must not split it in two, or the
        // counts stop matching and nothing is repaired.
        let out = restore_foreign_scripts(
            "hello \u{627}\u{644}\u{633}\u{644}\u{627}\u{645}",
            "Bonjour \u{627}\u{644}\u{633}\u{644}\u{667}\u{645}",
            "en_US",
            "fr_FR",
        );
        assert_eq!(out, "Bonjour \u{627}\u{644}\u{633}\u{644}\u{627}\u{645}");
    }

    #[test]
    fn a_script_either_language_writes_is_left_alone() {
        // Translating *into* Arabic must not have its output overwritten with
        // the source's Arabic, and the same for the source side.
        let src = "\u{627}\u{644}\u{633}\u{644}\u{627}\u{645}";
        let tgt = "\u{645}\u{631}\u{62d}\u{628}\u{627}";
        assert_eq!(restore_foreign_scripts(src, tgt, "en_US", "ar_AE"), tgt);
        assert_eq!(restore_foreign_scripts(src, tgt, "ar_AE", "fr_FR"), tgt);
    }

    #[test]
    fn repair_only_fires_when_the_runs_line_up() {
        let src = "\u{627}\u{644}\u{633} and \u{645}\u{631}\u{62d}";
        // Two runs in, one out: ambiguous, so nothing is touched.
        let one = "Bonjour \u{627}\u{644}\u{646}";
        assert_eq!(restore_foreign_scripts(src, one, "en_US", "fr_FR"), one);
        // None out: nothing to repair, and nothing invented.
        assert_eq!(
            restore_foreign_scripts(src, "Bonjour", "en_US", "fr_FR"),
            "Bonjour"
        );
    }
    use super::*;

    fn kinds(s: &str) -> Vec<(Kind, &str)> {
        spans(s).into_iter().map(|s| (s.kind, s.text)).collect()
    }

    /// Every span the live framework was observed to preserve.
    #[test]
    fn the_probed_identifiers_are_all_detected() {
        assert_eq!(
            kinds("visit https://www.example.com today"),
            [(Kind::Url, "https://www.example.com")]
        );
        assert_eq!(
            kinds("email me at john.smith@example.com"),
            [(Kind::Email, "john.smith@example.com")]
        );
        assert_eq!(
            kinds("follow @SomeHandle for updates"),
            [(Kind::Handle, "@SomeHandle")]
        );
        assert_eq!(
            kinds("the hashtag is #WWDC2026"),
            [(Kind::Hashtag, "#WWDC2026")]
        );
        assert_eq!(
            kinds("the file is README.md in the repo"),
            [(Kind::Filename, "README.md")]
        );
        assert_eq!(kinds("my flight AA123 leaves"), [(Kind::Code, "AA123")]);
        assert_eq!(
            kinds("the code is ABC-123-XYZ"),
            [(Kind::Code, "ABC-123-XYZ")]
        );
        assert_eq!(
            kinds("call me on +1 415 555 0123"),
            [(Kind::Phone, "+1 415 555 0123")]
        );
    }

    /// The two probes that did *not* survive, and must not be claimed.
    #[test]
    fn currency_and_ordinary_words_are_not_protected() {
        // `$99.99` becomes `99,99 $` in French. That is localization, and
        // protecting it would make the output wrong.
        assert!(kinds("it costs $99.99 in total").is_empty());
        // `cargo build --release` is words; the OS translates it.
        assert!(kinds("run cargo build --release now").is_empty());
    }

    #[test]
    fn plain_numbers_and_words_are_left_alone() {
        assert!(kinds("i have 100 cherries and 3 pears").is_empty());
        assert!(kinds("the woman of my dreams").is_empty());
        assert!(
            kinds("version 2.0").is_empty(),
            "a bare decimal is not a file"
        );
    }

    #[test]
    fn trailing_punctuation_is_not_part_of_the_span() {
        assert_eq!(
            kinds("go to https://example.com."),
            [(Kind::Url, "https://example.com")]
        );
    }

    #[test]
    fn a_lost_span_is_reported_and_a_kept_one_is_not() {
        let src = "the file is README.md in the repo";
        assert!(lost(src, "Le fichier est README.md dans le dépôt").is_empty());
        let bad = lost(src, "Le fichier est LISEZMOI.md dans le dépôt");
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].text, "README.md");
    }
}
