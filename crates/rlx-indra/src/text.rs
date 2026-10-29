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

//! Text: the recognizer vocabularies, the Unicode properties the pipeline
//! tests (Python's `str.isalpha` & co.), and right-to-left reordering.
//!
//! A CTC recognizer reads every line left to right, so a line with Arabic or
//! Hebrew comes out in visual order; [`flip`] turns it into logical order (it
//! is its own inverse). A small subset of UAX #9: no embeddings or isolates,
//! neutrals resolved by runs — the rule the upstream labels were made with.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use unicode_bidi::BidiClass;
use unicode_general_category::{GeneralCategory as Gc, get_general_category};
use unicode_normalization::UnicodeNormalization;

/// A recognizer's tokens: class 0 is the CTC blank, class `i + 1` is `chars[i]`.
#[derive(Clone, Debug)]
pub struct Vocab {
    pub chars: Vec<String>,
    pub index: HashMap<String, u32>,
    has_commands: bool,
    jamo: bool,
}

#[derive(Deserialize)]
struct VocabJson {
    #[serde(default)]
    blank: u32,
    chars: Vec<String>,
}

impl Vocab {
    pub fn new(chars: Vec<String>) -> Self {
        let index = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i as u32 + 1))
            .collect::<HashMap<_, _>>();
        let has_commands = chars.iter().any(|c| c.chars().count() > 1);
        let jamo = index.contains_key("\u{1100}");
        Self {
            chars,
            index,
            has_commands,
            jamo,
        }
    }

    /// `meta.json`'s `vocab*` entries: a JSON string holding `{blank, chars}`.
    pub fn from_json(s: &str) -> Result<Self> {
        let v: VocabJson = serde_json::from_str(s)?;
        if v.blank != 0 {
            bail!("only blank = 0 vocabularies are supported");
        }
        Ok(Self::new(v.chars))
    }

    pub fn classes(&self) -> usize {
        self.chars.len() + 1
    }

    pub fn token(&self, id: u32) -> Option<&str> {
        (id > 0)
            .then(|| self.chars.get(id as usize - 1))
            .flatten()
            .map(String::as_str)
    }

    pub fn space(&self) -> Option<u32> {
        self.index.get(" ").copied()
    }

    /// Text of class ids: jamo composed into syllables, and a space after a
    /// LaTeX command that a letter follows (`\alpha x`, not `\alphax`).
    pub fn decode(&self, ids: &[u32]) -> String {
        let toks: Vec<&str> = ids.iter().filter_map(|&i| self.token(i)).collect();
        if self.jamo {
            return toks.concat().nfc().collect();
        }
        if !self.has_commands {
            return toks.concat();
        }
        let mut out = String::new();
        for (k, t) in toks.iter().enumerate() {
            out.push_str(t);
            let next = toks.get(k + 1).and_then(|n| n.chars().next());
            let cmd = t.len() > 2 && t.starts_with('\\') && t[1..].chars().all(is_alpha);
            if cmd && next.is_some_and(is_alpha) {
                out.push(' ');
            }
        }
        out
    }
}

// ── Python character predicates ─────────────────────────────────────────────

pub fn category(c: char) -> Gc {
    get_general_category(c)
}

/// `str.isalpha`: a letter (`L*`).
pub fn is_alpha(c: char) -> bool {
    matches!(
        category(c),
        Gc::UppercaseLetter
            | Gc::LowercaseLetter
            | Gc::TitlecaseLetter
            | Gc::ModifierLetter
            | Gc::OtherLetter
    )
}

/// `str.isdigit`: decimal digits plus the digit-valued superscripts,
/// subscripts and circled digits.
pub fn is_digit(c: char) -> bool {
    category(c) == Gc::DecimalNumber
        || matches!(c, '²' | '³' | '¹' | '\u{2070}' | '\u{2074}'..='\u{2079}' | '\u{2080}'..='\u{2089}'
            | '\u{2460}'..='\u{2468}' | '\u{2474}'..='\u{247C}' | '\u{2488}'..='\u{2490}'
            | '\u{24F5}'..='\u{24FD}' | '\u{24EA}' | '\u{24FF}' | '\u{2776}'..='\u{277E}'
            | '\u{2780}'..='\u{2788}' | '\u{278A}'..='\u{2792}' | '\u{1369}'..='\u{1371}' | '\u{19DA}')
}

pub fn is_upper(c: char) -> bool {
    c.is_uppercase()
}

pub fn is_lower(c: char) -> bool {
    c.is_lowercase()
}

pub fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn is_mark(c: char) -> bool {
    matches!(category(c), Gc::NonspacingMark | Gc::EnclosingMark)
}

/// East Asian Wide or Fullwidth (counts twice in a line's length).
pub fn is_wide(c: char) -> bool {
    unicode_width::UnicodeWidthChar::width(c) == Some(2)
}

/// `" ".join(text.split())`: whitespace runs to one space, trimmed.
pub fn squash_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn nfc(text: &str) -> String {
    text.nfc().collect()
}

// ── right to left ───────────────────────────────────────────────────────────

pub fn is_rtl(c: char) -> bool {
    matches!(unicode_bidi::bidi_class(c), BidiClass::R | BidiClass::AL)
}

/// Strong left-to-right, or a digit (digits read left to right in RTL text).
fn is_ltr(c: char) -> bool {
    matches!(
        unicode_bidi::bidi_class(c),
        BidiClass::L | BidiClass::EN | BidiClass::AN
    )
}

pub fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

/// More right-to-left letters than strong left-to-right ones.
pub fn rtl_base(text: &str) -> bool {
    let r = text.chars().filter(|&c| is_rtl(c)).count();
    let l = text
        .chars()
        .filter(|&c| unicode_bidi::bidi_class(c) == BidiClass::L)
        .count();
    r > l
}

/// Base characters with their combining marks.
fn clusters(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in text.chars() {
        match out.last_mut() {
            Some(last) if is_mark(c) => last.push(c),
            _ => out.push(c.to_string()),
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    R,
    L,
    N,
}

fn kind(cl: &str) -> Kind {
    let c = cl.chars().next().unwrap_or(' ');
    if is_rtl(c) {
        Kind::R
    } else if is_ltr(c) {
        Kind::L
    } else {
        Kind::N
    }
}

/// `[start, end)` spans from a `strong` cluster to the last `strong` one
/// before a cluster of kind `stop`.
fn runs(kinds: &[Kind], strong: Kind, stop: Kind) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let (mut start, mut last) = (None, 0);
    for i in 0..=kinds.len() {
        let k = kinds.get(i).copied();
        if k == Some(strong) {
            start.get_or_insert(i);
            last = i;
        } else if (k.is_none() || k == Some(stop))
            && let Some(s) = start.take()
        {
            spans.push((s, last + 1));
        }
    }
    spans
}

fn mirror(cl: &str) -> String {
    const PAIRS: [(char, char); 12] = [
        ('(', ')'),
        (')', '('),
        ('[', ']'),
        (']', '['),
        ('{', '}'),
        ('}', '{'),
        ('<', '>'),
        ('>', '<'),
        ('«', '»'),
        ('»', '«'),
        ('‹', '›'),
        ('›', '‹'),
    ];
    let mut chars = cl.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let m = PAIRS
        .iter()
        .find(|(a, _)| *a == first)
        .map_or(first, |(_, b)| *b);
    std::iter::once(m).chain(chars).collect()
}

/// Logical ↔ visual order. `base_rtl`: the paragraph direction when known
/// (`None` decides by [`rtl_base`]).
pub fn flip(text: &str, base_rtl: Option<bool>) -> String {
    if !has_rtl(text) {
        return text.to_string();
    }
    let cls = clusters(text);
    let kinds: Vec<Kind> = cls.iter().map(|c| kind(c)).collect();
    if base_rtl.unwrap_or_else(|| rtl_base(text)) {
        // Units: LTR runs kept whole, every other cluster alone; reverse them.
        let mut units: Vec<String> = Vec::new();
        let mut spans = runs(&kinds, Kind::L, Kind::R);
        spans.push((cls.len(), cls.len()));
        let mut i = 0;
        for (a, b) in spans {
            for c in &cls[i..a] {
                units.push(mirror(c));
            }
            if a < b {
                units.push(cls[a..b].concat());
            }
            i = b;
        }
        units.reverse();
        return units.concat();
    }
    let mut out = cls.clone();
    for (a, b) in runs(&kinds, Kind::R, Kind::L) {
        for (k, c) in cls[a..b].iter().rev().enumerate() {
            out[a + k] = mirror(c);
        }
    }
    out.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_is_its_own_inverse() {
        for s in [
            "abc",
            "שלום עולם",
            "מחיר 25 ש\"ח (כולל)",
            "see العربية here",
            "1 + אב",
        ] {
            assert_eq!(flip(&flip(s, None), None), s, "{s}");
        }
        assert_eq!(flip("םולש", Some(true)), "שלום");
        assert_eq!(flip("abc דגה xyz", Some(false)), "abc הגד xyz");
    }

    #[test]
    fn latex_decode_spacing() {
        let v = Vocab::new(vec!["\\alpha".into(), "x".into(), "+".into()]);
        assert_eq!(v.decode(&[1, 2, 3, 1, 3]), "\\alpha x+\\alpha+");
    }
}
