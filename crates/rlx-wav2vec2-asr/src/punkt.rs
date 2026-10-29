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

//! NLTK 3.10's Punkt sentence splitter (`nltk.tokenize.punkt`), applying a
//! trained `punkt_tab` model: `span_tokenize` with boundary realignment,
//! the first-pass (type) and second-pass (collocation, orthographic,
//! sentence-starter, initial/number) annotations. WhisperX splits each
//! transcript segment into sentences with the `english` model, which is
//! embedded here (`punkt_tab/english/*`, nltk_data).
//!
//! The regular expressions are NLTK's, written out compactly (they are
//! `re.VERBOSE` in NLTK) and run on `fancy-regex` (lookahead, lazy
//! quantifiers), with Python's `match` anchored by `\A`.

use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const ORTHO_BEG_UC: u32 = 1 << 1;
const ORTHO_MID_UC: u32 = 1 << 2;
const ORTHO_UNK_UC: u32 = 1 << 3;
const ORTHO_BEG_LC: u32 = 1 << 4;
const ORTHO_MID_LC: u32 = 1 << 5;
const ORTHO_UNK_LC: u32 = 1 << 6;
const ORTHO_UC: u32 = ORTHO_BEG_UC + ORTHO_MID_UC + ORTHO_UNK_UC;
const ORTHO_LC: u32 = ORTHO_BEG_LC + ORTHO_MID_LC + ORTHO_UNK_LC;

const SENT_END: [char; 3] = ['.', '?', '!'];
const PUNCTUATION: [&str; 6] = [";", ":", ",", ".", "!", "?"];

/// A trained Punkt model (`PunktParameters`).
pub struct PunktParams {
    abbrev_types: HashSet<String>,
    collocations: HashSet<(String, String)>,
    sent_starters: HashSet<String>,
    ortho_context: HashMap<String, u32>,
}

impl PunktParams {
    /// `load_punkt_params`: the four `punkt_tab` files.
    pub fn from_tab(abbrev: &str, collocations: &str, starters: &str, ortho: &str) -> Self {
        let set = |s: &str| s.lines().map(str::to_string).collect::<HashSet<_>>();
        let tups = |s: &str| {
            s.lines()
                .map(|l| l.split('\t').map(str::to_string).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        PunktParams {
            abbrev_types: set(abbrev),
            collocations: tups(collocations)
                .into_iter()
                .filter(|t| t.len() == 2)
                .map(|t| (t[0].clone(), t[1].clone()))
                .collect(),
            sent_starters: set(starters),
            ortho_context: tups(ortho)
                .into_iter()
                .filter(|t| t.len() == 2)
                .map(|t| (t[0].clone(), t[1].parse().unwrap_or(0)))
                .collect(),
        }
    }

    /// The `english` model shipped with nltk_data (`punkt_tab`).
    pub fn english() -> &'static PunktParams {
        static P: OnceLock<PunktParams> = OnceLock::new();
        P.get_or_init(|| {
            PunktParams::from_tab(
                include_str!("../data/punkt_tab/english/abbrev_types.txt"),
                include_str!("../data/punkt_tab/english/collocations.tab"),
                include_str!("../data/punkt_tab/english/sent_starters.txt"),
                include_str!("../data/punkt_tab/english/ortho_context.tab"),
            )
        })
    }
}

struct Res {
    word: Regex,
    period_context: Regex,
    realign: Regex,
    numeric: Regex,
    ellipsis: Regex,
    initial: Regex,
}

fn res() -> &'static Res {
    static R: OnceLock<Res> = OnceLock::new();
    R.get_or_init(|| {
        let non_word = r#"(?:[)";}\]\*:@'\({\[\x{2018}\x{2019}\x{201c}\x{201d}\x{ab}\x{bb}\?!])"#;
        let multi = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)";
        let word_start = r#"[^\("`{\[:;&\#\*@\)}\]\-,]"#;
        let word = format!(r"({multi}|(?={word_start})\S+?(?=\s|$|{non_word}|{multi}|,(?=$|\s|{non_word}|{multi}))|\S)");
        let period_context = format!(r"[\.\?!](?=(?P<after_tok>{non_word}|\s+(?P<next_tok>\S+)))");
        Res {
            word: Regex::new(&word).unwrap(),
            period_context: Regex::new(&period_context).unwrap(),
            realign: Regex::new(r#"(?m)\A["')\]}\x{2018}\x{2019}\x{201c}\x{201d}\x{ab}\x{bb}]+?(?:\s+|(?=--)|$)"#).unwrap(),
            numeric: Regex::new(r"^-?[\.,]?\d[\d,\.-]*\.?$").unwrap(),
            ellipsis: Regex::new(r"\A\.\.+$").unwrap(),
            initial: Regex::new(r"\A[^\W\d]\.$").unwrap(),
        }
    })
}

#[derive(Debug)]
struct Token {
    tok: String,
    typ: String,
    period_final: bool,
    sentbreak: bool,
    abbr: bool,
    ellipsis: bool,
}

impl Token {
    fn new(tok: &str) -> Self {
        let lower = tok.to_lowercase();
        let typ = if res().numeric.is_match(&lower).unwrap_or(false) {
            "##number##".to_string()
        } else {
            lower
        };
        Token {
            tok: tok.to_string(),
            typ,
            period_final: tok.ends_with('.'),
            sentbreak: false,
            abbr: false,
            ellipsis: false,
        }
    }
    fn type_no_period(&self) -> &str {
        if self.typ.chars().count() > 1 && self.typ.ends_with('.') {
            &self.typ[..self.typ.len() - 1]
        } else {
            &self.typ
        }
    }
    fn type_no_sentperiod(&self) -> &str {
        if self.sentbreak {
            self.type_no_period()
        } else {
            &self.typ
        }
    }
    fn first_upper(&self) -> bool {
        self.tok.chars().next().is_some_and(char::is_uppercase)
    }
    fn first_lower(&self) -> bool {
        self.tok.chars().next().is_some_and(char::is_lowercase)
    }
    fn is_initial(&self) -> bool {
        res().initial.is_match(&self.tok).unwrap_or(false)
    }
}

#[derive(PartialEq)]
enum Ortho {
    Yes,
    No,
    Unknown,
}

pub struct Punkt {
    params: &'static PunktParams,
}

impl Punkt {
    pub fn english() -> Self {
        Punkt {
            params: PunktParams::english(),
        }
    }

    /// `_tokenize_words` (line starts / paragraph starts do not affect the
    /// decisions `span_tokenize` makes, so they are not tracked).
    fn tokenize_words(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        for line in text.split('\n') {
            if line.trim().is_empty() {
                continue;
            }
            for m in res().word.find_iter(line).flatten() {
                out.push(Token::new(m.as_str()));
            }
        }
        out
    }

    fn first_pass(&self, t: &mut Token) {
        let tok = t.tok.as_str();
        if tok.len() == 1 && SENT_END.contains(&tok.chars().next().unwrap()) {
            t.sentbreak = true;
        } else if res().ellipsis.is_match(tok).unwrap_or(false) {
            t.ellipsis = true;
        } else if t.period_final && !tok.ends_with("..") {
            let stem = tok[..tok.len() - 1].to_lowercase();
            let last = stem.rsplit('-').next().unwrap_or("");
            if self.params.abbrev_types.contains(&stem) || self.params.abbrev_types.contains(last) {
                t.abbr = true;
            } else {
                t.sentbreak = true;
            }
        }
    }

    fn ortho_heuristic(&self, t: &Token) -> Ortho {
        if PUNCTUATION.contains(&t.tok.as_str()) {
            return Ortho::No;
        }
        let oc = self
            .params
            .ortho_context
            .get(t.type_no_sentperiod())
            .copied()
            .unwrap_or(0);
        if t.first_upper() && (oc & ORTHO_LC) != 0 && (oc & ORTHO_MID_UC) == 0 {
            return Ortho::Yes;
        }
        if t.first_lower() && ((oc & ORTHO_UC) != 0 || (oc & ORTHO_BEG_LC) == 0) {
            return Ortho::No;
        }
        Ortho::Unknown
    }

    fn second_pass(&self, t1: &mut Token, t2: &Token) {
        if !t1.period_final {
            return;
        }
        let typ = t1.type_no_period().to_string();
        let next_typ = t2.type_no_sentperiod().to_string();
        let initial = t1.is_initial();
        if self
            .params
            .collocations
            .contains(&(typ.clone(), next_typ.clone()))
        {
            t1.sentbreak = false;
            t1.abbr = true;
            return;
        }
        if (t1.abbr || t1.ellipsis) && !initial {
            if self.ortho_heuristic(t2) == Ortho::Yes {
                t1.sentbreak = true;
                return;
            }
            if t2.first_upper() && self.params.sent_starters.contains(&next_typ) {
                t1.sentbreak = true;
                return;
            }
        }
        if initial || typ == "##number##" {
            let s = self.ortho_heuristic(t2);
            if s == Ortho::No {
                t1.sentbreak = false;
                t1.abbr = true;
                return;
            }
            if s == Ortho::Unknown
                && initial
                && t2.first_upper()
                && (self
                    .params
                    .ortho_context
                    .get(&next_typ)
                    .copied()
                    .unwrap_or(0)
                    & ORTHO_LC)
                    == 0
            {
                t1.sentbreak = false;
                t1.abbr = true;
            }
        }
    }

    /// `text_contains_sentbreak`: a break followed by any further token.
    fn contains_sentbreak(&self, text: &str) -> bool {
        let mut toks = self.tokenize_words(text);
        for t in toks.iter_mut() {
            self.first_pass(t);
        }
        let mut found = false;
        for i in 0..toks.len() {
            if i + 1 < toks.len() {
                let (a, b) = toks.split_at_mut(i + 1);
                self.second_pass(&mut a[i], &b[0]);
            }
            if found {
                return true;
            }
            if toks[i].sentbreak {
                found = true;
            }
        }
        false
    }

    /// `_match_potential_end_contexts`: (match start, match end, next_tok
    /// start, context). Offsets are in characters.
    fn end_contexts(
        &self,
        text: &str,
        cidx: &[usize],
    ) -> Vec<(usize, usize, Option<usize>, String)> {
        let ch = |b: usize| cidx.partition_point(|&x| x < b);
        let chars: Vec<char> = text.chars().collect();
        let sub = |a: usize, b: usize| chars[a..b].iter().collect::<String>();
        let mut out = Vec::new();
        let (mut prev_start, mut prev_stop) = (0usize, 0usize);
        let mut prev: Option<(usize, usize, Option<usize>, String)> = None; // (start, end, next_tok, group+after)
        for caps in res().period_context.captures_iter(text).flatten() {
            let m = caps.get(0).unwrap();
            let (ms, me) = (ch(m.start()), ch(m.end()));
            let after = caps
                .name("after_tok")
                .map(|g| g.as_str().to_string())
                .unwrap_or_default();
            let next_tok = caps.name("next_tok").map(|g| ch(g.start()));
            let before: &[char] = &chars[prev_stop..ms];
            let mut idx = (0..before.len())
                .rev()
                .find(|&i| " \t\n\r\x0b\x0c".contains(before[i]))
                .unwrap_or(0);
            if idx != 0 {
                idx += prev_stop + 1;
            } else {
                idx = prev_start;
            }
            if let Some((ps, pe, pn, pg)) = prev.take() {
                if prev_stop <= idx {
                    out.push((ps, pe, pn, sub(prev_start, prev_stop) + &pg));
                }
            }
            prev = Some((ms, me, next_tok, m.as_str().to_string() + &after));
            prev_start = idx;
            prev_stop = ms;
        }
        if let Some((ps, pe, pn, pg)) = prev {
            out.push((ps, pe, pn, sub(prev_start, prev_stop) + &pg));
        }
        out
    }

    /// `span_tokenize(text)` with `realign_boundaries=True`: sentence spans in
    /// character offsets (Python string indices).
    pub fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let cidx: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        let chars: Vec<char> = text.chars().collect();
        let mut slices = Vec::new();
        let mut last_break = 0usize;
        for (_s, e, next_tok, ctx) in self.end_contexts(text, &cidx) {
            if self.contains_sentbreak(&ctx) {
                slices.push((last_break, e));
                last_break = next_tok.unwrap_or(e);
            }
        }
        let rstripped = text
            .trim_end_matches(|c: char| c.is_whitespace())
            .chars()
            .count();
        slices.push((last_break, rstripped));
        // _realign_boundaries
        let sub = |a: usize, b: usize| -> String {
            if a < b {
                chars[a..b].iter().collect()
            } else {
                String::new()
            }
        };
        let mut out = Vec::new();
        let mut realign = 0usize;
        for i in 0..slices.len() {
            let s1 = (slices[i].0 + realign, slices[i].1);
            if i + 1 == slices.len() {
                if !sub(s1.0, s1.1).is_empty() {
                    out.push(s1);
                }
                continue;
            }
            let s2 = slices[i + 1];
            let s2text = sub(s2.0, s2.1);
            match res().realign.find(&s2text).ok().flatten() {
                Some(m) => {
                    let g = m.as_str();
                    let kept = g
                        .trim_end_matches(|c: char| c.is_whitespace())
                        .chars()
                        .count();
                    out.push((s1.0, s2.0 + kept));
                    realign = s2text[..m.end()].chars().count();
                }
                None => {
                    realign = 0;
                    if !sub(s1.0, s1.1).is_empty() {
                        out.push(s1);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nltk_doc_examples() {
        let p = Punkt::english();
        let t = "Hello! How are you?";
        assert_eq!(p.span_tokenize(t), vec![(0, 6), (7, 19)]);
        let t = "(Sent1.) Sent2.";
        let spans = p.span_tokenize(t);
        let s: Vec<String> = spans
            .iter()
            .map(|&(a, b)| t.chars().skip(a).take(b - a).collect())
            .collect();
        assert_eq!(s, vec!["(Sent1.)", "Sent2."]);
    }
}
