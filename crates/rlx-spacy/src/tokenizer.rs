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

//! spaCy 3.8's rule-based `Tokenizer` (`spacy/tokenizer.pyx`), loaded from a
//! pipeline's serialized `tokenizer` file: the whitespace walk, special cases
//! (exact-string exceptions), prefix/suffix splitting with the special-case
//! checks, infixes, `url_match`, and the `PhraseMatcher` pass that re-merges
//! special cases the affix pass split apart (`_apply_special_cases`).
//!
//! The pipeline's regular expressions are Python `re` patterns; they are
//! rewritten for `fancy-regex` (`\uXXXX` / `\UXXXXXXXX` escapes). Offsets are
//! in characters (Python string indices).

use anyhow::{Context, Result};
use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// One token: its text, whether a space follows, the `NORM` a special case
/// assigns (if any), and its character offset.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub text: String,
    pub whitespace: bool,
    pub norm: Option<String>,
    pub idx: usize,
}

#[derive(Clone, Debug)]
struct Special {
    orth: String,
    norm: Option<String>,
}

pub struct Tokenizer {
    prefix: Option<Regex>,
    suffix: Option<Regex>,
    infix: Option<Regex>,
    url: Option<Regex>,
    specials: HashMap<String, Vec<Special>>,
    /// Special-case token sequences for the post pass, by first token.
    phrases: HashMap<String, Vec<Vec<String>>>,
    max_phrase: usize,
}

/// Python `str.isspace()` for one character.
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Python `re` syntax -> `fancy-regex` syntax.
fn translate(p: &str) -> String {
    let cs: Vec<char> = p.chars().collect();
    let mut o = String::with_capacity(p.len());
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '\\' && i + 1 < cs.len() {
            match cs[i + 1] {
                'u' if i + 5 < cs.len() + 0
                    && cs[i + 2..i + 6].iter().all(|c| c.is_ascii_hexdigit()) =>
                {
                    o.push_str(&format!(
                        "\\x{{{}}}",
                        cs[i + 2..i + 6].iter().collect::<String>()
                    ));
                    i += 6;
                    continue;
                }
                'U' if i + 9 < cs.len() + 0
                    && cs[i + 2..i + 10].iter().all(|c| c.is_ascii_hexdigit()) =>
                {
                    o.push_str(&format!(
                        "\\x{{{}}}",
                        cs[i + 2..i + 10].iter().collect::<String>()
                    ));
                    i += 10;
                    continue;
                }
                c if !c.is_ascii_alphanumeric() && !"\\.^$|?*+()[]{}-#&~ ".contains(c) => {
                    // Python allows escaping any non-alphanumeric; keep it literal.
                    o.push_str(&fancy_regex::escape(&c.to_string()));
                    i += 2;
                    continue;
                }
                c => {
                    o.push('\\');
                    o.push(c);
                    i += 2;
                    continue;
                }
            }
        }
        o.push(cs[i]);
        i += 1;
    }
    o
}

fn compile(v: &rmpv::Value) -> Result<Option<Regex>> {
    match v.as_str() {
        Some(s) if !s.is_empty() => {
            Ok(Some(Regex::new(&translate(s)).with_context(|| {
                format!("regex {}", &s[..s.len().min(80)])
            })?))
        }
        _ => Ok(None),
    }
}

fn get<'a>(m: &'a rmpv::Value, key: &str) -> Option<&'a rmpv::Value> {
    m.as_map()?
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

/// Character length of a byte range of `s`.
fn chars(s: &str) -> usize {
    s.chars().count()
}

/// `s[a:]` / `s[:b]` in characters.
fn csub(s: &str, a: usize, b: usize) -> &str {
    let mut it = s
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(s.len()));
    let ba = it.clone().nth(a).unwrap_or(s.len());
    let bb = it.nth(b).unwrap_or(s.len());
    &s[ba..bb.max(ba)]
}

impl Tokenizer {
    /// Load `<pipeline>/tokenizer` (msgpack).
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let v = rmpv::decode::read_value(&mut bytes.as_slice())?;
        let mut specials = HashMap::new();
        if let Some(ex) = get(&v, "exceptions").and_then(|e| e.as_map()) {
            for (k, toks) in ex {
                let chunk = k.as_str().context("exception key")?.to_string();
                let mut seq = Vec::new();
                for t in toks.as_array().context("exception tokens")? {
                    let m = t.as_map().context("exception token")?;
                    let attr = |id: u64| {
                        m.iter()
                            .find(|(k, _)| k.as_u64() == Some(id))
                            .and_then(|(_, v)| v.as_str())
                            .map(str::to_string)
                    };
                    seq.push(Special {
                        orth: attr(65).context("ORTH")?,
                        norm: attr(67),
                    });
                }
                specials.insert(chunk, seq);
            }
        }
        let mut t = Tokenizer {
            prefix: compile(get(&v, "prefix_search").unwrap_or(&rmpv::Value::Nil))?,
            suffix: compile(get(&v, "suffix_search").unwrap_or(&rmpv::Value::Nil))?,
            infix: compile(get(&v, "infix_finditer").unwrap_or(&rmpv::Value::Nil))?,
            url: compile(get(&v, "url_match").unwrap_or(&rmpv::Value::Nil))?,
            specials,
            phrases: HashMap::new(),
            max_phrase: 0,
        };
        let faster = get(&v, "faster_heuristics")
            .and_then(|b| b.as_bool())
            .unwrap_or(true);
        // add_special_case: register in the PhraseMatcher when affixes or a
        // space could split the chunk.
        let mut chunks: Vec<String> = t.specials.keys().cloned().collect();
        chunks.sort();
        for c in chunks {
            if !faster
                || t.find_prefix(&c) > 0
                || t.has_infix(&c)
                || t.find_suffix(&c) > 0
                || c.contains(' ')
            {
                let toks: Vec<String> = t
                    .tokenize_affixes(&c, false)
                    .into_iter()
                    .map(|x| x.text)
                    .collect();
                if toks.is_empty() {
                    continue;
                }
                t.max_phrase = t.max_phrase.max(toks.len());
                let e = t.phrases.entry(toks[0].clone()).or_default();
                if !e.contains(&toks) {
                    e.push(toks);
                }
            }
        }
        Ok(t)
    }

    fn find_prefix(&self, s: &str) -> usize {
        match &self.prefix {
            Some(r) => r.find(s).ok().flatten().map_or(0, |m| chars(m.as_str())),
            None => 0,
        }
    }

    fn find_suffix(&self, s: &str) -> usize {
        match &self.suffix {
            Some(r) => r.find(s).ok().flatten().map_or(0, |m| chars(m.as_str())),
            None => 0,
        }
    }

    fn has_infix(&self, s: &str) -> bool {
        self.infix
            .as_ref()
            .is_some_and(|r| r.find(s).ok().flatten().is_some())
    }

    fn push_special(&self, out: &mut Vec<Token>, key: &str) -> bool {
        match self.specials.get(key) {
            Some(seq) => {
                for s in seq {
                    out.push(Token {
                        text: s.orth.clone(),
                        whitespace: false,
                        norm: s.norm.clone(),
                        idx: 0,
                    });
                }
                true
            }
            None => false,
        }
    }

    fn lex(text: &str) -> Token {
        Token {
            text: text.to_string(),
            whitespace: false,
            norm: None,
            idx: 0,
        }
    }

    /// `_split_affixes` + `_attach_tokens` for one non-space span.
    fn tokenize_span(&self, span: &str, out: &mut Vec<Token>, sp: bool) {
        let mut prefixes: Vec<String> = Vec::new();
        let mut suffixes: Vec<String> = Vec::new();
        let mut s: String = span.to_string();
        let mut last = usize::MAX;
        while !s.is_empty() && chars(&s) != last {
            if sp && self.specials.contains_key(&s) {
                break;
            }
            last = chars(&s);
            let pre_len = self.find_prefix(&s);
            let (mut prefix, mut minus_pre) = (String::new(), String::new());
            if pre_len != 0 {
                prefix = csub(&s, 0, pre_len).to_string();
                minus_pre = csub(&s, pre_len, usize::MAX).to_string();
                if !minus_pre.is_empty() && sp && self.specials.contains_key(&minus_pre) {
                    s = minus_pre;
                    prefixes.push(prefix);
                    break;
                }
            }
            let suf_len = self.find_suffix(csub(&s, pre_len, usize::MAX));
            let (mut suffix, mut minus_suf) = (String::new(), String::new());
            if suf_len != 0 {
                let n = chars(&s);
                suffix = csub(&s, n - suf_len, n).to_string();
                minus_suf = csub(&s, 0, n - suf_len).to_string();
                if !minus_suf.is_empty() && sp && self.specials.contains_key(&minus_suf) {
                    s = minus_suf;
                    suffixes.push(suffix);
                    break;
                }
            }
            let n = chars(&s);
            if pre_len != 0 && suf_len != 0 && pre_len + suf_len <= n {
                s = csub(&s, pre_len, n - suf_len).to_string();
                prefixes.push(std::mem::take(&mut prefix));
                suffixes.push(std::mem::take(&mut suffix));
            } else if pre_len != 0 {
                s = minus_pre;
                prefixes.push(prefix);
            } else if suf_len != 0 {
                s = minus_suf;
                suffixes.push(suffix);
            }
        }
        for p in &prefixes {
            out.push(Self::lex(p));
        }
        if !s.is_empty() {
            if sp && self.push_special(out, &s) {
            } else if self
                .url
                .as_ref()
                .is_some_and(|r| r.is_match(&s).unwrap_or(false))
            {
                out.push(Self::lex(&s));
            } else {
                let matches: Vec<(usize, usize)> = match &self.infix {
                    Some(r) => r
                        .find_iter(&s)
                        .flatten()
                        .map(|m| (m.start(), m.end()))
                        .collect(),
                    None => Vec::new(),
                };
                if matches.is_empty() {
                    out.push(Self::lex(&s));
                } else {
                    let mut start = 0usize; // byte offsets
                    let start_before = start;
                    for (a, b) in matches {
                        if a == start_before {
                            continue;
                        }
                        if a != start {
                            out.push(Self::lex(&s[start..a]));
                        }
                        if a != b {
                            out.push(Self::lex(&s[a..b]));
                        }
                        start = b;
                    }
                    if start < s.len() {
                        out.push(Self::lex(&s[start..]));
                    }
                }
            }
        }
        for x in suffixes.iter().rev() {
            out.push(Self::lex(x));
        }
    }

    /// `_tokenize_affixes(string, with_special_cases)`.
    fn tokenize_affixes(&self, string: &str, sp: bool) -> Vec<Token> {
        let mut out: Vec<Token> = Vec::new();
        let cs: Vec<(usize, char)> = string.char_indices().collect();
        if cs.is_empty() {
            return out;
        }
        let mut in_ws = py_isspace(cs[0].1);
        let mut start = 0usize; // char index
        let byte = |ci: usize| cs.get(ci).map_or(string.len(), |&(b, _)| b);
        let emit = |out: &mut Vec<Token>, a: usize, b: usize| {
            let span = &string[byte(a)..byte(b)];
            let first = out.len();
            if !(sp && self.push_special(out, span)) {
                self.tokenize_span(span, out, sp);
            }
            let _ = first;
        };
        for (i, &(_, uc)) in cs.iter().enumerate() {
            if py_isspace(uc) != in_ws {
                if start < i {
                    emit(&mut out, start, i);
                }
                if uc == ' ' {
                    if let Some(t) = out.last_mut() {
                        t.whitespace = true;
                    }
                    start = i + 1;
                } else {
                    start = i;
                }
                in_ws = !in_ws;
            }
        }
        if start < cs.len() {
            emit(&mut out, start, cs.len());
            let last = cs.last().unwrap().1;
            if let Some(t) = out.last_mut() {
                t.whitespace = last == ' ' && !in_ws;
            }
        }
        out
    }

    /// `_apply_special_cases`: PhraseMatcher over the special-case token
    /// sequences, longest (then leftmost) non-overlapping spans replaced by
    /// their exception tokens.
    fn apply_special_cases(&self, toks: Vec<Token>) -> Vec<Token> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for i in 0..toks.len() {
            if let Some(cands) = self.phrases.get(&toks[i].text) {
                for c in cands {
                    let e = i + c.len();
                    if e <= toks.len() && toks[i..e].iter().zip(c).all(|(t, o)| &t.text == o) {
                        spans.push((i, e));
                    }
                }
            }
        }
        if spans.is_empty() {
            return toks;
        }
        spans.sort();
        spans.dedup();
        // len_start_cmp: shorter first, ties by start descending; walk from the end.
        spans.sort_by(|a, b| {
            let (la, lb) = (a.1 - a.0, b.1 - b.0);
            la.cmp(&lb).then(b.0.cmp(&a.0))
        });
        let mut seen: HashSet<usize> = HashSet::new();
        let mut kept: Vec<(usize, usize)> = Vec::new();
        for &(s, e) in spans.iter().rev() {
            if !seen.contains(&s) && !seen.contains(&(e - 1)) {
                kept.push((s, e));
            }
            for k in s..e {
                seen.insert(k);
            }
        }
        kept.sort();
        let mut out = Vec::with_capacity(toks.len());
        let mut i = 0;
        let mut ki = 0;
        while i < toks.len() {
            if ki < kept.len() && kept[ki].0 == i {
                let (s, e) = kept[ki];
                ki += 1;
                let text: String = toks[s..e]
                    .iter()
                    .enumerate()
                    .map(|(k, t)| {
                        if k + 1 < e - s && t.whitespace {
                            format!("{} ", t.text)
                        } else {
                            t.text.clone()
                        }
                    })
                    .collect();
                match self.specials.get(&text) {
                    Some(seq) => {
                        let final_ws = toks[e - 1].whitespace;
                        for x in seq {
                            out.push(Token {
                                text: x.orth.clone(),
                                whitespace: false,
                                norm: x.norm.clone(),
                                idx: 0,
                            });
                        }
                        out.last_mut().unwrap().whitespace = final_ws;
                    }
                    None => out.extend_from_slice(&toks[s..e]),
                }
                i = e;
            } else {
                out.push(toks[i].clone());
                i += 1;
            }
        }
        out
    }

    /// `nlp.tokenizer(text)`.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let toks = self.tokenize_affixes(text, true);
        let mut toks = self.apply_special_cases(toks);
        let mut idx = 0;
        for t in toks.iter_mut() {
            t.idx = idx;
            idx += chars(&t.text) + t.whitespace as usize;
        }
        toks
    }
}
