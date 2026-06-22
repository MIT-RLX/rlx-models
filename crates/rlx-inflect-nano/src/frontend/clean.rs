//! `clean_tinytts_text` — normalize text into punctuation TinyTTS has symbols for.
//! Ported from `tinytts_text_cleaning.py`.

use once_cell::sync::Lazy;
use regex::Regex;

static RE_WS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static RE_WS_BEFORE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+([,.!?…])").unwrap());
static RE_REPEAT_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"([,.!?…])([,.!?…])+").unwrap());

fn translate(c: char) -> Option<char> {
    // None => drop the char; Some(x) => replace.
    match c {
        '\u{2018}' | '\u{2019}' => Some('\''),
        '\u{201c}' | '\u{201d}' => None, // removed
        '\u{2014}' | '\u{2013}' => Some(','),
        ';' | ':' => Some(','),
        '\n' => Some('.'),
        other => Some(other),
    }
}

pub fn clean_tinytts_text(text: &str) -> String {
    // str.translate
    let mut s: String = text.chars().filter_map(translate).collect();
    // replace "..." with "…"
    s = s.replace("...", "…");
    // collapse whitespace + trim
    s = RE_WS.replace_all(&s, " ").trim().to_string();
    // drop whitespace before punctuation
    s = RE_WS_BEFORE_PUNCT.replace_all(&s, "$1").to_string();
    // collapse runs of punctuation to the last one (re: `([,.!?…]){2,}` -> `\1`)
    s = RE_REPEAT_PUNCT
        .replace_all(&s, |caps: &regex::Captures| {
            caps[0].chars().last().unwrap().to_string()
        })
        .to_string();
    s
}
