//! `normalize_text` = lower → expand_time → normalize_numbers → expand_abbreviations.
//! Ports `tiny_tts/text/english.py::normalize_text` + the english_utils helpers.

use once_cell::sync::Lazy;
use regex::{Captures, Regex};

use super::numbers;

// number_norm.py regexes
static COMMA_NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"([0-9][0-9,]+[0-9])").unwrap());
static CURRENCY: Lazy<Regex> = Lazy::new(|| Regex::new(r"(£|\$|¥)([0-9,\.]*[0-9]+)").unwrap());
static DECIMAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"([0-9]+\.[0-9]+)").unwrap());
static ORDINAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"[0-9]+(st|nd|rd|th)").unwrap());
static NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"-?[0-9]+").unwrap());

// time_norm.py regex (verbatim, incl. the double-escaped a.m./p.m. branches that
// only effectively match bare "am"/"pm" — preserved for parity).
static TIME: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)\b(?P<hour>(0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3])):(?P<min>[0-5][0-9])\s*(?P<ap>a\\.m\\.|am|pm|p\\.m\\.|a\\.m|p\\.m)?\b",
    )
    .unwrap()
});

const ABBREVIATIONS: &[(&str, &str)] = &[
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
];

static ABBREV_RES: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    ABBREVIATIONS
        .iter()
        .map(|(a, r)| (Regex::new(&format!(r"(?i)\b{a}\.")).unwrap(), *r))
        .collect()
});

fn expand_number_word(num_str: &str) -> String {
    match num_str.parse::<i64>() {
        Ok(n) => numbers::expand_number(n),
        Err(_) => num_str.to_string(),
    }
}

fn normalize_numbers(text: &str) -> String {
    let s = COMMA_NUMBER.replace_all(text, |c: &Captures| c[1].replace(',', ""));
    let s = CURRENCY.replace_all(&s, |c: &Captures| {
        let sym = c[1].chars().next().unwrap();
        numbers::expand_currency(sym, &c[2])
    });
    let s = DECIMAL.replace_all(&s, |c: &Captures| c[1].replace('.', " point "));
    let s = ORDINAL.replace_all(&s, |c: &Captures| {
        let m = &c[0];
        let n: u64 = m[..m.len() - 2].parse().unwrap_or(0);
        numbers::ordinal(n)
    });
    NUMBER
        .replace_all(&s, |c: &Captures| expand_number_word(&c[0]))
        .into_owned()
}

fn expand_time(text: &str) -> String {
    TIME.replace_all(text, |c: &Captures| {
        let mut hour: i64 = c["hour"].parse().unwrap_or(0);
        let mut past_noon = hour >= 12;
        if hour > 12 {
            hour -= 12;
        } else if hour == 0 {
            hour = 12;
            past_noon = true;
        }
        let mut out = vec![numbers::cardinal(hour)];
        let minute: i64 = c["min"].parse().unwrap_or(0);
        if minute > 0 {
            if minute < 10 {
                out.push("oh".to_string());
            }
            out.push(numbers::cardinal(minute));
        }
        match c.name("ap") {
            None => out.push(if past_noon {
                "p m".to_string()
            } else {
                "a m".to_string()
            }),
            Some(ap) => {
                for ch in ap.as_str().replace('.', "").chars() {
                    out.push(ch.to_string());
                }
            }
        }
        out.join(" ")
    })
    .into_owned()
}

fn expand_abbreviations(text: &str) -> String {
    let mut s = text.to_string();
    for (re, rep) in ABBREV_RES.iter() {
        s = re.replace_all(&s, *rep).into_owned();
    }
    s
}

pub fn normalize_text(text: &str) -> String {
    let s = text.to_lowercase();
    let s = expand_time(&s);
    let s = normalize_numbers(&s);
    expand_abbreviations(&s)
}
