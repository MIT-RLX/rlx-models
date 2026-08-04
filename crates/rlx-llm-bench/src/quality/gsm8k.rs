// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! GSM8K answer extraction + numeric matching.
//!
//! Gold answers use the canonical `#### <number>` suffix; model outputs are
//! free text, so we take the **last** number they emit (the convention every
//! GSM8K harness follows). Numbers are normalized — commas and a leading `$`
//! stripped — then compared as floats so `1,000`, `1000`, and `1000.0` all match.

/// Pull the gold numeric answer out of a GSM8K `answer` string (`"…#### 42"`).
/// Falls back to the last number in the string if no `####` marker is present.
pub fn extract_gold(answer: &str) -> Option<String> {
    if let Some(idx) = answer.rfind("####") {
        let tail = &answer[idx + 4..];
        if let Some(n) = first_number(tail) {
            return Some(n);
        }
    }
    last_number(answer)
}

/// Pull the predicted numeric answer out of free-form model output: the last
/// number it wrote.
pub fn extract_pred(output: &str) -> Option<String> {
    last_number(output)
}

/// Do two extracted answer strings denote the same number? Compares as floats
/// when both parse, else falls back to normalized string equality.
pub fn answers_match(a: &str, b: &str) -> bool {
    let na = normalize(a);
    let nb = normalize(b);
    match (na.parse::<f64>(), nb.parse::<f64>()) {
        (Ok(x), Ok(y)) => (x - y).abs() < 1e-6,
        _ => na == nb,
    }
}

/// Strip grouping commas and a leading currency `$`/whitespace so `"$1,024 "`
/// becomes `"1024"`.
fn normalize(s: &str) -> String {
    s.trim()
        .trim_start_matches('$')
        .chars()
        .filter(|c| *c != ',')
        .collect::<String>()
        .trim()
        .to_string()
}

/// First number token scanning left→right (used just after a `####` marker).
fn first_number(s: &str) -> Option<String> {
    numbers(s).into_iter().next()
}

/// Last number token in the string (the model's final answer).
fn last_number(s: &str) -> Option<String> {
    numbers(s).into_iter().next_back()
}

/// Extract number-like tokens (optional sign, digits with grouping commas, an
/// optional decimal part), each normalized. Percent signs and units are left
/// behind. Kept dependency-free — a hand-rolled scan rather than a regex crate.
fn numbers(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        let start_sign = (c == '-' || c == '+')
            && i + 1 < bytes.len()
            && (bytes[i + 1] as char).is_ascii_digit();
        if c.is_ascii_digit() || start_sign {
            let start = i;
            if start_sign {
                i += 1;
            }
            let mut seen_dot = false;
            while i < bytes.len() {
                let d = bytes[i] as char;
                if d.is_ascii_digit() || d == ',' {
                    i += 1;
                } else if d == '.'
                    && !seen_dot
                    && i + 1 < bytes.len()
                    && (bytes[i + 1] as char).is_ascii_digit()
                {
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(normalize(&s[start..i]));
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gold_from_hash_marker() {
        assert_eq!(extract_gold("blah\n#### 42").as_deref(), Some("42"));
        assert_eq!(
            extract_gold("steps... #### 1,024 units").as_deref(),
            Some("1024")
        );
    }

    #[test]
    fn pred_takes_last_number() {
        assert_eq!(
            extract_pred("First 3, then 7, the answer is 18.").as_deref(),
            Some("18")
        );
        assert_eq!(extract_pred("no numbers here"), None);
        assert_eq!(extract_pred("total: $2,500.50").as_deref(), Some("2500.50"));
    }

    #[test]
    fn matching_is_numeric() {
        assert!(answers_match("1,000", "1000"));
        assert!(answers_match("42", "42.0"));
        assert!(!answers_match("42", "43"));
        assert!(answers_match("-5", "-5"));
    }
}
