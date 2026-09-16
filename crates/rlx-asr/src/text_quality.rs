//! Lightweight CTC transcript garbage heuristics (folded-path junk).

/// Detect n-gram repetition loops.
pub fn has_repetition_loop(text: &str, min_repeats: usize) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < min_repeats * 2 {
        return false;
    }
    for n in [2usize, 3, 4] {
        if words.len() < n * min_repeats {
            continue;
        }
        for i in 0..=words.len().saturating_sub(n * min_repeats) {
            let gram: Vec<&str> = words[i..i + n].to_vec();
            let mut rep = 1usize;
            let mut j = i + n;
            while j + n <= words.len() && words[j..j + n] == gram[..] {
                rep += 1;
                j += n;
            }
            if rep >= min_repeats {
                return true;
            }
        }
    }
    false
}

fn normalize_word(w: &str) -> String {
    w.trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
}

const JUNK_TOKENS: &[&str] = &[
    "large", "six", "run", "where", "mall", "zero", "list", "born", "road", "ct", "asc", "ike",
];

/// Heuristic for folded-CTC junk (fragment loops, ellipsis, glued tokens).
pub fn is_garbage(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.len() <= 2 {
        for w in &words {
            if JUNK_TOKENS.contains(&normalize_word(w).as_str()) {
                return true;
            }
        }
    }
    if has_repetition_loop(t, 3) {
        return true;
    }
    if words.len() >= 4 {
        let content = words
            .iter()
            .filter(|w| {
                let n = normalize_word(w);
                n.len() >= 3 && !JUNK_TOKENS.contains(&n.as_str())
            })
            .count();
        for n in [2usize, 3] {
            if words.len() < n * 2 {
                continue;
            }
            for i in 0..=words.len().saturating_sub(n * 2) {
                let gram: Vec<String> = words[i..i + n].iter().map(|w| normalize_word(w)).collect();
                if gram.iter().any(|g| g.is_empty()) {
                    continue;
                }
                let mut rep = 1usize;
                let mut j = i + n;
                while j + n <= words.len() {
                    let next: Vec<String> =
                        words[j..j + n].iter().map(|w| normalize_word(w)).collect();
                    if next != gram {
                        break;
                    }
                    rep += 1;
                    j += n;
                }
                // Need 3+ repeats when the phrase already has real lexical content
                // (CE-MLP often ends with a short "to do to do" stutter).
                let need = if content >= 3 { 3 } else { 2 };
                if rep >= need {
                    return true;
                }
            }
        }
    }
    let spaces = t.chars().filter(|c| c.is_whitespace()).count();
    if t.len() >= 80 && spaces * 100 / t.len().max(1) < 3 {
        return true;
    }
    let chars: Vec<char> = t.chars().collect();
    if !chars.is_empty() {
        let uniq: std::collections::HashSet<char> = chars.iter().copied().collect();
        if uniq.len() == 1 {
            return true;
        }
        if chars.len() >= 8 && uniq.len() <= 2 {
            return true;
        }
    }
    let ellipsis = t.chars().filter(|&c| c == '…').count();
    // Count bare '.' tokens / runs, ignoring a trailing "..." / sentence period.
    let core = t.trim_end_matches('.').trim_end();
    let dots = core.chars().filter(|&c| c == '.').count();
    // Ellipsis spam, or many bare dots mid-string (not a single sentence-final period).
    if ellipsis * 100 / t.len().max(1) > 5 {
        return true;
    }
    if dots >= 3 && dots * 100 / core.len().max(1) > 8 {
        return true;
    }
    if words.len() >= 3 {
        let mut uniq = std::collections::HashSet::new();
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut content = 0usize;
        for w in &words {
            let n = normalize_word(w);
            if n.len() >= 3 && !JUNK_TOKENS.contains(&n.as_str()) {
                content += 1;
            }
            uniq.insert(n.clone());
            *counts.entry(n).or_insert(0) += 1;
        }
        let ratio = uniq.len() as f32 / words.len() as f32;
        if ratio < 0.35 {
            return true;
        }
        let max_rep = counts.values().copied().max().unwrap_or(0);
        let max_rep_content = counts
            .iter()
            .filter(|(k, _)| {
                k.len() >= 3
                    && !matches!(
                        k.as_str(),
                        "what" | "this" | "that" | "with" | "from" | "have" | "been"
                    )
            })
            .map(|(_, &v)| v)
            .max()
            .unwrap_or(0);
        // Ignore short stutter tokens ("to to to") when scoring dominance.
        if content < 1 && max_rep >= 3 && max_rep * 100 / words.len() >= 30 {
            return true;
        }
        if max_rep_content >= 4 && max_rep_content * 100 / words.len() >= 35 {
            return true;
        }
    }
    false
}

/// Collapse immediate word repeats (`Ask Ask` → `Ask`) after CTC decode.
pub fn collapse_consecutive_words(text: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut prev_norm = String::new();
    for w in text.split_whitespace() {
        let n = normalize_word(w);
        if !n.is_empty() && n == prev_norm {
            continue;
        }
        out.push(w);
        prev_norm = n;
    }
    out.join(" ")
}

const TRAILING_STUTTER: &[&str] = &["to", "do", "a", "the", "and", "of", "is", "in", "or"];

/// Drop trailing function-word stutter (`… Ask not what to do to do to` → `… Ask not what`).
pub fn strip_trailing_stutter(text: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    while words.len() > 1 {
        let n = normalize_word(words[words.len() - 1]);
        if TRAILING_STUTTER.contains(&n.as_str()) {
            let content_left = words[..words.len() - 1].iter().any(|w| {
                let x = normalize_word(w);
                x.len() >= 4 && !JUNK_TOKENS.contains(&x.as_str())
            });
            if !content_left {
                break;
            }
            words.pop();
            continue;
        }
        break;
    }
    words.join(" ")
}

/// Drop leading filler / bare punctuation (`to notellow…` → `notellow…`).
pub fn strip_leading_stutter(text: &str) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    while words.len() > 1 {
        let n = normalize_word(words[0]);
        if n.is_empty()
            || TRAILING_STUTTER.contains(&n.as_str())
            || matches!(n.as_str(), "comp" | "we" | "my")
        {
            let content_left = words[1..].iter().any(|w| {
                let x = normalize_word(w);
                x.len() >= 4 && !JUNK_TOKENS.contains(&x.as_str())
            });
            if !content_left {
                break;
            }
            words.remove(0);
            continue;
        }
        break;
    }
    words.join(" ")
}

/// True when a chunk has almost no lexical content (safe to drop after chunk 0).
pub fn is_weak_chunk(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    let words: Vec<String> = t
        .split_whitespace()
        .map(normalize_word)
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return true;
    }
    let content = words
        .iter()
        .filter(|w| {
            w.len() >= 4
                && !JUNK_TOKENS.contains(&w.as_str())
                && !TRAILING_STUTTER.contains(&w.as_str())
                && !matches!(w.as_str(), "what" | "you" | "for" | "can" | "we" | "this")
        })
        .count();
    if content == 0 {
        return true;
    }
    let stop = words
        .iter()
        .filter(|w| {
            TRAILING_STUTTER.contains(&w.as_str())
                || matches!(w.as_str(), "what" | "you" | "for" | "can" | "we" | "this")
                || w.len() <= 2
        })
        .count();
    content <= 1 && stop * 100 / words.len() >= 50 && words.len() >= 3
}

/// Count strong lexical tokens (used to gate non-first chunks).
pub fn content_word_count(text: &str) -> usize {
    text.split_whitespace()
        .map(normalize_word)
        .filter(|w| {
            w.len() >= 4
                && !JUNK_TOKENS.contains(&w.as_str())
                && !TRAILING_STUTTER.contains(&w.as_str())
                && !matches!(
                    w.as_str(),
                    "what"
                        | "you"
                        | "for"
                        | "can"
                        | "we"
                        | "this"
                        | "that"
                        | "with"
                        | "from"
                        | "have"
                        | "been"
                        | "were"
                        | "their"
                        | "there"
                        | "about"
                        | "which"
                        | "other"
                        | "because"
                )
        })
        .count()
}

/// Cap how often a strong content word may appear (CE-MLP long-form spam).
pub fn cap_content_repeats(text: &str, max_n: usize) -> String {
    let max_n = max_n.max(1);
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out: Vec<&str> = Vec::new();
    for w in text.split_whitespace() {
        let n = normalize_word(w);
        let strong = n.len() >= 4
            && !JUNK_TOKENS.contains(&n.as_str())
            && !TRAILING_STUTTER.contains(&n.as_str())
            && !matches!(
                n.as_str(),
                "what"
                    | "this"
                    | "that"
                    | "with"
                    | "from"
                    | "have"
                    | "been"
                    | "were"
                    | "their"
                    | "there"
                    | "about"
                    | "which"
                    | "other"
                    | "because"
            );
        if strong {
            let c = counts.entry(n).or_insert(0);
            *c += 1;
            if *c > max_n {
                continue;
            }
        }
        out.push(w);
    }
    out.join(" ")
}

/// Normalize folded-CTC text: collapse repeats, strip leading/trailing stutter,
/// then cap content-word spam on long transcripts.
pub fn cleanup_transcript(text: &str) -> String {
    let t = strip_trailing_stutter(&strip_leading_stutter(&collapse_consecutive_words(text)));
    let words = t.split_whitespace().count();
    if words > 40 {
        cap_content_repeats(&t, 3)
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_large_single() {
        assert!(is_garbage("large"));
    }

    #[test]
    fn detects_six_run() {
        assert!(is_garbage("six run six run six run"));
    }

    #[test]
    fn detects_glued() {
        assert!(is_garbage(
            "largect Mall large Mall largect largeike largect largect Mall largect"
        ));
    }

    #[test]
    fn clean_passes() {
        assert!(!is_garbage("Well, I don't wish to see it anymore."));
    }

    #[test]
    fn partial_ce_mlp_phrase_passes() {
        assert!(!is_garbage(". eh ask not"));
        assert!(!is_garbage("to America eh ask not"));
        assert!(!is_garbage("not what you"));
        assert!(!is_garbage(". to language to to to"));
        assert!(!is_garbage(". Hello this a the and..."));
        assert!(!is_garbage(
            "ellow Americ frans Ask ask not what to do to do to"
        ));
        assert!(!is_garbage(
            "Ask Ask not not what your country you can do what Ask Ask what"
        ));
        assert!(!is_garbage(
            "Hello this K this is a testwherehes speech this synt this <segE> in this"
        ));
    }

    #[test]
    fn collapse_consecutive() {
        assert_eq!(
            collapse_consecutive_words("Ask Ask not not what your country"),
            "Ask not what your country"
        );
    }

    #[test]
    fn trailing_stutter_and_weak_chunk() {
        assert_eq!(
            strip_trailing_stutter("ellow Americ frans Ask not what to do to do to"),
            "ellow Americ frans Ask not what"
        );
        assert_eq!(
            strip_leading_stutter("to notellow Americans Ask not what"),
            "notellow Americans Ask not what"
        );
        assert_eq!(
            strip_leading_stutter("comp Ask not what your country"),
            "Ask not what your country"
        );
        assert!(is_weak_chunk("to do to do to"));
        assert!(is_weak_chunk("what to what"));
        assert!(!is_weak_chunk("Ask not what your country"));
        assert_eq!(
            cleanup_transcript("Ask Ask not not what your country you can do what to to"),
            "Ask not what your country you can do what"
        );
        assert!(content_word_count("Ask not what your country") >= 2);
    }
}
