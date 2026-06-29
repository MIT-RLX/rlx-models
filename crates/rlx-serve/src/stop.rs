// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Host-side multi-token stop-sequence matching.
//!
//! rlx runners stop on a single EOS *token id*; OpenAI `stop` strings may
//! span several tokens, so they must be matched on the **decoded text**.

/// Earliest byte index at which any non-empty stop string begins in `text`,
/// or `None` if none occur. The returned index is the cut point: text up to
/// (but not including) it is the visible output.
pub fn first_stop(text: &str, stops: &[String]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s.as_str()) {
            best = Some(best.map_or(pos, |b| b.min(pos)));
        }
    }
    best
}

/// Longest suffix of `text` that is a strict prefix of some stop string.
/// Used in streaming to hold back bytes that *might* complete into a stop
/// sequence on the next token, so we never emit a partial stop string.
pub fn pending_holdback(text: &str, stops: &[String]) -> usize {
    let mut hold = 0usize;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        // Try the longest suffix of `text` that is a proper prefix of `s`.
        let max = text.len().min(s.len().saturating_sub(1));
        let mut k = max;
        while k > 0 {
            // Respect char boundaries.
            if text.is_char_boundary(text.len() - k) && s.is_char_boundary(k) {
                if text[text.len() - k..] == s[..k] {
                    hold = hold.max(k);
                    break;
                }
            }
            k -= 1;
        }
    }
    hold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_earliest_stop() {
        let stops = vec!["END".to_string(), "stop".to_string()];
        assert_eq!(first_stop("abc stop def END", &stops), Some(4));
        assert_eq!(first_stop("nothing here", &stops), None);
    }

    #[test]
    fn empty_stops_ignored() {
        assert_eq!(first_stop("hello", &["".to_string()]), None);
    }

    #[test]
    fn holdback_detects_partial_suffix() {
        // "ST" is a prefix of "STOP" → hold back 2 bytes.
        assert_eq!(pending_holdback("blah ST", &["STOP".to_string()]), 2);
        // No partial overlap → hold back nothing.
        assert_eq!(pending_holdback("blah xy", &["STOP".to_string()]), 0);
        // Full match isn't a *proper* prefix; holdback caps at len-1.
        assert_eq!(pending_holdback("STOP", &["STOP".to_string()]), 0);
    }
}
