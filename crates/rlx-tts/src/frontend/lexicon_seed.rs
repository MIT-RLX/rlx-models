//! Lexicon seeding for the Hydra frontend.
//!
//! Load order (later wins):
//! 1. builtin digits / titles
//! 2. bundle `lexicon.txt`
//! 3. `nashville_isym_phones.json`
//! 4. neural-adapter Nashville hardcodes (`from`, `will`, …)
//! 5. round-trip overrides (OOV words TorchN G2P mis-pronounces)

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

pub(crate) fn seed_builtin_lexicon(lexicon: &mut HashMap<String, Vec<String>>) {
    // LHP orthography for digits + common spoken content words. Bundle
    // `lexicon.txt` overlays these when present.
    let entries: &[(&str, &[&str])] = &[
        ("zero", &["z", "i", "r", "o"]),
        ("one", &["w", "a", "n"]),
        ("two", &["t", "u"]),
        ("three", &["T", "r", "i"]),
        ("four", &["f", "o", "r"]),
        ("five", &["f", "a", "i", "v"]),
        ("six", &["s", "i", "k", "s"]),
        ("seven", &["s", "e", "v", "e", "n"]),
        ("eight", &["e", "i", "t"]),
        ("nine", &["n", "a", "i", "n"]),
        ("mister", &["m", "i", "s", "t", "e", "r"]),
        ("missus", &["m", "i", "s", "e", "s"]),
        ("miz", &["m", "i", "z"]),
        ("doctor", &["d", "a", "k", "t", "e", "r"]),
        ("saint", &["s", "e", "i", "n", "t"]),
        ("and", &["a", "n", "d"]),
        ("percent", &["p", "e", "r", "s", "e", "n", "t"]),
        ("at", &["a", "t"]),
        ("number", &["n", "a", "m", "b", "e", "r"]),
        ("hi", &["h", "a", "i"]),
        ("hello", &["h", "e", "l", "o"]),
        ("from", &["f", "r", "o", "m"]),
        ("call", &["k", "o", "l"]),
    ];
    for (word, phones) in entries {
        lexicon
            .entry((*word).to_string())
            .or_insert_with(|| phones.iter().map(|p| (*p).to_string()).collect());
    }
}

/// Whisper / demo round-trip overrides.
///
/// TorchN G2P returns garbage for these OOVs (e.g. `fox`→siz, `lazy`→speaker,
/// `rlx`→ve). Phones follow `nashville_isym` analogs (`box`, `jump`, `rise`, …).
pub(crate) fn seed_roundtrip_overrides(lexicon: &mut HashMap<String, Vec<String>>) {
    let entries: &[(&str, &[&str])] = &[
        ("fox", &["f", "a:", "k", "s"]),
        ("lazy", &["l", "J:", "z", "i"]),
        ("jumps", &["G", "^:", "m", "p", "s"]),
        ("speech", &["s", "p", "i:", "C"]),
        ("sounding", &["s", "@:", "n", "d", "I", "N"]),
        ("sunrise", &["s", "^:", "n", "r", "Y:", "z"]),
        ("riverbank", &["r", "I:", "v", "e", "b", "145:", "N", "k"]),
        ("artificial", &["a:", "r", "t", "$", "f", "I:", "S", "L"]),
        (
            "intelligence",
            &["I", "n", "t", "E:", "l", "$", "G", "$", "n", "s"],
        ),
        (
            "applications",
            &["145", "P", "l", "I", "K", "J:", "S", "$", "n", "z"],
        ),
        ("rust", &["r", "^:", "s", "t"]),
        // Brand acronym: letter names ar + ell + ex (ASR may still spell as one word).
        ("rlx", &["a:", "r", "E:", "l", "E:", "k", "s"]),
    ];
    for (word, phones) in entries {
        lexicon.insert(
            (*word).to_string(),
            phones.iter().map(|p| (*p).to_string()).collect(),
        );
    }
}

/// `examples/harvest_nashville_isyms` (`nashville_isym_phones.json`).
pub(crate) fn load_nashville_isym_phones(path: &Path, lexicon: &mut HashMap<String, Vec<String>>) {
    if !path.is_file() {
        return;
    }
    let Ok(raw) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return;
    };
    let Some(phones) = v.get("phones").and_then(|p| p.as_object()) else {
        return;
    };
    for (word, seq) in phones {
        let Some(arr) = seq.as_array() else {
            continue;
        };
        let toks: Vec<String> = arr
            .iter()
            .filter_map(|t| t.as_str().map(|s| s.to_string()))
            .filter(|s| {
                !s.is_empty()
                    && !matches!(
                        s.as_str(),
                        "." | "!" | "?" | "_" | "#" | "~" | "," | ";" | ":"
                    )
            })
            .collect();
        if !toks.is_empty() {
            lexicon.insert(word.to_ascii_lowercase(), toks);
        }
    }
}

pub(crate) fn load_lexicon(path: &Path, lexicon: &mut HashMap<String, Vec<String>>) -> Result<()> {
    for line in std::fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, rest) = if let Some((w, r)) = line.split_once('\t') {
            (w.to_string(), r.to_string())
        } else {
            let mut parts = line.split_whitespace();
            let Some(w) = parts.next() else {
                continue;
            };
            (w.to_string(), parts.collect::<Vec<_>>().join(" "))
        };
        let phones: Vec<String> = rest.split_whitespace().map(|s| s.to_string()).collect();
        if !phones.is_empty() {
            lexicon.insert(word.to_ascii_lowercase(), phones);
        }
    }
    Ok(())
}
