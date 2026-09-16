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

//! Output punctuation normalisation.
//!
//! The shipped phrasebooks are typeset — curly quotes, no-break spaces before
//! French `?`, `…` — but the framework's *returned* text is not. Counted over
//! 37 816 reference translations from the live framework:
//!
//! ```text
//!   ’ U+2019   0 occurrences      “ U+201C   2 occurrences
//!   ‘ U+2018   0                  ” U+201D   3
//!   NBSP       0                  — em dash  6
//!   NNBSP      0                  » U+00BB   3
//!   … U+2026   0
//!   – en dash  0
//!   « U+00AB   0
//! ```
//!
//! The left column is normalised away without exception; the right column
//! survives. So this maps exactly the characters with zero survivors and
//! deliberately leaves the others alone — normalising `—` or `”` too would be
//! a guess that the data contradicts.
//!
//! The fullwidth block is handled the same way, and it is **not** uniform:
//! fullwidth CJK punctuation survives (`？` 40, `！` 5, `（）` 3, `，` 3, `；` 3,
//! `：` 2, `～` 2) while fullwidth solidus and hyphen-minus never do, despite
//! appearing 12 times each in the en→zh phrasebooks. So `／`/`－` are folded to
//! ASCII and the CJK marks are left alone — folding `？` to `?` would break
//! Chinese typography.
//!
//! `–`, `«`, `［`, `］`, `＝`, `＋` also never survive, but no covered
//! translation exercises them, so their replacement is unknown and they are
//! left untouched rather than guessed at. [`UNMAPPED_ABSENT`] records that.

/// Characters that never survive in live output but whose replacement is not
/// determined by any observed translation.
pub const UNMAPPED_ABSENT: &[char] = &[
    '\u{2013}', '\u{00AB}', '\u{FF3B}', '\u{FF3D}', '\u{FF1D}', '\u{FF0B}',
];

/// Applies the framework's output punctuation normalisation for `target`.
///
/// `target` is the target locale or language code. The only script-dependent
/// rule found so far is the ellipsis: across 75 542 live translations `…`
/// survives **only** in Arabic output (4 occurrences), and is folded to `...`
/// everywhere else. That exception was invisible in an earlier 37 816-line
/// sample that happened to contain no Arabic-target directions — a reminder
/// that "zero survivors" is only as strong as the corpus behind it.
pub fn normalize_punctuation_for(text: &str, target: &str) -> String {
    let lang = target.split(['-', '_']).next().unwrap_or(target);
    let keep_ellipsis = lang == "ar";
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars() {
        let mapped: &str = match ch {
            '\u{2019}' | '\u{2018}' => "'",
            '\u{00A0}' | '\u{202F}' => " ",
            '\u{2026}' if !keep_ellipsis => "...",
            '\u{FF0F}' => "/",
            '\u{FF0D}' => "-",
            _ => {
                // Runs of spaces collapse: no live output among 75 542 contains
                // a double space, though the phrasebooks do.
                if ch == ' ' {
                    if last_space {
                        continue;
                    }
                    last_space = true;
                } else {
                    last_space = false;
                }
                out.push(ch);
                continue;
            }
        };
        for m in mapped.chars() {
            if m == ' ' {
                if last_space {
                    continue;
                }
                last_space = true;
            } else {
                last_space = false;
            }
            out.push(m);
        }
    }
    out
}

/// [`normalize_punctuation_for`] with no language-specific rules applied.
pub fn normalize_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            // Curly single quotes become the ASCII apostrophe.
            '\u{2019}' | '\u{2018}' => out.push('\''),
            // No-break and narrow no-break spaces become an ordinary space.
            // French typography puts one before `?`/`!`; the API returns U+0020.
            '\u{00A0}' | '\u{202F}' => out.push(' '),
            // The ellipsis is spelled out.
            '\u{2026}' => out.push_str("..."),
            // Fullwidth solidus and hyphen-minus fold to ASCII. Fullwidth CJK
            // punctuation deliberately does not — see the module docs.
            '\u{FF0F}' => out.push('/'),
            '\u{FF0D}' => out.push('-'),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curly_single_quotes_become_ascii() {
        assert_eq!(
            normalize_punctuation("J\u{2019}adore l\u{2019}été"),
            "J'adore l'été"
        );
        assert_eq!(normalize_punctuation("\u{2018}"), "'");
    }

    #[test]
    fn no_break_spaces_become_ordinary_spaces() {
        assert_eq!(normalize_punctuation("hamburger\u{a0}?"), "hamburger ?");
        assert_eq!(normalize_punctuation("1\u{202f}million"), "1 million");
    }

    #[test]
    fn ellipsis_is_spelled_out() {
        assert_eq!(normalize_punctuation("Анти\u{2026}"), "Анти...");
        assert_eq!(normalize_punctuation("\u{2026}脊背的"), "...脊背的");
    }

    #[test]
    fn arabic_keeps_its_ellipsis() {
        // The one script-dependent rule measured: `…` survives only in Arabic.
        assert_eq!(
            normalize_punctuation_for("ذو \u{2026}", "ar_AE"),
            "ذو \u{2026}"
        );
        assert_eq!(
            normalize_punctuation_for("Анти\u{2026}", "ru_RU"),
            "Анти..."
        );
    }

    #[test]
    fn runs_of_spaces_collapse() {
        assert_eq!(normalize_punctuation_for("a  b", "en"), "a b");
        assert_eq!(normalize_punctuation_for("a\u{a0} b", "en"), "a b");
        assert_eq!(normalize_punctuation_for("a b", "en"), "a b");
    }

    #[test]
    fn fullwidth_solidus_and_hyphen_fold_to_ascii() {
        assert_eq!(normalize_punctuation("模拟\u{ff0d}数字的"), "模拟-数字的");
        assert_eq!(
            normalize_punctuation("在船上\u{ff0f}车上\u{ff0f}飞机上"),
            "在船上/车上/飞机上"
        );
    }

    #[test]
    fn fullwidth_cjk_punctuation_is_preserved() {
        // These survive in live output; folding them would break typography.
        for s in [
            "你好\u{ff1f}",
            "好\u{ff01}",
            "a\u{ff0c}b",
            "\u{ff08}x\u{ff09}",
            "\u{ff5e}",
        ] {
            assert_eq!(normalize_punctuation(s), s, "{s} must be preserved");
        }
    }

    #[test]
    fn characters_that_survive_are_left_alone() {
        // Double quotes, em dash and » all appear in live output, so mapping
        // them would introduce a difference rather than remove one.
        for s in ["\u{201c}quoted\u{201d}", "a \u{2014} b", "\u{bb}"] {
            assert_eq!(normalize_punctuation(s), s, "{s} must be preserved");
        }
    }

    #[test]
    fn absent_but_unmapped_characters_are_left_alone() {
        for ch in UNMAPPED_ABSENT {
            let s = ch.to_string();
            assert_eq!(normalize_punctuation(&s), s, "{ch:?} has no known mapping");
        }
    }

    #[test]
    fn ordinary_text_is_unchanged() {
        for s in ["hello world", "夏が好きです", "", "1 million"] {
            assert_eq!(normalize_punctuation(s), s);
        }
    }
}
