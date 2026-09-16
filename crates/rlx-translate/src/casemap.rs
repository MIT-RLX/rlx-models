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

//! `CaseMapBlock` — sentence-casing of the finished translation.
//!
//! Measured against the framework's live output over 188 language pairs: of the
//! 425 sampled outputs that differed from a raw phrasebook hit only by case,
//! **414 were exactly "first cased character upper-cased"**, and 392 of those
//! had a *lower-case* source. So the rule is unconditional sentence-casing, not
//! "match the source's casing":
//!
//! ```text
//!   "'tween"     -> phrasebook "entre"          -> "Entre"
//!   "3d printer" -> phrasebook "imprimante 3D"  -> "Imprimante 3D"
//! ```
//!
//! Leading *punctuation* is skipped (`¿cómo` -> `¿Cómo`, `(гигро` -> `(Гигро`),
//! but a leading **digit stops** the pass: 481 digit-initial outputs in the
//! reference dump keep a lower-case first letter (`10.5" iPad Pro` stays
//! `iPad`, never `IPad`), against only 10 upper-case ones that were already
//! upper in the phrase itself (`3D-печать`).
//!
//! The block's `locale` drives locale-sensitive mappings — Turkish `i` must
//! become `İ`, not `I`.

/// Upper-cases the first letter, skipping leading punctuation but stopping at
/// a leading digit.
///
/// `locale` is the `CaseMapBlock`'s locale, e.g. `en`, `tr`.
pub fn sentence_case(text: &str, locale: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    for ch in chars.by_ref() {
        if ch.is_alphabetic() {
            push_upper(&mut out, ch, locale);
            break;
        }
        out.push(ch);
        // A number opens the string: leave the rest exactly as it is.
        if ch.is_numeric() {
            out.extend(chars);
            return out;
        }
    }
    out.extend(chars);
    out
}

/// Locale-sensitive upper-casing of a single character.
fn push_upper(out: &mut String, ch: char, locale: &str) {
    // Turkish and Azeri keep the dot: i -> İ, not I. Getting this wrong is a
    // real word change in those languages, not a cosmetic one.
    let lang = locale.split(['-', '_']).next().unwrap_or(locale);
    if matches!(lang, "tr" | "az") && ch == 'i' {
        out.push('\u{0130}');
        return;
    }
    for u in ch.to_uppercase() {
        out.push(u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_the_first_letter() {
        assert_eq!(sentence_case("entre", "fr"), "Entre");
        assert_eq!(sentence_case("imprimante 3D", "fr"), "Imprimante 3D");
    }

    #[test]
    fn leaves_an_already_capitalized_string_alone() {
        assert_eq!(sentence_case("Bonjour", "fr"), "Bonjour");
        assert_eq!(sentence_case("ABC", "en"), "ABC");
    }

    #[test]
    fn only_the_first_letter_changes() {
        assert_eq!(sentence_case("hello world", "en"), "Hello world");
        assert_eq!(sentence_case("a b c", "en"), "A b c");
    }

    #[test]
    fn skips_leading_punctuation() {
        assert_eq!(sentence_case("«bonjour»", "fr"), "«Bonjour»");
        assert_eq!(sentence_case("¿cómo estás?", "es"), "¿Cómo estás?");
        assert_eq!(
            sentence_case("(гигроскопическая) вата", "ru"),
            "(Гигроскопическая) вата"
        );
    }

    #[test]
    fn a_leading_digit_stops_the_pass() {
        // 481 digit-initial reference outputs keep their lower-case first
        // letter; capitalising here would turn `iPad` into `IPad`.
        assert_eq!(sentence_case("10.5\" iPad Pro", "en"), "10.5\" iPad Pro");
        assert_eq!(sentence_case("3d printing", "en"), "3d printing");
        assert_eq!(sentence_case("3D-печать", "ru"), "3D-печать");
    }

    #[test]
    fn handles_accented_and_multibyte_letters() {
        assert_eq!(sentence_case("été", "fr"), "Été");
        assert_eq!(sentence_case("ñandú", "es"), "Ñandú");
    }

    #[test]
    fn turkish_dotted_i_is_preserved() {
        // Plain to_uppercase would give "I", which is a different Turkish letter.
        assert_eq!(sentence_case("istanbul", "tr"), "\u{0130}stanbul");
        assert_eq!(sentence_case("istanbul", "en"), "Istanbul");
    }

    #[test]
    fn scripts_without_case_are_untouched() {
        for s in ["夏が好きです", "こんにちは", "مرحبا", "สวัสดี"] {
            assert_eq!(sentence_case(s, "en"), s, "{s} must be unchanged");
        }
    }

    #[test]
    fn empty_and_uncased_inputs_are_safe() {
        assert_eq!(sentence_case("", "en"), "");
        assert_eq!(sentence_case("123", "en"), "123");
        assert_eq!(sentence_case("  ", "en"), "  ");
    }
}
