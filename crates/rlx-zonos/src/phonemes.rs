// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Zonos espeak phoneme vocabulary (from Zyphra/Zonos `conditioning.py`).

pub const PAD_ID: i64 = 0;
pub const UNK_ID: i64 = 1;
pub const BOS_ID: i64 = 2;
pub const EOS_ID: i64 = 3;

const PUNCTUATION: &str = ";:,.!?¡¿—…\"«»“”() *~-/\\&";
const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const LETTERS_IPA: &str = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

fn symbols() -> Vec<char> {
    let mut v = Vec::new();
    v.extend(PUNCTUATION.chars());
    v.extend(LETTERS.chars());
    v.extend(LETTERS_IPA.chars());
    v
}

fn symbol_to_id(c: char) -> i64 {
    // Rebuild map once — vocab is tiny.
    static MAP: std::sync::OnceLock<std::collections::HashMap<char, i64>> =
        std::sync::OnceLock::new();
    let m = MAP.get_or_init(|| {
        symbols()
            .into_iter()
            .enumerate()
            .map(|(i, ch)| (ch, (i as i64) + 4))
            .collect()
    });
    m.get(&c).copied().unwrap_or(UNK_ID)
}

/// Map an espeak IPA / symbol string to Zonos phoneme ids: `[BOS, …, EOS]`.
pub fn tokenize_phoneme_string(phonemes: &str) -> Vec<i64> {
    let mut ids = Vec::with_capacity(phonemes.chars().count() + 2);
    ids.push(BOS_ID);
    for c in phonemes.chars() {
        ids.push(symbol_to_id(c));
    }
    ids.push(EOS_ID);
    ids
}

/// Phonemize English text with espeak-ng (via `rlx-kittentts`), then tokenize.
///
/// List commas are flattened to spaces before G2P: keeping `,` as a phoneme
/// token (it is in the Zonos vocab) tends to collapse “courage, kindness,” into
/// mush under default rate-15 sampling, while “courage and kindness” is clear.
#[cfg(feature = "espeak")]
pub fn phonemize_en(text: &str) -> anyhow::Result<Vec<i64>> {
    let cleaned: String = text
        .chars()
        .map(|c| if c == ',' { ' ' } else { c })
        .collect();
    let ipa = rlx_kittentts::phonemize::phonemize_lang("en-us", &cleaned)
        .or_else(|_| rlx_kittentts::phonemize::phonemize_lang("en", &cleaned))
        .map_err(|e| anyhow::anyhow!("espeak phonemize: {e}"))?;
    Ok(tokenize_phoneme_string(&ipa))
}

#[cfg(not(feature = "espeak"))]
pub fn phonemize_en(_text: &str) -> anyhow::Result<Vec<i64>> {
    anyhow::bail!("rlx-zonos built without `espeak` feature (needed for G2P)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bos_eos_wrap() {
        let ids = tokenize_phoneme_string("həˈloʊ");
        assert_eq!(ids[0], BOS_ID);
        assert_eq!(*ids.last().unwrap(), EOS_ID);
        assert!(ids.len() > 2);
    }
}
