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

//! LuxTTS text tokenizer. LuxTTS (ZipVoice `EspeakTokenizer`) phonemizes with
//! espeak-ng and — because English phoneme symbols are single characters in
//! `tokens.txt` — maps each phoneme *character* to its id (OOV dropped). We
//! reuse the bundled espeak-ng phonemizer from `rlx-kittentts`.

use anyhow::Result;

use crate::config::Tokens;

/// espeak language for the phonemizer (LuxTTS uses `en-us`).
pub const DEFAULT_LANG: &str = "en-us";

/// Phonemize `text` with espeak and map each phoneme char to its token id.
#[cfg(feature = "espeak")]
pub fn encode(text: &str, tokens: &Tokens, lang: &str) -> Result<Vec<i64>> {
    let ipa = phonemize_with_fallback(lang, text)?;
    Ok(ipa.chars().filter_map(|c| tokens.id_of(c)).collect())
}

#[cfg(not(feature = "espeak"))]
pub fn encode(_text: &str, _tokens: &Tokens, _lang: &str) -> Result<Vec<i64>> {
    anyhow::bail!("rlx-luxtts built without the `espeak` feature")
}

/// Phonemize, degrading `en-us`→`en` if the requested table is unavailable
/// (the bundled espeak data ships English only).
#[cfg(feature = "espeak")]
fn phonemize_with_fallback(primary: &str, text: &str) -> Result<String> {
    use anyhow::Context;
    use rlx_kittentts::phonemize::phonemize_lang;

    let mut langs = vec![primary];
    for f in ["en-us", "en"] {
        if !langs.contains(&f) {
            langs.push(f);
        }
    }
    let mut last = None;
    for lang in &langs {
        match phonemize_lang(lang, text) {
            Ok(ipa) => return Ok(ipa),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("no espeak language")))
        .with_context(|| format!("espeak phonemize failed for {text:?}"))
}
