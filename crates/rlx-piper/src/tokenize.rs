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

//! Piper phoneme tokenizer: espeak-ng phonemes → ids via `phoneme_id_map`,
//! wrapped `^ … $` with a `_` pad after every phoneme (Piper's convention).

use anyhow::Result;

use crate::config::PiperConfig;

const BOS: &str = "^";
const EOS: &str = "$";
const PAD: &str = "_";

/// Map an espeak phoneme string to Piper input ids.
pub fn phonemes_to_ids(phonemes: &str, cfg: &PiperConfig) -> Vec<i64> {
    let pad = cfg.id_of(PAD).unwrap_or(0);
    let mut ids = Vec::new();
    if let Some(bos) = cfg.id_of(BOS) {
        ids.push(bos);
    }
    for c in phonemes.chars() {
        let key = c.to_string();
        if let Some(mapped) = cfg.phoneme_id_map.get(&key) {
            ids.extend_from_slice(mapped);
            ids.push(pad);
        }
        // phonemes absent from the map are dropped
    }
    if let Some(eos) = cfg.id_of(EOS) {
        ids.push(eos);
    }
    ids
}

/// Phonemize `text` with espeak, then tokenize (needs the `espeak` feature).
#[cfg(feature = "espeak")]
pub fn encode(text: &str, cfg: &PiperConfig) -> Result<Vec<i64>> {
    let ipa = phonemize_with_fallback(&cfg.espeak_voice, text)?;
    Ok(phonemes_to_ids(&ipa, cfg))
}

#[cfg(not(feature = "espeak"))]
pub fn encode(_text: &str, _cfg: &PiperConfig) -> Result<Vec<i64>> {
    anyhow::bail!("rlx-piper built without the `espeak` feature")
}

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
