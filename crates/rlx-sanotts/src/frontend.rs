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

//! Text → Piper phoneme ids.
//!
//! sanoTTS inherits Piper's frontend wholesale: espeak-ng IPA (with stress,
//! punctuation preserved), NFD-decomposed to individual codepoints, then looked
//! up one codepoint at a time in the voice's `phoneme_id_map` and framed as
//! `^ _ (p _)* $`.
//!
//! Note the leading pad: sanoTTS emits `[BOS, PAD]` before the first phoneme,
//! which is one pad more than [`rlx_piper`](../../rlx-piper)'s framing. The
//! duration/acoustic models were trained on this exact framing, so it is not
//! interchangeable.

use anyhow::Result;
use unicode_normalization::UnicodeNormalization;

use crate::config::{BOS_ID, EOS_ID, PAD_ID, PhonemeTable};

/// Piper's punctuation set — the `phoneme_id_map` keys that are ASCII
/// punctuation rather than IPA symbols, and so the marks that survive
/// phonemization into the id stream.
///
/// Informational: espeak-ng's phonemizer mode preserves punctuation itself, so
/// nothing here configures it. It documents what [`phonemes_to_ids`] will map.
pub const PUNCTUATION_MARKS: &[char] = &['!', '\'', '(', ')', ',', '-', '.', ':', ';', '?', '"'];

/// Map an espeak phoneme string to Piper ids using sanoTTS's framing.
///
/// The string is NFD-decomposed first, so combining stress/length marks become
/// their own codepoints — which is how the table is keyed. Codepoints missing
/// from the table are dropped, matching Piper.
pub fn phonemes_to_ids(phonemes: &str, table: &PhonemeTable) -> Vec<i64> {
    let mut ids = vec![BOS_ID, PAD_ID];
    for ch in phonemes.nfd() {
        if let Some(&id) = table.id_map.get(&ch) {
            ids.push(id);
            ids.push(PAD_ID);
        }
    }
    ids.push(EOS_ID);
    ids
}

/// Number of framing ids in an otherwise-empty sequence (`^`, `_`, `$`).
pub const FRAMING_IDS: usize = 3;

/// Phonemize `text` with espeak-ng, then map to ids.
#[cfg(feature = "espeak")]
pub fn text_to_phoneme_ids(text: &str, table: &PhonemeTable) -> Result<Vec<i64>> {
    let clean = text.trim();
    if clean.is_empty() {
        anyhow::bail!("text is empty");
    }
    let ipa = phonemize(clean, &table.espeak_voice)?;
    let ids = phonemes_to_ids(&ipa, table);
    if ids.len() <= FRAMING_IDS {
        anyhow::bail!("phonemization produced no usable phonemes for: {text:?}");
    }
    Ok(ids)
}

#[cfg(not(feature = "espeak"))]
pub fn text_to_phoneme_ids(_text: &str, _table: &PhonemeTable) -> Result<Vec<i64>> {
    anyhow::bail!(
        "rlx-sanotts was built without the `espeak` feature; synthesize from \
         phoneme ids instead (see Synthesizer::synthesize_ids)"
    )
}

/// espeak-ng IPA for `text`, keeping Piper's punctuation and stress marks.
///
/// This is `espeak_ng`'s phonemizer-compatible mode: punctuation preserved,
/// clause separators flattened to spaces, stress marks retained — the same
/// three settings sanoTTS passes to the Python `phonemizer` package.
#[cfg(feature = "espeak")]
pub fn phonemize(text: &str, espeak_voice: &str) -> Result<String> {
    let out = with_engine(espeak_voice, |engine| {
        engine
            .text_to_phonemes_phonemizer(text)
            .map_err(|e| anyhow::anyhow!("espeak-ng phonemize ({espeak_voice}) failed: {e}"))
    })?;
    // phonemizer emits a trailing separator that Piper's own espeak bridge does
    // not; upstream rstrips before decomposing, so we do too.
    let out = out.trim_end().to_string();
    if out.is_empty() {
        anyhow::bail!("espeak-ng produced no phonemes for text: {text:?}");
    }
    Ok(out)
}

#[cfg(feature = "espeak")]
mod engine {
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    use anyhow::{Context, Result, anyhow};

    /// Overrides the bundled data directory (see [`super::set_espeak_data_path`]).
    static DATA_PATH: OnceLock<PathBuf> = OnceLock::new();
    /// Languages already unpacked into the working data directory.
    static INSTALLED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

    pub fn set_data_path(path: &Path) {
        let _ = DATA_PATH.set(path.to_path_buf());
    }

    /// espeak selects data by language code, so `en-us` needs the `en` pack.
    fn base_language(voice: &str) -> &str {
        voice.split(['-', '_']).next().unwrap_or(voice)
    }

    /// Data directory with `lang`'s dictionary installed. When the caller has
    /// supplied a system `espeak-ng-data`, it is used as-is.
    pub fn data_dir(lang: &str) -> Result<PathBuf> {
        if let Some(user) = DATA_PATH.get() {
            return Ok(user.clone());
        }
        let base = base_language(lang).to_ascii_lowercase();
        let dir = std::env::temp_dir().join("rlx-sanotts-espeak-ng-data");
        let mut installed = INSTALLED
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .map_err(|_| anyhow!("espeak data-install mutex poisoned"))?;
        if installed.contains(&base) {
            return Ok(dir);
        }
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create espeak data dir {}", dir.display()))?;
        install(&dir, &base)?;
        installed.insert(base);
        Ok(dir)
    }

    /// With every language bundled, espeak-ng exposes only the install-everything
    /// entry point — its per-language installer is gated on the individual
    /// `bundled-data-<lang>` features, so asking it for `id` would fail even
    /// though the dictionary is right there in the binary.
    #[cfg(feature = "espeak-all-languages")]
    fn install(dir: &Path, _base: &str) -> Result<()> {
        espeak_ng::install_bundled_data(dir)
            .map_err(|e| anyhow!("install bundled espeak-ng data into {}: {e}", dir.display()))
    }

    #[cfg(not(feature = "espeak-all-languages"))]
    fn install(dir: &Path, base: &str) -> Result<()> {
        espeak_ng::install_bundled_language(dir, base).map_err(|e| {
            anyhow!(
                "this build of rlx-sanotts has no bundled espeak-ng data for {base:?} ({e}); \
                 enable the `espeak-all-languages` feature, or point \
                 rlx_sanotts::frontend::set_espeak_data_path at a system espeak-ng-data"
            )
        })
    }
}

#[cfg(feature = "espeak")]
pub use engine::set_data_path as set_espeak_data_path;

/// Run `f` against an engine for `espeak_voice`, falling back to regional
/// variants of a bare language code.
///
/// Older voice packages (e.g. kristin) were trained when a bare `en` was a
/// selectable espeak voice; newer espeak-ng only exposes regional variants as
/// primary voices. That changes the accent slightly, never the phoneme table.
#[cfg(feature = "espeak")]
fn with_engine<T>(espeak_voice: &str, f: impl Fn(&espeak_ng::EspeakNg) -> Result<T>) -> Result<T> {
    let mut candidates = vec![espeak_voice.to_string()];
    if !espeak_voice.contains('-') {
        candidates.push(format!("{espeak_voice}-us"));
        candidates.push(format!("{espeak_voice}-gb"));
    }
    let dir = engine::data_dir(espeak_voice)?;
    let mut last = None;
    for candidate in &candidates {
        match espeak_ng::EspeakNg::with_data_dir(candidate, &dir) {
            Ok(engine) => match f(&engine) {
                Ok(v) => return Ok(v),
                Err(e) => last = Some(e),
            },
            Err(e) => last = Some(anyhow::anyhow!("{e}")),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("no espeak language")))
        .map_err(|e| anyhow::anyhow!("espeak-ng has no usable voice for {espeak_voice:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn table() -> PhonemeTable {
        let mut id_map = HashMap::new();
        id_map.insert('_', PAD_ID);
        id_map.insert('^', BOS_ID);
        id_map.insert('$', EOS_ID);
        id_map.insert('a', 10);
        id_map.insert('\u{0303}', 11); // combining tilde
        PhonemeTable {
            espeak_voice: "en-us".into(),
            id_map,
        }
    }

    #[test]
    fn framing_has_a_leading_pad() {
        assert_eq!(phonemes_to_ids("a", &table()), vec![1, 0, 10, 0, 2]);
    }

    #[test]
    fn unknown_codepoints_are_dropped() {
        assert_eq!(
            phonemes_to_ids("aZa", &table()),
            vec![1, 0, 10, 0, 10, 0, 2]
        );
    }

    #[test]
    fn nfd_splits_precomposed_marks() {
        // U+00E3 (ã) decomposes to 'a' + combining tilde, both in the table.
        assert_eq!(
            phonemes_to_ids("\u{00e3}", &table()),
            vec![1, 0, 10, 0, 11, 0, 2]
        );
    }
}
