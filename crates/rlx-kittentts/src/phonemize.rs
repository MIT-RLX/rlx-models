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

//! Text → IPA via the pure-Rust [`espeak-ng`](https://crates.io/crates/espeak-ng) crate.
//!
//! Compiled only when the **`espeak`** Cargo feature is enabled. Without it, public
//! functions return an informative error so IPA-only builds stay dependency-free.

use std::path::{Path, PathBuf};

use anyhow::Result;
#[cfg(not(feature = "espeak"))]
use anyhow::anyhow;
use once_cell::sync::OnceCell;

/// Optional runtime path to espeak-ng data (see [`set_data_path`]).
static DATA_PATH: OnceCell<PathBuf> = OnceCell::new();

/// Default espeak voice / language tag for [`phonemize`].
/// Matches KittenTTS phonemizer (`EspeakBackend(language="en-us", …)`).
pub const DEFAULT_LANG: &str = "en-us";

/// Set the espeak-ng data directory before the first [`phonemize`] call.
///
/// With `bundled-data-en`, data is extracted automatically; call this only to
/// override (e.g. system `/usr/share/espeak-ng-data` or a full language pack).
pub fn set_data_path(path: &Path) {
    let _ = DATA_PATH.set(path.to_path_buf());
}

#[cfg(feature = "espeak")]
mod inner {
    use std::path::PathBuf;

    use anyhow::{Result, anyhow};
    use once_cell::sync::OnceCell;

    use super::{DATA_PATH, DEFAULT_LANG};

    static BUNDLED_DATA_DIR: OnceCell<PathBuf> = OnceCell::new();

    fn get_data_dir() -> Result<&'static PathBuf> {
        if let Some(user_dir) = DATA_PATH.get() {
            return Ok(BUNDLED_DATA_DIR.get_or_init(|| user_dir.clone()));
        }

        BUNDLED_DATA_DIR.get_or_try_init(|| {
            let cache_dir = std::env::temp_dir().join("rlx-kittentts-espeak-ng-data");
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| anyhow!("failed to create espeak-ng data dir: {e}"))?;
            // Bundled pack is tagged `en`; `en-us` / `en-gb` voices share that dict data.
            espeak_ng::install_bundled_language(&cache_dir, "en").map_err(|e| {
                anyhow!("failed to install bundled espeak-ng data for en: {e}")
            })?;
            Ok(cache_dir)
        })
    }

    fn create_engine(lang: &str) -> Result<espeak_ng::EspeakNg> {
        let data_dir = get_data_dir()?;
        espeak_ng::EspeakNg::with_data_dir(lang, data_dir)
            .map_err(|e| anyhow!("espeak-ng init for '{lang}' failed: {e}"))
    }

    pub(super) fn is_available() -> bool {
        create_engine(DEFAULT_LANG).is_ok()
    }

    pub(super) fn run_phonemize_lang(lang: &str, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }
        let engine = create_engine(lang)?;
        // KittenTTS uses phonemizer with preserve_punctuation + strip; match that
        // (not raw espeak-ng --ipa, which drops punctuation and emits clause newlines).
        let ipa = engine
            .text_to_phonemes(text)
            .map_err(|e| anyhow!("espeak-ng phonemize ({lang}) failed: {e}"))?;
        Ok(ipa.trim().to_owned())
    }
}

/// Returns `true` when espeak-ng initialises successfully (`espeak` feature on).
pub fn is_espeak_available() -> bool {
    #[cfg(feature = "espeak")]
    {
        inner::is_available()
    }
    #[cfg(not(feature = "espeak"))]
    {
        false
    }
}

/// Convert English text to IPA (`en` voice).
pub fn phonemize(text: &str) -> Result<String> {
    phonemize_lang(DEFAULT_LANG, text)
}

/// Convert `text` to IPA using an espeak language / voice tag (e.g. `en`, `en-us`).
pub fn phonemize_lang(lang: &str, text: &str) -> Result<String> {
    #[cfg(feature = "espeak")]
    {
        inner::run_phonemize_lang(lang, text)
    }
    #[cfg(not(feature = "espeak"))]
    {
        let _ = (lang, text);
        Err(anyhow!(
            "phonemize_lang() requires the `espeak` Cargo feature.\n\
             Rebuild with: cargo build -p rlx-kittentts --features espeak\n\
             Or pass IPA via --ipa / generate_from_ipa()."
        ))
    }
}

#[cfg(all(test, feature = "espeak"))]
mod tests {
    use super::*;

    #[test]
    fn espeak_available_with_bundled_en() {
        assert!(is_espeak_available());
    }

    #[test]
    fn phonemize_hello_world() {
        let ipa = phonemize("Hello world").expect("phonemize");
        assert!(!ipa.is_empty(), "expected IPA for Hello world");
        assert!(
            ipa.contains('h') || ipa.contains('l') || ipa.contains('w'),
            "unexpected IPA: {ipa}"
        );
    }

    #[test]
    fn phonemize_preserves_punctuation() {
        let ipa = phonemize("Hello world,").expect("phonemize");
        assert!(
            ipa.trim_end().ends_with(','),
            "expected trailing comma (phonemizer mode), got: {ipa:?}"
        );
        assert!(
            !ipa.contains('\n'),
            "expected flattened clauses, got: {ipa:?}"
        );
    }
}
