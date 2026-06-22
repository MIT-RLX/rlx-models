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

//! Downloaded bench clips with paired reference transcripts (`just fetch-whisper-bench`).

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};

/// Default cache dir: `<repo>/.cache/whisper-bench`.
pub fn bench_cache_dir() -> PathBuf {
    std::env::var("RLX_WHISPER_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_bench_cache_dir())
}

fn default_bench_cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-bench")
}

/// OpenAI Whisper test clip (16 kHz mono). Override with `RLX_WHISPER_WAV`.
pub fn jfk_wav_path() -> PathBuf {
    std::env::var("RLX_WHISPER_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| bench_cache_dir().join("jfk_16k.wav"))
}

/// Greedy reference transcript for [`jfk_wav_path`]. Override with `RLX_WHISPER_REFERENCE`.
pub fn jfk_reference_path() -> PathBuf {
    std::env::var("RLX_WHISPER_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| bench_cache_dir().join("jfk.reference.txt"))
}

/// Embedded fallback when the reference file has not been fetched yet.
pub const JFK_REFERENCE: &str = " And so my fellow Americans ask not what your country can do for you ask what you can do for your country.";

/// Load the JFK reference string (file if present, else embedded constant).
pub fn load_jfk_reference() -> Result<String> {
    let path = jfk_reference_path();
    if path.is_file() {
        std::fs::read_to_string(&path)
            .with_context(|| format!("read reference {path:?}"))
            .map(|s| s.trim_end_matches(['\r', '\n']).to_string())
    } else {
        Ok(JFK_REFERENCE.to_string())
    }
}

/// Ensure wav + reference exist; useful from tests before running ASR.
pub fn ensure_jfk_fixture() -> Result<(PathBuf, String)> {
    let wav = jfk_wav_path();
    ensure!(
        wav.is_file(),
        "missing JFK wav at {wav:?}; run `just fetch-whisper-bench`"
    );
    let reference = load_jfk_reference()?;
    Ok((wav, reference))
}

/// Collapse whitespace for stable one-to-one comparison (Whisper may emit a leading space).
pub fn normalize_transcript(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Exact match after whitespace normalization.
pub fn transcripts_match(got: &str, want: &str) -> bool {
    normalize_transcript(got) == normalize_transcript(want)
}

/// Assert greedy decode matches the paired reference transcript.
pub fn assert_transcript_matches_reference(got: &str, want: &str) {
    assert!(
        transcripts_match(got, want),
        "transcript mismatch.\n  got:  {got:?}\n  want: {want:?}\n  \
         (normalized got:  {:?})\n  (normalized want: {:?})",
        normalize_transcript(got),
        normalize_transcript(want),
    );
}

/// Read manifest path if present.
pub fn manifest_path() -> PathBuf {
    bench_cache_dir().join("manifest.json")
}

/// Return `(wav, reference)` for clip id `jfk` only (extensible via manifest later).
pub fn clip_paths(id: &str) -> Result<(PathBuf, PathBuf)> {
    if id == "jfk" {
        return Ok((jfk_wav_path(), jfk_reference_path()));
    }
    bail!("unknown whisper bench clip {id:?}")
}

/// Load reference text for a bench clip.
pub fn load_reference_for(id: &str) -> Result<String> {
    match id {
        "jfk" => load_jfk_reference(),
        other => bail!("unknown whisper bench clip {other:?}"),
    }
}

/// Check fixture files on disk (for preflight / CLI).
pub fn fixture_status(dir: &Path) -> Result<()> {
    let wav = dir.join("jfk_16k.wav");
    let reference = dir.join("jfk.reference.txt");
    ensure!(wav.is_file(), "missing {wav:?}");
    ensure!(reference.is_file(), "missing {reference:?}");
    let _ = load_jfk_reference()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(
            normalize_transcript("  And  so my fellow Americans  "),
            "And so my fellow Americans"
        );
    }

    #[test]
    fn embedded_jfk_reference_matches_openai_whisper() {
        assert!(JFK_REFERENCE.starts_with(' '));
        assert!(transcripts_match(
            JFK_REFERENCE,
            "And so my fellow Americans ask not what your country can do for you ask what you can do for your country."
        ));
    }
}
