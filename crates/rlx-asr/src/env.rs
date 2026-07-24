// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Env knobs and weight layout for `rlx-asr`.
//!
//! ```text
//! weights/asr/ (`just fetch-rlx-asr` → [eugenehp/rlx-asr](https://huggingface.co/eugenehp/rlx-asr))
//!   model.rlxp       # sole runtime asset (`just asr-pack-rlxp`)
//!   model.gguf       # legacy GGUF pack
//!   manifest.json    # optional listing
//! ```
//!
//! Units, silence fbank, etiquette, TP FSTs, encoder/decoder/codebook/LS are
//! all inside the GGUF. Pack sources: `.cache/asr` / `RLX_ASR_PACK_SRC`.

use std::path::{Path, PathBuf};

pub fn timing() -> bool {
    matches!(
        std::env::var("RLX_ASR_TIMING").as_deref(),
        Ok("1") | Ok("true")
    )
}

pub fn asr_dir_env() -> Option<PathBuf> {
    std::env::var_os("RLX_ASR_DIR").map(Into::into)
}

fn looks_like_asr_root(p: &Path) -> bool {
    p.join("model.rlxp").is_file()
        || p.join("model.gguf").is_file()
        || p.join("manifest.json").is_file()
}

pub fn default_asr_roots() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let crate_dir = PathBuf::from(manifest);
        if let Some(repo) = crate_dir.parent().and_then(|p| p.parent()) {
            out.push(repo.join("weights/asr"));
        }
    }
    out.push(PathBuf::from("weights/asr"));
    out
}

/// Resolve the ASR root: `RLX_ASR_DIR` (if set), else `weights/asr`.
pub fn asr_dir() -> PathBuf {
    if let Some(p) = asr_dir_env() {
        if looks_like_asr_root(&p) || p.is_dir() {
            return p;
        }
    }
    for c in default_asr_roots() {
        if looks_like_asr_root(&c) || c.is_dir() {
            return c;
        }
    }
    PathBuf::from("weights/asr")
}

#[derive(Debug, Clone)]
pub struct AsrPaths {
    pub root: PathBuf,
}

impl AsrPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
        }
    }

    pub fn resolve() -> Self {
        Self::new(asr_dir())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Preferred single-file weight pack (`.rlxp` or legacy GGUF).
    pub fn pack(&self) -> Option<PathBuf> {
        crate::gguf_io::resolve_pack_path(&self.root)
    }

    /// Legacy GGUF-only resolver.
    pub fn gguf(&self) -> Option<PathBuf> {
        crate::gguf_io::resolve_gguf_path(&self.root)
    }

    // --- legacy pack-source paths (usually absent in a GGUF-only tree) ---

    pub fn units_txt(&self) -> PathBuf {
        first_file(&[self.root.join("units.txt"), self.root.join("misc/units.txt")])
            .unwrap_or_else(|| self.root.join("units.txt"))
    }

    pub fn silence_fbank_txt(&self) -> PathBuf {
        first_file(&[
            self.root.join("silence-fbank.txt"),
            self.root.join("misc/silence-fbank.txt"),
        ])
        .unwrap_or_else(|| self.root.join("silence-fbank.txt"))
    }

    pub fn tp_dir(&self) -> Option<PathBuf> {
        first_dir(&[self.root.join("TP")])
    }

    pub fn etiquette_json(&self) -> Option<PathBuf> {
        first_file(&[self.root.join("etiquette.json")])
    }

    pub fn encoder_dir(&self) -> PathBuf {
        self.root.join("encoder")
    }

    pub fn codebook_dir(&self) -> PathBuf {
        self.root.join("codebook")
    }

    pub fn ls_dir(&self) -> PathBuf {
        self.root.join("ls")
    }

    pub fn decoder_dir(&self) -> PathBuf {
        self.root.join("decoder")
    }
}

fn first_file(cands: &[PathBuf]) -> Option<PathBuf> {
    cands.iter().find(|p| p.is_file()).cloned()
}

fn first_dir(cands: &[PathBuf]) -> Option<PathBuf> {
    cands.iter().find(|p| p.is_dir()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_resolve_without_panic() {
        let p = AsrPaths::resolve();
        assert!(!p.root.as_os_str().is_empty());
        let _ = p.gguf();
    }

    #[test]
    fn gguf_only_tree() {
        let p = AsrPaths::resolve();
        if !p.root.join("model.gguf").is_file() {
            return;
        }
        assert!(p.gguf().is_some());
    }
}
