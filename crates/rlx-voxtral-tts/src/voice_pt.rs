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

//! Convert HuggingFace `voice_embedding/*.pt` (bf16 zip) to native `.f32`.

use crate::voice::VoiceEmbedding;
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use std::io::Read;
use std::path::{Path, PathBuf};
use zip::ZipArchive;

const PT_DATA_ENTRY: &str = "voice_embed/data/0";

pub fn convert_preset_voices(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let voice_dir = model_dir.join("voice_embedding");
    ensure!(
        voice_dir.is_dir(),
        "missing voice_embedding/ under {}",
        model_dir.display()
    );
    let mut written = Vec::new();
    for entry in std::fs::read_dir(&voice_dir).with_context(|| voice_dir.display().to_string())? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("pt") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("voice pt filename")?;
        let out = path.with_extension("f32");
        let emb = load_voice_pt(&path)?;
        ensure!(
            emb.n_tokens > 0,
            "{} produced empty embedding",
            path.display()
        );
        emb.save_f32(&out)?;
        eprintln!(
            "[rlx-voxtral-tts] {} -> {} ({} x hidden={})",
            path.file_name().unwrap().to_string_lossy(),
            out.file_name().unwrap().to_string_lossy(),
            emb.n_tokens,
            emb.hidden
        );
        written.push(out);
        let _ = name;
    }
    if written.is_empty() {
        bail!(
            "no voice_embedding/*.pt found under {}",
            model_dir.display()
        );
    }
    Ok(written)
}

pub fn load_voice_pt(path: &Path) -> Result<VoiceEmbedding> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("voice")
        .to_string();
    let data = read_pt_bf16_payload(path)?;
    let floats: Vec<f32> = data.iter().map(|v| f32::from(*v)).collect();
    const HIDDEN: usize = 3072;
    ensure!(
        !floats.is_empty() && floats.len().is_multiple_of(HIDDEN),
        "voice pt {}: invalid flat length {} (expected multiple of {HIDDEN})",
        path.display(),
        floats.len()
    );
    Ok(VoiceEmbedding {
        name,
        n_tokens: floats.len() / HIDDEN,
        hidden: HIDDEN,
        data: floats,
    })
}

fn read_pt_bf16_payload(path: &Path) -> Result<Vec<bf16>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut archive = ZipArchive::new(file).with_context(|| format!("zip {}", path.display()))?;
    let mut entry = archive
        .by_name(PT_DATA_ENTRY)
        .with_context(|| format!("missing {PT_DATA_ENTRY} in {}", path.display()))?;
    let mut raw = Vec::new();
    entry
        .read_to_end(&mut raw)
        .with_context(|| format!("read {PT_DATA_ENTRY} from {}", path.display()))?;
    ensure!(
        raw.len().is_multiple_of(2),
        "voice pt payload size {} is not bf16-aligned",
        raw.len()
    );
    Ok(raw
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<PathBuf> {
        std::env::var("RLX_VOXTRAL_TTS_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.join("voice_embedding/neutral_female.pt").is_file())
    }

    #[test]
    fn load_neutral_female_pt_shape() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with voice_embedding/*.pt");
            return;
        };
        let emb = load_voice_pt(&dir.join("voice_embedding/neutral_female.pt")).expect("load pt");
        assert_eq!(emb.n_tokens, 218);
        assert_eq!(emb.hidden, 3072);
        assert_eq!(emb.data.len(), 218 * 3072);
    }
}
