// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Built-in voices: what the LM prompt and the S3Gen decoder are conditioned on.
//!
//! A voice reaches the model three ways at once, and each comes from a file the
//! repository already ships, so no speaker encoder runs for a built-in voice:
//!
//! | what | consumer | source |
//! |---|---|---|
//! | transcript + codec tokens of the reference clip | LM prompt prefix | `voices/voices.json`, `voices/*.npz` |
//! | 512-d pyannote x-vector | LM `spk_proj` → position 0 | `voices/*.npz` (`embedding`) |
//! | prompt tokens, 24 kHz log-mel, 192-d CAM++ x-vector | S3Gen flow | `cpp/default/voices.json` |
//!
//! The last file is the conditioning `s3gen.embed_ref` derives from the
//! reference wav (first 10 s), precomputed for the llama.cpp fork; it matches a
//! fresh `embed_ref` to 4e-5 on the mel and 3e-6 on the x-vector.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::value::RawValue;

/// One entry of `voices/voices.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct VoiceSpec {
    pub voice_id: String,
    pub reference: String,
    pub artifacts: String,
    pub transcript: String,
    #[serde(default)]
    pub duration_s: f32,
}

/// What S3Gen's flow is conditioned on: the reference clip's own tokens and
/// mel (the in-context prompt) and its CAM++ x-vector.
#[derive(Debug, Clone)]
pub struct S3Conditioning {
    pub prompt_token: Vec<u32>,
    /// `[frames, 80]` row-major log-mel at 24 kHz / hop 480.
    pub prompt_feat: Vec<f32>,
    pub prompt_frames: usize,
    /// 192-d CAM++ embedding (not yet normalized).
    pub xvector: Vec<f32>,
}

/// Everything a built-in voice contributes to one generation.
#[derive(Debug, Clone)]
pub struct Voice {
    pub name: String,
    pub transcript: String,
    /// The reference clip as 0-based S3 codec ids, for the LM prompt.
    pub reference_tokens: Vec<u32>,
    /// 512-d pyannote x-vector for `spk_proj`.
    pub speaker_embedding: Vec<f32>,
    /// Conditioning for the released S3Gen flow (≤ 10 s of reference).
    pub s3: S3Conditioning,
    /// The reference clip, for conditioning a decoder that wants a different
    /// cut of it (the student: 4 s, ≤ 112 tokens).
    pub reference_wav: Option<PathBuf>,
}

/// Lazily-read voice library of one repository.
pub struct VoiceLibrary {
    voices_dir: PathBuf,
    specs: BTreeMap<String, VoiceSpec>,
    s3_path: PathBuf,
}

#[derive(Deserialize)]
struct S3Entry {
    prompt_token: Vec<Vec<u32>>,
    prompt_feat: Vec<Vec<Vec<f32>>>,
    embedding: Vec<Vec<f32>>,
}

impl VoiceLibrary {
    /// `voices_json` is `voices/voices.json`; `s3_json` the precomputed S3Gen
    /// conditioning (`cpp/default/voices.json`).
    pub fn open(voices_json: &Path, s3_json: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(voices_json)
            .with_context(|| format!("read {}", voices_json.display()))?;
        let specs: BTreeMap<String, VoiceSpec> =
            serde_json::from_str(&text).context("parse voices.json")?;
        ensure!(
            s3_json.is_file(),
            "missing S3Gen voice conditioning {} (fetch cpp/default/voices.json)",
            s3_json.display()
        );
        Ok(Self {
            voices_dir: voices_json
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
            specs,
            s3_path: s3_json.to_path_buf(),
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.specs.keys().map(String::as_str)
    }

    pub fn spec(&self, name: &str) -> Option<&VoiceSpec> {
        self.specs.get(name)
    }

    /// Reference clip (24 kHz-or-other wav) of a voice.
    pub fn reference_wav(&self, name: &str) -> Option<PathBuf> {
        self.specs
            .get(name)
            .map(|s| self.voices_dir.join(&s.reference))
    }

    pub fn load(&self, name: &str) -> Result<Voice> {
        let spec = self.specs.get(name).with_context(|| {
            format!(
                "voice {name:?} not available; {} voices, e.g. {:?}",
                self.specs.len(),
                self.specs.keys().take(8).collect::<Vec<_>>()
            )
        })?;
        let npz = self.voices_dir.join(&spec.artifacts);
        let mut arrays = read_npz(&npz)?;
        let speaker_embedding = arrays
            .remove("embedding")
            .with_context(|| format!("{}: no `embedding`", npz.display()))?
            .into_f32()?;
        let reference_tokens = arrays
            .remove("reference_tokens")
            .with_context(|| format!("{}: no `reference_tokens`", npz.display()))?
            .into_u32()?;
        let s3 = self.s3_conditioning(name)?;
        Ok(Voice {
            name: name.to_string(),
            transcript: spec.transcript.clone(),
            reference_tokens,
            speaker_embedding,
            s3,
            reference_wav: Some(self.voices_dir.join(&spec.reference)),
        })
    }

    /// Parse only `name`'s entry of the (100 MB) conditioning file.
    fn s3_conditioning(&self, name: &str) -> Result<S3Conditioning> {
        let text = std::fs::read_to_string(&self.s3_path)
            .with_context(|| format!("read {}", self.s3_path.display()))?;
        let all: BTreeMap<&str, &RawValue> =
            serde_json::from_str(&text).context("parse S3Gen voice conditioning")?;
        let raw = all
            .get(name)
            .with_context(|| format!("{name:?} has no S3Gen conditioning"))?;
        let e: S3Entry = serde_json::from_str(raw.get()).context("conditioning entry")?;
        let prompt_token = e.prompt_token.into_iter().next().unwrap_or_default();
        let feat = e.prompt_feat.into_iter().next().unwrap_or_default();
        let prompt_frames = feat.len();
        ensure!(
            feat.iter().all(|r| r.len() == 80),
            "prompt_feat rows must be 80 mel bins"
        );
        let xvector = e.embedding.into_iter().next().unwrap_or_default();
        ensure!(
            xvector.len() == 192,
            "x-vector has {} values, want 192",
            xvector.len()
        );
        Ok(S3Conditioning {
            prompt_token,
            prompt_feat: feat.into_iter().flatten().collect(),
            prompt_frames,
            xvector,
        })
    }
}

/// One array of an `.npz`.
#[derive(Debug, Clone)]
pub enum NpyArray {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl NpyArray {
    pub fn into_f32(self) -> Result<Vec<f32>> {
        match self {
            Self::F32(v) => Ok(v),
            _ => bail!("expected a float32 array"),
        }
    }

    pub fn into_u32(self) -> Result<Vec<u32>> {
        let v: Vec<i64> = match self {
            Self::I32(v) => v.into_iter().map(i64::from).collect(),
            Self::I64(v) => v,
            Self::F32(_) => bail!("expected an integer array"),
        };
        v.into_iter()
            .map(|x| u32::try_from(x).context("negative token id"))
            .collect()
    }
}

/// Read every member of an `.npz` (C-order `<f4`, `<i4` or `<i8` arrays).
pub fn read_npz(path: &Path) -> Result<BTreeMap<String, NpyArray>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(f).with_context(|| format!("zip {}", path.display()))?;
    let mut out = BTreeMap::new();
    for i in 0..zip.len() {
        let mut member = zip.by_index(i)?;
        let name = member.name().trim_end_matches(".npy").to_string();
        let mut bytes = Vec::new();
        member.read_to_end(&mut bytes)?;
        out.insert(
            name.clone(),
            parse_npy(&bytes).with_context(|| format!("{}:{name}", path.display()))?,
        );
    }
    Ok(out)
}

fn parse_npy(data: &[u8]) -> Result<NpyArray> {
    ensure!(
        data.len() >= 10 && &data[..6] == b"\x93NUMPY",
        "not an NPY array"
    );
    let (hlen, start) = match data[6] {
        1 => (u16::from_le_bytes([data[8], data[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
            12,
        ),
        v => bail!("NPY version {v}"),
    };
    let header = std::str::from_utf8(&data[start..start + hlen]).context("NPY header")?;
    ensure!(
        !header.contains("'fortran_order': True"),
        "Fortran-order NPY"
    );
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .context("NPY descr")?;
    let body = &data[start + hlen..];
    Ok(match descr {
        "<f4" => NpyArray::F32(
            body.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        "<i4" => NpyArray::I32(
            body.as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        "<i8" => NpyArray::I64(
            body.as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect(),
        ),
        other => bail!("NPY dtype {other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, payload: &[u8], n: usize) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn parses_float_and_int_arrays() {
        let f: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(
            matches!(parse_npy(&npy("<f4", &f, 2)).unwrap(), NpyArray::F32(v) if v == [1.5, -2.0])
        );
        let i: Vec<u8> = [6405i32, 4299]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            parse_npy(&npy("<i4", &i, 2)).unwrap().into_u32().unwrap(),
            [6405, 4299]
        );
    }
}
