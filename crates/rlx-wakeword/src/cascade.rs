// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Optional cascade stages: speaker gate and ASR confirm.
//!
//! Without feature `speaker-id`, [`SpeakerGate`] defaults to [`Unsupported`].
//! With `--features speaker-id`, use `SpeakerIdGate` (cosine over enrolled embeddings).

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsupported;

impl core::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cascade stage not enabled in this build")
    }
}

impl std::error::Error for Unsupported {}

/// Speaker enrollment gate. Without `speaker-id`, always unsupported.
pub trait SpeakerGate {
    fn score(&self, pcm: &[f32]) -> Result<f32> {
        let _ = pcm;
        bail!(Unsupported)
    }

    fn best_match(&self, pcm: &[f32]) -> Result<(String, f32)> {
        let _ = pcm;
        bail!(Unsupported)
    }
}

/// Future ASR confirm on candidates.
pub trait AsrConfirm {
    fn confirm(&self, pcm: &[f32], phrase_id: &str) -> Result<bool> {
        let _ = (pcm, phrase_id);
        bail!(Unsupported)
    }
}

pub struct NullSpeaker;
impl SpeakerGate for NullSpeaker {}

pub struct NullAsr;
impl AsrConfirm for NullAsr {}

#[cfg(feature = "speaker-id")]
pub mod speaker {
    //! Speaker enrollment gate (feature `speaker-id`).
    //!
    //! Enroll fixed-size embeddings and score candidates with cosine similarity.
    //! Build with `--features speaker-id`.

    use super::SpeakerGate;
    use anyhow::{Result, ensure};

    #[derive(Debug, Clone)]
    pub struct EnrolledSpeaker {
        pub id: String,
        pub embedding: Vec<f32>,
    }

    #[derive(Debug, Clone)]
    pub struct SpeakerIdConfig {
        /// Minimum cosine similarity to accept.
        pub threshold: f32,
        /// If true, reject candidates when no enrolled speaker matches.
        pub require_match: bool,
    }

    impl Default for SpeakerIdConfig {
        fn default() -> Self {
            Self {
                threshold: 0.65,
                require_match: false,
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct SpeakerIdGate {
        pub cfg: SpeakerIdConfig,
        pub speakers: Vec<EnrolledSpeaker>,
    }

    impl SpeakerIdGate {
        pub fn new(cfg: SpeakerIdConfig) -> Self {
            Self {
                cfg,
                speakers: Vec::new(),
            }
        }

        pub fn enroll(&mut self, id: impl Into<String>, embedding: Vec<f32>) -> Result<()> {
            ensure!(!embedding.is_empty(), "empty embedding");
            let id = id.into();
            if let Some(s) = self.speakers.iter_mut().find(|s| s.id == id) {
                s.embedding = embedding;
            } else {
                self.speakers.push(EnrolledSpeaker { id, embedding });
            }
            Ok(())
        }

        /// Placeholder embedding from PCM energy stats (replace with a real speaker encoder).
        pub fn embed_from_pcm(pcm: &[f32], dim: usize) -> Vec<f32> {
            let dim = dim.max(8);
            let mut out = vec![0.0f32; dim];
            if pcm.is_empty() {
                return out;
            }
            let n = pcm.len();
            for (i, slot) in out.iter_mut().enumerate() {
                let mut s = 0.0f32;
                let mut c = 0usize;
                let mut j = i;
                while j < n {
                    s += pcm[j].abs();
                    c += 1;
                    j += dim;
                }
                *slot = if c == 0 { 0.0 } else { s / c as f32 };
            }
            let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut out {
                *x /= norm;
            }
            out
        }

        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let n = a.len().min(b.len());
            if n == 0 {
                return 0.0;
            }
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..n {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            dot / (na.sqrt().max(1e-8) * nb.sqrt().max(1e-8))
        }
    }

    impl SpeakerGate for SpeakerIdGate {
        fn score(&self, pcm: &[f32]) -> Result<f32> {
            let (_id, sim) = self.best_match(pcm)?;
            Ok(sim)
        }

        fn best_match(&self, pcm: &[f32]) -> Result<(String, f32)> {
            ensure!(!self.speakers.is_empty(), "no enrolled speakers");
            let dim = self.speakers[0].embedding.len();
            let emb = Self::embed_from_pcm(pcm, dim);
            let mut best_id = self.speakers[0].id.clone();
            let mut best = -1.0f32;
            for s in &self.speakers {
                let sim = Self::cosine(&emb, &s.embedding);
                if sim > best {
                    best = sim;
                    best_id = s.id.clone();
                }
            }
            Ok((best_id, best))
        }
    }
}
