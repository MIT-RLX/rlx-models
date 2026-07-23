// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! End-to-end ASR session (RLX-native weights only).

use crate::beam::StreamingCtcBeam;
use crate::effective_decoder::EffectiveStep1;
use crate::encoder::Encoder;
use crate::frontend;
use crate::search::{aed_start_tokens, argmax_token, ctc_first_pass};
use crate::spec::{AED_CACHE_IN_ELEMS, BEAM, BLANK, ENC_ELEMS, EOS, SOS, VOCAB};
use crate::textproc::{Etiquette, Hammer};
use crate::units::Units;
use crate::vad::Vad;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub token_ids: Vec<u32>,
}

/// Streaming first-pass state: accumulate CTC frames, emit partial text.
pub struct StreamingAsr {
    pub units: Units,
    pub ctc: StreamingCtcBeam,
    pub hammer: Hammer,
    pub etiquette: Option<Etiquette>,
}

impl StreamingAsr {
    pub fn new(units: Units) -> Self {
        Self {
            units,
            ctc: StreamingCtcBeam::new(BLANK as usize, BEAM),
            hammer: Hammer { fsts: Vec::new() },
            etiquette: None,
        }
    }

    pub fn reset(&mut self) {
        self.ctc.reset();
    }

    pub fn push_frame(&mut self, logp_row: &[f32]) -> Result<String> {
        if logp_row.len() != VOCAB {
            bail!("expected {VOCAB} log-probs, got {}", logp_row.len());
        }
        self.ctc.push(logp_row);
        Ok(self.partial_text())
    }

    pub fn push_frames(&mut self, logp: &[f32], n_frames: usize) -> Result<String> {
        if logp.len() < n_frames * VOCAB {
            bail!("logp too short");
        }
        self.ctc.push_many(logp, n_frames, VOCAB);
        Ok(self.partial_text())
    }

    pub fn partial_text(&self) -> String {
        let ids: Vec<u32> = self
            .ctc
            .partial_ids()
            .into_iter()
            .map(|x| x as u32)
            .filter(|&t| t != SOS && t != EOS && t >= 4)
            .collect();
        let mut text = self.units.decode(&ids);
        text = self.hammer.apply(&text);
        if let Some(eti) = &self.etiquette {
            text = eti.apply(&text);
        }
        text
    }

    pub fn finish(&mut self) -> Transcript {
        let (ids, _score) = self.ctc.best();
        let token_ids: Vec<u32> = ids.into_iter().map(|x| x as u32).collect();
        let text_ids: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&t| t != SOS && t != EOS && t >= 4)
            .collect();
        let mut text = self.units.decode(&text_ids);
        text = self.hammer.apply(&text);
        if let Some(eti) = &self.etiquette {
            text = eti.apply(&text);
        }
        Transcript { text, token_ids }
    }
}

pub struct AsrSession {
    pub dir: PathBuf,
    pub units: Units,
    pub vad: Vad,
    pub encoder: Encoder,
    pub hammer: Hammer,
    pub etiquette: Option<Etiquette>,
    /// Native AED (embed / Ah / W_out under `decoder/`).
    decoder: Option<EffectiveStep1>,
}

impl AsrSession {
    /// Load from `model.gguf` under `dir` (`RLX_ASR_DIR` / `weights/asr`).
    /// Loose sidecars are optional pack leftovers.
    pub fn load(dir: &Path) -> Result<Self> {
        let paths = crate::AsrPaths::new(dir);
        let gguf = paths
            .gguf()
            .and_then(|p| crate::gguf_io::AsrGguf::open(p).ok());

        let units = if let Some(ref g) = gguf {
            if let Some(pieces) = g.units() {
                Units::from_pieces(pieces)
            } else if paths.units_txt().is_file() {
                Units::load(&paths.units_txt())?
            } else {
                bail!("rlx-asr.units missing in GGUF and no units.txt under {}", dir.display());
            }
        } else {
            Units::load(&paths.units_txt())
                .with_context(|| format!("units.txt under {}", dir.display()))?
        };

        let hammer = if let Some(ref g) = gguf {
            g.load_hammer("en_US").unwrap_or(Hammer { fsts: Vec::new() })
        } else if let Some(tp) = paths.tp_dir() {
            Hammer::load_dir(&tp, "en_US")?
        } else {
            Hammer { fsts: Vec::new() }
        };
        let etiquette = if let Some(ref g) = gguf {
            g.etiquette_json()
                .map(|s| Etiquette::from_json_str(s))
                .transpose()?
        } else if let Some(p) = paths.etiquette_json() {
            Some(Etiquette::load(&p)?)
        } else {
            None
        };
        let decoder = if let Some(ref g) = gguf {
            g.load_effective_step1().ok()
        } else {
            EffectiveStep1::load_bins(&paths.decoder_dir()).ok()
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            units,
            vad: Vad::default(),
            encoder: Encoder::default(),
            hammer,
            etiquette,
            decoder,
        })
    }

    /// Transcribe mono PCM with energy VAD + stub/native encoder + native AED (or CTC).
    pub fn transcribe(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Transcript> {
        let trimmed = self.vad.trim(pcm, sample_rate).unwrap_or_else(|_| pcm.to_vec());
        let pcm_use = if trimmed.is_empty() {
            pcm
        } else {
            trimmed.as_slice()
        };
        let mel = frontend::log_mel_fbank(pcm_use, sample_rate)?;
        let enc = if mel.is_empty() {
            crate::encoder::EncoderOutputs {
                wp_logprob: vec![0.0; VOCAB],
                encoder_cache: vec![0.0; ENC_ELEMS],
                n_frames: 1,
            }
        } else {
            self.encoder.forward_stub(&mel)?
        };
        let ctc_hyps = ctc_first_pass(&enc.wp_logprob, enc.n_frames);
        let mut tokens = aed_start_tokens();
        let mut decoded: Vec<u32> = vec![SOS];
        if let Some(dec) = self.decoder.as_ref() {
            for _step in 0..64 {
                let logprob = dec.logprob_with_encoder(tokens[0], &enc.encoder_cache)?;
                let next = argmax_token(&logprob, 0);
                decoded.push(next);
                if next == EOS {
                    break;
                }
                tokens = [next; BEAM];
            }
        } else if let Some((ids, _)) = ctc_hyps.first() {
            decoded.extend(ids.iter().map(|&x| x as u32));
        } else {
            bail!("no native AED under decoder/ and empty CTC");
        }
        let text_ids: Vec<u32> = decoded
            .iter()
            .copied()
            .filter(|&t| t != SOS && t != EOS && t >= 4)
            .collect();
        let mut text = self.units.decode(&text_ids);
        text = self.hammer.apply(&text);
        if let Some(eti) = &self.etiquette {
            text = eti.apply(&text);
        }
        Ok(Transcript {
            text,
            token_ids: decoded,
        })
    }

    /// AED step against a provided encoder_cache (debug / joint decode).
    pub fn aed_step(
        &mut self,
        tokens: &[u32; BEAM],
        encoder_cache: &[f32],
    ) -> Result<Vec<f32>> {
        let (lp, _) = self.aed_step_full(tokens, encoder_cache, None)?;
        Ok(lp)
    }

    /// Native AED step. `cache_in` is ignored (history-free effective maps).
    pub fn aed_step_full(
        &mut self,
        tokens: &[u32; BEAM],
        encoder_cache: &[f32],
        _cache_in: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let dec = self
            .decoder
            .as_ref()
            .context("native AED not loaded (decoder/embed.bin + effective maps)")?;
        let lp = dec.logprob_with_encoder(tokens[0], encoder_cache)?;
        Ok((lp, vec![0.0; AED_CACHE_IN_ELEMS]))
    }
}
