// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! End-to-end ASR session (RLX-native weights only).

use crate::beam::StreamingCtcBeam;
use crate::effective_decoder::EffectiveStep1;
use crate::encoder::Encoder;
use crate::frontend;
use crate::search::{aed_start_tokens, argmax_token, ctc_first_pass_beam};
use crate::spec::{AED_CACHE_IN_ELEMS, BEAM, BLANK, EOS, SOS, VOCAB};
use crate::textproc::{Etiquette, Hammer};
use crate::units::Units;
use crate::vad::Vad;
use anyhow::{Context, Result, bail};
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
        text = crate::text_quality::cleanup_transcript(&text);
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
        text = crate::text_quality::cleanup_transcript(&text);
        if let Some(eti) = &self.etiquette {
            text = eti.apply(&text);
        }
        Transcript { text, token_ids }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TranscribeOpts {
    /// Skip energy trim — use when the caller already VAD-segmented the PCM.
    pub skip_vad: bool,
    /// CTC prefix beam width (default 16 for folded encoder).
    pub ctc_beam: usize,
}

impl TranscribeOpts {
    pub fn segment() -> Self {
        Self {
            skip_vad: true,
            ctc_beam: 16,
        }
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
    /// Folded Conformer + CTC head (preferred over stub encoder).
    folded: Option<crate::folded_encoder::FoldedEncoder>,
    /// Full 28-layer stack (`RLX_ASR_ENCODER=native`).
    native: Option<crate::native_encoder::NativeEncoder>,
}

impl AsrSession {
    /// Load from `model.gguf` under `dir` (`RLX_ASR_DIR` / `weights/asr`).
    /// Loose sidecars are optional pack leftovers.
    pub fn load(dir: &Path) -> Result<Self> {
        let paths = crate::AsrPaths::new(dir);
        let pack = paths
            .pack()
            .and_then(|p| crate::gguf_io::AsrPack::open(p).ok());

        let units = if let Some(ref g) = pack {
            if let Some(pieces) = g.units() {
                Units::from_pieces(pieces)
            } else if paths.units_txt().is_file() {
                Units::load(&paths.units_txt())?
            } else {
                bail!(
                    "units missing in pack and no units.txt under {}",
                    dir.display()
                );
            }
        } else {
            Units::load(&paths.units_txt())
                .with_context(|| format!("units.txt under {}", dir.display()))?
        };

        let hammer = if let Some(ref g) = pack {
            g.load_hammer("en_US")
                .unwrap_or(Hammer { fsts: Vec::new() })
        } else if let Some(tp) = paths.tp_dir() {
            Hammer::load_dir(&tp, "en_US")?
        } else {
            Hammer { fsts: Vec::new() }
        };
        let etiquette = if let Some(ref g) = pack {
            g.etiquette_json()
                .map(|s| Etiquette::from_json_str(&s))
                .transpose()?
        } else if let Some(p) = paths.etiquette_json() {
            Some(Etiquette::load(&p)?)
        } else {
            None
        };
        let decoder = if let Some(ref g) = pack {
            g.load_effective_step1().ok()
        } else {
            EffectiveStep1::load_bins(&paths.decoder_dir()).ok()
        };
        let folded = pack
            .as_ref()
            .filter(|p| crate::folded_encoder::FoldedEncoder::is_available(p))
            .and_then(|p| crate::folded_encoder::FoldedEncoder::from_pack(p).ok());
        let native = pack
            .as_ref()
            .filter(|p| crate::native_encoder::NativeEncoder::is_available(p))
            .and_then(|p| crate::native_encoder::NativeEncoder::from_pack(p).ok());
        Ok(Self {
            dir: dir.to_path_buf(),
            units,
            vad: Vad::default(),
            encoder: Encoder::default(),
            hammer,
            etiquette,
            decoder,
            folded,
            native,
        })
    }

    /// Transcribe mono PCM with energy VAD + folded/native encoder + CTC/AED.
    pub fn transcribe(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Transcript> {
        self.transcribe_opts(pcm, sample_rate, TranscribeOpts::default())
    }

    /// Transcribe with optional VAD skip / beam width (for pre-segmented chunks).
    pub fn transcribe_opts(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        opts: TranscribeOpts,
    ) -> Result<Transcript> {
        let pcm_buf;
        let pcm_use: &[f32] = if opts.skip_vad || !crate::env::vad_enabled() {
            pcm
        } else {
            let trimmed = self
                .vad
                .trim(pcm, sample_rate)
                .unwrap_or_else(|_| pcm.to_vec());
            if trimmed.is_empty() {
                pcm
            } else {
                pcm_buf = trimmed;
                pcm_buf.as_slice()
            }
        };
        self.transcribe_pcm(pcm_use, sample_rate, opts)
    }

    fn transcribe_pcm(
        &mut self,
        pcm_use: &[f32],
        sample_rate: u32,
        opts: TranscribeOpts,
    ) -> Result<Transcript> {
        let mel = frontend::log_mel_fbank(pcm_use, sample_rate)?;
        if mel.is_empty() {
            return Ok(Transcript {
                text: String::new(),
                token_ids: vec![SOS],
            });
        }

        let mut mel = mel;
        if (self.folded.is_some() || self.native.is_some())
            && let Some(ref folded) = self.folded
        {
            folded.preprocess_mel(&mut mel);
        }
        if let Some(row) = mel.first()
            && crate::env::timing()
        {
            eprintln!(
                "[rlx-asr] mel frames={} f0=[{:.4}, {:.4}, {:.4}] h_map={} mlp={}",
                mel.len(),
                row.first().copied().unwrap_or(0.0),
                row.get(1).copied().unwrap_or(0.0),
                row.get(2).copied().unwrap_or(0.0),
                self.folded.as_ref().map(|f| f.has_h_map()).unwrap_or(false),
                self.folded
                    .as_ref()
                    .map(|f| f.has_body_mlp())
                    .unwrap_or(false),
            );
        }

        let probe =
            crate::env::early_abort() && mel.len() > crate::folded_encoder::FOLDED_MEL_FRAMES;
        if probe {
            // Probe folded only — native 28-layer is ~10× slower and not at parity yet.
            if let Some(ref folded) = self.folded {
                let enc = folded.forward_mel_limited(&mel, Some(1))?;
                let text = self.decode_enc(&enc, opts)?;
                if crate::text_quality::is_garbage(&text.text)
                    || crate::text_quality::has_repetition_loop(&text.text, 6)
                {
                    return Ok(Transcript {
                        text: String::new(),
                        token_ids: vec![SOS],
                    });
                }
            }
        }

        // Folded path: decode each mel window separately and keep strong chunks only.
        // Concatenating CTC log-probs across padded tails injects "to do to do" stutter.
        if self.folded.is_some()
            && !matches!(crate::env::encoder_mode(), crate::env::EncoderMode::Native)
        {
            let tr = self.transcribe_folded_chunked(&mel, opts)?;
            if crate::text_quality::is_garbage(&tr.text)
                || crate::text_quality::has_repetition_loop(&tr.text, 6)
            {
                if crate::env::timing() {
                    eprintln!("[rlx-asr] filtered garbage transcript: {:?}", tr.text);
                }
                return Ok(Transcript {
                    text: String::new(),
                    token_ids: vec![SOS],
                });
            }
            return Ok(tr);
        }

        let enc = self.forward_encoder(&mel)?;
        if crate::env::timing() {
            eprintln!("[rlx-asr] encoder frames={}", enc.n_frames);
        }

        let tr = self.decode_enc(&enc, opts)?;
        if crate::text_quality::is_garbage(&tr.text)
            || crate::text_quality::has_repetition_loop(&tr.text, 6)
        {
            if crate::env::timing() {
                eprintln!("[rlx-asr] filtered garbage transcript: {:?}", tr.text);
            }
            return Ok(Transcript {
                text: String::new(),
                token_ids: vec![SOS],
            });
        }
        Ok(tr)
    }

    /// Per-window folded CTC decode; drops weak/garbage tail chunks before join.
    fn transcribe_folded_chunked(
        &mut self,
        mel: &[Vec<f32>],
        opts: TranscribeOpts,
    ) -> Result<Transcript> {
        let chunks = {
            let folded = self.folded.as_ref().context("folded encoder required")?;
            crate::folded_encoder::mel_windows(
                mel,
                crate::folded_encoder::FOLDED_MEL_FRAMES,
                crate::folded_encoder::FOLDED_MEL_FRAMES,
            )
            .into_iter()
            .map(|chunk| folded.forward_chunk(&chunk))
            .collect::<Result<Vec<_>>>()?
        };
        if crate::env::timing() {
            eprintln!("[rlx-asr] folded chunks={} (per-chunk gate)", chunks.len());
        }
        let mut parts: Vec<String> = Vec::new();
        let mut token_ids: Vec<u32> = vec![SOS];
        let mut seen_content: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (i, (enc, logp)) in chunks.into_iter().enumerate() {
            let n_frames = logp.len() / crate::spec::VOCAB;
            let mut cache = vec![0f32; crate::spec::AED_WINDOW_FRAMES * crate::spec::DECODER_DIM];
            let copy = (enc.len() / crate::spec::DECODER_DIM).min(crate::spec::AED_WINDOW_FRAMES);
            for t in 0..copy {
                let src = &enc[t * crate::spec::DECODER_DIM..(t + 1) * crate::spec::DECODER_DIM];
                cache[t * crate::spec::DECODER_DIM..(t + 1) * crate::spec::DECODER_DIM]
                    .copy_from_slice(src);
            }
            let enc_out = crate::encoder::EncoderOutputs {
                wp_logprob: logp,
                encoder_cache: cache,
                n_frames,
            };
            let tr = self.decode_enc(&enc_out, opts)?;
            let text = crate::text_quality::cleanup_transcript(&tr.text);
            let weak = crate::text_quality::is_weak_chunk(&text)
                || crate::text_quality::is_garbage(&text)
                || crate::text_quality::has_repetition_loop(&text, 4)
                // Later windows need real lexical density; padded tails are noisy.
                || (i > 0 && crate::text_quality::content_word_count(&text) < 2);
            if weak {
                if crate::env::timing() {
                    eprintln!("[rlx-asr] drop chunk {i}: {:?}", text);
                }
                // Always keep a non-empty first chunk even if weak — better than silence.
                if i == 0 && parts.is_empty() && !text.is_empty() {
                    parts.push(text.clone());
                    token_ids.extend(
                        tr.token_ids
                            .iter()
                            .copied()
                            .filter(|&t| t != SOS && t != EOS),
                    );
                }
                continue;
            }
            // Long-form: require chunks to introduce new lexical content.
            if i > 0 && !parts.is_empty() {
                let new_words: Vec<String> = text
                    .split_whitespace()
                    .map(|w| {
                        w.trim_matches(|c: char| !c.is_ascii_alphanumeric())
                            .to_ascii_lowercase()
                    })
                    .filter(|w| w.len() >= 4)
                    .collect();
                let novel = new_words
                    .iter()
                    .filter(|w| !seen_content.contains(*w))
                    .count();
                if novel < 2 {
                    if crate::env::timing() {
                        eprintln!("[rlx-asr] drop redundant chunk {i}: {:?}", text);
                    }
                    continue;
                }
            }
            for w in text.split_whitespace() {
                let n = w
                    .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                    .to_ascii_lowercase();
                if n.len() >= 4 {
                    seen_content.insert(n);
                }
            }
            parts.push(text);
            token_ids.extend(
                tr.token_ids
                    .iter()
                    .copied()
                    .filter(|&t| t != SOS && t != EOS),
            );
            // Soft cap: ~45s of accepted speech windows.
            if parts.len() >= 12 {
                break;
            }
        }
        let text = crate::text_quality::cleanup_transcript(&parts.join(" "));
        Ok(Transcript { text, token_ids })
    }

    fn forward_encoder(&self, mel: &[Vec<f32>]) -> Result<crate::encoder::EncoderOutputs> {
        match crate::env::encoder_mode() {
            crate::env::EncoderMode::Native if self.native.is_some() => {
                self.native.as_ref().unwrap().forward_mel(mel)
            }
            _ if self.folded.is_some() => self.folded.as_ref().unwrap().forward_mel(mel),
            _ => self.encoder.forward_stub(mel),
        }
    }

    fn decode_enc(
        &mut self,
        enc: &crate::encoder::EncoderOutputs,
        opts: TranscribeOpts,
    ) -> Result<Transcript> {
        let ctc_beam = if opts.ctc_beam > 0 {
            opts.ctc_beam
        } else if self.folded.is_some() || self.native.is_some() {
            16
        } else {
            BEAM
        };
        let ctc_hyps = ctc_first_pass_beam(&enc.wp_logprob, enc.n_frames, ctc_beam);
        let mut decoded: Vec<u32> = vec![SOS];

        if let Some((ids, _)) = ctc_hyps.first() {
            decoded.extend(ids.iter().map(|&x| x as u32));
        } else if let Some(dec) = self.decoder.as_ref() {
            let mut tokens = aed_start_tokens();
            for _step in 0..256 {
                let logprob = dec.logprob_with_encoder(tokens[0], &enc.encoder_cache)?;
                let next = argmax_token(&logprob, 0);
                decoded.push(next);
                if next == EOS {
                    break;
                }
                tokens = [next; BEAM];
            }
        } else {
            bail!("empty CTC and no native AED");
        }
        let text_ids: Vec<u32> = decoded
            .iter()
            .copied()
            .filter(|&t| t != SOS && t != EOS && t >= 4)
            .collect();
        let mut text = self.units.decode(&text_ids);
        text = self.hammer.apply(&text);
        text = crate::text_quality::cleanup_transcript(&text);
        if let Some(eti) = &self.etiquette {
            text = eti.apply(&text);
        }
        Ok(Transcript {
            text,
            token_ids: decoded,
        })
    }

    /// AED step against a provided encoder_cache (debug / joint decode).
    pub fn aed_step(&mut self, tokens: &[u32; BEAM], encoder_cache: &[f32]) -> Result<Vec<f32>> {
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
