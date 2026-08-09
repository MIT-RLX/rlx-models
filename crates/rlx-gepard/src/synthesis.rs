// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard TTS synthesizer — AR frame generation from text.
//!
//! Supports both eager CPU inference (for testing/validation) and
//! compiled RLX IR graphs (for production multi-backend inference).

use crate::gepard_decoder::GepardDecoder;
use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use safetensors::SafeTensors;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::backbone::{
    BackboneWeights, GepardKvCache, backbone_decode_step, backbone_prefill, matvec,
};
use crate::codec_ops::NUM_CHANNELS;
use crate::compiled_session::GepardCompiledSession;
use crate::config::GepardConfig;
use crate::tokenizer::GepardTokenizer;
use crate::weights::{GepardOverlay, load_safetensors_bytes};

/// Map CLI device to the execution device (no silent remaps).
fn resolve_exec_device(requested: Device) -> Device {
    requested
}

const SAMPLE_RATE: u32 = 22050;
/// Production defaults from gepard-train MODEL_GUIDE §13.3.
const STOP_THRESHOLD: f32 = 0.5;
const MIN_FRAMES: usize = 8;
const MAX_FRAMES_CAP: usize = 2000;
const TEMPERATURE: f32 = 0.4;
const FRAME_RATE_HZ: f32 = 21.5;

/// Inference knobs (MODEL_GUIDE §6.3 / gepard-inference `config.yaml`).
#[derive(Debug, Clone)]
pub struct InferOpts {
    pub max_frames: usize,
    pub stop_threshold: f32,
    pub temperature: f32,
    /// Argmax codebook heads (parity / deterministic).
    pub greedy: bool,
    /// RNG seed for temperature sampling (`fastrand`).
    pub seed: u64,
    /// >1.0 penalises tokens seen in the recent window per head (1.0 = off).
    pub repetition_penalty: f32,
    /// Recent frames tracked for repetition penalty (0 = all history).
    pub repetition_window: usize,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self {
            max_frames: MAX_FRAMES_CAP,
            stop_threshold: STOP_THRESHOLD,
            temperature: TEMPERATURE,
            greedy: false,
            seed: 54,
            repetition_penalty: 1.0,
            repetition_window: 32,
        }
    }
}

/// Pick a validated sampling seed for `text` (fox=54, long paragraph=4).
pub fn default_seed_for_text(text: &str) -> u64 {
    if text.split_whitespace().count() > 12 {
        4
    } else {
        54
    }
}
pub fn suggest_max_frames(text: &str, opts_cap: usize) -> usize {
    let words = text.split_whitespace().count().max(1);
    // ~0.35 s/word × frame rate × 1.4 headroom.
    let est = ((words as f32) * 0.35 * FRAME_RATE_HZ * 1.4).ceil() as usize;
    est.clamp(MIN_FRAMES, opts_cap.min(MAX_FRAMES_CAP))
}

// ── audio-frame embedding ─────────────────────────────────────────────────────

/// Embed one audio frame through the audio-embed stack:
/// 32 × lookup → concat `[hidden]` → Linear+GELU+Linear → affine-free LN → × scale.
pub fn embed_audio_frame(
    channels: &[u32],
    overlay: &GepardOverlay,
    audio_embed_dim: usize,
    hidden_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(channels.len(), NUM_CHANNELS);
    debug_assert_eq!(audio_embed_dim * NUM_CHANNELS, hidden_size);

    let mut concat = vec![0.0f32; hidden_size];
    for ch in 0..NUM_CHANNELS {
        let code = channels[ch] as usize;
        let table = &overlay.audio_embeddings[ch];
        let row = &table[code * audio_embed_dim..(code + 1) * audio_embed_dim];
        concat[ch * audio_embed_dim..(ch + 1) * audio_embed_dim].copy_from_slice(row);
    }

    let mut h = matvec(
        &overlay.audio_proj_w0,
        &concat,
        Some(&overlay.audio_proj_b0),
        hidden_size,
        hidden_size,
    );
    for v in h.iter_mut() {
        *v = gelu(*v);
    }

    let mut h = matvec(
        &overlay.audio_proj_w2,
        &h,
        Some(&overlay.audio_proj_b2),
        hidden_size,
        hidden_size,
    );

    // Affine-free LayerNorm (mean-center + RMS) — not RMSNorm.
    let mean = h.iter().sum::<f32>() / h.len() as f32;
    let var = h
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / h.len() as f32;
    let inv = (var + 1e-5).sqrt().recip();
    for v in h.iter_mut() {
        *v = (*v - mean) * inv;
    }
    for v in h.iter_mut() {
        *v *= overlay.audio_embed_scale;
    }
    h
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

// ── codebook head sampling ────────────────────────────────────────────────────

pub fn sample_all_heads(h: &[f32], overlay: &GepardOverlay, vocabs: &[u32]) -> Vec<u32> {
    sample_all_heads_temp(h, overlay, vocabs, 0.0, 1.0, None, 32)
}

/// Sample all codebook heads with optional temperature.  temperature=0 → argmax.
pub fn sample_all_heads_temp(
    h: &[f32],
    overlay: &GepardOverlay,
    vocabs: &[u32],
    temperature: f32,
    repetition_penalty: f32,
    recent_frames: Option<&[Vec<u32>]>,
    repetition_window: usize,
) -> Vec<u32> {
    (0..NUM_CHANNELS)
        .map(|ch| {
            let v = vocabs[ch] as usize;
            let mut logits = matvec(
                &overlay.codebook_weights[ch],
                h,
                Some(&overlay.codebook_biases[ch]),
                h.len(),
                v,
            );
            if repetition_penalty != 1.0 {
                if let Some(frames) = recent_frames {
                    let start = if repetition_window == 0 {
                        0
                    } else {
                        frames.len().saturating_sub(repetition_window)
                    };
                    for frame in &frames[start..] {
                        let tok = frame[ch] as usize;
                        if tok < logits.len() {
                            if logits[tok] > 0.0 {
                                logits[tok] /= repetition_penalty;
                            } else {
                                logits[tok] *= repetition_penalty;
                            }
                        }
                    }
                }
            }
            if temperature <= 0.0 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            } else {
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits
                    .iter()
                    .map(|&x| ((x - max_l) / temperature).exp())
                    .collect();
                let sum: f32 = exps.iter().sum();
                let r = (fastrand::f32() * sum).min(sum - 1e-7);
                let mut acc = 0.0f32;
                for (i, e) in exps.iter().enumerate() {
                    acc += e;
                    if acc >= r {
                        return i as u32;
                    }
                }
                (v - 1) as u32
            }
        })
        .collect()
}

pub fn stop_probability(h: &[f32], overlay: &GepardOverlay) -> f32 {
    let logit = matvec(
        &overlay.stop_weight,
        h,
        Some(&overlay.stop_bias),
        h.len(),
        1,
    );
    1.0 / (1.0 + (-logit[0]).exp())
}

// ── synthesizer ───────────────────────────────────────────────────────────────

pub struct GepardSynthesizer {
    bundle_path: PathBuf,
    device: Device,
    tokenizer: GepardTokenizer,
    /// Backend Qwen3.5 AR when `device != cpu` or `RLX_GEPARD_COMPILED=1`.
    compiled_session: Option<RefCell<GepardCompiledSession>>,
    /// Audio-head overlay (cached — avoid re-loading the 1 GiB safetensors on
    /// every `synthesize`, which doubled RSS and jetsam'd Metal long).
    overlay: GepardOverlay,
    gepard_decoder: Option<GepardDecoder>,
    pub opts: InferOpts,
}

impl GepardSynthesizer {
    pub fn new<P: AsRef<Path>>(bundle_path: P) -> Result<Self> {
        Self::open(bundle_path, Device::Cpu)
    }

    /// Open a checkpoint directory on `device` (backbone AR + NanoCodec on `device`
    /// when not CPU; CPU uses eager backbone unless `RLX_GEPARD_COMPILED=1`).
    pub fn open<P: AsRef<Path>>(bundle_path: P, device: Device) -> Result<Self> {
        let device = resolve_exec_device(device);
        Self::open_with_compiled(bundle_path, device, use_backend_ar(device))
    }

    /// Open with an explicit compiled-AR choice (tests / parity).
    pub fn open_with_compiled<P: AsRef<Path>>(
        bundle_path: P,
        device: Device,
        compiled: bool,
    ) -> Result<Self> {
        let device = resolve_exec_device(device);
        let bp = bundle_path.as_ref().to_path_buf();
        let cfg = GepardConfig::from_path(&bp).unwrap_or_else(|_| default_config());
        let tokenizer = GepardTokenizer::load(&bp, &cfg)
            .with_context(|| format!("load tokenizer under {}", bp.display()))?;
        let gepard_decoder = Self::load_gepard_decoder(&bp, device).ok();
        let compiled_session = if compiled {
            Some(RefCell::new(GepardCompiledSession::new(device, &cfg, &bp)?))
        } else {
            None
        };
        let model_path = bp.join("model.safetensors");
        anyhow::ensure!(
            model_path.is_file(),
            "gepard: missing model.safetensors under {}",
            bp.display()
        );
        let bytes = load_safetensors_bytes(&model_path)?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse {}", model_path.display()))?;
        let overlay = GepardOverlay::load(&st, cfg.num_audio_heads())?;
        Ok(Self {
            bundle_path: bp,
            device,
            tokenizer,
            compiled_session,
            overlay,
            gepard_decoder,
            opts: InferOpts::default(),
        })
    }

    pub fn with_opts(mut self, opts: InferOpts) -> Self {
        self.opts = opts;
        self
    }

    /// Create synthesizer for a named device (`cpu` / `metal` / `mlx` / …).
    pub fn with_device<P: AsRef<Path>>(bundle_path: P, device: &str) -> Result<Self> {
        Self::open(bundle_path, parse_device_name(device))
    }

    fn load_gepard_decoder(bundle_path: &Path, device: Device) -> Result<GepardDecoder> {
        let decoder_path = bundle_path.join("nano_dec_1.89kbps.safetensors");
        let bytes = std::fs::read(&decoder_path)
            .with_context(|| format!("read {}", decoder_path.display()))?;
        GepardDecoder::from_safetensors_on(&bytes, device).context("load Gepard HiFi-GAN decoder")
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn config(&self) -> GepardConfig {
        let p = self.bundle_path.join("gepard_config.json");
        if p.is_file() {
            GepardConfig::from_path(&p).unwrap_or_else(|_| default_config())
        } else {
            default_config()
        }
    }

    pub fn synthesize(&self, text: &str, voice_desc: &str) -> Result<Vec<f32>> {
        self.synthesize_with_reference(text, voice_desc, None)
    }

    /// Synthesize with optional reference codes for voice cloning.
    pub fn synthesize_with_reference(
        &self,
        text: &str,
        voice_desc: &str,
        ref_codes: Option<&[u32]>,
    ) -> Result<Vec<f32>> {
        let _ = voice_desc;
        let cfg = self.config();
        let model_path = self.bundle_path.join("model.safetensors");
        if !model_path.is_file() {
            if self.bundle_path.join("weights.safetensors").is_file() {
                bail!(
                    "overlay-only weights.safetensors is no longer supported (sine path removed); \
                     place model.safetensors from nineninesix/gepard-1.0 under {}",
                    self.bundle_path.display()
                );
            }
            bail!("No model.safetensors in {}", self.bundle_path.display());
        }
        self.synthesize_with_model_ref(text, &cfg, &model_path, ref_codes)
    }

    fn synthesize_with_model_ref(
        &self,
        text: &str,
        cfg: &GepardConfig,
        model_path: &Path,
        ref_codes: Option<&[u32]>,
    ) -> Result<Vec<f32>> {
        if ref_codes.is_some() {
            return self.synthesize_eager_ref(text, cfg, model_path, ref_codes);
        }
        if let Some(session) = &self.compiled_session {
            return self.synthesize_compiled(text, cfg, model_path, session);
        }
        self.synthesize_eager_ref(text, cfg, model_path, ref_codes)
    }

    fn synthesize_eager_ref(
        &self,
        text: &str,
        cfg: &GepardConfig,
        model_path: &Path,
        ref_codes: Option<&[u32]>,
    ) -> Result<Vec<f32>> {
        let bytes = load_safetensors_bytes(model_path)?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse {}", model_path.display()))?;

        let overlay = GepardOverlay::load(&st, cfg.num_audio_heads())?;
        let backbone = BackboneWeights::load(&st, &cfg.backbone)?;
        let vocabs = cfg.codec.channel_vocabs();
        let hidden = cfg.hidden_size();

        // Tokenise and embed text (includes TextRepeater + SOS).
        let text_ids = self.tokenize(text)?;
        let text_embeds = backbone.embed_tokens(&text_ids);

        // Prefix only when reference codes are provided (MODEL_GUIDE §13.3 / §11.6).
        // Text-only generation omits null_prefix entirely.
        let (inputs, n_prefill) = if let Some(codes) = ref_codes {
            let speaker_prefix = if let Some(qformer) = &overlay.qformer {
                use crate::qformer::qformer_forward;
                let max_ref_frames = 256;
                let t_ref = (codes.len() / NUM_CHANNELS).min(max_ref_frames);
                let truncated = &codes[..t_ref * NUM_CHANNELS];
                eprintln!(
                    "[gepard] Q-Former prefix: {t_ref} frames (from {})",
                    codes.len() / NUM_CHANNELS
                );
                qformer_forward(truncated, t_ref, qformer, 1e-6)
            } else {
                eprintln!("[gepard] no Q-Former weights; falling back to null_prefix");
                overlay.null_prefix.clone().unwrap_or_default()
            };
            let n_prefix = speaker_prefix.len() / hidden;
            let mut v = speaker_prefix;
            v.extend_from_slice(&text_embeds);
            (v, n_prefix + text_ids.len())
        } else {
            (text_embeds, text_ids.len())
        };

        anyhow::ensure!(n_prefill > 0, "empty prefill (no text tokens)");

        let mut kv = GepardKvCache::new(cfg.backbone.num_hidden_layers);
        let all_h = backbone_prefill(&inputs, n_prefill, &backbone, &mut kv);

        // SOS hidden seeds frame 0 (MODEL_GUIDE §13.3).
        let mut h = all_h[(n_prefill - 1) * hidden..n_prefill * hidden].to_vec();

        let temp = if self.opts.greedy {
            0.0
        } else {
            self.opts.temperature
        };
        if !self.opts.greedy {
            fastrand::seed(self.opts.seed);
        }
        let max_frames = suggest_max_frames(text, self.opts.max_frames);
        let stop_th = self.opts.stop_threshold;
        let rep_pen = self.opts.repetition_penalty;
        let rep_win = self.opts.repetition_window;

        // Official runner: first frame from SOS hidden, then decode→stop→sample.
        let mut frames = Vec::with_capacity(max_frames.min(512));
        let first = sample_all_heads_temp(&h, &overlay, &vocabs, temp, rep_pen, None, rep_win);
        frames.push(first);

        for step in 1..max_frames {
            let frame_emb =
                embed_audio_frame(&frames[step - 1], &overlay, cfg.audio_embed_dim, hidden);
            h = backbone_decode_step(&frame_emb, &backbone, &mut kv);

            let stop_p = stop_probability(&h, &overlay);
            if stop_p > stop_th {
                break;
            }

            let recent = if rep_pen != 1.0 {
                Some(frames.as_slice())
            } else {
                None
            };
            let next = sample_all_heads_temp(&h, &overlay, &vocabs, temp, rep_pen, recent, rep_win);
            frames.push(next);
        }

        eprintln!(
            "[gepard] AR {} frames (max={max_frames} stop_th={stop_th} temp={temp} rep={rep_pen})",
            frames.len()
        );

        frames_to_audio_with_codec(&frames, self.gepard_decoder.as_ref())
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn synthesize_compiled(
        &self,
        text: &str,
        cfg: &GepardConfig,
        _model_path: &Path,
        session: &RefCell<GepardCompiledSession>,
    ) -> Result<Vec<f32>> {
        let overlay = &self.overlay;
        let vocabs = cfg.codec.channel_vocabs();
        let hidden = cfg.hidden_size();

        let text_ids = self.tokenize(text)?;
        let mut compiled = session.borrow_mut();
        let text_embeds = compiled.embed_tokens(&text_ids);

        let (inputs, n_prefill) = (text_embeds, text_ids.len());
        anyhow::ensure!(n_prefill > 0, "empty prefill (no text tokens)");

        let (mut h, mut cache) = compiled.prefill_hidden(&inputs, n_prefill)?;

        let temp = if self.opts.greedy {
            0.0
        } else {
            self.opts.temperature
        };
        if !self.opts.greedy {
            fastrand::seed(self.opts.seed);
        }
        // Stay inside compiled max_seq (prefill + frames).
        let seq_room = compiled
            .max_seq()
            .saturating_sub(n_prefill)
            .saturating_sub(8)
            .max(MIN_FRAMES);
        let max_frames = suggest_max_frames(text, self.opts.max_frames).min(seq_room);
        let stop_th = self.opts.stop_threshold;
        let rep_pen = self.opts.repetition_penalty;
        let rep_win = self.opts.repetition_window;

        let ar_t0 = Instant::now();
        let mut frames = Vec::with_capacity(max_frames.min(512));
        let first = sample_all_heads_temp(&h, overlay, &vocabs, temp, rep_pen, None, rep_win);
        frames.push(first);

        for step in 1..max_frames {
            let frame_emb =
                embed_audio_frame(&frames[step - 1], overlay, cfg.audio_embed_dim, hidden);
            h = compiled.decode_hidden(&mut cache, &frame_emb)?;

            let stop_p = stop_probability(&h, overlay);
            if stop_p > stop_th {
                break;
            }

            let recent = if rep_pen != 1.0 {
                Some(frames.as_slice())
            } else {
                None
            };
            let next = sample_all_heads_temp(&h, overlay, &vocabs, temp, rep_pen, recent, rep_win);
            frames.push(next);
        }
        let ar_ms = ar_t0.elapsed().as_secs_f64() * 1000.0;
        compiled.record_ar_timing(ar_ms, frames.len());

        let timing = compiled.last_timing.clone();
        eprintln!(
            "[gepard] compiled AR {} frames (max={max_frames} stop_th={stop_th} temp={temp} rep={rep_pen}) \
             prefill={:.0}ms ar={:.0}ms ({:.1} fps)",
            frames.len(),
            timing.prefill_ms,
            timing.ar_decode_ms,
            if timing.ar_decode_ms > 0.0 {
                frames.len() as f64 / (timing.ar_decode_ms / 1000.0)
            } else {
                0.0
            }
        );

        frames_to_audio_with_codec(&frames, self.gepard_decoder.as_ref())
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Prefill text region: HF tokenize + TextRepeater + SOS.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.build_prompt_ids(text)
    }

    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create {}", path.display()))?;
        for &s in audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn default_config() -> GepardConfig {
    serde_json::from_str(
        r#"{"backbone":{"hidden_size":1024,"num_hidden_layers":14,
        "num_attention_heads":8,"num_key_value_heads":2,
        "intermediate_size":2816,"vocab_size":152064}}"#,
    )
    .unwrap()
}

fn use_backend_ar(device: Device) -> bool {
    // Eager CPU AR override (debug / parity A/B).
    if matches!(
        std::env::var("RLX_GEPARD_EAGER_AR").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    ) {
        return false;
    }
    device != Device::Cpu || prefer_compiled()
}

fn prefer_compiled() -> bool {
    matches!(
        std::env::var("RLX_GEPARD_COMPILED").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn parse_device_name(s: &str) -> Device {
    match s.trim().to_ascii_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        _ => Device::Cpu,
    }
}

fn frames_to_audio_with_codec(
    frames: &[Vec<u32>],
    dec: Option<&GepardDecoder>,
) -> Result<Vec<f32>, String> {
    match dec {
        Some(d) => Ok(d.decode(frames)),
        None => Err(
            "Gepard NanoCodec decoder weights are required (place nano_dec*.safetensors under weights/tts/gepard); \
             sine-wave fallback has been removed"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    fn bundle() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/gepard")
    }

    #[test]
    fn test_embed_audio_frame_shape() {
        let num_ch = 32;
        let dim = 32;
        let hidden = 1024;
        let vocabs: Vec<u32> = crate::codec_ops::FSQ_LEVELS
            .iter()
            .cloned()
            .cycle()
            .take(num_ch)
            .collect();
        let overlay = GepardOverlay {
            audio_embeddings: vocabs
                .iter()
                .map(|&v| vec![0.0f32; v as usize * dim])
                .collect(),
            audio_proj_w0: vec![0.0; hidden * hidden],
            audio_proj_b0: vec![0.0; hidden],
            audio_proj_w2: vec![0.0; hidden * hidden],
            audio_proj_b2: vec![0.0; hidden],
            audio_embed_scale: 1.0,
            codebook_weights: vocabs
                .iter()
                .map(|&v| vec![0.0; v as usize * hidden])
                .collect(),
            codebook_biases: vocabs.iter().map(|&v| vec![0.0; v as usize]).collect(),
            stop_weight: vec![0.0; hidden],
            stop_bias: vec![0.0; 1],
            null_prefix: None,
            qformer: None,
        };
        let emb = embed_audio_frame(&vec![0u32; num_ch], &overlay, dim, hidden);
        assert_eq!(emb.len(), hidden);
    }

    #[test]
    fn test_synthesis_produces_audio() {
        let bundle = bundle();
        if !bundle.join("model.safetensors").is_file() || !bundle.join("tokenizer.json").is_file() {
            return;
        }
        let synth = GepardSynthesizer::new(&bundle).expect("open");
        let audio = synth.synthesize("Hello from Gepard.", "").unwrap();
        assert!(!audio.is_empty());
        assert!(audio.iter().any(|v| v.abs() > 1e-4));
    }

    #[test]
    fn test_synthesis_deterministic() {
        let bundle = bundle();
        if !bundle.join("model.safetensors").is_file() || !bundle.join("tokenizer.json").is_file() {
            return;
        }
        let synth = GepardSynthesizer::new(&bundle).expect("open");
        let a1 = synth.synthesize("same text", "").unwrap();
        let a2 = synth.synthesize("same text", "").unwrap();
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_tokenize_uses_hf_specials() {
        let bundle = bundle();
        if !bundle.join("tokenizer.json").is_file() {
            return;
        }
        let synth = GepardSynthesizer::new(&bundle).expect("open");
        let ids = synth.tokenize("Hi.").expect("tok");
        let sp = synth.config().special_tokens;
        assert_eq!(ids[0], sp.start_of_text);
        assert_eq!(*ids.last().unwrap(), sp.start_of_speech);
        assert!(ids.contains(&sp.end_of_text));
        // Must not be the old char-hash scheme.
        assert!(ids.iter().any(|&id| id > 1000));
    }

    #[test]
    fn test_synthesis_varies_with_text() {
        let bundle = bundle();
        if !bundle.join("model.safetensors").is_file() || !bundle.join("tokenizer.json").is_file() {
            return;
        }
        let synth = GepardSynthesizer::new(&bundle).expect("open");
        let a = synth.synthesize("Hello.", "").unwrap();
        let b = synth.synthesize("Different text here.", "").unwrap();
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    }
}
