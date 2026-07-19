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

//! Kokoro-82M runner over RLX (graph-split encoder + decoder).
//!
//! Kokoro's exported graph is monolithic and has a data-dependent length
//! regulator (`NonZero`/`Range` alignment) + an ISTFT (`NonZero`/`ScatterND`
//! overlap-add) that don't fit a single static-shape RLX compile. So the model
//! is **graph-split** into two fixed-shape subgraphs with the dynamic pieces
//! done in Rust:
//!
//! ```text
//! encoder.onnx      [input_ids, style, speed]
//!                 → prosody [1,640,seq]  (/encoder/Transpose_output_0)
//!                 → text    [1,512,seq]  (/encoder/text_encoder/Transpose_2_output_0)
//!                 → dur     [1,seq] i64  (/encoder/Cast_output_0)
//!   ── Rust length regulator: repeat_interleave columns by `dur` ──
//!                 → en  [1,640,frames]   asr [1,512,frames]   (frames = Σ dur)
//! decoder_raw.onnx  [en, asr, style] → raw waveform (pre-ISTFT-normalization)
//!   ── Rust ISTFT overlap-add normalization: edge-divide by `window_sum`,
//!      ×(n_fft/hop), crop n_fft/2 each end ──
//!                 → waveform [24 kHz mono]
//! ```
//!
//! With the `onnx` feature (default), the encoder prefers onnxruntime on CPU
//! (fast duration/prosody path); the decoder stays on RLX. Force the RLX
//! encoder with `RLX_KOKORO_NATIVE_ENC=1` (Whisper-gated fox 6/6 on CPU).
//!
//! The split bundle (`encoder.onnx`, `decoder_raw.onnx`, `window_sum.f32`) is
//! produced by `scripts/split_kokoro.py`. Decoder (and optional RLX encoder)
//! import through `rlx-onnx-import` → rlx-ir via rlx-tiny-tts's
//! `compile_named`/`run_typed` harness.

use std::path::{Path, PathBuf};
#[cfg(feature = "onnx")]
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_runtime::{CompileOptions, DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

#[cfg(feature = "onnx")]
use ort::session::Session;
#[cfg(feature = "onnx")]
use ort::value::Tensor;

use crate::config::{ModelLayout, SAMPLE_RATE};
use crate::model::{MIN_AUDIBLE_PEAK, peak_amplitude};
use crate::tokenize::Vocab;
use crate::voices::VoiceBank;

/// Graph names inside the split bundle (`<name>.onnx`).
const ENCODER: &str = "encoder";
const DECODER_RAW: &str = "decoder_raw";

/// Subdirectory (under the model's `onnx/`) holding the native split bundle.
pub const SPLIT_SUBDIR: &str = "rlx-split";

/// ISTFT overlap-add: n_fft/hop overlap factor and the crop of n_fft/2 each end.
const OVERLAP_FACTOR: f32 = 4.0; // n_fft(20) / hop(5)
const ISTFT_CROP: usize = 10; // n_fft / 2

/// Native Kokoro runner: graph-split subgraphs on an RLX backend, plus optional
/// ORT encoder when the `onnx` feature is enabled.
///
/// With `onnx`, the encoder defaults to ORT CPU (cheap; bit-close to the
/// export) while the heavy decoder stays on the requested RLX device. Force
/// the RLX encoder with `RLX_KOKORO_NATIVE_ENC=1` (also Whisper-valid).
pub struct NativeKokoro {
    model: TinyModel,
    /// ISTFT overlap-add normalization window (`window_sum`, len n_fft=20).
    window_sum: Vec<f32>,
    vocab: Vocab,
    voices: VoiceBank,
    device: Device,
    /// Device for the RLX encoder graph (when not using ORT encoder).
    enc_device: Device,
    #[cfg(feature = "onnx")]
    ort_encoder: Option<Mutex<Session>>,
}

/// Map a CLI/requested device onto the device graphs actually run on.
///
/// Encoder stays on CPU by default (duration rounding); decoder runs on
/// `requested`. Delegates CoreML unit pinning to
/// [`rlx_tiny_tts::resolve_tts_device`].
pub fn resolve_native_device(requested: Device) -> Device {
    rlx_tiny_tts::resolve_tts_device(requested)
}

impl NativeKokoro {
    /// Load a Kokoro model directory for native inference on `device`. Expects
    /// the same layout as the ort path (`onnx/model.onnx`, `tokenizer.json`,
    /// `voices/`) **plus** the split bundle under `onnx/<SPLIT_SUBDIR>/`
    /// (`encoder.onnx`, `decoder_raw.onnx`, `window_sum.f32`) produced by
    /// `scripts/split_kokoro.py`.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let layout = ModelLayout::resolve(model_dir, "model.onnx")?;
        let split_dir = layout.onnx.parent().unwrap_or(model_dir).join(SPLIT_SUBDIR);
        let vocab = Vocab::load(&layout.tokenizer)?;
        let voices = VoiceBank::load_dir(&layout.voices_dir)?;
        Self::load_split(&split_dir, vocab, voices, resolve_native_device(device))
    }

    /// Directory-based convenience: `load(dir, Cpu)`.
    pub fn load_from_dir(model_dir: &Path) -> Result<Self> {
        Self::load(model_dir, Device::Cpu)
    }

    /// Load directly from an explicit split-bundle directory + preloaded
    /// vocab/voices (for custom layouts).
    pub fn load_split(
        split_dir: &Path,
        vocab: Vocab,
        voices: VoiceBank,
        device: Device,
    ) -> Result<Self> {
        let dir = split_dir.to_path_buf();
        for f in [
            format!("{ENCODER}.onnx"),
            format!("{DECODER_RAW}.onnx"),
            "window_sum.f32".to_string(),
        ] {
            anyhow::ensure!(
                dir.join(&f).is_file(),
                "native Kokoro bundle missing {f} in {} (run scripts/split_kokoro.py)",
                dir.display()
            );
        }
        let window_sum = read_f32(&dir.join("window_sum.f32"))?;
        // BundleConfig is only used by the generic harness for VITS-specific glue
        // we don't exercise here; a placeholder with the right sample rate suffices.
        let cfg = BundleConfig {
            model: String::new(),
            sample_rate: SAMPLE_RATE,
            add_blank: false,
            language: "EN".into(),
            speakers: Default::default(),
            default_speaker: None,
            noise_scale: 0.0,
            noise_scale_w: 0.0,
            length_scale: 1.0,
            inter_channels: 0,
            gin_channels: 0,
        };
        let enc_device = match std::env::var("RLX_KOKORO_ENC_DEVICE").as_deref() {
            Ok("gpu") | Ok("device") => device,
            _ => Device::Cpu,
        };
        #[cfg(feature = "onnx")]
        let ort_encoder = {
            let force_native = matches!(
                std::env::var("RLX_KOKORO_NATIVE_ENC").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            );
            if force_native {
                None
            } else {
                let enc_path = dir.join(format!("{ENCODER}.onnx"));
                match rlx_kittentts::build_onnx_session(&enc_path, Device::Cpu) {
                    Ok(b) => {
                        eprintln!(
                            "[kokoro] encoder on onnxruntime/{} (decoder on rlx/{device:?}); \
                             set RLX_KOKORO_NATIVE_ENC=1 for full-native",
                            b.ort_ep
                        );
                        Some(Mutex::new(b.session))
                    }
                    Err(e) => {
                        eprintln!(
                            "[kokoro] ORT encoder unavailable ({e:#}); using RLX encoder on {enc_device:?}"
                        );
                        None
                    }
                }
            }
        };
        Ok(Self {
            model: TinyModel::new(dir, cfg),
            window_sum,
            vocab,
            voices,
            device,
            enc_device,
            #[cfg(feature = "onnx")]
            ort_encoder,
        })
    }

    /// Execution device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Model sample rate (24 kHz).
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Sorted list of available voice names.
    pub fn voice_names(&self) -> Vec<String> {
        self.voices.names()
    }

    /// The phoneme vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Synthesize from a phoneme (IPA) string using a named voice — the native
    /// analogue of [`crate::Kokoro::infer_phonemes`].
    pub fn infer_phonemes(&self, phonemes: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let voice_data = self.voices.get(voice).with_context(|| {
            format!(
                "voice '{voice}' not found. Available: {:?}",
                self.voice_names()
            )
        })?;
        let content_len = self.vocab.content_len(phonemes);
        anyhow::ensure!(
            content_len > 0,
            "phoneme input produced no in-vocabulary tokens: {phonemes:?}"
        );
        let ids = self.vocab.to_input_ids(phonemes);
        let style = voice_data.style_row(content_len).to_vec();
        let audio = self.infer(&ids, &style, speed)?;
        let peak = peak_amplitude(&audio);
        anyhow::ensure!(
            peak >= MIN_AUDIBLE_PEAK,
            "synthesized audio is silent (peak={peak:.2e})"
        );
        Ok(audio)
    }

    /// Synthesize from plain text (espeak-ng G2P). The espeak language is derived
    /// from the voice prefix. Native analogue of
    /// [`crate::Kokoro::generate_from_text`].
    #[cfg(feature = "espeak")]
    pub fn generate_from_text(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let lang = crate::config::voice_lang(voice);
        let processed = rlx_kittentts::preprocess::TextPreprocessor::new().process(text);
        let ipa = rlx_kittentts::phonemize::phonemize_lang(lang, &processed)
            .or_else(|_| rlx_kittentts::phonemize::phonemize_lang("en-us", &processed))
            .context("espeak phonemize failed")?;
        self.infer_phonemes(&ipa, voice, speed)
    }

    /// Run the full native pipeline: phoneme ids + style → 24 kHz waveform.
    ///
    /// `style` is the 256-d `ref_s` voice row; `speed` scales duration
    /// (`>1.0` faster). Mirrors the ort path bit-for-bit at the graph boundaries.
    pub fn infer(&self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        anyhow::ensure!(!ids.is_empty(), "empty phoneme id sequence");
        let seq = ids.len();

        // ── 1. Encoder → prosody / text / durations ─────────────────────────
        let (prosody, text, dur) = self.encode(ids, style, speed)?;
        anyhow::ensure!(dur.len() == seq, "durations len {} != seq {seq}", dur.len());
        if let Ok(dir) = std::env::var("RLX_KOK_DUMP") {
            let _ = std::fs::write(format!("{dir}/prosody.f32"), f32_bytes(&prosody));
            let _ = std::fs::write(format!("{dir}/text.f32"), f32_bytes(&text));
            let _ = std::fs::write(
                format!("{dir}/dur.i64"),
                dur.iter()
                    .flat_map(|&d| (d as i64).to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            eprintln!(
                "[dump] prosody={} text={} dur={:?}",
                prosody.len(),
                text.len(),
                dur
            );
        }

        // ── 2. Rust length regulator: repeat_interleave columns by dur ──────
        let frames: usize = dur.iter().sum();
        anyhow::ensure!(frames > 0, "total predicted duration is zero");
        let en = length_regulate(&prosody, 640, seq, &dur, frames);
        let asr = length_regulate(&text, 512, seq, &dur, frames);

        // ── 3. Decoder graph → raw (pre-ISTFT-normalization) waveform ───────
        // MPSGraph produces a corrupt result for this decoder's repeated
        // channel-normalization sequence. The Metal thunk schedule is correct.
        let mut dec_opts = CompileOptions::default();
        dec_opts.disable_mpsgraph = matches!(self.device, Device::Metal);
        let mut dec = self
            .model
            .compile_named_with_options(
                DECODER_RAW,
                self.device,
                frames,
                &[("unk__357", 1), ("unk__368", frames)],
                dec_opts,
            )
            .context("compile Kokoro decoder_raw graph")?;
        let dec_out = dec.run_typed(&[
            ("/encoder/MatMul_output_0", &f32_bytes(&en), DType::F32),
            ("/encoder/MatMul_1_output_0", &f32_bytes(&asr), DType::F32),
            ("style", &f32_bytes(style), DType::F32),
        ]);
        anyhow::ensure!(!dec_out.is_empty(), "decoder produced no output");
        if let Ok(dir) = std::env::var("RLX_KOK_DUMP") {
            for (i, (bytes, dtype)) in dec_out.iter().enumerate() {
                let path = format!("{dir}/decoder_{i:03}_{dtype:?}.bin");
                std::fs::write(&path, bytes)
                    .with_context(|| format!("write decoder output tap {path}"))?;
                if *dtype == DType::F32 {
                    let values = as_f32(bytes);
                    let peak = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
                    eprintln!("[dump] decoder[{i}] f32={} peak={peak:.6e}", values.len());
                } else {
                    eprintln!("[dump] decoder[{i}] {dtype:?}={} bytes", bytes.len());
                }
            }
        }
        let raw = as_f32(&dec_out[0].0);
        // Metal (even with MPSGraph disabled) can emit sparse ±Inf in the
        // ISTFTNet generator on longer utterances. Replace non-finite samples
        // before overlap-add so the waveform stays audible and Whisper-valid.
        let raw: Vec<f32> = raw
            .into_iter()
            .map(|v| if v.is_finite() { v } else { 0.0 })
            .collect();

        // ── 4. Rust ISTFT overlap-add normalization + soft peak limit ──────
        let mut wav = istft_normalize(&raw, &self.window_sum);
        soft_peak_limit(&mut wav, 0.95);
        Ok(wav)
    }

    fn encode(
        &self,
        ids: &[i64],
        style: &[f32],
        speed: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>)> {
        #[cfg(feature = "onnx")]
        if let Some(enc) = &self.ort_encoder {
            return self.encode_ort(enc, ids, style, speed);
        }
        self.encode_rlx(ids, style, speed)
    }

    fn encode_rlx(
        &self,
        ids: &[i64],
        style: &[f32],
        speed: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>)> {
        let seq = ids.len();
        let mut enc = self
            .model
            .compile_named(ENCODER, self.enc_device, seq, &[("sequence_length", seq)])
            .context("compile Kokoro encoder graph")?;
        let enc_out = enc.run_typed(&[
            ("input_ids", &i64_bytes(ids), DType::I64),
            ("style", &f32_bytes(style), DType::F32),
            ("speed", &f32_bytes(&[speed]), DType::F32),
        ]);
        anyhow::ensure!(
            enc_out.len() >= 3,
            "encoder produced {} outputs (need 3)",
            enc_out.len()
        );
        let prosody = as_f32(&enc_out[0].0);
        let text = as_f32(&enc_out[1].0);
        let dur = read_usize_dyn(&enc_out[2].0, enc_out[2].1);
        Ok((prosody, text, dur))
    }

    #[cfg(feature = "onnx")]
    fn encode_ort(
        &self,
        enc: &Mutex<Session>,
        ids: &[i64],
        style: &[f32],
        speed: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>)> {
        let seq = ids.len();
        let mut session = enc.lock().unwrap_or_else(|e| e.into_inner());
        let ids_t =
            Tensor::from_array(([1usize, seq], ids.to_vec())).context("ort encoder input_ids")?;
        let style_t = Tensor::from_array(([1usize, 256], style.to_vec())).context("ort style")?;
        let speed_t = Tensor::from_array(([1usize], vec![speed])).context("ort speed")?;
        let outs = session
            .run(ort::inputs![
                "input_ids" => ids_t,
                "style" => style_t,
                "speed" => speed_t,
            ])
            .context("ort encoder run")?;
        let prosody = tensor_f32(&outs[0])?;
        let text = tensor_f32(&outs[1])?;
        let dur = tensor_usize(&outs[2])?;
        Ok((prosody, text, dur))
    }
}

/// Repeat each column of `x` (`[1, c, seq]`, row-major) `dur[i]` times along the
/// last axis → `[1, c, frames]`. This is the StyleTTS2 length regulator: the
/// `MatMul`-with-alignment-matrix upsample reduces to a per-token repeat.
fn length_regulate(x: &[f32], c: usize, seq: usize, dur: &[usize], frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * frames];
    for ch in 0..c {
        let src = &x[ch * seq..ch * seq + seq];
        let dst = &mut out[ch * frames..ch * frames + frames];
        let mut f = 0;
        for (i, &d) in dur.iter().enumerate() {
            let v = src[i];
            for _ in 0..d {
                dst[f] = v;
                f += 1;
            }
        }
    }
    out
}

/// ISTFT overlap-add normalization (the piece excluded from `decoder_raw`):
/// edge positions covered by a partial window are divided by `window_sum`;
/// scale by the n_fft/hop overlap factor; crop n_fft/2 samples each end.
/// Verified bit-exact vs onnxruntime's full ISTFT.
fn istft_normalize(raw: &[f32], window_sum: &[f32]) -> Vec<f32> {
    let l = raw.len();
    let mut scat = raw.to_vec();
    for (j, &w) in window_sum.iter().enumerate().take(l) {
        if w > 1.18e-38 {
            scat[j] = raw[j] / w;
        }
    }
    let end = l.saturating_sub(ISTFT_CROP);
    scat[ISTFT_CROP.min(l)..end.max(ISTFT_CROP.min(l))]
        .iter()
        .map(|&s| s * OVERLAP_FACTOR)
        .collect()
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    Ok(as_f32(&bytes))
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Read a logically-integer graph output as `usize`, respecting the dtype the
/// runtime actually returned. The duration output is I64 in the graph, but the
/// GPU backends (Metal/MLX/wgpu/CoreML) materialize it as F32 — reading raw
/// bytes as i64 then yields half the count of garbage values (the classic
/// "durations len 16 != seq 32"). Clamp negatives to 0 (rounded for f32).
fn read_usize_dyn(bytes: &[u8], dt: DType) -> Vec<usize> {
    match dt {
        DType::F32 => as_f32(bytes)
            .iter()
            .map(|&x| x.round().max(0.0) as usize)
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()).max(0) as usize)
            .collect(),
        _ => bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()).max(0) as usize)
            .collect(),
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Soft peak limit so ISTFT peaks above 1.0 do not hard-clip the WAV writer.
fn soft_peak_limit(wav: &mut [f32], ceiling: f32) {
    let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if peak > ceiling && peak.is_finite() && peak > 0.0 {
        let s = ceiling / peak;
        for x in wav.iter_mut() {
            *x *= s;
        }
    }
}

#[cfg(feature = "onnx")]
fn tensor_f32(v: &ort::value::DynValue) -> Result<Vec<f32>> {
    let (_shape, data) = v
        .try_extract_tensor::<f32>()
        .context("extract ort f32 tensor")?;
    Ok(data.to_vec())
}

#[cfg(feature = "onnx")]
fn tensor_usize(v: &ort::value::DynValue) -> Result<Vec<usize>> {
    if let Ok((_shape, data)) = v.try_extract_tensor::<i64>() {
        return Ok(data.iter().map(|&x| x.max(0) as usize).collect());
    }
    if let Ok((_shape, data)) = v.try_extract_tensor::<i32>() {
        return Ok(data.iter().map(|&x| x.max(0) as usize).collect());
    }
    if let Ok((_shape, data)) = v.try_extract_tensor::<f32>() {
        return Ok(data.iter().map(|&x| x.round().max(0.0) as usize).collect());
    }
    anyhow::bail!("duration tensor: unsupported ort dtype")
}

/// Default location for a native split bundle beside a model directory.
pub fn default_split_dir(model_dir: &Path) -> PathBuf {
    model_dir.join("rlx-split")
}
