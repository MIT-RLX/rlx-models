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

//! Native (RLX, no onnxruntime) MOSS-TTS-Nano synthesis. Three nested subgraphs
//! (`prefill`, `local_fixed_sampled_frame`, `moss_audio_tokenizer_decode_full`)
//! lower to rlx-ir. The between-frame global update reuses `prefill` on the
//! growing sequence padded to a fixed compile length (`attention_mask` masks the
//! pad; the last real position's `global_hidden` matches a KV-cached
//! `decode_step`) so nothing needs dynamic-shape KV.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_runtime::{CompiledGraph, DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use tokenizers::Tokenizer;

/// Map CLI device to the execution device (no silent remaps).
fn resolve_exec_device(requested: Device) -> Device {
    requested
}

use crate::config::Manifest;
use crate::dsp::{TightenOpts, tighten_pauses};

const HIDDEN: usize = 768;

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn as_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
/// Read a run_typed output that is *logically* integer but whose backing dtype
/// varies by backend: CPU returns true I32/I64, but the GPU backends (Metal /
/// MLX / wgpu) may materialize an integer graph output as F32. Convert per the
/// dtype the runtime actually returned so `sample_frame`'s codes are correct on
/// every device (a raw i32 reinterpret of an f32 buffer yields IEEE bit patterns
/// like 1123811328 for 126.0 — the cross-backend divergence this fixes).
fn as_i32_dyn(b: &[u8], dt: DType) -> Vec<i32> {
    match dt {
        DType::F32 => as_f32(b).iter().map(|&x| x.round() as i32).collect(),
        DType::I64 => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
            .collect(),
        _ => as_i32(b),
    }
}

/// SplitMix64 → uniform f32 in [0,1). Matches `model::MossNano`'s sampler RNG so
/// native and ORT paths draw the same sequence for a given seed.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Per-call synthesis options for the native path.
#[derive(Debug, Clone, Copy)]
pub struct NativeOpts {
    pub seed: u64,
    /// Hard cap on generated frames; also bounds the prefill compile length.
    pub max_frames: usize,
    /// Post-decode pause polish. `None` = defaults ([`TightenOpts::default`]);
    /// set `max_internal_pause_ms: 0` to only trim lead/trail.
    pub tighten: Option<TightenOpts>,
}
impl Default for NativeOpts {
    fn default() -> Self {
        Self {
            seed: 0,
            max_frames: 96,
            tighten: Some(TightenOpts::default()),
        }
    }
}

pub struct MossNative {
    moss: TinyModel,
    codec: TinyModel,
    local: std::sync::Mutex<rlx_runtime::CompiledGraph>,
    manifest: Manifest,
    tokenizer: Tokenizer,
    device: Device,
}

impl MossNative {
    /// Load on the CPU backend (see `load_on` for other devices).
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    /// Prefer `moss-nano.rlxp`, then legacy `moss-nano.gguf`, else a materialized dir.
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        crate::gguf_bundle::open_path(dir, device)
    }

    /// Load from an already-materialized directory (`graphs/*.rlxp` or legacy ONNX).
    pub fn load_loose(dir: &Path, device: Device) -> Result<Self> {
        let device = resolve_exec_device(rlx_tiny_tts::resolve_tts_device(device));
        let cfg = || BundleConfig {
            model: String::new(),
            sample_rate: 48000,
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
        let moss = TinyModel::new(dir.to_path_buf(), cfg());
        // Codec graph lives under top-level `graphs/` in native packs (and under
        // `codec/` for legacy ONNX). Prefer the shared root so one resolve path works.
        let codec_root = if dir
            .join("graphs/moss_audio_tokenizer_decode_full.rlxp")
            .is_file()
        {
            dir.to_path_buf()
        } else {
            dir.join("codec")
        };
        let codec = TinyModel::new(codec_root, cfg());
        let manifest = Manifest::load(&dir.join("browser_poc_manifest.json"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("open tokenizer.json: {e}"))?;
        // The per-frame sampler (`fixed_sampled_frame`) selects discrete codes by
        // argmax / inverse-CDF over the local-decoder logits. That is exquisitely
        // sensitive to floating-point order: a sub-ULP logit difference flips the
        // sampled token, and GPU kernels are not bit-identical to CPU (Metal
        // happens to match; MLX/wgpu need not). Since it runs on the (host-side)
        // prefill hidden state and is cheap relative to the growing re-prefill, we
        // pin it to CPU so the sampled code stream — hence the whole render — is
        // bit-identical on every backend. The expensive prefill + codec stay on
        // `device` (except Metal prefill — see `component_device`).
        let sampler_dev = match std::env::var("RLX_MOSS_SAMPLER_DEVICE").as_deref() {
            Ok("gpu") | Ok("device") => device,
            _ => Device::Cpu,
        };
        let local = moss
            .compile_named(
                "moss_tts_local_fixed_sampled_frame",
                sampler_dev,
                1,
                &[("batch", 1)],
            )
            .map_err(|e| anyhow::anyhow!("compile local: {e:#}"))?;
        Ok(Self {
            moss,
            codec,
            local: std::sync::Mutex::new(local),
            manifest,
            tokenizer,
            device,
        })
    }

    /// Device for a named graph. Metal/wgpu f32-uniform prefill drifts enough to
    /// flip AR codes — keep those on CPU. CUDA/ROCm prefill runs on-device (arena
    /// empty-tensor + Expand-zero fixes); override with `RLX_MOSS_PREFILL_DEVICE`.
    /// CUDA/ROCm codec stays on CPU by default (hang/NaN historically); force with
    /// `RLX_MOSS_CODEC_DEVICE=gpu`.
    fn component_device(&self, component: &str) -> Device {
        if component == "moss_tts_prefill" {
            match std::env::var("RLX_MOSS_PREFILL_DEVICE").as_deref() {
                Ok("gpu") | Ok("device") => return self.device,
                Ok("cpu") => return Device::Cpu,
                _ => {}
            }
            if matches!(self.device, Device::Metal | Device::Gpu) {
                return Device::Cpu;
            }
        }
        if component == "moss_audio_tokenizer_decode_full" {
            match std::env::var("RLX_MOSS_CODEC_DEVICE").as_deref() {
                Ok("gpu") | Ok("device") => return self.device,
                Ok("cpu") => return Device::Cpu,
                _ => {}
            }
            if matches!(self.device, Device::Cuda | Device::Rocm) {
                return Device::Cpu;
            }
        }
        self.device
    }

    pub fn voice_names(&self) -> Vec<String> {
        self.manifest.voice_names()
    }
    /// The prompt (reference) audio codes for a builtin `voice` — the conditioning
    /// codes fed into `generate_codes`.
    pub fn voice_prompt_codes(&self, voice: &str) -> Result<Vec<Vec<i32>>> {
        let v = self
            .manifest
            .voice(voice)
            .with_context(|| format!("unknown voice '{voice}'; have {:?}", self.voice_names()))?;
        Ok(v.prompt_audio_codes.clone())
    }
    pub fn sample_rate(&self) -> u32 {
        48000
    }
    pub fn channels(&self) -> u16 {
        2
    }
    /// Resolved execution device (after Vulkan remaps, etc.).
    pub fn device(&self) -> Device {
        self.device
    }

    fn encode_text(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect())
    }

    /// Build the `[seq, row_width]` prompt rows (flat) — mirrors `model::MossNano`.
    fn build_input_ids(&self, text: &str, voice_codes: &[Vec<i32>]) -> Result<Vec<i32>> {
        let c = &self.manifest.tts_config;
        let t = &self.manifest.prompt_templates;
        let rw = c.row_width();
        let pad = c.audio_pad_token_id;
        let mut rows: Vec<i32> = Vec::new();
        let push_text = |tok: i32, rows: &mut Vec<i32>| {
            rows.push(tok);
            rows.extend(std::iter::repeat_n(pad, rw - 1));
        };
        for &tok in &t.user_prompt_prefix_token_ids {
            push_text(tok, &mut rows);
        }
        push_text(c.audio_start_token_id, &mut rows);
        for frame in voice_codes {
            anyhow::ensure!(
                frame.len() == c.n_vq,
                "voice frame width {} != n_vq {}",
                frame.len(),
                c.n_vq
            );
            rows.push(c.audio_user_slot_token_id);
            rows.extend_from_slice(frame);
        }
        push_text(c.audio_end_token_id, &mut rows);
        for &tok in &t.user_prompt_after_reference_token_ids {
            push_text(tok, &mut rows);
        }
        for tok in self.encode_text(text)? {
            push_text(tok, &mut rows);
        }
        for &tok in &t.assistant_prompt_prefix_token_ids {
            push_text(tok, &mut rows);
        }
        push_text(c.audio_start_token_id, &mut rows);
        Ok(rows)
    }

    /// Run `prefill` (compiled at `max_seq`) on `rows` (flat `[seq*rw]`) padded to
    /// `max_seq`; return the last real row's `global_hidden` (`[HIDDEN]`).
    /// One fused sampled frame → (should_continue, 16 codebook tokens).
    fn sample_frame(
        &self,
        hidden: &[f32],
        rep_mask: &[i32],
        rng: &mut Rng,
    ) -> Result<(bool, Vec<i32>)> {
        let nvq = self.manifest.tts_config.n_vq;
        let aru = vec![rng.uniform()];
        let aut: Vec<f32> = (0..nvq).map(|_| rng.uniform()).collect();
        let mut g = self.local.lock().unwrap();
        let out = g.run_typed(&[
            ("global_hidden", &f32_le(hidden), DType::F32),
            ("repetition_seen_mask", &i32_le(rep_mask), DType::I32),
            ("assistant_random_u", &f32_le(&aru), DType::F32),
            ("audio_random_u", &f32_le(&aut), DType::F32),
        ]);
        let should = as_i32_dyn(&out[0].0, out[0].1)[0] != 0;
        let frame = as_i32_dyn(&out[1].0, out[1].1);
        Ok((should, frame))
    }

    /// Round sequence length up to a compile bucket so we re-run prefill on a
    /// small pad early in the utterance instead of always paying for
    /// `prompt + max_frames`. AotCache keys on the exact bucket, so each size
    /// compiles once. Without this, Trump (~100 audio rows) + 96 max frames
    /// re-runs a ~200-token transformer every frame → multi-minute fox synth.
    fn prefill_bucket(seq: usize) -> usize {
        const BUCKET: usize = 32;
        seq.max(1).div_ceil(BUCKET) * BUCKET
    }

    /// Run `prefill` (compiled at `max_seq`) on `rows` padded to `max_seq`; return the
    /// last real row's `global_hidden`. `prefill`'s `attention_mask` masks the pad, so
    /// the last real position is bit-identical to a KV-cached step (verified cos 1.0).
    fn prefill_last_hidden(
        &self,
        g: &mut CompiledGraph,
        max_seq: usize,
        rows: &[i32],
    ) -> Result<Vec<f32>> {
        let rw = self.manifest.tts_config.row_width();
        let pad = self.manifest.tts_config.audio_pad_token_id;
        let seq = rows.len() / rw;
        anyhow::ensure!(
            seq <= max_seq,
            "sequence {seq} exceeds compiled max {max_seq}"
        );
        let mut ids = vec![pad; max_seq * rw];
        ids[..rows.len()].copy_from_slice(rows);
        let mut mask = vec![0i32; max_seq];
        for m in mask.iter_mut().take(seq) {
            *m = 1;
        }
        let out = g.run_typed(&[
            ("input_ids", &i32_le(&ids), DType::I32),
            ("attention_mask", &i32_le(&mask), DType::I32),
        ]);
        let hidden = as_f32(&out[0].0);
        let last = seq - 1;
        Ok(hidden[last * HIDDEN..(last + 1) * HIDDEN].to_vec())
    }

    fn compile_prefill(&self, max_seq: usize) -> Result<CompiledGraph> {
        self.moss
            .compile_named(
                "moss_tts_prefill",
                self.component_device("moss_tts_prefill"),
                max_seq,
                &[("batch", 1), ("prefill_seq", max_seq)],
            )
            .map_err(|e| anyhow::anyhow!("compile prefill@{max_seq}: {e:#}"))
    }

    /// Debug: the `global_hidden` of the last prompt row after `prefill`
    /// (before any frame is sampled). Used for cross-backend parity isolation.
    pub fn debug_prefill_hidden(
        &self,
        text: &str,
        voice_codes: &[Vec<i32>],
        max_frames: usize,
    ) -> Result<Vec<f32>> {
        let rw = self.manifest.tts_config.row_width();
        let rows = self.build_input_ids(text, voice_codes)?;
        let prompt_seq = rows.len() / rw;
        let max_seq = prompt_seq + max_frames;
        let mut prefill = self.compile_prefill(max_seq)?;
        self.prefill_last_hidden(&mut prefill, max_seq, &rows)
    }

    /// Generate `[n_frames][n_vq]` audio codes. Prefill is recompiled on 32-token
    /// buckets as the sequence grows (see `Self::prefill_bucket`) and re-run each
    /// frame — the `decode_step` KV cache can't substitute: it does NOT mask past
    /// keys (past_valid_lengths only sets the new token's RoPE index; verified in
    /// ORT too), so a fixed-shape padded KV would attend to the zero-pad and diverge.
    pub fn generate_codes(
        &self,
        text: &str,
        voice_codes: &[Vec<i32>],
        opts: &NativeOpts,
    ) -> Result<Vec<Vec<i32>>> {
        let c = self.manifest.tts_config.clone();
        let rw = c.row_width();
        let mut rows = self.build_input_ids(text, voice_codes)?;
        let prompt_seq = rows.len() / rw;
        let hard_cap = prompt_seq + opts.max_frames;
        let mut bucket = Self::prefill_bucket(prompt_seq + 1).min(hard_cap);
        let mut prefill = self.compile_prefill(bucket)?;
        let mut rng = Rng::new(opts.seed);
        let mut rep_mask = vec![0i32; c.n_vq * 1024];
        let mut frames: Vec<Vec<i32>> = Vec::new();
        for _ in 0..opts.max_frames {
            let seq = rows.len() / rw;
            let need = Self::prefill_bucket(seq).min(hard_cap);
            if need != bucket {
                bucket = need;
                prefill = self.compile_prefill(bucket)?;
            }
            let hidden = self.prefill_last_hidden(&mut prefill, bucket, &rows)?;
            let (should, frame) = self.sample_frame(&hidden, &rep_mask, &mut rng)?;
            if !should {
                break;
            }
            for (ch, &code) in frame.iter().enumerate() {
                if (code as usize) < 1024 {
                    rep_mask[ch * 1024 + code as usize] = 1;
                }
            }
            rows.push(c.audio_assistant_slot_token_id);
            rows.extend_from_slice(&frame);
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Decode `[n_frames][n_vq]` codes → interleaved-stereo f32 @ 48 kHz.
    pub fn decode_codes(&self, frames: &[Vec<i32>]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let n = frames.len();
        let nq = self.manifest.tts_config.n_vq;
        let flat: Vec<i32> = frames.iter().flatten().copied().collect();
        let mut g = self
            .codec
            .compile_named(
                "moss_audio_tokenizer_decode_full",
                self.component_device("moss_audio_tokenizer_decode_full"),
                n,
                &[("batch", 1), ("code_length", n)],
            )
            .map_err(|e| anyhow::anyhow!("compile codec: {e:#}"))?;
        let out = g.run_typed(&[
            ("audio_codes", &i32_le(&flat), DType::I32),
            ("audio_code_lengths", &i32_le(&[n as i32]), DType::I32),
        ]);
        let audio = as_f32(&out[0].0); // [ch, samples] flattened (batch 1)
        let ch = self.channels() as usize;
        let samples = audio.len() / ch;
        let mut inter = vec![0.0f32; audio.len()];
        for c in 0..ch {
            for i in 0..samples {
                inter[i * ch + c] = audio[c * samples + i];
            }
        }
        let _ = nq;
        Ok(inter)
    }

    /// Synthesize `text` in a builtin `voice` → interleaved-stereo f32 @ 48 kHz.
    pub fn synthesize(&self, text: &str, voice: &str, opts: &NativeOpts) -> Result<Vec<f32>> {
        let v = self
            .manifest
            .voice(voice)
            .with_context(|| format!("unknown voice '{voice}'; have {:?}", self.voice_names()))?;
        let codes = self.generate_codes(text, &v.prompt_audio_codes.clone(), opts)?;
        let mut audio = self.decode_codes(&codes)?;
        // Peak-limit to avoid hard-clipping in the 16-bit WAV: the codec (bit-exact
        // vs onnxruntime) legitimately peaks above 1.0, so scale the whole clip down
        // to a 0.98 ceiling rather than clamping individual samples (which distorts).
        let peak = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if peak > 0.98 {
            let g = 0.98 / peak;
            for s in &mut audio {
                *s *= g;
            }
        }
        // Trim lead/trail silence and clamp long mid-utterance holes (MOSS AR
        // often inserts 0.5–0.8 s pauses that Whisper still hears through but
        // feel like stumbles to a listener).
        if let Some(topts) = opts.tighten {
            audio = tighten_pauses(&audio, self.sample_rate(), self.channels() as usize, topts);
        }
        Ok(audio)
    }

    /// Write interleaved-stereo f32 to a 16-bit WAV.
    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channels(),
            sample_rate: self.sample_rate(),
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

/// Default weights dir helper.
pub fn default_dir() -> PathBuf {
    PathBuf::from(crate::DEFAULT_LOCAL_DIR)
}
