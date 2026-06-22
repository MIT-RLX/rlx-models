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

//! Native RLX inference via [`kitten_tts_mini_rlx`] (Rust graph + safetensors/GGUF).

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use kitten_tts_mini_rlx::compile_profile::{
    InferMode, apply_device_runtime_defaults, infer_mode, prefer_metal_device, prewarm_buckets,
    prewarm_enabled, seq_compile_cache_capacity,
};
use kitten_tts_mini_rlx::{CachedSeqGraphs, native_weights_available};
use rlx_runtime::{DType, Device};

enum CompileBackend {
    Weights(kitten_tts_mini_rlx::NativeSeqCompileCache),
    Bundle(kitten_tts_mini_rlx::bundle_compile::SeqCompileCache),
}

/// Compiled Kitten graph with per-sequence compile cache (ORT-style `[1, seq]`).
pub struct NativeEngine {
    pub device: Device,
    backend: CompileBackend,
    pub sequence_length: usize,
    pub max_waveform_samples: usize,
    pub style_dim: usize,
    style_bytes: Mutex<Vec<u8>>,
    speed_bytes: Mutex<Vec<u8>>,
}

/// Crossfade when stitching native IPA chunks (10 ms at 24 kHz).
const NATIVE_CHUNK_CROSSFADE: usize = 240;

impl NativeEngine {
    pub fn load(
        weights_dir: &Path,
        device: Device,
        sequence_length: usize,
        max_waveform_samples: usize,
    ) -> Result<Self> {
        let weights_dir = weights_dir
            .canonicalize()
            .with_context(|| format!("resolve native weights dir {weights_dir:?}"))?;
        let device = prefer_metal_device(device);
        if !rlx_runtime::is_available(device) {
            anyhow::bail!(
                "native device {device:?} is not available in this build — rebuild with \
                 `cargo build -p rlx-kittentts --features native-fast,metal --release` \
                 (or pass `--device cpu`)"
            );
        }
        apply_device_runtime_defaults(device);

        let force_bundle = std::env::var("KITTEN_RLX_FORCE_BUNDLE").is_ok_and(|v| {
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        });
        let force_weights = std::env::var("KITTEN_RLX_FORCE_WEIGHTS").is_ok_and(|v| {
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        });
        let has_weights = native_weights_available(&weights_dir);
        let has_bundle =
            kitten_tts_mini_rlx::bundle_compile::bundle_dir_near_weights(&weights_dir).is_some();

        let cache_capacity = seq_compile_cache_capacity();
        let backend = if force_weights && has_weights {
            CompileBackend::Weights(kitten_tts_mini_rlx::NativeSeqCompileCache::new(
                device,
                weights_dir.clone(),
                sequence_length,
                max_waveform_samples,
                cache_capacity,
            )?)
        } else if force_bundle || (has_bundle && (!has_weights || !force_weights)) {
            let bundle = kitten_tts_mini_rlx::bundle_compile::bundle_dir_near_weights(&weights_dir)
                .with_context(|| {
                    format!("find rlx_bundle or native weights near {weights_dir:?}")
                })?;
            CompileBackend::Bundle(kitten_tts_mini_rlx::bundle_compile::SeqCompileCache::new(
                device,
                bundle,
                sequence_length,
                max_waveform_samples,
                cache_capacity,
            ))
        } else if has_weights {
            CompileBackend::Weights(kitten_tts_mini_rlx::NativeSeqCompileCache::new(
                device,
                weights_dir.clone(),
                sequence_length,
                max_waveform_samples,
                cache_capacity,
            )?)
        } else {
            let bundle = kitten_tts_mini_rlx::bundle_compile::bundle_dir_near_weights(&weights_dir)
                .with_context(|| {
                    format!("find rlx_bundle or native weights near {weights_dir:?}")
                })?;
            CompileBackend::Bundle(kitten_tts_mini_rlx::bundle_compile::SeqCompileCache::new(
                device,
                bundle,
                sequence_length,
                max_waveform_samples,
                cache_capacity,
            ))
        };

        if std::env::var("KITTEN_RLX_RNG_SEED").is_err()
            && std::env::var("KITTEN_RLX_PARITY").is_err()
        {
            kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_RNG_SEED", "42");
        }

        eprintln!(
            "[kittentts] native infer mode: {} (KITTEN_RLX_INFER=production|parity)",
            match infer_mode() {
                InferMode::Production => "production",
                InferMode::Parity => "parity",
            }
        );

        let engine = Self {
            device,
            backend,
            sequence_length,
            max_waveform_samples,
            style_dim: 256,
            style_bytes: Mutex::new(vec![0u8; 256 * 4]),
            speed_bytes: Mutex::new(1.0f32.to_le_bytes().to_vec()),
        };
        let load_t0 = std::time::Instant::now();
        engine.prewarm_default()?;
        if kitten_tts_mini_rlx::compile_profile::parity_thunk_profile_enabled() {
            let compile_seq =
                kitten_tts_mini_rlx::compile_profile::compile_slot_length(sequence_length);
            let graphs = engine.graphs_for_token_len(compile_seq.min(sequence_length))?;
            kitten_tts_mini_rlx::bundle_compile::run_parity_thunk_profile(
                &graphs,
                compile_seq.min(sequence_length),
                compile_seq,
            )?;
        }
        if kitten_tts_mini_rlx::compile_profile::env_flag("KITTEN_RLX_TIMING") {
            eprintln!(
                "[kittentts] native load+prewarm {:.3}s",
                load_t0.elapsed().as_secs_f64()
            );
        }
        Ok(engine)
    }

    fn prewarm_default(&self) -> Result<()> {
        if !prewarm_enabled() {
            return Ok(());
        }
        let buckets = prewarm_buckets(self.sequence_length);
        match &self.backend {
            CompileBackend::Weights(c) => c.prewarm(&buckets)?,
            CompileBackend::Bundle(c) => c.prewarm(&buckets)?,
        }
        Ok(())
    }

    fn graphs_for_token_len(&self, token_len: usize) -> Result<CachedSeqGraphs> {
        match &self.backend {
            CompileBackend::Weights(c) => c
                .cached_graphs_for_seq(token_len)
                .with_context(|| format!("compile native graph for seq={token_len}")),
            CompileBackend::Bundle(c) => c
                .cached_graphs_for_seq(token_len)
                .with_context(|| format!("compile bundle graph for seq={token_len}")),
        }
    }

    /// Like [`infer`](Self::infer), but supplies per-chunk ORT duration tensors aligned with
    /// [`infer_opts::chunk_padded_ids_with_offsets`] (isolated ORT runs per chunk).
    pub fn infer_with_chunk_ort_durations(
        &self,
        ids: &[i64],
        style: &[f32],
        speed: f32,
        chunk_durations: &[Vec<i64>],
    ) -> Result<Vec<f32>> {
        let chunks = crate::infer_opts::chunk_plan(ids, self.sequence_length);
        if chunks.len() != chunk_durations.len() {
            anyhow::bail!(
                "chunk duration count {} != chunk count {}",
                chunk_durations.len(),
                chunks.len()
            );
        }
        let exact_target = chunk_durations
            .iter()
            .try_fold(0usize, |acc, d| {
                crate::infer_opts::waveform_samples_from_duration(d, d.len())
                    .map(|n| acc.saturating_add(n))
            })
            .filter(|&n| n > 0);

        let mut audio = Vec::new();
        for ((chunk, _start), chunk_dur) in chunks.iter().zip(chunk_durations.iter()) {
            let part = self.infer_chunk(chunk, style, speed, Some(chunk_dur.as_slice()))?;
            audio.extend_from_slice(&part);
        }

        if let Some(target) = exact_target {
            if audio.len() > target {
                audio.truncate(target);
            }
        }
        Ok(audio)
    }

    /// Run the native graph: `input_ids` `[1, seq]`, `style` `[1, 256]`, `speed` `[1]`.
    ///
    /// When `ort_duration` is provided (full padded id sequence), duration carry is seeded
    /// from ORT and each chunk is trimmed to the exact ONNX sample count.
    pub fn infer(
        &self,
        ids: &[i64],
        style: &[f32],
        speed: f32,
        ort_duration: Option<&[i64]>,
    ) -> Result<Vec<f32>> {
        let chunks = crate::infer_opts::chunk_plan(ids, self.sequence_length);
        let exact_target = ort_duration
            .and_then(|d| crate::infer_opts::waveform_samples_from_duration(d, ids.len()));

        let mut audio = if chunks.len() == 1 {
            let (chunk, start) = &chunks[0];
            let chunk_dur =
                ort_duration.map(|d| crate::infer_opts::ort_duration_slice(d, *start, chunk.len()));
            self.infer_chunk(chunk, style, speed, chunk_dur.as_deref())?
        } else {
            let mut audio = Vec::new();
            for (chunk, start) in &chunks {
                let chunk_dur = ort_duration
                    .map(|d| crate::infer_opts::ort_duration_slice(d, *start, chunk.len()));
                let part = self.infer_chunk(chunk, style, speed, chunk_dur.as_deref())?;
                if ort_duration.is_some() {
                    audio.extend_from_slice(&part);
                } else {
                    crossfade_append(&mut audio, &part);
                }
            }
            audio
        };

        if let Some(target) = exact_target {
            if audio.len() > target {
                audio.truncate(target);
            }
        }
        Ok(audio)
    }

    fn infer_chunk(
        &self,
        ids: &[i64],
        style: &[f32],
        speed: f32,
        ort_duration: Option<&[i64]>,
    ) -> Result<Vec<f32>> {
        if ids.len() > self.sequence_length {
            anyhow::bail!(
                "IPA length {} exceeds compiled sequence_length {} (re-load with a larger --seq-len)",
                ids.len(),
                self.sequence_length
            );
        }
        if style.len() != self.style_dim {
            anyhow::bail!("style dim {} != expected {}", style.len(), self.style_dim);
        }

        let compile_seq = kitten_tts_mini_rlx::compile_profile::compile_slot_length(ids.len());
        let active_tokens = ids.len();
        let mut ids_padded: Vec<i64> = ids.to_vec();
        if compile_seq > ids.len() {
            ids_padded.resize(compile_seq, 0);
        }
        let ids_bytes: Vec<u8> = ids_padded.iter().flat_map(|v| v.to_le_bytes()).collect();

        {
            let mut style_bytes = self.style_bytes.lock().expect("style_bytes mutex");
            for (i, &v) in style.iter().enumerate() {
                style_bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        {
            let mut speed_bytes = self.speed_bytes.lock().expect("speed_bytes mutex");
            speed_bytes.copy_from_slice(&speed.to_le_bytes());
        }

        kitten_tts_mini_rlx::opts::set_compile_sequence_length(compile_seq);

        let style_bytes = self.style_bytes.lock().expect("style_bytes mutex");
        let speed_bytes = self.speed_bytes.lock().expect("speed_bytes mutex");

        let graphs = self.graphs_for_token_len(ids.len())?;

        kitten_tts_mini_rlx::bundle_compile::shape_all_graphs_for_infer(
            &graphs,
            active_tokens,
            compile_seq,
        )
        .context("runtime graph shapes")?;

        let infer_t0 = std::time::Instant::now();
        let inputs: [(&str, &[u8], DType); 3] = [
            ("input_ids", ids_bytes.as_slice(), DType::I64),
            ("style", style_bytes.as_slice(), DType::F32),
            ("speed", speed_bytes.as_slice(), DType::F32),
        ];

        let carry_seed = ort_duration.and_then(|d| {
            if crate::ort_duration::ort_duration_carry_seed_enabled(compile_seq) {
                Some(crate::infer_opts::duration_carry_bytes(d, compile_seq))
            } else {
                None
            }
        });
        let align_bytes =
            ort_duration.map(|d| crate::infer_opts::duration_carry_bytes(d, compile_seq));

        let outputs = kitten_tts_mini_rlx::bundle_compile::run_kitten_inference(
            &graphs,
            &inputs,
            carry_seed.as_deref(),
            align_bytes.as_deref(),
        );

        if kitten_tts_mini_rlx::compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
            if let Some((dur_bytes, dur_dt)) = outputs.get(1) {
                if *dur_dt == DType::I64 {
                    let dur = decode_i64_tensor(dur_bytes);
                    let sum: i64 = dur[..ids.len().min(dur.len())]
                        .iter()
                        .copied()
                        .filter(|&d| d > 0 && d < 10_000)
                        .sum();
                    eprintln!(
                        "[kittentts] duration active={:?} sum={sum} compile_seq={compile_seq}",
                        &dur[..ids.len().min(dur.len())]
                    );
                }
            }
            if let Some((wave_bytes, wave_dt)) = outputs.first() {
                eprintln!(
                    "[kittentts] waveform raw dtype={wave_dt:?} bytes={} samples={}",
                    wave_bytes.len(),
                    wave_bytes.len() / 4
                );
            }
        }

        if kitten_tts_mini_rlx::compile_profile::env_flag("KITTEN_RLX_TIMING") {
            eprintln!(
                "[kittentts] native infer {:.3}s (seq={})",
                infer_t0.elapsed().as_secs_f64(),
                ids.len()
            );
        }

        let (waveform_bytes, waveform_dt) = outputs
            .first()
            .context("native Kitten graph returned no outputs")?;
        if waveform_dt != &DType::F32 {
            anyhow::bail!("expected waveform f32, got {waveform_dt:?}");
        }

        let mut audio = decode_f32_waveform(waveform_bytes);
        if kitten_tts_mini_rlx::compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
            let raw_peak = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "[kittentts] waveform raw peak={raw_peak:.4e} len={} (before trim)",
                audio.len()
            );
        }
        if let Some(dur) = ort_duration {
            trim_waveform_exact(&mut audio, dur, ids.len());
        } else if let Some((dur_bytes, dur_dt)) = outputs.get(1) {
            if *dur_dt == DType::I64 {
                trim_waveform_from_duration(&mut audio, dur_bytes, ids.len());
            }
        }
        Ok(audio)
    }
}

/// Trim to exact ONNX length from a trusted duration slice (no inflated-sum guard).
fn trim_waveform_exact(audio: &mut Vec<f32>, duration: &[i64], token_len: usize) {
    let Some(len) = crate::infer_opts::waveform_samples_from_duration(duration, token_len) else {
        return;
    };
    if len < audio.len() {
        audio.truncate(len);
    }
}

fn decode_f32_waveform(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_i64_tensor(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("i64 chunk")))
        .collect()
}

/// ONNX emits `sum(duration[:seq]) * 600` waveform samples (24 kHz vocoder hop).
fn trim_waveform_from_duration(audio: &mut Vec<f32>, dur_bytes: &[u8], token_len: usize) -> bool {
    let duration = decode_i64_tensor(dur_bytes);
    let n = token_len.min(duration.len());
    let sum: i64 = duration[..n]
        .iter()
        .copied()
        .filter(|&d| d > 0 && d < 10_000)
        .sum();
    // Ignore unconverged / inflated duration from wide compile slots (ORT avg ~2–6 per token).
    const MAX_AVG_DURATION_PER_TOKEN: i64 = 12;
    if sum <= 0 || sum > token_len as i64 * MAX_AVG_DURATION_PER_TOKEN {
        return false;
    }
    let Some(len) = crate::infer_opts::waveform_samples_from_duration(&duration, token_len) else {
        return false;
    };
    if len < audio.len() {
        audio.truncate(len);
    }
    true
}

fn crossfade_append(dst: &mut Vec<f32>, chunk: &[f32]) {
    if dst.is_empty() {
        dst.extend_from_slice(chunk);
        return;
    }
    let fade = NATIVE_CHUNK_CROSSFADE.min(dst.len()).min(chunk.len());
    if fade == 0 {
        dst.extend_from_slice(chunk);
        return;
    }
    for i in 0..fade {
        let t = i as f32 / fade as f32;
        let tail_idx = dst.len() - fade + i;
        dst[tail_idx] = dst[tail_idx] * (1.0 - t) + chunk[i] * t;
    }
    dst.extend_from_slice(&chunk[fade..]);
}
