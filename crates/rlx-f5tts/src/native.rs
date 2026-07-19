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

//! Native RLX F5-TTS: the three ONNX graphs (`F5_Preprocess`, `F5_Transformer`,
//! `F5_Decode`) imported + compiled + run through rlx-tiny-tts — **no ONNX
//! Runtime at runtime**, on any RLX backend. Mirrors rlx-supertonic's multi-
//! subgraph pattern; f16 tensors flow between graphs as raw bytes (kept in f16
//! across the NFE loop — no per-step f16⇄f32 churn).

use std::path::Path;

use anyhow::{Context, Result};
use half::f16;
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

use crate::config::{DEFAULT_MAX_DURATION, HOP_LENGTH, Layout, Vocab};
use crate::dsp::{preprocess_ref_audio, soft_peak_limit};
use crate::model::InferOpts;
use crate::tokenize::{encode, normalize_ref_text, text_len};

/// `cat_mel_text` width = 100 mel + 512 text-embed. Binds the `text_embed_len`
/// dim_param of the preprocess/transformer graphs.
const TEXT_EMBED_LEN: usize = 612;

/// Native F5-TTS engine over RLX (ort-free).
pub struct F5Native {
    model: TinyModel,
    vocab: Vocab,
    device: Device,
}

/// Prefer the requested device. Preprocess/decode/DiT all stay on true Vulkan
/// when asked (`rlx-vulkan` activation striping). Optional DiT hybrid via
/// [`dit_exec_device`] (`RLX_F5_CUDA_DIT` / `RLX_F5_CPU_DIT`).
fn resolve_exec_device(requested: Device) -> Device {
    rlx_tiny_tts::resolve_tts_device(requested)
}

impl F5Native {
    /// Load from a model directory (resolves `F5_*.onnx` + `vocab.txt`).
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let device = resolve_exec_device(rlx_tiny_tts::resolve_tts_device(device));
        let layout = Layout::resolve(dir)?;
        let vocab = Vocab::load(&layout.dir)?;
        let cfg = BundleConfig {
            model: String::new(),
            sample_rate: crate::config::SAMPLE_RATE,
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
        Ok(Self {
            model: TinyModel::new(layout.dir, cfg),
            vocab,
            device,
        })
    }

    /// Device used for preprocess / decode (the caller-requested device).
    pub fn execution_device(&self) -> Device {
        self.device
    }

    /// Device used for the NFE DiT loop (may fall back to CPU — see
    /// [`dit_exec_device`]).
    pub fn dit_device(&self) -> Device {
        dit_exec_device(self.device)
    }

    pub fn sample_rate(&self) -> u32 {
        crate::config::SAMPLE_RATE
    }

    /// Compile (AOT-cached) one graph at the given seq length + named dims.
    fn compile(
        &self,
        comp: &'static str,
        device: Device,
        length: usize,
        named: &[(&str, usize)],
    ) -> Result<rlx_runtime::CompiledGraph> {
        self.model
            .compile_named(comp, device, length, named)
            .map_err(|e| anyhow::anyhow!("compile {comp} on {device:?}: {e:#}"))
    }

    /// Compile + run a one-shot graph, returning ALL outputs as raw `(bytes,
    /// dtype)` — f16 stays f16 so it feeds the next graph without conversion.
    fn run(
        &self,
        comp: &'static str,
        device: Device,
        length: usize,
        named: &[(&str, usize)],
        inputs: &[(&str, &[u8], DType)],
    ) -> Result<Vec<(Vec<u8>, DType)>> {
        Ok(self.compile(comp, device, length, named)?.run_typed(inputs))
    }

    /// Synthesize `gen_text` in the voice of `ref_audio`/`ref_text` (24 kHz mono).
    ///
    /// Long `gen_text` is split into sentence-sized chunks so each DiT compile
    /// stays under [`DEFAULT_MAX_DURATION`] mel frames (override with
    /// `RLX_F5_MAX_DURATION`). Chunks are concatenated with a short crossfade.
    pub fn synthesize(
        &self,
        gen_text: &str,
        ref_audio: &[f32],
        ref_text: &str,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        let ref_text_n = normalize_ref_text(ref_text);
        let ref_audio_p = preprocess_ref_audio(ref_audio, self.sample_rate());
        let n = ref_audio_p.len();
        let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
        let ref_tl = text_len(&ref_text_n).max(1) as f64;
        let cap = max_duration_cap();

        let chunks = chunk_gen_text(gen_text, ref_audio_len, ref_tl, opts.speed, cap);
        if chunks.len() > 1 {
            eprintln!(
                "[f5tts] chunking gen_text into {} parts (max_duration cap={cap})",
                chunks.len()
            );
        }
        let mut out = Vec::new();
        let n_chunks = chunks.len();
        for (i, chunk) in chunks.iter().enumerate() {
            let pcm = self.synthesize_one(chunk, &ref_audio_p, &ref_text_n, opts)?;
            let peak = pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            if peak < 1e-3 {
                // With the RAM-bounded `max_duration` cap, a long paragraph is
                // split into many comma-delimited fragments; a short one
                // occasionally denoises to ~silence. Skip it rather than fail the
                // whole utterance. A lone silent chunk (single-chunk synth) is a
                // real break and still errors.
                anyhow::ensure!(
                    n_chunks > 1,
                    "native F5 synthesized silence (peak={peak:.2e})"
                );
                eprintln!(
                    "[f5tts] chunk {}/{n_chunks} near-silent (peak={peak:.2e}); skipping",
                    i + 1
                );
                continue;
            }
            if out.is_empty() {
                out = pcm;
            } else {
                crossfade_append(&mut out, &pcm, (self.sample_rate() / 50) as usize); // ~20 ms
            }
        }
        anyhow::ensure!(
            !out.is_empty(),
            "native F5 synthesized silence (all {n_chunks} chunk(s) empty)"
        );
        Ok(out)
    }

    fn synthesize_one(
        &self,
        gen_text: &str,
        ref_audio: &[f32],
        ref_text: &str,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        // `ref_text` / `ref_audio` already normalized by [`synthesize`].
        let text_ids = encode(ref_text, gen_text, &self.vocab);
        anyhow::ensure!(!text_ids.is_empty(), "empty text");
        let n = ref_audio.len();

        // Reference duration estimate (ONNX export uses hop+1).
        let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
        let ref_tl = text_len(ref_text).max(1) as f64;
        let gen_tl = text_len(gen_text) as f64;
        let mut max_duration =
            (ref_audio_len + (ref_audio_len / ref_tl * gen_tl / opts.speed as f64)) as usize;
        let cap = max_duration_cap();
        if max_duration > cap {
            eprintln!(
                "[f5tts] clamping max_duration {max_duration} → {cap} (set RLX_F5_MAX_DURATION)"
            );
            max_duration = cap;
        }
        anyhow::ensure!(max_duration > 0, "zero duration");
        let d = max_duration;
        let t = text_ids.len();

        // ── 1. preprocess → 7 conditioning tensors ─────────────────────────────
        let (audio_b, audio_dt) = float_bytes(ref_audio, self.device);
        let text_ids_b: Vec<u8> = text_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let md_b = (max_duration as i64).to_le_bytes().to_vec();
        let pre_named = &[
            ("audio_len", n),
            ("text_ids_len", t),
            ("max_duration", d),
            ("text_embed_len", TEXT_EMBED_LEN),
        ];
        let pre = self.run(
            "F5_Preprocess",
            self.device,
            d,
            pre_named,
            &[
                ("audio", &audio_b, audio_dt),
                ("text_ids", &text_ids_b, DType::I32),
                ("max_duration", &md_b, DType::I64),
            ],
        )?;
        anyhow::ensure!(
            pre.len() >= 7,
            "preprocess expected 7 outputs, got {}",
            pre.len()
        );
        if std::env::var("RLX_F5_DBG").is_ok() {
            let names = [
                "noise",
                "rope_cos",
                "rope_sin",
                "cat_mel",
                "cat_mel_drop",
                "qk",
                "ref_len",
            ];
            for (i, (b, dt)) in pre.iter().enumerate() {
                let v = to_f32(b, *dt);
                let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                eprintln!(
                    "[f5-dbg] pre[{}]={} elems={} peak={pk:.4} first={:?}",
                    i,
                    names.get(i).unwrap_or(&"?"),
                    v.len(),
                    &v[..3.min(v.len())]
                );
            }
            eprintln!("[f5-dbg] d(max_duration)={d} t(text_ids)={t} n(audio)={n}");
        }
        if let Ok(dir) = std::env::var("RLX_F5_DUMP_PP") {
            let _ = std::fs::create_dir_all(&dir);
            let names = [
                "noise",
                "rope_cos",
                "rope_sin",
                "cat_mel_text",
                "cat_mel_text_drop",
                "qk_rotated_empty",
                "ref_signal_len",
            ];
            for (i, (b, dt)) in pre.iter().enumerate() {
                let bytes = f16_bytes(&to_f32(b, *dt));
                let _ = std::fs::write(format!("{dir}/pp_rlx_{}.f16", names[i]), bytes);
            }
            eprintln!("[f5-dbg] dumped rlx preprocess to {dir} (md={d} t={t} n={n})");
        }
        if std::env::var("RLX_F5_STOP_AFTER_PRE").is_ok() {
            return Ok(vec![0.0; 100]);
        }
        // Output order (graph decl): noise, rope_cos, rope_sin, cat_mel_text,
        // cat_mel_text_drop, qk_rotated_empty, ref_signal_len.
        let dit_dev = dit_exec_device(self.device);
        if dit_dev != self.device {
            eprintln!(
                "[f5tts] DiT on {dit_dev:?} (preprocess/decode on {:?}; set RLX_F5_CPU_DIT=1 for CPU DiT)",
                self.device
            );
        }
        let (mut noise, mut noise_dt) = as_feed(&pre[0].0, pre[0].1, dit_dev);
        let (rope_cos, rope_cos_dt) = as_feed(&pre[1].0, pre[1].1, dit_dev);
        let (rope_sin, rope_sin_dt) = as_feed(&pre[2].0, pre[2].1, dit_dev);
        let (cat_mel, cat_mel_dt) = as_feed(&pre[3].0, pre[3].1, dit_dev);
        let (cat_mel_drop, cat_mel_drop_dt) = as_feed(&pre[4].0, pre[4].1, dit_dev);
        let (qk, qk_dt) = as_feed(&pre[5].0, pre[5].1, dit_dev);
        let ref_signal_len = i64_scalar(&pre[6].0);

        // ── 2. flow-matching loop (transformer folds CFG + ODE step) ────────────
        // DakeQQ export: linspace(0,1,NFE) → delta_t has length NFE-1. Inference
        // runs `range(0, NFE-1)` so time_step ∈ [0, NFE-2]. An extra step indexes
        // past delta_t and adds hiss.
        let ode_steps = opts.nfe.saturating_sub(1).max(1);
        let tf_named = &[("max_duration", d), ("text_embed_len", TEXT_EMBED_LEN)];
        let mut tf = self.compile("F5_Transformer", dit_dev, d, tf_named)?;
        // Re-upload all feeds every NFE step. Arena buffer reuse can stomp
        // graph-input slots after their last read within a run, so pinning
        // conditioning (or relying on a noise D2D feed alone) left later steps
        // reading garbage — CUDA NFE=32 collapsed to cos≈0 / fox 0/6.
        for step in 0..ode_steps {
            let ts_b = (step as i32).to_le_bytes().to_vec();
            let out = tf.run_typed(&[
                ("noise", &noise, noise_dt),
                ("rope_cos", &rope_cos, rope_cos_dt),
                ("rope_sin", &rope_sin, rope_sin_dt),
                ("cat_mel_text", &cat_mel, cat_mel_dt),
                ("cat_mel_text_drop", &cat_mel_drop, cat_mel_drop_dt),
                ("qk_rotated_empty", &qk, qk_dt),
                ("time_step", &ts_b, DType::I32),
            ]);
            let (out_bytes, out_dt) = out.into_iter().next().context("transformer: no output")?;
            let (n2, ndt) = as_feed(&out_bytes, out_dt, dit_dev);
            noise = n2;
            noise_dt = ndt;
            if std::env::var("RLX_F5_DBG").is_ok() {
                let all = std::env::var("RLX_F5_DBG_ALL_STEPS").is_ok();
                if all || step == 0 || step + 1 == ode_steps {
                    let v = to_f32(&noise, noise_dt);
                    let pk = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                    let nan_ct = v.iter().filter(|x| x.is_nan()).count();
                    eprintln!(
                        "[f5-dbg] denoised step {step}/{ode_steps} peak={pk:.4} nan={nan_ct}"
                    );
                }
            }
        }

        // ── 3. decode → waveform ────────────────────────────────────────────────
        let (noise_dec, noise_dec_dt) = as_feed(&noise, noise_dt, self.device);
        let rl_b = ref_signal_len.to_le_bytes().to_vec();
        let dec = self.run(
            "F5_Decode",
            self.device,
            d,
            &[("max_duration", d)],
            &[
                ("denoised", &noise_dec, noise_dec_dt),
                ("ref_signal_len", &rl_b, DType::I64),
            ],
        )?;
        let (bytes, dt) = dec.into_iter().next().context("decode: no output")?;
        let mut wav = to_f32(&bytes, dt);
        // The ONNX decode graph crops the reference prefix via `ref_signal_len`.
        // After onnx-import the Slice sometimes keeps the full padded length —
        // drop the reference hop frames so the WAV matches ORT (gen-only).
        let skip = (ref_signal_len.max(0) as usize).saturating_mul(HOP_LENGTH);
        if wav.len() > skip + HOP_LENGTH {
            wav = wav[skip..].to_vec();
        }
        soft_peak_limit(&mut wav, 0.95);
        // Silence is judged by the caller (`synthesize`): a single near-silent
        // chunk must not sink a multi-chunk utterance.
        Ok(wav)
    }
}

fn max_duration_cap() -> usize {
    std::env::var("RLX_F5_MAX_DURATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_DURATION)
        .clamp(256, 4096)
}

fn estimate_duration(ref_audio_len: f64, ref_tl: f64, gen_text: &str, speed: f32) -> usize {
    let gen_tl = text_len(gen_text) as f64;
    (ref_audio_len + (ref_audio_len / ref_tl.max(1.0) * gen_tl / speed.max(0.1) as f64)) as usize
}

/// Split `gen_text` so each piece's estimated mel length stays ≤ `cap`.
fn chunk_gen_text(
    gen_text: &str,
    ref_audio_len: f64,
    ref_tl: f64,
    speed: f32,
    cap: usize,
) -> Vec<String> {
    let trimmed = gen_text.trim();
    if trimmed.is_empty() {
        return vec![String::new()];
    }
    if estimate_duration(ref_audio_len, ref_tl, trimmed, speed) <= cap {
        return vec![trimmed.to_string()];
    }
    // Prefer sentence boundaries; fall back to comma / space packs.
    let mut sentences: Vec<&str> = Vec::new();
    let mut start = 0;
    let bytes = trimmed.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if matches!(b, b'.' | b'!' | b'?') {
            let end = (i + 1).min(trimmed.len());
            let s = trimmed[start..end].trim();
            if !s.is_empty() {
                sentences.push(s);
            }
            start = end;
        }
    }
    if start < trimmed.len() {
        let s = trimmed[start..].trim();
        if !s.is_empty() {
            sentences.push(s);
        }
    }
    if sentences.is_empty() {
        sentences.push(trimmed);
    }

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for s in sentences {
        let candidate = if cur.is_empty() {
            s.to_string()
        } else {
            format!("{cur} {s}")
        };
        if estimate_duration(ref_audio_len, ref_tl, &candidate, speed) <= cap {
            cur = candidate;
            continue;
        }
        if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if estimate_duration(ref_audio_len, ref_tl, s, speed) <= cap {
            cur = s.to_string();
        } else {
            // Hard-split oversized sentence by chars.
            for piece in hard_split(s, ref_audio_len, ref_tl, speed, cap) {
                out.push(piece);
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

fn hard_split(s: &str, ref_audio_len: f64, ref_tl: f64, speed: f32, cap: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in s.split_whitespace() {
        let candidate = if cur.is_empty() {
            w.to_string()
        } else {
            format!("{cur} {w}")
        };
        if estimate_duration(ref_audio_len, ref_tl, &candidate, speed) <= cap {
            cur = candidate;
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur = w.to_string();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn crossfade_append(dst: &mut Vec<f32>, src: &[f32], fade: usize) {
    let fade = fade.min(dst.len()).min(src.len());
    if fade == 0 {
        dst.extend_from_slice(src);
        return;
    }
    let d0 = dst.len() - fade;
    for i in 0..fade {
        let a = (i as f32 + 1.0) / (fade as f32 + 1.0);
        dst[d0 + i] = dst[d0 + i] * (1.0 - a) + src[i] * a;
    }
    dst.extend_from_slice(&src[fade..]);
}

fn f16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

fn i64_scalar(b: &[u8]) -> i64 {
    if b.len() >= 8 {
        i64::from_le_bytes(b[..8].try_into().unwrap())
    } else if b.len() >= 4 {
        // Prefer integer reinterpret; if the arena widened to f32 the bits are
        // a float (typical small integer → exact f32).
        let as_i = i32::from_le_bytes(b[..4].try_into().unwrap()) as i64;
        let as_f = f32::from_le_bytes(b[..4].try_into().unwrap());
        if as_f.is_finite() && as_f.abs() < 1.0e7 && (as_f - as_f.round()).abs() < 1e-3 {
            as_f.round() as i64
        } else {
            as_i
        }
    } else {
        0
    }
}

/// Where the NFE DiT loop should run.
///
/// Default: same as `requested`. Metal/MLX match CPU across the full NFE chain
/// (fox 6/6). Discrete GPUs use f32-uniform ScatterNd with `force_indices_f32`.
/// Opt out with `RLX_F5_CPU_DIT=1`. Apple wgpu DiT still drifts; when Metal is
/// available, route DiT there unless `RLX_F5_WGPU_DIT=1`.
fn dit_exec_device(requested: Device) -> Device {
    if env_truthy("RLX_F5_CPU_DIT") {
        return Device::Cpu;
    }
    if env_truthy("RLX_F5_GPU_DIT") {
        if matches!(requested, Device::Gpu)
            && !env_truthy("RLX_F5_WGPU_DIT")
            && rlx_runtime::is_available(Device::Metal)
        {
            return Device::Metal;
        }
        return requested;
    }
    if matches!(requested, Device::Gpu)
        && !env_truthy("RLX_F5_WGPU_DIT")
        && rlx_runtime::is_available(Device::Metal)
    {
        return Device::Metal;
    }
    // Vulkan DiT: activation striping in rlx-vulkan keeps acts within
    // maxStorageBufferRange. Stay on Vulkan by default. Opt into the old
    // hybrid with `RLX_F5_CUDA_DIT=1` (DiT on CUDA, preprocess/decode Vulkan)
    // or force CPU with `RLX_F5_CPU_DIT=1`.
    if matches!(requested, Device::Vulkan) && env_truthy("RLX_F5_CUDA_DIT") {
        if rlx_runtime::is_available(Device::Cuda) {
            eprintln!(
                "[f5tts] DiT on Cuda (preprocess/decode on {:?}; unset RLX_F5_CUDA_DIT for true Vulkan DiT)",
                requested
            );
            return Device::Cuda;
        }
        eprintln!("[f5tts] DiT on Cpu (RLX_F5_CUDA_DIT set but CUDA unavailable)");
        return Device::Cpu;
    }
    requested
}

fn env_truthy(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// Pack floats for a graph input slot.
///
/// CPU keeps the ONNX f16 edge. Metal/MLX/wgpu/CUDA run `prepare_f32_exec_graph`
/// — keep activations as F32 between DiT steps so we don't re-quantize through
/// f16 every NFE (that compounds with any residual backend error).
fn float_bytes(x: &[f32], device: Device) -> (Vec<u8>, DType) {
    match device {
        Device::Metal
        | Device::Mlx
        | Device::Gpu
        | Device::Vulkan
        | Device::Cuda
        | Device::Rocm
        | Device::Ane => (x.iter().flat_map(|v| v.to_le_bytes()).collect(), DType::F32),
        _ => (f16_bytes(x), DType::F16),
    }
}

/// Re-pack a graph output into the f16 edge the next F5 graph expects.
fn as_feed(bytes: &[u8], dt: DType, device: Device) -> (Vec<u8>, DType) {
    let f = to_f32(bytes, dt);
    float_bytes(&f, device)
}

fn to_f32(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F16 => b
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}
