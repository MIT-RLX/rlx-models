// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native (ort-free) Soprano 1.1: Qwen3 AR backbone (KV) + 32 kHz vocoder.
//!
//! Nested `graphs/*.rlxp` (or legacy local ONNX) lowered via `rlx-onnx-import`.
//! Prompt format matches the web demo: `[STOP][TEXT]…[START]`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use rlx_ir::DType;
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, Device};
use tokenizers::Tokenizer;

pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/soprano";
pub const SAMPLE_RATE: u32 = 32_000;
pub const HIDDEN: usize = 512;
pub const N_LAYERS: usize = 17;
pub const KV_HEADS: usize = 1;
pub const HEAD_DIM: usize = 128;
pub const VOCAB: usize = 8192;
pub const EOS: i64 = 3;
pub const TOKEN_SIZE: usize = 2048;
pub const RECEPTIVE_FIELD: usize = 4;

const BACKBONE: &str = "soprano_backbone_kv_fp32";
const DECODER: &str = "soprano_decoder_fp32";

fn should_decompose_decoder_ct(device: Device) -> bool {
    matches!(device, Device::Ane | Device::Metal)
}

/// Backbone full-recompute is finite only for `seq ≤ 32` and `seq == 128` (RLX
/// import/runtime quirk). Pad anything in between up to 128.
fn bucket_seq(n: usize) -> usize {
    if n <= 32 { n } else { 128 }
}

#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub greedy: bool,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.3,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.2,
            seed: 1337,
            greedy: false,
        }
    }
}

pub struct NativeSoprano {
    dir: PathBuf,
    cache: AotCache,
    tok: Tokenizer,
    device: Device,
    /// Vocos ISTFT uses ScatterElements; MLX host-kernel path can explode peaks.
    /// Pin the decoder to CPU when the backbone runs on MLX.
    dec_device: Device,
    graphs: Mutex<HashMap<String, CompiledGraph>>,
}

fn i64_le(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Soprano prompt wrapper (matches KevinAHM/soprano-web-onnx).
pub fn format_prompt(text: &str) -> String {
    let t = text.trim();
    format!("[STOP][TEXT]{t}[START]")
}

impl NativeSoprano {
    /// Prefer `soprano.rlxp`, then legacy `soprano.gguf`, else a materialized dir.
    pub fn open(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        crate::gguf_bundle::open_path(dir.as_ref(), device)
    }

    /// Load from an already-materialized directory (`graphs/*.rlxp` or legacy ONNX).
    pub fn open_loose(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        // Vocos ISTFT overlap-add uses ScatterElements (registered for HIR lower).
        rlx_onnx_import::onnx_scatter::register_onnx_scatter_elements_kernel();
        let dir = dir.as_ref().to_path_buf();
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let cache = AotCache::new(std::env::temp_dir().join(format!("rlx_soprano_{device:?}")));
        let dec_device = match device {
            Device::Mlx => Device::Cpu,
            other => other,
        };
        if dec_device != device {
            eprintln!(
                "[soprano] Vocos decoder on {dec_device:?} (backbone on {device:?}; MLX ScatterElements unstable)"
            );
        }
        Ok(Self {
            dir,
            cache,
            tok,
            device,
            dec_device,
            graphs: Mutex::new(HashMap::new()),
        })
    }

    pub fn open_default(device: Device) -> Result<Self> {
        Self::open(DEFAULT_LOCAL_DIR, device)
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn encode_prompt(&self, text: &str) -> Result<Vec<i64>> {
        let prompt = format_prompt(text);
        let enc = self
            .tok
            .encode(prompt, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
    }

    fn compile_backbone(&self, past: usize, seq: usize) -> Result<CompiledGraph> {
        let path = rlx_tiny_tts::model::resolve_component_path(&self.dir, BACKBONE)?;
        let named: Vec<(&str, usize)> = vec![
            ("batch_size", 1),
            ("sequence_length", seq),
            ("past_sequence_length", past),
            ("past_sequence_length + sequence_length", past + seq),
        ];
        let (hir, mut params, _r) =
            rlx_tiny_tts::model::import_graph_named(&path, BACKBONE, seq.max(1), false, &named)
                .with_context(|| format!("import {BACKBONE} past={past} seq={seq}"))?;
        let key = format!("sop_{BACKBONE}_{:?}_p{past}_s{seq}", self.device);
        let mut g = self
            .cache
            .compile_hir_cached(&key, self.device, hir, &CompileOptions::default())
            .map_err(|e| anyhow!("compile backbone: {e}"))?;
        for (n, d) in params.drain() {
            g.set_param(&n, &d);
        }
        g.finalize_params();
        Ok(g)
    }

    fn compile_decoder(&self, t: usize) -> Result<CompiledGraph> {
        let path = rlx_tiny_tts::model::resolve_component_path(&self.dir, DECODER)?;
        let audio_len = TOKEN_SIZE.saturating_mul(t.saturating_sub(1));
        let buf_len = TOKEN_SIZE * t;
        let frames = 4 * t - 3;
        let named: Vec<(&str, usize)> = vec![
            ("s53", t),
            ("2048*s53", buf_len),
            ("2048*s53 - 2048", audio_len),
            ("4*s53 - 3", frames),
            ("8192*s53 - 6144", 8192 * t - 6144),
        ];
        let (hir, mut params, _r) = rlx_tiny_tts::model::import_graph_named(
            &path,
            DECODER,
            t.max(1),
            should_decompose_decoder_ct(self.dec_device),
            &named,
        )
        .with_context(|| format!("import {DECODER} t={t}"))?;
        let key = format!("sop_{DECODER}_{:?}_t{t}", self.dec_device);
        let mut g = self
            .cache
            .compile_hir_cached(&key, self.dec_device, hir, &CompileOptions::default())
            .map_err(|e| anyhow!("compile decoder: {e}"))?;
        for (n, d) in params.drain() {
            g.set_param(&n, &d);
        }
        g.finalize_params();
        Ok(g)
    }

    fn graph_mut<'a>(
        cache: &'a mut HashMap<String, CompiledGraph>,
        key: &str,
        build: impl FnOnce() -> Result<CompiledGraph>,
    ) -> Result<&'a mut CompiledGraph> {
        if !cache.contains_key(key) {
            cache.insert(key.to_string(), build()?);
        }
        Ok(cache.get_mut(key).unwrap())
    }

    /// Run backbone over `input_ids` with empty past KV (full recompute).
    ///
    /// ONNX→HIR currently mis-broadcasts attention masks when `past_sequence_length > 1`
    /// (head dim becomes `128 + past`). Prefill and decode therefore always use `past=0`
    /// and the full token prefix — fine for short TTS utterances.
    ///
    /// Sequence lengths are bucketed via [`bucket_seq`] (pad to 128 when `n > 32`).
    fn backbone_step(
        &self,
        input_ids: &[i64],
        opts: &InferOpts,
        seen: &mut [bool],
        rng: &mut Rng,
    ) -> Result<(i64, Vec<f32>)> {
        let real = input_ids.len();
        anyhow::ensure!(real > 0, "empty input_ids");
        anyhow::ensure!(
            real <= 128,
            "sequence length {real} exceeds supported bucket 128"
        );
        let seq = bucket_seq(real);
        let past_len = 0usize;
        let past_kv: Vec<(Vec<f32>, Vec<f32>)> =
            (0..N_LAYERS).map(|_| (Vec::new(), Vec::new())).collect();
        let key = format!("bb_p{past_len}_s{seq}");
        let mut cache = self.graphs.lock().unwrap();
        let g = Self::graph_mut(&mut cache, &key, || self.compile_backbone(past_len, seq))?;

        let mut ids = input_ids.to_vec();
        ids.resize(seq, 0);
        let mut attn = vec![0i64; seq];
        for a in attn.iter_mut().take(real) {
            *a = 1;
        }
        let mut pos = vec![0i64; seq];
        for (i, p) in pos.iter_mut().enumerate().take(real) {
            *p = i as i64;
        }
        let mut inputs: Vec<(String, Vec<u8>, DType)> = Vec::with_capacity(3 + 2 * N_LAYERS);
        inputs.push(("input_ids".into(), i64_le(&ids), DType::I64));
        inputs.push(("attention_mask".into(), i64_le(&attn), DType::I64));
        inputs.push(("position_ids".into(), i64_le(&pos), DType::I64));
        for i in 0..N_LAYERS {
            let (k, v) = &past_kv[i];
            inputs.push((format!("past_key_values.{i}.key"), f32_le(k), DType::F32));
            inputs.push((format!("past_key_values.{i}.value"), f32_le(v), DType::F32));
        }
        let refs: Vec<(&str, &[u8], DType)> = inputs
            .iter()
            .map(|(n, b, d)| (n.as_str(), b.as_slice(), *d))
            .collect();
        let outs = g.run_typed(&refs);
        anyhow::ensure!(
            outs.len() > 1 + 2 * N_LAYERS,
            "backbone outs {} < expected",
            outs.len()
        );
        let logits = as_f32(&outs[0].0);
        anyhow::ensure!(
            logits.len() >= seq * VOCAB,
            "logits len {} < seq*vocab {}",
            logits.len(),
            seq * VOCAB
        );
        // Read the last *real* token (ignore pad positions).
        let row_base = (real - 1) * VOCAB;
        let row = &logits[row_base..row_base + VOCAB];
        if !row.iter().any(|x| x.is_finite()) {
            return Err(anyhow!(
                "backbone logits all non-finite at real={real} bucket={seq}"
            ));
        }
        let next = if opts.greedy {
            argmax(row) as i64
        } else {
            sample_top_p(
                row,
                opts.temperature,
                opts.top_p,
                opts.top_k,
                opts.repetition_penalty,
                seen,
                rng,
            ) as i64
        };
        let hidden_all = as_f32(&outs[1 + 2 * N_LAYERS].0);
        anyhow::ensure!(
            hidden_all.len() >= seq * HIDDEN,
            "hidden len {} < seq*hidden",
            hidden_all.len()
        );
        let h_base = (real - 1) * HIDDEN;
        let last_h = hidden_all[h_base..h_base + HIDDEN].to_vec();
        Ok((next, last_h))
    }

    /// Vocode latents `[T][512]` → PCM @ 32 kHz.
    ///
    /// When `crop_offline` is set, keep the last `(T-1)*TOKEN_SIZE` samples
    /// (Python offline `infer` crop).
    pub fn decode_latents(&self, latents: &[Vec<f32>], crop_offline: bool) -> Result<Vec<f32>> {
        let t = latents.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        // Channel-major [1, 512, T] as in the web demo.
        let mut buf = vec![0f32; HIDDEN * t];
        for (w, h) in latents.iter().enumerate() {
            for d in 0..HIDDEN {
                buf[d * t + w] = h[d];
            }
        }
        let key = format!("dec_t{t}");
        let mut cache = self.graphs.lock().unwrap();
        let g = Self::graph_mut(&mut cache, &key, || self.compile_decoder(t))?;
        let outs = g.run_typed(&[("hidden_states", &f32_le(&buf), DType::F32)]);
        let mut pcm = as_f32(&outs[0].0);
        for s in &mut pcm {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        if crop_offline {
            let keep = t.saturating_sub(1).saturating_mul(TOKEN_SIZE);
            if keep > 0 && pcm.len() > keep {
                pcm = pcm[pcm.len() - keep..].to_vec();
            }
        }
        Ok(pcm)
    }

    /// Prefill logit row over the prompt (last real position), for backend parity digs.
    pub fn prefill_logits(&self, text: &str) -> Result<Vec<f32>> {
        let prompt = self.encode_prompt(text.trim())?;
        anyhow::ensure!(!prompt.is_empty(), "empty tokenization");
        let real = prompt.len();
        let seq = bucket_seq(real);
        let key = format!("bb_p0_s{seq}");
        let mut cache = self.graphs.lock().unwrap();
        let g = Self::graph_mut(&mut cache, &key, || self.compile_backbone(0, seq))?;
        let mut ids = prompt;
        ids.resize(seq, 0);
        let mut attn = vec![0i64; seq];
        for a in attn.iter_mut().take(real) {
            *a = 1;
        }
        let mut pos = vec![0i64; seq];
        for (i, p) in pos.iter_mut().enumerate().take(real) {
            *p = i as i64;
        }
        let past_kv: Vec<(Vec<f32>, Vec<f32>)> =
            (0..N_LAYERS).map(|_| (Vec::new(), Vec::new())).collect();
        let mut inputs: Vec<(String, Vec<u8>, DType)> = Vec::with_capacity(3 + 2 * N_LAYERS);
        inputs.push(("input_ids".into(), i64_le(&ids), DType::I64));
        inputs.push(("attention_mask".into(), i64_le(&attn), DType::I64));
        inputs.push(("position_ids".into(), i64_le(&pos), DType::I64));
        for i in 0..N_LAYERS {
            let (k, v) = &past_kv[i];
            inputs.push((format!("past_key_values.{i}.key"), f32_le(k), DType::F32));
            inputs.push((format!("past_key_values.{i}.value"), f32_le(v), DType::F32));
        }
        let refs: Vec<(&str, &[u8], DType)> = inputs
            .iter()
            .map(|(n, b, d)| (n.as_str(), b.as_slice(), *d))
            .collect();
        let outs = g.run_typed(&refs);
        let logits = as_f32(&outs[0].0);
        let row_base = (real - 1) * VOCAB;
        Ok(logits[row_base..row_base + VOCAB].to_vec())
    }

    /// Autoregressive audio-token ids after the text prompt (greedy or sampled).
    /// Prefill sample is included; `EOS` ends the list when emitted.
    pub fn generate_audio_tokens(&self, text: &str, opts: &InferOpts) -> Result<Vec<i64>> {
        let (_latents, toks) = self.generate_latents(text, opts)?;
        Ok(toks)
    }

    /// AR latents `[T][512]` plus the audio token stream that produced them.
    pub fn generate_latents(
        &self,
        text: &str,
        opts: &InferOpts,
    ) -> Result<(Vec<Vec<f32>>, Vec<i64>)> {
        let text = text.trim();
        if text.is_empty() {
            return Err(anyhow!("empty text"));
        }
        let prompt = self.encode_prompt(text)?;
        anyhow::ensure!(!prompt.is_empty(), "empty tokenization");

        let mut seen = vec![false; VOCAB];
        for &id in &prompt {
            if (0..VOCAB as i64).contains(&id) {
                seen[id as usize] = true;
            }
        }
        let mut rng = Rng::new(opts.seed);
        let mut latents: Vec<Vec<f32>> = Vec::new();
        let mut audio_toks: Vec<i64> = Vec::new();
        let mut ids = prompt;

        // Prefill (i==0): sample first audio token; discard hidden.
        let (mut next, _) = self.backbone_step(&ids, opts, &mut seen, &mut rng)?;
        audio_toks.push(next);
        if next == EOS {
            return Ok((latents, audio_toks));
        }
        if (0..VOCAB as i64).contains(&next) {
            seen[next as usize] = true;
        }
        ids.push(next);

        for _step in 0..opts.max_new_tokens {
            // Full-recompute path only supports seq ≤ 128 (see `bucket_seq`).
            if ids.len() >= 128 {
                break;
            }
            let (tok, hidden) = self.backbone_step(&ids, opts, &mut seen, &mut rng)?;
            audio_toks.push(tok);
            // Demo: push last_hidden when i>0 && sampled != EOS.
            if tok != EOS {
                latents.push(hidden);
            }
            next = tok;
            if (0..VOCAB as i64).contains(&next) {
                seen[next as usize] = true;
            }
            ids.push(next);
            if next == EOS {
                break;
            }
        }

        Ok((latents, audio_toks))
    }

    /// Full synthesis: text → AR audio tokens/latents → PCM @ 32 kHz.
    ///
    /// Long prompts are sentence-/word-chunked: the ONNX full-recompute backbone
    /// only supports `seq ≤ 128`, so a long prompt can leave zero room for AR
    /// and yield empty latents.
    pub fn synthesize(&self, text: &str, opts: &InferOpts) -> Result<Vec<f32>> {
        let text = text.trim();
        if text.is_empty() {
            return Err(anyhow!("empty text"));
        }
        const MAX_PROMPT_TOKENS: usize = 80;
        let prompt_len = self.encode_prompt(text)?.len();
        if prompt_len > MAX_PROMPT_TOKENS {
            return self.synthesize_chunked(text, opts, MAX_PROMPT_TOKENS);
        }
        let (latents, _) = self.generate_latents(text, opts)?;
        if latents.is_empty() {
            return Err(anyhow!(
                "no audio latents produced (prompt_tokens={prompt_len}; first sample may be EOS)"
            ));
        }
        self.decode_latents(&latents, true)
    }

    fn synthesize_chunked(
        &self,
        text: &str,
        opts: &InferOpts,
        max_prompt_tokens: usize,
    ) -> Result<Vec<f32>> {
        let chunks = split_utterances(text, max_prompt_tokens, |chunk| {
            self.encode_prompt(chunk)
                .map(|ids| ids.len())
                .unwrap_or(usize::MAX)
        });
        anyhow::ensure!(!chunks.is_empty(), "soprano: no text chunks after split");
        let mut pcm = Vec::new();
        let gap = (SAMPLE_RATE as usize) / 20; // ~50 ms between clauses
        for (i, chunk) in chunks.iter().enumerate() {
            let (latents, _) = self
                .generate_latents(chunk, opts)
                .with_context(|| format!("soprano chunk {}/{}: {chunk:?}", i + 1, chunks.len()))?;
            if latents.is_empty() {
                return Err(anyhow!(
                    "no audio latents produced for chunk {}/{}: {chunk:?}",
                    i + 1,
                    chunks.len()
                ));
            }
            let part = self.decode_latents(&latents, true)?;
            if !pcm.is_empty() {
                pcm.extend(std::iter::repeat_n(0.0f32, gap));
            }
            pcm.extend(part);
        }
        Ok(pcm)
    }

    pub fn write_wav(audio: &[f32], path: impl AsRef<Path>, sample_rate: u32) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path.as_ref(), spec)
            .with_context(|| format!("create {}", path.as_ref().display()))?;
        for &s in audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

/// Split `text` into chunks whose prompt token count stays under `max_prompt_tokens`.
/// Prefers sentence boundaries, then commas, then words.
fn split_utterances(
    text: &str,
    max_prompt_tokens: usize,
    prompt_tokens: impl Fn(&str) -> usize,
) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '.' | '?' | '!') {
            let t = buf.trim();
            if !t.is_empty() {
                sentences.push(t.to_string());
            }
            buf.clear();
        }
    }
    let t = buf.trim();
    if !t.is_empty() {
        sentences.push(t.to_string());
    }
    if sentences.is_empty() {
        sentences.push(text.trim().to_string());
    }

    let mut out: Vec<String> = Vec::new();
    for sent in sentences {
        if prompt_tokens(&sent) <= max_prompt_tokens {
            out.push(sent);
            continue;
        }
        let clauses: Vec<&str> = sent
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let mut cur = String::new();
        for clause in clauses {
            let trial = if cur.is_empty() {
                clause.to_string()
            } else {
                format!("{cur}, {clause}")
            };
            if !cur.is_empty() && prompt_tokens(&trial) > max_prompt_tokens {
                out.push(std::mem::take(&mut cur));
                cur = clause.to_string();
            } else {
                cur = trial;
            }
            if prompt_tokens(&cur) > max_prompt_tokens {
                let words: Vec<String> = cur.split_whitespace().map(str::to_string).collect();
                cur.clear();
                let mut pack = String::new();
                for w in words {
                    let trial = if pack.is_empty() {
                        w.clone()
                    } else {
                        format!("{pack} {w}")
                    };
                    if !pack.is_empty() && prompt_tokens(&trial) > max_prompt_tokens {
                        out.push(std::mem::take(&mut pack));
                        pack = w;
                    } else {
                        pack = trial;
                    }
                }
                cur = pack;
            }
        }
        let t = cur.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    }
    out.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn sample_top_p(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rep_pen: f32,
    seen: &[bool],
    rng: &mut Rng,
) -> usize {
    let temp = temperature.max(1e-5);
    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut s = s / temp;
            if seen[i] && rep_pen != 1.0 {
                s = if s < 0.0 { s * rep_pen } else { s / rep_pen };
            }
            (i, s)
        })
        .collect();
    scores.sort_unstable_by(|a, b| match (a.1.is_finite(), b.1.is_finite()) {
        (true, true) => b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => std::cmp::Ordering::Equal,
    });
    let k = if top_k == 0 {
        scores.len()
    } else {
        top_k.min(scores.len())
    };
    scores.truncate(k);
    if scores.is_empty() || !scores[0].1.is_finite() {
        return argmax(logits);
    }
    let mx = scores[0].1;
    let mut probs: Vec<(usize, f32)> = scores.iter().map(|&(i, s)| (i, (s - mx).exp())).collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    for p in &mut probs {
        p.1 /= sum.max(1e-20);
    }
    let cutoff = top_p.clamp(0.0, 1.0);
    let mut cum = 0.0f32;
    let mut last = 0usize;
    for (ki, (_, p)) in probs.iter().enumerate() {
        cum += *p;
        last = ki;
        if cum >= cutoff {
            break;
        }
    }
    let target = rng.next_f32() * cum.min(1.0);
    let mut running = 0.0f32;
    for &(i, p) in probs.iter().take(last + 1) {
        running += p;
        if running >= target {
            return i;
        }
    }
    probs[last].0
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64 / u64::MAX as f64) as f32
    }
}

/// Peak absolute amplitude.
pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}
