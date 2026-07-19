// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native (ort-free) Parler-TTS synthesis: T5 text encoder → 9-codebook
//! delay-pattern decoder AR loop → un-delay → Descript-DAC decode → PCM.
//!
//! Both transformer graphs are imported from ONNX to rlx-ir via `rlx-onnx-import`
//! and run on any rlx backend (CPU/Metal/MLX/wgpu/CoreML). ort is a DEV-dependency
//! only (parity validation in `examples/native_parity.rs`), never at runtime.
//!
//! The decoder has NO KV cache and no separate transcript input (this export
//! conditions purely through the T5 encoder). Rather than recompile per step
//! (re-prefill), we compile the decoder ONCE at a fixed `max_steps` length and
//! refill a padded `[1,9,max]` buffer each step — the baked causal mask (`Trilu`)
//! makes `logits[:, step, :]` identical to the growing-prefix result.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use rlx_dac::codec::DacCodec;
use rlx_dac::codes::DacCodes;
use rlx_ir::DType;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, Device};
use tokenizers::Tokenizer;

const K: usize = 9; // codebooks
const BOS: i64 = 1025; // decoder_start_token_id
const PAD: i64 = 1024; // pad_token_id (== eos)
const VOCAB: usize = 1088; // per-codebook vocab
const DAC_MAX: i64 = 1024; // valid DAC code range is [0, 1023]

/// Default voice description when the caller does not supply one.
pub const DEFAULT_DESCRIPTION: &str =
    "A clear female voice speaks slowly with moderate pitch and no background noise.";

/// Sampling / loop knobs.
#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub max_steps: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
    /// Greedy argmax (ignores temperature/top_k) — repetitive but deterministic.
    pub greedy: bool,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self {
            max_steps: 172,
            temperature: 1.0,
            top_k: 50,
            seed: 0x50415254,
            greedy: false,
        }
    }
}

pub struct NativeParler {
    dir: PathBuf,
    cache: AotCache,
    tok: Tokenizer,
    device: Device,
    dac: DacCodec,
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

impl NativeParler {
    /// `dir` holds `onnx/{text_encoder,decoder}.onnx` + `tokenizer.json`.
    /// `dac_dir` holds the Descript-DAC `config.json` + `model.safetensors`.
    pub fn open(dir: impl AsRef<Path>, dac_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let dac = DacCodec::open_on(dac_dir.as_ref(), device)
            .with_context(|| format!("open DAC at {}", dac_dir.as_ref().display()))?;
        let cache = AotCache::new(std::env::temp_dir().join("rlx_parler_native"));
        Ok(Self {
            dir,
            cache,
            tok,
            device,
            dac,
        })
    }

    /// Import + compile a component graph with its params bound, ready to run.
    fn compile(
        &self,
        component: &str,
        named: &[(&str, usize)],
        seq: usize,
    ) -> Result<CompiledGraph> {
        let path = self.dir.join("onnx").join(format!("{component}.onnx"));
        let mut named_lengths: HashMap<String, usize> =
            named.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        named_lengths.entry("batch_size".into()).or_insert(1);
        let opts = ImportOptions {
            sequence_length: seq,
            named_lengths,
            strict: false,
            ..Default::default()
        };
        let (hir, mut params, _r, _m) =
            build_hir_from_onnx_file(&path, opts).with_context(|| format!("import {component}"))?;
        let key = format!("parler_{component}_s{seq}_{:?}", self.device);
        let mut g = self
            .cache
            .compile_hir_cached(&key, self.device, hir, &CompileOptions::default())
            .map_err(|e| anyhow!("compile {component}: {e}"))?;
        for (n, d) in params.drain() {
            g.set_param(&n, &d);
        }
        g.finalize_params();
        Ok(g)
    }

    /// Encode a string through the T5 encoder → flat `[enc_len * d_model]`.
    ///
    /// The current ONNX export has no separate `prompt_input_ids` input on the
    /// decoder (true Parler routes the transcript as a prompt prefix and the
    /// voice description through T5). We therefore feed the **transcript** to
    /// the encoder so the AR loop has content to speak; `description` still
    /// seeds sampling for voice variety until a re-export lands prompt embeds.
    fn encode(&self, text: &str) -> Result<(Vec<f32>, usize)> {
        let mut ids: Vec<i64> = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .iter()
            .map(|&i| i as i64)
            .collect();
        ids.push(1); // T5 eos
        let n = ids.len();
        let mask: Vec<i64> = vec![1; n];
        let mut enc = self.compile("text_encoder", &[("sequence_length", n), ("t", n)], n)?;
        let out = enc.run_typed(&[
            ("input_ids", &i64_le(&ids), DType::I64),
            ("attention_mask", &i64_le(&mask), DType::I64),
        ]);
        Ok((as_f32(&out[0].0), n))
    }

    /// Full pipeline: transcript (+ optional voice description) → PCM @ DAC rate.
    pub fn synthesize(&self, text: &str, description: &str, opts: &InferOpts) -> Result<Vec<f32>> {
        let text = text.trim();
        if text.is_empty() {
            return Err(anyhow!("empty transcript"));
        }
        let (hs, enc_len) = self.encode(text)?;
        let hs_bytes = f32_le(&hs);
        let emask: Vec<i64> = vec![1; enc_len];
        let emask_bytes = i64_le(&emask);
        let t = opts.max_steps;

        // Decoder compiled ONCE at fixed length `t`; refilled each step.
        let mut dec = self.compile(
            "decoder",
            &[
                ("t", t),
                ("et", enc_len),
                ("sequence_length", t),
                ("encoder_sequence_length", enc_len),
            ],
            t,
        )?;

        // decoder_input_ids [1, 9, t]: position 0 = BOS (all codebooks), rest PAD.
        let mut dids = vec![PAD; K * t];
        for k in 0..K {
            dids[k * t] = BOS; // [k, 0]
        }
        // Mix description into the RNG seed so voice strings change sampling without
        // requiring the (not-yet-exported) prompt-embed path.
        let seed = opts.seed ^ hash64(description);
        let mut rng = Rng::new(seed);
        // per-generation-step codes [9] (delayed, as fed to the decoder)
        let mut steps: Vec<[i64; K]> = Vec::with_capacity(t);

        for step in 0..t {
            let out = dec.run_typed(&[
                ("decoder_input_ids", &i64_le(&dids), DType::I64),
                ("encoder_hidden_states", &hs_bytes, DType::F32),
                ("encoder_attention_mask", &emask_bytes, DType::I64),
            ]);
            // logits: [9, t, VOCAB] = [codebook, seq, vocab]; take position `step`.
            let logits = as_f32(&out[0].0);
            let mut nxt = [PAD; K];
            for k in 0..K {
                if step < k {
                    nxt[k] = BOS; // delay: codebook k not real until step >= k
                    continue;
                }
                let base = k * t * VOCAB + step * VOCAB;
                let row = &logits[base..base + VOCAB];
                nxt[k] = if opts.greedy {
                    argmax(row) as i64
                } else {
                    sample(row, opts.temperature, opts.top_k, &mut rng) as i64
                };
            }
            steps.push(nxt);
            // stop once every codebook is producing EOS/PAD (past the delay ramp)
            if step >= K && nxt.iter().all(|&c| c == PAD) {
                break;
            }
            // write next codes into the buffer at position step+1 (if room)
            if step + 1 < t {
                for k in 0..K {
                    dids[k * t + (step + 1)] = nxt[k];
                }
            }
        }

        // Un-delay: codebook k was produced delayed by k → shift left by k. Then
        // clamp the tail to positions where every codebook is a valid DAC code.
        let n_steps = steps.len();
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); K];
        for k in 0..K {
            for p in 0..n_steps.saturating_sub(k) {
                let c = steps[p + k][k];
                rows[k].push(c as u32);
            }
        }
        // Trim to the shortest valid run: stop at the first frame with any code ≥ 1024.
        let min_len = rows.iter().map(|r| r.len()).min().unwrap_or(0);
        let mut valid = 0usize;
        'outer: for f in 0..min_len {
            for r in &rows {
                if r[f] as i64 >= DAC_MAX {
                    break 'outer;
                }
            }
            valid = f + 1;
        }
        for r in &mut rows {
            r.truncate(valid);
        }
        if valid == 0 {
            return Err(anyhow!(
                "decoder produced no valid DAC frames (all codes ≥ {DAC_MAX})"
            ));
        }

        if std::env::var_os("RLX_PARLER_DEBUG").is_some() {
            for (k, r) in rows.iter().enumerate() {
                let uniq: std::collections::HashSet<u32> = r.iter().copied().collect();
                let mn = r.iter().copied().min().unwrap_or(0);
                let mx = r.iter().copied().max().unwrap_or(0);
                eprintln!(
                    "  cb{k}: {} frames, {} unique, range[{mn},{mx}], head={:?}",
                    r.len(),
                    uniq.len(),
                    &r[..r.len().min(10)]
                );
            }
        }

        // DAC decode: [quantizer][frame] → PCM.
        let codes = DacCodes::from_quantizer_layout(rows);
        let pcm = self.dac.decode_codes(&codes).context("DAC decode")?;
        Ok(pcm)
    }

    pub fn sample_rate(&self) -> u32 {
        self.dac.sample_rate()
    }

    /// Write mono PCM as 16-bit WAV at the DAC sample rate.
    pub fn write_wav(&self, audio: &[f32], path: impl AsRef<Path>) -> Result<()> {
        write_wav(audio, self.sample_rate(), path.as_ref())
    }
}

/// Write mono f32 PCM as 16-bit integer WAV.
pub fn write_wav(audio: &[f32], sample_rate: u32, path: &Path) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &sample in audio {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
            .with_context(|| format!("write sample to {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("finalize {}", path.display()))?;
    Ok(())
}

fn hash64(s: &str) -> u64 {
    // FNV-1a 64 — stable, no extra deps.
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// Temperature + top-k multinomial sample over logits.
fn sample(row: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> usize {
    let temp = temperature.max(1e-4);
    // top-k indices by logit
    let mut idx: Vec<usize> = (0..row.len()).collect();
    let k = top_k.clamp(1, row.len());
    idx.select_nth_unstable_by(k - 1, |&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    idx.truncate(k);
    let maxl = idx
        .iter()
        .map(|&i| row[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((row[i] - maxl) / temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    let r = rng.next_f32();
    let mut acc = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[j];
        }
    }
    idx[k - 1]
}

/// Small deterministic RNG (splitmix64) — reproducible synthesis, no `rand` dep.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E3779B97F4A7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
