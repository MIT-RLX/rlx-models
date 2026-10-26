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

//! Moonshine v2 streaming ASR — `moonshine-ai/moonshine-streaming-{tiny,small,medium}`.
//!
//! ```text
//!  16 kHz PCM ─► 80-sample frames ─► per-frame CMVN ─► asinh(e^log_k · x)   (host)
//!      ─► linear 80→d ─► SiLU ─► causal conv k5/s2 ─► SiLU ─► causal conv k5/s2   (50 Hz)
//!      ─► N × [LN ─► sliding-window self-attn ─► LN ─► GELU MLP] ─► LN
//!      ─► + learned position embedding ─► proj to decoder width
//!      ─► Moonshine decoder (RoPE self-attn + cross-attn), greedy
//! ```
//!
//! What is new against Moonshine v1 ([`rlx_moonshine`]) is the encoder: a
//! cheap frame frontend instead of the raw-PCM conv stack, no positional
//! encoding at all inside the encoder (the "ergodic" encoder of
//! arXiv:2602.12241), and attention bounded per layer by a `(left, right)`
//! window — `(16, 4)` lets a frame see 15 frames back and 3 ahead, `(16, 0)`
//! is causal. Position enters only afterwards, as a learned embedding the
//! decoder side adds. The decoder is v1's unchanged, so it is built by
//! [`rlx_moonshine::flow::build_decoder_hidden_built`].
//!
//! LayerNorms in the encoder carry a unit offset (`γ + 1`, no bias); the
//! offset is folded into γ on load.

#[cfg(feature = "export")]
pub mod export;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::weight_map::WeightMap;
use rlx_core::{AssetSource, SafetensorsSource};
use rlx_flow::CompileProfile;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, Op, PadMode};
use rlx_ir::ops::attention::attention_kind_op;
use rlx_ir::{DType, Shape};
use rlx_moonshine::MoonshineConfig;
use rlx_moonshine::MoonshineWeightPrefix;
use rlx_moonshine::weight_source::CloningWeightSource;
use rlx_runtime::{CompiledGraph, Device};

pub const SAMPLE_RATE: usize = 16_000;

/// `nn.LayerNorm` default.
const LN_EPS: f32 = 1e-5;
/// `MoonshineStreamingFrameCMVN` default.
pub(crate) const CMVN_EPS: f32 = 1e-6;
/// Causal conv kernel and stride in the frontend.
pub(crate) const CONV_K: usize = 5;
pub(crate) const CONV_STRIDE: usize = 2;
/// Additive bias for a masked attention score (finite: no all-masked rows here,
/// but a NaN would be worse than a wrong row).
pub(crate) const MASKED: f32 = -1e9;
/// Decoder graphs are compiled for prefix lengths rounded up to this.
pub(crate) const DECODER_BUCKET: usize = 16;

/// Encoder hyper-parameters (`encoder_config`).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit: `moonshine-streaming-small` has 620-wide hidden states and
    /// 8 × 64 = 512-wide attention.
    pub head_dim: usize,
    pub intermediate_size: usize,
    /// Per layer `(left, right)`: a query at `q` sees keys `k` with
    /// `0 ≤ q − k < left` or `0 < k − q < right`.
    pub sliding_windows: Vec<(usize, usize)>,
    /// Samples per frontend frame (`sample_rate · frame_ms / 1000` = 80).
    pub frame_len: usize,
}

/// The whole checkpoint's configuration.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub encoder: EncoderConfig,
    /// Decoder side, in the v1 shape [`rlx_moonshine`]'s builder takes.
    pub decoder: MoonshineConfig,
}

impl StreamingConfig {
    /// Parse a `moonshine_streaming` `config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("moonshine-streaming: read {}", path.display()))?;
        Self::from_json(&raw)
    }

    /// Parse the text of a `moonshine_streaming` `config.json`.
    pub fn from_json(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)?;
        if v["model_type"].as_str() != Some("moonshine_streaming") {
            bail!(
                "moonshine-streaming: model_type is {:?}, expected \"moonshine_streaming\"",
                v["model_type"]
            );
        }
        let u = |v: &serde_json::Value, k: &str| -> Result<usize> {
            v[k].as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("moonshine-streaming: missing {k}"))
        };
        let e = &v["encoder_config"];
        let sliding_windows = e["sliding_windows"]
            .as_array()
            .ok_or_else(|| anyhow!("moonshine-streaming: missing sliding_windows"))?
            .iter()
            .map(|w| {
                Ok((
                    w[0].as_u64().ok_or_else(|| anyhow!("bad window"))? as usize,
                    w[1].as_u64().ok_or_else(|| anyhow!("bad window"))? as usize,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let sample_rate = e["sample_rate"].as_u64().unwrap_or(SAMPLE_RATE as u64) as f64;
        let frame_ms = e["frame_ms"].as_f64().unwrap_or(5.0);
        let encoder = EncoderConfig {
            hidden_size: u(e, "hidden_size")?,
            num_hidden_layers: u(e, "num_hidden_layers")?,
            num_attention_heads: u(e, "num_attention_heads")?,
            num_key_value_heads: u(e, "num_key_value_heads")?,
            head_dim: u(e, "head_dim")?,
            intermediate_size: u(e, "intermediate_size")?,
            sliding_windows,
            frame_len: (sample_rate * frame_ms / 1000.0).round() as usize,
        };
        if encoder.sliding_windows.len() != encoder.num_hidden_layers {
            bail!("moonshine-streaming: one sliding window per encoder layer expected");
        }
        if encoder.num_key_value_heads != encoder.num_attention_heads {
            bail!("moonshine-streaming: grouped-query encoder attention is not supported");
        }

        let rope = &v["rope_parameters"];
        let hidden = u(&v, "hidden_size")?;
        let heads = u(&v, "num_attention_heads")?;
        if let Some(hd) = v["head_dim"].as_u64()
            && hd as usize * heads != hidden
        {
            bail!("moonshine-streaming: decoder head_dim {hd} × {heads} heads != hidden {hidden}");
        }
        let decoder = MoonshineConfig {
            hidden_size: hidden,
            encoder_num_hidden_layers: encoder.num_hidden_layers,
            decoder_num_hidden_layers: u(&v, "num_hidden_layers")?,
            encoder_num_attention_heads: encoder.num_attention_heads,
            decoder_num_attention_heads: heads,
            encoder_intermediate_size: encoder.intermediate_size,
            decoder_intermediate_size: u(&v, "intermediate_size")?,
            vocab_size: u(&v, "vocab_size")?,
            rope_theta: rope["rope_theta"].as_f64().unwrap_or(10_000.0),
            partial_rotary_factor: rope["partial_rotary_factor"].as_f64().unwrap_or(1.0),
            max_position_embeddings: u(&v, "max_position_embeddings")?,
            attention_bias: v["attention_bias"].as_bool().unwrap_or(false),
            pad_head_dim_to_multiple_of: v["pad_head_dim_to_multiple_of"]
                .as_u64()
                .map(|x| x as usize),
            bos_token_id: v["bos_token_id"].as_u64().unwrap_or(1) as u32,
            eos_token_id: v["eos_token_id"].as_u64().unwrap_or(2) as u32,
            pad_token_id: v["pad_token_id"].as_u64().unwrap_or(0) as u32,
            decoder_start_token_id: v["decoder_start_token_id"].as_u64().unwrap_or(1) as u32,
        };
        Ok(Self { encoder, decoder })
    }
}

/// Host frontend: zero-pad to whole frames, then per-frame CMVN and asinh
/// compression. Returns `[frames · frame_len]` and the number of frames that
/// hold only real samples (a trailing partial frame is masked out, as the HF
/// embedder does with `padding_mask.sum() // frame_len`).
pub fn frontend(pcm: &[f32], frame_len: usize, log_k: f32) -> (Vec<f32>, usize) {
    let frames = pcm.len().div_ceil(frame_len);
    let valid = pcm.len() / frame_len;
    let k = log_k.exp();
    let mut out = vec![0f32; frames * frame_len];
    out[..pcm.len()].copy_from_slice(pcm);
    for f in out.chunks_mut(frame_len) {
        let mean = f.iter().sum::<f32>() / frame_len as f32;
        f.iter_mut().for_each(|x| *x -= mean);
        let var = f.iter().map(|x| x * x).sum::<f32>() / frame_len as f32;
        let rms = (var + CMVN_EPS).sqrt();
        f.iter_mut().for_each(|x| *x = (k * *x / rms).asinh());
    }
    (out, valid)
}

/// Output length of one causal conv (left pad `k − 1`).
fn conv_out_len(t: usize) -> usize {
    (t - 1) / CONV_STRIDE + 1
}

/// Validity of each conv output: valid when any input under the kernel was
/// (the HF mask conv, `conv1d(pad(mask), ones) > 0`).
fn conv_mask(mask: &[bool]) -> Vec<bool> {
    (0..conv_out_len(mask.len()))
        .map(|t| {
            let end = t * CONV_STRIDE; // inclusive, in unpadded coordinates
            let start = end.saturating_sub(CONV_K - 1);
            mask[start..=end.min(mask.len() - 1)].iter().any(|&m| m)
        })
        .collect()
}

/// `[1, heads, t, t]` additive bias for a `(left, right)` window over the
/// first `live` keys.
fn window_bias(heads: usize, t: usize, live: &[bool], (left, right): (usize, usize)) -> Vec<f32> {
    let mut one = vec![MASKED; t * t];
    for q in 0..t {
        for k in 0..t {
            let seen = if k <= q { q - k < left } else { k - q < right };
            if seen && live[k] {
                one[q * t + k] = 0.0;
            }
        }
    }
    one.repeat(heads)
}

/// A loaded checkpoint.
pub struct MoonshineStreaming {
    cfg: StreamingConfig,
    device: Device,
    weights: WeightMap,
    pfx: MoonshineWeightPrefix,
    log_k: f32,
    /// `[vocab, d_dec]` token embedding (decoder input, host gather).
    embed: Vec<f32>,
    /// `[vocab, d_dec]` LM head (`proj_out`, untied).
    lm_head: Vec<f32>,
    encoders: HashMap<usize, CompiledGraph>,
    decoders: HashMap<(usize, usize), CompiledGraph>,
    #[cfg(feature = "tokenizer")]
    tokenizer: tokenizers::Tokenizer,
}

/// Encoder output for one utterance.
pub struct Encoded {
    /// After `final_norm`, `[enc_seq · d_enc]`.
    pub hidden: Vec<f32>,
    /// What the decoder cross-attends to: `proj(hidden + pos_emb)`, `[enc_seq · d_dec]`.
    pub decoder_states: Vec<f32>,
    pub enc_seq: usize,
}

impl MoonshineStreaming {
    /// Open a checkpoint directory (`config.json`, `model.safetensors`, `tokenizer.json`).
    pub fn open(dir: impl Into<PathBuf>, device: Device) -> Result<Self> {
        Self::open_source(&AssetSource::dir(dir.into()), device)
    }

    /// Open from any [`AssetSource`] — a directory, a `.rlxpack`, or in-memory
    /// files (the browser path) — under the checkpoint's own file names.
    pub fn open_source(src: &AssetSource, device: Device) -> Result<Self> {
        rlx_core::validate_browser_capable_device("moonshine-streaming", device)?;
        let cfg = StreamingConfig::from_json(&src.read_to_string("config.json")?)?;
        let weights = match SafetensorsSource::from_asset(src, "model.safetensors")? {
            SafetensorsSource::File(p) => {
                rlx_core::weights::open_map(&p)
                    .with_context(|| format!("moonshine-streaming: load {}", p.display()))?
                    .1
            }
            bytes => WeightMap::from_weight_loader(bytes.loader()?.as_mut())?,
        };
        let pfx = MoonshineWeightPrefix::detect(&weights);
        let get = |k: &str| -> Result<Vec<f32>> {
            weights
                .get(k)
                .map(|(d, _)| d.to_vec())
                .ok_or_else(|| anyhow!("moonshine-streaming: missing {k}"))
        };
        let log_k = get("model.encoder.embedder.comp.log_k")?[0];
        let embed = get(&pfx.dec_embed_tokens())?;
        let lm_head = match &pfx.proj_out {
            Some(k) => get(k)?,
            None => embed.clone(),
        };
        #[cfg(feature = "tokenizer")]
        let tokenizer = tokenizers::Tokenizer::from_bytes(src.read("tokenizer.json")?)
            .map_err(|e| anyhow!("moonshine-streaming: load tokenizer.json: {e}"))?;
        Ok(Self {
            cfg,
            device,
            weights,
            pfx,
            log_k,
            embed,
            lm_head,
            encoders: HashMap::new(),
            decoders: HashMap::new(),
            #[cfg(feature = "tokenizer")]
            tokenizer,
        })
    }

    pub fn config(&self) -> &StreamingConfig {
        &self.cfg
    }

    /// Run the encoder over 16 kHz mono PCM in `[-1, 1]`.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Encoded> {
        let job = self.prepare_encode(pcm)?;
        let g = self.encoders.get_mut(&job.n_frames).expect("prepared");
        job.finish(g.run(&job.inputs()))
    }

    /// Greedy decode; returns the token ids including the start token, and the
    /// end token when one was produced within `max_new_tokens`.
    pub fn generate(&mut self, enc: &Encoded, max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut seq = vec![self.cfg.decoder.decoder_start_token_id];
        while seq.len() < self.decode_limit(max_new_tokens) {
            let (key, embeds) = self.prepare_step(&seq, enc)?;
            let g = self.decoders.get_mut(&key).expect("prepared");
            let out = g.run(&[
                ("decoder_inputs_embeds", embeds.as_slice()),
                ("encoder_hidden", enc.decoder_states.as_slice()),
            ]);
            if self.finish_step(out, &mut seq)? {
                break;
            }
        }
        Ok(seq)
    }

    /// Transcribe 16 kHz mono PCM. `max_new_tokens` defaults to 6.5 tokens per
    /// second of audio (plus slack), the rate Moonshine's own runtime uses.
    #[cfg(feature = "tokenizer")]
    pub fn transcribe(&mut self, pcm: &[f32], max_new_tokens: Option<usize>) -> Result<String> {
        let enc = self.encode(pcm)?;
        let ids = self.generate(&enc, token_budget(pcm.len(), max_new_tokens))?;
        self.detokenize(&ids)
    }

    /// [`encode`](Self::encode) on wasm WebGPU, where reading results back
    /// from the GPU cannot block.
    #[cfg(target_arch = "wasm32")]
    pub async fn encode_async(&mut self, pcm: &[f32]) -> Result<Encoded> {
        let job = self.prepare_encode(pcm)?;
        let g = self.encoders.get_mut(&job.n_frames).expect("prepared");
        let out = g.run_async(&job.inputs()).await.map_err(|e| anyhow!(e))?;
        job.finish(out)
    }

    /// [`generate`](Self::generate) on wasm WebGPU.
    #[cfg(target_arch = "wasm32")]
    pub async fn generate_async(
        &mut self,
        enc: &Encoded,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut seq = vec![self.cfg.decoder.decoder_start_token_id];
        while seq.len() < self.decode_limit(max_new_tokens) {
            let (key, embeds) = self.prepare_step(&seq, enc)?;
            let g = self.decoders.get_mut(&key).expect("prepared");
            let out = g
                .run_async(&[
                    ("decoder_inputs_embeds", embeds.as_slice()),
                    ("encoder_hidden", enc.decoder_states.as_slice()),
                ])
                .await
                .map_err(|e| anyhow!(e))?;
            if self.finish_step(out, &mut seq)? {
                break;
            }
        }
        Ok(seq)
    }

    /// [`transcribe`](Self::transcribe) on wasm WebGPU.
    #[cfg(all(target_arch = "wasm32", feature = "tokenizer"))]
    pub async fn transcribe_async(
        &mut self,
        pcm: &[f32],
        max_new_tokens: Option<usize>,
    ) -> Result<String> {
        let enc = self.encode_async(pcm).await?;
        let ids = self
            .generate_async(&enc, token_budget(pcm.len(), max_new_tokens))
            .await?;
        self.detokenize(&ids)
    }

    #[cfg(feature = "tokenizer")]
    fn detokenize(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("moonshine-streaming: detokenize: {e}"))
    }

    /// Encoder inputs for `pcm`, compiling the encoder for its length if needed.
    fn prepare_encode(&mut self, pcm: &[f32]) -> Result<EncodeJob> {
        let fl = self.cfg.encoder.frame_len;
        if pcm.len() < fl {
            bail!("moonshine-streaming: need at least {fl} samples");
        }
        let (frames, valid) = frontend(pcm, fl, self.log_k);
        let n_frames = frames.len() / fl;
        let m0: Vec<bool> = (0..n_frames).map(|i| i < valid).collect();
        let m1 = conv_mask(&m0);
        let m2 = conv_mask(&m1);
        let t = m2.len();
        let f32s = |m: &[bool]| -> Vec<f32> { m.iter().map(|&b| f32::from(u8::from(b))).collect() };

        let heads = self.cfg.encoder.num_attention_heads;
        let biases: Vec<(String, Vec<f32>)> = self
            .distinct_windows()
            .iter()
            .map(|&w| (window_input(w), window_bias(heads, t, &m2, w)))
            .collect();

        if !self.encoders.contains_key(&n_frames) {
            // Encoders are compiled per input length; on wasm32 keep one
            // (memory never shrinks there).
            if cfg!(target_arch = "wasm32") {
                self.encoders.clear();
            }
            let g = self.compile_encoder(n_frames)?;
            self.encoders.insert(n_frames, g);
        }
        Ok(EncodeJob {
            n_frames,
            enc_seq: t,
            frames,
            masks: [f32s(&m0), f32s(&m1), f32s(&m2)],
            biases,
        })
    }

    fn decode_limit(&self, max_new_tokens: usize) -> usize {
        (max_new_tokens + 1).min(self.cfg.decoder.max_position_embeddings)
    }

    /// Decoder graph key and input embeddings for the next step after `seq`.
    fn prepare_step(&mut self, seq: &[u32], enc: &Encoded) -> Result<((usize, usize), Vec<f32>)> {
        let d = self.cfg.decoder.hidden_size;
        let cap = seq.len().div_ceil(DECODER_BUCKET) * DECODER_BUCKET;
        let key = (cap, enc.enc_seq);
        if !self.decoders.contains_key(&key) {
            if cfg!(target_arch = "wasm32") {
                self.decoders.clear();
            }
            let g = self.compile_decoder(cap, enc.enc_seq)?;
            self.decoders.insert(key, g);
        }
        let mut embeds = vec![0f32; cap * d];
        for (i, &tok) in seq.iter().enumerate() {
            let row = tok as usize * d;
            embeds[i * d..(i + 1) * d].copy_from_slice(&self.embed[row..row + d]);
        }
        Ok((key, embeds))
    }

    /// Append the argmax token; true when it ends the sequence.
    fn finish_step(&self, out: Vec<Vec<f32>>, seq: &mut Vec<u32>) -> Result<bool> {
        let d = self.cfg.decoder.hidden_size;
        let hidden = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("decoder: no output"))?;
        let cur = seq.len();
        let next = argmax_row(&self.lm_head, &hidden[(cur - 1) * d..cur * d]);
        seq.push(next);
        Ok(next == self.cfg.decoder.eos_token_id)
    }

    fn distinct_windows(&self) -> Vec<(usize, usize)> {
        let mut w = self.cfg.encoder.sliding_windows.clone();
        w.sort_unstable();
        w.dedup();
        w
    }

    fn compile_encoder(&self, n_frames: usize) -> Result<CompiledGraph> {
        let mut src = CloningWeightSource(&self.weights);
        let built = build_encoder(&self.cfg, &mut src, &self.distinct_windows(), n_frames)?;
        rlx_core::flow_util::compile_built(built, self.device)
    }

    fn compile_decoder(&self, cap: usize, enc_seq: usize) -> Result<CompiledGraph> {
        let mut src = CloningWeightSource(&self.weights);
        let built = rlx_moonshine::flow::build_decoder_hidden_built(
            &self.cfg.decoder,
            &mut src,
            &self.pfx,
            1,
            cap,
            enc_seq,
        )?;
        rlx_core::flow_util::compile_built(built, self.device)
    }
}

/// Encoder inputs for one utterance, built by `MoonshineStreaming::prepare_encode`.
struct EncodeJob {
    n_frames: usize,
    enc_seq: usize,
    frames: Vec<f32>,
    /// Frame, conv1 and conv2 validity, as 0/1.
    masks: [Vec<f32>; 3],
    biases: Vec<(String, Vec<f32>)>,
}

impl EncodeJob {
    fn inputs(&self) -> Vec<(&str, &[f32])> {
        let mut v: Vec<(&str, &[f32])> = vec![
            ("frames", &self.frames),
            ("frame_mask", &self.masks[0]),
            ("conv1_mask", &self.masks[1]),
            ("conv2_mask", &self.masks[2]),
        ];
        v.extend(self.biases.iter().map(|(n, b)| (n.as_str(), b.as_slice())));
        v
    }

    fn finish(&self, out: Vec<Vec<f32>>) -> Result<Encoded> {
        let mut out = out.into_iter();
        let hidden = out
            .next()
            .ok_or_else(|| anyhow!("encoder: no hidden output"))?;
        let decoder_states = out
            .next()
            .ok_or_else(|| anyhow!("encoder: no decoder-state output"))?;
        Ok(Encoded {
            hidden,
            decoder_states,
            enc_seq: self.enc_seq,
        })
    }
}

/// 6.5 tokens per second of audio plus slack, unless the caller set a budget.
#[cfg(feature = "tokenizer")]
fn token_budget(samples: usize, max_new_tokens: Option<usize>) -> usize {
    max_new_tokens
        .unwrap_or_else(|| (samples as f64 / SAMPLE_RATE as f64 * 6.5).ceil() as usize + 8)
}

pub(crate) fn window_input((left, right): (usize, usize)) -> String {
    format!("attn_bias_{left}_{right}")
}

/// `argmax_v (W[v] · h)` over a row-major `[vocab, d]` head.
fn argmax_row(w: &[f32], h: &[f32]) -> u32 {
    let d = h.len();
    let mut best = (0usize, f32::NEG_INFINITY);
    for (v, row) in w.chunks_exact(d).enumerate() {
        let s: f32 = row.iter().zip(h).map(|(a, b)| a * b).sum();
        if s > best.1 {
            best = (v, s);
        }
    }
    best.0 as u32
}

/// Encoder graph: `frames [1, F, frame_len]` → (`hidden [1, T, d_enc]`,
/// `decoder_states [1, T, d_dec]`).
pub(crate) fn build_encoder(
    cfg: &StreamingConfig,
    w: &mut dyn rlx_flow::WeightSource,
    windows: &[(usize, usize)],
    n_frames: usize,
) -> Result<rlx_flow::BuiltModel> {
    let e = &cfg.encoder;
    let d = e.hidden_size;
    let t1 = conv_out_len(n_frames);
    let t2 = conv_out_len(t1);
    let f = DType::F32;
    let mut hir =
        HirModule::new("moonshine_streaming_encoder").with_fusion_policy(FusionPolicy::Direct);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();

    let frames = hir.input("frames", Shape::new(&[1, n_frames, e.frame_len], f));
    let frame_mask = hir.input("frame_mask", Shape::new(&[1, n_frames, 1], f));
    let conv1_mask = hir.input("conv1_mask", Shape::new(&[1, 1, t1], f));
    let conv2_mask = hir.input("conv2_mask", Shape::new(&[1, 1, t2], f));
    let mut bias_in = HashMap::new();
    for &win in windows {
        let name = window_input(win);
        let id = hir.input(&name, Shape::new(&[1, e.num_attention_heads, t2, t2], f));
        bias_in.insert(win, id);
    }

    let mut p = Params {
        hir: &mut hir,
        params: &mut params,
        w,
    };
    let pre = "model.encoder";

    // linear 80 → d, SiLU, zero frames past the real audio.
    let x = p.linear(frames, &format!("{pre}.embedder.linear.weight"), None)?;
    let x = p.g().silu(x);
    let x = p.g().mul(x, frame_mask);
    let x = p.g().transpose_(x, vec![0, 2, 1]); // [1, d, F]
    let x = p.causal_conv(x, &format!("{pre}.embedder.conv1"), d, 2 * d, n_frames)?;
    let x = p.g().mul(x, conv1_mask);
    let x = p.g().silu(x);
    let x = p.causal_conv(x, &format!("{pre}.embedder.conv2"), 2 * d, d, t1)?;
    let x = p.g().mul(x, conv2_mask);
    let mut x = p.g().transpose_(x, vec![0, 2, 1]); // [1, T, d]

    let nh = e.num_attention_heads;
    let hd = e.head_dim;
    for (l, win) in e.sliding_windows.iter().enumerate() {
        let lp = format!("{pre}.layers.{l}");
        let residual = x;
        let h = p.unit_ln(x, &format!("{lp}.input_layernorm.gamma"), d)?;
        let q = p.linear(h, &format!("{lp}.self_attn.q_proj.weight"), None)?;
        let k = p.linear(h, &format!("{lp}.self_attn.k_proj.weight"), None)?;
        let v = p.linear(h, &format!("{lp}.self_attn.v_proj.weight"), None)?;
        let attn = p.g().inner().mir(
            attention_kind_op(
                nh,
                hd,
                None,
                MaskKind::Bias,
                Some((hd as f32).powf(-0.5)),
                None,
            ),
            vec![q, k, v, bias_in[win]],
            Shape::new(&[1, t2, nh * hd], f),
        );
        let o = p.linear(attn, &format!("{lp}.self_attn.o_proj.weight"), None)?;
        x = p.g().add(residual, o);

        let residual = x;
        let h = p.unit_ln(x, &format!("{lp}.post_attention_layernorm.gamma"), d)?;
        let h = p.linear(
            h,
            &format!("{lp}.mlp.fc1.weight"),
            Some(&format!("{lp}.mlp.fc1.bias")),
        )?;
        let h = p.g().gelu(h);
        let h = p.linear(
            h,
            &format!("{lp}.mlp.fc2.weight"),
            Some(&format!("{lp}.mlp.fc2.bias")),
        )?;
        x = p.g().add(residual, h);
    }
    let hidden = p.unit_ln(x, &format!("{pre}.final_norm.gamma"), d)?;

    // Decoder side: learned positions (encoder width), then project.
    let (pos, pos_shape) = p.w.take("model.decoder.pos_emb.weight", false)?;
    let pos_rows = pos[..t2 * pos_shape[1]].to_vec();
    let pos_id = p.register("model.decoder.pos_emb.rows", pos_rows, &[1, t2, d]);
    let states = p.g().add(hidden, pos_id);
    let states = if p.w.has("model.decoder.proj.weight") {
        p.linear(states, "model.decoder.proj.weight", None)?
    } else {
        states
    };

    hir.outputs = vec![hidden, states];
    rlx_core::flow_util::built_from_hir_with_profile(hir, params, CompileProfile::encoder())
}

/// Param registration over a HIR module.
struct Params<'a> {
    hir: &'a mut HirModule,
    params: &'a mut HashMap<String, Vec<f32>>,
    w: &'a mut dyn rlx_flow::WeightSource,
}

impl Params<'_> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    fn register(&mut self, key: &str, data: Vec<f32>, dims: &[usize]) -> HirNodeId {
        let id = self.hir.param(key, Shape::new(dims, DType::F32));
        self.params.insert(key.to_string(), data);
        id
    }

    fn load(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let (data, shape) = self.w.take(key, transpose)?;
        Ok(self.register(key, data, &shape))
    }

    /// HF `nn.Linear` (`[out, in]` weight).
    fn linear(&mut self, x: HirNodeId, w: &str, b: Option<&str>) -> Result<HirNodeId> {
        let wt = self.load(w, true)?;
        let y = self.g().mm(x, wt);
        match b {
            Some(b) => {
                let bias = self.load(b, false)?;
                Ok(self.g().add(y, bias))
            }
            None => Ok(y),
        }
    }

    /// `LayerNorm(elementwise_affine=False) · (γ + 1)`.
    fn unit_ln(&mut self, x: HirNodeId, gamma_key: &str, d: usize) -> Result<HirNodeId> {
        let (g, _) = self.w.take(gamma_key, false)?;
        let g: Vec<f32> = g.iter().map(|v| v + 1.0).collect();
        let gamma = self.register(gamma_key, g, &[d]);
        let beta = self.register(&format!("{gamma_key}.zero_beta"), vec![0.0; d], &[d]);
        Ok(self.g().ln(x, gamma, beta, LN_EPS))
    }

    /// Causal Conv1d over `[1, c_in, t]`: left pad `k − 1`, stride 2, bias.
    fn causal_conv(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        t: usize,
    ) -> Result<HirNodeId> {
        let t_out = conv_out_len(t);
        // Time on the LAST axis (`[1, C, 1, T]`): WebGL lays a tensor out as
        // `rest × last_dim` texels, so `[1, C, T, 1]` became a `C·T × 1`
        // texture — 332 800 rows for 5 s of audio, past `MAX_TEXTURE_SIZE`.
        let x = self.g().reshape_(x, vec![1, c_in as i64, 1, t as i64]);
        let x = self.g().pad_(
            x,
            vec![[0, 0], [0, 0], [0, 0], [CONV_K - 1, 0]],
            PadMode::Constant(0.0),
        );
        let (wd, _) = self.w.take(&format!("{prefix}.weight"), false)?;
        let wk = format!("{prefix}.weight");
        let w = self.register(&wk, wd, &[c_out, c_in, 1, CONV_K]);
        let conv = self.g().add_node(
            Op::Conv {
                kernel_size: vec![1, CONV_K],
                stride: vec![1, CONV_STRIDE],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![x, w],
            Shape::new(&[1, c_out, 1, t_out], DType::F32),
        );
        let y = self.g().reshape_(conv, vec![1, c_out as i64, t_out as i64]);
        let (bd, _) = self.w.take(&format!("{prefix}.bias"), false)?;
        let b = self.register(&format!("{prefix}.bias"), bd, &[1, c_out, 1]);
        Ok(self.g().add(y, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_lengths_match_hf() {
        // HF `_get_feat_extract_output_lengths`: two `(n − 1) // 2 + 1`.
        for frames in [1usize, 2, 5, 80, 81, 1037] {
            let want = ((frames - 1) / 2 + 1 - 1) / 2 + 1;
            assert_eq!(conv_out_len(conv_out_len(frames)), want, "frames={frames}");
        }
    }

    #[test]
    fn window_bias_follows_left_right_bounds() {
        let live = vec![true; 8];
        let b = window_bias(1, 8, &live, (3, 2));
        let seen = |q: usize, k: usize| b[q * 8 + k] == 0.0;
        assert!(seen(4, 4) && seen(4, 2) && !seen(4, 1), "left: 0 ≤ q−k < 3");
        assert!(seen(4, 5) && !seen(4, 6), "right: 0 < k−q < 2");
        let causal = window_bias(1, 8, &live, (3, 0));
        assert_eq!(causal[4 * 8 + 5], MASKED, "right = 0 is causal");
    }

    #[test]
    fn frontend_masks_a_trailing_partial_frame() {
        let (frames, valid) = frontend(&vec![0.1; 170], 80, 0.0);
        assert_eq!((frames.len(), valid), (240, 2));
        assert_eq!(conv_mask(&[true, true, false]), vec![true, true]);
    }
}
