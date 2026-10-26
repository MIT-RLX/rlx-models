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

//! The engine: tokenize a request, run one bidirectional prefill, read the
//! hidden state at every marker through the readout.
//!
//! # Forward pass
//!
//! The backbone is Qwen3.5 with its full-attention layers made
//! **bidirectional** ([`rlx_qwen35::Qwen35Flow::bidirectional_attention`]);
//! the Gated DeltaNet layers stay causal. Sequences are right-padded to a
//! [`seq_bucket`] so a steady workload stops recompiling, and the
//! `attn_key_mask` input keeps the pads out of the bidirectional layers
//! (the causal ones cannot read forward). That makes a padded run equal to
//! an exactly-sized one, which `rlx-qwen35/tests/bidirectional_attention.rs`
//! pins.
//!
//! # Readout
//!
//! The GGUF stores the `[255, hidden]` readout as `output.weight`, zero
//! outside the 255 code tokens. Only those rows are ever needed, so they are
//! dequantized once at load and applied on the host to the few marker rows —
//! the 248k-wide LM head never runs.
//!
//! # Memory
//!
//! Weights stay packed (Q8_0 matmuls dequantize per block in-kernel), and
//! the embedding table is gathered row by row from the packed bytes, so the
//! resident cost is about the size of the file plus the compiled arenas.
//! Compiled shapes share one weight buffer where the backend can
//! ([`CompiledGraph::share_params_from`]); otherwise only one shape is kept,
//! since each would hold its own copy. Metal shares only externalized float
//! weights, never quantized ones, so the Q8_0 release runs single-slot
//! there and pays a compile (2-4 s) whenever the length bucket changes.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::ordered_json::Json;
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_qwen35::{ATTN_KEY_MASK, Qwen35Config, Qwen35Flow, Qwen35VisionEncoder, Qwen35Weights};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::protocol::{
    self, Escaper, LEGACY_MAX_PIXELS, MARKER, MAX_OPTIONS, MIN_PIXELS, PROTOCOL_VERSION, Request,
    RequestError, TEMPLATE,
};
use crate::vision::{self, Preprocessed, RgbImage};

/// Most compiled shapes kept at once (when they share one weight buffer).
pub const MAX_CACHED_SHAPES: usize = 6;

/// The compiled length for an `n`-token sequence: a power of two from 64 to
/// 512, then a multiple of 512. Padding costs compute (the pads run through
/// every layer) but each new length costs a compile (seconds on a GPU), and
/// a decision workload's lengths cluster, so a few coarse shapes win. Image
/// prompts run 100-2000 tokens; multiples of 1024 there padded a 1435-token
/// request to 2048, ~2 s of a 7 s Metal prefill.
pub fn seq_bucket(n: usize) -> usize {
    if n <= 512 {
        n.max(64).next_power_of_two()
    } else {
        n.div_ceil(512) * 512
    }
}

/// The `pplx_decider.*` protocol data a checkpoint carries.
#[derive(Debug, Clone, PartialEq)]
pub struct DeciderMeta {
    pub temperature: f64,
    /// Answer codes in option order (`A`, `B`, …, `AA`, …).
    pub codes: Vec<String>,
    /// Token id of each code; the readout's rows.
    pub token_ids: Vec<u32>,
    pub escape_tokens: Vec<String>,
    pub max_pixels: usize,
}

fn meta<'a>(file: &'a GgufFile, key: &str) -> Result<&'a MetaValue> {
    file.metadata
        .get(&format!("pplx_decider.{key}"))
        .ok_or_else(|| {
            anyhow!("GGUF has no pplx_decider.{key}; is this a pplx-decider checkpoint?")
        })
}

fn meta_str(file: &GgufFile, key: &str) -> Result<String> {
    meta(file, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("pplx_decider.{key} is not a string"))
}

fn meta_strs(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    meta(file, key)?
        .as_array()
        .ok_or_else(|| anyhow!("pplx_decider.{key} is not an array"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("pplx_decider.{key} holds a non-string"))
        })
        .collect()
}

impl DeciderMeta {
    /// Read and check the protocol data in a GGUF.
    ///
    /// Refuses anything but protocol v3 with a non-causal backbone: an
    /// earlier protocol renders a different prompt, and a causal pass over
    /// this checkpoint returns plausible but wrong probabilities.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let version = meta(file, "protocol_version")?
            .as_u32()
            .ok_or_else(|| anyhow!("pplx_decider.protocol_version is not an integer"))?;
        if version != PROTOCOL_VERSION {
            bail!("pplx_decider protocol v{version}; only v{PROTOCOL_VERSION} is implemented");
        }
        let template = meta_str(file, "template")?;
        if template != TEMPLATE {
            bail!("pplx_decider template {template:?}; expected {TEMPLATE:?}");
        }
        let marker = meta_str(file, "marker")?;
        if marker != MARKER {
            bail!("pplx_decider marker {marker:?}; expected {MARKER:?}");
        }
        if Qwen35Config::gguf_attention_causal(file) {
            bail!(
                "this GGUF declares causal attention, but protocol v3 was trained \
                 with non-causal full-attention layers"
            );
        }
        // Stored as an F64; never default it — 1.0 keeps every argmax and
        // silently miscalibrates every probability.
        let temperature = match meta(file, "temperature")? {
            MetaValue::F64(t) => *t,
            MetaValue::F32(t) => f64::from(*t),
            other => bail!("pplx_decider.temperature is {other:?}"),
        };
        if !temperature.is_finite() || temperature <= 0.0 {
            bail!("pplx_decider.temperature must be positive and finite, got {temperature}");
        }
        let codes = meta_strs(file, "codes")?;
        let token_ids = meta(file, "token_ids")?
            .as_array()
            .ok_or_else(|| anyhow!("pplx_decider.token_ids is not an array"))?
            .iter()
            .map(|v| {
                v.as_u32()
                    .ok_or_else(|| anyhow!("pplx_decider.token_ids holds a non-integer"))
            })
            .collect::<Result<Vec<_>>>()?;
        if codes.len() != MAX_OPTIONS || token_ids.len() != codes.len() {
            bail!(
                "pplx_decider has {} codes and {} token ids; expected {MAX_OPTIONS} of each",
                codes.len(),
                token_ids.len()
            );
        }
        let escape_tokens = match meta_strs(file, "escape_tokens") {
            Ok(t) => t,
            Err(_) => protocol::DEFAULT_ESCAPE_TOKENS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };
        let max_pixels = meta(file, "max_pixels")
            .ok()
            .and_then(MetaValue::as_u64)
            .map_or(LEGACY_MAX_PIXELS, |v| v as usize);
        Ok(Self {
            temperature,
            codes,
            token_ids,
            escape_tokens,
            max_pixels,
        })
    }
}

/// Raw readout logits of one question's valid options (before temperature).
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionLogits {
    pub id: String,
    pub logits: Vec<f32>,
}

/// One tokenized sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// Token ids, each image's `<|image_pad|>` expanded to its token count.
    pub input_ids: Vec<u32>,
    /// Position of each question's marker, in question order.
    pub markers: Vec<usize>,
    /// Merged grid `(rows, cols)` of each image, in prompt order.
    pub image_grids: Vec<(usize, usize)>,
    /// Which positions hold image tokens.
    pub is_image: Vec<bool>,
}

impl Encoded {
    /// M-RoPE positions `[t, h, w, 0]` of every token.
    pub fn positions(&self) -> Result<Vec<[usize; 4]>> {
        vision::mrope_positions(&self.is_image, &self.image_grids)
    }
}

/// A request's sequences, plus its images preprocessed once for all of them.
#[derive(Debug, Clone)]
pub struct Plan {
    pub images: Vec<Preprocessed>,
    pub chunks: Vec<(Request, Encoded)>,
}

/// Options for [`PplxDecider::open_with`].
#[derive(Debug, Clone, Default)]
pub struct OpenOptions {
    /// `tokenizer.json` to use instead of the one embedded in the GGUF.
    pub tokenizer: Option<PathBuf>,
    /// Per-sequence token limit; longer requests are split into question
    /// chunks, each with the full state. Default: the backbone's window.
    pub max_length: Option<usize>,
    /// Vision projector GGUF. Default: the one `mmproj-*.gguf` next to the
    /// backbone, loaded on the first request with images.
    pub mmproj: Option<PathBuf>,
    /// Directory relative image paths resolve against (default: the
    /// working directory, as the reference does).
    pub image_base: Option<PathBuf>,
    /// Image pixel cap. Default: the checkpoint's `max_pixels`.
    pub max_pixels: Option<usize>,
}

/// A loaded pplx-decider checkpoint.
pub struct PplxDecider {
    tokenizer: tokenizers::Tokenizer,
    meta: DeciderMeta,
    escaper: Escaper,
    marker_id: u32,
    cfg: Qwen35Config,
    weights: Qwen35Weights,
    loader: GgufLoader,
    /// `[MAX_OPTIONS, hidden]`, rows in code order.
    readout: Vec<f32>,
    device: Device,
    max_length: usize,
    /// Compiled graphs by length, most recently used last. More than one
    /// only while they share the first one's weight buffer.
    compiled: Vec<(usize, CompiledGraph)>,
    image_pad_id: u32,
    mmproj: Option<PathBuf>,
    image_base: Option<PathBuf>,
    max_pixels: usize,
    /// Loaded on the first request with images.
    vision: Option<Qwen35VisionEncoder>,
    /// Whether compiled shapes could share one weight buffer on this
    /// device; `None` until the second shape is compiled.
    shares_weights: Option<bool>,
}

/// The single `mmproj*.gguf` beside `gguf`, if there is exactly one.
pub(crate) fn sibling_mmproj(gguf: &Path) -> Option<PathBuf> {
    let dir = gguf.parent()?;
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "gguf")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj"))
        })
        .collect();
    (found.len() == 1).then(|| found.remove(0))
}

/// The `.gguf` to load from `path`: the file itself, or the single backbone
/// GGUF in a directory (the `mmproj-*` vision projector is skipped).
pub fn resolve_gguf(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(path)
        .with_context(|| format!("reading {}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "gguf")
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj"))
        })
        .collect();
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => bail!("no backbone .gguf in {}", path.display()),
        _ => bail!(
            "several .gguf files in {}; pass one: {:?}",
            path.display(),
            found
        ),
    }
}

impl PplxDecider {
    /// Open a GGUF (or a directory holding one) on `device`.
    pub fn open(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::open_with(path, device, &OpenOptions::default())
    }

    pub fn open_with(path: impl AsRef<Path>, device: Device, opts: &OpenOptions) -> Result<Self> {
        let gguf = resolve_gguf(path.as_ref())?;
        let gguf_str = gguf
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 path {}", gguf.display()))?;
        let mut loader = GgufLoader::from_file(gguf_str)?;
        let meta = DeciderMeta::from_gguf(loader.file())?;
        let mut cfg = Qwen35Config::from_gguf(loader.file())?;
        // Qwen3.5 interleaves its M-RoPE sections (`t h w t h w …`, HF
        // `mrope_interleaved`). A GGUF does not record it — llama.cpp derives
        // it from the architecture — and `from_gguf` leaves it off. Text
        // rotates the same either way (t = h = w); image tokens do not, and
        // the contiguous layout moves a logit by 0.24 on the image suite.
        cfg.mrope_interleaved = true;
        if cfg.nextn_predict_layers != 0 {
            bail!("unexpected MTP layers in a decider checkpoint");
        }

        let tokenizer = match &opts.tokenizer {
            Some(p) => tokenizers::Tokenizer::from_file(p)
                .map_err(|e| anyhow!("{e}"))
                .with_context(|| format!("loading {}", p.display()))?,
            None => gguf_tokenizer(&gguf)?,
        };
        let marker_id = tokenizer
            .token_to_id(MARKER)
            .ok_or_else(|| anyhow!("tokenizer has no {MARKER}"))?;
        for (code, &id) in meta.codes.iter().zip(&meta.token_ids) {
            let enc = tokenizer
                .encode(code.as_str(), false)
                .map_err(|e| anyhow!("{e}"))?;
            if enc.get_ids() != [id] {
                bail!(
                    "answer code {code:?} tokenizes to {:?}, but the checkpoint reads \
                     token {id}; tokenizer and weights disagree",
                    enc.get_ids()
                );
            }
        }

        let readout = read_readout(&loader, &meta.token_ids, cfg.hidden_size)?;
        let device = rlx_core::flow_bridge::packed_gguf_execution_device(device);
        // The CPU backend's dequant cache keeps an f32 copy of every packed
        // weight it touches (~19 GB for this 4.8B model, per compiled shape,
        // since entries are keyed by arena offset). That pays off for decode,
        // where each weight is reused for thousands of one-row matmuls; a
        // decider only ever prefills, where the dequant is amortized over the
        // whole sequence. Measured: a 1.4k-token request 21.5 s / 5.3 GB
        // without it vs 24.8 s / 19.1 GB with it, and nine requests over
        // several lengths overflowed its 16 GB cap and thrashed for minutes.
        // An explicit `RLX_DEQUANT_CACHE` setting wins.
        if device == Device::Cpu && rlx_ir::env::var_result("RLX_DEQUANT_CACHE").is_err() {
            rlx_ir::env::set("RLX_DEQUANT_CACHE", "0");
        }
        let weights = Qwen35Weights::from_loader_packed(&mut loader, &cfg)?;
        let escaper = Escaper::new(&meta.escape_tokens);
        let max_length = opts.max_length.unwrap_or(cfg.max_position_embeddings);
        let image_pad_id = tokenizer
            .token_to_id("<|image_pad|>")
            .ok_or_else(|| anyhow!("tokenizer has no <|image_pad|>"))?;
        let mmproj = opts.mmproj.clone().or_else(|| sibling_mmproj(&gguf));
        let max_pixels = opts.max_pixels.unwrap_or(meta.max_pixels);
        Ok(Self {
            tokenizer,
            meta,
            escaper,
            marker_id,
            cfg,
            weights,
            loader,
            readout,
            device,
            max_length,
            compiled: Vec::new(),
            image_pad_id,
            mmproj,
            image_base: opts.image_base.clone(),
            max_pixels,
            vision: None,
            shares_weights: None,
        })
    }

    pub fn meta(&self) -> &DeciderMeta {
        &self.meta
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Validate a request against this checkpoint.
    pub fn request(&self, value: &Json) -> Result<Request> {
        Request::from_json(value, self.meta.codes.len())
    }

    /// The rendered prompt of a request.
    pub fn prompt_text(&self, req: &Request) -> String {
        protocol::prompt_text(req, &self.meta.codes, &self.escaper)
    }

    /// Decode and preprocess a request's images (the processor's pixels).
    pub fn prepare_images(&self, req: &Request) -> Result<Vec<Preprocessed>> {
        req.images
            .iter()
            .map(|entry| {
                let img = RgbImage::open(entry, self.image_base.as_deref())?;
                vision::preprocess(&img, MIN_PIXELS, self.max_pixels)
            })
            .collect()
    }

    /// Tokenize a request as one sequence.
    pub fn encode(&self, req: &Request) -> Result<Encoded> {
        let images = self.prepare_images(req)?;
        self.encode_prepared(req, &images)
    }

    /// [`Self::encode`] with the images already preprocessed: each
    /// `<|image_pad|>` becomes as many tokens as its image has merged
    /// patches, which is what the processor does to the text.
    pub fn encode_prepared(&self, req: &Request, images: &[Preprocessed]) -> Result<Encoded> {
        let text = self.prompt_text(req);
        let enc = self
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow!("{e}"))?;
        let mut input_ids = Vec::with_capacity(enc.len());
        let mut is_image = Vec::with_capacity(enc.len());
        let mut next = images.iter();
        for &t in enc.get_ids() {
            if t == self.image_pad_id {
                let img = next
                    .next()
                    .ok_or_else(|| anyhow!("more image placeholders than images"))?;
                input_ids.extend(std::iter::repeat_n(t, img.n_tokens()));
                is_image.extend(std::iter::repeat_n(true, img.n_tokens()));
            } else {
                input_ids.push(t);
                is_image.push(false);
            }
        }
        if next.next().is_some() {
            bail!("fewer image placeholders than images");
        }
        let markers: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.marker_id).then_some(i))
            .collect();
        if markers.len() != req.questions.len() {
            bail!(
                "{} readout markers for {} questions after tokenization",
                markers.len(),
                req.questions.len()
            );
        }
        Ok(Encoded {
            input_ids,
            markers,
            image_grids: images.iter().map(|i| (i.grid_h, i.grid_w)).collect(),
            is_image,
        })
    }

    /// `Encoder.plan`: one sequence, or question chunks when the request is
    /// longer than `max_length`. All chunks are encoded before any forward
    /// pass, so an over-length single question fails the whole request.
    pub fn plan(&self, req: &Request) -> Result<Plan> {
        fn split(
            me: &PplxDecider,
            req: &Request,
            images: &[Preprocessed],
            out: &mut Vec<(Request, Encoded)>,
        ) -> Result<()> {
            let enc = me.encode_prepared(req, images)?;
            if enc.input_ids.len() <= me.max_length {
                out.push((req.clone(), enc));
                return Ok(());
            }
            let n = req.questions.len();
            if n == 1 {
                bail!(RequestError(format!(
                    "Question {:?} needs {} tokens with the state and images, over the \
                     {}-token limit; shorten the state, lower the image resolution or \
                     raise max_length",
                    req.questions[0].id,
                    enc.input_ids.len(),
                    me.max_length
                )));
            }
            split(me, &req.subset(0..n / 2), images, out)?;
            split(me, &req.subset(n / 2..n), images, out)
        }
        let images = self.prepare_images(req)?;
        let mut chunks = Vec::new();
        split(self, req, &images, &mut chunks)?;
        Ok(Plan { images, chunks })
    }

    /// The vision tower's output for each image, `[n_tokens, hidden]` rows
    /// in merged-token order.
    pub fn vision_embeddings(&mut self, images: &[Preprocessed]) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let hidden = self.cfg.hidden_size;
        if self.vision.is_none() {
            let path = self.mmproj.clone().ok_or_else(|| {
                RequestError(
                    "this request has images, but no vision projector was found; pass \
                     --mmproj (mmproj-decider-*.gguf ships with the GGUF release)"
                        .into(),
                )
            })?;
            // Fused vision attention unless the caller chose: measured on the
            // decider's own image sizes it is the faster option on every
            // backend tried (Metal 2.1 s vs 3.0 s and MLX 35 s vs 57 s for a
            // 1408x896 image, MLX 52 GB vs 95 GB peak), while rlx-qwen35
            // defaults GPUs to the decomposed form.
            if rlx_ir::env::var_result("RLX_QWEN35_VISION_SDPA_DECOMP").is_err() {
                rlx_ir::env::set("RLX_QWEN35_VISION_SDPA_DECOMP", "0");
            }
            let enc = Qwen35VisionEncoder::load(&path, self.device)
                .with_context(|| format!("loading {}", path.display()))?;
            let c = enc.config();
            if c.llm_hidden_size != hidden
                || c.patch_size != vision::PATCH
                || c.n_merge != vision::MERGE
            {
                bail!(
                    "{} projects to {} with patch {} / merge {}; the backbone needs \
                     {hidden} with patch {} / merge {}",
                    path.display(),
                    c.llm_hidden_size,
                    c.patch_size,
                    c.n_merge,
                    vision::PATCH,
                    vision::MERGE
                );
            }
            self.vision = Some(enc);
        }
        let enc = self.vision.as_mut().expect("loaded above");
        images
            .iter()
            .map(|img| {
                let out = enc.encode_nchw(&img.nchw, img.resized.width, img.resized.height)?;
                if out.n_tokens != img.n_tokens() || out.embeddings.len() != img.n_tokens() * hidden
                {
                    bail!(
                        "vision tower returned {} tokens for a {}x{} grid",
                        out.n_tokens,
                        img.grid_h,
                        img.grid_w
                    );
                }
                Ok(out.embeddings)
            })
            .collect()
    }

    /// Raw readout logits of every question's valid options.
    pub fn logits(&mut self, req: &Request) -> Result<Vec<QuestionLogits>> {
        Ok(self.run(req)?.0)
    }

    /// Answer one request: `{"answers": {...}, "usage": {...}}`, the same
    /// JSON the reference `Decider.decide` returns. `temperature` overrides
    /// the calibrated one.
    pub fn decide(&mut self, req: &Request, temperature: Option<f64>) -> Result<Json> {
        let t = temperature.unwrap_or(self.meta.temperature);
        if !t.is_finite() || t <= 0.0 {
            bail!("temperature must be positive and finite");
        }
        let (logits, input_tokens, sequences) = self.run(req)?;
        let answers = req
            .questions
            .iter()
            .zip(&logits)
            .map(|(q, l)| {
                let p = protocol::probabilities(&l.logits, t);
                (q.id.clone(), protocol::answer(q, &p))
            })
            .collect();
        Ok(Json::Object(vec![
            ("answers".into(), Json::Object(answers)),
            (
                "usage".into(),
                Json::Object(vec![
                    ("input_tokens".into(), Json::UInt(input_tokens as u64)),
                    ("output_tokens".into(), Json::UInt(0)),
                    ("sequences".into(), Json::UInt(sequences as u64)),
                ]),
            ),
        ]))
    }

    /// `(logits per question in request order, input tokens, sequences)`.
    fn run(&mut self, req: &Request) -> Result<(Vec<QuestionLogits>, usize, usize)> {
        let plan = self.plan(req)?;
        let vision = self.vision_embeddings(&plan.images)?;
        let mut by_id = std::collections::HashMap::new();
        let mut tokens = 0usize;
        for (chunk, enc) in &plan.chunks {
            tokens += enc.input_ids.len();
            let rows = self.marker_hidden(enc, &vision)?;
            let h = self.cfg.hidden_size;
            for (k, q) in chunk.questions.iter().enumerate() {
                let hid = &rows[k * h..(k + 1) * h];
                let logits = (0..q.keys.len())
                    .map(|o| dot(&self.readout[o * h..(o + 1) * h], hid))
                    .collect();
                by_id.insert(q.id.clone(), logits);
            }
        }
        let out = req
            .questions
            .iter()
            .map(|q| QuestionLogits {
                id: q.id.clone(),
                logits: by_id.remove(&q.id).expect("every question planned"),
            })
            .collect();
        Ok((out, tokens, plan.chunks.len()))
    }

    /// Final-norm hidden states at the markers, `[n_markers, hidden]`.
    ///
    /// The natural bisect point against the reference: it separates "the
    /// trunk is wrong" from "the readout is wrong". `vision` holds each
    /// image's [`Self::vision_embeddings`], which replace the image tokens'
    /// embeddings in order.
    pub fn marker_hidden(&mut self, enc: &Encoded, vision: &[Vec<f32>]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let n = enc.input_ids.len();
        let seq = seq_bucket(n);
        let positions = enc.positions()?;
        if vision.len() != enc.image_grids.len() {
            bail!(
                "{} vision embeddings for {} images",
                vision.len(),
                enc.image_grids.len()
            );
        }
        self.ensure_compiled(seq)?;

        let mut embeds = vec![0f32; seq * h];
        self.embed_into(&enc.input_ids, &mut embeds[..n * h])?;
        let mut rows = vision.iter().flat_map(|v| v.chunks_exact(h));
        for (i, _) in enc.is_image.iter().enumerate().filter(|(_, b)| **b) {
            let row = rows.next().ok_or_else(|| anyhow!("too few vision rows"))?;
            embeds[i * h..(i + 1) * h].copy_from_slice(row);
        }
        if rows.next().is_some() {
            bail!("more vision rows than image tokens");
        }
        let mut ids = vec![0f32; seq];
        for (d, &t) in ids.iter_mut().zip(&enc.input_ids) {
            *d = t as f32;
        }
        let mut mask = vec![0f32; seq];
        mask[..n].fill(1.0);
        // `rope_dim_count / 2`, not `key_length / 2`: Qwen3.5 rotates 64 of
        // 256 head dims, and the wider stride still returns plausible output.
        let rot_half = self.cfg.rope_dim_count / 2;
        let (cos, sin) =
            rlx_qwen35::mrope_prefill_feeds(&self.cfg, seq, Some(&positions), rot_half);

        let (_, graph) = self.compiled.last_mut().expect("compiled above");
        let out = graph.run(&[
            ("input_ids", &ids[..]),
            ("hidden", &embeds[..]),
            ("rope_cos", &cos[..]),
            ("rope_sin", &sin[..]),
            (ATTN_KEY_MASK, &mask[..]),
        ]);
        let hidden = out
            .first()
            .ok_or_else(|| anyhow!("prefill graph produced no outputs"))?;
        if hidden.len() != seq * h {
            bail!(
                "expected [1, {seq}, {h}] hidden values, got {}",
                hidden.len()
            );
        }
        let mut rows = Vec::with_capacity(enc.markers.len() * h);
        for &m in &enc.markers {
            rows.extend_from_slice(&hidden[m * h..(m + 1) * h]);
        }
        Ok(rows)
    }

    fn embed_into(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        let h = self.cfg.hidden_size;
        if self.weights.embed_is_lazy() {
            for (row, &id) in out.chunks_exact_mut(h).zip(ids) {
                self.weights
                    .embed_row_into(Some(&self.loader as &dyn WeightLoader), id, row)?;
            }
        } else {
            let table = self.weights.token_embd();
            for (row, &id) in out.chunks_exact_mut(h).zip(ids) {
                let i = id as usize;
                row.copy_from_slice(&table[i * h..(i + 1) * h]);
            }
        }
        Ok(())
    }

    /// Make the graph for `seq` the last entry of `self.compiled`.
    fn ensure_compiled(&mut self, seq: usize) -> Result<()> {
        if let Some(i) = self.compiled.iter().position(|(s, _)| *s == seq) {
            let hit = self.compiled.remove(i);
            self.compiled.push(hit);
            return Ok(());
        }
        if self.compiled.len() >= MAX_CACHED_SHAPES {
            self.compiled.remove(0);
        }
        // Where shapes cannot share weights, the next shape replaces the
        // current one anyway — drop it before compiling, not after, or the
        // switch holds two full weight copies at once.
        if self.shares_weights == Some(false) {
            self.compiled.clear();
        }
        let (built, packed) = Qwen35Flow::new(&self.cfg)
            .weights(&self.weights)
            .batch(1)
            .seq(seq)
            .runtime_mrope()
            .export_normed_hidden()
            .bidirectional_attention(true)
            .build_with_weights(&self.weights)
            .with_context(|| format!("building the decider graph at seq {seq}"))?;
        let (hir, params) = built.into_parts()?;
        let device = self.device;
        let profile = rlx_qwen35::qwen35_profile_default(false);
        let opts = rlx_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &profile, device,
        );
        let mut graph = rlx_core::flow_bridge::packed_gguf_compile_guard(device, || {
            Session::new(device).compile_hir_with(hir, &opts)
        })
        .with_context(|| format!("compiling the decider graph at seq {seq}"))?;
        // The weight layout does not depend on the length, so a cached shape
        // can lend its buffer instead of a second upload.
        let shared = match self.compiled.first() {
            Some((_, donor)) => {
                let ok = graph.share_params_from(donor);
                self.shares_weights = Some(ok);
                ok
            }
            None => false,
        };
        if shared {
            self.compiled.push((seq, graph));
            return Ok(());
        }
        // No sharing here: keep one shape rather than one weight copy each.
        self.compiled.clear();
        for (name, data) in params {
            graph.set_param(&name, &data);
        }
        // Stream each packed tensor through one scratch buffer instead of
        // borrowing the mmap, which would fault the whole file in and keep a
        // second copy resident (see rlx-qwen35's `upload_packed_opt`).
        let mut scratch = Vec::new();
        for (param, (key, _, _)) in &packed {
            if self.loader.read_tensor_bytes_into(key, &mut scratch)? {
                graph.set_param_typed(param, &scratch, rlx_ir::DType::U8);
            } else {
                let bytes = self
                    .loader
                    .tensor_bytes_borrowed(key)
                    .ok_or_else(|| anyhow!("packed tensor {key} has no bytes"))?;
                graph.set_param_typed(param, bytes, rlx_ir::DType::U8);
            }
        }
        self.compiled.push((seq, graph));
        Ok(())
    }

    /// Lengths currently compiled, least recently used first.
    pub fn compiled_lengths(&self) -> Vec<usize> {
        self.compiled.iter().map(|(s, _)| *s).collect()
    }
}

/// The tokenizer embedded in the GGUF, with the processor's pre-tokenizer.
///
/// The release's GGUF says `tokenizer.ggml.pre = qwen35`, whose split keeps
/// combining marks (`\p{M}`) inside words, but the checkpoint's own
/// `tokenizer.json` — what the model was trained and evaluated with — splits
/// with Qwen2's `\p{L}+`. They disagree on text with marks NFC does not
/// compose (Devanagari, Thai, …); follow the processor.
pub fn gguf_tokenizer(gguf: &Path) -> Result<tokenizers::Tokenizer> {
    let json = rlx_qwen35::gguf_tokenizer_json(gguf)?;
    let mut tok = tokenizers::Tokenizer::from_bytes(&json).map_err(|e| anyhow!("{e}"))?;
    tok.with_pre_tokenizer(Some(rlx_qwen35::qwen_pre_tokenizer(
        rlx_qwen35::QWEN2_SPLIT_REGEX,
    )?));
    Ok(tok)
}

/// Dequantize the readout's rows (`output.weight` at the code tokens).
pub(crate) fn read_readout(
    loader: &GgufLoader,
    token_ids: &[u32],
    hidden: usize,
) -> Result<Vec<f32>> {
    let t = loader
        .file()
        .tensors
        .get("output.weight")
        .ok_or_else(|| anyhow!("GGUF has no output.weight (the readout)"))?;
    let vocab = t.shape.get(1).copied().unwrap_or(0);
    if t.shape.first() != Some(&hidden) {
        bail!("output.weight is {:?}; expected [{hidden}, vocab]", t.shape);
    }
    let bytes = loader
        .tensor_bytes_borrowed("output.weight")
        .ok_or_else(|| anyhow!("output.weight has no bytes"))?;
    let row_bytes = rlx_gguf::bytes_for_public(t.dtype, hidden)
        .ok_or_else(|| anyhow!("no row size for {:?}", t.dtype))?;
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &id in token_ids {
        let id = id as usize;
        if id >= vocab {
            bail!("code token {id} is outside the {vocab}-row readout");
        }
        let row = &bytes[id * row_bytes..(id + 1) * row_bytes];
        out.extend(rlx_gguf::dequant_typed(
            t.dtype,
            row,
            hidden,
            "output.weight",
        )?);
    }
    Ok(out)
}

/// f64 accumulation: the readout is 2560 terms, and the reference's f32
/// matmul is itself a blocked sum, so neither order is "the" answer; f64
/// keeps this side's error far below the trunk's.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>() as f32
}

#[cfg(test)]
mod tests {
    use super::seq_bucket;

    #[test]
    fn buckets() {
        assert_eq!(seq_bucket(1), 64);
        assert_eq!(seq_bucket(64), 64);
        assert_eq!(seq_bucket(65), 128);
        assert_eq!(seq_bucket(512), 512);
        assert_eq!(seq_bucket(567), 1024);
        assert_eq!(seq_bucket(1024), 1024);
        assert_eq!(seq_bucket(1025), 1536);
        assert_eq!(seq_bucket(1435), 1536);
        assert_eq!(seq_bucket(3000), 3072);
    }
}
