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

//! [`LiquidCodec`]: the loaded checkpoint, its compiled graphs, encode and
//! decode (whole clips or streamed).

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use rlx_core::audio_codec::{AudioCodec, CodecInfo, RvqCodes};
use rlx_lfm3_audio::Istft;
use rlx_lfm3_audio::graph::{self, Gb, WeightStorage};
use rlx_lfm3_audio::trunk::{self, StepState, TrunkSpec};
use rlx_runtime::{CompiledGraph, Device};

use crate::config::LiquidCodecConfig;
use crate::weights::CodecWeights;
use crate::{fsq, graphs, stft};

/// Codec frames per encoder bucket (2 s); graphs are built per bucket size.
const ENCODER_BUCKET: usize = 25;
/// Codec frames per decoder bucket.
const DECODER_BUCKET: usize = 25;
/// New frames per decoder graph run; longer inputs are decoded in pieces.
const DECODER_CHUNK: usize = 250;
/// Additive attention bias of a hidden key in the streaming step.
const MASKED: f32 = -1e9;
/// Clips up to this many frames (30 s) encode in one graph; longer ones in
/// KV-cached chunks, so encoder memory grows linearly, not quadratically.
const ENCODER_WHOLE_MAX: usize = 375;
/// Frames per KV-cached encoder chunk.
const ENCODER_CHUNK: usize = 64;
/// Encoder attention caches grow in steps of this many frames.
const ENCODER_KV_ROUND: usize = 256;

/// The one-frame streaming decoder graph and the caches it carries.
struct StepGraph {
    compiled: CompiledGraph,
    state: StepState,
    max_kv: usize,
}

/// LiquidCodec: 24 kHz mono speech ↔ `n_q` tokens per 80 ms frame.
pub struct LiquidCodec {
    cfg: LiquidCodecConfig,
    weights: CodecWeights,
    encoder_spec: TrunkSpec,
    decoder_spec: TrunkSpec,
    /// Codec frames of history that fix every decoder output row.
    decoder_context: usize,
    device: Device,
    storage: WeightStorage,
    encoders: RefCell<HashMap<usize, CompiledGraph>>,
    decoders: RefCell<HashMap<usize, CompiledGraph>>,
    step: RefCell<Option<StepGraph>>,
    /// Encoder step graphs by `(frames, max_kv)`.
    encoder_steps: RefCell<HashMap<(usize, usize), StepGraph>>,
    max_whole_frames: usize,
}

impl LiquidCodec {
    /// Load a checkpoint: the `LiquidAI/Liquid-Codec` directory, an
    /// LFM3-Audio directory (its `audio_tokenizer/`), or a
    /// `model.safetensors` (with `config.json` beside it, else the released
    /// shape).
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let weights = CodecWeights::open(path)?;
        let dir = weights
            .path()
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let cfg = LiquidCodecConfig::load(&dir.join("config.json"))?;
        let encoder_spec = graphs::stack_spec(&weights, "enc.", None)?;
        let decoder_spec = graphs::stack_spec(&weights, "dec.", Some(cfg.decoder_window))?;
        let reach = graphs::reach(&decoder_spec).expect("the decoder attends in a window");
        let decoder_context = reach.div_ceil(cfg.stft_frame_stack);
        Ok(Self {
            cfg,
            weights,
            encoder_spec,
            decoder_spec,
            decoder_context,
            device,
            storage: graph::default_storage(),
            encoders: RefCell::default(),
            decoders: RefCell::default(),
            step: RefCell::default(),
            encoder_steps: RefCell::default(),
            max_whole_frames: ENCODER_WHOLE_MAX,
        })
    }

    pub fn config(&self) -> &LiquidCodecConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Codec frames of history a decoder row depends on (see
    /// [`graphs::reach`]); long clips decode in pieces with this much
    /// overlap.
    pub fn decoder_context(&self) -> usize {
        self.decoder_context
    }

    /// Store dense weights as `storage` in graphs built from now on (`Auto`
    /// keeps the bf16 checkpoint at 2 B/elem). Drops the compiled graphs.
    pub fn set_weight_storage(&mut self, storage: WeightStorage) {
        self.storage = storage;
        self.encoders.get_mut().clear();
        self.decoders.get_mut().clear();
        *self.step.get_mut() = None;
        self.encoder_steps.get_mut().clear();
    }

    /// Encode clips longer than `frames` codec frames in KV-cached chunks of
    /// 64 frames (default 375 = 30 s). One graph over a whole clip holds
    /// attention scores for every pair of frames; the chunks hold 64 rows of
    /// them, so memory grows with the clip length instead of its square.
    /// Both give the same latents up to float rounding.
    pub fn set_max_whole_frames(&mut self, frames: usize) {
        self.max_whole_frames = frames.max(1);
    }

    fn compile(
        &self,
        name: &str,
        build: impl FnOnce(&mut Gb) -> Result<()>,
    ) -> Result<CompiledGraph> {
        let mut w = self.weights.loader();
        let mut gb = Gb::new(name, &mut w, self.storage).dense_only(graph::dense_only(self.device));
        build(&mut gb)?;
        graph::compile(gb, self.device, self.weights.path())
    }

    /// The quantizer latents `[frames, n_q * d]` of `pcm` (24 kHz; the tail
    /// is zero-padded to a whole frame), before the FSQ bound.
    pub fn encode_latents(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            bail!("liquid-codec: empty audio");
        }
        let fs = self.cfg.frame_size();
        let frames = pcm.len().div_ceil(fs);
        let mut audio = pcm.to_vec();
        audio.resize(frames * fs, 0.0);
        let lat_dim = self.cfg.n_q * self.cfg.codebook_dim();
        let width = 3 * self.cfg.bins();
        let frame_feats = self.cfg.stft_frame_stack * width;
        if frames > self.max_whole_frames {
            // Features a chunk at a time: a long clip's whole STFT is the
            // largest buffer otherwise (346 MB at 10 minutes).
            let mut stft = stft::Stft::new(&self.cfg);
            let steps = audio.len() / self.cfg.stft_hop_length;
            let mut st = EncoderState::default();
            let mut lat =
                Vec::with_capacity(frames.div_ceil(ENCODER_CHUNK) * ENCODER_CHUNK * lat_dim);
            let mut feats = vec![0f32; ENCODER_CHUNK * frame_feats];
            for chunk in 0..frames.div_ceil(ENCODER_CHUNK) {
                let first = chunk * ENCODER_CHUNK * self.cfg.stft_frame_stack;
                for (i, row) in feats.chunks_exact_mut(width).enumerate() {
                    if first + i < steps {
                        stft.frame(first + i, |j| stft::reflect(&audio, j), row);
                    } else {
                        row.fill(0.0);
                    }
                }
                lat.extend(self.encoder_advance(&mut st, &feats, ENCODER_CHUNK, frames)?);
            }
            lat.truncate(frames * lat_dim);
            return Ok(lat);
        }
        let mut feats = stft::features(&audio, &self.cfg);
        let bucket = frames.div_ceil(ENCODER_BUCKET) * ENCODER_BUCKET;
        feats.resize(bucket * frame_feats, 0.0);
        let mut cache = self.encoders.borrow_mut();
        let g = match cache.entry(bucket) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(self.compile("liquid_codec_encoder", |gb| {
                graphs::encoder(gb, &self.cfg, &self.encoder_spec, bucket)
            })?),
        };
        let mut lat = g.run(&[("features", &feats)]).swap_remove(0);
        lat.truncate(frames * lat_dim);
        Ok(lat)
    }

    /// Run the encoder over `frames` more codec frames of `feats` from `st`,
    /// growing the caches to hold at least `capacity` frames (and always
    /// the ones written). Returns their latents.
    fn encoder_advance(
        &self,
        st: &mut EncoderState,
        feats: &[f32],
        frames: usize,
        capacity: usize,
    ) -> Result<Vec<f32>> {
        let need = (st.pos + frames).max(capacity);
        if need > st.max_kv {
            st.max_kv = need.max(2 * st.max_kv).div_ceil(ENCODER_KV_ROUND) * ENCODER_KV_ROUND;
            st.slots.resize(st.max_kv, None);
        }
        let mut cache = self.encoder_steps.borrow_mut();
        // A grown cache outdates the smaller graphs of this row count; each
        // holds its own copy of the weights.
        cache.retain(|&(f, kv), _| f != frames || kv >= st.max_kv);
        let g = match cache.entry((frames, st.max_kv)) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let mut state = None;
                let compiled = self.compile("liquid_codec_encoder_step", |gb| {
                    state = Some(graphs::encoder_step(
                        gb,
                        &self.cfg,
                        &self.encoder_spec,
                        frames,
                        st.max_kv,
                    )?);
                    Ok(())
                })?;
                e.insert(StepGraph {
                    compiled,
                    state: state.expect("built above"),
                    max_kv: st.max_kv,
                })
            }
        };
        // A grown cache keeps its rows: slot = position while nothing wraps.
        for (name, &size) in g.state.names.iter().zip(&g.state.sizes) {
            st.caches.entry(name.clone()).or_default().resize(size, 0.0);
        }
        let mut outs = run_step(
            g,
            &self.encoder_spec,
            &mut st.slots,
            &mut st.caches,
            st.pos,
            frames,
            ("features", feats),
        )?;
        st.pos += frames;
        Ok(outs.swap_remove(0))
    }

    /// A streaming encoder: push 24 kHz samples as they arrive, get the
    /// frames they complete. Codes match [`Self::encode`] of the whole
    /// signal.
    pub fn encode_stream(&self) -> EncodeStream<'_> {
        EncodeStream {
            codec: self,
            stft: stft::Stft::new(&self.cfg),
            samples: Vec::new(),
            base: 0,
            received: 0,
            next_row: 0,
            rows: Vec::new(),
            st: EncoderState::default(),
        }
    }

    /// Tokens `[frames][n_q]` of 24 kHz `pcm`, one frame per 1920 samples
    /// (the tail zero-padded). `audio_codes[0].T` of `LiquidCodecModel.encode`.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        Ok(fsq::encode(&self.encode_latents(pcm)?, &self.cfg))
    }

    /// The decoder output `[(codes.len() - context) * stack, 2 * bins]` for
    /// all but the first `context` frames of `codes` (which only set the
    /// state).
    fn spectrum(&self, codes: &[Vec<u32>], context: usize) -> Result<Vec<f32>> {
        let n = codes.len();
        let bucket = n.div_ceil(DECODER_BUCKET) * DECODER_BUCKET;
        let lat_dim = self.cfg.n_q * self.cfg.codebook_dim();
        let mut lat = fsq::decode(codes, &self.cfg)?;
        lat.resize(bucket * lat_dim, 0.0);
        let mut cache = self.decoders.borrow_mut();
        let g = match cache.entry(bucket) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(self.compile("liquid_codec_decoder", |gb| {
                graphs::decoder(gb, &self.cfg, &self.decoder_spec, bucket)
            })?),
        };
        let spec = g.run(&[("latents", &lat)]).swap_remove(0);
        let row = self.cfg.stft_frame_stack * 2 * self.cfg.bins();
        Ok(spec[context * row..n * row].to_vec())
    }

    /// Build the one-frame step graph on first use. Its attention caches are
    /// a ring of `window - 1` past positions plus the frame's own rows.
    fn step_graph(&self) -> Result<std::cell::RefMut<'_, StepGraph>> {
        let mut slot = self.step.borrow_mut();
        if slot.is_none() {
            let stack = self.cfg.stft_frame_stack;
            let max_kv = (self.cfg.decoder_window - 1 + stack).div_ceil(stack) * stack;
            let mut state = None;
            let compiled = self.compile("liquid_codec_decoder_step", |gb| {
                state = Some(graphs::decoder_step(
                    gb,
                    &self.cfg,
                    &self.decoder_spec,
                    1,
                    max_kv,
                )?);
                Ok(())
            })?;
            *slot = Some(StepGraph {
                compiled,
                state: state.expect("built above"),
                max_kv,
            });
        }
        Ok(std::cell::RefMut::map(slot, |s| {
            s.as_mut().expect("built above")
        }))
    }

    /// A streaming decoder: push frames as they arrive, get the samples no
    /// later frame changes. Each frame is one small graph run against the
    /// carried attention and convolution state.
    pub fn decode_stream(&self) -> DecodeStream<'_> {
        DecodeStream {
            codec: self,
            pos: 0,
            slots: Vec::new(),
            caches: HashMap::new(),
            istft: Istft::new(self.cfg.stft_n_fft, self.cfg.stft_hop_length),
        }
    }

    /// 24 kHz audio of tokens `[frames][n_q]`, 1920 samples per frame
    /// (`LiquidCodecModel.decode(...).audio_values[0]`). Clips longer than
    /// 250 frames run in pieces that overlap by [`Self::decoder_context`]
    /// frames, which gives the same rows as one graph over everything.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let mut istft = Istft::new(self.cfg.stft_n_fft, self.cfg.stft_hop_length);
        let mut out = Vec::new();
        let mut start = 0;
        while start < codes.len() {
            let end = (start + DECODER_CHUNK).min(codes.len());
            let context = start.min(self.decoder_context);
            let spec = self.spectrum(&codes[start - context..end], context)?;
            out.extend(istft.push(&spec));
            start = end;
        }
        out.extend(istft.finish());
        Ok(out)
    }
}

/// [`LiquidCodec::decode_stream`]: the decoder advanced one codec frame
/// (6 STFT steps) per graph run, with the attention keys and values of the
/// last `window - 1` steps in a ring and the short-convolution states
/// carried between runs, as `LiquidCodecStreamingCache` does. The inverse
/// STFT holds back the samples later frames still add to. The samples match
/// [`LiquidCodec::decode`] of all the frames at once.
pub struct DecodeStream<'a> {
    codec: &'a LiquidCodec,
    /// STFT steps decoded so far.
    pos: usize,
    /// The position held by each attention-cache slot.
    slots: Vec<Option<usize>>,
    caches: HashMap<String, Vec<f32>>,
    istft: Istft,
}

impl DecodeStream<'_> {
    /// Decode `codes` (any number of frames, including none) and return the
    /// finished samples: 1920 per frame, less the first 480 of the stream
    /// (and 960 held back until the next push or [`Self::finish`]).
    pub fn push(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let c = self.codec;
        let spec = &c.decoder_spec;
        let rows = c.cfg.stft_frame_stack;
        let mut g = c.step_graph()?;
        if self.caches.is_empty() {
            self.slots = vec![None; g.max_kv];
            for (name, &size) in g.state.names.iter().zip(&g.state.sizes) {
                self.caches.insert(name.clone(), vec![0.0; size]);
            }
        }
        let mut out = Vec::new();
        for frame in codes {
            let lat = fsq::decode(std::slice::from_ref(frame), &c.cfg)?;
            let mut outs = run_step(
                &mut g,
                spec,
                &mut self.slots,
                &mut self.caches,
                self.pos,
                rows,
                ("latents", &lat),
            )?;
            out.extend(self.istft.push(&outs.swap_remove(0)));
            self.pos += rows;
        }
        drop(g);
        Ok(out)
    }

    /// End the stream and return the remaining samples.
    pub fn finish(self) -> Vec<f32> {
        self.istft.finish()
    }
}

/// Positions, cache slots and carried tensors of an encoder run.
#[derive(Default)]
struct EncoderState {
    pos: usize,
    max_kv: usize,
    slots: Vec<Option<usize>>,
    caches: HashMap<String, Vec<f32>>,
}

/// One run of a [`trunk::step`] graph: `rows` positions from `pos`, written
/// to cache slot `position % max_kv` (a ring when positions outrun the
/// slots), each query seeing the slots whose position is in its window.
/// Feeds `input` and the carried state, stores the returned state and gives
/// back the other outputs.
fn run_step(
    g: &mut StepGraph,
    spec: &TrunkSpec,
    slots: &mut [Option<usize>],
    caches: &mut HashMap<String, Vec<f32>>,
    pos: usize,
    rows: usize,
    input: (&str, &[f32]),
) -> Result<Vec<Vec<f32>>> {
    let max_kv = g.max_kv;
    let mut feeds: Vec<(String, Vec<f32>)> = Vec::new();
    for (r, rope) in spec.ropes.iter().enumerate() {
        let (cos, sin) = trunk::rope_tables(*rope, spec.head_dim, pos..pos + rows);
        feeds.push((format!("rope{r}_cos"), cos));
        feeds.push((format!("rope{r}_sin"), sin));
    }
    let mut write = vec![0f32; max_kv * rows];
    let mut keep = vec![1f32; max_kv];
    for j in 0..rows {
        let s = (pos + j) % max_kv;
        write[s * rows + j] = 1.0;
        keep[s] = 0.0;
        slots[s] = Some(pos + j);
    }
    feeds.push(("kv_write".into(), write));
    feeds.push(("kv_keep".into(), keep));
    for w in spec.windows() {
        let seen =
            |q: usize, p: Option<usize>| p.is_some_and(|p| p <= q && w.is_none_or(|w| p + w > q));
        // One row takes a 0/1 key mask, several an additive bias.
        let m: Vec<f32> = if rows == 1 {
            slots.iter().map(|&p| seen(pos, p) as u8 as f32).collect()
        } else {
            (0..rows)
                .flat_map(|j| slots.iter().map(move |&p| (j, p)))
                .map(|(j, p)| if seen(pos + j, p) { 0.0 } else { MASKED })
                .collect()
        };
        feeds.push((trunk::mask_name(w), m));
    }
    let mut inputs: Vec<(&str, &[f32])> = vec![input];
    inputs.extend(feeds.iter().map(|(n, v)| (n.as_str(), v.as_slice())));
    for name in &g.state.names {
        inputs.push((name.as_str(), caches[name].as_slice()));
    }
    let mut outs = g.compiled.run(&inputs);
    let state = outs.split_off(outs.len() - g.state.names.len());
    for (name, next) in g.state.names.iter().zip(state) {
        caches.insert(name.clone(), next);
    }
    Ok(outs)
}

/// [`LiquidCodec::encode_stream`]. An STFT frame is computed once the
/// samples under its window have arrived, and a codec frame is encoded as
/// soon as its 6 STFT frames exist — 480 samples (20 ms) after the frame
/// ends. Only the stream's last frame needs the zero padding and right-edge
/// reflection, which [`Self::finish`] applies, so the codes match
/// [`LiquidCodec::encode`] of the whole signal.
pub struct EncodeStream<'a> {
    codec: &'a LiquidCodec,
    stft: stft::Stft,
    /// Samples from absolute index `base` on.
    samples: Vec<f32>,
    base: usize,
    received: usize,
    /// Next STFT frame to compute.
    next_row: usize,
    /// STFT features not yet encoded.
    rows: Vec<f32>,
    st: EncoderState,
}

impl EncodeStream<'_> {
    /// Add samples; returns the codes of the frames they complete.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        self.samples.extend_from_slice(pcm);
        self.received += pcm.len();
        let mut out = Vec::new();
        while self.stft.end_of(self.next_row) <= self.received {
            self.add_row(None, &mut out)?;
        }
        // Keep what the next window reads (and the start, for its reflection).
        let lo = (self.next_row * self.codec.cfg.stft_hop_length).saturating_sub(self.stft.pad());
        let lo = if self.next_row * self.codec.cfg.stft_hop_length < self.stft.pad() {
            0
        } else {
            lo
        };
        if lo > self.base {
            self.samples.drain(..lo - self.base);
            self.base = lo;
        }
        Ok(out)
    }

    /// Codes of the rest: the tail zero-padded to a whole frame and
    /// reflected at the end, as [`LiquidCodec::encode`] does.
    pub fn finish(mut self) -> Result<Vec<Vec<u32>>> {
        let mut out = Vec::new();
        if self.received == 0 {
            return Ok(out);
        }
        let fs = self.codec.cfg.frame_size();
        let total = self.received.div_ceil(fs) * fs;
        self.samples.resize(total - self.base, 0.0);
        let steps = total / self.codec.cfg.stft_hop_length;
        while self.next_row < steps {
            self.add_row(Some(total), &mut out)?;
        }
        Ok(out)
    }

    /// Compute STFT frame `next_row` (reflecting at `end` when given) and
    /// encode a codec frame when one is complete.
    fn add_row(&mut self, end: Option<usize>, out: &mut Vec<Vec<u32>>) -> Result<()> {
        let cfg = &self.codec.cfg;
        let width = 3 * cfg.bins();
        let at = self.rows.len();
        self.rows.resize(at + width, 0.0);
        let (samples, base) = (&self.samples, self.base as isize);
        let sample = |i: isize| -> f32 {
            let i = match end {
                Some(n) if i >= n as isize => 2 * (n as isize - 1) - i,
                _ => i,
            };
            samples[(i.abs() - base) as usize]
        };
        self.stft.frame(self.next_row, sample, &mut self.rows[at..]);
        self.next_row += 1;
        if self.rows.len() == cfg.stft_frame_stack * width {
            let lat = self.codec.encoder_advance(&mut self.st, &self.rows, 1, 0)?;
            out.extend(fsq::encode(&lat, cfg));
            self.rows.clear();
        }
        Ok(())
    }
}

impl AudioCodec for LiquidCodec {
    fn info(&self) -> CodecInfo {
        CodecInfo {
            sample_rate: self.cfg.sampling_rate,
            frame_rate: self.cfg.frame_rate(),
            hop_length: self.cfg.frame_size(),
            channels: 1,
            max_quantizers: self.cfg.n_q,
            codebook_size: self.cfg.codebook_size(),
        }
    }

    fn device(&self) -> Device {
        self.device
    }

    /// Every frame carries all `n_q` groups of one finite scalar quantizer
    /// (not a residual stack), so only the full count is accepted.
    fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<RvqCodes> {
        if let Some(n) = num_quantizers
            && n != self.cfg.n_q
        {
            bail!(
                "liquid-codec: {n} codebooks requested; every frame needs all {} (grouped FSQ, not RVQ)",
                self.cfg.n_q
            );
        }
        Ok(RvqCodes::new(self.encode(pcm)?, self.cfg.n_q))
    }

    fn decode_codes(&self, codes: &RvqCodes) -> Result<Vec<f32>> {
        if codes.num_quantizers != self.cfg.n_q {
            return Err(anyhow!(
                "liquid-codec: {} codebooks per frame, the codec has {}",
                codes.num_quantizers,
                self.cfg.n_q
            ));
        }
        self.decode(&codes.frames)
    }
}
