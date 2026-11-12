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

//! The fish codec (`codec.pth`, shared by S2-Pro, Drama 3 and Audio8): a causal
//! 44.1 kHz DAC whose RVQ sits behind a 4× temporal downsample with a
//! transformer on each side.
//!
//! ```text
//! decode: codes [10, T] ─ VQ lookup (host) ─ post transformer (8 layers,
//!         causal, window 128) ─ 2 × (ConvTranspose k2 s2 + ConvNeXt) ─
//!         DAC decoder (1536 ch, rates 8·8·4·2, causal) ─ tanh ─ [2048·T]
//! encode: pcm ─ DAC encoder (rates 2·4·8·8, 4-layer transformer, causal) ─
//!         2 × (Conv k2 s2 + ConvNeXt) ─ pre transformer ─ RVQ search (host)
//! ```
//!
//! Code 0 is the semantic codebook (4096 entries), codes 1..=9 the residual
//! ones (1024 each); every codebook is 8-dimensional and projected to 1024.
//! The decoder transformer the config declares is never registered by the
//! reference `DecoderBlock` and the checkpoint has no weights for it.
//!
//! Audio8's `codec.pth` matches the others tensor for tensor except for the
//! post transformer, which is narrower (GQA 16/8, FFN 1216): each
//! transformer's geometry is read from its own weights.
//!
//! Layout: everything runs in rows (`[1, T, C]`); convs are sums of matmuls
//! over shifted views (see the conv half below).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, NodeId, PadMode, RopeStyle, Shape};
use rlx_runtime::{CompiledGraph, Device};
use rlx_torch_ckpt::PtModel;

use crate::nn::{self, AttnSpec, BlockSpec, Builder, Keys, TensorSource};

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 44_100;
/// Waveform samples per code frame (512 hop × 4 quantizer downsample).
pub const FRAME_SAMPLES: usize = 2048;
/// Codebooks per frame: one semantic + nine residual.
pub const NUM_CODEBOOKS: usize = 10;
/// Semantic codebook entries (code 0).
pub const SEMANTIC_SIZE: usize = 4096;
/// Residual codebook entries (codes 1..).
pub const RESIDUAL_SIZE: usize = 1024;

const LATENT: usize = 1024;
const CODE_DIM: usize = 8;
/// Frames per compiled decode/encode graph are rounded up to this. The codec
/// is causal end to end, so zero frames appended after the real ones cannot
/// change them; they are trimmed from the output.
const FRAME_BUCKET: usize = 32;

const F32: DType = DType::F32;

/// One codebook: `in_proj` (encode) and `out_proj` (both directions).
struct Codebook {
    entries: Vec<f32>,
    size: usize,
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl Codebook {
    fn load(src: &CodecWeights, prefix: &str) -> Result<Self> {
        let (entries, shape) = get(src, &format!("{prefix}.codebook.weight"))?;
        ensure!(
            shape.len() == 2 && shape[1] == CODE_DIM,
            "{prefix}: codebook {shape:?}"
        );
        let (in_w, _) = folded_conv(src, &format!("{prefix}.in_proj"))?;
        let (out_w, _) = folded_conv(src, &format!("{prefix}.out_proj"))?;
        Ok(Self {
            size: shape[0],
            entries,
            in_w,
            in_b: get(src, &format!("{prefix}.in_proj.bias"))?.0,
            out_w,
            out_b: get(src, &format!("{prefix}.out_proj.bias"))?.0,
        })
    }

    /// `out_proj(entries[code])` added into `z[t·LATENT ..]` (row layout).
    fn add_decoded(&self, code: usize, z: &mut [f32]) {
        let e = &self.entries[code * CODE_DIM..(code + 1) * CODE_DIM];
        for (o, slot) in z.iter_mut().enumerate() {
            let w = &self.out_w[o * CODE_DIM..(o + 1) * CODE_DIM];
            *slot += self.out_b[o] + w.iter().zip(e).map(|(a, b)| a * b).sum::<f32>();
        }
    }

    /// Nearest entry by cosine similarity of the projected residual, exactly
    /// as `VectorQuantize.decode_latents` scores it.
    fn nearest(&self, residual: &[f32]) -> usize {
        let mut e = [0f32; CODE_DIM];
        for (d, slot) in e.iter_mut().enumerate() {
            let w = &self.in_w[d * LATENT..(d + 1) * LATENT];
            *slot = self.in_b[d] + w.iter().zip(residual).map(|(a, b)| a * b).sum::<f32>();
        }
        l2_normalize(&mut e);
        let e2: f32 = e.iter().map(|v| v * v).sum();
        let mut best = (f32::INFINITY, 0usize);
        for i in 0..self.size {
            let mut c = [0f32; CODE_DIM];
            c.copy_from_slice(&self.entries[i * CODE_DIM..(i + 1) * CODE_DIM]);
            l2_normalize(&mut c);
            let dot: f32 = e.iter().zip(&c).map(|(a, b)| a * b).sum();
            let c2: f32 = c.iter().map(|v| v * v).sum();
            let dist = e2 - 2.0 * dot + c2;
            if dist < best.0 {
                best = (dist, i);
            }
        }
        best.1
    }
}

/// The codec's tensors: the published `codec.pth`, or its safetensors export
/// (`codec_decoder.safetensors` / `codec_encoder.safetensors`, `rlx-fish
/// export-web`) — the same names and values, readable in parts, which a
/// browser can range-fetch.
pub enum CodecWeights {
    Torch(PtModel),
    Safetensors(SafetensorsCheckpoint),
}

impl CodecWeights {
    fn shape_of(&self, key: &str) -> Option<Vec<usize>> {
        match self {
            Self::Torch(m) => m.shape_of(key).map(<[usize]>::to_vec),
            Self::Safetensors(c) => c.shape_of(key),
        }
    }

    /// Every tensor name.
    pub fn keys(&self) -> Vec<String> {
        match self {
            Self::Torch(m) => m.names(),
            Self::Safetensors(c) => c.keys().map(str::to_string).collect(),
        }
    }
}

impl TensorSource for CodecWeights {
    fn tensor(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Torch(m) => TensorSource::tensor(m, key),
            Self::Safetensors(c) => TensorSource::tensor(c, key),
        }
    }
}

/// A checkpoint tensor as `(f32 data, shape)`.
fn get(src: &CodecWeights, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    TensorSource::tensor(src, key)
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= n;
    }
}

/// A conv weight with weight-norm folded in: `g · v / ‖v‖`, the norm taken
/// over every axis but the first (torch `weight_norm(dim=0)`). Reads either
/// parametrization spelling, or a plain `.weight`. Returns `(w, shape)`.
fn folded_conv(src: &CodecWeights, prefix: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let (g_key, v_key) = if src
        .shape_of(&format!("{prefix}.parametrizations.weight.original0"))
        .is_some()
    {
        (
            format!("{prefix}.parametrizations.weight.original0"),
            format!("{prefix}.parametrizations.weight.original1"),
        )
    } else if src.shape_of(&format!("{prefix}.weight_g")).is_some() {
        (format!("{prefix}.weight_g"), format!("{prefix}.weight_v"))
    } else {
        return get(src, &format!("{prefix}.weight"));
    };
    let (g, _) = get(src, &g_key)?;
    let (mut v, shape) = get(src, &v_key)?;
    let per = v.len() / shape[0];
    ensure!(
        g.len() == shape[0],
        "{prefix}: weight_norm g {} vs {shape:?}",
        g.len()
    );
    for (row, gain) in v.chunks_mut(per).zip(&g) {
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let s = gain / n;
        for x in row {
            *x *= s;
        }
    }
    Ok((v, shape))
}

/// The codec: host codebooks plus compiled decode/encode graphs per bucket.
pub struct FishCodec {
    ckpt: CodecWeights,
    device: Device,
    semantic: Codebook,
    residual: Vec<Codebook>,
    /// Post transformer + upsample graphs, by frame bucket.
    decoders: HashMap<usize, CompiledGraph>,
    /// The DAC decoder over one window (`DAC_CONTEXT + DAC_ROWS` rows).
    dac: Option<CompiledGraph>,
    encoders: HashMap<usize, CompiledGraph>,
}

impl FishCodec {
    /// Open `codec.pth` (a file, or a model directory holding one — or, in a
    /// directory without one, its `codec_*.safetensors` export).
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let file = if path.is_dir() {
            path.join("codec.pth")
        } else {
            path.to_path_buf()
        };
        if path.is_dir() && !file.is_file() {
            let parts: Vec<std::path::PathBuf> =
                ["codec_decoder.safetensors", "codec_encoder.safetensors"]
                    .iter()
                    .map(|n| path.join(n))
                    .filter(|p| p.is_file())
                    .collect();
            ensure!(
                !parts.is_empty(),
                "{}: no codec.pth or codec_*.safetensors",
                path.display()
            );
            let files = parts
                .iter()
                .map(|p| {
                    let name = p.file_name().unwrap().to_string_lossy().into_owned();
                    Ok((name, std::fs::read(p)?))
                })
                .collect::<Result<Vec<_>>>()?;
            return Self::from_weights(
                CodecWeights::Safetensors(SafetensorsCheckpoint::from_bytes(files)?),
                device,
            );
        }
        let ckpt = PtModel::open(&file).with_context(|| format!("open {}", file.display()))?;
        Self::from_weights(CodecWeights::Torch(ckpt), device)
    }

    /// The codec from tensors already in memory.
    pub fn from_weights(ckpt: CodecWeights, device: Device) -> Result<Self> {
        let semantic = Codebook::load(&ckpt, "quantizer.semantic_quantizer.quantizers.0")?;
        ensure!(
            semantic.size == SEMANTIC_SIZE,
            "semantic codebook has {} entries",
            semantic.size
        );
        let residual = (0..NUM_CODEBOOKS - 1)
            .map(|i| Codebook::load(&ckpt, &format!("quantizer.quantizer.quantizers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ckpt,
            device,
            semantic,
            residual,
            decoders: HashMap::new(),
            dac: None,
            encoders: HashMap::new(),
        })
    }

    /// Codes `[NUM_CODEBOOKS][frames]` (codebook-major, like the reference's
    /// `[10, T]`) → mono f32 PCM at [`SAMPLE_RATE`], `2048 · frames` samples.
    pub fn decode(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let z = self.dequantize(codes)?;
        let frames = codes[0].len();
        self.decode_latent(&z, frames)
    }

    /// The summed codebook vectors `[frames, 1024]` (row layout) — what
    /// `DownsampleResidualVectorQuantize.decode` hands its post transformer.
    pub fn dequantize(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        ensure!(
            codes.len() == NUM_CODEBOOKS,
            "need {NUM_CODEBOOKS} codebooks, got {}",
            codes.len()
        );
        let frames = codes[0].len();
        ensure!(
            codes.iter().all(|c| c.len() == frames),
            "ragged code matrix"
        );
        let mut z = vec![0f32; frames * LATENT];
        for t in 0..frames {
            let row = &mut z[t * LATENT..(t + 1) * LATENT];
            // The reference clamps out-of-range codes rather than failing.
            let c0 = (codes[0][t] as usize).min(SEMANTIC_SIZE - 1);
            self.semantic.add_decoded(c0, row);
            for (cb, book) in self.residual.iter().enumerate() {
                let c = (codes[cb + 1][t] as usize).min(RESIDUAL_SIZE - 1);
                book.add_decoded(c, row);
            }
        }
        Ok(z)
    }

    /// Every codebook's `out_proj(entry) + bias` rows, the semantic book's
    /// then each residual one's: `[4096 + 9 · 1024, 1024]` — a code's row
    /// is what [`Self::dequantize`] adds for it.
    #[cfg(feature = "export")]
    pub(crate) fn decoded_rows(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for book in std::iter::once(&self.semantic).chain(&self.residual) {
            for code in 0..book.size {
                let mut row = vec![0f32; LATENT];
                book.add_decoded(code, &mut row);
                out.extend(row);
            }
        }
        out
    }

    /// The checkpoint the decoder graphs read.
    #[cfg(feature = "export")]
    pub(crate) fn checkpoint(&self) -> &CodecWeights {
        &self.ckpt
    }

    /// Decode a `[frames, 1024]` latent.
    ///
    /// The post transformer and upsample run over the whole utterance (they
    /// are small: 1024 channels at 4 rows a frame). The DAC decoder then runs
    /// window by window: each window is `DAC_CONTEXT` rows of history and
    /// `DAC_ROWS` new rows, and keeps the new rows' samples. Its convs are
    /// causal and reach 19 rows back (`dac_receptive_rows`), fewer than the
    /// history, so every kept sample sees exactly what the whole-utterance
    /// graph would. The first window starts at the real beginning, where the
    /// convs' zero padding is the whole graph's, and keeps all its rows.
    pub fn decode_latent(&mut self, z: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.decode_latent_from(z, frames, 0)
    }

    /// The samples of [`Self::decode`] from sample `from` on — the same
    /// samples, without decoding the ones before. The upsample stage runs over
    /// every frame (its attention looks back up to 128 frames), the DAC
    /// decoder only over the windows holding samples at or after `from`.
    /// Decoding a growing utterance this way as it is generated plays it as it
    /// is made: the codec is causal, so frames added later cannot change the
    /// samples already returned.
    pub fn decode_from(&mut self, codes: &[Vec<u32>], from: usize) -> Result<Vec<f32>> {
        let z = self.dequantize(codes)?;
        self.decode_latent_from(&z, codes[0].len(), from)
    }

    fn decode_latent_from(&mut self, z: &[f32], frames: usize, from: usize) -> Result<Vec<f32>> {
        let input = self.upsample_prepare(z, frames)?;
        let bucket = upsample_bucket(frames);
        let mut h = self
            .decoders
            .get_mut(&bucket)
            .expect("upsample graph")
            .run(&[("z", &input)]);
        let h = Self::upsample_finish(h.swap_remove(0), frames);
        self.dac_prepare()?;
        let rows = h.len() / LATENT;
        let mut wav = Vec::with_capacity(rows * DAC_HOP);
        for (start, keep) in dac_windows(rows) {
            if dac_window_end(start) * DAC_HOP <= from {
                dac_skip(keep, start, rows, &mut wav);
                continue;
            }
            let input = dac_input(&h, start);
            let out = self
                .dac
                .as_mut()
                .expect("dac graph")
                .run(&[("h", &input)])
                .swap_remove(0);
            dac_keep(&out, keep, rows, &mut wav);
        }
        Ok(wav.split_off(from.min(wav.len())))
    }

    /// [`Self::decode_latent`] with async runs (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn decode_latent_async(&mut self, z: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.decode_latent_from_async(z, frames, 0).await
    }

    /// [`Self::decode_from`] with async runs.
    #[cfg(target_family = "wasm")]
    pub async fn decode_from_async(&mut self, codes: &[Vec<u32>], from: usize) -> Result<Vec<f32>> {
        let z = self.dequantize(codes)?;
        self.decode_latent_from_async(&z, codes[0].len(), from)
            .await
    }

    #[cfg(target_family = "wasm")]
    async fn decode_latent_from_async(
        &mut self,
        z: &[f32],
        frames: usize,
        from: usize,
    ) -> Result<Vec<f32>> {
        let input = self.upsample_prepare(z, frames)?;
        let bucket = upsample_bucket(frames);
        let mut h = {
            let g = self.decoders.get_mut(&bucket).expect("upsample graph");
            g.run_async(&[("z", &input)])
                .await
                .map_err(anyhow::Error::msg)?
        };
        let h = Self::upsample_finish(h.swap_remove(0), frames);
        self.dac_prepare()?;
        let rows = h.len() / LATENT;
        let mut wav = Vec::with_capacity(rows * DAC_HOP);
        for (start, keep) in dac_windows(rows) {
            if dac_window_end(start) * DAC_HOP <= from {
                dac_skip(keep, start, rows, &mut wav);
                continue;
            }
            let input = dac_input(&h, start);
            let mut out = {
                let g = self.dac.as_mut().expect("dac graph");
                g.run_async(&[("h", &input)])
                    .await
                    .map_err(anyhow::Error::msg)?
            };
            dac_keep(&out.swap_remove(0), keep, rows, &mut wav);
        }
        Ok(wav.split_off(from.min(wav.len())))
    }

    /// [`Self::decode`] with async runs.
    #[cfg(target_family = "wasm")]
    pub async fn decode_async(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let z = self.dequantize(codes)?;
        self.decode_latent_async(&z, codes[0].len()).await
    }

    /// The post transformer + upsample graph for `frames`, and its input.
    fn upsample_prepare(&mut self, z: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(
            frames > 0 && z.len() == frames * LATENT,
            "latent is {} values for {frames} frames",
            z.len()
        );
        let bucket = upsample_bucket(frames);
        if !self.decoders.contains_key(&bucket) {
            let (b, out) = build_upsampled(&self.ckpt, bucket)?;
            let g = compile(self.device, b, out, &self.ckpt)?;
            self.decoders.insert(bucket, g);
        }
        let mut input = vec![0f32; bucket * LATENT];
        input[..z.len()].copy_from_slice(z);
        Ok(input)
    }

    /// `h` `[4·frames, 1024]` from the upsample graph's output.
    fn upsample_finish(mut h: Vec<f32>, frames: usize) -> Vec<f32> {
        h.truncate(4 * frames * LATENT);
        h
    }

    /// The DAC decoder's window graph.
    fn dac_prepare(&mut self) -> Result<()> {
        if self.dac.is_none() {
            let (b, out) = build_dac(&self.ckpt, DAC_CONTEXT + DAC_ROWS)?;
            self.dac = Some(compile(self.device, b, out, &self.ckpt)?);
        }
        Ok(())
    }

    /// Decode with the whole-utterance graph (`build_decoder`): what
    /// [`Self::decode_latent`] must equal.
    #[doc(hidden)]
    pub fn decode_latent_whole(&mut self, z: &[f32], frames: usize) -> Result<Vec<f32>> {
        let bucket = frames.div_ceil(FRAME_BUCKET) * FRAME_BUCKET;
        let (b, out) = build_decoder(&self.ckpt, bucket)?;
        let mut g = compile(self.device, b, out, &self.ckpt)?;
        let mut input = vec![0f32; bucket * LATENT];
        input[..z.len()].copy_from_slice(z);
        let mut wav = g.run(&[("z", &input)]).swap_remove(0);
        wav.truncate(frames * FRAME_SAMPLES);
        Ok(wav)
    }

    /// Mono f32 PCM at [`SAMPLE_RATE`] → codes `[NUM_CODEBOOKS][frames]`,
    /// `frames = ceil(len / 2048)`.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        let latent = self.encode_latent(pcm)?;
        Ok(self.quantize(&latent))
    }

    /// The residual codebook search over a pre-quantizer latent `[frames, 1024]`.
    fn quantize(&self, latent: &[f32]) -> Vec<Vec<u32>> {
        let frames = latent.len() / LATENT;
        let mut codes: Vec<Vec<u32>> = (0..NUM_CODEBOOKS)
            .map(|_| Vec::with_capacity(frames))
            .collect();
        let mut decoded = vec![0f32; LATENT];
        for t in 0..frames {
            let mut residual = latent[t * LATENT..(t + 1) * LATENT].to_vec();
            let books = std::iter::once(&self.semantic).chain(&self.residual);
            for (cb, book) in books.enumerate() {
                let c = book.nearest(&residual);
                codes[cb].push(c as u32);
                decoded.fill(0.0);
                book.add_decoded(c, &mut decoded);
                for (r, d) in residual.iter_mut().zip(&decoded) {
                    *r -= d;
                }
            }
        }
        codes
    }

    /// The pre-quantizer latent `[frames, 1024]` for `pcm`.
    ///
    /// Runs in strict f32 ([`rlx_runtime::with_strict_f32`]): the codes are
    /// a nearest-codebook search, and CUDA's default TF32 moves the latent
    /// enough (rel-L2 7e-3) to change 82 of 1120 codes of a 5 s clip — the
    /// prompt a cloned voice is built from.
    pub fn encode_latent(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        rlx_runtime::with_strict_f32(|| self.encode_latent_inner(pcm))
    }

    fn encode_latent_inner(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let (bucket, input) = self.encode_prepare(pcm)?;
        let g = self.encoders.get_mut(&bucket).expect("encoder graph");
        let mut latent = g.run(&[("pcm", &input)]).swap_remove(0);
        latent.truncate(pcm.len().div_ceil(FRAME_SAMPLES) * LATENT);
        Ok(latent)
    }

    /// [`Self::encode`] with an async run (WebGPU in a browser). Strict f32
    /// is CUDA's concern; a browser has no CUDA.
    #[cfg(target_family = "wasm")]
    pub async fn encode_async(&mut self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        let (bucket, input) = self.encode_prepare(pcm)?;
        let mut latent = {
            let g = self.encoders.get_mut(&bucket).expect("encoder graph");
            g.run_async(&[("pcm", &input)])
                .await
                .map_err(anyhow::Error::msg)?
                .swap_remove(0)
        };
        latent.truncate(pcm.len().div_ceil(FRAME_SAMPLES) * LATENT);
        Ok(self.quantize(&latent))
    }

    fn encode_prepare(&mut self, pcm: &[f32]) -> Result<(usize, Vec<f32>)> {
        ensure!(!pcm.is_empty(), "empty audio");
        let frames = pcm.len().div_ceil(FRAME_SAMPLES);
        let bucket = frames.div_ceil(FRAME_BUCKET) * FRAME_BUCKET;
        if !self.encoders.contains_key(&bucket) {
            let (b, out) = build_encoder(&self.ckpt, bucket * FRAME_SAMPLES)?;
            let g = compile(self.device, b, out, &self.ckpt)?;
            self.encoders.insert(bucket, g);
        }
        let mut input = vec![0f32; bucket * FRAME_SAMPLES];
        input[..pcm.len()].copy_from_slice(pcm);
        Ok((bucket, input))
    }

    /// Build every decode graph an utterance of up to `frames` frames uses,
    /// ahead of it: a stream then pays no graph build between pieces.
    pub fn prepare(&mut self, frames: usize) -> Result<()> {
        let mut b = FRAME_BUCKET;
        while b <= upsample_bucket(frames) {
            if !self.decoders.contains_key(&b) {
                let (builder, out) = build_upsampled(&self.ckpt, b)?;
                let g = compile(self.device, builder, out, &self.ckpt)?;
                self.decoders.insert(b, g);
            }
            b *= 2;
        }
        self.dac_prepare()
    }

    /// Write the codec as `codec_decoder.safetensors` (what [`Self::decode`]
    /// reads) and `codec_encoder.safetensors` (what [`Self::encode`] reads)
    /// into `dir`: the checkpoint's own tensors, names and f32 values, so a
    /// browser can fetch the half it needs, in parts. Returns their sizes.
    pub fn export_safetensors(&self, dir: &Path) -> Result<[u64; 2]> {
        let keys = self.ckpt.keys();
        let codebooks = |k: &str| {
            k.starts_with("quantizer.quantizer.") || k.starts_with("quantizer.semantic_quantizer.")
        };
        let decoder = |k: &str| {
            codebooks(k)
                || k.starts_with("quantizer.post_module.")
                || k.starts_with("quantizer.upsample.")
                || k.starts_with("decoder.")
        };
        let encoder = |k: &str| {
            codebooks(k)
                || k.starts_with("quantizer.pre_module.")
                || k.starts_with("quantizer.downsample.")
                || k.starts_with("encoder.")
        };
        let mut sizes = [0u64; 2];
        for (slot, (name, keep)) in [
            (
                "codec_decoder.safetensors",
                &decoder as &dyn Fn(&str) -> bool,
            ),
            ("codec_encoder.safetensors", &encoder),
        ]
        .into_iter()
        .enumerate()
        {
            let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
            for k in keys.iter().filter(|k| keep(k)) {
                let (data, shape) = get(&self.ckpt, k)?;
                tensors.push((
                    k.clone(),
                    data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    shape,
                ));
            }
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            let views = tensors
                .iter()
                .map(|(k, bytes, shape)| {
                    let view = safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.clone(),
                        bytes,
                    )?;
                    Ok((k.clone(), view))
                })
                .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
            let path = dir.join(name);
            safetensors::serialize_to_file(views, None, &path)
                .with_context(|| format!("write {}", path.display()))?;
            sizes[slot] = std::fs::metadata(&path)?.len();
        }
        Ok(sizes)
    }

    /// Drop every compiled graph (they are rebuilt on demand).
    pub fn release(&mut self) {
        self.decoders.clear();
        self.dac = None;
        self.encoders.clear();
    }
}

fn compile(
    device: Device,
    mut b: Builder,
    out: NodeId,
    src: &CodecWeights,
) -> Result<CompiledGraph> {
    b.g.set_outputs(vec![out]);
    let profile = rlx_flow::CompileProfile::default();
    let name = b.g.name.clone();
    let t = crate::clock();
    let mut g = rlx_core::flow_bridge::compile_graph_with_profile(device, b.g, &profile)?;
    let t_compile = crate::elapsed(t);
    let t = crate::clock();
    nn::upload(&mut g, b.params, src)?;
    if rlx_ir::env::flag("RLX_FISH_VERBOSE") {
        eprintln!(
            "codec: {name}: compile {:.1} s, weights {:.1} s",
            t_compile.as_secs_f32(),
            crate::elapsed(t).as_secs_f32()
        );
    }
    Ok(g)
}

// ── transformer half (rows) ─────────────────────────────────────────────

/// Head width of every codec transformer.
const HEAD_DIM: usize = 64;

/// `(n_head, n_kv, ffn)` of the transformer under `prefix`, from the shapes of
/// its first layer: `wo` is `[dim, n_head·d]`, `wqkv` `[(n_head + 2·n_kv)·d,
/// dim]` and `w1` `[ffn, dim]`.
fn transformer_geometry(src: &CodecWeights, prefix: &str) -> Result<(usize, usize, usize)> {
    let shape = |name: &str| {
        let key = format!("{prefix}layers.0.{name}.weight");
        src.shape_of(&key)
            .with_context(|| format!("codec: missing {key}"))
    };
    let (wo, wqkv, w1) = (
        shape("attention.wo")?,
        shape("attention.wqkv")?,
        shape("feed_forward.w1")?,
    );
    let q_w = wo[1];
    ensure!(
        q_w % HEAD_DIM == 0 && wqkv[0] > q_w && (wqkv[0] - q_w) % (2 * HEAD_DIM) == 0,
        "{prefix}: wqkv {wqkv:?} does not fit wo {wo:?}"
    );
    Ok((q_w / HEAD_DIM, (wqkv[0] - q_w) / (2 * HEAD_DIM), w1[0]))
}

/// The codec's `WindowLimitedTransformer`: causal, `window` keys per query,
/// LayerScale, RoPE from bf16 tables, a final RMSNorm. `x: [1, t, dim]`.
#[allow(clippy::too_many_arguments)]
fn transformer(
    b: &mut Builder,
    src: &CodecWeights,
    prefix: &str,
    x: NodeId,
    t: usize,
    layers: usize,
    dim: usize,
    window: Option<usize>,
) -> Result<NodeId> {
    let (n_head, n_kv, ffn) = transformer_geometry(src, prefix)?;
    let spec = BlockSpec {
        dim,
        ffn,
        eps: 1e-5,
        attn: AttnSpec {
            n_head,
            n_kv,
            head_dim: HEAD_DIM,
            qk_norm: false,
            qkv_bias: false,
            rope_dim: HEAD_DIM,
            rope_style: RopeStyle::GptJ,
            output_gate: false,
        },
        layer_scale: true,
        zero_centered: false,
    };
    let positions: Vec<usize> = (0..t).collect();
    let (cos, sin) = nn::rope_tables(&positions, HEAD_DIM, 10_000.0, true);
    let cos = b.data(&format!("{prefix}rope_cos.{t}"), cos, &[t, HEAD_DIM / 2]);
    let sin = b.data(&format!("{prefix}rope_sin.{t}"), sin, &[t, HEAD_DIM / 2]);
    // `make_window_limited_mask`: key j is visible to query i when
    // i - window < j <= i, i.e. `window` keys including itself.
    let mask = match window {
        Some(w) if w < t => MaskKind::SlidingWindow(w - 1),
        _ => MaskKind::Causal,
    };
    let mut h = x;
    for l in 0..layers {
        h = nn::block(
            b,
            &spec,
            &format!("{prefix}layers.{l}."),
            h,
            t,
            cos,
            sin,
            Keys::Own(mask),
        )
        .x;
    }
    Ok(b.rms_norm(h, &format!("{prefix}norm.weight"), dim, 1e-5, false))
}

/// `ConvNeXtBlock` in rows: causal depthwise k7, LayerNorm, MLP ×4 with exact
/// GELU, LayerScale, residual.
fn convnext(
    b: &mut Builder,
    src: &CodecWeights,
    prefix: &str,
    x: NodeId,
    t: usize,
    c: usize,
) -> Result<NodeId> {
    let (w, shape) = get(src, &format!("{prefix}.dwconv.conv.weight"))?;
    ensure!(shape == [c, 1, 7], "{prefix}: dwconv {shape:?}");
    let xp =
        b.g.pad_(x, vec![[0, 0], [6, 0], [0, 0]], PadMode::Constant(0.0));
    let mut acc = None;
    for j in 0..7 {
        let tap: Vec<f32> = (0..c).map(|ch| w[ch * 7 + j]).collect();
        let tw = b.data(&format!("{prefix}.dwconv.tap{j}"), tap, &[c]);
        let s = b.g.narrow_(xp, 1, j, t);
        let m = b.g.mul(s, tw);
        acc = Some(match acc {
            None => m,
            Some(a) => b.g.add(a, m),
        });
    }
    let bias = b.weight(
        &format!("{prefix}.dwconv.conv.bias"),
        &[c],
        nn::TensorOp::Raw,
    );
    let y = b.g.add(acc.expect("taps"), bias);
    let gamma = b.weight(&format!("{prefix}.norm.weight"), &[c], nn::TensorOp::Raw);
    let beta = b.weight(&format!("{prefix}.norm.bias"), &[c], nn::TensorOp::Raw);
    let y = b.g.ln(y, gamma, beta, 1e-6);
    let y = b.linear_bias(y, &format!("{prefix}.pwconv1"), c, 4 * c);
    let y = b.g.gelu(y);
    let y = b.linear_bias(y, &format!("{prefix}.pwconv2"), 4 * c, c);
    let ls = b.weight(&format!("{prefix}.gamma"), &[c], nn::TensorOp::Raw);
    let y = b.g.mul(y, ls);
    Ok(b.g.add(x, y))
}

// ── conv half (rows) ────────────────────────────────────────────────────
//
// Every conv is a sum of matmuls over shifted views of `[1, T, C]` rows: a
// k-tap conv is `Σ_j x[t + j·d] · W_j` with `W_j = w[:, :, j]ᵀ`. Matmuls run
// at full speed on every backend, where direct 1-D conv kernels may not (the
// Metal conv on this stack ran ~10× under the CPU). One matmul per tap, not
// an im2col concat: at 44.1 kHz an im2col of the 96-channel stage is 2 GB.

/// `[cin, cout]` tap `j` of a conv weight `[cout, cin, k]`.
fn conv_tap(w: &[f32], c_out: usize, c_in: usize, k: usize, j: usize) -> Vec<f32> {
    let mut t = vec![0f32; c_in * c_out];
    for o in 0..c_out {
        for i in 0..c_in {
            t[i * c_out + o] = w[(o * c_in + i) * k + j];
        }
    }
    t
}

fn add_bias(
    b: &mut Builder,
    src: &CodecWeights,
    prefix: &str,
    y: NodeId,
    c: usize,
) -> Result<NodeId> {
    let (bias, _) = get(src, &format!("{prefix}.bias"))?;
    ensure!(
        bias.len() == c,
        "{prefix}.bias: {} for {c} channels",
        bias.len()
    );
    let bn = b.data(&format!("{prefix}.b"), bias, &[c]);
    Ok(b.g.add(y, bn))
}

/// `Σ_j narrow(x, j·step) @ W_j` over `taps` with `rows` output rows.
fn tap_sum(b: &mut Builder, x: NodeId, taps: Vec<(usize, NodeId)>, rows: usize) -> NodeId {
    let mut acc = None;
    for (off, w) in taps {
        let v = b.g.narrow_(x, 1, off, rows);
        let m = b.g.mm(v, w);
        acc = Some(match acc {
            None => m,
            Some(a) => b.g.add(a, m),
        });
    }
    acc.expect("at least one tap")
}

/// Causal `Conv1d` on rows `[1, t, c_in]`. Stride 1 (any dilation), or
/// `k = 2·stride` (the encoder's downsamplers). Returns `(y, c_out, t_out)`.
fn causal_conv(
    b: &mut Builder,
    src: &CodecWeights,
    prefix: &str,
    x: NodeId,
    t: usize,
    stride: usize,
    dilation: usize,
) -> Result<(NodeId, usize, usize)> {
    let (w, shape) = folded_conv(src, prefix)?;
    let (c_out, c_in, k) = (shape[0], shape[1], shape[2]);
    let pad = (k - 1) * dilation + 1 - stride;
    let xp = if pad > 0 {
        b.g.pad_(x, vec![[0, 0], [pad, 0], [0, 0]], PadMode::Constant(0.0))
    } else {
        x
    };
    if stride == 1 {
        let taps = (0..k)
            .map(|j| {
                let wj = b.data(
                    &format!("{prefix}.tap{j}"),
                    conv_tap(&w, c_out, c_in, k, j),
                    &[c_in, c_out],
                );
                (j * dilation, wj)
            })
            .collect();
        let y = tap_sum(b, xp, taps, t);
        return Ok((add_bias(b, src, prefix, y, c_out)?, c_out, t));
    }
    // `k = 2s`, left pad `s`: output `t'` reads padded samples
    // `[t'·s, t'·s + 2s)` — rows `t'` and `t' + 1` of the padded input
    // regrouped `s` samples per row.
    let s = stride;
    ensure!(
        k == 2 * s && dilation == 1 && t.is_multiple_of(s),
        "{prefix}: strided conv k {k} s {s} on {t}"
    );
    let t_out = t / s;
    let r =
        b.g.reshape_(xp, vec![1, (t_out + 1) as i64, (s * c_in) as i64]);
    let half = |b: &mut Builder, first: usize, name: &str| {
        let mut m = vec![0f32; s * c_in * c_out];
        for j in 0..s {
            let tap = conv_tap(&w, c_out, c_in, k, first + j);
            m[j * c_in * c_out..(j + 1) * c_in * c_out].copy_from_slice(&tap);
        }
        b.data(&format!("{prefix}.{name}"), m, &[s * c_in, c_out])
    };
    let wa = half(b, 0, "lo");
    let wb = half(b, s, "hi");
    let y = tap_sum(b, r, vec![(0, wa), (1, wb)], t_out);
    Ok((add_bias(b, src, prefix, y, c_out)?, c_out, t_out))
}

/// Causal `ConvTranspose1d` with `k = 2·stride` on rows, keeping the first
/// `t · stride` outputs (`CausalTransConvNet` trims the right `k - stride`).
/// Output `t·s + p` is `x[t]·w[:, :, p] + x[t-1]·w[:, :, p+s]`: two matmuls
/// onto all `s` phases at once, then a reshape interleaves them.
fn causal_conv_transpose(
    b: &mut Builder,
    src: &CodecWeights,
    prefix: &str,
    x: NodeId,
    t: usize,
    stride: usize,
) -> Result<(NodeId, usize, usize)> {
    let (w, shape) = folded_conv(src, prefix)?;
    let (c_in, c_out, k) = (shape[0], shape[1], shape[2]);
    let s = stride;
    ensure!(k == 2 * s, "{prefix}: kernel {k} is not 2·stride {s}");
    let phases = |b: &mut Builder, first: usize, name: &str| {
        let mut m = vec![0f32; c_in * s * c_out];
        for i in 0..c_in {
            for p in 0..s {
                for o in 0..c_out {
                    m[i * s * c_out + p * c_out + o] = w[(i * c_out + o) * k + first + p];
                }
            }
        }
        b.data(&format!("{prefix}.{name}"), m, &[c_in, s * c_out])
    };
    let w_cur = phases(b, 0, "cur");
    let w_prev = phases(b, s, "prev");
    let xp =
        b.g.pad_(x, vec![[0, 0], [1, 0], [0, 0]], PadMode::Constant(0.0));
    let y = tap_sum(b, xp, vec![(1, w_cur), (0, w_prev)], t);
    let t_out = t * s;
    let y = b.g.reshape_(y, vec![1, t_out as i64, c_out as i64]);
    Ok((add_bias(b, src, prefix, y, c_out)?, c_out, t_out))
}

/// Snake: `x + sin(α·x)² / (α + 1e-9)`, per channel (last axis).
fn snake(
    b: &mut Builder,
    src: &CodecWeights,
    key: &str,
    x: NodeId,
    c: usize,
    t: usize,
) -> Result<NodeId> {
    let (alpha, shape) = get(src, key)?;
    ensure!(alpha.len() == c, "{key}: {shape:?} for {c} channels");
    let inv: Vec<f32> = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
    let a = b.data(&format!("{key}.a"), alpha, &[c]);
    let ia = b.data(&format!("{key}.ia"), inv, &[c]);
    let ax = b.g.mul(x, a);
    let s =
        b.g.activation(rlx_ir::op::Activation::Sin, ax, Shape::new(&[1, t, c], F32));
    let s2 = b.g.mul(s, s);
    let sc = b.g.mul(s2, ia);
    Ok(b.g.add(x, sc))
}

/// `ResidualUnit(causal)`: snake → conv k7 (dilated) → snake → conv k1, + x.
fn residual_unit(
    b: &mut Builder,
    src: &CodecWeights,
    prefix: &str,
    x: NodeId,
    c: usize,
    t: usize,
    dilation: usize,
) -> Result<NodeId> {
    let h = snake(b, src, &format!("{prefix}.block.0.alpha"), x, c, t)?;
    let (h, _, _) = causal_conv(b, src, &format!("{prefix}.block.1.conv"), h, t, 1, dilation)?;
    let h = snake(b, src, &format!("{prefix}.block.2.alpha"), h, c, t)?;
    let (h, _, _) = causal_conv(b, src, &format!("{prefix}.block.3.conv"), h, t, 1, 1)?;
    Ok(b.g.add(x, h))
}

pub(crate) fn build_decoder(src: &CodecWeights, frames: usize) -> Result<(Builder, NodeId)> {
    let mut b = Builder::new("fish_codec_decode");
    let z = b.input("z", &[1, frames, LATENT]);
    let (h, t) = upsampled(&mut b, src, z, frames)?;
    let out = dac(&mut b, src, h, t)?;
    Ok((b, out))
}

/// The post transformer and the 4× upsample, on their own: `z` `[1, frames,
/// 1024]` → `h` `[1, 4·frames, 1024]`, the rows [`build_dac`] reads.
fn build_upsampled(src: &CodecWeights, frames: usize) -> Result<(Builder, NodeId)> {
    let mut b = Builder::new("fish_codec_upsample");
    let z = b.input("z", &[1, frames, LATENT]);
    let (h, _) = upsampled(&mut b, src, z, frames)?;
    Ok((b, h))
}

/// The DAC decoder on its own: `h` `[1, rows, 1024]` → `rows · 512` samples.
fn build_dac(src: &CodecWeights, rows: usize) -> Result<(Builder, NodeId)> {
    let mut b = Builder::new("fish_codec_dac");
    let h = b.input("h", &[1, rows, LATENT]);
    let out = dac(&mut b, src, h, rows)?;
    Ok((b, out))
}

/// Samples per `h` row (the DAC decoder's upsampling: 8·8·4·2).
const DAC_HOP: usize = 512;
/// The DAC decoder's strides, first block first.
const DAC_STRIDES: [usize; 4] = [8, 8, 4, 2];
/// Its residual units' dilations, each a k7 conv.
const DAC_DILATIONS: [usize; 3] = [1, 3, 9];

/// `h` rows before a sample's own row that the DAC decoder reads: every conv
/// is causal, so walk back from the output (the final k7 conv, then each
/// block's residual units and its `k = 2s` transposed conv, then the first
/// k7 conv). 19 for this decoder.
const fn dac_receptive_rows() -> usize {
    let k7 = 6;
    let mut ctx = k7;
    let mut i = DAC_STRIDES.len();
    while i > 0 {
        i -= 1;
        let mut j = 0;
        while j < DAC_DILATIONS.len() {
            ctx += k7 * DAC_DILATIONS[j];
            j += 1;
        }
        ctx = ctx.div_ceil(DAC_STRIDES[i]) + 1;
    }
    ctx + k7
}

/// `h` rows one DAC run keeps, and the rows of history it is given before
/// them. The DAC decoder holds the codec's large activations (1536 channels
/// at 1/512 of the sample rate down to 96 at the full rate), so it runs over
/// a fixed window instead of the whole utterance: memory stays that of
/// `DAC_CONTEXT + DAC_ROWS` rows (1.1 s of audio) however long the speech.
// 64: one window per 16 frames, the step `rlx-fish audio8 --play` decodes
// at, so a streamed piece runs exactly one window (128 ran ~1.5 of 160 rows
// a step: 1.5x real time on a shared Metal GPU against ~4x).
const DAC_ROWS: usize = 64;
const DAC_CONTEXT: usize = 32;
const _: () = assert!(DAC_CONTEXT >= dac_receptive_rows());

/// The DAC windows over `rows` `h` rows: `(first input row, leading output
/// rows to drop)`. The first starts at row 0 and keeps everything; each later
/// one starts `DAC_CONTEXT` rows before the first row it keeps.
fn dac_windows(rows: usize) -> Vec<(usize, usize)> {
    let window = DAC_CONTEXT + DAC_ROWS;
    let mut out = vec![(0, 0)];
    let mut start = window;
    while start < rows {
        out.push((start - DAC_CONTEXT, DAC_CONTEXT));
        start += DAC_ROWS;
    }
    out
}

/// One window's input: `h` rows from `from`, zero past the end.
fn dac_input(h: &[f32], from: usize) -> Vec<f32> {
    let window = DAC_CONTEXT + DAC_ROWS;
    let rows = h.len() / LATENT;
    let mut input = vec![0f32; window * LATENT];
    let have = rows.saturating_sub(from).min(window);
    input[..have * LATENT].copy_from_slice(&h[from * LATENT..(from + have) * LATENT]);
    input
}

/// The frames an upsample graph is built for: a power of two from
/// `FRAME_BUCKET` up. A growing utterance (streamed) crosses few of them —
/// five to 512 frames — instead of a new graph every 32 frames; the padding
/// is free of effect (the stage is causal) and cheap (it is the small half).
fn upsample_bucket(frames: usize) -> usize {
    frames.max(1).next_power_of_two().max(FRAME_BUCKET)
}

/// The row after the last one a window starting at `start` keeps.
fn dac_window_end(start: usize) -> usize {
    start + DAC_CONTEXT + DAC_ROWS
}

/// Stand-in samples for a window not run, where its samples would go (cut
/// before returning: only windows wholly before `from` are skipped).
fn dac_skip(keep: usize, start: usize, rows: usize, wav: &mut Vec<f32>) {
    let n = (dac_window_end(start) - start - keep) * DAC_HOP;
    let want = rows * DAC_HOP - wav.len();
    wav.resize(wav.len() + n.min(want), 0.0);
}

/// A window's samples after its `keep` history rows, up to `rows · 512` in all.
fn dac_keep(out: &[f32], keep: usize, rows: usize, wav: &mut Vec<f32>) {
    let kept = &out[keep * DAC_HOP..];
    let want = rows * DAC_HOP - wav.len();
    wav.extend_from_slice(&kept[..kept.len().min(want)]);
}

/// `[1, frames, 1024]` → `([1, 4·frames, 1024], 4·frames)`.
fn upsampled(
    b: &mut Builder,
    src: &CodecWeights,
    z: NodeId,
    frames: usize,
) -> Result<(NodeId, usize)> {
    let mut h = transformer(
        b,
        src,
        "quantizer.post_module.",
        z,
        frames,
        8,
        LATENT,
        Some(128),
    )?;

    // Upsample: ConvTranspose(k2, s2) in rows is one matmul onto both output
    // phases, `[t, C] @ [C, 2C]`, then a reshape that interleaves them.
    let mut t = frames;
    for i in 0..2 {
        let prefix = format!("quantizer.upsample.{i}");
        let (w, shape) = get(src, &format!("{prefix}.0.conv.weight"))?;
        ensure!(shape == [LATENT, LATENT, 2], "{prefix}: {shape:?}");
        let mut wp = vec![0f32; LATENT * 2 * LATENT];
        for ic in 0..LATENT {
            for oc in 0..LATENT {
                for p in 0..2 {
                    wp[ic * 2 * LATENT + p * LATENT + oc] = w[(ic * LATENT + oc) * 2 + p];
                }
            }
        }
        let wn = b.data(&format!("{prefix}.0.w"), wp, &[LATENT, 2 * LATENT]);
        let y = b.g.mm(h, wn);
        t *= 2;
        let y = b.g.reshape_(y, vec![1, t as i64, LATENT as i64]);
        let bias = b.weight(
            &format!("{prefix}.0.conv.bias"),
            &[LATENT],
            nn::TensorOp::Raw,
        );
        let y = b.g.add(y, bias);
        h = convnext(b, src, &format!("{prefix}.1"), y, t, LATENT)?;
    }
    Ok((h, t))
}

/// The DAC decoder on `h` `[1, t, 1024]`.
fn dac(b: &mut Builder, src: &CodecWeights, h: NodeId, t: usize) -> Result<NodeId> {
    let (mut x, mut c, mut t) = causal_conv(b, src, "decoder.model.0.conv", h, t, 1, 1)?;
    for (i, stride) in DAC_STRIDES.into_iter().enumerate() {
        let p = format!("decoder.model.{}.block", i + 1);
        let s = snake(b, src, &format!("{p}.0.alpha"), x, c, t)?;
        (x, c, t) = causal_conv_transpose(b, src, &format!("{p}.1.conv"), s, t, stride)?;
        for (j, dil) in DAC_DILATIONS.into_iter().enumerate() {
            x = residual_unit(b, src, &format!("{p}.{}", j + 2), x, c, t, dil)?;
        }
    }
    let x = snake(b, src, "decoder.model.5.alpha", x, c, t)?;
    let (x, _, _) = causal_conv(b, src, "decoder.model.6.conv", x, t, 1, 1)?;
    Ok(b.g.tanh(x))
}

fn build_encoder(src: &CodecWeights, samples: usize) -> Result<(Builder, NodeId)> {
    if !samples.is_multiple_of(FRAME_SAMPLES) {
        bail!("encoder input must be a multiple of {FRAME_SAMPLES} samples");
    }
    let mut b = Builder::new("fish_codec_encode");
    let pcm = b.input("pcm", &[1, samples, 1]);
    let (mut x, mut c, mut t) =
        causal_conv(&mut b, src, "encoder.block.0.conv", pcm, samples, 1, 1)?;
    for (i, stride) in [2usize, 4, 8, 8].into_iter().enumerate() {
        let p = format!("encoder.block.{}.block", i + 1);
        for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
            x = residual_unit(&mut b, src, &format!("{p}.{j}"), x, c, t, dil)?;
        }
        x = snake(&mut b, src, &format!("{p}.3.alpha"), x, c, t)?;
        (x, c, t) = causal_conv(&mut b, src, &format!("{p}.4.conv"), x, t, stride, 1)?;
        if i == 3 {
            x = transformer(&mut b, src, &format!("{p}.5."), x, t, 4, c, Some(512))?;
        }
    }
    let x = snake(&mut b, src, "encoder.block.5.alpha", x, c, t)?;
    let (x, c, t) = causal_conv(&mut b, src, "encoder.block.6.conv", x, t, 1, 1)?;
    ensure!(c == LATENT, "encoder latent has {c} channels");

    // Downsample: causal Conv(k2, s2) in rows is `[t/2, 2C] @ [2C, C]`.
    let mut h = x;
    let mut t = t;
    for i in 0..2 {
        let prefix = format!("quantizer.downsample.{i}");
        let (w, shape) = get(src, &format!("{prefix}.0.conv.weight"))?;
        ensure!(shape == [LATENT, LATENT, 2], "{prefix}: {shape:?}");
        let mut wp = vec![0f32; 2 * LATENT * LATENT];
        for oc in 0..LATENT {
            for ic in 0..LATENT {
                for k in 0..2 {
                    wp[(k * LATENT + ic) * LATENT + oc] = w[(oc * LATENT + ic) * 2 + k];
                }
            }
        }
        t /= 2;
        let r = b.g.reshape_(h, vec![1, t as i64, (2 * LATENT) as i64]);
        let wn = b.data(&format!("{prefix}.0.w"), wp, &[2 * LATENT, LATENT]);
        let y = b.g.mm(r, wn);
        let bias = b.weight(
            &format!("{prefix}.0.conv.bias"),
            &[LATENT],
            nn::TensorOp::Raw,
        );
        let y = b.g.add(y, bias);
        h = convnext(&mut b, src, &format!("{prefix}.1"), y, t, LATENT)?;
    }
    let out = transformer(
        &mut b,
        src,
        "quantizer.pre_module.",
        h,
        t,
        8,
        LATENT,
        Some(128),
    )?;
    Ok((b, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dac_windows_carry_enough_history() {
        assert_eq!(dac_receptive_rows(), 19);
        assert!(DAC_CONTEXT >= dac_receptive_rows());
        assert_eq!(DAC_STRIDES.iter().product::<usize>(), DAC_HOP);
        assert_eq!(4 * DAC_HOP, FRAME_SAMPLES);
    }

    #[test]
    fn frame_geometry() {
        assert_eq!(FRAME_SAMPLES, 512 * 4);
        assert_eq!(SAMPLE_RATE as f32 / FRAME_SAMPLES as f32, 21.533203);
    }
}
