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

//! `AutoencoderKLMiniMaxH3` — the video VAE.
//!
//! Asymmetric by design: a 3-D **causal CNN** encoder compresses pixels 16x
//! spatially and 4x temporally into 24 latent channels, while decoding is done
//! by a 36-layer **ViT** rather than a mirrored CNN. Every latent voxel is one
//! token; the decoder attends over them with full self-attention and then
//! expands each token into a `4 x 16 x 16` pixel block.
//!
//! ## Register tokens
//!
//! `num_register_tokens` learned tokens plus a single all-zero token are
//! appended to the sequence, all pinned at rotary position `0`, attended over,
//! and dropped again before the patch projection. They give the attention
//! somewhere to park global information instead of distorting a real voxel — the
//! same trick DINOv2 introduced.
//!
//! ## Rotary grid
//!
//! Coordinates are length-normalized to `[-1, 1)` per axis — `2 * (i + 0.5) / n - 1`
//! — and scaled by `2 * pi`, so the decoder's positions are *resolution
//! independent*: the same latent voxel occupies the same angle whether the clip
//! is 48 or 84 tokens wide. That is the opposite convention from the DiT, whose
//! grid is scaled by a fixed 32.
//!
//! ## Temporal chunking
//!
//! A full clip is far too long for one attention document — a 37-frame 768x1344
//! latent is ~150k tokens. The VAE therefore decodes `clip_length` pixel frames
//! at a time from `tokens_chunk_size` latent frames, dropping `token_drop`
//! frames of overlap and cross-fading the seam.
//! [`H3VideoVaeConfig::chunk_geometry`] derives that schedule and
//! [`H3VideoDecoder::decode_clip`] drives it.

use crate::config::H3VideoVaeConfig;
use crate::rope::RopeTables;
use crate::vae_video_encoder::Volume;
use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, plugin_named};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// Pixel frames one latent frame expands to along time.
pub const PATCH_SIZE_T: usize = 4;
/// Pixel extent one latent voxel expands to along height and width.
pub const PATCH_SIZE: usize = 16;

/// Number of rotary frequencies per axis in the ViT decoder.
///
/// The reference builds `arange(0, 1, 2 * num_axes / dim)` over
/// `dim = attention_head_dim * rope_dim_ratio`, which is
/// `dim / (2 * num_axes)` entries.
#[must_use]
pub fn decoder_rope_freq_dim(cfg: &H3VideoVaeConfig) -> usize {
    let dim = (cfg.decoder_attention_head_dim as f32 * cfg.decoder_rope_dim_ratio) as usize;
    dim / 6
}

/// The temporal chunking geometry a decode runs under.
///
/// `clip_length` (17) is deliberately **not** a multiple of the 4x temporal
/// compression, so the decoder has to re-derive two things the encoder left
/// implicit: a leading pad of `(-clip_length) % 4 = 3` pixel frames that every
/// decoded window carries, and the `token_drop` overlap between consecutive
/// windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkGeometry {
    /// Latent frames each decoded window covers — `tokens_chunk_size + token_overlap`.
    ///
    /// This is the same for **every** window, so one compiled graph serves a
    /// whole clip.
    pub window_frames: usize,
    /// Latent frames a window advances by.
    pub tokens_chunk_size: usize,
    /// Extra latent frames a window reads past its stride.
    pub token_overlap: usize,
    /// Leading pixel frames every decoded window discards.
    pub frame_pre_padding: usize,
    /// Pixel frames consecutive windows cross-fade over.
    pub frame_overlap: usize,
    /// Pixel frames one window contributes after the pre-pad is cut.
    pub frames_per_window: usize,
    /// Number of windows.
    pub num_chunks: usize,
    /// Latent frames repeated at the tail to fill the last window.
    pub pad_tokens: usize,
    /// Trailing pixel frames to cut, produced by `pad_tokens`.
    pub pad_frames: usize,
    /// Pixel frames the clip decodes to.
    pub num_pixel_frames: usize,
}

impl ChunkGeometry {
    /// Whether the clip is long enough to fill a decode window.
    ///
    /// `token_drop` costs a whole window's worth of latent frames up front, so
    /// clips below `tokens_chunk_size * 2 - token_drop` latent frames yield no
    /// windows at all. The released generation range never reaches down there —
    /// the shortest video is 5 seconds — but a caller decoding a stray short
    /// clip deserves an error rather than an empty result.
    #[must_use]
    pub fn is_decodable(&self) -> bool {
        self.num_chunks > 0
    }
}

impl H3VideoVaeConfig {
    /// Derive the chunking geometry for a latent clip.
    #[must_use]
    pub fn chunk_geometry(&self, num_latent_frames: usize) -> ChunkGeometry {
        let ratio = self.temporal_compression().max(1);
        let frame_pre_padding = self.clip_length.next_multiple_of(ratio) - self.clip_length;
        // Only a zero `clip_length` could make this zero, which is not a
        // checkpoint anyone ships; clamping once keeps every division below safe.
        let tokens_chunk_size = self.clip_length.div_ceil(ratio).max(1);
        let token_overlap =
            (tokens_chunk_size - self.token_drop % tokens_chunk_size) % tokens_chunk_size;
        let frame_overlap = (token_overlap * ratio).saturating_sub(frame_pre_padding);
        let chunk_num_frames = tokens_chunk_size * ratio;
        let frames_per_window = chunk_num_frames.saturating_sub(frame_pre_padding);

        let num_tokens = num_latent_frames + self.token_drop;
        let pad_tokens = (tokens_chunk_size - num_tokens % tokens_chunk_size) % tokens_chunk_size;
        let num_chunks = ((num_tokens + pad_tokens) / tokens_chunk_size)
            .saturating_sub(usize::from(self.token_drop > 0));

        // The repeated tail latents produce pixel frames nobody asked for. A
        // chunk's last latent frame only covers `clip_length % ratio` of them.
        let intra_tail = self.clip_length % ratio;
        let pad_frames: usize = (0..pad_tokens)
            .map(|k| {
                if intra_tail != 0 && (num_latent_frames + k).is_multiple_of(tokens_chunk_size) {
                    intra_tail
                } else {
                    ratio
                }
            })
            .sum();

        // Every window appends `frames_per_window`; the trailing overlap window
        // adds `frame_overlap` more.
        let raw = num_chunks * frames_per_window
            + if num_chunks > 0 && self.token_drop > 0 {
                frame_overlap
            } else {
                0
            };
        ChunkGeometry {
            window_frames: tokens_chunk_size + token_overlap,
            tokens_chunk_size,
            token_overlap,
            frame_pre_padding,
            frame_overlap,
            frames_per_window,
            num_chunks,
            pad_tokens,
            pad_frames,
            num_pixel_frames: raw.saturating_sub(pad_frames),
        }
    }
}

/// Length-normalized rotary coordinates of one decoder chunk, in the packed
/// `(t, h, w)` order the ViT attends over, with the register and zero tokens
/// pinned at the origin.
///
/// Angles are pre-multiplied by `2 * pi` so [`RopeTables`] can build the same
/// tables it does for the DiT.
#[must_use]
pub fn decoder_position_ids(
    frames: usize,
    height: usize,
    width: usize,
    num_suffix_tokens: usize,
) -> Vec<f64> {
    let norm = |i: usize, n: usize| 2.0 * ((i as f64 + 0.5) / n as f64) - 1.0;
    let two_pi = std::f64::consts::TAU;
    let mut out = Vec::with_capacity((frames * height * width + num_suffix_tokens) * 3);
    for t in 0..frames {
        for h in 0..height {
            for w in 0..width {
                out.push(two_pi * norm(t, frames));
                out.push(two_pi * norm(h, height));
                out.push(two_pi * norm(w, width));
            }
        }
    }
    // Register tokens and the zero token all sit at position 0.
    out.extend(std::iter::repeat_n(0.0, num_suffix_tokens * 3));
    out
}

/// The video VAE's ViT decoder, compiled for one chunk geometry.
pub struct H3VideoDecoder {
    compiled: CompiledGraph,
    cfg: H3VideoVaeConfig,
    frames: usize,
    height: usize,
    width: usize,
    num_tokens: usize,
    device: Device,
}

impl H3VideoDecoder {
    #[must_use]
    pub fn config(&self) -> &H3VideoVaeConfig {
        &self.cfg
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Latent voxels this graph was compiled for.
    #[must_use]
    pub fn geometry(&self) -> (usize, usize, usize) {
        (self.frames, self.height, self.width)
    }

    /// Sequence length the decoder attends over: one token per latent voxel
    /// plus the register tokens and the zero token.
    #[must_use]
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Decode one chunk of latents to pixels.
    ///
    /// `latents` is `[latent_channels * frames * height * width]` in `(C, T, H, W)`
    /// order. The result is `[out_channels, frames * 4, height * 16, width * 16]`.
    pub fn decode_chunk(&mut self, latents: &[f32]) -> Result<Vec<f32>> {
        let c = self.cfg.latent_channels;
        let voxels = self.frames * self.height * self.width;
        ensure!(
            latents.len() == c * voxels,
            "latents len {} != {c} channels × {voxels} voxels",
            latents.len()
        );
        // The decoder reads tokens as `(T, H, W, C)`; the caller hands us
        // `(C, T, H, W)`.
        let mut tokens = vec![0.0f32; voxels * c];
        for ch in 0..c {
            for v in 0..voxels {
                tokens[v * c + ch] = latents[ch * voxels + v];
            }
        }

        let suffix = self.cfg.decoder_num_register_tokens + 1;
        let pos = decoder_position_ids(self.frames, self.height, self.width, suffix);
        let tables = RopeTables::build(
            &pos,
            decoder_rope_freq_dim(&self.cfg),
            self.cfg.decoder_rope_theta,
        )?;

        let outs = self.compiled.run(&[
            ("tokens", &tokens),
            ("cos", &tables.cos),
            ("sin", &tables.sin),
        ]);
        let flat = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("the video decoder returned no output"))?;

        let patch = self.cfg.out_channels * PATCH_SIZE_T * PATCH_SIZE * PATCH_SIZE;
        ensure!(
            flat.len() >= voxels * patch,
            "decoder produced {} values, expected at least {}",
            flat.len(),
            voxels * patch
        );
        self.unpatchify(&flat[..voxels * patch])
    }

    /// `[voxels, out_channels * 4 * 16 * 16]` to `[C, T*4, H*16, W*16]`.
    fn unpatchify(&self, rows: &[f32]) -> Result<Vec<f32>> {
        let (t, h, w) = (self.frames, self.height, self.width);
        let oc = self.cfg.out_channels;
        let (ot, oh, ow) = (t * PATCH_SIZE_T, h * PATCH_SIZE, w * PATCH_SIZE);
        let patch = oc * PATCH_SIZE_T * PATCH_SIZE * PATCH_SIZE;
        let mut out = vec![0.0f32; oc * ot * oh * ow];
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    let row = ((ti * h + hi) * w + wi) * patch;
                    for c in 0..oc {
                        for dt in 0..PATCH_SIZE_T {
                            for dh in 0..PATCH_SIZE {
                                for dw in 0..PATCH_SIZE {
                                    let src = row
                                        + ((c * PATCH_SIZE_T + dt) * PATCH_SIZE + dh) * PATCH_SIZE
                                        + dw;
                                    let dst = ((c * ot + ti * PATCH_SIZE_T + dt) * oh
                                        + hi * PATCH_SIZE
                                        + dh)
                                        * ow
                                        + wi * PATCH_SIZE
                                        + dw;
                                    out[dst] = rows[src];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Decode a whole latent clip, window by window.
    ///
    /// `latents` is `[latent_channels, num_latent_frames, height, width]`.
    /// Returns `[out_channels, num_pixel_frames, height * 16, width * 16]`.
    ///
    /// Consecutive windows overlap by [`ChunkGeometry::frame_overlap`] pixel
    /// frames and are linearly cross-faded, which is what keeps the seams
    /// invisible. The decoder must be compiled for
    /// [`ChunkGeometry::window_frames`] latent frames — every window is that
    /// size, including the last, because the tail is padded by repetition.
    pub fn decode_clip(&mut self, latents: &[f32], num_latent_frames: usize) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let g = cfg.chunk_geometry(num_latent_frames);
        let (c, h, w) = (cfg.latent_channels, self.height, self.width);
        ensure!(
            self.frames == g.window_frames,
            "this decoder is compiled for {} latent frames; a clip of {num_latent_frames} needs windows of {}",
            self.frames,
            g.window_frames
        );
        ensure!(
            latents.len() == c * num_latent_frames * h * w,
            "latents hold {} values for {c}x{num_latent_frames}x{h}x{w}",
            latents.len()
        );
        ensure!(
            g.is_decodable(),
            "a clip of {num_latent_frames} latent frames is shorter than one decode window \
             ({} latent frames after the {}-frame token drop); decode at least {} latent frames",
            g.window_frames,
            cfg.token_drop,
            g.tokens_chunk_size * 2 - cfg.token_drop.min(g.tokens_chunk_size * 2)
        );

        // Repeat the last latent frame to fill the final window.
        let padded_frames = num_latent_frames + g.pad_tokens;
        let hw = h * w;
        let mut z = vec![0.0f32; c * padded_frames * hw];
        for ch in 0..c {
            for t in 0..padded_frames {
                let src_t = t.min(num_latent_frames - 1);
                let src = (ch * num_latent_frames + src_t) * hw;
                let dst = (ch * padded_frames + t) * hw;
                z[dst..dst + hw].copy_from_slice(&latents[src..src + hw]);
            }
        }

        let (oh, ow) = (h * PATCH_SIZE, w * PATCH_SIZE);
        let frame_px = cfg.out_channels * oh * ow;
        let chunk_num_frames = g.tokens_chunk_size * cfg.temporal_compression().max(1);

        let mut out: Vec<f32> = Vec::with_capacity(g.num_pixel_frames * frame_px);
        let mut carried: Option<Vec<f32>> = None;

        for i in 0..g.num_chunks {
            let start = i * g.tokens_chunk_size;
            // Slice the window out of the padded clip.
            let mut window = vec![0.0f32; c * g.window_frames * hw];
            for ch in 0..c {
                for t in 0..g.window_frames {
                    let src = (ch * padded_frames + start + t) * hw;
                    let dst = (ch * g.window_frames + t) * hw;
                    window[dst..dst + hw].copy_from_slice(&z[src..src + hw]);
                }
            }
            let clip = self.decode_chunk(&window)?;
            let clip_frames = clip.len() / frame_px;

            // Each window yields the body plus, when `token_drop` created an
            // overlap, the next window's leading segment.
            let segments = usize::from(cfg.token_drop > 0) + 1;
            for j in 0..segments {
                let fs = j * chunk_num_frames;
                if fs >= clip_frames {
                    break;
                }
                let fe = (fs + chunk_num_frames).min(clip_frames);
                // Drop the implicit leading pad every window carries.
                let fs = (fs + g.frame_pre_padding).min(fe);
                if fs >= fe {
                    continue;
                }
                let seg = frames_slice(&clip, fs, fe, frame_px);
                if j == 0 {
                    let seg = match carried.take() {
                        Some(prev) => blend_frames(&prev, &seg, g.frame_overlap, frame_px),
                        None => seg,
                    };
                    out.extend_from_slice(&seg);
                } else {
                    carried = Some(seg);
                }
            }
        }
        if let Some(tail) = carried {
            out.extend_from_slice(&tail);
        }

        // Cut the pixel frames the repeated tail latents invented.
        let want = g.num_pixel_frames * frame_px;
        ensure!(
            out.len() >= want,
            "decode produced {} pixel frames, expected at least {}",
            out.len() / frame_px,
            g.num_pixel_frames
        );
        out.truncate(want);
        Ok(out)
    }
}

/// Copy pixel frames `[start, end)` out of a decoded window.
fn frames_slice(clip: &[f32], start: usize, end: usize, frame_px: usize) -> Vec<f32> {
    clip[start * frame_px..end * frame_px].to_vec()
}

/// Linearly cross-fade `prev`'s frames into the head of `next`.
///
/// `prev` is the carried overlap; the first `extent` frames of `next` cover the
/// same instants, so they are mixed rather than concatenated.
fn blend_frames(prev: &[f32], next: &[f32], extent: usize, frame_px: usize) -> Vec<f32> {
    let prev_frames = prev.len() / frame_px;
    let next_frames = next.len() / frame_px;
    let extent = extent.min(prev_frames).min(next_frames);
    if extent == 0 {
        return next.to_vec();
    }
    let mut out = next.to_vec();
    for f in 0..extent {
        let wb = f as f32 / extent as f32;
        let wa = 1.0 - wb;
        let a =
            &prev[(prev_frames - extent + f) * frame_px..(prev_frames - extent + f + 1) * frame_px];
        let b = &mut out[f * frame_px..(f + 1) * frame_px];
        for (dst, &src) in b.iter_mut().zip(a) {
            *dst = src * wa + *dst * wb;
        }
    }
    out
}

/// Compile the ViT decoder for one chunk geometry.
pub fn compile_video_decoder(
    cfg: &H3VideoVaeConfig,
    weights: &mut WeightMap,
    device: Device,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<H3VideoDecoder> {
    cfg.validate()?;
    ensure!(
        frames > 0 && height > 0 && width > 0,
        "the decoder chunk must be non-empty"
    );
    let voxels = frames * height * width;
    let num_tokens = voxels + cfg.decoder_num_register_tokens + 1;

    let built = build_decoder_flow(cfg, weights, voxels, num_tokens)
        .context("MiniMax-H3: build video VAE decoder flow")?;
    let typed = built.typed_params.clone();
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    let opts =
        rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);

    Ok(H3VideoDecoder {
        compiled,
        cfg: cfg.clone(),
        frames,
        height,
        width,
        num_tokens,
        device,
    })
}

fn build_decoder_flow(
    cfg: &H3VideoVaeConfig,
    weights: &mut WeightMap,
    voxels: usize,
    num_tokens: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let dim = cfg.decoder_hidden_size();
    let nh = cfg.decoder_num_attention_heads;
    let hd = cfg.decoder_attention_head_dim;
    let freq_dim = decoder_rope_freq_dim(cfg);
    let half = 3 * freq_dim;
    let n_rot = 2 * half;
    let ffn = cfg.decoder_ffn_mult * dim;
    let eps = cfg.decoder_norm_eps;
    let n_reg = cfg.decoder_num_register_tokens;
    let patch = cfg.out_channels * PATCH_SIZE_T * PATCH_SIZE * PATCH_SIZE;

    let flow = ModelFlow::new("minimax_h3_video_decoder")
        .with_profile(CompileProfile::encoder())
        .input("tokens", Shape::new(&[1, voxels, cfg.latent_channels], f))
        .input("cos", Shape::new(&[num_tokens, half], f))
        .input("sin", Shape::new(&[num_tokens, half], f))
        .stage(plugin_named("bind", move |emit, h| {
            let cos = emit.flow_input("cos")?;
            let sin = emit.flow_input("sin")?;
            emit.set_named("vae_cos", cos.hir_id());
            emit.set_named("vae_sin", sin.hir_id());
            let z_dim = emit.synth_zeros("vae_zeros_dim", dim);
            let z_head = emit.synth_zeros("vae_zeros_head", hd);
            let o_head = emit.synth_param(
                "vae_ones_head",
                vec![1.0f32; hd],
                Shape::new(&[hd], DType::F32),
            );
            emit.set_named("vae_zeros_dim", z_dim);
            emit.set_named("vae_zeros_head", z_head);
            emit.set_named("vae_ones_head", o_head);
            Ok(h)
        }))
        // Project the latents and append the register + zero tokens.
        .stage(plugin_named("proj_in", move |emit, _h| {
            let tokens = emit.flow_input("tokens")?;
            // The latents pass through `post_quant_conv` — a 1x1 Conv3d, i.e. a
            // plain linear map on the channel axis — *before* the ViT. Skipping
            // it is invisible in shapes and produces plausible-looking but wrong
            // pixels, so it is applied here rather than folded into `proj_in`.
            let pq_w = emit.load_param("post_quant_conv.weight", true)?;
            let pq_b = emit.load_param("post_quant_conv.bias", false)?;
            let w = emit.load_param("decoder.proj_in.weight", true)?;
            let b = emit.load_param("decoder.proj_in.bias", false)?;
            let reg = emit.load_param("decoder.register_tokens", false)?;
            let zero_token = emit.synth_zeros("vae_zero_token", dim);
            let mut gb = HirMut::new(emit.hir());
            let q = gb.mm(tokens.hir_id(), pq_w);
            let q = gb.add(q, pq_b);
            let x = gb.mm(q, w);
            let x = gb.add(x, b);
            let reg3 = gb.reshape_(reg, vec![1, n_reg as i64, dim as i64]);
            let zero3 = gb.reshape_(zero_token, vec![1, 1, dim as i64]);
            let cat = gb.concat_(vec![x, reg3, zero3], 1);
            Ok(Some(emit.wrap(cat, Shape::new(&[1, num_tokens, dim], f))))
        }))
        .repeat_layers(cfg.decoder_num_layers, move |blk| {
            decoder_block(blk, num_tokens, dim, nh, hd, n_rot, ffn, eps)
        })
        .stage(plugin_named("out", move |emit, h| {
            let x = h.ok_or_else(|| anyhow!("the video decoder stack produced no output"))?;
            let ng = emit.load_param("decoder.norm_out.weight", false)?;
            let nb = emit.load_param("decoder.norm_out.bias", false)?;
            let pw = emit.load_param("decoder.proj_out.weight", true)?;
            let pb = emit.load_param("decoder.proj_out.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let n = gb.ln(x.hir_id(), ng, nb, eps);
            let y = gb.mm(n, pw);
            let y = gb.add(y, pb);
            // Drop the register and zero tokens.
            let y = gb.narrow_(y, 1, 0, voxels);
            Ok(Some(emit.wrap(y, Shape::new(&[1, voxels, patch], f))))
        }))
        .output("pixels");

    flow.build_with(&mut WeightMapSource(weights), None)
}

#[allow(clippy::too_many_arguments)]
fn decoder_block(
    blk: usize,
    n: usize,
    dim: usize,
    nh: usize,
    hd: usize,
    n_rot: usize,
    ffn: usize,
    eps: f32,
) -> FlowStage {
    let p = format!("decoder.transformer_blocks.{blk}");
    plugin_named(format!("vae_block{blk}"), move |emit, h| {
        let x = h.ok_or_else(|| anyhow!("video decoder block {blk} needs a hidden state"))?;
        let f = DType::F32;
        let shape = Shape::new(&[1, n, dim], f);
        let inner = nh * hd;

        let n1 = emit.load_param(&format!("{p}.norm1.weight"), false)?;
        let qw = emit.load_param(&format!("{p}.attn.to_q.weight"), true)?;
        let qb = emit.load_param(&format!("{p}.attn.to_q.bias"), false)?;
        let kw = emit.load_param(&format!("{p}.attn.to_k.weight"), true)?;
        let kb = emit.load_param(&format!("{p}.attn.to_k.bias"), false)?;
        let vw = emit.load_param(&format!("{p}.attn.to_v.weight"), true)?;
        let vb = emit.load_param(&format!("{p}.attn.to_v.bias"), false)?;
        let ow = emit.load_param(&format!("{p}.attn.to_out.0.weight"), true)?;
        let ob = emit.load_param(&format!("{p}.attn.to_out.0.bias"), false)?;
        let s1 = emit.load_param(&format!("{p}.scale1"), false)?;
        let n2 = emit.load_param(&format!("{p}.norm2.weight"), false)?;
        let fw = emit.load_param(&format!("{p}.ff.net.0.proj.weight"), true)?;
        let fb = emit.load_param(&format!("{p}.ff.net.0.proj.bias"), false)?;
        let fw2 = emit.load_param(&format!("{p}.ff.net.2.weight"), true)?;
        let fb2 = emit.load_param(&format!("{p}.ff.net.2.bias"), false)?;
        let s2 = emit.load_param(&format!("{p}.scale2"), false)?;

        let z_dim = emit.named("vae_zeros_dim")?;
        let z_head = emit.named("vae_zeros_head")?;
        let o_head = emit.named("vae_ones_head")?;
        let cos = emit.named("vae_cos")?;
        let sin = emit.named("vae_sin")?;

        let mut gb = HirMut::new(emit.hir());
        let residual = x.hir_id();
        let hx = gb.rms_norm(residual, n1, z_dim, eps);

        let q = {
            let m = gb.mm(hx, qw);
            gb.add(m, qb)
        };
        let k = {
            let m = gb.mm(hx, kw);
            gb.add(m, kb)
        };
        let v = {
            let m = gb.mm(hx, vw);
            gb.add(m, vb)
        };
        // The decoder's Q/K norms carry no learnable affine, so gamma is ones.
        let mut qk = |t| {
            let flat = gb.reshape_(t, vec![1, (n * nh) as i64, hd as i64]);
            let nrm = gb.rms_norm(flat, o_head, z_head, eps);
            let back = gb.reshape_(nrm, vec![1, n as i64, inner as i64]);
            crate::rope::emit_partial_rope(&mut gb, back, cos, sin, n, nh, hd, n_rot)
        };
        let q = qk(q);
        let k = qk(k);
        let attn = gb.attention_kind(
            q,
            k,
            v,
            nh,
            hd,
            MaskKind::None,
            Shape::new(&[1, n, inner], f),
        );
        let attn = {
            let m = gb.mm(attn, ow);
            gb.add(m, ob)
        };
        // A per-channel gate rather than a plain residual — `scale1` starts at
        // zero, so an untrained block is the identity.
        let scaled = gb.mul(attn, s1);
        let x1 = gb.add(residual, scaled);

        let hx = gb.rms_norm(x1, n2, z_dim, eps);
        let up = {
            let m = gb.mm(hx, fw);
            gb.add(m, fb)
        };
        let value = gb.narrow_(up, 2, 0, ffn);
        let gate = gb.narrow_(up, 2, ffn, ffn);
        let act = gb.silu(gate);
        let prod = gb.mul(value, act);
        let down = {
            let m = gb.mm(prod, fw2);
            gb.add(m, fb2)
        };
        let scaled = gb.mul(down, s2);
        let out = gb.add(x1, scaled);
        Ok(Some(emit.wrap(out, shape)))
    })
}

/// Every parameter key the ViT decoder reads.
#[must_use]
pub fn decoder_parameter_keys(cfg: &H3VideoVaeConfig) -> Vec<String> {
    let mut keys = vec![
        "post_quant_conv.weight".to_string(),
        "post_quant_conv.bias".to_string(),
        "decoder.proj_in.weight".to_string(),
        "decoder.proj_in.bias".to_string(),
        "decoder.register_tokens".to_string(),
        "decoder.norm_out.weight".to_string(),
        "decoder.norm_out.bias".to_string(),
        "decoder.proj_out.weight".to_string(),
        "decoder.proj_out.bias".to_string(),
    ];
    for b in 0..cfg.decoder_num_layers {
        let p = format!("decoder.transformer_blocks.{b}");
        for s in [
            "norm1.weight",
            "attn.to_q.weight",
            "attn.to_q.bias",
            "attn.to_k.weight",
            "attn.to_k.bias",
            "attn.to_v.weight",
            "attn.to_v.bias",
            "attn.to_out.0.weight",
            "attn.to_out.0.bias",
            "scale1",
            "norm2.weight",
            "ff.net.0.proj.weight",
            "ff.net.0.proj.bias",
            "ff.net.2.weight",
            "ff.net.2.bias",
            "scale2",
        ] {
            keys.push(format!("{p}.{s}"));
        }
    }
    keys
}

/// Undo the per-channel latent normalization the DiT operates in.
pub fn denormalize_latents(
    latents: &mut [f32],
    cfg: &H3VideoVaeConfig,
    voxels: usize,
) -> Result<()> {
    if cfg.latents_mean.is_empty() || cfg.latents_std.is_empty() {
        return Ok(());
    }
    ensure!(
        latents.len() == cfg.latent_channels * voxels,
        "latents len {} != {} channels × {voxels} voxels",
        latents.len(),
        cfg.latent_channels
    );
    for c in 0..cfg.latent_channels {
        let (m, s) = (cfg.latents_mean[c], cfg.latents_std[c]);
        for v in &mut latents[c * voxels..(c + 1) * voxels] {
            *v = *v * s + m;
        }
    }
    Ok(())
}

/// Apply the per-channel latent normalization the DiT operates in.
pub fn normalize_latents(latents: &mut [f32], cfg: &H3VideoVaeConfig, voxels: usize) -> Result<()> {
    if cfg.latents_mean.is_empty() || cfg.latents_std.is_empty() {
        return Ok(());
    }
    ensure!(
        latents.len() == cfg.latent_channels * voxels,
        "latents len {} != {} channels × {voxels} voxels",
        latents.len(),
        cfg.latent_channels
    );
    for c in 0..cfg.latent_channels {
        let (m, s) = (cfg.latents_mean[c], cfg.latents_std[c]);
        let inv = if s != 0.0 { 1.0 / s } else { 0.0 };
        for v in &mut latents[c * voxels..(c + 1) * voxels] {
            *v = (*v - m) * inv;
        }
    }
    Ok(())
}

/// Convert decoded pixels back to display range.
///
/// The VAE works on ImageNet-normalized values; this undoes that and clamps to
/// `[0, 1]`.
pub fn to_display_range(pixels: &mut [f32], channels: usize) -> Result<()> {
    ensure!(
        channels > 0 && pixels.len().is_multiple_of(channels),
        "pixel buffer of {} does not divide into {channels} channels",
        pixels.len()
    );
    let per = pixels.len() / channels;
    for c in 0..channels.min(3) {
        let (m, s) = (crate::layout::PIXEL_MEAN[c], crate::layout::PIXEL_STD[c]);
        for v in &mut pixels[c * per..(c + 1) * per] {
            *v = (*v * s + m).clamp(0.0, 1.0);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> H3VideoVaeConfig {
        serde_json::from_str(
            r#"{"in_channels":3,"out_channels":3,"latent_channels":24,
                "block_out_channels":[128,256,256,512,512,1024],"layers_per_block":2,
                "spatial_downsample_factors":[2,2,2,2,1,1],
                "temporal_downsample_factors":[1,2,2,1,1,1],
                "norm_num_groups":32,"norm_eps":1e-06,"spatial_padding_mode":"reflect",
                "decoder_num_layers":36,"decoder_num_attention_heads":32,
                "decoder_attention_head_dim":64,"decoder_num_register_tokens":4,
                "decoder_ffn_mult":4,"decoder_rope_theta":100.0,
                "decoder_rope_dim_ratio":0.75,"decoder_norm_eps":1e-05,
                "clip_length":17,"token_drop":3}"#,
        )
        .unwrap()
    }

    #[test]
    fn decoder_geometry_matches_the_released_config() {
        let c = cfg();
        assert_eq!(c.decoder_hidden_size(), 2048);
        // dim = 64 * 0.75 = 48, and 48 / 6 = 8 frequencies per axis.
        assert_eq!(decoder_rope_freq_dim(&c), 8);
        // 3 axes x 8, doubled = 48 of the 64 head channels rotated.
        assert_eq!(2 * 3 * decoder_rope_freq_dim(&c), 48);
        assert!(2 * 3 * decoder_rope_freq_dim(&c) < c.decoder_attention_head_dim);
    }

    #[test]
    fn parameter_keys_cover_the_released_decoder() {
        let c = cfg();
        let keys = decoder_parameter_keys(&c);
        // post_quant_conv (2) + 7 top-level + 36 blocks x 16 keys.
        assert_eq!(keys.len(), 2 + 7 + 36 * 16);
        assert!(
            keys.contains(&"post_quant_conv.weight".to_string()),
            "the decode path starts at post_quant_conv, not proj_in"
        );
        assert!(keys.contains(&"decoder.register_tokens".to_string()));
        assert!(keys.contains(&"decoder.transformer_blocks.35.scale2".to_string()));
    }

    #[test]
    fn position_ids_are_length_normalized_and_centred() {
        let pos = decoder_position_ids(2, 2, 2, 0);
        assert_eq!(pos.len(), 8 * 3);
        let two_pi = std::f64::consts::TAU;
        // First voxel: (0.5/2)*2 - 1 = -0.5 on every axis.
        for k in 0..3 {
            assert!((pos[k] - two_pi * -0.5).abs() < 1e-12);
        }
        // Last voxel: (1.5/2)*2 - 1 = 0.5 on every axis.
        for k in 0..3 {
            assert!((pos[7 * 3 + k] - two_pi * 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn position_ids_are_resolution_independent() {
        // The same relative voxel lands on the same angle at any resolution.
        let a = decoder_position_ids(1, 4, 4, 0);
        let b = decoder_position_ids(1, 8, 8, 0);
        // Voxel (0, 0, 0) of both grids: -1 + 1/n differs, but the *range* is
        // the same and the first entry is closer to -1 as n grows.
        assert!(a[1] > b[1], "a finer grid must start closer to -1");
        assert!(a[1] < 0.0 && b[1] < 0.0);
        let last_a = a[(15) * 3 + 1];
        let last_b = b[(63) * 3 + 1];
        assert!(last_a < last_b, "a finer grid must end closer to +1");
    }

    #[test]
    fn suffix_tokens_sit_at_the_origin() {
        let pos = decoder_position_ids(1, 2, 2, 5);
        assert_eq!(pos.len(), (4 + 5) * 3);
        for k in 0..5 * 3 {
            assert_eq!(pos[4 * 3 + k], 0.0, "suffix token {k} is not at the origin");
        }
    }

    #[test]
    fn chunk_geometry_matches_the_released_derivation() {
        let c = cfg();
        let g = c.chunk_geometry(37);
        // clip_length 17 is not a multiple of the 4x temporal compression.
        assert_eq!(g.frame_pre_padding, 3);
        assert_eq!(g.tokens_chunk_size, 5);
        assert_eq!(g.token_overlap, 2);
        assert_eq!(g.frame_overlap, 5);
        assert_eq!(g.window_frames, 7, "every window reads 5 + 2 latent frames");
        assert_eq!(g.frames_per_window, 17);
        assert_eq!(g.num_chunks, 7);
        assert_eq!(g.pad_tokens, 0);
        // 7 windows x 17 frames + the 5-frame trailing overlap = 124, which is
        // exactly the pixel frame count 37 latent frames were encoded from.
        assert_eq!(g.num_pixel_frames, 124);
    }

    #[test]
    fn chunk_geometry_round_trips_the_frame_relation() {
        // Whatever `video_latent_num_frames` maps a pixel count to, the decoder
        // has to map back to the same pixel count.
        let c = cfg();
        for pixels in [22usize, 39, 56, 124, 141, 226, 362] {
            let latents = crate::layout::video_latent_num_frames(pixels).unwrap();
            let g = c.chunk_geometry(latents);
            assert!(
                g.is_decodable(),
                "{latents} latent frames should be decodable"
            );
            assert_eq!(
                g.num_pixel_frames, pixels,
                "{latents} latent frames should decode to {pixels} pixel frames"
            );
        }
        // Below one window the scheme yields nothing — the reference has the
        // same floor, and the 5 s minimum duration keeps generation clear of it.
        let g = c.chunk_geometry(crate::layout::video_latent_num_frames(5).unwrap());
        assert!(
            !g.is_decodable(),
            "a 5-frame clip is shorter than one window"
        );
    }

    #[test]
    fn every_window_is_the_same_size() {
        // This is what lets one compiled graph decode a whole clip.
        let c = cfg();
        for latents in [7usize, 12, 37, 102] {
            let g = c.chunk_geometry(latents);
            assert_eq!(g.window_frames, 7);
            assert!(g.is_decodable());
            // The last window must land exactly on the padded clip.
            let padded = latents + g.pad_tokens;
            let last_end = (g.num_chunks - 1) * g.tokens_chunk_size + g.window_frames;
            assert_eq!(
                last_end, padded,
                "windows must tile the padded clip exactly"
            );
        }
    }

    #[test]
    fn tiles_cover_the_released_canvas_exactly() {
        // 768x1344 at the released 256-pixel tiles with 64-pixel minimum
        // overlap: the slack is spread in whole 16-pixel steps so every
        // boundary stays latent-aligned.
        let y = split_tiles(768, TILE_MIN_SIZE, TILE_MIN_OVERLAP, 16).unwrap();
        assert_eq!(y.count(), 4);
        assert_eq!(y.covered(), 768);
        assert_eq!(y.overlaps, vec![96, 80, 80]);
        assert_eq!(y.starts, vec![0, 160, 336, 512]);

        let x = split_tiles(1344, TILE_MIN_SIZE, TILE_MIN_OVERLAP, 16).unwrap();
        assert_eq!(x.count(), 7);
        assert_eq!(x.covered(), 1344);

        // Every overlap is a whole number of latent voxels.
        for o in y.overlaps.iter().chain(&x.overlaps) {
            assert_eq!(o % 16, 0, "overlap {o} is not latent-aligned");
            assert!(*o >= TILE_MIN_OVERLAP, "overlap {o} below the minimum");
        }
    }

    #[test]
    fn tiling_covers_many_canvases_exactly() {
        for length in [256usize, 272, 512, 768, 1024, 1344, 1920] {
            let a = split_tiles(length, TILE_MIN_SIZE, TILE_MIN_OVERLAP, 16).unwrap();
            assert_eq!(a.covered(), length, "tiles must cover {length} exactly");
            // Tiles must be contiguous and in order.
            for i in 0..a.count() - 1 {
                assert_eq!(a.starts[i + 1], a.starts[i] + a.length - a.overlaps[i]);
            }
            // The last tile must end exactly on the axis.
            assert_eq!(a.starts[a.count() - 1] + a.length, length);
        }
    }

    #[test]
    fn a_canvas_below_one_tile_is_a_single_tile() {
        let a = split_tiles(128, TILE_MIN_SIZE, TILE_MIN_OVERLAP, 16).unwrap();
        assert_eq!(a.count(), 1);
        assert_eq!(a.length, 128);
        assert!(a.overlaps.is_empty());
        assert_eq!(a.covered(), 128);
    }

    #[test]
    fn stitching_reassembles_a_split_volume() {
        // Tile a known volume, then stitch it back. With identical overlapping
        // content the cross-fade is a no-op, so the result must be the original.
        let (c, t, h, w) = (2usize, 2usize, 8usize, 8usize);
        let src =
            Volume::from_data(c, t, h, w, (0..c * t * h * w).map(|i| i as f32).collect()).unwrap();
        let y = split_tiles(h, 4, 2, 1).unwrap();
        let x = split_tiles(w, 4, 2, 1).unwrap();

        let mut grid = Vec::new();
        for &ys in &y.starts {
            let mut row = Vec::new();
            for &xs in &x.starts {
                let mut tile = Volume::new(c, t, y.length, x.length);
                for ci in 0..c {
                    for ti in 0..t {
                        for dy in 0..y.length {
                            for dx in 0..x.length {
                                let s = ((ci * t + ti) * h + ys + dy) * w + xs + dx;
                                let d = ((ci * t + ti) * y.length + dy) * x.length + dx;
                                tile.data[d] = src.data[s];
                            }
                        }
                    }
                }
                row.push(tile);
            }
            grid.push(row);
        }
        let out = stitch_tiles(&grid, &y.overlaps, &x.overlaps).unwrap();
        assert_eq!((out.height, out.width), (h, w));
        for (a, b) in out.data.iter().zip(&src.data) {
            assert!((a - b).abs() < 1e-4, "stitch changed a value: {a} vs {b}");
        }
    }

    #[test]
    fn latent_normalization_round_trips() {
        let mut c = cfg();
        c.latents_mean = (0..24).map(|i| i as f32 * 0.1 - 1.0).collect();
        c.latents_std = (0..24).map(|i| 1.0 + i as f32 * 0.05).collect();
        let voxels = 4;
        let mut x: Vec<f32> = (0..24 * voxels).map(|i| (i % 7) as f32).collect();
        let original = x.clone();
        normalize_latents(&mut x, &c, voxels).unwrap();
        assert_ne!(x, original);
        denormalize_latents(&mut x, &c, voxels).unwrap();
        for (a, b) in x.iter().zip(&original) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn display_range_undoes_imagenet_normalization() {
        let mut px = vec![0.0f32; 3];
        to_display_range(&mut px, 3).unwrap();
        for (c, v) in px.iter().enumerate() {
            assert!((v - crate::layout::PIXEL_MEAN[c]).abs() < 1e-6);
        }
        // Values outside the range are clamped rather than wrapped.
        let mut px = vec![100.0f32, -100.0, 0.0];
        to_display_range(&mut px, 3).unwrap();
        assert_eq!(px[0], 1.0);
        assert_eq!(px[1], 0.0);
    }
}

// ---------------------------------------------------------------------------
// Spatial tiling
// ---------------------------------------------------------------------------

/// Default tile geometry, in **pixels**. Tiling is on by default in the
/// released pipeline, so these are the settings a normal decode runs under.
pub const TILE_MIN_SIZE: usize = 256;
/// Default minimum overlap between neighbouring tiles, in pixels.
pub const TILE_MIN_OVERLAP: usize = 64;

/// Tile placement along one axis.
///
/// Every tile is the same `length`; the last one starts earlier rather than
/// being clipped, which is what lets a single compiled graph serve them all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileAxis {
    /// Pixel offset of each tile.
    pub starts: Vec<usize>,
    /// Pixel extent of every tile.
    pub length: usize,
    /// Overlap between tile `i` and tile `i + 1`, in pixels.
    pub overlaps: Vec<usize>,
}

impl TileAxis {
    #[must_use]
    pub fn count(&self) -> usize {
        self.starts.len()
    }

    /// Total extent the stitched tiles cover.
    #[must_use]
    pub fn covered(&self) -> usize {
        self.starts.len() * self.length - self.overlaps.iter().sum::<usize>()
    }
}

/// Lay `tile_size`-wide tiles over `length` pixels.
///
/// The tile count is the smallest whose union covers `length` while keeping
/// every overlap at least `min_overlap`. The slack is then handed out
/// round-robin over the overlaps in whole `ratio` steps, so **every tile
/// boundary stays latent-aligned** — an overlap that is not a multiple of the
/// spatial compression would put a tile edge in the middle of a latent voxel.
pub fn split_tiles(
    length: usize,
    tile_size: usize,
    min_overlap: usize,
    ratio: usize,
) -> Result<TileAxis> {
    ensure!(
        length > 0 && tile_size > 0 && ratio > 0,
        "tiling needs positive extents"
    );
    if tile_size >= length {
        return Ok(TileAxis {
            starts: vec![0],
            length,
            overlaps: Vec::new(),
        });
    }
    ensure!(
        min_overlap < tile_size,
        "a minimum overlap of {min_overlap} does not fit in a {tile_size}-pixel tile"
    );

    let mut num_tiles = length.div_ceil(tile_size);
    // Grow until the tiles can actually cover the axis at the minimum overlap.
    while tile_size * num_tiles < min_overlap * (num_tiles - 1) + length {
        num_tiles += 1;
    }

    let mut overlaps = vec![min_overlap; num_tiles - 1];
    let remaining = tile_size * num_tiles - overlaps.iter().sum::<usize>() - length;
    for i in 0..remaining / ratio {
        overlaps[i % (num_tiles - 1)] += ratio;
    }

    let mut starts = vec![0usize];
    for i in 0..num_tiles - 1 {
        starts.push(starts[starts.len() - 1] + tile_size - overlaps[i]);
    }
    let axis = TileAxis {
        starts,
        length: tile_size,
        overlaps,
    };
    ensure!(
        axis.covered() == length,
        "tiles cover {} pixels, expected {length}",
        axis.covered()
    );
    Ok(axis)
}

/// Blend the tail of `a` into the head of `b` along `axis` (2 = height,
/// 3 = width of a `(C, T, H, W)` volume), returning a volume shaped like `b`.
fn blend_axis(a: &Volume, b: &Volume, extent: usize, axis: usize) -> Result<Volume> {
    let size_b = if axis == 2 { b.height } else { b.width };
    let size_a = if axis == 2 { a.height } else { a.width };
    let extent = extent.min(size_a).min(size_b);
    if extent == 0 {
        return Ok(b.clone());
    }
    let mut out = b.clone();
    for c in 0..b.channels {
        for t in 0..b.frames {
            for k in 0..extent {
                let wb = k as f32 / extent as f32;
                let wa = 1.0 - wb;
                if axis == 2 {
                    for x in 0..b.width {
                        let ia =
                            ((c * a.frames + t) * a.height + (size_a - extent + k)) * a.width + x;
                        let ib = ((c * b.frames + t) * b.height + k) * b.width + x;
                        out.data[ib] = a.data[ia] * wa + b.data[ib] * wb;
                    }
                } else {
                    for y in 0..b.height {
                        let ia =
                            ((c * a.frames + t) * a.height + y) * a.width + (size_a - extent + k);
                        let ib = ((c * b.frames + t) * b.height + y) * b.width + k;
                        out.data[ib] = a.data[ia] * wa + b.data[ib] * wb;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Trim `n` samples off the end of `axis` (2 = height, 3 = width).
fn trim_axis(v: &Volume, n: usize, axis: usize) -> Volume {
    let (h, w) = if axis == 2 {
        (v.height.saturating_sub(n), v.width)
    } else {
        (v.height, v.width.saturating_sub(n))
    };
    let mut out = Volume::new(v.channels, v.frames, h, w);
    for c in 0..v.channels {
        for t in 0..v.frames {
            for y in 0..h {
                for x in 0..w {
                    let src = ((c * v.frames + t) * v.height + y) * v.width + x;
                    let dst = ((c * out.frames + t) * h + y) * w + x;
                    out.data[dst] = v.data[src];
                }
            }
        }
    }
    out
}

/// Stitch a grid of decoded tiles into one volume.
///
/// Each tile is cross-faded with its **original** upper and left neighbours —
/// not with the already-blended versions — and then its trailing overlap is
/// trimmed, exactly as the reference does.
pub fn stitch_tiles(
    tiles: &[Vec<Volume>],
    height_overlaps: &[usize],
    width_overlaps: &[usize],
) -> Result<Volume> {
    ensure!(
        !tiles.is_empty() && !tiles[0].is_empty(),
        "no tiles to stitch"
    );
    let rows = tiles.len();
    let cols = tiles[0].len();
    let (channels, frames) = (tiles[0][0].channels, tiles[0][0].frames);

    let mut result_rows: Vec<Volume> = Vec::with_capacity(rows);
    for i in 0..rows {
        ensure!(tiles[i].len() == cols, "tile grid row {i} is ragged");
        let mut pieces: Vec<Volume> = Vec::with_capacity(cols);
        for j in 0..cols {
            let mut tile = tiles[i][j].clone();
            if i > 0 {
                tile = blend_axis(&tiles[i - 1][j], &tile, height_overlaps[i - 1], 2)?;
            }
            if j > 0 {
                tile = blend_axis(&tiles[i][j - 1], &tile, width_overlaps[j - 1], 3)?;
            }
            if i + 1 < rows {
                tile = trim_axis(&tile, height_overlaps[i], 2);
            }
            if j + 1 < cols {
                tile = trim_axis(&tile, width_overlaps[j], 3);
            }
            pieces.push(tile);
        }
        result_rows.push(concat_axis(&pieces, 3, channels, frames)?);
    }
    concat_axis(&result_rows, 2, channels, frames)
}

/// Concatenate volumes along `axis` (2 = height, 3 = width).
fn concat_axis(parts: &[Volume], axis: usize, channels: usize, frames: usize) -> Result<Volume> {
    ensure!(!parts.is_empty(), "nothing to concatenate");
    let (h, w) = if axis == 2 {
        let h: usize = parts.iter().map(|p| p.height).sum();
        ensure!(
            parts.iter().all(|p| p.width == parts[0].width),
            "height-concatenated parts must share a width"
        );
        (h, parts[0].width)
    } else {
        let w: usize = parts.iter().map(|p| p.width).sum();
        ensure!(
            parts.iter().all(|p| p.height == parts[0].height),
            "width-concatenated parts must share a height"
        );
        (parts[0].height, w)
    };
    let mut out = Volume::new(channels, frames, h, w);
    let mut off = 0usize;
    for p in parts {
        for c in 0..channels {
            for t in 0..frames {
                for y in 0..p.height {
                    for x in 0..p.width {
                        let (dy, dx) = if axis == 2 {
                            (off + y, x)
                        } else {
                            (y, off + x)
                        };
                        let src = ((c * p.frames + t) * p.height + y) * p.width + x;
                        let dst = ((c * frames + t) * h + dy) * w + dx;
                        out.data[dst] = p.data[src];
                    }
                }
            }
        }
        off += if axis == 2 { p.height } else { p.width };
    }
    Ok(out)
}

/// Decode a latent clip in spatial tiles.
///
/// `decoder` must be compiled for `(window_frames, tile_size / 16, tile_size / 16)`.
/// Tiles are laid out in **pixel** space and mapped back onto the latent grid,
/// which is why [`split_tiles`] keeps every boundary latent-aligned.
///
/// Without this a 768x1344 clip is one ~28k-token attention per window; with the
/// released 256-pixel tiles it is 4 x 7 windows of ~1.8k tokens.
pub fn decode_clip_tiled(
    decoder: &mut H3VideoDecoder,
    latents: &[f32],
    num_latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    tile_size: usize,
    min_overlap: usize,
) -> Result<Volume> {
    let cfg = decoder.config().clone();
    let ratio = cfg.spatial_compression();
    let c = cfg.latent_channels;
    ensure!(
        latents.len() == c * num_latent_frames * latent_height * latent_width,
        "latents hold {} values for {c}x{num_latent_frames}x{latent_height}x{latent_width}",
        latents.len()
    );
    let y = split_tiles(latent_height * ratio, tile_size, min_overlap, ratio)?;
    let x = split_tiles(latent_width * ratio, tile_size, min_overlap, ratio)?;
    let (tile_lh, tile_lw) = (y.length / ratio, x.length / ratio);
    ensure!(
        decoder.geometry() == (decoder.geometry().0, tile_lh, tile_lw),
        "the decoder is compiled for {:?} but the tiles are {tile_lh}x{tile_lw} latent",
        decoder.geometry()
    );

    let g = cfg.chunk_geometry(num_latent_frames);
    let hw = latent_height * latent_width;
    let mut grid: Vec<Vec<Volume>> = Vec::with_capacity(y.count());
    for &ys in &y.starts {
        let mut row: Vec<Volume> = Vec::with_capacity(x.count());
        for &xs in &x.starts {
            let (ly, lx) = (ys / ratio, xs / ratio);
            // Slice the latent sub-volume this tile covers.
            let mut sub = vec![0.0f32; c * num_latent_frames * tile_lh * tile_lw];
            for ch in 0..c {
                for t in 0..num_latent_frames {
                    for dy in 0..tile_lh {
                        let src = (ch * num_latent_frames + t) * hw + (ly + dy) * latent_width + lx;
                        let dst = ((ch * num_latent_frames + t) * tile_lh + dy) * tile_lw;
                        sub[dst..dst + tile_lw].copy_from_slice(&latents[src..src + tile_lw]);
                    }
                }
            }
            let pixels = decoder.decode_clip(&sub, num_latent_frames)?;
            row.push(Volume::from_data(
                cfg.out_channels,
                g.num_pixel_frames,
                tile_lh * PATCH_SIZE,
                tile_lw * PATCH_SIZE,
                pixels,
            )?);
        }
        grid.push(row);
    }
    let out = stitch_tiles(&grid, &y.overlaps, &x.overlaps)?;
    ensure!(
        out.height == latent_height * ratio && out.width == latent_width * ratio,
        "stitched {}x{}, expected {}x{}",
        out.height,
        out.width,
        latent_height * ratio,
        latent_width * ratio
    );
    Ok(out)
}
