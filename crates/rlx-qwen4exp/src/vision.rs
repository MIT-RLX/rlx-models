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

//! The multimodal seam: text ids + images in, one embedding stream out.
//!
//! The tower itself is [`crate::vit`]. This module is the three things around
//! it that are qwen4exp-specific rather than Qwen3-VL-generic.
//!
//! ## 1. The embedding stream is host-assembled
//!
//! Build the text flow with [`crate::FlowOptions::host_embed`] and it takes an
//! `inputs_embeds` input instead of gathering `token_embd` in the graph. The
//! graph then never has to know an image was there — and a 1.2 GB embedding
//! table stays off the device.
//!
//! ## 2. The PLE hash sees an image as one token id
//!
//! A vision row has no token id, but PLE still needs a row for every position.
//! The reference substitutes `image_token_id` (`248056`); llama.cpp carries it
//! as `qwen4exp.ple.image_token_id` for exactly this and falls back to EOS when
//! the key is absent. [`MultimodalBatch::ple_token_ids`] is the result, and it
//! is **not** the same array as the one fed to `inputs_embeds` — hashing the
//! raw placeholder ids instead shifts every n-gram across an image.
//!
//! ## 3. An image consumes `max(nx, ny)` positions, not `nx · ny`
//!
//! M-RoPE gives a vision token `[t, t + row, t + col, 0]`, so a whole image
//! advances the text position by the larger grid side — not by its token count.
//! Getting that wrong de-synchronizes every position after the image.
//!
//! ```no_run
//! use rlx_qwen4exp::vision::{ImageSpan, VisionEncoder, assemble_multimodal};
//! # fn main() -> anyhow::Result<()> {
//! # let cfg = rlx_qwen4exp::Qwen4ExpConfig::qwen38_flash_next();
//! # let embed_table: Vec<f32> = vec![];
//! # let (rgb, w, h) = (vec![0u8; 3], 1usize, 1usize);
//! let mut encoder = VisionEncoder::open(&cfg, "mmproj-F16.gguf", rlx_runtime::Device::Cpu)?;
//! let out = encoder.encode_rgb(&rgb, w, h)?;
//!
//! // `<|vision_start|> <|image_pad|> … <|vision_end|>`: the span is the pads.
//! let ids: Vec<u32> = vec![/* … */];
//! let batch = assemble_multimodal(
//!     &cfg,
//!     &ids,
//!     &embed_table,
//!     &[ImageSpan::from_encode(1, &out)?],
//! )?;
//! # let _ = batch;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result, bail};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

use crate::config::Qwen4ExpConfig;
use crate::rope::text_section_pos;
use crate::vit::{
    IMAGE_INPUT, VISION_ROPE_COS_INPUT, VISION_ROPE_SIN_INPUT, VisionConfig,
    build_qwen4exp_vision_flow, preprocess_rgb, vision_position_hw, vision_rope_tables,
};

/// The projector the published `mmproj` declares.
pub const PROJECTOR_TYPE: &str = "qwen3vl_merger";

/// One image's placeholder span in the token stream, and its embeddings.
#[derive(Debug, Clone, Copy)]
pub struct ImageSpan<'a> {
    /// Index of the first placeholder token (the first `<|image_pad|>`).
    pub start: usize,
    /// Output grid width, **after** the spatial merge.
    pub nx: usize,
    /// Output grid height, after the merge.
    pub ny: usize,
    /// `[nx · ny, hidden]` projected rows, row-major over the grid.
    pub embeds: &'a [f32],
}

impl ImageSpan<'_> {
    /// Placeholder tokens this image occupies.
    pub fn n_tokens(&self) -> usize {
        self.nx * self.ny
    }
    /// M-RoPE positions this image consumes — `max(nx, ny)`, not `nx · ny`.
    pub fn n_positions(&self) -> usize {
        self.nx.max(self.ny)
    }
}

impl<'a> ImageSpan<'a> {
    /// Build a span from a [`VisionEncodeOutput`].
    ///
    /// `start` is the index of the first `<|image_pad|>`. The encoder reports
    /// the post-merge grid, which is also the placeholder count.
    pub fn from_encode(start: usize, out: &'a VisionEncodeOutput) -> Result<Self> {
        if out.grid_x * out.grid_y != out.n_tokens {
            bail!(
                "qwen4exp: the encoder reports a {}x{} grid but {} tokens",
                out.grid_x,
                out.grid_y,
                out.n_tokens
            );
        }
        Ok(Self {
            start,
            nx: out.grid_x,
            ny: out.grid_y,
            embeds: &out.embeddings,
        })
    }
}

/// One image's projected rows, and the grid they came from.
#[derive(Debug, Clone)]
pub struct VisionEncodeOutput {
    /// `[n_tokens · llm_hidden]`, row-major over the post-merge grid.
    pub embeddings: Vec<f32>,
    /// Post-merge grid width and height.
    pub grid_x: usize,
    pub grid_y: usize,
    /// `grid_x · grid_y` — the placeholder count this image needs.
    pub n_tokens: usize,
}

/// The vision tower, compiled per distinct smart-resized image size.
///
/// One graph is built for each `(w, h)` the encoder is asked for and then
/// reused: the second photo at a given size skips both the compile and the
/// cold first run, since the parameters are already resident. Images of
/// *different* sizes each cost a compile, which is part of why
/// [`crate::vit::smart_resize`] aligns to a merge block rather than a pixel.
pub struct VisionEncoder {
    cfg: VisionConfig,
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: Device,
    cache: HashMap<(usize, usize), CompiledGraph>,
}

impl VisionEncoder {
    /// Load a mmproj GGUF and check it is this model's tower.
    pub fn open(text: &Qwen4ExpConfig, path: &str, device: Device) -> Result<Self> {
        let cfg = VisionConfig::from_gguf_path(path)?;
        check_mmproj(text, &cfg)?;
        let mut loader = rlx_core::weight_loader::load_from_path(path)
            .with_context(|| format!("qwen4exp vision: opening {path:?}"))?;
        let map = WeightMap::from_weight_loader_dequant_all(loader.as_mut())
            .with_context(|| format!("qwen4exp vision: reading {path:?}"))?;
        Ok(Self::from_tensors(cfg, weight_map_to_tensors(&map), device))
    }

    /// An encoder over an already-loaded tensor map — the seam a test uses.
    pub fn from_tensors(
        cfg: VisionConfig,
        tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
        device: Device,
    ) -> Self {
        Self {
            cfg,
            tensors,
            device,
            cache: HashMap::new(),
        }
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Image sizes with a compiled graph resident.
    pub fn compiled_sizes(&self) -> usize {
        self.cache.len()
    }

    /// Encode a packed RGB8 buffer: smart resize, normalize, run the tower.
    pub fn encode_rgb(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionEncodeOutput> {
        if rgb.len() < w * h * 3 {
            bail!(
                "qwen4exp vision: a {w}x{h} RGB buffer needs {} bytes, got {}",
                w * h * 3,
                rgb.len()
            );
        }
        let (nchw, tw, th) = preprocess_rgb(rgb, w, h, &self.cfg);
        let (gx, gy) = self.cfg.grid(tw, th);
        let n_tokens = self.cfg.n_out_tokens(tw, th);

        if !self.cache.contains_key(&(tw, th)) {
            // Each build consumes the weights it reads, so the map is cloned
            // per size rather than drained.
            let mut weights = WeightMap::from_tensors(self.tensors.clone());
            let built = build_qwen4exp_vision_flow(&self.cfg, &mut weights, tw, th)?;
            let compiled = compile_built(built, self.device)?;
            self.cache.insert((tw, th), compiled);
        }

        let position_hw = vision_position_hw(&self.cfg, tw, th);
        let (cos, sin) = vision_rope_tables(&self.cfg, &position_hw);
        let session = self.cache.get_mut(&(tw, th)).expect("just inserted");
        let embeddings = session
            .run(&[
                (IMAGE_INPUT, nchw.as_slice()),
                (VISION_ROPE_COS_INPUT, cos.as_slice()),
                (VISION_ROPE_SIN_INPUT, sin.as_slice()),
            ])
            .pop()
            .context("qwen4exp vision: the tower produced no output")?;

        let want = n_tokens * self.cfg.llm_hidden_size;
        if embeddings.len() != want {
            bail!(
                "qwen4exp vision: the tower returned {} values, expected {want} \
                 ({n_tokens} tokens x {})",
                embeddings.len(),
                self.cfg.llm_hidden_size
            );
        }
        Ok(VisionEncodeOutput {
            embeddings,
            // The POST-merge grid: that is what the M-RoPE coordinates and the
            // placeholder span are counted in.
            grid_x: gx / self.cfg.n_merge,
            grid_y: gy / self.cfg.n_merge,
            n_tokens,
        })
    }
}

/// Copy a [`WeightMap`] out into a plain map, so one set of weights can build
/// several differently-sized graphs.
fn weight_map_to_tensors(map: &WeightMap) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    map.keys()
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|k| {
            let (d, s) = map.get(&k).expect("key came from `keys`");
            (k, (d.to_vec(), s.to_vec()))
        })
        .collect()
}

/// Everything the host has to compute for a multimodal prefill.
#[derive(Debug, Clone)]
pub struct MultimodalBatch {
    /// `[seq · hidden]` — the `inputs_embeds` graph input, with vision rows
    /// spliced over the placeholders.
    pub inputs_embeds: Vec<f32>,
    /// `[seq]` — token ids **for the PLE hash**, with `image_token_id`
    /// substituted at every vision position. Not the same as the prompt ids.
    pub ple_token_ids: Vec<u32>,
    /// `[seq]` — M-RoPE section positions, `[t, h, w, 0]` per token.
    pub section_positions: Vec<[usize; 4]>,
    pub seq: usize,
}

/// Validate a mmproj against the text config.
///
/// The checks are cheap and the failure they prevent is not: a mismatched
/// `projection_dim` splices rows of the wrong width into the embedding stream,
/// which a `[seq, hidden]` buffer happily accepts.
pub fn check_mmproj(cfg: &Qwen4ExpConfig, mm: &VisionConfig) -> Result<()> {
    if mm.projector_type != PROJECTOR_TYPE {
        bail!(
            "qwen4exp: mmproj declares projector_type {:?}, expected {PROJECTOR_TYPE:?} \
             (the vision tower is an unmodified Qwen3-VL ViT)",
            mm.projector_type
        );
    }
    if mm.llm_hidden_size != cfg.hidden_size {
        bail!(
            "qwen4exp: mmproj projects to {} but the text model is {} wide",
            mm.llm_hidden_size,
            cfg.hidden_size
        );
    }
    // `VisionConfig::validate` already refuses a deepstack mmproj; re-stated
    // here so the message names the seam rather than the tower.
    if !mm.deepstack_layers.is_empty() {
        bail!(
            "qwen4exp: this mmproj declares {} deepstack layers; the published \
             checkpoint has none and the multimodal seam has no splice point \
             for the extra features",
            mm.deepstack_layers.len()
        );
    }
    Ok(())
}

/// The token id the PLE hash uses for a vision position.
///
/// `qwen4exp.ple.image_token_id` when the converter wrote it, else EOS — which
/// is what llama.cpp falls back to for files written before that key existed.
pub fn ple_vision_token(cfg: &Qwen4ExpConfig) -> u32 {
    if cfg.ple_image_token_id != 0 {
        cfg.ple_image_token_id
    } else {
        cfg.ple_eos_token_id
    }
}

/// Assemble a multimodal prefill: splice vision rows in, substitute the PLE
/// token ids, and lay out the 3-D M-RoPE positions.
///
/// `embed_table` is `token_embd` as `[vocab, hidden]` (the orientation the GGUF
/// loader reports). `images` must be sorted by `start` and non-overlapping.
pub fn assemble_multimodal(
    cfg: &Qwen4ExpConfig,
    token_ids: &[u32],
    embed_table: &[f32],
    images: &[ImageSpan<'_>],
) -> Result<MultimodalBatch> {
    let hidden = cfg.hidden_size;
    let seq = token_ids.len();
    if embed_table.len() < cfg.vocab_size * hidden {
        bail!(
            "qwen4exp: the embedding table is {} values, expected {} ({} x {hidden})",
            embed_table.len(),
            cfg.vocab_size * hidden,
            cfg.vocab_size
        );
    }

    // ── text rows ──
    let mut inputs_embeds = vec![0f32; seq * hidden];
    for (t, &id) in token_ids.iter().enumerate() {
        let src = (id as usize)
            .checked_mul(hidden)
            .filter(|o| o + hidden <= embed_table.len())
            .with_context(|| format!("qwen4exp: token id {id} is out of the embedding table"))?;
        inputs_embeds[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&embed_table[src..src + hidden]);
    }

    // ── vision rows over the placeholders, and the PLE substitution ──
    let vision_tok = ple_vision_token(cfg);
    let mut ple_token_ids = token_ids.to_vec();
    let mut last_end = 0usize;
    for img in images {
        let n = img.n_tokens();
        if img.start < last_end {
            bail!(
                "qwen4exp: image spans must be sorted and disjoint; one starts at \
                 {} but the previous ended at {last_end}",
                img.start
            );
        }
        if img.start + n > seq {
            bail!(
                "qwen4exp: an image span [{}, {}) runs past the {seq}-token prompt",
                img.start,
                img.start + n
            );
        }
        if img.embeds.len() != n * hidden {
            bail!(
                "qwen4exp: an image has {} embedding values, expected {n} x {hidden}",
                img.embeds.len()
            );
        }
        inputs_embeds[img.start * hidden..(img.start + n) * hidden].copy_from_slice(img.embeds);
        for p in &mut ple_token_ids[img.start..img.start + n] {
            *p = vision_tok;
        }
        last_end = img.start + n;
    }

    // ── M-RoPE positions ──
    //
    // Text advances by one. An image's tokens get `[t, t + row, t + col, 0]`
    // and the whole image advances `t` by `max(nx, ny)`.
    let mut section_positions = Vec::with_capacity(seq);
    let mut pos = 0usize;
    let mut t = 0usize;
    let mut it = images.iter().peekable();
    while t < seq {
        match it.peek().filter(|img| img.start == t) {
            Some(img) => {
                let n = img.n_tokens();
                for i in 0..n {
                    // `[t, t + row, t + col, 0]`, row-major over the grid.
                    section_positions.push([pos, pos + i / img.nx, pos + i % img.nx, 0]);
                }
                pos += img.n_positions();
                t += n;
                it.next();
            }
            _ => {
                section_positions.push(text_section_pos(pos));
                pos += 1;
                t += 1;
            }
        }
    }
    debug_assert_eq!(section_positions.len(), seq);

    Ok(MultimodalBatch {
        inputs_embeds,
        ple_token_ids,
        section_positions,
        seq,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Qwen4ExpConfig {
        crate::synth::tiny_config(2048)
    }

    fn table(cfg: &Qwen4ExpConfig) -> Vec<f32> {
        (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| i as f32)
            .collect()
    }

    #[test]
    fn a_text_only_batch_is_the_embedding_gather() {
        let c = cfg();
        let tbl = table(&c);
        let ids = [3u32, 1, 4];
        let b = assemble_multimodal(&c, &ids, &tbl, &[]).unwrap();
        assert_eq!(b.seq, 3);
        assert_eq!(b.ple_token_ids, ids);
        for (t, &id) in ids.iter().enumerate() {
            let want = &tbl[id as usize * c.hidden_size..(id as usize + 1) * c.hidden_size];
            assert_eq!(
                &b.inputs_embeds[t * c.hidden_size..(t + 1) * c.hidden_size],
                want
            );
        }
        assert_eq!(
            b.section_positions,
            vec![[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 0]]
        );
    }

    /// The vision rows replace the placeholders, the PLE ids become the image
    /// token, and the text after the image resumes at `max(nx, ny)` — not at
    /// `nx · ny`.
    #[test]
    fn an_image_splices_rows_and_advances_positions_by_the_grid_side() {
        let c = cfg();
        let h = c.hidden_size;
        let tbl = table(&c);
        // `a <pad × 6> b` with a 3×2 grid.
        let ids: Vec<u32> = vec![5, 0, 0, 0, 0, 0, 0, 7];
        let embeds: Vec<f32> = (0..6 * h).map(|i| -(i as f32) - 1.0).collect();
        let b = assemble_multimodal(
            &c,
            &ids,
            &tbl,
            &[ImageSpan {
                start: 1,
                nx: 3,
                ny: 2,
                embeds: &embeds,
            }],
        )
        .unwrap();

        assert_eq!(&b.inputs_embeds[h..7 * h], embeds.as_slice());
        let img_tok = ple_vision_token(&c);
        assert_eq!(
            b.ple_token_ids,
            vec![5, img_tok, img_tok, img_tok, img_tok, img_tok, img_tok, 7]
        );

        // Row-major over the grid: token i is at (row i/nx, col i%nx).
        assert_eq!(b.section_positions[0], [0, 0, 0, 0]);
        assert_eq!(b.section_positions[1], [1, 1, 1, 0]);
        assert_eq!(b.section_positions[3], [1, 1, 3, 0], "row 0, col 2");
        assert_eq!(b.section_positions[4], [1, 2, 1, 0], "row 1, col 0");
        // 6 tokens, but only max(3, 2) = 3 positions consumed.
        assert_eq!(b.section_positions[7], [4, 4, 4, 0]);
    }

    #[test]
    fn overlapping_or_overrunning_spans_are_rejected() {
        let c = cfg();
        let h = c.hidden_size;
        let tbl = table(&c);
        let ids: Vec<u32> = vec![1; 6];
        let e2: Vec<f32> = vec![0.0; 2 * h];
        let spans = [
            ImageSpan {
                start: 1,
                nx: 2,
                ny: 1,
                embeds: &e2,
            },
            ImageSpan {
                start: 2,
                nx: 2,
                ny: 1,
                embeds: &e2,
            },
        ];
        assert!(assemble_multimodal(&c, &ids, &tbl, &spans).is_err());

        let spans = [ImageSpan {
            start: 5,
            nx: 2,
            ny: 1,
            embeds: &e2,
        }];
        assert!(assemble_multimodal(&c, &ids, &tbl, &spans).is_err());
    }

    #[test]
    fn a_mismatched_mmproj_is_rejected() {
        let c = cfg();
        let base = VisionConfig {
            patch_size: 16,
            n_embd: 1152,
            n_head: 16,
            n_layer: 27,
            image_size: 768,
            n_ff: 4304,
            n_merge: 2,
            eps: 1e-6,
            projector_type: PROJECTOR_TYPE.to_string(),
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            llm_hidden_size: c.hidden_size,
            image_min_pixels: 1024,
            image_max_pixels: 1 << 20,
            rope_theta: 10_000.0,
            deepstack_layers: vec![],
        };
        check_mmproj(&c, &base).unwrap();

        let mut wrong_width = base.clone();
        wrong_width.llm_hidden_size = c.hidden_size + 8;
        assert!(check_mmproj(&c, &wrong_width).is_err());

        let mut wrong_proj = base.clone();
        wrong_proj.projector_type = "mlp".into();
        assert!(check_mmproj(&c, &wrong_proj).is_err());

        let mut deepstack = base.clone();
        deepstack.deepstack_layers = vec![3, 7];
        assert!(check_mmproj(&c, &deepstack).is_err());
    }
}
