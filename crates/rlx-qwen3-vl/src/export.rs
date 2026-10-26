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

//! Export a Hugging Face Qwen3-VL checkpoint (dense) as an `.rlxp` package
//! rlx runs alone, through rlx-qwen3's decoder exporter: the language
//! model's step and prompt functions (Qwen3 graphs; its tables read at
//! `position + rope_offset`), and a `media` entry — token embeddings, the
//! vision tower as a stage taking an RGB image of the exported size, and a
//! prefill from embeddings, interleaved multimodal RoPE rows and the
//! tower's deepstack features (added to the hidden states after the first
//! layers, at the image's positions).
//!
//! The tower is built for that one size: the pixels' rearrangement into
//! the processor's patch order, the learned position embedding (bilinearly
//! interpolated to the grid) and its 2-D RoPE rows are constants of it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::decoder::{Built, Decoder, MediaParts, export_decoder, prompt_function};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::weight_loader::WeightLoader;
use rlx_flow::blocks::{CustomStage, LmHeadStage, Qwen3DecoderSpec, qwen3_prefill_layer_side};
use rlx_flow::{FlowStage, ModelFlow, SideOutputs};
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{DType, Graph, GraphExt, HirGraphExt, NodeId, Shape};
use rlx_qwen3::Qwen3Config;
use serde_json::{Value, json};

/// The media functions' symbols: the embedding's token count, the media
/// prefill's length.
const EMBED_N: u32 = 10;
const MEDIA_SEQ: u32 = 11;
/// The checkpoint's language model, as rlx-qwen3's graphs name it.
const LM_PREFIX: &str = "model.language_model.";

/// The vision tower's settings (`config.json`'s `vision_config`).
#[derive(Debug, Clone)]
struct Vision {
    hidden: usize,
    heads: usize,
    depth: usize,
    patch: usize,
    merge: usize,
    /// The learned position grid's side (`√num_position_embeddings`).
    grid_side: usize,
    deepstack: Vec<usize>,
    mean: [f32; 3],
    std: [f32; 3],
}

struct Qwen3VlDecoder {
    /// `qwen3_vl`, or `qwen3_vl_moe`.
    family: String,
    lm: Qwen3Config,
    vision: Vision,
    sections: Vec<usize>,
    /// The processor's pixel bounds: an image is resized to a multiple of a
    /// merged patch within them.
    pixels: (usize, usize),
    image_token: u64,
    /// The video processor's bounds, over all of a video's frames.
    video_pixels: (usize, usize),
    video_token: u64,
    media_sizes: Vec<usize>,
}

/// The checkpoint with its language model's tensors named as rlx-qwen3's
/// graphs read them (`model.layers.…` is `model.language_model.layers.…`).
struct LmNames<'a>(&'a mut dyn WeightLoader);

impl LmNames<'_> {
    fn key(key: &str) -> String {
        match key.strip_prefix("model.") {
            Some(rest) if !rest.starts_with("language_model.") && !rest.starts_with("visual.") => {
                format!("{LM_PREFIX}{rest}")
            }
            _ => key.to_string(),
        }
    }
}

impl WeightLoader for LmNames<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0.take(&Self::key(key))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0.take_transposed(&Self::key(key))
    }
    fn stored_dtype(&self, key: &str) -> Option<DType> {
        self.0.stored_dtype(&Self::key(key))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.0
            .remaining_keys()
            .into_iter()
            .map(|k| match k.strip_prefix(LM_PREFIX) {
                Some(rest) => format!("model.{rest}"),
                None => k,
            })
            .collect()
    }
}

impl Decoder for Qwen3VlDecoder {
    fn family(&self) -> String {
        self.family.clone()
    }
    fn exported_by(&self) -> &str {
        "rlx-qwen3-vl"
    }
    fn max_positions(&self) -> usize {
        self.lm.max_position_embeddings
    }
    fn layers(&self) -> usize {
        self.lm.num_hidden_layers
    }
    fn vocab_size(&self) -> usize {
        self.lm.vocab_size
    }
    fn rope_inputs(&self) -> Vec<(String, String, usize)> {
        vec![("rope_cos".into(), "rope_sin".into(), self.lm.head_dim / 2)]
    }
    fn rope_row(&self, pos: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![rlx_qwen3::generator::compute_rope_slice(&self.lm, pos)]
    }
    fn step(&self, weights: &mut dyn WeightLoader, batch: usize, upper: usize) -> Result<Built> {
        rlx_qwen3::builder::build_qwen3_decode_graph_sized_ext(
            &self.lm,
            &mut LmNames(weights),
            batch,
            upper,
            true,
        )
    }
    fn prompt(&self, weights: &mut dyn WeightLoader, batch: usize, seq: usize) -> Result<Built> {
        rlx_qwen3::builder::build_qwen3_graph_sized_last_logits(
            &self.lm,
            &mut LmNames(weights),
            batch,
            seq,
            true,
        )
    }
    fn prompt_rope_params(&self) -> Vec<(String, String)> {
        vec![("rope.cos".into(), "rope.sin".into())]
    }
    fn rope_offset(&self) -> bool {
        true
    }
    fn media(
        &self,
        weights: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
        rows: usize,
        take: &mut dyn FnMut(HashMap<String, Vec<f32>>) -> Result<()>,
    ) -> Result<Option<MediaParts>> {
        let mut graphs = Vec::new();
        let embed = rlx_ir::dynamic::generalize(
            &embed_graph(&self.lm, 37),
            &embed_graph(&self.lm, 53),
            &[(EMBED_N, 37, 53)],
        )
        .map_err(anyhow::Error::msg)?;
        graphs.push(embed);

        let deepstack = self.vision.deepstack.len();
        let (mut prefill, p) = prompt_function(
            |seq| {
                let mut wm = weights()?;
                prefill_graph(&self.lm, &mut LmNames(wm.as_mut()), seq, deepstack)
            },
            MEDIA_SEQ,
            &[],
            &mut |p| take(p),
        )
        .map_err(anyhow::Error::msg)?;
        prefill.name = "prefill_embeds".into();
        take(p)?;
        graphs.push(prefill);

        let (vision, p) = vision_function(&self.vision, weights, 1)?;
        take(p)?;
        graphs.push(vision);
        let (mut video, p) = vision_function(&self.vision, weights, 2)?;
        video.name = "video".into();
        take(p)?;
        graphs.push(video);

        let mut outputs = vec!["logits".to_string()];
        for i in 0..self.layers() {
            outputs.push(format!("past_k_{i}"));
            outputs.push(format!("past_v_{i}"));
        }
        let extra: Vec<String> = (0..deepstack).map(|i| format!("deepstack_{i}")).collect();
        let mut prefill_inputs = json!({"prefill_hidden": "prompt_embeds", "last": "last",
                                        "rope_cos": "rope_cos", "rope_sin": "rope_sin"});
        let mut stage_outputs = vec!["image_embeds".to_string()];
        for e in &extra {
            prefill_inputs[e] = json!(e);
            stage_outputs.push(e.clone());
        }
        let sizes: Vec<usize> = self
            .media_sizes
            .iter()
            .copied()
            .filter(|&s| s < rows)
            .collect();
        let (min_pixels, max_pixels) = self.pixels;
        let (video_min, video_max) = self.video_pixels;
        let multiple = self.vision.patch * self.vision.merge;
        let entry = json!({
            "embed": {"function": "embed", "inputs": {"input_ids": "ids"}, "outputs": ["embeds"]},
            "prefill": {"function": "prefill_embeds", "inputs": prefill_inputs,
                        "outputs": outputs, "sizes": sizes},
            "mrope": {"sections": self.sections, "layout": "interleaved",
                      "cos": "rope_cos.table", "sin": "rope_sin.table"},
            "kinds": {"image": {
                "token": self.image_token,
                "input": "image",
                "grid": {"patch": self.vision.patch * self.vision.merge},
                "stages": [{"function": "vision", "inputs": {"image": "image"},
                            "outputs": stage_outputs}],
                "output": "image_embeds",
                "rows": extra,
                // Any size: resized as the processor does (the nearest
                // multiple of a merged patch, within its pixel bounds), then
                // the tower specialized to its grid.
                "resize": {"multiple": self.vision.patch * self.vision.merge,
                           "min_pixels": min_pixels, "max_pixels": max_pixels,
                           "filter": "bicubic"},
                "contract": {"kind": "image", "dtype": "u8", "shape": [null, null, 3], "color": "rgb"},
            }, "video": {
                // Sampled frames, two at a time through the tower; each pair
                // its own run of placeholders, after its timestamp.
                "token": self.video_token,
                "input": "video",
                "grid": {"patch": multiple},
                "frames": {"patch": 2, "layout": "runs"},
                "stages": [{"function": "video", "inputs": {"video": "video"},
                            "outputs": stage_outputs}],
                "output": "image_embeds",
                "rows": extra,
                "resize": {"multiple": multiple, "min_pixels": video_min,
                           "max_pixels": video_max, "filter": "bicubic",
                           "over_frames": true},
                "contract": {"kind": "video", "dtype": "u8", "shape": [null, null, null, 3], "color": "rgb"},
            }},
        });
        Ok(Some(MediaParts {
            graphs,
            dims: vec![
                ("n".into(), EMBED_N),
                ("media_seq".into(), MEDIA_SEQ),
                ("grid_h".into(), GRID_H),
                ("grid_w".into(), GRID_W),
            ],
            entry,
        }))
    }
}

/// `input_ids [n]` to the embedding table's rows, `[n, hidden]`.
fn embed_graph(cfg: &Qwen3Config, n: usize) -> Graph {
    let mut g = Graph::new("embed");
    let ids = g.input("input_ids", Shape::new(&[n], DType::I32));
    let table = g.param(
        "model.embed_tokens.weight",
        Shape::new(&[cfg.vocab_size, cfg.hidden_size], DType::F32),
    );
    let rows = g.gather(table, ids, 0, Shape::new(&[n, cfg.hidden_size], DType::F32));
    g.set_outputs(vec![rows]);
    g
}

/// The language model from embedded rows (`prefill_hidden`), each token's
/// RoPE rows (`rope_cos` / `rope_sin`) and `deepstack` feature inputs, the
/// `i`-th added to the hidden states after layer `i`: the last row's
/// logits, then each layer's K and V.
fn prefill_graph(
    cfg: &Qwen3Config,
    wm: &mut dyn WeightLoader,
    seq: usize,
    deepstack: usize,
) -> Result<Built> {
    let f = DType::F32;
    let h = cfg.hidden_size;
    let half = cfg.head_dim / 2;
    let hidden_shape = Shape::new(&[1, seq, h], f);
    let spec = Qwen3DecoderSpec {
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        eps: cfg.rms_norm_eps as f32,
        hidden_shape: hidden_shape.clone(),
        batch: 1,
        seq,
        qk_norm: cfg.qk_norm,
        attention_bias: cfg.attention_bias,
        mask: MaskKind::Causal,
        moe: None,
    };
    let kv = SideOutputs::new();
    let mut flow = ModelFlow::new("qwen3_vl_prefill")
        .input("prefill_hidden", hidden_shape.clone())
        .input("rope_cos", Shape::new(&[seq, half], f))
        .input("rope_sin", Shape::new(&[seq, half], f));
    for i in 0..deepstack {
        flow = flow.input(format!("deepstack_{i}"), hidden_shape.clone());
    }
    let tied = cfg.tie_word_embeddings;
    flow = flow
        .plugin_named("qwen3vl.rope.bind", |emit, _| {
            let cos = emit.flow_input("rope_cos")?.hir_id();
            let sin = emit.flow_input("rope_sin")?.hir_id();
            emit.state.rope_cos = Some(cos);
            emit.state.rope_sin = Some(sin);
            Ok(None)
        })
        .plugin_named("qwen3vl.adopt_hidden", move |emit, _| {
            // A tied head reads the embedding table.
            if tied {
                let _ = emit.load_param("model.embed_tokens.weight", false)?;
            }
            Ok(Some(emit.flow_input("prefill_hidden")?))
        })
        .zero_beta_named("zero_beta", h)
        .zero_beta_named("zero_beta.head", cfg.head_dim);
    flow = flow.repeat_layers(cfg.num_hidden_layers, {
        let spec = spec.clone();
        let sink = kv.clone();
        let moes: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|l| cfg.layer_moe(l))
            .collect();
        move |i| {
            let mut spec = spec.clone();
            spec.moe = moes[i].clone();
            let layer = qwen3_prefill_layer_side(i, spec, &sink, &SideOutputs::new(), true, false);
            if i >= deepstack {
                return layer;
            }
            let name = format!("deepstack_{i}");
            FlowStage::Sequence(vec![
                layer,
                FlowStage::Custom(CustomStage::named(
                    format!("qwen3vl.{name}"),
                    move |emit, val| {
                        let val = val.context("deepstack after a layer with no output")?;
                        let x = emit.flow_input(&name)?.hir_id();
                        let shape = val.shape.clone();
                        let sum = rlx_ir::hir::HirMut::new(emit.hir()).add(val.hir_id(), x);
                        Ok(Some(emit.wrap(sum, shape)))
                    },
                )),
            ])
        }
    });
    flow = flow
        .gather_last_token_at(1, seq)
        .final_norm(cfg.rms_norm_eps as f32);
    flow = if tied {
        flow.raw_stage(FlowStage::LmHead(LmHeadStage {
            weight_key: None,
            tie_word_embeddings: true,
            vocab_size: cfg.vocab_size,
            hidden_size: h,
            tied_param_name: "qwen3.lm_head.tied_t".into(),
        }))
        .output("logits")
    } else {
        flow.raw_stage(FlowStage::LmHead(LmHeadStage::separate(
            "lm_head.weight",
            cfg.vocab_size,
            h,
        )))
        .output("logits")
    };
    let built = flow.build(&mut rlx_core::flow_bridge::WeightLoaderSource(wm))?;
    let built = built.with_extra_hir_outputs(kv.drain());
    rlx_core::flow_util::graph_from_built(built)
}

/// The vision tower's symbols: the merged grid's rows and columns.
const GRID_H: u32 = 12;
const GRID_W: u32 = 13;
/// The longest side, in patches, the tower's in-graph positions reach.
const IOTA: usize = 1024;

/// The vision tower for an `[mh·32, mw·32, 3]` image's pixel values
/// (0–255), symbolic in the merged grid `(mh, mw)`: built at two grids no
/// other dim has, generalized, and checked at a third. `image_embeds`
/// `[mh·mw, out]`, then each deepstack feature.
/// With `frames` 2, the tower for a video's frame pair (`video`, `[2, mh·32,
/// mw·32, 3]`) instead.
fn vision_function(
    v: &Vision,
    wm: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
    frames: usize,
) -> Result<Built> {
    let (a, params) = vision_graph(v, wm, 5, 7, frames)?;
    let (b, _) = vision_graph(v, wm, 6, 11, frames)?;
    let g = rlx_ir::dynamic::generalize(&a, &b, &[(GRID_H, 5, 6), (GRID_W, 7, 11)])
        .map_err(|e| anyhow::anyhow!("the vision tower is not symbolic in its grid: {e}"))?;
    let (c, _) = vision_graph(v, wm, 4, 9, frames)?;
    let mut at = rlx_ir::DimBinding::new();
    at.set(GRID_H, 4);
    at.set(GRID_W, 9);
    let (got, want) = (
        format!("{}", rlx_ir::dynamic::specialize(&g, &at)),
        format!("{c}"),
    );
    if let Some((a, b)) = got.lines().zip(want.lines()).find(|(a, b)| a != b) {
        bail!(
            "the vision tower generalized from two grids is not the one built at a third: \
             `{a}` where it is `{b}`"
        );
    }
    Ok((g, params))
}

/// The vision tower for an `[mh·32, mw·32, 3]` image (merge 2, patch 16):
/// its positions computed in the graph from the grid, so the same graph
/// serves every size.
fn vision_graph(
    v: &Vision,
    weights: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
    mh: usize,
    mw: usize,
    frames: usize,
) -> Result<Built> {
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    // The tower's tensors, each read from the checkpoint when used.
    let wm = std::cell::RefCell::new(weights()?);
    let get = |k: &str| -> Result<(Vec<f32>, Vec<usize>)> {
        wm.borrow_mut()
            .take(&format!("model.visual.{k}"))
            .with_context(|| format!("the checkpoint has no model.visual.{k}"))
    };
    let (p, m, d) = (v.patch, v.merge, v.hidden);
    let (gh, gw) = (mh * m, mw * m);
    let n = gh * gw;
    let head_dim = d / v.heads;
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut g = Graph::new("vision");
    let mut param = |g: &mut Graph, name: &str, data: Vec<f32>, dims: &[usize]| -> NodeId {
        params.insert(name.into(), data);
        g.param(name, f(dims))
    };
    let mut linear = |g: &mut Graph,
                      param: &mut dyn FnMut(&mut Graph, &str, Vec<f32>, &[usize]) -> NodeId,
                      x: NodeId,
                      key: &str|
     -> Result<NodeId> {
        let (wt, ws) = get(&format!("{key}.weight"))?;
        let (b, _) = get(&format!("{key}.bias"))?;
        let [o, i] = ws[..] else {
            bail!("{key}.weight is {ws:?}")
        };
        let wp = param(
            g,
            &format!("vision.{key}.weight.t"),
            rlx_core::weight_map::transpose_2d(&wt, o, i),
            &[i, o],
        );
        let bp = param(g, &format!("model.visual.{key}.bias"), b, &[o]);
        let y = g.mm(x, wp);
        Ok(g.add(y, bp))
    };
    let norm = |g: &mut Graph,
                param: &mut dyn FnMut(&mut Graph, &str, Vec<f32>, &[usize]) -> NodeId,
                x: NodeId,
                key: &str|
     -> Result<NodeId> {
        let (gw_, gs) = get(&format!("{key}.weight"))?;
        let (b, _) = get(&format!("{key}.bias"))?;
        let gp = param(g, &format!("model.visual.{key}.weight"), gw_, &gs);
        let bp = param(g, &format!("model.visual.{key}.bias"), b, &gs);
        let shape = g.shape(x).clone();
        Ok(g.layer_norm(x, gp, bp, -1, 1e-6, shape))
    };
    let constant = |g: &mut Graph, data: &[f32], dims: &[usize]| -> NodeId {
        g.add_node(
            rlx_ir::Op::Constant {
                data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            },
            vec![],
            f(dims),
        )
    };

    // Pixels: normalized, then the processor's patches (`[n, 3·t·p·p]`,
    // channel, then frame, then pixel). Both frames of a still image are the
    // same, so its patch is `[n, 3·p·p]` against the frames' kernels summed.
    let (cw, cs) = get("patch_embed.proj.weight")?;
    let [o, c, t, kh, kw] = cs[..] else {
        bail!("patch_embed.proj.weight is {cs:?}")
    };
    if frames > 1 && frames != t {
        bail!("a video's frames go {t} at a time, not {frames}");
    }
    let px = if frames > 1 {
        g.input("video", f(&[frames, gh * p, gw * p, 3]))
    } else {
        g.input("image", f(&[gh * p, gw * p, 3]))
    };
    let scale: Vec<f32> = (0..3).map(|c| 1.0 / (255.0 * v.std[c])).collect();
    let shift: Vec<f32> = (0..3).map(|c| -v.mean[c] / v.std[c]).collect();
    let s = param(&mut g, "vision.pixel_scale", scale, &[3]);
    let b = param(&mut g, "vision.pixel_shift", shift, &[3]);
    let x = g.mul(px, s);
    let x = g.add(x, b);
    let (x, pw) = if frames > 1 {
        let x = g.reshape_(
            x,
            vec![
                t as i64, mh as i64, m as i64, p as i64, mw as i64, m as i64, p as i64, 3,
            ],
        );
        let x = g.transpose_(x, vec![1, 4, 2, 5, 7, 0, 3, 6]);
        let x = g.reshape_(x, vec![n as i64, (3 * t * p * p) as i64]);
        let pw = param(
            &mut g,
            "vision.patch_embed.weight.video.t",
            rlx_core::weight_map::transpose_2d(&cw, o, c * t * kh * kw),
            &[c * t * kh * kw, o],
        );
        (x, pw)
    } else {
        let x = g.reshape_(
            x,
            vec![
                mh as i64, m as i64, p as i64, mw as i64, m as i64, p as i64, 3,
            ],
        );
        let x = g.transpose_(x, vec![0, 3, 1, 4, 6, 2, 5]);
        let x = g.reshape_(x, vec![n as i64, (3 * p * p) as i64]);
        let mut summed = vec![0f32; o * c * kh * kw];
        for oc in 0..o * c {
            for f_ in 0..t {
                let at = (oc * t + f_) * kh * kw;
                for k in 0..kh * kw {
                    summed[oc * kh * kw + k] += cw[at + k];
                }
            }
        }
        let pw = param(
            &mut g,
            "vision.patch_embed.weight.t",
            rlx_core::weight_map::transpose_2d(&summed, o, c * kh * kw),
            &[c * kh * kw, o],
        );
        (x, pw)
    };
    let (pb, _) = get("patch_embed.proj.bias")?;
    let pb = param(&mut g, "model.visual.patch_embed.proj.bias", pb, &[o]);
    let x = g.mm(x, pw);
    let x = g.add(x, pb);

    // Each patch's row and column, in patch order (merge blocks row by row,
    // 2×2 within): `block·m + offset`, broadcast over the grid.
    let iota: Vec<f32> = (0..IOTA).map(|i| i as f32).collect();
    let iota = param(&mut g, "vision.iota", iota, &[IOTA]);
    let grid4 = f(&[mh, mw, m, m]);
    let mf = constant(&mut g, &[m as f32], &[1]);
    let offsets: Vec<f32> = (0..m).map(|i| i as f32).collect();
    let coord = |g: &mut Graph, blocks: usize, block_dims: [usize; 4], off_dims: [usize; 4]| {
        let idx = g.narrow_(iota, 0, 0, blocks);
        let idx = g.reshape_(idx, block_dims.iter().map(|&x| x as i64).collect());
        let idx = g.mul(idx, mf);
        let off = constant(g, &offsets, &off_dims);
        let idx = g.add(idx, off);
        let idx = g.add_node(
            rlx_ir::Op::Expand {
                target_shape: vec![mh as i64, mw as i64, m as i64, m as i64],
            },
            vec![idx],
            grid4.clone(),
        );
        g.reshape_(idx, vec![n as i64])
    };
    let rows = coord(&mut g, mh, [mh, 1, 1, 1], [1, 1, m, 1]);
    let cols = coord(&mut g, mw, [1, mw, 1, 1], [1, 1, 1, m]);

    // The learned position grid, bilinearly sampled at `linspace(0, side -
    // 1, gh)` × `(…, gw)`, as HF's `fast_pos_embed_interpolate`.
    let side = v.grid_side;
    let last = constant(&mut g, &[(side - 1) as f32], &[1]);
    let sidef = constant(&mut g, &[side as f32], &[1]);
    let one = constant(&mut g, &[1.0], &[1]);
    let sample = |g: &mut Graph, at: NodeId| {
        let top = g.reduce(at, rlx_ir::op::ReduceOp::Max, vec![0], false, f(&[]));
        let y = g.mul(at, last);
        let y = g.div(y, top);
        let y0 = g.activation(Activation::Floor, y, f(&[n]));
        let up = g.add(y0, one);
        let y1 = g.binary(rlx_ir::op::BinaryOp::Min, up, last, f(&[n]));
        let dy = g.sub(y, y0);
        (y0, y1, dy)
    };
    let (y0, y1, dy) = sample(&mut g, rows);
    let (x0, x1, dx) = sample(&mut g, cols);
    let (table, _) = get("pos_embed.weight")?;
    let table = param(
        &mut g,
        "model.visual.pos_embed.weight",
        table,
        &[side * side, d],
    );
    let ndy = g.sub(one, dy);
    let ndx = g.sub(one, dx);
    let mut pos = None;
    for (yy, xx, wy, wx) in [
        (y0, x0, ndy, ndx),
        (y0, x1, ndy, dx),
        (y1, x0, dy, ndx),
        (y1, x1, dy, dx),
    ] {
        let at = g.mul(yy, sidef);
        let at = g.add(at, xx);
        let at = g.cast(at, DType::I64);
        let rows = g.gather(table, at, 0, f(&[n, d]));
        let wt = g.mul(wy, wx);
        let wt = g.reshape_(wt, vec![n as i64, 1]);
        let term = g.mul(rows, wt);
        pos = Some(match pos {
            None => term,
            Some(acc) => g.add(acc, term),
        });
    }
    let x = g.add(x, pos.expect("four corners"));

    // 2-D RoPE: a patch's row frequencies, then its column's.
    let dim = head_dim / 2;
    let inv: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32))
        .collect();
    let inv = constant(&mut g, &inv, &[1, dim / 2]);
    let angles = |g: &mut Graph, at: NodeId| {
        let at = g.reshape_(at, vec![n as i64, 1]);
        g.mul(at, inv)
    };
    let ar = angles(&mut g, rows);
    let ac = angles(&mut g, cols);
    let ang = g.concat(vec![ar, ac], 1, f(&[n, dim]));
    let cos = g.activation(Activation::Cos, ang, f(&[n, dim]));
    let sin = g.activation(Activation::Sin, ang, f(&[n, dim]));

    let mut hidden = g.reshape_(x, vec![1, n as i64, d as i64]);
    let mut deep = Vec::new();
    for l in 0..v.depth {
        let pre = format!("blocks.{l}");
        let y = norm(&mut g, &mut param, hidden, &format!("{pre}.norm1"))?;
        let qkv = linear(&mut g, &mut param, y, &format!("{pre}.attn.qkv"))?;
        let part = |g: &mut Graph, i: usize| {
            let q = g.narrow_(qkv, 2, i * d, d);
            g.rope(q, cos, sin, head_dim)
        };
        let q = part(&mut g, 0);
        let k = part(&mut g, 1);
        let vv = g.narrow_(qkv, 2, 2 * d, d);
        let shape = g.shape(q).clone();
        let a = g.attention_kind(q, k, vv, v.heads, head_dim, MaskKind::None, shape);
        let a = linear(&mut g, &mut param, a, &format!("{pre}.attn.proj"))?;
        hidden = g.add(hidden, a);
        let y = norm(&mut g, &mut param, hidden, &format!("{pre}.norm2"))?;
        let y = linear(&mut g, &mut param, y, &format!("{pre}.mlp.linear_fc1"))?;
        let shape = g.shape(y).clone();
        let y = g.activation(Activation::GeluApprox, y, shape);
        let y = linear(&mut g, &mut param, y, &format!("{pre}.mlp.linear_fc2"))?;
        hidden = g.add(hidden, y);
        if let Some(j) = v.deepstack.iter().position(|&i| i == l) {
            // Merged first, then normed.
            let y = g.reshape_(hidden, vec![(n / (m * m)) as i64, (d * m * m) as i64]);
            let key = format!("deepstack_merger_list.{j}");
            let y = norm(&mut g, &mut param, y, &format!("{key}.norm"))?;
            deep.push(merger_mlp(&mut g, &mut param, &mut linear, y, &key)?);
        }
    }
    // Normed, then merged.
    let y = norm(&mut g, &mut param, hidden, "merger.norm")?;
    let y = g.reshape_(y, vec![(n / (m * m)) as i64, (d * m * m) as i64]);
    let out = merger_mlp(&mut g, &mut param, &mut linear, y, "merger")?;
    let mut outs = vec![out];
    outs.extend(deep);
    g.set_outputs(outs);
    Ok((g, params))
}

/// A merger's `fc1 → GELU (erf) → fc2`.
fn merger_mlp(
    g: &mut Graph,
    param: &mut dyn FnMut(&mut Graph, &str, Vec<f32>, &[usize]) -> NodeId,
    linear: &mut dyn FnMut(
        &mut Graph,
        &mut dyn FnMut(&mut Graph, &str, Vec<f32>, &[usize]) -> NodeId,
        NodeId,
        &str,
    ) -> Result<NodeId>,
    x: NodeId,
    key: &str,
) -> Result<NodeId> {
    let y = linear(g, param, x, &format!("{key}.linear_fc1"))?;
    let shape = g.shape(y).clone();
    let y = g.activation(Activation::Gelu, y, shape);
    linear(g, param, y, &format!("{key}.linear_fc2"))
}

/// What [`export_rlxp`] builds besides [`ExportOptions`].
#[derive(Debug, Clone)]
pub struct VlExportOptions {
    /// Prompt lengths the media prefill pads to.
    pub media_sizes: Vec<usize>,
}

impl Default for VlExportOptions {
    fn default() -> Self {
        Self {
            media_sizes: vec![256, 512, 1024, 2048],
        }
    }
}

/// Export the dense Qwen3-VL checkpoint in `dir` (`config.json`,
/// `preprocessor_config.json`, the safetensors, `tokenizer.json`,
/// `tokenizer_config.json`) as the package `out`.
pub fn export_rlxp(
    dir: &Path,
    out: &Path,
    opts: &ExportOptions,
    vl: &VlExportOptions,
) -> Result<ExportReport> {
    let text = std::fs::read_to_string(dir.join("config.json"))?;
    let raw: Value = serde_json::from_str(&text)?;
    let family = raw
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if !matches!(family.as_str(), "qwen3_vl" | "qwen3_vl_moe") {
        bail!("model_type {family:?}: this exporter builds qwen3_vl and qwen3_vl_moe");
    }
    let lm = Qwen3Config::from_json_str(&text)?;
    let tc = &raw["text_config"];
    if lm.is_moe() {
        // Every layer's MLP routed (the released layout `layer_moe` builds).
        let step = tc.get("decoder_sparse_step").and_then(Value::as_u64);
        let dense = tc
            .get("mlp_only_layers")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty());
        if step.is_some_and(|s| s != 1) || dense {
            bail!(
                "MoE layers every {} layer(s), dense layers {}: only every layer's MLP routed is built",
                step.unwrap_or(1),
                tc["mlp_only_layers"]
            );
        }
    }
    let scaling = tc.get("rope_scaling").or_else(|| tc.get("rope_parameters"));
    if !scaling
        .and_then(|s| s.get("mrope_interleaved"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        bail!("Qwen3-VL's multimodal RoPE is interleaved; this config says otherwise");
    }
    let sections: Vec<usize> = scaling
        .and_then(|s| s.get("mrope_section"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|x| x as usize)
                .collect()
        })
        .unwrap_or_else(|| vec![24, 20, 20]);
    let vc = &raw["vision_config"];
    let u = |k: &str| -> Result<usize> {
        vc.get(k)
            .and_then(Value::as_u64)
            .map(|x| x as usize)
            .with_context(|| format!("vision_config.{k}"))
    };
    let pre: Value = std::fs::read(dir.join("preprocessor_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    let three = |k: &str| -> [f32; 3] {
        pre.get(k)
            .and_then(Value::as_array)
            .filter(|a| a.len() == 3)
            .map(|a| {
                let f = |i: usize| a[i].as_f64().unwrap_or(0.5) as f32;
                [f(0), f(1), f(2)]
            })
            .unwrap_or([0.5; 3])
    };
    let positions = u("num_position_embeddings")?;
    let side = (positions as f64).sqrt() as usize;
    if side * side != positions {
        bail!("vision_config.num_position_embeddings {positions} is not a square grid");
    }
    if vc.get("hidden_act").and_then(Value::as_str) != Some("gelu_pytorch_tanh") {
        bail!(
            "vision_config.hidden_act {}: this exporter builds gelu_pytorch_tanh",
            vc["hidden_act"]
        );
    }
    let vision = Vision {
        hidden: u("hidden_size")?,
        heads: u("num_heads")?,
        depth: u("depth")?,
        patch: u("patch_size")?,
        merge: u("spatial_merge_size")?,
        grid_side: side,
        deepstack: vc
            .get("deepstack_visual_indexes")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default(),
        mean: three("image_mean"),
        std: three("image_std"),
    };
    let edge = |k: &str, d: usize| {
        pre.pointer(&format!("/size/{k}"))
            .and_then(Value::as_u64)
            .map_or(d, |x| x as usize)
    };
    let video_pre: Value = std::fs::read(dir.join("video_preprocessor_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    // Without the file transformers makes the video processor from
    // `preprocessor_config.json`: the image's bounds.
    let video_edge = |k: &str, d: usize| {
        video_pre
            .pointer(&format!("/size/{k}"))
            .and_then(Value::as_u64)
            .map_or(d, |x| x as usize)
    };
    let model = Qwen3VlDecoder {
        family,
        lm,
        vision,
        sections,
        pixels: (edge("shortest_edge", 65536), edge("longest_edge", 16777216)),
        image_token: raw
            .get("image_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(151655),
        video_pixels: (
            video_edge("shortest_edge", edge("shortest_edge", 65536)),
            video_edge("longest_edge", edge("longest_edge", 16777216)),
        ),
        video_token: raw
            .get("video_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(151656),
        media_sizes: vl.media_sizes.clone(),
    };
    export_decoder(&model, dir, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `qwen3-vl` runner's name.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("qwen3-vl", "image-text-to-text", |dir, out, opts| {
            export_rlxp(dir, out, opts, &VlExportOptions::default())
        })
        .claims(&["qwen3_vl", "qwen3_vl_moe"]),
    );
}
