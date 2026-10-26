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

//! Export a Hugging Face Qwen2.5-VL checkpoint as an `.rlxp` package rlx
//! runs alone, through rlx-qwen3's decoder exporter: the language model's
//! step (its tables read at `position + rope_offset`) and prompt functions,
//! and a `media` entry: token embeddings, a prefill that takes embeddings
//! and each token's multimodal RoPE rows, and the vision tower as a stage
//! that takes an RGB image (`u8 [height, width, 3]`) of the exported size
//! and returns its rows of the language model's input.
//!
//! The vision graph is built for that one size: its window order, window
//! mask and RoPE rows are constants of the size, frozen into it, and the
//! pixels' rescale and normalization are its first ops.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::export::decoder::{Built, Decoder, MediaParts, export_decoder, prompt_function};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_qwen3::Qwen3Config;
use serde_json::{Value, json};

use crate::config::{Qwen25VlConfig, Qwen25VlHfConfig};
use crate::lm_flow::{Qwen25VlPrefillOpts, build_qwen25_vl_prefill_mrope_built};
use crate::vision::MmProjConfig;
use crate::vision::preprocess::DEFAULT_ATTN_WINDOW_SIZE;

/// The media functions' symbols: the embedding's token count, the media
/// prefill's length.
const EMBED_N: u32 = 10;
const MEDIA_SEQ: u32 = 11;

/// The `<|image_pad|>` id, when the config does not say.
const IMAGE_TOKEN: u64 = 151655;

struct VlDecoder {
    vl: Qwen25VlConfig,
    vision: MmProjConfig,
    /// Window attention's window, in patches a side.
    window: usize,
    image_token: u64,
    video_token: u64,
    /// The video processor's bounds, each frame's.
    video_pixels: (usize, usize),
    /// Time positions between a video's frame pairs.
    video_t_step: usize,
    /// Prompt lengths the media prefill pads to.
    media_sizes: Vec<usize>,
}

impl VlDecoder {
    fn lm(&self) -> &Qwen3Config {
        &self.vl.lm.lm
    }
}

impl Decoder for VlDecoder {
    fn family(&self) -> String {
        "qwen2_5_vl".into()
    }
    fn exported_by(&self) -> &str {
        "rlx-qwen25-vl"
    }
    fn max_positions(&self) -> usize {
        self.lm().max_position_embeddings
    }
    fn layers(&self) -> usize {
        self.lm().num_hidden_layers
    }
    fn vocab_size(&self) -> usize {
        self.lm().vocab_size
    }
    fn rope_inputs(&self) -> Vec<(String, String, usize)> {
        vec![("rope_cos".into(), "rope_sin".into(), self.lm().head_dim / 2)]
    }
    fn rope_row(&self, pos: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![rlx_qwen3::generator::compute_rope_slice(self.lm(), pos)]
    }
    fn step(&self, weights: &mut dyn WeightLoader, batch: usize, upper: usize) -> Result<Built> {
        rlx_qwen3::builder::build_qwen3_decode_graph_sized_ext(
            self.lm(),
            weights,
            batch,
            upper,
            true,
        )
    }
    fn prompt(&self, weights: &mut dyn WeightLoader, batch: usize, seq: usize) -> Result<Built> {
        rlx_qwen3::builder::build_qwen3_graph_sized_last_logits(
            self.lm(),
            weights,
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

        // Token ids to the language model's input rows.
        let embed = rlx_ir::dynamic::generalize(
            &embed_graph(self.lm(), 37),
            &embed_graph(self.lm(), 53),
            &[(EMBED_N, 37, 53)],
        )
        .map_err(anyhow::Error::msg)?;
        graphs.push(embed);

        // The prompt from its rows and RoPE rows; its lm head reads the
        // step's tied table.
        let vl_lm = self.vl.lm.clone();
        let tied = "qwen25vl.lm_head.tied_t".to_string();
        let (mut prefill, mut p) = prompt_function(
            |seq| {
                let mut wm = weights()?;
                let mut opts = Qwen25VlPrefillOpts::vlm_prefill(1, seq);
                opts.with_lm_head = true;
                rlx_core::flow_util::graph_from_built(build_qwen25_vl_prefill_mrope_built(
                    &vl_lm,
                    wm.as_mut(),
                    &opts,
                    None,
                )?)
            },
            MEDIA_SEQ,
            std::slice::from_ref(&tied),
            &mut |p| take(p),
        )
        .map_err(anyhow::Error::msg)?;
        prefill.name = "prefill_embeds".into();
        if self.lm().tie_word_embeddings {
            rename_param(
                &mut prefill,
                &mut p,
                "qwen25vl.lm_head.tied_t",
                "qwen3.lm_head.tied_t",
            );
        }
        take(p)?;
        graphs.push(prefill);

        // The vision tower at the exported size.
        let (vision, p) = self.vision_function(weights, 1)?;
        take(p)?;
        graphs.push(vision);
        let (mut video, p) = self.vision_function(weights, 2)?;
        video.name = "video".into();
        take(p)?;
        graphs.push(video);

        let layers = self.layers();
        let mut outputs = vec!["logits".to_string()];
        for i in 0..layers {
            outputs.push(format!("past_k_{i}"));
            outputs.push(format!("past_v_{i}"));
        }
        let sections = &self.vl.lm.mrope_sections[..3];
        let patch = self.vision.patch_size * self.vision.n_merge;
        let sizes: Vec<usize> = self
            .media_sizes
            .iter()
            .copied()
            .filter(|&s| s < rows)
            .collect();
        let entry = json!({
            "embed": {"function": "embed", "inputs": {"input_ids": "ids"}, "outputs": ["embeds"]},
            "prefill": {
                "function": "prefill_embeds",
                "inputs": {"prefill_hidden": "prompt_embeds", "last": "last",
                           "rope_cos": "rope_cos", "rope_sin": "rope_sin"},
                "outputs": outputs,
                "sizes": sizes,
            },
            "mrope": {"sections": sections, "cos": "rope_cos.table", "sin": "rope_sin.table"},
            "kinds": {"image": {
                "token": self.image_token,
                "input": "image",
                "grid": {"patch": patch},
                "stages": [{"function": "vision", "inputs": {"image": "image"},
                            "outputs": ["image_embeds"]}],
                "output": "image_embeds",
                // Any size: resized as the processor does, then the tower
                // specialized to its grid.
                "resize": {"multiple": patch, "min_pixels": self.vision.image_min_pixels,
                           "max_pixels": self.vision.image_max_pixels, "filter": "bicubic"},
                "contract": {"kind": "image", "dtype": "u8", "shape": [null, null, 3], "color": "rgb"},
            }, "video": {
                // Sampled frames, two at a time through the tower, in one run
                // of placeholders: pair `g` at time position `g · t_step`
                // (`tokens_per_second · 2 / fps`, at the processor's 2 fps).
                "token": self.video_token,
                "input": "video",
                "grid": {"patch": patch},
                "frames": {"patch": 2, "layout": "joined", "t_step": self.video_t_step},
                "stages": [{"function": "video", "inputs": {"video": "video"},
                            "outputs": ["image_embeds"]}],
                "output": "image_embeds",
                "resize": {"multiple": patch, "min_pixels": self.video_pixels.0,
                           "max_pixels": self.video_pixels.1, "filter": "bicubic"},
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

/// Param `from` of `g` (and `params`) as `to`.
fn rename_param(g: &mut Graph, params: &mut HashMap<String, Vec<f32>>, from: &str, to: &str) {
    let ids: Vec<_> = g
        .nodes()
        .iter()
        .filter(|n| matches!(&n.op, Op::Param { name } if name == from))
        .map(|n| n.id)
        .collect();
    for id in ids {
        let node = g.node_mut(id);
        node.op = Op::Param { name: to.into() };
        node.name = Some(to.into());
    }
    if let Some(v) = params.remove(from) {
        params.insert(to.into(), v);
    }
}

/// The vision tower's symbols: the merged grid's rows and columns.
const GRID_H: u32 = 12;
const GRID_W: u32 = 13;
/// The longest side, in patches, the tower's in-graph positions reach.
const IOTA: usize = 1024;

impl VlDecoder {
    /// The vision tower symbolic in its merged grid `(mh, mw)`: built at two
    /// grids no other dim has, generalized, and checked at a third. With
    /// `frames` 2, the tower for a video's frame pair (`video`, `[2, …]`).
    fn vision_function(
        &self,
        wm: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
        frames: usize,
    ) -> Result<Built> {
        let (a, params) = self.vision_graph(wm, 5, 7, frames)?;
        let (b, _) = self.vision_graph(wm, 6, 11, frames)?;
        let g = rlx_ir::dynamic::generalize(&a, &b, &[(GRID_H, 5, 6), (GRID_W, 7, 11)])
            .map_err(|e| anyhow::anyhow!("the vision tower is not symbolic in its grid: {e}"))?;
        let (c, _) = self.vision_graph(wm, 4, 9, frames)?;
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

    /// The vision tower for an `[mh·m·p, mw·m·p, 3]` image's pixel values
    /// (0–255): its patches in merge order (2×2 blocks, as the processor
    /// lays them out), each patch's 2-D RoPE rows and window computed from
    /// the grid. Window attention is a bias admitting only a token's own
    /// window — the same attention as HF's reordering into windows, with
    /// the tokens left where they are.
    fn vision_graph(
        &self,
        weights: &dyn Fn() -> Result<Box<dyn WeightLoader>>,
        mh: usize,
        mw: usize,
        frames: usize,
    ) -> Result<Built> {
        let cfg = &self.vision;
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        // The tower's tensors, each read from the checkpoint when used.
        let wm = std::cell::RefCell::new(weights()?);
        let get = |k: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            wm.borrow_mut()
                .take(&format!("visual.{k}"))
                .with_context(|| format!("the checkpoint has no visual.{k}"))
        };
        let (p, m, d) = (cfg.patch_size, cfg.n_merge, cfg.n_embd);
        let (gh, gw) = (mh * m, mw * m);
        let n = gh * gw;
        let heads = cfg.n_head;
        let head_dim = d / heads;
        let eps = cfg.eps as f32;
        let mut params: HashMap<String, Vec<f32>> = HashMap::new();
        let mut g = Graph::new("vision");
        let mut param = |g: &mut Graph, name: &str, data: Vec<f32>, dims: &[usize]| -> NodeId {
            params.insert(name.into(), data);
            g.param(name, f(dims))
        };
        let constant = |g: &mut Graph, data: &[f32], dims: &[usize]| -> NodeId {
            g.add_node(
                Op::Constant {
                    data: data.iter().flat_map(|x| x.to_le_bytes()).collect(),
                },
                vec![],
                f(dims),
            )
        };
        // `x · Wᵀ + b` for a torch `[out, in]` weight.
        let linear = |g: &mut Graph,
                      param: &mut dyn FnMut(&mut Graph, &str, Vec<f32>, &[usize]) -> NodeId,
                      x: NodeId,
                      key: &str|
         -> Result<NodeId> {
            let (wt, ws) = get(&format!("{key}.weight"))?;
            let [o, i] = ws[..] else {
                bail!("{key}.weight is {ws:?}")
            };
            let wp = param(
                g,
                &format!("vision.{key}.weight.t"),
                rlx_core::weight_map::transpose_2d(&wt, o, i),
                &[i, o],
            );
            let y = g.mm(x, wp);
            Ok(match get(&format!("{key}.bias")) {
                Ok((b, _)) => {
                    let bp = param(g, &format!("visual.{key}.bias"), b, &[o]);
                    g.add(y, bp)
                }
                Err(_) => y,
            })
        };
        let zeros = vec![0f32; d];
        let norm = |g: &mut Graph,
                    param: &mut dyn FnMut(&mut Graph, &str, Vec<f32>, &[usize]) -> NodeId,
                    x: NodeId,
                    key: &str|
         -> Result<NodeId> {
            let (gw_, gs) = get(&format!("{key}.weight"))?;
            let gp = param(g, &format!("visual.{key}.weight"), gw_, &gs);
            let bp = param(g, "vision.zero_beta", zeros.clone(), &gs);
            Ok(g.rms_norm(x, gp, bp, eps))
        };

        // Pixels: normalized, then the processor's patches (`[n, 3·t·p·p]`,
        // channel, then frame, then pixel). Both frames of a still image are
        // the same, so its patch is `[n, 3·p·p]` against the frames' kernels
        // summed.
        let (cw, cs) = get("patch_embed.proj.weight")?;
        let [o, c, t, kh, kw] = cs[..] else {
            bail!("visual.patch_embed.proj.weight is {cs:?}")
        };
        if frames > 1 && frames != t {
            bail!("a video's frames go {t} at a time, not {frames}");
        }
        let px = if frames > 1 {
            g.input("video", f(&[frames, gh * p, gw * p, 3]))
        } else {
            g.input("image", f(&[gh * p, gw * p, 3]))
        };
        let scale: Vec<f32> = (0..3).map(|c| 1.0 / (255.0 * cfg.image_std[c])).collect();
        let shift: Vec<f32> = (0..3)
            .map(|c| -cfg.image_mean[c] / cfg.image_std[c])
            .collect();
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
        let x = g.mm(x, pw);

        // Each patch's row and column, in merge order.
        let iota: Vec<f32> = (0..IOTA).map(|i| i as f32).collect();
        let iota = param(&mut g, "vision.iota", iota, &[IOTA]);
        let mf = constant(&mut g, &[m as f32], &[1]);
        let offsets: Vec<f32> = (0..m).map(|i| i as f32).collect();
        let grid4 = f(&[mh, mw, m, m]);
        let coord = |g: &mut Graph, blocks: usize, block_dims: [usize; 4], off_dims: [usize; 4]| {
            let idx = g.narrow_(iota, 0, 0, blocks);
            let idx = g.reshape_(idx, block_dims.iter().map(|&x| x as i64).collect());
            let idx = g.mul(idx, mf);
            let off = constant(g, &offsets, &off_dims);
            let idx = g.add(idx, off);
            let idx = g.add_node(
                Op::Expand {
                    target_shape: vec![mh as i64, mw as i64, m as i64, m as i64],
                },
                vec![idx],
                grid4.clone(),
            );
            g.reshape_(idx, vec![n as i64])
        };
        let rows = coord(&mut g, mh, [mh, 1, 1, 1], [1, 1, m, 1]);
        let cols = coord(&mut g, mw, [1, mw, 1, 1], [1, 1, 1, m]);

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

        // Each patch's window (`window` patches a side), and the bias that
        // keeps attention inside it.
        let win = constant(&mut g, &[self.window as f32], &[1]);
        let one = constant(&mut g, &[1.0], &[1]);
        let wr = g.div(rows, win);
        let wr = g.activation(Activation::Floor, wr, f(&[n]));
        let wc = g.div(cols, win);
        let wc = g.activation(Activation::Floor, wc, f(&[n]));
        let last_col = g.reduce(cols, rlx_ir::op::ReduceOp::Max, vec![0], false, f(&[]));
        let across = g.div(last_col, win);
        let shape = g.shape(across).clone();
        let across = g.activation(Activation::Floor, across, shape);
        let across = g.add(across, one);
        let id = g.mul(wr, across);
        let id = g.add(id, wc);
        // Both sides `[n, n]`: not every backend's compare broadcasts.
        let square = |g: &mut Graph, x: NodeId| {
            g.add_node(
                Op::Expand {
                    target_shape: vec![n as i64, n as i64],
                },
                vec![x],
                f(&[n, n]),
            )
        };
        let qid = g.reshape_(id, vec![n as i64, 1]);
        let qid = square(&mut g, qid);
        let kid = g.reshape_(id, vec![1, n as i64]);
        let kid = square(&mut g, kid);
        let same = g.add_node(
            Op::Compare(rlx_ir::op::CmpOp::Eq),
            vec![qid, kid],
            Shape::new(&[n, n], DType::Bool),
        );
        let same = g.cast(same, DType::F32);
        let big = constant(&mut g, &[1e9], &[1]);
        let off = g.sub(same, one);
        let bias = g.mul(off, big);
        let bias = g.reshape_(bias, vec![1, 1, n as i64, n as i64]);
        let bias = g.add_node(
            Op::Expand {
                target_shape: vec![1, heads as i64, n as i64, n as i64],
            },
            vec![bias],
            f(&[1, heads, n, n]),
        );

        let mut hidden = g.reshape_(x, vec![1, n as i64, d as i64]);
        for l in 0..cfg.n_layer {
            let pre = format!("blocks.{l}");
            let full = cfg.n_wa_pattern == 0 || (l + 1) % cfg.n_wa_pattern == 0;
            let y = norm(&mut g, &mut param, hidden, &format!("{pre}.norm1"))?;
            let qkv = linear(&mut g, &mut param, y, &format!("{pre}.attn.qkv"))?;
            let part = |g: &mut Graph, i: usize| {
                let q = g.narrow_(qkv, 2, i * d, d);
                g.rope(q, cos, sin, head_dim)
            };
            let q = part(&mut g, 0);
            let k = part(&mut g, 1);
            let v = g.narrow_(qkv, 2, 2 * d, d);
            let shape = g.shape(q).clone();
            let a = if full {
                g.attention_kind(q, k, v, heads, head_dim, MaskKind::None, shape)
            } else {
                g.attention_bias(q, k, v, bias, heads, head_dim, shape)
            };
            let a = linear(&mut g, &mut param, a, &format!("{pre}.attn.proj"))?;
            hidden = g.add(hidden, a);
            let y = norm(&mut g, &mut param, hidden, &format!("{pre}.norm2"))?;
            let gate = linear(&mut g, &mut param, y, &format!("{pre}.mlp.gate_proj"))?;
            let shape = g.shape(gate).clone();
            let gate = g.activation(Activation::Silu, gate, shape);
            let up = linear(&mut g, &mut param, y, &format!("{pre}.mlp.up_proj"))?;
            let y = g.mul(gate, up);
            let y = linear(&mut g, &mut param, y, &format!("{pre}.mlp.down_proj"))?;
            hidden = g.add(hidden, y);
        }
        // Normed, merged, then the merger's MLP.
        let y = norm(&mut g, &mut param, hidden, "merger.ln_q")?;
        let y = g.reshape_(y, vec![(n / (m * m)) as i64, (d * m * m) as i64]);
        let y = linear(&mut g, &mut param, y, "merger.mlp.0")?;
        let shape = g.shape(y).clone();
        let y = g.activation(Activation::Gelu, y, shape);
        let y = linear(&mut g, &mut param, y, "merger.mlp.2")?;
        g.set_outputs(vec![y]);
        Ok((g, params))
    }
}

/// The vision tower's settings from `config.json`'s `vision_config` and
/// `preprocessor_config.json`.
fn hf_vision_config(raw: &Value, pre: &Value, lm_hidden: usize) -> Result<MmProjConfig> {
    let v = raw
        .get("vision_config")
        .context("config.json has no vision_config")?;
    let n = |k: &str| -> Result<usize> {
        v.get(k)
            .and_then(Value::as_u64)
            .map(|x| x as usize)
            .with_context(|| format!("vision_config.{k}"))
    };
    let three = |k: &str, d: [f32; 3]| -> [f32; 3] {
        pre.get(k)
            .and_then(Value::as_array)
            .filter(|a| a.len() == 3)
            .map(|a| {
                let f = |i: usize| a[i].as_f64().unwrap_or(0.0) as f32;
                [f(0), f(1), f(2)]
            })
            .unwrap_or(d)
    };
    // Every `n_wa_pattern`-th block attends to the whole image.
    let full: Vec<usize> = v
        .get("fullatt_block_indexes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|x| x as usize)
                .collect()
        })
        .unwrap_or_default();
    let n_wa_pattern = full.first().map_or(0, |f| f + 1);
    if full.iter().any(|f| (f + 1) % n_wa_pattern.max(1) != 0) {
        bail!("vision_config.fullatt_block_indexes {full:?} is not every n-th block");
    }
    let merge = n("spatial_merge_size")?;
    let patch = n("patch_size")?;
    let token_pixels = merge * merge * patch * patch;
    Ok(MmProjConfig {
        patch_size: patch,
        n_embd: n("hidden_size")?,
        n_head: n("num_heads")?,
        n_layer: n("depth")?,
        image_size: 448,
        image_min_pixels: pre
            .get("min_pixels")
            .and_then(Value::as_u64)
            .map_or(4 * token_pixels, |x| x as usize),
        image_max_pixels: pre
            .get("max_pixels")
            .and_then(Value::as_u64)
            .map_or(1280 * token_pixels, |x| x as usize),
        n_merge: merge,
        eps: 1e-6,
        projector_type: "qwen2.5vl_merger".into(),
        image_mean: three("image_mean", [0.481_454_66, 0.457_827_5, 0.408_210_73]),
        image_std: three("image_std", [0.268_629_54, 0.261_302_6, 0.275_777_1]),
        spatial_merge_size: merge,
        llm_hidden_size: v
            .get("out_hidden_size")
            .and_then(Value::as_u64)
            .map_or(lm_hidden, |x| x as usize),
        n_ff: n("intermediate_size")?,
        n_wa_pattern,
        use_silu: true,
        use_rms_norm: true,
    })
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
            media_sizes: vec![320, 512, 1024, 2048],
        }
    }
}

/// Export the Qwen2.5-VL checkpoint in `dir` (`config.json`,
/// `preprocessor_config.json`, the safetensors, `tokenizer.json`,
/// `tokenizer_config.json`) as the package `out`.
pub fn export_rlxp(
    dir: &Path,
    out: &Path,
    opts: &ExportOptions,
    vl: &VlExportOptions,
) -> Result<ExportReport> {
    let config = Qwen25VlHfConfig::from_file(&dir.join("config.json"))?.into_runtime()?;
    if config.lm.lm.is_moe() {
        bail!("a mixture-of-experts language model is not exported");
    }
    let raw: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    if raw.get("model_type").and_then(Value::as_str) != Some("qwen2_5_vl") {
        bail!(
            "model_type {}: this exporter builds qwen2_5_vl",
            raw["model_type"]
        );
    }
    let pre: Value = std::fs::read(dir.join("preprocessor_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    let vision = hf_vision_config(&raw, &pre, config.lm.lm.hidden_size)?;
    let window = raw
        .pointer("/vision_config/window_size")
        .and_then(Value::as_u64)
        .map_or(DEFAULT_ATTN_WINDOW_SIZE, |w| w as usize)
        / vision.patch_size;
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
    // `tokens_per_second · ⌊temporal_patch / fps⌋` per frame pair, as
    // transformers' rope index has it, at the processor's 2 fps.
    let tokens_per_second = raw
        .pointer("/vision_config/tokens_per_second")
        .and_then(Value::as_f64)
        .unwrap_or(2.0);
    let model = VlDecoder {
        video_token: raw
            .get("video_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(IMAGE_TOKEN + 1),
        video_pixels: (
            video_edge("shortest_edge", vision.image_min_pixels),
            video_edge("longest_edge", vision.image_max_pixels),
        ),
        video_t_step: tokens_per_second as usize,
        image_token: raw
            .get("image_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(IMAGE_TOKEN),
        vl: config,
        vision,
        window,
        media_sizes: vl.media_sizes.clone(),
    };
    export_decoder(&model, dir, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("qwen25-vl", "image-text-to-text", |dir, out, opts| {
            export_rlxp(dir, out, opts, &VlExportOptions::default())
        })
        .claims(&["qwen2_5_vl"]),
    );
}
