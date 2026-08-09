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

//! `MiniMaxH3Transformer3DModel` — the joint video + audio flow-matching DiT.
//!
//! One stack of 50 blocks runs full self-attention over a **single packed
//! sequence** holding text, conditioning media, audio and target video rows.
//! There is no cross-attention. Modality-specific behaviour comes from exactly
//! three places: the two input patch projections, the per-row AdaLN modality
//! tag, and the two output heads.
//!
//! ## Scatter without a scatter op
//!
//! The reference builds the packed buffer with three `index_copy` calls. The
//! three index sets partition the sequence, so the same result is a single
//! **gather** from the concatenation `[text | video | audio]` under a
//! permutation the host computes once per layout — see
//! [`H3DitLayout::scatter_perm`]. That keeps the whole forward inside one
//! compiled graph.
//!
//! ## The AdaLN table
//!
//! Every block projects the shared timestep embedding to
//! `6 * hidden_size * 3` and reshapes it to one row per `(timestep, modality)`
//! pair, addressed by `timestep_index * 3 + tag`. Those projections hold ~13B of
//! the model's ~33B parameters, so they dominate both the weight footprint and
//! the per-step bandwidth. This port gathers the six modulation vectors in one
//! pass over the `(seq_len, 6 * hidden)` table rather than six, then narrows.
//!
//! ## Timestep bucketing
//!
//! One forward serves rows at up to four distinct noise levels (target video,
//! target audio, conditioning video, reference audio). Which of them coincide
//! changes from step to step, so the graph is compiled for a fixed
//! [`MAX_TIMESTEPS`] bucket and the host zero-pads the unused rows — they are
//! never gathered, so they cannot affect the result.

use crate::config::{H3TransformerConfig, MODALITY_NUM};
use crate::layout::{PackedLayout, RowTimesteps};
use anyhow::{Context, Result, anyhow, ensure};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, plugin_named};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

/// Distinct noise levels one forward can serve: target video, target audio,
/// conditioning video, reference audio.
pub const MAX_TIMESTEPS: usize = 4;

/// Base of the sinusoidal timestep embedding.
const TIME_MAX_PERIOD: f32 = 10_000.0;

/// The host-side description of one packed layout, in the form the compiled
/// graph consumes.
///
/// All index buffers are `f32`-encoded: rlx feeds integer graph inputs through
/// the same `&[f32]` channel as everything else and narrows them at the
/// boundary.
#[derive(Debug, Clone)]
pub struct H3DitLayout {
    pub seq_len: usize,
    pub num_text_rows: usize,
    pub num_video_rows: usize,
    pub num_audio_rows: usize,
    /// For each sequence position, its row in `concat([text, video, audio])`.
    pub scatter_perm: Vec<f32>,
    /// Sequence positions of the video rows, in output order.
    pub video_gather: Vec<f32>,
    /// Sequence positions of the audio rows, in output order.
    pub audio_gather: Vec<f32>,
    /// `timestep_index * 3 + modality_tag` per row.
    pub adaln_indices: Vec<f32>,
    /// Per-row index into the distinct timestep list.
    pub timestep_indices: Vec<f32>,
    /// The sinusoidal timestep embedding, `[MAX_TIMESTEPS * freq_dim]`,
    /// zero-padded beyond the distinct count.
    pub time_freq: Vec<f32>,
}

impl H3DitLayout {
    /// Build the graph-facing layout from a [`PackedLayout`] and the per-row
    /// timestep assignment of one step.
    pub fn new(
        layout: &PackedLayout,
        rows: &RowTimesteps,
        cfg: &H3TransformerConfig,
    ) -> Result<Self> {
        let seq_len = layout.sequence_length();
        ensure!(
            rows.indices.len() == seq_len,
            "row timestep indices ({}) do not match the sequence length ({seq_len})",
            rows.indices.len()
        );
        ensure!(
            rows.timesteps.len() <= MAX_TIMESTEPS,
            "{} distinct timesteps exceeds the compiled bucket of {MAX_TIMESTEPS}",
            rows.timesteps.len()
        );

        let n_text = layout.text_indices.len();
        let n_video = layout.video_indices.len();
        let n_audio = layout.audio_indices.len();

        // Invert the three index lists into one gather permutation over
        // `concat([text, video, audio])`.
        let mut scatter = vec![f32::NAN; seq_len];
        for (j, &p) in layout.text_indices.iter().enumerate() {
            scatter[p] = j as f32;
        }
        for (j, &p) in layout.video_indices.iter().enumerate() {
            scatter[p] = (n_text + j) as f32;
        }
        for (j, &p) in layout.audio_indices.iter().enumerate() {
            scatter[p] = (n_text + n_video + j) as f32;
        }
        ensure!(
            scatter.iter().all(|v| v.is_finite()),
            "the packed layout leaves some rows unassigned"
        );

        let adaln_indices = rows
            .indices
            .iter()
            .zip(&layout.token_tags)
            .map(|(&ti, &tag)| (ti as usize * MODALITY_NUM + tag as usize) as f32)
            .collect();

        Ok(Self {
            seq_len,
            num_text_rows: n_text,
            num_video_rows: n_video,
            num_audio_rows: n_audio,
            scatter_perm: scatter,
            video_gather: layout.video_indices.iter().map(|&i| i as f32).collect(),
            audio_gather: layout.audio_indices.iter().map(|&i| i as f32).collect(),
            adaln_indices,
            timestep_indices: rows.indices.iter().map(|&i| i as f32).collect(),
            time_freq: timestep_embedding(&rows.timesteps, cfg.freq_dim),
        })
    }

    /// The shape signature a compiled graph is keyed by.
    #[must_use]
    pub fn signature(&self) -> (usize, usize, usize, usize) {
        (
            self.seq_len,
            self.num_text_rows,
            self.num_video_rows,
            self.num_audio_rows,
        )
    }
}

/// Sinusoidal timestep embedding, zero-padded to [`MAX_TIMESTEPS`] rows.
///
/// Mirrors `diffusers.models.embeddings.get_timestep_embedding` with
/// `flip_sin_to_cos = True` and `downscale_freq_shift = 0`, so each row is
/// `[cos(...), sin(...)]`. H3 consumes timesteps unscaled on `[0, 1]`.
#[must_use]
pub fn timestep_embedding(timesteps: &[f32], freq_dim: usize) -> Vec<f32> {
    let half = freq_dim / 2;
    let mut out = vec![0.0f32; MAX_TIMESTEPS * freq_dim];
    for (r, &t) in timesteps.iter().enumerate().take(MAX_TIMESTEPS) {
        for i in 0..half {
            let exponent = -(TIME_MAX_PERIOD.ln()) * (i as f32) / (half as f32);
            let angle = t * exponent.exp();
            // flip_sin_to_cos: the cosine block leads.
            out[r * freq_dim + i] = angle.cos();
            out[r * freq_dim + half + i] = angle.sin();
        }
    }
    out
}

/// Everything one DiT evaluation needs beyond the compiled weights.
#[derive(Debug, Clone)]
pub struct H3DitInputs<'a> {
    /// `[num_video_rows * video_patch_dim]`, patchified and ordered to match
    /// `video_indices`.
    pub video_rows: &'a [f32],
    /// `[num_audio_rows * audio_in_channels]`, ordered to match `audio_indices`.
    pub audio_rows: &'a [f32],
    /// `[num_text_rows * text_dim]` conditioning from the text encoder.
    pub text_rows: &'a [f32],
    /// RoPE tables for the packed grid, `[seq_len * half]` each.
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    pub layout: &'a H3DitLayout,
}

/// The velocity prediction of one DiT evaluation.
#[derive(Debug, Clone)]
pub struct H3DitOutput {
    /// `[num_video_rows * video_patch_dim]`, in `video_indices` order.
    pub video: Vec<f32>,
    /// `[num_audio_rows * audio_in_channels]`, in `audio_indices` order.
    pub audio: Vec<f32>,
}

/// A DiT compiled for one fixed packed-sequence shape.
pub struct CompiledH3Dit {
    compiled: CompiledGraph,
    cfg: H3TransformerConfig,
    seq_len: usize,
    n_text: usize,
    n_video: usize,
    n_audio: usize,
    device: Device,
}

impl CompiledH3Dit {
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub fn config(&self) -> &H3TransformerConfig {
        &self.cfg
    }

    /// The `(seq_len, n_text, n_video, n_audio)` this graph was compiled for.
    #[must_use]
    pub fn signature(&self) -> (usize, usize, usize, usize) {
        (self.seq_len, self.n_text, self.n_video, self.n_audio)
    }

    /// Run one evaluation.
    pub fn forward(&mut self, inputs: &H3DitInputs<'_>) -> Result<H3DitOutput> {
        let c = &self.cfg;
        let l = inputs.layout;
        ensure!(
            l.signature() == self.signature(),
            "layout {:?} does not match the compiled shape {:?}",
            l.signature(),
            self.signature()
        );
        let vpd = c.video_patch_dim();
        let half = 3 * c.rope_freq_dim;
        ensure!(
            inputs.video_rows.len() == self.n_video * vpd,
            "video_rows len {} != {} rows × {vpd}",
            inputs.video_rows.len(),
            self.n_video
        );
        ensure!(
            inputs.audio_rows.len() == self.n_audio * c.audio_in_channels,
            "audio_rows len {} != {} rows × {}",
            inputs.audio_rows.len(),
            self.n_audio,
            c.audio_in_channels
        );
        ensure!(
            inputs.text_rows.len() == self.n_text * c.text_dim,
            "text_rows len {} != {} rows × {}",
            inputs.text_rows.len(),
            self.n_text,
            c.text_dim
        );
        ensure!(
            inputs.cos.len() == self.seq_len * half && inputs.sin.len() == self.seq_len * half,
            "rope tables must be [seq_len {} × half {half}]",
            self.seq_len
        );

        let outs = self.compiled.run(&[
            ("video_rows", inputs.video_rows),
            ("audio_rows", inputs.audio_rows),
            ("text_rows", inputs.text_rows),
            ("time_freq", &l.time_freq),
            ("cos", inputs.cos),
            ("sin", inputs.sin),
            ("scatter_perm", &l.scatter_perm),
            ("adaln_idx", &l.adaln_indices),
            ("timestep_idx", &l.timestep_indices),
            ("video_idx", &l.video_gather),
            ("audio_idx", &l.audio_gather),
        ]);
        let flat = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("compiled DiT returned no output"))?;
        split_velocity(flat, self.n_video * vpd, self.n_audio * c.audio_in_channels)
    }
}

/// Split the packed `[video rows | audio rows]` output back into the two heads.
pub fn split_velocity(
    mut flat: Vec<f32>,
    video_len: usize,
    audio_len: usize,
) -> Result<H3DitOutput> {
    ensure!(
        flat.len() >= video_len + audio_len,
        "velocity buffer holds {} values, expected at least {} video + {} audio",
        flat.len(),
        video_len,
        audio_len
    );
    let audio = flat[video_len..video_len + audio_len].to_vec();
    flat.truncate(video_len);
    Ok(H3DitOutput { video: flat, audio })
}

/// Build and compile a DiT for one fixed packed-sequence shape.
pub fn compile_dit(
    cfg: &H3TransformerConfig,
    weights: &mut WeightMap,
    device: Device,
    seq_len: usize,
    n_text: usize,
    n_video: usize,
    n_audio: usize,
) -> Result<CompiledH3Dit> {
    cfg.validate()?;
    ensure!(
        seq_len > 0 && n_text > 0,
        "the packed sequence and its text stream must be non-empty"
    );
    ensure!(
        n_text + n_video + n_audio == seq_len,
        "text ({n_text}) + video ({n_video}) + audio ({n_audio}) rows must cover the sequence ({seq_len})"
    );

    let built = build_dit_flow(cfg, weights, seq_len, n_text, n_video, n_audio)
        .context("MiniMax-H3: build DiT flow")?;
    let typed = built.typed_params.clone();
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    let opts =
        rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);

    Ok(CompiledH3Dit {
        compiled,
        cfg: cfg.clone(),
        seq_len,
        n_text,
        n_video,
        n_audio,
        device,
    })
}

fn build_dit_flow(
    cfg: &H3TransformerConfig,
    weights: &mut WeightMap,
    seq_len: usize,
    n_text: usize,
    n_video: usize,
    n_audio: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let i32d = DType::I32;
    let hidden = cfg.hidden_size;
    let inner = cfg.inner_dim();
    let nh = cfg.num_attention_heads;
    let hd = cfg.attention_head_dim;
    let n_rot = cfg.rope_rotary_dim();
    let half = n_rot / 2;
    let ffn = cfg.ffn_dim;
    let vpd = cfg.video_patch_dim();
    let aic = cfg.audio_in_channels;
    let ted = cfg.time_embed_dim;
    let num_layers = cfg.num_layers;
    let num_refiner = cfg.num_refiner_layers;
    let norm_eps = cfg.norm_eps;
    let qk_eps = cfg.qk_norm_eps;
    let final_eps = cfg.final_norm_eps;
    let adaln_rows = MAX_TIMESTEPS * MODALITY_NUM;

    let flow = ModelFlow::new("minimax_h3_dit")
        .with_profile(CompileProfile::encoder())
        .input("video_rows", Shape::new(&[1, n_video.max(1), vpd], f))
        .input("audio_rows", Shape::new(&[1, n_audio.max(1), aic], f))
        .input("text_rows", Shape::new(&[1, n_text, cfg.text_dim], f))
        .input(
            "time_freq",
            Shape::new(&[1, MAX_TIMESTEPS, cfg.freq_dim], f),
        )
        .input("cos", Shape::new(&[seq_len, half], f))
        .input("sin", Shape::new(&[seq_len, half], f))
        .input("scatter_perm", Shape::new(&[seq_len], i32d))
        .input("adaln_idx", Shape::new(&[seq_len], i32d))
        .input("timestep_idx", Shape::new(&[seq_len], i32d))
        .input("video_idx", Shape::new(&[n_video.max(1)], i32d))
        .input("audio_idx", Shape::new(&[n_audio.max(1)], i32d))
        // 1. Bind the rotary tables and the shared zero betas.
        .stage(plugin_named("bind", move |emit, hidden_in| {
            let cos = emit.flow_input("cos")?;
            let sin = emit.flow_input("sin")?;
            emit.set_named("h3_cos", cos.hir_id());
            emit.set_named("h3_sin", sin.hir_id());
            let z_hidden = emit.synth_zeros("h3_zeros_hidden", hidden);
            let z_head = emit.synth_zeros("h3_zeros_head", hd);
            emit.set_named("h3_zeros_hidden", z_hidden);
            emit.set_named("h3_zeros_head", z_head);
            Ok(hidden_in)
        }))
        // 2. Timestep embedding, shared by every AdaLN projection.
        .stage(plugin_named("time_embed", move |emit, hidden_in| {
            let freq = emit.flow_input("time_freq")?;
            let w1 = emit.load_param("time_embedder.linear_1.weight", true)?;
            let b1 = emit.load_param("time_embedder.linear_1.bias", false)?;
            let w2 = emit.load_param("time_embedder.linear_2.weight", true)?;
            let b2 = emit.load_param("time_embedder.linear_2.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let h = gb.mm(freq.hir_id(), w1);
            let h = gb.add(h, b1);
            let h = gb.silu(h);
            let h = gb.mm(h, w2);
            let temb = gb.add(h, b2); // [1, MAX_TIMESTEPS, time_embed_dim]
            emit.set_named("h3_temb", temb);
            Ok(hidden_in)
        }))
        // 3. Project each modality, refine the text stream, and pack.
        .stage(plugin_named("project_and_pack", move |emit, _hidden_in| {
            let video_in = emit.flow_input("video_rows")?;
            let audio_in = emit.flow_input("audio_rows")?;
            let text_in = emit.flow_input("text_rows")?;

            let pw = emit.load_param("proj_in.weight", true)?;
            let pb = emit.load_param("proj_in.bias", false)?;
            let aw = emit.load_param("audio_proj_in.weight", true)?;
            let ab = emit.load_param("audio_proj_in.bias", false)?;
            let cw = emit.load_param("context_embedder.weight", true)?;
            let cb = emit.load_param("context_embedder.bias", false)?;

            let mut gb = HirMut::new(emit.hir());
            let v = gb.mm(video_in.hir_id(), pw);
            let v = gb.add(v, pb);
            let a = gb.mm(audio_in.hir_id(), aw);
            let a = gb.add(a, ab);
            let t = gb.mm(text_in.hir_id(), cw);
            let t = gb.add(t, cb);
            emit.set_named("h3_video_embeds", v);
            emit.set_named("h3_audio_embeds", a);
            Ok(Some(emit.wrap(t, Shape::new(&[1, n_text, hidden], f))))
        }))
        // 4. Token refiner over the text stream.
        .repeat_layers(num_refiner, move |blk| {
            refiner_block(blk, n_text, hidden, inner, nh, hd, ffn, norm_eps, qk_eps)
        })
        .stage(plugin_named(
            "refiner_out_and_pack",
            move |emit, hidden_in| {
                let x = hidden_in.ok_or_else(|| anyhow!("token refiner produced no output"))?;
                let g = emit.load_param("token_refiner.final_norm.weight", false)?;
                let zeros = emit.named("h3_zeros_hidden")?;
                let v = emit.named("h3_video_embeds")?;
                let a = emit.named("h3_audio_embeds")?;
                let perm = emit.flow_input("scatter_perm")?;
                let mut gb = HirMut::new(emit.hir());
                let t = gb.rms_norm(x.hir_id(), g, zeros, final_eps);
                // `[text | video | audio]` in the order `scatter_perm` addresses.
                let cat = gb.concat_(vec![t, v, a], 1);
                let packed = gb.gather_(cat, perm.hir_id(), 1);
                Ok(Some(
                    emit.wrap(packed, Shape::new(&[1, seq_len, hidden], f)),
                ))
            },
        ))
        // 5. The block stack.
        .repeat_layers(num_layers, move |blk| {
            transformer_block(
                blk, seq_len, hidden, inner, nh, hd, n_rot, ffn, ted, adaln_rows, norm_eps, qk_eps,
            )
        })
        // 6. Shared output norm and the two per-modality heads.
        .stage(plugin_named("out", move |emit, hidden_in| {
            let x = hidden_in.ok_or_else(|| anyhow!("block stack produced no output"))?;
            let ng = emit.load_param("norm_out.norm.weight", false)?;
            let lw = emit.load_param("norm_out.linear.weight", true)?;
            let lb = emit.load_param("norm_out.linear.bias", false)?;
            let vw = emit.load_param("proj_out.weight", true)?;
            let vb = emit.load_param("proj_out.bias", false)?;
            let aw = emit.load_param("audio_proj_out.weight", true)?;
            let ab = emit.load_param("audio_proj_out.bias", false)?;
            let zeros = emit.named("h3_zeros_hidden")?;
            let temb = emit.named("h3_temb")?;
            let tidx = emit.flow_input("timestep_idx")?;
            let vidx = emit.flow_input("video_idx")?;
            let aidx = emit.flow_input("audio_idx")?;

            let mut gb = HirMut::new(emit.hir());
            // shift, scale — in that order, as in LTX2 / Wan output layers.
            let s = gb.silu(temb);
            let m = gb.mm(s, lw);
            let m = gb.add(m, lb);
            let table = gb.reshape_(m, vec![MAX_TIMESTEPS as i64, (2 * hidden) as i64]);
            let sel = gb.gather_(table, tidx.hir_id(), 0); // [seq_len, 2*hidden]
            let shift = gb.narrow_(sel, 1, 0, hidden);
            let scale = gb.narrow_(sel, 1, hidden, hidden);
            let shift = gb.reshape_(shift, vec![1, seq_len as i64, hidden as i64]);
            let scale = gb.reshape_(scale, vec![1, seq_len as i64, hidden as i64]);

            let h = gb.rms_norm(x.hir_id(), ng, zeros, final_eps);
            let h = modulate(&mut gb, h, scale, shift);

            let video = gb.mm(h, vw);
            let video = gb.add(video, vb);
            let video = gb.gather_(video, vidx.hir_id(), 1);

            let audio = gb.mm(h, aw);
            let audio = gb.add(audio, ab);
            let audio = gb.gather_(audio, aidx.hir_id(), 1);

            // The two heads leave as one buffer: `[video rows | audio rows]`,
            // each flattened. A flow stage returns a single value, and packing
            // them here is cheaper than compiling and running two graphs over
            // the same 50-block stack. `split_velocity` undoes it on the host.
            let nv = n_video.max(1);
            let na = n_audio.max(1);
            let video_flat = gb.reshape_(video, vec![1, (nv * vpd) as i64]);
            let audio_flat = gb.reshape_(audio, vec![1, (na * aic) as i64]);
            let both = gb.concat_(vec![video_flat, audio_flat], 1);
            Ok(Some(
                emit.wrap(both, Shape::new(&[1, nv * vpd + na * aic], f)),
            ))
        }))
        .output("velocity");

    flow.build_with(&mut WeightMapSource(weights), None)
}

/// `x * (1 + scale) + shift`, written as `x*scale + x + shift` so no constant
/// one has to be materialized.
fn modulate(gb: &mut HirMut<'_>, x: HirNodeId, scale: HirNodeId, shift: HirNodeId) -> HirNodeId {
    let scaled = gb.mul(x, scale);
    let summed = gb.add(scaled, x);
    gb.add(summed, shift)
}

/// Per-head RMS norm over the query / key projection, then partial RoPE.
#[allow(clippy::too_many_arguments)]
fn norm_and_rope(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    gamma: HirNodeId,
    zeros: HirNodeId,
    rope: Option<(HirNodeId, HirNodeId)>,
    n: usize,
    nh: usize,
    hd: usize,
    inner: usize,
    n_rot: usize,
    eps: f32,
) -> HirNodeId {
    let flat = gb.reshape_(x, vec![1, (n * nh) as i64, hd as i64]);
    let normed = gb.rms_norm(flat, gamma, zeros, eps);
    let back = gb.reshape_(normed, vec![1, n as i64, inner as i64]);
    match rope {
        Some((cos, sin)) => crate::rope::emit_partial_rope(gb, back, cos, sin, n, nh, hd, n_rot),
        None => back,
    }
}

/// SwiGLU feed-forward: `proj` emits `2 * ffn_dim`, the first half is the value
/// and the second is the gate — `out = value * silu(gate)`.
fn swiglu_ff(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    proj_w: HirNodeId,
    out_w: HirNodeId,
    ffn: usize,
) -> HirNodeId {
    let up = gb.mm(x, proj_w);
    let value = gb.narrow_(up, 2, 0, ffn);
    let gate = gb.narrow_(up, 2, ffn, ffn);
    let act = gb.silu(gate);
    let prod = gb.mul(value, act);
    gb.mm(prod, out_w)
}

/// A plain pre-norm block refining the projected text stream: no AdaLN, no RoPE.
#[allow(clippy::too_many_arguments)]
fn refiner_block(
    blk: usize,
    n_text: usize,
    hidden: usize,
    inner: usize,
    nh: usize,
    hd: usize,
    ffn: usize,
    norm_eps: f32,
    qk_eps: f32,
) -> FlowStage {
    let p = format!("token_refiner.refiner_blocks.{blk}");
    plugin_named(format!("refiner{blk}"), move |emit, hidden_in| {
        let x = hidden_in.ok_or_else(|| anyhow!("refiner block {blk} needs a hidden state"))?;
        let f = DType::F32;
        let shape = Shape::new(&[1, n_text, hidden], f);

        let n1 = emit.load_param(&format!("{p}.norm1.weight"), false)?;
        let wq = emit.load_param(&format!("{p}.attn.to_q.weight"), true)?;
        let wk = emit.load_param(&format!("{p}.attn.to_k.weight"), true)?;
        let wv = emit.load_param(&format!("{p}.attn.to_v.weight"), true)?;
        let gq = emit.load_param(&format!("{p}.attn.norm_q.weight"), false)?;
        let gk = emit.load_param(&format!("{p}.attn.norm_k.weight"), false)?;
        let wo = emit.load_param(&format!("{p}.attn.to_out.0.weight"), true)?;
        let n2 = emit.load_param(&format!("{p}.norm2.weight"), false)?;
        let ff_in = emit.load_param(&format!("{p}.ff.net.0.proj.weight"), true)?;
        let ff_out = emit.load_param(&format!("{p}.ff.net.2.weight"), true)?;
        let z_hidden = emit.named("h3_zeros_hidden")?;
        let z_head = emit.named("h3_zeros_head")?;

        let mut gb = HirMut::new(emit.hir());
        let residual = x.hir_id();
        let h = gb.rms_norm(residual, n1, z_hidden, norm_eps);
        let q = gb.mm(h, wq);
        let k = gb.mm(h, wk);
        let v = gb.mm(h, wv);
        let q = norm_and_rope(
            &mut gb, q, gq, z_head, None, n_text, nh, hd, inner, 0, qk_eps,
        );
        let k = norm_and_rope(
            &mut gb, k, gk, z_head, None, n_text, nh, hd, inner, 0, qk_eps,
        );
        let attn = gb.attention_kind(
            q,
            k,
            v,
            nh,
            hd,
            MaskKind::None,
            Shape::new(&[1, n_text, inner], f),
        );
        let attn = gb.mm(attn, wo);
        let x1 = gb.add(residual, attn);

        let h = gb.rms_norm(x1, n2, z_hidden, norm_eps);
        let ff = swiglu_ff(&mut gb, h, ff_in, ff_out, ffn);
        let out = gb.add(x1, ff);
        Ok(Some(emit.wrap(out, shape)))
    })
}

/// One MiniMax-H3 block: pre-norm attention and feed-forward, each modulated by
/// AdaLN parameters selected per row from the `(timestep, modality)` table.
#[allow(clippy::too_many_arguments)]
fn transformer_block(
    blk: usize,
    seq_len: usize,
    hidden: usize,
    inner: usize,
    nh: usize,
    hd: usize,
    n_rot: usize,
    ffn: usize,
    ted: usize,
    adaln_rows: usize,
    norm_eps: f32,
    qk_eps: f32,
) -> FlowStage {
    let p = format!("transformer_blocks.{blk}");
    plugin_named(format!("block{blk}"), move |emit, hidden_in| {
        let x = hidden_in.ok_or_else(|| anyhow!("block {blk} needs a hidden state"))?;
        let f = DType::F32;
        let shape = Shape::new(&[1, seq_len, hidden], f);

        let aw = emit.load_param(&format!("{p}.adaln_proj.linear.weight"), true)?;
        let ab = emit.load_param(&format!("{p}.adaln_proj.linear.bias"), false)?;
        let n1 = emit.load_param(&format!("{p}.norm1.weight"), false)?;
        let wq = emit.load_param(&format!("{p}.attn.to_q.weight"), true)?;
        let wk = emit.load_param(&format!("{p}.attn.to_k.weight"), true)?;
        let wv = emit.load_param(&format!("{p}.attn.to_v.weight"), true)?;
        let gq = emit.load_param(&format!("{p}.attn.norm_q.weight"), false)?;
        let gk = emit.load_param(&format!("{p}.attn.norm_k.weight"), false)?;
        let wo = emit.load_param(&format!("{p}.attn.to_out.0.weight"), true)?;
        let n2 = emit.load_param(&format!("{p}.norm2.weight"), false)?;
        let ff_in = emit.load_param(&format!("{p}.ff.net.0.proj.weight"), true)?;
        let ff_out = emit.load_param(&format!("{p}.ff.net.2.weight"), true)?;

        let temb = emit.named("h3_temb")?;
        let z_hidden = emit.named("h3_zeros_hidden")?;
        let z_head = emit.named("h3_zeros_head")?;
        let cos = emit.named("h3_cos")?;
        let sin = emit.named("h3_sin")?;
        let adaln_idx = emit.flow_input("adaln_idx")?;

        let mut gb = HirMut::new(emit.hir());

        // The block's modulation table: one row per (timestep, modality) pair,
        // gathered once for all six vectors and narrowed afterwards.
        let s = gb.silu(temb);
        let m = gb.mm(s, aw);
        let m = gb.add(m, ab); // [1, MAX_TIMESTEPS, 6*hidden*MODALITY_NUM]
        let table = gb.reshape_(m, vec![adaln_rows as i64, (6 * hidden) as i64]);
        let sel = gb.gather_(table, adaln_idx.hir_id(), 0); // [seq_len, 6*hidden]
        let take = |gb: &mut HirMut<'_>, k: usize| {
            let n = gb.narrow_(sel, 1, k * hidden, hidden);
            gb.reshape_(n, vec![1, seq_len as i64, hidden as i64])
        };
        let shift_msa = take(&mut gb, 0);
        let scale_msa = take(&mut gb, 1);
        let gate_msa = take(&mut gb, 2);
        let shift_mlp = take(&mut gb, 3);
        let scale_mlp = take(&mut gb, 4);
        let gate_mlp = take(&mut gb, 5);

        // Self-attention.
        let residual = x.hir_id();
        let h = gb.rms_norm(residual, n1, z_hidden, norm_eps);
        let h = modulate(&mut gb, h, scale_msa, shift_msa);
        let q = gb.mm(h, wq);
        let k = gb.mm(h, wk);
        let v = gb.mm(h, wv);
        let q = norm_and_rope(
            &mut gb,
            q,
            gq,
            z_head,
            Some((cos, sin)),
            seq_len,
            nh,
            hd,
            inner,
            n_rot,
            qk_eps,
        );
        let k = norm_and_rope(
            &mut gb,
            k,
            gk,
            z_head,
            Some((cos, sin)),
            seq_len,
            nh,
            hd,
            inner,
            n_rot,
            qk_eps,
        );
        let attn = gb.attention_kind(
            q,
            k,
            v,
            nh,
            hd,
            MaskKind::None,
            Shape::new(&[1, seq_len, inner], f),
        );
        let attn = gb.mm(attn, wo);
        let x1 = gb.gated_residual(residual, attn, gate_msa);

        // Feed-forward.
        let h = gb.rms_norm(x1, n2, z_hidden, norm_eps);
        let h = modulate(&mut gb, h, scale_mlp, shift_mlp);
        let ff = swiglu_ff(&mut gb, h, ff_in, ff_out, ffn);
        let out = gb.gated_residual(x1, ff, gate_mlp);

        let _ = ted;
        Ok(Some(emit.wrap(out, shape)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{H3Geometry, KeyframeAnchor, build_packed_sequence, build_row_timesteps};

    fn tiny_cfg() -> H3TransformerConfig {
        H3TransformerConfig {
            num_attention_heads: 2,
            attention_head_dim: 16,
            hidden_size: 24,
            num_layers: 2,
            num_refiner_layers: 1,
            ffn_dim: 32,
            in_channels: 4,
            audio_in_channels: 6,
            patch_size: [1, 2, 2],
            text_dim: 8,
            freq_dim: 16,
            time_embed_hidden_dim: 24,
            time_embed_dim: 12,
            rope_freq_dim: 2,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    #[test]
    fn timestep_embedding_is_padded_and_bounded() {
        let e = timestep_embedding(&[0.0, 0.5], 16);
        assert_eq!(e.len(), MAX_TIMESTEPS * 16);
        // t = 0 gives cos = 1, sin = 0 across the block.
        for i in 0..8 {
            assert!((e[i] - 1.0).abs() < 1e-6);
            assert!(e[8 + i].abs() < 1e-6);
        }
        // Unused rows stay zero.
        assert!(e[2 * 16..].iter().all(|&v| v == 0.0));
        assert!(e.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn timestep_embedding_leads_with_cosine() {
        // flip_sin_to_cos puts the cosine block first; at t = 1 the leading
        // entry is cos(1) and the second-half entry is sin(1).
        let e = timestep_embedding(&[1.0], 8);
        assert!((e[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((e[4] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn dit_layout_inverts_the_index_lists() {
        let g = H3Geometry::resolve(768, 1344, 124, 16, 2).unwrap();
        let layout = build_packed_sequence(&[1u32; 6], &g, [1, 2, 2], &[]).unwrap();
        let rows = build_row_timesteps(&layout, 0.2, 0.3, 0.999, 1.0).unwrap();
        let cfg = H3TransformerConfig::default();
        let l = H3DitLayout::new(&layout, &rows, &cfg).unwrap();

        assert_eq!(l.seq_len, layout.sequence_length());
        assert_eq!(l.scatter_perm.len(), l.seq_len);
        // The permutation must be a bijection onto [0, seq_len).
        let mut seen = vec![false; l.seq_len];
        for &p in &l.scatter_perm {
            let i = p as usize;
            assert!(!seen[i], "scatter_perm repeats {i}");
            seen[i] = true;
        }
        assert!(seen.into_iter().all(|b| b));
        // Text rows map to the head of the concatenation.
        for (j, &pos) in layout.text_indices.iter().enumerate() {
            assert_eq!(l.scatter_perm[pos], j as f32);
        }
    }

    #[test]
    fn adaln_indices_stay_inside_the_table() {
        let g = H3Geometry::resolve(768, 1344, 124, 16, 2).unwrap();
        let layout =
            build_packed_sequence(&[1u32; 4], &g, [1, 2, 2], &[KeyframeAnchor::First]).unwrap();
        let rows = build_row_timesteps(&layout, 0.2, 0.3, 0.999, 1.0).unwrap();
        let cfg = H3TransformerConfig::default();
        let l = H3DitLayout::new(&layout, &rows, &cfg).unwrap();
        let limit = (MAX_TIMESTEPS * MODALITY_NUM) as f32;
        assert!(l.adaln_indices.iter().all(|&i| i >= 0.0 && i < limit));
        assert!(
            l.timestep_indices
                .iter()
                .all(|&i| i >= 0.0 && (i as usize) < MAX_TIMESTEPS)
        );
    }

    #[test]
    fn dit_layout_rejects_too_many_distinct_timesteps() {
        let g = H3Geometry::resolve(768, 1344, 124, 16, 2).unwrap();
        let layout = build_packed_sequence(&[1u32; 4], &g, [1, 2, 2], &[]).unwrap();
        let rows = RowTimesteps {
            timesteps: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            indices: vec![0; layout.sequence_length()],
        };
        let cfg = H3TransformerConfig::default();
        assert!(H3DitLayout::new(&layout, &rows, &cfg).is_err());
    }

    #[test]
    fn tiny_config_is_self_consistent() {
        let c = tiny_cfg();
        c.validate().unwrap();
        assert_eq!(c.inner_dim(), 32);
        assert_eq!(c.video_patch_dim(), 16);
        assert_eq!(c.rope_rotary_dim(), 12);
        assert!(c.rope_rotary_dim() <= c.attention_head_dim);
    }
}
