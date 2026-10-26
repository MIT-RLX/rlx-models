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

//! The Mimi decoder as a streaming block graph.
//!
//! `latent [1, 32, F, 1]` (de-normalized) → quantizer projection → depthwise
//! transposed-conv upsample (×16) → projected transformer (sliding window
//! `context`) → SEANet decoder → `F · 1920` samples.
//!
//! Every op in that chain is causal, which is what makes upstream's
//! frame-by-frame streaming decoder possible; this graph is the same streaming
//! decoder with the per-module state made explicit:
//!
//! - a causal conv keeps its last `(k-1)·dilation + 1 - stride` input samples
//!   and prepends them to the next block (zeros at the start, i.e. the
//!   left padding);
//! - a transposed conv keeps the `k - stride` samples that overlap the next
//!   block and adds them to the next block's head;
//! - the transformer keeps the keys/values of the last `context - 1` positions
//!   and takes a host-built additive mask for the sliding window.
//!
//! The graph is built once for `F` frames; any utterance is decoded block by
//! block, the last block zero-padded (padding only reaches samples after the
//! real ones, which are cut).

use anyhow::{Result, bail};
use rlx_ir::NodeId;
use rlx_ir::infer::GraphExt;
use rlx_runtime::{CompiledGraph, Device};

use super::{Builder, Tensor, compile, get, rope_tables};
use crate::checkpoint::Checkpoint;
use crate::spec::{MimiTransformerSpec, ModelSpec};

const LN_EPS: f32 = 1e-5;

/// One state tensor carried between blocks.
#[derive(Debug, Clone)]
struct StateSlot {
    name: String,
    len: usize,
}

struct Ctx<'a> {
    b: Builder,
    ck: &'a Checkpoint,
    states: Vec<StateSlot>,
    state_out: Vec<NodeId>,
    /// `[1, 1, 1, 1]` input, 1 on a stream's first block: replicate-padded
    /// convs then take their left context from the first sample.
    first: Option<NodeId>,
}

impl Ctx<'_> {
    fn state(&mut self, shape: &[usize]) -> NodeId {
        let name = format!("state{}", self.states.len());
        let id = self.b.input(&name, shape);
        self.states.push(StateSlot {
            name,
            len: shape.iter().product(),
        });
        id
    }

    fn set_state_out(&mut self, x: NodeId) {
        let own = self.b.own(x);
        self.state_out.push(own);
    }

    /// Causal conv with carried left context. `x` is `[1, C, T, 1]`.
    fn conv(&mut self, x: NodeId, prefix: &str, stride: usize, dilation: usize) -> Result<NodeId> {
        self.conv_padded(x, prefix, stride, dilation, false)
    }

    /// [`Self::conv`]; `replicate` pads a stream's start with its first sample
    /// (upstream `pad_mode="replicate"`) rather than zeros.
    fn conv_padded(
        &mut self,
        x: NodeId,
        prefix: &str,
        stride: usize,
        dilation: usize,
        replicate: bool,
    ) -> Result<NodeId> {
        let w = get(self.ck, &format!("{prefix}.weight"))?;
        let (c_out, c_in, k) = w.dims3()?;
        let bias = self.ck.opt(&format!("{prefix}.bias"))?;
        let dims = self.b.dims(x);
        if dims[1] != c_in {
            bail!(
                "{prefix}: input has {} channels, weight wants {c_in}",
                dims[1]
            );
        }
        let ctx = (k - 1) * dilation + 1 - stride;
        let x = if ctx > 0 {
            let mut prev = self.state(&[1, c_in, ctx, 1]);
            if replicate {
                let first = self.first.ok_or_else(|| {
                    anyhow::anyhow!("{prefix}: replicate pad needs a `first` input")
                })?;
                let one = self.b.scalar(1.0);
                let keep = self.b.g.sub(one, first);
                let head = self.b.g.narrow_(x, 2, 0, 1);
                let head = self.b.expand(head, &[1, c_in, ctx, 1]);
                let a = self.b.g.mul(prev, keep);
                let h = self.b.g.mul(head, first);
                prev = self.b.g.add(a, h);
            }
            let cat = self.b.g.concat_(vec![prev, x], 2);
            let len = self.b.dims(cat)[2];
            let keep = self.b.g.narrow_(cat, 2, len - ctx, ctx);
            self.set_state_out(keep);
            cat
        } else {
            x
        };
        let y = self
            .b
            .conv1d(x, &w.data, [c_out, c_in, k], stride, dilation, 1);
        Ok(match bias {
            Some(bias) => self.b.channel_bias(y, &bias.data),
            None => y,
        })
    }

    /// Transposed conv with the carried overlap tail.
    fn conv_tr(&mut self, x: NodeId, prefix: &str, stride: usize, groups: usize) -> Result<NodeId> {
        let w = get(self.ck, &format!("{prefix}.weight"))?;
        let (_, c_out_g, k) = w.dims3()?;
        let c_out = c_out_g * groups;
        let bias = self.ck.opt(&format!("{prefix}.bias"))?;
        let t = self.b.dims(x)[2];
        let y = self.b.conv_transpose1d_full(x, &w, stride, groups)?;
        let tail = k - stride;
        let out_len = t * stride;
        let y = if tail > 0 {
            let partial = self.state(&[1, c_out, tail, 1]);
            let z = self.b.zeros_ct(c_out, out_len);
            let carry = self.b.g.concat_(vec![partial, z], 2);
            let y = self.b.g.add(y, carry);
            let next = self.b.g.narrow_(y, 2, out_len, tail);
            self.set_state_out(next);
            self.b.g.narrow_(y, 2, 0, out_len)
        } else {
            y
        };
        Ok(match bias {
            Some(bias) => self.b.channel_bias(y, &bias.data),
            None => y,
        })
    }

    /// `[T, C] @ Wᵀ` for a torch Linear weight.
    fn linear(&mut self, x: NodeId, name: &str) -> Result<NodeId> {
        let w = get(self.ck, name)?;
        self.b.linear(x, &w, None)
    }

    /// `ProjectedTransformer` on `[1, C, T, 1]`, returning the same layout.
    fn projected_transformer(
        &mut self,
        x: NodeId,
        tp: &str,
        tf: &MimiTransformerSpec,
        t: usize,
        cos: NodeId,
        sin: NodeId,
        mask: NodeId,
    ) -> Result<NodeId> {
        let c = self.b.dims(x)[1];
        let tc = self.b.g.reshape_(x, vec![c as i64, t as i64]);
        let mut h = self.b.g.transpose_(tc, vec![1, 0]);
        if self.ck.has(&format!("{tp}.input_proj.weight")) {
            h = self.linear(h, &format!("{tp}.input_proj.weight"))?;
        }
        for l in 0..tf.num_layers {
            let prefix = format!("{tp}.transformer.layers.{l}");
            h = self.transformer_layer(h, &prefix, tf, t, cos, sin, mask)?;
        }
        if self.ck.has(&format!("{tp}.output_projs.0.weight")) {
            h = self.linear(h, &format!("{tp}.output_projs.0.weight"))?;
        }
        let c = self.b.dims(h)[1];
        let ct = self.b.g.transpose_(h, vec![1, 0]);
        Ok(self.b.g.reshape_(ct, vec![1, c as i64, t as i64, 1]))
    }

    /// SEANet residual block (`ELU, conv k, ELU, conv 1`, plus the skip).
    fn resnet_block(&mut self, x: NodeId, rb: &str, dilation: usize) -> Result<NodeId> {
        let v = self.b.elu(x);
        let v = self.conv(v, &format!("{rb}.1.conv"), 1, dilation)?;
        let v = self.b.elu(v);
        let v = self.conv(v, &format!("{rb}.3.conv"), 1, 1)?;
        Ok(self.b.g.add(x, v))
    }

    fn transformer_layer(
        &mut self,
        x: NodeId,
        prefix: &str,
        tf: &MimiTransformerSpec,
        t: usize,
        cos: NodeId,
        sin: NodeId,
        mask: NodeId,
    ) -> Result<NodeId> {
        let d = tf.d_model;
        let h = tf.num_heads;
        let hd = d / h;
        let p = tf.context - 1;

        let n1w = get(self.ck, &format!("{prefix}.norm1.weight"))?;
        let n1b = get(self.ck, &format!("{prefix}.norm1.bias"))?;
        let n = self.b.layer_norm(x, Some((&n1w, &n1b)), LN_EPS);
        let in_proj = get(self.ck, &format!("{prefix}.self_attn.in_proj.weight"))?;
        if in_proj.shape != [3 * d, d] {
            bail!("{prefix}.self_attn.in_proj.weight is {:?}", in_proj.shape);
        }
        let part = |i: usize| Tensor {
            data: in_proj.data[i * d * d..(i + 1) * d * d].to_vec(),
            shape: vec![d, d],
        };
        let q = self.b.linear(n, &part(0), None)?;
        let k = self.b.linear(n, &part(1), None)?;
        let v = self.b.linear(n, &part(2), None)?;
        let heads = vec![t as i64, h as i64, hd as i64];
        let q = self.b.g.reshape_(q, heads.clone());
        let k = self.b.g.reshape_(k, heads);
        let q = self.b.rope_interleaved(q, cos, sin);
        let k = self.b.rope_interleaved(k, cos, sin);
        let k = self.b.g.reshape_(k, vec![t as i64, d as i64]);

        let past_k = self.state(&[p, d]);
        let past_v = self.state(&[p, d]);
        let k_full = self.b.g.concat_(vec![past_k, k], 0);
        let v_full = self.b.g.concat_(vec![past_v, v], 0);
        let keep_k = self.b.g.narrow_(k_full, 0, t, p);
        let keep_v = self.b.g.narrow_(v_full, 0, t, p);
        self.set_state_out(keep_k);
        self.set_state_out(keep_v);

        let s = p + t;
        let qh = self.b.g.transpose_(q, vec![1, 0, 2]); // [H, T, hd]
        let kh = self
            .b
            .g
            .reshape_(k_full, vec![s as i64, h as i64, hd as i64]);
        let kh = self.b.g.transpose_(kh, vec![1, 2, 0]); // [H, hd, S]
        let vh = self
            .b
            .g
            .reshape_(v_full, vec![s as i64, h as i64, hd as i64]);
        let vh = self.b.g.transpose_(vh, vec![1, 0, 2]); // [H, S, hd]
        let scores = self.b.g.mm(qh, kh);
        let scale = self.b.scalar(1.0 / (hd as f32).sqrt());
        let scores = self.b.g.mul(scores, scale);
        let scores = self.b.g.add(scores, mask);
        let probs = self.b.g.sm(scores, -1);
        let o = self.b.g.mm(probs, vh); // [H, T, hd]
        let o = self.b.g.transpose_(o, vec![1, 0, 2]);
        let o = self.b.g.reshape_(o, vec![t as i64, d as i64]);
        let o = self.linear(o, &format!("{prefix}.self_attn.out_proj.weight"))?;
        let ls1 = get(self.ck, &format!("{prefix}.layer_scale_1.scale"))?;
        let ls1 = self.b.param(ls1.data, &[1, d]);
        let o = self.b.g.mul(o, ls1);
        let x = self.b.g.add(x, o);

        let n2w = get(self.ck, &format!("{prefix}.norm2.weight"))?;
        let n2b = get(self.ck, &format!("{prefix}.norm2.bias"))?;
        let n2 = self.b.layer_norm(x, Some((&n2w, &n2b)), LN_EPS);
        let m = self.linear(n2, &format!("{prefix}.linear1.weight"))?;
        let m = self.b.g.gelu_approx(m);
        let m = self.linear(m, &format!("{prefix}.linear2.weight"))?;
        let ls2 = get(self.ck, &format!("{prefix}.layer_scale_2.scale"))?;
        let ls2 = self.b.param(ls2.data, &[1, d]);
        let m = self.b.g.mul(m, ls2);
        Ok(self.b.g.add(x, m))
    }
}

/// A built block graph: the graph plus the state layout it expects.
pub(crate) struct Block {
    pub builder: Builder,
    states: Vec<StateSlot>,
}

fn new_ctx<'a>(ck: &'a Checkpoint, name: &str) -> Ctx<'a> {
    Ctx {
        b: Builder::new(name),
        ck,
        states: Vec::new(),
        state_out: Vec::new(),
        first: None,
    }
}

/// Transformer window inputs (`attn_mask`, `rope_cos`, `rope_sin`) for a
/// block of `t` positions.
fn window_inputs(cx: &mut Ctx, tf: &MimiTransformerSpec, t: usize) -> (NodeId, NodeId, NodeId) {
    let hd = tf.d_model / tf.num_heads;
    let mask = cx.b.input("attn_mask", &[1, t, tf.context - 1 + t]);
    let cos = cx.b.input("rope_cos", &[t, 1, hd / 2]);
    let sin = cx.b.input("rope_sin", &[t, 1, hd / 2]);
    (mask, cos, sin)
}

fn finish(mut cx: Ctx, out: NodeId) -> Block {
    let out = cx.b.own(out);
    let mut outs = vec![out];
    outs.extend(cx.state_out.iter().copied());
    cx.b.g.set_outputs(outs);
    Block {
        builder: cx.b,
        states: cx.states,
    }
}

/// Decoder block: `latent [1, 32, F, 1]` → audio `[1, 1, F·frame_size, 1]`.
pub(crate) fn build_decoder_block(
    ck: &Checkpoint,
    spec: &ModelSpec,
    frames: usize,
) -> Result<Block> {
    let m = &spec.mimi;
    let steps = m.steps_per_frame();
    let tf = m.decoder_transformer();
    let sea = m.decoder_seanet();
    let t = frames * steps;

    let mut cx = new_ctx(ck, &format!("pocket_mimi_decode_f{frames}"));
    let latent = cx.b.input("x", &[1, spec.latent_dim(), frames, 1]);
    let (mask, cos, sin) = window_inputs(&mut cx, tf, t);

    // Quantizer output projection (1×1, no bias).
    let qw = get(ck, "mimi.quantizer.output_proj.weight")?;
    let (qo, qi, _) = qw.dims3()?;
    let mut x = cx.b.conv1d(latent, &qw.data, [qo, qi, 1], 1, 1, 1);

    // Frame-rate upsample: depthwise transposed conv, groups = channels.
    x = cx.conv_tr(x, "mimi.upsample.convtr.convtr", steps, sea.dimension)?;
    x = cx.projected_transformer(x, "mimi.decoder_transformer", tf, t, cos, sin, mask)?;

    // SEANet decoder.
    let dp = "mimi.decoder.model";
    let mut idx = 0;
    x = cx.conv(x, &format!("{dp}.{idx}.conv"), 1, 1)?;
    idx += 1;
    for &ratio in &sea.ratios {
        x = cx.b.elu(x);
        idx += 1;
        x = cx.conv_tr(x, &format!("{dp}.{idx}.convtr"), ratio, 1)?;
        idx += 1;
        for j in 0..sea.n_residual_layers {
            let dil = sea.dilation_base.pow(j as u32);
            x = cx.resnet_block(x, &format!("{dp}.{idx}.block"), dil)?;
            idx += 1;
        }
    }
    x = cx.b.elu(x);
    idx += 1;
    x = cx.conv(x, &format!("{dp}.{idx}.conv"), 1, 1)?;
    let samples = frames * m.frame_size();
    if cx.b.dims(x) != [1, m.channels, samples, 1] {
        bail!("decoder block output is {:?}", cx.b.dims(x));
    }
    Ok(finish(cx, x))
}

/// Encoder block: audio `[1, 1, F·frame_size, 1]` → unquantized latent
/// `[1, 32, F, 1]` (SEANet encoder → transformer → replicate-padded downsample).
pub(crate) fn build_encoder_block(
    ck: &Checkpoint,
    spec: &ModelSpec,
    frames: usize,
) -> Result<Block> {
    let m = &spec.mimi;
    let steps = m.steps_per_frame();
    let tf = &m.transformer;
    let sea = &m.seanet;
    let t = frames * steps;

    let mut cx = new_ctx(ck, &format!("pocket_mimi_encode_f{frames}"));
    let audio =
        cx.b.input("x", &[1, m.channels, frames * m.frame_size(), 1]);
    let first = cx.b.input("first", &[1, 1, 1, 1]);
    cx.first = Some(first);
    let (mask, cos, sin) = window_inputs(&mut cx, tf, t);

    let ep = "mimi.encoder.model";
    let mut idx = 0;
    let mut x = cx.conv(audio, &format!("{ep}.{idx}.conv"), 1, 1)?;
    idx += 1;
    for &ratio in sea.ratios.iter().rev() {
        for j in 0..sea.n_residual_layers {
            let dil = sea.dilation_base.pow(j as u32);
            x = cx.resnet_block(x, &format!("{ep}.{idx}.block"), dil)?;
            idx += 1;
        }
        x = cx.b.elu(x);
        idx += 1;
        x = cx.conv(x, &format!("{ep}.{idx}.conv"), ratio, 1)?;
        idx += 1;
    }
    x = cx.b.elu(x);
    idx += 1;
    x = cx.conv(x, &format!("{ep}.{idx}.conv"), 1, 1)?;
    if cx.b.dims(x)[2] != t {
        bail!(
            "encoder SEANet output is {:?}, expected {t} steps",
            cx.b.dims(x)
        );
    }
    x = cx.projected_transformer(x, "mimi.encoder_transformer", tf, t, cos, sin, mask)?;
    x = cx.conv_padded(x, "mimi.downsample.conv.conv", steps, 1, true)?;
    if cx.b.dims(x) != [1, spec.latent_dim(), frames, 1] {
        bail!("encoder block output is {:?}", cx.b.dims(x));
    }
    Ok(finish(cx, x))
}

/// A compiled block graph and the state it carries between calls.
struct Stream {
    compiled: CompiledGraph,
    states: Vec<StateSlot>,
    state: Vec<Vec<f32>>,
    /// Transformer positions consumed so far.
    position: usize,
    steps: usize,
    context: usize,
    head_dim: usize,
    max_period: f32,
}

impl Stream {
    fn new(block: Block, tf: &MimiTransformerSpec, steps: usize, device: Device) -> Result<Self> {
        let compiled = compile(block.builder, device)?;
        let state = block.states.iter().map(|s| vec![0.0; s.len]).collect();
        Ok(Self {
            compiled,
            states: block.states,
            state,
            position: 0,
            steps,
            context: tf.context,
            head_dim: tf.d_model / tf.num_heads,
            max_period: tf.max_period,
        })
    }

    fn reset(&mut self) {
        for s in &mut self.state {
            s.fill(0.0);
        }
        self.position = 0;
    }

    /// Run one block of `frames` frames; `x` is the block input, `extra` any
    /// further named inputs. Returns the block output.
    fn run(&mut self, frames: usize, x: &[f32], extra: &[(&str, &[f32])]) -> Result<Vec<f32>> {
        let t = frames * self.steps;
        let p = self.context - 1;
        let p0 = self.position;
        let (cos, sin) = rope_tables(p0..p0 + t, self.head_dim, self.max_period);
        let mut mask = vec![f32::NEG_INFINITY; t * (p + t)];
        for qi in 0..t {
            let qpos = (p0 + qi) as i64;
            for s in 0..p + t {
                // Absolute position of key slot `s` (past slots may predate 0).
                let kpos = p0 as i64 - p as i64 + s as i64;
                if kpos >= 0 && kpos <= qpos && qpos - kpos < self.context as i64 {
                    mask[qi * (p + t) + s] = 0.0;
                }
            }
        }
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("x", x),
            ("attn_mask", &mask),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
        ];
        inputs.extend_from_slice(extra);
        for (slot, data) in self.states.iter().zip(&self.state) {
            inputs.push((slot.name.as_str(), data));
        }
        let mut outs = self.compiled.run(&inputs).into_iter();
        let out = outs
            .next()
            .ok_or_else(|| anyhow::anyhow!("block returned no output"))?;
        for (i, s) in self.state.iter_mut().enumerate() {
            let next = outs
                .next()
                .ok_or_else(|| anyhow::anyhow!("block returned too few states"))?;
            if next.len() != s.len() {
                bail!("state {i}: {} values, expected {}", next.len(), s.len());
            }
            *s = next;
        }
        self.position += t;
        Ok(out)
    }
}

/// Streaming Mimi decoder on one device.
pub struct MimiDecoder {
    stream: Stream,
    frames: usize,
    ldim: usize,
    frame_size: usize,
}

impl MimiDecoder {
    /// Compile the decoder for blocks of `frames` latent frames.
    pub fn new(ck: &Checkpoint, spec: &ModelSpec, frames: usize, device: Device) -> Result<Self> {
        if frames == 0 {
            bail!("decoder block needs at least one frame");
        }
        let block = build_decoder_block(ck, spec, frames)?;
        let m = &spec.mimi;
        Ok(Self {
            stream: Stream::new(block, m.decoder_transformer(), m.steps_per_frame(), device)?,
            frames,
            ldim: spec.latent_dim(),
            frame_size: m.frame_size(),
        })
    }

    /// Latent frames per graph call.
    pub fn block_frames(&self) -> usize {
        self.frames
    }

    /// Forget all history (a fresh decoder state).
    pub fn reset(&mut self) {
        self.stream.reset();
    }

    /// Decode one block. `latents` holds `n ≤ block_frames` de-normalized
    /// frames, row-major `[n, ldim]`; returns `n · frame_size` samples. A short
    /// block is zero-padded; its state is then only good for the end of a stream.
    pub fn decode_block(&mut self, latents: &[f32]) -> Result<Vec<f32>> {
        let n = latents.len() / self.ldim;
        if n == 0 || n * self.ldim != latents.len() || n > self.frames {
            bail!(
                "decode_block: {} values is not 1..={} frames of {}",
                latents.len(),
                self.frames,
                self.ldim
            );
        }
        let f = self.frames;
        // [n, ldim] → channel-major [1, ldim, F, 1].
        let mut x = vec![0.0f32; self.ldim * f];
        for i in 0..n {
            for c in 0..self.ldim {
                x[c * f + i] = latents[i * self.ldim + c];
            }
        }
        let mut audio = self.stream.run(f, &x, &[])?;
        if audio.len() != f * self.frame_size {
            bail!(
                "decoder returned {} samples, expected {}",
                audio.len(),
                f * self.frame_size
            );
        }
        audio.truncate(n * self.frame_size);
        Ok(audio)
    }

    /// Decode a whole utterance from a fresh state.
    pub fn decode_all(&mut self, latents: &[f32]) -> Result<Vec<f32>> {
        self.reset();
        let mut out = Vec::new();
        for block in latents.chunks(self.frames * self.ldim) {
            out.extend(self.decode_block(block)?);
        }
        Ok(out)
    }
}

/// Mimi encoder (voice prompts → unquantized latents) on one device.
pub struct MimiEncoder {
    stream: Stream,
    frames: usize,
    ldim: usize,
    frame_size: usize,
}

impl MimiEncoder {
    /// Compile the encoder for blocks of `frames` latent frames.
    pub fn new(ck: &Checkpoint, spec: &ModelSpec, frames: usize, device: Device) -> Result<Self> {
        if frames == 0 {
            bail!("encoder block needs at least one frame");
        }
        if spec.mimi.channels != 1 {
            bail!("only mono Mimi is supported");
        }
        let block = build_encoder_block(ck, spec, frames)?;
        let m = &spec.mimi;
        Ok(Self {
            stream: Stream::new(block, &m.transformer, m.steps_per_frame(), device)?,
            frames,
            ldim: spec.latent_dim(),
            frame_size: m.frame_size(),
        })
    }

    /// `MimiModel.encode_to_latent` of mono 24 kHz `audio`: returns
    /// `[ceil(len / frame_size), 32]` row-major.
    pub fn encode(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        self.stream.reset();
        let total = audio.len().div_ceil(self.frame_size);
        let block_samples = self.frames * self.frame_size;
        let mut out = Vec::with_capacity(total * self.ldim);
        for (bi, chunk) in audio.chunks(block_samples).enumerate() {
            let mut x = chunk.to_vec();
            x.resize(block_samples, 0.0);
            let first = [if bi == 0 { 1.0f32 } else { 0.0 }];
            let y = self.stream.run(self.frames, &x, &[("first", &first)])?;
            // [1, ldim, F, 1] → rows.
            let n = (total - bi * self.frames).min(self.frames);
            for i in 0..n {
                for c in 0..self.ldim {
                    out.push(y[c * self.frames + i]);
                }
            }
        }
        Ok(out)
    }
}
