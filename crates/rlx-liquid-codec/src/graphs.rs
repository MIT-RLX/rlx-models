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

//! The two graphs: encoder (STFT features → quantizer latents) and decoder
//! (FSQ latents → log-magnitude + phase per STFT frame).
//!
//! Everything is causal — the front-end convolution is left-padded, the
//! frame-stack convolution does not overlap frames, both LFM3 stacks look
//! only backwards — so a graph built for a padded bucket gives the same rows
//! for the real frames, and the decoder's sliding window bounds how far back
//! any output row reads ([`TrunkSpec`] → [`reach`]).

use anyhow::{Result, anyhow, bail};
use rlx_ir::NodeId;
use rlx_ir::infer::GraphExt;
use rlx_lfm3_audio::graph::Gb;
use rlx_lfm3_audio::trunk::{self, MMPROJ_NAMES, Mixer, Rope, StepState, TrunkSpec};

use crate::config::LiquidCodecConfig;
use crate::weights::CodecWeights;

/// RoPE base of both stacks (`LiquidCodecConfig.lfm_config`).
const ROPE_THETA: f64 = 1_000_000.0;
/// RMSNorm epsilon of both stacks.
const NORM_EPS: f32 = 1e-5;

/// One LFM3 stack (`enc.` or `dec.`), its layout read from the tensors:
/// attention where `attn_q` exists, short convolution elsewhere.
pub fn stack_spec(w: &CodecWeights, stack: &str, window: Option<usize>) -> Result<TrunkSpec> {
    let prefix = format!("{stack}blk.");
    let n = (0..)
        .take_while(|i| w.has(&format!("{prefix}{i}.ln1.weight")))
        .count();
    if n == 0 {
        bail!("liquid-codec: no {prefix}* layers");
    }
    let is_attn = |i: usize| w.has(&format!("{prefix}{i}.attn_q.weight"));
    let attn = (0..n)
        .find(|&i| is_attn(i))
        .ok_or_else(|| anyhow!("liquid-codec: {stack} stack has no attention layer"))?;
    let conv = (0..n)
        .find(|&i| !is_attn(i))
        .ok_or_else(|| anyhow!("liquid-codec: {stack} stack has no convolution layer"))?;
    let head_dim = w.shape(&format!("{prefix}{attn}.attn_q_norm.weight"))?[0];
    let rows = |name: &str| -> Result<usize> { Ok(w.shape(&format!("{prefix}{attn}.{name}"))?[0]) };
    let conv_shape = w.shape(&format!("{prefix}{conv}.conv_dw.weight"))?;
    Ok(TrunkSpec {
        prefix: prefix.clone(),
        final_norm: format!("{stack}output_norm.weight"),
        names: MMPROJ_NAMES,
        hidden: w.shape(&format!("{stack}blk.0.ln1.weight"))?[0],
        heads: rows("attn_q.weight")? / head_dim,
        kv_heads: rows("attn_k.weight")? / head_dim,
        head_dim,
        conv_kernel: *conv_shape.last().expect("conv weight rank"),
        eps: NORM_EPS,
        layers: (0..n)
            .map(|i| {
                if is_attn(i) {
                    Mixer::Attn { rope: 0, window }
                } else {
                    Mixer::Conv
                }
            })
            .collect(),
        ropes: vec![Rope {
            theta: ROPE_THETA,
            partial: 1.0,
        }],
        causal: true,
        gate: w.has(&format!("{stack}blk.{attn}.attn_gate.weight")),
        sandwich: w.has(&format!("{stack}blk.0.attn_post_norm.weight")),
    })
}

/// How many positions back one output row of `spec` can read: each
/// convolution adds `kernel - 1`, each attention layer `window - 1`
/// (`None` = unbounded).
pub fn reach(spec: &TrunkSpec) -> Option<usize> {
    spec.layers.iter().try_fold(0, |acc, m| match *m {
        Mixer::Conv => Some(acc + spec.conv_kernel - 1),
        Mixer::Attn { window, .. } => window.map(|w| acc + w - 1),
    })
}

/// `x @ W_jᵀ` summed over taps: the weight `[out, in, taps]` of a
/// `Conv1d`, applied to `taps` row-shifted views of `x` (`[rows + taps - 1,
/// in]` → `[rows, out]`).
fn conv_taps(gb: &mut Gb, x: NodeId, key: &str, rows: usize) -> Result<NodeId> {
    let (w, shape) = gb.take(key)?;
    conv_taps_of(gb, x, key, &w, &shape, rows)
}

fn conv_taps_of(
    gb: &mut Gb,
    x: NodeId,
    key: &str,
    w: &[f32],
    shape: &[usize],
    rows: usize,
) -> Result<NodeId> {
    let [out, inp, taps] = shape[..] else {
        bail!("{key}: expected [out, in, taps], got {shape:?}");
    };
    let mut acc = None;
    for j in 0..taps {
        let mut m = vec![0f32; inp * out];
        for o in 0..out {
            for c in 0..inp {
                m[c * out + o] = w[(o * inp + c) * taps + j];
            }
        }
        let wj = gb.constant(&format!("{key}#tap{j}"), m, &[inp, out]);
        let xj = if taps == 1 {
            x
        } else {
            gb.g.narrow_(x, 0, j, rows)
        };
        let y = gb.g.mm(xj, wj);
        acc = Some(match acc {
            None => y,
            Some(a) => gb.g.add(a, y),
        });
    }
    Ok(acc.expect("at least one tap"))
}

fn bias(gb: &mut Gb, x: NodeId, key: &str) -> Result<NodeId> {
    let (b, _) = gb.take(key)?;
    let n = b.len();
    let b = gb.constant(key, b, &[1, n]);
    Ok(gb.g.add(x, b))
}

/// The front end over `frames` codec frames of `features` (`[frames *
/// stack, 3 * bins]`) → `[frames, encoder_dim]`. `history` gives the rows
/// before the first one to the causal width-3 convolution (zeros at the
/// start of a clip); returns the output and the last such rows, the history
/// of the next call.
fn front_end(
    gb: &mut Gb,
    cfg: &LiquidCodecConfig,
    x: NodeId,
    frames: usize,
    history: impl FnOnce(&mut Gb, usize) -> NodeId,
) -> Result<(NodeId, NodeId)> {
    let stack = cfg.stft_frame_stack;
    let steps = frames * stack;
    let conv_dim = cfg.encoder_conv_dim;
    // in_proj: a 1×1 convolution.
    let h = conv_taps(gb, x, "frontend.in_proj.weight", steps)?;
    let h = bias(gb, h, "frontend.in_proj.bias")?;
    let h = gb.g.silu(h);
    // conv: kernel 3, padded 2 on both sides and cut back to the input
    // length — i.e. causal.
    let (w, shape) = gb.take("frontend.conv.weight")?;
    let taps = *shape.last().unwrap_or(&1);
    let past = history(gb, taps - 1);
    let h = gb.g.concat_(vec![past, h], 0);
    let next = gb.g.narrow_(h, 0, steps, taps - 1);
    let h = conv_taps_of(gb, h, "frontend.conv.weight", &w, &shape, steps)?;
    let h = bias(gb, h, "frontend.conv.bias")?;
    let h = gb.g.silu(h);
    // out: kernel = stride = frame stack, one codec frame per `stack` rows;
    // as rows, `[frames, stack * conv_dim]` against the taps laid side by side.
    let h =
        gb.g.reshape_(h, vec![frames as i64, (stack * conv_dim) as i64]);
    let (w, shape) = gb.take("frontend.out.weight")?;
    let [out, inp, k] = shape[..] else {
        bail!("frontend.out.weight: {shape:?}");
    };
    if inp != conv_dim || k != stack {
        bail!("frontend.out.weight {shape:?} vs conv dim {conv_dim}, frame stack {stack}");
    }
    let mut m = vec![0f32; k * inp * out];
    for o in 0..out {
        for c in 0..inp {
            for j in 0..k {
                m[(j * inp + c) * out + o] = w[(o * inp + c) * k + j];
            }
        }
    }
    let wm = gb.constant("frontend.out.weight", m, &[k * inp, out]);
    let h = gb.g.mm(h, wm);
    Ok((bias(gb, h, "frontend.out.bias")?, next))
}

/// The encoder stack's output `[frames, encoder_dim]` → quantizer latents.
fn to_latents(gb: &mut Gb, y: NodeId, frames: usize) -> Result<NodeId> {
    let y = gb.linear_bias(y, "to_quantizer.weight", "to_quantizer.bias", frames)?;
    gb.linear_bias(
        y,
        "quantizer.project_in.weight",
        "quantizer.project_in.bias",
        frames,
    )
}

/// Name of the front end's convolution history in [`encoder_step`].
pub const FRONT_END_STATE: &str = "frontend_conv_in";

/// Encoder over `frames` codec frames: input `features` `[frames * stack, 3
/// * bins]` ([`crate::stft::features`]), output the quantizer latents
/// `[frames, n_q * d]` (`project_in`'s output, before the bound).
pub fn encoder(
    gb: &mut Gb,
    cfg: &LiquidCodecConfig,
    spec: &TrunkSpec,
    frames: usize,
) -> Result<()> {
    let conv_dim = cfg.encoder_conv_dim;
    let x = gb.input("features", &[frames * cfg.stft_frame_stack, 3 * cfg.bins()]);
    let (h, _) = front_end(gb, cfg, x, frames, |gb, rows| {
        gb.shared(
            &format!("liquid.causal_pad.{rows}x{conv_dim}"),
            &[rows, conv_dim],
            || vec![0.0; rows * conv_dim],
        )
    })?;
    let y = trunk::sequence(gb, spec, h, frames)?;
    let lat = to_latents(gb, y, frames)?;
    gb.g.set_outputs(vec![lat]);
    Ok(())
}

/// The encoder advanced by `frames` codec frames against `max_kv`-slot
/// attention caches ([`trunk::step`]) and the front end's convolution
/// history ([`FRONT_END_STATE`], `[2, conv_dim]`). Outputs: the latents,
/// then the carried state in the returned [`StepState`] order.
pub fn encoder_step(
    gb: &mut Gb,
    cfg: &LiquidCodecConfig,
    spec: &TrunkSpec,
    frames: usize,
    max_kv: usize,
) -> Result<StepState> {
    let conv_dim = cfg.encoder_conv_dim;
    let x = gb.input("features", &[frames * cfg.stft_frame_stack, 3 * cfg.bins()]);
    let mut taps = 0;
    let (h, fe_next) = front_end(gb, cfg, x, frames, |gb, rows| {
        taps = rows;
        gb.input(FRONT_END_STATE, &[rows, conv_dim])
    })?;
    let (y, caches, mut state) = trunk::step(gb, spec, h, frames, max_kv)?;
    let lat = to_latents(gb, y, frames)?;
    let mut outs = vec![lat];
    outs.extend(caches);
    outs.push(fe_next);
    state.names.push(FRONT_END_STATE.into());
    state.sizes.push(taps * conv_dim);
    gb.g.set_outputs(outs);
    Ok(state)
}

/// Latents `[frames, n_q * d]` → the decoder stack's input `[frames *
/// stack, hidden]` (`project_out`, `from_quantizer`, nearest upsampling).
fn decoder_input(
    gb: &mut Gb,
    cfg: &LiquidCodecConfig,
    spec: &TrunkSpec,
    frames: usize,
) -> Result<NodeId> {
    let stack = cfg.stft_frame_stack;
    let lat = gb.input("latents", &[frames, cfg.n_q * cfg.codebook_dim()]);
    let x = gb.linear_bias(
        lat,
        "quantizer.project_out.weight",
        "quantizer.project_out.bias",
        frames,
    )?;
    let x = gb.linear_bias(x, "from_quantizer.weight", "from_quantizer.bias", frames)?;
    // Nearest upsampling by the frame stack: every frame repeated.
    let h = spec.hidden;
    let x = gb.g.reshape_(x, vec![frames as i64, 1, h as i64]);
    let x = gb.g.concat_(std::iter::repeat_n(x, stack).collect(), 1);
    Ok(gb.g.reshape_(x, vec![(frames * stack) as i64, h as i64]))
}

/// One streaming decoder step over `frames` new codec frames against
/// `max_kv`-slot attention caches and the convolution states
/// ([`trunk::step`]: its inputs besides `latents`, its cache outputs after
/// the spectrum `[frames * stack, 2 * bins]`).
pub fn decoder_step(
    gb: &mut Gb,
    cfg: &LiquidCodecConfig,
    spec: &TrunkSpec,
    frames: usize,
    max_kv: usize,
) -> Result<StepState> {
    let steps = frames * cfg.stft_frame_stack;
    let x = decoder_input(gb, cfg, spec, frames)?;
    let (y, caches, state) = trunk::step(gb, spec, x, steps, max_kv)?;
    let out = gb.linear_bias(y, "decoder.lin.weight", "decoder.lin.bias", steps)?;
    let mut outs = vec![out];
    outs.extend(caches);
    gb.g.set_outputs(outs);
    Ok(state)
}

/// Decoder over `frames` codec frames: input `latents` `[frames, n_q * d]`
/// ([`crate::fsq::decode`]), output `[frames * stack, 2 * bins]`
/// (log-magnitudes, then phases) for [`rlx_lfm3_audio::Istft`].
pub fn decoder(
    gb: &mut Gb,
    cfg: &LiquidCodecConfig,
    spec: &TrunkSpec,
    frames: usize,
) -> Result<()> {
    let steps = frames * cfg.stft_frame_stack;
    let x = decoder_input(gb, cfg, spec, frames)?;
    let y = trunk::sequence(gb, spec, x, steps)?;
    let out = gb.linear_bias(y, "decoder.lin.weight", "decoder.lin.bias", steps)?;
    gb.g.set_outputs(vec![out]);
    Ok(())
}
