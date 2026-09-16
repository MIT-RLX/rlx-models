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

//! The TEN-VAD network as an rlx HIR graph — CNN frontend, two stacked LSTMs,
//! two dense layers — so it runs natively on every rlx backend
//! (cpu/metal/mlx/cuda/rocm/wgpu/vulkan) through a [`Session`].
//!
//! Two shapes of the same network are built:
//!
//! * [`Shape::Streaming`] scores one frame per call and threads the four LSTM
//!   states as ordinary graph inputs/outputs. The recurrence is unrolled to a
//!   single cell step (matmul + narrow + elementwise), so nothing beyond the
//!   universally supported op set is needed.
//! * [`Shape::Batch`] scores `T` frames at once. The CNN is per-frame, so `T`
//!   becomes the batch axis there and the time axis of a real [`Op::Lstm`]
//!   scan afterwards — one dispatch instead of `T`, which is what makes GPU
//!   execution worthwhile.
//!
//! Both produce identical probabilities for a run that starts from zero state.
//!
//! Layout note: after the first (genuinely 2-D) conv every activation is a 1-D
//! sequence. Those are carried as `[N, C, L, 1]` — length in H — because that
//! is the form every backend's `Conv`/`Pool` lowering agrees on; MLX in
//! particular reads the stride/padding of the length axis from index 0.

use anyhow::{Context, Result};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op, ReduceOp};
use rlx_ir::{DType, Graph, HirGraphExt, Shape as IrShape};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::weights::{CONV_CH, DENSE_HIDDEN, LSTM1_INPUT, TenVadWeights};
use crate::{CONTEXT_FRAMES, FEATURE_LEN, HIDDEN};

const F32: DType = DType::F32;
/// Width after the 3×3 valid conv over the 41-wide feature axis.
const W0: usize = FEATURE_LEN - 2;
/// After max-pool (k=3, s=2), then the two stride-2 separable convs.
const W1: usize = (W0 - 3) / 2 + 1;
const W2: usize = (W1 + 2 - 3) / 2 + 1;
const W3: usize = (W2 + 1 - 3) / 2 + 1;

/// Which graph shape to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// One frame per run, LSTM state threaded through graph inputs/outputs.
    Streaming,
    /// `T` frames per run, LSTM state starts at zero.
    Batch(usize),
    /// `T` frames per run with the LSTM state **carried** across runs, so a
    /// stream can be scored `T` frames at a time and still be one continuous
    /// sequence. `T = 1` degenerates to [`Self::Streaming`] but through the
    /// native recurrence op, which is slower at that size — see
    /// the private `Ctx::lstm_step`. Useful from a handful of frames up.
    Chunk(usize),
}

impl Shape {
    fn frames(self) -> usize {
        match self {
            Self::Streaming => 1,
            Self::Batch(t) | Self::Chunk(t) => t,
        }
    }

    /// Whether the graph keeps LSTM state in params across runs.
    fn carries_state(self) -> bool {
        matches!(self, Self::Chunk(_))
    }
}

/// Param names holding the carried LSTM state, in `[h1, c1, h2, c2]` order.
pub(crate) const STATE_PARAMS: [&str; 4] = ["state.h1", "state.c1", "state.h2", "state.c2"];

type NamedTensors = Vec<(String, Vec<f32>)>;

struct Ctx<'a, 'b> {
    g: &'a mut HirMut<'b>,
    params: NamedTensors,
    next: usize,
}

impl Ctx<'_, '_> {
    fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "param shape {shape:?} vs {} values",
            data.len()
        );
        let name = format!("w{}", self.next);
        self.next += 1;
        let id = self.g.param(name.clone(), IrShape::new(shape, F32));
        self.params.push((name, data));
        id
    }

    /// A param under a caller-chosen name, so it can be rewritten later.
    fn named_param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let id = self.g.param(name.to_string(), IrShape::new(shape, F32));
        self.params.push((name.to_string(), data));
        id
    }

    fn zeros(&mut self, shape: &[usize]) -> HirNodeId {
        let n: usize = shape.iter().product();
        self.param(vec![0.0; n], shape)
    }

    fn relu(&mut self, x: HirNodeId, shape: &[usize]) -> HirNodeId {
        self.g
            .activation(Activation::Relu, x, IrShape::new(shape, F32))
    }

    /// Broadcast-add a per-channel bias to an `[N, C, H, W]` tensor.
    ///
    /// Kept in the canonical `bias[C] → Reshape([1,C,1,1]) → Expand → Add`
    /// form: `rlx-fusion`'s conv-bias-act matcher peels the wrappers and
    /// requires the source to be the rank-1 `[C]` vector, and a *bare* rank-1
    /// operand would broadcast along W instead of C.
    fn add_bias_nchw(&mut self, x: HirNodeId, bias: &[f32], shape: &[usize; 4]) -> HirNodeId {
        let b = self.param(bias.to_vec(), &[shape[1]]);
        let b = self.g.reshape_(b, vec![1, shape[1] as i64, 1, 1]);
        let target: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let be = self.g.add_node(
            Op::Expand {
                target_shape: target,
            },
            vec![b],
            IrShape::new(shape, F32),
        );
        self.g.add(x, be)
    }

    /// Broadcast-add a `[n]` bias to a `[rows, n]` tensor.
    ///
    /// Added **bare**, with no `Expand`: `rlx-fusion`'s matmul-bias-act matcher
    /// reads the rank of the `Add`'s operand directly (it does not peel
    /// wrappers the way the conv matcher does), so an expanded `[rows, n]`
    /// bias is reported as `BiasRankTooHigh` and the whole
    /// `matmul → add → act` chain stays unfused. The implicit trailing-dim
    /// broadcast is the same computation.
    fn add_bias_2d(&mut self, x: HirNodeId, bias: &[f32], rows: usize) -> HirNodeId {
        let n = bias.len();
        let b = self.param(bias.to_vec(), &[n]);
        let _ = rows;
        self.g.add(x, b)
    }

    fn conv(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        kernel: [usize; 2],
        stride: [usize; 2],
        groups: usize,
        out: &[usize; 4],
    ) -> HirNodeId {
        self.g.add_node(
            Op::Conv {
                kernel_size: kernel.to_vec(),
                stride: stride.to_vec(),
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups,
            },
            vec![x, w],
            IrShape::new(out, F32),
        )
    }

    /// Zero-pad the H (length) axis by `(left, right)`. Explicit concat rather
    /// than the conv's own `padding`, which is symmetric-only and which the
    /// backends read off different axes for the `[N, C, L, 1]` layout.
    fn pad_len(&mut self, x: HirNodeId, n: usize, c: usize, pad: (usize, usize)) -> HirNodeId {
        let mut parts = Vec::new();
        if pad.0 > 0 {
            parts.push(self.zeros(&[n, c, pad.0, 1]));
        }
        parts.push(x);
        if pad.1 > 0 {
            parts.push(self.zeros(&[n, c, pad.1, 1]));
        }
        if parts.len() == 1 {
            return x;
        }
        self.g.concat_(parts, 2)
    }

    /// `depthwise(k=3, stride 2) → pointwise(1×1) + bias → relu` on `[N, C, H, 1]`.
    fn separable(
        &mut self,
        x: HirNodeId,
        n: usize,
        h_in: usize,
        pad: (usize, usize),
        dw: &[f32],
        pw: &[f32],
        bias: &[f32],
    ) -> (HirNodeId, usize) {
        let padded = self.pad_len(x, n, CONV_CH, pad);
        let h_pad = h_in + pad.0 + pad.1;
        let h_out = (h_pad - 3) / 2 + 1;
        let wd = self.param(dw.to_vec(), &[CONV_CH, 1, 3, 1]);
        let y = self.conv(padded, wd, [3, 1], [2, 1], CONV_CH, &[n, CONV_CH, h_out, 1]);
        let wp = self.param(pw.to_vec(), &[CONV_CH, CONV_CH, 1, 1]);
        let y = self.conv(y, wp, [1, 1], [1, 1], 1, &[n, CONV_CH, h_out, 1]);
        let y = self.add_bias_nchw(y, bias, &[n, CONV_CH, h_out, 1]);
        (self.relu(y, &[n, CONV_CH, h_out, 1]), h_out)
    }

    /// CNN frontend: `[N, 3, 41]` features → `[N, 80]` LSTM input.
    fn conv_stack(&mut self, feat: HirNodeId, n: usize, w: &TenVadWeights) -> HirNodeId {
        let x = self.g.reshape_(
            feat,
            vec![n as i64, 1, CONTEXT_FRAMES as i64, FEATURE_LEN as i64],
        );
        let k = self.param(w.conv0_depthwise.clone(), &[1, 1, 3, 3]);
        let y = self.conv(x, k, [3, 3], [1, 1], 1, &[n, 1, 1, W0]);
        let p = self.param(w.conv0_pointwise.clone(), &[CONV_CH, 1, 1, 1]);
        let y = self.conv(y, p, [1, 1], [1, 1], 1, &[n, CONV_CH, 1, W0]);
        let y = self.add_bias_nchw(y, &w.conv0_bias, &[n, CONV_CH, 1, W0]);
        let y = self.relu(y, &[n, CONV_CH, 1, W0]);

        // Length W → H for the rest of the stack.
        let y = self.g.transpose_(y, vec![0, 1, 3, 2]);
        let y = self.g.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![3, 1],
                stride: vec![2, 1],
                padding: vec![0, 0],
            },
            vec![y],
            IrShape::new(&[n, CONV_CH, W1, 1], F32),
        );

        let (y, h) = self.separable(
            y,
            n,
            W1,
            (1, 1),
            &w.sep1_depthwise,
            &w.sep1_pointwise,
            &w.sep1_bias,
        );
        debug_assert_eq!(h, W2);
        let (y, h) = self.separable(
            y,
            n,
            W2,
            (0, 1),
            &w.sep2_depthwise,
            &w.sep2_pointwise,
            &w.sep2_bias,
        );
        debug_assert_eq!(h, W3);

        // `[N, C, L, 1]` → `[N, L, C]` → flat `[N, L*C]`, matching the ONNX
        // `Squeeze → Transpose(0,2,1) → Reshape` that feeds the LSTM.
        let y = self
            .g
            .reshape_(y, vec![n as i64, CONV_CH as i64, W3 as i64]);
        let y = self.g.transpose_(y, vec![0, 2, 1]);
        self.g.reshape_(y, vec![n as i64, LSTM1_INPUT as i64])
    }

    /// One LSTM cell step, `[1, input] → (h, c)`, gate order `i, f, g, o`.
    ///
    /// `x` and `h` are concatenated so the two gate projections become a
    /// *single* `[1, in+H] @ [in+H, 4H]` matmul with a bare rank-1 bias, which
    /// `rlx-fusion` collapses into one `FusedMatMulBiasAct`. Kept as two
    /// separate matmuls the pass sees `add(matmul, matmul)` and fuses nothing.
    ///
    /// Deliberately *not* `Op::Lstm { carry }`, even though that is one native
    /// node and is correct on every backend as of rlx 0.2.15 (this crate found
    /// and drove the fix for the missing `hn`/`cn` write-back on Metal, MLX and
    /// wgpu). Measured on a 7.6 s clip, streaming RTF — a native recurrence
    /// kernel carries an input-projection pass and a barriered sequential
    /// dispatch that only pay off over a long sequence, and `seq = 1` is the
    /// worst case for it:
    ///
    /// | | cpu | metal | mlx | wgpu |
    /// |---|---|---|---|---|
    /// | fused cell (this) | **328×** | **6.3×** | **35×** | 11× |
    /// | `Op::Lstm { carry }` | 274× | 3.6× | 20× | **20×** |
    ///
    /// Only wgpu prefers the native op; CPU is the recommended streaming
    /// device and is ~15× faster than any GPU here, so the cell wins.
    #[allow(clippy::too_many_arguments)]
    fn lstm_step(
        &mut self,
        x: HirNodeId,
        h: HirNodeId,
        c: HirNodeId,
        w_ih: &[f32],
        w_hh: &[f32],
        bias: &[f32],
        input: usize,
    ) -> (HirNodeId, HirNodeId) {
        let gates = 4 * HIDDEN;
        let rows = input + HIDDEN;
        let w_cat = self.param(concat_gate_weights(w_ih, w_hh, input), &[rows, gates]);
        let xh = self.g.concat_(vec![x, h], 1);
        let gshape = IrShape::new(&[1, gates], F32);
        let z = self.g.add_node(Op::MatMul, vec![xh, w_cat], gshape);
        let z = self.add_bias_2d(z, bias, 1);

        let hs = IrShape::new(&[1, HIDDEN], F32);
        let mut gate = [z; 4];
        for (k, slot) in gate.iter_mut().enumerate() {
            *slot = self.g.narrow_(z, 1, k * HIDDEN, HIDDEN);
        }
        let i = self.g.activation(Activation::Sigmoid, gate[0], hs.clone());
        let f = self.g.activation(Activation::Sigmoid, gate[1], hs.clone());
        let gg = self.g.activation(Activation::Tanh, gate[2], hs.clone());
        let o = self.g.activation(Activation::Sigmoid, gate[3], hs.clone());

        let fc = self.g.mul(f, c);
        let ig = self.g.mul(i, gg);
        let c_out = self.g.add(fc, ig);
        let tc = self.g.activation(Activation::Tanh, c_out, hs);
        let h_out = self.g.mul(o, tc);
        (h_out, c_out)
    }

    /// `concat(h2, h1) → dense(128→32) → relu → dense(32→1) → sigmoid`.
    fn head(&mut self, h2: HirNodeId, h1: HirNodeId, rows: usize, w: &TenVadWeights) -> HirNodeId {
        let cat = self.g.concat_(vec![h2, h1], 1);
        let d1 = self.param(w.dense1_weight.clone(), &[2 * HIDDEN, DENSE_HIDDEN]);
        let y = self.g.add_node(
            Op::MatMul,
            vec![cat, d1],
            IrShape::new(&[rows, DENSE_HIDDEN], F32),
        );
        let y = self.add_bias_2d(y, &w.dense1_bias, rows);
        let y = self.relu(y, &[rows, DENSE_HIDDEN]);
        let d2 = self.param(w.dense2_weight.clone(), &[DENSE_HIDDEN, 1]);
        let y = self
            .g
            .add_node(Op::MatMul, vec![y, d2], IrShape::new(&[rows, 1], F32));
        let y = self.add_bias_2d(y, &w.dense2_bias, rows);
        self.g
            .activation(Activation::Sigmoid, y, IrShape::new(&[rows, 1], F32))
    }
}

/// Stack `w_ih [4H, in]` and `w_hh [4H, H]` into the transposed
/// `[in + H, 4H]` the concatenated gate matmul wants.
fn concat_gate_weights(w_ih: &[f32], w_hh: &[f32], input: usize) -> Vec<f32> {
    let gates = 4 * HIDDEN;
    let rows = input + HIDDEN;
    let mut out = vec![0.0; rows * gates];
    for g in 0..gates {
        for r in 0..input {
            out[r * gates + g] = w_ih[g * input + r];
        }
        for r in 0..HIDDEN {
            out[(input + r) * gates + g] = w_hh[g * HIDDEN + r];
        }
    }
    out
}

/// Build the rlx-ir graph and its parameter map.
///
/// Public so tooling outside inference can consume the same graph the runtime
/// does — notably `rlx-fpga`, which lowers it to RTL.
pub fn build_graph(shape: Shape, w: &TenVadWeights) -> Result<(Graph, NamedTensors)> {
    build(shape, w)
}

fn build(shape: Shape, w: &TenVadWeights) -> Result<(Graph, NamedTensors)> {
    let n = shape.frames();
    anyhow::ensure!(n > 0, "graph needs at least one frame");
    let mut hir = HirModule::new("ten_vad");
    let mut g = HirMut::new(&mut hir);
    let mut ctx = Ctx {
        g: &mut g,
        params: Vec::new(),
        next: 0,
    };

    let feat = ctx
        .g
        .input("feat", IrShape::new(&[n, CONTEXT_FRAMES, FEATURE_LEN], F32));
    let x = ctx.conv_stack(feat, n, w);

    let outputs = match shape {
        Shape::Streaming => {
            let hs = IrShape::new(&[1, HIDDEN], F32);
            let h1 = ctx.g.input("h1", hs.clone());
            let c1 = ctx.g.input("c1", hs.clone());
            let h2 = ctx.g.input("h2", hs.clone());
            let c2 = ctx.g.input("c2", hs);
            let (h1n, c1n) = ctx.lstm_step(
                x,
                h1,
                c1,
                &w.lstm1_weight_ih,
                &w.lstm1_weight_hh,
                &w.lstm1_bias,
                LSTM1_INPUT,
            );
            let (h2n, c2n) = ctx.lstm_step(
                h1n,
                h2,
                c2,
                &w.lstm2_weight_ih,
                &w.lstm2_weight_hh,
                &w.lstm2_bias,
                HIDDEN,
            );
            let prob = ctx.head(h2n, h1n, 1, w);
            vec![prob, h1n, c1n, h2n, c2n]
        }
        Shape::Chunk(t) => {
            // `[T, 80]` is already `[batch = 1, seq = T, input]` in memory.
            let seq = ctx.g.reshape_(x, vec![1, t as i64, LSTM1_INPUT as i64]);
            let out1 = lstm_carry(&mut ctx, seq, t, LSTM1_INPUT, 1, w);
            let out2 = lstm_carry(&mut ctx, out1, t, HIDDEN, 2, w);
            let h1 = ctx.g.reshape_(out1, vec![t as i64, HIDDEN as i64]);
            let h2 = ctx.g.reshape_(out2, vec![t as i64, HIDDEN as i64]);
            vec![ctx.head(h2, h1, t, w)]
        }
        Shape::Batch(t) => {
            // `[T, 80]` is already `[batch = 1, seq = T, input]` in memory.
            let seq = ctx.g.reshape_(x, vec![1, t as i64, LSTM1_INPUT as i64]);
            let out1 = lstm_scan(
                &mut ctx,
                seq,
                t,
                LSTM1_INPUT,
                &w.lstm1_weight_ih,
                &w.lstm1_weight_hh,
                &w.lstm1_bias,
            );
            let out2 = lstm_scan(
                &mut ctx,
                out1,
                t,
                HIDDEN,
                &w.lstm2_weight_ih,
                &w.lstm2_weight_hh,
                &w.lstm2_bias,
            );
            let h1 = ctx.g.reshape_(out1, vec![t as i64, HIDDEN as i64]);
            let h2 = ctx.g.reshape_(out2, vec![t as i64, HIDDEN as i64]);
            vec![ctx.head(h2, h1, t, w)]
        }
    };

    let params = ctx.params;
    hir.set_outputs(outputs);
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("ten-vad lower: {e}"))?;
    Ok((graph, params))
}

/// Single-layer forward `Op::Lstm` over `[1, T, input]` → `[1, T, HIDDEN]`.
fn lstm_scan(
    ctx: &mut Ctx,
    x: HirNodeId,
    t: usize,
    input: usize,
    w_ih: &[f32],
    w_hh: &[f32],
    bias: &[f32],
) -> HirNodeId {
    let gates = 4 * HIDDEN;
    let wi = ctx.param(w_ih.to_vec(), &[gates, input]);
    let wh = ctx.param(w_hh.to_vec(), &[gates, HIDDEN]);
    let b = ctx.param(bias.to_vec(), &[gates]);
    // `lstm` lives on `HirModule`; `HirMut::inner()` is the way through.
    ctx.g.inner().lstm(
        x,
        wi,
        wh,
        b,
        HIDDEN,
        1,
        false,
        IrShape::new(&[1, t, HIDDEN], F32),
    )
}

/// A single-layer `Op::Lstm { carry }` over `[1, T, input]` → `[1, T, HIDDEN]`.
///
/// The whole recurrence is one node with each backend's native kernel behind
/// it, and `h0`/`c0` are params the op overwrites in place so consecutive runs
/// continue the sequence. That in-place write-back was honoured only by CPU
/// until rlx 0.2.15 — Metal and wgpu seeded and silently never advanced, MLX
/// discarded the host kernel's result — which this crate found and drove the
/// fix for. [`TenVadModel::reset_state`] rewrites the params.
fn lstm_carry(
    ctx: &mut Ctx,
    x: HirNodeId,
    t: usize,
    input: usize,
    layer: usize,
    w: &TenVadWeights,
) -> HirNodeId {
    let (w_ih, w_hh, bias) = if layer == 1 {
        (&w.lstm1_weight_ih, &w.lstm1_weight_hh, &w.lstm1_bias)
    } else {
        (&w.lstm2_weight_ih, &w.lstm2_weight_hh, &w.lstm2_bias)
    };
    let gates = 4 * HIDDEN;
    let wi = ctx.param(w_ih.clone(), &[gates, input]);
    let wh = ctx.param(w_hh.clone(), &[gates, HIDDEN]);
    let b = ctx.param(bias.clone(), &[gates]);
    // `[L*D, batch, hidden]`
    let h0 = ctx.named_param(
        STATE_PARAMS[(layer - 1) * 2],
        vec![0.0; HIDDEN],
        &[1, 1, HIDDEN],
    );
    let c0 = ctx.named_param(
        STATE_PARAMS[(layer - 1) * 2 + 1],
        vec![0.0; HIDDEN],
        &[1, 1, HIDDEN],
    );
    ctx.g.inner().lstm_carry(
        x,
        wi,
        wh,
        b,
        h0,
        c0,
        HIDDEN,
        1,
        false,
        IrShape::new(&[1, t, HIDDEN], F32),
    )
}

/// A compiled TEN-VAD network bound to one device.
pub struct TenVadModel {
    compiled: CompiledGraph,
    shape: Shape,
    device: Device,
}

impl TenVadModel {
    pub fn new(device: Device, shape: Shape, weights: &TenVadWeights) -> Result<Self> {
        let (graph, params) = build(shape, weights)?;
        let session = Session::new(device);
        let mut compiled = session.compile(graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Ok(Self {
            compiled,
            shape,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn frames(&self) -> usize {
        self.shape.frames()
    }

    /// Score one frame, threading the LSTM state through `state`.
    pub fn step(&mut self, feat: &[f32], state: &mut LstmState) -> Result<f32> {
        debug_assert_eq!(self.shape, Shape::Streaming);
        anyhow::ensure!(
            feat.len() == CONTEXT_FRAMES * FEATURE_LEN,
            "expected {} features, got {}",
            CONTEXT_FRAMES * FEATURE_LEN,
            feat.len()
        );
        let out = self.compiled.run(&[
            ("feat", feat),
            ("h1", &state.h1),
            ("c1", &state.c1),
            ("h2", &state.h2),
            ("c2", &state.c2),
        ]);
        anyhow::ensure!(
            out.len() == 5,
            "streaming graph returned {} outputs",
            out.len()
        );
        let prob = *out[0].first().context("empty probability output")?;
        state.h1.copy_from_slice(&out[1]);
        state.c1.copy_from_slice(&out[2]);
        state.h2.copy_from_slice(&out[3]);
        state.c2.copy_from_slice(&out[4]);
        Ok(prob)
    }

    /// Zero the carried LSTM state ([`Shape::Chunk`] only) — a new utterance,
    /// or the periodic reset `resetFrameNum` performs upstream.
    pub fn reset_state(&mut self) {
        debug_assert!(self.shape.carries_state());
        let zeros = [0.0f32; HIDDEN];
        for name in STATE_PARAMS {
            self.compiled.set_param(name, &zeros);
        }
    }

    /// Whether this graph keeps LSTM state across runs.
    pub fn carries_state(&self) -> bool {
        self.shape.carries_state()
    }

    /// Score `frames()` frames, returning one probability each.
    pub fn run_batch(&mut self, feat: &[f32]) -> Result<Vec<f32>> {
        let t = self.frames();
        anyhow::ensure!(
            feat.len() == t * CONTEXT_FRAMES * FEATURE_LEN,
            "expected {} features for {t} frames, got {}",
            t * CONTEXT_FRAMES * FEATURE_LEN,
            feat.len()
        );
        let mut out = self.compiled.run(&[("feat", feat)]);
        let probs = out.pop().context("batch graph returned no output")?;
        anyhow::ensure!(
            probs.len() == t,
            "expected {t} probabilities, got {}",
            probs.len()
        );
        Ok(probs)
    }
}

/// The four LSTM state vectors carried between streaming frames.
#[derive(Debug, Clone)]
pub struct LstmState {
    pub h1: Vec<f32>,
    pub c1: Vec<f32>,
    pub h2: Vec<f32>,
    pub c2: Vec<f32>,
}

impl Default for LstmState {
    fn default() -> Self {
        Self {
            h1: vec![0.0; HIDDEN],
            c1: vec![0.0; HIDDEN],
            h2: vec![0.0; HIDDEN],
            c2: vec![0.0; HIDDEN],
        }
    }
}

impl LstmState {
    pub fn clear(&mut self) {
        self.h1.fill(0.0);
        self.c1.fill(0.0);
        self.h2.fill(0.0);
        self.c2.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = assert!(W0 == 39 && W1 == 19 && W2 == 10 && W3 == 5);

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.31).sin() * 1.7).collect()
    }

    #[test]
    fn streaming_and_batch_agree() {
        let w = TenVadWeights::embedded();
        let t = 7usize;
        let feats = ramp(t * CONTEXT_FRAMES * FEATURE_LEN);

        let mut stream = TenVadModel::new(Device::Cpu, Shape::Streaming, w).unwrap();
        let mut state = LstmState::default();
        let per_frame: Vec<f32> = (0..t)
            .map(|i| {
                let n = CONTEXT_FRAMES * FEATURE_LEN;
                stream.step(&feats[i * n..(i + 1) * n], &mut state).unwrap()
            })
            .collect();

        let mut batch = TenVadModel::new(Device::Cpu, Shape::Batch(t), w).unwrap();
        let all = batch.run_batch(&feats).unwrap();

        for (i, (a, b)) in per_frame.iter().zip(&all).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "frame {i}: streaming {a} vs batch {b}"
            );
        }
    }

    /// Every fusable pattern in either graph shape must actually fuse.
    ///
    /// `assert_fusion_clean` fails the compile if `rlx-fusion` leaves a
    /// recognised pattern on the table — which is how the rank-2 bias that kept
    /// all four gate/dense matmuls unfused was found. `RLX_FUSION_REPORT=1` on
    /// any run prints the same tally with reasons.
    #[test]
    fn both_graph_shapes_are_fully_fused() {
        let w = TenVadWeights::embedded();
        for shape in [Shape::Streaming, Shape::Batch(4)] {
            let (graph, params) = build(shape, w).expect("build");
            let session = Session::new(Device::Cpu);
            let opts = rlx_runtime::CompileOptions::default().assert_fusion_clean(true);
            let mut compiled = session.compile_with(graph, &opts);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            compiled.finalize_params();
        }
    }

    #[test]
    fn probabilities_are_in_range() {
        let w = TenVadWeights::embedded();
        let mut m = TenVadModel::new(Device::Cpu, Shape::Batch(4), w).unwrap();
        for p in m
            .run_batch(&ramp(4 * CONTEXT_FRAMES * FEATURE_LEN))
            .unwrap()
        {
            assert!((0.0..=1.0).contains(&p), "probability {p} out of range");
        }
    }

    /// The threaded state must actually advance, and clearing it must undo that.
    #[test]
    fn threaded_state_advances_and_resets() {
        let w = TenVadWeights::embedded();
        let mut m = TenVadModel::new(Device::Cpu, Shape::Streaming, w).unwrap();
        let mut state = LstmState::default();
        let feat = ramp(CONTEXT_FRAMES * FEATURE_LEN);
        let first = m.step(&feat, &mut state).unwrap();
        let second = m.step(&feat, &mut state).unwrap();
        assert!((first - second).abs() > 1e-6, "state did not advance");
        state.clear();
        let after = m.step(&feat, &mut state).unwrap();
        assert!(
            (first - after).abs() < 1e-6,
            "clearing did not restore state"
        );
    }
}
