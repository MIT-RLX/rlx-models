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

//! CPU kernels for native Kitten ops (registered before `compile_hir`).

use std::sync::Arc;

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

use crate::lstm::{LstmAttrs, dynamic_lstm_f32, dynamic_quantize_lstm};
use crate::qmatmul::{qmatmul_uint8_act_f32_weight_into, qmatmul_uint8_act_i8_weight_into};
use crate::scatter::{scatter_elements, scatter_nd_inplace_limited};

pub const DYNAMIC_QUANTIZE_LSTM: &str = "onnx.DynamicQuantizeLSTM";
pub const SCATTER_ND: &str = "onnx.ScatterND";
pub const SCATTER_ELEMENTS: &str = "onnx.ScatterElements";
pub const CONCAT_FROM_SEQUENCE: &str = "onnx.KittenConcatFromSequence";
/// Generic import lowering name (same kernel).
pub const CONCAT_FROM_SEQUENCE_ONNX: &str = "onnx.ConcatFromSequence";
pub const RANDOM_NORMAL_LIKE: &str = "onnx.RandomNormalLike";
pub const RANDOM_UNIFORM_LIKE: &str = "onnx.RandomUniformLike";
pub const Q_MATMUL: &str = "onnx.QMatMul";
pub const Q_MATMUL_BAKED: &str = "onnx.QMatMulBaked";
pub const ACT_COPY: &str = "onnx.ActCopy";
/// Copy F0 features into the `/If` stub buffer (batch row 0 → `[1,T]`).
pub const F0_IF_BYPASS: &str = "onnx.F0IfBypass";
/// Trim F0 cast to runtime mel length (`2 ×` alignment frames).
pub const F0_IF_SELECT: &str = "onnx.F0IfSelect";
pub const ALIGNMENT_SCATTER_INDICES: &str = "onnx.AlignmentScatterIndices";
pub const DYNAMIC_QUANTIZE_LINEAR: &str = "onnx.DynamicQuantizeLinearExport";

#[derive(Debug, Clone, Copy)]
struct LstmKernelAttrs {
    hidden_size: usize,
    bidirectional: bool,
}

fn parse_lstm_attrs(attrs: &[u8]) -> LstmKernelAttrs {
    if attrs.len() >= 8 {
        let hidden_size = u32::from_le_bytes(attrs[0..4].try_into().unwrap()) as usize;
        let bidirectional = attrs[4] != 0;
        return LstmKernelAttrs {
            hidden_size,
            bidirectional,
        };
    }
    LstmKernelAttrs {
        hidden_size: 256,
        bidirectional: true,
    }
}

fn shape_usize(sh: &rlx_ir::Shape) -> Vec<usize> {
    let mut out = Vec::with_capacity(sh.rank());
    for d in sh.dims() {
        match d {
            rlx_ir::Dim::Static(n) => out.push(*n),
            rlx_ir::Dim::Dynamic(_) => return Vec::new(),
        }
    }
    out
}

fn shape_for_buffer(buf_len: usize, shape: &[usize]) -> Vec<usize> {
    let want: usize = shape.iter().product::<usize>().max(1);
    if want == buf_len {
        shape.to_vec()
    } else {
        vec![buf_len.max(1)]
    }
}

fn read_zp_u8(inp: &CpuTensorRef<'_>, name: &str) -> Result<u8, String> {
    match inp.shape().dtype() {
        rlx_ir::DType::U8 => Ok(inp.expect_u8(name)?[0]),
        rlx_ir::DType::I8 => Ok(inp.expect_i8(name)?[0] as u8),
        rlx_ir::DType::I32 => Ok(inp.expect_i32(name)?[0].clamp(0, 255) as u8),
        rlx_ir::DType::F32 => Ok(inp.expect_f32(name)?[0].round().clamp(0.0, 255.0) as u8),
        dt => Err(format!("{name}: expected zp as U8/I8/I32/F32, got {dt:?}")),
    }
}

fn read_zp(inp: &CpuTensorRef<'_>, name: &str) -> Result<Vec<i32>, String> {
    match inp.shape().dtype() {
        rlx_ir::DType::I32 => Ok(inp.expect_i32(name)?.to_vec()),
        rlx_ir::DType::I8 => Ok(inp.expect_i8(name)?.iter().map(|&x| x as i32).collect()),
        rlx_ir::DType::F32 => Ok(inp
            .expect_f32(name)?
            .iter()
            .map(|&x| x.round() as i32)
            .collect()),
        dt => Err(format!("{name}: expected zp as I32/I8/F32, got {dt:?}")),
    }
}

struct DynamicQuantizeLstmKernel;

impl CpuKernel for DynamicQuantizeLstmKernel {
    fn name(&self) -> &str {
        DYNAMIC_QUANTIZE_LSTM
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        // ONNX lists optional sequence_lens / initial_h / initial_c / P; bundle
        // lowering drops empty names so we may see 11 tensors instead of 12.
        // Quant params are always the last four inputs.
        if inputs.len() < 8 {
            return Err(format!("expected at least 8 inputs, got {}", inputs.len()));
        }
        let ka = parse_lstm_attrs(attrs);
        let x_dims = shape_usize(inputs[0].shape());
        let y_shape = output.shape().clone();
        let y_dims = shape_usize(&y_shape);
        let x = inputs[0].expect_f32("X")?;
        let b = inputs[3].expect_f32("B")?;
        let y = output.expect_f32_mut("Y")?;
        let runtime_seq = crate::opts::compile_sequence_length_from_env().filter(|&n| n > 0);
        let runtime_mel = crate::opts::runtime_mel_frames();
        let x_dims_buf: Option<[usize; 3]> = if x_dims.len() == 3 && x_dims.iter().all(|&d| d > 0) {
            let (mut seq, mut batch) = if x_dims[0] >= x_dims[1] {
                (x_dims[0], x_dims[1].max(1))
            } else {
                (x_dims[1], x_dims[0].max(1))
            };
            let mut input_size = x_dims[2];
            if let Some(rt) = runtime_seq {
                seq = seq.min(rt);
            }
            if let Some(mel) = runtime_mel {
                // `/shared/Transpose` → LSTM X is `[mel, batch, 640]`; compile slots may be wider.
                if input_size == 640 {
                    seq = mel.min(seq);
                }
            }
            if runtime_seq.is_none() && input_size * seq * batch != x.len() && !x.is_empty() {
                let h = ka.hidden_size.max(1);
                let w_shape = shape_usize(inputs[1].shape());
                let from_w = if w_shape.len() >= 2 && h > 0 {
                    w_shape[1]
                } else {
                    0
                };
                if from_w > 0 {
                    input_size = from_w;
                    batch = batch.max(1);
                    seq = x.len() / (batch * input_size);
                }
            }
            Some([seq, batch, input_size])
        } else if y_dims.len() == 4 && y_dims.iter().all(|&d| d > 0) {
            // ONNX Y: [seq, num_directions, batch, hidden]
            let (mut seq, batch) = if y_dims[0] >= y_dims[2] {
                (y_dims[0], y_dims[2].max(1))
            } else {
                (y_dims[2], y_dims[0].max(1))
            };
            let input_size = if x_dims.len() == 3 {
                x_dims[2]
            } else {
                x.len().checked_div(seq * batch).unwrap_or(0)
            };
            if let Some(rt) = runtime_seq {
                seq = seq.min(rt);
            }
            Some([seq, batch, input_size])
        } else {
            let dirs = if ka.bidirectional { 2 } else { 1 };
            let h = ka.hidden_size.max(1);
            let batch = 1usize;
            let mut seq = y.len() / (dirs * batch * h);
            if let Some(rt) = runtime_seq {
                seq = seq.min(rt);
            }
            let input_size = if seq > 0 && batch > 0 {
                x.len() / (seq * batch)
            } else {
                0
            };
            if seq > 0 && input_size > 0 {
                Some([seq, batch, input_size])
            } else {
                None
            }
        };
        let x_dims_ref = x_dims_buf.as_ref().map(|b| b.as_slice());
        let attrs = LstmAttrs {
            hidden_size: ka.hidden_size,
            bidirectional: ka.bidirectional,
        };
        let run = if inputs[1].shape().dtype() == rlx_ir::DType::I8 {
            let w = inputs[1].expect_i8("W")?;
            let r = inputs[2].expect_i8("R")?;
            let n = inputs.len();
            let w_scale = inputs[n - 4].expect_f32("W_scale")?;
            let w_zp = read_zp(&inputs[n - 3], "W_zp")?;
            let r_scale = inputs[n - 2].expect_f32("R_scale")?;
            let r_zp = read_zp(&inputs[n - 1], "R_zp")?;
            dynamic_quantize_lstm(
                x, x_dims_ref, w, r, b, w_scale, &w_zp, r_scale, &r_zp, attrs, y,
            )
        } else {
            let w = inputs[1].expect_f32("W")?;
            let r = inputs[2].expect_f32("R")?;
            dynamic_lstm_f32(x, x_dims_ref, w, r, b, attrs, y)
        };
        run?;
        // Duration `/lstm/LSTM_quant` has 10 tensor inputs (no extra state outs).
        let is_duration_lstm = inputs.len() == 10;
        if std::env::var("KITTEN_RLX_DEBUG_PRED_LSTM").is_ok()
            && is_duration_lstm
            && ka.hidden_size == 256
            && ka.bidirectional
            && x_dims_ref.is_some_and(|d| d.len() == 3 && d[2] == 640)
        {
            let h = ka.hidden_size;
            let seq = x_dims_ref.unwrap()[0];
            eprintln!(
                "pred_lstm X dims={:?} x[0..4]={:?}",
                x_dims_ref,
                &x[..4.min(x.len())]
            );
            eprintln!(
                "pred_lstm Y dims={y_dims:?} t0[0..4]={:?} t1[0..4]={:?}",
                &y[..4.min(y.len())],
                if seq > 1 && y.len() > h {
                    &y[h..(h + 4).min(y.len())]
                } else {
                    &[] as &[f32]
                }
            );
        }
        Ok(())
    }
}

struct ScatterNdKernel;

impl CpuKernel for ScatterNdKernel {
    fn name(&self) -> &str {
        SCATTER_ND
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 3 {
            return Err(format!("expected 3 inputs, got {}", inputs.len()));
        }
        let out = output.expect_f32_mut("output")?;
        let indices = inputs[1].expect_i64("indices")?;
        let updates = inputs[2].expect_f32("updates")?;
        if let Some(data) = inputs[0].as_f32() {
            if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                let n = data.len().min(out.len());
                out[..n].copy_from_slice(&data[..n]);
            }
        }
        let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].shape()));
        let indices_shape = shape_for_buffer(indices.len(), &shape_usize(inputs[1].shape()));
        let index_depth = indices_shape
            .last()
            .copied()
            .filter(|&d| d > 0)
            .unwrap_or(1);
        let max_updates = (index_depth == 2)
            .then(crate::opts::runtime_mel_frames)
            .flatten()
            .filter(|&frames| {
                indices_shape
                    .first()
                    .is_some_and(|&rows| rows > frames && rows > 32)
            });
        if std::env::var("KITTEN_RLX_DEBUG_SCATTER").is_ok_and(|v| v == "1") {
            eprintln!(
                "[scatter] data_shape={data_shape:?} indices_shape={indices_shape:?} \
                 indices_len={} updates_len={} out_len={} max_updates={max_updates:?}",
                indices.len(),
                updates.len(),
                out.len()
            );
            let show = (max_updates.unwrap_or(indices.len() / index_depth) * index_depth)
                .min(indices.len())
                .min(8);
            eprintln!("[scatter] indices_t0={:?}", &indices[..show]);
        }
        scatter_nd_inplace_limited(
            out,
            &data_shape,
            indices,
            &indices_shape,
            updates,
            max_updates,
        );
        Ok(())
    }
}

struct ScatterElementsKernel;

impl CpuKernel for ScatterElementsKernel {
    fn name(&self) -> &str {
        SCATTER_ELEMENTS
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 3 {
            return Err(format!("expected 3 inputs, got {}", inputs.len()));
        }
        let axis = if attrs.len() >= 4 {
            i32::from_le_bytes(attrs[0..4].try_into().unwrap())
        } else {
            0
        };
        if inputs[0].shape().dtype() == rlx_ir::DType::I64 {
            let out = output.expect_i64_mut("output")?;
            let indices = inputs[1].expect_i64("indices")?;
            let updates = inputs[2].expect_i64("updates")?;
            if let Some(data) = inputs[0].as_i64() {
                if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                    let n = data.len().min(out.len());
                    out[..n].copy_from_slice(&data[..n]);
                }
            }
            let n = out.len().min(indices.len()).min(updates.len());
            for i in 0..n {
                let j = indices[i].max(0) as usize;
                if j < out.len() {
                    out[j] = updates[i];
                }
            }
            let _ = axis;
        } else {
            let out = output.expect_f32_mut("output")?;
            let indices = inputs[1].expect_i64("indices")?;
            let updates = inputs[2].expect_f32("updates")?;
            if let Some(data) = inputs[0].as_f32() {
                if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                    let n = data.len().min(out.len());
                    out[..n].copy_from_slice(&data[..n]);
                }
            }
            let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].shape()));
            scatter_elements(out, &data_shape, indices, updates, axis);
        }
        Ok(())
    }
}

fn parse_tag(attrs: &[u8]) -> u64 {
    if attrs.len() >= 16 {
        u64::from_le_bytes(attrs[8..16].try_into().unwrap())
    } else {
        0
    }
}

struct RandomNormalLikeKernel;

impl CpuKernel for RandomNormalLikeKernel {
    fn name(&self) -> &str {
        RANDOM_NORMAL_LIKE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.is_empty() {
            return Err("RandomNormalLike: missing shape input".into());
        }
        let mean = if attrs.len() >= 4 {
            f32::from_le_bytes(attrs[0..4].try_into().unwrap())
        } else {
            0.0
        };
        let scale = if attrs.len() >= 8 {
            f32::from_le_bytes(attrs[4..8].try_into().unwrap())
        } else {
            1.0
        };
        let tag = parse_tag(attrs);
        let out = output.expect_f32_mut("output")?;
        let opts = crate::random::rng_options_from_env();
        if matches!(opts.backend, rlx_ir::RngBackend::Zero) {
            out.fill(0.0);
        } else {
            crate::random::fill_normal_with_opts(out, mean, scale, opts, tag);
        }
        Ok(())
    }
}

struct RandomUniformLikeKernel;

impl CpuKernel for RandomUniformLikeKernel {
    fn name(&self) -> &str {
        RANDOM_UNIFORM_LIKE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.is_empty() {
            return Err("RandomUniformLike: missing shape input".into());
        }
        let low = if attrs.len() >= 4 {
            f32::from_le_bytes(attrs[0..4].try_into().unwrap())
        } else {
            0.0
        };
        let high = if attrs.len() >= 8 {
            f32::from_le_bytes(attrs[4..8].try_into().unwrap())
        } else {
            1.0
        };
        let tag = parse_tag(attrs);
        let out = output.expect_f32_mut("output")?;
        let opts = crate::random::rng_options_from_env();
        if matches!(opts.backend, rlx_ir::RngBackend::Zero) {
            out.fill(0.0);
        } else {
            crate::random::fill_uniform_with_opts(out, low, high, opts, tag);
        }
        Ok(())
    }
}

struct DynamicQuantizeLinearExportKernel;

const F0_PREDICTOR_CHANNELS: usize = 256;

fn dynamic_quantize_f0_mel_layout(act: &[f32], channels: usize, time: usize) -> (Vec<u8>, f32, u8) {
    let mel = crate::opts::runtime_mel_frames().unwrap_or(time).min(time);
    let mut prefix = Vec::with_capacity(channels * mel);
    for h in 0..channels {
        let base = h * time;
        prefix.extend_from_slice(&act[base..base + mel]);
    }
    let (q_pre, scale, zp) = crate::qmatmul::dynamic_quantize_uint8(&prefix);
    let mut q = vec![0u8; act.len()];
    for h in 0..channels {
        let base = h * time;
        for i in 0..mel {
            q[base + i] = q_pre[h * mel + i];
        }
    }
    (q, scale, zp)
}

fn dynamic_quantize_f0_mel(act: &[f32]) -> (Vec<u8>, f32, u8) {
    let t = act.len() / F0_PREDICTOR_CHANNELS;
    let mel = crate::opts::runtime_mel_frames().unwrap_or(t).min(t);
    let mut prefix = Vec::with_capacity(F0_PREDICTOR_CHANNELS * mel);
    for h in 0..F0_PREDICTOR_CHANNELS {
        let base = h * t;
        prefix.extend_from_slice(&act[base..base + mel]);
    }
    let (q_pre, scale, zp) = crate::qmatmul::dynamic_quantize_uint8(&prefix);
    let mut q = vec![0u8; act.len()];
    for h in 0..F0_PREDICTOR_CHANNELS {
        let base = h * t;
        for i in 0..mel {
            q[base + i] = q_pre[h * mel + i];
        }
    }
    (q, scale, zp)
}

impl CpuKernel for DynamicQuantizeLinearExportKernel {
    fn name(&self) -> &str {
        DYNAMIC_QUANTIZE_LINEAR
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.is_empty() {
            return Err("DynamicQuantizeLinearExport expected 1 input".into());
        }
        let act = inputs[0].expect_f32("X")?;
        let dims = shape_usize(inputs[0].shape());
        let (q, scale, zp) = if dims.len() == 3 {
            let channels = dims[1];
            let time = dims[2];
            if (channels == 512 || channels == F0_PREDICTOR_CHANNELS)
                && crate::opts::runtime_mel_frames().is_some()
            {
                dynamic_quantize_f0_mel_layout(act, channels, time)
            } else {
                crate::qmatmul::dynamic_quantize_uint8(act)
            }
        } else if act.len().is_multiple_of(F0_PREDICTOR_CHANNELS)
            && crate::opts::runtime_mel_frames().is_some()
        {
            dynamic_quantize_f0_mel(act)
        } else {
            crate::qmatmul::dynamic_quantize_uint8(act)
        };
        let which = attrs.first().copied().unwrap_or(0);
        match which {
            0 => {
                let out = output.expect_u8_mut("quantized")?;
                if out.len() != q.len() {
                    return Err(format!("quantized size {} != {}", out.len(), q.len()));
                }
                out.copy_from_slice(&q);
            }
            1 => {
                let out = output.expect_f32_mut("scale")?;
                out[0] = scale;
            }
            2 => {
                let out = output.expect_u8_mut("zero_point")?;
                out[0] = zp;
            }
            _ => return Err(format!("unknown DQL export slot {which}")),
        }
        Ok(())
    }
}

struct ActCopyKernel;

struct F0IfBypassKernel;

struct F0IfSelectKernel;

impl CpuKernel for F0IfSelectKernel {
    fn name(&self) -> &str {
        F0_IF_SELECT
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let f0 = inputs
            .first()
            .ok_or("F0IfSelect: missing f0 input")?
            .expect_f32("f0")?;
        let align = inputs
            .get(1)
            .ok_or("F0IfSelect: missing alignment input")?
            .expect_i64("align")?;
        let mel = align.first().copied().unwrap_or(0).max(0) as usize;
        let out = output.expect_f32_mut("out")?;
        out.fill(0.0);
        if f0.len() == out.len() {
            let n = mel.min(f0.len());
            out[..n].copy_from_slice(&f0[..n]);
            return Ok(());
        }
        // `[1,1,T]` activations into `[1,1,T_stub]` (If stub from lower_if_stub).
        if f0.len() > out.len() && !out.is_empty() && f0.len() % out.len() == 0 {
            let chunk = f0.len() / out.len();
            let frames = mel.min(out.len());
            for (i, slot) in out.iter_mut().enumerate().take(frames) {
                *slot = f0[i * chunk];
            }
            return Ok(());
        }
        if mel > out.len() {
            let n = out.len().min(f0.len());
            out[..n].copy_from_slice(&f0[..n]);
            return Ok(());
        }
        let n = mel.min(f0.len()).min(out.len());
        out[..n].copy_from_slice(&f0[..n]);
        Ok(())
    }
}

impl CpuKernel for F0IfBypassKernel {
    fn name(&self) -> &str {
        F0_IF_BYPASS
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs
            .first()
            .ok_or("F0IfBypass: missing f0 input")?
            .expect_f32("f0")?;
        let out = output.expect_f32_mut("out")?;
        if x.len() == out.len() {
            out.copy_from_slice(x);
            return Ok(());
        }
        // `[1,1,T]` activations feeding a `[1,T]` If stub (same flat length when T matches).
        if x.len() > out.len() && x.len() % out.len() == 0 {
            let chunk = x.len() / out.len();
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = x[i * chunk];
            }
            return Ok(());
        }
        Err(format!(
            "F0IfBypass size {} != {} (squeeze {} -> {})",
            x.len(),
            out.len(),
            x.len(),
            out.len()
        ))
    }
}

impl CpuKernel for ActCopyKernel {
    fn name(&self) -> &str {
        ACT_COPY
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("x")?;
        let out = output.expect_f32_mut("out")?;
        if x.len() != out.len() {
            return Err(format!("ActCopy size {} != {}", x.len(), out.len()));
        }
        out.copy_from_slice(x);
        Ok(())
    }
}

struct QMatMulKernel;

impl CpuKernel for QMatMulKernel {
    fn name(&self) -> &str {
        Q_MATMUL
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 6 {
            return Err(format!("QMatMul expected 6 inputs, got {}", inputs.len()));
        }
        let act_q: &[u8] = match inputs[0].shape().dtype() {
            rlx_ir::DType::U8 => inputs[0].expect_u8("act_q")?,
            rlx_ir::DType::I8 => {
                let i = inputs[0].expect_i8("act_q")?;
                unsafe { std::slice::from_raw_parts(i.as_ptr() as *const u8, i.len()) }
            }
            dt => return Err(format!("act_q: expected U8/I8, got {dt:?}")),
        };
        let act_scale = inputs[1].expect_f32("act_scale")?[0];
        let act_zp = read_zp_u8(&inputs[2], "act_zp")?;
        let act_shape = shape_usize(inputs[0].shape());
        let out = output.expect_f32_mut("out")?;
        if inputs[3].shape().dtype() == rlx_ir::DType::F32 {
            let w_f32 = inputs[3].expect_f32("w_baked")?;
            let w_shape = shape_usize(inputs[3].shape());
            qmatmul_uint8_act_f32_weight_into(
                act_q, &act_shape, act_scale, act_zp, w_f32, &w_shape, out,
            );
            return Ok(());
        }
        let w = inputs[3].expect_i8("w_quantized")?;
        let w_scale = inputs[4].expect_f32("w_scale")?[0];
        let w_zp = read_zp(&inputs[5], "w_zp")?[0];
        let w_shape = shape_usize(inputs[3].shape());
        if crate::qmatmul_gpu::try_qmatmul_uint8_gpu_into(
            act_q, &act_shape, act_scale, act_zp, w, &w_shape, w_scale, w_zp, out,
        ) {
            return Ok(());
        }
        qmatmul_uint8_act_i8_weight_into(
            act_q, &act_shape, act_scale, act_zp, w, &w_shape, w_scale, w_zp, out,
        );
        Ok(())
    }
}

struct QMatMulBakedKernel;

impl CpuKernel for QMatMulBakedKernel {
    fn name(&self) -> &str {
        Q_MATMUL_BAKED
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 4 {
            return Err(format!(
                "QMatMulBaked expected 4 inputs, got {}",
                inputs.len()
            ));
        }
        let act_q: &[u8] = match inputs[0].shape().dtype() {
            rlx_ir::DType::U8 => inputs[0].expect_u8("act_q")?,
            rlx_ir::DType::I8 => {
                let i = inputs[0].expect_i8("act_q")?;
                unsafe { std::slice::from_raw_parts(i.as_ptr() as *const u8, i.len()) }
            }
            dt => return Err(format!("act_q: expected U8/I8, got {dt:?}")),
        };
        let act_scale = inputs[1].expect_f32("act_scale")?[0];
        let act_zp = read_zp_u8(&inputs[2], "act_zp")?;
        let w_f32 = inputs[3].expect_f32("w_baked")?;
        let act_shape = shape_usize(inputs[0].shape());
        let w_shape = shape_usize(inputs[3].shape());
        let out = output.expect_f32_mut("out")?;
        qmatmul_uint8_act_f32_weight_into(
            act_q, &act_shape, act_scale, act_zp, w_f32, &w_shape, out,
        );
        Ok(())
    }
}

struct AlignmentScatterIndicesKernel;

impl CpuKernel for AlignmentScatterIndicesKernel {
    fn name(&self) -> &str {
        ALIGNMENT_SCATTER_INDICES
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 2 {
            return Err(format!(
                "AlignmentScatterIndices expected 2 inputs, got {}",
                inputs.len()
            ));
        }
        let token_ids = inputs[0].expect_i64("token_ids")?;
        let align = inputs[1].expect_i64("align")?;
        let out = output.expect_i64_mut("indices")?;
        let frames = align.first().copied().unwrap_or(0).max(0) as usize;
        crate::alignment::alignment_scatter_index_pairs(token_ids, frames, out);
        Ok(())
    }
}

pub fn register_native_kernels() {
    rlx_cpu::onnx_ref::register_onnx_reference_kernels();
    register_cpu_kernel(Arc::new(DynamicQuantizeLinearExportKernel));
    register_cpu_kernel(Arc::new(ActCopyKernel));
    register_cpu_kernel(Arc::new(F0IfBypassKernel));
    register_cpu_kernel(Arc::new(F0IfSelectKernel));
    register_cpu_kernel(Arc::new(QMatMulKernel));
    register_cpu_kernel(Arc::new(QMatMulBakedKernel));
    register_cpu_kernel(Arc::new(DynamicQuantizeLstmKernel));
    register_cpu_kernel(Arc::new(ScatterNdKernel));
    register_cpu_kernel(Arc::new(ScatterElementsKernel));
    register_cpu_kernel(Arc::new(AlignmentScatterIndicesKernel));
    register_cpu_kernel(Arc::new(RandomNormalLikeKernel));
    register_cpu_kernel(Arc::new(RandomUniformLikeKernel));
    crate::gpu_kernels::register_gpu_kernels();
}

pub fn lstm_attrs_bytes(hidden_size: usize, bidirectional: bool) -> Vec<u8> {
    let mut v = vec![0u8; 8];
    v[0..4].copy_from_slice(&(hidden_size as u32).to_le_bytes());
    v[4] = u8::from(bidirectional);
    v
}

pub fn scatter_elements_attrs_bytes(axis: i32) -> Vec<u8> {
    axis.to_le_bytes().to_vec()
}
