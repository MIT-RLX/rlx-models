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

use crate::alignment::concat_alignment_durations;
use crate::lstm::{LstmAttrs, dynamic_lstm_f32, dynamic_quantize_lstm};
use crate::qmatmul::{dynamic_quantize_uint8, qmatmul_uint8_act_i8_weight};
use crate::random::{fill_normal, fill_uniform, normal_seed, uniform_seed};
use crate::scatter::{scatter_elements, scatter_nd_inplace};

pub const DYNAMIC_QUANTIZE_LSTM: &str = "onnx.DynamicQuantizeLSTM";
pub const SCATTER_ND: &str = "onnx.ScatterND";
pub const SCATTER_ELEMENTS: &str = "onnx.ScatterElements";
pub const CONCAT_FROM_SEQUENCE: &str = "onnx.KittenConcatFromSequence";
/// Generic import lowering name (same kernel).
pub const CONCAT_FROM_SEQUENCE_ONNX: &str = "onnx.ConcatFromSequence";
pub const RANDOM_NORMAL_LIKE: &str = "onnx.RandomNormalLike";
pub const RANDOM_UNIFORM_LIKE: &str = "onnx.RandomUniformLike";
pub const Q_MATMUL: &str = "onnx.QMatMul";
pub const ACT_COPY: &str = "onnx.ActCopy";
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
                out.copy_from_slice(data);
            }
        }
        let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].shape()));
        let indices_shape = shape_for_buffer(indices.len(), &shape_usize(inputs[1].shape()));
        scatter_nd_inplace(out, &data_shape, indices, &indices_shape, updates);
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
                    out.copy_from_slice(data);
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
        if std::env::var("KITTEN_RLX_RNG_SEED").is_ok() {
            fill_normal(out, mean, scale, normal_seed(tag));
        } else {
            out.fill(0.0);
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
        if std::env::var("KITTEN_RLX_RNG_SEED").is_ok() {
            fill_uniform(out, low, high, uniform_seed(tag));
        } else {
            out.fill(0.0);
        }
        Ok(())
    }
}

struct ConcatFromSequenceKernel;

impl ConcatFromSequenceKernel {
    fn run(inputs: &[CpuTensorRef<'_>], output: CpuTensorMut<'_>) -> Result<(), String> {
        if inputs.len() < 4 {
            return Err(format!("expected 4 inputs, got {}", inputs.len()));
        }
        let duration_mask = inputs[0].expect_i64("duration_mask")?;
        let range_ids = inputs[1].expect_i64("range_ids")?;
        let split_lens = inputs[2].expect_i64("split_lens")?;
        let trip = inputs[3].expect_i64("trip_count")?;
        let out = output.expect_i64_mut("output")?;
        let mut trip_count = trip.first().copied().unwrap_or(0).max(0) as usize;
        if trip_count <= 1 {
            if let Some(n) = crate::opts::compile_sequence_length_from_env() {
                if n > 1 {
                    trip_count = n;
                }
            }
        }
        trip_count = trip_count.min(out.len()).min(256);
        out.fill(0);
        concat_alignment_durations(duration_mask, range_ids, split_lens, trip_count, out);
        Ok(())
    }
}

impl CpuKernel for ConcatFromSequenceKernel {
    fn name(&self) -> &str {
        CONCAT_FROM_SEQUENCE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        Self::run(inputs, output)
    }
}

struct ConcatFromSequenceOnnxAlias;

impl CpuKernel for ConcatFromSequenceOnnxAlias {
    fn name(&self) -> &str {
        CONCAT_FROM_SEQUENCE_ONNX
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        ConcatFromSequenceKernel::run(inputs, output)
    }
}

struct DynamicQuantizeLinearExportKernel;

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
        let (q, scale, zp) = dynamic_quantize_uint8(act);
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
        let act_q = inputs[0].expect_u8("act_q")?;
        let act_scale = inputs[1].expect_f32("act_scale")?[0];
        let act_zp = read_zp_u8(&inputs[2], "act_zp")?;
        let w = inputs[3].expect_i8("w_quantized")?;
        let w_scale = inputs[4].expect_f32("w_scale")?[0];
        let w_zp = read_zp(&inputs[5], "w_zp")?[0];
        let act_shape = shape_usize(inputs[0].shape());
        let w_shape = shape_usize(inputs[3].shape());
        let out = output.expect_f32_mut("out")?;
        let vals = qmatmul_uint8_act_i8_weight(
            act_q, &act_shape, act_scale, act_zp, w, &w_shape, w_scale, w_zp,
        );
        if vals.len() != out.len() {
            return Err(format!(
                "QMatMul size mismatch: computed {} vs output {}",
                vals.len(),
                out.len()
            ));
        }
        out.copy_from_slice(&vals);
        Ok(())
    }
}

pub fn register_native_kernels() {
    register_cpu_kernel(Arc::new(DynamicQuantizeLinearExportKernel));
    register_cpu_kernel(Arc::new(ActCopyKernel));
    register_cpu_kernel(Arc::new(QMatMulKernel));
    register_cpu_kernel(Arc::new(DynamicQuantizeLstmKernel));
    register_cpu_kernel(Arc::new(ScatterNdKernel));
    register_cpu_kernel(Arc::new(ScatterElementsKernel));
    register_cpu_kernel(Arc::new(ConcatFromSequenceKernel));
    register_cpu_kernel(Arc::new(ConcatFromSequenceOnnxAlias));
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
