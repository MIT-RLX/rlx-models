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

//! Metal / MLX host delegates for Kitten custom ONNX ops (unified memory).

#![cfg_attr(
    not(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "mlx", target_os = "macos")
    )),
    allow(dead_code, unused_imports)
)]

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
use std::sync::Arc;

use rlx_ir::DType;

use crate::alignment::concat_alignment_durations;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
use crate::kernels::{CONCAT_FROM_SEQUENCE, DYNAMIC_QUANTIZE_LSTM, SCATTER_ELEMENTS, SCATTER_ND};
use crate::lstm::{LstmAttrs, dynamic_lstm_f32, dynamic_quantize_lstm};
use crate::scatter::{scatter_elements, scatter_nd_inplace};

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

unsafe fn typed<'a, T: Copy>(
    bytes: &'a [u8],
    shape: &rlx_ir::Shape,
    want: DType,
    role: &str,
) -> Result<&'a [T], String> {
    if shape.dtype() != want {
        return Err(format!(
            "{role}: expected {want:?}, got {:?}",
            shape.dtype()
        ));
    }
    let n = shape.num_elements().unwrap_or(0);
    let want_bytes = n * std::mem::size_of::<T>();
    if bytes.len() < want_bytes {
        return Err(format!(
            "{role}: buffer too small (have {}, want {want_bytes})",
            bytes.len()
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), n) })
}

unsafe fn typed_mut<'a, T: Copy>(
    bytes: &'a mut [u8],
    shape: &rlx_ir::Shape,
    want: DType,
    role: &str,
) -> Result<&'a mut [T], String> {
    if shape.dtype() != want {
        return Err(format!(
            "{role}: expected {want:?}, got {:?}",
            shape.dtype()
        ));
    }
    let n = shape.num_elements().unwrap_or(0);
    let want_bytes = n * std::mem::size_of::<T>();
    if bytes.len() < want_bytes {
        return Err(format!(
            "{role}: buffer too small (have {}, want {want_bytes})",
            bytes.len()
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), n) })
}

fn run_dynamic_quantize_lstm(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    attrs: &[u8],
) -> Result<(), String> {
    if inputs.len() < 8 {
        return Err(format!("expected at least 8 inputs, got {}", inputs.len()));
    }
    let ka = parse_lstm_attrs(attrs);
    unsafe {
        let x = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "X")?;
        let b = typed::<f32>(inputs[3].0, inputs[3].1, DType::F32, "B")?;
        let y = typed_mut::<f32>(output.0, output.1, DType::F32, "Y")?;
        let x_dims = shape_usize(inputs[0].1);
        let y_dims = shape_usize(output.1);
        let x_dims_buf: Option<[usize; 3]> = if x_dims.len() == 3 && x_dims.iter().all(|&d| d > 0) {
            let (seq, batch) = if x_dims[0] >= x_dims[1] {
                (x_dims[0], x_dims[1].max(1))
            } else {
                (x_dims[1], x_dims[0].max(1))
            };
            let denom = seq * batch;
            let input_size = x.len().checked_div(denom).unwrap_or(0);
            Some([seq, batch, input_size])
        } else if y_dims.len() == 4 && y_dims.iter().all(|&d| d > 0) {
            let (seq, batch) = if y_dims[0] >= y_dims[2] {
                (y_dims[0], y_dims[2].max(1))
            } else {
                (y_dims[2], y_dims[0].max(1))
            };
            let denom = seq * batch;
            let input_size = x.len().checked_div(denom).unwrap_or(0);
            Some([seq, batch, input_size])
        } else {
            let dirs = if ka.bidirectional { 2 } else { 1 };
            let h = ka.hidden_size.max(1);
            let batch = 1usize;
            let seq = y.len() / (dirs * batch * h);
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
        if inputs[1].1.dtype() == DType::I8 {
            let w = typed::<i8>(inputs[1].0, inputs[1].1, DType::I8, "W")?;
            let r = typed::<i8>(inputs[2].0, inputs[2].1, DType::I8, "R")?;
            let n = inputs.len();
            let w_scale = typed::<f32>(inputs[n - 4].0, inputs[n - 4].1, DType::F32, "W_scale")?;
            let w_zp = typed::<i32>(inputs[n - 3].0, inputs[n - 3].1, DType::I32, "W_zp")?;
            let r_scale = typed::<f32>(inputs[n - 2].0, inputs[n - 2].1, DType::F32, "R_scale")?;
            let r_zp = typed::<i32>(inputs[n - 1].0, inputs[n - 1].1, DType::I32, "R_zp")?;
            dynamic_quantize_lstm(
                x, x_dims_ref, w, r, b, w_scale, w_zp, r_scale, r_zp, attrs, y,
            )
        } else {
            let w = typed::<f32>(inputs[1].0, inputs[1].1, DType::F32, "W")?;
            let r = typed::<f32>(inputs[2].0, inputs[2].1, DType::F32, "R")?;
            dynamic_lstm_f32(x, x_dims_ref, w, r, b, attrs, y)
        }
    }
}

fn run_scatter_nd(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.len() < 3 {
        return Err(format!("expected 3 inputs, got {}", inputs.len()));
    }
    unsafe {
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
        let indices = typed::<i64>(inputs[1].0, inputs[1].1, DType::I64, "indices")?;
        let updates = typed::<f32>(inputs[2].0, inputs[2].1, DType::F32, "updates")?;
        if inputs[0].1.dtype() == DType::F32 {
            let data = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "data")?;
            if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                out.copy_from_slice(data);
            }
        }
        let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].1));
        let indices_shape = shape_for_buffer(indices.len(), &shape_usize(inputs[1].1));
        scatter_nd_inplace(out, &data_shape, indices, &indices_shape, updates);
    }
    Ok(())
}

fn run_scatter_elements(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
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
    unsafe {
        if inputs[0].1.dtype() == DType::I64 {
            let out = typed_mut::<i64>(output.0, output.1, DType::I64, "output")?;
            let indices = typed::<i64>(inputs[1].0, inputs[1].1, DType::I64, "indices")?;
            let updates = typed::<i64>(inputs[2].0, inputs[2].1, DType::I64, "updates")?;
            if inputs[0].1.dtype() == DType::I64 {
                let data = typed::<i64>(inputs[0].0, inputs[0].1, DType::I64, "data")?;
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
            let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
            let indices = typed::<i64>(inputs[1].0, inputs[1].1, DType::I64, "indices")?;
            let updates = typed::<f32>(inputs[2].0, inputs[2].1, DType::F32, "updates")?;
            if inputs[0].1.dtype() == DType::F32 {
                let data = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "data")?;
                if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                    out.copy_from_slice(data);
                }
            }
            let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].1));
            scatter_elements(out, &data_shape, indices, updates, axis);
        }
    }
    Ok(())
}

fn run_concat_from_sequence(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.len() < 4 {
        return Err(format!("expected 4 inputs, got {}", inputs.len()));
    }
    unsafe {
        let duration_mask = typed::<i64>(inputs[0].0, inputs[0].1, DType::I64, "duration_mask")?;
        let range_ids = typed::<i64>(inputs[1].0, inputs[1].1, DType::I64, "range_ids")?;
        let split_lens = typed::<i64>(inputs[2].0, inputs[2].1, DType::I64, "split_lens")?;
        let trip = typed::<i64>(inputs[3].0, inputs[3].1, DType::I64, "trip_count")?;
        let out = typed_mut::<i64>(output.0, output.1, DType::I64, "output")?;
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
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use rlx_metal::op_registry::{MetalKernel, register_metal_kernel};

    macro_rules! metal_kernel {
        ($struct:ident, $op_name:expr, $run:ident) => {
            #[derive(Debug)]
            struct $struct;
            impl MetalKernel for $struct {
                fn name(&self) -> &str {
                    $op_name
                }
                fn execute(
                    &self,
                    inputs: &[(&[u8], &rlx_ir::Shape)],
                    output: (&mut [u8], &rlx_ir::Shape),
                    attrs: &[u8],
                ) -> Result<(), String> {
                    $run(inputs, output, attrs)
                }
            }
        };
    }

    metal_kernel!(
        DynamicQuantizeLstmMetal,
        DYNAMIC_QUANTIZE_LSTM,
        run_dynamic_quantize_lstm
    );
    metal_kernel!(ScatterNdMetal, SCATTER_ND, run_scatter_nd);
    metal_kernel!(ScatterElementsMetal, SCATTER_ELEMENTS, run_scatter_elements);
    metal_kernel!(
        ConcatFromSequenceMetal,
        CONCAT_FROM_SEQUENCE,
        run_concat_from_sequence
    );

    pub fn register() {
        register_metal_kernel(Arc::new(DynamicQuantizeLstmMetal));
        register_metal_kernel(Arc::new(ScatterNdMetal));
        register_metal_kernel(Arc::new(ScatterElementsMetal));
        register_metal_kernel(Arc::new(ConcatFromSequenceMetal));
    }
}

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod mlx {
    use super::*;
    use rlx_ir::Shape;
    use rlx_mlx::array::{Array, MlxError};
    use rlx_mlx::op_registry::{MlxKernel, register_mlx_kernel};

    fn shape_dims(shape: &rlx_ir::Shape) -> Result<Vec<usize>, MlxError> {
        let mut out = Vec::with_capacity(shape.rank());
        for d in shape.dims() {
            match d {
                rlx_ir::Dim::Static(n) => out.push(*n),
                rlx_ir::Dim::Dynamic(_) => {
                    return Err(MlxError(
                        "dynamic dims unsupported in kitten mlx kernel".into(),
                    ));
                }
            }
        }
        Ok(out)
    }

    fn dtype_from_bytes(len: usize, nelems: usize) -> DType {
        let es = if nelems > 0 { len / nelems } else { 4 };
        match es {
            1 => DType::U8,
            4 => DType::F32,
            8 => DType::I64,
            _ => DType::F32,
        }
    }

    fn run_host<F>(
        inputs: &[&Array],
        output_shape: &rlx_ir::Shape,
        attrs: &[u8],
        run: F,
    ) -> Result<Array, MlxError>
    where
        F: FnOnce(
            &[(&[u8], &rlx_ir::Shape)],
            (&mut [u8], &rlx_ir::Shape),
            &[u8],
        ) -> Result<(), String>,
    {
        let mut owned = Vec::with_capacity(inputs.len());
        let mut views = Vec::with_capacity(inputs.len());
        for arr in inputs {
            let dims = arr.shape()?;
            let bytes = arr.to_bytes()?;
            let nelems = arr.num_elements()?;
            let dtype = dtype_from_bytes(bytes.len(), nelems);
            owned.push((bytes, Shape::new(&dims, dtype)));
        }
        for (bytes, shape) in &owned {
            views.push((bytes.as_slice(), shape));
        }
        let out_dims = shape_dims(output_shape)?;
        let out_nelems = output_shape.num_elements().unwrap_or(1).max(1);
        let out_bytes = out_nelems * output_shape.dtype().size_bytes();
        let mut out_buf = vec![0u8; out_bytes];
        let out_shape = output_shape.clone();
        run(&views, (&mut out_buf, &out_shape), attrs).map_err(MlxError)?;
        Array::from_bytes(&out_buf, &out_dims, output_shape.dtype())
    }

    macro_rules! mlx_kernel {
        ($struct:ident, $op_name:expr, $run:ident) => {
            #[derive(Debug)]
            struct $struct;
            impl MlxKernel for $struct {
                fn name(&self) -> &str {
                    $op_name
                }
                fn execute(
                    &self,
                    inputs: &[&Array],
                    output_shape: &rlx_ir::Shape,
                    attrs: &[u8],
                ) -> Result<Array, MlxError> {
                    run_host(inputs, output_shape, attrs, $run)
                }
            }
        };
    }

    mlx_kernel!(
        DynamicQuantizeLstmMlx,
        DYNAMIC_QUANTIZE_LSTM,
        run_dynamic_quantize_lstm
    );
    mlx_kernel!(ScatterNdMlx, SCATTER_ND, run_scatter_nd);
    mlx_kernel!(ScatterElementsMlx, SCATTER_ELEMENTS, run_scatter_elements);
    mlx_kernel!(
        ConcatFromSequenceMlx,
        CONCAT_FROM_SEQUENCE,
        run_concat_from_sequence
    );

    pub fn register() {
        register_mlx_kernel(Arc::new(DynamicQuantizeLstmMlx));
        register_mlx_kernel(Arc::new(ScatterNdMlx));
        register_mlx_kernel(Arc::new(ScatterElementsMlx));
        register_mlx_kernel(Arc::new(ConcatFromSequenceMlx));
    }
}

pub fn register_gpu_kernels() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    metal::register();
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    mlx::register();
}
