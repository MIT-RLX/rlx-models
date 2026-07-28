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
use crate::kernels::{
    ACT_COPY, ALIGNMENT_SCATTER_INDICES, CONCAT_FROM_SEQUENCE, CONCAT_FROM_SEQUENCE_ONNX,
    DYNAMIC_QUANTIZE_LINEAR, DYNAMIC_QUANTIZE_LSTM, F0_IF_BYPASS, F0_IF_SELECT, F0_NCHW_UNSQUEEZE,
    F0_NEAREST_UPSAMPLE, Q_MATMUL, Q_MATMUL_BAKED, RANDOM_NORMAL_LIKE, RANDOM_UNIFORM_LIKE,
    SCATTER_ELEMENTS, SCATTER_ND,
};
use crate::lstm::{LstmAttrs, dynamic_lstm_f32, dynamic_quantize_lstm};
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
use crate::qmatmul::{
    dynamic_quantize_uint8, qmatmul_uint8_act_f32_weight_into, qmatmul_uint8_act_i8_weight_into,
};
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
use crate::random::{fill_normal_with_opts, fill_uniform_with_opts, rng_options_from_env};
use crate::scatter::{scatter_elements, scatter_nd_inplace};
use rlx_ir::RngBackend;

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

/// Scatter/gather indices at the MLX lazy boundary are often F32-encoded
/// (4 B/elem) even when the IR dtype is I64.
fn decode_i64_indices(bytes: &[u8], shape: &rlx_ir::Shape) -> Result<Vec<i64>, String> {
    let n = shape.num_elements().unwrap_or(0);
    if n == 0 {
        return Ok(Vec::new());
    }
    let es = bytes.len() / n;
    match (shape.dtype(), es) {
        (DType::I64, 8) => Ok(bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        (DType::I32, 4) => Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
            .collect()),
        (DType::F32, 4) | (_, 4) => Ok(bytes
            .chunks_exact(4)
            .map(|c| {
                let f = f32::from_le_bytes(c.try_into().unwrap());
                // Prefer float-encoded integers (Metal f32-uniform arena).
                // Fall back to i32 bitcast when the slot still holds raw
                // integer bits (legacy uploads → denormals like 4e-45 for 3).
                if f.is_finite() && f.abs() < 1.0e7 && (f - f.round()).abs() < 1.0e-3 {
                    f.round() as i64
                } else {
                    i32::from_le_bytes(c.try_into().unwrap()) as i64
                }
            })
            .collect()),
        (DType::I64, _) if bytes.len() >= n * 8 => Ok(bytes
            .chunks_exact(8)
            .take(n)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        _ => Err(format!(
            "indices: cannot decode {} elems from {} bytes (dtype={:?})",
            n,
            bytes.len(),
            shape.dtype()
        )),
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
            let w_zp = read_zp_vec(inputs[n - 3], "W_zp")?;
            let r_scale = typed::<f32>(inputs[n - 2].0, inputs[n - 2].1, DType::F32, "R_scale")?;
            let r_zp = read_zp_vec(inputs[n - 1], "R_zp")?;
            dynamic_quantize_lstm(
                x, x_dims_ref, w, r, b, w_scale, &w_zp, r_scale, &r_zp, attrs, y,
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
    let indices = decode_i64_indices(inputs[1].0, inputs[1].1)?;
    unsafe {
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
        let updates = typed::<f32>(inputs[2].0, inputs[2].1, DType::F32, "updates")?;
        if inputs[0].1.dtype() == DType::F32 {
            let data = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "data")?;
            if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                out.copy_from_slice(data);
            }
        }
        let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].1));
        let indices_shape = shape_for_buffer(indices.len(), &shape_usize(inputs[1].1));
        scatter_nd_inplace(out, &data_shape, &indices, &indices_shape, updates);
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
    let indices = decode_i64_indices(inputs[1].0, inputs[1].1)?;
    unsafe {
        if inputs[0].1.dtype() == DType::I64 {
            let updates = typed::<i64>(inputs[2].0, inputs[2].1, DType::I64, "updates")?;
            let data = typed::<i64>(inputs[0].0, inputs[0].1, DType::I64, "data")?;
            // Compute the scatter in i64, then write into whatever dtype the
            // output arena slot actually is. On the GPU/arena backends (Metal
            // f32-uniform) the importer types this scatter's output F32, so a
            // hard `typed_mut::<i64>` fails ("output: expected I64, got F32").
            let mut res: Vec<i64> = data.to_vec();
            let n = res.len().min(indices.len()).min(updates.len());
            for i in 0..n {
                let j = indices[i].max(0) as usize;
                if j < res.len() {
                    res[j] = updates[i];
                }
            }
            match output.1.dtype() {
                DType::I64 => {
                    let out = typed_mut::<i64>(output.0, output.1, DType::I64, "output")?;
                    let n = res.len().min(out.len());
                    out[..n].copy_from_slice(&res[..n]);
                }
                DType::I32 => {
                    let out = typed_mut::<i32>(output.0, output.1, DType::I32, "output")?;
                    let n = res.len().min(out.len());
                    for k in 0..n {
                        out[k] = res[k] as i32;
                    }
                }
                DType::F32 => {
                    let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
                    let n = res.len().min(out.len());
                    for k in 0..n {
                        out[k] = res[k] as f32;
                    }
                }
                other => {
                    return Err(format!(
                        "ScatterElements: unsupported output dtype {other:?} for i64 data"
                    ));
                }
            }
            let _ = axis;
        } else {
            let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
            let updates = typed::<f32>(inputs[2].0, inputs[2].1, DType::F32, "updates")?;
            if inputs[0].1.dtype() == DType::F32 {
                let data = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "data")?;
                if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
                    out.copy_from_slice(data);
                }
            }
            let data_shape = shape_for_buffer(out.len(), &shape_usize(inputs[0].1));
            scatter_elements(out, &data_shape, &indices, updates, axis);
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
    // Metal/MLX f32-uniform arenas often type i64 duration tensors as F32.
    // Decode flexibly — same strategy as ScatterElements / `decode_i64_indices`.
    let duration_mask = decode_i64_indices(inputs[0].0, inputs[0].1)?;
    let range_ids = decode_i64_indices(inputs[1].0, inputs[1].1)?;
    let split_lens = decode_i64_indices(inputs[2].0, inputs[2].1)?;
    let trip = decode_i64_indices(inputs[3].0, inputs[3].1)?;
    let trip_count = rlx_cpu::onnx_control_flow::resolve_concat_trip_count(
        &trip,
        duration_mask.len(),
        split_lens.len(),
    );
    let out_len = output.1.num_elements().unwrap_or(0).max(1);
    let mut res = vec![0i64; out_len];
    concat_alignment_durations(
        &duration_mask,
        &range_ids,
        &split_lens,
        trip_count,
        &mut res,
    );
    unsafe {
        match output.1.dtype() {
            DType::I64 => {
                let out = typed_mut::<i64>(output.0, output.1, DType::I64, "output")?;
                let n = res.len().min(out.len());
                out[..n].copy_from_slice(&res[..n]);
            }
            DType::F32 => {
                let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
                let n = res.len().min(out.len());
                for k in 0..n {
                    out[k] = res[k] as f32;
                }
            }
            DType::I32 => {
                let out = typed_mut::<i32>(output.0, output.1, DType::I32, "output")?;
                let n = res.len().min(out.len());
                for k in 0..n {
                    out[k] = res[k] as i32;
                }
            }
            other => {
                return Err(format!(
                    "ConcatFromSequence: unsupported output dtype {other:?}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn parse_rng_tag(attrs: &[u8]) -> u64 {
    if attrs.len() >= 16 {
        u64::from_le_bytes(attrs[8..16].try_into().unwrap())
    } else {
        0
    }
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn read_zp_u8_bytes(inp: (&[u8], &rlx_ir::Shape), name: &str) -> Result<u8, String> {
    unsafe {
        match inp.1.dtype() {
            DType::U8 => Ok(typed::<u8>(inp.0, inp.1, DType::U8, name)?[0]),
            DType::I8 => Ok(typed::<i8>(inp.0, inp.1, DType::I8, name)?[0] as u8),
            DType::I32 => Ok(typed::<i32>(inp.0, inp.1, DType::I32, name)?[0].clamp(0, 255) as u8),
            DType::F32 => Ok(typed::<f32>(inp.0, inp.1, DType::F32, name)?[0]
                .round()
                .clamp(0.0, 255.0) as u8),
            dt => Err(format!("{name}: expected zp as U8/I8/I32/F32, got {dt:?}")),
        }
    }
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn read_zp_bytes(inp: (&[u8], &rlx_ir::Shape), name: &str) -> Result<i32, String> {
    Ok(read_zp_vec(inp, name)?[0])
}

fn read_zp_vec(inp: (&[u8], &rlx_ir::Shape), name: &str) -> Result<Vec<i32>, String> {
    unsafe {
        Ok(match inp.1.dtype() {
            DType::I32 => typed::<i32>(inp.0, inp.1, DType::I32, name)?.to_vec(),
            DType::I8 => typed::<i8>(inp.0, inp.1, DType::I8, name)?
                .iter()
                .map(|&x| x as i32)
                .collect(),
            DType::F32 => typed::<f32>(inp.0, inp.1, DType::F32, name)?
                .iter()
                .map(|&x| x.round() as i32)
                .collect(),
            dt => return Err(format!("{name}: expected zp as I32/I8/F32, got {dt:?}")),
        })
    }
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_dynamic_quantize_linear_export(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    attrs: &[u8],
) -> Result<(), String> {
    if inputs.is_empty() {
        return Err("DynamicQuantizeLinearExport expected 1 input".into());
    }
    unsafe {
        let act = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "X")?;
        let (q, scale, zp) = dynamic_quantize_uint8(act);
        let which = attrs.first().copied().unwrap_or(0);
        match which {
            0 => {
                let out = typed_mut::<u8>(output.0, output.1, DType::U8, "quantized")?;
                if out.len() != q.len() {
                    return Err(format!("quantized size {} != {}", out.len(), q.len()));
                }
                out.copy_from_slice(&q);
            }
            1 => {
                let out = typed_mut::<f32>(output.0, output.1, DType::F32, "scale")?;
                out[0] = scale;
            }
            2 => {
                let out = typed_mut::<u8>(output.0, output.1, DType::U8, "zero_point")?;
                out[0] = zp;
            }
            _ => return Err(format!("unknown DQL export slot {which}")),
        }
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_act_copy(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    unsafe {
        let x = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "x")?;
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "out")?;
        if x.len() != out.len() {
            return Err(format!("ActCopy size {} != {}", x.len(), out.len()));
        }
        out.copy_from_slice(x);
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_f0_if_select(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.len() < 2 {
        return Err(format!(
            "F0IfSelect expected 2 inputs, got {}",
            inputs.len()
        ));
    }
    let align = decode_i64_indices(inputs[1].0, inputs[1].1)?;
    let mel_from_align = align.first().copied().unwrap_or(0).max(0) as usize;
    let mel = if mel_from_align > 0 {
        mel_from_align
    } else {
        crate::opts::runtime_mel_frames().unwrap_or(0)
    };
    let valid = mel.saturating_mul(2);
    unsafe {
        let f0 = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "f0")?;
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "out")?;
        if std::env::var("RLX_KITTEN_F0_DEBUG").is_ok() {
            let s: Vec<String> = [0usize, 10, 20, 35, 70, 128, 130, 256, 640, 1000]
                .iter()
                .filter(|&&i| i < f0.len())
                .map(|&i| format!("[{i}]={:.1}", f0[i]))
                .collect();
            eprintln!(
                "[f0if-gpu] mel={mel} mel_cap={:?} wave_cap={:?} valid={valid} f0.len={} out.len={} samples: {}",
                crate::opts::runtime_mel_cap(),
                crate::opts::runtime_wave_cap(),
                f0.len(),
                out.len(),
                s.join(" ")
            );
        }
        out.fill(0.0);
        if f0.len() == out.len() {
            let n = valid.min(f0.len());
            out[..n].copy_from_slice(&f0[..n]);
            return Ok(());
        }
        if f0.len() > out.len() && !out.is_empty() && f0.len() % out.len() == 0 {
            let chunk = f0.len() / out.len();
            let frames = valid.min(out.len());
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
        let n = valid.min(f0.len()).min(out.len());
        out[..n].copy_from_slice(&f0[..n]);
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_f0_nchw_unsqueeze(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.is_empty() {
        return Err("F0NchwUnsqueeze: missing input".into());
    }
    unsafe {
        let x = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "f0")?;
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "out")?;
        out.fill(0.0);
        let n = x.len().min(out.len());
        out[..n].copy_from_slice(&x[..n]);
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_f0_nearest_upsample(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    attrs: &[u8],
) -> Result<(), String> {
    if inputs.is_empty() {
        return Err("F0NearestUpsample: missing input".into());
    }
    let scale = if attrs.len() >= 4 {
        u32::from_le_bytes(attrs[0..4].try_into().unwrap()) as usize
    } else {
        300
    }
    .max(1);
    unsafe {
        let x = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "f0")?;
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "out")?;
        out.fill(0.0);
        let t_out = out.len();
        for (i, &v) in x.iter().enumerate() {
            let base = i.saturating_mul(scale);
            if base >= t_out {
                break;
            }
            for k in 0..scale {
                let j = base + k;
                if j >= t_out {
                    break;
                }
                out[j] = v;
            }
        }
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_f0_if_bypass(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.is_empty() {
        return Err("F0IfBypass: missing f0 input".into());
    }
    unsafe {
        let x = typed::<f32>(inputs[0].0, inputs[0].1, DType::F32, "f0")?;
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "out")?;
        if x.len() == out.len() {
            out.copy_from_slice(x);
            return Ok(());
        }
        if x.len() > out.len() && x.len() % out.len() == 0 {
            let chunk = x.len() / out.len();
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = x[i * chunk];
            }
            return Ok(());
        }
        return Err(format!(
            "F0IfBypass size {} != {} (squeeze {} -> {})",
            x.len(),
            out.len(),
            x.len(),
            out.len()
        ));
    }
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_qmatmul(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.len() < 6 {
        return Err(format!("QMatMul expected 6 inputs, got {}", inputs.len()));
    }
    unsafe {
        let act_q: &[u8] = match inputs[0].1.dtype() {
            DType::U8 => typed::<u8>(inputs[0].0, inputs[0].1, DType::U8, "act_q")?,
            DType::I8 => {
                let i = typed::<i8>(inputs[0].0, inputs[0].1, DType::I8, "act_q")?;
                std::slice::from_raw_parts(i.as_ptr() as *const u8, i.len())
            }
            dt => return Err(format!("act_q: expected U8/I8, got {dt:?}")),
        };
        let act_scale = typed::<f32>(inputs[1].0, inputs[1].1, DType::F32, "act_scale")?[0];
        let act_zp = read_zp_u8_bytes(inputs[2], "act_zp")?;
        let act_shape = shape_for_buffer(act_q.len(), &shape_usize(inputs[0].1));
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "out")?;
        if inputs[3].1.dtype() == DType::F32 {
            let w_f32 = typed::<f32>(inputs[3].0, inputs[3].1, DType::F32, "w_baked")?;
            let w_shape = shape_for_buffer(w_f32.len(), &shape_usize(inputs[3].1));
            qmatmul_uint8_act_f32_weight_into(
                act_q, &act_shape, act_scale, act_zp, w_f32, &w_shape, out,
            );
            return Ok(());
        }
        let w: &[i8] = match inputs[3].1.dtype() {
            DType::I8 => typed::<i8>(inputs[3].0, inputs[3].1, DType::I8, "w_quantized")?,
            DType::U8 => {
                let u = typed::<u8>(inputs[3].0, inputs[3].1, DType::U8, "w_quantized")?;
                std::slice::from_raw_parts(u.as_ptr() as *const i8, u.len())
            }
            dt => return Err(format!("w_quantized: expected I8/U8, got {dt:?}")),
        };
        let w_scale = typed::<f32>(inputs[4].0, inputs[4].1, DType::F32, "w_scale")?[0];
        let w_zp = read_zp_bytes(inputs[5], "w_zp")?;
        let w_shape = shape_for_buffer(w.len(), &shape_usize(inputs[3].1));
        if crate::qmatmul_gpu::try_qmatmul_uint8_gpu_into(
            act_q, &act_shape, act_scale, act_zp, w, &w_shape, w_scale, w_zp, out,
        ) {
            return Ok(());
        }
        qmatmul_uint8_act_i8_weight_into(
            act_q, &act_shape, act_scale, act_zp, w, &w_shape, w_scale, w_zp, out,
        );
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_qmatmul_baked(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.len() < 4 {
        return Err(format!(
            "QMatMulBaked expected 4 inputs, got {}",
            inputs.len()
        ));
    }
    unsafe {
        let act_q: &[u8] = match inputs[0].1.dtype() {
            DType::U8 => typed::<u8>(inputs[0].0, inputs[0].1, DType::U8, "act_q")?,
            DType::I8 => {
                let i = typed::<i8>(inputs[0].0, inputs[0].1, DType::I8, "act_q")?;
                std::slice::from_raw_parts(i.as_ptr() as *const u8, i.len())
            }
            dt => return Err(format!("act_q: expected U8/I8, got {dt:?}")),
        };
        let act_scale = typed::<f32>(inputs[1].0, inputs[1].1, DType::F32, "act_scale")?[0];
        let act_zp = read_zp_u8_bytes(inputs[2], "act_zp")?;
        let w_f32 = typed::<f32>(inputs[3].0, inputs[3].1, DType::F32, "w_baked")?;
        let act_shape = shape_for_buffer(act_q.len(), &shape_usize(inputs[0].1));
        let w_shape = shape_for_buffer(w_f32.len(), &shape_usize(inputs[3].1));
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "out")?;
        qmatmul_uint8_act_f32_weight_into(
            act_q, &act_shape, act_scale, act_zp, w_f32, &w_shape, out,
        );
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_alignment_scatter_indices(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
    _attrs: &[u8],
) -> Result<(), String> {
    if inputs.len() < 2 {
        return Err(format!(
            "AlignmentScatterIndices expected 2 inputs, got {}",
            inputs.len()
        ));
    }
    let token_ids = decode_i64_indices(inputs[0].0, inputs[0].1)?;
    let align = decode_i64_indices(inputs[1].0, inputs[1].1)?;
    let frames = align.first().copied().unwrap_or(0).max(0) as usize;
    let mut res = vec![0i64; output.1.num_elements().unwrap_or(0).max(1)];
    crate::alignment::alignment_scatter_index_pairs(&token_ids, frames, &mut res);
    unsafe {
        match output.1.dtype() {
            DType::I64 => {
                let out = typed_mut::<i64>(output.0, output.1, DType::I64, "indices")?;
                let n = res.len().min(out.len());
                out[..n].copy_from_slice(&res[..n]);
            }
            DType::F32 => {
                let out = typed_mut::<f32>(output.0, output.1, DType::F32, "indices")?;
                let n = res.len().min(out.len());
                for k in 0..n {
                    out[k] = res[k] as f32;
                }
            }
            DType::I32 => {
                let out = typed_mut::<i32>(output.0, output.1, DType::I32, "indices")?;
                let n = res.len().min(out.len());
                for k in 0..n {
                    out[k] = res[k] as i32;
                }
            }
            other => {
                return Err(format!(
                    "AlignmentScatterIndices: unsupported output dtype {other:?}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_random_normal_like(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
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
    let tag = parse_rng_tag(attrs);
    unsafe {
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
        let opts = rng_options_from_env();
        if matches!(opts.backend, RngBackend::Zero) {
            out.fill(0.0);
        } else {
            fill_normal_with_opts(out, mean, scale, opts, tag);
        }
    }
    Ok(())
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "mlx", target_os = "macos")
))]
fn run_random_uniform_like(
    inputs: &[(&[u8], &rlx_ir::Shape)],
    output: (&mut [u8], &rlx_ir::Shape),
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
    let tag = parse_rng_tag(attrs);
    unsafe {
        let out = typed_mut::<f32>(output.0, output.1, DType::F32, "output")?;
        let opts = rng_options_from_env();
        if matches!(opts.backend, RngBackend::Zero) {
            out.fill(0.0);
        } else {
            fill_uniform_with_opts(out, low, high, opts, tag);
        }
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
        AlignmentScatterIndicesMetal,
        ALIGNMENT_SCATTER_INDICES,
        run_alignment_scatter_indices
    );
    metal_kernel!(
        ConcatFromSequenceMetal,
        CONCAT_FROM_SEQUENCE,
        run_concat_from_sequence
    );
    metal_kernel!(
        ConcatFromSequenceOnnxMetal,
        CONCAT_FROM_SEQUENCE_ONNX,
        run_concat_from_sequence
    );
    metal_kernel!(
        DynamicQuantizeLinearExportMetal,
        DYNAMIC_QUANTIZE_LINEAR,
        run_dynamic_quantize_linear_export
    );
    metal_kernel!(ActCopyMetal, ACT_COPY, run_act_copy);
    metal_kernel!(F0IfBypassMetal, F0_IF_BYPASS, run_f0_if_bypass);
    metal_kernel!(F0IfSelectMetal, F0_IF_SELECT, run_f0_if_select);
    metal_kernel!(
        F0NearestUpsampleMetal,
        F0_NEAREST_UPSAMPLE,
        run_f0_nearest_upsample
    );
    metal_kernel!(
        F0NchwUnsqueezeMetal,
        F0_NCHW_UNSQUEEZE,
        run_f0_nchw_unsqueeze
    );
    metal_kernel!(QMatMulMetal, Q_MATMUL, run_qmatmul);
    metal_kernel!(QMatMulBakedMetal, Q_MATMUL_BAKED, run_qmatmul_baked);
    metal_kernel!(
        RandomNormalLikeMetal,
        RANDOM_NORMAL_LIKE,
        run_random_normal_like
    );
    metal_kernel!(
        RandomUniformLikeMetal,
        RANDOM_UNIFORM_LIKE,
        run_random_uniform_like
    );

    pub fn register() {
        register_metal_kernel(Arc::new(DynamicQuantizeLstmMetal));
        register_metal_kernel(Arc::new(ScatterNdMetal));
        register_metal_kernel(Arc::new(ScatterElementsMetal));
        register_metal_kernel(Arc::new(AlignmentScatterIndicesMetal));
        register_metal_kernel(Arc::new(ConcatFromSequenceMetal));
        register_metal_kernel(Arc::new(ConcatFromSequenceOnnxMetal));
        register_metal_kernel(Arc::new(DynamicQuantizeLinearExportMetal));
        register_metal_kernel(Arc::new(ActCopyMetal));
        register_metal_kernel(Arc::new(F0IfBypassMetal));
        register_metal_kernel(Arc::new(F0IfSelectMetal));
        register_metal_kernel(Arc::new(F0NearestUpsampleMetal));
        register_metal_kernel(Arc::new(F0NchwUnsqueezeMetal));
        register_metal_kernel(Arc::new(QMatMulMetal));
        register_metal_kernel(Arc::new(QMatMulBakedMetal));
        register_metal_kernel(Arc::new(RandomNormalLikeMetal));
        register_metal_kernel(Arc::new(RandomUniformLikeMetal));
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
            1 => DType::I8,
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
        let out_nelems = out_dims.iter().product::<usize>().max(1);
        let out_bytes = out_nelems * output_shape.dtype().size_bytes();
        let mut out_buf = vec![0u8; out_bytes];
        let out_shape = Shape::new(&out_dims, output_shape.dtype());
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
        AlignmentScatterIndicesMlx,
        ALIGNMENT_SCATTER_INDICES,
        run_alignment_scatter_indices
    );
    mlx_kernel!(
        ConcatFromSequenceMlx,
        CONCAT_FROM_SEQUENCE,
        run_concat_from_sequence
    );
    mlx_kernel!(
        ConcatFromSequenceOnnxMlx,
        CONCAT_FROM_SEQUENCE_ONNX,
        run_concat_from_sequence
    );
    mlx_kernel!(
        DynamicQuantizeLinearExportMlx,
        DYNAMIC_QUANTIZE_LINEAR,
        run_dynamic_quantize_linear_export
    );
    mlx_kernel!(ActCopyMlx, ACT_COPY, run_act_copy);
    mlx_kernel!(F0IfBypassMlx, F0_IF_BYPASS, run_f0_if_bypass);
    mlx_kernel!(F0IfSelectMlx, F0_IF_SELECT, run_f0_if_select);
    mlx_kernel!(
        F0NearestUpsampleMlx,
        F0_NEAREST_UPSAMPLE,
        run_f0_nearest_upsample
    );
    mlx_kernel!(F0NchwUnsqueezeMlx, F0_NCHW_UNSQUEEZE, run_f0_nchw_unsqueeze);
    mlx_kernel!(QMatMulMlx, Q_MATMUL, run_qmatmul);
    mlx_kernel!(QMatMulBakedMlx, Q_MATMUL_BAKED, run_qmatmul_baked);
    mlx_kernel!(
        RandomNormalLikeMlx,
        RANDOM_NORMAL_LIKE,
        run_random_normal_like
    );
    mlx_kernel!(
        RandomUniformLikeMlx,
        RANDOM_UNIFORM_LIKE,
        run_random_uniform_like
    );

    pub fn register() {
        register_mlx_kernel(Arc::new(DynamicQuantizeLstmMlx));
        register_mlx_kernel(Arc::new(ScatterNdMlx));
        register_mlx_kernel(Arc::new(ScatterElementsMlx));
        register_mlx_kernel(Arc::new(AlignmentScatterIndicesMlx));
        register_mlx_kernel(Arc::new(ConcatFromSequenceMlx));
        register_mlx_kernel(Arc::new(ConcatFromSequenceOnnxMlx));
        register_mlx_kernel(Arc::new(DynamicQuantizeLinearExportMlx));
        register_mlx_kernel(Arc::new(ActCopyMlx));
        register_mlx_kernel(Arc::new(F0IfBypassMlx));
        register_mlx_kernel(Arc::new(F0IfSelectMlx));
        register_mlx_kernel(Arc::new(F0NearestUpsampleMlx));
        register_mlx_kernel(Arc::new(F0NchwUnsqueezeMlx));
        register_mlx_kernel(Arc::new(QMatMulMlx));
        register_mlx_kernel(Arc::new(QMatMulBakedMlx));
        register_mlx_kernel(Arc::new(RandomNormalLikeMlx));
        register_mlx_kernel(Arc::new(RandomUniformLikeMlx));
    }
}

pub fn register_gpu_kernels() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    metal::register();
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    mlx::register();
    #[cfg(feature = "cuda")]
    cuda::register();
    // WGPU/Vulkan: custom kitten ops dispatch via the shared CPU kernel registry
    // during host segments; no separate wgpu shader pack yet.
}

/// Cuda raw-GPU customs (no D2H). Takes precedence over the CPU host kernels.
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use rlx_cuda::cuda_gpu_kernels::{CudaGpuKernel, register_cuda_gpu_kernel};
    use rlx_ir::Shape;
    use std::sync::Arc;

    /// Rank-3 InstanceNorm over the active time window only (AdaIN).
    ///
    /// Inputs: X `[N,C,T]`, gamma `[C]`, beta `[C]`.
    /// Extras: `e0=N, e1=C, e2=T, e3=active` (resolved at launch from Kitten opts).
    ///
    /// Launch: **one block per (n,c) row**; threads cooperate on mean/var over
    /// `active`, then write all `T` in parallel. The old one-thread-per-row
    /// serial scan was ~5 ms/call on large generator axes.
    const INORM_CU: &str = r#"
extern "C" __global__ void rlx_custom(
    float* arena,
    unsigned out_off, unsigned out_len, unsigned n_inputs,
    unsigned in0_off, unsigned in0_len,
    unsigned in1_off, unsigned in1_len,
    unsigned in2_off, unsigned in2_len,
    unsigned in3_off, unsigned in3_len,
    unsigned e0, unsigned e1, unsigned e2, unsigned e3)
{
    (void)n_inputs; (void)in0_len; (void)in1_len; (void)in2_len;
    (void)in3_off; (void)in3_len; (void)out_len;
    unsigned N = e0;
    unsigned C = e1;
    unsigned T = e2;
    unsigned active = e3;
    if (N == 0 || C == 0 || T == 0 || active == 0) return;
    if (active > T) active = T;
    unsigned row = blockIdx.x;
    unsigned rows = N * C;
    if (row >= rows) return;
    unsigned ni = row / C;
    unsigned ci = row % C;
    unsigned base = (ni * C + ci) * T;
    float* x = arena + in0_off + base;
    float* y = arena + out_off + base;
    unsigned tid = threadIdx.x;
    unsigned bdim = blockDim.x;
    __shared__ float smem[256];

    // Mean over active (strided + block reduce).
    float partial = 0.f;
    for (unsigned j = tid; j < active; j += bdim) partial += x[j];
    smem[tid] = partial;
    __syncthreads();
    for (unsigned s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)active;
    __syncthreads();

    // Variance over active.
    partial = 0.f;
    for (unsigned j = tid; j < active; j += bdim) {
        float d = x[j] - mean;
        partial += d * d;
    }
    smem[tid] = partial;
    __syncthreads();
    for (unsigned s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / (float)active + 1e-5f);
    float g = arena[in1_off + ci];
    float b = arena[in2_off + ci];
    // Apply active-window stats to the full axis (matches CPU kernel).
    for (unsigned j = tid; j < T; j += bdim) {
        y[j] = (x[j] - mean) * inv * g + b;
    }
}
"#;

    #[derive(Debug)]
    struct KittenInstanceNormActiveCuda;

    impl CudaGpuKernel for KittenInstanceNormActiveCuda {
        fn name(&self) -> &str {
            crate::kernels::KITTEN_INSTANCE_NORM_ACTIVE
        }
        fn cuda_c(&self) -> &str {
            INORM_CU
        }
        fn block_size(&self) -> u32 {
            256
        }
        fn extras(&self, attrs: &[u8], out_shape: &Shape) -> [u32; 4] {
            let dims = out_shape.dims();
            if dims.len() != 3 {
                return [0, 0, 0, 0];
            }
            let n = dims[0].unwrap_static().max(1) as u32;
            let c = dims[1].unwrap_static().max(1) as u32;
            let t = dims[2].unwrap_static() as u32;
            let is_generator = attrs.get(4).copied().unwrap_or(0) != 0;
            let active =
                crate::kernels::instance_norm_resolve_active(t as usize, is_generator) as u32;
            [n, c, t, active.max(1)]
        }
        fn launch_elems(&self, _out_len: u32, extras: [u32; 4]) -> u32 {
            // One cooperative block per (n, c) row.
            extras[0].saturating_mul(extras[1]).max(1)
        }
        fn grid_blocks(&self, launch_elems: u32, _block_size: u32) -> u32 {
            launch_elems.max(1)
        }
    }

    pub fn register() {
        register_cuda_gpu_kernel(Arc::new(KittenInstanceNormActiveCuda));
        // Active-seq clamp for on-device DynamicQuantizeLSTM (matches CPU kernel).
        rlx_cuda::dyn_quant_lstm_gpu::set_seq_resolver(kitten_lstm_active_seq);
    }

    fn kitten_lstm_active_seq(raw_seq: usize, input_size: usize) -> usize {
        let runtime_seq = crate::opts::compile_sequence_length_from_env().filter(|&n| n > 0);
        let runtime_mel = crate::opts::runtime_mel_frames();
        let mut seq = raw_seq;
        if let Some(rt) = runtime_seq {
            if input_size == 640 && raw_seq > rt {
                seq = runtime_mel.map_or(raw_seq, |mel| mel.min(raw_seq));
            } else {
                let active = crate::opts::runtime_active_tokens()
                    .filter(|&a| a > 0)
                    .unwrap_or(rt);
                seq = seq.min(active);
            }
        } else if let Some(mel) = runtime_mel {
            if input_size == 640 {
                seq = mel.min(seq);
            }
        }
        seq.clamp(1, raw_seq.max(1))
    }
}
