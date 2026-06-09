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

//! ONNX ScatterND / ScatterElements reference kernels.

fn flat_offset(indices: &[usize], strides: &[usize]) -> usize {
    let mut off = 0usize;
    for (i, &s) in strides.iter().enumerate() {
        off += indices.get(i).copied().unwrap_or(0) * s;
    }
    off
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i].max(1));
    }
    strides
}

fn shape_from_buffer(buf_len: usize, shape: &[usize]) -> Vec<usize> {
    let want: usize = shape.iter().product::<usize>().max(1);
    if want == buf_len {
        return shape.to_vec();
    }
    vec![buf_len.max(1)]
}

pub fn scatter_nd_inplace(
    out: &mut [f32],
    data_shape: &[usize],
    indices: &[i64],
    indices_shape: &[usize],
    updates: &[f32],
) {
    if out.is_empty() || indices.is_empty() {
        return;
    }
    let data_shape = shape_from_buffer(out.len(), data_shape);
    let index_depth = indices_shape
        .last()
        .copied()
        .filter(|&d| d > 0)
        .unwrap_or(1);
    if !indices.len().is_multiple_of(index_depth) {
        return;
    }
    let num_updates = indices.len() / index_depth;
    let data_strides = row_major_strides(&data_shape);
    let update_elems = updates.len().checked_div(num_updates).unwrap_or(0);
    for u in 0..num_updates {
        let base = u * index_depth;
        if base + index_depth > indices.len() {
            break;
        }
        let mut idx = vec![0usize; data_shape.len()];
        for (j, slot) in idx.iter_mut().enumerate().take(index_depth) {
            *slot = indices[base + j].max(0) as usize;
        }
        let dst = flat_offset(&idx, &data_strides);
        let up_off = u * update_elems;
        for k in 0..update_elems {
            let p = dst.saturating_add(k);
            if p < out.len() && up_off + k < updates.len() {
                out[p] = updates[up_off + k];
            }
        }
    }
}

pub fn scatter_elements_i64(
    data: &[i64],
    data_shape: &[usize],
    indices: &[i64],
    updates: &[i64],
    axis: i32,
    out: &mut [i64],
) {
    let _ = data;
    if out.is_empty() {
        return;
    }
    let data_shape = shape_from_buffer(out.len(), data_shape);
    if data_shape.len() == 1 {
        for (i, &idx) in indices.iter().enumerate() {
            let j = idx.max(0) as usize;
            if j < out.len() && i < updates.len() {
                out[j] = updates[i];
            }
        }
        return;
    }
    let rank = data_shape.len();
    let axis = if axis < 0 { rank as i32 + axis } else { axis } as usize;
    let axis = axis.min(rank.saturating_sub(1));
    let outer: usize = data_shape[..axis].iter().product::<usize>();
    let inner: usize = data_shape[axis..].iter().skip(1).product::<usize>().max(1);
    let axis_dim = data_shape.get(axis).copied().unwrap_or(1);
    for o in 0..outer {
        for i in 0..inner {
            let flat_i = o * axis_dim * inner + i;
            if flat_i >= indices.len() {
                continue;
            }
            let row = indices[flat_i].max(0) as usize;
            let dst = o * axis_dim * inner + row.min(axis_dim.saturating_sub(1)) * inner + i;
            if dst < out.len() && flat_i < updates.len() {
                out[dst] = updates[flat_i];
            }
        }
    }
}

pub fn scatter_elements(
    out: &mut [f32],
    data_shape: &[usize],
    indices: &[i64],
    updates: &[f32],
    axis: i32,
) {
    if out.is_empty() {
        return;
    }
    let data_shape = shape_from_buffer(out.len(), data_shape);
    if data_shape.len() == 1 {
        for (i, &idx) in indices.iter().enumerate() {
            let j = idx.max(0) as usize;
            if j < out.len() && i < updates.len() {
                out[j] = updates[i];
            }
        }
        return;
    }
    let rank = data_shape.len();
    let axis = if axis < 0 { rank as i32 + axis } else { axis } as usize;
    let axis = axis.min(rank.saturating_sub(1));
    let outer: usize = data_shape[..axis].iter().product::<usize>();
    let inner: usize = data_shape[axis..].iter().skip(1).product::<usize>().max(1);
    let axis_dim = data_shape.get(axis).copied().unwrap_or(1);
    for o in 0..outer {
        for i in 0..inner {
            let flat_i = o * axis_dim * inner + i;
            if flat_i >= indices.len() {
                continue;
            }
            let row = indices[flat_i].max(0) as usize;
            let dst = o * axis_dim * inner + row.min(axis_dim.saturating_sub(1)) * inner + i;
            if dst < out.len() && flat_i < updates.len() {
                out[dst] = updates[flat_i];
            }
        }
    }
}
