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

//! In-place KV cache commits (reuse layer buffers across decode steps).

/// Copy bucketed decode K/V into existing layer buffers (resize in place, no fresh `Vec` per step).
pub fn commit_kv_layers(
    layers_k: &mut [Vec<f32>],
    layers_v: &mut [Vec<f32>],
    new_k: &[Vec<f32>],
    new_v: &[Vec<f32>],
) {
    for (i, (dst_k, dst_v)) in layers_k.iter_mut().zip(layers_v.iter_mut()).enumerate() {
        let src_k = &new_k[i];
        let src_v = &new_v[i];
        if dst_k.len() != src_k.len() {
            dst_k.resize(src_k.len(), 0.0);
        }
        if dst_v.len() != src_v.len() {
            dst_v.resize(src_v.len(), 0.0);
        }
        dst_k.copy_from_slice(src_k);
        dst_v.copy_from_slice(src_v);
    }
}
