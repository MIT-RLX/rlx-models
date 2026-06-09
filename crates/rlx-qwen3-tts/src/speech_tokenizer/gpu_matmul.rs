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

//! GPU matmul for speech conv gemm paths (`col @ w^T`).

use crate::compile_opts::{metal_compile_guard, metal_mpsgraph_run_guard};
use anyhow::{Context, Result};
use ndarray::{Array2, ArrayView2};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

struct MatmulEntry {
    graph: CompiledGraph,
}

/// Cached `[m,k] @ [n,k]^T → [m,n]` matmul on GPU (Metal/CUDA/MLX).
pub struct GpuMatmulCache {
    device: Device,
    entries: HashMap<(usize, usize, usize), MatmulEntry>,
}

impl GpuMatmulCache {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: HashMap::new(),
        }
    }

    pub fn available(device: Device) -> bool {
        crate::gpu_pipeline::speech_conv_use_gpu(device)
    }

    fn ensure(&mut self, m: usize, k: usize, n: usize) -> Result<&mut MatmulEntry> {
        let key = (m, k, n);
        if !self.entries.contains_key(&key) {
            let f = DType::F32;
            let mut g = Graph::new("speech_conv_matmul");
            let a = g.input("a", Shape::new(&[m, k], f));
            let b = g.param("b", Shape::new(&[n, k], f));
            let bt = g.transpose_(b, vec![1, 0]);
            let out = g.matmul(a, bt, Shape::new(&[m, n], f));
            g.set_outputs(vec![out]);
            let session = Session::new(self.device);
            let graph = metal_compile_guard(self.device, || session.compile(g));
            self.entries.insert(key, MatmulEntry { graph });
        }
        Ok(self.entries.get_mut(&key).expect("matmul cache entry"))
    }

    /// Touch compile cache for one matmul shape (no-op if already built).
    pub fn touch_shape(&mut self, m: usize, k: usize, n: usize) -> Result<()> {
        let _ = self.ensure(m, k, n)?;
        Ok(())
    }

    /// Pre-compile conv gemm shapes for an utterance with `n_codec_frames` at 12 Hz.
    pub fn warmup_for_codec_frames(
        &mut self,
        n_codec_frames: usize,
        pre_conv: (usize, usize, usize),
        upsample: &[(usize, usize, usize, usize)],
        convnext: &[(usize, usize, usize)],
        decoder_entry: (usize, usize, usize),
        residual_units: &[(usize, usize, usize, usize, usize)],
        final_conv: (usize, usize, usize),
    ) -> Result<()> {
        let t = n_codec_frames.max(1);
        let mut lengths = vec![t];
        let mut cur = t;
        for &(in_ch, out_ch, k, stride) in upsample {
            let t_raw = (cur - 1) * stride + k;
            let end = t_raw - k.saturating_sub(stride);
            cur = end.max(1);
            lengths.push(cur);
            let _ = (in_ch, out_ch);
        }

        let causal_t_out = |t_in: usize, k: usize, stride: usize, dilation: usize| -> usize {
            let effective_k = (k - 1) * dilation + 1;
            let pad_left = effective_k.saturating_sub(stride);
            let n_frames = ((t_in as f32 - effective_k as f32 + pad_left as f32) / stride as f32
                + 1.0)
                .floor() as isize;
            let n_frames = n_frames.max(1) as usize;
            let ideal_len = (n_frames - 1) * stride + effective_k - pad_left;
            let extra_right = ideal_len.saturating_sub(t_in);
            let t_pad = t_in + pad_left + extra_right;
            (t_pad - effective_k) / stride + 1
        };

        let (pc_in, pc_k, pc_out) = pre_conv;
        {
            let t_out = causal_t_out(t, pc_k, 1, 1);
            self.touch_shape(t_out, pc_in * pc_k, pc_out)?;
        }

        for (stage_i, &(in_ch, out_ch, k, stride)) in upsample.iter().enumerate() {
            let t_in = lengths[stage_i];
            if t_in > 0 {
                self.touch_shape(t_in, in_ch, out_ch)?;
            }
            if let Some(&(dw_k, pw1_out, pw2_out)) = convnext.get(stage_i) {
                let t_after_up = lengths.get(stage_i + 1).copied().unwrap_or(t_in);
                let t_out = causal_t_out(t_after_up, dw_k, 1, 1);
                self.touch_shape(t_out, in_ch * dw_k, in_ch)?;
                if t_after_up > 0 {
                    self.touch_shape(t_after_up, in_ch, pw1_out)?;
                    self.touch_shape(t_after_up, pw1_out, pw2_out)?;
                }
            }
            let _ = (k, stride, out_ch);
        }

        let dec_t = *lengths.last().unwrap_or(&t);
        let (de_in, de_k, de_out) = decoder_entry;
        {
            let t_out = causal_t_out(dec_t, de_k, 1, 1);
            self.touch_shape(t_out, de_in * de_k, de_out)?;
        }
        for &(c1_in, c1_out, c1_k, c1_dil, c2_k) in residual_units {
            let t1 = causal_t_out(dec_t, c1_k, 1, c1_dil);
            self.touch_shape(t1, c1_in * c1_k, c1_out)?;
            let t2 = causal_t_out(t1, c2_k, 1, 1);
            self.touch_shape(t2, c1_out * c2_k, c1_out)?;
        }
        let (fc_in, fc_k, fc_out) = final_conv;
        {
            let t_out = causal_t_out(dec_t, fc_k, 1, 1);
            self.touch_shape(t_out, fc_in * fc_k, fc_out)?;
        }

        Ok(())
    }

    /// `col [m,k] · w^T` with `w [n,k]` → `[m,n]` (channel-major conv gemm).
    pub fn matmul_bt(&mut self, col: ArrayView2<f32>, w: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (m, k) = col.dim();
        let (n, k_w) = w.dim();
        anyhow::ensure!(k == k_w, "matmul_bt k mismatch {k} vs {k_w}");
        let device = self.device;
        let entry = self.ensure(m, k, n)?;
        let w_flat = flatten_row_major(w)?;
        entry.graph.set_param("b", &w_flat);
        let col_buf = flatten_row_major(col)?;
        let out_flat = metal_mpsgraph_run_guard(device, || {
            entry
                .graph
                .run(&[("a", &col_buf)])
                .into_iter()
                .next()
                .context("gpu matmul: no output")
        })?;
        anyhow::ensure!(out_flat.len() == m * n, "gpu matmul output len");
        Ok(Array2::from_shape_vec((m, n), out_flat)?)
    }
}

/// Row-major `[m,k]` flat buffer (copies when the view is not memory-order contiguous).
fn flatten_row_major(v: ArrayView2<f32>) -> Result<Vec<f32>> {
    if let Some(s) = v.as_slice_memory_order() {
        return Ok(s.to_vec());
    }
    let (m, k) = v.dim();
    let mut out = vec![0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            out[i * k + j] = v[[i, j]];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rlx_cpu::blas::sgemm_bt;
    use rlx_runtime::is_available;

    #[test]
    fn matmul_bt_matches_sgemm_bt_on_metal() {
        let device = Device::Metal;
        if !GpuMatmulCache::available(device) || !is_available(device) {
            return;
        }
        let m = 17usize;
        let k = 11usize;
        let n = 13usize;
        let mut col = vec![0f32; m * k];
        let mut w = vec![0f32; n * k];
        for i in 0..col.len() {
            col[i] = ((i % 19) as f32 - 9.0) * 0.03;
        }
        for i in 0..w.len() {
            w[i] = ((i % 23) as f32 - 11.0) * 0.02;
        }
        // Conv layout: sgemm_bt(w, col, …, out_ch=n, patch=k, t_out=m) → [n, m] channel-major.
        let mut cpu_out = vec![0f32; n * m];
        sgemm_bt(&w, &col, &mut cpu_out, n, k, m, 1.0);

        let col_a = Array2::from_shape_vec((m, k), col).unwrap();
        let w_a = Array2::from_shape_vec((n, k), w).unwrap();
        let mut cache = GpuMatmulCache::new(device);
        let gpu_out = cache
            .matmul_bt(col_a.view(), w_a.view())
            .expect("gpu matmul");

        let mut max_abs = 0f32;
        for ti in 0..m {
            for oc in 0..n {
                let cpu = cpu_out[oc * m + ti];
                let gpu = gpu_out[[ti, oc]];
                max_abs = max_abs.max((cpu - gpu).abs());
            }
        }
        assert!(
            max_abs < 1e-3,
            "matmul_bt diverged from sgemm_bt on Metal: max_abs={max_abs}"
        );
    }

    #[test]
    fn matmul_bt_param_update_on_metal() {
        let device = Device::Metal;
        if !GpuMatmulCache::available(device) || !is_available(device) {
            return;
        }
        let m = 8usize;
        let k = 6usize;
        let n = 5usize;
        let col =
            Array2::from_shape_vec((m, k), (0..m * k).map(|i| i as f32 * 0.01).collect()).unwrap();
        let w0 = Array2::from_shape_vec((n, k), vec![1.0; n * k]).unwrap();
        let w1 =
            Array2::from_shape_vec((n, k), (0..n * k).map(|i| (i as f32 + 1.0) * 0.1).collect())
                .unwrap();
        let mut cache = GpuMatmulCache::new(device);
        let out0 = cache.matmul_bt(col.view(), w0.view()).expect("run0");
        let out1 = cache.matmul_bt(col.view(), w1.view()).expect("run1");
        let mut cpu1 = vec![0f32; n * m];
        sgemm_bt(
            w1.as_slice().unwrap(),
            col.as_slice().unwrap(),
            &mut cpu1,
            n,
            k,
            m,
            1.0,
        );
        let mut max_abs = 0f32;
        for ti in 0..m {
            for oc in 0..n {
                let expect = cpu1[oc * m + ti];
                max_abs = max_abs.max((out1[[ti, oc]] - expect).abs());
            }
        }
        assert!(
            max_abs < 1e-3,
            "second matmul_bt with updated param diverged: max_abs={max_abs}"
        );
        assert!(
            (out0[[0, 0]] - out1[[0, 0]]).abs() > 1e-4,
            "expected different outputs for w0 vs w1"
        );
    }
}
