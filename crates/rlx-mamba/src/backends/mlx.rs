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

//! Apple MLX backend.
//!
//! Each tensor wraps an `mlx::Array`. Linears use MLX's native `matmul`;
//! element-wise ops compose from MLX primitives (`add`, `mul`, `unary`,
//! `sub`). The depthwise causal conv1d and the selective scan are
//! computed host-side after a `to_f32` readback — MLX has no native
//! selective-scan op exposed at this layer (the IR path in
//! `rlx-mlx/src/lower.rs` unrolls it but isn't available here).
//!
//! MLX is lazy: arrays are computation graphs until `eval()` (or any
//! readback) forces them. We rely on `to_f32` to flush at backend
//! boundaries, which keeps the bench numbers faithful.
//!
//! Build:  cargo build -p rlx-mamba --features mlx

use crate::backend::{MambaBackend, MambaTensor};
use crate::backends::ssm_dispatch;
use anyhow::{Context, Result, anyhow};
use rlx_ir::DType;
use rlx_mlx::Array;
use rlx_mlx::ops;
use rlx_runtime::Device;
// Note: `rlx_mlx::ops::unary` takes a private `MlxUnary` enum from
// `rlx_mlx::ffi`, so it's not callable from outside the crate. We work
// around by routing unaries through host-side compute (via `to_f32` +
// CPU loop + re-upload). Linears still use MLX's native `matmul`,
// which is what dominates compute for typical Mamba1 sizes.

pub struct MlxTensor {
    pub array: Array,
    pub len: usize,
}
impl MambaTensor for MlxTensor {}

pub struct MlxBackend;

impl MlxBackend {
    pub fn new() -> Result<Self> {
        // Touch the runtime to validate it actually loads.
        let _ = Array::from_f32_slice(&[0.0], &[1], DType::F32)
            .map_err(|e| anyhow!("rlx-mamba: MLX runtime unavailable: {e:?}"))?;
        Ok(Self)
    }

    pub fn available() -> bool {
        Array::from_f32_slice(&[0.0], &[1], DType::F32).is_ok()
    }

    fn from_host(data: &[f32]) -> Result<MlxTensor> {
        let array = Array::from_f32_slice(data, &[data.len()], DType::F32)
            .map_err(|e| anyhow!("mlx from_host: {e:?}"))?;
        Ok(MlxTensor {
            array,
            len: data.len(),
        })
    }
}

fn err<T>(e: rlx_mlx::MlxError) -> anyhow::Result<T> {
    Err(anyhow!("mlx: {e:?}"))
}

fn reshape2(a: &Array, rows: i32, cols: i32) -> Result<Array> {
    ops::reshape(a, &[rows, cols]).or_else(err)
}

impl MambaBackend for MlxBackend {
    type Tensor = MlxTensor;

    fn name(&self) -> &'static str {
        "mlx"
    }
    fn is_available(&self) -> bool {
        true
    }

    fn scan_device(&self) -> Device {
        ssm_dispatch::execution_device("mlx")
    }

    fn upload(&mut self, data: &[f32]) -> Result<Self::Tensor> {
        Self::from_host(data)
    }

    fn alloc(&mut self, len: usize) -> Result<Self::Tensor> {
        // MLX has no "uninit" alloc; create a zeroed buffer.
        Self::from_host(&vec![0.0f32; len])
    }

    fn read_to_host(&mut self, t: &Self::Tensor) -> Result<Vec<f32>> {
        t.array.to_f32().or_else(err).context("read_to_host")
    }

    fn sgemm_bias(
        &mut self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        out: &mut Self::Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let a2 = reshape2(&a.array, m as i32, k as i32)?;
        let b2 = reshape2(&b.array, k as i32, n as i32)?;
        let mut c = ops::matmul(&a2, &b2).or_else(err)?;
        if let Some(bt) = bias {
            // Bias is [n]; matmul output is [m, n]; broadcasted add.
            let bias2 = reshape2(&bt.array, 1, n as i32)?;
            c = ops::add(&c, &bias2).or_else(err)?;
        }
        // Materialize as flat [m*n].
        let flat = ops::reshape(&c, &[(m * n) as i32]).or_else(err)?;
        out.array = flat;
        out.len = m * n;
        Ok(())
    }

    fn sgemm_bt(
        &mut self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        out: &mut Self::Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let a2 = reshape2(&a.array, m as i32, k as i32)?;
        let b2 = reshape2(&b.array, n as i32, k as i32)?;
        let bt = ops::transpose(&b2, &[1, 0]).or_else(err)?;
        let c = ops::matmul(&a2, &bt).or_else(err)?;
        let flat = ops::reshape(&c, &[(m * n) as i32]).or_else(err)?;
        out.array = flat;
        out.len = m * n;
        Ok(())
    }

    fn silu_in_place(&mut self, t: &mut Self::Tensor, _len: usize) -> Result<()> {
        let mut h = t.array.to_f32().or_else(err)?;
        for v in &mut h {
            *v = *v / (1.0 + (-*v).exp());
        }
        *t = Self::from_host(&h)?;
        Ok(())
    }

    fn softplus_in_place(&mut self, t: &mut Self::Tensor, _len: usize) -> Result<()> {
        let mut h = t.array.to_f32().or_else(err)?;
        for v in &mut h {
            let ax = v.abs();
            *v = v.max(0.0) + (1.0 + (-ax).exp()).ln();
        }
        *t = Self::from_host(&h)?;
        Ok(())
    }

    fn neg_exp(&mut self, a: &Self::Tensor, out: &mut Self::Tensor, len: usize) -> Result<()> {
        let mut h = a.array.to_f32().or_else(err)?;
        for v in &mut h {
            *v = -v.exp();
        }
        *out = Self::from_host(&h)?;
        out.len = len;
        Ok(())
    }

    fn mul(
        &mut self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        out: &mut Self::Tensor,
        len: usize,
    ) -> Result<()> {
        out.array = ops::mul(&a.array, &b.array).or_else(err)?;
        out.len = len;
        Ok(())
    }

    fn add_assign(&mut self, out: &mut Self::Tensor, a: &Self::Tensor, len: usize) -> Result<()> {
        out.array = ops::add(&out.array, &a.array).or_else(err)?;
        out.len = len;
        Ok(())
    }

    fn causal_conv1d(
        &mut self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
        out: &mut Self::Tensor,
        batch: usize,
        seq: usize,
        d_inner: usize,
        k: usize,
    ) -> Result<()> {
        // Host-side reference (MLX's `conv1d` exists but its calling
        // convention for groups=d_inner depthwise needs careful weight
        // reshaping; the host path is correct and simple).
        let x_h = x.array.to_f32().or_else(err)?;
        let w_h = weight.array.to_f32().or_else(err)?;
        let b_h = bias.array.to_f32().or_else(err)?;
        let mut o = vec![0.0f32; batch * seq * d_inner];
        for b in 0..batch {
            for t in 0..seq {
                for c in 0..d_inner {
                    let mut acc = b_h[c];
                    for i in 0..k {
                        let src_t = t as isize - (k as isize - 1) + i as isize;
                        if src_t >= 0 && (src_t as usize) < seq {
                            let v = x_h[b * seq * d_inner + (src_t as usize) * d_inner + c];
                            acc += w_h[c * k + i] * v;
                        }
                    }
                    o[b * seq * d_inner + t * d_inner + c] = acc;
                }
            }
        }
        *out = Self::from_host(&o)?;
        Ok(())
    }

    fn selective_scan(
        &mut self,
        u: &Self::Tensor,
        dt_raw: &Self::Tensor,
        b_mat: &Self::Tensor,
        c_mat: &Self::Tensor,
        a_log: &Self::Tensor,
        d_skip: &Self::Tensor,
        out: &mut Self::Tensor,
        batch: usize,
        seq: usize,
        h: usize,
        n: usize,
    ) -> Result<()> {
        let u_h = u.array.to_f32().or_else(err)?;
        let dt_h = dt_raw.array.to_f32().or_else(err)?;
        let b_h = b_mat.array.to_f32().or_else(err)?;
        let c_h = c_mat.array.to_f32().or_else(err)?;
        let a_h = a_log.array.to_f32().or_else(err)?;
        let dskip_h = d_skip.array.to_f32().or_else(err)?;
        let mut o = vec![0.0f32; batch * seq * h];
        ssm_dispatch::run_prefill_scan(
            self.scan_device(),
            &mut o,
            &u_h,
            &dt_h,
            &b_h,
            &c_h,
            &a_h,
            &dskip_h,
            batch,
            seq,
            h,
            n,
        )?;
        *out = Self::from_host(&o)?;
        Ok(())
    }
}
