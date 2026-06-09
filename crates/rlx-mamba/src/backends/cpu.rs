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

//! CPU backend: f32 sgemm via `rlx_cpu::blas`, scalar element-wise
//! kernels in Rust. Always available. This is the same algorithm
//! used by [`crate::Mamba1Block::forward`] — extracted behind the
//! [`MambaBackend`] trait so other backends can be swapped in without
//! touching the algorithm.

use crate::backend::{MambaBackend, MambaTensor};
use crate::backends::ssm_dispatch;
use anyhow::{Result, ensure};
use rlx_cpu::blas;
use rlx_runtime::Device;

/// CPU "tensor" — just an owned f32 vector.
#[derive(Debug)]
pub struct CpuTensor(pub Vec<f32>);
impl MambaTensor for CpuTensor {}

#[derive(Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl MambaBackend for CpuBackend {
    type Tensor = CpuTensor;

    fn name(&self) -> &'static str {
        "cpu"
    }
    fn is_available(&self) -> bool {
        true
    }

    fn scan_device(&self) -> Device {
        Device::Cpu
    }
    fn upload(&mut self, data: &[f32]) -> Result<Self::Tensor> {
        Ok(CpuTensor(data.to_vec()))
    }
    fn alloc(&mut self, len: usize) -> Result<Self::Tensor> {
        Ok(CpuTensor(vec![0.0; len]))
    }
    fn read_to_host(&mut self, t: &Self::Tensor) -> Result<Vec<f32>> {
        Ok(t.0.clone())
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
        ensure!(a.0.len() == m * k, "sgemm_bias a shape");
        ensure!(b.0.len() == k * n, "sgemm_bias b shape");
        ensure!(out.0.len() == m * n, "sgemm_bias out shape");
        blas::sgemm(&a.0, &b.0, &mut out.0, m, k, n);
        if let Some(bv) = bias {
            ensure!(bv.0.len() == n, "sgemm_bias bias shape");
            for r in 0..m {
                let row = &mut out.0[r * n..(r + 1) * n];
                for c in 0..n {
                    row[c] += bv.0[c];
                }
            }
        }
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
        blas::sgemm_bt(&a.0, &b.0, &mut out.0, m, k, n, 1.0);
        Ok(())
    }

    fn silu_in_place(&mut self, t: &mut Self::Tensor, len: usize) -> Result<()> {
        ensure!(t.0.len() == len, "silu len");
        for v in &mut t.0 {
            *v = *v / (1.0 + (-*v).exp());
        }
        Ok(())
    }

    fn softplus_in_place(&mut self, t: &mut Self::Tensor, len: usize) -> Result<()> {
        ensure!(t.0.len() == len, "softplus len");
        for v in &mut t.0 {
            let ax = v.abs();
            *v = v.max(0.0) + (1.0 + (-ax).exp()).ln();
        }
        Ok(())
    }

    fn neg_exp(&mut self, a_log: &Self::Tensor, out: &mut Self::Tensor, len: usize) -> Result<()> {
        ensure!(a_log.0.len() == len && out.0.len() == len, "neg_exp len");
        for (o, &x) in out.0.iter_mut().zip(a_log.0.iter()) {
            *o = -x.exp();
        }
        Ok(())
    }

    fn mul(
        &mut self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        out: &mut Self::Tensor,
        len: usize,
    ) -> Result<()> {
        ensure!(
            a.0.len() == len && b.0.len() == len && out.0.len() == len,
            "mul len"
        );
        for i in 0..len {
            out.0[i] = a.0[i] * b.0[i];
        }
        Ok(())
    }

    fn add_assign(&mut self, out: &mut Self::Tensor, a: &Self::Tensor, len: usize) -> Result<()> {
        ensure!(out.0.len() == len && a.0.len() == len, "add_assign len");
        for i in 0..len {
            out.0[i] += a.0[i];
        }
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
        for b in 0..batch {
            for t in 0..seq {
                for c in 0..d_inner {
                    let mut acc = bias.0[c];
                    for i in 0..k {
                        let src_t = t as isize - (k as isize - 1) + i as isize;
                        if src_t >= 0 && (src_t as usize) < seq {
                            let v = x.0[b * seq * d_inner + (src_t as usize) * d_inner + c];
                            acc += weight.0[c * k + i] * v;
                        }
                    }
                    out.0[b * seq * d_inner + t * d_inner + c] = acc;
                }
            }
        }
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
        ssm_dispatch::run_prefill_scan(
            self.scan_device(),
            &mut out.0,
            &u.0,
            &dt_raw.0,
            &b_mat.0,
            &c_mat.0,
            &a_log.0,
            &d_skip.0,
            batch,
            seq,
            h,
            n,
        )
    }
}
