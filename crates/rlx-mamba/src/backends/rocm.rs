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

//! AMD ROCm / HIP backend.
//!
//! Same shape as the CUDA backend (per-shape matmul graph cached via
//! `RocmExecutable::compile`, dispatched through `.run(&[...])`).
//! Element-wise / scan / conv1d run host-side, matching CUDA today;
//! once the IR-graph approach lands, the native `Op::SelectiveScan`
//! lowering in `rlx-rocm` (sharing `.cu` sources with `rlx-cuda` via
//! hipRTC) takes over the scan.
//!
//! Where this is verified:
//! - **Compile**: `Dockerfile.rocm-check` builds against the real HIP
//!   runtime inside the official `rocm/dev-ubuntu-22.04` image.
//! - **Runtime**: requires an AMD host with `/dev/kfd` accessible.
//!   `rlx_rocm::is_available()` probes the live HIP runtime; falls
//!   back to false everywhere else (including this Mac and the
//!   NVIDIA rig).

use crate::backend::{MambaBackend, MambaTensor};
use crate::backends::ssm_dispatch;
use anyhow::{Result, bail};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

pub struct RocmTensor(pub Vec<f32>);
impl MambaTensor for RocmTensor {}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct GemmKey {
    m: usize,
    k: usize,
    n: usize,
    has_bias: bool,
}

pub struct RocmBackend {
    cache: HashMap<GemmKey, rlx_rocm::RocmExecutable>,
}

impl RocmBackend {
    pub fn new() -> Result<Self> {
        if !rlx_rocm::is_available() {
            bail!("rlx-mamba: ROCm/HIP runtime not present (no AMD GPU or HIP libs missing)");
        }
        Ok(Self {
            cache: HashMap::new(),
        })
    }

    pub fn available() -> bool {
        rlx_rocm::is_available()
    }

    fn compile_sgemm(&mut self, key: GemmKey) -> &mut rlx_rocm::RocmExecutable {
        self.cache.entry(key).or_insert_with(|| {
            let mut g = Graph::new("mamba1.sgemm");
            let x = g.input("x", Shape::new(&[key.m, key.k], DType::F32));
            let w = g.input("w", Shape::new(&[key.k, key.n], DType::F32));
            let y = g.matmul(x, w, Shape::new(&[key.m, key.n], DType::F32));
            let out = if key.has_bias {
                let b = g.input("b", Shape::new(&[key.n], DType::F32));
                g.binary(
                    rlx_ir::op::BinaryOp::Add,
                    y,
                    b,
                    Shape::new(&[key.m, key.n], DType::F32),
                )
            } else {
                y
            };
            g.set_outputs(vec![out]);
            rlx_rocm::RocmExecutable::compile(g)
        })
    }
}

impl MambaBackend for RocmBackend {
    type Tensor = RocmTensor;
    fn name(&self) -> &'static str {
        "rocm"
    }
    fn is_available(&self) -> bool {
        true
    }

    fn scan_device(&self) -> Device {
        ssm_dispatch::execution_device("rocm")
    }

    fn upload(&mut self, data: &[f32]) -> Result<Self::Tensor> {
        Ok(RocmTensor(data.to_vec()))
    }
    fn alloc(&mut self, len: usize) -> Result<Self::Tensor> {
        Ok(RocmTensor(vec![0.0; len]))
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
        let key = GemmKey {
            m,
            k,
            n,
            has_bias: bias.is_some(),
        };
        let a_data = a.0.as_slice();
        let b_data = b.0.as_slice();
        let bias_data = bias.map(|t| t.0.as_slice());
        let exe = self.compile_sgemm(key);
        let inputs: Vec<(&str, &[f32])> = match bias_data {
            Some(bd) => vec![("x", a_data), ("w", b_data), ("b", bd)],
            None => vec![("x", a_data), ("w", b_data)],
        };
        let mut outs = exe.run(&inputs);
        out.0 = outs.remove(0);
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
        let mut bt = vec![0.0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                bt[c * n + r] = b.0[r * k + c];
            }
        }
        let bt_t = RocmTensor(bt);
        self.sgemm_bias(a, &bt_t, None, out, m, k, n)
    }

    fn silu_in_place(&mut self, t: &mut Self::Tensor, _len: usize) -> Result<()> {
        for v in &mut t.0 {
            *v = *v / (1.0 + (-*v).exp());
        }
        Ok(())
    }
    fn softplus_in_place(&mut self, t: &mut Self::Tensor, _len: usize) -> Result<()> {
        for v in &mut t.0 {
            let ax = v.abs();
            *v = v.max(0.0) + (1.0 + (-ax).exp()).ln();
        }
        Ok(())
    }
    fn neg_exp(&mut self, a: &Self::Tensor, out: &mut Self::Tensor, len: usize) -> Result<()> {
        if a.0.len() != len || out.0.len() != len {
            bail!("neg_exp len");
        }
        for (o, &v) in out.0.iter_mut().zip(a.0.iter()) {
            *o = -v.exp();
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
        for i in 0..len {
            out.0[i] = a.0[i] * b.0[i];
        }
        Ok(())
    }
    fn add_assign(&mut self, out: &mut Self::Tensor, a: &Self::Tensor, len: usize) -> Result<()> {
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
