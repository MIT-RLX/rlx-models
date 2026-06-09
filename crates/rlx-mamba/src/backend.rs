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

//! Backend trait — the seam every accelerator implementation must
//! provide so the same `Mamba1Block` driver can run on CPU, Metal, MLX,
//! CUDA, wgpu, or ROCm without per-backend forks.
//!
//! Backends do not see the block's algorithm. They just expose enough
//! tensor-level primitives that the algorithm — written once in
//! [`crate::block::forward_with`] — can run on top.
//!
//! Linears and conv use per-backend matmul primitives; the **selective scan**
//! is a single compiled `rlx_ssm::MambaScanStage` graph per backend (see
//! [`crate::scan`] and [`Self::selective_scan`]). CUDA/wgpu run
//! `Op::SelectiveScan` on device; Metal/MLX fall back to the CPU reference
//! path for the scan while keeping device matmuls.

use anyhow::Result;
use rlx_runtime::Device;

/// Device-side tensor handle. Backends choose their own representation
/// — for CPU this is `Vec<f32>`; for Metal it might be `(arena_offset,
/// len)`; for CUDA a `CudaSlice<f32>`. The driver only ever passes
/// these back to the same backend that allocated them.
pub trait MambaTensor: Sized {}

/// One backend implementation. The trait operates on rank-3 row-major
/// f32 buffers; shapes are passed alongside since most backends don't
/// carry shape metadata in their tensor handles.
///
/// All operations are *synchronous from the caller's perspective* —
/// the backend is free to queue work internally, but a `read_to_host`
/// must flush before returning.
pub trait MambaBackend {
    type Tensor: MambaTensor;

    /// Human-readable name (used in bench labels & error messages).
    fn name(&self) -> &'static str;

    /// Whether this backend is *runnable* on the current machine. Used
    /// by benchmarks to skip backends whose required hardware/driver
    /// isn't present. CPU always returns `true`.
    fn is_available(&self) -> bool;

    /// RLX device used for [`Self::selective_scan`] (`MambaScanStage` flow).
    fn scan_device(&self) -> Device {
        Device::Cpu
    }

    /// Upload an `f32` buffer to the device.
    fn upload(&mut self, data: &[f32]) -> Result<Self::Tensor>;

    /// Allocate an uninitialized `len`-element f32 tensor.
    fn alloc(&mut self, len: usize) -> Result<Self::Tensor>;

    /// Read a tensor back to host memory.
    fn read_to_host(&mut self, t: &Self::Tensor) -> Result<Vec<f32>>;

    /// `out[m, n] = a[m, k] @ b[k, n] + bias[n]` (bias broadcast across
    /// rows). All buffers row-major. `bias` may be a zero-length slice
    /// to mean "no bias" — backends should treat that as plain gemm.
    fn sgemm_bias(
        &mut self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        out: &mut Self::Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>;

    /// `out[m, n] = a[m, k] @ b^T` where `b` is `[n, k]`.
    fn sgemm_bt(
        &mut self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        out: &mut Self::Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>;

    /// In-place SiLU.
    fn silu_in_place(&mut self, t: &mut Self::Tensor, len: usize) -> Result<()>;

    /// In-place softplus: `x ← ln(1 + exp(x))`.
    fn softplus_in_place(&mut self, t: &mut Self::Tensor, len: usize) -> Result<()>;

    /// Element-wise: `out ← exp(-exp(a_log))` (Mamba's A parameterization).
    fn neg_exp(&mut self, a_log: &Self::Tensor, out: &mut Self::Tensor, len: usize) -> Result<()>;

    /// `out ← a * b` element-wise (same length).
    fn mul(
        &mut self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        out: &mut Self::Tensor,
        len: usize,
    ) -> Result<()>;

    /// `out ← out + a` element-wise.
    fn add_assign(&mut self, out: &mut Self::Tensor, a: &Self::Tensor, len: usize) -> Result<()>;

    /// Causal depthwise conv1d over a `[batch, seq, d_inner]` tensor.
    /// `weight` is `[d_inner, k]`, `bias` is `[d_inner]`.
    /// Equivalent to PyTorch `Conv1d(d_inner, d_inner, k, groups=d_inner,
    /// padding=(k-1, k-1))` followed by `narrow(2, 0, seq)`.
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
    ) -> Result<()>;

    /// Mamba1 selective scan via `rlx_ssm::MambaScanStage` (prefill, stateless).
    ///
    /// `u` — `conv_out` `[batch, seq, d_inner]` after SiLU.
    /// `dt_raw` — `delta @ dt_proj + bias` **before** softplus `[batch, seq, d_inner]`.
    /// `a_log` — `[d_inner, d_state]`.
    #[allow(clippy::too_many_arguments)]
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
        d_inner: usize,
        d_state: usize,
    ) -> Result<()>;
}
