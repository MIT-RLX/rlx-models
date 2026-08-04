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

//! Apple Metal backend.
//!
//! Architecture: one growable `metal::Buffer` in shared-storage mode,
//! bump-allocated per `upload`/`alloc`. The matmuls go through
//! `rlx_metal::blas::metal_sgemm` / `metal_sgemm_bias` (custom MSL
//! kernels, picked per-shape via `hw_model().pick_sgemm`). Element-wise
//! ops, the depthwise causal conv1d, and the selective scan run on the
//! CPU side via the same buffer's `contents()` pointer — Apple Silicon's
//! unified memory makes that a memory-fence-only operation, not a copy.
//!
//! This is the "first useful" Metal backend: matmuls dispatch to the
//! GPU (where most of the FLOPs live for typical Mamba1 configs); the
//! rest stays on CPU until a native MSL selective-scan kernel and
//! per-op MSL element-wise kernels are wired into `rlx-metal`. The
//! survey in `claude-code-guide` notes both kernels are achievable
//! (the WGSL/CUDA scan ports are the reference).
//!
//! Build:  cargo build -p rlx-mamba --features metal

use crate::backend::{MambaBackend, MambaTensor};
use crate::backends::ssm_dispatch;
use anyhow::{Result, anyhow, bail};
use metal::{Buffer, CommandBufferRef, CommandQueue};
use rlx_metal::blas as mblas;
use rlx_metal::device::metal_device;
use rlx_runtime::Device;
use std::ffi::c_void;

/// Bump-allocated handle into the shared Metal arena. `len` is in f32
/// elements, `offset` is in bytes (matches the units the Metal kernel
/// wrappers expect).
pub struct MetalTensor {
    pub offset: usize,
    pub len: usize,
}
impl MambaTensor for MetalTensor {}

pub struct MetalBackend {
    device: &'static rlx_metal::device::MetalDevice,
    arena: Buffer,
    capacity_bytes: usize,
    cursor: usize, // bytes
    /// Pre-allocated zero-bias region (length = max single-vector size
    /// we ever encounter). Used when callers pass `bias = None` but we
    /// dispatch to the fused `metal_sgemm_bias` kernel anyway.
    zero_bias_off: usize,
    zero_bias_cap: usize,
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = metal_device()
            .ok_or_else(|| anyhow!("rlx-mamba: no Metal device available on this host"))?;
        // Initial 16 MiB; we'll grow as needed.
        let capacity_bytes = 16 * 1024 * 1024;
        let arena = device.alloc_shared(capacity_bytes);
        // Reserve the first 4 KiB as our "always-zero" region (covers any
        // bias up to 1024 f32). The shared buffer is zero-initialized.
        let zero_bias_cap = 1024 * 4;
        Ok(Self {
            device,
            arena,
            capacity_bytes,
            cursor: zero_bias_cap,
            zero_bias_off: 0,
            zero_bias_cap,
        })
    }

    /// Whether Metal initialization can complete on this host.
    pub fn available() -> bool {
        metal_device().is_some()
    }

    fn grow_to(&mut self, needed_bytes: usize) {
        if needed_bytes <= self.capacity_bytes {
            return;
        }
        let mut new_cap = self.capacity_bytes;
        while new_cap < needed_bytes {
            new_cap *= 2;
        }
        let new_arena = self.device.alloc_shared(new_cap);
        // Copy existing content over (only the [0..cursor] range matters).
        unsafe {
            let src = self.arena.contents() as *const u8;
            let dst = new_arena.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, self.cursor);
        }
        self.arena = new_arena;
        self.capacity_bytes = new_cap;
    }

    fn bump_alloc(&mut self, len_f32: usize) -> MetalTensor {
        let bytes = len_f32 * std::mem::size_of::<f32>();
        // 16-byte align (Metal/SIMD requires natural alignment for vec4).
        let aligned_cursor = (self.cursor + 15) & !15;
        let new_cursor = aligned_cursor + bytes;
        self.grow_to(new_cursor);
        let offset = aligned_cursor;
        self.cursor = new_cursor;
        MetalTensor {
            offset,
            len: len_f32,
        }
    }

    fn host_slice(&self, t: &MetalTensor) -> &[f32] {
        unsafe {
            let base = self.arena.contents() as *const u8;
            let ptr = base.add(t.offset) as *const f32;
            std::slice::from_raw_parts(ptr, t.len)
        }
    }

    fn host_slice_mut(&mut self, t: &MetalTensor) -> &mut [f32] {
        unsafe {
            let base = self.arena.contents() as *mut u8;
            let ptr = base.add(t.offset) as *mut f32;
            std::slice::from_raw_parts_mut(ptr, t.len)
        }
    }

    /// Encode + commit + wait for a single GPU dispatch. Mamba's
    /// algorithm is sequential at the host level, so flushing per-call
    /// keeps the semantics simple; batching multiple sgemms into one
    /// command buffer is an obvious follow-up optimization.
    fn run_gpu<F: FnOnce(&CommandBufferRef, &metal::ComputeCommandEncoderRef)>(
        queue: &CommandQueue,
        f: F,
    ) {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        f(cmd, enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

impl MambaBackend for MetalBackend {
    type Tensor = MetalTensor;

    fn name(&self) -> &'static str {
        "metal"
    }

    fn scan_device(&self) -> Device {
        ssm_dispatch::execution_device("metal")
    }
    fn is_available(&self) -> bool {
        true
    }

    fn upload(&mut self, data: &[f32]) -> Result<Self::Tensor> {
        let t = self.bump_alloc(data.len());
        let bytes = std::mem::size_of_val(data);
        unsafe {
            let dst = (self.arena.contents() as *mut u8).add(t.offset);
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const c_void as *const u8, dst, bytes);
        }
        Ok(t)
    }

    fn alloc(&mut self, len: usize) -> Result<Self::Tensor> {
        let t = self.bump_alloc(len);
        // Zero-initialize so reads-before-writes don't see garbage.
        for v in self.host_slice_mut(&t) {
            *v = 0.0;
        }
        Ok(t)
    }

    fn read_to_host(&mut self, t: &Self::Tensor) -> Result<Vec<f32>> {
        Ok(self.host_slice(t).to_vec())
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
        if a.len != m * k || b.len != k * n || out.len != m * n {
            bail!("metal sgemm_bias: shape mismatch m={m} k={k} n={n}");
        }
        let a_off = a.offset;
        let b_off = b.offset;
        let c_off = out.offset;
        // Resolve bias offset before borrowing self for the dispatch.
        let bias_off = match bias {
            Some(t) => t.offset,
            None if n * std::mem::size_of::<f32>() <= self.zero_bias_cap => self.zero_bias_off,
            None => {
                // Need a wider zero region than the preallocated one.
                let zb = self.alloc(n)?;
                zb.offset
            }
        };
        let queue = self.device.queue.clone();
        let arena = self.arena.clone();
        Self::run_gpu(&queue, |_cmd, enc| {
            mblas::metal_sgemm_bias(
                enc,
                &arena,
                a_off,
                b_off,
                bias_off,
                c_off,
                m,
                k,
                n,
                mblas::FusedAct::None,
            );
        });
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
        // No fused sgemm_bt on the rlx-metal MSL path yet; transpose
        // host-side. This only fires for `lm_head` against a tied
        // embedding, which the Mamba1 block doesn't hit, so the slow
        // path is fine until/unless a network-level driver shows up.
        let b_host = self.host_slice(b).to_vec();
        let mut bt = vec![0.0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                bt[c * n + r] = b_host[r * k + c];
            }
        }
        let bt_t = self.upload(&bt)?;
        self.sgemm_bias(a, &bt_t, None, out, m, k, n)
    }

    fn silu_in_place(&mut self, t: &mut Self::Tensor, len: usize) -> Result<()> {
        let s = self.host_slice_mut(t);
        if s.len() != len {
            bail!("silu len");
        }
        for v in s {
            *v = *v / (1.0 + (-*v).exp());
        }
        Ok(())
    }

    fn softplus_in_place(&mut self, t: &mut Self::Tensor, len: usize) -> Result<()> {
        let s = self.host_slice_mut(t);
        if s.len() != len {
            bail!("softplus len");
        }
        for v in s {
            let ax = v.abs();
            *v = v.max(0.0) + (1.0 + (-ax).exp()).ln();
        }
        Ok(())
    }

    fn neg_exp(&mut self, a: &Self::Tensor, out: &mut Self::Tensor, len: usize) -> Result<()> {
        let a_host = self.host_slice(a).to_vec();
        let o = self.host_slice_mut(out);
        if o.len() != len || a_host.len() != len {
            bail!("neg_exp len");
        }
        for i in 0..len {
            o[i] = -a_host[i].exp();
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
        let a_host = self.host_slice(a).to_vec();
        let b_host = self.host_slice(b).to_vec();
        let o = self.host_slice_mut(out);
        if o.len() != len {
            bail!("mul len");
        }
        for i in 0..len {
            o[i] = a_host[i] * b_host[i];
        }
        Ok(())
    }

    fn add_assign(&mut self, out: &mut Self::Tensor, a: &Self::Tensor, len: usize) -> Result<()> {
        let a_host = self.host_slice(a).to_vec();
        let o = self.host_slice_mut(out);
        if o.len() != len {
            bail!("add_assign len");
        }
        for i in 0..len {
            o[i] += a_host[i];
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
        let x_h = self.host_slice(x).to_vec();
        let w_h = self.host_slice(weight).to_vec();
        let b_h = self.host_slice(bias).to_vec();
        let o = self.host_slice_mut(out);
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
        let device = self.scan_device();
        let u_h = self.host_slice(u).to_vec();
        let dt_h = self.host_slice(dt_raw).to_vec();
        let b_h = self.host_slice(b_mat).to_vec();
        let c_h = self.host_slice(c_mat).to_vec();
        let a_h = self.host_slice(a_log).to_vec();
        let dskip_h = self.host_slice(d_skip).to_vec();
        let o = self.host_slice_mut(out);
        ssm_dispatch::run_prefill_scan(
            device, o, &u_h, &dt_h, &b_h, &c_h, &a_h, &dskip_h, batch, seq, h, n,
        )
    }
}
