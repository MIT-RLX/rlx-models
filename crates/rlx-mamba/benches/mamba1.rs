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

//! Multi-backend Mamba1 prefill bench.
//!
//! Each available `MambaBackend` produces one criterion line. Backends
//! whose `is_available()` returns `false` are skipped with a notice,
//! so the bench can run uniformly on any host (CPU is always present;
//! accelerator backends light up as their natives are wired).
//!
//! For the head-to-head comparison against `burn-mamba`, see
//! `crates/rlx-mamba-bench`.

use criterion::{Criterion, criterion_group, criterion_main};
use rlx_mamba::{
    CpuBackend, Mamba1Block, Mamba1Config, Mamba1ResidentBlock, MambaBackend, mamba1_forward,
};
use std::hint::black_box;

const D_MODEL: usize = 128;
const SEQ: usize = 256;
const BATCH: usize = 1;

fn bench_backend<B: MambaBackend + 'static>(c: &mut Criterion, mut backend: B, label: &str) {
    if !backend.is_available() {
        eprintln!("[bench] {label}: backend not available, skipping");
        return;
    }
    let cfg = Mamba1Config::new(D_MODEL);
    let block = Mamba1Block::random_for_bench(cfg, 0xA110CA7E);
    let resident = Mamba1ResidentBlock::upload(&mut backend, &block).unwrap();
    let input: Vec<f32> = (0..BATCH * SEQ * D_MODEL)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();

    let bench_label = format!("rlx_mamba::forward[{label}](d{D_MODEL}_s{SEQ})");
    c.bench_function(&bench_label, |b| {
        b.iter(|| {
            let y = mamba1_forward(&mut backend, &resident, black_box(&input), BATCH, SEQ).unwrap();
            black_box(y);
        });
    });
}

fn bench_cpu(c: &mut Criterion) {
    bench_backend(c, CpuBackend::new(), "cpu");
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn bench_metal(c: &mut Criterion) {
    match rlx_mamba::backends::metal::MetalBackend::new() {
        Ok(b) => bench_backend(c, b, "metal"),
        Err(e) => eprintln!("[bench] metal: init failed: {e}"),
    }
}
#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn bench_metal(_: &mut Criterion) {}

#[cfg(all(feature = "mlx", any(target_os = "macos", target_os = "ios")))]
fn bench_mlx(c: &mut Criterion) {
    match rlx_mamba::backends::mlx::MlxBackend::new() {
        Ok(b) => bench_backend(c, b, "mlx"),
        Err(e) => eprintln!("[bench] mlx: init failed: {e}"),
    }
}
#[cfg(not(all(feature = "mlx", any(target_os = "macos", target_os = "ios"))))]
fn bench_mlx(_: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn bench_cuda(c: &mut Criterion) {
    match rlx_mamba::backends::cuda::CudaBackend::new() {
        Ok(b) => bench_backend(c, b, "cuda"),
        Err(e) => eprintln!("[bench] cuda: init failed: {e}"),
    }
}
#[cfg(not(feature = "cuda"))]
fn bench_cuda(_: &mut Criterion) {}

#[cfg(feature = "wgpu")]
fn bench_wgpu(c: &mut Criterion) {
    match rlx_mamba::backends::wgpu::WgpuBackend::new() {
        Ok(b) => bench_backend(c, b, "wgpu"),
        Err(e) => eprintln!("[bench] wgpu: init failed: {e}"),
    }
}
#[cfg(not(feature = "wgpu"))]
fn bench_wgpu(_: &mut Criterion) {}

#[cfg(feature = "rocm")]
fn bench_rocm(c: &mut Criterion) {
    match rlx_mamba::backends::rocm::RocmBackend::new() {
        Ok(b) => bench_backend(c, b, "rocm"),
        Err(e) => eprintln!("[bench] rocm: init failed: {e}"),
    }
}
#[cfg(not(feature = "rocm"))]
fn bench_rocm(_: &mut Criterion) {}

criterion_group!(
    benches,
    bench_cpu,
    bench_metal,
    bench_mlx,
    bench_cuda,
    bench_wgpu,
    bench_rocm
);
criterion_main!(benches);
