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

//! Native RLX `Op::Fft` graphs — forward and inverse, shared by bench / ablation / variants.

use crate::config::{FftLearnConfig, TransformDir};
use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

pub fn build_rlx_fft_forward_graph(cfg: &FftLearnConfig) -> Graph {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("rlx_op_fft_fwd");
    let signal = g.input("signal", Shape::new(&[batch, n], f));
    let zeros = g.sub(signal, signal);
    let block = g.concat_(vec![signal, zeros], 1);
    let out = g.fft(block, false);
    g.set_outputs(vec![out]);
    g
}

pub fn build_rlx_fft_inverse_graph(cfg: &FftLearnConfig) -> Graph {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("rlx_op_fft_inv");
    let spectrum = g.input("spectrum", Shape::new(&[batch, n * 2], f));
    let out = g.fft(spectrum, true);
    g.set_outputs(vec![out]);
    g
}

pub fn compile_rlx_fft(
    cfg: &FftLearnConfig,
    dir: TransformDir,
    device: Device,
) -> Result<CompiledGraph> {
    let graph = if dir.is_forward() {
        build_rlx_fft_forward_graph(cfg)
    } else {
        build_rlx_fft_inverse_graph(cfg)
    };
    Ok(Session::new(device).compile(graph))
}

/// Real `[batch, n]` → interleaved spectrum via compiled forward `Op::Fft`.
pub fn rlx_fft_forward(
    exec: &mut CompiledGraph,
    signal: &[f32],
    batch: usize,
    n_fft: usize,
) -> Vec<f32> {
    let block = rlx_fft_forward_block(exec, signal, batch, n_fft);
    crate::reference::block_to_interleaved(&block, batch, n_fft)
}

/// Real `[batch, n]` → RLX FFT block layout `[batch, 2*n]` (re ∥ im planes).
pub fn rlx_fft_forward_block(
    exec: &mut CompiledGraph,
    signal: &[f32],
    _batch: usize,
    _n_fft: usize,
) -> Vec<f32> {
    exec.run(&[("signal", signal)]).remove(0)
}

/// Phase 1 — read FFT block output from arena after `run_slots` (one copy, no interleaved convert).
pub fn rlx_fft_forward_block_in_arena(
    exec: &mut CompiledGraph,
    signal: &[f32],
    batch: usize,
    n_fft: usize,
) -> Vec<f32> {
    let slots = exec.run_slots(&[signal]);
    let (off, len) = slots[0];
    let ptr = exec.arena_ptr();
    let mut block = vec![0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr.add(off) as *const f32, block.as_mut_ptr(), len);
    }
    let _ = (batch, n_fft);
    block
}

/// Block spectrum `[batch, n*2]` (re plane ∥ im plane) → interleaved via inverse `Op::Fft`.
pub fn rlx_fft_inverse_block(
    exec: &mut CompiledGraph,
    spectrum_block: &[f32],
    batch: usize,
    n_fft: usize,
) -> Vec<f32> {
    let block = exec.run(&[("spectrum", spectrum_block)]).remove(0);
    crate::reference::block_to_interleaved(&block, batch, n_fft)
}

/// Interleaved complex spectrum → block layout for inverse RLX FFT input.
pub fn interleaved_to_block(interleaved: &[f32], batch: usize, n_fft: usize) -> Vec<f32> {
    let mut block = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        for i in 0..n_fft {
            let src = b * n_fft * 2 + i * 2;
            let dst = b * n_fft * 2;
            block[dst + i] = interleaved[src];
            block[dst + n_fft + i] = interleaved[src + 1];
        }
    }
    block
}
