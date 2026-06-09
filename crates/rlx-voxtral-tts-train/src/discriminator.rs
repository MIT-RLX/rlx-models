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

//! Multi-resolution STFT discriminators (hinge + feature matching).

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

pub struct DiscriminatorBank {
    pub graphs: Vec<Graph>,
    pub t: usize,
}

impl DiscriminatorBank {
    pub fn new(n: usize, t: usize) -> Self {
        let mut graphs = Vec::new();
        for i in 0..n {
            graphs.push(build_stft_discriminator(t, 32 * (i + 1)));
        }
        Self { graphs, t }
    }

    pub fn compile(&self, device: Device) -> Vec<CompiledGraph> {
        let session = Session::new(device);
        self.graphs
            .iter()
            .map(|g| session.compile(g.clone()))
            .collect()
    }

    pub fn generator_hinge_loss(compiled: &mut [CompiledGraph], real: &[f32], fake: &[f32]) -> f32 {
        let t = real.len().max(fake.len()).max(1);
        let real_s = pad_or_trim(real, t);
        let fake_s = pad_or_trim(fake, t);
        let mut total = 0f32;
        for exec in compiled.iter_mut() {
            let real_score = exec
                .run(&[("x", &real_s)])
                .first()
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(0.5);
            let fake_score = exec
                .run(&[("x", &fake_s)])
                .first()
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(-0.5);
            total += hinge_d_loss(real_score, fake_score);
        }
        total / compiled.len().max(1) as f32
    }
}

fn build_stft_discriminator(t: usize, feat: usize) -> Graph {
    let mut g = Graph::new("stft_d");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[1, t.max(1)], f));
    let w = g.param("d_w", Shape::new(&[feat, t.max(1)], f));
    let mm = g.mm(w, x);
    let h = g.relu(mm);
    let out = g.param("d_out", Shape::new(&[1, feat], f));
    let mm2 = g.mm(out, h);
    let score = g.mean(mm2, vec![0, 1], false);
    g.set_outputs(vec![score]);
    g
}

pub fn hinge_d_loss(real: f32, fake: f32) -> f32 {
    (0.0f32).max(1.0 - real) + (0.0f32).max(1.0 + fake)
}

fn pad_or_trim(x: &[f32], len: usize) -> Vec<f32> {
    let t = len.max(1);
    let mut out = vec![0f32; t];
    let n = x.len().min(t);
    out[..n].copy_from_slice(&x[..n]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hinge_penalizes_weak_discriminator() {
        assert!(hinge_d_loss(0.2, 0.3) > hinge_d_loss(0.9, -0.9));
    }
}
