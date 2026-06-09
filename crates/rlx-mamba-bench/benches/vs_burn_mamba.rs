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

//! Head-to-head: `rlx-mamba` vs `burn-mamba` on the parallel prefill
//! path of a single Mamba1 block. Same hyperparameters (d_model=128,
//! d_state=16, d_conv=4, expand=2, seq=256, batch=1), same input.
//!
//! Weights aren't shared — both sides use their own random init — so
//! this measures kernel/runtime throughput, not numerical equivalence
//! (the `forward_step_parity` test in rlx-mamba covers correctness).
//!
//! Run with:  cargo bench -p rlx-mamba-bench

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn_mamba::mamba1::prelude::{Mamba1, Mamba1Config as BurnMamba1Config};
use criterion::{Criterion, criterion_group, criterion_main};
use rlx_mamba::{Mamba1Block, Mamba1Config};
use std::hint::black_box;

const D_MODEL: usize = 128;
const SEQ: usize = 256;
const BATCH: usize = 1;

fn input_vec() -> Vec<f32> {
    (0..BATCH * SEQ * D_MODEL)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect()
}

fn bench_rlx(c: &mut Criterion) {
    let cfg = Mamba1Config::new(D_MODEL);
    let block = Mamba1Block::random_for_bench(cfg, 0xA110CA7E);
    let input = input_vec();
    c.bench_function("rlx_mamba::forward(d128_s256)", |b| {
        b.iter(|| {
            let y = block.forward(black_box(&input), BATCH, SEQ).unwrap();
            black_box(y);
        });
    });
}

fn bench_burn(c: &mut Criterion) {
    type B = NdArray<f32>;
    let device = Default::default();
    let burn_cfg = BurnMamba1Config::new(D_MODEL)
        .with_d_state(16)
        .with_d_conv(4)
        .with_expand(2);
    let model: Mamba1<B> = burn_cfg.init(&device);
    let data = TensorData::new(input_vec(), [BATCH, SEQ, D_MODEL]);

    c.bench_function("burn_mamba::forward(d128_s256)", |b| {
        b.iter(|| {
            let x = Tensor::<B, 3>::from_data(data.clone(), &device);
            let y = model.forward(black_box(x));
            black_box(y);
        });
    });
}

criterion_group!(benches, bench_rlx, bench_burn);
criterion_main!(benches);
