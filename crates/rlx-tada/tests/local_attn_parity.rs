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

//! Parity vs `tada.modules.encoder.LocalAttentionEncoder`.
//!
//! This is the block where the port could plausibly be wrong and still look
//! healthy: the RoPE is interleaved rather than rotate-half, the residual is
//! post-norm in both sub-blocks, and the GELU is exact rather than the tanh
//! approximation. Any one of those substitutions produces a graph that runs,
//! stays finite, and generates the wrong audio — so it is pinned against a real
//! forward pass from the upstream module with randomized weights and a v2
//! segment mask.
//!
//! The fixture is a 2-layer, 32-wide, 4-head instance saved from torch 2.11
//! with `transformers` 5.3.

use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{HirGraphExt, Shape};
mod common;
use common::device;
use rlx_tada::builder::{Ctx, F32, compile, lower};
use rlx_tada::local_attn::LocalAttnStack;
use rlx_tada::mask::encoder_segment_bias;
use rlx_tada::weights::TensorStore;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    d_model: usize,
    layers: usize,
    heads: usize,
    seq: usize,
    token_mask: Vec<u8>,
    x: Vec<f32>,
    y: Vec<f32>,
}

#[test]
fn matches_the_upstream_local_attention_encoder() {
    let meta: Fixture = serde_json::from_str(include_str!("fixtures/local_attn_reference.json"))
        .expect("parse local_attn_reference.json");
    let weights = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/local_attn_reference.safetensors");
    let store = std::sync::Arc::new(TensorStore::open(&weights).expect("open fixture weights"));

    let stack = LocalAttnStack::load(
        store,
        "local_attention_encoder",
        meta.layers,
        meta.heads,
        1e-5,
    )
    .expect("load stack");
    assert_eq!(stack.d_model, meta.d_model);

    let mut hir = HirModule::new("local_attn_parity");
    let mut g = HirMut::new(&mut hir);
    let mut ctx = Ctx::new(&mut g);
    let x = ctx
        .g
        .input("x", Shape::new(&[1, meta.seq, meta.d_model], F32));
    let bias = ctx.param(
        encoder_segment_bias(&meta.token_mask),
        &[1, 1, meta.seq, meta.seq],
    );
    let bias = ctx.g.expand_(
        bias,
        vec![1, meta.heads as i64, meta.seq as i64, meta.seq as i64],
    );
    let out = stack
        .build(&mut ctx, x, bias, meta.seq)
        .expect("build stack");
    let params = ctx.into_params();
    hir.set_outputs(vec![out]);
    let graph = lower(hir, "local_attn_parity").unwrap();
    let mut compiled = compile(device(), graph, params);

    let got = compiled
        .run(&[("x", &meta.x)])
        .into_iter()
        .next()
        .expect("no output");
    assert_eq!(got.len(), meta.y.len());

    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(&meta.y).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    // Activations here have magnitude ~2, so 1e-3 is well inside f32
    // accumulation noise while being far below any structural mistake (a
    // rotate-half RoPE or a tanh GELU moves this by whole units).
    assert!(
        worst < 1e-3,
        "worst deviation {worst} at index {at} (got {}, want {})",
        got[at],
        meta.y[at]
    );
}
