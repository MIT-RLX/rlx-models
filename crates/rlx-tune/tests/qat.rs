// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Quantization-aware training through `DpConfig::qat`: a model trained with
//! its forward in f4 must beat the same model trained in f32 when both are
//! evaluated in f4 — the point of QAT.

use std::collections::HashMap;

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompileOptions, Device, NumberFormat, Session};
use rlx_tune::{DpConfig, ParamSlot, Trainer};

const N: usize = 64;
const IN: usize = 8;
const HID: usize = 32;

fn param(g: &mut Graph, name: &str, dims: &[usize]) -> NodeId {
    g.add_node(
        Op::Param { name: name.into() },
        vec![],
        Shape::new(dims, DType::F32),
    )
}

/// `mean((tanh(x W1) W2 - y)^2)`.
fn mlp_loss() -> (Graph, Vec<ParamSlot>) {
    let mut g = Graph::new("mlp");
    let x = g.input("x", Shape::new(&[N, IN], DType::F32));
    let y = g.input("y", Shape::new(&[N, 1], DType::F32));
    let w1 = param(&mut g, "w1", &[IN, HID]);
    let w2 = param(&mut g, "w2", &[HID, 1]);
    let h = g.add_node(Op::MatMul, vec![x, w1], Shape::new(&[N, HID], DType::F32));
    let h = g.activation(Activation::Tanh, h, Shape::new(&[N, HID], DType::F32));
    let p = g.add_node(Op::MatMul, vec![h, w2], Shape::new(&[N, 1], DType::F32));
    let d = g.binary(BinaryOp::Sub, p, y, Shape::new(&[N, 1], DType::F32));
    let sq = g.binary(BinaryOp::Mul, d, d, Shape::new(&[N, 1], DType::F32));
    let loss = g.reduce(
        sq,
        ReduceOp::Mean,
        vec![0, 1],
        false,
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![loss]);
    let slots = vec![
        ParamSlot {
            name: "w1".into(),
            node: w1,
        },
        ParamSlot {
            name: "w2".into(),
            node: w2,
        },
    ];
    (g, slots)
}

fn data() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..N * IN)
        .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
        .collect();
    let y: Vec<f32> = (0..N)
        .map(|r| {
            let row = &x[r * IN..(r + 1) * IN];
            (row[0] * 1.7 - row[3]).sin() + 0.3 * row[5]
        })
        .collect();
    (x, y)
}

fn init() -> HashMap<String, Vec<f32>> {
    let w = |n: usize, k: f32| {
        (0..n)
            .map(|i| ((i * 7919 % 97) as f32 / 97.0 - 0.5) * k)
            .collect()
    };
    HashMap::from([("w1".into(), w(IN * HID, 0.8)), ("w2".into(), w(HID, 0.4))])
}

fn train(qat: Option<NumberFormat>, steps: usize) -> (HashMap<String, Vec<f32>>, f32, f32) {
    let (g, slots) = mlp_loss();
    let (x, y) = data();
    let mut cfg = DpConfig::new(0.01);
    cfg.qat = qat;
    let mut t = Trainer::new(g, &slots, &init(), steps, None, &cfg).unwrap();
    let mut batch =
        |_: usize, _: usize| vec![("x".to_string(), x.clone()), ("y".to_string(), y.clone())];
    let first = t.step(&mut batch).unwrap().loss;
    let mut last = first;
    for _ in 1..steps {
        last = t.step(&mut batch).unwrap().loss;
    }
    (t.params(), first, last)
}

/// The loss with the forward run in `format` (weights rounded on upload).
fn eval(params: &HashMap<String, Vec<f32>>, format: NumberFormat) -> f32 {
    let (g, _) = mlp_loss();
    let (x, y) = data();
    let mut cg =
        Session::new(Device::Cpu).compile_with(g, &CompileOptions::new().number_format(format));
    for (n, v) in params {
        cg.set_param(n, v);
    }
    cg.run(&[("x", &x), ("y", &y)]).remove(0)[0]
}

#[test]
fn qat_beats_post_training_rounding() {
    let f4: NumberFormat = "f4".parse().unwrap();
    let steps = 400;
    let (plain, p0, p1) = train(None, steps);
    let (qat, q0, q1) = train(Some(f4), steps);
    let plain_f4 = eval(&plain, f4);
    let qat_f4 = eval(&qat, f4);
    eprintln!(
        "f32 training: loss {p0:.4} -> {p1:.4}, evaluated in f4: {plain_f4:.4}\n\
         f4 QAT:       loss {q0:.4} -> {q1:.4}, evaluated in f4: {qat_f4:.4}"
    );
    assert!(q1 < 0.5 * q0, "QAT did not train: {q0} -> {q1}");
    assert!(
        qat_f4 < plain_f4,
        "QAT ({qat_f4}) must beat rounding an f32-trained model ({plain_f4}) in f4"
    );
}
