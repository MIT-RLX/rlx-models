// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Phase-2 gate: the DINO loss differentiates and trains (via `rlx-tune`), the
//! projection head compiles/runs to the right shape, and the host-side
//! multi-crop + teacher-target helpers behave.

use std::collections::HashMap;

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompileOptions, Device, Session};
use rlx_tune::{Adam, ParamSlot, train};

use rlx_vit_elastic::dino::{
    CropConfig, DinoHeadConfig, Rng, build_dino_head, build_dino_loss, multi_crop, pair_mask,
    teacher_targets,
};

const F: DType = DType::F32;

#[test]
fn dino_loss_trains_student_toward_teacher() {
    let (ns, nt, k, ng) = (4usize, 2usize, 6usize, 2usize);
    let mut g = Graph::new("dino_loss_test");
    let student = g.param("student", Shape::new(&[ns, k], F));
    let tt = g.input("teacher_targets", Shape::new(&[nt, k], F));
    let pm = g.input("pair_mask", Shape::new(&[nt, ns], F));
    let (mask, active) = pair_mask(ng, ns);
    let loss = build_dino_loss(&mut g, student, tt, pm, 0.1, active);
    g.set_outputs(vec![loss]);

    // Teacher target rows: sharp distributions peaked at index t.
    let mut teacher_logits = vec![0.0f32; nt * k];
    for t in 0..nt {
        teacher_logits[t * k + t] = 6.0;
    }
    let targets = teacher_targets(&teacher_logits, nt, k, 0.04, &vec![0.0; k]);

    let mut params = HashMap::new();
    params.insert("student".to_string(), vec![0.0; ns * k]);
    let wrt = vec![ParamSlot {
        name: "student".into(),
        node: student,
    }];
    let inputs = vec![
        ("teacher_targets".to_string(), targets),
        ("pair_mask".to_string(), mask),
    ];
    let mut opt = Adam::new(0.2);
    let losses = train(g, &wrt, &mut params, &inputs, &mut opt, 150, None).unwrap();

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(first.is_finite() && last.is_finite());
    assert!(
        last < first * 0.5,
        "DINO loss did not decrease: {first} -> {last}"
    );
}

#[test]
fn dino_head_forward_shape() {
    let (n, in_dim, out_k) = (3usize, 16usize, 20usize);
    let cfg = DinoHeadConfig::small(in_dim, out_k);
    let mut g = Graph::new("head");
    let x = g.input("x", Shape::new(&[n, in_dim], F));
    let mut params = Vec::new();
    let y = build_dino_head(&mut g, x, &cfg, "head", &mut params);
    g.set_outputs(vec![y]);

    let init = rlx_vit_elastic::dino::init_head_params(&cfg, "head", 7);
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &CompileOptions::new());
    for p in &params {
        compiled.set_param(&p.name, &init[&p.name]);
    }
    let xd: Vec<f32> = (0..n * in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let out = compiled.run(&[("x", xd.as_slice())]);
    assert_eq!(out[0].len(), n * out_k);
    assert!(out[0].iter().all(|v| v.is_finite()));
}

#[test]
fn multi_crop_shapes() {
    let cfg = CropConfig {
        n_global: 2,
        n_local: 3,
        img_size: 32,
        ..Default::default()
    };
    let mut rng = Rng::new(1);
    let rgb: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 251) as u8).collect();
    let crops = multi_crop(&mut rng, &rgb, 64, 64, &cfg);
    assert_eq!(crops.len(), 5);
    for c in &crops {
        assert_eq!(c.len(), 3 * 32 * 32);
        assert!(c.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn teacher_targets_are_distributions() {
    let (rows, dim) = (3usize, 5usize);
    let logits: Vec<f32> = (0..rows * dim).map(|i| (i as f32 * 0.3).cos()).collect();
    let t = teacher_targets(&logits, rows, dim, 0.05, &vec![0.0; dim]);
    for r in 0..rows {
        let s: f32 = t[r * dim..(r + 1) * dim].iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "row {r} sums to {s}");
        assert!(t[r * dim..(r + 1) * dim].iter().all(|&v| v >= 0.0));
    }
}
