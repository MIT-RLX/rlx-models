//! `Op::Attention` prefill parity — wgpu vs CPU — for Gemma 4 E2B configs.
//!
//! E2B global (full-attention) layers: nh=8, nkv=1, head_dim=512. The builder
//! materializes K/V to 8 heads via `repeat_kv_packed` (concat 8 copies) before
//! the attention op, so the op sees MHA. This probe covers both the raw op-GQA
//! path and the E2B "concat-then-attend" path, at head_dim 256 (sliding) vs 512
//! (global).

use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

// mode: 0 = pass K/V directly (op does GQA if nkv<nh)
//       1 = concat `nh/nkv` copies of K/V first (E2B repeat_kv path → MHA)
fn run_case(nh: usize, nkv: usize, hd: usize, seq: usize, mode: u32) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let b = 1usize;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let group = nh / nkv;
    // `mode` high bits carry a magnitude multiplier (mag = mode >> 4) to probe
    // the sharp-softmax regime (Gemma4 scale=1.0 → logits ≈ hd).
    let mag = ((mode >> 4).max(1)) as f32;
    let mode = mode & 0xF;
    let q: Vec<f32> = (0..b * seq * q_dim)
        .map(|i| ((i as f32) * 0.011).sin() * mag)
        .collect();
    let k: Vec<f32> = (0..b * seq * kv_dim)
        .map(|i| ((i as f32) * 0.013).cos() * mag)
        .collect();
    let v: Vec<f32> = (0..b * seq * kv_dim)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();

    let mut g = Graph::new("attn");
    let qi = g.input("q", Shape::new(&[b, seq, q_dim], DType::F32));
    let ki = g.input("k", Shape::new(&[b, seq, kv_dim], DType::F32));
    let vi = g.input("v", Shape::new(&[b, seq, kv_dim], DType::F32));
    let (kk, vv) = if mode == 1 && group > 1 {
        // repeat_kv_packed: concat `group` copies of each kv head along last axis
        let kc = g.concat_(vec![ki; group], 2);
        let vc = g.concat_(vec![vi; group], 2);
        (kc, vc)
    } else {
        (ki, vi)
    };
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            mask_kind: MaskKind::Causal,
            score_scale: Some(1.0),
            attn_logit_softcap: None,
        },
        vec![qi, kk, vv],
        Shape::new(&[b, seq, q_dim], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.run(&[
            ("q", q.as_slice()),
            ("k", k.as_slice()),
            ("v", v.as_slice()),
        ])
        .remove(0)
    };
    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("attn mode={mode} nh={nh} nkv={nkv} hd={hd}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

fn run_rope(nh: usize, hd: usize, n_rot: usize, seq: usize) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Gpu) {
        return None;
    }
    let b = 1usize;
    let dim = nh * hd;
    let half = hd / 2; // cos/sin row stride is head_dim/2 (matches CPU tab_half)
    let x: Vec<f32> = (0..b * seq * dim)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let cs: Vec<f32> = (0..seq * half).map(|i| ((i as f32) * 0.02).cos()).collect();
    let sn: Vec<f32> = (0..seq * half).map(|i| ((i as f32) * 0.02).sin()).collect();
    let mut g = Graph::new("rope");
    let xi = g.input("x", Shape::new(&[b, seq, dim], DType::F32));
    let ci = g.input("cos", Shape::new(&[seq, half], DType::F32));
    let si = g.input("sin", Shape::new(&[seq, half], DType::F32));
    let y = g.rope_n(xi, ci, si, hd, n_rot);
    g.set_outputs(vec![y]);
    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.run(&[
            ("x", x.as_slice()),
            ("cos", cs.as_slice()),
            ("sin", sn.as_slice()),
        ])
        .remove(0)
    };
    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("rope nh={nh} hd={hd} n_rot={n_rot}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

fn run_rms(nh: usize, hd: usize, seq: usize) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Gpu) {
        return None;
    }
    let b = 1usize;
    let x: Vec<f32> = (0..b * seq * nh * hd)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let gamma: Vec<f32> = (0..hd).map(|i| 1.0 + ((i as f32) * 0.001).sin()).collect();
    let beta: Vec<f32> = vec![0.0; hd];
    let mut g = Graph::new("rms");
    let xi = g.input("x", Shape::new(&[b, seq, nh, hd], DType::F32));
    let gi = g.input("g", Shape::new(&[hd], DType::F32));
    let bi = g.input("b", Shape::new(&[hd], DType::F32));
    let y = g.rms_norm(xi, gi, bi, 1e-6);
    g.set_outputs(vec![y]);
    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.run(&[
            ("x", x.as_slice()),
            ("g", gamma.as_slice()),
            ("b", beta.as_slice()),
        ])
        .remove(0)
    };
    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("rms nh={nh} hd={hd}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

#[test]
fn wgpu_rms_norm_head_dim() {
    for &hd in &[256usize, 512] {
        if let Some(ma) = run_rms(8, hd, 5) {
            eprintln!(
                "  -> rms hd={hd}: {}",
                if ma <= 1e-2 { "OK" } else { "FAIL" }
            );
        }
    }
}

#[test]
fn wgpu_rope_head_dim() {
    // (hd, n_rot): full rotation (n_rot==hd) = sliding layers; partial
    // (n_rot<hd) = Gemma 4 E2B global layers (the suspect).
    let cases: &[(usize, usize)] = &[(256, 256), (512, 512), (512, 256), (512, 128)];
    for &(hd, n_rot) in cases {
        if let Some(ma) = run_rope(8, hd, n_rot, 5) {
            eprintln!(
                "  -> rope hd={hd} n_rot={n_rot}: {}",
                if ma <= 1e-2 { "OK" } else { "FAIL" }
            );
        }
    }
}

#[test]
fn wgpu_attention_e2b_paths() {
    // (nh, nkv, hd, mode)
    let cases: &[(usize, usize, usize, u32)] = &[
        (8, 8, 512, 0), // MHA hd512 (regression check)
        (8, 1, 256, 0), // op-GQA hd256 (was broken 1.9 → should be fixed)
        (8, 1, 512, 0), // op-GQA hd512 (was broken 1.2 → should be fixed)
        (8, 2, 512, 0), // op-GQA group=4 hd512
    ];
    for &(nh, nkv, hd, mode) in cases {
        let Some(max_abs) = run_case(nh, nkv, hd, 5, mode) else {
            return;
        };
        eprintln!(
            "  -> mode={mode} hd={hd}: {}",
            if max_abs <= 1e-2 { "OK" } else { "FAIL" }
        );
    }
}
