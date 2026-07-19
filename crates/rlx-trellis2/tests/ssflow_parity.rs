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

//! Real-weight parity: host DiT reference vs the upstream PyTorch dump.
//!
//! Gated on two env vars (skipped otherwise so CI stays green without the
//! 2.6 GB checkpoint):
//!   * `RLX_TRELLIS2_SSFLOW_CKPT` — path to `ss_flow_img_dit_1_3B_64_bf16.safetensors`
//!   * `RLX_TRELLIS2_SSFLOW_REF`  — path to the `ss_flow_ref.safetensors` golden dump
//!     produced by `scripts/ss_flow_ref.py`.

use rlx_trellis2::config::{DitConfig, DitKind};
use rlx_trellis2::dit_host::{DitDump, dit_forward};
use rlx_trellis2::rope::grid_coords;
use rlx_trellis2::sampler::{SamplerConfig, flow_euler_sample};
use safetensors::SafeTensors;
use std::path::Path;

fn ssflow_cfg() -> DitConfig {
    DitConfig {
        kind: DitKind::SparseStructureFlow,
        args: serde_json::from_str(
            r#"{"resolution":16,"in_channels":8,"out_channels":8,"model_channels":1536,
                "cond_channels":1024,"num_blocks":30,"num_heads":12,"mlp_ratio":5.3334,
                "pe_mode":"rope","share_mod":true,"initialization":"scaled",
                "qk_rms_norm":true,"qk_rms_norm_cross":true,"dtype":"bfloat16"}"#,
        )
        .unwrap(),
    }
}

fn read_f32(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|_| panic!("golden tensor {name} missing"));
    let bytes = t.data();
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v, t.shape().to_vec())
}

fn maxabs(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0.0f32;
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / a.len() as f32)
}

/// cosine similarity (a robust magnitude-independent parity check)
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[test]
fn ssflow_dit_matches_pytorch() {
    let (Ok(ckpt), Ok(refp)) = (
        std::env::var("RLX_TRELLIS2_SSFLOW_CKPT"),
        std::env::var("RLX_TRELLIS2_SSFLOW_REF"),
    ) else {
        eprintln!(
            "skipping ssflow parity: set RLX_TRELLIS2_SSFLOW_CKPT and RLX_TRELLIS2_SSFLOW_REF"
        );
        return;
    };

    // config: hard-coded to the 4B ss_flow checkpoint (matches ckpts/*.json).
    let cfg = ssflow_cfg();

    let wm = rlx_core::load_weight_map(Path::new(&ckpt), &[]).expect("load ckpt");

    let bytes = std::fs::read(&refp).expect("read golden");
    let st = SafeTensors::deserialize(&bytes).expect("parse golden");

    let (x, x_shape) = read_f32(&st, "x"); // [1, 8, 16,16,16]
    let (t, _) = read_f32(&st, "t");
    let (cond, cond_shape) = read_f32(&st, "cond"); // [1, Lc, 1024]
    let n_cond = cond_shape[1];
    let res = x_shape[2];
    let n_pos = res * res * res;
    let in_ch = x_shape[1];

    // x [1,C,D,H,W] -> tokens [n_pos, C] channels-last:  tok[p*C+c] = x[c*n_pos+p]
    let mut tokens = vec![0.0f32; n_pos * in_ch];
    for c in 0..in_ch {
        for p in 0..n_pos {
            tokens[p * in_ch + c] = x[c * n_pos + p];
        }
    }
    let coords = grid_coords(res);

    let mut dump = DitDump::default();
    let out = dit_forward(
        &cfg,
        &wm,
        &tokens,
        &coords,
        n_pos,
        &cond,
        n_cond,
        t[0],
        Some(&mut dump),
    )
    .expect("forward");

    // reference intermediates are [1, n_pos, C] channels-last already
    let (g_input, _) = read_f32(&st, "after_input");
    let (g_block0, _) = read_f32(&st, "after_block0");
    let (g_finalln, _) = read_f32(&st, "after_final_ln");
    let (g_out, _) = read_f32(&st, "out"); // [1,8,16,16,16]

    // our `out` is channels-last [n_pos, 8]; golden is [1,8,16,16,16] -> reorder
    let out_ch = 8usize;
    let mut out_cdhw = vec![0.0f32; n_pos * out_ch];
    for p in 0..n_pos {
        for c in 0..out_ch {
            out_cdhw[c * n_pos + p] = out[p * out_ch + c];
        }
    }

    let (mi, ai) = maxabs(&dump.after_input, &g_input);
    let (mb, ab) = maxabs(&dump.after_block0, &g_block0);
    let (mf, af) = maxabs(&dump.after_final_ln, &g_finalln);
    let (mo, ao) = maxabs(&out_cdhw, &g_out);
    println!(
        "after_input : maxabs {mi:.3e} mean {ai:.3e} cos {:.6}",
        cosine(&dump.after_input, &g_input)
    );
    println!(
        "after_block0: maxabs {mb:.3e} mean {ab:.3e} cos {:.6}",
        cosine(&dump.after_block0, &g_block0)
    );
    println!(
        "after_finaln: maxabs {mf:.3e} mean {af:.3e} cos {:.6}",
        cosine(&dump.after_final_ln, &g_finalln)
    );
    println!(
        "out         : maxabs {mo:.3e} mean {ao:.3e} cos {:.6}",
        cosine(&out_cdhw, &g_out)
    );

    // block0 activations are O(1e4) in magnitude; compare by cosine + relative.
    assert!(
        cosine(&dump.after_input, &g_input) > 0.99999,
        "after_input cos too low"
    );
    assert!(
        cosine(&dump.after_block0, &g_block0) > 0.9999,
        "after_block0 cos too low"
    );
    assert!(cosine(&out_cdhw, &g_out) > 0.9999, "out cos too low: {mo}");
}

/// Reshape a `[C, n_pos]` (C,D,H,W flattened) latent to channels-last tokens.
fn cdhw_to_tokens(x: &[f32], n_pos: usize, ch: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; n_pos * ch];
    for c in 0..ch {
        for p in 0..n_pos {
            t[p * ch + c] = x[c * n_pos + p];
        }
    }
    t
}

fn tokens_to_cdhw(t: &[f32], n_pos: usize, ch: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; n_pos * ch];
    for p in 0..n_pos {
        for c in 0..ch {
            x[c * n_pos + p] = t[p * ch + c];
        }
    }
    x
}

#[test]
fn ssflow_sampler_matches_pytorch() {
    let (Ok(ckpt), Ok(refp)) = (
        std::env::var("RLX_TRELLIS2_SSFLOW_CKPT"),
        std::env::var("RLX_TRELLIS2_SSFLOW_SAMPLE_REF"),
    ) else {
        eprintln!(
            "skipping sampler parity: set RLX_TRELLIS2_SSFLOW_CKPT + RLX_TRELLIS2_SSFLOW_SAMPLE_REF"
        );
        return;
    };
    let cfg = ssflow_cfg();
    let wm = rlx_core::load_weight_map(Path::new(&ckpt), &[]).expect("load ckpt");
    let bytes = std::fs::read(&refp).expect("read golden");
    let st = SafeTensors::deserialize(&bytes).expect("parse golden");

    let (noise, _) = read_f32(&st, "noise"); // [1,8,16,16,16]
    let (cond, cshape) = read_f32(&st, "cond");
    let (sample, _) = read_f32(&st, "sample");
    let n_cond = cshape[1];
    let res = 16usize;
    let n_pos = res * res * res;
    let in_ch = cfg.args.in_channels;
    let out_ch = cfg.args.out_channels;
    let coords = grid_coords(res);
    let neg_cond = vec![0.0f32; cond.len()];

    let scfg = SamplerConfig {
        sigma_min: 1e-5,
        steps: 12,
        guidance_strength: 7.5,
        guidance_rescale: 0.7,
        guidance_interval: [0.6, 1.0],
        rescale_t: 5.0,
    };

    let model_v = |x_t: &[f32], t_scaled: f32, cnd: &[f32]| -> Vec<f32> {
        let tokens = cdhw_to_tokens(x_t, n_pos, in_ch);
        let out = dit_forward(
            &cfg, &wm, &tokens, &coords, n_pos, cnd, n_cond, t_scaled, None,
        )
        .expect("forward");
        tokens_to_cdhw(&out, n_pos, out_ch)
    };

    let got = flow_euler_sample(model_v, &noise, &cond, &neg_cond, &scfg);
    let (m, a) = maxabs(&got, &sample);
    let c = cosine(&got, &sample);
    println!("sample: maxabs {m:.3e} mean {a:.3e} cos {c:.6}");
    assert!(c > 0.9999, "sampler cos too low: {c}");
    assert!(m < 5e-2, "sampler maxabs too high: {m}");
}
