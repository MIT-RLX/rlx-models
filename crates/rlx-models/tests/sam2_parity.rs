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

//! SAM 2 image-encoder + decoder parity test against the official
//! `facebookresearch/sam2` PyTorch package.
//!
//! Unlike SAM v1, there is no Rust-side reference (candle has no SAM 2
//! implementation). The reference path runs as a subprocess invocation
//! of `tests/sam2_parity_helpers/dump_reference.py`, which dumps every
//! intermediate to a temp directory as raw f32 LE blobs. The Rust test
//! then loads them and compares against the rlx-models output
//! element-wise.
//!
//! Two reference transports, picked by env at test time:
//!
//!   - **Local Python** (default): `python` on `PATH` must have
//!     `sam2 + torch + numpy` installed. Easiest for one-off
//!     debugging on a dev box.
//!   - **Docker** (`RLX_SAM2_DOCKER=1`): hermetic build via
//!     `tests/sam2_parity_helpers/Dockerfile`. Recommended for CI and
//!     for cross-machine reproducibility — pins sam2 / torch versions
//!     so parity numbers stay stable when upstream pushes a release.
//!
//! Both transports support `RLX_SAM2_DEVICE=cpu|cuda`. Under Docker
//! the wrapper threads `--gpus all` through automatically when
//! `cuda` is selected (requires NVIDIA Container Toolkit on the host).
//!
//! ## Running — Docker, CPU
//!
//! ```bash
//! tests/sam2_parity_helpers/build.sh cpu
//! huggingface-cli download facebook/sam2-hiera-base-plus \
//!   sam2_hiera_base_plus.safetensors --local-dir /tmp/rlx_sam2
//!
//! RLX_SAM2_DOCKER=1 \
//!   RLX_SAM2_WEIGHTS=/tmp/rlx_sam2/sam2_hiera_base_plus.safetensors \
//!   RLX_SAM2_CONFIG=sam2_hiera_b+ \
//!   cargo test -p rlx-models --features parity-pytorch --release \
//!     sam2_encoder_parity_vs_pytorch -- --nocapture
//! ```
//!
//! ## Running — Docker, CUDA
//!
//! ```bash
//! tests/sam2_parity_helpers/build.sh gpu
//! RLX_SAM2_DOCKER=1 RLX_SAM2_DEVICE=cuda \
//!   RLX_SAM2_WEIGHTS=/tmp/rlx_sam2/sam2_hiera_base_plus.safetensors \
//!   RLX_SAM2_CONFIG=sam2_hiera_b+ \
//!   cargo test -p rlx-models --features parity-pytorch --release \
//!     sam2_encoder_parity_vs_pytorch -- --nocapture
//! ```
//!
//! ## Running — local Python (no Docker)
//!
//! ```bash
//! pip install sam2 torch numpy   # or use a venv
//! RLX_SAM2_WEIGHTS=/tmp/rlx_sam2/sam2_hiera_base_plus.safetensors \
//!   RLX_SAM2_CONFIG=sam2_hiera_b+ \
//!   cargo test -p rlx-models --features parity-pytorch --release \
//!     sam2_encoder_parity_vs_pytorch -- --nocapture
//! ```
//!
//! Without `RLX_SAM2_WEIGHTS` set, the test prints a skip notice and
//! returns Ok — same skip semantics as SAM v1's `sam_parity.rs`.

#![cfg(feature = "parity-pytorch")]
mod compile_support;

use anyhow::{Context, Result, anyhow, ensure};
use rlx_models::sam2::{
    SAM2_IMG_SIZE, SAM2_PROMPT_GRID, Sam2Config, Sam2HieraConfig, apply_fpn_neck,
    assemble_patch_tokens, build_sam2_image_encoder_graph, mask_decoder_forward,
    memory_attention_forward, memory_encoder_forward, preprocess_image, prompt_encoder_forward,
};
use rlx_models::{
    WeightMap,
    sam2::{
        mask_decoder::extract_mask_decoder_weights,
        memory_attention::extract_memory_attention_weights,
        memory_encoder::extract_memory_encoder_weights,
        prompt_encoder::{SAM2_MASK_IN_CHANS, extract_prompt_encoder_weights},
    },
};
use rlx_runtime::{Device, Session};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Cosine distance is the primary functional-parity gate: 1e-8 means
/// "directionally bit-perfect" — any larger and the model would
/// produce different downstream outputs. Max-abs-diff is a secondary
/// signal for spotting single-point outliers; on 50+-layer encoders
/// (Hiera-large) it can hit ~1e-2 even when cos_dist is ~1e-10 (a
/// few f32 accumulation outliers in an otherwise bit-perfect map).
const TOL_COS: f64 = 1e-7;
/// Loose cap on max-abs-diff for warnings only. Doesn't fail the
/// test — cos_dist is the gate.
const WARN_MAX_DIFF: f32 = 5e-2;
const TOL_MASK: f32 = 1e-2;
const TOL_IOU: f32 = 1e-3;

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_SAM2_WEIGHTS")
}

fn ref_config_name() -> String {
    rlx_ir::env::var("RLX_SAM2_CONFIG").unwrap_or_else(|| "sam2_hiera_b+".to_string())
}

fn cfg_for_ref_name(name: &str) -> Sam2HieraConfig {
    match name {
        "sam2_hiera_t" | "sam2_hiera_tiny" => Sam2HieraConfig::tiny(),
        "sam2_hiera_s" | "sam2_hiera_small" => Sam2HieraConfig::small(),
        "sam2_hiera_b+" | "sam2_hiera_base_plus" => Sam2HieraConfig::base_plus(),
        "sam2_hiera_l" | "sam2_hiera_large" => Sam2HieraConfig::large(),
        other => panic!("unknown sam2 config name: {other}"),
    }
}

/// Deterministic 1024×1024 RGB-u8 image — same sine recipe as the
/// DINOv2 / SAM v1 parity tests.
fn synthesize_image_u8() -> Vec<u8> {
    let n = SAM2_IMG_SIZE * SAM2_IMG_SIZE * 3;
    let mut v = vec![0u8; n];
    for y in 0..SAM2_IMG_SIZE {
        for x in 0..SAM2_IMG_SIZE {
            for c in 0..3 {
                let fx = x as f32 / SAM2_IMG_SIZE as f32;
                let fy = y as f32 / SAM2_IMG_SIZE as f32;
                let phase = (c as f32) * 0.7;
                let s = (6.28 * fx + phase).sin() * (3.14 * fy + phase).cos();
                let val = ((s + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                v[(y * SAM2_IMG_SIZE + x) * 3 + c] = val;
            }
        }
    }
    v
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0f32;
    let mut idx = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, idx)
}

/// Cosine distance = 1 - cos_similarity = 1 - (a·b) / (||a|| * ||b||).
/// Computed in f64 to avoid catastrophic cancellation on near-parallel
/// vectors. Reported alongside max-abs-diff for full parity picture.
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..a.len() {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (1.0 - dot / denom).max(0.0)
    }
}

fn read_f32_blob(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let mut f = fs::File::open(path).with_context(|| format!("opening reference blob {path:?}"))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() == expected_len * 4,
        "blob {path:?}: expected {} bytes (={} f32s), got {}",
        expected_len * 4,
        expected_len,
        bytes.len()
    );
    let mut out = vec![0f32; expected_len];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

fn write_f32_blob(path: &Path, data: &[f32]) -> Result<()> {
    let mut f = fs::File::create(path)?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn dump_reference(
    weights: &str,
    cfg_name: &str,
    image_nchw: &[f32],
    points: Option<(&[f32], &[f32])>,
) -> Result<PathBuf> {
    let tmp = env::temp_dir().join(format!("rlx_sam2_parity_{}", std::process::id()));
    fs::create_dir_all(&tmp)?;
    let img_bin = tmp.join("image.f32");
    write_f32_blob(&img_bin, image_nchw)?;

    // Two transports, picked via env:
    //   RLX_SAM2_DOCKER=1            → run via run-ref.sh (hermetic, GPU-capable)
    //   anything else (default)      → run via local `python` on PATH
    let use_docker = rlx_ir::env::var("RLX_SAM2_DOCKER").as_deref() == Some("1");
    let device = rlx_ir::env::var("RLX_SAM2_DEVICE").unwrap_or_else(|| "cpu".to_string());

    let mut cmd = if use_docker {
        let mut c = Command::new("bash");
        c.arg("tests/sam2_parity_helpers/run-ref.sh");
        c.env("RLX_SAM2_DEVICE", &device);
        // Image tag is overridable; default picked by run-ref.sh.
        if let Some(tag) = rlx_ir::env::var("RLX_SAM2_IMAGE_TAG") {
            c.env("RLX_SAM2_IMAGE_TAG", tag);
        }
        c
    } else {
        let mut c =
            Command::new(rlx_ir::env::var("RLX_SAM2_PYTHON").unwrap_or_else(|| "python".into()));
        c.arg("tests/sam2_parity_helpers/dump_reference.py");
        c.env("RLX_SAM2_DEVICE", &device);
        c
    };
    cmd.env("RLX_SAM2_WEIGHTS", weights)
        .env("RLX_SAM2_CONFIG", cfg_name)
        .env("RLX_SAM2_IMAGE_BIN", img_bin.as_os_str())
        .env("RLX_SAM2_OUT_DIR", tmp.as_os_str());
    if let Some((pts, lbls)) = points {
        let pts_bin = tmp.join("points.f32");
        let lbl_bin = tmp.join("labels.f32");
        write_f32_blob(&pts_bin, pts)?;
        write_f32_blob(&lbl_bin, lbls)?;
        cmd.env("RLX_SAM2_RUN_DECODER", "1")
            .env("RLX_SAM2_POINTS", pts_bin.as_os_str())
            .env("RLX_SAM2_LABELS", lbl_bin.as_os_str());
    }
    let status = cmd.status().with_context(|| {
        if use_docker {
            "spawning docker for sam2 reference dump (is `docker` on PATH and the image built? \
             run `tests/sam2_parity_helpers/build.sh` first)"
        } else {
            "spawning python for sam2 reference dump (is `python` on PATH with sam2 + torch?)"
        }
    })?;
    ensure!(
        status.success(),
        "reference dump subprocess failed: exit={:?}",
        status.code()
    );
    Ok(tmp)
}

fn run_rlx_encoder(
    weights: &str,
    cfg: &Sam2HieraConfig,
    image_nchw: &[f32],
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut wm = WeightMap::from_file(weights)?;
    let (graph, params, pre, fpn) = build_sam2_image_encoder_graph(cfg, &mut wm)?;
    let mut compiled = compile_support::compile_sam(Device::Cpu, graph, params);
    let hidden = assemble_patch_tokens(&pre, image_nchw)?;
    let stages = compiled.run(&[("hidden", hidden.as_slice())]);
    // FPN host-side.
    let stage_hw: Vec<(usize, usize)> = (0..cfg.stages.len())
        .map(|s| (cfg.grid_size_at_stage(s), cfg.grid_size_at_stage(s)))
        .collect();
    let stage_dims: Vec<usize> = (0..cfg.stages.len())
        .map(|s| cfg.embed_dim_at_stage(s))
        .collect();
    let mut fpn_ir = rlx_models::sam2::compile_fpn_neck_ir(
        &fpn,
        &stage_hw,
        &stage_dims,
        Device::Cpu,
        &rlx_flow::CompileProfile::sam2(),
    )?;
    let levels = apply_fpn_neck(&fpn, &mut fpn_ir, &stages, &stage_hw, &stage_dims)?;
    let fpn_flat: Vec<Vec<f32>> = levels.into_iter().map(|l| l.features).collect();
    Ok((stages, fpn_flat))
}

/// Reference dumps Hiera stage outputs as NCHW `[1, dim, h, w]`. Ours
/// come out as BHWC `[1, h, w, dim]` flattened. Convert for diff.
fn bhwc_to_nchw_flat(bhwc: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                out[ch * h * w + y * w + x] = bhwc[(y * w + x) * c + ch];
            }
        }
    }
    out
}

#[test]
fn sam2_encoder_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping — set RLX_SAM2_WEIGHTS=/path/to/sam2_hiera_*.safetensors");
        return Ok(());
    };
    let cfg_name = ref_config_name();
    let cfg = cfg_for_ref_name(&cfg_name);

    let image_u8 = synthesize_image_u8();
    let image_nchw = preprocess_image(&image_u8, SAM2_IMG_SIZE, SAM2_IMG_SIZE);

    let ref_dir = dump_reference(&weights, &cfg_name, &image_nchw, None)?;

    // ── Pre-encoder bisect: patch_embed alone, then patch+pos ──
    let mut wm = WeightMap::from_file(&weights)?;
    let (_g, _p, pre, _fpn) = build_sam2_image_encoder_graph(&cfg, &mut wm)?;
    // Reproduce host-side patch embed *without* adding position embedding.
    let mut patch_only = pre.pos_embed_full.clone();
    for v in patch_only.iter_mut() {
        *v = 0.0; // disable pos contribution
    }
    let pre_no_pos = rlx_models::sam2::Sam2PreprocessWeights {
        patch_proj_w: pre.patch_proj_w.clone(),
        patch_proj_b: pre.patch_proj_b.clone(),
        pos_embed_full: patch_only,
        embed_dim: pre.embed_dim,
        grid: pre.grid,
    };
    let patch_bhwc = assemble_patch_tokens(&pre_no_pos, &image_nchw)?;
    let grid = pre.grid;
    let dim0 = pre.embed_dim;
    let ref_patch = read_f32_blob(&ref_dir.join("patch_embed.f32"), dim0 * grid * grid)?;
    // Reference is NHWC [1, grid, grid, dim]; ours is BHWC same shape — flat layout matches.
    let (pd, pi) = max_abs_diff(&patch_bhwc, &ref_patch);
    eprintln!("[patch_embed bisect] max |Δ| = {pd:.4e} at idx {pi}");

    // Then with pos:
    let patch_plus_pos = assemble_patch_tokens(&pre, &image_nchw)?;
    let ref_pp = read_f32_blob(&ref_dir.join("patch_plus_pos.f32"), dim0 * grid * grid)?;
    let (ppd, ppi) = max_abs_diff(&patch_plus_pos, &ref_pp);
    eprintln!("[patch+pos bisect]    max |Δ| = {ppd:.4e} at idx {ppi}");

    // ── Block-0 substep bisect ──
    // Compute norm1(patch_plus_pos) by hand (LN over last axis, with
    // eps=1e-6 per cfg). Patch+pos is BHWC `[1, grid, grid, dim]`
    // flat; LN normalizes per spatial position over channel.
    let bisect_path = ref_dir.join("block0_post_norm1.f32");
    eprintln!(
        "[block0 bisect] checking {bisect_path:?} exists={}",
        bisect_path.exists()
    );
    if bisect_path.exists() {
        let mut wm_b0 = WeightMap::from_file(&weights)?;
        // Drain encoder body weights to position the cursor at block 0's norm1.
        // We re-extract just the keys we need.
        let (n1_g, _) = wm_b0.take("image_encoder.trunk.blocks.0.norm1.weight")?;
        let (n1_b, _) = wm_b0.take("image_encoder.trunk.blocks.0.norm1.bias")?;
        let dim = pre.embed_dim;
        let n_pos = grid * grid;
        let mut x = patch_plus_pos.clone();
        let eps = cfg.layer_norm_eps as f32;
        for p in 0..n_pos {
            let row = &mut x[p * dim..(p + 1) * dim];
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for k in 0..dim {
                row[k] = (row[k] - mean) * inv * n1_g[k] + n1_b[k];
            }
        }
        let ref_n1 = read_f32_blob(&bisect_path, dim * n_pos)?;
        let (nd, ni) = max_abs_diff(&x, &ref_n1);
        eprintln!("[block0 post_norm1]   max |Δ| = {nd:.4e} at idx {ni}");

        // ── window_partition bisect ──
        // Reference window_partition for tiny block 0 (ws=8, no pad):
        //   x: [1, 256, 256, 96] → reshape [1, 32, 8, 32, 8, 96]
        //   → permute(0,1,3,2,4,5) → [1, 32, 32, 8, 8, 96] → reshape [1024, 8, 8, 96]
        let ws = 8usize;
        let nh = grid / ws;
        let nw = grid / ws;
        let n_win = nh * nw;
        // Source x is [grid, grid, dim] flat (B=1 collapsed).
        // We use the just-normed `x` from above.
        let mut partitioned = vec![0f32; n_win * ws * ws * dim];
        for wi in 0..nh {
            for wj in 0..nw {
                let win_idx = wi * nw + wj;
                for ly in 0..ws {
                    for lx in 0..ws {
                        let gy = wi * ws + ly;
                        let gx = wj * ws + lx;
                        let src_off = (gy * grid + gx) * dim;
                        let dst_off = ((win_idx * ws + ly) * ws + lx) * dim;
                        partitioned[dst_off..dst_off + dim]
                            .copy_from_slice(&x[src_off..src_off + dim]);
                    }
                }
            }
        }
        let ref_part = read_f32_blob(
            &ref_dir.join("block0_post_partition.f32"),
            n_win * ws * ws * dim,
        )?;
        let (pd, pi) = max_abs_diff(&partitioned, &ref_part);
        eprintln!("[block0 post_partition] max |Δ| = {pd:.4e} at idx {pi}");

        // ── attn windowed bisect ──
        // Need QKV weights + proj weights. Load + apply via host-side
        // multi-head attention to the partitioned tensor.
        let (qkv_w, _) = wm_b0.take("image_encoder.trunk.blocks.0.attn.qkv.weight")?; // [3*dim, dim]
        let (qkv_b, _) = wm_b0.take("image_encoder.trunk.blocks.0.attn.qkv.bias")?;
        let (proj_w, _) = wm_b0.take("image_encoder.trunk.blocks.0.attn.proj.weight")?; // [dim, dim]
        let (proj_b, _) = wm_b0.take("image_encoder.trunk.blocks.0.attn.proj.bias")?;
        let num_heads = 1usize;
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let s_kv = ws * ws;

        // qkv linear: per token (B*S = n_win*s_kv tokens) compute Wx+b.
        let mut qkv = vec![0f32; n_win * s_kv * 3 * dim];
        for t in 0..n_win * s_kv {
            let src = &partitioned[t * dim..(t + 1) * dim];
            for o in 0..3 * dim {
                let mut acc = qkv_b[o];
                for k in 0..dim {
                    acc += src[k] * qkv_w[o * dim + k];
                }
                qkv[t * 3 * dim + o] = acc;
            }
        }
        // Reshape to (n_win, s_kv, 3, nh, dh) and split.
        let extract = |comp_idx: usize| -> Vec<f32> {
            // q: comp_idx=0, k=1, v=2
            let mut out = vec![0f32; n_win * s_kv * dim];
            for w in 0..n_win {
                for s in 0..s_kv {
                    for d_ in 0..dim {
                        out[(w * s_kv + s) * dim + d_] =
                            qkv[((w * s_kv + s) * 3 + comp_idx) * dim + d_];
                    }
                }
            }
            out
        };
        let q = extract(0);
        let k = extract(1);
        let v = extract(2);
        // Single head: scores = q @ k^T per window.
        let mut attn_out = vec![0f32; n_win * s_kv * dim];
        for w in 0..n_win {
            let mut scores = vec![0f32; s_kv * s_kv];
            for i in 0..s_kv {
                for j in 0..s_kv {
                    let mut acc = 0f32;
                    for d_ in 0..dim {
                        acc += q[(w * s_kv + i) * dim + d_] * k[(w * s_kv + j) * dim + d_];
                    }
                    scores[i * s_kv + j] = acc * scale;
                }
            }
            // Softmax over last axis.
            for i in 0..s_kv {
                let row = &mut scores[i * s_kv..(i + 1) * s_kv];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0f32;
                for vv in row.iter_mut() {
                    *vv = (*vv - m).exp();
                    s += *vv;
                }
                for vv in row.iter_mut() {
                    *vv /= s;
                }
            }
            // out = scores @ v
            for i in 0..s_kv {
                for d_ in 0..dim {
                    let mut acc = 0f32;
                    for j in 0..s_kv {
                        acc += scores[i * s_kv + j] * v[(w * s_kv + j) * dim + d_];
                    }
                    attn_out[(w * s_kv + i) * dim + d_] = acc;
                }
            }
        }
        // Output projection.
        let mut proj_out = vec![0f32; n_win * s_kv * dim];
        for t in 0..n_win * s_kv {
            let src = &attn_out[t * dim..(t + 1) * dim];
            for o in 0..dim {
                let mut acc = proj_b[o];
                for ki in 0..dim {
                    acc += src[ki] * proj_w[o * dim + ki];
                }
                proj_out[t * dim + o] = acc;
            }
        }
        // Compare against block0_post_attn_windowed.f32.
        // Reference's `attn` output shape is [n_win, ws, ws, dim] = [1024, 8, 8, 96] = same flat layout.
        let ref_attn = read_f32_blob(
            &ref_dir.join("block0_post_attn_windowed.f32"),
            n_win * s_kv * dim,
        )?;
        let (ad, ai) = max_abs_diff(&proj_out, &ref_attn);
        eprintln!("[block0 post_attn]    max |Δ| = {ad:.4e} at idx {ai}");

        // ── IR-only mask-unit-attention bisect ──
        // Build a minimal IR graph that runs JUST the mask-unit attention
        // (matmul → reshape→transpose→narrow split q/k/v → SDPA → output proj)
        // on the already-partitioned input. Same algorithm my handcoded
        // bisect just verified to 3.6e-7. If the IR version diverges, the
        // bug is in IR composition, not algorithm.
        use rlx_ir::infer::GraphExt as _;
        use rlx_ir::{DType as IrDType, Graph as IrGraph, Shape as IrShape};
        let mut gattn = IrGraph::new("sam2_block0_attn_isolated");
        let x_in = gattn.input("x", IrShape::new(&[n_win, ws, ws, dim], IrDType::F32));
        // qkv weight: transposed [dim, 3*dim] (row-major sgemm convention).
        let qkv_w_t: Vec<f32> = {
            let mut t = vec![0f32; dim * 3 * dim];
            for o in 0..3 * dim {
                for k in 0..dim {
                    t[k * 3 * dim + o] = qkv_w[o * dim + k];
                }
            }
            t
        };
        let qkv_w_p = gattn.param("qkv_w", IrShape::new(&[dim, 3 * dim], IrDType::F32));
        let qkv_b_p = gattn.param("qkv_b", IrShape::new(&[3 * dim], IrDType::F32));
        let qkv_mm = gattn.mm(x_in, qkv_w_p);
        let qkv_node = gattn.add(qkv_mm, qkv_b_p);
        // Reshape [n_win, ws, ws, 3*dim] → [n_win, s_kv, 3, nh, dh]
        let qkv5 = gattn.reshape_(
            qkv_node,
            vec![
                n_win as i64,
                s_kv as i64,
                3,
                num_heads as i64,
                head_dim as i64,
            ],
        );
        let qkv_perm = gattn.transpose_(qkv5, vec![2, 0, 1, 3, 4]); // [3, n_win, s_kv, nh, dh]
        let q_split = gattn.narrow_(qkv_perm, 0, 0, 1);
        let q_node = gattn.reshape_(
            q_split,
            vec![n_win as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        );
        let k_split = gattn.narrow_(qkv_perm, 0, 1, 1);
        let k_node = gattn.reshape_(
            k_split,
            vec![n_win as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        );
        let v_split = gattn.narrow_(qkv_perm, 0, 2, 1);
        let v_node = gattn.reshape_(
            v_split,
            vec![n_win as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        );
        // Transpose to [B, nh, S, dh] then flatten head into batch.
        let q_t = gattn.transpose_(q_node, vec![0, 2, 1, 3]);
        let k_t = gattn.transpose_(k_node, vec![0, 2, 1, 3]);
        let v_t = gattn.transpose_(v_node, vec![0, 2, 1, 3]);
        let q_flat = gattn.reshape_(
            q_t,
            vec![(n_win * num_heads) as i64, s_kv as i64, head_dim as i64],
        );
        let k_flat = gattn.reshape_(
            k_t,
            vec![(n_win * num_heads) as i64, s_kv as i64, head_dim as i64],
        );
        let v_flat = gattn.reshape_(
            v_t,
            vec![(n_win * num_heads) as i64, s_kv as i64, head_dim as i64],
        );
        // Scale + scores
        let scale_p = gattn.param("scale", IrShape::new(&[1], IrDType::F32));
        let q_scaled = gattn.mul(q_flat, scale_p);
        let k_for_mm = gattn.transpose_(k_flat, vec![0, 2, 1]); // [B·nh, dh, S]
        let scores = gattn.mm(q_scaled, k_for_mm);
        let attn_w = gattn.sm(scores, -1);
        let attn_v = gattn.mm(attn_w, v_flat);
        // Reshape back: [B·nh, S, dh] → [B, nh, S, dh] → [B, S, nh, dh] → [B, ws, ws, dim]
        let r = gattn.reshape_(
            attn_v,
            vec![n_win as i64, num_heads as i64, s_kv as i64, head_dim as i64],
        );
        let r = gattn.transpose_(r, vec![0, 2, 1, 3]);
        let merged = gattn.reshape_(r, vec![n_win as i64, ws as i64, ws as i64, dim as i64]);
        // Output projection.
        let proj_w_t: Vec<f32> = {
            let mut t = vec![0f32; dim * dim];
            for o in 0..dim {
                for k in 0..dim {
                    t[k * dim + o] = proj_w[o * dim + k];
                }
            }
            t
        };
        let proj_w_p = gattn.param("proj_w", IrShape::new(&[dim, dim], IrDType::F32));
        let proj_b_p = gattn.param("proj_b", IrShape::new(&[dim], IrDType::F32));
        let proj_mm = gattn.mm(merged, proj_w_p);
        let out_node = gattn.add(proj_mm, proj_b_p);
        gattn.set_outputs(vec![out_node]);

        let opts = rlx_models::flow_bridge::compile_options_sam_encoder(Device::Cpu);
        let mut compiled = Session::new(Device::Cpu).compile_with(gattn, &opts);
        compiled.set_param("qkv_w", &qkv_w_t);
        compiled.set_param("qkv_b", &qkv_b);
        compiled.set_param("proj_w", &proj_w_t);
        compiled.set_param("proj_b", &proj_b);
        compiled.set_param("scale", &[scale]);

        let outputs = compiled.run(&[("x", partitioned.as_slice())]);
        let ir_attn = outputs.into_iter().next().unwrap();
        let (iad, iai) = max_abs_diff(&ir_attn, &ref_attn);
        eprintln!("[block0 IR-attn]      max |Δ| = {iad:.4e} at idx {iai}");

        // ── IR-only window_partition bisect ──
        // Reshape [1, grid, grid, dim] → [1, nh, ws, nw, ws, dim]
        // → transpose(0,1,3,2,4,5) → [1, nh, nw, ws, ws, dim]
        // → reshape [n_win, ws, ws, dim].
        // Input: norm1(patch+pos) (bit-perfect to ref).
        let mut gp = IrGraph::new("sam2_block0_partition_isolated");
        let x_in = gp.input("x", IrShape::new(&[1, grid, grid, dim], IrDType::F32));
        let r5 = gp.reshape_(
            x_in,
            vec![1, nh as i64, ws as i64, nw as i64, ws as i64, dim as i64],
        );
        let perm = gp.transpose_(r5, vec![0, 1, 3, 2, 4, 5]);
        let wp = gp.reshape_(perm, vec![n_win as i64, ws as i64, ws as i64, dim as i64]);
        gp.set_outputs(vec![wp]);
        let opts = rlx_models::flow_bridge::compile_options_sam_encoder(Device::Cpu);
        let mut cp = Session::new(Device::Cpu).compile_with(gp, &opts);
        let outs = cp.run(&[("x", x.as_slice())]); // x = handcoded norm1 (bit-perfect)
        let ir_part = outs.into_iter().next().unwrap();
        let ref_part2 = read_f32_blob(
            &ref_dir.join("block0_post_partition.f32"),
            n_win * ws * ws * dim,
        )?;
        let (pd2, pi2) = max_abs_diff(&ir_part, &ref_part2);
        eprintln!("[block0 IR-partition] max |Δ| = {pd2:.4e} at idx {pi2}");

        // ── Full block-0 IR test (mirror multi_scale_block exactly) ──
        // If this matches stage_0 cleanly, the bug is in how multi-block
        // chaining happens in the full encoder.
        let mut gb = IrGraph::new("sam2_block0_full_isolated");
        let x_in = gb.input("x", IrShape::new(&[1, grid, grid, dim], IrDType::F32));
        let n1g_p = gb.param("n1g", IrShape::new(&[dim], IrDType::F32));
        let n1b_p = gb.param("n1b", IrShape::new(&[dim], IrDType::F32));
        let normed = gb.ln(x_in, n1g_p, n1b_p, eps);
        // Window-partition.
        let r5 = gb.reshape_(
            normed,
            vec![1, nh as i64, ws as i64, nw as i64, ws as i64, dim as i64],
        );
        let perm = gb.transpose_(r5, vec![0, 1, 3, 2, 4, 5]);
        let wnd = gb.reshape_(perm, vec![n_win as i64, ws as i64, ws as i64, dim as i64]);
        // mask_unit_attention.
        let qkv_w2 = gb.param("qkv_w", IrShape::new(&[dim, 3 * dim], IrDType::F32));
        let qkv_b2 = gb.param("qkv_b", IrShape::new(&[3 * dim], IrDType::F32));
        let qkv_mm = gb.mm(wnd, qkv_w2);
        let qkv_v = gb.add(qkv_mm, qkv_b2);
        let qkv5 = gb.reshape_(
            qkv_v,
            vec![
                n_win as i64,
                s_kv as i64,
                3,
                num_heads as i64,
                head_dim as i64,
            ],
        );
        let qkvp = gb.transpose_(qkv5, vec![2, 0, 1, 3, 4]);
        let qs = gb.narrow_(qkvp, 0, 0, 1);
        let q = gb.reshape_(
            qs,
            vec![n_win as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        );
        let ks = gb.narrow_(qkvp, 0, 1, 1);
        let kk = gb.reshape_(
            ks,
            vec![n_win as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        );
        let vs = gb.narrow_(qkvp, 0, 2, 1);
        let vv = gb.reshape_(
            vs,
            vec![n_win as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        );
        let qt = gb.transpose_(q, vec![0, 2, 1, 3]);
        let kt = gb.transpose_(kk, vec![0, 2, 1, 3]);
        let vt = gb.transpose_(vv, vec![0, 2, 1, 3]);
        let qfl = gb.reshape_(
            qt,
            vec![(n_win * num_heads) as i64, s_kv as i64, head_dim as i64],
        );
        let kfl = gb.reshape_(
            kt,
            vec![(n_win * num_heads) as i64, s_kv as i64, head_dim as i64],
        );
        let vfl = gb.reshape_(
            vt,
            vec![(n_win * num_heads) as i64, s_kv as i64, head_dim as i64],
        );
        let sp = gb.param("scale", IrShape::new(&[1], IrDType::F32));
        let qsc = gb.mul(qfl, sp);
        let kmm = gb.transpose_(kfl, vec![0, 2, 1]);
        let sc = gb.mm(qsc, kmm);
        let aw = gb.sm(sc, -1);
        let av = gb.mm(aw, vfl);
        let rsh = gb.reshape_(
            av,
            vec![n_win as i64, num_heads as i64, s_kv as i64, head_dim as i64],
        );
        let rsh = gb.transpose_(rsh, vec![0, 2, 1, 3]);
        let mer = gb.reshape_(rsh, vec![n_win as i64, ws as i64, ws as i64, dim as i64]);
        let pw = gb.param("proj_w", IrShape::new(&[dim, dim], IrDType::F32));
        let pb = gb.param("proj_b", IrShape::new(&[dim], IrDType::F32));
        let pmm = gb.mm(mer, pw);
        let attn_p = gb.add(pmm, pb);
        // window_unpartition: [n_win, ws, ws, dim] → [1, nh, nw, ws, ws, dim]
        //   → transpose(0,1,3,2,4,5) → [1, nh, ws, nw, ws, dim] → [1, grid, grid, dim]
        let r6 = gb.reshape_(
            attn_p,
            vec![1, nh as i64, nw as i64, ws as i64, ws as i64, dim as i64],
        );
        let p6 = gb.transpose_(r6, vec![0, 1, 3, 2, 4, 5]);
        let unp = gb.reshape_(p6, vec![1, grid as i64, grid as i64, dim as i64]);
        // residual
        let post_attn = gb.add(x_in, unp);
        // norm2
        let n2g_p = gb.param("n2g", IrShape::new(&[dim], IrDType::F32));
        let n2b_p = gb.param("n2b", IrShape::new(&[dim], IrDType::F32));
        let n2 = gb.ln(post_attn, n2g_p, n2b_p, eps);
        // MLP: dim → 4*dim → dim with GELU (erf).
        let hidden = 4 * dim;
        let m1w_p = gb.param("m1w", IrShape::new(&[dim, hidden], IrDType::F32));
        let m1b_p = gb.param("m1b", IrShape::new(&[hidden], IrDType::F32));
        let m2w_p = gb.param("m2w", IrShape::new(&[hidden, dim], IrDType::F32));
        let m2b_p = gb.param("m2b", IrShape::new(&[dim], IrDType::F32));
        let up_mm = gb.mm(n2, m1w_p);
        let up = gb.add(up_mm, m1b_p);
        let act = gb.gelu(up);
        let dw_mm = gb.mm(act, m2w_p);
        let dw = gb.add(dw_mm, m2b_p);
        let block_out = gb.add(post_attn, dw);
        gb.set_outputs(vec![block_out]);
        // Load weights via a fresh WeightMap.
        let mut wm2 = WeightMap::from_file(&weights)?;
        let (n1g, _) = wm2.take("image_encoder.trunk.blocks.0.norm1.weight")?;
        let (n1b, _) = wm2.take("image_encoder.trunk.blocks.0.norm1.bias")?;
        let (n2g, _) = wm2.take("image_encoder.trunk.blocks.0.norm2.weight")?;
        let (n2b, _) = wm2.take("image_encoder.trunk.blocks.0.norm2.bias")?;
        let (m1w, _) = wm2.take("image_encoder.trunk.blocks.0.mlp.layers.0.weight")?;
        let (m1b, _) = wm2.take("image_encoder.trunk.blocks.0.mlp.layers.0.bias")?;
        let (m2w, _) = wm2.take("image_encoder.trunk.blocks.0.mlp.layers.1.weight")?;
        let (m2b, _) = wm2.take("image_encoder.trunk.blocks.0.mlp.layers.1.bias")?;
        // Transpose row-major.
        let m1w_t: Vec<f32> = {
            let mut t = vec![0f32; dim * hidden];
            for o in 0..hidden {
                for k in 0..dim {
                    t[k * hidden + o] = m1w[o * dim + k];
                }
            }
            t
        };
        let m2w_t: Vec<f32> = {
            let mut t = vec![0f32; hidden * dim];
            for o in 0..dim {
                for k in 0..hidden {
                    t[k * dim + o] = m2w[o * hidden + k];
                }
            }
            t
        };
        let opts = rlx_models::flow_bridge::compile_options_sam_encoder(Device::Cpu);
        let mut cb = Session::new(Device::Cpu).compile_with(gb, &opts);
        cb.set_param("n1g", &n1g);
        cb.set_param("n1b", &n1b);
        cb.set_param("qkv_w", &qkv_w_t);
        cb.set_param("qkv_b", &qkv_b);
        cb.set_param("proj_w", &proj_w_t);
        cb.set_param("proj_b", &proj_b);
        cb.set_param("scale", &[scale]);
        cb.set_param("n2g", &n2g);
        cb.set_param("n2b", &n2b);
        cb.set_param("m1w", &m1w_t);
        cb.set_param("m1b", &m1b);
        cb.set_param("m2w", &m2w_t);
        cb.set_param("m2b", &m2b);
        let bo = cb.run(&[("x", patch_plus_pos.as_slice())]);
        let ir_block0 = bo.into_iter().next().unwrap();
        // Reference stage_0 is NCHW [1, 96, 256, 256] = same total size; ours is BHWC.
        // Transpose to NCHW for comparison.
        let ir_block0_nchw = bhwc_to_nchw_flat(&ir_block0, grid, grid, dim);
        let ref_s0 = read_f32_blob(&ref_dir.join("encoder_stage_0.f32"), dim * grid * grid)?;
        let (bd, bi) = max_abs_diff(&ir_block0_nchw, &ref_s0);
        eprintln!("[block0 IR-FULL]      max |Δ| = {bd:.4e} at idx {bi}");

        // ── Block-1 (Q-pool) substep bisect ──
        // Block 1 is the first q_pool boundary: dim 96 → 192, spatial
        // 256 → 128, num_heads 1 → 2. window_size_old = 8 (stage 0),
        // window_size_new = 8/2 = 4 (after q_pool).
        let blk1_input_path = ref_dir.join("block1_input.f32");
        if blk1_input_path.exists() {
            // Reference's block1_input is in NHWC [1, 256, 256, 96] flat.
            let n_in = grid * grid * dim;
            let ref_b1_input = read_f32_blob(&blk1_input_path, n_in)?;

            // Use the reference block 0 output as block 1's input (so we
            // compare block 1 in isolation, not its sensitivity to block 0).
            let dim_in_b1 = dim;
            let dim_out_b1 = dim * 2; // 192
            let num_heads_b1 = 2;
            let head_dim_b1 = dim_out_b1 / num_heads_b1; // 96 — same as block 0
            let scale_b1 = 1.0 / (head_dim_b1 as f32).sqrt();
            let ws1 = 8usize; // q_pool block reads from PRE-pool stage 0's ws
            let nh1 = grid / ws1; // 32
            let nw1 = grid / ws1; // 32
            let n_win_b1 = nh1 * nw1; // 1024
            let s_kv_b1 = ws1 * ws1; // 64

            // ── norm1 host-side compare ──
            let (n1g_b1, _) = wm2.take("image_encoder.trunk.blocks.1.norm1.weight")?;
            let (n1b_b1, _) = wm2.take("image_encoder.trunk.blocks.1.norm1.bias")?;
            let mut n1_b1 = ref_b1_input.clone();
            let n_pos_b1 = grid * grid;
            for p in 0..n_pos_b1 {
                let row = &mut n1_b1[p * dim_in_b1..(p + 1) * dim_in_b1];
                let mean: f32 = row.iter().sum::<f32>() / dim_in_b1 as f32;
                let var: f32 =
                    row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim_in_b1 as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for k in 0..dim_in_b1 {
                    row[k] = (row[k] - mean) * inv * n1g_b1[k] + n1b_b1[k];
                }
            }
            let ref_n1_b1 = read_f32_blob(
                &ref_dir.join("block1_post_norm1.f32"),
                dim_in_b1 * grid * grid,
            )?;
            let (d, i) = max_abs_diff(&n1_b1, &ref_n1_b1);
            eprintln!("[block1 post_norm1]   max |Δ| = {d:.4e} at idx {i}");

            // ── shortcut: proj(normed) then max-pool 2x2 ──
            let (proj_w_b1, _) = wm2.take("image_encoder.trunk.blocks.1.proj.weight")?; // [dim_out, dim_in]
            let (proj_b_b1, _) = wm2.take("image_encoder.trunk.blocks.1.proj.bias")?;
            let mut proj_out_b1 = vec![0f32; n_pos_b1 * dim_out_b1];
            for t in 0..n_pos_b1 {
                let src = &n1_b1[t * dim_in_b1..(t + 1) * dim_in_b1];
                for o in 0..dim_out_b1 {
                    let mut acc = proj_b_b1[o];
                    for k in 0..dim_in_b1 {
                        acc += src[k] * proj_w_b1[o * dim_in_b1 + k];
                    }
                    proj_out_b1[t * dim_out_b1 + o] = acc;
                }
            }
            let ref_proj_b1 = read_f32_blob(
                &ref_dir.join("block1_shortcut_pre_pool.f32"),
                n_pos_b1 * dim_out_b1,
            )?;
            let (d, i) = max_abs_diff(&proj_out_b1, &ref_proj_b1);
            eprintln!("[block1 shortcut_proj] max |Δ| = {d:.4e} at idx {i}");

            // Max-pool 2x2 stride 2 over BHWC spatial dims.
            let g2 = grid / 2; // 128
            let mut shortcut_pooled = vec![f32::NEG_INFINITY; g2 * g2 * dim_out_b1];
            for y in 0..g2 {
                for x in 0..g2 {
                    for c in 0..dim_out_b1 {
                        let mut m = f32::NEG_INFINITY;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let sy = y * 2 + dy;
                                let sx = x * 2 + dx;
                                let v = proj_out_b1[(sy * grid + sx) * dim_out_b1 + c];
                                if v > m {
                                    m = v;
                                }
                            }
                        }
                        shortcut_pooled[(y * g2 + x) * dim_out_b1 + c] = m;
                    }
                }
            }
            let ref_shortcut = read_f32_blob(
                &ref_dir.join("block1_shortcut_pooled.f32"),
                g2 * g2 * dim_out_b1,
            )?;
            let (d, i) = max_abs_diff(&shortcut_pooled, &ref_shortcut);
            eprintln!("[block1 shortcut_pool] max |Δ| = {d:.4e} at idx {i}");

            // ── Host-side block-1 attention (q_pool=true) bisect ──
            // window_partition: same recipe as block 0 but with input
            // = n1_b1 and ws=8 (still stage 0 windowing).
            let mut partitioned_b1 = vec![0f32; n_win_b1 * s_kv_b1 * dim_in_b1];
            for wi in 0..nh1 {
                for wj in 0..nw1 {
                    let win_idx = wi * nw1 + wj;
                    for ly in 0..ws1 {
                        for lx in 0..ws1 {
                            let gy = wi * ws1 + ly;
                            let gx = wj * ws1 + lx;
                            let src_off = (gy * grid + gx) * dim_in_b1;
                            let dst_off = ((win_idx * ws1 + ly) * ws1 + lx) * dim_in_b1;
                            partitioned_b1[dst_off..dst_off + dim_in_b1]
                                .copy_from_slice(&n1_b1[src_off..src_off + dim_in_b1]);
                        }
                    }
                }
            }
            // qkv linear: dim_in → 3·dim_out
            let (qkv_w_b1, _) = wm2.take("image_encoder.trunk.blocks.1.attn.qkv.weight")?;
            let (qkv_b_b1, _) = wm2.take("image_encoder.trunk.blocks.1.attn.qkv.bias")?;
            let mut qkv_b1 = vec![0f32; n_win_b1 * s_kv_b1 * 3 * dim_out_b1];
            for t in 0..n_win_b1 * s_kv_b1 {
                let src = &partitioned_b1[t * dim_in_b1..(t + 1) * dim_in_b1];
                for o in 0..3 * dim_out_b1 {
                    let mut acc = qkv_b_b1[o];
                    for k in 0..dim_in_b1 {
                        acc += src[k] * qkv_w_b1[o * dim_in_b1 + k];
                    }
                    qkv_b1[t * 3 * dim_out_b1 + o] = acc;
                }
            }
            // Reshape to (n_win, s_kv, 3, nh, dh). Extract q/k/v.
            let extract_b1 = |comp_idx: usize| -> Vec<f32> {
                let mut out = vec![0f32; n_win_b1 * s_kv_b1 * dim_out_b1];
                for w in 0..n_win_b1 {
                    for s in 0..s_kv_b1 {
                        for d_ in 0..dim_out_b1 {
                            out[(w * s_kv_b1 + s) * dim_out_b1 + d_] =
                                qkv_b1[((w * s_kv_b1 + s) * 3 + comp_idx) * dim_out_b1 + d_];
                        }
                    }
                }
                out
            };
            let q_b1 = extract_b1(0);
            let k_b1 = extract_b1(1);
            let v_b1 = extract_b1(2);
            // Q-pool: reshape Q to [n_win, ws1, ws1, dim_out], max-pool 2x2.
            let ws1_new = ws1 / 2; // 4
            let s_q_b1 = ws1_new * ws1_new; // 16
            let mut q_pooled = vec![f32::NEG_INFINITY; n_win_b1 * s_q_b1 * dim_out_b1];
            for w in 0..n_win_b1 {
                for y in 0..ws1_new {
                    for x in 0..ws1_new {
                        for c in 0..dim_out_b1 {
                            let mut m = f32::NEG_INFINITY;
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let sy = y * 2 + dy;
                                    let sx = x * 2 + dx;
                                    let v = q_b1[((w * ws1 + sy) * ws1 + sx) * dim_out_b1 + c];
                                    if v > m {
                                        m = v;
                                    }
                                }
                            }
                            q_pooled[((w * ws1_new + y) * ws1_new + x) * dim_out_b1 + c] = m;
                        }
                    }
                }
            }
            // Multi-head attention per window: nh=2, dh=96.
            // q_pooled: [n_win, s_q, nh*dh]  (flat). k, v: [n_win, s_kv, nh*dh].
            let mut attn_out_b1 = vec![0f32; n_win_b1 * s_q_b1 * dim_out_b1];
            for w in 0..n_win_b1 {
                for h in 0..num_heads_b1 {
                    let mut scores = vec![0f32; s_q_b1 * s_kv_b1];
                    for i in 0..s_q_b1 {
                        for j in 0..s_kv_b1 {
                            let mut acc = 0f32;
                            for d_ in 0..head_dim_b1 {
                                let q_off = (w * s_q_b1 + i) * dim_out_b1 + h * head_dim_b1 + d_;
                                let k_off = (w * s_kv_b1 + j) * dim_out_b1 + h * head_dim_b1 + d_;
                                acc += q_pooled[q_off] * k_b1[k_off];
                            }
                            scores[i * s_kv_b1 + j] = acc * scale_b1;
                        }
                    }
                    for i in 0..s_q_b1 {
                        let row = &mut scores[i * s_kv_b1..(i + 1) * s_kv_b1];
                        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut s = 0f32;
                        for vv in row.iter_mut() {
                            *vv = (*vv - m).exp();
                            s += *vv;
                        }
                        for vv in row.iter_mut() {
                            *vv /= s;
                        }
                    }
                    for i in 0..s_q_b1 {
                        for d_ in 0..head_dim_b1 {
                            let mut acc = 0f32;
                            for j in 0..s_kv_b1 {
                                let v_off = (w * s_kv_b1 + j) * dim_out_b1 + h * head_dim_b1 + d_;
                                acc += scores[i * s_kv_b1 + j] * v_b1[v_off];
                            }
                            attn_out_b1[(w * s_q_b1 + i) * dim_out_b1 + h * head_dim_b1 + d_] = acc;
                        }
                    }
                }
            }
            // Output projection.
            let (proj_w_attn_b1, _) = wm2.take("image_encoder.trunk.blocks.1.attn.proj.weight")?;
            let (proj_b_attn_b1, _) = wm2.take("image_encoder.trunk.blocks.1.attn.proj.bias")?;
            let mut proj_out_attn_b1 = vec![0f32; n_win_b1 * s_q_b1 * dim_out_b1];
            for t in 0..n_win_b1 * s_q_b1 {
                let src = &attn_out_b1[t * dim_out_b1..(t + 1) * dim_out_b1];
                for o in 0..dim_out_b1 {
                    let mut acc = proj_b_attn_b1[o];
                    for k in 0..dim_out_b1 {
                        acc += src[k] * proj_w_attn_b1[o * dim_out_b1 + k];
                    }
                    proj_out_attn_b1[t * dim_out_b1 + o] = acc;
                }
            }
            // Reference's block1_post_attn_windowed is shape [n_win, ws_new, ws_new, dim_out].
            let ref_attn_b1 = read_f32_blob(
                &ref_dir.join("block1_post_attn_windowed.f32"),
                n_win_b1 * s_q_b1 * dim_out_b1,
            )?;
            let (d, i) = max_abs_diff(&proj_out_attn_b1, &ref_attn_b1);
            eprintln!("[block1 post_attn]    max |Δ| = {d:.4e} at idx {i}");

            // ── Full block-1 IR isolated test ──
            // Use the production multi_scale_block via a 1-block encoder
            // graph would be complex; instead reuse the in-place IR
            // sequence with q_pool=true, mirroring multi_scale_block.
            let mut gb1 = IrGraph::new("sam2_block1_full_isolated");
            let g_in = gb1.input("x", IrShape::new(&[1, grid, grid, dim_in_b1], IrDType::F32));
            let n1g_p1 = gb1.param("n1g", IrShape::new(&[dim_in_b1], IrDType::F32));
            let n1b_p1 = gb1.param("n1b", IrShape::new(&[dim_in_b1], IrDType::F32));
            let normed_1 = gb1.ln(g_in, n1g_p1, n1b_p1, eps);
            // Shortcut: proj(normed) then qpool 2x2.
            let proj_w_pp = gb1.param(
                "shortcut_proj_w",
                IrShape::new(&[dim_in_b1, dim_out_b1], IrDType::F32),
            );
            let proj_b_pp = gb1.param("shortcut_proj_b", IrShape::new(&[dim_out_b1], IrDType::F32));
            let proj_mm_1 = gb1.mm(normed_1, proj_w_pp);
            let projected_1 = gb1.add(proj_mm_1, proj_b_pp);
            // qpool 2x2 over the FULL spatial grid (b=1 here).
            let qpool_rs = gb1.reshape_(
                projected_1,
                vec![
                    1,
                    (grid / 2) as i64,
                    2,
                    (grid / 2) as i64,
                    2,
                    dim_out_b1 as i64,
                ],
            );
            let qpool_red = gb1.reduce(
                qpool_rs,
                rlx_ir::op::ReduceOp::Max,
                vec![2, 4],
                false,
                IrShape::new(&[1, grid / 2, grid / 2, dim_out_b1], IrDType::F32),
            );
            let shortcut_1 = qpool_red;
            // Window-partition normed_1.
            let r5 = gb1.reshape_(
                normed_1,
                vec![
                    1,
                    nh1 as i64,
                    ws1 as i64,
                    nw1 as i64,
                    ws1 as i64,
                    dim_in_b1 as i64,
                ],
            );
            let perm = gb1.transpose_(r5, vec![0, 1, 3, 2, 4, 5]);
            let wnd = gb1.reshape_(
                perm,
                vec![n_win_b1 as i64, ws1 as i64, ws1 as i64, dim_in_b1 as i64],
            );
            // mask_unit_attention with q_pool=true.
            let qkv_w_t_b1: Vec<f32> = {
                let mut t = vec![0f32; dim_in_b1 * 3 * dim_out_b1];
                for o in 0..3 * dim_out_b1 {
                    for k in 0..dim_in_b1 {
                        t[k * 3 * dim_out_b1 + o] = qkv_w_b1[o * dim_in_b1 + k];
                    }
                }
                t
            };
            let qkv_wp = gb1.param(
                "qkv_w",
                IrShape::new(&[dim_in_b1, 3 * dim_out_b1], IrDType::F32),
            );
            let qkv_bp = gb1.param("qkv_b", IrShape::new(&[3 * dim_out_b1], IrDType::F32));
            let qkv_mm = gb1.mm(wnd, qkv_wp);
            let qkv_v = gb1.add(qkv_mm, qkv_bp);
            let qkv5 = gb1.reshape_(
                qkv_v,
                vec![
                    n_win_b1 as i64,
                    s_kv_b1 as i64,
                    3,
                    num_heads_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            let qkvp = gb1.transpose_(qkv5, vec![2, 0, 1, 3, 4]);
            let qs = gb1.narrow_(qkvp, 0, 0, 1);
            let q = gb1.reshape_(
                qs,
                vec![
                    n_win_b1 as i64,
                    s_kv_b1 as i64,
                    num_heads_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            let ks = gb1.narrow_(qkvp, 0, 1, 1);
            let kk = gb1.reshape_(
                ks,
                vec![
                    n_win_b1 as i64,
                    s_kv_b1 as i64,
                    num_heads_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            let vs = gb1.narrow_(qkvp, 0, 2, 1);
            let vv = gb1.reshape_(
                vs,
                vec![
                    n_win_b1 as i64,
                    s_kv_b1 as i64,
                    num_heads_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            // Q-pool: reshape back to [n_win, ws, ws, dim_out] then qpool.
            let q_spat = gb1.reshape_(
                q,
                vec![n_win_b1 as i64, ws1 as i64, ws1 as i64, dim_out_b1 as i64],
            );
            let ws1_new = ws1 / 2; // 4
            let s_q_b1 = ws1_new * ws1_new; // 16
            let q_pool_rs = gb1.reshape_(
                q_spat,
                vec![
                    n_win_b1 as i64,
                    ws1_new as i64,
                    2,
                    ws1_new as i64,
                    2,
                    dim_out_b1 as i64,
                ],
            );
            let q_pool_red = gb1.reduce(
                q_pool_rs,
                rlx_ir::op::ReduceOp::Max,
                vec![2, 4],
                false,
                IrShape::new(&[n_win_b1, ws1_new, ws1_new, dim_out_b1], IrDType::F32),
            );
            let q_pooled = gb1.reshape_(
                q_pool_red,
                vec![
                    n_win_b1 as i64,
                    s_q_b1 as i64,
                    num_heads_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            // Transpose Q/K/V to [B, nh, S, dh] then merge head into batch.
            let qt = gb1.transpose_(q_pooled, vec![0, 2, 1, 3]);
            let kt = gb1.transpose_(kk, vec![0, 2, 1, 3]);
            let vt = gb1.transpose_(vv, vec![0, 2, 1, 3]);
            let qfl = gb1.reshape_(
                qt,
                vec![
                    (n_win_b1 * num_heads_b1) as i64,
                    s_q_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            let kfl = gb1.reshape_(
                kt,
                vec![
                    (n_win_b1 * num_heads_b1) as i64,
                    s_kv_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            let vfl = gb1.reshape_(
                vt,
                vec![
                    (n_win_b1 * num_heads_b1) as i64,
                    s_kv_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            let sp = gb1.param("scale", IrShape::new(&[1], IrDType::F32));
            let qsc = gb1.mul(qfl, sp);
            let kmm = gb1.transpose_(kfl, vec![0, 2, 1]);
            let sc = gb1.mm(qsc, kmm);
            let aw = gb1.sm(sc, -1);
            let av = gb1.mm(aw, vfl);
            let rsh = gb1.reshape_(
                av,
                vec![
                    n_win_b1 as i64,
                    num_heads_b1 as i64,
                    s_q_b1 as i64,
                    head_dim_b1 as i64,
                ],
            );
            let rsh = gb1.transpose_(rsh, vec![0, 2, 1, 3]);
            let mer = gb1.reshape_(
                rsh,
                vec![
                    n_win_b1 as i64,
                    ws1_new as i64,
                    ws1_new as i64,
                    dim_out_b1 as i64,
                ],
            );
            let proj_w_attn_b1_t: Vec<f32> = {
                let mut t = vec![0f32; dim_out_b1 * dim_out_b1];
                for o in 0..dim_out_b1 {
                    for k in 0..dim_out_b1 {
                        t[k * dim_out_b1 + o] = proj_w_attn_b1[o * dim_out_b1 + k];
                    }
                }
                t
            };
            let pw = gb1.param(
                "attn_proj_w",
                IrShape::new(&[dim_out_b1, dim_out_b1], IrDType::F32),
            );
            let pb = gb1.param("attn_proj_b", IrShape::new(&[dim_out_b1], IrDType::F32));
            let pmm = gb1.mm(mer, pw);
            let attn_p = gb1.add(pmm, pb);
            // window_unpartition with ws_new.
            let grid_new = grid / 2; // 128
            let r6 = gb1.reshape_(
                attn_p,
                vec![
                    1,
                    nh1 as i64,
                    nw1 as i64,
                    ws1_new as i64,
                    ws1_new as i64,
                    dim_out_b1 as i64,
                ],
            );
            let p6 = gb1.transpose_(r6, vec![0, 1, 3, 2, 4, 5]);
            let unp = gb1.reshape_(
                p6,
                vec![1, grid_new as i64, grid_new as i64, dim_out_b1 as i64],
            );
            let post_attn_1 = gb1.add(shortcut_1, unp);
            // norm2 + MLP.
            let (n2g_b1, _) = wm2.take("image_encoder.trunk.blocks.1.norm2.weight")?;
            let (n2b_b1, _) = wm2.take("image_encoder.trunk.blocks.1.norm2.bias")?;
            let (m1w_b1, _) = wm2.take("image_encoder.trunk.blocks.1.mlp.layers.0.weight")?;
            let (m1b_b1, _) = wm2.take("image_encoder.trunk.blocks.1.mlp.layers.0.bias")?;
            let (m2w_b1, _) = wm2.take("image_encoder.trunk.blocks.1.mlp.layers.1.weight")?;
            let (m2b_b1, _) = wm2.take("image_encoder.trunk.blocks.1.mlp.layers.1.bias")?;
            let hidden_b1 = 4 * dim_out_b1;
            let m1w_b1_t: Vec<f32> = {
                let mut t = vec![0f32; dim_out_b1 * hidden_b1];
                for o in 0..hidden_b1 {
                    for k in 0..dim_out_b1 {
                        t[k * hidden_b1 + o] = m1w_b1[o * dim_out_b1 + k];
                    }
                }
                t
            };
            let m2w_b1_t: Vec<f32> = {
                let mut t = vec![0f32; hidden_b1 * dim_out_b1];
                for o in 0..dim_out_b1 {
                    for k in 0..hidden_b1 {
                        t[k * dim_out_b1 + o] = m2w_b1[o * hidden_b1 + k];
                    }
                }
                t
            };
            let n2g_pp = gb1.param("n2g", IrShape::new(&[dim_out_b1], IrDType::F32));
            let n2b_pp = gb1.param("n2b", IrShape::new(&[dim_out_b1], IrDType::F32));
            let m1w_pp = gb1.param("m1w", IrShape::new(&[dim_out_b1, hidden_b1], IrDType::F32));
            let m1b_pp = gb1.param("m1b", IrShape::new(&[hidden_b1], IrDType::F32));
            let m2w_pp = gb1.param("m2w", IrShape::new(&[hidden_b1, dim_out_b1], IrDType::F32));
            let m2b_pp = gb1.param("m2b", IrShape::new(&[dim_out_b1], IrDType::F32));
            let n2 = gb1.ln(post_attn_1, n2g_pp, n2b_pp, eps);
            let up_mm = gb1.mm(n2, m1w_pp);
            let up = gb1.add(up_mm, m1b_pp);
            let act = gb1.gelu(up);
            let dw_mm = gb1.mm(act, m2w_pp);
            let dw = gb1.add(dw_mm, m2b_pp);
            let block1_out = gb1.add(post_attn_1, dw);
            gb1.set_outputs(vec![block1_out]);

            let opts = rlx_models::flow_bridge::compile_options_sam_encoder(Device::Cpu);
            let mut cb1 = Session::new(Device::Cpu).compile_with(gb1, &opts);
            let shortcut_proj_w_t: Vec<f32> = {
                let mut t = vec![0f32; dim_in_b1 * dim_out_b1];
                for o in 0..dim_out_b1 {
                    for k in 0..dim_in_b1 {
                        t[k * dim_out_b1 + o] = proj_w_b1[o * dim_in_b1 + k];
                    }
                }
                t
            };
            cb1.set_param("n1g", &n1g_b1);
            cb1.set_param("n1b", &n1b_b1);
            cb1.set_param("shortcut_proj_w", &shortcut_proj_w_t);
            cb1.set_param("shortcut_proj_b", &proj_b_b1);
            cb1.set_param("qkv_w", &qkv_w_t_b1);
            cb1.set_param("qkv_b", &qkv_b_b1);
            cb1.set_param("scale", &[scale_b1]);
            cb1.set_param("attn_proj_w", &proj_w_attn_b1_t);
            cb1.set_param("attn_proj_b", &proj_b_attn_b1);
            cb1.set_param("n2g", &n2g_b1);
            cb1.set_param("n2b", &n2b_b1);
            cb1.set_param("m1w", &m1w_b1_t);
            cb1.set_param("m1b", &m1b_b1);
            cb1.set_param("m2w", &m2w_b1_t);
            cb1.set_param("m2b", &m2b_b1);

            let outs = cb1.run(&[("x", ref_b1_input.as_slice())]);
            let ir_b1 = outs.into_iter().next().unwrap();
            let ref_b1_out = read_f32_blob(
                &ref_dir.join("block1_output.f32"),
                ((grid / 2) * (grid / 2)) * dim_out_b1,
            )?;
            let (d, i) = max_abs_diff(&ir_b1, &ref_b1_out);
            eprintln!("[block1 IR-FULL]      max |Δ| = {d:.4e} at idx {i}");
        }
    }

    let (rlx_stages, rlx_fpn) = run_rlx_encoder(&weights, &cfg, &image_nchw)?;

    // ── Encoder stages ── log all then assert (bisect-friendly).
    let mut worst_stage_cos: (usize, f64) = (0, 0.0);
    let mut worst_stage_diff: (usize, f32) = (0, 0.0);
    for (s, rlx) in rlx_stages.iter().enumerate() {
        let h = cfg.grid_size_at_stage(s);
        let w = h;
        let dim = cfg.embed_dim_at_stage(s);
        let ref_blob = read_f32_blob(&ref_dir.join(format!("encoder_stage_{s}.f32")), dim * h * w)?;
        let rlx_nchw = bhwc_to_nchw_flat(rlx, h, w, dim);
        let (diff, idx) = max_abs_diff(&rlx_nchw, &ref_blob);
        let cos = cosine_distance(&rlx_nchw, &ref_blob);
        eprintln!(
            "[sam2 encoder stage {s}] {} f32 values; max |Δ| = {diff:.4e} at idx {idx}; cos_dist = {cos:.3e}",
            ref_blob.len()
        );
        if cos > worst_stage_cos.1 {
            worst_stage_cos = (s, cos);
        }
        if diff > worst_stage_diff.1 {
            worst_stage_diff = (s, diff);
        }
    }

    // ── FPN levels ──
    let mut worst_fpn_cos: (usize, f64) = (0, 0.0);
    let mut worst_fpn_diff: (usize, f32) = (0, 0.0);
    let ref_num_fpn = (0..rlx_fpn.len())
        .take_while(|i| ref_dir.join(format!("fpn_level_{i}.f32")).exists())
        .count();
    for (i, rlx) in rlx_fpn.iter().take(ref_num_fpn).enumerate() {
        let ref_blob = read_f32_blob(&ref_dir.join(format!("fpn_level_{i}.f32")), rlx.len())?;
        let (diff, idx) = max_abs_diff(rlx, &ref_blob);
        let cos = cosine_distance(rlx, &ref_blob);
        eprintln!(
            "[sam2 fpn lvl {i}] {} f32 values; max |Δ| = {diff:.4e} at idx {idx}; cos_dist = {cos:.3e}",
            ref_blob.len()
        );
        if cos > worst_fpn_cos.1 {
            worst_fpn_cos = (i, cos);
        }
        if diff > worst_fpn_diff.1 {
            worst_fpn_diff = (i, diff);
        }
    }
    if ref_num_fpn < rlx_fpn.len() {
        eprintln!(
            "[sam2 fpn] reference emitted {ref_num_fpn} levels (scalp dropped \
             the coarsest); skipping rlx levels {ref_num_fpn}..{}",
            rlx_fpn.len()
        );
    }

    // Cosine distance is the functional-parity gate.
    ensure!(
        worst_stage_cos.1 <= TOL_COS,
        "worst encoder stage cos_dist failure at stage {}: {:.3e} > {TOL_COS:.3e}",
        worst_stage_cos.0,
        worst_stage_cos.1,
    );
    ensure!(
        worst_fpn_cos.1 <= TOL_COS,
        "worst FPN cos_dist failure at level {}: {:.3e} > {TOL_COS:.3e}",
        worst_fpn_cos.0,
        worst_fpn_cos.1,
    );
    if worst_stage_diff.1 > WARN_MAX_DIFF {
        eprintln!(
            "warning: worst encoder stage max |Δ| = {:.4e} > {WARN_MAX_DIFF:.4e} \
             (cos_dist still bit-perfect, so functionally correct)",
            worst_stage_diff.1
        );
    }
    Ok(())
}

#[test]
fn sam2_decoder_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping — set RLX_SAM2_WEIGHTS");
        return Ok(());
    };
    let cfg_name = ref_config_name();
    let hiera = cfg_for_ref_name(&cfg_name);
    let full_cfg = Sam2Config {
        hiera: hiera.clone(),
        ..Sam2Config::hiera_base_plus()
    };

    let image_u8 = synthesize_image_u8();
    let image_nchw = preprocess_image(&image_u8, SAM2_IMG_SIZE, SAM2_IMG_SIZE);

    // Single foreground point at image center.
    let points: Vec<f32> = vec![SAM2_IMG_SIZE as f32 * 0.5, SAM2_IMG_SIZE as f32 * 0.5];
    let labels: Vec<f32> = vec![1.0];

    let ref_dir = dump_reference(&weights, &cfg_name, &image_nchw, Some((&points, &labels)))?;

    // ── Run rlx encoder + FPN ──
    let mut wm = WeightMap::from_file(&weights)?;
    let (graph, params, pre, fpn) = build_sam2_image_encoder_graph(&hiera, &mut wm)?;
    let mut compiled = compile_support::compile_sam(Device::Cpu, graph, params);
    let hidden = assemble_patch_tokens(&pre, &image_nchw)?;
    let stages = compiled.run(&[("hidden", hidden.as_slice())]);
    let stage_hw: Vec<(usize, usize)> = (0..hiera.stages.len())
        .map(|s| (hiera.grid_size_at_stage(s), hiera.grid_size_at_stage(s)))
        .collect();
    let stage_dims: Vec<usize> = (0..hiera.stages.len())
        .map(|s| hiera.embed_dim_at_stage(s))
        .collect();
    let mut fpn_ir = rlx_models::sam2::compile_fpn_neck_ir(
        &fpn,
        &stage_hw,
        &stage_dims,
        Device::Cpu,
        &rlx_flow::CompileProfile::sam2(),
    )?;
    let levels = apply_fpn_neck(&fpn, &mut fpn_ir, &stages, &stage_hw, &stage_dims)?;

    let prompt_w = extract_prompt_encoder_weights(
        &mut wm,
        full_cfg.decoder.transformer_dim,
        SAM2_MASK_IN_CHANS,
    )?;
    let mask_w = extract_mask_decoder_weights(&mut wm, &full_cfg.decoder)?;
    let _ = extract_memory_encoder_weights(&mut wm, &full_cfg.memory_encoder)?;
    let _ = extract_memory_attention_weights(&mut wm, &full_cfg.memory)?;

    let prompt = prompt_encoder_forward(
        &prompt_w,
        Some((&points, &labels)),
        /*boxes=*/ None,
        /*masks=*/ None,
    )?;

    // ── Prompt encoder bisect ──
    let ref_sparse_path = ref_dir.join("sparse_embeddings.f32");
    if ref_sparse_path.exists() {
        let ref_sparse = read_f32_blob(&ref_sparse_path, prompt.sparse_embeddings.len())?;
        let (sd, si) = max_abs_diff(&prompt.sparse_embeddings, &ref_sparse);
        let sc = cosine_distance(&prompt.sparse_embeddings, &ref_sparse);
        eprintln!("[prompt sparse]      max |Δ| = {sd:.4e} at idx {si}; cos_dist = {sc:.3e}");
        let ref_dense = read_f32_blob(
            &ref_dir.join("dense_embeddings.f32"),
            prompt.dense_embeddings.len(),
        )?;
        let (dd, di) = max_abs_diff(&prompt.dense_embeddings, &ref_dense);
        let dc = cosine_distance(&prompt.dense_embeddings, &ref_dense);
        eprintln!("[prompt dense]       max |Δ| = {dd:.4e} at idx {di}; cos_dist = {dc:.3e}");
        let ref_pe = read_f32_blob(&ref_dir.join("decoder_image_pe.f32"), prompt.image_pe.len())?;
        let (pd, pi) = max_abs_diff(&prompt.image_pe, &ref_pe);
        let pc = cosine_distance(&prompt.image_pe, &ref_pe);
        eprintln!("[prompt image_pe]    max |Δ| = {pd:.4e} at idx {pi}; cos_dist = {pc:.3e}");
    }

    let lvl_stride16 = &levels[2];
    let lvl_stride8 = &levels[1];
    let lvl_stride4 = &levels[0];
    let high_res_features = if mask_w.use_high_res_features {
        Some((
            lvl_stride4.features.as_slice(),
            lvl_stride8.features.as_slice(),
        ))
    } else {
        None
    };

    // image_pe comes from the prompt encoder's random-Fourier PE
    // (`sam_prompt_encoder.get_dense_pe()`), NOT from the FpnNeck's
    // sinusoidal pos. The two are mathematically different — reference
    // uses the former.
    let mut upscale = rlx_models::sam2::upscale_ir::Sam2MaskUpscaleCompiled::compile(
        &mask_w,
        SAM2_PROMPT_GRID,
        Device::Cpu,
    )?;
    let mut hyper_matmul = rlx_models::mask_hyper_matmul_ir::MaskHyperMatmulCompiled::compile(
        mask_w.num_mask_tokens,
        full_cfg.decoder.transformer_dim / 8,
        SAM2_PROMPT_GRID,
        Device::Cpu,
    )?;
    let mut hyper_mlps_ir =
        rlx_models::sam2::mlp_ir::compile_hyper_mlps(&mask_w.hyper_mlps, Device::Cpu)?;
    let mut iou_head_ir =
        rlx_models::sam2::mlp_ir::compile_hyper_mlp(&mask_w.iou_head, Device::Cpu)?;
    let mut obj_score_head_ir = rlx_models::sam2::mlp_ir::compile_optional_hyper_mlp(
        &mask_w.obj_score_head,
        1,
        Device::Cpu,
    )?;
    let obj_ptr_rows = rlx_models::sam2::mlp_ir::obj_ptr_proj_rows(
        mask_w.num_mask_tokens,
        mask_w.use_multimask_token_for_obj_ptr,
    );
    let mut obj_ptr_proj_ir = rlx_models::sam2::mlp_ir::compile_optional_hyper_mlp(
        &mask_w.obj_ptr_proj,
        obj_ptr_rows,
        Device::Cpu,
    )?;
    let s_tok = if mask_w.obj_score_token.is_some() {
        1
    } else {
        0
    };
    let base_q_n = s_tok + 1 + mask_w.num_mask_tokens;
    let mut tw_ir = rlx_models::sam2::transformer_ir::compile_two_way_transformer(
        &mask_w.transformer,
        base_q_n,
        SAM2_PROMPT_GRID,
        Device::Cpu,
    )?;
    let dec = mask_decoder_forward(
        &mask_w,
        &mut upscale,
        Some(&mut hyper_matmul),
        Some(&mut hyper_mlps_ir),
        Some(&mut iou_head_ir),
        obj_score_head_ir.as_mut(),
        obj_ptr_proj_ir.as_mut(),
        Some(&mut tw_ir),
        &lvl_stride16.features,
        &prompt.image_pe,
        &prompt.sparse_embeddings,
        prompt.num_sparse_tokens,
        &prompt.dense_embeddings,
        high_res_features,
        /*multimask_output=*/ true,
        SAM2_PROMPT_GRID,
    )?;

    // ── TwoWayTransformer bisect (run just the transformer with
    //     the reference inputs the mask decoder produces) ──
    let tw_hs_path = ref_dir.join("tw_hs.f32");
    if tw_hs_path.exists() {
        use rlx_models::sam2::transformer::two_way_transformer_forward as twt;
        let e = full_cfg.decoder.transformer_dim;
        let nm = full_cfg.decoder.num_mask_tokens;
        let g = SAM2_PROMPT_GRID;
        // Build tokens = cat(obj_score, iou, mask, sparse)
        let n_out = 1 + 1 + nm; // pred_obj_scores=true → 1+1+4=6
        let q_n = n_out + prompt.num_sparse_tokens;
        let mut tokens = Vec::with_capacity(q_n * e);
        tokens.extend_from_slice(mask_w.obj_score_token.as_ref().unwrap());
        tokens.extend_from_slice(&mask_w.iou_token);
        tokens.extend_from_slice(&mask_w.mask_tokens);
        tokens.extend_from_slice(&prompt.sparse_embeddings);
        // src = image_emb + dense_emb
        let mut src = lvl_stride16.features.clone();
        for i in 0..src.len() {
            src[i] += prompt.dense_embeddings[i];
        }
        let (hs, src_post) = twt(
            &mask_w.transformer,
            &src,
            &prompt.image_pe,
            &tokens,
            1,
            e,
            g,
            g,
            q_n,
        );
        let ref_hs = read_f32_blob(&tw_hs_path, hs.len())?;
        let ref_src = read_f32_blob(&ref_dir.join("tw_src.f32"), src_post.len())?;
        let (hd, hi) = max_abs_diff(&hs, &ref_hs);
        let hc = cosine_distance(&hs, &ref_hs);
        eprintln!("[tw_transformer hs]  max |Δ| = {hd:.4e} at idx {hi}; cos_dist = {hc:.3e}");
        let (sd, si) = max_abs_diff(&src_post, &ref_src);
        let sc = cosine_distance(&src_post, &ref_src);
        eprintln!("[tw_transformer src] max |Δ| = {sd:.4e} at idx {si}; cos_dist = {sc:.3e}");

        // Verify the decoder input image_embed matches reference.
        let ref_emb = read_f32_blob(
            &ref_dir.join("decoder_image_embed.f32"),
            lvl_stride16.features.len(),
        )?;
        let (d, i) = max_abs_diff(&lvl_stride16.features, &ref_emb);
        let c = cosine_distance(&lvl_stride16.features, &ref_emb);
        eprintln!("[decoder image_embed] max |Δ| = {d:.4e} at idx {i}; cos_dist = {c:.3e}");

        // ── Per-step layer-0 bisect ──
        use rlx_models::sam2::transformer::layer_norm_last;
        use rlx_models::sam2::transformer::sam2_attention_forward as attn_fn;
        let layer0 = &mask_w.transformer.layers[0];
        // Self-attn with skip_first_layer_pe=True: queries = attn(q, q, q).
        let q_sa = attn_fn(
            &layer0.self_attn,
            &tokens,
            q_n,
            &tokens,
            q_n,
            &tokens,
            q_n,
            1,
        );
        let ref_sa = read_f32_blob(&ref_dir.join("tw_l0_post_self_attn.f32"), q_sa.len())?;
        let (d, i) = max_abs_diff(&q_sa, &ref_sa);
        let c = cosine_distance(&q_sa, &ref_sa);
        eprintln!("[l0 self_attn]       max |Δ| = {d:.4e} at idx {i}; cos_dist = {c:.3e}");
        // Then norm1.
        let mut q_n1 = q_sa.clone();
        layer_norm_last(&mut q_n1, q_n, e, &layer0.norm1_g, &layer0.norm1_b, 1e-5);
        let ref_n1 = read_f32_blob(&ref_dir.join("tw_l0_post_norm1.f32"), q_n1.len())?;
        let (d, i) = max_abs_diff(&q_n1, &ref_n1);
        let c = cosine_distance(&q_n1, &ref_n1);
        eprintln!("[l0 norm1]           max |Δ| = {d:.4e} at idx {i}; cos_dist = {c:.3e}");

        // Cross-attn token→image. q = q_n1 + tokens (query_pe), k = keys+image_pe, v = keys.
        // src is image_emb + dense_emb in NCHW, need flat BHWC sequence.
        let mut keys_seq = vec![0f32; src.len()];
        let mut keys_pe_seq = vec![0f32; src.len()];
        let k_n_seq = g * g;
        for y in 0..g {
            for x in 0..g {
                for ch in 0..e {
                    let src_idx = ch * g * g + y * g + x;
                    let dst_idx = (y * g + x) * e + ch;
                    keys_seq[dst_idx] = src[src_idx];
                    keys_pe_seq[dst_idx] = prompt.image_pe[src_idx];
                }
            }
        }
        let mut q_pe = q_n1.clone();
        for i in 0..q_pe.len() {
            q_pe[i] += tokens[i];
        }
        let mut k_pe = keys_seq.clone();
        for i in 0..k_pe.len() {
            k_pe[i] += keys_pe_seq[i];
        }
        // Compare k_pe (input to k_proj) first.
        let ref_kpe = read_f32_blob(&ref_dir.join("tw_l0_cross_t2i_kpe.f32"), k_pe.len())?;
        let (d, i) = max_abs_diff(&k_pe, &ref_kpe);
        let c = cosine_distance(&k_pe, &ref_kpe);
        eprintln!("[l0 cross_t2i k_pe]  max |Δ| = {d:.4e} at idx {i}; cos_dist = {c:.3e}");

        // Dump k_proj alone for bisect.
        use rlx_models::sam2::transformer::linear as host_linear;
        let cross_w = &layer0.cross_token_to_image;
        let k_id = cross_w.internal_dim;
        let kp = host_linear(
            &k_pe,
            &cross_w.k_w,
            &cross_w.k_b,
            k_n_seq,
            cross_w.embed_dim,
            k_id,
        );
        let ref_kp = read_f32_blob(&ref_dir.join("tw_l0_cross_t2i_kproj.f32"), kp.len())?;
        let (d, i) = max_abs_diff(&kp, &ref_kp);
        let c = cosine_distance(&kp, &ref_kp);
        eprintln!("[l0 cross_t2i kproj] max |Δ| = {d:.4e} at idx {i}; cos_dist = {c:.3e}");

        let ca1 = attn_fn(
            &layer0.cross_token_to_image,
            &q_pe,
            q_n,
            &k_pe,
            k_n_seq,
            &keys_seq,
            k_n_seq,
            1,
        );
        let ref_ca1 = read_f32_blob(&ref_dir.join("tw_l0_post_cross_t2i.f32"), ca1.len())?;
        let (d, i) = max_abs_diff(&ca1, &ref_ca1);
        let c = cosine_distance(&ca1, &ref_ca1);
        eprintln!("[l0 cross_t2i]       max |Δ| = {d:.4e} at idx {i}; cos_dist = {c:.3e}");
    }

    // ── Compare ──
    let ref_masks = read_f32_blob(&ref_dir.join("mask_logits.f32"), dec.masks.len())?;
    let ref_iou = read_f32_blob(&ref_dir.join("iou_pred.f32"), dec.iou_pred.len())?;
    let (m_diff, m_idx) = max_abs_diff(&dec.masks, &ref_masks);
    let m_cos = cosine_distance(&dec.masks, &ref_masks);
    let (i_diff, i_idx) = max_abs_diff(&dec.iou_pred, &ref_iou);
    let i_cos = cosine_distance(&dec.iou_pred, &ref_iou);
    eprintln!("[sam2 decoder masks] max |Δ| = {m_diff:.4e} at idx {m_idx}; cos_dist = {m_cos:.3e}");
    eprintln!("[sam2 decoder iou]   max |Δ| = {i_diff:.4e} at idx {i_idx}; cos_dist = {i_cos:.3e}");

    let obj_path = ref_dir.join("object_score.f32");
    if obj_path.exists() {
        let ref_obj = read_f32_blob(&obj_path, dec.object_score_logits.len())?;
        let (od, oi) = max_abs_diff(&dec.object_score_logits, &ref_obj);
        eprintln!(
            "[sam2 decoder obj]   max |Δ| = {od:.4e} at idx {oi}; rlx={:?} ref={:?}",
            dec.object_score_logits, ref_obj
        );
    }

    ensure!(
        m_cos <= TOL_COS,
        "mask cos_dist failure: {m_cos:.3e} > {TOL_COS:.3e}"
    );
    ensure!(
        i_cos <= TOL_COS,
        "iou cos_dist failure: {i_cos:.3e} > {TOL_COS:.3e}"
    );

    Ok(())
}

#[test]
fn sam2_memory_encoder_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping — set RLX_SAM2_WEIGHTS");
        return Ok(());
    };
    let cfg_name = ref_config_name();
    let hiera = cfg_for_ref_name(&cfg_name);
    let full_cfg = Sam2Config {
        hiera: hiera.clone(),
        ..Sam2Config::hiera_base_plus()
    };

    let image_u8 = synthesize_image_u8();
    let image_nchw = preprocess_image(&image_u8, SAM2_IMG_SIZE, SAM2_IMG_SIZE);

    // Tell the reference to dump memory encoder I/O.
    let tmp = env::temp_dir().join(format!("rlx_sam2_memenc_{}", std::process::id()));
    fs::create_dir_all(&tmp)?;
    let img_bin = tmp.join("image.f32");
    write_f32_blob(&img_bin, &image_nchw)?;

    let use_docker = rlx_ir::env::var("RLX_SAM2_DOCKER").as_deref() == Some("1");
    let device = rlx_ir::env::var("RLX_SAM2_DEVICE").unwrap_or_else(|| "cpu".to_string());
    let mut cmd = if use_docker {
        let mut c = Command::new("bash");
        c.arg("tests/sam2_parity_helpers/run-ref.sh");
        c.env("RLX_SAM2_DEVICE", &device);
        c
    } else {
        let mut c =
            Command::new(rlx_ir::env::var("RLX_SAM2_PYTHON").unwrap_or_else(|| "python".into()));
        c.arg("tests/sam2_parity_helpers/dump_reference.py");
        c.env("RLX_SAM2_DEVICE", &device);
        c
    };
    cmd.env("RLX_SAM2_WEIGHTS", &weights)
        .env("RLX_SAM2_CONFIG", &cfg_name)
        .env("RLX_SAM2_IMAGE_BIN", img_bin.as_os_str())
        .env("RLX_SAM2_OUT_DIR", tmp.as_os_str())
        .env("RLX_SAM2_RUN_MEMORY_ENCODER", "1");
    let status = cmd.status()?;
    ensure!(status.success(), "memenc ref dump failed");

    let mut wm = WeightMap::from_file(&weights)?;
    // Drain non-memory-encoder weights so the next call cursor lands clean.
    let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&hiera, &mut wm)?;
    let _ = extract_prompt_encoder_weights(
        &mut wm,
        full_cfg.decoder.transformer_dim,
        SAM2_MASK_IN_CHANS,
    )?;
    let _ = extract_mask_decoder_weights(&mut wm, &full_cfg.decoder)?;
    let mut mem_w = extract_memory_encoder_weights(&mut wm, &full_cfg.memory_encoder)?;
    rlx_models::sam2::memory_encoder::compile_memory_encoder_ir(
        &mut mem_w,
        SAM2_IMG_SIZE,
        SAM2_IMG_SIZE,
        SAM2_PROMPT_GRID,
        SAM2_PROMPT_GRID,
        rlx_runtime::Device::Cpu,
        &rlx_flow::CompileProfile::sam2(),
    )?;

    // Inputs.
    let pix_feat = read_f32_blob(
        &tmp.join("memenc_pix_feat.f32"),
        full_cfg.memory_encoder.in_dim * SAM2_PROMPT_GRID * SAM2_PROMPT_GRID,
    )?;
    let mask_full = read_f32_blob(&tmp.join("memenc_mask.f32"), SAM2_IMG_SIZE * SAM2_IMG_SIZE)?;
    let out = memory_encoder_forward(
        &mut mem_w,
        &pix_feat,
        &mask_full,
        SAM2_PROMPT_GRID,
        SAM2_PROMPT_GRID,
        /*skip_mask_sigmoid=*/ false,
    )?;

    let ref_feat = read_f32_blob(&tmp.join("memenc_features.f32"), out.features.len())?;
    let ref_pos = read_f32_blob(&tmp.join("memenc_pos.f32"), out.pos.len())?;
    let (fd, fi) = max_abs_diff(&out.features, &ref_feat);
    let fc = cosine_distance(&out.features, &ref_feat);
    eprintln!("[memenc features] max |Δ| = {fd:.4e} at idx {fi}; cos_dist = {fc:.3e}");
    let (pd, pi) = max_abs_diff(&out.pos, &ref_pos);
    let pc = cosine_distance(&out.pos, &ref_pos);
    eprintln!("[memenc pos]      max |Δ| = {pd:.4e} at idx {pi}; cos_dist = {pc:.3e}");

    ensure!(
        fc <= TOL_COS,
        "memenc features cos_dist {fc:.3e} > {TOL_COS:.3e}"
    );
    ensure!(
        pc <= TOL_COS,
        "memenc pos cos_dist {pc:.3e} > {TOL_COS:.3e}"
    );
    Ok(())
}

#[test]
fn sam2_memory_attention_parity_vs_pytorch() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping — set RLX_SAM2_WEIGHTS");
        return Ok(());
    };
    let cfg_name = ref_config_name();
    let hiera = cfg_for_ref_name(&cfg_name);
    let full_cfg = Sam2Config {
        hiera: hiera.clone(),
        ..Sam2Config::hiera_base_plus()
    };

    let image_u8 = synthesize_image_u8();
    let image_nchw = preprocess_image(&image_u8, SAM2_IMG_SIZE, SAM2_IMG_SIZE);

    let tmp = env::temp_dir().join(format!("rlx_sam2_memattn_{}", std::process::id()));
    fs::create_dir_all(&tmp)?;
    let img_bin = tmp.join("image.f32");
    write_f32_blob(&img_bin, &image_nchw)?;

    let use_docker = rlx_ir::env::var("RLX_SAM2_DOCKER").as_deref() == Some("1");
    let device = rlx_ir::env::var("RLX_SAM2_DEVICE").unwrap_or_else(|| "cpu".to_string());
    let mut cmd = if use_docker {
        let mut c = Command::new("bash");
        c.arg("tests/sam2_parity_helpers/run-ref.sh");
        c.env("RLX_SAM2_DEVICE", &device);
        c
    } else {
        let mut c =
            Command::new(rlx_ir::env::var("RLX_SAM2_PYTHON").unwrap_or_else(|| "python".into()));
        c.arg("tests/sam2_parity_helpers/dump_reference.py");
        c.env("RLX_SAM2_DEVICE", &device);
        c
    };
    cmd.env("RLX_SAM2_WEIGHTS", &weights)
        .env("RLX_SAM2_CONFIG", &cfg_name)
        .env("RLX_SAM2_IMAGE_BIN", img_bin.as_os_str())
        .env("RLX_SAM2_OUT_DIR", tmp.as_os_str())
        .env("RLX_SAM2_RUN_MEMORY_ATTENTION", "1");
    let status = cmd.status()?;
    ensure!(status.success(), "memattn ref dump failed");

    let mut wm = WeightMap::from_file(&weights)?;
    let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&hiera, &mut wm)?;
    let _ = extract_prompt_encoder_weights(
        &mut wm,
        full_cfg.decoder.transformer_dim,
        SAM2_MASK_IN_CHANS,
    )?;
    let _ = extract_mask_decoder_weights(&mut wm, &full_cfg.decoder)?;
    let _ = extract_memory_encoder_weights(&mut wm, &full_cfg.memory_encoder)?;
    let attn_w = extract_memory_attention_weights(&mut wm, &full_cfg.memory)?;

    // Inputs are dumped as (S, B, C). Current frame queries are the
    // stride-16 features: S = 64*64 = 4096. Memory bank is also 64*64.
    let n_img = 64 * 64;
    let d = full_cfg.memory.d_model;
    let kv = full_cfg.memory.kv_in_dim;
    let n_mem = 64 * 64;

    let curr_sbc = read_f32_blob(&tmp.join("memattn_curr.f32"), n_img * d)?;
    let curr_pos_sbc = read_f32_blob(&tmp.join("memattn_curr_pos.f32"), n_img * d)?;
    let mem_sbc = read_f32_blob(&tmp.join("memattn_mem.f32"), n_mem * kv)?;
    let mem_pos_sbc = read_f32_blob(&tmp.join("memattn_mem_pos.f32"), n_mem * kv)?;
    let ref_out_sbc = read_f32_blob(&tmp.join("memattn_output.f32"), n_img * d)?;

    // Our memory_attention_forward expects [N, C] flat (B=1 collapsed).
    // (S, B=1, C) layout matches [N, C] when B=1 since the memory layout
    // is identical row-major.
    let out = memory_attention_forward(
        &attn_w,
        &curr_sbc,
        &curr_pos_sbc,
        &mem_sbc,
        &mem_pos_sbc,
        n_img,
        n_mem,
        kv,
        /*num_obj_ptr_tokens=*/ 0,
    )?;

    let (od, oi) = max_abs_diff(&out, &ref_out_sbc);
    let oc = cosine_distance(&out, &ref_out_sbc);
    eprintln!("[memattn output] max |Δ| = {od:.4e} at idx {oi}; cos_dist = {oc:.3e}");
    ensure!(oc <= TOL_COS, "memattn cos_dist {oc:.3e} > {TOL_COS:.3e}");
    Ok(())
}

#[allow(dead_code)]
fn _silence_unused_helpers() {
    let _ = memory_encoder_forward;
    let _ = memory_attention_forward;
    let _ = anyhow!("");
}
