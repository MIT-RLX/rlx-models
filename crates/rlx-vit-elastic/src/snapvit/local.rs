// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Local Hessian diagonal (SnapViT Eq. 2/3): the mean over a small calibration
//! set of the squared gradient of the DINO SSL loss, aggregated from
//! per-parameter to per-structure (per attention head / per FFN channel).

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use rlx_autodiff::grad_with_loss;
use rlx_ir::NodeId;
use rlx_runtime::{CompileOptions, Device, Session};

use crate::dino::{CropConfig, Rng, multi_crop, stack_crops, teacher_targets};
use crate::vit::config::{FfnKind, VitConfig};
use crate::vit::preprocess::assemble_hidden;
use crate::vit::runner::VitRunner;
use crate::vit::weights::LoadedVit;

use super::loss::build_snapvit_loss;

/// A calibration image: HWC u8 pixels + source dimensions.
#[derive(Clone)]
pub struct CalibImage {
    pub rgb: Vec<u8>,
    pub h: usize,
    pub w: usize,
}

/// SnapViT SSL hyperparameters (defaults follow DINO).
#[derive(Clone)]
pub struct SnapVitConfig {
    pub crops: CropConfig,
    pub temp_s: f32,
    pub temp_t: f32,
    pub seed: u64,
}

impl SnapVitConfig {
    pub fn new(img_size: usize) -> Self {
        Self {
            crops: CropConfig {
                img_size,
                ..Default::default()
            },
            temp_s: 0.1,
            temp_t: 0.04,
            seed: 0xA5A5,
        }
    }
}

/// Per-structure local sensitivity scores.
#[derive(Clone, Debug)]
pub struct LocalScores {
    /// Per attention head, `[num_layers · num_heads]` (layer-major).
    pub head: Vec<f32>,
    /// Per FFN inner channel, `[num_layers · inner]` (layer-major).
    pub ffn: Vec<f32>,
}

/// The device to run a **backward** (gradient) graph on. Every backend now runs
/// this graph's autodiff correctly and is honored as-is:
///   - CUDA: fixed via the `rlx-cuda` unfuse rank-4 `AttentionBackward` promotion.
///   - Metal: fixed via the `rlx-metal` LayerNorm variance clamp
///     (`fmax(0, E[x²]−E[x]²)`) — the one-pass variance could go slightly
///     negative on large inputs → `rsqrt(neg)=NaN`, poisoning the whole forward.
///   - MLX / wgpu / Vulkan: already correct.
pub fn backward_device(device: Device, _what: &str) -> Device {
    device
}

fn l2norm_host(v: &[f32]) -> Vec<f32> {
    let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt() + 1e-12;
    v.iter().map(|x| x / n).collect()
}

/// Compute the local Hessian-diagonal per-structure scores over `images`.
pub fn compute_local_scores(
    cfg: &VitConfig,
    loaded: &LoadedVit,
    images: &[CalibImage],
    sc: &SnapVitConfig,
    device: Device,
) -> Result<LocalScores> {
    if images.is_empty() {
        return Err(anyhow!("compute_local_scores: no calibration images"));
    }
    let n_crops = sc.crops.n_crops();
    let n_global = sc.crops.n_global;
    let h = cfg.hidden_size;

    // Local-scores is a gradient (calibration) step, so it runs entirely on CPU:
    // RLX's Metal/GPU autodiff has a NaN in the transpose/narrow backward kernels
    // (the DINO head's layout ops trip it). The forward-only xNES fitness — the
    // bulk of the work — still uses the requested `device`.
    let grad_device = backward_device(device, "snapvit local scores");

    // Teacher forward runner (same weights → self-distillation targets).
    let teacher_loaded = LoadedVit {
        params: loaded.params.clone(),
        preprocess: loaded.preprocess.clone(),
    };
    let mut runner = VitRunner::from_loaded(cfg.clone(), teacher_loaded, grad_device, n_crops)?;

    let snap = build_snapvit_loss(cfg, n_crops, n_global, sc.temp_s);
    let wrt_ids: Vec<NodeId> = snap.params.iter().map(|p| p.node).collect();
    let backward = grad_with_loss(&snap.graph, &wrt_ids);
    let mut bw = Session::new(grad_device).compile_with(backward, &CompileOptions::new());
    for p in &snap.params {
        bw.set_param(
            &p.name,
            loaded
                .params
                .get(&p.name)
                .ok_or_else(|| anyhow!("missing param {}", p.name))?,
        );
    }

    let ones_head = vec![1.0f32; cfg.num_hidden_layers * h];
    let ones_ffn = vec![1.0f32; cfg.num_hidden_layers * cfg.ffn_inner()];
    let (mask, _) = crate::dino::pair_mask(n_global, n_crops);
    let center = vec![0.0f32; h];
    let seed_bytes = [1.0f32];

    // Accumulator of squared gradients, aligned with snap.params.
    let mut accum: Vec<Vec<f32>> = snap
        .params
        .iter()
        .map(|p| vec![0.0f32; p.dims.iter().product()])
        .collect();

    let mut rng = Rng::new(sc.seed);
    for img in images {
        let crops = multi_crop(&mut rng, &img.rgb, img.h, img.w, &sc.crops);
        let stacked = stack_crops(&crops);
        let hidden = assemble_hidden(&loaded.preprocess, &stacked, n_crops)?;

        // Teacher targets from the global crops' (L2-normed) embeddings.
        runner.reset_masks();
        let emb = runner.embed_hidden(&hidden)?;
        let mut tlogits = vec![0.0f32; n_global * h];
        for (t, e) in emb.iter().take(n_global).enumerate() {
            tlogits[t * h..(t + 1) * h].copy_from_slice(&l2norm_host(e));
        }
        let targets = teacher_targets(&tlogits, n_global, h, sc.temp_t, &center);

        let outs = bw.run(&[
            ("hidden", hidden.as_slice()),
            ("head_mask", ones_head.as_slice()),
            ("ffn_mask", ones_ffn.as_slice()),
            ("teacher_targets", targets.as_slice()),
            ("pair_mask", mask.as_slice()),
            ("d_output", seed_bytes.as_slice()),
        ]);
        // outs = [loss, grad(params[0]), grad(params[1]), ...].
        for (i, acc) in accum.iter_mut().enumerate() {
            if let Some(gv) = outs.get(1 + i) {
                let n = acc.len().min(gv.len());
                for j in 0..n {
                    acc[j] += gv[j] * gv[j];
                }
            }
        }
    }

    let inv = 1.0 / images.len() as f32;
    for acc in accum.iter_mut() {
        for a in acc.iter_mut() {
            *a *= inv;
        }
    }

    let by_name: HashMap<&str, &Vec<f32>> = snap
        .params
        .iter()
        .zip(&accum)
        .map(|(p, a)| (p.name.as_str(), a))
        .collect();
    Ok(aggregate_structures(cfg, &by_name))
}

/// Sum the per-parameter diagonal into per-head and per-FFN-channel scores.
fn aggregate_structures(cfg: &VitConfig, by_name: &HashMap<&str, &Vec<f32>>) -> LocalScores {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let inner = cfg.ffn_inner();
    let depth = cfg.num_hidden_layers;

    let mut head = vec![0f32; depth * nh];
    let mut ffn = vec![0f32; depth * inner];
    let g = |name: &str| by_name.get(name).copied();

    for l in 0..depth {
        let lp = format!("blocks.{l}");
        // ---- heads ----
        let qkv = g(&format!("{lp}.attn.qkv.weight")); // [H, 3H] row-major
        let qkvb = g(&format!("{lp}.attn.qkv.bias")); // [3H]
        let proj = g(&format!("{lp}.attn.proj.weight")); // [H, H] (in rows, out cols)
        for hh in 0..nh {
            let mut s = 0.0f32;
            if let Some(qkv) = qkv {
                for blk in 0..3 {
                    let base = blk * h + hh * hd;
                    for c in base..base + hd {
                        for r in 0..h {
                            s += qkv[r * 3 * h + c];
                        }
                    }
                }
            }
            if let Some(qkvb) = qkvb {
                for blk in 0..3 {
                    let base = blk * h + hh * hd;
                    for c in base..base + hd {
                        s += qkvb[c];
                    }
                }
            }
            if let Some(proj) = proj {
                for r in hh * hd..(hh + 1) * hd {
                    for c in 0..h {
                        s += proj[r * h + c];
                    }
                }
            }
            head[l * nh + hh] = s;
        }

        // ---- FFN channels ----
        let fc2 = g(&format!("{lp}.mlp.fc2.weight")); // [inner, H] (in rows, out cols)
        match cfg.ffn_kind {
            FfnKind::Gelu => {
                let fc1 = g(&format!("{lp}.mlp.fc1.weight")); // [H, inner]
                let fc1b = g(&format!("{lp}.mlp.fc1.bias"));
                for cc in 0..inner {
                    let mut s = 0.0f32;
                    if let Some(fc1) = fc1 {
                        for r in 0..h {
                            s += fc1[r * inner + cc];
                        }
                    }
                    if let Some(fc1b) = fc1b {
                        s += fc1b[cc];
                    }
                    if let Some(fc2) = fc2 {
                        for c in 0..h {
                            s += fc2[cc * h + c];
                        }
                    }
                    ffn[l * inner + cc] = s;
                }
            }
            FfnKind::PackedSwiGLU => {
                let val = g(&format!("{lp}.mlp.fc1_value.weight")); // [H, inner]
                let gate = g(&format!("{lp}.mlp.fc1_gate.weight"));
                let valb = g(&format!("{lp}.mlp.fc1_value.bias"));
                let gateb = g(&format!("{lp}.mlp.fc1_gate.bias"));
                for cc in 0..inner {
                    let mut s = 0.0f32;
                    for r in 0..h {
                        if let Some(val) = val {
                            s += val[r * inner + cc];
                        }
                        if let Some(gate) = gate {
                            s += gate[r * inner + cc];
                        }
                    }
                    if let Some(valb) = valb {
                        s += valb[cc];
                    }
                    if let Some(gateb) = gateb {
                        s += gateb[cc];
                    }
                    if let Some(fc2) = fc2 {
                        for c in 0..h {
                            s += fc2[cc * h + c];
                        }
                    }
                    ffn[l * inner + cc] = s;
                }
            }
        }
    }

    LocalScores { head, ffn }
}
