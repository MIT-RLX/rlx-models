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

//! Swin Transformer vision backbone (CPU-native), matching HF
//! `GroundingDinoConvEncoder` (a `SwinBackbone`). Produces three multi-scale
//! feature maps (stages 2/3/4) for the neck.

use crate::config::SwinConfig;
use crate::ir;
use crate::nn::{self, AttnBias};
use crate::swin_ir::WindowAttnWeights;
use crate::weights::{get, get_with_shape};
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, HirGraphExt, HirModule, HirMut, HirNodeId, Shape};
use rlx_runtime::Device;

const NEG: f32 = -100.0; // HF uses -100 for masked window attention entries.

/// One multi-scale feature map, channels-first `[c, h, w]`.
#[derive(Debug, Clone)]
pub struct FeatureMap {
    pub data: Vec<f32>,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

struct Block {
    dim: usize,
    heads: usize,
    shift: usize,
    ln_before_w: Vec<f32>,
    ln_before_b: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    rel_bias_table: Vec<f32>, // [(2ws-1)^2, heads]
    rel_index: Vec<usize>,    // [ws2, ws2]
    attn_out_w: Vec<f32>,
    attn_out_b: Vec<f32>,
    ln_after_w: Vec<f32>,
    ln_after_b: Vec<f32>,
    inter_w: Vec<f32>,
    inter_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

struct Downsample {
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    reduction_w: Vec<f32>, // [2d, 4d], no bias
}

struct Stage {
    blocks: Vec<Block>,
    downsample: Option<Downsample>,
}

/// Swin backbone weights + config.
pub struct SwinBackbone {
    cfg: SwinConfig,
    patch_w: Vec<f32>, // [embed_dim, 3, p, p]
    patch_b: Vec<f32>,
    emb_norm_w: Vec<f32>,
    emb_norm_b: Vec<f32>,
    stages: Vec<Stage>,
    /// Output LayerNorms keyed by stage index (1-based), for `out_indices`.
    out_norms: Vec<(usize, Vec<f32>, Vec<f32>)>,
    eps: f32,
    /// Device the compute (LN/attention/FFN/patch-merge) graphs run on.
    device: Device,
}

impl SwinBackbone {
    /// Build for CPU (numerically equivalent to the native reference path).
    pub fn from_weights(wm: &WeightMap, cfg: SwinConfig) -> Result<Self> {
        Self::from_weights_on(wm, cfg, Device::Cpu)
    }

    pub fn from_weights_on(wm: &WeightMap, cfg: SwinConfig, device: Device) -> Result<Self> {
        let p = "model.backbone.conv_encoder.model.";
        let ws = cfg.window_size;
        let mut stages = Vec::with_capacity(cfg.num_stages());
        for s in 0..cfg.num_stages() {
            let dim = cfg.stage_dim(s);
            let heads = cfg.num_heads[s];
            let mut blocks = Vec::with_capacity(cfg.depths[s]);
            for b in 0..cfg.depths[s] {
                let bp = format!("{p}encoder.layers.{s}.blocks.{b}.");
                let shift = if b % 2 == 1 { ws / 2 } else { 0 };
                let (rel_idx_f, _) =
                    get_with_shape(wm, &format!("{bp}attention.self.relative_position_index"))?;
                let rel_index: Vec<usize> = rel_idx_f.iter().map(|&v| v as usize).collect();
                blocks.push(Block {
                    dim,
                    heads,
                    shift,
                    ln_before_w: get(wm, &format!("{bp}layernorm_before.weight"))?,
                    ln_before_b: get(wm, &format!("{bp}layernorm_before.bias"))?,
                    q_w: get(wm, &format!("{bp}attention.self.query.weight"))?,
                    q_b: get(wm, &format!("{bp}attention.self.query.bias"))?,
                    k_w: get(wm, &format!("{bp}attention.self.key.weight"))?,
                    k_b: get(wm, &format!("{bp}attention.self.key.bias"))?,
                    v_w: get(wm, &format!("{bp}attention.self.value.weight"))?,
                    v_b: get(wm, &format!("{bp}attention.self.value.bias"))?,
                    rel_bias_table: get(
                        wm,
                        &format!("{bp}attention.self.relative_position_bias_table"),
                    )?,
                    rel_index,
                    attn_out_w: get(wm, &format!("{bp}attention.output.dense.weight"))?,
                    attn_out_b: get(wm, &format!("{bp}attention.output.dense.bias"))?,
                    ln_after_w: get(wm, &format!("{bp}layernorm_after.weight"))?,
                    ln_after_b: get(wm, &format!("{bp}layernorm_after.bias"))?,
                    inter_w: get(wm, &format!("{bp}intermediate.dense.weight"))?,
                    inter_b: get(wm, &format!("{bp}intermediate.dense.bias"))?,
                    out_w: get(wm, &format!("{bp}output.dense.weight"))?,
                    out_b: get(wm, &format!("{bp}output.dense.bias"))?,
                });
            }
            let downsample = if s < cfg.num_stages() - 1 {
                let dp = format!("{p}encoder.layers.{s}.downsample.");
                Some(Downsample {
                    norm_w: get(wm, &format!("{dp}norm.weight"))?,
                    norm_b: get(wm, &format!("{dp}norm.bias"))?,
                    reduction_w: get(wm, &format!("{dp}reduction.weight"))?,
                })
            } else {
                None
            };
            stages.push(Stage { blocks, downsample });
        }
        let out_norms = cfg
            .out_indices
            .iter()
            .map(|&idx| {
                let w = get(wm, &format!("{p}hidden_states_norms.stage{idx}.weight"))?;
                let b = get(wm, &format!("{p}hidden_states_norms.stage{idx}.bias"))?;
                Ok((idx, w, b))
            })
            .collect::<Result<Vec<_>>>()?;
        let eps = cfg.layer_norm_eps as f32;
        Ok(Self {
            patch_w: get(
                wm,
                &format!("{p}embeddings.patch_embeddings.projection.weight"),
            )?,
            patch_b: get(
                wm,
                &format!("{p}embeddings.patch_embeddings.projection.bias"),
            )?,
            emb_norm_w: get(wm, &format!("{p}embeddings.norm.weight"))?,
            emb_norm_b: get(wm, &format!("{p}embeddings.norm.bias"))?,
            cfg,
            stages,
            out_norms,
            eps,
            device,
        })
    }

    /// Forward a single normalized image, channels-first `[3, h, w]`.
    /// Returns the feature maps selected by `out_indices`, in order. Compute
    /// runs as HIR graphs on `self.device`; geometry stays on the host.
    pub fn forward(&self, pixel_values: &[f32], h: usize, w: usize) -> Vec<FeatureMap> {
        self.forward_ir(pixel_values, h, w)
            .expect("swin graph forward on the model device should not fail")
    }

    /// Fallible graph forward (see [`Self::forward`]).
    pub fn forward_ir(&self, pixel_values: &[f32], h: usize, w: usize) -> Result<Vec<FeatureMap>> {
        let embed_dim = self.cfg.embed_dim;
        let ps = self.cfg.patch_size;
        let (img, hp, wp) = pad_image_chw(pixel_values, 3, h, w, ps);
        let gh = hp / ps;
        let gw = wp / ps;

        // Patch embed conv (host geometry), then embedding LayerNorm (graph).
        let mut x = patch_embed_conv(&img, 3, hp, wp, &self.patch_w, &self.patch_b, embed_dim, ps);
        x = ln_ir(
            self.device,
            &x,
            gh * gw,
            embed_dim,
            &self.emb_norm_w,
            &self.emb_norm_b,
            self.eps,
        )?;

        let prof = std::env::var("RLX_GDINO_PROFILE").is_ok();
        let (mut cur_h, mut cur_w) = (gh, gw);
        let mut feature_maps = Vec::new();
        for (s, stage) in self.stages.iter().enumerate() {
            let st = std::time::Instant::now();
            let nb = stage.blocks.len();
            for block in &stage.blocks {
                x = self.run_block_ir(block, &x, cur_h, cur_w)?;
            }
            if prof {
                eprintln!(
                    "[gdino-profile]   swin.stage{s}: {:.3}s ({nb} blocks, {cur_h}x{cur_w})",
                    st.elapsed().as_secs_f64()
                );
            }
            let stage_name = s + 1;
            if let Some((_, nw, nb)) = self.out_norms.iter().find(|(i, _, _)| *i == stage_name) {
                let dim = self.cfg.stage_dim(s);
                let normed = ln_ir(self.device, &x, cur_h * cur_w, dim, nw, nb, self.eps)?;
                feature_maps.push(tokens_to_chw(&normed, cur_h, cur_w, dim));
            }
            if let Some(ds) = &stage.downsample {
                let dim = self.cfg.stage_dim(s);
                let (nx, nh, nw_) =
                    patch_merge_ir(self.device, &x, cur_h, cur_w, dim, ds, self.eps)?;
                x = nx;
                cur_h = nh;
                cur_w = nw_;
            }
        }
        Ok(feature_maps)
    }

    /// Native (eager `nn::`) forward — retained as the parity oracle for
    /// [`Self::forward_ir`].
    pub fn forward_native(&self, pixel_values: &[f32], h: usize, w: usize) -> Vec<FeatureMap> {
        let embed_dim = self.cfg.embed_dim;
        let ps = self.cfg.patch_size;
        // Pad image so H,W divisible by patch size.
        let (img, hp, wp) = pad_image_chw(pixel_values, 3, h, w, ps);
        let gh = hp / ps;
        let gw = wp / ps;

        // Patch embed conv (stride = kernel = ps), → tokens [gh*gw, embed_dim].
        let mut x = patch_embed_conv(&img, 3, hp, wp, &self.patch_w, &self.patch_b, embed_dim, ps);
        x = nn::layer_norm(&x, &self.emb_norm_w, &self.emb_norm_b, embed_dim, self.eps);

        let (mut cur_h, mut cur_w) = (gh, gw);
        let mut feature_maps = Vec::new();

        for (s, stage) in self.stages.iter().enumerate() {
            for block in &stage.blocks {
                x = self.run_block(block, &x, cur_h, cur_w);
            }
            // "stage{s+1}" output (before downsampling) is a candidate feature map.
            let stage_name = s + 1;
            if let Some((_, nw, nb)) = self.out_norms.iter().find(|(i, _, _)| *i == stage_name) {
                let dim = self.cfg.stage_dim(s);
                let normed = nn::layer_norm(&x, nw, nb, dim, self.eps);
                feature_maps.push(tokens_to_chw(&normed, cur_h, cur_w, dim));
            }
            // Downsample for the next stage.
            if let Some(ds) = &stage.downsample {
                let dim = self.cfg.stage_dim(s);
                let (nx, nh, nw_) = patch_merge(&x, cur_h, cur_w, dim, ds, self.eps);
                x = nx;
                cur_h = nh;
                cur_w = nw_;
            }
        }
        feature_maps
    }

    /// One Swin transformer block on tokens `x [h*w, dim]`.
    fn run_block(&self, block: &Block, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let dim = block.dim;
        let ws = self.cfg.window_size;
        let n = h * w;
        let shortcut = x.to_vec();
        let normed = nn::layer_norm(x, &block.ln_before_w, &block.ln_before_b, dim, self.eps);

        // Pad feature map to a multiple of the window size.
        let (mut feat, hp, wp) = pad_tokens(&normed, h, w, dim, ws);

        // Cyclic shift.
        if block.shift > 0 {
            feat = roll(
                &feat,
                hp,
                wp,
                dim,
                -(block.shift as isize),
                -(block.shift as isize),
            );
        }

        // Window partition → windows [nW, ws*ws, dim].
        let (windows, n_wh, n_ww) = window_partition(&feat, hp, wp, dim, ws);
        let n_win = n_wh * n_ww;
        let ws2 = ws * ws;

        // Precompute per-head relative position bias [heads, ws2, ws2].
        let rel_bias = self.relative_bias(block, ws2);

        // Shifted-window attention mask [n_win, ws2, ws2] (or None).
        let win_mask = if block.shift > 0 {
            Some(window_attn_mask(hp, wp, ws, block.shift))
        } else {
            None
        };

        // Attention per window.
        let mut attn_out = vec![0f32; n_win * ws2 * dim];
        for wi in 0..n_win {
            let win = &windows[wi * ws2 * dim..(wi + 1) * ws2 * dim];
            let bias: Vec<f32> = match &win_mask {
                None => rel_bias.clone(),
                Some(m) => {
                    // rel_bias[h,i,j] + mask[wi,i,j] (broadcast over heads).
                    let mut b = rel_bias.clone();
                    let mw = &m[wi * ws2 * ws2..(wi + 1) * ws2 * ws2];
                    for hh in 0..block.heads {
                        for ij in 0..ws2 * ws2 {
                            b[hh * ws2 * ws2 + ij] += mw[ij];
                        }
                    }
                    b
                }
            };
            let out = nn::mha(
                win,
                win,
                win,
                ws2,
                ws2,
                dim,
                block.heads,
                &block.q_w,
                &block.q_b,
                &block.k_w,
                &block.k_b,
                &block.v_w,
                &block.v_b,
                &block.attn_out_w,
                &block.attn_out_b,
                AttnBias::PerHead(&bias),
            );
            attn_out[wi * ws2 * dim..(wi + 1) * ws2 * dim].copy_from_slice(&out);
        }

        // Merge windows → [hp, wp, dim].
        let mut merged = window_reverse(&attn_out, n_wh, n_ww, ws, dim);

        // Reverse cyclic shift.
        if block.shift > 0 {
            merged = roll(
                &merged,
                hp,
                wp,
                dim,
                block.shift as isize,
                block.shift as isize,
            );
        }

        // Crop padding back to [h, w, dim] → tokens [n, dim].
        let cropped = crop_tokens(&merged, hp, wp, h, w, dim);

        // Residual.
        let mut hidden = vec![0f32; n * dim];
        for i in 0..n * dim {
            hidden[i] = shortcut[i] + cropped[i];
        }

        // FFN: ln_after → intermediate(gelu) → output, residual.
        let inter_dim = block.inter_b.len();
        let normed2 = nn::layer_norm(&hidden, &block.ln_after_w, &block.ln_after_b, dim, self.eps);
        let mut inter = nn::linear(&normed2, n, dim, &block.inter_w, inter_dim, &block.inter_b);
        nn::gelu_erf(&mut inter);
        let ffn = nn::linear(&inter, n, inter_dim, &block.out_w, dim, &block.out_b);
        for i in 0..n * dim {
            hidden[i] += ffn[i];
        }
        hidden
    }

    /// One Swin block as a **single** fused HIR graph: `ln_before` → pad → cyclic
    /// shift → window partition → batched window attention → window reverse →
    /// reverse shift → crop → residual → `ln_after` → FFN → residual. All window
    /// geometry (pad/roll/partition/reverse/crop) is expressed with graph
    /// reshape/transpose/narrow/concat ops so vision tokens stay on-device across
    /// the whole block — no per-sub-op host round-trip. The additive attention
    /// bias (relative-position table + shift mask) is data-independent of the
    /// activations, so it's precomputed on the host and passed in. Numerically
    /// equivalent to [`Self::run_block`] (the parity oracle).
    fn run_block_ir(&self, block: &Block, x: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        let dim = block.dim;
        let ws = self.cfg.window_size;
        let n = h * w;
        let ws2 = ws * ws;
        let heads = block.heads;
        let shift = block.shift;

        // Host geometry sizes (must match `pad_tokens` / `window_partition`).
        let hp = h.div_ceil(ws) * ws;
        let wp = w.div_ceil(ws) * ws;
        let (n_wh, n_ww) = (hp / ws, wp / ws);
        let n_win = n_wh * n_ww;

        // Host-precomputed additive bias [n_win, heads, ws2, ws2] = relative-pos
        // bias (+ shift region mask) — independent of the activations.
        let rel_bias = self.relative_bias(block, ws2);
        let win_mask = if shift > 0 {
            Some(window_attn_mask(hp, wp, ws, shift))
        } else {
            None
        };
        let mut bias = vec![0f32; n_win * heads * ws2 * ws2];
        for wi in 0..n_win {
            for hh in 0..heads {
                for ij in 0..ws2 * ws2 {
                    let mut b = rel_bias[hh * ws2 * ws2 + ij];
                    if let Some(m) = &win_mask {
                        b += m[wi * ws2 * ws2 + ij];
                    }
                    bias[(wi * heads + hh) * ws2 * ws2 + ij] = b;
                }
            }
        }

        let mut hir = HirModule::new("swin_block");
        let mut params = ir::Params::new();
        let mut g = HirMut::new(&mut hir);
        let x_n = g.input("x", Shape::new(&[n, dim], DType::F32));
        let bias_n = g.input("bias", Shape::new(&[n_win, heads, ws2, ws2], DType::F32));

        // ln_before → [h, w, dim].
        let normed = ir::layer_norm(
            &mut g,
            &mut params,
            "lnb",
            x_n,
            &block.ln_before_w,
            &block.ln_before_b,
            self.eps,
        );
        let mut feat = g.reshape_(normed, vec![h as i64, w as i64, dim as i64]);

        // Pad bottom/right with zeros (`pad_tokens`): concat zero rows/cols.
        if hp > h {
            params.insert("padh".into(), vec![0f32; (hp - h) * w * dim]);
            let zh = g.param("padh", Shape::new(&[hp - h, w, dim], DType::F32));
            feat = g.concat_(vec![feat, zh], 0);
        }
        if wp > w {
            params.insert("padw".into(), vec![0f32; hp * (wp - w) * dim]);
            let zw = g.param("padw", Shape::new(&[hp, wp - w, dim], DType::F32));
            feat = g.concat_(vec![feat, zw], 1);
        }

        // Cyclic shift (forward = roll by -shift): out[y]=in[(y+shift)%L].
        if shift > 0 {
            feat = roll_graph(&mut g, feat, 0, hp, shift);
            feat = roll_graph(&mut g, feat, 1, wp, shift);
        }

        // Window partition [hp, wp, dim] → [n_win, ws2, dim] → flatten [n_win·ws2, dim].
        let feat = g.reshape_(
            feat,
            vec![n_wh as i64, ws as i64, n_ww as i64, ws as i64, dim as i64],
        );
        let feat = g.transpose_(feat, vec![0, 2, 1, 3, 4]);
        let windows = g.reshape_(feat, vec![(n_win * ws2) as i64, dim as i64]);

        // Batched window attention.
        let wa = WindowAttnWeights {
            q_w: block.q_w.clone(),
            q_b: block.q_b.clone(),
            k_w: block.k_w.clone(),
            k_b: block.k_b.clone(),
            v_w: block.v_w.clone(),
            v_b: block.v_b.clone(),
            o_w: block.attn_out_w.clone(),
            o_b: block.attn_out_b.clone(),
        };
        let attn = crate::swin_ir::build_window_attn(
            &mut g,
            &mut params,
            &wa,
            "wa",
            windows,
            bias_n,
            n_win,
            ws2,
            dim,
            heads,
        );

        // Window reverse [n_win·ws2, dim] → [hp, wp, dim].
        let merged = g.reshape_(
            attn,
            vec![n_wh as i64, n_ww as i64, ws as i64, ws as i64, dim as i64],
        );
        let merged = g.transpose_(merged, vec![0, 2, 1, 3, 4]);
        let mut merged = g.reshape_(merged, vec![hp as i64, wp as i64, dim as i64]);

        // Reverse cyclic shift (roll by +shift): out[y]=in[(y-shift)%L].
        if shift > 0 {
            merged = roll_graph(&mut g, merged, 0, hp, hp - shift);
            merged = roll_graph(&mut g, merged, 1, wp, wp - shift);
        }
        // Crop padding back to [h, w, dim] → tokens [n, dim].
        if hp > h {
            merged = g.narrow_(merged, 0, 0, h);
        }
        if wp > w {
            merged = g.narrow_(merged, 1, 0, w);
        }
        let cropped = g.reshape_(merged, vec![n as i64, dim as i64]);

        // Residual 1 (on the original input) → ln_after → ReLU/GELU-FFN → residual 2.
        let hidden = g.add(x_n, cropped);
        let inter_dim = block.inter_b.len();
        let normed2 = ir::layer_norm(
            &mut g,
            &mut params,
            "lna",
            hidden,
            &block.ln_after_w,
            &block.ln_after_b,
            self.eps,
        );
        let f1 = ir::linear(
            &mut g,
            &mut params,
            "fc1",
            normed2,
            dim,
            inter_dim,
            &block.inter_w,
            &block.inter_b,
            1.0,
        );
        let act = g.gelu(f1);
        let f2 = ir::linear(
            &mut g,
            &mut params,
            "fc2",
            act,
            inter_dim,
            dim,
            &block.out_w,
            &block.out_b,
            1.0,
        );
        let out = g.add(hidden, f2);
        g.set_outputs(vec![out]);

        let outs = ir::compile_and_run(hir, params, self.device, &[("x", x), ("bias", &bias)])?;
        Ok(outs.into_iter().next().unwrap_or_default())
    }

    /// Gather per-head relative position bias `[heads, ws2, ws2]`.
    fn relative_bias(&self, block: &Block, ws2: usize) -> Vec<f32> {
        let heads = block.heads;
        let mut bias = vec![0f32; heads * ws2 * ws2];
        for i in 0..ws2 {
            for j in 0..ws2 {
                let idx = block.rel_index[i * ws2 + j];
                for hh in 0..heads {
                    bias[(hh * ws2 + i) * ws2 + j] = block.rel_bias_table[idx * heads + hh];
                }
            }
        }
        bias
    }
}

// ---- geometry helpers ----

/// Pad a CHW image so H,W are multiples of `m` (zeros, bottom/right).
fn pad_image_chw(x: &[f32], c: usize, h: usize, w: usize, m: usize) -> (Vec<f32>, usize, usize) {
    let hp = h.div_ceil(m) * m;
    let wp = w.div_ceil(m) * m;
    if hp == h && wp == w {
        return (x.to_vec(), h, w);
    }
    let mut out = vec![0f32; c * hp * wp];
    for ch in 0..c {
        for y in 0..h {
            for xx in 0..w {
                out[(ch * hp + y) * wp + xx] = x[(ch * h + y) * w + xx];
            }
        }
    }
    (out, hp, wp)
}

/// Patch-embed conv: stride = kernel = `ps`, no padding. Input CHW, output
/// tokens `[gh*gw, out_c]` (channel-last per token).
fn patch_embed_conv(
    img: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [out_c, in_c, ps, ps]
    bias: &[f32],
    out_c: usize,
    ps: usize,
) -> Vec<f32> {
    let gh = h / ps;
    let gw = w / ps;
    let ntok = gh * gw;
    let k = in_c * ps * ps;
    // im2col (channel-last per token) + BLAS: out[token, oc] = Σ_k colT[token,k]·weight[oc,k].
    // `weight` is [out_c, in_c, ps, ps] = [out_c, k], so this is a B-transposed
    // GEMM. Replaces a naive 5-deep loop — the patch embed was part of Swin's
    // host-side "outside stages" cost on every backend.
    let mut col_t = vec![0f32; ntok * k];
    for oy in 0..gh {
        for ox in 0..gw {
            let row = &mut col_t[(oy * gw + ox) * k..(oy * gw + ox) * k + k];
            for ic in 0..in_c {
                for ky in 0..ps {
                    for kx in 0..ps {
                        row[(ic * ps + ky) * ps + kx] =
                            img[(ic * h + oy * ps + ky) * w + ox * ps + kx];
                    }
                }
            }
        }
    }
    let mut tokens = vec![0f32; ntok * out_c];
    rlx_cpu::blas::sgemm_bt(&col_t, weight, &mut tokens, ntok, k, out_c, 1.0);
    for t in 0..ntok {
        let tok = &mut tokens[t * out_c..(t + 1) * out_c];
        for (oc, v) in tok.iter_mut().enumerate() {
            *v += bias[oc];
        }
    }
    tokens
}

/// Tokens `[h*w, c]` → CHW `[c, h, w]`.
fn tokens_to_chw(x: &[f32], h: usize, w: usize, c: usize) -> FeatureMap {
    let mut data = vec![0f32; c * h * w];
    for y in 0..h {
        for xx in 0..w {
            for ch in 0..c {
                data[(ch * h + y) * w + xx] = x[(y * w + xx) * c + ch];
            }
        }
    }
    FeatureMap { data, c, h, w }
}

/// Pad token grid `[h*w, c]` to `[hp*wp, c]` (zeros bottom/right).
fn pad_tokens(x: &[f32], h: usize, w: usize, c: usize, m: usize) -> (Vec<f32>, usize, usize) {
    let hp = h.div_ceil(m) * m;
    let wp = w.div_ceil(m) * m;
    if hp == h && wp == w {
        return (x.to_vec(), h, w);
    }
    let mut out = vec![0f32; hp * wp * c];
    for y in 0..h {
        for xx in 0..w {
            let src = &x[(y * w + xx) * c..(y * w + xx) * c + c];
            out[(y * wp + xx) * c..(y * wp + xx) * c + c].copy_from_slice(src);
        }
    }
    (out, hp, wp)
}

/// Crop token grid `[hp*wp, c]` back to `[h*w, c]`.
fn crop_tokens(x: &[f32], hp: usize, wp: usize, h: usize, w: usize, c: usize) -> Vec<f32> {
    if hp == h && wp == w {
        return x.to_vec();
    }
    let mut out = vec![0f32; h * w * c];
    for y in 0..h {
        for xx in 0..w {
            let src = &x[(y * wp + xx) * c..(y * wp + xx) * c + c];
            out[(y * w + xx) * c..(y * w + xx) * c + c].copy_from_slice(src);
        }
    }
    out
}

/// Cyclic roll of `[h*w, c]` grid by (`sh`, `sw`) along (height, width).
fn roll(x: &[f32], h: usize, w: usize, c: usize, sh: isize, sw: isize) -> Vec<f32> {
    let mut out = vec![0f32; h * w * c];
    for y in 0..h {
        let sy = ((y as isize - sh).rem_euclid(h as isize)) as usize;
        for xx in 0..w {
            let sx = ((xx as isize - sw).rem_euclid(w as isize)) as usize;
            let src = &x[(sy * w + sx) * c..(sy * w + sx) * c + c];
            out[(y * w + xx) * c..(y * w + xx) * c + c].copy_from_slice(src);
        }
    }
    out
}

/// Cyclic roll along `axis` of a graph tensor of length `len`: returns a node
/// whose index `y` equals input index `(y + r) % len` (`0 < r < len`). Built from
/// `narrow_`/`concat_`. With `r = shift` this is the forward shift (host
/// `roll(-shift)`); with `r = len - shift` it's the reverse (host `roll(+shift)`).
fn roll_graph(g: &mut HirMut<'_>, x: HirNodeId, axis: usize, len: usize, r: usize) -> HirNodeId {
    if r == 0 || r == len {
        return x;
    }
    let head = g.narrow_(x, axis, r, len - r);
    let tail = g.narrow_(x, axis, 0, r);
    g.concat_(vec![head, tail], axis)
}

/// Partition `[h*w, c]` into windows `[n_wh*n_ww, ws*ws, c]`.
fn window_partition(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    ws: usize,
) -> (Vec<f32>, usize, usize) {
    let n_wh = h / ws;
    let n_ww = w / ws;
    let mut out = vec![0f32; n_wh * n_ww * ws * ws * c];
    for wy in 0..n_wh {
        for wx in 0..n_ww {
            let win = (wy * n_ww + wx) * ws * ws * c;
            for iy in 0..ws {
                for ix in 0..ws {
                    let gy = wy * ws + iy;
                    let gx = wx * ws + ix;
                    let src = &x[(gy * w + gx) * c..(gy * w + gx) * c + c];
                    let dst = win + (iy * ws + ix) * c;
                    out[dst..dst + c].copy_from_slice(src);
                }
            }
        }
    }
    (out, n_wh, n_ww)
}

/// Inverse of [`window_partition`] → `[h*w, c]` with `h = n_wh*ws`, `w = n_ww*ws`.
fn window_reverse(win: &[f32], n_wh: usize, n_ww: usize, ws: usize, c: usize) -> Vec<f32> {
    let h = n_wh * ws;
    let w = n_ww * ws;
    let mut out = vec![0f32; h * w * c];
    for wy in 0..n_wh {
        for wx in 0..n_ww {
            let wbase = (wy * n_ww + wx) * ws * ws * c;
            for iy in 0..ws {
                for ix in 0..ws {
                    let gy = wy * ws + iy;
                    let gx = wx * ws + ix;
                    let src = wbase + (iy * ws + ix) * c;
                    out[(gy * w + gx) * c..(gy * w + gx) * c + c]
                        .copy_from_slice(&win[src..src + c]);
                }
            }
        }
    }
    out
}

/// Shifted-window attention mask `[n_win, ws2, ws2]` with 0 (allowed) / -100.
fn window_attn_mask(hp: usize, wp: usize, ws: usize, shift: usize) -> Vec<f32> {
    // Region id per grid cell (HF img_mask construction).
    let mut img = vec![0i32; hp * wp];
    let h_slices = [(0, hp - ws), (hp - ws, hp - shift), (hp - shift, hp)];
    let w_slices = [(0, wp - ws), (wp - ws, wp - shift), (wp - shift, wp)];
    let mut cnt = 0i32;
    for (h0, h1) in h_slices {
        for (w0, w1) in w_slices {
            for y in h0..h1 {
                for x in w0..w1 {
                    img[y * wp + x] = cnt;
                }
            }
            cnt += 1;
        }
    }
    // Partition region ids into windows and compute pairwise mask.
    let (regions, n_wh, n_ww) = {
        let f: Vec<f32> = img.iter().map(|&v| v as f32).collect();
        window_partition(&f, hp, wp, 1, ws)
    };
    let n_win = n_wh * n_ww;
    let ws2 = ws * ws;
    let mut mask = vec![0f32; n_win * ws2 * ws2];
    for wi in 0..n_win {
        for i in 0..ws2 {
            for j in 0..ws2 {
                let ri = regions[wi * ws2 + i];
                let rj = regions[wi * ws2 + j];
                if (ri - rj).abs() > f32::EPSILON {
                    mask[(wi * ws2 + i) * ws2 + j] = NEG;
                }
            }
        }
    }
    mask
}

/// PatchMerging: `[h*w, dim]` → `[(h/2)*(w/2), 2*dim]`.
fn patch_merge(
    x: &[f32],
    h: usize,
    w: usize,
    dim: usize,
    ds: &Downsample,
    eps: f32,
) -> (Vec<f32>, usize, usize) {
    // Pad to even H,W (HF maybe_pad).
    let (xp, hp, wp) = {
        let needs = h % 2 == 1 || w % 2 == 1;
        if needs {
            pad_tokens(x, h, w, dim, 2)
        } else {
            (x.to_vec(), h, w)
        }
    };
    let nh = hp / 2;
    let nw = wp / 2;
    let cat = 4 * dim;
    // Concatenate the four 2×2 sub-grids in HF order: (0::2,0::2),(1::2,0::2),(0::2,1::2),(1::2,1::2).
    let mut merged = vec![0f32; nh * nw * cat];
    let offsets = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)];
    for y in 0..nh {
        for xx in 0..nw {
            let dst = (y * nw + xx) * cat;
            for (qi, (oy, ox)) in offsets.iter().enumerate() {
                let gy = 2 * y + oy;
                let gx = 2 * xx + ox;
                let src = &xp[(gy * wp + gx) * dim..(gy * wp + gx) * dim + dim];
                merged[dst + qi * dim..dst + qi * dim + dim].copy_from_slice(src);
            }
        }
    }
    let normed = nn::layer_norm(&merged, &ds.norm_w, &ds.norm_b, cat, eps);
    let out = nn::linear(&normed, nh * nw, cat, &ds.reduction_w, 2 * dim, &[]);
    (out, nh, nw)
}

// ---- graph (on-device) compute helpers ----

/// LayerNorm over the last `dim` of `[rows, dim]`, run as a one-node graph.
fn ln_ir(
    device: Device,
    x: &[f32],
    rows: usize,
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    let mut hir = HirModule::new("swin_ln");
    let mut params = ir::Params::new();
    let mut g = HirMut::new(&mut hir);
    let x_n = g.input("x", Shape::new(&[rows, dim], DType::F32));
    let out = ir::layer_norm(&mut g, &mut params, "ln", x_n, gamma, beta, eps);
    g.set_outputs(vec![out]);
    let outs = ir::compile_and_run(hir, params, device, &[("x", x)])?;
    Ok(outs.into_iter().next().unwrap_or_default())
}

/// PatchMerging (graph): host concat of the four 2×2 sub-grids, then
/// `LN → reduction linear (no bias)` as one graph. `[h*w, dim]` →
/// `[(h/2)*(w/2), 2*dim]`. Mirrors [`patch_merge`].
fn patch_merge_ir(
    device: Device,
    x: &[f32],
    h: usize,
    w: usize,
    dim: usize,
    ds: &Downsample,
    eps: f32,
) -> Result<(Vec<f32>, usize, usize)> {
    let (xp, hp, wp) = {
        let needs = h % 2 == 1 || w % 2 == 1;
        if needs {
            pad_tokens(x, h, w, dim, 2)
        } else {
            (x.to_vec(), h, w)
        }
    };
    let nh = hp / 2;
    let nw = wp / 2;
    let cat = 4 * dim;
    let mut merged = vec![0f32; nh * nw * cat];
    let offsets = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)];
    for y in 0..nh {
        for xx in 0..nw {
            let dst = (y * nw + xx) * cat;
            for (qi, (oy, ox)) in offsets.iter().enumerate() {
                let gy = 2 * y + oy;
                let gx = 2 * xx + ox;
                let src = &xp[(gy * wp + gx) * dim..(gy * wp + gx) * dim + dim];
                merged[dst + qi * dim..dst + qi * dim + dim].copy_from_slice(src);
            }
        }
    }

    let mut hir = HirModule::new("swin_patch_merge");
    let mut params = ir::Params::new();
    let mut g = HirMut::new(&mut hir);
    let m_n = g.input("m", Shape::new(&[nh * nw, cat], DType::F32));
    let normed = ir::layer_norm(&mut g, &mut params, "ln", m_n, &ds.norm_w, &ds.norm_b, eps);
    let out = ir::linear(
        &mut g,
        &mut params,
        "red",
        normed,
        cat,
        2 * dim,
        &ds.reduction_w,
        &[],
        1.0,
    );
    g.set_outputs(vec![out]);
    let outs = ir::compile_and_run(hir, params, device, &[("m", &merged)])?;
    Ok((outs.into_iter().next().unwrap_or_default(), nh, nw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_partition_roundtrip() {
        let (h, w, c, ws) = (4, 6, 2, 2);
        let x: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let (win, nwh, nww) = window_partition(&x, h, w, c, ws);
        assert_eq!((nwh, nww), (2, 3));
        let back = window_reverse(&win, nwh, nww, ws, c);
        assert_eq!(back, x);
    }

    #[test]
    fn roll_is_invertible() {
        let (h, w, c) = (4, 4, 1);
        let x: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let r = roll(&x, h, w, c, -2, -1);
        let back = roll(&r, h, w, c, 2, 1);
        assert_eq!(back, x);
    }

    #[test]
    fn shifted_mask_blocks_cross_region() {
        // 4x4 grid, window 2, shift 1 → some windows straddle regions.
        let m = window_attn_mask(4, 4, 2, 1);
        // At least one masked (-100) entry must exist.
        assert!(m.iter().any(|&v| v < -1.0));
        // Diagonal entries (i==j) are always same-region → 0.
        let ws2 = 4;
        let n_win = m.len() / (ws2 * ws2);
        for wi in 0..n_win {
            for i in 0..ws2 {
                assert_eq!(m[(wi * ws2 + i) * ws2 + i], 0.0);
            }
        }
    }
}
