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

//! SAM 2 memory encoder — host-side.
//!
//! Mirrors `sam2/modeling/memory_encoder.py` exactly:
//!
//! ```text
//!   MemoryEncoder(pix_feat, masks):
//!     masks = sigmoid(masks) if not skip_mask_sigmoid
//!     masks = MaskDownSampler(masks)        # 1×1024×1024 → 256×64×64
//!     pix_feat = pix_feat_proj(pix_feat)    # 1×1 conv 256→256
//!     x = pix_feat + masks
//!     x = Fuser(x)                          # 2 × CXBlock
//!     x = out_proj(x)                       # 1×1 conv 256→out_dim (64)
//!     pos = PositionEmbeddingSine(x)        # sinusoidal 2-D PE
//!     return (x, pos)
//! ```
//!
//! `MaskDownSampler` is a stack of `log_stride(total_stride)` blocks of
//! `Conv2d(k,s,p) → LayerNorm2d → GELU` that grow the channel dim by
//! `stride²` each step (1 → 4 → 16 → 64 → 256 for the default
//! stride=2, total_stride=16). A final 1×1 conv projects to
//! `embed_dim=in_dim=256`.
//!
//! `Fuser` is a ConvNeXt-style stack — depthwise Conv k=7 → LN →
//! pointwise Linear (4× expansion) → GELU → pointwise Linear → optional
//! per-channel `gamma` (LayerScale) → residual.

use super::config::{SAM2_IMG_SIZE, Sam2MemoryEncoderConfig};
use super::memory_mask_ir::{
    Sam2MemoryConv1x1Compiled, Sam2MemoryFuserCompiled, Sam2MemoryMaskDownCompiled,
    Sam2MemoryPrefixCompiled,
};
use super::prompt_encoder::{conv2d_1x1, gelu_erf_inplace, layernorm2d_nchw, sigmoid_inplace};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;
use std::f32::consts::PI;

// ─── Weight structs ─────────────────────────────────────────────────

pub struct Sam2MaskDownSamplerWeights {
    /// Per-level `(conv_w, conv_b, ln_g, ln_b)` for the down-sampling
    /// Conv → LN2d → GELU pattern.
    pub levels: Vec<DownSampleLevel>,
    /// Final 1×1 conv `[embed_dim, last_chans]`.
    pub final_conv_w: Vec<f32>,
    pub final_conv_b: Vec<f32>,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub embed_dim: usize,
}

pub struct DownSampleLevel {
    pub conv_w: Vec<f32>, // [out_c, in_c, k, k]
    pub conv_b: Vec<f32>, // [out_c]
    pub ln_g: Vec<f32>,   // [out_c]
    pub ln_b: Vec<f32>,
    pub in_c: usize,
    pub out_c: usize,
}

pub struct Sam2CXBlockWeights {
    pub dw_conv_w: Vec<f32>, // depthwise [dim, 1, k, k]
    pub dw_conv_b: Vec<f32>, // [dim]
    pub ln_g: Vec<f32>,
    pub ln_b: Vec<f32>,
    pub pw1_w: Vec<f32>, // [4·dim, dim]
    pub pw1_b: Vec<f32>, // [4·dim]
    pub pw2_w: Vec<f32>, // [dim, 4·dim]
    pub pw2_b: Vec<f32>, // [dim]
    /// LayerScale per-channel gain (optional in reference; present
    /// when `layer_scale_init_value > 0`).
    pub gamma: Option<Vec<f32>>,
    pub dim: usize,
    pub kernel: usize,
    pub padding: usize,
}

pub struct Sam2FuserWeights {
    /// Optional input-projection 1×1 conv (rarely used).
    pub input_proj_w: Option<Vec<f32>>,
    pub input_proj_b: Option<Vec<f32>>,
    pub layers: Vec<Sam2CXBlockWeights>,
    pub dim: usize,
}

pub struct Sam2MemoryEncoderWeights {
    pub mask_downsampler: Sam2MaskDownSamplerWeights,
    pub prefix: Option<Sam2MemoryPrefixCompiled>,
    pub mask_down: Option<Sam2MemoryMaskDownCompiled>,
    pub pix_proj: Option<Sam2MemoryConv1x1Compiled>,
    pub fuser_ir: Option<Sam2MemoryFuserCompiled>,
    pub out_proj_ir: Option<Sam2MemoryConv1x1Compiled>,
    pub pix_feat_proj_w: Vec<f32>, // [in_dim, in_dim, 1, 1]
    pub pix_feat_proj_b: Vec<f32>,
    pub fuser: Sam2FuserWeights,
    /// `out_proj`: 1×1 conv `in_dim → out_dim`. None when in_dim == out_dim
    /// (PyTorch `nn.Identity` in the reference).
    pub out_proj_w: Option<Vec<f32>>,
    pub out_proj_b: Option<Vec<f32>>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub pe_num_pos_feats: usize,
    pub pe_temperature: f32,
}

// ─── Weight extraction ─────────────────────────────────────────────

pub fn extract_memory_encoder_weights(
    weights: &mut WeightMap,
    cfg: &Sam2MemoryEncoderConfig,
) -> Result<Sam2MemoryEncoderWeights> {
    let mask_downsampler = extract_mask_downsampler(weights, cfg)?;

    let (pix_feat_proj_w, sh) = weights.take("memory_encoder.pix_feat_proj.weight")?;
    ensure!(
        sh == vec![cfg.in_dim, cfg.in_dim, 1, 1],
        "pix_feat_proj.weight shape {sh:?} not [{}, {}, 1, 1]",
        cfg.in_dim,
        cfg.in_dim
    );
    let (pix_feat_proj_b, _) = weights.take("memory_encoder.pix_feat_proj.bias")?;

    let fuser = extract_fuser(weights, cfg)?;

    let (out_proj_w, out_proj_b) = if cfg.in_dim == cfg.out_dim {
        (None, None)
    } else {
        let (w, sh) = weights.take("memory_encoder.out_proj.weight")?;
        ensure!(
            sh == vec![cfg.out_dim, cfg.in_dim, 1, 1],
            "out_proj.weight shape {sh:?} not [{}, {}, 1, 1]",
            cfg.out_dim,
            cfg.in_dim
        );
        let (b, _) = weights.take("memory_encoder.out_proj.bias")?;
        (Some(w), Some(b))
    };

    Ok(Sam2MemoryEncoderWeights {
        mask_downsampler,
        prefix: None,
        mask_down: None,
        pix_proj: None,
        fuser_ir: None,
        out_proj_ir: None,
        pix_feat_proj_w,
        pix_feat_proj_b,
        fuser,
        out_proj_w,
        out_proj_b,
        in_dim: cfg.in_dim,
        out_dim: cfg.out_dim,
        pe_num_pos_feats: cfg.pe_num_pos_feats,
        pe_temperature: cfg.pe_temperature,
    })
}

fn extract_mask_downsampler(
    weights: &mut WeightMap,
    cfg: &Sam2MemoryEncoderConfig,
) -> Result<Sam2MaskDownSamplerWeights> {
    // num_layers = log_stride(total_stride). Reference asserts
    // `stride ** num_layers == total_stride`.
    let mut num_layers = 0;
    let mut acc = 1usize;
    while acc < cfg.mask_downsampler_total_stride {
        acc *= cfg.mask_downsampler_stride;
        num_layers += 1;
    }
    ensure!(
        acc == cfg.mask_downsampler_total_stride,
        "mask_downsampler total_stride {} must be a power of stride {}",
        cfg.mask_downsampler_total_stride,
        cfg.mask_downsampler_stride
    );

    let mut levels = Vec::with_capacity(num_layers);
    let mut in_c = 1usize;
    let stride2 = cfg.mask_downsampler_stride * cfg.mask_downsampler_stride;
    // Reference's MaskDownSampler `encoder` is `nn.Sequential` of
    // groups (Conv2d, LayerNorm2d, GELU) per level, plus a final
    // 1×1 conv. Group index increment is 3 per level.
    for li in 0..num_layers {
        let out_c = in_c * stride2;
        let conv_idx = li * 3;
        let ln_idx = conv_idx + 1;
        let (conv_w, sh) = weights.take(&format!(
            "memory_encoder.mask_downsampler.encoder.{conv_idx}.weight"
        ))?;
        ensure!(
            sh == vec![
                out_c,
                in_c,
                cfg.mask_downsampler_kernel,
                cfg.mask_downsampler_kernel
            ],
            "mask_downsampler conv {li} weight shape {sh:?} not [{out_c}, {in_c}, {}, {}]",
            cfg.mask_downsampler_kernel,
            cfg.mask_downsampler_kernel
        );
        let (conv_b, _) = weights.take(&format!(
            "memory_encoder.mask_downsampler.encoder.{conv_idx}.bias"
        ))?;
        let (ln_g, _) = weights.take(&format!(
            "memory_encoder.mask_downsampler.encoder.{ln_idx}.weight"
        ))?;
        let (ln_b, _) = weights.take(&format!(
            "memory_encoder.mask_downsampler.encoder.{ln_idx}.bias"
        ))?;
        levels.push(DownSampleLevel {
            conv_w,
            conv_b,
            ln_g,
            ln_b,
            in_c,
            out_c,
        });
        in_c = out_c;
    }
    // Final 1×1 conv goes at index num_layers*3.
    let final_idx = num_layers * 3;
    let (final_conv_w, sh) = weights.take(&format!(
        "memory_encoder.mask_downsampler.encoder.{final_idx}.weight"
    ))?;
    ensure!(
        sh == vec![cfg.in_dim, in_c, 1, 1],
        "mask_downsampler final conv weight shape {sh:?} not [{}, {in_c}, 1, 1]",
        cfg.in_dim
    );
    let (final_conv_b, _) = weights.take(&format!(
        "memory_encoder.mask_downsampler.encoder.{final_idx}.bias"
    ))?;

    Ok(Sam2MaskDownSamplerWeights {
        levels,
        final_conv_w,
        final_conv_b,
        kernel: cfg.mask_downsampler_kernel,
        stride: cfg.mask_downsampler_stride,
        padding: cfg.mask_downsampler_padding,
        embed_dim: cfg.in_dim,
    })
}

fn extract_fuser(
    weights: &mut WeightMap,
    cfg: &Sam2MemoryEncoderConfig,
) -> Result<Sam2FuserWeights> {
    let (input_proj_w, input_proj_b) = if cfg.fuser_input_projection {
        let (w, sh) = weights.take("memory_encoder.fuser.proj.weight")?;
        ensure!(
            sh == vec![cfg.fuser_dim, cfg.fuser_dim, 1, 1],
            "fuser.proj.weight shape {sh:?} not [{}, {}, 1, 1]",
            cfg.fuser_dim,
            cfg.fuser_dim
        );
        let (b, _) = weights.take("memory_encoder.fuser.proj.bias")?;
        (Some(w), Some(b))
    } else {
        (None, None)
    };

    let mut layers = Vec::with_capacity(cfg.fuser_num_layers);
    for i in 0..cfg.fuser_num_layers {
        let p = format!("memory_encoder.fuser.layers.{i}");
        let (dw_conv_w, sh) = weights.take(&format!("{p}.dwconv.weight"))?;
        // Depthwise conv: groups=dim → weight shape [dim, 1, k, k].
        let dim = cfg.fuser_dim;
        let k = cfg.fuser_kernel;
        if cfg.fuser_use_dwconv {
            ensure!(
                sh == vec![dim, 1, k, k],
                "{p}.dwconv.weight shape {sh:?} not [{dim}, 1, {k}, {k}]"
            );
        } else {
            ensure!(
                sh == vec![dim, dim, k, k],
                "{p}.dwconv.weight shape {sh:?} not [{dim}, {dim}, {k}, {k}]"
            );
        }
        let (dw_conv_b, _) = weights.take(&format!("{p}.dwconv.bias"))?;
        let (ln_g, _) = weights.take(&format!("{p}.norm.weight"))?;
        let (ln_b, _) = weights.take(&format!("{p}.norm.bias"))?;
        let (pw1_w, sh) = weights.take(&format!("{p}.pwconv1.weight"))?;
        ensure!(
            sh == vec![4 * dim, dim],
            "{p}.pwconv1.weight shape {sh:?} not [{}, {dim}]",
            4 * dim
        );
        let (pw1_b, _) = weights.take(&format!("{p}.pwconv1.bias"))?;
        let (pw2_w, _) = weights.take(&format!("{p}.pwconv2.weight"))?;
        let (pw2_b, _) = weights.take(&format!("{p}.pwconv2.bias"))?;
        let gamma = if cfg.fuser_layer_scale_init_value > 0.0 {
            let (g, _) = weights.take(&format!("{p}.gamma"))?;
            Some(g)
        } else {
            None
        };
        layers.push(Sam2CXBlockWeights {
            dw_conv_w,
            dw_conv_b,
            ln_g,
            ln_b,
            pw1_w,
            pw1_b,
            pw2_w,
            pw2_b,
            gamma,
            dim,
            kernel: k,
            padding: cfg.fuser_padding,
        });
    }
    Ok(Sam2FuserWeights {
        input_proj_w,
        input_proj_b,
        layers,
        dim: cfg.fuser_dim,
    })
}

/// Compile memory-encoder IR subgraphs (mask down, pix 1×1, fuser, optional out 1×1).
pub fn compile_memory_encoder_ir(
    weights: &mut Sam2MemoryEncoderWeights,
    mask_in_h: usize,
    mask_in_w: usize,
    feat_h: usize,
    feat_w: usize,
    device: Device,
    profile: &rlx_flow::CompileProfile,
) -> Result<()> {
    weights.prefix = Some(Sam2MemoryPrefixCompiled::compile_with_profile(
        &weights.mask_downsampler,
        weights.in_dim,
        mask_in_h,
        mask_in_w,
        feat_h,
        feat_w,
        &weights.pix_feat_proj_w,
        &weights.pix_feat_proj_b,
        device,
        profile,
    )?);
    weights.fuser_ir = Some(Sam2MemoryFuserCompiled::compile_with_profile(
        &weights.fuser,
        feat_h,
        feat_w,
        device,
        profile,
    )?);
    if let (Some(opw), Some(opb)) = (&weights.out_proj_w, &weights.out_proj_b) {
        weights.out_proj_ir = Some(Sam2MemoryConv1x1Compiled::compile_with_profile(
            weights.in_dim,
            weights.out_dim,
            feat_h,
            feat_w,
            opw,
            opb,
            device,
            profile,
        )?);
    }
    Ok(())
}

/// Back-compat alias for mask-downsampler-only compile.
pub fn compile_memory_mask_ir(
    weights: &mut Sam2MemoryEncoderWeights,
    mask_in_h: usize,
    mask_in_w: usize,
    device: Device,
) -> Result<()> {
    compile_memory_encoder_ir(
        weights,
        mask_in_h,
        mask_in_w,
        mask_in_h / total_stride(&weights.mask_downsampler),
        mask_in_w / total_stride(&weights.mask_downsampler),
        device,
        &rlx_flow::CompileProfile::sam2(),
    )
}

// ─── Forward ────────────────────────────────────────────────────────

pub struct Sam2MemoryEncoderOutput {
    /// `[out_dim, h, w]` memory feature map (typically 64×64×64).
    pub features: Vec<f32>,
    /// `[2·pe_num_pos_feats, h, w]` sinusoidal PE matching `features`.
    pub pos: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// Run the SAM 2 memory encoder.
///
/// `pix_feat`: stride-16 features `[in_dim, h, w]` (typically 256×64×64
/// from the FpnNeck level 2).
/// `masks`: mask logits `[1, H_full, W_full]` (or sigmoid probs, with
/// `skip_mask_sigmoid=true`). H_full = W_full = `SAM2_IMG_SIZE` (1024).
/// After MaskDownSampler the masks are at stride `total_stride=16`,
/// giving shape `[in_dim, h, w]` matching pix_feat.
pub fn memory_encoder_forward(
    w: &mut Sam2MemoryEncoderWeights,
    pix_feat: &[f32],
    masks: &[f32],
    pix_h: usize,
    pix_w: usize,
    skip_mask_sigmoid: bool,
) -> Result<Sam2MemoryEncoderOutput> {
    ensure!(
        pix_feat.len() == w.in_dim * pix_h * pix_w,
        "pix_feat len {} ≠ in_dim·h·w ({}·{pix_h}·{pix_w})",
        pix_feat.len(),
        w.in_dim
    );
    let in_h = SAM2_IMG_SIZE;
    let in_w = SAM2_IMG_SIZE;
    ensure!(
        masks.len() == in_h * in_w,
        "masks len {} ≠ H·W ({in_h}·{in_w}); pass a full-resolution mask",
        masks.len()
    );

    // 1) Sigmoid (optional).
    let mut m: Vec<f32> = masks.to_vec();
    if !skip_mask_sigmoid {
        sigmoid_inplace(&mut m);
    }

    // 2–4) MaskDownSampler + pix_feat_proj + add (fused or split).
    let x = if let Some(ref mut prefix) = w.prefix {
        prefix.run(&m, pix_feat)?
    } else {
        let m_down = if let Some(ref mut md) = w.mask_down {
            md.run(&m)?
        } else {
            mask_downsampler_forward(&w.mask_downsampler, &m, in_h, in_w)?
        };
        let down_h = in_h / total_stride(&w.mask_downsampler);
        let down_w = in_w / total_stride(&w.mask_downsampler);
        ensure!(
            down_h == pix_h && down_w == pix_w,
            "mask after downsampling ({down_h}×{down_w}) doesn't match pix_feat ({pix_h}×{pix_w})"
        );
        let mut x = if let Some(ref mut p) = w.pix_proj {
            p.run(pix_feat)?
        } else {
            conv2d_1x1(
                pix_feat,
                w.in_dim,
                w.in_dim,
                pix_h,
                pix_w,
                &w.pix_feat_proj_w,
                &w.pix_feat_proj_b,
            )
        };
        for i in 0..x.len() {
            x[i] += m_down[i];
        }
        x
    };

    // 5) Fuser.
    let x = if let Some(ref mut f) = w.fuser_ir {
        f.run(&x)?
    } else {
        fuser_forward(&w.fuser, x, pix_h, pix_w)
    };

    // 6) Optional out_proj.
    let features = if let Some(ref mut o) = w.out_proj_ir {
        o.run(&x)?
    } else if let (Some(opw), Some(opb)) = (&w.out_proj_w, &w.out_proj_b) {
        conv2d_1x1(&x, w.in_dim, w.out_dim, pix_h, pix_w, opw, opb)
    } else {
        x
    };

    // 7) Sinusoidal PE.
    let pos = sinusoidal_pos_2d(2 * w.pe_num_pos_feats, pix_h, pix_w, w.pe_temperature);

    Ok(Sam2MemoryEncoderOutput {
        features,
        pos,
        h: pix_h,
        w: pix_w,
    })
}

fn total_stride(d: &Sam2MaskDownSamplerWeights) -> usize {
    d.stride.pow(d.levels.len() as u32)
}

/// MaskDownSampler forward. `in`: `[1, H, W]`. Repeats
/// Conv(k,s,p) → LN2d → GELU `num_levels` times, then a final 1×1 conv
/// to `embed_dim`.
fn mask_downsampler_forward(
    w: &Sam2MaskDownSamplerWeights,
    input: &[f32],
    h: usize,
    ww: usize,
) -> Result<Vec<f32>> {
    let mut cur = input.to_vec();
    let mut cur_c = 1usize;
    let mut cur_h = h;
    let mut cur_w = ww;
    for level in &w.levels {
        let out_h = (cur_h + 2 * w.padding - w.kernel) / w.stride + 1;
        let out_w = (cur_w + 2 * w.padding - w.kernel) / w.stride + 1;
        cur = conv2d_general(
            &cur,
            cur_c,
            level.out_c,
            cur_h,
            cur_w,
            w.kernel,
            w.stride,
            w.padding,
            &level.conv_w,
            &level.conv_b,
        );
        cur_c = level.out_c;
        cur_h = out_h;
        cur_w = out_w;
        layernorm2d_nchw(
            &mut cur,
            cur_c,
            cur_h,
            cur_w,
            &level.ln_g,
            &level.ln_b,
            1e-6,
        );
        gelu_erf_inplace(&mut cur);
    }
    // Final 1×1 conv.
    let out = conv2d_1x1(
        &cur,
        cur_c,
        w.embed_dim,
        cur_h,
        cur_w,
        &w.final_conv_w,
        &w.final_conv_b,
    );
    Ok(out)
}

fn fuser_forward(w: &Sam2FuserWeights, mut x: Vec<f32>, h: usize, ww: usize) -> Vec<f32> {
    if let (Some(pw), Some(pb)) = (&w.input_proj_w, &w.input_proj_b) {
        x = conv2d_1x1(&x, w.dim, w.dim, h, ww, pw, pb);
    }
    for layer in &w.layers {
        x = cx_block_forward(layer, x, h, ww);
    }
    x
}

fn cx_block_forward(w: &Sam2CXBlockWeights, x: Vec<f32>, h: usize, ww: usize) -> Vec<f32> {
    let dim = w.dim;
    // Depthwise conv k×k pad=padding.
    let mut y = conv2d_depthwise_k_pad(
        &x,
        dim,
        h,
        ww,
        w.kernel,
        w.padding,
        &w.dw_conv_w,
        &w.dw_conv_b,
    );
    // LN over channel dim (NCHW per spatial pos).
    layernorm2d_nchw(&mut y, dim, h, ww, &w.ln_g, &w.ln_b, 1e-6);
    // Permute NCHW → NHWC, apply pointwise Linear(dim → 4·dim) → GELU
    // → Linear(4·dim → dim), permute back.
    let mut nhwc = vec![0f32; h * ww * dim];
    for c in 0..dim {
        for yy in 0..h {
            for xx in 0..ww {
                nhwc[(yy * ww + xx) * dim + c] = y[c * h * ww + yy * ww + xx];
            }
        }
    }
    let four_d = 4 * dim;
    let mut up = vec![0f32; h * ww * four_d];
    for r in 0..h * ww {
        for o in 0..four_d {
            let mut acc = w.pw1_b[o];
            for k in 0..dim {
                acc += nhwc[r * dim + k] * w.pw1_w[o * dim + k];
            }
            up[r * four_d + o] = acc;
        }
    }
    gelu_erf_inplace(&mut up);
    let mut down = vec![0f32; h * ww * dim];
    for r in 0..h * ww {
        for o in 0..dim {
            let mut acc = w.pw2_b[o];
            for k in 0..four_d {
                acc += up[r * four_d + k] * w.pw2_w[o * four_d + k];
            }
            down[r * dim + o] = acc;
        }
    }
    if let Some(gamma) = &w.gamma {
        for r in 0..h * ww {
            for c in 0..dim {
                down[r * dim + c] *= gamma[c];
            }
        }
    }
    // Permute NHWC → NCHW, add residual.
    let mut out = x;
    for c in 0..dim {
        for yy in 0..h {
            for xx in 0..ww {
                out[c * h * ww + yy * ww + xx] += down[(yy * ww + xx) * dim + c];
            }
        }
    }
    out
}

// ─── Generic conv helpers ───────────────────────────────────────────

/// Generic 2-D conv NCHW: `[in_c, h, w]` → `[out_c, h', w']` with
/// arbitrary kernel/stride/padding (no dilation).
fn conv2d_general(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    p: usize,
    weight: &[f32], // [out_c, in_c, k, k]
    bias: &[f32],   // [out_c]
) -> Vec<f32> {
    let out_h = (h + 2 * p - k) / s + 1;
    let out_w = (w + 2 * p - k) / s + 1;
    let mut out = vec![0f32; out_c * out_h * out_w];
    for oc in 0..out_c {
        let b = bias[oc];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = b;
                for ic in 0..in_c {
                    for ky in 0..k {
                        let iy = oy as isize * s as isize + ky as isize - p as isize;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..k {
                            let ix = ox as isize * s as isize + kx as isize - p as isize;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let v = input[ic * h * w + iy as usize * w + ix as usize];
                            let w_idx = ((oc * in_c + ic) * k + ky) * k + kx;
                            acc += v * weight[w_idx];
                        }
                    }
                }
                out[oc * out_h * out_w + oy * out_w + ox] = acc;
            }
        }
    }
    out
}

/// Depthwise 2-D conv k×k stride=1 padding=p. Weight `[dim, 1, k, k]`.
fn conv2d_depthwise_k_pad(
    input: &[f32],
    dim: usize,
    h: usize,
    w: usize,
    k: usize,
    p: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; dim * h * w];
    for c in 0..dim {
        let b = bias[c];
        let w_base = c * k * k; // weight is [dim, 1, k, k], so per-channel offset = c·k·k
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = b;
                for ky in 0..k {
                    let iy = oy as isize + ky as isize - p as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let ix = ox as isize + kx as isize - p as isize;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let v = input[c * h * w + iy as usize * w + ix as usize];
                        acc += v * weight[w_base + ky * k + kx];
                    }
                }
                out[c * h * w + oy * w + ox] = acc;
            }
        }
    }
    out
}

/// Reference `PositionEmbeddingSine` forward — same code path as the
/// FpnNeck PE but kept here so the memory-encoder output owns its PE
/// generator with its own `temperature` config option.
pub(super) fn sinusoidal_pos_2d(d_model: usize, h: usize, w: usize, temperature: f32) -> Vec<f32> {
    let nf = d_model / 2;
    let scale: f32 = 2.0 * PI;
    let eps: f32 = 1e-6;
    let mut out = vec![0f32; d_model * h * w];
    let mut dim_t = vec![0f32; nf];
    for i in 0..nf {
        let exp = 2.0 * ((i / 2) as f32) / (nf as f32);
        dim_t[i] = temperature.powf(exp);
    }
    for y in 0..h {
        let y_emb = ((y + 1) as f32) / ((h as f32) + eps) * scale;
        for x in 0..w {
            let x_emb = ((x + 1) as f32) / ((w as f32) + eps) * scale;
            for i in 0..nf {
                let py = y_emb / dim_t[i];
                let v = if i % 2 == 0 { py.sin() } else { py.cos() };
                out[i * h * w + y * w + x] = v;
            }
            for i in 0..nf {
                let px = x_emb / dim_t[i];
                let v = if i % 2 == 0 { px.sin() } else { px.cos() };
                out[(nf + i) * h * w + y * w + x] = v;
            }
        }
    }
    out
}
