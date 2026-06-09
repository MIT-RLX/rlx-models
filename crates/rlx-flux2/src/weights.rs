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

//! FLUX.2 transformer weight extraction from safetensors.

use super::adapt::prepare_weight_map;
use super::config::Flux2Config;
use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub w_t: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct RmsNormWeight {
    pub scale: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Flux2FeedForwardWeights {
    pub linear_in: LinearWeights,
    pub linear_out: LinearWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2DualAttnWeights {
    pub to_q: LinearWeights,
    pub to_k: LinearWeights,
    pub to_v: LinearWeights,
    pub norm_q: RmsNormWeight,
    pub norm_k: RmsNormWeight,
    pub add_q: LinearWeights,
    pub add_k: LinearWeights,
    pub add_v: LinearWeights,
    pub norm_added_q: RmsNormWeight,
    pub norm_added_k: RmsNormWeight,
    pub to_out: LinearWeights,
    pub to_add_out: LinearWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2ParallelAttnWeights {
    pub to_qkv_mlp: LinearWeights,
    pub norm_q: RmsNormWeight,
    pub norm_k: RmsNormWeight,
    pub to_out: LinearWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2DoubleBlockWeights {
    pub attn: Flux2DualAttnWeights,
    pub ff: Flux2FeedForwardWeights,
    pub ff_context: Flux2FeedForwardWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2SingleBlockWeights {
    pub attn: Flux2ParallelAttnWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2TimestepGuidanceWeights {
    pub timestep_linear1: LinearWeights,
    pub timestep_linear2: LinearWeights,
    pub guidance_linear1: Option<LinearWeights>,
    pub guidance_linear2: Option<LinearWeights>,
}

#[derive(Debug, Clone)]
pub struct Flux2ModulationWeights {
    pub linear: LinearWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2NormOutWeights {
    pub linear: LinearWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2Weights {
    pub x_embedder: LinearWeights,
    pub context_embedder: LinearWeights,
    pub time_guidance: Flux2TimestepGuidanceWeights,
    /// Second timestep embedder for flow-map dual-time (`t` vs `t′`). When `None`, dual-time
    /// forwards reuse [`Self::time_guidance`] for both (averaged embedding).
    pub time_guidance_target: Option<Flux2TimestepGuidanceWeights>,
    pub double_mod_img: Flux2ModulationWeights,
    pub double_mod_txt: Flux2ModulationWeights,
    pub single_mod: Flux2ModulationWeights,
    pub transformer_blocks: Vec<Flux2DoubleBlockWeights>,
    pub single_transformer_blocks: Vec<Flux2SingleBlockWeights>,
    pub norm_out: Flux2NormOutWeights,
    pub proj_out: LinearWeights,
}

/// Load denoiser weights from `.safetensors` or a single-file `.gguf` (BFL / ComfyUI naming).
pub fn load_flux2_weight_map(path: &Path) -> Result<WeightMap> {
    rlx_core::load_weight_map(path, rlx_core::FLUX_GGUF_ARCHES)
}

pub fn load_flux2_weights(path: &str, cfg: &Flux2Config) -> Result<Flux2Weights> {
    let wm = load_flux2_weight_map(Path::new(path))?;
    extract_flux2_weights(prepare_weight_map(wm), cfg)
}

pub fn extract_flux2_weights(wm: WeightMap, cfg: &Flux2Config) -> Result<Flux2Weights> {
    extract_flux2_weights_with_opts(wm, cfg, ExtractFlux2Opts::default())
}

pub fn extract_flux2_weights_with_opts(
    mut wm: WeightMap,
    cfg: &Flux2Config,
    opts: ExtractFlux2Opts<'_>,
) -> Result<Flux2Weights> {
    let guidance_embeds = cfg.guidance_embeds
        && (wm.has("time_guidance_embed.guidance_embedder.linear_1.weight")
            || wm.has("guidance_in.in_layer.weight"));

    let x_embedder =
        load_linear_with_opts(&mut wm, "x_embedder.weight", "x_embedder.bias", false, opts)?;
    let context_embedder = load_linear_with_opts(
        &mut wm,
        "context_embedder.weight",
        "context_embedder.bias",
        false,
        opts,
    )?;
    let time_guidance =
        load_time_guidance_block(&mut wm, "time_guidance_embed", guidance_embeds, opts)?;
    let time_guidance_target =
        try_load_time_guidance_block(&mut wm, "time_guidance_embed_target", guidance_embeds, opts)?
            .or_else(|| {
                if opts.dual_time_embedder {
                    Some(time_guidance.clone())
                } else {
                    None
                }
            });
    let double_mod_img = Flux2ModulationWeights {
        linear: load_linear_with_opts(
            &mut wm,
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_img.linear.bias",
            false,
            opts,
        )?,
    };
    let double_mod_txt = Flux2ModulationWeights {
        linear: load_linear_with_opts(
            &mut wm,
            "double_stream_modulation_txt.linear.weight",
            "double_stream_modulation_txt.linear.bias",
            false,
            opts,
        )?,
    };
    let single_mod = Flux2ModulationWeights {
        linear: load_linear_with_opts(
            &mut wm,
            "single_stream_modulation.linear.weight",
            "single_stream_modulation.linear.bias",
            false,
            opts,
        )?,
    };

    let mut transformer_blocks = Vec::with_capacity(cfg.num_layers);
    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        transformer_blocks.push(Flux2DoubleBlockWeights {
            attn: load_dual_attn(&mut wm, &p, opts)?,
            ff: load_ff(&mut wm, &format!("{p}.ff"), opts)?,
            ff_context: load_ff(&mut wm, &format!("{p}.ff_context"), opts)?,
        });
    }

    let mut single_transformer_blocks = Vec::with_capacity(cfg.num_single_layers);
    for i in 0..cfg.num_single_layers {
        let p = format!("single_transformer_blocks.{i}");
        single_transformer_blocks.push(Flux2SingleBlockWeights {
            attn: load_parallel_attn(&mut wm, &p, opts)?,
        });
    }

    let norm_out = Flux2NormOutWeights {
        linear: load_linear_with_opts(
            &mut wm,
            "norm_out.linear.weight",
            "norm_out.linear.bias",
            true,
            opts,
        )?,
    };
    let proj_out = load_linear_with_opts(&mut wm, "proj_out.weight", "proj_out.bias", false, opts)?;

    Ok(Flux2Weights {
        x_embedder,
        context_embedder,
        time_guidance,
        time_guidance_target,
        double_mod_img,
        double_mod_txt,
        single_mod,
        transformer_blocks,
        single_transformer_blocks,
        norm_out,
        proj_out,
    })
}

fn load_ff(
    wm: &mut WeightMap,
    prefix: &str,
    opts: ExtractFlux2Opts<'_>,
) -> Result<Flux2FeedForwardWeights> {
    Ok(Flux2FeedForwardWeights {
        linear_in: load_linear_with_opts(
            wm,
            &format!("{prefix}.linear_in.weight"),
            &format!("{prefix}.linear_in.bias"),
            true,
            opts,
        )?,
        linear_out: load_linear_with_opts(
            wm,
            &format!("{prefix}.linear_out.weight"),
            &format!("{prefix}.linear_out.bias"),
            true,
            opts,
        )?,
    })
}

fn load_dual_attn(
    wm: &mut WeightMap,
    prefix: &str,
    opts: ExtractFlux2Opts<'_>,
) -> Result<Flux2DualAttnWeights> {
    let ap = format!("{prefix}.attn");
    Ok(Flux2DualAttnWeights {
        to_q: load_linear_with_opts(
            wm,
            &format!("{ap}.to_q.weight"),
            &format!("{ap}.to_q.bias"),
            true,
            opts,
        )?,
        to_k: load_linear_with_opts(
            wm,
            &format!("{ap}.to_k.weight"),
            &format!("{ap}.to_k.bias"),
            true,
            opts,
        )?,
        to_v: load_linear_with_opts(
            wm,
            &format!("{ap}.to_v.weight"),
            &format!("{ap}.to_v.bias"),
            true,
            opts,
        )?,
        norm_q: load_rms(wm, &format!("{ap}.norm_q.weight"))?,
        norm_k: load_rms(wm, &format!("{ap}.norm_k.weight"))?,
        add_q: load_linear_with_opts(
            wm,
            &format!("{ap}.add_q_proj.weight"),
            &format!("{ap}.add_q_proj.bias"),
            true,
            opts,
        )?,
        add_k: load_linear_with_opts(
            wm,
            &format!("{ap}.add_k_proj.weight"),
            &format!("{ap}.add_k_proj.bias"),
            true,
            opts,
        )?,
        add_v: load_linear_with_opts(
            wm,
            &format!("{ap}.add_v_proj.weight"),
            &format!("{ap}.add_v_proj.bias"),
            true,
            opts,
        )?,
        norm_added_q: load_rms(wm, &format!("{ap}.norm_added_q.weight"))?,
        norm_added_k: load_rms(wm, &format!("{ap}.norm_added_k.weight"))?,
        to_out: load_linear_with_opts(
            wm,
            &format!("{ap}.to_out.0.weight"),
            &format!("{ap}.to_out.0.bias"),
            true,
            opts,
        )?,
        to_add_out: load_linear_with_opts(
            wm,
            &format!("{ap}.to_add_out.weight"),
            &format!("{ap}.to_add_out.bias"),
            true,
            opts,
        )?,
    })
}

fn load_parallel_attn(
    wm: &mut WeightMap,
    prefix: &str,
    opts: ExtractFlux2Opts<'_>,
) -> Result<Flux2ParallelAttnWeights> {
    let ap = format!("{prefix}.attn");
    Ok(Flux2ParallelAttnWeights {
        to_qkv_mlp: load_linear_with_opts(
            wm,
            &format!("{ap}.to_qkv_mlp_proj.weight"),
            &format!("{ap}.to_qkv_mlp_proj.bias"),
            true,
            opts,
        )?,
        norm_q: load_rms(wm, &format!("{ap}.norm_q.weight"))?,
        norm_k: load_rms(wm, &format!("{ap}.norm_k.weight"))?,
        to_out: load_linear_with_opts(
            wm,
            &format!("{ap}.to_out.weight"),
            &format!("{ap}.to_out.bias"),
            true,
            opts,
        )?,
    })
}

pub(crate) fn load_rms(wm: &mut WeightMap, key: &str) -> Result<RmsNormWeight> {
    let (scale, shape) = wm.take(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 1, "{key}: expected 1D scale");
    Ok(RmsNormWeight { scale })
}

#[derive(Copy, Clone, Default)]
pub struct ExtractFlux2Opts<'a> {
    pub typed_linears: Option<&'a crate::typed_linear::TypedLinearStore>,
    pub packed_linears: Option<&'a crate::packed::Flux2PackedParams>,
    /// Clone [`Flux2TimestepGuidanceWeights`] for `t′` when no `time_guidance_embed_target` tensors.
    pub dual_time_embedder: bool,
}

fn load_time_guidance_block(
    wm: &mut WeightMap,
    prefix: &str,
    guidance_embeds: bool,
    opts: ExtractFlux2Opts<'_>,
) -> Result<Flux2TimestepGuidanceWeights> {
    Ok(Flux2TimestepGuidanceWeights {
        timestep_linear1: load_linear_with_opts(
            wm,
            &format!("{prefix}.timestep_embedder.linear_1.weight"),
            &format!("{prefix}.timestep_embedder.linear_1.bias"),
            true,
            opts,
        )?,
        timestep_linear2: load_linear_with_opts(
            wm,
            &format!("{prefix}.timestep_embedder.linear_2.weight"),
            &format!("{prefix}.timestep_embedder.linear_2.bias"),
            true,
            opts,
        )?,
        guidance_linear1: if guidance_embeds {
            Some(load_linear_with_opts(
                wm,
                &format!("{prefix}.guidance_embedder.linear_1.weight"),
                &format!("{prefix}.guidance_embedder.linear_1.bias"),
                true,
                opts,
            )?)
        } else {
            None
        },
        guidance_linear2: if guidance_embeds {
            Some(load_linear_with_opts(
                wm,
                &format!("{prefix}.guidance_embedder.linear_2.weight"),
                &format!("{prefix}.guidance_embedder.linear_2.bias"),
                true,
                opts,
            )?)
        } else {
            None
        },
    })
}

fn try_load_time_guidance_block(
    wm: &mut WeightMap,
    prefix: &str,
    guidance_embeds: bool,
    opts: ExtractFlux2Opts<'_>,
) -> Result<Option<Flux2TimestepGuidanceWeights>> {
    let w1 = format!("{prefix}.timestep_embedder.linear_1.weight");
    if !wm.has(&w1) {
        return Ok(None);
    }
    Ok(Some(load_time_guidance_block(
        wm,
        prefix,
        guidance_embeds,
        opts,
    )?))
}

pub(crate) fn load_linear(
    wm: &mut WeightMap,
    w_key: &str,
    b_key: &str,
    expect_bias: bool,
) -> Result<LinearWeights> {
    load_linear_with_opts(wm, w_key, b_key, expect_bias, ExtractFlux2Opts::default())
}

pub(crate) fn load_linear_with_opts(
    wm: &mut WeightMap,
    w_key: &str,
    b_key: &str,
    _expect_bias: bool,
    opts: ExtractFlux2Opts<'_>,
) -> Result<LinearWeights> {
    let prefix = w_key.strip_suffix(".weight").unwrap_or(w_key);
    if !wm.has(w_key) {
        if let Some(tl) = opts.typed_linears.and_then(|t| t.get(prefix)) {
            return Ok(LinearWeights {
                w_t: Vec::new(),
                in_dim: tl.in_dim,
                out_dim: tl.out_dim,
                bias: tl.bias.clone(),
            });
        }
        if let Some(p) = opts.packed_linears.and_then(|m| m.get_nvfp4(prefix)) {
            return Ok(LinearWeights {
                w_t: Vec::new(),
                in_dim: p.in_dim,
                out_dim: p.out_dim,
                bias: p.bias.clone(),
            });
        }
        if let Some(p) = opts.packed_linears.and_then(|m| m.get_gguf(prefix)) {
            return Ok(LinearWeights {
                w_t: Vec::new(),
                in_dim: p.in_dim,
                out_dim: p.out_dim,
                bias: p.bias.clone(),
            });
        }
    }
    let (w_t, shape) = wm
        .take_transposed(w_key)
        .with_context(|| format!("missing {w_key}"))?;
    ensure!(shape.len() == 2, "{w_key}: expected 2D");
    let out_dim = shape[1];
    let in_dim = shape[0];
    let bias = if wm.has(b_key) {
        let (b, bshape) = wm.take(b_key)?;
        ensure!(bshape == vec![out_dim], "{b_key}: bias shape");
        b
    } else {
        vec![0.0f32; out_dim]
    };
    Ok(LinearWeights {
        w_t,
        in_dim,
        out_dim,
        bias,
    })
}
