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

//! FLUX.2 VAE decoder HIR (`flux2_vae_decode` trunk on GPU backends).

use super::config::Flux2VaeConfig;
use super::weights::{
    AttnBlockWeights, Conv2dWeight, DownEncoderBlockWeights, Flux2VaeWeights, GroupNormWeight,
    ResnetBlockWeights, UpDecoderBlockWeights,
};
use crate::builder::Flux2GraphParams;
use crate::compile_util::{
    compile_hir_cached, flux2_vae_decoder_aot_key, flux2_vae_encoder_aot_key,
};
use anyhow::Result;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{DType, Op, Shape};
use rlx_runtime::Device;

pub struct Flux2VaeGraph {
    pub hir: HirModule,
    pub params: Flux2GraphParams,
}

pub fn build_flux2_vae_hir(
    cfg: &Flux2VaeConfig,
    weights: &Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<Flux2VaeGraph> {
    let lc = cfg.latent_channels;
    let f = DType::F32;
    let mut hir =
        HirModule::new("flux2_vae_decoder").with_fusion_policy(rlx_ir::hir::FusionPolicy::Direct);
    let mut params = Flux2GraphParams::new();
    let latents = hir.input("latents", Shape::new(&[batch, lc, h, w], f));
    let mut b = VaeHirBuilder::from_emit_parts(&mut hir, &mut params, cfg, weights, batch, h, w);
    let (out, _, _, _) = b.emit_decoder(latents)?;
    hir.outputs = vec![out];
    Ok(Flux2VaeGraph { hir, params })
}

pub fn build_flux2_vae_encoder_hir(
    cfg: &Flux2VaeConfig,
    weights: &Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<Flux2VaeGraph> {
    let in_c = cfg.in_channels;
    let f = DType::F32;
    let mut hir =
        HirModule::new("flux2_vae_encoder").with_fusion_policy(rlx_ir::hir::FusionPolicy::Direct);
    let mut params = Flux2GraphParams::new();
    let rgb = hir.input("rgb", Shape::new(&[batch, in_c, h, w], f));
    let mut b = VaeHirBuilder::from_emit_parts(&mut hir, &mut params, cfg, weights, batch, h, w);
    let out = b.emit_encoder(rgb)?;
    hir.outputs = vec![out];
    Ok(Flux2VaeGraph { hir, params })
}

pub fn compile_flux2_vae_hir(
    cfg: &Flux2VaeConfig,
    weights: &Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
    device: Device,
    aot: Option<&rlx_runtime::AotCache>,
) -> Result<(rlx_runtime::CompiledGraph, Flux2GraphParams)> {
    crate::device::assert_flux2_device_available(device)?;
    let g = build_flux2_vae_hir(cfg, weights, batch, h, w)?;
    let key = flux2_vae_decoder_aot_key(device, batch, h, w);
    let mut compiled = compile_hir_cached(
        device,
        aot,
        &key,
        g.hir,
        &crate::compile_util::flux2_compile_profile(),
    )?;
    for (name, data) in &g.params {
        compiled.set_param(name, data);
    }
    Ok((compiled, g.params))
}

pub fn compile_flux2_vae_encoder_hir(
    cfg: &Flux2VaeConfig,
    weights: &Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
    device: Device,
    aot: Option<&rlx_runtime::AotCache>,
) -> Result<(rlx_runtime::CompiledGraph, Flux2GraphParams)> {
    crate::device::assert_flux2_device_available(device)?;
    let g = build_flux2_vae_encoder_hir(cfg, weights, batch, h, w)?;
    let key = flux2_vae_encoder_aot_key(device, batch, h, w);
    let mut compiled = compile_hir_cached(
        device,
        aot,
        &key,
        g.hir,
        &crate::compile_util::flux2_compile_profile(),
    )?;
    for (name, data) in &g.params {
        compiled.set_param(name, data);
    }
    Ok((compiled, g.params))
}

pub(crate) struct VaeHirBuilder<'a> {
    hir: &'a mut HirModule,
    params: &'a mut Flux2GraphParams,
    cfg: &'a Flux2VaeConfig,
    weights: &'a Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
    f: DType,
    eps: f32,
    groups: usize,
}

impl<'a> VaeHirBuilder<'a> {
    pub(crate) fn from_emit_parts(
        hir: &'a mut HirModule,
        params: &'a mut Flux2GraphParams,
        cfg: &'a Flux2VaeConfig,
        weights: &'a Flux2VaeWeights,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Self {
        Self {
            hir,
            params,
            cfg,
            weights,
            batch,
            h,
            w,
            f: DType::F32,
            eps: 1e-6,
            groups: cfg.norm_num_groups,
        }
    }

    pub(crate) fn emit_decoder(
        &mut self,
        mut x: HirNodeId,
    ) -> Result<(HirNodeId, usize, usize, usize)> {
        let lc = self.cfg.latent_channels;
        let mut channels = lc;
        let mut h = self.h;
        let mut w = self.w;

        if let Some(pqc) = &self.weights.post_quant_conv {
            x = self.conv2d_bias(x, pqc, "post_quant_conv", channels, h, w)?;
            channels = pqc.out_c;
        }
        x = self.conv2d_bias(x, &self.weights.conv_in, "conv_in", channels, h, w)?;
        channels = self.weights.conv_in.out_c;

        for (i, resnet) in self.weights.mid_resnets.iter().enumerate() {
            x = self.resnet_block(x, resnet, &format!("mid.{i}"), channels, h, w)?;
            channels = resnet.conv2.out_c;
        }
        if let Some(attn) = &self.weights.mid_attn {
            x = self.spatial_attention(x, attn, "mid.attn", channels, h, w)?;
        }

        for (i, block) in self.weights.up_blocks.iter().enumerate() {
            let (cur, c, hh, ww) = self.up_block(x, block, &format!("up.{i}"), channels, h, w)?;
            x = cur;
            channels = c;
            h = hh;
            w = ww;
        }

        let shape = self.nchw(channels, h, w);
        x = self.group_norm(
            x,
            &self.weights.conv_norm_out,
            "conv_norm_out",
            shape.clone(),
        )?;
        x = self.g().activation(Activation::Silu, x, shape.clone());
        x = self.conv2d_bias(x, &self.weights.conv_out, "conv_out", channels, h, w)?;
        let out_c = self.weights.conv_out.out_c;
        Ok((x, out_c, h, w))
    }

    fn nchw(&self, c: usize, h: usize, w: usize) -> Shape {
        Shape::new(&[self.batch, c, h, w], self.f)
    }

    fn register_param(&mut self, name: &str, data: Vec<f32>, shape: Shape) -> HirNodeId {
        let id = self.hir.param(name, shape);
        self.params.insert(name.to_string(), data);
        id
    }

    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    pub(crate) fn emit_encoder(&mut self, mut x: HirNodeId) -> Result<HirNodeId> {
        let in_c = self.cfg.in_channels;
        let mut channels = in_c;
        let mut h = self.h;
        let mut w = self.w;

        x = self.conv2d_bias(
            x,
            &self.weights.encoder_conv_in,
            "encoder.conv_in",
            channels,
            h,
            w,
        )?;
        channels = self.weights.encoder_conv_in.out_c;

        for (i, block) in self.weights.encoder_down_blocks.iter().enumerate() {
            let (cur, c, hh, ww) =
                self.down_block(x, block, &format!("encoder.down.{i}"), channels, h, w)?;
            x = cur;
            channels = c;
            h = hh;
            w = ww;
        }

        for (i, resnet) in self.weights.encoder_mid_resnets.iter().enumerate() {
            x = self.resnet_block(x, resnet, &format!("encoder.mid.{i}"), channels, h, w)?;
            channels = resnet.conv2.out_c;
        }
        if let Some(attn) = &self.weights.encoder_mid_attn {
            x = self.spatial_attention(x, attn, "encoder.mid.attn", channels, h, w)?;
        }

        let shape = self.nchw(channels, h, w);
        x = self.group_norm(
            x,
            &self.weights.encoder_conv_norm_out,
            "encoder.conv_norm_out",
            shape.clone(),
        )?;
        x = self.g().activation(Activation::Silu, x, shape.clone());
        x = self.conv2d_bias(
            x,
            &self.weights.encoder_conv_out,
            "encoder.conv_out",
            channels,
            h,
            w,
        )?;
        channels = self.weights.encoder_conv_out.out_c;

        x = self.conv2d_bias(x, &self.weights.quant_conv, "quant_conv", channels, h, w)?;
        let mean_c = self.weights.quant_conv.out_c / 2;
        Ok(self.g().narrow_(x, 1, 0, mean_c))
    }

    fn group_norm(
        &mut self,
        x: HirNodeId,
        gn: &GroupNormWeight,
        name: &str,
        shape: Shape,
    ) -> Result<HirNodeId> {
        let c = shape.dim(1).unwrap_static();
        let g = self.register_param(
            &format!("{name}.weight"),
            gn.gamma.clone(),
            Shape::new(&[c], self.f),
        );
        let b = self.register_param(
            &format!("{name}.bias"),
            gn.beta.clone(),
            Shape::new(&[c], self.f),
        );
        let groups = self.groups;
        let eps = self.eps;
        Ok(self.g().group_norm(x, g, b, groups, eps))
    }

    fn conv2d_bias(
        &mut self,
        x: HirNodeId,
        conv: &Conv2dWeight,
        name: &str,
        _in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<HirNodeId> {
        let is_1x1 = conv.weight.len() == conv.out_c * conv.in_c;
        let (kh, kw) = if is_1x1 { (1, 1) } else { (3, 3) };
        let (pad, stride) = if is_1x1 {
            ([0, 0], [1, 1])
        } else {
            ([1, 1], [1, 1])
        };
        let w_shape = if is_1x1 {
            Shape::new(&[conv.out_c, conv.in_c, 1, 1], self.f)
        } else {
            Shape::new(&[conv.out_c, conv.in_c, 3, 3], self.f)
        };
        let weight = self.register_param(&format!("{name}.weight"), conv.weight.clone(), w_shape);
        let out_shape = self.nchw(conv.out_c, h, w);
        let y = self
            .g()
            .conv2d(x, weight, [kh, kw], stride, pad, 1, out_shape.clone());
        let bias = self.register_param(
            &format!("{name}.bias"),
            conv.bias.clone(),
            Shape::new(&[conv.out_c], self.f),
        );
        let bias4 = self.g().reshape_(bias, vec![1, conv.out_c as i64, 1, 1]);
        let batch = self.batch;
        let expanded = self.g().add_node(
            Op::Expand {
                target_shape: vec![batch as i64, conv.out_c as i64, h as i64, w as i64],
            },
            vec![bias4],
            out_shape.clone(),
        );
        Ok(self.g().add(y, expanded))
    }

    fn resnet_block(
        &mut self,
        x: HirNodeId,
        b: &ResnetBlockWeights,
        name: &str,
        in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<HirNodeId> {
        let shape = self.nchw(in_c, h, w);
        let mut residual = x;
        let mut h1 = self.group_norm(x, &b.norm1, &format!("{name}.norm1"), shape.clone())?;
        h1 = self.g().activation(Activation::Silu, h1, shape.clone());
        h1 = self.conv2d_bias(h1, &b.conv1, &format!("{name}.conv1"), in_c, h, w)?;
        let c1 = b.conv1.out_c;
        let s1 = self.nchw(c1, h, w);
        h1 = self.group_norm(h1, &b.norm2, &format!("{name}.norm2"), s1.clone())?;
        h1 = self.g().activation(Activation::Silu, h1, s1.clone());
        h1 = self.conv2d_bias(h1, &b.conv2, &format!("{name}.conv2"), c1, h, w)?;
        let out_c = b.conv2.out_c;
        if let Some(sc) = &b.shortcut {
            residual = self.conv2d_bias(residual, sc, &format!("{name}.shortcut"), in_c, h, w)?;
        }
        let _out_shape = self.nchw(out_c, h, w);
        Ok(self.g().add(h1, residual))
    }

    fn spatial_attention(
        &mut self,
        x: HirNodeId,
        attn: &AttnBlockWeights,
        name: &str,
        channels: usize,
        h: usize,
        w: usize,
    ) -> Result<HirNodeId> {
        let shape = self.nchw(channels, h, w);
        let normed = self.group_norm(x, &attn.norm, &format!("{name}.norm"), shape.clone())?;
        let q = self.conv2d_bias(normed, &attn.to_q, &format!("{name}.to_q"), channels, h, w)?;
        let k = self.conv2d_bias(normed, &attn.to_k, &format!("{name}.to_k"), channels, h, w)?;
        let v = self.conv2d_bias(normed, &attn.to_v, &format!("{name}.to_v"), channels, h, w)?;
        let seq = h * w;
        let batch = self.batch;
        let bsh = Shape::new(&[batch, seq, channels], self.f);
        let q2 = self
            .g()
            .reshape_(q, vec![batch as i64, seq as i64, channels as i64]);
        let k2 = self
            .g()
            .reshape_(k, vec![batch as i64, seq as i64, channels as i64]);
        let v2 = self
            .g()
            .reshape_(v, vec![batch as i64, seq as i64, channels as i64]);
        let fixed = self
            .g()
            .attention_kind(q2, k2, v2, 1, channels, MaskKind::None, bsh.clone());
        let fixed4 = self.g().reshape_(
            fixed,
            vec![batch as i64, channels as i64, h as i64, w as i64],
        );
        let proj = self.conv2d_bias(
            fixed4,
            &attn.to_out,
            &format!("{name}.to_out"),
            channels,
            h,
            w,
        )?;
        Ok(self.g().add(x, proj))
    }

    fn up_block(
        &mut self,
        x: HirNodeId,
        block: &UpDecoderBlockWeights,
        name: &str,
        mut in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<(HirNodeId, usize, usize, usize)> {
        let mut cur = x;
        for (j, resnet) in block.resnets.iter().enumerate() {
            let out_c = resnet.conv2.out_c;
            cur = self.resnet_block(cur, resnet, &format!("{name}.resnet.{j}"), in_c, h, w)?;
            in_c = out_c;
        }
        let mut out_h = h;
        let mut out_w = w;
        if let Some(up) = &block.upsample {
            let uped = self.g().resize_nearest_2x(cur);
            out_h = h * 2;
            out_w = w * 2;
            cur = self.conv2d_bias(uped, up, &format!("{name}.upsample"), in_c, out_h, out_w)?;
            in_c = up.out_c;
        }
        Ok((cur, in_c, out_h, out_w))
    }

    fn down_block(
        &mut self,
        x: HirNodeId,
        block: &DownEncoderBlockWeights,
        name: &str,
        mut in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<(HirNodeId, usize, usize, usize)> {
        let mut cur = x;
        for (j, resnet) in block.resnets.iter().enumerate() {
            let out_c = resnet.conv2.out_c;
            cur = self.resnet_block(cur, resnet, &format!("{name}.resnet.{j}"), in_c, h, w)?;
            in_c = out_c;
        }
        let mut out_h = h;
        let mut out_w = w;
        if let Some(down) = &block.downsample {
            out_h = (h + 1 - 3) / 2 + 1;
            out_w = (w + 1 - 3) / 2 + 1;
            cur = self.conv2d_downsample(
                cur,
                down,
                &format!("{name}.downsample"),
                in_c,
                h,
                w,
                out_h,
                out_w,
            )?;
            in_c = down.out_c;
        }
        Ok((cur, in_c, out_h, out_w))
    }

    fn conv2d_downsample(
        &mut self,
        x: HirNodeId,
        conv: &Conv2dWeight,
        name: &str,
        _in_c: usize,
        _h: usize,
        _w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Result<HirNodeId> {
        let w_shape = Shape::new(&[conv.out_c, conv.in_c, 3, 3], self.f);
        let weight = self.register_param(&format!("{name}.weight"), conv.weight.clone(), w_shape);
        let out_shape = self.nchw(conv.out_c, out_h, out_w);
        let y = self
            .g()
            .conv2d(x, weight, [3, 3], [2, 2], [1, 1], 1, out_shape.clone());
        let bias = self.register_param(
            &format!("{name}.bias"),
            conv.bias.clone(),
            Shape::new(&[conv.out_c], self.f),
        );
        let bias4 = self.g().reshape_(bias, vec![1, conv.out_c as i64, 1, 1]);
        let batch = self.batch;
        let expanded = self.g().add_node(
            Op::Expand {
                target_shape: vec![batch as i64, conv.out_c as i64, out_h as i64, out_w as i64],
            },
            vec![bias4],
            out_shape.clone(),
        );
        Ok(self.g().add(y, expanded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vae::{Flux2VaeConfig, flux2_vae_decode, synthetic_vae_weights};
    use rlx_runtime::Device;

    #[test]
    fn vae_hir_lowers() {
        let cfg = Flux2VaeConfig::tiny();
        let w = synthetic_vae_weights(&cfg);
        let g = build_flux2_vae_hir(&cfg, &w, 1, 4, 4).unwrap();
        g.hir.lower_to_mir().expect("lower");
    }

    #[test]
    fn compiled_vae_encoder_matches_native() {
        let cfg = Flux2VaeConfig::tiny();
        let w = synthetic_vae_weights(&cfg);
        let batch = 1usize;
        let h = 32usize;
        let w_px = 32usize;
        let rgb: Vec<f32> = (0..batch * 3 * h * w_px)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let native =
            super::super::encoder::flux2_vae_encode(&w, &cfg, &rgb, batch, h, w_px).unwrap();

        let (mut compiled, _) =
            compile_flux2_vae_encoder_hir(&cfg, &w, batch, h, w_px, Device::Cpu, None).unwrap();
        let mut out = compiled.run(&[("rgb", rgb.as_slice())]).remove(0);
        if cfg.scaling_factor != 1.0 || cfg.shift_factor != 0.0 {
            for v in &mut out {
                *v = (*v - cfg.shift_factor) * cfg.scaling_factor;
            }
        }

        assert_eq!(out.len(), native.len());
        let max = out
            .iter()
            .zip(&native)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 5e-2, "HIR encoder vs native max_abs_diff={max}");
    }

    #[test]
    fn compiled_vae_matches_native() {
        let cfg = Flux2VaeConfig::tiny();
        let w = synthetic_vae_weights(&cfg);
        let batch = 1usize;
        let h = 4usize;
        let w_px = 4usize;
        let latents = vec![0.1f32; batch * cfg.latent_channels * h * w_px];

        let native = flux2_vae_decode(&w, &cfg, &latents, batch, h, w_px).unwrap();

        let (mut compiled, _) =
            compile_flux2_vae_hir(&cfg, &w, batch, h, w_px, Device::Cpu, None).unwrap();
        let out = compiled.run(&[("latents", latents.as_slice())]).remove(0);

        assert_eq!(out.len(), native.len());
        let up = 2usize.pow(cfg.block_out_channels.len().saturating_sub(1) as u32);
        assert_eq!(out.len(), batch * cfg.out_channels * h * up * w_px * up);
        let max = out
            .iter()
            .zip(&native)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 2e-2, "HIR vs native VAE max_abs_diff={max}");
    }
}
