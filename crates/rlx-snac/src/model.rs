// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SNAC decoder weights (flat row-major f32), loadable from a folded-weight_norm
// safetensors export or constructed randomly for cross-backend parity tests.

use crate::config::SnacConfig;
use anyhow::{Context, Result};
use safetensors::SafeTensors;

#[derive(Clone)]
pub struct ResidualUnitW {
    pub snake1_alpha: Vec<f32>, // [dim]
    pub conv1_w: Vec<f32>,      // [dim, dim/groups, 7]
    pub conv1_b: Vec<f32>,      // [dim]
    pub conv1_dilation: usize,
    pub snake2_alpha: Vec<f32>, // [dim]
    pub conv2_w: Vec<f32>,      // [dim, dim, 1]
    pub conv2_b: Vec<f32>,      // [dim]
    pub groups: usize,
    pub dim: usize,
}

#[derive(Clone)]
pub struct DecoderBlockW {
    pub snake_alpha: Vec<f32>, // [in_dim]
    pub in_dim: usize,
    pub out_dim: usize,
    pub stride: usize,
    pub upsample_w: Vec<f32>, // [in_dim, out_dim, 2*stride]
    pub upsample_b: Vec<f32>, // [out_dim]
    pub noise_w: Vec<f32>,    // [out_dim, out_dim, 1]
    pub residual_units: Vec<ResidualUnitW>,
}

#[derive(Clone)]
pub struct VqW {
    pub codebook: Vec<f32>,          // [codebook_size, codebook_dim]
    pub out_proj_w: Vec<f32>,        // [latent, codebook_dim, 1]
    pub out_proj_b: Vec<f32>,        // [latent]
    pub in_proj_w: Option<Vec<f32>>, // [codebook_dim, latent, 1] (encode only)
    pub in_proj_b: Option<Vec<f32>>, // [codebook_dim]
    pub stride: usize,
}

/// One encoder downsampling block: 3 residual units → snake → strided conv.
#[derive(Clone)]
pub struct EncoderBlockW {
    pub residual_units: Vec<ResidualUnitW>,
    pub snake_alpha: Vec<f32>,  // [input_dim]
    pub downsample_w: Vec<f32>, // [output_dim, input_dim, 2*stride]
    pub downsample_b: Vec<f32>, // [output_dim]
    pub input_dim: usize,
    pub output_dim: usize,
    pub stride: usize,
}

/// SNAC encoder conv stack: stem → blocks → final depthwise conv → latent.
#[derive(Clone)]
pub struct EncoderW {
    pub stem_w: Vec<f32>, // [encoder_dim, 1, 7]
    pub stem_b: Vec<f32>,
    pub blocks: Vec<EncoderBlockW>,
    pub final_w: Vec<f32>, // [latent, latent/groups, 7] (depthwise → [latent,1,7])
    pub final_b: Vec<f32>,
    pub final_groups: usize,
}

#[derive(Clone)]
pub struct SnacWeights {
    pub config: SnacConfig,
    pub quantizers: Vec<VqW>,
    pub init_dw_w: Vec<f32>, // [latent, 1, 7]
    pub init_dw_b: Vec<f32>,
    pub init_pw_w: Vec<f32>, // [decoder_dim, latent, 1]
    pub init_pw_b: Vec<f32>,
    pub blocks: Vec<DecoderBlockW>,
    pub final_snake_alpha: Vec<f32>, // [final_dim]
    pub final_conv_w: Vec<f32>,      // [1, final_dim, 7]
    pub final_conv_b: Vec<f32>,      // [1]
    pub encoder: Option<EncoderW>,   // present when loaded from a full checkpoint
}

impl SnacWeights {
    pub fn latent(&self) -> usize {
        self.config.latent_dim()
    }

    pub fn final_dim(&self) -> usize {
        self.config.decoder_dim / 2usize.pow(self.config.decoder_rates.len() as u32)
    }

    /// Load a folded-weight_norm safetensors export (keys per the SNAC ref:
    /// `decoder.model.*`, `quantizer.quantizers.*`).
    pub fn from_safetensors(bytes: &[u8], config: SnacConfig) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse SNAC safetensors")?;
        let get = |name: &str| -> Result<Vec<f32>> { tensor_f32(&st, name) };

        let ddim = config.decoder_dim;

        let mut quantizers = Vec::new();
        for (i, &stride) in config.vq_strides.iter().enumerate() {
            let p = format!("quantizer.quantizers.{i}");
            quantizers.push(VqW {
                codebook: get(&format!("{p}.codebook.weight"))?,
                out_proj_w: get(&format!("{p}.out_proj.weight"))?,
                out_proj_b: get(&format!("{p}.out_proj.bias"))?,
                in_proj_w: get(&format!("{p}.in_proj.weight")).ok(),
                in_proj_b: get(&format!("{p}.in_proj.bias")).ok(),
                stride,
            });
        }

        // Encoder (present in a full checkpoint): block.0 stem, block.1..N
        // EncoderBlocks, block.{N+1} final depthwise conv.
        let encoder = if st.tensor("encoder.block.0.weight").is_ok() {
            let edim = config.encoder_dim;
            let mut d_model = edim;
            let mut blocks = Vec::new();
            for (i, &stride) in config.encoder_rates.iter().enumerate() {
                let input_dim = d_model;
                d_model *= 2;
                let output_dim = d_model;
                let groups = if config.depthwise { input_dim } else { 1 };
                let bp = format!("encoder.block.{}.block", i + 1);
                let ru = |j: usize, dil: usize| -> Result<ResidualUnitW> {
                    let rp = format!("{bp}.{j}.block");
                    Ok(ResidualUnitW {
                        snake1_alpha: get(&format!("{rp}.0.alpha"))?,
                        conv1_w: get(&format!("{rp}.1.weight"))?,
                        conv1_b: get(&format!("{rp}.1.bias"))?,
                        conv1_dilation: dil,
                        snake2_alpha: get(&format!("{rp}.2.alpha"))?,
                        conv2_w: get(&format!("{rp}.3.weight"))?,
                        conv2_b: get(&format!("{rp}.3.bias"))?,
                        groups,
                        dim: input_dim,
                    })
                };
                blocks.push(EncoderBlockW {
                    residual_units: vec![ru(0, 1)?, ru(1, 3)?, ru(2, 9)?],
                    snake_alpha: get(&format!("{bp}.3.alpha"))?,
                    downsample_w: get(&format!("{bp}.4.weight"))?,
                    downsample_b: get(&format!("{bp}.4.bias"))?,
                    input_dim,
                    output_dim,
                    stride,
                });
            }
            let final_idx = config.encoder_rates.len() + 1;
            let final_groups = if config.depthwise { d_model } else { 1 };
            Some(EncoderW {
                stem_w: get("encoder.block.0.weight")?,
                stem_b: get("encoder.block.0.bias")?,
                final_w: get(&format!("encoder.block.{final_idx}.weight"))?,
                final_b: get(&format!("encoder.block.{final_idx}.bias"))?,
                final_groups,
                blocks,
            })
        } else {
            None
        };

        let mut blocks = Vec::new();
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let in_dim = ddim / 2usize.pow(i as u32);
            let out_dim = ddim / 2usize.pow(i as u32 + 1);
            let groups = if config.depthwise { out_dim } else { 1 };
            let prefix = format!("decoder.model.{}.block", i + 2);
            let ru = |j: usize, dil: usize| -> Result<ResidualUnitW> {
                let rp = format!("{prefix}.{j}.block");
                Ok(ResidualUnitW {
                    snake1_alpha: get(&format!("{rp}.0.alpha"))?,
                    conv1_w: get(&format!("{rp}.1.weight"))?,
                    conv1_b: get(&format!("{rp}.1.bias"))?,
                    conv1_dilation: dil,
                    snake2_alpha: get(&format!("{rp}.2.alpha"))?,
                    conv2_w: get(&format!("{rp}.3.weight"))?,
                    conv2_b: get(&format!("{rp}.3.bias"))?,
                    groups,
                    dim: out_dim,
                })
            };
            blocks.push(DecoderBlockW {
                snake_alpha: get(&format!("{prefix}.0.alpha"))?,
                in_dim,
                out_dim,
                stride,
                upsample_w: get(&format!("{prefix}.1.weight"))?,
                upsample_b: get(&format!("{prefix}.1.bias"))?,
                noise_w: get(&format!("{prefix}.2.linear.weight"))?,
                residual_units: vec![ru(3, 1)?, ru(4, 3)?, ru(5, 9)?],
            });
        }

        Ok(Self {
            init_dw_w: get("decoder.model.0.weight")?,
            init_dw_b: get("decoder.model.0.bias")?,
            init_pw_w: get("decoder.model.1.weight")?,
            init_pw_b: get("decoder.model.1.bias")?,
            final_snake_alpha: get("decoder.model.6.alpha")?,
            final_conv_w: get("decoder.model.7.weight")?,
            final_conv_b: get("decoder.model.7.bias")?,
            blocks,
            quantizers,
            encoder,
            config,
        })
    }

    /// Deterministic random weights for cross-backend parity tests.
    pub fn random(config: SnacConfig, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let latent = config.latent_dim();
        let ddim = config.decoder_dim;

        let quantizers = config
            .vq_strides
            .iter()
            .map(|&stride| VqW {
                codebook: r.vec(config.codebook_size * config.codebook_dim, 0.3),
                out_proj_w: r.vec(latent * config.codebook_dim, 0.2),
                out_proj_b: r.vec(latent, 0.05),
                in_proj_w: None,
                in_proj_b: None,
                stride,
            })
            .collect();

        let mut blocks = Vec::new();
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let in_dim = ddim / 2usize.pow(i as u32);
            let out_dim = ddim / 2usize.pow(i as u32 + 1);
            let groups = if config.depthwise { out_dim } else { 1 };
            let ru = |r: &mut Lcg, dil: usize| ResidualUnitW {
                snake1_alpha: r.alpha(out_dim),
                conv1_w: r.vec(out_dim * (out_dim / groups) * 7, 0.1),
                conv1_b: r.vec(out_dim, 0.02),
                conv1_dilation: dil,
                snake2_alpha: r.alpha(out_dim),
                conv2_w: r.vec(out_dim * out_dim, 0.1),
                conv2_b: r.vec(out_dim, 0.02),
                groups,
                dim: out_dim,
            };
            blocks.push(DecoderBlockW {
                snake_alpha: r.alpha(in_dim),
                in_dim,
                out_dim,
                stride,
                upsample_w: r.vec(in_dim * out_dim * (2 * stride), 0.1),
                upsample_b: r.vec(out_dim, 0.02),
                noise_w: r.vec(out_dim * out_dim, 0.1),
                residual_units: vec![ru(&mut r, 1), ru(&mut r, 3), ru(&mut r, 9)],
            });
        }
        let final_dim = ddim / 2usize.pow(config.decoder_rates.len() as u32);

        Self {
            init_dw_w: r.vec(latent * 7, 0.1),
            init_dw_b: r.vec(latent, 0.02),
            init_pw_w: r.vec(ddim * latent, 0.1),
            init_pw_b: r.vec(ddim, 0.02),
            final_snake_alpha: r.alpha(final_dim),
            final_conv_w: r.vec(final_dim * 7, 0.1),
            final_conv_b: r.vec(1, 0.0),
            blocks,
            quantizers,
            encoder: None,
            config,
        }
    }
}

// Drop the latent_dim shadow that `check` referenced; keep struct minimal.
// (Removed the stray field by not declaring it.)

fn tensor_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    let t = st
        .tensor(name)
        .with_context(|| format!("missing SNAC tensor {name}"))?;
    let raw = t.data();
    Ok(match t.dtype() {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        dt => anyhow::bail!("SNAC tensor {name}: unsupported dtype {dt:?} (export as F32)"),
    })
}

pub(crate) struct Lcg(pub u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
    fn alpha(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 0.5 + self.next() * 0.2).collect()
    }
}
