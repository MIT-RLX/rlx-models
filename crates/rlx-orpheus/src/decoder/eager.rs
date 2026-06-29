// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! SNAC 24 kHz decoder — eager CPU inference from safetensors weights.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, ArrayView2, s};
use safetensors::SafeTensors;

use super::config::SnacConfig;
use super::ops::{
    conv_transpose1d, conv1d, embed_codes, noise_block, noise_block_zero, repeat_interleave_time,
    residual_add, snake1d,
};

/// Output sample rate (Hz).
pub const SAMPLE_RATE: u32 = 24_000;

/// PCM samples per Orpheus streaming slice (`2048` = one quarter of a 8192-sample SNAC frame).
pub const SAMPLES_PER_FRAME: usize = 2048;

fn load_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("Missing weight: {name}"))?;
    let raw = view.data();
    use safetensors::tensor::Dtype;
    Ok(match view.dtype() {
        Dtype::F32 => {
            assert!(
                raw.len() % 4 == 0,
                "F32 tensor byte length not divisible by 4"
            );
            let n = raw.len() / 4;
            let mut out = Vec::with_capacity(n);
            #[cfg(target_endian = "little")]
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), out.as_mut_ptr() as *mut u8, raw.len());
                out.set_len(n);
            }
            #[cfg(not(target_endian = "little"))]
            {
                out.extend(
                    raw.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );
            }
            out
        }
        dt => bail!("Tensor {name}: unsupported dtype {dt:?} (expected F32)"),
    })
}

fn as1d(data: Vec<f32>, n: usize) -> Array1<f32> {
    Array1::from_shape_vec(n, data).expect("1-D shape mismatch")
}

fn as2d(data: Vec<f32>, rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_vec((rows, cols), data).expect("2-D shape mismatch")
}

fn as3d(data: Vec<f32>, d0: usize, d1: usize, d2: usize) -> Array3<f32> {
    Array3::from_shape_vec((d0, d1, d2), data).expect("3-D shape mismatch")
}

pub(crate) struct ResidualUnitWeights {
    pub(crate) snake1_alpha: Array1<f32>,
    pub(crate) conv1_w: Array3<f32>,
    pub(crate) conv1_b: Array1<f32>,
    pub(crate) conv1_pad: usize,
    pub(crate) conv1_dilation: usize,
    pub(crate) snake2_alpha: Array1<f32>,
    pub(crate) conv2_w: Array3<f32>,
    pub(crate) conv2_b: Array1<f32>,
    pub(crate) groups: usize,
}

pub(crate) struct DecoderBlockWeights {
    pub(crate) snake_alpha: Array1<f32>,
    pub(crate) upsample_w: Array3<f32>,
    pub(crate) upsample_b: Array1<f32>,
    pub(crate) noise_w: Array3<f32>,
    pub(crate) stride: usize,
    pub(crate) residual_units: [ResidualUnitWeights; 3],
}

struct VectorQuantizeWeights {
    codebook: Array2<f32>,
    out_proj_w: Array3<f32>,
    out_proj_b: Array1<f32>,
    stride: usize,
}

pub(crate) struct SnacDecoderInner {
    pub(crate) config: SnacConfig,
    quantizers: Vec<VectorQuantizeWeights>,
    pub(crate) init_dw_w: Array3<f32>,
    pub(crate) init_dw_b: Array1<f32>,
    pub(crate) init_pw_w: Array3<f32>,
    pub(crate) init_pw_b: Array1<f32>,
    pub(crate) blocks: Vec<DecoderBlockWeights>,
    pub(crate) final_snake_alpha: Array1<f32>,
    pub(crate) final_conv_w: Array3<f32>,
    pub(crate) final_conv_b: Array1<f32>,
}

fn load_alpha(st: &SafeTensors<'_>, key: &str) -> Result<Array1<f32>> {
    let data = load_f32(st, key)?;
    let c = data.len();
    Ok(as1d(data, c))
}

fn load_residual_unit(
    st: &SafeTensors<'_>,
    prefix: &str,
    dim: usize,
    groups: usize,
    dilation: usize,
) -> Result<ResidualUnitWeights> {
    let pad = ((7 - 1) * dilation) / 2;
    Ok(ResidualUnitWeights {
        snake1_alpha: load_alpha(st, &format!("{prefix}.0.alpha"))?,
        conv1_w: as3d(
            load_f32(st, &format!("{prefix}.1.weight"))?,
            dim,
            dim / groups,
            7,
        ),
        conv1_b: as1d(load_f32(st, &format!("{prefix}.1.bias"))?, dim),
        conv1_pad: pad,
        conv1_dilation: dilation,
        snake2_alpha: load_alpha(st, &format!("{prefix}.2.alpha"))?,
        conv2_w: as3d(load_f32(st, &format!("{prefix}.3.weight"))?, dim, dim, 1),
        conv2_b: as1d(load_f32(st, &format!("{prefix}.3.bias"))?, dim),
        groups,
    })
}

fn residual_unit_forward(x: ArrayView2<f32>, w: &ResidualUnitWeights) -> Array2<f32> {
    let mut h = snake1d(x, w.snake1_alpha.view());
    h = conv1d(
        h.view(),
        w.conv1_w.view(),
        Some(w.conv1_b.view()),
        w.conv1_pad,
        w.groups,
        w.conv1_dilation,
    );
    h = snake1d(h.view(), w.snake2_alpha.view());
    h = conv1d(h.view(), w.conv2_w.view(), Some(w.conv2_b.view()), 0, 1, 1);
    residual_add(x, h.view())
}

fn load_decoder_block(
    st: &SafeTensors<'_>,
    prefix: &str,
    input_dim: usize,
    output_dim: usize,
    stride: usize,
    depthwise: bool,
) -> Result<DecoderBlockWeights> {
    let k = 2 * stride;
    let ru_groups = if depthwise { output_dim } else { 1 };
    let block = format!("{prefix}.block");
    Ok(DecoderBlockWeights {
        snake_alpha: load_alpha(st, &format!("{block}.0.alpha"))?,
        upsample_w: as3d(
            load_f32(st, &format!("{block}.1.weight"))?,
            input_dim,
            output_dim,
            k,
        ),
        upsample_b: as1d(load_f32(st, &format!("{block}.1.bias"))?, output_dim),
        noise_w: as3d(
            load_f32(st, &format!("{block}.2.linear.weight"))?,
            output_dim,
            output_dim,
            1,
        ),
        stride,
        residual_units: [
            load_residual_unit(st, &format!("{block}.3.block"), output_dim, ru_groups, 1)?,
            load_residual_unit(st, &format!("{block}.4.block"), output_dim, ru_groups, 3)?,
            load_residual_unit(st, &format!("{block}.5.block"), output_dim, ru_groups, 9)?,
        ],
    })
}

fn decoder_block_forward(
    x: ArrayView2<f32>,
    w: &DecoderBlockWeights,
    noise: &mut NoiseState,
) -> Array2<f32> {
    let mut h = snake1d(x, w.snake_alpha.view());
    let padding = w.stride.div_ceil(2);
    let output_padding = w.stride % 2;
    h = conv_transpose1d(
        h.view(),
        w.upsample_w.view(),
        Some(w.upsample_b.view()),
        w.stride,
        padding,
        output_padding,
        1,
    );
    if noise.enabled() {
        let t = h.shape()[1];
        let n = noise.next_plane(t);
        h = noise_block(h.view(), w.noise_w.view(), n.view());
    } else {
        h = noise_block_zero(h.view());
    }
    for ru in &w.residual_units {
        h = residual_unit_forward(h.view(), ru);
    }
    h
}

fn load_inner(st: &SafeTensors<'_>, config: SnacConfig) -> Result<SnacDecoderInner> {
    let latent = config.latent_dim();
    let decoder_dim = config.decoder_dim;

    let quantizers = config
        .vq_strides
        .iter()
        .enumerate()
        .map(|(i, &stride)| {
            let prefix = format!("quantizer.quantizers.{i}");
            Ok(VectorQuantizeWeights {
                codebook: as2d(
                    load_f32(st, &format!("{prefix}.codebook.weight"))?,
                    config.codebook_size,
                    config.codebook_dim,
                ),
                out_proj_w: as3d(
                    load_f32(st, &format!("{prefix}.out_proj.weight"))?,
                    latent,
                    config.codebook_dim,
                    1,
                ),
                out_proj_b: as1d(load_f32(st, &format!("{prefix}.out_proj.bias"))?, latent),
                stride,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let init_dw_w = as3d(load_f32(st, "decoder.model.0.weight")?, latent, 1, 7);
    let init_dw_b = as1d(load_f32(st, "decoder.model.0.bias")?, latent);
    let init_pw_w = as3d(
        load_f32(st, "decoder.model.1.weight")?,
        decoder_dim,
        latent,
        1,
    );
    let init_pw_b = as1d(load_f32(st, "decoder.model.1.bias")?, decoder_dim);

    let blocks = config
        .decoder_rates
        .iter()
        .enumerate()
        .map(|(i, &stride)| {
            let input_dim = decoder_dim / 2_usize.pow(i as u32);
            let output_dim = decoder_dim / 2_usize.pow(i as u32 + 1);
            load_decoder_block(
                st,
                &format!("decoder.model.{}", i + 2),
                input_dim,
                output_dim,
                stride,
                config.depthwise,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let final_dim = decoder_dim / 2_usize.pow(config.decoder_rates.len() as u32);
    let final_snake_alpha = load_alpha(st, "decoder.model.6.alpha")?;
    let final_conv_w = as3d(load_f32(st, "decoder.model.7.weight")?, 1, final_dim, 7);
    let final_conv_b = as1d(load_f32(st, "decoder.model.7.bias")?, 1);

    Ok(SnacDecoderInner {
        config,
        quantizers,
        init_dw_w,
        init_dw_b,
        init_pw_w,
        init_pw_b,
        blocks,
        final_snake_alpha,
        final_conv_w,
        final_conv_b,
    })
}

pub(crate) fn from_codes_inner(
    inner: &SnacDecoderInner,
    codes_0: &[i32],
    codes_1: &[i32],
    codes_2: &[i32],
) -> Result<Array2<f32>> {
    let base_len = codes_0.len();
    let finest = inner.config.vq_strides[0];
    let all_codes = [codes_0, codes_1, codes_2];
    for (i, codes) in all_codes.iter().enumerate() {
        let stride = inner.config.vq_strides[i];
        if codes.len() * stride != base_len * finest {
            bail!(
                "codes_{i} length {} inconsistent with codes_0={base_len} (need len*stride={})",
                codes.len(),
                base_len * finest
            );
        }
    }
    let mut z_q: Option<Array2<f32>> = None;
    for (q, codes) in inner.quantizers.iter().zip(all_codes) {
        validate_codes(codes, inner.config.codebook_size)?;
        let z_p = embed_codes(codes, q.codebook.view());
        let mut z_q_i = conv1d(
            z_p.view(),
            q.out_proj_w.view(),
            Some(q.out_proj_b.view()),
            0,
            1,
            1,
        );
        if q.stride > 1 {
            z_q_i = repeat_interleave_time(z_q_i.view(), q.stride);
        }
        z_q = Some(match z_q {
            None => z_q_i,
            Some(acc) => &acc + &z_q_i,
        });
    }
    Ok(z_q.expect("quantizers non-empty"))
}

fn validate_codes(codes: &[i32], codebook_size: usize) -> Result<()> {
    for (i, &c) in codes.iter().enumerate() {
        if !(0..codebook_size as i32).contains(&c) {
            bail!("code index {c} at position {i} out of range 0..{codebook_size}");
        }
    }
    Ok(())
}

/// Per-decode noise planes for SNAC NoiseBlocks (PyTorch `torch.randn(1,1,T)` per block).
pub(crate) struct NoiseState {
    block: usize,
    fixed: Option<Vec<Vec<f32>>>,
    rng: Option<rand::rngs::StdRng>,
    enabled: bool,
}

impl NoiseState {
    fn disabled() -> Self {
        Self {
            block: 0,
            fixed: None,
            rng: None,
            enabled: false,
        }
    }

    fn from_reference_dir(dir: &Path) -> Result<Option<Self>> {
        let mut planes = Vec::new();
        for i in 0..4 {
            let path = dir.join(format!("ref_noise_{i}.npy"));
            if !path.is_file() {
                return Ok(None);
            }
            planes.push(load_npy_f32(&path)?);
        }
        Ok(Some(Self {
            block: 0,
            fixed: Some(planes),
            rng: None,
            enabled: true,
        }))
    }

    pub(crate) fn seeded(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            block: 0,
            fixed: None,
            rng: Some(rand::rngs::StdRng::seed_from_u64(seed)),
            enabled: true,
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn next_plane(&mut self, t: usize) -> Array1<f32> {
        if let Some(fixed) = &self.fixed {
            if self.block < fixed.len() {
                let lane = &fixed[self.block];
                if lane.len() >= t {
                    self.block += 1;
                    return Array1::from(lane[..t].to_vec());
                }
            }
            // Reference fixtures cover four SNAC blocks only — longer utterances need RNG noise.
            self.fixed = None;
            if self.rng.is_none() {
                use rand::SeedableRng;
                self.rng = Some(rand::rngs::StdRng::seed_from_u64(42));
            }
        }
        let rng = self
            .rng
            .as_mut()
            .expect("seeded noise requires ORPHEUS_SNAC_SEED");
        use rand_distr::{Distribution, StandardNormal};
        self.block += 1;
        let dist = StandardNormal;
        Array1::from(
            (0..t)
                .map(|_| {
                    let v: f64 = dist.sample(rng);
                    v as f32
                })
                .collect::<Vec<_>>(),
        )
    }
}

pub(crate) fn load_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(
        bytes.starts_with(b"\x93NUMPY"),
        "not npy: {}",
        path.display()
    );
    let header_len = u16::from_le_bytes(bytes[8..10].try_into()?) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len])?;
    let shape_start = header.find('(').context("npy shape")?;
    let shape_end = header[shape_start..].find(')').context("npy shape")? + shape_start;
    let shape_str = &header[shape_start + 1..shape_end];
    let elem_count: usize = if shape_str.trim().is_empty() {
        0
    } else {
        shape_str
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .product()
    };
    let data_off = 10 + header_len;
    let mut out = Vec::with_capacity(elem_count);
    for chunk in bytes[data_off..].chunks_exact(4).take(elem_count) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Audio samples after the full decoder stack for `t_latent` latent steps.
#[cfg(feature = "coreml")]
pub(crate) fn audio_len_for_latent(decoder_rates: &[usize], t_latent: usize) -> usize {
    let mut t = t_latent;
    for &stride in decoder_rates {
        let k = 2 * stride;
        let padding = stride.div_ceil(2);
        let output_padding = stride % 2;
        t = (t - 1) * stride + k - 2 * padding + output_padding;
    }
    t
}

/// Time width after each decoder block upsample (noise plane lengths).
#[cfg(feature = "coreml")]
pub(crate) fn decoder_block_times(decoder_rates: &[usize], t_latent: usize) -> [usize; 4] {
    let mut t = t_latent;
    let mut out = [0usize; 4];
    for (slot, &stride) in decoder_rates.iter().enumerate() {
        let k = 2 * stride;
        let padding = stride.div_ceil(2);
        let output_padding = stride % 2;
        t = (t - 1) * stride + k - 2 * padding + output_padding;
        out[slot] = t;
    }
    out
}

fn decode_latent(
    inner: &SnacDecoderInner,
    z_q: ArrayView2<f32>,
    noise: &mut NoiseState,
) -> Array2<f32> {
    decode_latent_stages(inner, z_q, noise)
        .into_iter()
        .last()
        .map(|(_, x)| x)
        .expect("decode_latent_stages non-empty")
}

/// Decoder forward with named stage outputs (for Python layer parity tests).
pub(crate) fn decode_latent_stages(
    inner: &SnacDecoderInner,
    z_q: ArrayView2<f32>,
    noise: &mut NoiseState,
) -> Vec<(&'static str, Array2<f32>)> {
    let mut stages = Vec::new();
    let mut x = if inner.config.depthwise {
        conv1d(
            z_q,
            inner.init_dw_w.view(),
            Some(inner.init_dw_b.view()),
            3,
            inner.config.latent_dim(),
            1,
        )
    } else {
        conv1d(
            z_q,
            inner.init_dw_w.view(),
            Some(inner.init_dw_b.view()),
            3,
            1,
            1,
        )
    };
    stages.push(("init_dw", x.clone()));
    x = conv1d(
        x.view(),
        inner.init_pw_w.view(),
        Some(inner.init_pw_b.view()),
        0,
        1,
        1,
    );
    stages.push(("init_pw", x.clone()));
    for (i, block) in inner.blocks.iter().enumerate() {
        x = decoder_block_forward(x.view(), block, noise);
        stages.push((DECODER_BLOCK_STAGE_NAMES[i], x.clone()));
    }
    x = snake1d(x.view(), inner.final_snake_alpha.view());
    stages.push(("final_snake", x.clone()));
    x = conv1d(
        x.view(),
        inner.final_conv_w.view(),
        Some(inner.final_conv_b.view()),
        3,
        1,
        1,
    );
    stages.push(("final_conv", x.clone()));
    x = x.mapv(|v| v.tanh());
    stages.push(("output", x));
    stages
}

const DECODER_BLOCK_STAGE_NAMES: [&str; 4] = ["block_0", "block_1", "block_2", "block_3"];

fn config_path_for_weights(path: &Path) -> PathBuf {
    path.parent()
        .map(|dir| dir.join("snac_24khz_decoder_config.json"))
        .unwrap_or_else(|| PathBuf::from("snac_24khz_decoder_config.json"))
}

/// SNAC neural codec decoder for Orpheus speech tokens.
pub struct SnacDecoder {
    inner: SnacDecoderInner,
    weights_path: PathBuf,
    noise_seed: Option<u64>,
    noise_ref_dir: Option<PathBuf>,
}

impl SnacDecoder {
    /// Load decoder weights + JSON config from `path` (`.safetensors`).
    pub fn from_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            bail!("SNAC decoder weights not found: {}", path.display());
        }

        let cfg_path = config_path_for_weights(path);
        let config = SnacConfig::from_file(&cfg_path).with_context(|| {
            format!(
                "SNAC config not found next to weights (expected {})",
                cfg_path.display()
            )
        })?;

        let bytes =
            std::fs::read(path).with_context(|| format!("read SNAC weights {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse safetensors {}", path.display()))?;
        let inner = load_inner(&st, config)
            .with_context(|| format!("load SNAC weights from {}", path.display()))?;

        let noise_seed = std::env::var("ORPHEUS_SNAC_SEED")
            .ok()
            .and_then(|s| s.parse().ok());
        let noise_ref_dir = std::env::var("ORPHEUS_SNAC_REF_DIR")
            .ok()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .filter(|p| p.is_dir());

        Ok(Self {
            inner,
            weights_path: path.to_path_buf(),
            noise_seed,
            noise_ref_dir,
        })
    }

    /// Override the PyTorch-style noise seed (`torch.manual_seed` before decode).
    pub fn with_noise_seed(mut self, seed: u64) -> Self {
        self.noise_seed = Some(seed);
        self
    }

    fn make_noise_state(&self) -> Result<NoiseState> {
        if !self.inner.config.noise {
            return Ok(NoiseState::disabled());
        }
        if let Some(dir) = &self.noise_ref_dir {
            if let Some(state) = NoiseState::from_reference_dir(dir)? {
                return Ok(state);
            }
        }
        if let Some(seed) = self.noise_seed {
            return Ok(NoiseState::seeded(seed));
        }
        // PyTorch SNAC uses per-block `torch.randn`; a fixed seed per decode call
        // matches upstream closely enough for audible output without ref fixtures.
        Ok(NoiseState::seeded(42))
    }

    /// Decode three RVQ codebook streams to mono PCM at [`SAMPLE_RATE`].
    pub fn decode_codes(
        &self,
        codes_0: &[i32],
        codes_1: &[i32],
        codes_2: &[i32],
    ) -> Result<Vec<f32>> {
        if codes_0.is_empty() {
            return Ok(Vec::new());
        }
        let z_q = from_codes_inner(&self.inner, codes_0, codes_1, codes_2)?;
        let mut noise = self.make_noise_state()?;
        let audio = decode_latent(&self.inner, z_q.view(), &mut noise);
        Ok(audio.slice(s![0, ..]).to_vec())
    }

    /// Decode interleaved Orpheus frame tokens (7 per frame) via [`crate::tokens::pack_orpheus_codes`].
    pub fn decode_orpheus_frames(&self, frame_tokens: &[i32]) -> Result<Vec<f32>> {
        let (c0, c1, c2) = crate::tokens::pack_orpheus_codes(frame_tokens)
            .ok_or_else(|| anyhow::anyhow!("invalid Orpheus frame token layout"))?;
        self.decode_codes(&c0, &c1, &c2)
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn inner(&self) -> &SnacDecoderInner {
        &self.inner
    }

    pub fn config(&self) -> &SnacConfig {
        &self.inner.config
    }

    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ref_dir() -> Option<PathBuf> {
        std::env::var("ORPHEUS_SNAC_REF_DIR")
            .ok()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
    }

    #[test]
    fn decode_matches_reference_when_weights_available() {
        let Some(weights) = super::super::decoder_weights_path_if_available() else {
            eprintln!("skip decode_matches_reference: set ORPHEUS_SNAC_PATH");
            return;
        };
        let Some(ref_dir) = ref_dir() else {
            eprintln!(
                "skip decode_matches_reference: set ORPHEUS_SNAC_REF_DIR=/tmp/rlx-weights/snac"
            );
            return;
        };

        let codes_json: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(ref_dir.join("ref_codes.json")).unwrap())
                .unwrap();
        let c0: Vec<i32> = codes_json["codes_0"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let c1: Vec<i32> = codes_json["codes_1"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let c2: Vec<i32> = codes_json["codes_2"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();

        let dec = SnacDecoder::from_file(&weights).expect("SnacDecoder::from_file");

        let z_q = from_codes_inner(&dec.inner, &c0, &c1, &c2).expect("from_codes");
        let ref_z_q = load_npy_f32(&ref_dir.join("act_z_q.npy")).expect("act_z_q.npy");
        assert_eq!(z_q.len(), ref_z_q.len());
        let zq_max = z_q
            .iter()
            .zip(ref_z_q.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        eprintln!("z_q max abs diff vs act_z_q: {zq_max:.6}");
        assert!(zq_max < 1e-3, "z_q parity failed: max diff {zq_max}");

        let audio = dec.decode_codes(&c0, &c1, &c2).expect("decode_codes");
        assert_eq!(audio.len(), 8192);
        let ref_audio = load_npy_f32(&ref_dir.join("ref_decode.npy")).expect("ref_decode.npy");
        assert_eq!(audio.len(), ref_audio.len());
        let pcm_max = audio
            .iter()
            .zip(ref_audio.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        eprintln!("pcm max abs diff vs ref_decode: {pcm_max:.6}");
        assert!(
            pcm_max < 1e-3,
            "PCM parity failed: max diff {pcm_max} (need ref_noise_*.npy from export script)"
        );
        for (i, &s) in audio.iter().enumerate() {
            assert!(s.is_finite(), "non-finite sample at {i}: {s}");
        }
    }

    #[test]
    fn snac_decoder_stages_match_python_reference() {
        let Some(weights) = super::super::decoder_weights_path_if_available() else {
            eprintln!("skip snac_decoder_stages_match_python_reference: set ORPHEUS_SNAC_PATH");
            return;
        };
        let Some(ref_dir) = ref_dir() else {
            eprintln!("skip snac_decoder_stages_match_python_reference: set ORPHEUS_SNAC_REF_DIR");
            return;
        };

        let codes_json: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(ref_dir.join("ref_codes.json")).unwrap())
                .unwrap();
        let c0: Vec<i32> = codes_json["codes_0"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let c1: Vec<i32> = codes_json["codes_1"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let c2: Vec<i32> = codes_json["codes_2"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();

        let dec = SnacDecoder::from_file(&weights).expect("SnacDecoder::from_file");
        let z_q = from_codes_inner(&dec.inner, &c0, &c1, &c2).expect("from_codes");
        let mut noise = dec.make_noise_state().expect("noise");
        let stages = decode_latent_stages(&dec.inner, z_q.view(), &mut noise);

        for (name, got) in stages {
            let path = ref_dir.join(format!("ref_stage_{name}.npy"));
            if !path.is_file() {
                eprintln!(
                    "skip stage {name}: missing {} (re-run export with --parity-layers)",
                    path.display()
                );
                continue;
            }
            let expect = load_npy_f32(&path).expect("stage npy");
            assert_eq!(
                got.len(),
                expect.len(),
                "stage {name} element count mismatch"
            );
            let max_diff = got
                .iter()
                .zip(expect.iter())
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max);
            eprintln!("stage {name} max abs diff vs python: {max_diff:.6}");
            assert!(
                max_diff < 1e-3,
                "stage {name} parity failed: max diff {max_diff}"
            );
        }
    }

    #[test]
    fn decode_orpheus_fixture_center_matches_reference() {
        let Some(weights) = super::super::decoder_weights_path_if_available() else {
            eprintln!("skip decode_orpheus_fixture_center: set ORPHEUS_SNAC_PATH");
            return;
        };
        let Some(ref_dir) = ref_dir() else {
            eprintln!("skip decode_orpheus_fixture_center: set ORPHEUS_SNAC_REF_DIR");
            return;
        };

        let fixture = include_str!("../../tests/fixtures/orpheus_hi_codes.txt");
        let mut lines = fixture
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
        let _count: usize = lines.next().unwrap().trim().parse().unwrap();
        let codes: Vec<i32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();

        let dec = SnacDecoder::from_file(&weights).expect("SnacDecoder::from_file");
        let backend = crate::decoder::SnacBackend::Eager(dec);
        let pcm = crate::decode_orpheus_codes(&backend, &codes).expect("decode_orpheus_codes");
        assert_eq!(pcm.len(), 4 * super::super::SAMPLES_PER_FRAME);

        let ref_audio = load_npy_f32(&ref_dir.join("ref_decode.npy")).expect("ref_decode.npy");
        let max_diff = pcm
            .iter()
            .zip(ref_audio.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        eprintln!("fixture full decode max abs diff vs ref_decode: {max_diff:.6}");
        assert!(
            max_diff < 1e-3,
            "fixture PCM full decode parity failed: max diff {max_diff}"
        );
    }
}
