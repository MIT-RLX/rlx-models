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

//! Shape / texture sparse VAE decoders (`SparseUnetVaeDecoder` and its
//! `FlexiDualGridVaeDecoder` subclass, `trellis2/models/sc_vaes/*`).
//!
//! Both are octree-upsampling U-Nets over active voxels:
//!   * `from_latent` (SparseLinear) lifts the latent to `model_channels[0]`;
//!   * each stage runs `num_blocks[i]` `SparseConvNeXtBlock3d`s, then (except the
//!     last) a `SparseResBlockC2S3d` that predicts a per-voxel subdivision and
//!     splits each active voxel into its occupied octants (×2 resolution);
//!   * a final non-affine LayerNorm + `output_layer` (SparseLinear) produces the
//!     per-voxel outputs (shape: 7 dual-grid channels; texture: 6 PBR channels).
//!
//! The shape decoder (`pred_subdiv = true`) returns the subdivision logits per
//! up-stage as `subs`, which the texture decoder (`pred_subdiv = false`)
//! consumes as `guide_subs` so both meshes share topology.

use crate::config::SparseVaeConfig;
use crate::sparse::{
    SparseTensor, channel2spatial, layer_norm, repeat_interleave_channels, silu, sparse_linear,
    subdiv_to_bits, submanifold_conv3d,
};
use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;

fn get<'a>(wm: &'a WeightMap, key: &str) -> Result<&'a [f32]> {
    wm.get(key)
        .map(|(d, _)| d)
        .with_context(|| format!("missing weight {key}"))
}

/// `SparseConvNeXtBlock3d`: `x + mlp(norm(conv(x)))`. conv is submanifold 3³;
/// norm is affine LayerNorm; mlp is `Linear→SiLU→Linear` (ratio 4).
fn convnext_block(
    wm: &WeightMap,
    prefix: &str,
    x: &SparseTensor,
    ch: usize,
) -> Result<SparseTensor> {
    let cw = get(wm, &format!("{prefix}.conv.weight"))?;
    let cb = get(wm, &format!("{prefix}.conv.bias"))?;
    let h = submanifold_conv3d(x, cw, cb, ch);
    let nw = get(wm, &format!("{prefix}.norm.weight"))?;
    let nb = get(wm, &format!("{prefix}.norm.bias"))?;
    let hn = layer_norm(&h.feats, h.n(), ch, Some(nw), Some(nb));
    // mlp
    let m0w = get(wm, &format!("{prefix}.mlp.0.weight"))?;
    let m0b = get(wm, &format!("{prefix}.mlp.0.bias"))?;
    let hidden = m0w.len() / ch;
    let mut up = sparse_linear(&hn, h.n(), ch, m0w, hidden, Some(m0b));
    silu(&mut up);
    let m2w = get(wm, &format!("{prefix}.mlp.2.weight"))?;
    let m2b = get(wm, &format!("{prefix}.mlp.2.bias"))?;
    let down = sparse_linear(&up, h.n(), hidden, m2w, ch, Some(m2b));
    Ok(x.add(&x.replace(down)))
}

/// `SparseResBlockC2S3d`: predict subdivision, conv1 (ch→out·8), C2S-upsample
/// both `h` and the residual, conv2 (out→out), add. Returns `(h_up, subdiv)`.
fn res_block_c2s(
    wm: &WeightMap,
    prefix: &str,
    x: &SparseTensor,
    ch: usize,
    out_ch: usize,
) -> Result<(SparseTensor, Vec<f32>)> {
    // subdivision from the (pre-norm) input
    let sw = get(wm, &format!("{prefix}.to_subdiv.weight"))?;
    let sb = get(wm, &format!("{prefix}.to_subdiv.bias"))?;
    let subdiv = sparse_linear(&x.feats, x.n(), ch, sw, 8, Some(sb));
    let bits = subdiv_to_bits(&subdiv, x.n());

    // main branch: norm1(affine) -> silu -> conv1(ch -> out*8)
    let n1w = get(wm, &format!("{prefix}.norm1.weight"))?;
    let n1b = get(wm, &format!("{prefix}.norm1.bias"))?;
    let mut h = layer_norm(&x.feats, x.n(), ch, Some(n1w), Some(n1b));
    silu(&mut h);
    let h = x.replace(h);
    let c1w = get(wm, &format!("{prefix}.conv1.weight"))?;
    let c1b = get(wm, &format!("{prefix}.conv1.bias"))?;
    let h = submanifold_conv3d(&h, c1w, c1b, out_ch * 8);

    // C2S upsample both branches with the same subdivision
    let h_up = channel2spatial(&h, &bits); // out*8 -> out channels
    let x_up = channel2spatial(x, &bits); // ch -> ch/8 channels
    if h_up.n() == 0 {
        bail!("shape VAE C2S produced 0 voxels (all subdivision logits ≤ 0 at {prefix})");
    }

    // norm2(non-affine) -> silu -> conv2(out -> out)
    let mut h2 = layer_norm(&h_up.feats, h_up.n(), out_ch, None, None);
    silu(&mut h2);
    let h2 = h_up.replace(h2);
    let c2w = get(wm, &format!("{prefix}.conv2.weight"))?;
    let c2b = get(wm, &format!("{prefix}.conv2.bias"))?;
    let h2 = submanifold_conv3d(&h2, c2w, c2b, out_ch);

    // skip: repeat_interleave(x_up, out/(ch/8))
    let k = out_ch
        .checked_div(x_up.c)
        .filter(|k| *k > 0 && out_ch == x_up.c * k)
        .with_context(|| {
            format!(
                "{prefix}: skip expand out_ch={out_ch} not divisible by x_up.c={}",
                x_up.c
            )
        })?;
    let skip = repeat_interleave_channels(&x_up.feats, x_up.n(), x_up.c, k);
    let out = h2.add(&h2.replace(skip));
    Ok((out, subdiv))
}

/// Decoded shape output: per-voxel 7-channel features on the finest voxel grid,
/// plus the subdivision logits per up-stage (topology guide for texturing).
pub struct DecodedShape {
    pub voxels: SparseTensor,
    /// One `[N_stage_input, 8]` subdivision-logit tensor per up-stage.
    pub subs: Vec<SparseTensor>,
}

/// Topology-only cascade upsample (`SparseUnetVaeDecoder.upsample`).
///
/// Runs the first `upsample_times` decoder stages (blocks + C2S up) and returns
/// the active voxel coordinates after that many octree levels — used to seed
/// the high-resolution shape DiT without allocating the final 7-channel output.
pub fn upsample_coords(
    cfg: &SparseVaeConfig,
    wm: &WeightMap,
    latent: &SparseTensor,
    upsample_times: usize,
) -> Result<Vec<[i32; 3]>> {
    let mc = &cfg.args.model_channels;
    let nb = &cfg.args.num_blocks;
    let fl_w = get(wm, "from_latent.weight")?;
    let fl_b = get(wm, "from_latent.bias")?;
    let feats = sparse_linear(
        &latent.feats,
        latent.n(),
        cfg.args.latent_channels,
        fl_w,
        mc[0],
        Some(fl_b),
    );
    let mut h = SparseTensor::new(feats, latent.coords.clone(), mc[0]);
    let n_stages = nb.len();
    for i in 0..n_stages {
        if i == upsample_times {
            return Ok(h.coords);
        }
        for j in 0..nb[i] {
            h = convnext_block(wm, &format!("blocks.{i}.{j}"), &h, mc[i])?;
        }
        if i < n_stages - 1 {
            let prefix = format!("blocks.{i}.{}", nb[i]);
            let (hup, _sub) = res_block_c2s(wm, &prefix, &h, mc[i], mc[i + 1])?;
            h = hup;
        }
    }
    Ok(h.coords)
}

/// Run `SparseUnetVaeDecoder` / `FlexiDualGridVaeDecoder` over a shape latent.
/// `guide_subs`, if given, replaces predicted subdivisions (texture path).
pub fn decode(
    cfg: &SparseVaeConfig,
    wm: &WeightMap,
    latent: &SparseTensor,
    guide_subs: Option<&[SparseTensor]>,
) -> Result<DecodedShape> {
    let mc = &cfg.args.model_channels; // [1024,512,256,128,64]
    let nb = &cfg.args.num_blocks; // [4,16,8,4,0]
    let out_ch = cfg.decoder_out_channels();

    // from_latent: SparseLinear(latent_channels -> mc[0])
    let fl_w = get(wm, "from_latent.weight")?;
    let fl_b = get(wm, "from_latent.bias")?;
    let feats = sparse_linear(
        &latent.feats,
        latent.n(),
        cfg.args.latent_channels,
        fl_w,
        mc[0],
        Some(fl_b),
    );
    let mut h = SparseTensor::new(feats, latent.coords.clone(), mc[0]);

    let mut subs = Vec::new();
    let n_stages = nb.len();
    for i in 0..n_stages {
        for j in 0..nb[i] {
            h = convnext_block(wm, &format!("blocks.{i}.{j}"), &h, mc[i])?;
        }
        if i < n_stages - 1 {
            // the up block is the last module in the stage (index nb[i]).
            let prefix = format!("blocks.{i}.{}", nb[i]);
            if let Some(g) = guide_subs {
                // texture path: use provided subdivision, no prediction stored.
                let bits = subdiv_to_bits(&g[i].feats, g[i].n());
                h = res_block_c2s_guided(wm, &prefix, &h, mc[i], mc[i + 1], &bits)?;
            } else {
                let (hup, sub) = res_block_c2s(wm, &prefix, &h, mc[i], mc[i + 1])?;
                subs.push(h.replace(sub));
                h = hup;
            }
        }
    }

    // final non-affine LayerNorm + output_layer
    let hn = layer_norm(&h.feats, h.n(), *mc.last().unwrap(), None, None);
    let ow = get(wm, "output_layer.weight")?;
    let ob = get(wm, "output_layer.bias")?;
    let of = sparse_linear(&hn, h.n(), *mc.last().unwrap(), ow, out_ch, Some(ob));
    Ok(DecodedShape {
        voxels: SparseTensor::new(of, h.coords.clone(), out_ch),
        subs,
    })
}

/// `SparseResBlockC2S3d` with an externally provided subdivision (texture
/// decoder, `pred_subdiv = false`, so no `to_subdiv`).
fn res_block_c2s_guided(
    wm: &WeightMap,
    prefix: &str,
    x: &SparseTensor,
    ch: usize,
    out_ch: usize,
    bits: &[u8],
) -> Result<SparseTensor> {
    let n1w = get(wm, &format!("{prefix}.norm1.weight"))?;
    let n1b = get(wm, &format!("{prefix}.norm1.bias"))?;
    let mut h = layer_norm(&x.feats, x.n(), ch, Some(n1w), Some(n1b));
    silu(&mut h);
    let h = x.replace(h);
    let c1w = get(wm, &format!("{prefix}.conv1.weight"))?;
    let c1b = get(wm, &format!("{prefix}.conv1.bias"))?;
    let h = submanifold_conv3d(&h, c1w, c1b, out_ch * 8);

    let h_up = channel2spatial(&h, bits);
    let x_up = channel2spatial(x, bits);
    if h_up.n() == 0 {
        bail!("texture VAE C2S produced 0 voxels (empty guide subdivision at {prefix})");
    }

    let mut h2 = layer_norm(&h_up.feats, h_up.n(), out_ch, None, None);
    silu(&mut h2);
    let h2 = h_up.replace(h2);
    let c2w = get(wm, &format!("{prefix}.conv2.weight"))?;
    let c2b = get(wm, &format!("{prefix}.conv2.bias"))?;
    let h2 = submanifold_conv3d(&h2, c2w, c2b, out_ch);

    let k = out_ch
        .checked_div(x_up.c)
        .filter(|k| *k > 0 && out_ch == x_up.c * k)
        .with_context(|| {
            format!(
                "{prefix}: skip expand out_ch={out_ch} not divisible by x_up.c={}",
                x_up.c
            )
        })?;
    let skip = repeat_interleave_channels(&x_up.feats, x_up.n(), x_up.c, k);
    Ok(h2.add(&h2.replace(skip)))
}
