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

//! **Real-CUGAN** (`upcunet_v3`) — the anime upscaler, and the one network in
//! this crate that is shaped nothing like the rest of it.
//!
//! Two U-Nets in series, the second correcting the first:
//!
//! ```text
//!   x → reflect-pad → UNet1 → u          (u is already at the output scale)
//!                     UNet2(u) → v
//!                     out = v + crop(u)
//! ```
//!
//! # Valid padding, and why the crops are load-bearing
//!
//! Every convolution here is `padding=0`, so the feature map *shrinks* at each
//! step and the two sides of a U-Net skip no longer line up. The reference
//! reconciles them with negative `F.pad` — a centre crop — and the constants
//! (4, 16, 20, 1) are not slack: they are exactly the accumulated shrinkage of
//! the branch they rejoin. `crop` is therefore checked against the sibling it
//! is about to be added to, because an off-by-one here produces a correctly
//! shaped tensor built from misaligned pixels.
//!
//! The input pad (18 at ×2, 14 at ×3, 19 at ×4, 38 at ×2-fast) is what buys
//! back the shrinkage so the whole network is size-preserving end to end.
//!
//! # `alpha`
//!
//! The reference takes a forward-pass `alpha` that scales the deepest UNet2
//! branch, trading denoising against detail. It defaults to 1 and every
//! released model is evaluated at 1, so it is pinned here — note that this is
//! unrelated to [`crate::AlphaMode`], which is about transparency.
//!
//! # `pro`
//!
//! The "pro" checkpoints register a `pro` buffer and expect input in
//! `[0.15, 0.85]` rather than `[0, 1]`. The buffer's *value* is unused; its
//! presence is the flag.

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::PadMode;

use crate::config::{CuganVariant, ModelConfig};
use crate::nn::Builder;

/// Every activation in the network.
const SLOPE: f32 = 0.1;

/// The `alpha` the released models are evaluated at.
const ALPHA: f32 = 1.0;

/// `pro` models are trained on a compressed input range.
const PRO_SCALE: f32 = 0.7;
const PRO_SHIFT: f32 = 0.15;

/// Centre-crop `n` pixels from each spatial edge — the reference's negative
/// `F.pad`.
fn crop(b: &mut Builder, x: HirNodeId, n: usize) -> Result<HirNodeId> {
    if n == 0 {
        return Ok(x);
    }
    let [_, _, h, w] = b.dims4(x);
    ensure!(
        h > 2 * n && w > 2 * n,
        "crop of {n} leaves nothing of a {h}×{w} feature map"
    );
    let y = b.g().narrow_(x, 2, n, h - 2 * n);
    Ok(b.g().narrow_(y, 3, n, w - 2 * n))
}

/// Add two branches of a U-Net skip, refusing to broadcast.
///
/// The whole hazard in a valid-padded U-Net is a skip whose two sides have
/// drifted apart. Most mismatches would be caught by a shape check downstream,
/// but a *broadcastable* one (a `1` extent, or equal H with unequal W) would
/// not — so the sizes are compared here, where the constant that produced them
/// is still in view.
fn join(b: &mut Builder, what: &str, a: HirNodeId, c: HirNodeId) -> Result<HirNodeId> {
    let (da, dc) = (b.dims4(a), b.dims4(c));
    ensure!(
        da == dc,
        "{what}: U-Net skip branches disagree — {da:?} against {dc:?}; the crop \
         constant does not match the accumulated valid-convolution shrinkage"
    );
    Ok(b.g().add(a, c))
}

/// Squeeze-and-excitation over a global spatial average.
fn se_block(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    bias: bool,
) -> Result<HirNodeId> {
    let [_, c, _, _] = b.dims4(x);
    ensure!(c % 8 == 0, "SE block: {c} channels do not reduce by 8");
    let pooled = b.g().mean(x, vec![2, 3], true);
    let h = b.conv(
        wm,
        &format!("{prefix}.conv1"),
        pooled,
        c / 8,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        bias,
    )?;
    let h = b.relu(h);
    let h = b.conv(
        wm,
        &format!("{prefix}.conv2"),
        h,
        c,
        [1, 1],
        [1, 1],
        [0, 0],
        1,
        bias,
    )?;
    let gate = b.sigmoid(h);
    Ok(b.g().mul(x, gate))
}

/// `UNetConv`: two valid 3×3s with leaky ReLUs, optionally SE-gated.
fn unet_conv(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    mid: usize,
    out: usize,
    se: bool,
) -> Result<HirNodeId> {
    let h = valid3x3(b, wm, &format!("{prefix}.conv.0"), x, mid)?;
    let h = b.leaky_relu(h, SLOPE);
    let h = valid3x3(b, wm, &format!("{prefix}.conv.2"), h, out)?;
    let h = b.leaky_relu(h, SLOPE);
    if se {
        // `UNetConv` builds its SE with `bias=True`, overriding the block's own
        // default — the released weights carry those biases.
        se_block(b, wm, &format!("{prefix}.seblock"), h, true)
    } else {
        Ok(h)
    }
}

/// A `padding=0` 3×3 — the only kind this network has.
fn valid3x3(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    out: usize,
) -> Result<HirNodeId> {
    b.conv(wm, prefix, x, out, [3, 3], [1, 1], [0, 0], 1, true)
}

/// The shallow U-Net. `x3_deconv` selects the ×3 tail, whose only difference is
/// a 5×5 stride-3 transposed convolution where the others use 4×4 stride-2.
fn unet1(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    out: usize,
    deconv: bool,
    x3_deconv: bool,
) -> Result<HirNodeId> {
    let x1 = unet_conv(b, wm, &format!("{prefix}.conv1"), x, 32, 64, false)?;
    let x2 = b.conv(
        wm,
        &format!("{prefix}.conv1_down"),
        x1,
        64,
        [2, 2],
        [2, 2],
        [0, 0],
        1,
        true,
    )?;
    let x1 = crop(b, x1, 4)?;
    let x2 = b.leaky_relu(x2, SLOPE);
    let x2 = unet_conv(b, wm, &format!("{prefix}.conv2"), x2, 128, 64, true)?;
    let x2 = b.conv_transpose(
        wm,
        &format!("{prefix}.conv2_up"),
        x2,
        64,
        [2, 2],
        [2, 2],
        [0, 0],
    )?;
    let x2 = b.leaky_relu(x2, SLOPE);

    let x3 = join(b, "UNet1 conv3", x1, x2)?;
    let x3 = valid3x3(b, wm, &format!("{prefix}.conv3"), x3, 64)?;
    let x3 = b.leaky_relu(x3, SLOPE);

    bottom(b, wm, prefix, x3, out, deconv, x3_deconv)
}

/// The deep U-Net.
fn unet2(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    out: usize,
    deconv: bool,
) -> Result<HirNodeId> {
    let x1 = unet_conv(b, wm, &format!("{prefix}.conv1"), x, 32, 64, false)?;
    let x2 = b.conv(
        wm,
        &format!("{prefix}.conv1_down"),
        x1,
        64,
        [2, 2],
        [2, 2],
        [0, 0],
        1,
        true,
    )?;
    let x1 = crop(b, x1, 16)?;
    let x2 = b.leaky_relu(x2, SLOPE);
    let x2 = unet_conv(b, wm, &format!("{prefix}.conv2"), x2, 64, 128, true)?;

    let x3 = b.conv(
        wm,
        &format!("{prefix}.conv2_down"),
        x2,
        128,
        [2, 2],
        [2, 2],
        [0, 0],
        1,
        true,
    )?;
    let x2 = crop(b, x2, 4)?;
    let x3 = b.leaky_relu(x3, SLOPE);
    let x3 = unet_conv(b, wm, &format!("{prefix}.conv3"), x3, 256, 128, true)?;
    let x3 = b.conv_transpose(
        wm,
        &format!("{prefix}.conv3_up"),
        x3,
        128,
        [2, 2],
        [2, 2],
        [0, 0],
    )?;
    let x3 = b.leaky_relu(x3, SLOPE);

    let x4 = join(b, "UNet2 conv4", x2, x3)?;
    let x4 = unet_conv(b, wm, &format!("{prefix}.conv4"), x4, 64, 64, true)?;
    // `x4 *= alpha` in the reference. Pinned at 1, so this folds away.
    let x4 = if ALPHA == 1.0 {
        x4
    } else {
        let s = b.scalar_like("cugan_alpha", ALPHA, x4);
        b.g().mul(x4, s)
    };
    let x4 = b.conv_transpose(
        wm,
        &format!("{prefix}.conv4_up"),
        x4,
        64,
        [2, 2],
        [2, 2],
        [0, 0],
    )?;
    let x4 = b.leaky_relu(x4, SLOPE);

    let x5 = join(b, "UNet2 conv5", x1, x4)?;
    let x5 = valid3x3(b, wm, &format!("{prefix}.conv5"), x5, 64)?;
    let x5 = b.leaky_relu(x5, SLOPE);

    bottom(b, wm, prefix, x5, out, deconv, false)
}

/// The tail shared by both U-Nets: a transposed convolution that doubles (or,
/// at ×3, triples) the resolution, or a plain valid 3×3 that does not.
fn bottom(
    b: &mut Builder,
    wm: &mut WeightMap,
    prefix: &str,
    x: HirNodeId,
    out: usize,
    deconv: bool,
    x3_deconv: bool,
) -> Result<HirNodeId> {
    let key = format!("{prefix}.conv_bottom");
    if !deconv {
        return valid3x3(b, wm, &key, x, out);
    }
    if x3_deconv {
        b.conv_transpose(wm, &key, x, out, [5, 5], [3, 3], [2, 2])
    } else {
        b.conv_transpose(wm, &key, x, out, [4, 4], [2, 2], [3, 3])
    }
}

/// Build Real-CUGAN.
pub fn build(
    b: &mut Builder,
    wm: &mut WeightMap,
    cfg: &ModelConfig,
    variant: CuganVariant,
    pro: bool,
    x_in: HirNodeId,
) -> Result<HirNodeId> {
    let [_, _, h0, w0] = b.dims4(x_in);
    let step = variant.size_multiple();
    ensure!(
        h0 % step == 0 && w0 % step == 0,
        "Real-CUGAN {} needs input dimensions that are multiples of {step}, got {h0}×{w0}",
        variant.name()
    );

    // `x * 0.7 + 0.15`, undone at the very end.
    let x = if pro {
        let s = b.scalar_like("pro_scale", PRO_SCALE, x_in);
        let o = b.scalar_like("pro_shift", PRO_SHIFT, x_in);
        let t = b.g().mul(x_in, s);
        b.g().add(t, o)
    } else {
        x_in
    };

    let pad = variant.input_pad();
    // A reflection needs something to reflect: mirroring `pad` pixels across an
    // edge reads `pad` pixels inward, so an input narrower than the pad has no
    // defined answer and PyTorch refuses it. The shrinkage from the valid
    // convolutions is *not* what binds here — the halo always more than covers
    // it — so this is the only floor the graph has.
    ensure!(
        h0 > pad && w0 > pad,
        "Real-CUGAN {} reflect-pads by {pad}, which a {h0}×{w0} tile cannot \
         supply; use a tile of at least {}",
        variant.name(),
        pad + 2
    );
    let padded = b.g().pad_(
        x,
        vec![[0, 0], [0, 0], [pad, pad], [pad, pad]],
        PadMode::Reflect,
    );

    let out = match variant {
        CuganVariant::X2 | CuganVariant::X3 => {
            let x3_deconv = variant == CuganVariant::X3;
            let u = unet1(b, wm, "unet1", padded, cfg.out_ch, true, x3_deconv)?;
            let v = unet2(b, wm, "unet2", u, cfg.out_ch, false)?;
            let u = crop(b, u, 20)?;
            join(b, "Real-CUGAN trunk", v, u)?
        }
        CuganVariant::X4 | CuganVariant::X2Fast => {
            // Both feed a 64-channel intermediate into a pixel-shuffled tail.
            let stem = if variant == CuganVariant::X2Fast {
                // `inv` is a PixelUnshuffle(2); the reference pads by 38 first
                // so the unshuffled map still carries the usual 19-pixel halo.
                b.pixel_unshuffle(padded, 2)?
            } else {
                padded
            };
            let u = unet1(b, wm, "unet1", stem, 64, true, false)?;
            let v = unet2(b, wm, "unet2", u, 64, false)?;
            let u = crop(b, u, 20)?;
            let h = join(b, "Real-CUGAN trunk", v, u)?;
            let h = valid3x3(b, wm, "conv_final", h, 4 * cfg.out_ch)?;
            let h = crop(b, h, 1)?;
            let h = b.pixel_shuffle(h, 2)?;
            // The ×4 and fast variants add a nearest-neighbour copy of the
            // *original* input, not of the padded one.
            let ident = b.nearest_upsample(x, cfg.scale);
            join(b, "Real-CUGAN identity", h, ident)?
        }
    };

    if pro {
        let s = b.scalar_like("pro_unscale", 1.0 / PRO_SCALE, out);
        let o = b.scalar_like("pro_unshift", -PRO_SHIFT, out);
        let t = b.g().add(out, o);
        Ok(b.g().mul(t, s))
    } else {
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    /// The pad and crop constants have to leave the network size-preserving.
    /// Worked through here in closed form so a change to either constant fails
    /// this rather than a shape error deep inside a build.
    fn unet1_out(h: usize, deconv_2x: bool) -> usize {
        let h = h - 4; // conv1: two valid 3×3
        let d = h / 2; // conv1_down
        let d = d - 4; // conv2
        let up = d * 2; // conv2_up
        assert_eq!(up, h - 8, "conv2_up must rejoin the 4-cropped x1");
        let h = up - 2; // conv3
        if deconv_2x { h * 2 - 4 } else { h }
    }

    fn unet2_out(h: usize) -> usize {
        let a = h - 4; // conv1
        let d = a / 2; // conv1_down
        let d = d - 4; // conv2
        let e = d / 2; // conv2_down
        let e = e - 4; // conv3
        let up = e * 2; // conv3_up
        assert_eq!(up, d - 8, "conv3_up must rejoin the 4-cropped x2");
        let f = up - 4; // conv4
        let up2 = f * 2; // conv4_up
        assert_eq!(up2, a - 32, "conv4_up must rejoin the 16-cropped x1");
        up2 - 2 - 2 // conv5, then the valid conv_bottom
    }

    #[test]
    fn x2_is_size_preserving() {
        for h in [64usize, 96, 128] {
            let padded = h + 2 * 18;
            let u = unet1_out(padded, true);
            assert_eq!(u, 2 * h + 40);
            assert_eq!(unet2_out(u), 2 * h);
            // The trunk skip crops `u` by 20 on each side to match.
            assert_eq!(u - 40, 2 * h);
        }
    }

    #[test]
    fn x4_is_size_preserving() {
        for h in [64usize, 96, 128] {
            let padded = h + 2 * 19;
            let u = unet1_out(padded, true);
            assert_eq!(u, 2 * h + 44);
            let v = unet2_out(u);
            assert_eq!(v, 2 * h + 4);
            assert_eq!(u - 40, v, "the 20-crop must match UNet2's output");
            // conv_final is valid (−2), then a 1-crop, then PixelShuffle(2).
            assert_eq!((v - 2 - 2) * 2, 4 * h);
        }
    }
}
