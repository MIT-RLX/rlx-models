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

//! Tier-1 compile profiles and bucket keys for Whisper graphs.

use crate::builder::WhisperGraphOpts;
use crate::config::WhisperConfig;
use crate::flow::{
    WhisperEncoderFlow, build_whisper_align_hidden_built_ext_opts, build_whisper_cross_kv_built,
    build_whisper_decode_step_built_ext_opts, build_whisper_decoder_prefill_built_ext_opts,
};
use crate::fused::{FusedDecoderWeights, FusedEncoderWeights};
use crate::weights::WhisperWeightPrefix;
use anyhow::Result;
use rlx_core::device_supports_gpu_kv;
use rlx_core::flow_bridge::{compile_options_for_profile, profile_near_weights};
use rlx_core::weight_map::WeightMap;
use rlx_flow::BuiltModel;
use rlx_flow::CompileProfile;
use rlx_opt::PrecisionPolicy;
use rlx_runtime::compile_cache::BucketedCompileCache;
use rlx_runtime::{CompileOptions, Device, Precision};
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

type WeightTensor = (Vec<f32>, Vec<usize>);
type WhisperWeightMap = HashMap<String, WeightTensor>;

/// Non-bucketed compile cache key (prefill graphs indexed by `(batch, dec_seq)`).
pub fn decode_cache_key(batch: usize, seq: usize) -> u64 {
    ((batch as u64) << 32) | (seq as u64)
}

/// Encoder / cross graphs keyed by `(geometry_epoch, batch)`.
pub fn geometry_cache_key(geometry_epoch: u64, batch: usize) -> u64 {
    (geometry_epoch << 32) | batch as u64
}

/// Prefill graphs also depend on encoder sequence length (cross KV width).
pub fn prefill_cache_key(geometry_epoch: u64, batch: usize, dec_seq: usize) -> u64 {
    geometry_epoch ^ ((batch as u64) << 24) ^ (dec_seq as u64)
}

/// Align-hidden graphs keyed by `(geometry_epoch, dec_seq, max_layer)`.
pub fn align_cache_key(geometry_epoch: u64, dec_seq: usize, max_layer: usize) -> u64 {
    geometry_epoch ^ ((dec_seq as u64) << 20) ^ (max_layer as u64)
}

/// Power-of-two decode buckets; graphs take runtime `pos_ix` + bucket self-attn mask.
pub fn decode_bucket_ladder(device: Device, max_past: u64) -> BucketedCompileCache {
    let max_past = max_past.max(1);
    #[allow(clippy::single_range_in_vec_init)]
    let mut ranges: Vec<Range<u64>> = vec![0..2];
    let mut start = 2u64;
    let mut extent = 4u64;
    loop {
        ranges.push(start..(extent + 1));
        if extent >= max_past {
            break;
        }
        start = extent + 1;
        extent = extent.saturating_mul(2).max(start + 1);
    }
    BucketedCompileCache::with_policy(device, ranges, None)
}

/// MPSGraph rejects some fused attention reshapes on Metal decode graphs.
pub(crate) fn metal_safe_decode_profile(
    device: Device,
    mut profile: CompileProfile,
) -> CompileProfile {
    if device == Device::Metal {
        profile.fusion.skip = true;
        profile.backend.metal.skip_fusion = true;
        profile.backend.metal.unfuse_regions = true;
    }
    profile
}

fn apply_f16_compute(opts: &mut CompileOptions, f16: bool) {
    if f16 {
        opts.precision = Precision::F16;
        opts.policy = Some(PrecisionPolicy::AutoMixed);
    }
}

/// Resolved compile options for encoder, cross, prefill, bucketed decode, and align-hidden.
#[derive(Debug, Clone)]
pub struct WhisperCompileOpts {
    pub encoder: CompileOptions,
    pub cross: CompileOptions,
    pub prefill: CompileOptions,
    pub decode: CompileOptions,
    /// Align-hidden forward (same profile as encoder; runs on [`whisper_encoder_device`]).
    pub align: CompileOptions,
}

/// GPU-resident KV when decode runs on the same device (not CPU-decode fallback).
pub fn whisper_use_gpu_kv(lm_device: Device, decode_device: Device) -> bool {
    device_supports_gpu_kv(decode_device) && decode_device == lm_device
}

/// Device for decoder stages (cross, prefill, bucketed decode).
///
/// Metal / MLX / Vulkan keep decoder graphs on CPU until attention parity
/// matches the CPU reference; encoder may still run on the requested device.
///
/// On `--device metal`, typical routing is:
///
/// | Stage | Device |
/// |-------|--------|
/// | Mel encoder | Metal |
/// | Cross / prefill / decode | CPU |
/// | DTW align-hidden | Metal |
pub fn whisper_decoder_device(lm_device: Device) -> Device {
    match lm_device {
        Device::Metal | Device::Mlx | Device::Vulkan => Device::Cpu,
        other => other,
    }
}

/// Device for the mel encoder graph.
pub fn whisper_encoder_device(lm_device: Device) -> Device {
    match lm_device {
        // MLX encoder self-attn hits (1500,384) vs (1320,384) broadcast mismatch on JFK.
        Device::Mlx => Device::Cpu,
        other => other,
    }
}

/// Disable MPSGraph for LM-shaped compiles on Metal (attention reshape coverage).
pub fn metal_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device == Device::Metal {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        let out = f();
        rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
        out
    } else {
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::Device;

    #[test]
    fn decode_bucket_ladder_includes_past_zero() {
        let cache = decode_bucket_ladder(Device::Cpu, 448);
        assert!(cache.bucket_for(0).is_some());
        assert!(cache.bucket_for(1).is_some());
        assert!(cache.bucket_for(200).is_some());
    }

    #[test]
    fn decode_bucket_ladder_is_logarithmic() {
        let cache = decode_bucket_ladder(Device::Cpu, 448);
        assert!(
            cache.total_buckets() <= 12,
            "expected O(log n) buckets, got {}",
            cache.total_buckets()
        );
        assert_eq!(cache.bucket_for(0), cache.bucket_for(1));
        assert_eq!(cache.bucket_for(5), cache.bucket_for(8));
    }

    #[test]
    fn stage_device_routing() {
        assert_eq!(whisper_encoder_device(Device::Cpu), Device::Cpu);
        assert_eq!(whisper_decoder_device(Device::Cpu), Device::Cpu);
        assert_eq!(whisper_encoder_device(Device::Metal), Device::Metal);
        assert_eq!(whisper_decoder_device(Device::Metal), Device::Cpu);
        assert_eq!(whisper_encoder_device(Device::Mlx), Device::Cpu);
        assert_eq!(whisper_decoder_device(Device::Mlx), Device::Cpu);
        assert_eq!(whisper_encoder_device(Device::Gpu), Device::Gpu);
        assert_eq!(whisper_decoder_device(Device::Gpu), Device::Gpu);
        assert_eq!(whisper_decoder_device(Device::Vulkan), Device::Cpu);
    }
}

impl WhisperCompileOpts {
    pub fn new(device: Device, f16_compute: bool, weights: &Path) -> Self {
        let encoder_profile =
            profile_near_weights(weights, "whisper.rlx.toml", CompileProfile::encoder());
        let decode_profile = metal_safe_decode_profile(device, CompileProfile::gemma_decode());
        let prefill_profile = metal_safe_decode_profile(device, CompileProfile::llama32_prefill());

        let mut encoder = compile_options_for_profile(&encoder_profile, device);
        let mut cross = compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut prefill = compile_options_for_profile(&prefill_profile, device);
        let mut decode = compile_options_for_profile(&decode_profile, device);

        apply_f16_compute(&mut encoder, f16_compute);
        apply_f16_compute(&mut cross, f16_compute);
        apply_f16_compute(&mut prefill, f16_compute);
        apply_f16_compute(&mut decode, f16_compute);

        let align = encoder.clone();
        Self {
            encoder,
            cross,
            prefill,
            decode,
            align,
        }
    }
}

/// Shared checkpoint + graph options (cheap `Clone` via `Arc` weights).
#[derive(Clone)]
pub struct WhisperGraphCtx {
    pub cfg: WhisperConfig,
    pub pfx: WhisperWeightPrefix,
    pub weights: Arc<WhisperWeightMap>,
    pub enc_seq: usize,
    pub mel_frames: usize,
    pub graph_opts: WhisperGraphOpts,
    pub fused: Option<FusedDecoderWeights>,
    pub fused_enc: Option<FusedEncoderWeights>,
}

impl WhisperGraphCtx {
    pub fn weight_map(&self) -> WeightMap {
        WeightMap::from_tensors((*self.weights).clone())
    }

    /// Clone only the tensors a build may need, dropping keys that *definitely* belong
    /// to the other half. `build()` drains weights via `take()`, so the full map was
    /// cloned on every build; this halves that transient allocation. Conservative —
    /// any key not clearly encoder/decoder-only is kept in both subsets, so no build
    /// can ever miss a tensor it needs.
    fn weight_map_excluding(&self, exclude: impl Fn(&str) -> bool) -> WeightMap {
        let mut m: WhisperWeightMap = HashMap::with_capacity(self.weights.len());
        for (k, v) in self.weights.iter() {
            if !exclude(k) {
                m.insert(k.clone(), v.clone());
            }
        }
        WeightMap::from_tensors(m)
    }

    /// Encoder-only subset (drops decoder + fused-decoder tensors).
    fn weight_map_encoder(&self) -> WeightMap {
        let dec = self.pfx.decoder.clone();
        self.weight_map_excluding(move |k| k.starts_with(&dec) || k.starts_with("fused.dec"))
    }

    /// Decoder-family subset for cross / prefill / decode / align (drops encoder tensors).
    fn weight_map_decoder(&self) -> WeightMap {
        let enc = self.pfx.encoder.clone();
        self.weight_map_excluding(move |k| k.starts_with(&enc) || k.starts_with("fused.enc"))
    }

    pub fn build_encoder(&self, batch: usize) -> Result<BuiltModel> {
        self.build_encoder_sized(batch, self.mel_frames)
    }

    pub fn build_encoder_sized(&self, batch: usize, mel_frames: usize) -> Result<BuiltModel> {
        let mut wm = self.weight_map_encoder();
        WhisperEncoderFlow::new_opts(
            &self.cfg,
            &wm,
            batch,
            mel_frames,
            self.graph_opts,
            self.fused_enc.as_ref(),
        )
        .build(&mut wm)
    }

    pub fn build_cross(&self, batch: usize) -> Result<BuiltModel> {
        self.build_cross_sized(batch, self.enc_seq)
    }

    pub fn build_cross_sized(&self, batch: usize, enc_seq: usize) -> Result<BuiltModel> {
        let mut wm = self.weight_map_decoder();
        build_whisper_cross_kv_built(&self.cfg, &mut wm, &self.pfx, batch, enc_seq)
    }

    pub fn build_prefill(&self, batch: usize, dec_seq: usize) -> Result<BuiltModel> {
        self.build_prefill_sized(batch, dec_seq, self.enc_seq)
    }

    pub fn build_prefill_sized(
        &self,
        batch: usize,
        dec_seq: usize,
        enc_seq: usize,
    ) -> Result<BuiltModel> {
        let mut wm = self.weight_map_decoder();
        build_whisper_decoder_prefill_built_ext_opts(
            &self.cfg,
            &mut wm,
            &self.pfx,
            batch,
            dec_seq,
            enc_seq,
            true,
            self.graph_opts,
            self.fused.as_ref(),
        )
    }

    pub fn build_decode_step(&self, batch: usize, bucket_upper: usize) -> Result<BuiltModel> {
        self.build_decode_step_sized(batch, bucket_upper, self.enc_seq)
    }

    pub fn build_decode_step_sized(
        &self,
        batch: usize,
        bucket_upper: usize,
        enc_seq: usize,
    ) -> Result<BuiltModel> {
        let mut wm = self.weight_map_decoder();
        build_whisper_decode_step_built_ext_opts(
            &self.cfg,
            &mut wm,
            &self.pfx,
            batch,
            bucket_upper,
            enc_seq,
            true,
            self.graph_opts,
            self.fused.as_ref(),
        )
    }

    /// Compile align-hidden graph (decoder through alignment layer, self-attn only on last layer).
    pub fn build_align_hidden_sized(
        &self,
        batch: usize,
        dec_seq: usize,
        enc_seq: usize,
        max_layer: usize,
    ) -> Result<BuiltModel> {
        let mut wm = self.weight_map_decoder();
        build_whisper_align_hidden_built_ext_opts(
            &self.cfg,
            &mut wm,
            &self.pfx,
            batch,
            dec_seq,
            enc_seq,
            max_layer,
            self.graph_opts,
            self.fused.as_ref(),
        )
    }
}
