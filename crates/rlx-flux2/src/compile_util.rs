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

//! Shared FLUX.2 HIR compile helpers (AOT cache keys + profile-aware compile).

use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_ir::hir::HirModule;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::{AotCache, CompiledGraph, Device, Session};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use rlx_core::flow_bridge::compile_options_from_profile;

/// Compile HIR with tier-1 profile options (fusion, passes, device target).
pub fn compile_hir_cached(
    device: Device,
    aot: Option<&AotCache>,
    key: &str,
    hir: HirModule,
    profile: &CompileProfile,
) -> Result<CompiledGraph> {
    let opts = compile_options_from_profile(profile, device, KernelDispatchConfig::default());
    if let Some(cache) = aot {
        Ok(cache.compile_hir_cached(key, device, hir, &opts)?)
    } else {
        Ok(Session::new(device).compile_hir_with(hir, &opts)?)
    }
}

/// Default FLUX.2 compile profile.
pub fn flux2_compile_profile() -> CompileProfile {
    CompileProfile::flux2()
}

pub fn aot_cache_from_dir(dir: Option<&Path>) -> Option<AotCache> {
    dir.map(AotCache::new)
}

pub fn hash_f32_slice(v: &[f32]) -> u64 {
    let mut h = DefaultHasher::new();
    v.len().hash(&mut h);
    for x in v.iter().take(64) {
        x.to_bits().hash(&mut h);
    }
    if v.len() > 64 {
        if let (Some(a), Some(b)) = (v.first(), v.last()) {
            a.to_bits().hash(&mut h);
            b.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

pub fn flux2_denoiser_aot_key(
    device: Device,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: &[f32],
    txt_ids: &[f32],
    nvfp4: bool,
) -> String {
    format!(
        "flux2_denoiser_{device:?}_b{batch}_is{img_seq}_ts{txt_seq}_nv{nvfp4}_i{}_t{}",
        hash_f32_slice(img_ids),
        hash_f32_slice(txt_ids),
    )
}

pub fn flux2_text_encoder_aot_key(device: Device, batch: usize, txt_seq: usize) -> String {
    format!("flux2_te_{device:?}_b{batch}_ts{txt_seq}")
}

pub fn flux2_vae_decoder_aot_key(device: Device, batch: usize, h: usize, w: usize) -> String {
    format!("flux2_vae_dec_{device:?}_b{batch}_{h}x{w}")
}

pub fn flux2_vae_encoder_aot_key(device: Device, batch: usize, h: usize, w: usize) -> String {
    format!("flux2_vae_enc_{device:?}_b{batch}_{h}x{w}")
}

pub fn flux2_cfg_aot_key(device: Device, batch: usize, img_seq: usize, out_dim: usize) -> String {
    format!("flux2_cfg_{device:?}_b{batch}_is{img_seq}_od{out_dim}")
}
