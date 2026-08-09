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

//! LM weight storage precision / quant for Unlimited-OCR MoE.
//!
//! - [`LmWeightPrecision::F32`] — parity path (exact host F32 pack).
//! - [`LmWeightPrecision::F16`] / `Bf16` — half-size host cache.
//! - [`LmWeightPrecision::Q8_0`] / `Q4_0` — GGUF block quants on host
//!   (~⅛–¼ of F32); kept packed in IR via Dequant*MatMul.
//! - [`LmWeightPrecision::Auto`] — cascade F32 → F16 → Q8_0 → Q4_0 by RAM.

use crate::config::UnlimitedOcrConfig;
use anyhow::{Result, bail};
use std::fmt;

/// How to store packed LM weights on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LmWeightPrecision {
    /// Full F32 host cache — bit-exact vs the eager reference path.
    F32,
    /// IEEE F16 host cache for large mats (~½ RAM vs F32).
    F16,
    /// BF16 host cache for large mats (~½ RAM vs F32).
    Bf16,
    /// GGUF Q8_0 block quant (~1.06 bytes/elem).
    Q8_0,
    /// GGUF Q4_0 block quant (~0.56 bytes/elem).
    Q4_0,
    /// Prefer F32 when RAM allows, else F16, then Q8_0, then Q4_0.
    #[default]
    Auto,
}

impl LmWeightPrecision {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            "f16" | "fp16" | "float16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "q8_0" | "q8" | "q80" => Ok(Self::Q8_0),
            "q4_0" | "q4" | "q40" => Ok(Self::Q4_0),
            "auto" => Ok(Self::Auto),
            other => bail!("unknown lm precision {other:?} (expected f32|f16|bf16|q8_0|q4_0|auto)"),
        }
    }
}

impl fmt::Display for LmWeightPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q4_0 => write!(f, "q4_0"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Concrete storage dtype after resolving [`LmWeightPrecision::Auto`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedLmPrecision {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
}

impl ResolvedLmPrecision {
    /// Approximate bytes per element for packed large mats (×1000 scale for quants).
    pub fn bytes_per_elem_milli(self) -> u64 {
        match self {
            Self::F32 => 4000,
            Self::F16 | Self::Bf16 => 2000,
            // Q8_0: 34 bytes / 32 elems
            Self::Q8_0 => 1063,
            // Q4_0: 18 bytes / 32 elems
            Self::Q4_0 => 563,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }
}

impl fmt::Display for ResolvedLmPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Extra multiplier on packed-weight bytes to leave room for compile
/// param clones, vision towers, KV, and OS headroom (F32 / half IR paths
/// still materialize F32 params).
pub const PACK_COMPILE_HEADROOM: f64 = 2.75;

/// Headroom when Q8_0/Q4_0 stay packed in IR (`Dequant*MatMul` + U8 typed params).
pub const PACK_COMPILE_HEADROOM_IR_PACKED: f64 = 1.35;

/// Fraction of *available* RAM we are willing to claim for LM pack+compile.
pub const USABLE_RAM_FRACTION: f64 = 0.70;

/// Fixed allowance (bytes) reserved outside the LM pack estimate for
/// SAM+CLIP+projector host tensors and working sets.
pub const VISION_WORKING_SET_BYTES: u64 = 2 << 30; // 2 GiB

/// Estimate host bytes for the packed LM cache at `prec`.
///
/// Embeddings / norms / routers stay F32. For [`ResolvedLmPrecision::Q4_0`],
/// attention, `lm_head`, dense MLP, and shared experts soft-pack as F16
/// (routed experts stay Q4); other precisions apply uniformly to large mats.
pub fn estimate_packed_lm_bytes(cfg: &UnlimitedOcrConfig, prec: ResolvedLmPrecision) -> u64 {
    let h = cfg.hidden_size as u64;
    let v = cfg.vocab_size as u64;
    let bpe_m = prec.bytes_per_elem_milli();
    let scale = |n: u64, milli: u64| (n * milli).div_ceil(1000);
    // Q4 soft-pack: quality-sensitive mats stay F16 on host.
    let soft_m = match prec {
        ResolvedLmPrecision::Q4_0 => ResolvedLmPrecision::F16.bytes_per_elem_milli(),
        _ => bpe_m,
    };

    let mut bytes = v * h * 4; // embed F32
    bytes += h * 4; // final norm F32
    bytes += scale(v * h, soft_m); // lm_head (F16 under Q4)

    for layer in 0..cfg.num_hidden_layers as u64 {
        bytes += h * 4 * 2; // layernorms F32
        bytes += scale(h * h * 4, soft_m); // q,k,v,o (F16 under Q4)
        if (layer as usize) < cfg.first_k_dense_replace {
            let ff = cfg.intermediate_size as u64;
            bytes += scale(ff * h * 2 + h * ff, soft_m);
        } else {
            let n_e = cfg.n_routed_experts as u64;
            let moe_ff = cfg.moe_intermediate_size as u64;
            let shared_ff = moe_ff * cfg.n_shared_experts as u64;
            bytes += n_e * h * 4; // router F32
            bytes += scale(shared_ff * h * 2 + h * shared_ff, soft_m);
            bytes += scale(n_e * (h * moe_ff * 2 + moe_ff * h), bpe_m); // routed
        }
    }
    bytes
}

/// F32 IR bytes materialized for Q4 soft-pack mats (F16 host → F32 params).
fn estimate_q4_soft_ir_f32_bytes(cfg: &UnlimitedOcrConfig) -> u64 {
    let h = cfg.hidden_size as u64;
    let v = cfg.vocab_size as u64;
    let mut elems = v * h; // lm_head
    for layer in 0..cfg.num_hidden_layers as u64 {
        elems += h * h * 4; // attn
        if (layer as usize) < cfg.first_k_dense_replace {
            let ff = cfg.intermediate_size as u64;
            elems += ff * h * 2 + h * ff;
        } else {
            let moe_ff = cfg.moe_intermediate_size as u64;
            let shared_ff = moe_ff * cfg.n_shared_experts as u64;
            elems += shared_ff * h * 2 + h * shared_ff;
        }
    }
    elems * 4
}

/// Bytes we expect for pack+compile of the LM alone (excludes vision reserve).
pub fn estimate_pack_compile_need(cfg: &UnlimitedOcrConfig, prec: ResolvedLmPrecision) -> u64 {
    let packed = estimate_packed_lm_bytes(cfg, prec) as f64;
    match prec {
        ResolvedLmPrecision::Q8_0 => (packed * PACK_COMPILE_HEADROOM_IR_PACKED).ceil() as u64,
        // Soft F16 mats widen to F32 IR params on top of packed Q4 experts.
        ResolvedLmPrecision::Q4_0 => {
            let soft_ir = estimate_q4_soft_ir_f32_bytes(cfg) as f64;
            (packed * PACK_COMPILE_HEADROOM_IR_PACKED + soft_ir).ceil() as u64
        }
        ResolvedLmPrecision::F32 | ResolvedLmPrecision::F16 | ResolvedLmPrecision::Bf16 => {
            (packed * PACK_COMPILE_HEADROOM).ceil() as u64
        }
    }
}

/// Resolve [`LmWeightPrecision`] against available RAM (and optional overrides).
pub fn resolve_lm_precision(
    requested: LmWeightPrecision,
    cfg: &UnlimitedOcrConfig,
) -> ResolvedLmPrecision {
    resolve_lm_precision_with_ram(requested, cfg, available_ram_bytes())
}

/// Same as [`resolve_lm_precision`] with an explicit available-RAM budget (tests).
pub fn resolve_lm_precision_with_ram(
    requested: LmWeightPrecision,
    cfg: &UnlimitedOcrConfig,
    available_ram: u64,
) -> ResolvedLmPrecision {
    match requested {
        LmWeightPrecision::F32 => ResolvedLmPrecision::F32,
        LmWeightPrecision::F16 => ResolvedLmPrecision::F16,
        LmWeightPrecision::Bf16 => ResolvedLmPrecision::Bf16,
        LmWeightPrecision::Q8_0 => ResolvedLmPrecision::Q8_0,
        LmWeightPrecision::Q4_0 => ResolvedLmPrecision::Q4_0,
        LmWeightPrecision::Auto => {
            let usable = ((available_ram as f64) * USABLE_RAM_FRACTION) as u64;
            // Reserve vision/OS working set; cascade on remaining pack budget.
            let pack_budget = usable.saturating_sub(VISION_WORKING_SET_BYTES);
            let cascade = [
                ResolvedLmPrecision::F32,
                ResolvedLmPrecision::F16,
                ResolvedLmPrecision::Q8_0,
                ResolvedLmPrecision::Q4_0,
            ];
            for &prec in &cascade {
                if pack_budget >= estimate_pack_compile_need(cfg, prec) {
                    return prec;
                }
            }
            ResolvedLmPrecision::Q4_0
        }
    }
}

/// Human-readable decision log line for Auto / overrides.
pub fn precision_decision_message(
    requested: LmWeightPrecision,
    resolved: ResolvedLmPrecision,
    cfg: &UnlimitedOcrConfig,
) -> String {
    let avail = available_ram_bytes();
    let usable = ((avail as f64) * USABLE_RAM_FRACTION) as u64;
    let pack_budget = usable.saturating_sub(VISION_WORKING_SET_BYTES);
    let need = estimate_pack_compile_need(cfg, resolved);
    let soft = if resolved == ResolvedLmPrecision::Q4_0 {
        " soft=F16(attn,lm_head,dense,shared)+Q4(routed)"
    } else {
        ""
    };
    format!(
        "lm-precision requested={requested} resolved={resolved}{soft} \
         avail_ram={} pack_budget={} need≈{} (pack≈{})",
        fmt_bytes(avail),
        fmt_bytes(pack_budget),
        fmt_bytes(need),
        fmt_bytes(estimate_packed_lm_bytes(cfg, resolved)),
    )
}

fn fmt_bytes(n: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if n as f64 >= GIB {
        format!("{:.1}GiB", n as f64 / GIB)
    } else {
        format!("{:.0}MiB", n as f64 / MIB)
    }
}

/// Available physical RAM in bytes.
///
/// Override with `RLX_UNLIMITED_OCR_ASSUME_RAM_BYTES` (tests / forced Auto).
pub fn available_ram_bytes() -> u64 {
    if let Ok(s) = std::env::var("RLX_UNLIMITED_OCR_ASSUME_RAM_BYTES") {
        if let Ok(n) = s.parse::<u64>() {
            return n;
        }
    }
    platform_available_ram().unwrap_or(8 << 30)
}

fn platform_available_ram() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = macos_available_approx() {
            return Some(n);
        }
        macos_total_phys()
    }
    #[cfg(target_os = "linux")]
    {
        linux_mem_available().or_else(linux_mem_total)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn macos_total_phys() -> Option<u64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().parse().ok()
}

#[cfg(target_os = "macos")]
fn macos_available_approx() -> Option<u64> {
    let page_out = std::process::Command::new("pagesize").output().ok()?;
    let page_size: u64 = String::from_utf8_lossy(&page_out.stdout)
        .trim()
        .parse()
        .ok()?;
    let out = std::process::Command::new("vm_stat").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut free = 0u64;
    let mut inactive = 0u64;
    let mut speculative = 0u64;
    let mut purgeable = 0u64;
    for line in text.lines() {
        let take = |prefix: &str| -> Option<u64> {
            let rest = line.strip_prefix(prefix)?;
            let num: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
            num.parse().ok()
        };
        if let Some(n) = take("Pages free:") {
            free = n;
        } else if let Some(n) = take("Pages inactive:") {
            inactive = n;
        } else if let Some(n) = take("Pages speculative:") {
            speculative = n;
        } else if let Some(n) = take("Pages purgeable:") {
            purgeable = n;
        }
    }
    let pages = free
        .saturating_add(inactive)
        .saturating_add(speculative)
        .saturating_add(purgeable);
    if pages == 0 {
        return None;
    }
    Some(pages.saturating_mul(page_size))
}

#[cfg(target_os = "linux")]
fn linux_mem_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib.saturating_mul(1024));
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn linux_mem_total() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib.saturating_mul(1024));
        }
    }
    None
}

/// Cast F32 slice → little-endian F16 bytes.
pub fn f32_to_f16_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &v in src {
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

/// Cast F32 slice → little-endian BF16 bytes.
pub fn f32_to_bf16_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &v in src {
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

/// Widen little-endian F16 bytes → F32.
pub fn f16_bytes_to_f32(src: &[u8]) -> Result<Vec<f32>> {
    if !src.len().is_multiple_of(2) {
        bail!("f16 bytes length {} not multiple of 2", src.len());
    }
    let mut out = Vec::with_capacity(src.len() / 2);
    for chunk in src.chunks_exact(2) {
        out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    Ok(out)
}

/// Widen little-endian BF16 bytes → F32.
pub fn bf16_bytes_to_f32(src: &[u8]) -> Result<Vec<f32>> {
    if !src.len().is_multiple_of(2) {
        bail!("bf16 bytes length {} not multiple of 2", src.len());
    }
    let mut out = Vec::with_capacity(src.len() / 2);
    for chunk in src.chunks_exact(2) {
        out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    Ok(out)
}

const QK: usize = 32;

fn pad_to_block(src: &[f32]) -> Vec<f32> {
    let n = src.len().div_ceil(QK) * QK;
    if n == src.len() {
        return src.to_vec();
    }
    let mut v = vec![0f32; n];
    v[..src.len()].copy_from_slice(src);
    v
}

/// Quantize F32 → GGUF Q8_0 bytes (pads to 32-elem blocks).
pub fn f32_to_q8_0_bytes(src: &[f32]) -> Result<Vec<u8>> {
    let padded = pad_to_block(src);
    rlx_gguf::quantize(&padded, rlx_gguf::GgmlType::Q8_0)
}

/// Quantize F32 → GGUF Q4_0 bytes (pads to 32-elem blocks).
pub fn f32_to_q4_0_bytes(src: &[f32]) -> Result<Vec<u8>> {
    let padded = pad_to_block(src);
    rlx_gguf::quantize(&padded, rlx_gguf::GgmlType::Q4_0)
}

/// Dequant Q8_0 → F32, truncated to `nelems`.
pub fn q8_0_bytes_to_f32(src: &[u8], nelems: usize) -> Result<Vec<f32>> {
    let n = nelems.div_ceil(QK) * QK;
    let mut out = rlx_gguf::dequant_q8_0(src, n)?;
    out.truncate(nelems);
    Ok(out)
}

/// Dequant Q4_0 → F32, truncated to `nelems`.
pub fn q4_0_bytes_to_f32(src: &[u8], nelems: usize) -> Result<Vec<f32>> {
    let n = nelems.div_ceil(QK) * QK;
    let mut out = rlx_gguf::dequant_q4_0(src, n)?;
    out.truncate(nelems);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ClipTowerConfig, ProjectorConfig, SamTowerConfig, UnlimitedOcrVisionConfig,
    };

    fn tiny_cfg() -> UnlimitedOcrConfig {
        UnlimitedOcrConfig {
            model_type: "unlimited-ocr".into(),
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            n_routed_experts: 4,
            n_shared_experts: 2,
            num_experts_per_tok: 2,
            moe_intermediate_size: 32,
            intermediate_size: 64,
            first_k_dense_replace: 1,
            vocab_size: 128,
            max_position_embeddings: 256,
            sliding_window: 16,
            use_mla: false,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            hidden_act: "silu".into(),
            bos_token_id: 0,
            eos_token_id: 1,
            pad_token_id: 2,
            image_token_id: 3,
            v_head_dim: Some(16),
            vision_config: UnlimitedOcrVisionConfig {
                sam: SamTowerConfig::default(),
                clip: ClipTowerConfig::default(),
                image_size: 1024,
            },
            projector: ProjectorConfig {
                input_dim: 2048,
                n_embed: 64,
                projector_type: "linear".into(),
            },
            patch_size: 16,
            downsample_ratio: 4,
        }
    }

    #[test]
    fn auto_picks_f32_when_ram_is_huge() {
        let cfg = tiny_cfg();
        assert_eq!(
            resolve_lm_precision_with_ram(LmWeightPrecision::Auto, &cfg, 512 << 30),
            ResolvedLmPrecision::F32
        );
    }

    #[test]
    fn auto_cascades_to_quant_when_ram_is_tiny() {
        let cfg = tiny_cfg();
        // usable ≪ vision reserve → pack_budget 0 → Q4_0.
        assert_eq!(
            resolve_lm_precision_with_ram(LmWeightPrecision::Auto, &cfg, 512 << 20),
            ResolvedLmPrecision::Q4_0
        );
    }

    #[test]
    fn auto_picks_f16_between_f32_and_quant() {
        // Fat LM so F32 pack exceeds a mid-size budget while F16 still fits.
        let mut cfg = tiny_cfg();
        cfg.hidden_size = 2048;
        cfg.num_attention_heads = 16;
        cfg.num_key_value_heads = 16;
        cfg.num_hidden_layers = 12;
        cfg.n_routed_experts = 64;
        cfg.moe_intermediate_size = 2048;
        cfg.vocab_size = 50_000;
        cfg.v_head_dim = Some(128);
        cfg.projector.n_embed = 2048;

        let need_f32 = estimate_pack_compile_need(&cfg, ResolvedLmPrecision::F32);
        let need_f16 = estimate_pack_compile_need(&cfg, ResolvedLmPrecision::F16);
        assert!(need_f32 > need_f16);

        // Choose available RAM so pack_budget sits between need_f16 and need_f32.
        let pack_budget = (need_f16 + need_f32) / 2;
        let usable = pack_budget + VISION_WORKING_SET_BYTES;
        let available = ((usable as f64) / USABLE_RAM_FRACTION).ceil() as u64;
        assert_eq!(
            resolve_lm_precision_with_ram(LmWeightPrecision::Auto, &cfg, available),
            ResolvedLmPrecision::F16
        );
    }

    #[test]
    fn q4_estimate_accounts_for_soft_f16_mats() {
        let mut cfg = tiny_cfg();
        cfg.hidden_size = 1024;
        cfg.num_attention_heads = 8;
        cfg.num_key_value_heads = 8;
        cfg.num_hidden_layers = 4;
        cfg.n_routed_experts = 16;
        cfg.moe_intermediate_size = 512;
        cfg.vocab_size = 8_000;
        cfg.v_head_dim = Some(128);

        let q4 = estimate_packed_lm_bytes(&cfg, ResolvedLmPrecision::Q4_0);
        let q8 = estimate_packed_lm_bytes(&cfg, ResolvedLmPrecision::Q8_0);
        let f16 = estimate_packed_lm_bytes(&cfg, ResolvedLmPrecision::F16);
        // Soft Q4 (F16 head/attn/shared + Q4 routed) sits between pure Q8 and F16.
        assert!(q4 < q8, "soft Q4 host should beat Q8 ({q4} vs {q8})");
        assert!(q4 < f16, "soft Q4 host should beat F16 ({q4} vs {f16})");

        let need_q4 = estimate_pack_compile_need(&cfg, ResolvedLmPrecision::Q4_0);
        let need_q8 = estimate_pack_compile_need(&cfg, ResolvedLmPrecision::Q8_0);
        // Soft IR F32 materialization can push Q4 need above Q8 on small MoEs.
        assert!(
            need_q4 > q4,
            "Q4 need includes soft IR ({need_q4} vs pack {q4})"
        );
        let _ = need_q8;
    }

    #[test]
    fn f16_roundtrip() {
        let v = vec![0.0f32, 1.0, -0.5, std::f32::consts::PI];
        let b = f32_to_f16_bytes(&v);
        let back = f16_bytes_to_f32(&b).unwrap();
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }
    }

    #[test]
    fn q8_0_roundtrip() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let b = f32_to_q8_0_bytes(&v).unwrap();
        let back = q8_0_bytes_to_f32(&b, v.len()).unwrap();
        assert_eq!(back.len(), v.len());
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.15, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_0_roundtrip() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let b = f32_to_q4_0_bytes(&v).unwrap();
        let back = q4_0_bytes_to_f32(&b, v.len()).unwrap();
        assert_eq!(back.len(), v.len());
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.6, "{a} vs {b}");
        }
    }

    #[test]
    fn parse_precision() {
        assert_eq!(
            LmWeightPrecision::parse("f32").unwrap(),
            LmWeightPrecision::F32
        );
        assert_eq!(
            LmWeightPrecision::parse("q8_0").unwrap(),
            LmWeightPrecision::Q8_0
        );
        assert_eq!(
            LmWeightPrecision::parse("Q4").unwrap(),
            LmWeightPrecision::Q4_0
        );
        assert_eq!(
            LmWeightPrecision::parse("auto").unwrap(),
            LmWeightPrecision::Auto
        );
    }
}
