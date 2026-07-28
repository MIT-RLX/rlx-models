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

//! Laguna GGUF memory policy — packed by default; F32 expand opt-in.
//!
//! Full quant→F32 materialization is **off by default**. Use header sniff
//! (`--weights`) for the per-file `packed` vs `F32-expand` estimate (XS
//! `Q4_K_M` is typically ~20 GB packed → ~130 GB+ if every element is
//! widened). Prefer packed mmap generate.
//!
//! Contract:
//! - [`ALLOW_F32_EXPAND`] is the compile-time default (`false`).
//! - Runtime opt-in: [`allow_f32_expand`] via `RLX_LAGUNA_ALLOW_F32_EXPAND=1`
//!   or CLI `--allow-f32-expand`.
//! - Prefer [`open_gguf_header_only`] or
//!   [`crate::runner::LagunaPackedRunner::from_gguf_packed`].
//! - Shared loader (`rlx-core`) refuses Laguna **quant** `take` / WeightMap
//!   drain unless opted in; native F32/F16/BF16 may use `take_native_f32`.

use anyhow::{Result, bail};
use rlx_gguf::{GgufFile, bytes_for_public};
use std::path::Path;

/// Compile-time default: F32 expand is off. Prefer [`allow_f32_expand`] for
/// the effective runtime policy.
pub const ALLOW_F32_EXPAND: bool = false;

/// Effective policy: default off; true when
/// `RLX_LAGUNA_ALLOW_F32_EXPAND` is `1` / `true` / `on` / `yes`.
#[inline]
pub fn allow_f32_expand() -> bool {
    rlx_core::laguna_allow_f32_expand()
}

/// Estimated on-disk packed size vs F32 materialization for tensors listed
/// in a (header-only) GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufRamEstimate {
    pub tensor_count: usize,
    /// Sum of ggml block sizes for listed tensors (≈ packed footprint).
    pub packed_bytes: u64,
    /// `n_elements * 4` if every tensor were widened to F32 (opt-in path).
    pub f32_expand_bytes: u64,
}

impl GgufRamEstimate {
    pub fn packed_gb(self) -> f64 {
        self.packed_bytes as f64 / 1e9
    }

    pub fn f32_gb(self) -> f64 {
        self.f32_expand_bytes as f64 / 1e9
    }

    pub fn expand_ratio(self) -> f64 {
        if self.packed_bytes == 0 {
            return 0.0;
        }
        self.f32_expand_bytes as f64 / self.packed_bytes as f64
    }
}

/// Header-only open — metadata + tensor table without holding payloads.
pub fn open_gguf_header_only(path: impl AsRef<Path>) -> Result<GgufFile> {
    let path = path.as_ref();
    GgufFile::header_from_path(path)
        .map_err(|e| anyhow::anyhow!("rlx-laguna: header-only open {}: {e:#}", path.display()))
}

/// Estimate packed vs hypothetical F32 RAM (works on header-only files).
pub fn estimate_ram(raw: &GgufFile) -> GgufRamEstimate {
    let mut packed_bytes = 0u64;
    let mut f32_expand_bytes = 0u64;
    for t in raw.tensors.values() {
        let n = t.n_elements() as u64;
        f32_expand_bytes += n.saturating_mul(4);
        packed_bytes += bytes_for_public(t.dtype, t.n_elements())
            .map(|b| b as u64)
            .unwrap_or(n.saturating_mul(4));
    }
    GgufRamEstimate {
        tensor_count: raw.tensors.len(),
        packed_bytes,
        f32_expand_bytes,
    }
}

/// Error unless F32 expand is opted in.
///
/// Returns `Ok(())` when [`allow_f32_expand`] is true.
pub fn refuse_f32_expand(context: &str) -> Result<()> {
    if allow_f32_expand() {
        return Ok(());
    }
    bail!(
        "rlx-laguna: F32 weight expand disabled by default ({context}). \
         Prefer packed mmap (`--packed-load`) or header sniff (`--weights`) \
         for the packed vs F32-expand estimate. \
         Side tensors: `take_packed_metadata` / `take_native_f32`. \
         Opt in: `RLX_LAGUNA_ALLOW_F32_EXPAND=1` or `--allow-f32-expand`."
    )
}

/// Human-readable policy blurb for CLI / docs.
pub const PACKED_ONLY_POLICY: &str = "\
Laguna memory policy (F32 expand off by default):
- `--weights` opens GGUF **header-only** (prints packed vs F32-expand estimate)
- `--weights … --packed-load` mmaps GGUF; mats stay packed; norms/biases native F32 only
- Quant F32 drain / `dequant_f32` / `load_weight_map` on laguna → error unless opted in
- Opt in: `RLX_LAGUNA_ALLOW_F32_EXPAND=1` or `--allow-f32-expand` (see sniff `F32-expand≈`)
- Generate: packed host fused dequant+matmul (`--packed-load --max-tokens`); KV-cached decode; optional `--device metal|mlx`
";

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_gguf::{GgmlType, GgufWriter, MetaValue};

    fn write_tiny_q4k_gguf(path: &std::path::Path) {
        let mut w = GgufWriter::new();
        w.set_arch("laguna");
        w.set_meta("laguna.block_count", MetaValue::U32(1));
        w.set_meta("laguna.embedding_length", MetaValue::U32(16));
        w.set_meta("laguna.feed_forward_length", MetaValue::U32(32));
        w.set_meta("laguna.attention.head_count", MetaValue::U32(4));
        w.set_meta("laguna.attention.head_count_kv", MetaValue::U32(2));
        w.set_meta("laguna.expert_count", MetaValue::U32(4));
        w.set_meta("laguna.expert_used_count", MetaValue::U32(2));
        w.set_meta("laguna.expert_feed_forward_length", MetaValue::U32(8));
        let n = 256usize;
        let nbytes = bytes_for_public(GgmlType::Q4K, n).expect("q4k size");
        let bytes = vec![0u8; nbytes];
        w.add_tensor_bytes("blk.0.ffn_down.weight", vec![16, 16], GgmlType::Q4K, bytes)
            .unwrap();
        w.write_to_path(path).unwrap();
    }

    #[test]
    fn f32_expand_default_is_off() {
        const { assert!(!ALLOW_F32_EXPAND) };
        // Do not assert `!allow_f32_expand()` — parallel tests may set the env.
    }

    #[test]
    fn header_only_does_not_hold_tensor_payload() {
        let path = std::env::temp_dir().join("rlx_laguna_tiny_q4k_policy.gguf");
        write_tiny_q4k_gguf(&path);

        let hdr = open_gguf_header_only(&path).unwrap();
        let est = estimate_ram(&hdr);
        assert_eq!(est.tensor_count, 1);
        assert!(est.packed_bytes > 0);
        assert!(est.f32_expand_bytes > est.packed_bytes);
        assert!(hdr.dequant_f32("blk.0.ffn_down.weight").is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn refuse_f32_expand_errors_when_disabled() {
        if allow_f32_expand() {
            return;
        }
        let err = refuse_f32_expand("unit-test").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("disabled") || msg.contains("FORBIDDEN") || msg.contains("packed"));
        assert!(msg.contains("ALLOW_F32_EXPAND") || msg.contains("allow-f32-expand"));
    }
}
