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

//! Decode/MTP bucket ranges for KV-cache compile caches.
//!
//! Power-of-two ladders waste up to ~2× sequence length on padding I/O. For
//! grounding (past_len usually under 2k) we use 32-wide buckets up to 512, then
//! 128-wide — tighter upper bounds with a bounded number of compiled graphs.
//!
//! WebGPU/Vulkan use a **single** bucket per cache so we do not retain several
//! full 3B arenas (each graph duplicates weights in the backend arena).

use rlx_runtime::Device;
use std::ops::Range;

/// Bucket key ranges for `BucketedCompileCache::with`.
///
/// Each range `start..end` compiles a graph with `upper = end - 1`. Example:
/// past_len=200 → bucket `193..225` → upper=224 (vs 256 for a pure power-of-two ladder).
pub fn locateanything_kv_bucket_ranges(max_past: usize) -> Vec<Range<u64>> {
    let max_past = max_past.max(1) as u64;
    let mut ranges = Vec::new();
    let mut start = 1u64;
    let mut step = 32u64;
    loop {
        let end = (start + step).min(max_past + 1);
        ranges.push(start..end);
        if end > max_past {
            break;
        }
        start = end;
        if start >= 512 {
            step = 128;
        }
    }
    ranges
}

/// Bucket layout for a backend (wgpu/vulkan: one compile per cache to limit VRAM/RAM).
pub fn locateanything_kv_bucket_ranges_for_device(
    device: Device,
    max_past: usize,
) -> Vec<Range<u64>> {
    let max_past = max_past.max(1) as u64;
    if matches!(device, Device::Gpu | Device::Vulkan) {
        return std::iter::once(1..max_past + 1).collect();
    }
    locateanything_kv_bucket_ranges(max_past as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_uses_single_bucket() {
        let buckets = locateanything_kv_bucket_ranges_for_device(Device::Gpu, 1024);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].end - 1, 1024);
    }

    #[test]
    fn past_200_uses_upper_224_not_256() {
        let buckets = locateanything_kv_bucket_ranges(2048);
        let idx = buckets
            .iter()
            .position(|r| r.contains(&200))
            .expect("bucket");
        let upper = buckets[idx].end - 1;
        assert_eq!(upper, 224, "expected 32-wide bucket ending at 224");
    }
}
