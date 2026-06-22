# BF8 Dequant Analysis — Final Summary

## The Problem
Our `dequant_weights()` produces weight values that differ from libnc's `nc_convert` by up to 5000× per-element, resulting in 0.002 WAV correlation with the original tsac output.

## What Was Tested (13 formulas, all failed)

| Formula | Corr | Notes |
|---------|------|-------|
| `int8 * scale / 127` | 0.006 | Our best candidate |
| `(byte-128) * scale / 127` | -0.025 | Original formula |
| `int8 * scale / 255` | 0.006 | Same corr (K cancels) |
| `int8 * 2^((s-127)/127)` | 0.009 | Exponential |
| `int8 * (s-127) / 127` | 0.004 | Zero-centered |
| `int8 / (255-s+1)` | 0.032 | Inverse scale |
| `int8 / s * 127` | 0.039 | **Best** |

## Key Evidence
1. **First group**: consistent ratio of 4902.3 for all 14 elements (our/libnc)
2. **Subsequent groups**: ratio varies wildly (different grouping axis)
3. **Batch injection**: All 22 layers replaced with libnc weights → RMS 0.9999 (clipping)
4. **K=1 layers**: libnc norms ≈ 1.0 (L2 normalized)
5. **K>1 layers**: libnc norms = 2.5-4.5 (raw magnitude, NOT normalized)
6. **Group size**: model.0 uses group_size=14 (confirmed, no padding issues)

## Root Cause
libnc's `nc_reduce_sum_sqr` (0x8310) processes BF8 grouped data along a DIFFERENT axis than our dequant_weights. We group consecutive elements in [Ci,K,Co] linear order. libnc likely groups across K×Co interleaved.

**Evidence for grouping axis mismatch**:
- First group has CONSTANT ratio (scale applies to same elements)
- Second group onwards: ratio VARYING (elements shuffled differently)
- Unit-norm correlation: -0.000000 (after removing scale factor, values still don't match)

## Captured libnc Weight Files
14 decoder layer files in `docs/libnc_weights/`:
| File | Layer | Type |
|------|-------|------|
| `1024x7x1536.bin` | model.0 | conv1d |
| `768x16x1536.bin` | model.1.block.1 | convtr |
| `768x7x768.bin` | model.1 inner | conv1d K=7 |
| `768x1x768.bin` | model.1 inner | conv1d K=1 |
| `384x16x768.bin` | model.2.block.1 | convtr |
| ... | ... | ... |

## Recommended Fix Approach
1. Single-step GDB through `nc_reduce_sum_sqr` (0x8310) with type=11
2. Identify how GROUP_SIZE is determined from tensor metadata
3. Identify which axis is used for grouping
4. Implement matching grouping in dequant_weights
