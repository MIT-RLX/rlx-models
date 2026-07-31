// RLX — versatile ML compiler + runtime. GPLv3.
//! Host-side dequant for the **GA `deepseek-ai/DeepSeek-V4-Flash-0731` original**
//! checkpoint, whose weights are stored in the DeepSeek FP8/FP4 block-quant format
//! (`config.json`: `quant_method: fp8`, `weight_block_size: [128, 128]`,
//! `scale_fmt: ue8m0`, `expert_dtype: fp4`) rather than the MLX-affine repacks
//! (`mlx-community/DeepSeek-V4-Flash-{2,3,4}bit-DQ`) that the existing `MlxLoader`
//! path already handles.
//!
//! These are pure host kernels that materialize F32, feeding the same dense path
//! `build_deepseek_v4_stage`/`build_dspark_stage` already use for F32 weights.
//! They are deliberately implemented **here** (not as a new `QuantScheme` in
//! `rlx-ir`/`rlx-mlx-io`) so the GA-original path adds nothing to the shared
//! `../rlx` loader — which was under a large in-flight refactor at the time.
//!
//! Two formats, matching `inference/{kernel.py,convert.py}`:
//! * **FP8 block** (dense/attn Linears) — `e4m3fn` weights `[out, in]` with a
//!   `ue8m0` block scale `[ceil(out/128), ceil(in/128)]` (one power-of-two scale
//!   per 128×128 block). `fp8_gemm` applies `scales_a[m,·]·scales_b[n,·]`; the
//!   weight side is `w[o,i] = e4m3(o,i) · scale[o/128, i/128]`.
//! * **FP4 block** (MoE experts, `expert_dtype: fp4`) — `e2m1` values packed two
//!   per byte along `in` (`[·, in/2]`) with a `ue8m0` scale per 32 (`[·, in/32]`).
//!   This is exactly OCP MXFP4; `w[o,i] = E2M1[nibble(o,i)] · scale[o, i/32]`.
//!
//! Block scales are `ue8m0` bytes decoded with
//! [`rlx_mlx_io::mxfp4_scale_e8m0_to_f32`] (`2^(e-127)`); callers that hold f32
//! `weight_scale_inv` tensors instead pass them straight through.

/// OCP **E2M1** (FP4) value table — index by nibble. Matches the reference
/// `convert.py` `FP4_TABLE` (`[0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6]`).
pub const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one **`e4m3fn`** (FP8, sign / 4-bit exp bias-7 / 3-bit mantissa) byte to
/// f32. `e4m3fn` has no infinities; the sole NaN is `exp==15 && mantissa==7`
/// (`0x7F`/`0xFF`), and the max finite magnitude is 448 (`exp==15, mantissa==6`).
#[inline]
pub fn e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (b >> 3) & 0x0F;
    let mant = (b & 0x07) as f32;
    if exp == 0 {
        // subnormal: 2^(1-bias) · (mant/8) = 2^-6 · mant/8
        sign * (2f32.powi(-6)) * (mant / 8.0)
    } else if exp == 0x0F && mant == 7.0 {
        f32::NAN
    } else {
        sign * 2f32.powi(exp as i32 - 7) * (1.0 + mant / 8.0)
    }
}

/// Dequantize an **FP8 block-quant** weight `[rows, cols]` to F32.
/// `codes` are `e4m3fn` bytes (row-major, one per element); `block_scales` are the
/// already-decoded f32 block scales laid out `[ceil(rows/block_r), ceil(cols/block_c)]`
/// row-major (decode `ue8m0` bytes with [`rlx_mlx_io::mxfp4_scale_e8m0_to_f32`]
/// first, or pass an f32 `weight_scale_inv` verbatim). DeepSeek-V4 uses
/// `block_r == block_c == 128`.
pub fn dequant_fp8_block(
    codes: &[u8],
    block_scales: &[f32],
    rows: usize,
    cols: usize,
    block_r: usize,
    block_c: usize,
) -> Vec<f32> {
    assert_eq!(codes.len(), rows * cols, "fp8 codes size");
    let sc_cols = cols.div_ceil(block_c);
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let sr = r / block_r;
        for c in 0..cols {
            let s = block_scales[sr * sc_cols + c / block_c];
            out[r * cols + c] = e4m3fn_to_f32(codes[r * cols + c]) * s;
        }
    }
    out
}

/// Dequantize an **FP4 (E2M1) block-quant** weight `[rows, cols]` to F32.
/// `packed` holds two FP4 nibbles per byte along `cols` (`[rows, cols/2]`,
/// low nibble = even column); `block_scales` are decoded f32 scales laid out
/// `[rows, ceil(cols/block)]` row-major (DeepSeek-V4 experts: `block == 32`).
/// Equivalent to OCP MXFP4 dequant.
pub fn dequant_fp4_block(
    packed: &[u8],
    block_scales: &[f32],
    rows: usize,
    cols: usize,
    block: usize,
) -> Vec<f32> {
    assert_eq!(packed.len(), rows * cols.div_ceil(2), "fp4 packed size");
    let sc_cols = cols.div_ceil(block);
    let bytes_per_row = cols.div_ceil(2);
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let byte = packed[r * bytes_per_row + c / 2];
            let nib = if c % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            let s = block_scales[r * sc_cols + c / block];
            out[r * cols + c] = E2M1_TABLE[nib as usize] * s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_known_values() {
        // (byte, expected) — hand-derived from the e4m3fn spec.
        let cases = [
            (0x00u8, 0.0f32),
            (0x38, 1.0),   // exp 7, mant 0
            (0x3F, 1.875), // exp 7, mant 7 → 1+7/8
            (0x40, 2.0),   // exp 8
            (0x78, 256.0), // exp 15, mant 0 → 2^8
            (0x7E, 448.0), // exp 15, mant 6 → 2^8·1.75 (max finite)
            (0xB8, -1.0),  // sign, exp 7
            (0x01, 0.001953125), // subnormal 2^-9
        ];
        for (b, want) in cases {
            let got = e4m3fn_to_f32(b);
            assert!((got - want).abs() < 1e-6, "e4m3 {b:#04x} = {got} want {want}");
        }
        assert!(e4m3fn_to_f32(0x7F).is_nan());
        assert!(e4m3fn_to_f32(0xFF).is_nan());
    }

    #[test]
    fn fp8_block_dequant_matches_reference() {
        // 2 rows × 3 cols, 2×2 blocks ⇒ scale grid [1, 2].
        let (rows, cols, br, bc) = (2usize, 3usize, 2usize, 2usize);
        let codes: Vec<u8> = vec![0x38, 0x40, 0x3F, 0xB8, 0x78, 0x00];
        let scales = vec![2.0f32, 0.5]; // [ceil(2/2)=1, ceil(3/2)=2] → 2 scales
        let out = dequant_fp8_block(&codes, &scales, rows, cols, br, bc);
        // reference: value(code) · scale[row/2, col/2]
        let mut refv = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                refv[r * cols + c] = e4m3fn_to_f32(codes[r * cols + c]) * scales[c / bc];
            }
        }
        let max = out
            .iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-6, "fp8 block max err {max:e}");
        // spot check: [0,0] = 1.0·2.0 = 2.0; [0,2] = 1.875·0.5 = 0.9375
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[2] - 0.9375).abs() < 1e-6);
    }

    #[test]
    fn fp4_block_dequant_matches_reference() {
        // 1 row × 4 cols, block 2 ⇒ scale grid [1, 2]. Packed 2 nibbles/byte.
        let (rows, cols, block) = (1usize, 4usize, 2usize);
        // cols: [1, 2, 3, 4] → nibbles [1,2,3,4] = E2M1 [0.5,1.0,1.5,2.0]
        // byte0 = (col1<<4)|col0 = (2<<4)|1 = 0x21 ; byte1 = (4<<4)|3 = 0x43
        let packed = vec![0x21u8, 0x43];
        let scales = vec![2.0f32, 4.0]; // block0 (cols 0,1), block1 (cols 2,3)
        let out = dequant_fp4_block(&packed, &scales, rows, cols, block);
        let want = [0.5 * 2.0, 1.0 * 2.0, 1.5 * 4.0, 2.0 * 4.0];
        for i in 0..cols {
            assert!((out[i] - want[i]).abs() < 1e-6, "fp4 col {i} = {} want {}", out[i], want[i]);
        }
    }

    #[test]
    fn e8m0_scale_roundtrip() {
        // ue8m0 byte e decodes to 2^(e-127); reuse the shared rlx-mlx-io helper.
        assert!((rlx_mlx_io::mxfp4_scale_e8m0_to_f32(127) - 1.0).abs() < 1e-6);
        assert!((rlx_mlx_io::mxfp4_scale_e8m0_to_f32(128) - 2.0).abs() < 1e-6);
        assert!((rlx_mlx_io::mxfp4_scale_e8m0_to_f32(126) - 0.5).abs() < 1e-6);
    }
}
