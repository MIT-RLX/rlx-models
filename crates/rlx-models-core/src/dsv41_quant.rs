// RLX — versatile ML compiler + runtime. GPLv3.
//! Host-side dequantization for the released **DeepSeek-V4.1-Flash** checkpoint.
//!
//! `config.json` declares `quant_method: fp8`, `weight_block_size: [32, 32]`,
//! `scale_fmt: ue8m0`, `expert_dtype: fp4`, and the shards carry a companion
//! `<stem>.scale` next to every quantized `<stem>.weight`. Three different scale
//! layouts hide behind that one description, and they are told apart by the
//! *shape* of the scale tensor rather than by name:
//!
//! | tensor | weight | scale | layout |
//! |---|---|---|---|
//! | `attn.wq_a.weight` | `F8_E4M3 [1280, 5120]` | `[40, 160]` | one scale per 32×32 tile |
//! | `ffn.experts.N.w1.weight` | `I8 [2304, 2560]` | `[2304, 160]` | FP4 nibble pairs, one scale per row per 32 columns |
//! | `engram.embed.weight` | `F8_E4M3 [384006168, 256]` | `[384006168, 8]` | FP8, one scale per row per 32 columns |
//!
//! The third is the trap: it is FP8 like the first but scaled like the second,
//! because `ParallelEngramEmbedding` dequantizes whole rows on lookup rather than
//! feeding a blocked GEMM. Reading it with the 32×32 tile rule would apply one
//! row's scale to 32 rows of the table.
//!
//! Scales are `float8_e8m0fnu`. Note this is **not** the MLX E8M0 convention
//! ([`rlx_mlx_io::mxfp4_scale_e8m0_to_f32`] special-cases `0`); here the encoding
//! is the plain OCP one, `2^(b - 127)`, with `255` reserved for NaN.
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/{kernel.py,convert.py}`.

use crate::dsv4_quant::{E2M1_TABLE, e4m3fn_to_f32};
use anyhow::{Result, anyhow};

/// Default `weight_block_size` from the released `config.json`.
pub const DEFAULT_BLOCK: usize = 32;

/// Decode a `float8_e8m0fnu` byte: `2^(b - 127)`, `255` = NaN.
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        return f32::NAN;
    }
    // exact for the whole range: 2^-127 is normal in f32, 2^127 is finite
    f32::from_bits(((b as u32) << 23).max(1))
}

/// How a `<stem>.scale` tensor maps onto its weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleLayout {
    /// One scale per `block × block` tile — the blocked FP8 GEMM layout.
    Tile { block: usize },
    /// One scale per row per `block` columns — FP4 experts and the Engram table.
    RowGroups { block: usize },
}

/// Storage format of the codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeFormat {
    /// One `e4m3fn` byte per element.
    Fp8E4m3,
    /// Two `e2m1` nibbles per byte along the column axis; the low nibble is the
    /// even column.
    Fp4E2m1,
}

/// A quantized tensor's decoded description: logical shape plus how to read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantPlan {
    pub rows: usize,
    /// Logical column count — twice the stored one for FP4.
    pub cols: usize,
    pub format: CodeFormat,
    pub layout: ScaleLayout,
}

/// Work out how to read `<stem>.weight` given its stored shape and its scale's.
///
/// `packed_fp4` is true when the weight arrives as `I8`/`F4` (the experts), in
/// which case the stored column count is half the logical one. Ambiguity is an
/// error rather than a guess: a wrong layout is silently wrong output, not a
/// crash.
pub fn plan(
    weight_shape: &[usize],
    packed_fp4: bool,
    scale_shape: &[usize],
    block: usize,
) -> Result<QuantPlan> {
    if weight_shape.len() != 2 || scale_shape.len() != 2 {
        return Err(anyhow!(
            "deepseek_v41 quant: expected rank-2 weight and scale, got {weight_shape:?} and {scale_shape:?}"
        ));
    }
    if block == 0 {
        return Err(anyhow!("deepseek_v41 quant: weight_block_size must be > 0"));
    }
    let rows = weight_shape[0];
    let cols = if packed_fp4 {
        weight_shape[1] * 2
    } else {
        weight_shape[1]
    };
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    if sc != cols.div_ceil(block) {
        return Err(anyhow!(
            "deepseek_v41 quant: scale has {sc} column groups for {cols} columns at block {block} (expected {})",
            cols.div_ceil(block)
        ));
    }
    let tiled = rows.div_ceil(block);
    let layout = match (sr == rows, sr == tiled) {
        // rows == ceil(rows/block) only when rows <= 1, where the two layouts
        // coincide anyway; prefer the row-wise reading there.
        (true, _) => ScaleLayout::RowGroups { block },
        (false, true) => ScaleLayout::Tile { block },
        (false, false) => {
            return Err(anyhow!(
                "deepseek_v41 quant: scale has {sr} row groups for {rows} rows at block {block}; \
                 expected {rows} (row-wise) or {tiled} (tiled)"
            ));
        }
    };
    let format = if packed_fp4 {
        CodeFormat::Fp4E2m1
    } else {
        CodeFormat::Fp8E4m3
    };
    Ok(QuantPlan {
        rows,
        cols,
        format,
        layout,
    })
}

/// Dequantize to F32 row-major `[rows, cols]`. `scales` are raw `e8m0` bytes.
pub fn dequantize(codes: &[u8], scales: &[u8], p: &QuantPlan) -> Result<Vec<f32>> {
    let want_codes = match p.format {
        CodeFormat::Fp8E4m3 => p.rows * p.cols,
        CodeFormat::Fp4E2m1 => p.rows * p.cols.div_ceil(2),
    };
    if codes.len() != want_codes {
        return Err(anyhow!(
            "deepseek_v41 quant: {} code bytes for a {}×{} {:?} tensor (expected {want_codes})",
            codes.len(),
            p.rows,
            p.cols,
            p.format
        ));
    }
    let (block, row_wise) = match p.layout {
        ScaleLayout::Tile { block } => (block, false),
        ScaleLayout::RowGroups { block } => (block, true),
    };
    let sc_cols = p.cols.div_ceil(block);
    let sc_rows = if row_wise { p.rows } else { p.rows.div_ceil(block) };
    if scales.len() != sc_rows * sc_cols {
        return Err(anyhow!(
            "deepseek_v41 quant: {} scale bytes, expected {}",
            scales.len(),
            sc_rows * sc_cols
        ));
    }
    let mut out = vec![0f32; p.rows * p.cols];
    let stride = match p.format {
        CodeFormat::Fp8E4m3 => p.cols,
        CodeFormat::Fp4E2m1 => p.cols.div_ceil(2),
    };
    for r in 0..p.rows {
        let sr = if row_wise { r } else { r / block };
        for c in 0..p.cols {
            let s = e8m0_to_f32(scales[sr * sc_cols + c / block]);
            let v = match p.format {
                CodeFormat::Fp8E4m3 => e4m3fn_to_f32(codes[r * stride + c]),
                CodeFormat::Fp4E2m1 => {
                    let byte = codes[r * stride + c / 2];
                    let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    E2M1_TABLE[nib as usize]
                }
            };
            out[r * p.cols + c] = v * s;
        }
    }
    Ok(out)
}

/// A [`WeightLoader`](crate::weight_loader::WeightLoader) over a released
/// DeepSeek-V4.1 safetensors tree that dequantizes on the way out.
///
/// Every `<stem>.weight` with a companion `<stem>.scale` is read through
/// [`plan`] + [`dequantize`]; everything else (BF16 norms, F32 `attn_sink` and
/// the Hyper-Connection parameters, the BF16 vision tower) is converted
/// verbatim. The builder therefore never has to know which tensors were
/// quantized.
///
/// Note this materializes F32, so a routed-expert bank costs
/// `n_experts · inter · dim · 4` bytes — fine for a pipeline stage over the
/// attention path, not for resident 384-expert MoE layers of the full model.
pub struct DsV41Loader {
    ck: crate::safetensors_checkpoint::SafetensorsCheckpoint,
    block: usize,
    taken: std::collections::HashSet<String>,
}

impl DsV41Loader {
    /// Open a checkpoint directory. `block` is `quantization_config.weight_block_size[0]`
    /// ([`DEFAULT_BLOCK`] for the released config).
    pub fn open(dir: &std::path::Path, block: usize) -> Result<Self> {
        Ok(Self {
            ck: crate::safetensors_checkpoint::SafetensorsCheckpoint::open(dir)?,
            block,
            taken: std::collections::HashSet::new(),
        })
    }

    /// Convert an unquantized tensor's raw bytes to F32.
    fn plain_to_f32(key: &str, bytes: &[u8], dt: safetensors::Dtype) -> Result<Vec<f32>> {
        use safetensors::Dtype as D;
        Ok(match dt {
            D::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            D::BF16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            D::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            D::F64 => bytes
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            D::I64 => bytes
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            D::I32 => bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            other => {
                return Err(anyhow!(
                    "deepseek_v41: {key} has dtype {other:?} and no companion `.scale` to \
                     interpret it with"
                ));
            }
        })
    }

    fn load(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (bytes, dt, shape) = self.ck.tensor_raw(key)?;
        self.taken.insert(key.to_string());
        let scale_key = key.strip_suffix(".weight").map(|s| format!("{s}.scale"));
        let Some(sk) = scale_key.filter(|k| self.ck.contains(k)) else {
            return Ok((Self::plain_to_f32(key, &bytes, dt)?, shape));
        };
        let (sbytes, sdt, sshape) = self.ck.tensor_raw(&sk)?;
        if sdt != safetensors::Dtype::F8_E8M0 {
            return Err(anyhow!(
                "deepseek_v41: {sk} is {sdt:?}, expected F8_E8M0 (`scale_fmt: ue8m0`)"
            ));
        }
        // I8 / F4 codes are FP4 nibble pairs; F8_E4M3 codes are one per element.
        let packed_fp4 = matches!(dt, safetensors::Dtype::I8 | safetensors::Dtype::F4);
        if !packed_fp4 && dt != safetensors::Dtype::F8_E4M3 {
            return Err(anyhow!(
                "deepseek_v41: {key} has a `.scale` but dtype {dt:?} is neither FP8 nor packed FP4"
            ));
        }
        let p = plan(&shape, packed_fp4, &sshape, self.block)?;
        self.taken.insert(sk);
        let data = dequantize(&bytes, &sbytes, &p)?;
        Ok((data, vec![p.rows, p.cols]))
    }
}

impl crate::weight_loader::WeightLoader for DsV41Loader {
    fn format_id(&self) -> &'static str {
        "deepseek-v41"
    }

    fn len(&self) -> usize {
        self.ck.keys().count()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.load(key)
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.load(key)?;
        if s.len() != 2 {
            return Ok((d, s));
        }
        let (r, c) = (s[0], s[1]);
        let mut o = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = d[i * c + j];
            }
        }
        Ok((o, vec![c, r]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.ck
            .keys()
            .filter(|k| !self.taken.contains(*k))
            .map(str::to_string)
            .collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_is_a_plain_power_of_two() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(137), 1024.0);
        assert_eq!(e8m0_to_f32(117), 1.0 / 1024.0);
        assert!(e8m0_to_f32(255).is_nan());
        // the extremes stay finite and non-zero
        assert!(e8m0_to_f32(0) > 0.0 && e8m0_to_f32(0).is_finite());
        assert!(e8m0_to_f32(254).is_finite());
    }

    /// The real checkpoint's `(weight dtype, weight shape, scale shape)` triples,
    /// read from the safetensors headers of
    /// `deepseek-ai/DeepSeek-V4.1-Flash`. This is what pins the three layouts
    /// apart — in particular the Engram table, whose scale has one row per table
    /// row rather than per 32.
    #[test]
    fn layouts_match_the_released_checkpoint() {
        // (name, packed_fp4, weight shape, scale shape, expected layout)
        let cases: &[(&str, bool, [usize; 2], [usize; 2], ScaleLayout)] = &[
            ("attn.wq_a", false, [1280, 5120], [40, 160], ScaleLayout::Tile { block: 32 }),
            ("attn.wq_b", false, [32768, 1280], [1024, 40], ScaleLayout::Tile { block: 32 }),
            ("attn.wkv", false, [512, 5120], [16, 160], ScaleLayout::Tile { block: 32 }),
            ("attn.wo_a", false, [8192, 4096], [256, 128], ScaleLayout::Tile { block: 32 }),
            ("attn.wo_b", false, [5120, 8192], [160, 256], ScaleLayout::Tile { block: 32 }),
            ("shared_experts.w1", false, [2304, 5120], [72, 160], ScaleLayout::Tile { block: 32 }),
            ("shared_experts.w2", false, [5120, 2304], [160, 72], ScaleLayout::Tile { block: 32 }),
            ("indexer.wq_b", false, [4096, 1280], [128, 40], ScaleLayout::Tile { block: 32 }),
            ("engram.wkv", false, [25600, 6144], [800, 192], ScaleLayout::Tile { block: 32 }),
            ("mtp.main_proj", false, [5120, 15360], [160, 480], ScaleLayout::Tile { block: 32 }),
            // FP4 experts: stored columns are halved
            ("experts.w1", true, [2304, 2560], [2304, 160], ScaleLayout::RowGroups { block: 32 }),
            ("experts.w2", true, [5120, 1152], [5120, 72], ScaleLayout::RowGroups { block: 32 }),
            // the Engram table: FP8 codes, row-wise scales
            (
                "engram.embed",
                false,
                [384006168, 256],
                [384006168, 8],
                ScaleLayout::RowGroups { block: 32 },
            ),
        ];
        for (name, fp4, w, s, want) in cases {
            let p = plan(w, *fp4, s, DEFAULT_BLOCK).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(p.layout, *want, "{name} layout");
            assert_eq!(p.rows, w[0], "{name} rows");
            assert_eq!(p.cols, if *fp4 { w[1] * 2 } else { w[1] }, "{name} cols");
            assert_eq!(
                p.format,
                if *fp4 {
                    CodeFormat::Fp4E2m1
                } else {
                    CodeFormat::Fp8E4m3
                },
                "{name} format"
            );
        }
    }

    #[test]
    fn plan_rejects_a_scale_that_fits_neither_layout() {
        let err = plan(&[1000, 256], false, &[17, 8], 32).unwrap_err().to_string();
        assert!(err.contains("17 row groups"), "{err}");
        let err = plan(&[1024, 256], false, &[32, 9], 32).unwrap_err().to_string();
        assert!(err.contains("9 column groups"), "{err}");
    }

    #[test]
    fn tiled_fp8_applies_one_scale_per_tile() {
        // 3×5 at block 2 → scales [2, 3]
        let p = plan(&[3, 5], false, &[2, 3], 2).unwrap();
        assert_eq!(p.layout, ScaleLayout::Tile { block: 2 });
        let one = 0x38u8; // e4m3 1.0
        let codes = vec![one; 15];
        // scale tile (i, j) = 2^j, distinct per column group, same for both row groups
        let scales: Vec<u8> = (0..2).flat_map(|_| [127u8, 128, 129]).collect();
        let out = dequantize(&codes, &scales, &p).unwrap();
        for r in 0..3 {
            for c in 0..5 {
                let want = 2f32.powi(c as i32 / 2);
                assert_eq!(out[r * 5 + c], want, "r{r} c{c}");
            }
        }
    }

    #[test]
    fn row_wise_fp8_gives_every_row_its_own_scale() {
        // the Engram shape in miniature: 4 rows × 4 cols, block 2 → scales [4, 2]
        let p = plan(&[4, 4], false, &[4, 2], 2).unwrap();
        assert_eq!(p.layout, ScaleLayout::RowGroups { block: 2 });
        let codes = vec![0x38u8; 16]; // all 1.0
        let scales: Vec<u8> = (0..4).flat_map(|r| [127u8 + r as u8, 127]).collect();
        let out = dequantize(&codes, &scales, &p).unwrap();
        for r in 0..4 {
            assert_eq!(out[r * 4], 2f32.powi(r as i32), "row {r} group 0");
            assert_eq!(out[r * 4 + 1], 2f32.powi(r as i32), "row {r} group 0");
            assert_eq!(out[r * 4 + 2], 1.0, "row {r} group 1");
        }
        // Reading it as tiled would share one scale across each pair of rows —
        // the mistake this layout split exists to prevent. Row 1 then picks up
        // row 0's scale instead of its own.
        let tiled = QuantPlan {
            layout: ScaleLayout::Tile { block: 2 },
            ..p
        };
        let wrong = dequantize(&codes, &[127u8, 127, 129, 127], &tiled).unwrap();
        assert_eq!(wrong[4], 1.0);
        assert_eq!(out[4], 2.0);
    }

    #[test]
    fn fp4_nibbles_unpack_low_then_high() {
        // 1 row, 4 logical columns → 2 stored bytes; block 4 → scales [1, 1]
        let p = plan(&[1, 2], true, &[1, 1], 4).unwrap();
        assert_eq!(p.format, CodeFormat::Fp4E2m1);
        assert_eq!(p.cols, 4);
        // E2M1 table: 0->0.0, 2->1.0, 4->2.0, 0xC->-2.0
        let codes = vec![0x20u8, 0xC4];
        let out = dequantize(&codes, &[128u8], &p).unwrap(); // scale 2
        assert_eq!(out, vec![0.0, 2.0, 4.0, -4.0]);
    }

    #[test]
    fn dequantize_rejects_a_short_blob() {
        let p = plan(&[4, 4], false, &[4, 2], 2).unwrap();
        let err = dequantize(&[0u8; 15], &[127u8; 8], &p)
            .unwrap_err()
            .to_string();
        assert!(err.contains("15 code bytes"), "{err}");
    }
}
