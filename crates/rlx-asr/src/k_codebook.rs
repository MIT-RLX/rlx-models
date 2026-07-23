//! TEXT `linear_k` codebook — row-scale int8 dequant.
//!
//! Packing for layers L ∈ {0,7,14,21}:
//! ```text
//! linear_k int8[512,512] + scale[512]
//! W[i,j] = int8[i,j] * scale[i]                   // row-scale
//! ```
//! Affine group: `G = 7 + 8*(L/7)` → {7,15,23,31}.
//!
//! Weights under `weights/asr/codebook/by_layer/LXX/` (see [`crate::AsrPaths`]).

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::spec::DECODER_DIM;
use crate::weights::read_f32_bin;

/// Layers that ship named TEXT `linear_k` in the codebook pack.
pub const TEXT_K_LAYERS: [usize; 4] = [0, 7, 14, 21];

/// Affine quantized group id bound to TEXT K (SYMTAB).
pub fn affine_group(layer: usize) -> usize {
    7 + 8 * (layer / 7)
}

/// One layer's codebook: int8 weight + GOC row-scale (+ optional materialized W).
#[derive(Clone, Debug)]
pub struct TextKLayer {
    pub layer: usize,
    pub int8: Vec<i8>,
    pub scale: Vec<f32>,
    pub ones: Vec<f32>,
    /// Row-major f32 `[out,in]` = int8 * scale[row].
    pub weight: Vec<f32>,
}

impl TextKLayer {
    pub fn dequant(int8: &[i8], scale: &[f32]) -> Result<Vec<f32>> {
        if int8.len() != DECODER_DIM * DECODER_DIM {
            bail!("int8 len {} != {}", int8.len(), DECODER_DIM * DECODER_DIM);
        }
        if scale.len() != DECODER_DIM {
            bail!("scale len {} != {}", scale.len(), DECODER_DIM);
        }
        let mut w = vec![0f32; DECODER_DIM * DECODER_DIM];
        for i in 0..DECODER_DIM {
            let s = scale[i];
            let row = &int8[i * DECODER_DIM..(i + 1) * DECODER_DIM];
            let out = &mut w[i * DECODER_DIM..(i + 1) * DECODER_DIM];
            for j in 0..DECODER_DIM {
                out[j] = (row[j] as f32) * s;
            }
        }
        Ok(w)
    }

    /// `y = W @ x` with W row-major, without materializing f32 W:
    /// `y[i] = scale[i] * Σ_j int8[i,j] * x[j]` (GOC row-scale int8 matmul).
    pub fn matvec_goc(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != DECODER_DIM || y.len() != DECODER_DIM {
            bail!("matvec_goc: expected dim {DECODER_DIM}");
        }
        for i in 0..DECODER_DIM {
            let mut acc = 0.0f32;
            let row = &self.int8[i * DECODER_DIM..(i + 1) * DECODER_DIM];
            for j in 0..DECODER_DIM {
                acc += (row[j] as f32) * x[j];
            }
            y[i] = self.scale[i] * acc;
        }
        Ok(())
    }

    pub fn ones_all_one(&self) -> bool {
        self.ones.iter().all(|&v| (v - 1.0).abs() < 1e-5)
    }
}

/// Full TEXT K codebook pack (four layers).
#[derive(Clone, Debug)]
pub struct KCodebook {
    pub root: PathBuf,
    pub layers: HashMap<usize, TextKLayer>,
}

impl KCodebook {
    /// Load from `k_codebook/` (or `k_codebook/native/`) export directory.
    ///
    /// Accepts either:
    /// - `by_layer/LXX/{linear_k_int8.bin,linear_k_scale_fp32.bin}`
    /// - or the same under `native/by_layer/` when `dir` points at a native export
    pub fn load(dir: &Path) -> Result<Self> {
        let by = if dir.join("by_layer").is_dir() {
            dir.join("by_layer")
        } else {
            bail!("{}: missing by_layer/", dir.display());
        };
        let mut layers = HashMap::new();
        for &layer in &TEXT_K_LAYERS {
            let ld = by.join(format!("L{layer:02}"));
            if !ld.is_dir() {
                bail!("missing {}", ld.display());
            }
            let int8 = read_i8_bin(&ld.join("linear_k_int8.bin"))
                .or_else(|_| read_npy_i8(&ld.join("linear_k_int8.npy")))?;
            if int8.len() != DECODER_DIM * DECODER_DIM {
                bail!("L{layer}: int8 len {}", int8.len());
            }
            let scale = read_f32_bin(&ld.join("linear_k_scale_fp32.bin"))
                .or_else(|_| read_npy_f32(&ld.join("linear_k_scale_fp32.npy")))?;
            if scale.len() != DECODER_DIM {
                bail!("L{layer}: scale len {}", scale.len());
            }
            let ones = vec![1.0f32; DECODER_DIM];
            let weight = TextKLayer::dequant(&int8, &scale)?;
            let _ = fs::read_to_string(ld.join("k_codebook_meta.json"));
            layers.insert(
                layer,
                TextKLayer {
                    layer,
                    int8,
                    scale,
                    ones,
                    weight,
                },
            );
        }
        if layers.len() != TEXT_K_LAYERS.len() {
            bail!("expected 4 TEXT K layers, got {}", layers.len());
        }
        Ok(Self {
            root: dir.to_path_buf(),
            layers,
        })
    }

    /// Resolve under `RLX_ASR_DIR` / `.cache/asr` defaults.
    /// Resolve under [`crate::AsrPaths`] (`codebook/`, with legacy fallbacks).
    pub fn load_default(asr_dir: &Path) -> Result<Self> {
        let paths = crate::AsrPaths::new(asr_dir);
        let candidates = [paths.codebook_dir()];
        for c in &candidates {
            if c.join("by_layer").is_dir() {
                return Self::load(c);
            }
        }
        bail!(
            "TEXT K codebook not found under {} (expected codebook/by_layer)",
            asr_dir.display()
        );
    }

    pub fn get(&self, layer: usize) -> Option<&TextKLayer> {
        self.layers.get(&layer)
    }

    pub fn weight(&self, layer: usize) -> Option<&[f32]> {
        self.layers.get(&layer).map(|l| l.weight.as_slice())
    }

    /// Build f32 `[512,512]` map suitable for `NativeMhaWeights.wk` injection.
    pub fn weights_by_layer(&self) -> HashMap<usize, Vec<f32>> {
        self.layers
            .iter()
            .map(|(&layer, layer_w)| (layer, layer_w.weight.clone()))
            .collect()
    }
}

fn read_i8_bin(path: &Path) -> Result<Vec<i8>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes.iter().map(|&b| b as i8).collect())
}

fn read_npy_i8(path: &Path) -> Result<Vec<i8>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let data = npy_payload(&bytes)?;
    Ok(data.iter().map(|&b| b as i8).collect())
}

fn read_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let data = npy_payload(&bytes)?;
    if data.len() % 4 != 0 {
        bail!("{}: f32 npy size", path.display());
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn npy_payload(bytes: &[u8]) -> Result<&[u8]> {
    if bytes.len() < 128 || &bytes[0..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let ver_minor = bytes[7];
    let (header_len, header_end) = if ver_minor == 0 {
        let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (hl, 10 + hl)
    } else {
        let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (hl, 12 + hl)
    };
    let _ = header_len;
    if bytes.len() < header_end {
        bail!("short npy header");
    }
    Ok(&bytes[header_end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_group_formula() {
        assert_eq!(affine_group(0), 7);
        assert_eq!(affine_group(7), 15);
        assert_eq!(affine_group(14), 23);
        assert_eq!(affine_group(21), 31);
    }

    #[test]
    fn load_k_codebook_if_present() {
        let p = crate::AsrPaths::resolve().codebook_dir();
        if !p.join("by_layer/L00/linear_k_int8.bin").exists() {
            return;
        }
        let cb = KCodebook::load(&p).expect("load k codebook");
        assert_eq!(cb.layers.len(), 4);
        let l0 = cb.get(0).unwrap();
        assert!(l0.ones_all_one());
        assert_eq!(l0.weight.len(), 512 * 512);
        // GOC matvec ≡ dense matvec
        let x = vec![0.01f32; 512];
        let mut y_goc = vec![0f32; 512];
        l0.matvec_goc(&x, &mut y_goc).unwrap();
        let mut y_dense = vec![0f32; 512];
        for i in 0..512 {
            let mut s = 0.0f32;
            let row = &l0.weight[i * 512..(i + 1) * 512];
            for j in 0..512 {
                s += row[j] * x[j];
            }
            y_dense[i] = s;
        }
        let max_err = y_goc
            .iter()
            .zip(y_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "goc vs dense max_err={max_err}");
        // Re-encode: round(W/scale) ≈ int8
        for i in 0..512 {
            let s = l0.scale[i];
            if s.abs() < 1e-8 {
                continue;
            }
            for j in 0..512 {
                let q = (l0.weight[i * 512 + j] / s).round() as i32;
                assert_eq!(q as i8, l0.int8[i * 512 + j]);
            }
        }
    }
}
