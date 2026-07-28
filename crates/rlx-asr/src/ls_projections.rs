//! LS attention V projections for TEXT-K layers.
//!
//! `att_v` is `[28,64,8,16]` ⇒ `d_v=16`, `Wv` shape `[128,512]`.
//! Head-padded `[512,512]` embeds each head's 16-d V into the first 16 of the
//! 64-wide head slot (zeros elsewhere) so existing `[out,in]=512` matvecs work.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::k_codebook::TEXT_K_LAYERS;
use crate::spec::{DECODER_DIM, DECODER_HEAD_DIM, DECODER_HEADS};
use crate::weights::read_f32_bin;

/// `d_v` per head in the teacher att_v cache.
pub const ATT_V_HEAD_DIM: usize = 16;
/// `n_heads * d_v` = 128.
pub const ATT_V_OUT: usize = DECODER_HEADS * ATT_V_HEAD_DIM;

#[derive(Clone, Debug)]
pub struct LsVLayer {
    pub layer: usize,
    /// Row-major `[128, 512]`.
    pub weight_128: Vec<f32>,
    /// Row-major `[512, 512]` head-padded drop-in.
    pub weight_pad: Vec<f32>,
    /// Optional int8 `[128*512]` + row scale `[128]`.
    pub int8: Option<Vec<i8>>,
    pub scale: Option<Vec<f32>>,
}

impl LsVLayer {
    /// `y[128] = W @ x[512]` with W row-major `[128,512]`.
    pub fn matvec_128(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        if let (Some(int8), Some(scale)) = (self.int8.as_ref(), self.scale.as_ref()) {
            return self.matvec_goc_128(x, y, int8, scale);
        }
        if x.len() != DECODER_DIM || y.len() != ATT_V_OUT {
            bail!(
                "matvec_128: expected in={DECODER_DIM} out={ATT_V_OUT}, got {}/{}",
                x.len(),
                y.len()
            );
        }
        for i in 0..ATT_V_OUT {
            let mut s = 0.0f32;
            let row = &self.weight_128[i * DECODER_DIM..(i + 1) * DECODER_DIM];
            for j in 0..DECODER_DIM {
                s += row[j] * x[j];
            }
            y[i] = s;
        }
        Ok(())
    }

    /// GOC-style: `y[i] = scale[i] * Σ_j int8[i,j] * x[j]`.
    pub fn matvec_goc_128(
        &self,
        x: &[f32],
        y: &mut [f32],
        int8: &[i8],
        scale: &[f32],
    ) -> Result<()> {
        if x.len() != DECODER_DIM || y.len() != ATT_V_OUT {
            bail!("matvec_goc_128: bad lengths");
        }
        if int8.len() != ATT_V_OUT * DECODER_DIM || scale.len() != ATT_V_OUT {
            bail!("matvec_goc_128: bad int8/scale");
        }
        for i in 0..ATT_V_OUT {
            let mut acc = 0.0f32;
            let row = &int8[i * DECODER_DIM..(i + 1) * DECODER_DIM];
            for j in 0..DECODER_DIM {
                acc += (row[j] as f32) * x[j];
            }
            y[i] = scale[i] * acc;
        }
        Ok(())
    }

    /// Expand `[128]` head-packed V into `[512]` with zeros in unused head slots.
    pub fn expand_to_dim512(v128: &[f32], out512: &mut [f32]) -> Result<()> {
        if v128.len() != ATT_V_OUT || out512.len() != DECODER_DIM {
            bail!("expand_to_dim512: bad lengths");
        }
        out512.fill(0.0);
        for h in 0..DECODER_HEADS {
            let src = &v128[h * ATT_V_HEAD_DIM..(h + 1) * ATT_V_HEAD_DIM];
            let dst = &mut out512[h * DECODER_HEAD_DIM..h * DECODER_HEAD_DIM + ATT_V_HEAD_DIM];
            dst.copy_from_slice(src);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LsProjections {
    pub root: PathBuf,
    pub v_layers: HashMap<usize, LsVLayer>,
}

impl LsProjections {
    pub fn load(dir: &Path) -> Result<Self> {
        let by = if dir.join("by_layer").is_dir() {
            dir.join("by_layer")
        } else {
            bail!("{}: missing by_layer/", dir.display());
        };
        let mut v_layers = HashMap::new();
        for &layer in &TEXT_K_LAYERS {
            let ld = by.join(format!("L{layer:02}"));
            if !ld.is_dir() {
                continue;
            }
            let w128 = read_f32_bin(&ld.join("linear_v_ls_128x512.bin"))
                .with_context(|| format!("L{layer} linear_v_ls_128x512.bin"))?;
            if w128.len() != ATT_V_OUT * DECODER_DIM {
                bail!(
                    "L{layer} V len {} != {}",
                    w128.len(),
                    ATT_V_OUT * DECODER_DIM
                );
            }
            let wpad = match read_f32_bin(&ld.join("linear_v_ls_512x512_headpad.bin")) {
                Ok(w) if w.len() == DECODER_DIM * DECODER_DIM => w,
                _ => headpad_512(&w128),
            };
            let (int8, scale) = load_goc_optional(&ld, &w128);
            v_layers.insert(
                layer,
                LsVLayer {
                    layer,
                    weight_128: w128,
                    weight_pad: wpad,
                    int8,
                    scale,
                },
            );
        }
        if v_layers.is_empty() {
            bail!("{}: no TEXT-K V layers found", dir.display());
        }
        Ok(Self {
            root: dir.to_path_buf(),
            v_layers,
        })
    }

    pub fn load_default(asr_dir: &Path) -> Result<Self> {
        let p = crate::AsrPaths::new(asr_dir).ls_dir();
        Self::load(&p)
    }

    pub fn get_v(&self, layer: usize) -> Option<&LsVLayer> {
        self.v_layers.get(&layer)
    }
}

fn headpad_512(w128: &[f32]) -> Vec<f32> {
    let mut m = vec![0f32; DECODER_DIM * DECODER_DIM];
    for h in 0..DECODER_HEADS {
        for r in 0..ATT_V_HEAD_DIM {
            let src = &w128[(h * ATT_V_HEAD_DIM + r) * DECODER_DIM
                ..(h * ATT_V_HEAD_DIM + r + 1) * DECODER_DIM];
            let dst_row = h * DECODER_HEAD_DIM + r;
            m[dst_row * DECODER_DIM..(dst_row + 1) * DECODER_DIM].copy_from_slice(src);
        }
    }
    m
}

fn goc_quantize(w128: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut int8 = vec![0i8; ATT_V_OUT * DECODER_DIM];
    let mut scale = vec![0f32; ATT_V_OUT];
    for i in 0..ATT_V_OUT {
        let row = &w128[i * DECODER_DIM..(i + 1) * DECODER_DIM];
        let mut m = 0.0f32;
        for &v in row {
            m = m.max(v.abs());
        }
        let s = (m / 127.0).max(1e-12);
        scale[i] = s;
        let out = &mut int8[i * DECODER_DIM..(i + 1) * DECODER_DIM];
        for j in 0..DECODER_DIM {
            let q = (row[j] / s).round().clamp(-127.0, 127.0) as i8;
            out[j] = q;
        }
    }
    (int8, scale)
}

fn load_goc_optional(ld: &Path, w128: &[f32]) -> (Option<Vec<i8>>, Option<Vec<f32>>) {
    let i8p = ld.join("linear_v_ls_int8.bin");
    let sp = ld.join("linear_v_ls_scale_fp32.bin");
    if i8p.is_file() && sp.is_file() {
        let Ok(bytes) = fs::read(&i8p) else {
            return (None, None);
        };
        if bytes.len() != ATT_V_OUT * DECODER_DIM {
            return (None, None);
        }
        let int8: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        let Ok(scale) = read_f32_bin(&sp) else {
            return (None, None);
        };
        if scale.len() != ATT_V_OUT {
            return (None, None);
        }
        return (Some(int8), Some(scale));
    }
    // Synthesize GOC from f32 when int8+scale pack is absent.
    let (int8, scale) = goc_quantize(w128);
    (Some(int8), Some(scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goc_matches_f32_matvec() {
        let mut w = vec![0f32; ATT_V_OUT * DECODER_DIM];
        for i in 0..w.len() {
            w[i] = ((i % 97) as f32) * 0.01 - 0.4;
        }
        let (int8, scale) = goc_quantize(&w);
        let layer = LsVLayer {
            layer: 0,
            weight_128: w.clone(),
            weight_pad: vec![],
            int8: Some(int8),
            scale: Some(scale),
        };
        let x: Vec<f32> = (0..DECODER_DIM).map(|i| (i as f32) * 0.001).collect();
        let mut y_goc = vec![0f32; ATT_V_OUT];
        let mut y_f32 = vec![0f32; ATT_V_OUT];
        layer.matvec_128(&x, &mut y_goc).unwrap();
        // force f32 path
        let layer_f = LsVLayer {
            int8: None,
            scale: None,
            ..layer
        };
        layer_f.matvec_128(&x, &mut y_f32).unwrap();
        let mut max_err = 0.0f32;
        for i in 0..ATT_V_OUT {
            max_err = max_err.max((y_goc[i] - y_f32[i]).abs());
        }
        assert!(max_err < 0.05, "goc vs f32 max_err={max_err}");
    }

    #[test]
    fn expand_roundtrip_layout() {
        let mut v128 = vec![0f32; ATT_V_OUT];
        for i in 0..ATT_V_OUT {
            v128[i] = i as f32;
        }
        let mut out = vec![0f32; DECODER_DIM];
        LsVLayer::expand_to_dim512(&v128, &mut out).unwrap();
        for h in 0..DECODER_HEADS {
            for d in 0..ATT_V_HEAD_DIM {
                assert_eq!(
                    out[h * DECODER_HEAD_DIM + d],
                    (h * ATT_V_HEAD_DIM + d) as f32
                );
            }
            for d in ATT_V_HEAD_DIM..DECODER_HEAD_DIM {
                assert_eq!(out[h * DECODER_HEAD_DIM + d], 0.0);
            }
        }
    }

    #[test]
    fn load_optional() {
        let p = crate::AsrPaths::resolve().ls_dir();
        if !p.join("by_layer/L00/linear_v_ls_128x512.bin").exists() {
            return;
        }
        let ls = LsProjections::load(&p).expect("load ls projections");
        assert!(ls.get_v(0).is_some());
        let v = ls.get_v(0).unwrap();
        assert_eq!(v.weight_128.len(), ATT_V_OUT * DECODER_DIM);
        assert_eq!(v.weight_pad.len(), DECODER_DIM * DECODER_DIM);
    }
}
