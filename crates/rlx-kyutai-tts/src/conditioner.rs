//! Conditioner kernels for Kyutai TTS.
//!
//! Three families, all driven by the `conditioners` map in [`crate::config`]:
//!
//! | Kind | Source | Produces | Used by fuser as |
//! |------|--------|----------|------------------|
//! | [`LutConditioner`] | discrete bin id | `[dim]` lookup row | `sum` (CFG / control) |
//! | [`TensorConditioner`] | pre-computed `[dim]` embedding | proj→`[d_model]` | `cross` (speaker_wavs) |
//!
//! The native eager forms below are pure ndarray; they consume already-loaded
//! weights (no candle / safetensors dependency at this layer).

use crate::nn::{Embedding, linear};
use anyhow::{Result, bail};
use ndarray::{Array1, Array2};

/// Lookup-table conditioner — discrete bin id → `[dim]` embedding.
///
/// Used for `cfg` (7 bins × 16-D) and `control` (1 bin × 2048-D) in the
/// published Kyutai TTS 1.6B config.
#[derive(Debug, Clone)]
pub struct LutConditioner {
    /// `[n_bins, dim]` embedding table.
    pub table: Embedding,
    /// Possible bin string values, in the order they appear in `config.json`.
    pub possible_values: Vec<String>,
}

impl LutConditioner {
    /// Total bins.
    pub fn n_bins(&self) -> usize {
        self.table.weight.nrows()
    }

    /// Embedding dim.
    pub fn dim(&self) -> usize {
        self.table.weight.ncols()
    }

    /// Resolve a `possible_values` string (case-sensitive) to its bin id.
    pub fn bin_for(&self, value: &str) -> Result<u32> {
        self.possible_values
            .iter()
            .position(|s| s == value)
            .map(|p| p as u32)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no LUT bin for {value:?} (valid: {:?})",
                    self.possible_values
                )
            })
    }

    /// Forward with a literal bin id.
    pub fn forward_id(&self, id: u32) -> Result<Array1<f32>> {
        if (id as usize) >= self.n_bins() {
            bail!("LUT id {id} out of range [0, {})", self.n_bins());
        }
        Ok(self.table.forward_one(id))
    }

    /// Forward by `possible_values` string.
    pub fn forward_value(&self, value: &str) -> Result<Array1<f32>> {
        let id = self.bin_for(value)?;
        self.forward_id(id)
    }
}

/// Tensor (pre-computed embedding) conditioner.
///
/// The Kyutai TTS speaker conditioner reads a 512-D vector from `kyutai/tts-voices`
/// and projects it via an optional learned `output_proj` to whatever the
/// cross-attention expects. The projection is bias-free linear; if absent
/// the raw input is returned unchanged.
#[derive(Debug, Clone)]
pub struct TensorConditioner {
    /// Source embedding dim (e.g. 512 for `speaker_wavs`).
    pub input_dim: usize,
    /// Optional output projection `[input_dim, out_dim]`.
    pub output_proj: Option<Array2<f32>>,
}

impl TensorConditioner {
    /// Output dim after optional projection.
    pub fn output_dim(&self) -> usize {
        match &self.output_proj {
            Some(w) => w.nrows(), // linear() does x @ w^T → out = w.rows()
            None => self.input_dim,
        }
    }

    /// Forward a `[input_dim]` voice embedding.
    pub fn forward(&self, embedding: &Array1<f32>) -> Result<Array1<f32>> {
        if embedding.len() != self.input_dim {
            bail!(
                "tensor conditioner expects {} dims, got {}",
                self.input_dim,
                embedding.len()
            );
        }
        match &self.output_proj {
            None => Ok(embedding.clone()),
            Some(w) => {
                let x = embedding.view().insert_axis(ndarray::Axis(0));
                let y = linear(x, w);
                Ok(y.row(0).to_owned())
            }
        }
    }

    /// Forward a `[t, input_dim]` sequence of voice frames → `[t, output_dim]`.
    pub fn forward_seq(&self, frames: &Array2<f32>) -> Result<Array2<f32>> {
        if frames.ncols() != self.input_dim {
            bail!(
                "tensor conditioner expects {} dims/frame, got {}",
                self.input_dim,
                frames.ncols()
            );
        }
        match &self.output_proj {
            None => Ok(frames.clone()),
            Some(w) => Ok(linear(frames.view(), w)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn lut_fixture() -> LutConditioner {
        // 3 bins × 4 dim.
        let table = Embedding {
            weight: array![
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0]
            ],
        };
        LutConditioner {
            table,
            possible_values: vec!["lo".into(), "mid".into(), "hi".into()],
        }
    }

    #[test]
    fn lut_bin_for_resolves_value() {
        let lut = lut_fixture();
        assert_eq!(lut.bin_for("mid").unwrap(), 1);
        assert!(lut.bin_for("nope").is_err());
    }

    #[test]
    fn lut_forward_value_returns_correct_row() {
        let lut = lut_fixture();
        let v = lut.forward_value("hi").unwrap();
        assert_eq!(v, array![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn lut_forward_id_bounds_checked() {
        let lut = lut_fixture();
        assert!(lut.forward_id(2).is_ok());
        assert!(lut.forward_id(3).is_err());
    }

    #[test]
    fn tensor_conditioner_identity_when_no_proj() {
        let tc = TensorConditioner {
            input_dim: 3,
            output_proj: None,
        };
        let x = array![1.0, 2.0, 3.0];
        let y = tc.forward(&x).unwrap();
        assert_eq!(y, x);
        assert_eq!(tc.output_dim(), 3);
    }

    #[test]
    fn tensor_conditioner_projects_when_proj_set() {
        // 2-D in → 4-D out. linear does x @ w^T, so weights are [out, in].
        let w = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]];
        let tc = TensorConditioner {
            input_dim: 2,
            output_proj: Some(w),
        };
        assert_eq!(tc.output_dim(), 4);
        let x = array![3.0, 4.0];
        let y = tc.forward(&x).unwrap();
        // w[0]·[3,4] = 3; w[1]·x = 4; w[2]·x = 7; w[3]·x = 1
        assert_eq!(y, array![3.0, 4.0, 7.0, 1.0]);
    }

    #[test]
    fn tensor_conditioner_rejects_wrong_input_dim() {
        let tc = TensorConditioner {
            input_dim: 4,
            output_proj: None,
        };
        let bad = array![1.0, 2.0];
        assert!(tc.forward(&bad).is_err());
    }
}
