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

//! BiRefNet checkpoint access: every float tensor as f32, with eval-mode
//! BatchNorm folded into the convolution it follows.

use anyhow::{Context, Result, ensure};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use std::collections::HashMap;
use std::path::Path;

/// `name → (data, shape)`.
#[derive(Debug, Default)]
pub struct Weights {
    pub t: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

const BN_EPS: f32 = 1e-5;

impl Weights {
    /// Load `model.safetensors` (a file or the directory holding it).
    pub fn load(path: &Path) -> Result<Self> {
        let file = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_path_buf()
        };
        let ck = SafetensorsCheckpoint::open_file(&file)
            .with_context(|| format!("opening {}", file.display()))?;
        let mut t = HashMap::new();
        for k in ck.keys() {
            // Integer buffers (Swin's relative_position_index) are recomputed, not read.
            if matches!(
                ck.stored_dtype(k),
                Some(safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16)
            ) {
                t.insert(k.to_string(), ck.load_tensor_f32(k)?);
            }
        }
        Ok(Self { t })
    }

    pub fn has(&self, k: &str) -> bool {
        self.t.contains_key(k)
    }

    pub fn get(&self, k: &str) -> Result<&(Vec<f32>, Vec<usize>)> {
        self.t
            .get(k)
            .with_context(|| format!("BiRefNet checkpoint lacks `{k}`"))
    }

    /// Conv kernel `[O, I/g, kh, kw]` + bias `[O]` for `conv` (`<conv>.weight`,
    /// optional `<conv>.bias`), with BatchNorm `bn` (`<bn>.{weight,bias,
    /// running_mean,running_var}`) folded in when present.
    pub fn conv(&self, conv: &str, bn: Option<&str>) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>)> {
        let (w, shape) = self.get(&format!("{conv}.weight"))?.clone();
        let o = shape[0];
        let mut b = match self.t.get(&format!("{conv}.bias")) {
            Some((b, _)) => b.clone(),
            None => vec![0.0; o],
        };
        let mut w = w;
        if let Some(bn) = bn.filter(|bn| self.has(&format!("{bn}.running_mean"))) {
            let g = &self.get(&format!("{bn}.weight"))?.0;
            let be = &self.get(&format!("{bn}.bias"))?.0;
            let mu = &self.get(&format!("{bn}.running_mean"))?.0;
            let var = &self.get(&format!("{bn}.running_var"))?.0;
            ensure!(
                g.len() == o,
                "{bn}: {} channels for a {o}-channel conv",
                g.len()
            );
            let per = w.len() / o;
            for c in 0..o {
                let s = g[c] / (var[c] + BN_EPS).sqrt();
                w[c * per..(c + 1) * per].iter_mut().for_each(|v| *v *= s);
                b[c] = (b[c] - mu[c]) * s + be[c];
            }
        }
        Ok((w, shape, b))
    }
}
