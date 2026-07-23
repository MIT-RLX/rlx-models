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

//! Linear projector `2048 → 1280` + learned image newline / view separator.

use crate::config::ProjectorConfig;
use crate::weights::{IMAGE_NEWLINE, UnlimitedOcrWeightStore, VIEW_SEPARATOR};
use anyhow::{Context, Result, ensure};

#[derive(Debug, Clone, Default)]
pub struct Projector {
    pub in_features: usize,
    pub out_features: usize,
    weight: Option<Vec<f32>>, // [out, in]
    bias: Option<Vec<f32>>,
    pub image_newline: Option<Vec<f32>>,
    pub view_separator: Option<Vec<f32>>,
}

impl Projector {
    pub fn from_config(cfg: &ProjectorConfig) -> Self {
        Self {
            in_features: cfg.input_dim,
            out_features: cfg.n_embed,
            ..Default::default()
        }
    }

    pub fn load(&mut self, store: &UnlimitedOcrWeightStore) -> Result<()> {
        let keys = store.keys();
        let w_key = [
            "model.projector.layers.weight",
            "model.projector.weight",
            "model.projector.layers.0.weight",
        ]
        .into_iter()
        .find(|k| keys.contains(*k))
        .context("projector weight key")?;
        let b_key = [
            "model.projector.layers.bias",
            "model.projector.bias",
            "model.projector.layers.0.bias",
        ]
        .into_iter()
        .find(|k| keys.contains(*k));

        let mut want = vec![w_key, IMAGE_NEWLINE, VIEW_SEPARATOR];
        if let Some(b) = b_key {
            want.push(b);
        }
        let map = store.load_keys(&want)?;

        let (w, w_shape) = map.get(w_key).context("projector weight tensor")?;
        ensure!(
            w_shape == [self.out_features, self.in_features].as_slice()
                || w_shape == [self.in_features, self.out_features].as_slice(),
            "projector weight shape {w_shape:?}"
        );
        let mut weight = w.to_vec();
        if w_shape == [self.in_features, self.out_features].as_slice() {
            let mut t = vec![0f32; weight.len()];
            for o in 0..self.out_features {
                for i in 0..self.in_features {
                    t[o * self.in_features + i] = weight[i * self.out_features + o];
                }
            }
            weight = t;
        }
        self.weight = Some(weight);
        if let Some(bk) = b_key {
            let (b, _) = map.get(bk).context("projector bias")?;
            self.bias = Some(b.to_vec());
        }
        let (nl, _) = map.get(IMAGE_NEWLINE).context("image_newline")?;
        self.image_newline = Some(nl.to_vec());
        let (sep, _) = map.get(VIEW_SEPARATOR).context("view_seperator")?;
        self.view_separator = Some(sep.to_vec());
        Ok(())
    }

    pub fn forward(&self, features: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        let w = self.weight.as_ref().context("projector not loaded")?;
        ensure!(
            features.len() == n_tokens * self.in_features,
            "features len {} != {}*{}",
            features.len(),
            n_tokens,
            self.in_features
        );
        // `w` is [out, in] (PyTorch `nn.Linear` layout) — BLAS-backed
        // `y = x @ w^T (+ b)` needs no transpose at call time.
        let mut out = vec![0f32; n_tokens * self.out_features];
        rlx_core::host_kernels::matmul_bt(
            features,
            w,
            &mut out,
            n_tokens,
            self.in_features,
            self.out_features,
            1.0,
        );
        if let Some(b) = &self.bias {
            for row in out.chunks_mut(self.out_features) {
                for (v, bi) in row.iter_mut().zip(b.iter()) {
                    *v += *bi;
                }
            }
        }
        Ok(out)
    }

    pub fn newline(&self) -> Result<&[f32]> {
        self.image_newline
            .as_deref()
            .context("image_newline not loaded")
    }

    pub fn separator(&self) -> Result<&[f32]> {
        self.view_separator
            .as_deref()
            .context("view_separator not loaded")
    }
}
