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

//! Text embedding + projection (`talker.model.text_embedding`, `talker.text_projection`).

use crate::load::Qwen3TtsWeightStore;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView1};

pub struct TextEmbedder {
    embed: Array2<f32>,
    fc1_w: Array2<f32>,
    fc1_b: Array1<f32>,
    fc2_w: Array2<f32>,
    fc2_b: Array1<f32>,
    hidden_act: String,
}

impl TextEmbedder {
    pub fn open(store: &Qwen3TtsWeightStore) -> Result<Self> {
        let snap = store.tensor_snapshot(&[
            "talker.model.text_embedding.weight",
            "talker.text_projection.linear_fc1.weight",
            "talker.text_projection.linear_fc1.bias",
            "talker.text_projection.linear_fc2.weight",
            "talker.text_projection.linear_fc2.bias",
        ])?;
        let (ew, es) = snap.get("talker.model.text_embedding.weight").unwrap();
        let embed = Array2::from_shape_vec((es[0], es[1]), ew.clone()).context("text embed")?;
        let (w1, s1) = snap
            .get("talker.text_projection.linear_fc1.weight")
            .unwrap();
        let (b1, _) = snap.get("talker.text_projection.linear_fc1.bias").unwrap();
        let (w2, s2) = snap
            .get("talker.text_projection.linear_fc2.weight")
            .unwrap();
        let (b2, _) = snap.get("talker.text_projection.linear_fc2.bias").unwrap();
        Ok(Self {
            embed,
            fc1_w: Array2::from_shape_vec((s1[0], s1[1]), w1.clone())?,
            fc1_b: Array1::from_vec(b1.clone()),
            fc2_w: Array2::from_shape_vec((s2[0], s2[1]), w2.clone())?,
            fc2_b: Array1::from_vec(b2.clone()),
            hidden_act: "silu".into(),
        })
    }

    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let text_hidden = self.embed.shape()[1];
        let mut out = vec![0f32; ids.len() * text_hidden];
        for (i, &id) in ids.iter().enumerate() {
            let row = self.embed.row(id as usize);
            out[i * text_hidden..(i + 1) * text_hidden].copy_from_slice(row.as_slice().unwrap());
        }
        Ok(out)
    }

    pub fn project(&self, text_hidden: ArrayView1<f32>) -> Result<Vec<f32>> {
        let mut mid = vec![0f32; self.fc1_w.shape()[0]];
        for i in 0..mid.len() {
            mid[i] = text_hidden
                .iter()
                .zip(self.fc1_w.row(i).iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                + self.fc1_b[i];
        }
        if self.hidden_act == "silu" {
            for v in &mut mid {
                *v = silu(*v);
            }
        }
        let mut out = vec![0f32; self.fc2_w.shape()[0]];
        for i in 0..out.len() {
            out[i] = mid
                .iter()
                .zip(self.fc2_w.row(i).iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                + self.fc2_b[i];
        }
        Ok(out)
    }

    pub fn embed_token(&self, id: u32) -> Result<Vec<f32>> {
        let row = self.embed.row(id as usize);
        self.project(row)
    }

    pub fn embed_project_ids(&self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let text_hidden = self.embed.shape()[1];
        let flat = self.embed_ids(ids)?;
        let mut out = Vec::with_capacity(ids.len());
        for chunk in flat.chunks(text_hidden) {
            out.push(self.project(ndarray::ArrayView1::from(chunk))?);
        }
        Ok(out)
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
