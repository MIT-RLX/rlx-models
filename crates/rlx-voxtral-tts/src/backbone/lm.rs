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

//! Ministral language model (eager CPU, static KV cache).

use crate::backbone::layer::{DecoderLayer, LayerKv};
use crate::backbone::rope::{build_inv_freq, build_rope_tables};
use crate::config::TextConfig;
use crate::load::PREFIX_BACKBONE;
use crate::math::rms_norm;
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;

pub struct MinistralLm {
    layers: Vec<DecoderLayer>,
    norm: Array1<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    kv: Vec<LayerKv>,
    pos: usize,
    hidden: usize,
}

impl MinistralLm {
    pub fn from_tensors(
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        cfg: &TextConfig,
    ) -> Result<Self> {
        Self::from_tensors_with_lora(tensors, cfg, None)
    }

    pub fn from_tensors_with_lora(
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        cfg: &TextConfig,
        lora: Option<&crate::lora::LoraBank>,
    ) -> Result<Self> {
        let n_layers = cfg.num_hidden_layers;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let mut layer = DecoderLayer::load(tensors, &format!("{PREFIX_BACKBONE}{i}"), cfg)?;
            if let Some(bank) = lora {
                if let Some(adapt) = bank.layers.get(i).and_then(|x| x.as_ref()) {
                    layer.set_lora(adapt.clone(), bank.scale());
                }
            }
            layers.push(layer);
        }
        let norm = take1d(tensors, "norm.weight")?;
        let inv = build_inv_freq(cfg.rope_theta, cfg.head_dim);
        let (cos, sin) = build_rope_tables(&inv, cfg.max_position_embeddings);
        let kv = (0..n_layers)
            .map(|_| LayerKv {
                k: Array2::<f32>::zeros((0, 0)),
                v: Array2::<f32>::zeros((0, 0)),
            })
            .collect();
        Ok(Self {
            layers,
            norm,
            cos,
            sin,
            kv,
            pos: 0,
            hidden: cfg.hidden_size,
        })
    }

    pub fn reset_cache(&mut self) {
        self.pos = 0;
        for kv in &mut self.kv {
            kv.k = Array2::<f32>::zeros((0, 0));
            kv.v = Array2::<f32>::zeros((0, 0));
        }
    }

    /// Forward layer stack without final RMS norm (matches LoRA train graph).
    pub fn forward_pre_norm(&mut self, inputs_embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (seq, h) = inputs_embeds.dim();
        ensure!(h == self.hidden, "hidden size mismatch");
        let start = self.pos;
        let mut x = inputs_embeds.to_owned();
        for (layer, kv) in self.layers.iter().zip(self.kv.iter_mut()) {
            x = layer.forward(x.view(), &self.cos, &self.sin, start, kv)?;
        }
        self.pos += seq;
        Ok(x)
    }

    /// Forward `inputs_embeds` `[seq, hidden]`; returns final hidden states.
    pub fn forward(&mut self, inputs_embeds: ArrayView2<f32>) -> Result<Array2<f32>> {
        let x = self.forward_pre_norm(inputs_embeds)?;
        Ok(rms_norm(x.view(), self.norm.view(), 1e-5))
    }

    /// Hidden state at last token after forward.
    pub fn last_hidden(&self, hidden: &Array2<f32>) -> Array1<f32> {
        hidden.row(hidden.dim().0 - 1).to_owned()
    }
}

fn take1d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array1<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 1);
    Array1::from_shape_vec(shape[0], data.clone()).with_context(|| key.to_string())
}
