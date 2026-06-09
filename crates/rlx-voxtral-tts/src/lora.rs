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

//! LoRA adapters on Ministral attention + FFN projections (inference merge + eager apply).

use crate::config::TextConfig;
use crate::load::{PREFIX_BACKBONE, VoxtralTtsWeightStore};
use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use std::collections::HashMap;

pub const DEFAULT_LORA_ALPHA: f32 = 16.0;

const ALL_PROJS: [&str; 7] = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"];

#[derive(Debug, Clone)]
pub struct ProjLora {
    pub a: Array2<f32>,
    pub b: Array2<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct LayerLora {
    pub wq: Option<ProjLora>,
    pub wk: Option<ProjLora>,
    pub wv: Option<ProjLora>,
    pub wo: Option<ProjLora>,
    pub w1: Option<ProjLora>,
    pub w2: Option<ProjLora>,
    pub w3: Option<ProjLora>,
}

impl LayerLora {
    pub fn any(&self) -> bool {
        self.wq.is_some()
            || self.wk.is_some()
            || self.wv.is_some()
            || self.wo.is_some()
            || self.w1.is_some()
            || self.w2.is_some()
            || self.w3.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct LoraBank {
    pub layers: Vec<Option<LayerLora>>,
    pub alpha: f32,
    pub rank: usize,
}

impl LoraBank {
    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            return 1.0;
        }
        self.alpha / self.rank as f32
    }

    pub fn has_any(&self) -> bool {
        self.layers
            .iter()
            .any(|l| l.as_ref().is_some_and(|x| x.any()))
    }
}

pub fn has_lora_weights(keys: &std::collections::HashSet<String>) -> bool {
    keys.iter().any(|k| k.contains(".lora."))
}

pub fn load_lora_bank(store: &VoxtralTtsWeightStore, cfg: &TextConfig) -> Result<Option<LoraBank>> {
    if !has_lora_weights(store.keys()) {
        return Ok(None);
    }
    let alpha = std::env::var("RLX_VOXTRAL_TTS_LORA_ALPHA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LORA_ALPHA);
    let mut layers = vec![None; cfg.num_hidden_layers];
    let mut rank = 0usize;
    let snap = load_lora_snapshot(store)?;
    for i in 0..cfg.num_hidden_layers {
        let mut layer = LayerLora::default();
        for proj in ALL_PROJS {
            let a_key = format!("{PREFIX_BACKBONE}{i}.lora.{proj}_a");
            let b_key = format!("{PREFIX_BACKBONE}{i}.lora.{proj}_b");
            let (a_key, b_key) = if snap.contains_key(&a_key) {
                (a_key, b_key)
            } else {
                (format!("lora.{i}.{proj}_a"), format!("lora.{i}.{proj}_b"))
            };
            let Some((a_data, a_shape)) = snap.get(&a_key) else {
                continue;
            };
            let (b_data, b_shape) = snap
                .get(&b_key)
                .with_context(|| format!("missing {b_key}"))?;
            let a = array2(a_data, a_shape)?;
            let b = array2(b_data, b_shape)?;
            ensure!(
                a.nrows() == b.nrows(),
                "lora rank mismatch layer {i} {proj}"
            );
            rank = a.nrows().max(rank);
            let slot = ProjLora { a, b };
            match proj {
                "wq" => layer.wq = Some(slot),
                "wk" => layer.wk = Some(slot),
                "wv" => layer.wv = Some(slot),
                "wo" => layer.wo = Some(slot),
                "w1" => layer.w1 = Some(slot),
                "w2" => layer.w2 = Some(slot),
                "w3" => layer.w3 = Some(slot),
                _ => {}
            }
        }
        if layer.any() {
            layers[i] = Some(layer);
        }
    }
    if !layers.iter().any(|l| l.as_ref().is_some_and(|x| x.any())) {
        return Ok(None);
    }
    Ok(Some(LoraBank {
        layers,
        alpha,
        rank,
    }))
}

/// Merge LoRA delta into weight tensor (checkpoint layout `[out, in]`).
pub fn merge_lora_into_w(
    w: &mut Array2<f32>,
    a: &Array2<f32>,
    b: &Array2<f32>,
    scale: f32,
) -> Result<()> {
    let (out, inp) = w.dim();
    ensure!(a.ncols() == inp, "lora hidden mismatch");
    ensure!(b.ncols() == out, "lora out mismatch");
    ensure!(a.nrows() == b.nrows(), "lora rank mismatch");
    let rank = a.nrows();
    for j in 0..out {
        for i in 0..inp {
            let mut acc = 0f32;
            for r in 0..rank {
                acc += b[[r, j]] * a[[r, i]];
            }
            w[[j, i]] += scale * acc;
        }
    }
    Ok(())
}

pub fn apply_lora_linear(
    h: &Array2<f32>,
    w: &Array2<f32>,
    lora: Option<&ProjLora>,
    scale: f32,
) -> Array2<f32> {
    let mut out = h.dot(&w.t());
    if let Some(adapt) = lora {
        let mid = h.dot(&adapt.a.t());
        out += &(mid.dot(&adapt.b) * scale);
    }
    out
}

pub fn apply_lora_to_backbone(
    backbone: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    lora: &LoraBank,
) -> Result<()> {
    let scale = lora.scale();
    for (i, slot) in lora.layers.iter().enumerate() {
        let Some(adapt) = slot else { continue };
        for (proj, lora_opt, weight_suffix) in [
            ("wq", &adapt.wq, "attention.wq.weight"),
            ("wk", &adapt.wk, "attention.wk.weight"),
            ("wv", &adapt.wv, "attention.wv.weight"),
            ("wo", &adapt.wo, "attention.wo.weight"),
            ("w1", &adapt.w1, "feed_forward.w1.weight"),
            ("w2", &adapt.w2, "feed_forward.w2.weight"),
            ("w3", &adapt.w3, "feed_forward.w3.weight"),
        ] {
            let _ = proj;
            let Some(l) = lora_opt else { continue };
            let key = format!("{PREFIX_BACKBONE}{i}.{weight_suffix}");
            let Some((data, shape)) = backbone.get_mut(&key) else {
                continue;
            };
            ensure!(shape.len() == 2, "shape for {key}");
            let mut w = Array2::from_shape_vec((shape[0], shape[1]), data.clone())
                .with_context(|| format!("reshape {key}"))?;
            merge_lora_into_w(&mut w, &l.a, &l.b, scale)?;
            *data = w.into_raw_vec_and_offset().0;
        }
    }
    Ok(())
}

fn load_lora_snapshot(store: &VoxtralTtsWeightStore) -> Result<crate::load::WeightSnapshot> {
    let mut snap = store.tensor_snapshot(PREFIX_BACKBONE)?;
    for key in store.keys().iter().filter(|k| k.starts_with("lora.")) {
        if snap.contains_key(key) {
            continue;
        }
        let mut legacy = store.tensor_snapshot("")?;
        if let Some(v) = legacy.remove(key) {
            snap.insert(key.clone(), v);
        }
    }
    Ok(snap)
}

fn array2(data: &[f32], shape: &[usize]) -> Result<Array2<f32>> {
    ensure!(shape.len() == 2, "expected rank-2");
    Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).context("reshape lora")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_lora_updates_weight() {
        let mut w = Array2::<f32>::eye(2);
        let lora = ProjLora {
            a: Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap(),
            b: Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap(),
        };
        merge_lora_into_w(&mut w, &lora.a, &lora.b, 2.0).unwrap();
        assert!((w[[1, 0]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn layer_lora_tracks_ffn_projs() {
        let layer = LayerLora {
            w1: Some(ProjLora {
                a: Array2::zeros((2, 4)),
                b: Array2::zeros((2, 8)),
            }),
            ..Default::default()
        };
        assert!(layer.any());
    }
}
