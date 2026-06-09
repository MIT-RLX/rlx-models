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

//! Host-side AdamW optimizer (lifted from `rlx-umap`).

use crate::weights::WeightStore;

#[derive(Debug, Clone)]
pub struct AdamState {
    m: WeightStore,
    v: WeightStore,
    step: u64,
}

impl AdamState {
    pub fn new_like(weights: &WeightStore) -> Self {
        Self::new_for_names(weights.0.keys(), weights)
    }

    pub fn new_for_names<'a>(
        names: impl IntoIterator<Item = &'a String>,
        weights: &WeightStore,
    ) -> Self {
        let mut m = WeightStore::default();
        let mut v = WeightStore::default();
        for name in names {
            let Some(data) = weights.0.get(name) else {
                continue;
            };
            m.0.insert(name.clone(), vec![0.0; data.len()]);
            v.0.insert(name.clone(), vec![0.0; data.len()]);
        }
        Self { m, v, step: 0 }
    }

    pub fn step(
        &mut self,
        weights: &mut WeightStore,
        grads: &WeightStore,
        lr: f64,
        beta1: f64,
        beta2: f64,
        weight_decay: f32,
        eps: f64,
        max_norm: f32,
    ) {
        self.step += 1;
        let t = self.step as f64;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);
        let clip_scale = global_grad_clip_scale(grads, max_norm);

        for (name, g) in &grads.0 {
            let Some(w) = weights.0.get_mut(name) else {
                continue;
            };
            let m = self.m.0.get_mut(name).expect("adam m for param");
            let v = self.v.0.get_mut(name).expect("adam v for param");
            debug_assert_eq!(
                w.len(),
                g.len(),
                "adam weight/grad length mismatch for {name}: w={} g={}",
                w.len(),
                g.len()
            );
            let n = w.len().min(g.len()).min(m.len()).min(v.len());
            for i in 0..n {
                let gi = (g[i] + weight_decay * w[i]) * clip_scale;
                m[i] = (beta1 * m[i] as f64 + (1.0 - beta1) * gi as f64) as f32;
                v[i] = (beta2 * v[i] as f64 + (1.0 - beta2) * (gi as f64 * gi as f64)) as f32;
                let m_hat = m[i] as f64 / bc1;
                let v_hat = v[i] as f64 / bc2;
                w[i] -= (lr * m_hat / (v_hat.sqrt() + eps)) as f32;
            }
        }
    }
}

pub fn global_grad_clip_scale(grads: &WeightStore, max_norm: f32) -> f32 {
    let mut norm_sq = 0.0f32;
    for g in grads.0.values() {
        for gi in g {
            if gi.is_finite() {
                norm_sq += gi * gi;
            }
        }
    }
    let max_sq = max_norm * max_norm;
    if norm_sq > max_sq && norm_sq > 0.0 {
        max_norm / norm_sq.sqrt()
    } else {
        1.0
    }
}
