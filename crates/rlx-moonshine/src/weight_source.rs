// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Non-destructive [`WeightSource`] that clones tensors out of a [`WeightMap`].

use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_flow::WeightSource;

pub struct CloningWeightSource<'a>(pub &'a WeightMap);

impl WeightSource for CloningWeightSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .0
            .get(key)
            .ok_or_else(|| anyhow!("missing weight `{key}`"))?;
        if transpose {
            if shape.len() != 2 {
                return Err(anyhow!(
                    "transpose requested for non-2D tensor `{key}` (shape {shape:?})"
                ));
            }
            let (r, c) = (shape[0], shape[1]);
            let mut out = vec![0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    out[j * r + i] = data[i * c + j];
                }
            }
            Ok((out, vec![c, r]))
        } else {
            Ok((data.to_vec(), shape.to_vec()))
        }
    }

    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
}
