// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! UniAdapter configuration + initialization (`x' = x + s·ReLU(x·W_down)·W_up`).

use std::collections::HashMap;

use crate::vit::config::VitConfig;
use crate::vit::forward::AdapterOpts;

/// UniAdapter geometry.
#[derive(Clone, Copy, Debug)]
pub struct AdapterConfig {
    pub rank: usize,
    pub scale: f32,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            rank: 64,
            scale: 0.1,
        }
    }
}

impl AdapterConfig {
    pub fn opts(&self) -> AdapterOpts {
        AdapterOpts {
            rank: self.rank,
            scale: self.scale,
        }
    }
}

/// Initialize per-layer adapter params: `down` small-random, `up` zeros — so
/// the adapter is the identity at init (standard adapter/LoRA practice).
pub fn init_adapter_params(
    cfg: &VitConfig,
    ac: &AdapterConfig,
    seed: u32,
) -> HashMap<String, Vec<f32>> {
    let h = cfg.hidden_size;
    let r = ac.rank;
    let mut sd = seed;
    let fan = (h as f32).sqrt().recip();
    let mut p = HashMap::new();
    for l in 0..cfg.num_hidden_layers {
        sd = sd
            .wrapping_mul(1664525)
            .wrapping_add(1013904223)
            .wrapping_add(0x9E3779B9);
        let mut s = sd;
        let down: Vec<f32> = (0..h * r)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 2.0 * fan
            })
            .collect();
        p.insert(format!("blocks.{l}.adapter.down.weight"), down);
        p.insert(format!("blocks.{l}.adapter.up.weight"), vec![0.0; r * h]);
    }
    p
}
