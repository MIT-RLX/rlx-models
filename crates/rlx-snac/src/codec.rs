// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// High-level SNAC decoder: device + weights + per-length compiled-graph cache.

use crate::eager;
use crate::graph::{SnacDecoderGraph, SnacEncoderGraph};
use crate::model::SnacWeights;
use anyhow::Result;
use rlx_core::HierarchicalCodes;
use rlx_runtime::Device;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

/// SNAC encoder running on a chosen rlx backend (conv stack as a graph,
/// multi-scale RVQ on the host).
pub struct SnacEncoder {
    weights: SnacWeights,
    device: Device,
    graphs: RefCell<HashMap<usize, SnacEncoderGraph>>,
}

impl SnacEncoder {
    pub fn new(weights: SnacWeights, device: Device) -> Self {
        Self {
            weights,
            device,
            graphs: RefCell::new(HashMap::new()),
        }
    }

    pub fn from_safetensors_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let weights =
            SnacWeights::from_safetensors(&bytes, crate::config::SnacConfig::snac_24khz())?;
        anyhow::ensure!(
            weights.encoder.is_some(),
            "checkpoint has no encoder weights"
        );
        Ok(Self::new(weights, device))
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Encode mono PCM (length should be a multiple of the codec hop) to codes.
    pub fn encode(&self, pcm: &[f32]) -> Result<HierarchicalCodes> {
        let mut cache = self.graphs.borrow_mut();
        let g = match cache.get_mut(&pcm.len()) {
            Some(g) => g,
            None => {
                let g = SnacEncoderGraph::compile_for(self.device, &self.weights, pcm.len())?;
                cache.entry(pcm.len()).or_insert(g)
            }
        };
        let (latent, ld, t) = g.run(pcm)?;
        debug_assert_eq!(ld, self.weights.latent());
        eager::rvq_encode(&self.weights, &latent, t)
    }
}

/// SNAC decoder running on a chosen rlx backend. The multi-scale RVQ runs on the
/// host; the conv decoder stack is a graph compiled per latent length.
pub struct SnacDecoder {
    weights: SnacWeights,
    device: Device,
    graphs: RefCell<HashMap<usize, SnacDecoderGraph>>,
}

impl SnacDecoder {
    pub fn new(weights: SnacWeights, device: Device) -> Self {
        Self {
            weights,
            device,
            graphs: RefCell::new(HashMap::new()),
        }
    }

    /// Load from a folded-weight_norm safetensors export (see
    /// `scripts/gen_fixture.py`) with the SNAC 24 kHz config.
    pub fn from_safetensors_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let weights =
            SnacWeights::from_safetensors(&bytes, crate::config::SnacConfig::snac_24khz())?;
        Ok(Self::new(weights, device))
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn sample_rate(&self) -> u32 {
        self.weights.config.sampling_rate
    }

    /// Decode codes to PCM. `noise_seed` seeds the per-block noise planes; pass
    /// `None` for silent (zero) noise (deterministic, slightly duller output).
    pub fn decode(&self, codes: &HierarchicalCodes, noise_seed: Option<u64>) -> Result<Vec<f32>> {
        let (z_q, t_latent) = eager::from_codes(&self.weights, codes)?;
        let mut cache = self.graphs.borrow_mut();
        let g = match cache.get_mut(&t_latent) {
            Some(g) => g,
            None => {
                let g = SnacDecoderGraph::compile_for(self.device, &self.weights, t_latent)?;
                cache.entry(t_latent).or_insert(g)
            }
        };
        let noise = make_noise(&g.noise_lengths(), noise_seed);
        g.run(&z_q, &noise)
    }
}

fn make_noise(lengths: &[usize], seed: Option<u64>) -> Vec<Vec<f32>> {
    match seed {
        None => lengths.iter().map(|&t| vec![0.0; t]).collect(),
        Some(s) => {
            // Box–Muller from a deterministic LCG → standard normal noise.
            let mut state = s | 1;
            let mut next_u = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            lengths
                .iter()
                .map(|&t| {
                    (0..t)
                        .map(|_| {
                            let u1 = next_u().max(1e-12);
                            let u2 = next_u();
                            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
                        })
                        .collect()
                })
                .collect()
        }
    }
}
