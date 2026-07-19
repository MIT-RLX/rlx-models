//! SNAC decode backend — host eager safetensors, or native RLX conv decoder on
//! a GPU backend (Metal / MLX / wgpu / CUDA / CoreML) via the compiled HIR path.

use std::path::Path;

use anyhow::Result;
#[cfg(feature = "snac-rlx")]
use rlx_runtime::{Device, is_available};

use super::eager::SnacDecoder;

/// Where the SNAC conv decoder runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SnacExec {
    /// Host eager safetensors decoder (CPU, ndarray). Default.
    #[default]
    CpuEager,
    /// Native RLX CoreML (`Device::Ane`).
    Ane,
    /// Native RLX Metal.
    Metal,
    /// Native RLX MLX.
    Mlx,
    /// Native RLX wgpu (`Device::Gpu`).
    Wgpu,
    /// Native RLX CUDA.
    Cuda,
}

/// Load options for [`SnacBackend::open`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SnacLoadOptions {
    /// Execution target for the SNAC conv decoder.
    pub exec: SnacExec,
}

/// SNAC vocoder: host eager safetensors, or the compiled RLX graph on a GPU
/// backend (feature `snac-rlx`, pulled in by each GPU feature).
pub enum SnacBackend {
    Eager(SnacDecoder),
    #[cfg(feature = "snac-rlx")]
    Rlx(super::compiled::RlxSnacDecoder),
}

impl SnacBackend {
    pub fn open(path: &Path, opts: SnacLoadOptions) -> Result<Self> {
        #[cfg(feature = "snac-rlx")]
        {
            let device = match opts.exec {
                SnacExec::CpuEager => None,
                SnacExec::Ane => Some(Device::Ane),
                SnacExec::Metal => Some(Device::Metal),
                SnacExec::Mlx => Some(Device::Mlx),
                SnacExec::Wgpu => Some(Device::Gpu),
                SnacExec::Cuda => Some(Device::Cuda),
            };
            if let Some(dev) = device {
                if is_available(dev) {
                    return Ok(Self::Rlx(super::compiled::RlxSnacDecoder::load(path, dev)?));
                }
                eprintln!(
                    "[snac] requested RLX device {dev:?} unavailable — using CPU eager decoder"
                );
            }
        }
        #[cfg(not(feature = "snac-rlx"))]
        {
            if !matches!(opts.exec, SnacExec::CpuEager) {
                eprintln!(
                    "[snac] GPU SNAC requested but rlx-orpheus built without a `snac-rlx` \
                     backend — using CPU eager decoder"
                );
            }
        }
        Ok(Self::Eager(SnacDecoder::from_file(path)?))
    }

    pub fn decode_codes(
        &self,
        codes_0: &[i32],
        codes_1: &[i32],
        codes_2: &[i32],
    ) -> Result<Vec<f32>> {
        match self {
            Self::Eager(d) => d.decode_codes(codes_0, codes_1, codes_2),
            #[cfg(feature = "snac-rlx")]
            Self::Rlx(d) => d.decode_codes(codes_0, codes_1, codes_2),
        }
    }

    pub fn decode_orpheus_frames(&self, frame_tokens: &[i32]) -> Result<Vec<f32>> {
        match self {
            Self::Eager(d) => d.decode_orpheus_frames(frame_tokens),
            #[cfg(feature = "snac-rlx")]
            Self::Rlx(d) => d.decode_orpheus_frames(frame_tokens),
        }
    }
}
