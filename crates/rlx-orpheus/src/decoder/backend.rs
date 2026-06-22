//! SNAC decode backend — host eager safetensors or native RLX CoreML ([`Device::Ane`]).

use std::path::Path;

use anyhow::{Result, bail};
#[cfg(feature = "coreml")]
use rlx_runtime::{Device, is_available};

use super::eager::SnacDecoder;

/// Load options for [`SnacBackend::open`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SnacLoadOptions {
    /// Run the SNAC conv decoder on native RLX CoreML ([`Device::Ane`]).
    pub coreml: bool,
}

/// SNAC vocoder: host eager safetensors or compiled RLX graph (feature `coreml`).
pub enum SnacBackend {
    Eager(SnacDecoder),
    #[cfg(feature = "coreml")]
    Rlx(super::compiled::RlxSnacDecoder),
}

impl SnacBackend {
    pub fn open(path: &Path, opts: SnacLoadOptions) -> Result<Self> {
        if opts.coreml {
            #[cfg(feature = "coreml")]
            {
                if !is_available(Device::Ane) {
                    bail!(
                        "SNAC CoreML (Device::Ane) is not available — rebuild with \
                         `--features coreml` on macOS"
                    );
                }
                return Ok(Self::Rlx(super::compiled::RlxSnacDecoder::load(
                    path,
                    Device::Ane,
                )?));
            }
            #[cfg(not(feature = "coreml"))]
            {
                bail!(
                    "SNAC CoreML requested but rlx-orpheus was built without \
                     `--features coreml`"
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
            #[cfg(feature = "coreml")]
            Self::Rlx(d) => d.decode_codes(codes_0, codes_1, codes_2),
        }
    }

    pub fn decode_orpheus_frames(&self, frame_tokens: &[i32]) -> Result<Vec<f32>> {
        match self {
            Self::Eager(d) => d.decode_orpheus_frames(frame_tokens),
            #[cfg(feature = "coreml")]
            Self::Rlx(d) => d.decode_orpheus_frames(frame_tokens),
        }
    }
}
