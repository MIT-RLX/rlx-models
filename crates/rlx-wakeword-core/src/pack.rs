// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Embedded `.rlxw` pack header (`RLXW` magic).
//!
//! Payload dtypes: [`dtype::F32`], [`dtype::INT8`] (reserved), [`dtype::TERNARY`] (packed trits).

/// ASCII `RLXW`.
pub const PACK_MAGIC: [u8; 4] = *b"RLXW";
pub const PACK_VERSION: u16 = 1;

/// Payload dtype for [`PackHeader::dtype`].
pub mod dtype {
    pub const F32: u16 = 0;
    /// Reserved int8 path.
    pub const INT8: u16 = 1;
    /// Packed trits (2 bits/weight) + f32 biases — see `ternary::pack_trits`.
    pub const TERNARY: u16 = 2;
}

/// Reserved quantization scales for a future int8 path.
#[derive(Debug, Clone, Default)]
pub struct QuantScales {
    pub weight_scale: f32,
    pub act_scale: f32,
}

/// Fixed header for a flat `.rlxw` blob (little-endian fields).
#[derive(Debug, Clone, Copy)]
pub struct PackHeader {
    pub magic: [u8; 4],
    pub version: u16,
    /// 0 = f32, 1 = int8 (reserved), 2 = ternary packed trits.
    pub dtype: u16,
    pub n_phrases: u32,
    pub hop_samples: u32,
    pub header_bytes: u32,
    pub payload_bytes: u32,
}

impl PackHeader {
    pub const BYTES: usize = 28;

    pub fn new_f32(n_phrases: u32, hop_samples: u32, payload_bytes: u32) -> Self {
        Self {
            magic: PACK_MAGIC,
            version: PACK_VERSION,
            dtype: dtype::F32,
            n_phrases,
            hop_samples,
            header_bytes: Self::BYTES as u32,
            payload_bytes,
        }
    }

    pub fn new_ternary(n_phrases: u32, hop_samples: u32, payload_bytes: u32) -> Self {
        Self {
            magic: PACK_MAGIC,
            version: PACK_VERSION,
            dtype: dtype::TERNARY,
            n_phrases,
            hop_samples,
            header_bytes: Self::BYTES as u32,
            payload_bytes,
        }
    }

    pub fn write_to(&self, out: &mut [u8]) {
        assert!(out.len() >= Self::BYTES);
        out[0..4].copy_from_slice(&self.magic);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.dtype.to_le_bytes());
        out[8..12].copy_from_slice(&self.n_phrases.to_le_bytes());
        out[12..16].copy_from_slice(&self.hop_samples.to_le_bytes());
        out[16..20].copy_from_slice(&self.header_bytes.to_le_bytes());
        out[20..24].copy_from_slice(&self.payload_bytes.to_le_bytes());
        out[24..28].fill(0);
    }

    pub fn read_from(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::BYTES {
            return None;
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != PACK_MAGIC {
            return None;
        }
        Some(Self {
            magic,
            version: u16::from_le_bytes([buf[4], buf[5]]),
            dtype: u16::from_le_bytes([buf[6], buf[7]]),
            n_phrases: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            hop_samples: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            header_bytes: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
            payload_bytes: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = PackHeader::new_ternary(2, 640, 1000);
        let mut buf = [0u8; PackHeader::BYTES];
        h.write_to(&mut buf);
        let r = PackHeader::read_from(&buf).unwrap();
        assert_eq!(r.n_phrases, 2);
        assert_eq!(r.hop_samples, 640);
        assert_eq!(r.dtype, dtype::TERNARY);
        assert_eq!(r.version, PACK_VERSION);
    }
}
