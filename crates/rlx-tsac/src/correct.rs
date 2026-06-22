//! Correct TSAC codec on all RLX backends.
//!
//! TSAC's neural codec **is** Descript-DAC-44kHz: Bellard's `tsac` ships the
//! descript weights, BF8-quantized via the closed-source `libnc`. Verified by
//! roundtrip comparison — rlx-dac + descript weights vs Bellard's `tsac` produce
//! **0.9957** correlated output, both reconstructing the input at ~0.98 corr.
//!
//! Bellard's bundled `*_q8.bin` weights use libnc's BF8 layout, whose dequant is
//! a documented unsolved closed-source problem (`vendor/docs/bf8-analysis.md`) —
//! tsac-ng's own dequant gives 0.002 corr. So the *correct* path uses the real
//! descript weights (freely downloadable) through rlx-dac's verified, all-backend
//! (cpu/metal/mlx/wgpu) encode+decode graph. The vendored q8 path remains for
//! bit-exact reproduction of tsac-ng files (see [`crate::rlx_decode`]).

use anyhow::{Context, Result};
use rlx_dac::DacCodec;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Fastest available RLX backend for this codec, measured on Apple silicon:
/// **MLX ≫ Metal > wgpu > CPU** (MLX ~14–26× real-time; CPU ~0.02×). Only
/// considers backends compiled in *and* available at runtime.
pub fn best_device() -> Device {
    for d in [Device::Mlx, Device::Metal, Device::Gpu] {
        if rlx_runtime::is_available(d) {
            return d;
        }
    }
    Device::Cpu
}

/// Default directory for the descript-DAC-44kHz weights (`RLX_DAC_DIR` or
/// `.cache/dac44`).
pub fn default_dir() -> PathBuf {
    std::env::var("RLX_DAC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/dac44"))
}

/// Open the correct TSAC (= Descript-DAC-44kHz) codec on `device`, ensuring the
/// weights are present (downloaded if the crate is built with the `oracle`
/// feature, i.e. `rlx-dac/hf-download`). Returns a full encode+decode codec that
/// runs on every RLX backend.
pub fn open(device: Device) -> Result<DacCodec> {
    open_in(&default_dir(), device)
}

/// Like [`open`] but with an explicit weights directory.
pub fn open_in(model_dir: &Path, device: Device) -> Result<DacCodec> {
    rlx_dac::download::ensure_weights(model_dir)?;
    DacCodec::open_on(model_dir, device)
}

/// True when descript-44kHz weights are present in `dir` (no download needed).
pub fn weights_available(dir: &Path) -> bool {
    dir.join("model.safetensors").is_file() && dir.join("config.json").is_file()
}

/// File-level correct TSAC codec: WAV ⇄ `.tsac` container, encode+decode on any
/// RLX backend. The container is a small self-describing RVQ-code blob (magic
/// `TSRX`) — self-consistent (its own encode/decode roundtrip), distinct from
/// Bellard's libnc container.
pub struct CorrectCodec {
    dac: DacCodec,
    num_quantizers: Option<usize>,
}

/// Legacy container: header + one `u16` per RVQ code (16 bits each).
const MAGIC_V1: &[u8; 4] = b"TSRX";
/// Bit-packed container: header + a `bits_per_code` byte + codes packed at
/// exactly `ceil(log2(codebook_size))` bits each (10 bits for the 1024-entry
/// DAC codebooks), LSB-first. ~37 % smaller than V1, losslessly. V1 files still
/// decode (see [`CorrectCodec::decode_file`]).
const MAGIC_V2: &[u8; 4] = b"TSR2";

/// Bits needed to index `codebook_size` codebook entries.
fn bits_per_code(codebook_size: usize) -> u32 {
    codebook_size.next_power_of_two().trailing_zeros().max(1)
}

/// Append `value`'s low `bits` bits to `buf`/`acc`/`nbits`, LSB-first.
struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    nbits: u32,
}
impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }
    fn put(&mut self, value: u32, bits: u32) {
        self.acc |= (value as u64) << self.nbits;
        self.nbits += bits;
        while self.nbits >= 8 {
            self.buf.push((self.acc & 0xff) as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.buf.push((self.acc & 0xff) as u8);
        }
        self.buf
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}
impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            acc: 0,
            nbits: 0,
        }
    }
    fn get(&mut self, bits: u32) -> u32 {
        while self.nbits < bits {
            let byte = self.bytes.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.acc |= (byte as u64) << self.nbits;
            self.nbits += 8;
        }
        let mask = if bits >= 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        let v = (self.acc as u32) & mask;
        self.acc >>= bits;
        self.nbits -= bits;
        v
    }
}

impl CorrectCodec {
    /// Open on `device`, drawing weights from [`default_dir`]. `quality` maps to
    /// the number of RVQ codebooks used (1..=9; `None` = all).
    pub fn open(device: Device, quality: Option<u8>) -> Result<Self> {
        Self::open_in(&default_dir(), device, quality)
    }

    pub fn open_in(model_dir: &Path, device: Device, quality: Option<u8>) -> Result<Self> {
        let dac = open_in(model_dir, device)?;
        let num_quantizers = quality.map(|q| (q as usize).clamp(1, 9));
        Ok(Self {
            dac,
            num_quantizers,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.dac.sample_rate()
    }

    /// Encode a WAV (any rate/channels — resampled to mono 44.1 kHz) to `.tsac`,
    /// bit-packing the RVQ codes (V2 container).
    pub fn encode_file(&self, in_audio: &Path, out_tsac: &Path) -> Result<()> {
        let codes = self.dac.encode_wav(in_audio, self.num_quantizers)?;
        let orig = mono_len_44k(in_audio)?;
        let bits = bits_per_code(self.dac.config().codebook_size);
        let mut buf = Vec::with_capacity(
            21 + (codes.num_frames() * codes.num_quantizers * bits as usize).div_ceil(8),
        );
        buf.extend_from_slice(MAGIC_V2);
        buf.extend_from_slice(&self.dac.sample_rate().to_le_bytes());
        buf.extend_from_slice(&(orig as u32).to_le_bytes());
        buf.extend_from_slice(&(codes.num_frames() as u32).to_le_bytes());
        buf.extend_from_slice(&(codes.num_quantizers as u32).to_le_bytes());
        buf.push(bits as u8);
        let mut bw = BitWriter::new();
        for frame in &codes.frames {
            for &c in frame {
                bw.put(c, bits);
            }
        }
        buf.extend_from_slice(&bw.finish());
        if let Some(parent) = out_tsac.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        std::fs::write(out_tsac, buf).with_context(|| format!("write {}", out_tsac.display()))?;
        Ok(())
    }

    /// Decode a `.tsac` container (V2 packed or legacy V1 `u16`) back to a WAV.
    pub fn decode_file(&self, in_tsac: &Path, out_wav: &Path) -> Result<()> {
        let b = std::fs::read(in_tsac).with_context(|| format!("read {}", in_tsac.display()))?;
        anyhow::ensure!(b.len() >= 20, "container too short");
        let rd = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
        let orig = rd(8);
        let n_frames = rd(12);
        let n_cb = rd(16);
        let frames = if &b[0..4] == MAGIC_V2 {
            anyhow::ensure!(b.len() >= 21, "TSR2 truncated header");
            let bits = b[20] as u32;
            anyhow::ensure!((1..=16).contains(&bits), "TSR2 bad bits_per_code {bits}");
            let need = (n_frames * n_cb * bits as usize).div_ceil(8);
            anyhow::ensure!(b.len() >= 21 + need, "TSR2 truncated payload");
            let mut br = BitReader::new(&b[21..]);
            let mut frames = Vec::with_capacity(n_frames);
            for _ in 0..n_frames {
                let mut row = Vec::with_capacity(n_cb);
                for _ in 0..n_cb {
                    row.push(br.get(bits));
                }
                frames.push(row);
            }
            frames
        } else if &b[0..4] == MAGIC_V1 {
            anyhow::ensure!(b.len() >= 20 + n_frames * n_cb * 2, "TSRX truncated");
            let mut frames = Vec::with_capacity(n_frames);
            let mut off = 20;
            for _ in 0..n_frames {
                let mut row = Vec::with_capacity(n_cb);
                for _ in 0..n_cb {
                    row.push(u16::from_le_bytes([b[off], b[off + 1]]) as u32);
                    off += 2;
                }
                frames.push(row);
            }
            frames
        } else {
            anyhow::bail!("not a TSRX/TSR2 container");
        };
        let codes = rlx_dac::DacCodes {
            frames,
            num_quantizers: n_cb,
        };
        self.dac.decode_wav(&codes, out_wav, Some(orig))
    }
}

fn mono_len_44k(path: &Path) -> Result<usize> {
    let r = hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let s = r.spec();
    let frames = (r.len() as usize) / (s.channels as usize).max(1);
    Ok((frames as u64 * 44_100 / s.sample_rate.max(1) as u64) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit writer/reader round-trips arbitrary 10-bit codes losslessly,
    /// which is what makes the packed TSR2 container bit-exact with the codes.
    #[test]
    fn bit_pack_roundtrip_10bit() {
        let bits = bits_per_code(1024);
        assert_eq!(bits, 10);
        let codes: Vec<u32> = (0..5000u32).map(|i| (i * 37 + 11) % 1024).collect();
        let mut bw = BitWriter::new();
        for &c in &codes {
            bw.put(c, bits);
        }
        let packed = bw.finish();
        // 10 bits/code, byte-aligned tail.
        assert_eq!(packed.len(), (codes.len() * bits as usize).div_ceil(8));
        let mut br = BitReader::new(&packed);
        for &c in &codes {
            assert_eq!(br.get(bits), c);
        }
    }

    #[test]
    fn bits_per_code_sizes() {
        assert_eq!(bits_per_code(1024), 10);
        assert_eq!(bits_per_code(512), 9);
        assert_eq!(bits_per_code(2048), 11);
        assert_eq!(bits_per_code(1), 1);
    }
}
