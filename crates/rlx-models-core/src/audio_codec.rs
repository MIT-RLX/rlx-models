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

//! A unified interface for the workspace's neural audio codecs (rlx-mimi,
//! rlx-dac, and any RVQ tokenizer), so TTS/ASR consumers can swap codecs
//! without binding to a concrete struct. Covers the discrete-RVQ shape that
//! mimi and dac share; codec-agnostic bitrate control and resampling are
//! provided as trait defaults on top of [`crate::audio`].

use crate::audio::resample_linear;
use anyhow::Result;
use rlx_runtime::Device;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Discrete residual-vector-quantizer codes in frame-major layout:
/// `frames[t][q]` is the codebook index at time step `t` for quantizer `q`.
///
/// This is the representation rlx-mimi (`MimiCodes`) and rlx-dac (`DacCodes`)
/// already use internally; [`RvqCodes`] is the neutral exchange type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RvqCodes {
    /// `[num_frames][num_quantizers]` codebook indices.
    pub frames: Vec<Vec<u32>>,
    /// Quantizers (codebooks) per frame.
    pub num_quantizers: usize,
}

impl RvqCodes {
    pub fn new(frames: Vec<Vec<u32>>, num_quantizers: usize) -> Self {
        Self {
            frames,
            num_quantizers,
        }
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Transpose to quantizer-major `[num_quantizers][num_frames]` (the layout
    /// HF `MimiModel.encode` and the DAC reference use).
    pub fn to_quantizer_major(&self) -> Vec<Vec<u32>> {
        let nq = self.num_quantizers;
        let t = self.num_frames();
        let mut out = vec![Vec::with_capacity(t); nq];
        for frame in &self.frames {
            for (q, &code) in frame.iter().enumerate().take(nq) {
                out[q].push(code);
            }
        }
        out
    }

    /// Build from quantizer-major `[num_quantizers][num_frames]` rows.
    pub fn from_quantizer_major(rows: &[Vec<u32>]) -> Self {
        let nq = rows.len();
        let t = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut frames = Vec::with_capacity(t);
        for ti in 0..t {
            let mut frame = Vec::with_capacity(nq);
            for row in rows {
                frame.push(row.get(ti).copied().unwrap_or(0));
            }
            frames.push(frame);
        }
        Self {
            frames,
            num_quantizers: nq,
        }
    }
}

/// Multi-scale RVQ codes where each level runs at a different temporal rate
/// (e.g. SNAC: coarse level has fewer tokens than fine levels). Levels are
/// coarse-to-fine; lengths may differ between levels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HierarchicalCodes {
    /// One token stream per RVQ level (codebook indices).
    pub levels: Vec<Vec<u32>>,
}

impl HierarchicalCodes {
    pub fn new(levels: Vec<Vec<u32>>) -> Self {
        Self { levels }
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn level_lengths(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.len()).collect()
    }

    pub fn total_tokens(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.iter().all(|l| l.is_empty())
    }
}

/// Static description of a codec — enough to reason about rate, framing, and
/// bitrate without touching a concrete config struct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodecInfo {
    /// Audio sample rate (Hz) the codec consumes/produces.
    pub sample_rate: u32,
    /// Codec frames per second (Hz) — e.g. Mimi 12.5, DAC `sample_rate/hop`.
    pub frame_rate: f32,
    /// PCM samples per codec frame (`sample_rate / frame_rate`).
    pub hop_length: usize,
    /// Audio channels (1 for the mono codecs here).
    pub channels: usize,
    /// Codebooks (quantizers) available at full rate.
    pub max_quantizers: usize,
    /// Entries per codebook.
    pub codebook_size: usize,
}

impl CodecInfo {
    /// Bits carried by one codebook index (`log2(codebook_size)`).
    pub fn bits_per_codebook(&self) -> f32 {
        (self.codebook_size.max(2) as f32).log2()
    }

    /// Bits/sec contributed by a single codebook at this frame rate.
    pub fn bits_per_second_per_codebook(&self) -> f32 {
        self.bits_per_codebook() * self.frame_rate
    }

    /// Nominal bitrate (bits/sec) for `num_quantizers` codebooks (default: all).
    pub fn bitrate_bps(&self, num_quantizers: Option<usize>) -> f32 {
        let q = num_quantizers
            .unwrap_or(self.max_quantizers)
            .min(self.max_quantizers);
        q as f32 * self.bits_per_second_per_codebook()
    }

    /// The largest codebook count whose nominal bitrate stays at or below
    /// `target_bps`, clamped to `[1, max_quantizers]`. Codec-agnostic bitrate
    /// control: callers pass a bitrate, codecs pick the quantizer count.
    pub fn quantizers_for_bitrate(&self, target_bps: f32) -> usize {
        let per_q = self.bits_per_second_per_codebook();
        if per_q <= 0.0 {
            return 1;
        }
        let q = (target_bps / per_q).floor() as i64;
        q.clamp(1, self.max_quantizers as i64) as usize
    }

    /// Number of codec frames produced for `num_samples` PCM samples.
    pub fn frames_for_samples(&self, num_samples: usize) -> usize {
        if self.hop_length == 0 {
            return 0;
        }
        num_samples.div_ceil(self.hop_length)
    }
}

/// A neural audio codec: PCM ⇄ discrete RVQ codes, on a chosen backend.
///
/// `encode_pcm`/`decode_codes` are whole-clip. The provided methods add
/// bitrate-driven encoding and resampling front-ends on top, so every codec
/// gets consistent `--target-bitrate` and arbitrary-input-rate behaviour for
/// free.
pub trait AudioCodec {
    /// Static metadata (rate, framing, codebooks).
    fn info(&self) -> CodecInfo;

    /// Backend this codec executes on.
    fn device(&self) -> Device;

    /// Encode mono PCM (at [`CodecInfo::sample_rate`]) to RVQ codes. `None` uses
    /// the codec's full quantizer count.
    fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<RvqCodes>;

    /// Decode RVQ codes back to mono PCM at [`CodecInfo::sample_rate`].
    fn decode_codes(&self, codes: &RvqCodes) -> Result<Vec<f32>>;

    /// Convenience: the codec sample rate.
    fn sample_rate(&self) -> u32 {
        self.info().sample_rate
    }

    /// Encode at a target bitrate (bits/sec) rather than an explicit quantizer
    /// count — the codec maps the bitrate to codebooks via [`CodecInfo`].
    fn encode_pcm_bitrate(&self, pcm: &[f32], target_bps: f32) -> Result<RvqCodes> {
        let nq = self.info().quantizers_for_bitrate(target_bps);
        self.encode_pcm(pcm, Some(nq))
    }

    /// Encode PCM that arrives at `input_rate`, resampling to the codec rate
    /// first (shared linear resampler).
    fn encode_pcm_resampled(
        &self,
        pcm: &[f32],
        input_rate: u32,
        num_quantizers: Option<usize>,
    ) -> Result<RvqCodes> {
        let sr = self.info().sample_rate;
        if input_rate == sr {
            self.encode_pcm(pcm, num_quantizers)
        } else {
            let resampled = resample_linear(pcm, input_rate, sr);
            self.encode_pcm(&resampled, num_quantizers)
        }
    }

    /// Decode then resample the waveform to `output_rate`.
    fn decode_codes_resampled(&self, codes: &RvqCodes, output_rate: u32) -> Result<Vec<f32>> {
        let pcm = self.decode_codes(codes)?;
        let sr = self.info().sample_rate;
        Ok(if output_rate == sr {
            pcm
        } else {
            resample_linear(&pcm, sr, output_rate)
        })
    }

    /// Encode then decode, returning both the codes and the reconstruction.
    fn roundtrip_pcm(
        &self,
        pcm: &[f32],
        num_quantizers: Option<usize>,
    ) -> Result<(RvqCodes, Vec<f32>)> {
        let codes = self.encode_pcm(pcm, num_quantizers)?;
        let recon = self.decode_codes(&codes)?;
        Ok((codes, recon))
    }
}

/// Stats from a file-bitstream compression pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressStats {
    /// Size of the compressed bitstream on disk.
    pub compressed_bytes: u64,
    /// Encode wall time (ms).
    pub encode_ms: f64,
    /// Decode wall time (ms); `0.0` for an encode-only call.
    pub decode_ms: f64,
}

impl CompressStats {
    /// Compression ratio vs an uncompressed PCM size (`raw_pcm_bytes / compressed`).
    pub fn compression_ratio(&self, raw_pcm_bytes: u64) -> f64 {
        if self.compressed_bytes == 0 {
            return 0.0;
        }
        raw_pcm_bytes as f64 / self.compressed_bytes as f64
    }

    /// Effective bitrate (bits/sec) for an audio clip of `duration_secs`.
    pub fn bitrate_bps(&self, duration_secs: f64) -> f64 {
        if duration_secs <= 0.0 {
            return 0.0;
        }
        self.compressed_bytes as f64 * 8.0 / duration_secs
    }
}

/// A **file-level** audio compressor: an audio file ⇄ an opaque compressed
/// bitstream on disk. Unlike [`AudioCodec`], there are no in-memory PCM buffers
/// or discrete RVQ codes — the compressed form is a self-contained file
/// (e.g. TSAC's transformer-entropy-coded `.tsac`). This is the home for codecs
/// that don't expose a frame/quantizer structure.
pub trait FileCodec {
    /// Backend this codec executes on.
    fn device(&self) -> Device;

    /// Native sample rate the codec targets.
    fn sample_rate(&self) -> u32;

    /// Compress `in_audio` → `out_compressed`, returning size + timing.
    fn encode_file(&self, in_audio: &Path, out_compressed: &Path) -> Result<CompressStats>;

    /// Decompress `in_compressed` → `out_wav`.
    fn decode_file(&self, in_compressed: &Path, out_wav: &Path) -> Result<()>;

    /// Encode then decode through a temporary compressed file. Codecs with a
    /// cheaper native roundtrip may override this.
    fn roundtrip_file(&self, in_audio: &Path, out_wav: &Path) -> Result<CompressStats> {
        let stem = in_audio
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio");
        let tmp =
            std::env::temp_dir().join(format!("rlx-filecodec-{}-{stem}.bin", std::process::id()));
        let enc = self.encode_file(in_audio, &tmp)?;
        let t0 = std::time::Instant::now();
        let dec = self.decode_file(&tmp, out_wav);
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let _ = std::fs::remove_file(&tmp);
        dec?;
        Ok(CompressStats {
            compressed_bytes: enc.compressed_bytes,
            encode_ms: enc.encode_ms,
            decode_ms,
        })
    }
}

/// Drives a whole-clip [`AudioCodec`] in a chunked / streaming loop: push PCM
/// chunks to get codes, push code chunks to get PCM. Each chunk is processed
/// independently (no carried causal state) — this is exactly how moshi/kyutai
/// drive Mimi today (whole-clip calls on one-frame inputs).
///
/// For truly artifact-free, minimal-latency incremental decoding a codec needs
/// internal state carry (causal-conv ring buffers + transformer KV); see
/// rlx-qwen3-tts's `StreamingDecoder` for the reference design. This wrapper
/// gives consumers a uniform streaming API over the codecs that only expose
/// whole-clip encode/decode.
pub struct ChunkStreamer<'a, C: AudioCodec + ?Sized> {
    codec: &'a C,
    num_quantizers: Option<usize>,
    frames_emitted: usize,
    samples_emitted: usize,
}

impl<'a, C: AudioCodec + ?Sized> ChunkStreamer<'a, C> {
    pub fn new(codec: &'a C, num_quantizers: Option<usize>) -> Self {
        Self {
            codec,
            num_quantizers,
            frames_emitted: 0,
            samples_emitted: 0,
        }
    }

    /// Pick the quantizer count from a target bitrate (bits/sec).
    pub fn with_bitrate(codec: &'a C, target_bps: f32) -> Self {
        let nq = codec.info().quantizers_for_bitrate(target_bps);
        Self::new(codec, Some(nq))
    }

    /// Encode the next PCM chunk to codes.
    pub fn encode_chunk(&mut self, pcm: &[f32]) -> Result<RvqCodes> {
        let codes = self.codec.encode_pcm(pcm, self.num_quantizers)?;
        self.frames_emitted += codes.num_frames();
        Ok(codes)
    }

    /// Decode the next code chunk to PCM.
    pub fn decode_chunk(&mut self, codes: &RvqCodes) -> Result<Vec<f32>> {
        let pcm = self.codec.decode_codes(codes)?;
        self.samples_emitted += pcm.len();
        Ok(pcm)
    }

    pub fn frames_emitted(&self) -> usize {
        self.frames_emitted
    }

    pub fn samples_emitted(&self) -> usize {
        self.samples_emitted
    }

    pub fn reset(&mut self) {
        self.frames_emitted = 0;
        self.samples_emitted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mimi_info() -> CodecInfo {
        CodecInfo {
            sample_rate: 24_000,
            frame_rate: 12.5,
            hop_length: 1920,
            channels: 1,
            max_quantizers: 32,
            codebook_size: 2048,
        }
    }

    #[test]
    fn bitrate_monotonic_in_quantizers() {
        let info = mimi_info();
        let b8 = info.bitrate_bps(Some(8));
        let b16 = info.bitrate_bps(Some(16));
        assert!(b16 > b8);
        // 8 codebooks × 11 bits × 12.5 Hz = 1100 bps.
        assert!((b8 - 1100.0).abs() < 1.0, "got {b8}");
    }

    #[test]
    fn quantizers_for_bitrate_roundtrips() {
        let info = mimi_info();
        for q in 1..=info.max_quantizers {
            let bps = info.bitrate_bps(Some(q));
            // Exactly-q bitrate should map back to q (or q within rounding).
            let got = info.quantizers_for_bitrate(bps);
            assert!(got == q || got == q.saturating_sub(1), "q={q} -> {got}");
        }
    }

    #[test]
    fn quantizers_for_bitrate_clamps() {
        let info = mimi_info();
        assert_eq!(info.quantizers_for_bitrate(0.0), 1);
        assert_eq!(info.quantizers_for_bitrate(1.0), 1);
        assert_eq!(info.quantizers_for_bitrate(1e9), info.max_quantizers);
    }

    #[test]
    fn compress_stats_ratio_and_bitrate() {
        let s = CompressStats {
            compressed_bytes: 1_000,
            encode_ms: 5.0,
            decode_ms: 3.0,
        };
        // 8000 raw bytes / 1000 compressed = 8x.
        assert!((s.compression_ratio(8_000) - 8.0).abs() < 1e-9);
        // 1000 bytes over 1 second = 8000 bits/sec.
        assert!((s.bitrate_bps(1.0) - 8_000.0).abs() < 1e-6);
        // Degenerate inputs don't panic.
        let z = CompressStats {
            compressed_bytes: 0,
            encode_ms: 0.0,
            decode_ms: 0.0,
        };
        assert_eq!(z.compression_ratio(1000), 0.0);
        assert_eq!(s.bitrate_bps(0.0), 0.0);
    }

    #[test]
    fn rvq_codes_transpose_roundtrip() {
        let codes = RvqCodes::new(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]], 3);
        let qm = codes.to_quantizer_major();
        assert_eq!(qm, vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]]);
        assert_eq!(RvqCodes::from_quantizer_major(&qm), codes);
    }
}
