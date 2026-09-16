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

//! Streaming and batched TEN-VAD sessions — the `ten_vad.h` surface, plus a
//! whole-clip path that scores a 30 s window per graph dispatch.

use anyhow::{Result, ensure};
use rlx_runtime::Device;

use crate::frontend::{Frontend, pre_emphasis};
use crate::model::{LstmState, Shape, TenVadModel};
use crate::weights::TenVadWeights;
use crate::{
    CONTEXT_FRAMES, DEFAULT_CHUNK_FRAMES, DEFAULT_RESET_FRAMES, DEFAULT_THRESHOLD, FEATURE_LEN,
    HOP_SIZE, SAMPLE_RATE,
};

/// Spectra are computed on int16-scaled samples; undo that for the energy report.
const POWER_NORM: f32 = 32768.0 * 32768.0;
/// Smallest hop `ten_vad_create` accepts.
const MIN_HOP: usize = 32;

/// Session configuration. [`Default`] matches `TenVad(hop_size=256, threshold=0.5)`.
#[derive(Debug, Clone)]
pub struct TenVadConfig {
    /// Samples handed to [`TenVad::process_i16`] per call. Any value ≥ 32; the
    /// analysis hop stays 256, so smaller values buffer and larger ones may
    /// advance the model several times per call.
    pub hop_size: usize,
    /// Voice is reported when `probability > threshold`.
    pub threshold: f32,
    /// Backend the network graph compiles for.
    pub device: Device,
    /// Frames between LSTM state resets; `0` disables resetting.
    pub reset_frames: usize,
}

impl Default for TenVadConfig {
    fn default() -> Self {
        Self {
            hop_size: HOP_SIZE,
            threshold: DEFAULT_THRESHOLD,
            device: Device::Cpu,
            reset_frames: DEFAULT_RESET_FRAMES,
        }
    }
}

/// One call's worth of output.
#[derive(Debug, Clone, Copy)]
pub struct TenVadFrame {
    /// Voice probability in `[0, 1]`, or `-1.0` before the first internal
    /// 256-sample frame completes (only reachable when `hop_size < 256`).
    pub probability: f32,
    /// `probability > threshold`.
    pub voice: bool,
    /// Estimated pitch in Hz, `0.0` when the frame reads as unvoiced.
    pub pitch_hz: f32,
    /// RMS of this call's samples, in int16 units.
    pub frame_rms: f32,
    /// Mean square of this call's samples, normalized to `[0, 1]`.
    pub frame_energy: f32,
}

impl TenVadFrame {
    /// Whether [`Self::probability`] holds a real score yet.
    pub fn scored(&self) -> bool {
        self.probability >= 0.0
    }
}

/// Frame-at-a-time TEN-VAD, mirroring `ten_vad_process`.
pub struct TenVad {
    cfg: TenVadConfig,
    frontend: Frontend,
    model: TenVadModel,
    state: LstmState,
    /// FIFO of raw (and pre-emphasised) samples, drained 256 at a time.
    raw: Vec<f32>,
    emphasised: Vec<f32>,
    filled: usize,
    pre_emphasis_prev: f32,
    score: f32,
    pitch_hz: f32,
    since_reset: usize,
    scratch: Vec<f32>,
}

impl TenVad {
    pub fn new(cfg: TenVadConfig) -> Result<Self> {
        Self::with_weights(cfg, TenVadWeights::embedded())
    }

    pub fn with_weights(cfg: TenVadConfig, weights: &TenVadWeights) -> Result<Self> {
        ensure!(
            cfg.hop_size >= MIN_HOP,
            "hop_size {} is below the minimum of {MIN_HOP}",
            cfg.hop_size
        );
        ensure!(
            (0.0..=1.0).contains(&cfg.threshold),
            "threshold {} is outside [0, 1]",
            cfg.threshold
        );
        crate::device::ensure_backend_ready(cfg.device)?;
        let capacity = cfg.hop_size + HOP_SIZE;
        Ok(Self {
            frontend: Frontend::new(weights.core()),
            model: TenVadModel::new(cfg.device, Shape::Streaming, weights)?,
            state: LstmState::default(),
            raw: vec![0.0; capacity],
            emphasised: vec![0.0; capacity],
            filled: 0,
            pre_emphasis_prev: 0.0,
            // `-1` marks "no score yet", exactly as the C does.
            score: -1.0,
            pitch_hz: 0.0,
            since_reset: 0,
            scratch: vec![0.0; cfg.hop_size],
            cfg,
        })
    }

    pub fn hop_size(&self) -> usize {
        self.cfg.hop_size
    }

    pub fn threshold(&self) -> f32 {
        self.cfg.threshold
    }

    pub fn device(&self) -> Device {
        self.cfg.device
    }

    /// Clear all filter, context and LSTM state.
    pub fn reset(&mut self) {
        self.frontend.reset();
        self.state.clear();
        self.raw.fill(0.0);
        self.emphasised.fill(0.0);
        self.filled = 0;
        self.pre_emphasis_prev = 0.0;
        self.score = -1.0;
        self.pitch_hz = 0.0;
        self.since_reset = 0;
    }

    /// Score one hop of int16 PCM.
    pub fn process_i16(&mut self, frame: &[i16]) -> Result<TenVadFrame> {
        ensure!(
            frame.len() == self.cfg.hop_size,
            "expected {} samples, got {}",
            self.cfg.hop_size,
            frame.len()
        );
        for (dst, &s) in self.scratch.iter_mut().zip(frame) {
            *dst = s as f32;
        }
        let samples = std::mem::take(&mut self.scratch);
        let out = self.process_scaled(&samples);
        self.scratch = samples;
        out
    }

    /// Score one hop of normalized `[-1, 1]` PCM.
    pub fn process_f32(&mut self, frame: &[f32]) -> Result<TenVadFrame> {
        ensure!(
            frame.len() == self.cfg.hop_size,
            "expected {} samples, got {}",
            self.cfg.hop_size,
            frame.len()
        );
        for (dst, &s) in self.scratch.iter_mut().zip(frame) {
            *dst = s * 32768.0;
        }
        let samples = std::mem::take(&mut self.scratch);
        let out = self.process_scaled(&samples);
        self.scratch = samples;
        out
    }

    /// Score one hop already in int16 units (`[-32768, 32767]` as `f32`).
    pub fn process_scaled(&mut self, frame: &[f32]) -> Result<TenVadFrame> {
        ensure!(
            frame.len() == self.cfg.hop_size,
            "expected {} samples, got {}",
            self.cfg.hop_size,
            frame.len()
        );
        let energy: f32 = frame.iter().map(|&s| s * s).sum();
        let rms = (energy / frame.len() as f32).sqrt();

        let n = frame.len();
        ensure!(self.filled + n <= self.raw.len(), "input FIFO overflow");
        self.raw[self.filled..self.filled + n].copy_from_slice(frame);
        pre_emphasis(
            frame,
            &mut self.pre_emphasis_prev,
            &mut self.emphasised[self.filled..self.filled + n],
        );
        self.filled += n;

        while self.filled >= HOP_SIZE {
            // Destructure so `frontend` and the two FIFOs are borrowed as
            // disjoint fields — `self.frontend.push(&self.raw[..], ..)` would
            // borrow all of `self`.
            let Self {
                frontend,
                raw,
                emphasised,
                ..
            } = self;
            let info = frontend.push(&raw[..HOP_SIZE], &emphasised[..HOP_SIZE]);
            self.pitch_hz = info.pitch.freq_hz;
            self.score = self.model.step(self.frontend.context(), &mut self.state)?;

            self.since_reset += 1;
            if self.cfg.reset_frames != 0 && self.since_reset >= self.cfg.reset_frames {
                self.state.clear();
                self.since_reset = 0;
            }

            self.raw.copy_within(HOP_SIZE.., 0);
            self.emphasised.copy_within(HOP_SIZE.., 0);
            self.filled -= HOP_SIZE;
        }

        Ok(TenVadFrame {
            probability: self.score,
            voice: self.score > self.cfg.threshold,
            pitch_hz: self.pitch_hz,
            frame_rms: rms,
            frame_energy: energy / POWER_NORM,
        })
    }

    /// Score a whole clip frame by frame, dropping any partial trailing hop.
    pub fn probabilities_i16(&mut self, pcm: &[i16]) -> Result<Vec<f32>> {
        pcm.chunks_exact(self.cfg.hop_size)
            .map(|c| self.process_i16(c).map(|f| f.probability))
            .collect()
    }
}

/// Chunked TEN-VAD: the DSP still runs per frame, but the network scores
/// `chunk_frames` frames per graph dispatch.
///
/// The LSTM state is carried across chunks (`Shape::Chunk`), so the result is
/// one continuous sequence regardless of chunk size — identical to scoring the
/// clip in one shot, and identical to [`TenVad`] frame by frame.
///
/// Chunk size is a latency/throughput dial: a chunk adds `chunk_frames · 16 ms`
/// before its scores are available, and buys a large amount of throughput on
/// GPU, where a single 16 ms frame is pure launch latency. Network-only RTF on
/// a 7.6 s clip:
///
/// | frames/dispatch | cpu | metal | mlx | wgpu |
/// |---|---|---|---|---|
/// | 1 | 448× | 37× | 20× | 20× |
/// | 8 (128 ms) | 450× | 414× | 134× | 87× |
/// | 32 (512 ms) | **573×** | **1020×** | **380×** | **141×** |
pub struct TenVadBatch {
    device: Device,
    weights: TenVadWeights,
    frontend: Frontend,
    chunk: usize,
    reset_frames: usize,
    model: TenVadModel,
    /// Frames scored since the last LSTM state reset.
    since_reset: usize,
}

impl TenVadBatch {
    /// [`DEFAULT_CHUNK_FRAMES`] frames per dispatch.
    ///
    /// Not the 30 s reset window: chunk size and reset period are independent
    /// now that state carries across chunks, and a huge chunk makes short clips
    /// pay for a mostly-padded dispatch (476 frames padded to 1875 is 4× the
    /// work). 128 frames is past the knee of the throughput curve while
    /// wasting at most ~2 s of compute on the tail.
    pub fn new(device: Device) -> Result<Self> {
        Self::with_chunk(device, DEFAULT_CHUNK_FRAMES)
    }

    /// `chunk_frames` frames per dispatch. 8–32 is the useful streaming range.
    pub fn with_chunk(device: Device, chunk_frames: usize) -> Result<Self> {
        Self::with_weights(device, TenVadWeights::embedded().clone(), chunk_frames)
    }

    pub fn with_weights(device: Device, weights: TenVadWeights, chunk: usize) -> Result<Self> {
        ensure!(chunk > 0, "chunk must hold at least one frame");
        crate::device::ensure_backend_ready(device)?;
        let frontend = Frontend::new(weights.core());
        let model = TenVadModel::new(device, Shape::Chunk(chunk), &weights)?;
        Ok(Self {
            device,
            weights,
            frontend,
            chunk,
            reset_frames: DEFAULT_RESET_FRAMES,
            model,
            since_reset: 0,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Frames per graph dispatch.
    pub fn chunk_frames(&self) -> usize {
        self.chunk
    }

    /// Frames between LSTM state resets; `0` disables resetting.
    pub fn set_reset_frames(&mut self, frames: usize) {
        self.reset_frames = frames;
    }

    /// Clear the frontend and the carried LSTM state.
    pub fn reset(&mut self) {
        self.frontend.reset();
        self.model.reset_state();
        self.since_reset = 0;
    }

    /// Per-frame probabilities for `pcm` (int16 units), one per 256 samples.
    ///
    /// Starts from a cleared state, so repeated calls are independent clips.
    pub fn probabilities(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.reset();
        let frames = pcm.len() / HOP_SIZE;
        let stride = CONTEXT_FRAMES * FEATURE_LEN;
        let mut feats = Vec::with_capacity(frames * stride);
        let mut emph = vec![0.0f32; HOP_SIZE];
        let mut prev = 0.0f32;
        for raw in pcm.chunks_exact(HOP_SIZE) {
            pre_emphasis(raw, &mut prev, &mut emph);
            self.frontend.push(raw, &emph);
            feats.extend_from_slice(self.frontend.context());
        }

        let mut probs = Vec::with_capacity(frames);
        let mut padded = vec![0.0f32; self.chunk * stride];
        for block in feats.chunks(self.chunk * stride) {
            let n = block.len() / stride;
            // The reset period is honoured on the frame grid, so it must not
            // land mid-chunk; `probabilities` uses chunks that divide it.
            if self.reset_frames != 0 && self.since_reset >= self.reset_frames {
                self.model.reset_state();
                self.since_reset = 0;
            }
            let scored = if n == self.chunk {
                self.model.run_batch(block)?
            } else {
                // Short tail: pad to the compiled shape and trim. The pad
                // frames pollute the carried state, which is harmless because
                // nothing follows them.
                padded[..block.len()].copy_from_slice(block);
                padded[block.len()..].fill(0.0);
                let mut out = self.model.run_batch(&padded)?;
                out.truncate(n);
                out
            };
            self.since_reset += n;
            probs.extend(scored);
        }
        Ok(probs)
    }

    /// Convenience wrapper for int16 input.
    pub fn probabilities_i16(&mut self, pcm: &[i16]) -> Result<Vec<f32>> {
        let scaled: Vec<f32> = pcm.iter().map(|&s| s as f32).collect();
        self.probabilities(&scaled)
    }

    /// The weights this session was built from.
    pub fn weights(&self) -> &TenVadWeights {
        &self.weights
    }
}

/// Frames a clip of `samples` produces.
pub fn frame_count(samples: usize) -> usize {
    samples / HOP_SIZE
}

/// Duration of one frame in seconds (16 ms).
pub fn frame_seconds() -> f64 {
    HOP_SIZE as f64 / SAMPLE_RATE as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, f0: f32) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                ((t * f0 * std::f32::consts::TAU).sin() * 6000.0) as i16
            })
            .collect()
    }

    #[test]
    fn streaming_matches_batched() {
        let pcm = tone(HOP_SIZE * 40, 180.0);
        let mut stream = TenVad::new(TenVadConfig::default()).unwrap();
        let streamed = stream.probabilities_i16(&pcm).unwrap();

        let mut batch = TenVadBatch::new(Device::Cpu).unwrap();
        let batched = batch.probabilities_i16(&pcm).unwrap();

        assert_eq!(streamed.len(), batched.len());
        for (i, (a, b)) in streamed.iter().zip(&batched).enumerate() {
            assert!((a - b).abs() < 1e-4, "frame {i}: {a} vs {b}");
        }
    }

    #[test]
    fn small_hop_buffers_until_a_frame_completes() {
        let mut vad = TenVad::new(TenVadConfig {
            hop_size: 64,
            ..Default::default()
        })
        .unwrap();
        let pcm = tone(64 * 8, 200.0);
        let mut scored = Vec::new();
        for chunk in pcm.chunks_exact(64) {
            scored.push(vad.process_i16(chunk).unwrap().scored());
        }
        // 256 / 64 = 4 calls before the first score.
        assert_eq!(&scored[..4], &[false, false, false, true]);
        assert!(scored[4..].iter().all(|&s| s));
    }

    #[test]
    fn large_hop_advances_several_frames_per_call() {
        let hop = HOP_SIZE * 3;
        let mut fast = TenVad::new(TenVadConfig {
            hop_size: hop,
            ..Default::default()
        })
        .unwrap();
        let mut slow = TenVad::new(TenVadConfig::default()).unwrap();
        let pcm = tone(hop * 4, 220.0);
        let mut last_fast = 0.0;
        for chunk in pcm.chunks_exact(hop) {
            last_fast = fast.process_i16(chunk).unwrap().probability;
        }
        let mut last_slow = 0.0;
        for chunk in pcm.chunks_exact(HOP_SIZE) {
            last_slow = slow.process_i16(chunk).unwrap().probability;
        }
        assert!(
            (last_fast - last_slow).abs() < 1e-5,
            "{last_fast} vs {last_slow}"
        );
    }

    #[test]
    fn reset_reproduces_the_first_run() {
        let pcm = tone(HOP_SIZE * 12, 150.0);
        let mut vad = TenVad::new(TenVadConfig::default()).unwrap();
        let first = vad.probabilities_i16(&pcm).unwrap();
        vad.reset();
        let second = vad.probabilities_i16(&pcm).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_a_wrong_length_frame() {
        let mut vad = TenVad::new(TenVadConfig::default()).unwrap();
        assert!(vad.process_i16(&[0i16; 100]).is_err());
    }

    #[test]
    fn rejects_a_tiny_hop() {
        let cfg = TenVadConfig {
            hop_size: 16,
            ..Default::default()
        };
        assert!(TenVad::new(cfg).is_err());
    }
}
