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

//! `config.json` of the dual-AR checkpoints: S2-Pro and Audio8.
//! Playing speech while it is made: one output stream on the default device,
//! fed from a queue that generation pushes into.
//!
//! The device usually runs at another rate (48 kHz) than the codec (44.1 kHz).
//! [`Resampler`] carries its position from one pushed piece to the next, so
//! the pieces join as if they had been resampled whole: no seam at each push.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

/// Linear resampling across pushed pieces.
pub struct Resampler {
    step: f64,
    /// Position of the next output sample, in input samples, from `prev`.
    pos: f64,
    /// The last input sample of the previous piece.
    prev: Option<f32>,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        Self {
            step: from as f64 / to as f64,
            pos: 0.0,
            prev: None,
        }
    }

    /// The output for one more piece of input.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }
        // `x[i]` with `x[-1]` the previous piece's last sample.
        let at = |i: isize| -> f32 {
            if i < 0 {
                self.prev.unwrap_or(input[0])
            } else {
                input[i as usize]
            }
        };
        let origin = if self.prev.is_some() { -1.0 } else { 0.0 };
        let mut t = self.pos + origin;
        let last = input.len() as f64 - 1.0;
        while t <= last {
            let i = t.floor();
            let frac = (t - i) as f32;
            let (a, b) = (at(i as isize), at((i as isize + 1).min(last as isize)));
            out.push(a + (b - a) * frac);
            t += self.step;
        }
        // Next output relative to this piece's last sample.
        self.pos = t - last;
        self.prev = input.last().copied();
    }
}

/// One output stream draining a queue of samples at the device rate.
pub struct StreamPlayer {
    _stream: cpal::Stream,
    queue: Arc<Mutex<VecDeque<f32>>>,
    resampler: Resampler,
    /// Samples pushed (at the device rate) and the stream's sample rate.
    pushed: usize,
    out_rate: u32,
    /// Times the device found the queue empty after playback began.
    pub underruns: Arc<Mutex<usize>>,
    started: Arc<Mutex<bool>>,
}

impl StreamPlayer {
    /// A stream on the default output device for `src_rate` audio. It plays
    /// silence until [`Self::start`].
    pub fn open(src_rate: u32) -> Result<Self> {
        let device = cpal::default_host()
            .default_output_device()
            .context("no default output device")?;
        let supported = device
            .default_output_config()
            .context("default output config")?;
        let out_rate = supported.sample_rate().0;
        let channels = supported.channels() as usize;
        let cfg: cpal::StreamConfig = supported.config();
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let underruns = Arc::new(Mutex::new(0usize));
        let started = Arc::new(Mutex::new(false));
        let args = (queue.clone(), underruns.clone(), started.clone());
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build::<f32>(&device, &cfg, channels, args)?,
            SampleFormat::I16 => build::<i16>(&device, &cfg, channels, args)?,
            SampleFormat::U16 => build::<u16>(&device, &cfg, channels, args)?,
            other => bail!("unsupported output sample format {other:?}"),
        };
        stream.play().context("start the output stream")?;
        Ok(Self {
            _stream: stream,
            queue,
            resampler: Resampler::new(src_rate, out_rate),
            pushed: 0,
            out_rate,
            underruns,
            started,
        })
    }

    /// Queue more speech (at the source rate).
    pub fn push(&mut self, pcm: &[f32]) {
        let mut out = Vec::with_capacity(pcm.len() * 12 / 10 + 2);
        self.resampler.push(pcm, &mut out);
        self.pushed += out.len();
        lock(&self.queue).extend(out);
    }

    /// Begin playing what is queued.
    pub fn start(&self) {
        *lock(&self.started) = true;
    }

    /// Seconds queued and not yet played.
    pub fn buffered(&self) -> f64 {
        lock(&self.queue).len() as f64 / self.out_rate as f64
    }

    /// Start (if not yet) and block until everything queued has played. The
    /// queue running dry from here on is the end, not an underrun.
    pub fn finish(&self) {
        let before = *lock(&self.underruns);
        self.start();
        while !lock(&self.queue).is_empty() {
            std::thread::sleep(Duration::from_millis(20));
        }
        std::thread::sleep(Duration::from_millis(150));
        let mut u = lock(&self.underruns);
        *u = before.max((*u).saturating_sub(1));
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

type Shared = (
    Arc<Mutex<VecDeque<f32>>>,
    Arc<Mutex<usize>>,
    Arc<Mutex<bool>>,
);

fn build<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device,
    cfg: &cpal::StreamConfig,
    channels: usize,
    (queue, underruns, started): Shared,
) -> Result<cpal::Stream> {
    let mut was_playing = false;
    Ok(device.build_output_stream(
        cfg,
        move |out: &mut [T], _| {
            let on = *lock(&started);
            let mut q = lock(&queue);
            for frame in out.chunks_mut(channels.max(1)) {
                let s = if on { q.pop_front() } else { None };
                if on && s.is_none() && was_playing {
                    *lock(&underruns) += 1;
                    was_playing = false;
                }
                if s.is_some() {
                    was_playing = true;
                }
                let v = T::from_sample(s.unwrap_or(0.0));
                for c in frame.iter_mut() {
                    *c = v;
                }
            }
        },
        |e| eprintln!("output stream error: {e}"),
        None,
    )?)
}

#[cfg(test)]
mod tests {
    use super::Resampler;

    /// Pieces resample to what the whole input does.
    #[test]
    fn pieces_join_without_a_seam() {
        let x: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.013).sin()).collect();
        let mut whole = Vec::new();
        Resampler::new(44_100, 48_000).push(&x, &mut whole);
        let mut parts = Vec::new();
        let mut r = Resampler::new(44_100, 48_000);
        for piece in x.chunks(777) {
            r.push(piece, &mut parts);
        }
        assert_eq!(whole.len(), parts.len());
        let d = whole
            .iter()
            .zip(&parts)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(d < 1e-5, "max difference {d}");
    }
}
