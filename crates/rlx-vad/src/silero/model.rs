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

use super::weights::SileroWeights;
use crate::ops::{conv1d_nchw, fill_reflect_pad_right, lstm_cell_step, sigmoid};

const STFT_K: usize = 128;
const STFT_STRIDE: usize = 64;
const STFT_PAD_RIGHT: usize = 32;
const STFT_OUT_CH: usize = 130;
const MAG_BINS: usize = 65;
const HIDDEN: usize = 128;

pub struct LstmState {
    pub h: [f32; HIDDEN],
    pub c: [f32; HIDDEN],
}

impl Default for LstmState {
    fn default() -> Self {
        Self {
            h: [0.0; HIDDEN],
            c: [0.0; HIDDEN],
        }
    }
}

/// One streaming step: `frame` is `[context || chunk]` normalized f32 mono @ 16 kHz.
pub fn forward_frame(
    w: &SileroWeights,
    frame: &[f32],
    state: &mut LstmState,
    scratch: &mut SileroScratch,
) -> f32 {
    let t_in = frame.len();
    let t_padded = t_in + STFT_PAD_RIGHT;
    let (head, tail) = scratch.stft_in.split_at_mut(t_in);
    head.copy_from_slice(frame);
    fill_reflect_pad_right(head, &mut tail[..STFT_PAD_RIGHT]);
    let t_stft = conv1d_nchw(
        &scratch.stft_in[..t_padded],
        1,
        t_padded,
        &w.stft_conv,
        STFT_OUT_CH,
        STFT_K,
        STFT_STRIDE,
        0,
        None,
        &mut scratch.stft_out,
    );

    for ti in 0..t_stft {
        for b in 0..MAG_BINS {
            let re = scratch.stft_out[b * t_stft + ti];
            let im = scratch.stft_out[(b + MAG_BINS) * t_stft + ti];
            scratch.spec[b * t_stft + ti] = (re * re + im * im).sqrt();
        }
    }

    let mut t_cur = t_stft;
    relu_conv(
        &scratch.spec[..MAG_BINS * t_cur],
        MAG_BINS,
        t_cur,
        &w.conv1_w,
        &w.conv1_b,
        128,
        3,
        1,
        1,
        &mut scratch.buf_a,
        &mut t_cur,
    );
    relu_conv(
        &scratch.buf_a[..128 * t_cur],
        128,
        t_cur,
        &w.conv2_w,
        &w.conv2_b,
        64,
        3,
        2,
        1,
        &mut scratch.buf_b,
        &mut t_cur,
    );
    relu_conv(
        &scratch.buf_b[..64 * t_cur],
        64,
        t_cur,
        &w.conv3_w,
        &w.conv3_b,
        64,
        3,
        2,
        1,
        &mut scratch.buf_a,
        &mut t_cur,
    );
    relu_conv(
        &scratch.buf_a[..64 * t_cur],
        64,
        t_cur,
        &w.conv4_w,
        &w.conv4_b,
        128,
        3,
        1,
        1,
        &mut scratch.buf_b,
        &mut t_cur,
    );

    let last = t_cur.saturating_sub(1);
    for ch in 0..HIDDEN {
        scratch.lstm_in[ch] = scratch.buf_b[ch * t_cur + last];
    }

    let h_prev = state.h;
    let c_prev = state.c;
    let mut h_new = [0.0f32; HIDDEN];
    let mut c_new = [0.0f32; HIDDEN];
    lstm_cell_step(
        &scratch.lstm_in,
        &h_prev,
        &c_prev,
        &w.lstm_w_ih,
        &w.lstm_w_hh,
        &w.lstm_b_ih,
        &w.lstm_b_hh,
        HIDDEN,
        HIDDEN,
        &mut h_new,
        &mut c_new,
        &mut scratch.lstm_gates,
    );
    state.h = h_new;
    state.c = c_new;

    for i in 0..HIDDEN {
        scratch.lstm_in[i] = state.h[i].max(0.0);
    }

    let mut logit = w.final_b[0];
    for i in 0..HIDDEN {
        logit += scratch.lstm_in[i] * w.final_w[i];
    }
    sigmoid(logit)
}

fn relu_conv(
    x: &[f32],
    in_ch: usize,
    t_in: usize,
    w: &[f32],
    b: &[f32],
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    out: &mut [f32],
    t_out: &mut usize,
) {
    *t_out = conv1d_nchw(x, in_ch, t_in, w, out_ch, k, stride, pad, Some(b), out);
    for v in out.iter_mut().take(out_ch * *t_out) {
        *v = (*v).max(0.0);
    }
}

pub struct SileroScratch {
    stft_in: Vec<f32>,
    stft_out: Vec<f32>,
    spec: Vec<f32>,
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    lstm_in: [f32; HIDDEN],
    lstm_gates: Vec<f32>,
}

impl SileroScratch {
    pub fn for_max_frame(max_frame: usize) -> Self {
        let t_padded = max_frame + STFT_PAD_RIGHT;
        let t_stft = (t_padded.saturating_sub(STFT_K)) / STFT_STRIDE + 1;
        let spec_sz = MAG_BINS * t_stft.max(1);
        let buf_sz = 128 * t_stft.max(1);
        Self {
            stft_in: vec![0.0; t_padded],
            stft_out: vec![0.0; STFT_OUT_CH * t_stft.max(1)],
            spec: vec![0.0; spec_sz],
            buf_a: vec![0.0; buf_sz.max(64 * 4)],
            buf_b: vec![0.0; buf_sz.max(64 * 4)],
            lstm_in: [0.0; HIDDEN],
            lstm_gates: vec![0.0; HIDDEN * 4],
        }
    }
}

#[cfg(all(test, feature = "silero"))]
mod tests {
    use super::*;
    use crate::silero::SileroWeights;

    #[test]
    fn jfk_first_frame_reference_prob() {
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/jfk/jfk_rust_speech.wav");
        if !wav.is_file() {
            return;
        }
        let (sr, pcm) = crate::load_wav_mono_f32(&wav).expect("wav");
        let pcm = if sr == crate::SAMPLE_RATE_16K {
            pcm
        } else {
            crate::resample_linear(&pcm, sr, crate::SAMPLE_RATE_16K)
        };
        let w = SileroWeights::embedded();
        let mut state = LstmState::default();
        let mut scratch = SileroScratch::for_max_frame(576);
        let mut frame = vec![0.0f32; 576];
        frame[64..64 + 512].copy_from_slice(&pcm[..512]);
        let p = forward_frame(&w, &frame, &mut state, &mut scratch);
        assert!((p - 0.805).abs() < 0.05, "expected ~0.805, got {p}");
    }
}
