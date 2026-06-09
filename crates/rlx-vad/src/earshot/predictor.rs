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

//! Earshot default predictor (ported from pykeio/earshot, RLX BLAS matmul).

use super::weights;
use crate::ops::sigmoid;
use rlx_cpu::blas::sgemm;

pub struct Predictor {
    state: [f32; 128],
}

impl Default for Predictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Predictor {
    pub fn new() -> Self {
        Self { state: [0.0; 128] }
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub fn normalize(&self, features: &mut [f32]) {
        let w = weights::weights();
        let i_rms =
            1.0 / (features.iter().map(|x| x * x).sum::<f32>() / features.len() as f32).sqrt();
        for (i, v) in features.iter_mut().enumerate() {
            *v = w.norm_weight[i] * *v * i_rms;
        }
    }

    pub fn predict(&mut self, features: &[f32], buffer: &mut [f32]) -> f32 {
        let w = weights::weights();
        let (buffer1, buffer2) = buffer.split_at_mut(288);
        input_layer1(features, w, buffer1);
        input_layer2_3::<18, 9, false>(
            &buffer1[..288],
            w.layer2_kernel,
            w.layer2_weight,
            w.layer2_bias,
            &mut buffer2[..144],
        );
        input_layer2_3::<9, 5, true>(
            &buffer2[..144],
            w.layer3_kernel,
            w.layer3_weight,
            w.layer3_bias,
            &mut buffer1[..80],
        );
        mingru::<80>(
            &buffer1[..80],
            &self.state[..64],
            w.rnn1_weight,
            &mut buffer2[..128],
        );
        self.state[..64].copy_from_slice(&buffer2[..64]);
        mingru::<64>(
            &buffer2[..64],
            &self.state[64..128],
            w.rnn2_weight,
            &mut buffer1[..128],
        );
        self.state[64..128].copy_from_slice(&buffer1[..64]);
        output(w, &buffer2[..64], &buffer1[..64])
    }
}

fn input_layer1(features: &[f32], wt: &weights::ParsedWeights, output: &mut [f32]) {
    const NUM_FEATURES: usize = 40;
    const KERNEL_SIZE: usize = 3;
    const DEPTHWISE_NUM_FEATURES: usize = NUM_FEATURES - KERNEL_SIZE + 1;
    const OUT_CHANNELS: usize = 16;
    const POOL_KERNEL_SIZE: usize = 3;
    const POOL_STRIDE: usize = 2;
    const POOLED_COLS: usize = (DEPTHWISE_NUM_FEATURES - POOL_KERNEL_SIZE) / POOL_STRIDE + 1;

    output.fill(0.0);
    let mut row = [0.0_f32; DEPTHWISE_NUM_FEATURES];
    for ox in 0..DEPTHWISE_NUM_FEATURES {
        let mut sum = 0.0;
        for kh in 0..KERNEL_SIZE {
            for kw in 0..KERNEL_SIZE {
                let col = ox + kw;
                let input_idx = kh * NUM_FEATURES + col;
                sum += features[input_idx] * wt.layer1_kernel[kh * KERNEL_SIZE + kw];
            }
        }
        row[ox] = sum;
    }
    for c in 0..OUT_CHANNELS {
        let mut new_row = [0.0; DEPTHWISE_NUM_FEATURES];
        for ox in 0..DEPTHWISE_NUM_FEATURES {
            new_row[ox] = row[ox] * wt.layer1_weight[c] + wt.layer1_bias[c];
        }
        let out_row_offs = POOLED_COLS * c;
        for q in 0..POOLED_COLS {
            for x in 0..POOL_KERNEL_SIZE {
                let out_q = &mut output[out_row_offs + q];
                *out_q = (*out_q).max(new_row[q * POOL_STRIDE + x]);
            }
        }
    }
}

fn input_layer2_3<const IN_FEATURES: usize, const OUT_FEATURES: usize, const LAYER3: bool>(
    features: &[f32],
    kernel: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    const HORIZONTAL_KERNEL_SIZE: usize = 3;
    const STRIDE: usize = 2;
    const CHANNELS: usize = 16;

    output.fill(0.0);
    for ox in 0..OUT_FEATURES {
        let mut dw = [0.0; CHANNELS];
        for c in 0..CHANNELS {
            let mut sum = 0.0;
            for kw in 0..HORIZONTAL_KERNEL_SIZE {
                let ix = ox * STRIDE + kw;
                let ix = ix as isize - 1;
                if ix < 0 || ix >= IN_FEATURES as isize {
                    continue;
                }
                sum += features[c * IN_FEATURES + ix as usize]
                    * kernel[c * HORIZONTAL_KERNEL_SIZE + kw];
            }
            dw[c] = sum;
        }
        for oc in 0..CHANNELS {
            let mut ic = 0.0;
            for c in 0..CHANNELS {
                ic += dw[c] * weight[oc * CHANNELS + c];
            }
            let ptr = if !LAYER3 {
                &mut output[oc * OUT_FEATURES + ox]
            } else {
                &mut output[ox * CHANNELS + oc]
            };
            *ptr = (ic + bias[oc]).max(0.0);
        }
    }
}

fn mingru<const IN_DIM: usize>(features: &[f32], h: &[f32], weight: &[f32], out: &mut [f32]) {
    sgemm(weight, features, out, 128, IN_DIM, 1);
    for i in 0..64 {
        let g = (out[64 + i] * 0.25).clamp(0.0, 1.0);
        let v = &mut out[i];
        *v = (1.0 - g) * h[i] + g * *v;
    }
}

fn output(w: &weights::ParsedWeights, out_1: &[f32], out_2: &[f32]) -> f32 {
    let mut out = 0.0;
    for f in 0..64 {
        out += out_1[f] * w.output_weight[f];
    }
    for f in 0..64 {
        out += out_2[f] * w.output_weight[64 + f];
    }
    sigmoid(out)
}
