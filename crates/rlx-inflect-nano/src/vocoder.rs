//! Host-eager Snake HiFi-GAN generator (`snake_v2mid`), weight-norm folded.
//! Ported from `train_hifigan_oracle_v1.py::HifiGanGenerator`.

use anyhow::Result;
use ndarray::Array2;

use crate::config::VocoderConfig;
use crate::ops::{conv_transpose1d, conv1d};
use crate::weights::Weights;

pub struct Vocoder<'a> {
    w: &'a Weights,
    cfg: &'a VocoderConfig,
}

fn get_padding(k: usize, d: usize) -> usize {
    (k * d - d) / 2
}

impl<'a> Vocoder<'a> {
    pub fn new(w: &'a Weights, cfg: &'a VocoderConfig) -> Self {
        Self { w, cfg }
    }

    fn d(&self, name: &str) -> &[f32] {
        self.w.data(name).expect(name)
    }

    /// Snake activation in place: `x + sin(αx)²/α`, α = exp(log_alpha).clamp(1e-4,100), per channel.
    fn snake_inplace(&self, x: &mut Array2<f32>, log_alpha_key: &str) {
        let log_alpha = self.d(log_alpha_key); // [ch]
        let (ch, t) = x.dim();
        for c in 0..ch {
            let alpha = log_alpha[c].exp().clamp(1e-4, 100.0);
            let inv = 1.0 / alpha;
            for j in 0..t {
                let v = x[[c, j]];
                let s = (alpha * v).sin();
                x[[c, j]] = v + s * s * inv;
            }
        }
    }

    fn resblock(
        &self,
        mut x: Array2<f32>,
        idx: usize,
        ch: usize,
        k: usize,
        dils: &[usize],
    ) -> Array2<f32> {
        for (j, &dl) in dils.iter().enumerate() {
            let mut y = x.clone();
            self.snake_inplace(&mut y, &format!("resblocks.{idx}.acts1.{j}.log_alpha"));
            let y = conv1d(
                &y,
                self.d(&format!("resblocks.{idx}.convs1.{j}.weight")),
                ch,
                ch,
                k,
                Some(self.d(&format!("resblocks.{idx}.convs1.{j}.bias"))),
                1,
                get_padding(k, dl),
                dl,
                1,
            );
            let mut y = y;
            self.snake_inplace(&mut y, &format!("resblocks.{idx}.acts2.{j}.log_alpha"));
            let y = conv1d(
                &y,
                self.d(&format!("resblocks.{idx}.convs2.{j}.weight")),
                ch,
                ch,
                k,
                Some(self.d(&format!("resblocks.{idx}.convs2.{j}.bias"))),
                1,
                get_padding(k, 1),
                1,
                1,
            );
            x += &y;
        }
        x
    }

    /// `mel: [80, T]` → waveform `[n_samples]`.
    pub fn forward(&self, mel: &Array2<f32>) -> Result<Vec<f32>> {
        let init = self.cfg.upsample_initial_channel;
        let num_kernels = self.cfg.resblock_kernel_sizes.len();
        let num_up = self.cfg.upsample_rates.len();

        let mut x = conv1d(
            mel,
            self.d("conv_pre.weight"),
            init,
            self.cfg.num_mels,
            7,
            Some(self.d("conv_pre.bias")),
            1,
            3,
            1,
            1,
        ); // [init, T]

        for i in 0..num_up {
            let in_ch = init / (1 << i);
            let out_ch = init / (1 << (i + 1));
            let k = self.cfg.upsample_kernel_sizes[i];
            let rate = self.cfg.upsample_rates[i];
            self.snake_inplace(&mut x, &format!("up_acts.{i}.log_alpha"));
            x = conv_transpose1d(
                &x,
                self.d(&format!("ups.{i}.weight")),
                in_ch,
                out_ch,
                k,
                Some(self.d(&format!("ups.{i}.bias"))),
                rate,
                (k - rate) / 2,
            );
            // sum of resblocks / num_kernels
            let mut acc: Option<Array2<f32>> = None;
            for j in 0..num_kernels {
                let idx = i * num_kernels + j;
                let kk = self.cfg.resblock_kernel_sizes[j];
                let dils = &self.cfg.resblock_dilation_sizes[j];
                let r = self.resblock(x.clone(), idx, out_ch, kk, dils);
                acc = Some(match acc {
                    None => r,
                    Some(a) => a + &r,
                });
            }
            x = acc.unwrap();
            x.mapv_inplace(|v| v / num_kernels as f32);
        }

        self.snake_inplace(&mut x, "post_act.log_alpha");
        let final_ch = init / (1 << num_up);
        let x = conv1d(
            &x,
            self.d("conv_post.weight"),
            1,
            final_ch,
            7,
            Some(self.d("conv_post.bias")),
            1,
            3,
            1,
            1,
        ); // [1, n]
        Ok(x.row(0).iter().map(|v| v.tanh()).collect())
    }
}
