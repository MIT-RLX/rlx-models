use anyhow::{Context, Result};
use ndarray::Array2;

use crate::ops::{
    add_inplace, conv1d_same, embed_lookup, layer_norm, length_regulate, linear_in_out,
    multihead_self_attention, to_channels_first, to_time_major, view2,
};
use crate::weights::Weights;

const HIDDEN: usize = 256;
const HEADS: usize = 2;
const ATTN_DIM: usize = 512; // heads * head_dim with head_dim=256
const ATTN_SCALE: f32 = 0.0625; // 1/sqrt(256)
const LN_EPS: f32 = 1e-6;
const N_MELS: usize = 80;
const PHONE_VOCAB: usize = 84;
const PITCH_BUCKETS: usize = 256;
const DECODER_DILATIONS: [usize; 12] = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32];

/// Controllable variance scales (duration / pitch / energy). Defaults match
///
/// calibration). `pausev2` + `PhonewiseFeatureModule` supply per-phone
/// duration/pitch/energy overrides — wire those via
/// (`Neural phone feature duration: %s`).
#[derive(Debug, Clone)]
pub struct VarianceControls {
    pub duration_scale: f32,
    pub pitch_scale: f32,
    pub energy_scale: f32,
    pub duration_bias: f32,
    pub pitch_bias: f32,
    pub energy_bias: f32,
    /// `pause_min_duration` 70 ms @ hop 240 → 7 frames.
    pub pause_min_frames: usize,
    /// When set (len == phone count), replace the FS2 duration predictor
    pub phonewise_duration: Option<Vec<f32>>,
    pub phonewise_pitch: Option<Vec<f32>>,
    pub phonewise_energy: Option<Vec<f32>>,
}

impl Default for VarianceControls {
    fn default() -> Self {
        Self {
            duration_scale: 1.0,
            pitch_scale: 1.0,
            energy_scale: 1.0,
            duration_bias: 0.0,
            pitch_bias: 0.0,
            energy_bias: 0.0,
            pause_min_frames: 0,
            phonewise_duration: None,
            phonewise_pitch: None,
            phonewise_energy: None,
        }
    }
}

/// Word breaks (`#`) and punctuation keep predictor durations; applying
fn is_pause_marker_phone(id: usize) -> bool {
    id == 72
}

pub struct FastSpeech2<'a> {
    enc: &'a Weights,
    dec: &'a Weights,
}

impl<'a> FastSpeech2<'a> {
    pub fn new(enc: &'a Weights, dec: &'a Weights) -> Self {
        Self { enc, dec }
    }

    fn e(&self, name: &str) -> Result<&[f32]> {
        self.enc.data(name)
    }

    fn d(&self, name: &str) -> Result<&[f32]> {
        self.dec.data(name)
    }

    /// Phone ids → mel `[T, 80]`.
    pub fn infer(&self, phone_ids: &[usize], ctrl: &VarianceControls) -> Result<Array2<f32>> {
        let hid = self.encode(phone_ids, ctrl)?;
        self.decode(&hid)
    }

    pub fn encode(&self, phone_ids: &[usize], ctrl: &VarianceControls) -> Result<Array2<f32>> {
        anyhow::ensure!(!phone_ids.is_empty(), "empty phone id sequence");
        let embed = self.e("phone_embed.weight")?;
        let x_embed = embed_lookup(phone_ids, embed, PHONE_VOCAB, HIDDEN);
        let t = x_embed.dim().0;

        // Prenet: conv k=9 + ReLU → folded BN → + positional-encoding[:T].
        // net ending at `…/conv1d/Relu` matches ReLU (not linear) against GET.
        let prenet_w = self.e("encoder.prenet.weight")?;
        let prenet_b = self.e("encoder.prenet.bias")?;
        let x_cf = to_channels_first(&x_embed);
        let mut h_cf = conv1d_same(&x_cf, prenet_w, HIDDEN, HIDDEN, 9, Some(prenet_b), 1);
        if std::env::var_os("RLX_FS2_NO_PRENET_RELU").is_none() {
            for v in h_cf.iter_mut() {
                *v = v.max(0.0);
            }
        }
        let mut x = to_time_major(&h_cf);
        // Folded batch norm: y = x * mul + sub
        let mul = self.e("encoder.bn.mul")?;
        let sub = self.e("encoder.bn.sub")?;
        for mut row in x.rows_mut() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = *v * mul[j] + sub[j];
            }
        }
        // Add sinusoidal positional encoding [max_pos, HIDDEN] sliced to T.
        let pe = self.e("encoder.pos_enc")?;
        let pe = view2(pe, 256, HIDDEN);
        for i in 0..t {
            let prow = pe.row(i.min(255));
            let mut xr = x.row_mut(i);
            for j in 0..HIDDEN {
                xr[j] += prow[j];
            }
        }

        // 4 FFT blocks (post-norm)
        for i in 0..4 {
            x = self.fft_block(&x, i)?;
        }

        // Variance adaptors
        let mut dur = self.variance_predictor(&x, "duration")?;
        let mut pitch = self.variance_predictor(&x, "pitch")?;
        let mut energy = self.variance_predictor(&x, "energy")?;

        for v in dur.iter_mut() {
            *v = v.max(0.001);
        }
        for (v, s, b) in [
            (&mut dur, ctrl.duration_scale, ctrl.duration_bias),
            (&mut pitch, ctrl.pitch_scale, ctrl.pitch_bias),
            (&mut energy, ctrl.energy_scale, ctrl.energy_bias),
        ] {
            for x in v.iter_mut() {
                *x = *x * s + b;
            }
        }

        if let Some(ref pw) = ctrl.phonewise_duration {
            anyhow::ensure!(
                pw.len() == phone_ids.len(),
                "phonewise_duration len {} != phones {}",
                pw.len(),
                phone_ids.len()
            );
            for (d, &v) in dur.iter_mut().zip(pw.iter()) {
                *d = v.max(0.001);
            }
        }
        if let Some(ref pw) = ctrl.phonewise_pitch {
            anyhow::ensure!(
                pw.len() == phone_ids.len(),
                "phonewise_pitch length mismatch"
            );
            for (d, &v) in pitch.iter_mut().zip(pw.iter()) {
                *d = v;
            }
        }
        if let Some(ref pw) = ctrl.phonewise_energy {
            anyhow::ensure!(
                pw.len() == phone_ids.len(),
                "phonewise_energy length mismatch"
            );
            for (d, &v) in energy.iter_mut().zip(pw.iter()) {
                *d = v;
            }
        }

        // Identity diffs as the per-phone integers — not `round` of each float.
        let mut durs = Vec::with_capacity(phone_ids.len());
        let mut cum = 0.0f32;
        let mut prev = 0usize;
        for &v in &dur {
            cum += v.max(0.0);
            let c = cum.floor().max(0.0) as usize;
            durs.push(c.saturating_sub(prev));
            prev = c;
        }
        if durs.iter().sum::<usize>() == 0 {
            durs[0] = 1;
        }
        // Only `pau` (pause_marker), not `#` / `_` / `,`.
        if ctrl.pause_min_frames > 0 {
            for (i, &pid) in phone_ids.iter().enumerate() {
                if is_pause_marker_phone(pid) {
                    durs[i] = durs[i].max(ctrl.pause_min_frames);
                }
            }
        }
        if std::env::var_os("RLX_FS2_DEBUG_DURS").is_some() {
            eprintln!(
                "fs2 dur_raw[:20]={:?} durs={:?} sum={}",
                &dur[..dur.len().min(20)],
                durs,
                durs.iter().sum::<usize>()
            );
        }

        //   pitch_id = floor((log(pitch) - ln(40)) * 99.216) ∈ [0, 255]
        //   energy_id = floor((energy + 0.2) * 27.826) ∈ [0, 255]
        // Cast uses floor/ceil select (not round-half-up); for these positive
        // values that is truncating toward -∞.
        const LN40: f32 = 3.688_879_5;
        const PITCH_BUCKET_SCALE: f32 = 99.216_47;
        const ENERGY_BUCKET_SCALE: f32 = 27.826_088;
        const ENERGY_BUCKET_OFFSET: f32 = 0.2;
        let pitch_ids: Vec<usize> = pitch
            .iter()
            .map(|&v| {
                let v = v.max(1e-5);
                ((v.ln() - LN40) * PITCH_BUCKET_SCALE)
                    .floor()
                    .clamp(0.0, 255.0) as usize
            })
            .collect();
        let energy_ids: Vec<usize> = energy
            .iter()
            .map(|&v| {
                ((v + ENERGY_BUCKET_OFFSET) * ENERGY_BUCKET_SCALE)
                    .floor()
                    .clamp(0.0, 255.0) as usize
            })
            .collect();
        let pitch_e = embed_lookup(
            &pitch_ids,
            self.e("pitch_embed.weight")?,
            PITCH_BUCKETS,
            HIDDEN,
        );
        let energy_e = embed_lookup(
            &energy_ids,
            self.e("energy_embed.weight")?,
            PITCH_BUCKETS,
            HIDDEN,
        );
        add_inplace(&mut x, &pitch_e);
        add_inplace(&mut x, &energy_e);

        Ok(length_regulate(&x, &durs))
    }

    fn fft_block(&self, x: &Array2<f32>, i: usize) -> Result<Array2<f32>> {
        // Post-norm: x = LN0(x + attn(x)); x = LN1(x + ffn(x)).
        let ln0 = 2 * i;
        let ln1 = 2 * i + 1;
        let q = linear_in_out(
            x,
            self.e(&format!("encoder.block{i}.attn.q.weight"))?,
            HIDDEN,
            ATTN_DIM,
            None,
        );
        let k = linear_in_out(
            x,
            self.e(&format!("encoder.block{i}.attn.k.weight"))?,
            HIDDEN,
            ATTN_DIM,
            None,
        );
        let v = linear_in_out(
            x,
            self.e(&format!("encoder.block{i}.attn.v.weight"))?,
            HIDDEN,
            ATTN_DIM,
            None,
        );
        let ctx = multihead_self_attention(&q, &k, &v, HEADS, ATTN_SCALE);
        let mut attn = linear_in_out(
            &ctx,
            self.e(&format!("encoder.block{i}.attn.out.weight"))?,
            ATTN_DIM,
            HIDDEN,
            Some(self.e(&format!("encoder.block{i}.attn.out.bias"))?),
        );
        add_inplace(&mut attn, x);
        let x1 = self.ln(&attn, ln0)?;

        // FFN: Relu(conv1d k=9, 256→1024) → conv1d(k=1, 1024→256).
        // net ending at the FFN residual matches only with an explicit ReLU
        // after the first conv (same pattern as the prenet).
        let x1_cf = to_channels_first(&x1);
        let mut ff = conv1d_same(
            &x1_cf,
            self.e(&format!("encoder.block{i}.ffn0.weight"))?,
            1024,
            HIDDEN,
            9,
            Some(self.e(&format!("encoder.block{i}.ffn0.bias"))?),
            1,
        );
        if std::env::var_os("RLX_FS2_NO_FFN_RELU").is_none() {
            for v in ff.iter_mut() {
                *v = v.max(0.0);
            }
        }
        let ff = conv1d_same(
            &ff,
            self.e(&format!("encoder.block{i}.ffn1.weight"))?,
            HIDDEN,
            1024,
            1,
            Some(self.e(&format!("encoder.block{i}.ffn1.bias"))?),
            1,
        );
        let mut out = to_time_major(&ff);
        add_inplace(&mut out, &x1);
        self.ln(&out, ln1)
    }

    fn ln(&self, x: &Array2<f32>, idx: usize) -> Result<Array2<f32>> {
        let g = self.e(&format!("encoder.ln{idx}.gamma"))?;
        let b = self.e(&format!("encoder.ln{idx}.beta"))?;
        Ok(layer_norm(x, g, b, LN_EPS, false))
    }

    fn variance_predictor(&self, x: &Array2<f32>, name: &str) -> Result<Vec<f32>> {
        // matches only with ReLU after each conv (same as prenet / FFN).
        // Duration alone also has an explicit dense `Relu` after BiasAdd.
        let x_cf = to_channels_first(x);
        let mut h = conv1d_same(
            &x_cf,
            self.e(&format!("{name}.conv0.weight"))?,
            HIDDEN,
            HIDDEN,
            3,
            Some(self.e(&format!("{name}.conv0.bias"))?),
            1,
        );
        for v in h.iter_mut() {
            *v = v.max(0.0);
        }
        // LN after conv0: dur→8, pitch→9, energy→10; after conv1: 11/12/13.
        let (ln_a, ln_b) = match name {
            "duration" => (8, 11),
            "pitch" => (9, 12),
            "energy" => (10, 13),
            _ => (8, 11),
        };
        let mut h_tm = self.ln(&to_time_major(&h), ln_a)?;
        let mut h = conv1d_same(
            &to_channels_first(&h_tm),
            self.e(&format!("{name}.conv1.weight"))?,
            HIDDEN,
            HIDDEN,
            3,
            Some(self.e(&format!("{name}.conv1.bias"))?),
            1,
        );
        for v in h.iter_mut() {
            *v = v.max(0.0);
        }
        h_tm = self.ln(&to_time_major(&h), ln_b)?;
        let proj = linear_in_out(
            &h_tm,
            self.e(&format!("{name}.proj.weight"))?,
            HIDDEN,
            1,
            None,
        );
        let bias = self
            .e(&format!("{name}.bias"))
            .ok()
            .and_then(|b| b.first().copied())
            .unwrap_or(0.0);
        let mut out: Vec<f32> = proj.column(0).to_vec();
        for v in &mut out {
            *v += bias;
        }
        if name == "duration" {
            for v in &mut out {
                *v = v.max(0.0);
            }
        }
        Ok(out)
    }

    pub fn decode(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        // Sequential dilated conv stack with fused ReLU (`fused_activation_mode=0`),
        let mut h = to_channels_first(x);
        for (i, &dil) in DECODER_DILATIONS.iter().enumerate() {
            let w = self
                .d(&format!("decoder.dilated.{i}.weight"))
                .with_context(|| format!("decoder dilated {i}"))?;
            let b = self.d(&format!("decoder.dilated.{i}.bias"))?;
            h = conv1d_same(&h, w, HIDDEN, HIDDEN, 3, Some(b), dil);
            for v in h.iter_mut() {
                *v = v.max(0.0);
            }
        }
        let mel = conv1d_same(
            &h,
            self.d("decoder.mel.weight")?,
            N_MELS,
            HIDDEN,
            3,
            Some(self.d("decoder.mel.bias")?),
            1,
        );
        Ok(to_time_major(&mel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_controls_default() {
        let c = VarianceControls::default();
        assert_eq!(c.duration_scale, 1.0);
        assert_eq!(c.pitch_bias, 0.0);
    }

    #[test]
    fn pause_min_targets_pau_not_word_break() {
        assert!(is_pause_marker_phone(72)); // pau
        assert!(!is_pause_marker_phone(3)); // #
        assert!(!is_pause_marker_phone(5)); // ,
        assert!(!is_pause_marker_phone(0)); // _
        assert!(!is_pause_marker_phone(6)); // .
    }
}
