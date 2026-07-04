//! Host-eager `MicroFastSpeech.infer` — deterministic mel synthesis from token
//! ids. Ported op-for-op from `train_inflect_micro_fastspeech_v3_pitch.py`.

use anyhow::Result;
use ndarray::Array2;

use crate::config::{AcousticConfig, InferOpts};
use crate::ops::{
    GruWeights, bidirectional_gru, conv1d, layer_norm, linear, sigmoid_, silu_inplace, view2,
};
use crate::weights::Weights;

const LN_EPS: f32 = 1e-5;

pub struct Acoustic<'a> {
    w: &'a Weights,
    cfg: &'a AcousticConfig,
}

impl<'a> Acoustic<'a> {
    pub fn new(w: &'a Weights, cfg: &'a AcousticConfig) -> Self {
        Self { w, cfg }
    }

    fn d(&self, name: &str) -> &[f32] {
        self.w.data(name).expect(name)
    }

    /// LayerNorm → Linear(h,mid) → SiLU → Linear(mid,out), the `*_head` / proj pattern.
    fn mlp_head(&self, x: &Array2<f32>, prefix: &str, mid: usize, out: usize) -> Array2<f32> {
        let h = self.cfg.hidden;
        let y = layer_norm(
            x,
            self.d(&format!("{prefix}.0.weight")),
            self.d(&format!("{prefix}.0.bias")),
            LN_EPS,
        );
        let mut y = linear(
            &y,
            self.d(&format!("{prefix}.1.weight")),
            mid,
            h,
            Some(self.d(&format!("{prefix}.1.bias"))),
        );
        silu_inplace(&mut y);
        linear(
            &y,
            self.d(&format!("{prefix}.3.weight")),
            out,
            mid,
            Some(self.d(&format!("{prefix}.3.bias"))),
        )
    }

    /// Linear(in,mid) → SiLU → Linear(mid,out) projection (frame_proj / pitch_proj).
    fn mlp_proj(
        &self,
        x: &Array2<f32>,
        prefix: &str,
        inp: usize,
        mid: usize,
        out: usize,
    ) -> Array2<f32> {
        let mut y = linear(
            x,
            self.d(&format!("{prefix}.0.weight")),
            mid,
            inp,
            Some(self.d(&format!("{prefix}.0.bias"))),
        );
        silu_inplace(&mut y);
        linear(
            &y,
            self.d(&format!("{prefix}.2.weight")),
            out,
            mid,
            Some(self.d(&format!("{prefix}.2.bias"))),
        )
    }

    /// One ConvFFN block (encoder / decoder). `x: [T, hidden]`.
    fn conv_ffn(&self, x: &Array2<f32>, prefix: &str, ff_mid: usize) -> Array2<f32> {
        let h = self.cfg.hidden;
        let k = self.cfg.kernel_size;
        // depthwise conv (groups=h) → GLU gate → pointwise conv, on channel-major [h, T].
        let y = layer_norm(
            x,
            self.d(&format!("{prefix}.norm1.weight")),
            self.d(&format!("{prefix}.norm1.bias")),
            LN_EPS,
        );
        let y_cm = y.t().to_owned(); // [h, T]
        let dd = conv1d(
            &y_cm,
            self.d(&format!("{prefix}.depth.weight")),
            2 * h,
            1,
            k,
            Some(self.d(&format!("{prefix}.depth.bias"))),
            1,
            k / 2,
            1,
            h,
        ); // [2h, T]
        let t = dd.dim().1;
        let mut gated = Array2::<f32>::zeros((h, t));
        for c in 0..h {
            for j in 0..t {
                gated[[c, j]] = dd[[c, j]] * sigmoid_(dd[[h + c, j]]);
            }
        }
        let pt = conv1d(
            &gated,
            self.d(&format!("{prefix}.point.weight")),
            h,
            h,
            1,
            Some(self.d(&format!("{prefix}.point.bias"))),
            1,
            0,
            1,
            1,
        ); // [h, T]
        let mut x = x + &pt.t(); // residual, back to [T, h]

        // FFN: Linear(h, ff_mid) → SiLU → Linear(ff_mid, h)
        let y = layer_norm(
            &x,
            self.d(&format!("{prefix}.norm2.weight")),
            self.d(&format!("{prefix}.norm2.bias")),
            LN_EPS,
        );
        let mut y = linear(
            &y,
            self.d(&format!("{prefix}.ff.0.weight")),
            ff_mid,
            h,
            Some(self.d(&format!("{prefix}.ff.0.bias"))),
        );
        silu_inplace(&mut y);
        let y = linear(
            &y,
            self.d(&format!("{prefix}.ff.3.weight")),
            h,
            ff_mid,
            Some(self.d(&format!("{prefix}.ff.3.bias"))),
        );
        x += &y;
        x
    }

    fn encode(&self, phone: &[i64], tone: &[i64], lang: &[i64], speaker: i64) -> Array2<f32> {
        let h = self.cfg.hidden;
        let s = phone.len();
        let phone_emb = view2(self.d("phone.weight"), self.cfg.vocab_size, h);
        let tone_emb = view2(self.d("tone.weight"), self.cfg.tone_size, h);
        let lang_emb = view2(self.d("lang.weight"), self.cfg.lang_size, h);
        let mut x = Array2::<f32>::zeros((s, h));
        for i in 0..s {
            let pr = phone_emb.row(phone[i] as usize);
            let tr = tone_emb.row((tone[i] as usize).min(self.cfg.tone_size - 1));
            let lr = lang_emb.row((lang[i] as usize).min(self.cfg.lang_size - 1));
            for j in 0..h {
                x[[i, j]] = pr[j] + tr[j] + lr[j];
            }
        }
        // speaker_proj(speaker_emb) broadcast-added to every token.
        let spk_emb = view2(
            self.d("speaker.weight"),
            self.cfg.speaker_count,
            self.cfg.speaker_dim,
        );
        let spk_row = spk_emb.row(speaker as usize).to_owned();
        let spk_2d = spk_row
            .into_shape_with_order((1, self.cfg.speaker_dim))
            .unwrap();
        let spk_proj = linear(
            &spk_2d,
            self.d("speaker_proj.weight"),
            h,
            self.cfg.speaker_dim,
            Some(self.d("speaker_proj.bias")),
        );
        for i in 0..s {
            for j in 0..h {
                x[[i, j]] += spk_proj[[0, j]];
            }
        }
        for l in 0..self.cfg.encoder_layers {
            x = self.conv_ffn(&x, &format!("encoder.{l}"), h * 4);
        }
        x
    }

    pub fn infer(
        &self,
        phone: &[i64],
        tone: &[i64],
        lang: &[i64],
        speaker: i64,
        opts: &InferOpts,
    ) -> Result<Array2<f32>> {
        let h = self.cfg.hidden;
        let s = phone.len();
        let encoded = self.encode(phone, tone, lang, speaker);

        // predict_prosody
        let log_dur = self.mlp_head(&encoded, "duration_head", h, 1); // [S,1]
        let energy = self.mlp_head(&encoded, "energy_head", h / 2, 1);
        let bright = self.mlp_head(&encoded, "bright_head", h / 2, 1);
        let pitch = self.mlp_head(&encoded, "pitch_head", h, 2); // [S,2]

        // durations = round(expm1(log_dur).clamp(0,max)*scale).clamp_min(min)
        let mut durations = vec![0i64; s];
        for i in 0..s {
            let pd = (log_dur[[i, 0]]
                .exp_m1()
                .clamp(0.0, opts.max_duration as f32))
                * opts.effective_length_scale();
            let r = pd.round_ties_even() as i64;
            durations[i] = r.max(opts.min_duration);
        }

        // conditioned = encoded + energy_proj(energy*escale) + bright_proj(bright)
        let mut energy_s = Array2::<f32>::zeros((s, 1));
        let mut bright_s = Array2::<f32>::zeros((s, 1));
        let mut pitch_s = Array2::<f32>::zeros((s, 2));
        for i in 0..s {
            energy_s[[i, 0]] = energy[[i, 0]] * opts.energy_scale;
            bright_s[[i, 0]] = bright[[i, 0]];
            pitch_s[[i, 0]] = pitch[[i, 0]] * opts.pitch_scale;
            pitch_s[[i, 1]] = pitch[[i, 1]].clamp(0.0, 1.0);
        }
        let e_proj = linear(
            &energy_s,
            self.d("energy_proj.weight"),
            h,
            1,
            Some(self.d("energy_proj.bias")),
        );
        let b_proj = linear(
            &bright_s,
            self.d("bright_proj.weight"),
            h,
            1,
            Some(self.d("bright_proj.bias")),
        );
        let mut conditioned = encoded.clone();
        conditioned += &e_proj;
        conditioned += &b_proj;

        // length regulation + frame meta + local context + pitch expansion
        let total: i64 = durations.iter().sum();
        let t = (total as usize).min(self.cfg.max_frames).max(1);
        let token_count = s.max(1);
        let mut frames = Array2::<f32>::zeros((t, h));
        let mut frame_meta = Array2::<f32>::zeros((t, 8));
        let mut ctx = Array2::<f32>::zeros((t, 3 * h));
        let mut pitch_frame = Array2::<f32>::zeros((t, 2));
        let mut fi = 0usize;
        for i in 0..s {
            let dur = durations[i] as usize;
            if dur == 0 {
                continue;
            }
            let prev_i = i.saturating_sub(1);
            let next_i = (i + 1).min(s - 1);
            let token_pos = i as f32 / (token_count.max(2) - 1) as f32;
            let log1p_dur = (dur as f32).ln_1p() / 6.0;
            for j in 0..dur {
                if fi >= t {
                    break;
                }
                let rel = if dur <= 1 {
                    0.0
                } else {
                    j as f32 / (dur - 1) as f32
                };
                let inv_rel = 1.0 - rel;
                let center = 1.0 - (rel * 2.0 - 1.0).abs();
                frame_meta[[fi, 0]] = rel;
                frame_meta[[fi, 1]] = inv_rel;
                frame_meta[[fi, 2]] = center;
                frame_meta[[fi, 3]] = (rel * std::f32::consts::PI).sin();
                frame_meta[[fi, 4]] = (rel * std::f32::consts::PI).cos();
                frame_meta[[fi, 5]] = token_pos;
                frame_meta[[fi, 6]] = log1p_dur;
                frame_meta[[fi, 7]] = dur as f32 / 40.0;
                for c in 0..h {
                    frames[[fi, c]] = conditioned[[i, c]];
                    ctx[[fi, c]] = conditioned[[prev_i, c]];
                    ctx[[fi, h + c]] = conditioned[[i, c]];
                    ctx[[fi, 2 * h + c]] = conditioned[[next_i, c]];
                }
                pitch_frame[[fi, 0]] = pitch_s[[i, 0]];
                pitch_frame[[fi, 1]] = pitch_s[[i, 1]];
                fi += 1;
            }
        }

        let mut x = frames;
        x += &self.mlp_proj(&frame_meta, "frame_proj", 8, h, h);
        // add_local_context: Linear(3h,2h)→SiLU→Linear(2h,h)
        let mut lc = linear(
            &ctx,
            self.d("local_ctx.0.weight"),
            2 * h,
            3 * h,
            Some(self.d("local_ctx.0.bias")),
        );
        silu_inplace(&mut lc);
        let lc = linear(
            &lc,
            self.d("local_ctx.2.weight"),
            h,
            2 * h,
            Some(self.d("local_ctx.2.bias")),
        );
        x += &lc;

        // absolute frame position embedding
        let abs_emb = view2(self.d("abs_frame.weight"), self.cfg.abs_frame_bins, h);
        for f in 0..t {
            let pos = ((f * self.cfg.abs_frame_bins) / self.cfg.max_frames.max(1))
                .min(self.cfg.abs_frame_bins - 1);
            let row = abs_emb.row(pos);
            for c in 0..h {
                x[[f, c]] += row[c];
            }
        }

        // pitch projection
        if self.cfg.use_frame_pitch {
            x += &self.mlp_proj(&pitch_frame, "pitch_proj", 2, h, h);
        }

        // decoder blocks
        let ff_mid = h * self.cfg.decoder_ff_mult;
        for l in 0..self.cfg.decoder_layers {
            x = self.conv_ffn(&x, &format!("decoder.{l}"), ff_mid);
        }

        // bidirectional GRU residual: x = x + gru(x)
        let hid = h / 2;
        let fwd = GruWeights {
            w_ih: self.d("frame_gru.weight_ih_l0"),
            w_hh: self.d("frame_gru.weight_hh_l0"),
            b_ih: self.d("frame_gru.bias_ih_l0"),
            b_hh: self.d("frame_gru.bias_hh_l0"),
        };
        let rev = GruWeights {
            w_ih: self.d("frame_gru.weight_ih_l0_reverse"),
            w_hh: self.d("frame_gru.weight_hh_l0_reverse"),
            b_ih: self.d("frame_gru.bias_ih_l0_reverse"),
            b_hh: self.d("frame_gru.bias_hh_l0_reverse"),
        };
        let g = bidirectional_gru(&x, &fwd, &rev, hid, h);
        x += &g;

        // mel head → [T, 80], then transpose to channel-major [80, T]
        let mel_tc = self.mlp_head(&x, "mel_head", h, self.cfg.n_mels);
        let mel = mel_tc.t().to_owned(); // [80, T]

        // postnet: Conv1d(80,h,5,p2) Tanh Conv1d(h,h,5,p2) Tanh Conv1d(h,80,5,p2)
        let mut p = conv1d(
            &mel,
            self.d("postnet.0.weight"),
            h,
            self.cfg.n_mels,
            5,
            Some(self.d("postnet.0.bias")),
            1,
            2,
            1,
            1,
        );
        p.mapv_inplace(|v| v.tanh());
        let mut p = conv1d(
            &p,
            self.d("postnet.2.weight"),
            h,
            h,
            5,
            Some(self.d("postnet.2.bias")),
            1,
            2,
            1,
            1,
        );
        p.mapv_inplace(|v| v.tanh());
        let p = conv1d(
            &p,
            self.d("postnet.4.weight"),
            self.cfg.n_mels,
            h,
            5,
            Some(self.d("postnet.4.bias")),
            1,
            2,
            1,
            1,
        );

        let mut out = mel;
        out.scaled_add(self.cfg.postnet_scale, &p);
        Ok(out)
    }
}
