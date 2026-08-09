//! WaveRNN h448 vocoder (dual 256-way Gumbel).

use anyhow::Result;
use half::f16;
use ndarray::Array2;
use rlx_ir::{
    BnnsAesCtr128, NativeBnnsGumbelMax, Philox4x32, apple_bnns::NativeBnnsFullyConnected,
    argmax_f32, bnns_gumbel_from_uniform, gumbel_max_argmax_with, gumbel01_from_uniform,
};

use crate::ops::{f16_act, view2};
use crate::weights::Weights;

const HIDDEN: usize = 448;
const GATES: usize = 1344; // 3 * HIDDEN
const N_MELS: usize = 80;
const STEPS_PER_FRAME: usize = 120; // sub2: 120 steps × 2 samples = 240 hop
const SUB: usize = 224; // per-substream hidden (coarse / fine)
const SUB_GATES: usize = 672; // 3 * SUB

pub struct WaveRnn<'a> {
    w: &'a Weights,
}

/// RNG used for free-run `gumbel_max` when [`WaveRnnOpts::gumbel_noise`] is unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaveRnnRng {
    #[default]
    Bnns,
    /// Process-global BNNS `GumbelMax` activation.
    NativeBnns,
    /// Portable Philox4×32-10 (RLX-native alternative).
    Philox,
}

#[derive(Debug, Clone)]
pub struct WaveRnnOpts {
    /// When set, use these uniform `[0,1]` samples (length = n_samples * 512)
    /// for teacher-forced / deterministic parity. Otherwise sample with RNG.
    pub gumbel_noise: Option<Vec<f32>>,
    pub gumbel_alpha: f32,
    /// Free-run uses `argmax(logit + Gumbel(α,β))` with
    /// `Gumbel = −log(−log(α·U+β)+β)` — not `logit + β·Gumbel(0,1)`.
    /// Product NativeBnns Gumbel locks `0.01` (seed 16807, α=1).
    pub gumbel_beta: f32,
    /// Legacy scale for injected-noise / Philox `logit + temperature · Gumbel(0,1)`.
    /// Free-run BNNS ignores this and uses [`Self::gumbel_alpha`] / [`Self::gumbel_beta`].
    /// Teacher-force tests that inject uniforms typically set `1.0`.
    pub temperature: f32,
    /// Fix RNG seed for reproducible sampling when noise is not injected.
    /// Literal seed (including `0`) — not remapped to `0xDEADBEEF`.
    /// (`BNNSCreateRandomGenerator` → AES-CTR).
    /// For [`WaveRnnRng::Bnns`] + `Some(s)` this is
    /// `BNNSCreateRandomGeneratorWithSeed`.
    pub seed: Option<u64>,
    pub rng: WaveRnnRng,
    pub greedy: bool,
    pub inject_f16: bool,
    /// GRU f16 mode bits for coarse sub-stream (`bit0=z,1=r,2=n,3=y`).
    /// Product default: `0` (no post-nonlinear f16).
    pub gru_f16_coarse: Option<u32>,
    /// GRU f16 mode bits for fine sub-stream. Product default: `0`.
    pub gru_f16_fine: Option<u32>,
    /// f16 the coarse n-gate pre-activation (`ig + r·rg`) before `tanh`.
    pub gru_n_pre_coarse: bool,
    /// f16 the fine n-gate pre-activation before `tanh`.
    pub gru_n_pre_fine: bool,
    /// Teacher-force emitted PCM (length = n_samples). When set, coarse/fine
    /// come from this stream instead of Gumbel (state follows reference bits).
    pub force_pcm: Option<Vec<f32>>,
    /// When set, Gumbel samples from these instead of the local head (Gumbel
    /// stream identity probe).
    pub force_logits: Option<Vec<f32>>,
}

impl Default for WaveRnnOpts {
    fn default() -> Self {
        Self {
            gumbel_noise: None,
            gumbel_alpha: 1.0,
            gumbel_beta: 0.01,
            temperature: 0.01,
            seed: Some(0),
            rng: WaveRnnRng::Bnns,
            greedy: false,
            inject_f16: false,
            gru_f16_coarse: None,
            gru_f16_fine: None,
            gru_n_pre_coarse: false,
            gru_n_pre_fine: false,
            force_pcm: None,
            force_logits: None,
        }
    }
}

impl WaveRnnOpts {
    /// Product-path defaults: fused GRU cell + NativeBnns Gumbel on macOS.
    /// Bit feedback defaults to `(fine, 0)` for the coarse channel
    /// (`RLX_WR_COARSE_AR=1` restores dual-channel AR).
    /// Override with `RLX_WR_PORTABLE_BNNS=1` or set [`Self::rng`] explicitly.
    pub fn product_default() -> Self {
        let mut opts = Self::default();
        if std::env::var_os("RLX_WR_PORTABLE_BNNS").is_some() {
            opts.rng = WaveRnnRng::Bnns;
            return opts;
        }
        #[cfg(target_os = "macos")]
        {
            // Native BNNS process-global generator; product seed 16807.
            opts.rng = WaveRnnRng::NativeBnns;
            if opts.seed == Some(0) {
                opts.seed = Some(16_807);
            }
            // Product sampler: seed 16807, α=1, β=0.01; GRU mode-0 / no n_pre.
            opts.gumbel_beta = 0.01;
            opts.inject_f16 = false;
            opts.gru_f16_coarse = Some(0);
            opts.gru_f16_fine = Some(0);
            opts.gru_n_pre_coarse = false;
            opts.gru_n_pre_fine = false;
        }
        opts
    }

    /// Disable outer GRU f16 modes (mode-0 / no n_pre).
    pub fn no_gru_f16() -> Self {
        let mut opts = Self::product_default();
        opts.inject_f16 = false;
        opts.gru_f16_coarse = Some(0);
        opts.gru_f16_fine = Some(0);
        opts.gru_n_pre_coarse = false;
        opts.gru_n_pre_fine = false;
        opts
    }
}

/// Reused scratch for one WaveRNN utterance (avoids per-sample heap churn).
struct Scratch {
    i_mel: Vec<f32>,
    ig: Vec<f32>,
    rg: Vec<f32>,
    fine_ig: Vec<f32>,
    h_coarse: Vec<f32>,
    h_fine: Vec<f32>,
    head_hid: Vec<f32>,
    logits: Vec<f32>,
    hidden16: Vec<u16>,
    mel16: Vec<u16>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            i_mel: vec![0.0; GATES],
            ig: vec![0.0; GATES],
            rg: vec![0.0; GATES],
            fine_ig: vec![0.0; SUB_GATES],
            h_coarse: vec![0.0; SUB],
            h_fine: vec![0.0; SUB],
            head_hid: vec![0.0; SUB],
            logits: vec![0.0; 256],
            hidden16: vec![0; HIDDEN],
            mel16: vec![0; N_MELS],
        }
    }
}

impl<'a> WaveRnn<'a> {
    pub fn new(w: &'a Weights) -> Self {
        Self { w }
    }

    /// Mel `[T, 80]` → 24 kHz PCM in `[-1, 1]`.
    ///
    /// Sub2 subscale:
    /// - split GRU over coarse/fine halves of the 448-d state;
    /// - each step emits interleaved (coarse, fine) 8-bit samples;
    /// - fine input gates get `I_fine + coarse_scaled * scale_const`.
    ///
    /// Sampling uses BNNS AES-CTR (or Philox) `GumbelMax(α,β)`,
    /// not multinomial [`rlx_ir::Op::Sample`].
    pub fn infer(&self, mel: &Array2<f32>, opts: &WaveRnnOpts) -> Result<Vec<f32>> {
        let (n_frames, c) = mel.dim();
        anyhow::ensure!(c == N_MELS, "expected {N_MELS} mel bins, got {c}");
        let n_samples = n_frames * STEPS_PER_FRAME * 2;
        let mel_w = self.w.data("wavernn.mel_fc.weight")?;
        let mel_b = self.w.data("wavernn.mel_fc.bias")?;
        let bits_w = self.w.data("wavernn.bits_fc.weight")?;
        let hh_w = self.w.data("wavernn.gru_hh.weight")?;
        let hh_b = self.w.data("wavernn.gru_hh.bias")?;
        let inject = self.w.data("wavernn.scale_const")?; // [672]
        let c_fc0_w = self.w.data("wavernn.coarse_fc0.weight")?;
        let c_fc0_b = self.w.data("wavernn.coarse_fc0.bias")?;
        let c_fc1_w = self.w.data("wavernn.coarse_fc1.weight")?;
        let c_fc1_b = self.w.data("wavernn.coarse_fc1.bias")?;
        let f_fc0_w = self.w.data("wavernn.fine_fc0.weight")?;
        let f_fc0_b = self.w.data("wavernn.fine_fc0.bias")?;
        let f_fc1_w = self.w.data("wavernn.fine_fc1.weight")?;
        let f_fc1_b = self.w.data("wavernn.fine_fc1.bias")?;

        let mel_view = view2(mel_w, N_MELS, GATES);
        let bits_view = view2(bits_w, 2, GATES);
        let hh_view = view2(hh_w, HIDDEN, GATES);

        let mut h = vec![0.0f32; HIDDEN]; // [coarse(224), fine(224)]
        let mut prev_coarse = 0.0f32;
        let mut prev_fine = 0.0f32;
        let mut pcm = Vec::with_capacity(n_samples);
        let seed = opts.seed;
        let mut bnns = match seed {
            Some(s) => BnnsAesCtr128::new(s),
            None => BnnsAesCtr128::from_entropy(),
        };
        let mut philox = match seed {
            Some(s) => Philox4x32::new(s),
            None => Philox4x32::new(rlx_ir::bnns_entropy_seed()),
        };
        let mut native_bnns = (opts.rng == WaveRnnRng::NativeBnns)
            .then(|| NativeBnnsGumbelMax::new(seed.unwrap_or(16_807)))
            .flatten();
        let make_fc = |weights: &[f32], input, output, bias: Option<&[f32]>, relu| {
            let weights: Vec<u16> = weights
                .iter()
                .map(|&value| f16::from_f32(value).to_bits())
                .collect();
            NativeBnnsFullyConnected::new(&weights, input, output, bias, relu)
        };
        let use_native_fc =
            opts.rng == WaveRnnRng::NativeBnns && std::env::var_os("RLX_WR_PORTABLE_FC").is_none();
        let mut native_mel_fc = use_native_fc
            .then(|| make_fc(mel_w, N_MELS, GATES, Some(mel_b), false))
            .flatten();
        let mut native_bits_fc = use_native_fc
            .then(|| make_fc(bits_w, 2, GATES, None, false))
            .flatten();
        let mut native_hh_fc = use_native_fc
            .then(|| make_fc(hh_w, HIDDEN, GATES, Some(hh_b), false))
            .flatten();
        let mut native_coarse_head = use_native_fc
            .then(|| {
                Some((
                    make_fc(c_fc0_w, SUB, SUB, Some(c_fc0_b), true)?,
                    make_fc(c_fc1_w, SUB, 256, Some(c_fc1_b), false)?,
                ))
            })
            .flatten();
        let mut native_fine_head = use_native_fc
            .then(|| {
                Some((
                    make_fc(f_fc0_w, SUB, SUB, Some(f_fc0_b), true)?,
                    make_fc(f_fc1_w, SUB, 256, Some(f_fc1_b), false)?,
                ))
            })
            .flatten();
        let mut noise_i = 0usize;
        let alpha = opts.gumbel_alpha;
        let beta = opts.gumbel_beta.max(0.0);
        let inject_scale = opts.temperature.max(1e-8);
        let mut sample_bit = |logits: &[f32]| -> usize {
            if opts.greedy {
                return argmax_f32(logits);
            }
            if let Some(ref noise) = opts.gumbel_noise {
                return gumbel_max_argmax_with(logits, inject_scale, || {
                    let u = noise.get(noise_i).copied().unwrap_or(0.5);
                    noise_i += 1;
                    gumbel01_from_uniform(u)
                });
            }
            match opts.rng {
                WaveRnnRng::Bnns => bnns.bnns_gumbel_max_argmax(logits, alpha, beta),
                WaveRnnRng::NativeBnns => native_bnns
                    .as_mut()
                    .and_then(|sampler| sampler.argmax(logits, alpha, beta))
                    .unwrap_or_else(|| bnns.bnns_gumbel_max_argmax(logits, alpha, beta)),
                WaveRnnRng::Philox => gumbel_max_argmax_with(logits, 1.0, || {
                    bnns_gumbel_from_uniform(philox.next_f32(), alpha, beta)
                }),
            }
        };

        let mut scratch = Scratch::new();
        let flip_bits = std::env::var("RLX_WR_BITS_NOFLIP").is_err();
        let fixed_coarse = std::env::var("RLX_WR_FIXED_COARSE").is_ok();
        // product bit-feedback path
        // Coarse channel held at 0 in product bit-feedback.
        // Opt out with `RLX_WR_COARSE_AR=1` (or `RLX_WR_FIXED_COARSE=1`).
        let zero_coarse_bit = std::env::var_os("RLX_WR_COARSE_AR").is_none()
            && std::env::var_os("RLX_WR_FIXED_COARSE").is_none()
            && std::env::var_os("RLX_WR_PORTABLE_BNNS").is_none();

        for frame in 0..n_frames {
            let mel_row = mel.row(frame);
            // mel_fc computed once per frame (compute_every=120).
            for (dst, &v) in scratch.mel16.iter_mut().zip(mel_row.iter()) {
                *dst = f16::from_f32(v).to_bits();
            }
            let mut mel_buf = [0.0f32; N_MELS];
            for (dst, &v) in mel_buf.iter_mut().zip(mel_row.iter()) {
                *dst = v;
            }
            if let Some(fc) = native_mel_fc.as_mut() {
                if let Some(out) = fc.apply(&scratch.mel16) {
                    scratch.i_mel.copy_from_slice(&out[..GATES.min(out.len())]);
                } else {
                    mel_fc_portable(&mel_buf, mel_b, &mel_view, &mut scratch.i_mel);
                }
            } else {
                mel_fc_portable(&mel_buf, mel_b, &mel_view, &mut scratch.i_mel);
            }

            let frame_in_coarse = if zero_coarse_bit { 0.0 } else { prev_coarse };

            for _ in 0..STEPS_PER_FRAME {
                let coarse_bit = if zero_coarse_bit {
                    0.0
                } else if fixed_coarse {
                    frame_in_coarse
                } else {
                    prev_coarse
                };
                let (b0, b1) = if flip_bits {
                    (prev_fine, coarse_bit)
                } else {
                    (coarse_bit, prev_fine)
                };
                let b0 = if std::env::var_os("RLX_WR_BITS_NO_F16").is_some() {
                    b0
                } else {
                    f16_act(b0)
                };
                let b1 = if std::env::var_os("RLX_WR_BITS_NO_F16").is_some() {
                    b1
                } else {
                    f16_act(b1)
                };

                scratch.ig.copy_from_slice(&scratch.i_mel);
                let bits16 = [f16::from_f32(b0).to_bits(), f16::from_f32(b1).to_bits()];
                if let Some(fc) = native_bits_fc.as_mut() {
                    if let Some(bits_projection) = fc.apply(&bits16) {
                        for o in 0..GATES {
                            scratch.ig[o] += bits_projection[o];
                        }
                    } else {
                        for o in 0..GATES {
                            let col = bits_view.column(o);
                            scratch.ig[o] += b0 * col[0] + b1 * col[1];
                        }
                    }
                } else {
                    for o in 0..GATES {
                        let col = bits_view.column(o);
                        scratch.ig[o] += b0 * col[0] + b1 * col[1];
                    }
                }

                for (dst, &v) in scratch.hidden16.iter_mut().zip(h.iter()) {
                    *dst = f16::from_f32(v).to_bits();
                }
                if let Some(fc) = native_hh_fc.as_mut() {
                    if let Some(out) = fc.apply(&scratch.hidden16) {
                        scratch.rg.copy_from_slice(&out[..GATES.min(out.len())]);
                    } else {
                        hh_fc_portable(&h, hh_b, &hh_view, &mut scratch.rg);
                    }
                } else {
                    hh_fc_portable(&h, hh_b, &hh_view, &mut scratch.rg);
                }

                let coarse_mode = opts.gru_f16_coarse.unwrap_or_else(gru_f16_mode_default);
                gru_sub_into_mode(
                    &h[..SUB],
                    &scratch.ig[..SUB_GATES],
                    &scratch.rg[..SUB_GATES],
                    &mut scratch.h_coarse,
                    coarse_mode,
                    opts.gru_n_pre_coarse,
                );
                if !native_head_into(
                    &mut native_coarse_head,
                    &scratch.h_coarse,
                    &mut scratch.head_hid,
                    &mut scratch.logits,
                ) {
                    portable_head_into(
                        &scratch.h_coarse,
                        c_fc0_w,
                        c_fc0_b,
                        c_fc1_w,
                        c_fc1_b,
                        &mut scratch.head_hid,
                        &mut scratch.logits,
                    );
                }
                let step_i = pcm.len() / 2;
                if let Some(ref fl) = opts.force_logits {
                    let base = step_i * 512;
                    if base + 256 <= fl.len() {
                        scratch.logits.copy_from_slice(&fl[base..base + 256]);
                    }
                }
                let sampled_coarse = sample_bit(&scratch.logits);
                let coarse = if let Some(ref force) = opts.force_pcm {
                    let idx = pcm.len();
                    if idx < force.len() {
                        let target = force[idx];
                        (((target + 1.0) * 255.0 / 2.0).round() as i32).clamp(0, 255) as usize
                    } else {
                        sampled_coarse
                    }
                } else {
                    sampled_coarse
                };
                let coarse_scaled = if std::env::var_os("RLX_WR_SCALE_LEFT").is_some() {
                    (coarse as f32) * 2.0 / 255.0 - 1.0
                } else if std::env::var_os("RLX_WR_HALFBIN").is_some() {
                    ((coarse as f32) + 0.5) * (2.0 / 255.0) - 1.0
                } else {
                    (coarse as f32) * (2.0 / 255.0) - 1.0
                };
                let inject_f16 = opts.inject_f16 || std::env::var_os("RLX_WR_INJECT_F16").is_some();
                let coarse_inj = if inject_f16 {
                    f16_act(coarse_scaled)
                } else if std::env::var_os("RLX_WR_INJECT_LEFT").is_some() {
                    (coarse as f32) * 2.0 / 255.0 - 1.0
                } else {
                    coarse_scaled
                };
                let inject_w_f16 = std::env::var_os("RLX_WR_INJECT_W_F16").is_some();

                for j in 0..SUB_GATES {
                    let w = if inject_w_f16 {
                        f16_act(inject[j])
                    } else {
                        inject[j]
                    };
                    scratch.fine_ig[j] = scratch.ig[SUB_GATES + j] + coarse_inj * w;
                }
                if std::env::var_os("RLX_WR_FINE_IG_F16").is_some() {
                    for v in scratch.fine_ig.iter_mut() {
                        *v = f16_act(*v);
                    }
                }
                let fine_mode = opts.gru_f16_fine.unwrap_or_else(gru_f16_mode_default);
                gru_sub_into_mode(
                    &h[SUB..],
                    &scratch.fine_ig,
                    &scratch.rg[SUB_GATES..],
                    &mut scratch.h_fine,
                    fine_mode,
                    opts.gru_n_pre_fine,
                );
                if !native_head_into(
                    &mut native_fine_head,
                    &scratch.h_fine,
                    &mut scratch.head_hid,
                    &mut scratch.logits,
                ) {
                    portable_head_into(
                        &scratch.h_fine,
                        f_fc0_w,
                        f_fc0_b,
                        f_fc1_w,
                        f_fc1_b,
                        &mut scratch.head_hid,
                        &mut scratch.logits,
                    );
                }
                let step_i = pcm.len() / 2;
                if let Some(ref fl) = opts.force_logits {
                    let base = step_i * 512 + 256;
                    if base + 256 <= fl.len() {
                        scratch.logits.copy_from_slice(&fl[base..base + 256]);
                    }
                }
                let sampled_fine = sample_bit(&scratch.logits);
                // Coarse/fine not pushed yet: coarse@pcm.len(), fine@pcm.len()+1.
                let fine = if let Some(ref force) = opts.force_pcm {
                    let force_idx = pcm.len() + 1;
                    if force_idx < force.len() {
                        let target = force[force_idx];
                        (((target + 1.0) * 255.0 / 2.0).round() as i32).clamp(0, 255) as usize
                    } else {
                        sampled_fine
                    }
                } else {
                    sampled_fine
                };
                let fine_scaled = (fine as f32) * (2.0 / 255.0) - 1.0;

                h[..SUB].copy_from_slice(&scratch.h_coarse);
                h[SUB..].copy_from_slice(&scratch.h_fine);
                if std::env::var_os("RLX_WR_H_F16").is_some() {
                    for v in h.iter_mut() {
                        *v = f16_act(*v);
                    }
                }

                pcm.push(coarse_scaled);
                pcm.push(fine_scaled);
                if std::env::var_os("RLX_WR_PCM_F16").is_some() {
                    prev_coarse = f16_act(coarse_scaled);
                    prev_fine = f16_act(fine_scaled);
                } else {
                    prev_coarse = coarse_scaled;
                    prev_fine = fine_scaled;
                }
            }
        }
        Ok(pcm)
    }
}

fn mel_fc_portable(
    mel_row: &[f32],
    mel_b: &[f32],
    mel_view: &ndarray::ArrayView2<'_, f32>,
    out: &mut [f32],
) {
    for o in 0..GATES {
        let mut a = mel_b[o];
        let col = mel_view.column(o);
        for j in 0..N_MELS {
            a += f16_act(mel_row[j]) * col[j];
        }
        out[o] = a;
    }
}

fn hh_fc_portable(
    h: &[f32],
    hh_b: &[f32],
    hh_view: &ndarray::ArrayView2<'_, f32>,
    out: &mut [f32],
) {
    for o in 0..GATES {
        let mut a = hh_b[o];
        let col = hh_view.column(o);
        for j in 0..HIDDEN {
            a += f16_act(h[j]) * col[j];
        }
        out[o] = a;
    }
}

/// `y = f16_act(x) @ W + b` with `W: [inp, out]` row-major.
fn gemv_lpa(x: &[f32], w: &[f32], inp: usize, out: usize, b: Option<&[f32]>, y: &mut [f32]) {
    debug_assert_eq!(x.len(), inp);
    debug_assert_eq!(y.len(), out);
    debug_assert_eq!(w.len(), inp * out);
    for o in 0..out {
        let mut a = b.map(|bb| bb[o]).unwrap_or(0.0);
        for j in 0..inp {
            a += f16_act(x[j]) * w[j * out + o];
        }
        y[o] = a;
    }
}

fn portable_head_into(
    x: &[f32],
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    hid: &mut [f32],
    logits: &mut [f32],
) {
    gemv_lpa(x, w0, SUB, SUB, Some(b0), hid);
    for v in hid.iter_mut() {
        *v = v.max(0.0);
    }
    gemv_lpa(hid, w1, SUB, 256, Some(b1), logits);
}

fn native_head_into(
    layers: &mut Option<(NativeBnnsFullyConnected, NativeBnnsFullyConnected)>,
    x: &[f32],
    hid: &mut [f32],
    logits: &mut [f32],
) -> bool {
    let Some((fc0, fc1)) = layers.as_mut() else {
        return false;
    };
    let mut input = [0u16; SUB];
    for (dst, &v) in input.iter_mut().zip(x.iter()) {
        *dst = f16::from_f32(v).to_bits();
    }
    let Some(hidden) = fc0.apply(&input) else {
        return false;
    };
    for (dst, &v) in hid.iter_mut().zip(hidden.iter()) {
        *dst = v;
    }
    let mut hidden16 = [0u16; SUB];
    for (dst, &v) in hidden16.iter_mut().zip(hid.iter()) {
        *dst = f16::from_f32(v).to_bits();
    }
    let Some(out) = fc1.apply(&hidden16) else {
        return false;
    };
    logits[..out.len().min(256)].copy_from_slice(&out[..out.len().min(256)]);
    true
}

/// One sub-stream GRU step (product fused path).
#[allow(dead_code)]
fn gru_sub_into(prev: &[f32], ig: &[f32], rg: &[f32], out: &mut [f32]) {
    gru_sub_into_mode(prev, ig, rg, out, 0, false);
}

fn gru_f16_mode_default() -> u32 {
    0
}

fn gru_sub_into_mode(
    prev: &[f32],
    ig: &[f32],
    rg: &[f32],
    out: &mut [f32],
    mode: u32,
    n_pre_f16: bool,
) {
    debug_assert_eq!(prev.len(), SUB);
    debug_assert_eq!(out.len(), SUB);
    if crate::gru_rlx::eval_mode(prev, rg, ig, out, mode, n_pre_f16, false, false) {
        return;
    }
    for j in 0..SUB {
        let z = sigmoid_(ig[j] + rg[j]);
        let r = sigmoid_(ig[SUB + j] + rg[SUB + j]);
        let mut n_in = ig[2 * SUB + j] + r * rg[2 * SUB + j];
        if n_pre_f16 {
            n_in = f16_act(n_in);
        }
        let n = n_in.tanh();
        out[j] = z * prev[j] + (1.0 - z) * n;
    }
}

#[inline]
fn sigmoid_(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opts_default_seeded() {
        let o = WaveRnnOpts::default();
        assert_eq!(o.seed, Some(0));
        assert_eq!(o.rng, WaveRnnRng::Bnns);
        assert!(!o.greedy);
        assert!(o.gumbel_noise.is_none());
        assert_eq!(o.gumbel_alpha, 1.0);
        assert_eq!(o.gumbel_beta, 0.01);
        assert!(!o.inject_f16);
    }

    #[test]
    fn product_default_native_path() {
        #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
        let o = WaveRnnOpts::product_default();
        #[cfg(target_os = "macos")]
        {
            if std::env::var_os("RLX_WR_PORTABLE_BNNS").is_none() {
                assert_eq!(o.rng, WaveRnnRng::NativeBnns);
                assert_eq!(o.seed, Some(16_807));
                assert_eq!(o.gumbel_beta, 0.01);
                assert!(!o.inject_f16);
                assert_eq!(o.gru_f16_coarse, Some(0));
                assert_eq!(o.gru_f16_fine, Some(0));
                assert!(!o.gru_n_pre_coarse);
                assert!(!o.gru_n_pre_fine);
            }
        }
    }
}
