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

//! LuxTTS runner: 3 chained ONNX subgraphs + Rust glue mirroring the reference
//! `zipvoice/onnx_modeling.py::generate_cpu`. Voice cloning from a prompt wav +
//! its transcript; a 4-step anchor-ODE flow-matching sampler; the vocoder
//! spectral head (ONNX) + Rust ISTFT.

use std::path::Path;
#[cfg(feature = "onnx")]
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

use crate::config::{Layout, Tokens};
use crate::dsp::{self, N_MELS, VocosFbank};

#[cfg(feature = "onnx")]
use ort::{session::Session, value::Tensor};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A throwaway [`TinyModel`] config: only the onnx dir + graph cache are used
/// (we drive the subgraphs via `compile_named`/`run_typed`, never tiny-tts's
/// VITS glue), so these fields are irrelevant.
fn tiny_config() -> BundleConfig {
    BundleConfig {
        model: String::new(),
        sample_rate: dsp::SR,
        add_blank: true,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        length_scale: 1.0,
        inter_channels: N_MELS,
        gin_channels: N_MELS,
    }
}

pub const DEFAULT_NUM_STEP: usize = 4;
pub const DEFAULT_GUIDANCE: f32 = 3.0;
pub const DEFAULT_SPEED: f32 = 1.0;
pub const DEFAULT_T_SHIFT: f32 = 0.9;
pub const TARGET_RMS: f32 = 0.1;
pub const FEAT_SCALE: f32 = 0.1;

pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0, f32::max)
}

/// Deterministic Gaussian (xorshift128+ → Box–Muller).
struct Rng {
    s0: u64,
    s1: u64,
    spare: Option<f32>,
}
impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut n = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s0: n() | 1,
            s1: n() | 1,
            spare: None,
        }
    }
    fn u64(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        self.s0 = s0;
        s1 ^= s1 << 23;
        self.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.s1.wrapping_add(s0)
    }
    fn randn(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u = |x: u64| (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let mut u1 = u(self.u64());
        while u1 <= f64::MIN_POSITIVE {
            u1 = u(self.u64());
        }
        let u2 = u(self.u64());
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        self.spare = Some((r * th.sin()) as f32);
        (r * th.cos()) as f32
    }
}

/// Per-call synthesis options (reference defaults).
#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub num_step: usize,
    pub guidance_scale: f32,
    pub speed: f32,
    /// Internal speed multiplier applied to `speed` before the text encoder.
    /// LuxTTS uses 1.3 ("default too slow"); base ZipVoice uses 1.0.
    pub speed_mult: f32,
    pub t_shift: f32,
    pub seed: u64,
}
impl Default for InferOpts {
    fn default() -> Self {
        Self {
            num_step: DEFAULT_NUM_STEP,
            guidance_scale: DEFAULT_GUIDANCE,
            speed: DEFAULT_SPEED,
            speed_mult: 1.3,
            t_shift: DEFAULT_T_SHIFT,
            seed: 0,
        }
    }
}

/// `t_shift * τ / (1 + (t_shift-1) * τ)` over `linspace(0,1,num_step+1)`.
fn time_steps(num_step: usize, t_shift: f32) -> Vec<f32> {
    (0..=num_step)
        .map(|i| {
            let tau = i as f32 / num_step as f32;
            t_shift * tau / (1.0 + (t_shift - 1.0) * tau)
        })
        .collect()
}

/// Scale `wav` to `target` RMS; returns `(scaled, original_rms)`.
fn rms_norm(wav: &[f32], target: f32) -> (Vec<f32>, f32) {
    let ms = wav.iter().map(|x| x * x).sum::<f32>() / wav.len().max(1) as f32;
    let rms = ms.sqrt();
    let g = target / rms.max(1e-8);
    (wav.iter().map(|x| x * g).collect(), rms)
}

/// A loaded LuxTTS model. **Runs natively on RLX** — the encoder body, CFM
/// `fm_decoder`, and `vocoder_spec` subgraphs are imported to rlx-ir and
/// compiled per-device via [`TinyModel`] (no ONNX Runtime on the default path).
/// The derived-length token concat+pad and the scalar length regulator that the
/// original single `text_encoder` graph did internally are re-done in Rust
/// (see `synthesize`). `ort` is an opt-in `onnx` feature for parity validation.
pub struct LuxTts {
    /// Device for CFM (`encoder_body` + `fm_decoder`).
    device: Device,
    /// Device for `vocoder_spec` (same as [`Self::device`] today; kept so a
    /// future hybrid split does not churn the public API).
    voc_device: Device,
    tokens: Tokens,
    fbank: VocosFbank,
    /// native runner for `encoder_body.onnx` + `fm_decoder.onnx` (same dir).
    root_model: TinyModel,
    /// native runner for `vocoder_spec.onnx` (usually an `onnx/` subdir).
    voc_model: TinyModel,
    #[cfg(feature = "onnx")]
    text_encoder: Mutex<Session>,
    #[cfg(feature = "onnx")]
    fm_decoder_ort: Mutex<Session>,
    #[cfg(feature = "onnx")]
    vocoder_ort: Mutex<Session>,
}

impl LuxTts {
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        // End-to-end CoreML (`Device::Ane`) works when compute units avoid the
        // Neural-Engine BNNS path: default `CpuAndNeuralEngine` SIGSEGVs inside
        // `bnns::GraphCompile` on `encoder_body` / `fm_decoder`. CoreML GPU
        // (`RLX_COREML_UNITS=gpu`) or `all` compiles those graphs fine.
        let device = rlx_tiny_tts::resolve_tts_device(device);
        let voc_device = device;
        let layout = Layout::resolve(dir)?;
        let tokens = Tokens::load(&layout.dir)?;
        let fbank = VocosFbank::new();
        let enc_path = layout.encoder_body.as_ref().with_context(|| {
            format!(
                "encoder_body.onnx not found under {} — generate it with \
                 `python crates/rlx-luxtts/scripts/export_encoder_body.py \
                 {}/text_encoder.onnx {}/encoder_body.onnx`",
                layout.dir.display(),
                layout.dir.display(),
                layout.dir.display()
            )
        })?;
        let root_dir = enc_path.parent().unwrap_or(&layout.dir).to_path_buf();
        let voc_dir = layout
            .vocoder_spec
            .parent()
            .unwrap_or(&layout.dir)
            .to_path_buf();
        let root_model = TinyModel::new(root_dir, tiny_config());
        let voc_model = TinyModel::new(voc_dir, tiny_config());
        if device == voc_device {
            eprintln!("[luxtts] loaded 3 subgraphs on rlx-native/{device:?}");
        } else {
            eprintln!(
                "[luxtts] loaded subgraphs: CFM on rlx-native/{device:?}, \
                 vocoder on rlx-native/{voc_device:?}"
            );
        }

        #[cfg(not(feature = "onnx"))]
        {
            Ok(Self {
                device,
                voc_device,
                tokens,
                fbank,
                root_model,
                voc_model,
            })
        }
        #[cfg(feature = "onnx")]
        {
            // ORT sessions are only for synthesize_ort parity. CoreML EP often
            // fails on text_encoder — fall back those sessions to CPU.
            let ort_device = device;
            let build = |p: &Path| -> Result<(Session, String)> {
                let b = rlx_kittentts::build_onnx_session(p, ort_device)
                    .with_context(|| format!("session {}", p.display()))?;
                Ok((b.session, b.ort_ep))
            };
            let (text_encoder, _ep) = build(&layout.text_encoder)?;
            let (fm_decoder_ort, _) = build(&layout.fm_decoder)?;
            let (vocoder_ort, _) = build(&layout.vocoder_spec)?;
            Ok(Self {
                device,
                voc_device,
                tokens,
                fbank,
                root_model,
                voc_model,
                text_encoder: Mutex::new(text_encoder),
                fm_decoder_ort: Mutex::new(fm_decoder_ort),
                vocoder_ort: Mutex::new(vocoder_ort),
            })
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Device used for `vocoder_spec` (may differ from [`Self::device`] under
    /// CoreML hybrid).
    pub fn voc_device(&self) -> Device {
        self.voc_device
    }

    pub fn sample_rate(&self) -> u32 {
        dsp::SR
    }
    /// Native execution-provider label for CLI reporting.
    pub fn ort_ep(&self) -> String {
        if self.device == self.voc_device {
            format!("rlx-native/{:?}", self.device)
        } else {
            format!("rlx-native/cfm={:?}+voc={:?}", self.device, self.voc_device)
        }
    }

    /// Compile one native subgraph on `model` and run it, returning the first
    /// output as `f32`. `named` binds the graph's symbolic length dims.
    fn run1(
        model: &TinyModel,
        comp: &'static str,
        device: Device,
        length: usize,
        named: &[(&str, usize)],
        inputs: &[(&str, &[u8], DType)],
    ) -> Result<Vec<f32>> {
        let mut g = model
            .compile_named(comp, device, length, named)
            .map_err(|e| anyhow::anyhow!("compile {comp}: {e:#}"))?;
        let out = g.run_typed(inputs);
        let (bytes, _dt) = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("{comp}: no output"))?;
        Ok(as_f32(&bytes))
    }

    /// Clone `prompt_wav` (24 kHz mono) reading `prompt_text`, and speak `text`
    /// — **natively on RLX** (no ONNX Runtime). The single `text_encoder` graph
    /// is replaced by: Rust token concat+pad → native `encoder_body` → Rust
    /// length regulator; then the native CFM `fm_decoder` loop + native
    /// `vocoder_spec` + Rust ISTFT.
    #[cfg(feature = "espeak")]
    pub fn synthesize(
        &self,
        text: &str,
        prompt_wav: &[f32],
        prompt_text: &str,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        let dev = self.device;
        // 1. rms-norm prompt + extract its scaled log-mel [tp, 100].
        let (pw, prompt_rms) = rms_norm(prompt_wav, TARGET_RMS);
        let (mel, tp) = self.fbank.log_mel(&pw); // [100, tp] row-major
        anyhow::ensure!(tp > 0, "prompt audio too short");
        let prompt_feat = transpose_scale(&mel, N_MELS, tp, FEAT_SCALE); // [tp*100], time-major

        // 2. tokenize target + prompt transcript.
        let tokens = crate::tokenize::encode(text, &self.tokens, crate::tokenize::DEFAULT_LANG)?;
        let ptokens =
            crate::tokenize::encode(prompt_text, &self.tokens, crate::tokenize::DEFAULT_LANG)?;
        anyhow::ensure!(!tokens.is_empty(), "text tokenized to empty");
        // The graph substitutes an empty prompt with a single [0] token.
        let ptok = if ptokens.is_empty() { vec![0] } else { ptokens };
        let (tp_len, t_len) = (ptok.len(), tokens.len());

        // 3. native encoder body — input_ids = prompt ++ target ++ [pad 0].
        let mut input_ids = ptok;
        input_ids.extend_from_slice(&tokens);
        input_ids.push(0);
        let s = input_ids.len(); // tp_len + t_len + 1
        let enc = Self::run1(
            &self.root_model,
            "encoder_body",
            dev,
            s,
            &[("S", s)],
            &[("/Pad_output_0", &i64_bytes(&input_ids), DType::I64)],
        )?; // [1, S, 100] row-major

        // 4. Rust length regulator (repeat_interleave the S-1 real frames + tail
        //    of the last/pad frame; num_frames = the exported graph's formula).
        let speed_eff = (opts.speed * opts.speed_mult) as f64;
        let main_len = s - 1; // tp_len + t_len
        // Floor at `tp + 1` so a long prompt + short text (or high speed) never
        // collapses the CFM window to “prompt only” — that used to trip
        // `num_frames <= prompt` on matrix fox runs.
        let num_frames = ((tp as f64 / tp_len.max(1) as f64) * (tp_len + t_len) as f64
            / speed_eff.max(1e-6))
        .ceil() as usize;
        let num_frames = num_frames.max(tp + 1);
        anyhow::ensure!(num_frames > tp, "num_frames {num_frames} <= prompt {tp}");
        let repeat = (num_frames / main_len.max(1)).max(1);
        let mut text_condition = vec![0f32; num_frames * N_MELS];
        for f in 0..num_frames {
            let src = if f < repeat * main_len {
                f / repeat
            } else {
                s - 1
            };
            text_condition[f * N_MELS..(f + 1) * N_MELS]
                .copy_from_slice(&enc[src * N_MELS..(src + 1) * N_MELS]);
        }

        // 5. init noise + padded speech condition.
        let mut rng = Rng::new(opts.seed);
        let mut x: Vec<f32> = (0..num_frames * N_MELS).map(|_| rng.randn()).collect();
        let mut speech_cond = vec![0f32; num_frames * N_MELS];
        speech_cond[..tp * N_MELS].copy_from_slice(&prompt_feat);

        // 6. flow-matching loop — compile the CFM decoder once, run num_step×.
        let ts = time_steps(opts.num_step, opts.t_shift);
        let mut fm = self
            .root_model
            .compile_named("fm_decoder", dev, num_frames, &[("T", num_frames)])
            .map_err(|e| anyhow::anyhow!("compile fm_decoder: {e:#}"))?;
        for step in 0..opts.num_step {
            let (t_cur, t_next) = (ts[step], ts[step + 1]);
            let out = fm.run_typed(&[
                ("t", &f32_bytes(&[t_cur]), DType::F32),
                ("x", &f32_bytes(&x), DType::F32),
                ("text_condition", &f32_bytes(&text_condition), DType::F32),
                ("speech_condition", &f32_bytes(&speech_cond), DType::F32),
                (
                    "guidance_scale",
                    &f32_bytes(&[opts.guidance_scale]),
                    DType::F32,
                ),
            ]);
            let v = as_f32(&out.into_iter().next().context("fm_decoder: no output")?.0);
            for i in 0..x.len() {
                let x1 = x[i] + (1.0 - t_cur) * v[i];
                let x0 = x[i] - t_cur * v[i];
                x[i] = if step < opts.num_step - 1 {
                    (1.0 - t_next) * x0 + t_next * x1
                } else {
                    x1
                };
            }
        }

        // 7. drop prompt frames → generated mel [L,100] time-major; un-scale (÷0.1)
        //    and transpose to the vocoder layout [100,L].
        let l = num_frames - tp;
        let generated = &x[tp * N_MELS..]; // [L,100] time-major (frame, mel)
        let mut mel_lm = vec![0f32; N_MELS * l];
        for t in 0..l {
            for m in 0..N_MELS {
                mel_lm[m * l + t] = generated[t * N_MELS + m] / FEAT_SCALE;
            }
        }

        // 8. native vocoder spectral head → real/imag [513,L]; Rust ISTFT.
        //    Under CoreML hybrid, this is the only subgraph on Ane.
        let mut vg = self
            .voc_model
            .compile_named("vocoder_spec", self.voc_device, l, &[("l", l)])
            .map_err(|e| anyhow::anyhow!("compile vocoder_spec: {e:#}"))?;
        let vout = vg.run_typed(&[("mel", &f32_bytes(&mel_lm), DType::F32)]);
        anyhow::ensure!(vout.len() >= 2, "vocoder_spec must return real+imag");
        let real = as_f32(&vout[0].0);
        let imag = as_f32(&vout[1].0);
        let mut wav = dsp::istft(&real, &imag, l);

        // 9. volume match.
        if prompt_rms < TARGET_RMS {
            let g = prompt_rms / TARGET_RMS;
            for sm in &mut wav {
                *sm *= g;
            }
        }
        let peak = peak_amplitude(&wav);
        anyhow::ensure!(
            peak >= 1e-3,
            "synthesized audio is silent (peak={peak:.2e})"
        );
        Ok(wav)
    }

    /// ONNX-Runtime reference path (parity validation only). Mirrors the native
    /// `synthesize` op-for-op but drives the original single `text_encoder`
    /// graph + ort `fm_decoder`/`vocoder` sessions.
    #[cfg(all(feature = "onnx", feature = "espeak"))]
    pub fn synthesize_ort(
        &self,
        text: &str,
        prompt_wav: &[f32],
        prompt_text: &str,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        let (pw, prompt_rms) = rms_norm(prompt_wav, TARGET_RMS);
        let (mel, tp) = self.fbank.log_mel(&pw);
        anyhow::ensure!(tp > 0, "prompt audio too short");
        let prompt_feat = transpose_scale(&mel, N_MELS, tp, FEAT_SCALE);

        let tokens = crate::tokenize::encode(text, &self.tokens, crate::tokenize::DEFAULT_LANG)?;
        let ptokens =
            crate::tokenize::encode(prompt_text, &self.tokens, crate::tokenize::DEFAULT_LANG)?;
        anyhow::ensure!(!tokens.is_empty(), "text tokenized to empty");

        let (tc_shape, text_condition) = {
            let a = i64_t(&[1, tokens.len()], tokens.clone())?;
            let b = i64_t(
                &[1, ptokens.len().max(1)],
                if ptokens.is_empty() {
                    vec![0]
                } else {
                    ptokens.clone()
                },
            )?;
            let c = i64_scalar(tp as i64)?;
            let d = f32_scalar(opts.speed * opts.speed_mult)?;
            let mut s = self.text_encoder.lock().expect("poisoned");
            extract1(&s.run(ort::inputs![a, b, c, d]).context("text_encoder")?)?
        };
        let num_frames = tc_shape[1] as usize;
        anyhow::ensure!(num_frames > tp, "num_frames {num_frames} <= prompt {tp}");

        let mut rng = Rng::new(opts.seed);
        let mut x: Vec<f32> = (0..num_frames * N_MELS).map(|_| rng.randn()).collect();
        let mut speech_cond = vec![0f32; num_frames * N_MELS];
        speech_cond[..tp * N_MELS].copy_from_slice(&prompt_feat);

        let ts = time_steps(opts.num_step, opts.t_shift);
        for step in 0..opts.num_step {
            let (t_cur, t_next) = (ts[step], ts[step + 1]);
            let v = {
                let t = f32_scalar(t_cur)?;
                let xt = f32_t(&[1, num_frames, N_MELS], x.clone())?;
                let tc = f32_t(&[1, num_frames, N_MELS], text_condition.clone())?;
                let sc = f32_t(&[1, num_frames, N_MELS], speech_cond.clone())?;
                let g = f32_scalar(opts.guidance_scale)?;
                let mut s = self.fm_decoder_ort.lock().expect("poisoned");
                extract1(
                    &s.run(ort::inputs![t, xt, tc, sc, g])
                        .context("fm_decoder")?,
                )?
                .1
            };
            for i in 0..x.len() {
                let x1 = x[i] + (1.0 - t_cur) * v[i];
                let x0 = x[i] - t_cur * v[i];
                x[i] = if step < opts.num_step - 1 {
                    (1.0 - t_next) * x0 + t_next * x1
                } else {
                    x1
                };
            }
        }

        let l = num_frames - tp;
        let generated = &x[tp * N_MELS..];
        let mut mel_lm = vec![0f32; N_MELS * l];
        for t in 0..l {
            for m in 0..N_MELS {
                mel_lm[m * l + t] = generated[t * N_MELS + m] / FEAT_SCALE;
            }
        }

        let (real, imag) = {
            let m = f32_t(&[1, N_MELS, l], mel_lm)?;
            let mut s = self.vocoder_ort.lock().expect("poisoned");
            let out = s.run(ort::inputs![m]).context("vocoder")?;
            (
                out[0]
                    .try_extract_tensor::<f32>()
                    .context("real")?
                    .1
                    .to_vec(),
                out[1]
                    .try_extract_tensor::<f32>()
                    .context("imag")?
                    .1
                    .to_vec(),
            )
        };
        let mut wav = dsp::istft(&real, &imag, l);
        if prompt_rms < TARGET_RMS {
            let g = prompt_rms / TARGET_RMS;
            for sm in &mut wav {
                *sm *= g;
            }
        }
        Ok(wav)
    }

    /// Write mono 16-bit PCM WAV at 24 kHz.
    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: dsp::SR,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create {}", path.display()))?;
        for &s in audio {
            w.write_sample((s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                .context("wav write")?;
        }
        w.finalize().context("wav finalize")?;
        Ok(())
    }
}

/// `[rows, cols]` row-major, scaled, transposed to `[cols, rows]` row-major.
fn transpose_scale(src: &[f32], rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    // interpret src as [rows, cols]; output [cols, rows]
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c] * scale;
        }
    }
    out
}

#[cfg(feature = "onnx")]
fn f32_t(shape: &[usize], data: Vec<f32>) -> Result<Tensor<f32>> {
    Tensor::from_array((shape.to_vec(), data)).context("f32 tensor")
}
#[cfg(feature = "onnx")]
fn i64_t(shape: &[usize], data: Vec<i64>) -> Result<Tensor<i64>> {
    Tensor::from_array((shape.to_vec(), data)).context("i64 tensor")
}
#[cfg(feature = "onnx")]
fn f32_scalar(v: f32) -> Result<Tensor<f32>> {
    Tensor::from_array((Vec::<usize>::new(), vec![v])).context("f32 scalar")
}
#[cfg(feature = "onnx")]
fn i64_scalar(v: i64) -> Result<Tensor<i64>> {
    Tensor::from_array((Vec::<usize>::new(), vec![v])).context("i64 scalar")
}

/// Extract the first output as `(shape, data)`.
#[cfg(feature = "onnx")]
fn extract1(outputs: &ort::session::SessionOutputs) -> Result<(Vec<i64>, Vec<f32>)> {
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("extract output")?;
    Ok((shape.to_vec(), data.to_vec()))
}
