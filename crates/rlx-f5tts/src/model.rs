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

//! F5-TTS runner: three chained f16 ONNX subgraphs mirroring the reference
//! `F5-TTS-ONNX-Inference.py`. `preprocess(audio, text_ids, max_duration)`
//! produces the conditioning; the transformer is looped `nfe` times (it does
//! classifier-free guidance + the ODE step internally); `decode` runs the
//! Vocos vocoder (ISTFT folded in) to 24 kHz audio. Everything numeric lives in
//! the ONNX — the Rust side is text tokenization + the duration estimate + the
//! loop.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_runtime::Device;

use crate::config::{HOP_LENGTH, Layout, SAMPLE_RATE, Vocab};

#[cfg(feature = "onnx")]
use half::f16;
#[cfg(feature = "onnx")]
use ort::{session::Session, value::Tensor};

pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter().filter(|s| s.is_finite()).map(|s| s.abs()).fold(0.0, f32::max)
}

/// Per-call options.
#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub nfe: usize,
    pub speed: f32,
}
impl Default for InferOpts {
    fn default() -> Self {
        Self { nfe: crate::config::DEFAULT_NFE, speed: 1.0 }
    }
}

/// A loaded F5-TTS model (preprocess / transformer / decode sessions).
pub struct F5Tts {
    device: Device,
    vocab: Vocab,
    ort_ep: String,
    #[cfg(feature = "onnx")]
    preprocess: Mutex<Session>,
    #[cfg(feature = "onnx")]
    transformer: Mutex<Session>,
    #[cfg(feature = "onnx")]
    decode: Mutex<Session>,
}

impl F5Tts {
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let layout = Layout::resolve(dir)?;
        let vocab = Vocab::load(&layout.dir)?;
        #[cfg(not(feature = "onnx"))]
        {
            let _ = (device, &layout, &vocab);
            anyhow::bail!("rlx-f5tts built without the `onnx` feature");
        }
        #[cfg(feature = "onnx")]
        {
            let build = |p: &Path| -> Result<(Session, String)> {
                let b = rlx_kittentts::build_onnx_session(p, device)
                    .with_context(|| format!("session {}", p.display()))?;
                Ok((b.session, b.ort_ep))
            };
            let (preprocess, ep) = build(&layout.preprocess)?;
            let (transformer, _) = build(&layout.transformer)?;
            let (decode, _) = build(&layout.decode)?;
            eprintln!("[f5tts] loaded on {device:?} (ep={ep})");
            Ok(Self {
                device,
                vocab,
                ort_ep: ep,
                preprocess: Mutex::new(preprocess),
                transformer: Mutex::new(transformer),
                decode: Mutex::new(decode),
            })
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
    pub fn ort_ep(&self) -> &str {
        &self.ort_ep
    }

    /// Clone the voice in `ref_audio` (24 kHz mono, reading `ref_text`) and speak
    /// `gen_text`.
    #[cfg(feature = "onnx")]
    pub fn synthesize(
        &self,
        gen_text: &str,
        ref_audio: &[f32],
        ref_text: &str,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        use crate::tokenize::{encode, text_len};

        let text_ids = encode(ref_text, gen_text, &self.vocab);
        anyhow::ensure!(!text_ids.is_empty(), "empty text");
        let n = ref_audio.len();

        // Duration estimate (reference formula).
        let ref_audio_len = (n / HOP_LENGTH + 1) as f64;
        let ref_tl = text_len(ref_text).max(1) as f64;
        let gen_tl = text_len(gen_text) as f64;
        let max_duration =
            (ref_audio_len + (ref_audio_len / ref_tl * gen_tl / opts.speed as f64)) as i64;

        // 1. preprocess → conditioning (7 outputs, all f16 except ref_signal_len).
        let (mut noise, rope_cos, rope_sin, cat_mel, cat_mel_drop, qk_empty, ref_signal_len) = {
            let audio: Vec<f16> = ref_audio.iter().map(|&x| f16::from_f32(x)).collect();
            let a = Tensor::<f16>::from_array(([1usize, 1, n], audio)).context("audio")?;
            let ti = Tensor::<i32>::from_array(([1usize, text_ids.len()], text_ids.clone())).context("text_ids")?;
            let md = Tensor::<i64>::from_array((Vec::<usize>::new(), vec![max_duration])).context("max_duration")?;
            let mut s = self.preprocess.lock().expect("poisoned");
            let out = s.run(ort::inputs![a, ti, md]).context("preprocess")?;
            (
                ef16(&out, 0)?,
                ef16(&out, 1)?,
                ef16(&out, 2)?,
                ef16(&out, 3)?,
                ef16(&out, 4)?,
                ef16(&out, 5)?,
                out[6].try_extract_tensor::<i64>().context("ref_signal_len")?.1[0],
            )
        };

        // 2. flow-matching loop — the transformer folds in CFG + the ODE step.
        for step in 0..opts.nfe {
            let denoised = {
                let noi = f16_t(&noise.0, noise.1.clone())?;
                let rc = f16_t(&rope_cos.0, rope_cos.1.clone())?;
                let rs = f16_t(&rope_sin.0, rope_sin.1.clone())?;
                let cm = f16_t(&cat_mel.0, cat_mel.1.clone())?;
                let cmd = f16_t(&cat_mel_drop.0, cat_mel_drop.1.clone())?;
                let qk = f16_t(&qk_empty.0, qk_empty.1.clone())?;
                let ts = Tensor::<i32>::from_array((Vec::<usize>::new(), vec![step as i32])).context("time_step")?;
                let mut s = self.transformer.lock().expect("poisoned");
                let out = s.run(ort::inputs![noi, rc, rs, cm, cmd, qk, ts]).context("transformer")?;
                ef16(&out, 0)?
            };
            noise = denoised;
        }

        // 3. decode → waveform.
        let wav = {
            let d = f16_t(&noise.0, noise.1.clone())?;
            let rl = Tensor::<i64>::from_array((Vec::<usize>::new(), vec![ref_signal_len])).context("ref_signal_len")?;
            let mut s = self.decode.lock().expect("poisoned");
            let out = s.run(ort::inputs![d, rl]).context("decode")?;
            let (_shape, data) = out[0].try_extract_tensor::<f16>().context("output_audio")?;
            data.iter().map(|h| h.to_f32()).collect::<Vec<f32>>()
        };

        let peak = peak_amplitude(&wav);
        anyhow::ensure!(peak >= 1e-3, "synthesized audio is silent (peak={peak:.2e})");
        Ok(wav)
    }

    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
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

#[cfg(feature = "onnx")]
fn f16_t(shape: &[usize], data: Vec<f16>) -> Result<Tensor<f16>> {
    Tensor::from_array((shape.to_vec(), data)).context("f16 tensor")
}

/// Extract output `i` as `(shape, f16 data)`.
#[cfg(feature = "onnx")]
fn ef16(out: &ort::session::SessionOutputs, i: usize) -> Result<(Vec<usize>, Vec<f16>)> {
    let (shape, data) = out[i].try_extract_tensor::<f16>().with_context(|| format!("output {i}"))?;
    Ok((shape.iter().map(|&d| d.max(0) as usize).collect(), data.to_vec()))
}
