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

//! ChatterBox runner: 4 ONNX graphs (speech_encoder, embed_tokens,
//! language_model, conditional_decoder) driving the T3 autoregressive loop.

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use half::f16;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use rlx_runtime::Device;
use tokenizers::Tokenizer;

use crate::common::{
    HEAD_DIM, N_HEADS, N_LAYERS, Rng, SAMPLE_RATE, START_SPEECH, START_TEXT, STOP_TEXT, SynthOpts,
    is_eos, resample, sample,
};

pub struct ChatterBox {
    embed: Mutex<Session>,
    lm: Mutex<Session>,
    speech_encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    ort_ep: String,
}

impl ChatterBox {
    /// Load from a dir with `onnx/{embed_tokens,language_model_q4f16,speech_encoder,
    /// conditional_decoder}.onnx` (+ `.onnx_data`) and `tokenizer.json`.
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        // These graphs crash ORT's CoreML EP (metal/mlx); the CPU EP is the
        // validated path, so fall back to it for GPU devices.
        let ort_device = if matches!(device, Device::Cpu) {
            device
        } else {
            eprintln!("[chatterbox] CoreML EP is unstable for this model; using CPU EP");
            Device::Cpu
        };
        let build = |name: &str| -> Result<(Session, String)> {
            let p = dir.join("onnx").join(name);
            let b = rlx_kittentts::build_onnx_session(&p, ort_device)
                .with_context(|| format!("load {}", p.display()))?;
            Ok((b.session, b.ort_ep))
        };
        let dbg = std::env::var_os("RLX_CB_DEBUG").is_some();
        let (embed, ort_ep) = build("embed_tokens.onnx")?;
        if dbg {
            eprintln!("[cb] embed_tokens loaded");
        }
        // The 30-layer LM crashes ORT's Level3 graph optimizer with external
        // data — load it with optimization disabled.
        let lm = {
            let p = dir.join("onnx").join("language_model_fp16.onnx");
            ort::session::Session::builder()
                .context("ort builder")?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)
                .context("opt level")?
                .commit_from_file(&p)
                .with_context(|| format!("load {}", p.display()))?
        };
        if dbg {
            eprintln!("[cb] language_model loaded");
        }
        let (speech_encoder, _) = build("speech_encoder.onnx")?;
        if dbg {
            eprintln!("[cb] speech_encoder loaded");
        }
        let (decoder, _) = build("conditional_decoder.onnx")?;
        if dbg {
            eprintln!("[cb] conditional_decoder loaded");
        }
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))?;
        Ok(Self {
            embed: Mutex::new(embed),
            lm: Mutex::new(lm),
            speech_encoder: Mutex::new(speech_encoder),
            decoder: Mutex::new(decoder),
            tokenizer,
            ort_ep,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
    pub fn ort_ep(&self) -> &str {
        &self.ort_ep
    }

    /// `embed_tokens(ids, position_ids, exaggeration)` → `[1, n, 1024]` flat.
    fn embed(&self, ids: &[i64], pos0: usize, exag: f32) -> Result<Vec<f32>> {
        let n = ids.len();
        let pos: Vec<i64> = (pos0..pos0 + n).map(|p| p as i64).collect();
        let mut s = self.embed.lock().unwrap();
        let out = s
            .run(ort::inputs![
                "input_ids" => Tensor::<i64>::from_array(([1usize, n], ids.to_vec()))?,
                "position_ids" => Tensor::<i64>::from_array(([1usize, n], pos))?,
                "exaggeration" => Tensor::<f32>::from_array(([1usize], vec![exag]))?
            ])
            .context("embed_tokens")?;
        Ok(out[0].try_extract_tensor::<f32>()?.1.to_vec())
    }

    /// Synthesize `text` in the voice of `reference` (PCM at `ref_sr`). 24 kHz PCM.
    pub fn synthesize(
        &self,
        text: &str,
        reference: &[f32],
        ref_sr: u32,
        opts: &SynthOpts,
    ) -> Result<Vec<f32>> {
        // 1) reference → speech_encoder conditioning
        let ref24 = resample(reference, ref_sr, SAMPLE_RATE);
        let (audio_features, af_len, audio_tokens, speaker_embeddings, speaker_features, mel_len) = {
            let mut s = self.speech_encoder.lock().unwrap();
            let n = ref24.len();
            let out = s
                .run(ort::inputs!["audio_values" => Tensor::<f32>::from_array(([1usize, n], ref24))?])
                .context("speech_encoder")?;
            let names: Vec<String> = out.keys().map(|k| k.to_string()).collect();
            let get = |name: &str| names.iter().position(|k| k == name).unwrap();
            let (af_shape, af) = out[get("audio_features")].try_extract_tensor::<f32>()?;
            let (_, at) = out[get("audio_tokens")].try_extract_tensor::<i64>()?;
            let (_, se) = out[get("speaker_embeddings")].try_extract_tensor::<f32>()?;
            let (sf_shape, sf) = out[get("speaker_features")].try_extract_tensor::<f32>()?;
            (
                af.to_vec(),
                af_shape[1] as usize,
                at.to_vec(),
                se.to_vec(),
                sf.to_vec(),
                sf_shape[1] as usize,
            )
        };

        // 2) prompt embeds = cat(audio_features, text_embeds, start_speech_embed)
        let text_ids: Vec<i64> = std::iter::once(START_TEXT)
            .chain(
                self.tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
                    .get_ids()
                    .iter()
                    .map(|&i| i as i64),
            )
            .chain(std::iter::once(STOP_TEXT))
            .collect();
        let text_embeds = self.embed(&text_ids, 0, opts.exaggeration)?;
        let start_embed = self.embed(&[START_SPEECH], text_ids.len(), opts.exaggeration)?;
        let mut inputs_embeds = audio_features;
        inputs_embeds.extend_from_slice(&text_embeds);
        inputs_embeds.extend_from_slice(&start_embed);
        let mut seq = af_len + text_ids.len() + 1;

        // 3) prefill + AR loop
        let (logits, mut kv) = self.lm_forward(&inputs_embeds, seq, None)?;
        let mut rng = Rng::new(opts.seed);
        let mut generated: Vec<i64> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::new();
        let mut next = sample(&logits[logits.len() - 8194..], &seen, opts, &mut rng);
        for _ in 0..opts.max_frames {
            if is_eos(next) {
                break;
            }
            generated.push(next);
            seen.insert(next);
            let emb = self.embed(&[next], seq, opts.exaggeration)?;
            seq += 1;
            let (lg, nkv) = self.lm_forward(&emb, seq, Some(kv))?;
            kv = nkv;
            next = sample(&lg[lg.len() - 8194..], &seen, opts, &mut rng);
        }
        anyhow::ensure!(!generated.is_empty(), "no speech tokens generated");

        // 4) conditional_decoder: prompt tokens (mel_len/2) + generated → waveform
        let prompt_len = (mel_len / 2).min(audio_tokens.len());
        let mut codes: Vec<i64> = audio_tokens[..prompt_len].to_vec();
        codes.extend_from_slice(&generated);
        let mut s = self.decoder.lock().unwrap();
        let out = s
            .run(ort::inputs![
                "speech_tokens" => Tensor::<i64>::from_array(([1usize, codes.len()], codes))?,
                "speaker_embeddings" => Tensor::<f32>::from_array(([1usize, 192], speaker_embeddings))?,
                "speaker_features" => Tensor::<f32>::from_array(([1usize, mel_len, 80], speaker_features))?
            ])
            .context("conditional_decoder")?;
        Ok(out[0].try_extract_tensor::<f32>()?.1.to_vec())
    }

    /// One `language_model` call; returns (logits, present KV as [k0,v0,k1,v1,…]).
    fn lm_forward(
        &self,
        inputs_embeds: &[f32],
        seq: usize,
        past: Option<Vec<(Vec<usize>, Vec<f16>)>>,
    ) -> Result<(Vec<f32>, Vec<(Vec<usize>, Vec<f16>)>)> {
        let step = inputs_embeds.len() / 1024;
        let mut s = self.lm.lock().unwrap();
        let mut ins: Vec<(Cow<'static, str>, SessionInputValue)> =
            Vec::with_capacity(2 + 2 * N_LAYERS);
        ins.push((
            "inputs_embeds".into(),
            SessionInputValue::from(Tensor::<f32>::from_array((
                [1usize, step, 1024],
                inputs_embeds.to_vec(),
            ))?),
        ));
        ins.push((
            "attention_mask".into(),
            SessionInputValue::from(Tensor::<i64>::from_array(([1usize, seq], vec![1i64; seq]))?),
        ));
        let empty = || (vec![1usize, N_HEADS, 0, HEAD_DIM], Vec::<f16>::new());
        for i in 0..N_LAYERS {
            let (ks, kd) = past
                .as_ref()
                .map(|p| p[2 * i].clone())
                .unwrap_or_else(empty);
            let (vs, vd) = past
                .as_ref()
                .map(|p| p[2 * i + 1].clone())
                .unwrap_or_else(empty);
            ins.push((
                format!("past_key_values.{i}.key").into(),
                SessionInputValue::from(kv_tensor(ks, kd)?),
            ));
            ins.push((
                format!("past_key_values.{i}.value").into(),
                SessionInputValue::from(kv_tensor(vs, vd)?),
            ));
        }
        let out = s.run(ins).context("language_model")?;
        let names: Vec<String> = out.keys().map(|k| k.to_string()).collect();
        let logits = out[names.iter().position(|k| k == "logits").unwrap()]
            .try_extract_tensor::<f32>()?
            .1
            .to_vec();
        let mut kv = Vec::with_capacity(2 * N_LAYERS);
        for i in 0..N_LAYERS {
            for part in ["key", "value"] {
                let idx = names
                    .iter()
                    .position(|k| k == &format!("present.{i}.{part}"))
                    .unwrap();
                let (sh, d) = out[idx].try_extract_tensor::<f16>()?;
                kv.push((
                    sh.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                    d.to_vec(),
                ));
            }
        }
        Ok((logits, kv))
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
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

/// Build an f16 KV tensor; the empty-prefill case (a 0 dimension) goes through
/// ndarray, since ort's raw-data path rejects zero-size dimensions.
fn kv_tensor(shape: Vec<usize>, data: Vec<f16>) -> Result<Tensor<f16>> {
    if shape.contains(&0) {
        let a = ndarray::ArrayD::<f16>::from_shape_vec(ndarray::IxDyn(&shape), data)
            .context("empty kv ndarray")?;
        Ok(Tensor::from_array(a)?)
    } else {
        Ok(Tensor::from_array((shape, data))?)
    }
}
