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

//! MOSS-TTS-Nano runner: 4 ONNX graphs (prefill / decode_step / fused local
//! sampled-frame / codec decode) driving a hierarchical autoregressive loop.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use rlx_runtime::Device;
use tokenizers::Tokenizer;

use crate::config::{CodecInfo, Manifest};

const HIDDEN: usize = 768;
const N_LAYERS: usize = 12;

/// Per-call synthesis options.
#[derive(Debug, Clone, Copy)]
pub struct SynthOpts {
    pub seed: u64,
    pub max_frames: Option<usize>,
}

impl Default for SynthOpts {
    fn default() -> Self {
        Self {
            seed: 0,
            max_frames: None,
        }
    }
}

/// Small deterministic RNG (SplitMix64) → uniform f32 in [0, 1).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct MossNano {
    prefill: Mutex<Session>,
    decode: Mutex<Session>,
    local: Mutex<Session>,
    codec: Mutex<Session>,
    tokenizer: Tokenizer,
    manifest: Manifest,
    codec_info: CodecInfo,
    ort_ep: String,
}

impl MossNano {
    /// Load from a directory laid out like the ONNX release:
    /// `moss_tts_{prefill,decode_step,local_fixed_sampled_frame}.onnx`,
    /// `codec/moss_audio_tokenizer_decode_full.onnx`, `tokenizer.model`,
    /// `browser_poc_manifest.json` (+ their `.data` external weights).
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        // The hierarchical AR graphs misbehave on ORT's CoreML EP (metal/mlx);
        // the CPU EP is the validated path, so fall back to it for GPU devices.
        let ort_device = if matches!(device, Device::Cpu) {
            device
        } else {
            eprintln!("[moss-nano] CoreML EP is unstable for this model; using CPU EP");
            Device::Cpu
        };
        let build = |p: PathBuf| -> Result<(Session, String)> {
            let b = rlx_kittentts::build_onnx_session(&p, ort_device)
                .with_context(|| format!("load onnx {}", p.display()))?;
            Ok((b.session, b.ort_ep))
        };
        let (prefill, ort_ep) = build(dir.join("moss_tts_prefill.onnx"))?;
        let (decode, _) = build(dir.join("moss_tts_decode_step.onnx"))?;
        let (local, _) = build(dir.join("moss_tts_local_fixed_sampled_frame.onnx"))?;
        let (codec, _) = build(dir.join("codec/moss_audio_tokenizer_decode_full.onnx"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("open tokenizer.json: {e}"))?;
        let manifest = Manifest::load(&dir.join("browser_poc_manifest.json"))?;
        Ok(Self {
            prefill: Mutex::new(prefill),
            decode: Mutex::new(decode),
            local: Mutex::new(local),
            codec: Mutex::new(codec),
            tokenizer,
            manifest,
            codec_info: CodecInfo::default(),
            ort_ep,
        })
    }

    pub fn voice_names(&self) -> Vec<String> {
        self.manifest.voice_names()
    }
    pub fn sample_rate(&self) -> u32 {
        self.codec_info.sample_rate
    }
    pub fn channels(&self) -> u16 {
        self.codec_info.channels
    }
    pub fn ort_ep(&self) -> &str {
        &self.ort_ep
    }

    fn encode_text(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect())
    }

    /// Build the `[1, seq, row_width]` prompt input_ids for voice-clone mode.
    fn build_input_ids(&self, text: &str, voice_codes: &[Vec<i32>]) -> Result<(Vec<i32>, usize)> {
        let c = &self.manifest.tts_config;
        let t = &self.manifest.prompt_templates;
        let rw = c.row_width();
        let pad = c.audio_pad_token_id;
        let mut rows: Vec<i32> = Vec::new();
        let mut push_text = |tok: i32, rows: &mut Vec<i32>| {
            rows.push(tok);
            rows.extend(std::iter::repeat(pad).take(rw - 1));
        };
        // section 1: user prompt prefix + audio_start
        for &tok in &t.user_prompt_prefix_token_ids {
            push_text(tok, &mut rows);
        }
        push_text(c.audio_start_token_id, &mut rows);
        // section 2: reference audio rows (user slot)
        for frame in voice_codes {
            anyhow::ensure!(
                frame.len() == c.n_vq,
                "voice frame width {} != n_vq {}",
                frame.len(),
                c.n_vq
            );
            rows.push(c.audio_user_slot_token_id);
            rows.extend_from_slice(frame);
        }
        // section 3: audio_end + after-reference template + gen text + assistant prefix + audio_start
        push_text(c.audio_end_token_id, &mut rows);
        for &tok in &t.user_prompt_after_reference_token_ids {
            push_text(tok, &mut rows);
        }
        for tok in self.encode_text(text)? {
            push_text(tok, &mut rows);
        }
        for &tok in &t.assistant_prompt_prefix_token_ids {
            push_text(tok, &mut rows);
        }
        push_text(c.audio_start_token_id, &mut rows);

        let seq = rows.len() / rw;
        Ok((rows, seq))
    }

    /// Synthesize `text` in a builtin `voice`. Returns interleaved-stereo f32 @ 48 kHz.
    pub fn synthesize(&self, text: &str, voice: &str, opts: &SynthOpts) -> Result<Vec<f32>> {
        let voice = self
            .manifest
            .voice(voice)
            .with_context(|| format!("unknown voice '{voice}'; have {:?}", self.voice_names()))?;
        let voice_codes = voice.prompt_audio_codes.clone();
        let codes = self.generate_codes(text, &voice_codes, opts)?;
        self.decode_codes(&codes)
    }

    /// Run the AR loop → `[n_frames][n_vq]` audio codes.
    fn generate_codes(
        &self,
        text: &str,
        voice_codes: &[Vec<i32>],
        opts: &SynthOpts,
    ) -> Result<Vec<Vec<i32>>> {
        let c = self.manifest.tts_config.clone();
        let rw = c.row_width();
        let (input_ids, seq) = self.build_input_ids(text, voice_codes)?;
        let max_frames = opts
            .max_frames
            .unwrap_or(self.manifest.generation_defaults.max_new_frames);

        // ---- prefill ----
        let (mut hidden, mut kv) = {
            let mut s = self.prefill.lock().unwrap();
            let ids = Tensor::<i32>::from_array(([1, seq, rw], input_ids)).context("input_ids")?;
            let mask = Tensor::<i32>::from_array(([1, seq], vec![1i32; seq])).context("mask")?;
            let out = s
                .run(ort::inputs!["input_ids" => ids, "attention_mask" => mask])
                .context("prefill")?;
            let (hshape, hdata) = out[0]
                .try_extract_tensor::<f32>()
                .context("global_hidden")?;
            let hidden = last_row(hdata, &hshape); // [HIDDEN]
            let mut kv = Vec::with_capacity(2 * N_LAYERS);
            for i in 1..=2 * N_LAYERS {
                let (shape, data) = out[i]
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("kv {i}"))?;
                kv.push((
                    shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                    data.to_vec(),
                ));
            }
            (hidden, kv)
        };

        let mut rng = Rng::new(opts.seed);
        let mut rep_mask = vec![0i32; c.n_vq * 1024];
        let mut frames: Vec<Vec<i32>> = Vec::new();
        let mut past_len = seq as i32;

        for _ in 0..max_frames {
            // ---- fused local sampled frame ----
            let (should, frame) = {
                let mut s = self.local.lock().unwrap();
                let h =
                    Tensor::<f32>::from_array(([1, HIDDEN], hidden.clone())).context("hidden")?;
                let rm = Tensor::<i32>::from_array(([1, c.n_vq, 1024], rep_mask.clone()))
                    .context("rep_mask")?;
                let aru = Tensor::<f32>::from_array(([1], vec![rng.uniform()])).context("aru")?;
                let au: Vec<f32> = (0..c.n_vq).map(|_| rng.uniform()).collect();
                let aut = Tensor::<f32>::from_array(([1, c.n_vq], au)).context("au")?;
                let out = s
                    .run(ort::inputs![
                        "global_hidden" => h, "repetition_seen_mask" => rm,
                        "assistant_random_u" => aru, "audio_random_u" => aut
                    ])
                    .context("local")?;
                let should = out[0]
                    .try_extract_tensor::<i32>()
                    .context("should_continue")?
                    .1[0];
                let frame: Vec<i32> = out[1]
                    .try_extract_tensor::<i32>()
                    .context("frame_token_ids")?
                    .1
                    .to_vec();
                (should, frame)
            };
            if should == 0 {
                break;
            }
            for (ch, &code) in frame.iter().enumerate() {
                if (code as usize) < 1024 {
                    rep_mask[ch * 1024 + code as usize] = 1;
                }
            }
            frames.push(frame.clone());

            // ---- next assistant-slot row → decode_step ----
            let mut row = vec![c.audio_pad_token_id; rw];
            row[0] = c.audio_assistant_slot_token_id;
            row[1..].copy_from_slice(&frame);
            let (nh, nkv) = {
                let mut s = self.decode.lock().unwrap();
                let mut ins: Vec<(Cow<'static, str>, SessionInputValue)> =
                    Vec::with_capacity(2 + 2 * N_LAYERS);
                let row_t = Tensor::<i32>::from_array(([1, 1, rw], row)).context("row")?;
                let pvl = Tensor::<i32>::from_array(([1], vec![past_len])).context("pvl")?;
                ins.push(("input_ids".into(), SessionInputValue::from(row_t)));
                ins.push(("past_valid_lengths".into(), SessionInputValue::from(pvl)));
                for i in 0..N_LAYERS {
                    let (ks, kd) = &kv[2 * i];
                    let (vs, vd) = &kv[2 * i + 1];
                    let kt =
                        Tensor::<f32>::from_array((ks.clone(), kd.clone())).context("past_key")?;
                    let vt = Tensor::<f32>::from_array((vs.clone(), vd.clone()))
                        .context("past_value")?;
                    ins.push((format!("past_key_{i}").into(), SessionInputValue::from(kt)));
                    ins.push((
                        format!("past_value_{i}").into(),
                        SessionInputValue::from(vt),
                    ));
                }
                let out = s.run(ins).context("decode_step")?;
                let (hshape, hdata) = out[0]
                    .try_extract_tensor::<f32>()
                    .context("decode hidden")?;
                let nh = last_row(hdata, &hshape);
                let mut nkv = Vec::with_capacity(2 * N_LAYERS);
                for i in 1..=2 * N_LAYERS {
                    let (shape, data) = out[i]
                        .try_extract_tensor::<f32>()
                        .with_context(|| format!("kv {i}"))?;
                    nkv.push((
                        shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                        data.to_vec(),
                    ));
                }
                (nh, nkv)
            };
            hidden = nh;
            kv = nkv;
            past_len += 1;
        }
        Ok(frames)
    }

    /// Decode `[n_frames][n_vq]` codes → interleaved-stereo f32 @ 48 kHz.
    fn decode_codes(&self, frames: &[Vec<i32>]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let n = frames.len();
        let nq = self.manifest.tts_config.n_vq;
        let mut flat = Vec::with_capacity(n * nq);
        for f in frames {
            flat.extend_from_slice(f);
        }
        let mut s = self.codec.lock().unwrap();
        let codes = Tensor::<i32>::from_array(([1, n, nq], flat)).context("audio_codes")?;
        let lens = Tensor::<i32>::from_array(([1], vec![n as i32])).context("code_lengths")?;
        let out = s
            .run(ort::inputs!["audio_codes" => codes, "audio_code_lengths" => lens])
            .context("codec decode")?;
        let (shape, data) = out[0].try_extract_tensor::<f32>().context("audio")?; // [1, ch, samples]
        let ch = shape[1] as usize;
        let samples = shape[2] as usize;
        // planar [ch, samples] → interleaved
        let mut inter = vec![0.0f32; ch * samples];
        for c in 0..ch {
            for i in 0..samples {
                inter[i * ch + c] = data[c * samples + i];
            }
        }
        Ok(inter)
    }

    /// Write interleaved-stereo f32 to a WAV at the codec sample rate.
    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channels(),
            sample_rate: self.sample_rate(),
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

/// Extract the last time-step row from `[1, seq, HIDDEN]` flat data.
fn last_row(data: &[f32], shape: &[i64]) -> Vec<f32> {
    let hidden = *shape.last().unwrap() as usize;
    data[data.len() - hidden..].to_vec()
}
