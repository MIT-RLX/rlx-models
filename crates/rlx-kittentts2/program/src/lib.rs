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

//! KittenTTS 2's host logic as an RLX program, answering `synthesize`:
//! what `KittenTts2::generate` does — the text normalized (the crate's NeMo
//! grammars) and chunked, each chunk's prompt around the voice's reference
//! (the crate's `text`), the speech LM a token at a time (the package's
//! `generate` entry, the voice's speaker row as its `speaker` medium; the
//! crate's penalties and sampler), its codec tokens through S3Gen (the
//! `s3_encoder`, two meanflow steps of `s3_estimator`, `hift_f0`, the sine
//! source, `hift_decode`, the inverse STFT and the fade), the chunks joined.
//! `run` answers the ids the LM generates after `options.ids`.

#[allow(dead_code)]
#[path = "../../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../../src/normalize/mod.rs"]
mod normalize;
#[allow(dead_code)]
#[path = "../../src/sampling.rs"]
mod sampling;
#[allow(dead_code)]
#[path = "../../src/text.rs"]
mod text;

use std::cell::RefCell;
use std::collections::HashMap;

use config::{DecodePreset, TokenMap};
use rlx_program::{Message, Program, Tensor, host, run};
use sampling::{Rng, SamplerOpts, apply_penalties, sample};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";
const MEL: usize = 80;
/// S3Gen's codec vocabulary, and the silence it pads with.
const SPEECH_VOCAB: u32 = 6561;
const SILENCE_TOKEN: u32 = 4299;
/// Mel frames per codec token; samples per mel frame.
const TOKEN_MEL_RATIO: usize = 2;
const UPSAMPLE: usize = 480;
/// HiFT's source and inverse STFT.
const N_FFT: usize = 16;
const HOP: usize = 4;
const N_BINS: usize = N_FFT / 2 + 1;
const HARMONICS: usize = 9;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const AUDIO_LIMIT: f32 = 0.99;
/// Each chunk's token budget, as the runner sizes it.
const TOKENS_PER_SECOND: f32 = 25.0;
const CHARS_PER_SECOND: f32 = 20.0;
const CHUNK_BUDGET_SLACK: f32 = 1.8;
const MIN_CHUNK_BUDGET: usize = 200;

/// A built-in voice (the package's `voices/<name>.json`).
struct Voice {
    transcript: String,
    reference_tokens: Vec<u32>,
    speaker_embedding: Vec<f32>,
    prompt_token: Vec<u32>,
    prompt_feat: Vec<f32>,
    prompt_frames: usize,
    spks: Vec<f32>,
}

struct Kitten {
    tm: TokenMap,
    window: Option<usize>,
    use_reference_prompt: bool,
    emotion_control: Option<String>,
    chunk_chars: usize,
    chunk_min_chars: usize,
    presets: HashMap<String, DecodePreset>,
    default_preset: String,
    default_voice: String,
    sample_rate: u32,
    temb: Vec<Vec<f32>>,
    source_weight: Vec<f32>,
    source_bias: f32,
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_f64().map(|x| x as f32))
                .collect()
        })
        .unwrap_or_default()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default()
}

fn i32s(v: &[u32]) -> Vec<i32> {
    v.iter().map(|&t| t as i32).collect()
}

/// The `generate` entry's logits for `ids` (the speaker's row the first
/// time).
fn lm_logits(ids: &[u32], speaker: Option<&[f32]>) -> Result<Vec<f32>, String> {
    let ids = i32s(ids);
    let mut tensors = vec![Tensor::i32("input_ids", vec![1, ids.len()], &ids)];
    if let Some(s) = speaker {
        tensors.push(Tensor::f32("speaker", vec![s.len()], s));
    }
    let out = host(
        "entry.run",
        Message::new(json!({"entry": "generate"}), tensors),
    )?;
    Ok(out.tensors.first().ok_or("the LM gave no logits")?.to_f32())
}

/// `sine_source`: f0 (`[L]`, Hz) → `tanh(Linear(9 → 1))` of the harmonics'
/// sines (`[480 L]`), the overtones' phases and the noise drawn from `rng`.
fn sine_source(weight: &[f32], bias: f32, f0: &[f32], rng: &RefCell<Rng>) -> Vec<f32> {
    let mut phase = [0f32; HARMONICS];
    {
        let mut r = rng.borrow_mut();
        for p in phase.iter_mut().skip(1) {
            *p = ((r.uniform() * 2.0 - 1.0) * std::f64::consts::PI) as f32;
        }
    }
    let sample_rate = 24_000f32;
    let n = f0.len() * UPSAMPLE;
    let mut out = vec![0f32; n];
    let mut acc = [0f64; HARMONICS];
    for (i, o) in out.iter_mut().enumerate() {
        let f = f0[i / UPSAMPLE];
        let voiced = f > VOICED_THRESHOLD;
        let mut s = bias;
        for h in 0..HARMONICS {
            acc[h] += (f * (h + 1) as f32 / sample_rate) as f64;
            let theta = std::f64::consts::TAU * acc[h].rem_euclid(1.0);
            let mut v = if voiced {
                SINE_AMP * ((theta as f32) + phase[h]).sin()
            } else {
                0.0
            };
            let amp = if voiced { NOISE_STD } else { SINE_AMP / 3.0 };
            v += amp * rng.borrow_mut().normal();
            s += weight[h] * v;
        }
        *o = s.tanh();
    }
    out
}

/// The periodic Hann window.
fn hann() -> [f32; N_FFT] {
    let mut w = [0f32; N_FFT];
    for (i, v) in w.iter_mut().enumerate() {
        *v = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N_FFT as f64).cos() as f32;
    }
    w
}

/// `torch.stft(x, 16, 4, 16, hann, center=True, reflect)` → `[18, F]`.
fn stft(x: &[f32]) -> Vec<f32> {
    let half = N_FFT / 2;
    let n = x.len();
    let reflect = |i: isize| -> f32 {
        let j = if i < 0 {
            -i
        } else if i as usize >= n {
            2 * (n as isize - 1) - i
        } else {
            i
        };
        x[j as usize]
    };
    let frames = n / HOP + 1;
    let win = hann();
    let mut out = vec![0f32; 2 * N_BINS * frames];
    for f in 0..frames {
        let start = (f * HOP) as isize - half as isize;
        let seg: Vec<f64> = (0..N_FFT)
            .map(|j| (reflect(start + j as isize) * win[j]) as f64)
            .collect();
        for k in 0..N_BINS {
            let (mut re, mut im) = (0f64, 0f64);
            for (j, &v) in seg.iter().enumerate() {
                let a = std::f64::consts::TAU * (k * j) as f64 / N_FFT as f64;
                re += v * a.cos();
                im -= v * a.sin();
            }
            out[k * frames + f] = re as f32;
            out[(N_BINS + k) * frames + f] = im as f32;
        }
    }
    out
}

/// The decoder's `[18, F]` → waveform: `exp` magnitude (≤ 100), `sin`
/// phase, inverse STFT, clamped to ±0.99.
fn istft_head(x: &[f32], frames: usize) -> Vec<f32> {
    let mut spec = vec![0f64; 2 * N_BINS * frames];
    for k in 0..N_BINS {
        for f in 0..frames {
            let mag = x[k * frames + f].exp().min(100.0) as f64;
            let ph = x[(N_BINS + k) * frames + f].sin() as f64;
            spec[k * frames + f] = mag * ph.cos();
            spec[(N_BINS + k) * frames + f] = mag * ph.sin();
        }
    }
    let win = hann();
    let half = N_FFT / 2;
    let full = N_FFT + HOP * (frames - 1);
    let mut y = vec![0f64; full];
    let mut env = vec![0f64; full];
    for f in 0..frames {
        let re = |k: usize| spec[k * frames + f];
        let im = |k: usize| spec[(N_BINS + k) * frames + f];
        for j in 0..N_FFT {
            let mut s = re(0) + re(N_BINS - 1) * if j % 2 == 0 { 1.0 } else { -1.0 };
            for k in 1..N_BINS - 1 {
                let a = std::f64::consts::TAU * (k * j) as f64 / N_FFT as f64;
                s += 2.0 * (re(k) * a.cos() - im(k) * a.sin());
            }
            let wj = win[j] as f64;
            y[f * HOP + j] += s / N_FFT as f64 * wj;
            env[f * HOP + j] += wj * wj;
        }
    }
    (half..half + HOP * (frames - 1))
        .map(|i| {
            let v = (if env[i] > 1e-11 { y[i] / env[i] } else { y[i] }) as f32;
            v.clamp(-AUDIO_LIMIT, AUDIO_LIMIT)
        })
        .collect()
}

/// S3Gen's fade-in: 20 ms of silence, then a 20 ms raised-cosine ramp.
fn apply_trim_fade(wav: &mut [f32], sample_rate: u32) {
    let n = (sample_rate as usize) / 50;
    for (i, v) in wav.iter_mut().take(2 * n).enumerate() {
        let g = if i < n {
            0.0
        } else {
            let a = std::f32::consts::PI * (1.0 - (i - n) as f32 / (n - 1) as f32);
            (a.cos() + 1.0) / 2.0
        };
        *v *= g;
    }
}

impl Kitten {
    fn encode(&self, s: &str) -> Result<Vec<u32>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": s, "special": false})),
        )?;
        Ok(ids(&m.body["ids"]))
    }

    fn voice(&self, name: &str) -> Result<Voice, String> {
        let p = format!("/package/voices/{name}.json");
        let bytes = std::fs::read(&p).map_err(|_| format!("no voice '{name}'"))?;
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        Ok(Voice {
            transcript: v["transcript"].as_str().unwrap_or("").to_string(),
            reference_tokens: ids(&v["reference_tokens"]),
            speaker_embedding: floats(&v["speaker_embedding"]),
            prompt_token: ids(&v["prompt_token"]),
            prompt_feat: floats(&v["prompt_feat"]),
            prompt_frames: v["prompt_frames"].as_u64().unwrap_or(0) as usize,
            spks: floats(&v["spks"]),
        })
    }

    /// `SpeechLm::generate`: until `<speech_end>`, `[STOP]` or `max_new`.
    fn generate(
        &self,
        prompt: &[u32],
        speaker: &[f32],
        o: &SamplerOpts,
        max_new: usize,
        rng: &mut Rng,
    ) -> Result<Vec<u32>, String> {
        let mut logits = lm_logits(prompt, Some(speaker))?;
        let mut history = prompt.to_vec();
        let mut out = Vec::new();
        for step in 0..max_new {
            apply_penalties(&mut logits, &history, prompt.len(), &self.tm, o);
            let tok = sample(&logits, o, rng);
            out.push(tok);
            history.push(tok);
            if tok == self.tm.speech_end_id || tok == self.tm.stop_id || step + 1 == max_new {
                break;
            }
            logits = lm_logits(&history, None)?;
        }
        Ok(out)
    }

    /// `S3Gen::synthesize`: codec tokens → 24 kHz audio, or none.
    fn s3(&self, audio: &[u32], v: &Voice, rng: &RefCell<Rng>) -> Result<Option<Vec<f32>>, String> {
        let Some((_, mel)) = self.flow(audio, v, rng)? else {
            return Ok(None);
        };
        let gl = mel.len() / MEL;
        let f0 = run(
            "hift_f0",
            vec![Tensor::f32("mel", vec![1, MEL, gl, 1], &mel)],
        )?;
        let f0 = f0.first().ok_or("hift_f0 gave nothing")?.to_f32();
        let source = sine_source(&self.source_weight, self.source_bias, &f0, rng);
        let s_stft = stft(&source);
        let frames = source.len() / HOP + 1;
        let head = run(
            "hift_decode",
            vec![
                Tensor::f32("mel", vec![1, MEL, gl, 1], &mel),
                Tensor::f32("s_stft", vec![1, 2 * N_BINS, frames, 1], &s_stft),
            ],
        )?;
        let head = head.first().ok_or("hift_decode gave nothing")?.to_f32();
        let mut wav = istft_head(&head, frames);
        apply_trim_fade(&mut wav, self.sample_rate);
        Ok(Some(wav))
    }

    /// `S3Gen::flow_with_noise`: the encoder's `mu` `[80, L]` and the
    /// generated frames' mel `[80, L − prompt]`, the noise drawn from `rng`.
    fn flow(
        &self,
        audio: &[u32],
        v: &Voice,
        rng: &RefCell<Rng>,
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>, String> {
        let mut tokens: Vec<u32> = audio
            .iter()
            .copied()
            .filter(|&t| t < SPEECH_VOCAB)
            .collect();
        if tokens.is_empty() {
            return Ok(None);
        }
        tokens.extend([SILENCE_TOKEN; 3]);
        let l = TOKEN_MEL_RATIO * (v.prompt_token.len() + tokens.len());
        let noise: Vec<f32> = {
            let mut r = rng.borrow_mut();
            (0..MEL * l).map(|_| r.normal()).collect()
        };
        let mut all = v.prompt_token.clone();
        all.extend(&tokens);
        let mu = run(
            "s3_encoder",
            vec![Tensor::i32("tokens", vec![all.len()], &i32s(&all))],
        )?;
        let mu = mu.first().ok_or("the encoder gave nothing")?.to_f32();
        let p = v.prompt_frames;
        if p >= l {
            return Err(format!("the prompt's {p} frames cover all {l}"));
        }
        let mut cond = vec![0f32; MEL * l];
        for f in 0..p {
            for m in 0..MEL {
                cond[m * l + f] = v.prompt_feat[f * MEL + m];
            }
        }
        let steps = self.temb.len();
        let mut x = noise;
        for (step, temb) in self.temb.iter().enumerate() {
            let t = step as f32 / steps as f32;
            let r = (step + 1) as f32 / steps as f32;
            let s = |c: usize| vec![1, c, l, 1];
            let vel = run(
                "s3_estimator",
                vec![
                    Tensor::f32("x", s(MEL), &x),
                    Tensor::f32("mu", s(MEL), &mu),
                    Tensor::f32("cond", s(MEL), &cond),
                    Tensor::f32("spks", vec![1, MEL, 1, 1], &v.spks),
                    Tensor::f32("temb", vec![1, temb.len()], temb),
                ],
            )?;
            let vel = vel.first().ok_or("the estimator gave nothing")?.to_f32();
            for (xi, vi) in x.iter_mut().zip(&vel) {
                *xi += (r - t) * vi;
            }
        }
        let mel: Vec<f32> = (0..MEL)
            .flat_map(|m| x[m * l + p..(m + 1) * l].to_vec())
            .collect();
        Ok(Some((mu, mel)))
    }

    fn sampler(&self, o: &Value) -> Result<SamplerOpts, String> {
        let name = o["preset"].as_str().unwrap_or(&self.default_preset);
        let preset = *self
            .presets
            .get(name)
            .ok_or_else(|| format!("unknown preset {name:?}"))?;
        let mut s = SamplerOpts::from_preset(preset, self.window);
        let f = |k: &str| o[k].as_f64().map(|v| v as f32);
        s.temperature = f("temperature").unwrap_or(s.temperature);
        s.top_k = o["top_k"].as_u64().map_or(s.top_k, |v| v as usize);
        s.top_p = f("top_p").unwrap_or(s.top_p);
        s.min_p = f("min_p").unwrap_or(s.min_p);
        s.greedy = o["greedy"].as_bool().unwrap_or(false);
        Ok(s)
    }

    /// `KittenTts2::generate_with_voice`: the waveform, and every chunk's
    /// codec tokens. `options.codes`: those decoded, the LM skipped.
    fn speak(&self, input: &str, o: &Value) -> Result<(Vec<f32>, Vec<u32>), String> {
        if let Some(codes) = o["codes"].as_array() {
            let codes: Vec<u32> = codes
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect();
            let voice = self.voice(o["voice"].as_str().unwrap_or(&self.default_voice))?;
            let seed = o["seed"].as_u64().unwrap_or(0);
            let flow_rng = RefCell::new(Rng::new(seed ^ 0x5EED_F10E));
            let wav = self
                .s3(&codes, &voice, &flow_rng)?
                .ok_or("no codec tokens to decode")?;
            return Ok((wav, codes));
        }
        let spoken = text::normalize_text(input, o["normalize"].as_bool().unwrap_or(true));
        if spoken.trim().is_empty() {
            return Err("no text to speak".into());
        }
        let chunks = text::split_for_synthesis(
            &spoken,
            o["chunk_chars"]
                .as_u64()
                .map_or(self.chunk_chars, |v| v as usize),
            o["chunk_min_chars"]
                .as_u64()
                .map_or(self.chunk_min_chars, |v| v as usize),
        );
        let sampler = self.sampler(o)?;
        let voice = self.voice(o["voice"].as_str().unwrap_or(&self.default_voice))?;
        let use_ref = o["use_reference_prompt"]
            .as_bool()
            .unwrap_or(self.use_reference_prompt);
        let prefix = if use_ref {
            let t = self.encode(voice.transcript.trim())?;
            Some(text::build_reference_prefix(
                &self.tm,
                &t,
                &voice.reference_tokens,
            ))
        } else {
            None
        };
        let use_emotion = o["use_emotion"]
            .as_bool()
            .unwrap_or_else(|| text::has_expression_tag(input));
        let emotion = match (&self.emotion_control, use_emotion) {
            (Some(ctrl), true) => Some(self.encode(ctrl)?),
            _ => None,
        };
        let seed = o["seed"].as_u64().unwrap_or(0);
        let max_new = o["max_new_tokens"].as_u64().unwrap_or(1000) as usize;
        let mut rng = Rng::new(seed);
        let flow_rng = RefCell::new(Rng::new(seed ^ 0x5EED_F10E));
        let single = chunks.len() == 1;
        let mut wavs = Vec::new();
        let mut all = Vec::new();
        for chunk in &chunks {
            let budget = if single {
                max_new
            } else {
                let expected = chunk.chars().count() as f32 / CHARS_PER_SECOND
                    * TOKENS_PER_SECOND
                    * CHUNK_BUDGET_SLACK;
                (expected as usize).min(max_new).max(MIN_CHUNK_BUDGET)
            };
            let target = self.encode(chunk)?;
            let prompt = text::build_generation_prompt(
                &self.tm,
                &target,
                prefix.as_deref(),
                emotion.as_deref(),
            );
            let generated = self.generate(
                &prompt,
                &voice.speaker_embedding,
                &sampler,
                budget,
                &mut rng,
            )?;
            let audio: Vec<u32> = generated
                .iter()
                .filter(|&&t| self.tm.is_audio(t))
                .map(|&t| t - self.tm.audio_id_base)
                .collect();
            all.extend(&audio);
            if let Some(w) = self.s3(&audio, &voice, &flow_rng)? {
                wavs.push(w);
            }
        }
        let gap = o["chunk_gap_s"]
            .as_f64()
            .map_or(text::CHUNK_GAP_S, |v| v as f32);
        let wav = text::join_chunks(&wavs, self.sample_rate, gap)
            .ok_or_else(|| "the model produced no audio tokens".to_string())?;
        Ok((wav, all))
    }
}

impl Program for Kitten {
    fn init(config: Value) -> Result<Self, String> {
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let presets = config["presets"]
            .as_object()
            .ok_or("config has no presets")?
            .iter()
            .map(|(k, v)| {
                serde_json::from_value::<DecodePreset>(v.clone())
                    .map(|p| (k.clone(), p))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        Ok(Kitten {
            tm: serde_json::from_value(config["token_map"].clone()).map_err(|e| e.to_string())?,
            window: config["repetition_window"].as_u64().map(|v| v as usize),
            use_reference_prompt: config["use_reference_prompt"].as_bool().unwrap_or(true),
            emotion_control: config["emotion_control"].as_str().map(String::from),
            chunk_chars: u("chunk_chars")? as usize,
            chunk_min_chars: u("chunk_min_chars")? as usize,
            presets,
            default_preset: s("default_preset")?,
            default_voice: s("default_voice")?,
            sample_rate: u("sample_rate")? as u32,
            temb: config["temb"]
                .as_array()
                .ok_or("config has no temb")?
                .iter()
                .map(floats)
                .collect(),
            source_weight: floats(&config["source"]["weight"]),
            source_bias: config["source"]["bias"].as_f64().unwrap_or(0.0) as f32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        match task {
            "synthesize" => {
                let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
                let (wav, _) = self.speak(text, o)?;
                let n = wav.len();
                Ok(Message::new(
                    json!({"sample_rate": self.sample_rate}),
                    vec![Tensor::f32("audio", vec![n], &wav)],
                ))
            }
            // `run` with `options.codes` and `stage: "flow"`: S3Gen's `mu`
            // and mel for them.
            "run" if o["stage"].as_str() == Some("flow") => {
                let codes = ids(&o["codes"]);
                let voice = self.voice(o["voice"].as_str().unwrap_or(&self.default_voice))?;
                let rng = RefCell::new(Rng::new(o["seed"].as_u64().unwrap_or(0) ^ 0x5EED_F10E));
                let (mu, mel) = self.flow(&codes, &voice, &rng)?.ok_or("no codes")?;
                Ok(Message::tensors(vec![
                    Tensor::f32("mu", vec![mu.len()], &mu),
                    Tensor::f32("mel", vec![mel.len()], &mel),
                ]))
            }
            // `run` with `options.text`: the codec tokens it speaks.
            "run" if o["text"].is_string() => {
                let (_, codes) = self.speak(o["text"].as_str().unwrap_or(""), o)?;
                let n = codes.len();
                Ok(Message::tensors(vec![Tensor::i32(
                    "codes",
                    vec![n],
                    &i32s(&codes),
                )]))
            }
            // `run`: the ids generated after `options.ids`, the speaker
            // `options.speaker`.
            "run" => {
                let prompt = ids(&o["ids"]);
                let speaker = floats(&o["speaker"]);
                let mut rng = Rng::new(o["seed"].as_u64().unwrap_or(0));
                let max_new = o["max_new_tokens"].as_u64().unwrap_or(256) as usize;
                let out = self.generate(&prompt, &speaker, &self.sampler(o)?, max_new, &mut rng)?;
                let n = out.len();
                Ok(Message::tensors(vec![Tensor::i32(
                    "tokens",
                    vec![n],
                    &i32s(&out),
                )]))
            }
            _ => Err(format!(
                "KittenTTS 2 answers synthesize and run, not {task}"
            )),
        }
    }
}

rlx_program::program!(Kitten);
