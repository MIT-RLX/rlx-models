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

//! LuxTTS's host logic as an RLX program, answering `synthesize`: what
//! `LuxTts::synthesize` does around its graphs — the prompt's RMS and its
//! `center` padding, espeak-ng and the token table, the length regulator,
//! the seeded noise and the flow-matching update. The prompt's log-mel,
//! the encoder, the flow decoder, the vocoder and its ISTFT run in rlx
//! (`rlx_program::run`).
//!
//! The voice to clone is `options.prompt`: `{audio: [samples], sample_rate:
//! 24000, text}`.

use std::path::Path;

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

#[allow(dead_code, clippy::all)]
#[path = "../../src/config.rs"]
mod config;

const PACKAGE: &str = "/package";
const SAMPLE_RATE: u64 = 24_000;
const N_FFT: usize = 1024;
const HOP: usize = 256;
const N_MELS: usize = 100;

/// The runner's deterministic Gaussian (xorshift128+ → Box–Muller).
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

/// The options a call may set, the runner's defaults otherwise.
struct Opts {
    num_step: usize,
    guidance_scale: f32,
    speed: f32,
    speed_mult: f32,
    t_shift: f32,
    seed: u64,
}

struct Lux {
    tokens: config::Tokens,
    espeak_dir: String,
    defaults: Value,
    target_rms: f32,
    feat_scale: f32,
    /// The flow decoder's graphs: `fm_decoder_<frames mod this>`.
    fm_decoders: usize,
}

fn first(out: Vec<Tensor>, what: &str) -> Result<Vec<f32>, String> {
    Ok(out
        .first()
        .ok_or(format!("the {what} gave no output"))?
        .to_f32())
}

/// `x` reflected `pad` samples at each end (the edge sample not repeated).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(x[(pad - i).min(n - 1)]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[n.saturating_sub(2 + i)]);
    }
    out
}

impl Lux {
    fn phonemize_lang(&self, lang: &str, text: &str) -> Result<String, String> {
        if text.is_empty() {
            return Ok(String::new());
        }
        let engine = espeak_ng::EspeakNg::with_data_dir(lang, Path::new(&self.espeak_dir))
            .map_err(|e| format!("espeak-ng init for '{lang}': {e}"))?;
        const PUNCT: &[char] = &['.', ',', '!', '?', ';', ':'];
        let mut out = String::new();
        let mut span = String::new();
        let flush = |span: &mut String, out: &mut String| -> Result<(), String> {
            if span.trim().is_empty() {
                span.clear();
                return Ok(());
            }
            let ipa = engine
                .text_to_phonemes(span.trim())
                .map_err(|e| format!("espeak-ng phonemize ({lang}): {e}"))?;
            let ipa = ipa.replace('\n', " ");
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(ipa.trim());
            span.clear();
            Ok(())
        };
        for ch in text.chars() {
            if PUNCT.contains(&ch) {
                flush(&mut span, &mut out)?;
                out.push(ch);
            } else {
                span.push(ch);
            }
        }
        flush(&mut span, &mut out)?;
        Ok(out.trim().to_owned())
    }

    /// `text` phonemized (`en-us`, else `en`), each phoneme its id.
    fn encode(&self, text: &str) -> Result<Vec<i64>, String> {
        let mut last = String::from("no espeak language");
        for lang in ["en-us", "en"] {
            match self.phonemize_lang(lang, text) {
                Ok(ipa) => return Ok(ipa.chars().filter_map(|c| self.tokens.id_of(c)).collect()),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn speak(
        &self,
        text: &str,
        prompt: &[f32],
        prompt_text: &str,
        o: &Opts,
    ) -> Result<Vec<f32>, String> {
        // 1. The prompt at the target RMS, and its scaled log-mel.
        let ms = prompt.iter().map(|x| x * x).sum::<f32>() / prompt.len().max(1) as f32;
        let prompt_rms = ms.sqrt();
        let gain = self.target_rms / prompt_rms.max(1e-8);
        let pw: Vec<f32> = prompt.iter().map(|x| x * gain).collect();
        let padded = reflect_pad(&pw, N_FFT / 2);
        if padded.len() < N_FFT {
            return Err("the prompt audio is too short".into());
        }
        let tp = 1 + (padded.len() - N_FFT) / HOP;
        let signal = &padded[..HOP * (tp + N_FFT / HOP - 1)];
        let mel = first(
            run(
                "log_mel",
                vec![Tensor::f32("signal", vec![signal.len()], signal)],
            )?,
            "log-mel",
        )?;
        let prompt_feat: Vec<f32> = mel.iter().map(|v| v * self.feat_scale).collect();

        // 2. The prompt's transcript and the text, as ids.
        let tokens = self.encode(text)?;
        let ptokens = self.encode(prompt_text)?;
        if tokens.is_empty() {
            return Err("the text tokenized to nothing".into());
        }
        let ptok = if ptokens.is_empty() { vec![0] } else { ptokens };
        let (tp_len, t_len) = (ptok.len(), tokens.len());

        // 3. The encoder over prompt ++ text ++ [pad].
        let mut ids = ptok;
        ids.extend_from_slice(&tokens);
        ids.push(0);
        let s = ids.len();
        let idb: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let enc = first(
            run(
                "encoder_body",
                vec![Tensor::new("/Pad_output_0", "i64", vec![1, s], idb)],
            )?,
            "encoder",
        )?;

        // 4. The length regulator.
        let speed_eff = (o.speed * o.speed_mult) as f64;
        let main_len = s - 1;
        let num_frames = ((tp as f64 / tp_len.max(1) as f64) * (tp_len + t_len) as f64
            / speed_eff.max(1e-6))
        .ceil() as usize;
        let num_frames = num_frames.max(tp + 1);
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

        // 5. The noise, the padded speech condition, and the flow.
        let mut rng = Rng::new(o.seed);
        let mut x: Vec<f32> = (0..num_frames * N_MELS).map(|_| rng.randn()).collect();
        let mut speech_cond = vec![0f32; num_frames * N_MELS];
        speech_cond[..tp * N_MELS].copy_from_slice(&prompt_feat);
        let ts: Vec<f32> = (0..=o.num_step)
            .map(|i| {
                let tau = i as f32 / o.num_step as f32;
                o.t_shift * tau / (1.0 + (o.t_shift - 1.0) * tau)
            })
            .collect();
        let shape = vec![1, num_frames, N_MELS];
        let fm = format!("fm_decoder_{}", num_frames % self.fm_decoders);
        for step in 0..o.num_step {
            let (t_cur, t_next) = (ts[step], ts[step + 1]);
            let v = first(
                run(
                    &fm,
                    vec![
                        Tensor::f32("t", vec![], &[t_cur]),
                        Tensor::f32("x", shape.clone(), &x),
                        Tensor::f32("text_condition", shape.clone(), &text_condition),
                        Tensor::f32("speech_condition", shape.clone(), &speech_cond),
                        Tensor::f32("guidance_scale", vec![], &[o.guidance_scale]),
                    ],
                )?,
                "flow decoder",
            )?;
            for i in 0..x.len() {
                let x1 = x[i] + (1.0 - t_cur) * v[i];
                let x0 = x[i] - t_cur * v[i];
                x[i] = if step < o.num_step - 1 {
                    (1.0 - t_next) * x0 + t_next * x1
                } else {
                    x1
                };
            }
        }

        // 6. The generated frames, un-scaled, mel-major; the vocoder and its ISTFT.
        let l = num_frames - tp;
        let generated = &x[tp * N_MELS..];
        let mut mel_lm = vec![0f32; N_MELS * l];
        for t in 0..l {
            for m in 0..N_MELS {
                mel_lm[m * l + t] = generated[t * N_MELS + m] / self.feat_scale;
            }
        }
        let spec = run(
            "vocoder_spec",
            vec![Tensor::f32("mel", vec![1, N_MELS, l], &mel_lm)],
        )?;
        if spec.len() < 2 {
            return Err("the vocoder gave no real and imaginary parts".into());
        }
        let n_freq = N_FFT / 2 + 1;
        let mut wav = first(
            run(
                "istft",
                vec![
                    Tensor::f32("real", vec![n_freq, l], &spec[0].to_f32()),
                    Tensor::f32("imag", vec![n_freq, l], &spec[1].to_f32()),
                ],
            )?,
            "ISTFT",
        )?;

        // 7. The prompt's volume.
        if prompt_rms < self.target_rms {
            let g = prompt_rms / self.target_rms;
            for sm in &mut wav {
                *sm *= g;
            }
        }
        let peak = wav
            .iter()
            .fold(0f32, |a, s| if s.is_finite() { a.max(s.abs()) } else { a });
        if peak < 1e-3 {
            return Err(format!("the audio is silent (peak {peak:.2e})"));
        }
        Ok(wav)
    }
}

impl Program for Lux {
    fn init(config: Value) -> Result<Self, String> {
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Lux {
            tokens: config::Tokens::load(Path::new(PACKAGE)).map_err(|e| e.to_string())?,
            espeak_dir: config["espeak"]
                .as_str()
                .ok_or("config has no espeak")?
                .into(),
            target_rms: f("target_rms")?,
            feat_scale: f("feat_scale")?,
            defaults: config["defaults"].clone(),
            fm_decoders: config["fm_decoders"]
                .as_u64()
                .ok_or("config has no fm_decoders")? as usize,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("LuxTTS answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let prompt = &o["prompt"];
        let audio: Vec<f32> = prompt["audio"]
            .as_array()
            .ok_or("LuxTTS clones a voice: options.prompt.audio, its samples")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        if prompt["sample_rate"].as_u64().unwrap_or(SAMPLE_RATE) != SAMPLE_RATE {
            return Err(format!("the prompt must be {SAMPLE_RATE} Hz"));
        }
        let prompt_text = prompt["text"]
            .as_str()
            .ok_or("options.prompt.text, its transcript")?;
        let d = &self.defaults;
        let num = |k: &str| o[k].as_f64().or(d[k].as_f64()).unwrap_or(0.0);
        let opts = Opts {
            num_step: num("num_step") as usize,
            guidance_scale: num("guidance_scale") as f32,
            speed: num("speed") as f32,
            speed_mult: num("speed_mult") as f32,
            t_shift: num("t_shift") as f32,
            seed: o["seed"].as_u64().unwrap_or(0),
        };
        if opts.num_step == 0 {
            return Err("num_step must be at least 1".into());
        }
        let wav = self.speak(text, &audio, prompt_text, &opts)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": SAMPLE_RATE}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Lux);
