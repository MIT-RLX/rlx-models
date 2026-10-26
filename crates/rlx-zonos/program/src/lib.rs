// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Zonos's host logic as an RLX program, answering `synthesize`: what
//! `NativeZonos::synthesize` does — the text phonemized (espeak-ng, its
//! data read from the package; punctuation kept) to Zonos' symbol ids; the
//! six conditioning rows (the speaker, the emotion, fmax, pitch and
//! speaking-rate Fourier features, the language) and the unconditional
//! ones, through `prefix`; the delay-pattern loop over the backbone with
//! classifier-free guidance (`backbone` over the pair, `heads`), the
//! runner's sampler (repetition penalty, temperature, min-p, its xorshift
//! stream) and its EOS rules; DAC (`dac_lookup`, `dac_decode`). `run`
//! answers the aligned codes, `[9, frames]`.

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

const PAD_ID: i64 = 0;
const UNK_ID: i64 = 1;
const BOS_ID: i64 = 2;
const EOS_ID: i64 = 3;
const PUNCTUATION: &str = ";:,.!?¡¿—…\"«»“”() *~-/\\&";
const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const LETTERS_IPA: &str = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

struct Zonos {
    espeak: String,
    d: usize,
    n_q: usize,
    vocab: usize,
    eos: i64,
    masked: i64,
    languages: Vec<String>,
    defaults: Value,
    sample_rate: u32,
    speaker_weight: Vec<f32>,
    speaker_bias: Vec<f32>,
    language: Vec<f32>,
    uncond: Vec<Vec<f32>>,
    fourier: Vec<Vec<f32>>,
}

/// The runner's sampler stream (xorshift64).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn first(outs: Vec<Tensor>, what: &str) -> Result<Vec<f32>, String> {
    Ok(outs
        .into_iter()
        .next()
        .ok_or_else(|| format!("{what} gave nothing"))?
        .to_f32())
}

fn symbol_id(c: char) -> i64 {
    PUNCTUATION
        .chars()
        .chain(LETTERS.chars())
        .chain(LETTERS_IPA.chars())
        .position(|s| s == c)
        .map_or(UNK_ID, |i| i as i64 + 4)
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

fn argmax_excluding(row: &[f32], ban: usize) -> usize {
    let mut best_i = if ban == 0 { 1 } else { 0 };
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if i != ban && v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn softmax_inplace(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

/// The runner's sampling knobs for one request.
struct Opts {
    greedy: bool,
    temperature: f32,
    min_p: f32,
    repetition_penalty: f32,
    cfg_scale: f32,
}

impl Zonos {
    /// `phonemize_lang`: espeak-ng over the spans between punctuation, the
    /// marks kept in place; then Zonos' symbol ids between BOS and EOS.
    fn phonemes(&self, text: &str, lang: &str) -> Result<Vec<i64>, String> {
        let cleaned: String = text
            .chars()
            .map(|c| if c == ',' { ' ' } else { c })
            .collect();
        let engine = espeak_ng::EspeakNg::with_data_dir(lang, std::path::Path::new(&self.espeak))
            .or_else(|_| {
                espeak_ng::EspeakNg::with_data_dir("en", std::path::Path::new(&self.espeak))
            })
            .map_err(|e| format!("espeak-ng init: {e}"))?;
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
                .map_err(|e| format!("espeak-ng phonemize: {e}"))?
                .replace('\n', " ");
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(ipa.trim());
            span.clear();
            Ok(())
        };
        for ch in cleaned.chars() {
            if PUNCT.contains(&ch) {
                flush(&mut span, &mut out)?;
                out.push(ch);
            } else {
                span.push(ch);
            }
        }
        flush(&mut span, &mut out)?;
        let mut ids = vec![BOS_ID];
        ids.extend(out.trim().chars().map(symbol_id));
        ids.push(EOS_ID);
        let _ = PAD_ID;
        Ok(ids)
    }

    /// A Fourier conditioner's row: `[cos(2π · x̂ Wᵀ), sin(…)]`, `x̂` scaled
    /// to `[min, max]`.
    fn fourier(&self, k: usize, x: &[f32], min: f32, max: f32) -> Vec<f32> {
        let w = &self.fourier[k - 2];
        let half = self.d / 2;
        let n = x.len();
        let span = (max - min).max(1e-8);
        let xn: Vec<f32> = x.iter().map(|v| (v - min) / span).collect();
        let mut out = vec![0f32; self.d];
        for o in 0..half {
            let mut acc = 0.0f32;
            for i in 0..n {
                acc += xn[i] * w[o * n + i];
            }
            let f = 2.0 * std::f32::consts::PI * acc;
            out[o] = f.cos();
            out[half + o] = f.sin();
        }
        out
    }

    /// The six conditioning rows `[6 · d]`, or the unconditional ones.
    fn conditions(&self, o: &Value, unconditional: bool) -> Result<Vec<f32>, String> {
        if unconditional {
            return Ok(self.uncond.concat());
        }
        let num = |k: &str| {
            o[k].as_f64()
                .or_else(|| self.defaults[k].as_f64())
                .unwrap_or(0.0) as f32
        };
        let mut rows = Vec::with_capacity(6 * self.d);
        match o["speaker"].as_array() {
            Some(s) => {
                let s: Vec<f32> = s
                    .iter()
                    .filter_map(|v| v.as_f64().map(|v| v as f32))
                    .collect();
                if s.len() != 128 {
                    return Err("the speaker embedding is 128 values".into());
                }
                for r in 0..self.d {
                    let w = &self.speaker_weight[r * 128..(r + 1) * 128];
                    rows.push(
                        w.iter().zip(&s).map(|(a, b)| a * b).sum::<f32>() + self.speaker_bias[r],
                    );
                }
            }
            None => rows.extend(&self.uncond[0]),
        }
        let mut emo: Vec<f32> = match o["emotion"].as_array() {
            Some(e) => e
                .iter()
                .filter_map(|v| v.as_f64().map(|v| v as f32))
                .collect(),
            None => self.defaults["emotion"]
                .as_array()
                .map(|e| {
                    e.iter()
                        .filter_map(|v| v.as_f64().map(|v| v as f32))
                        .collect()
                })
                .unwrap_or_default(),
        };
        let sum: f32 = emo.iter().sum();
        if sum > 0.0 {
            for v in &mut emo {
                *v /= sum;
            }
        }
        rows.extend(self.fourier(2, &emo, 0.0, 1.0));
        rows.extend(self.fourier(3, &[num("fmax")], 0.0, 24_000.0));
        rows.extend(self.fourier(4, &[num("pitch_std")], 0.0, 400.0));
        rows.extend(self.fourier(5, &[num("speaking_rate")], 0.0, 40.0));
        let lang = o["language"]
            .as_str()
            .or_else(|| self.defaults["language"].as_str())
            .unwrap_or("en-us");
        let lid = self
            .languages
            .iter()
            .position(|l| l.eq_ignore_ascii_case(lang))
            .ok_or_else(|| format!("unsupported language {lang}"))?;
        let idx = lid + 1;
        rows.extend_from_slice(&self.language[idx * self.d..(idx + 1) * self.d]);
        Ok(rows)
    }

    fn prefix(&self, phonemes: &[i64], conditions: &[f32]) -> Result<Vec<f32>, String> {
        let ids: Vec<i32> = phonemes.iter().map(|&p| p as i32).collect();
        first(
            run(
                "prefix",
                vec![
                    Tensor::i32("phonemes", vec![ids.len()], &ids),
                    Tensor::f32("conditions", vec![6, self.d], conditions),
                ],
            )?,
            "prefix",
        )
    }

    fn embed(&self, codes: &[i64]) -> Result<Vec<f32>, String> {
        let c: Vec<i32> = codes.iter().map(|&v| v as i32).collect();
        first(
            run(
                "embed_codes",
                vec![Tensor::i32("codes", vec![self.n_q], &c)],
            )?,
            "embed_codes",
        )
    }

    /// The backbone over the pair, padded to `len`, and the heads: the
    /// guided logits `[9 · (codebook + 1)]`.
    fn logits(
        &self,
        cond: &[f32],
        uncond: &[f32],
        len: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let d = self.d;
        let n = cond.len() / d;
        let mut x = cond.to_vec();
        x.resize(len * d, 0.0);
        x.extend_from_slice(uncond);
        x.resize(2 * len * d, 0.0);
        let h = first(
            run(
                "backbone",
                vec![Tensor::f32("inputs_embeds", vec![2, len, d], &x)],
            )?,
            "backbone",
        )?;
        let mut last = h[(n - 1) * d..n * d].to_vec();
        last.extend_from_slice(&h[(len + n - 1) * d..(len + n) * d]);
        let l = first(
            run("heads", vec![Tensor::f32("h", vec![2, d], &last)])?,
            "heads",
        )?;
        let v = self.n_q * (self.vocab + 1);
        Ok((0..v)
            .map(|i| l[v + i] + (l[i] - l[v + i]) * scale)
            .collect())
    }

    fn sample(
        &self,
        logits: &[f32],
        o: &Opts,
        generated: Option<&[Vec<i64>]>,
        rng: &mut Rng,
    ) -> Vec<i64> {
        let v = self.vocab + 1;
        let mut out = vec![0i64; self.n_q];
        if o.greedy {
            for (q, slot) in out.iter_mut().enumerate() {
                *slot = argmax(&logits[q * v..(q + 1) * v]) as i64;
            }
            return out;
        }
        for (q, slot) in out.iter_mut().enumerate() {
            let mut row = logits[q * v..(q + 1) * v].to_vec();
            if o.repetition_penalty != 1.0
                && let Some(g) = generated
            {
                let window = 2usize.min(g[q].len());
                let start = g[q].len() - window;
                for &tok in &g[q][start..] {
                    let t = tok.clamp(0, (v - 1) as i64) as usize;
                    if row[t] <= 0.0 {
                        row[t] *= o.repetition_penalty;
                    } else {
                        row[t] /= o.repetition_penalty;
                    }
                }
            }
            let temp = o.temperature.max(1e-5);
            for x in &mut row {
                *x /= temp;
            }
            softmax_inplace(&mut row);
            if o.min_p > 0.0 {
                let top = row.iter().copied().fold(0.0f32, f32::max);
                let thr = o.min_p * top;
                let mut sum = 0.0f32;
                for x in &mut row {
                    if *x < thr {
                        *x = 0.0;
                    }
                    sum += *x;
                }
                if sum > 0.0 {
                    for x in &mut row {
                        *x /= sum;
                    }
                }
            }
            let r = rng.f32();
            let mut acc = 0.0f32;
            let mut pick = row.len().saturating_sub(1);
            for (i, &p) in row.iter().enumerate() {
                acc += p;
                if r <= acc {
                    pick = i;
                    break;
                }
            }
            *slot = pick as i64;
        }
        out
    }

    /// `generate_codes_compiled`: the aligned codes `[9][frames]`.
    fn codes(&self, phonemes: &[i64], req: &Value) -> Result<Vec<Vec<i64>>, String> {
        let d = self.d;
        let n_q = self.n_q;
        let num = |k: &str| {
            req[k]
                .as_f64()
                .or_else(|| self.defaults[k].as_f64())
                .unwrap_or(0.0)
        };
        let o = Opts {
            greedy: req["greedy"].as_bool().unwrap_or(false),
            temperature: num("temperature") as f32,
            min_p: num("min_p") as f32,
            repetition_penalty: num("repetition_penalty") as f32,
            cfg_scale: num("cfg_scale") as f32,
        };
        let max_new = req["max_new_tokens"].as_u64().unwrap_or(86 * 8) as usize;
        let mut rng = Rng::new(req["seed"].as_u64().unwrap_or(num("seed") as u64));
        let prefix_c = self.prefix(phonemes, &self.conditions(req, false)?)?;
        let prefix_u = self.prefix(phonemes, &self.conditions(req, true)?)?;
        let prefix_len = prefix_c.len() / d;

        // The delay pattern over `max_new` unknown frames.
        let t = max_new;
        let out_t = t + n_q;
        let mut delayed = vec![vec![self.masked; out_t]; n_q];
        for (k, row) in delayed.iter_mut().enumerate() {
            let mut src = vec![self.masked; out_t];
            for v in src.iter_mut().take(t) {
                *v = -1;
            }
            for i in 0..out_t {
                row[(i + k + 1) % out_t] = src[i];
            }
        }
        let column = |delayed: &[Vec<i64>], at: usize| -> Vec<i64> {
            delayed.iter().map(|r| r[at]).collect()
        };
        let row0 = self.embed(&column(&delayed, 0))?;
        let mut cond = prefix_c;
        let mut uncond = prefix_u;
        cond.extend_from_slice(&row0);
        uncond.extend_from_slice(&row0);
        // Every step's row, the context's length compiled once.
        let len = prefix_len + 1 + out_t;
        let logits = self.logits(&cond, &uncond, len, o.cfg_scale)?;
        let mut next = self.sample(&logits, &o, None, &mut rng);
        let write = |delayed: &mut [Vec<i64>], at: usize, next: &[i64]| {
            for (q, row) in delayed.iter_mut().enumerate() {
                if row[at] == -1 {
                    row[at] = next[q];
                }
            }
        };
        let mut offset = 1;
        write(&mut delayed, offset, &next);
        let rate = num("speaking_rate").clamp(5.0, 40.0) as f32;
        let min_frames = (((phonemes.len() as f32) * (86.0 / rate) * 1.05).ceil() as usize).max(32);
        let mut remaining = out_t - offset;
        let mut stopping = false;
        let v = self.vocab + 1;
        while remaining > 0 {
            let row = self.embed(&column(&delayed, offset - 1))?;
            cond.extend_from_slice(&row);
            uncond.extend_from_slice(&row);
            let mut logits = self.logits(&cond, &uncond, len, o.cfg_scale)?;
            for q in 1..n_q {
                logits[q * v + self.eos as usize] = f32::NEG_INFINITY;
            }
            let so_far: Option<Vec<Vec<i64>>> = if o.greedy {
                None
            } else {
                Some(delayed.iter().map(|r| r[..offset].to_vec()).collect())
            };
            next = self.sample(&logits, &o, so_far.as_deref(), &mut rng);
            let frames = offset.saturating_sub(n_q);
            if next[0] == self.eos {
                if !stopping && frames < min_frames {
                    next[0] = argmax_excluding(&logits[..v], self.eos as usize) as i64;
                } else {
                    remaining = remaining.min(n_q);
                    stopping = true;
                }
            }
            if stopping {
                let idx = (n_q - remaining).min(n_q - 1);
                for (q, slot) in next.iter_mut().enumerate() {
                    if q < idx {
                        *slot = self.masked;
                    } else if q == idx {
                        *slot = self.eos;
                    }
                }
            }
            write(&mut delayed, offset, &next);
            offset += 1;
            remaining -= 1;
        }
        // The delay reverted, the frames written, codes past the book 0.
        let t_out = offset.saturating_sub(n_q);
        let mut aligned = Vec::with_capacity(n_q);
        for (k, row) in delayed.iter().enumerate() {
            let mut a: Vec<i64> = row[k + 1..out_t - n_q + k + 1].to_vec();
            a.truncate(t_out);
            for x in &mut a {
                if *x >= self.vocab as i64 {
                    *x = 0;
                }
            }
            aligned.push(a);
        }
        Ok(aligned)
    }

    fn decode(&self, codes: &[Vec<i64>]) -> Result<Vec<f32>, String> {
        let (n, t) = (codes.len(), codes[0].len());
        let flat: Vec<i32> = codes.iter().flatten().map(|&c| c as i32).collect();
        let z = run("dac_lookup", vec![Tensor::i32("codes", vec![n, t], &flat)])?;
        let mut z = z.into_iter().next().ok_or("DAC gave no latent")?;
        z.name = "z".into();
        first(run("dac_decode", vec![z])?, "dac_decode")
    }
}

impl Program for Zonos {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        let cond = s("cond")?;
        let f = |name: &str| read_f32(&format!("{cond}/{name}.f32"));
        Ok(Zonos {
            espeak: s("espeak")?,
            d: u("d_model")? as usize,
            n_q: u("n_codebooks")? as usize,
            vocab: u("codebook_size")? as usize,
            eos: u("eos")? as i64,
            masked: u("masked")? as i64,
            languages: config["languages"]
                .as_array()
                .ok_or("config has no languages")?
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            defaults: config.clone(),
            sample_rate: u("sample_rate")? as u32,
            speaker_weight: f("speaker_weight")?,
            speaker_bias: f("speaker_bias")?,
            language: f("language")?,
            uncond: (1..=6)
                .map(|i| f(&format!("uncond_{i}")))
                .collect::<Result<_, _>>()?,
            fourier: (2..=5)
                .map(|i| f(&format!("fourier_{i}")))
                .collect::<Result<_, _>>()?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = match task {
            "synthesize" => req.body["text"].as_str().unwrap_or(""),
            "run" => o["text"].as_str().unwrap_or(""),
            _ => return Err(format!("Zonos answers synthesize and run, not {task}")),
        };
        // `options.phonemes`: symbol ids, the phonemizer skipped.
        let phonemes: Vec<i64> = match o["phonemes"].as_array() {
            Some(p) => p.iter().filter_map(|v| v.as_i64()).collect(),
            None => {
                let lang = o["language"].as_str().unwrap_or("en-us");
                self.phonemes(text, lang)?
            }
        };
        // `options.codes`: aligned codes `[9][frames]` to decode.
        let codes: Vec<Vec<i64>> = match o["codes"].as_array() {
            Some(rows) => rows
                .iter()
                .map(|r| {
                    r.as_array()
                        .map(|c| c.iter().filter_map(|v| v.as_i64()).collect())
                        .unwrap_or_default()
                })
                .collect(),
            None => self.codes(&phonemes, o)?,
        };
        if codes.is_empty() || codes[0].is_empty() {
            return Err("the backbone produced no codes".into());
        }
        if task == "run" {
            let (n, t) = (codes.len(), codes[0].len());
            let flat: Vec<i32> = codes.iter().flatten().map(|&c| c as i32).collect();
            return Ok(Message::tensors(vec![Tensor::i32(
                "codes",
                vec![n, t],
                &flat,
            )]));
        }
        let wav = self.decode(&codes)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Zonos);
