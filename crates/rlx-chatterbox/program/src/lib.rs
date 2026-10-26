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

//! ChatterBox's host logic as an RLX program, answering `synthesize`: what
//! `NativeChatterBox::synthesize` does — the reference (a `reference`
//! tensor, else the package's voice) resampled to 24 kHz and padded,
//! through `speech_encoder`; the prompt embeds (its audio features, the
//! text's and the start of speech's, `embed_tokens`); T3 over the whole
//! context each token (`t3`), sampled as the runner samples (the
//! repetition penalty, temperature, top-k, top-p, its splitmix stream),
//! each token embedded at its position; `flow_encoder`; the CFM Euler
//! solver over `cfm_estimator` with classifier-free guidance (its noise from
//! the same stream); the prompt's frames dropped; `hift_head`, the n_fft-16
//! inverse STFT and the onset polish. `run` answers the speech tokens.

use std::collections::HashSet;

use rlx_program::{Message, Program, Tensor, host, run, run_sized};
use serde_json::{Value, json};

struct ChatterBox {
    tokenizer: String,
    /// The default voice's conditioning, a file per output.
    voice: String,
    unit: usize,
    start_text: i64,
    stop_text: i64,
    start_speech: i64,
    hidden: usize,
    opts: Opts,
    cfg_rate: f32,
    flow_steps: usize,
    istft_nfft: usize,
    istft_hop: usize,
    sample_rate: u32,
}

#[derive(Clone, Copy)]
struct Opts {
    exaggeration: f32,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    max_frames: usize,
    seed: u64,
    greedy: bool,
}

struct Rng(u64);

impl Rng {
    fn new(s: u64) -> Self {
        Self(s.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let u1 = (self.uniform() as f32).max(1e-12);
        let u2 = self.uniform() as f32;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn is_eos(t: i64) -> bool {
    t == 2 || t == 6562
}

/// The runner's sampler: the repetition penalty on every token generated,
/// then greedy, or temperature, top-k, top-p and one draw — in f64.
fn sample(logits: &[f32], seen: &HashSet<i64>, o: &Opts, rng: &mut Rng) -> i64 {
    let mut l: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    for &t in seen {
        if let Some(v) = l.get_mut(t as usize) {
            *v = if *v > 0.0 {
                *v / o.repetition_penalty as f64
            } else {
                *v * o.repetition_penalty as f64
            };
        }
    }
    if o.greedy {
        let mut best = 0usize;
        for i in 1..l.len() {
            if l[i] > l[best] {
                best = i;
            }
        }
        return best as i64;
    }
    for v in l.iter_mut() {
        *v /= o.temperature as f64;
    }
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_unstable_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
    idx.truncate(o.top_k.max(1));
    let max = l[idx[0]];
    let mut probs: Vec<f64> = idx.iter().map(|&i| (l[i] - max).exp()).collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut cum = 0.0;
    let mut k = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= o.top_p as f64 {
            k = j + 1;
            break;
        }
    }
    idx.truncate(k);
    probs.truncate(k);
    let s: f64 = probs.iter().sum();
    let r = rng.uniform() * s;
    let mut acc = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[j] as i64;
        }
    }
    idx[idx.len() - 1] as i64
}

/// Linear-interpolated resampling, the runner's.
fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

fn cosine_t_span(n: usize) -> Vec<f32> {
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            1.0 - (t * 0.5 * std::f32::consts::PI).cos()
        })
        .collect()
}

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
        .collect()
}

/// HiFT's inverse STFT: magnitude and phase `[bins, frames]`, a direct
/// inverse DFT per frame, the periodic Hann window, overlap-added and
/// normalized, `n_fft / 2` trimmed each side — the runner's.
fn istft(mag: &[f32], phase: &[f32], n_frames: usize, n_fft: usize, hop: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let win = hann_periodic(n_fft);
    let full_len = n_fft + hop * (n_frames - 1);
    let mut out = vec![0f32; full_len];
    let mut wsum = vec![0f32; full_len];
    for f in 0..n_frames {
        let mut frame = vec![0f32; n_fft];
        for (n, fr) in frame.iter_mut().enumerate() {
            let mut acc = 0f32;
            for k in 0..n_fft {
                let (re, im) = if k < n_bins {
                    let b = k * n_frames + f;
                    (mag[b] * phase[b].cos(), mag[b] * phase[b].sin())
                } else {
                    let b = (n_fft - k) * n_frames + f;
                    (mag[b] * phase[b].cos(), -mag[b] * phase[b].sin())
                };
                let ang = std::f32::consts::TAU * (k * n) as f32 / n_fft as f32;
                acc += re * ang.cos() - im * ang.sin();
            }
            *fr = acc / n_fft as f32;
        }
        let base = f * hop;
        for i in 0..n_fft {
            out[base + i] += frame[i] * win[i];
            wsum[base + i] += win[i] * win[i];
        }
    }
    for (o, w) in out.iter_mut().zip(&wsum) {
        if *w > 1e-8 {
            *o /= *w;
        }
    }
    let pad = n_fft / 2;
    out[pad..full_len - pad].to_vec()
}

/// The runner's onset polish: a leading click and the silence after it
/// cut, 15 ms kept before the speech, a 5 ms fade in.
fn polish_onset(pcm: &[f32], sr: u32) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let peak = pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak < 1e-4 {
        return pcm.to_vec();
    }
    let win = ((sr as usize) / 100).max(1);
    let speech_thresh = (0.08 * peak).max(0.006);
    let silence_thresh = (0.025 * peak).max(0.002);
    let n_wins = pcm.len() / win;
    if n_wins == 0 {
        return pcm.to_vec();
    }
    let rms: Vec<f32> = (0..n_wins)
        .map(|i| {
            let s = &pcm[i * win..(i + 1) * win];
            (s.iter().map(|x| x * x).sum::<f32>() / win as f32).sqrt()
        })
        .collect();
    let mut i = 0usize;
    if rms[0] > speech_thresh {
        let mut j = 0usize;
        while j < 4 && j < rms.len() && rms[j] > silence_thresh {
            j += 1;
        }
        let mut sil = 0usize;
        while j + sil < rms.len() && rms[j + sil] <= silence_thresh {
            sil += 1;
        }
        if sil >= 8 {
            i = j;
        }
    }
    while i < rms.len() && rms[i] <= silence_thresh {
        i += 1;
    }
    if i >= rms.len() {
        i = 0;
    }
    let pad = ((sr as usize) * 15) / 1000;
    let start = (i * win).saturating_sub(pad);
    let mut out = pcm[start..].to_vec();
    let fade = ((sr as usize) * 5) / 1000;
    for i in 0..fade.min(out.len()) {
        out[i] *= i as f32 / fade as f32;
    }
    out
}

fn i64_tensor(name: &str, shape: Vec<usize>, v: &[i64]) -> Tensor {
    Tensor::new(
        name,
        "i64",
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

/// The output `name`, else the one at `at` (its place among the ONNX
/// graph's outputs: two of them one value share a name).
fn named<'a>(outs: &'a [Tensor], name: &str, at: usize) -> Result<&'a Tensor, String> {
    outs.iter()
        .find(|t| t.name == name)
        .or_else(|| outs.get(at))
        .ok_or_else(|| {
            let have: Vec<&str> = outs.iter().map(|t| t.name.as_str()).collect();
            format!("no output {name} (have {have:?})")
        })
}

/// What the reference conditions: `speech_encoder`'s four outputs.
struct Conditioning {
    audio_features: Vec<f32>,
    audio_tokens: Vec<i64>,
    speaker_embeddings: Vec<f32>,
    speaker_features: Vec<f32>,
}

impl ChatterBox {
    /// The reference at 24 kHz, at least 2 s, then whole units — 51 to 100
    /// of them (2 to 4 s: the encoder's graph is exact there; a longer
    /// reference is cut) — through the encoder.
    fn condition(&self, reference: &[f32], sr: u32) -> Result<Conditioning, String> {
        let mut pcm = resample(reference, sr, self.sample_rate);
        let units = pcm.len().max(48_000).div_ceil(self.unit).clamp(51, 100);
        pcm.resize(units * self.unit, 0.0);
        let n = pcm.len();
        let outs = run(
            "speech_encoder",
            vec![Tensor::f32("audio_values", vec![1, n], &pcm)],
        )?;
        Ok(Conditioning {
            audio_features: named(&outs, "audio_features", 0)?.to_f32(),
            audio_tokens: named(&outs, "audio_tokens", 1)?.to_i64(),
            speaker_embeddings: named(&outs, "speaker_embeddings", 2)?.to_f32(),
            speaker_features: named(&outs, "speaker_features", 3)?.to_f32(),
        })
    }

    /// `ids` embedded from position `pos0`.
    fn embed(&self, ids: &[i64], pos0: usize, exaggeration: f32) -> Result<Vec<f32>, String> {
        let n = ids.len();
        let pos: Vec<i64> = (pos0..pos0 + n).map(|p| p as i64).collect();
        let outs = run(
            "embed_tokens",
            vec![
                i64_tensor("input_ids", vec![1, n], ids),
                i64_tensor("position_ids", vec![1, n], &pos),
                Tensor::f32("exaggeration", vec![1], &[exaggeration]),
            ],
        )?;
        Ok(outs.first().ok_or("embed_tokens gave nothing")?.to_f32())
    }

    /// The text's ids: start, the tokenizer's, stop.
    fn text_ids(&self, text: &str) -> Result<Vec<i64>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer, "text": text, "special": false})),
        )?;
        let mut ids = vec![self.start_text];
        ids.extend(
            m.body["ids"]
                .as_array()
                .ok_or("no ids")?
                .iter()
                .filter_map(Value::as_i64),
        );
        ids.push(self.stop_text);
        Ok(ids)
    }

    /// The speech tokens T3 samples after the prompt.
    fn tokens(
        &self,
        c: &Conditioning,
        text: &str,
        o: &Opts,
        rng: &mut Rng,
    ) -> Result<Vec<i64>, String> {
        let h = self.hidden;
        let ids = self.text_ids(text)?;
        let mut embeds = c.audio_features.clone();
        embeds.extend(self.embed(&ids, 0, o.exaggeration)?);
        embeds.extend(self.embed(&[self.start_speech], ids.len(), o.exaggeration)?);
        let prompt = embeds.len() / h;
        let mut generated = Vec::new();
        let mut seen = HashSet::new();
        for seq in (prompt..).take(o.max_frames) {
            // The context padded to the prompt and every frame it may
            // generate, as the runner pads it: T3 is causal, so the last
            // real position's logits are the unpadded ones — one length
            // compiled per request, not one per token.
            let len = prompt + o.max_frames;
            let mut ctx = embeds[..seq * h].to_vec();
            ctx.resize(len * h, 0.0);
            let outs = run(
                "t3",
                vec![Tensor::f32("inputs_embeds", vec![1, len, h], &ctx)],
            )?;
            let logits = outs.first().ok_or("t3 gave no logits")?.to_f32();
            let vocab = logits.len() / len;
            let next = sample(&logits[(seq - 1) * vocab..seq * vocab], &seen, o, rng);
            if is_eos(next) {
                break;
            }
            generated.push(next);
            seen.insert(next);
            embeds.extend(self.embed(&[next], seq, o.exaggeration)?);
        }
        Ok(generated)
    }

    /// The speech tokens to audio: the flow encoder, the CFM solver, the
    /// vocoder.
    fn speak(
        &self,
        c: &Conditioning,
        generated: &[i64],
        rng: &mut Rng,
    ) -> Result<Vec<f32>, String> {
        let mel = self.mel(c, generated, rng)?;
        let gen_frames = mel.len() / 80;
        let hh = run(
            "hift_head",
            vec![Tensor::f32("speech_feat", vec![1, 80, gen_frames], &mel)],
        )?;
        let mag = named(&hh, "magnitude", 0)?.to_f32();
        let phase = named(&hh, "phase", 1)?.to_f32();
        let tw = mag.len() / (self.istft_nfft / 2 + 1);
        let wav = istft(&mag, &phase, tw, self.istft_nfft, self.istft_hop);
        Ok(polish_onset(&wav, self.sample_rate))
    }

    /// The generated frames' mel `[80, frames]`: the flow encoder, the CFM
    /// solver, the prompt's frames dropped.
    fn mel(&self, c: &Conditioning, generated: &[i64], rng: &mut Rng) -> Result<Vec<f32>, String> {
        let mel_len = c.speaker_features.len() / 80;
        let prompt_len = (mel_len / 2).min(c.audio_tokens.len());
        let prompt_token = &c.audio_tokens[..prompt_len];
        let (nt, np) = (generated.len(), prompt_token.len());
        let enc = run_sized(
            "flow_encoder",
            vec![
                i64_tensor("token", vec![1, nt], generated),
                i64_tensor("prompt_token", vec![1, np], prompt_token),
                Tensor::f32("prompt_feat", vec![1, mel_len, 80], &c.speaker_features),
                Tensor::f32("embedding", vec![1, 192], &c.speaker_embeddings),
            ],
            &[
                ("tokens", nt + np),
                ("prompt_tokens", np),
                ("prompt_frames", mel_len),
            ],
        )?;
        let mu = named(&enc, "mu", 0)?.to_f32();
        let mask = named(&enc, "mask", 1)?.to_f32();
        let spks = named(&enc, "spks", 2)?.to_f32();
        let cond = named(&enc, "cond", 3)?.to_f32();
        let tf = mu.len() / 80;
        let plane = 80 * tf;

        let mut x: Vec<f32> = (0..plane).map(|_| rng.normal()).collect();
        let cat0 = |a: &[f32]| {
            let mut v = a.to_vec();
            v.resize(2 * a.len(), 0.0);
            v
        };
        let (mu2, spks2, cond2) = (cat0(&mu), cat0(&spks), cat0(&cond));
        let mut mask2 = mask.clone();
        mask2.extend_from_slice(&mask);
        for w in cosine_t_span(self.flow_steps).windows(2) {
            let (a, b) = (w[0], w[1]);
            let mut x2 = x.clone();
            x2.extend_from_slice(&x);
            let outs = run(
                "cfm_estimator",
                vec![
                    Tensor::f32("x", vec![2, 80, tf], &x2),
                    Tensor::f32("mask", vec![2, 1, tf], &mask2),
                    Tensor::f32("mu", vec![2, 80, tf], &mu2),
                    Tensor::f32("t", vec![2], &[a, a]),
                    Tensor::f32("spks", vec![2, 80], &spks2),
                    Tensor::f32("cond", vec![2, 80, tf], &cond2),
                ],
            )?;
            let dxdt = named(&outs, "dxdt", 0)?.to_f32();
            for i in 0..plane {
                let d = (1.0 + self.cfg_rate) * dxdt[i] - self.cfg_rate * dxdt[plane + i];
                x[i] += (b - a) * d;
            }
        }
        let gen_frames = tf.saturating_sub(mel_len);
        if gen_frames == 0 {
            return Err("the flow decoder produced no generated frames".into());
        }
        let mut mel = Vec::with_capacity(80 * gen_frames);
        for ch in 0..80 {
            mel.extend_from_slice(&x[ch * tf + mel_len..ch * tf + tf]);
        }
        Ok(mel)
    }

    /// The request's reference (a `reference` tensor at
    /// `options.reference_sample_rate`) through the encoder, else the
    /// package's voice's conditioning.
    fn reference(&self, req: &Message) -> Result<Conditioning, String> {
        if let Some(t) = req.tensor("reference") {
            let sr = req.body["options"]["reference_sample_rate"]
                .as_u64()
                .unwrap_or(self.sample_rate as u64) as u32;
            return self.condition(&t.to_f32(), sr);
        }
        let read = |name: &str| {
            let p = format!("{}/{name}", self.voice);
            std::fs::read(&p).map_err(|e| format!("read {p}: {e}"))
        };
        let f32s = |b: Vec<u8>| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        Ok(Conditioning {
            audio_features: f32s(read("audio_features")?),
            audio_tokens: read("audio_tokens")?
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            speaker_embeddings: f32s(read("speaker_embeddings")?),
            speaker_features: f32s(read("speaker_features")?),
        })
    }

    fn opts(&self, o: &Value) -> Opts {
        let d = self.opts;
        let f = |k: &str, v: f32| o[k].as_f64().map_or(v, |x| x as f32);
        let u = |k: &str, v: usize| o[k].as_u64().map_or(v, |x| x as usize);
        Opts {
            exaggeration: f("exaggeration", d.exaggeration),
            temperature: f("temperature", d.temperature),
            top_p: f("top_p", d.top_p),
            top_k: u("top_k", d.top_k),
            repetition_penalty: f("repetition_penalty", d.repetition_penalty),
            max_frames: u("max_frames", d.max_frames),
            seed: o["seed"].as_u64().unwrap_or(d.seed),
            greedy: o["greedy"].as_bool().unwrap_or(d.greedy),
        }
    }
}

impl Program for ChatterBox {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        Ok(ChatterBox {
            tokenizer: s("tokenizer")?,
            voice: s("voice")?,
            unit: u("reference_unit")? as usize,
            start_text: u("start_text")? as i64,
            stop_text: u("stop_text")? as i64,
            start_speech: u("start_speech")? as i64,
            hidden: u("hidden")? as usize,
            opts: Opts {
                exaggeration: f("exaggeration")?,
                temperature: f("temperature")?,
                top_p: f("top_p")?,
                top_k: u("top_k")? as usize,
                repetition_penalty: f("repetition_penalty")?,
                max_frames: u("max_frames")? as usize,
                seed: u("seed")?,
                greedy: config["greedy"].as_bool().unwrap_or(false),
            },
            cfg_rate: f("cfg_rate")?,
            flow_steps: u("flow_steps")? as usize,
            istft_nfft: u("istft_nfft")? as usize,
            istft_hop: u("istft_hop")? as usize,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let opts = self.opts(o);
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().ok_or("run takes options.text")?,
            _ => return Err(format!("ChatterBox answers synthesize and run, not {task}")),
        };
        let c = self.reference(&req)?;
        let mut rng = Rng::new(opts.seed);
        let generated = self.tokens(&c, text, &opts, &mut rng)?;
        if generated.is_empty() {
            return Err("no speech tokens generated".into());
        }
        // `run` with `options.stage: "mel"`: the mel before the vocoder.
        if task == "run" && o["stage"].as_str() == Some("mel") {
            let mel = self.mel(&c, &generated, &mut rng)?;
            let n = mel.len() / 80;
            return Ok(Message::tensors(vec![Tensor::f32(
                "mel",
                vec![80, n],
                &mel,
            )]));
        }
        // `run`: the speech tokens, as `[n]` i32.
        if task == "run" {
            let t: Vec<i32> = generated.iter().map(|&x| x as i32).collect();
            let n = t.len();
            return Ok(Message::tensors(vec![Tensor::i32("tokens", vec![n], &t)]));
        }
        let wav = self.speak(&c, &generated, &mut rng)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(ChatterBox);
