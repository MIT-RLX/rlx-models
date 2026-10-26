// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! MetaVoice's host logic as an RLX program, answering `synthesize`: what
//! `MetaVoice::synthesize` does — the voice (a 16 kHz `reference`, else the
//! bundled Bria clip) trimmed, framed and embedded a 160-frame partial at a
//! time (`speaker`), the partials averaged; the text normalized and
//! BPE-encoded (the runner's tokenizer); the first stage over the prompt
//! (`fs_prefill`) and then a token at a time (`fs_step`, each layer's keys
//! and values kept in slots), its two rows' logits mixed by the guidance
//! scale, argmax or nucleus-sampled from the runner's xorshift stream, until
//! the end of audio; the coarse books split, the second stage's fine books
//! (`second_stage`), EnCodec (`encodec_lookup`, `encodec_pre`,
//! `encodec_lstm`, `encodec_post`), the silence trimmed and the peak
//! normalized. `run` answers the tokens and the codes (or, with
//! `options.stage: "speaker"`, the voice's embedding).

#[path = "../../src/tokenize.rs"]
mod tokenize;

use rlx_program::{Message, Program, Tensor, host, run, run_body, run_with};
use serde_json::{Value, json};

use tokenize::MetaTokenizer;

struct MetaVoice {
    tok: MetaTokenizer,
    voice: String,
    layers: usize,
    block_size: usize,
    second_block_size: usize,
    eos: u32,
    pad: u32,
    text_offset: u32,
    max_new_tokens: usize,
    guidance_scale: f32,
    temperature: f32,
    top_p: f32,
    seed: u64,
    greedy: bool,
    sample_rate: u32,
    codebooks: usize,
    n_fft: usize,
    hop: usize,
    partial: usize,
}

/// Collapse whitespace / strip: the runner's `normalize_text`.
fn normalize_text(text: &str) -> String {
    let t = text
        .replace(['\t', '\n', '\r', '*'], " ")
        .trim()
        .to_string();
    let mut out = String::with_capacity(t.len());
    let mut prev_space = false;
    for ch in t.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

/// The runner's `postprocess_pcm`: near-silence trimmed (20 ms kept), the
/// peak normalized to 0.95.
fn postprocess_pcm(pcm: &[f32], sample_rate: u32) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let peak = pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak < 1e-6 {
        return pcm.to_vec();
    }
    let thr = peak * 0.01;
    let mut lo = 0usize;
    while lo < pcm.len() && pcm[lo].abs() < thr {
        lo += 1;
    }
    let mut hi = pcm.len();
    while hi > lo && pcm[hi - 1].abs() < thr {
        hi -= 1;
    }
    let pad = (sample_rate as usize / 50).max(1);
    lo = lo.saturating_sub(pad);
    hi = (hi + pad).min(pcm.len());
    if lo >= hi {
        return pcm.to_vec();
    }
    let slice = &pcm[lo..hi];
    let peak2 = slice.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
    let gain = 0.95 / peak2;
    slice.iter().map(|&x| (x * gain).clamp(-1.0, 1.0)).collect()
}

/// The speaker encoder's `trim_top_db`: frames 20 dB under the loudest cut
/// from both ends.
fn trim_top_db(wav: &mut Vec<f32>, top_db: f32) {
    if wav.is_empty() {
        return;
    }
    let frame = 480usize;
    let hop = 160usize;
    let mut energies = Vec::new();
    let mut i = 0;
    while i + frame <= wav.len() {
        let e: f32 = wav[i..i + frame].iter().map(|x| x * x).sum::<f32>() / frame as f32;
        energies.push(e.max(1e-12).log10() * 10.0);
        i += hop;
    }
    if energies.is_empty() {
        return;
    }
    let peak = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let thr = peak - top_db;
    let first = energies.iter().position(|&e| e >= thr).unwrap_or(0);
    let last = energies
        .iter()
        .rposition(|&e| e >= thr)
        .unwrap_or(energies.len() - 1);
    let start = first * hop;
    let end = ((last + 1) * hop + frame).min(wav.len());
    *wav = wav[start..end].to_vec();
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    if x.is_empty() {
        return vec![0.0; pad * 2];
    }
    let mut out = Vec::with_capacity(x.len() + 2 * pad);
    for i in 0..pad {
        out.push(x[(pad - i).min(x.len() - 1)]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[x.len().saturating_sub(2 + i)]);
    }
    out
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= n;
    }
}

/// Lowest index of the largest value.
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// The runner's sampler stream.
struct XorShift64(u64);

impl XorShift64 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64 / u64::MAX as f64) as f32
    }
}

/// The runner's `sample_top_p`: softmax, the nucleus, one draw.
fn sample_top_p(logits: &mut [f32], top_p: f32, rng: &mut XorShift64) -> u32 {
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let mx = logits[order[0]];
    let mut sum = 0.0f32;
    for &i in &order {
        logits[i] = (logits[i] - mx).exp();
        sum += logits[i];
    }
    let inv = 1.0 / sum.max(1e-20);
    for &i in &order {
        logits[i] *= inv;
    }
    let cutoff = top_p.clamp(0.0, 1.0);
    let mut cum = 0.0f32;
    let mut last = 0usize;
    for (k, &i) in order.iter().enumerate() {
        cum += logits[i];
        last = k;
        if cum >= cutoff {
            break;
        }
    }
    let target = rng.next_f32() * cum.min(1.0);
    let mut running = 0.0f32;
    for &i in order.iter().take(last + 1) {
        running += logits[i];
        if running >= target {
            return i as u32;
        }
    }
    order[last] as u32
}

/// The first stage's interleaved tokens → codebooks 0 and 1.
fn extract_codebooks(tokens: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut c0 = Vec::new();
    let mut c1 = Vec::new();
    for &t in tokens {
        if t < 1024 {
            c0.push(t);
        } else if t < 2048 {
            c1.push(t - 1024);
        }
    }
    let n = c0.len().min(c1.len());
    c0.truncate(n);
    c1.truncate(n);
    (c0, c1)
}

fn pairs(v: &[(String, String)]) -> Vec<(&str, &str)> {
    v.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

fn first(outs: Vec<Tensor>, what: &str) -> Result<Tensor, String> {
    outs.into_iter()
        .next()
        .ok_or_else(|| format!("{what} gave nothing"))
}

fn u32s(v: &Value) -> Option<Vec<u32>> {
    v.as_array().map(|a| {
        a.iter()
            .filter_map(|x| x.as_u64().map(|x| x as u32))
            .collect()
    })
}

fn f32s(v: &Value) -> Option<Vec<f32>> {
    v.as_array().map(|a| {
        a.iter()
            .filter_map(|x| x.as_f64().map(|x| x as f32))
            .collect()
    })
}

impl MetaVoice {
    /// BPE ids plus `off`, then the end of text.
    fn encode(&self, text: &str, off: u32) -> Result<Vec<u32>, String> {
        let mut ids = self.tok.encode(text).map_err(|e| e.to_string())?;
        for id in &mut ids {
            *id += off;
        }
        ids.push(self.tok.eot_id() + off);
        Ok(ids)
    }

    /// A 16 kHz clip → its 256-d speaker embedding: each 160-frame partial
    /// (one every 77 frames, the last dropped under 75 % coverage) through
    /// `speaker`, L2-normalized, averaged, normalized again.
    fn speaker(&self, pcm: &[f32]) -> Result<Vec<f32>, String> {
        let mut wav = pcm.to_vec();
        trim_top_db(&mut wav, 20.0);
        if wav.len() < self.hop * 8 {
            return Err("reference wav too short after trim".into());
        }
        let padded = reflect_pad(&wav, self.n_fft / 2);
        let n_frames = if padded.len() >= self.n_fft {
            1 + (padded.len() - self.n_fft) / self.hop
        } else {
            0
        };
        let frame_step = ((16_000.0f32 / 1.3) / self.hop as f32).round() as usize;
        let frame_step = frame_step.max(1);
        let steps = (n_frames + frame_step).saturating_sub(self.partial).max(1);
        let mut starts: Vec<usize> = (0..steps).step_by(frame_step).collect();
        if let Some(&last) = starts.last() {
            let (wav_start, wav_stop) = (last * self.hop, (last + self.partial) * self.hop);
            let coverage = (n_frames * self.hop).saturating_sub(wav_start) as f32
                / (wav_stop - wav_start) as f32;
            if coverage < 0.75 && starts.len() > 1 {
                starts.pop();
            }
        }
        let mut acc = vec![0.0f32; 256];
        for &s in &starts {
            let mut frames = vec![0.0f32; self.partial * self.n_fft];
            for j in 0..self.partial {
                let f = s + j;
                if f < n_frames {
                    frames[j * self.n_fft..(j + 1) * self.n_fft]
                        .copy_from_slice(&padded[f * self.hop..f * self.hop + self.n_fft]);
                }
            }
            let mut emb = first(
                run(
                    "speaker",
                    vec![Tensor::f32(
                        "frames",
                        vec![self.partial, self.n_fft],
                        &frames,
                    )],
                )?,
                "speaker",
            )?
            .to_f32();
            l2_normalize(&mut emb);
            for (a, e) in acc.iter_mut().zip(&emb) {
                *a += e;
            }
        }
        let n = starts.len() as f32;
        for a in &mut acc {
            *a /= n;
        }
        l2_normalize(&mut acc);
        Ok(acc)
    }

    /// The voice: `options.speaker` (an embedding), a `reference` tensor or
    /// `options.reference` (16 kHz), else the bundled clip.
    fn voice(&self, req: &Message) -> Result<Vec<f32>, String> {
        let o = &req.body["options"];
        if let Some(e) = f32s(&o["speaker"]) {
            return Ok(e);
        }
        let pcm = match req.tensor("reference") {
            Some(t) => t.to_f32(),
            None => match f32s(&o["reference"]) {
                Some(p) => p,
                None => std::fs::read(&self.voice)
                    .map_err(|e| format!("read {}: {e}", self.voice))?
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
            },
        };
        self.speaker(&pcm)
    }

    /// Both guidance rows' logits mixed, then picked.
    fn pick(
        &self,
        logits: &[f32],
        greedy: bool,
        g: f32,
        temperature: f32,
        top_p: f32,
        rng: &mut XorShift64,
    ) -> u32 {
        let v = logits.len() / 2;
        let mut mixed: Vec<f32> = (0..v)
            .map(|i| g * logits[i] + (1.0 - g) * logits[v + i])
            .collect();
        if greedy {
            return argmax(&mixed);
        }
        for l in &mut mixed {
            *l /= temperature;
        }
        sample_top_p(&mut mixed, top_p, rng)
    }

    /// The first stage: the prompt's tokens and those it generates.
    fn generate(&self, prompt: &[u32], spk: &[f32], o: &Value) -> Result<Vec<u32>, String> {
        let max_new = o["max_new_tokens"]
            .as_u64()
            .map_or(self.max_new_tokens, |v| v as usize);
        let g = o["guidance_scale"]
            .as_f64()
            .map_or(self.guidance_scale, |v| v as f32);
        let temperature = o["temperature"]
            .as_f64()
            .map_or(self.temperature, |v| v as f32);
        let top_p = o["top_p"].as_f64().map_or(self.top_p, |v| v as f32);
        let greedy = o["greedy"].as_bool().unwrap_or(self.greedy) || temperature <= 0.0;
        let mut rng = XorShift64(o["seed"].as_u64().unwrap_or(self.seed).max(1));
        let spk_t = || Tensor::f32("spk", vec![1, spk.len()], spk);
        let names: Vec<(String, String)> = (0..self.layers)
            .flat_map(|i| [format!("k{i}"), format!("v{i}")])
            .map(|n| (n.clone(), format!("metavoice.{n}")))
            .collect();
        let nexts: Vec<(String, String)> = names
            .iter()
            .map(|(n, s)| (format!("{n}.next"), s.clone()))
            .collect();
        // One cache length a request, a multiple of 256 rows past the
        // prompt and what it may generate (one compile of each graph).
        let cap = (prompt.len() + max_new).div_ceil(256) * 256;
        let cap = cap.min(self.block_size);
        let slot_map = |v: &[(String, String)]| -> Value {
            v.iter()
                .map(|(a, b)| (a.clone(), Value::String(b.clone())))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        let mut tokens = prompt.to_vec();
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let mut logits = first(
            run_body(
                "fs_prefill",
                vec![Tensor::i32("ids", vec![ids.len()], &ids), spk_t()],
                json!({"keep": slot_map(&names), "dims": {"cap": cap}}),
            )?,
            "fs_prefill",
        )?
        .to_f32();
        let mut mask = vec![0.0f32; 2 * cap];
        for step in 0..max_new {
            if step > 0 {
                let last = *tokens.last().expect("a prompt") as i32;
                let pos = tokens.len() - 1;
                for row in 0..2 {
                    mask[row * cap..row * cap + pos + 1].fill(1.0);
                }
                logits = first(
                    run_with(
                        "fs_step",
                        vec![
                            Tensor::i32("id", vec![1], &[last]),
                            Tensor::f32("pos", vec![1], &[pos as f32]),
                            Tensor::f32("mask", vec![2, cap], &mask),
                            spk_t(),
                        ],
                        &pairs(&names),
                        &pairs(&nexts),
                    )?,
                    "fs_step",
                )?
                .to_f32();
            }
            let next = self.pick(&logits, greedy, g, temperature, top_p, &mut rng);
            tokens.push(next);
            if next == self.eos || tokens.len() >= cap {
                break;
            }
        }
        let slots: Vec<&str> = names.iter().map(|(_, s)| s.as_str()).collect();
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok(tokens)
    }

    /// The first stage's tokens → the 8 codebooks: 0 and 1 split from them,
    /// 2 … 7 from the second stage.
    fn codes(&self, text: &str, tokens: &[u32], spk: &[f32]) -> Result<Vec<Vec<u32>>, String> {
        let (c0, c1) = extract_codebooks(tokens);
        if c0.is_empty() {
            return Err(format!(
                "first stage produced no audio codes ({} tokens)",
                tokens.len()
            ));
        }
        let text_ids = self.encode(text, self.text_offset)?;
        let bs = self.second_block_size;
        let mut h0 = text_ids.clone();
        h0.extend_from_slice(&c0);
        h0.push(self.pad);
        let mut h1 = vec![self.pad; text_ids.len()];
        h1.extend_from_slice(&c1);
        h1.push(self.pad);
        h0.resize(bs, self.pad);
        h1.resize(bs, self.pad);
        let as_i32 = |v: &[u32]| v.iter().map(|&x| x as i32).collect::<Vec<_>>();
        let logits = first(
            run(
                "second_stage",
                vec![
                    Tensor::i32("h0", vec![bs], &as_i32(&h0)),
                    Tensor::i32("h1", vec![bs], &as_i32(&h1)),
                    Tensor::f32("spk", vec![1, spk.len()], spk),
                ],
            )?,
            "second_stage",
        )?;
        let vocab = logits.shape[2];
        let heads = logits.shape[0];
        let logits = logits.to_f32();
        let start = text_ids.len();
        let mut codes = vec![c0.clone(), c1];
        for h in 0..heads {
            let mut book = Vec::with_capacity(c0.len());
            for ti in start..(start + c0.len()).min(bs) {
                let row = &logits[(h * bs + ti) * vocab..(h * bs + ti + 1) * vocab];
                let id = argmax(row);
                book.push(if id >= self.pad { 0 } else { id });
            }
            codes.push(book);
        }
        if codes.len() != self.codebooks {
            return Err(format!(
                "expected {} codebooks, got {}",
                self.codebooks,
                codes.len()
            ));
        }
        Ok(codes)
    }

    /// Codebooks → 24 kHz audio, EnCodec, post-processed.
    fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        let t = codes.iter().map(Vec::len).min().unwrap_or(0);
        let n = codes.len();
        let flat: Vec<i32> = codes
            .iter()
            .flat_map(|b| b[..t].iter().map(|&c| c as i32))
            .collect();
        let mut x = first(
            run(
                "encodec_lookup",
                vec![Tensor::i32("codes", vec![n, t], &flat)],
            )?,
            "encodec_lookup",
        )?;
        for (f, input) in [
            ("encodec_pre", "zq"),
            ("encodec_lstm", "x"),
            ("encodec_post", "z"),
        ] {
            x.name = input.into();
            x = first(run(f, vec![x])?, f)?;
        }
        Ok(postprocess_pcm(&x.to_f32(), self.sample_rate))
    }
}

impl Program for MetaVoice {
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
        Ok(MetaVoice {
            tok: MetaTokenizer::load(s("tokenizer")?).map_err(|e| e.to_string())?,
            voice: s("voice")?,
            layers: u("layers")? as usize,
            block_size: u("block_size")? as usize,
            second_block_size: u("second_block_size")? as usize,
            eos: u("eos")? as u32,
            pad: u("pad")? as u32,
            text_offset: u("text_offset")? as u32,
            max_new_tokens: u("max_new_tokens")? as usize,
            guidance_scale: f("guidance_scale")?,
            temperature: f("temperature")?,
            top_p: f("top_p")?,
            seed: u("seed")?,
            greedy: config["greedy"].as_bool().unwrap_or(true),
            sample_rate: u("sample_rate")? as u32,
            codebooks: u("codebooks")? as usize,
            n_fft: u("spk_n_fft")? as usize,
            hop: u("spk_hop")? as usize,
            partial: u("spk_partial")? as usize,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().unwrap_or(""),
            _ => return Err(format!("MetaVoice answers synthesize and run, not {task}")),
        };
        let spk = self.voice(&req)?;
        if task == "run" && o["stage"].as_str() == Some("speaker") {
            return Ok(Message::tensors(vec![Tensor::f32(
                "speaker",
                vec![spk.len()],
                &spk,
            )]));
        }
        let text = normalize_text(text);
        if text.is_empty() {
            return Err("empty text".into());
        }
        // `options.tokens`: the first stage's tokens, the first stage skipped.
        let tokens = match u32s(&o["tokens"]) {
            Some(t) => t,
            None => {
                let prompt = self.encode(&text, self.tok.offset)?;
                self.generate(&prompt, &spk, o)?
            }
        };
        let codes = self.codes(&text, &tokens, &spk)?;
        if task == "run" {
            let t = codes[0].len();
            let flat: Vec<i32> = codes.iter().flatten().map(|&c| c as i32).collect();
            let toks: Vec<i32> = tokens.iter().map(|&c| c as i32).collect();
            return Ok(Message::tensors(vec![
                Tensor::i32("tokens", vec![toks.len()], &toks),
                Tensor::i32("codes", vec![codes.len(), t], &flat),
            ]));
        }
        let wav = self.decode(&codes)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(MetaVoice);
