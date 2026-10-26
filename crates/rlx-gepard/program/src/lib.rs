// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard's host logic as an RLX program, answering `synthesize`: what
//! `GepardSynthesizer`'s eager loop does — the text's ids (the host's
//! tokenizer, no specials) between start and end of text, repeated up to
//! the target length, then start of speech; the prompt through `prefill`
//! (its caches kept in slots, one length a request); the first frame from
//! its heads, each next from `step` over the last, until the stop head
//! passes its threshold or the frames run out (the text's word count's
//! budget); each channel argmax or sampled at the temperature from the
//! runner's `fastrand` stream; the frames dequantized and decoded
//! (`decode`). `run` answers the frames (`options.frames` decodes given
//! ones).

use rlx_program::{Message, Program, Tensor, host, run, run_body, run_with};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";

struct Repeat {
    enabled: bool,
    target_text_tokens: usize,
    apply_below: usize,
    max_repeats: usize,
}

struct Gepard {
    layers: usize,
    start_of_text: u32,
    end_of_text: u32,
    start_of_speech: u32,
    repeat: Repeat,
    vocabs: Vec<usize>,
    levels: Vec<u32>,
    max_context: usize,
    max_frames: usize,
    min_frames: usize,
    frame_rate: f32,
    stop_threshold: f32,
    temperature: f32,
    seed: u64,
    repetition_penalty: f32,
    repetition_window: usize,
    sample_rate: u32,
}

fn first(outs: Vec<Tensor>, what: &str) -> Result<Tensor, String> {
    outs.into_iter()
        .next()
        .ok_or_else(|| format!("{what} gave nothing"))
}

fn pairs(v: &[(String, String)]) -> Vec<(&str, &str)> {
    v.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

impl Gepard {
    /// The runner's `target_r`: how many times the text is repeated.
    fn target_r(&self, n: usize) -> usize {
        let c = &self.repeat;
        if !c.enabled || n == 0 || n >= c.apply_below || n >= c.target_text_tokens {
            return 1;
        }
        let r = (c.target_text_tokens as f64 / n as f64).ceil() as usize;
        r.clamp(1, c.max_repeats)
    }

    /// `[SOT text EOT] × r, SOS`.
    fn prompt(&self, text: &str) -> Result<Vec<u32>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": text, "special": false})),
        )?;
        let body: Vec<u32> = m.body["ids"]
            .as_array()
            .ok_or("no ids")?
            .iter()
            .filter_map(|v| v.as_u64().map(|v| v as u32))
            .collect();
        let r = self.target_r(body.len()).max(1);
        let mut block = vec![self.start_of_text];
        block.extend_from_slice(&body);
        block.push(self.end_of_text);
        let mut out = Vec::with_capacity(block.len() * r + 1);
        for _ in 0..r {
            out.extend_from_slice(&block);
        }
        out.push(self.start_of_speech);
        Ok(out)
    }

    /// The runner's `suggest_max_frames`.
    fn max_frames(&self, text: &str, cap: usize) -> usize {
        let words = text.split_whitespace().count().max(1);
        let est = ((words as f32) * 0.35 * self.frame_rate * 1.4).ceil() as usize;
        est.clamp(self.min_frames, cap.min(self.max_frames))
    }

    /// The runner's `sample_all_heads_temp`: each channel's head (penalized
    /// over the recent frames), argmax at temperature 0, else one draw.
    fn sample(
        &self,
        logits: &[f32],
        temperature: f32,
        recent: Option<&[Vec<u32>]>,
        rng: &mut fastrand::Rng,
    ) -> Vec<u32> {
        let mut off = 0;
        let mut frame = Vec::with_capacity(self.vocabs.len());
        for (ch, &v) in self.vocabs.iter().enumerate() {
            let mut l = logits[off..off + v].to_vec();
            off += v;
            if self.repetition_penalty != 1.0
                && let Some(frames) = recent
            {
                let start = if self.repetition_window == 0 {
                    0
                } else {
                    frames.len().saturating_sub(self.repetition_window)
                };
                for f in &frames[start..] {
                    let tok = f[ch] as usize;
                    if tok < l.len() {
                        if l[tok] > 0.0 {
                            l[tok] /= self.repetition_penalty;
                        } else {
                            l[tok] *= self.repetition_penalty;
                        }
                    }
                }
            }
            let id = if temperature <= 0.0 {
                l.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            } else {
                let max_l = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = l
                    .iter()
                    .map(|&x| ((x - max_l) / temperature).exp())
                    .collect();
                let sum: f32 = exps.iter().sum();
                let r = (rng.f32() * sum).min(sum - 1e-7);
                let mut acc = 0.0f32;
                let mut pick = (v - 1) as u32;
                for (i, e) in exps.iter().enumerate() {
                    acc += e;
                    if acc >= r {
                        pick = i as u32;
                        break;
                    }
                }
                pick
            };
            frame.push(id);
        }
        frame
    }

    /// The frames for `text`.
    fn frames(&self, text: &str, o: &Value) -> Result<Vec<Vec<u32>>, String> {
        let prompt = self.prompt(text)?;
        let greedy = o["greedy"].as_bool().unwrap_or(false);
        let temperature = if greedy {
            0.0
        } else {
            o["temperature"]
                .as_f64()
                .map_or(self.temperature, |v| v as f32)
        };
        let mut rng = fastrand::Rng::with_seed(o["seed"].as_u64().unwrap_or(self.seed));
        let threshold = o["stop_threshold"]
            .as_f64()
            .map_or(self.stop_threshold, |v| v as f32);
        let asked = o["max_frames"]
            .as_u64()
            .map_or(self.max_frames, |v| v as usize);
        let max_frames = self.max_frames(text, asked);
        // One cache length a request, a multiple of 256 rows.
        let cap = (prompt.len() + max_frames)
            .div_ceil(256)
            .saturating_mul(256)
            .min(self.max_context);
        let max_frames = max_frames.min(cap - prompt.len());

        let names: Vec<(String, String)> = (0..self.layers)
            .flat_map(|i| [format!("k{i}"), format!("v{i}")])
            .map(|n| (n.clone(), format!("gepard.{n}")))
            .collect();
        let nexts: Vec<(String, String)> = names
            .iter()
            .map(|(n, s)| (format!("{n}.next"), s.clone()))
            .collect();
        let keep: serde_json::Map<String, Value> = names
            .iter()
            .map(|(a, b)| (a.clone(), Value::String(b.clone())))
            .collect();
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let logits = first(
            run_body(
                "prefill",
                vec![Tensor::i32("ids", vec![ids.len()], &ids)],
                json!({"keep": keep, "dims": {"cap": cap}}),
            )?,
            "prefill",
        )?
        .to_f32();
        let penalized = self.repetition_penalty != 1.0;
        let mut frames = vec![self.sample(&logits, temperature, None, &mut rng)];
        let mut mask = vec![0.0f32; cap];
        for step in 1..max_frames {
            let pos = prompt.len() + step - 1;
            mask[..=pos].fill(1.0);
            let codes: Vec<i32> = frames[step - 1].iter().map(|&c| c as i32).collect();
            let logits = first(
                run_with(
                    "step",
                    vec![
                        Tensor::i32("codes", vec![codes.len()], &codes),
                        Tensor::f32("pos", vec![1], &[pos as f32]),
                        Tensor::f32("mask", vec![1, cap], &mask),
                    ],
                    &pairs(&names),
                    &pairs(&nexts),
                )?,
                "step",
            )?
            .to_f32();
            let stop = 1.0 / (1.0 + (-logits[logits.len() - 1]).exp());
            if stop > threshold {
                break;
            }
            let recent = penalized.then_some(frames.as_slice());
            let next = self.sample(&logits, temperature, recent, &mut rng);
            frames.push(next);
        }
        let slots: Vec<&str> = names.iter().map(|(_, s)| s.as_str()).collect();
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok(frames)
    }

    /// Frames → 22.05 kHz audio: each channel `(code − L/2) / (L/2)`, the
    /// decoder.
    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        let t = frames.len();
        let c = self.vocabs.len();
        let mut latent = vec![0.0f32; c * t];
        for (fi, frame) in frames.iter().enumerate() {
            for ch in 0..c {
                let l = self.levels[ch % self.levels.len()] as i32;
                let half = (l / 2).max(1) as f32;
                let code = (frame[ch] as i32).clamp(0, l - 1) as f32;
                latent[ch * t + fi] = (code - half) / half;
            }
        }
        Ok(first(
            run("decode", vec![Tensor::f32("latent", vec![c, t], &latent)])?,
            "decode",
        )?
        .to_f32())
    }
}

impl Program for Gepard {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        let r = &config["repeat"];
        let ru = |k: &str| r[k].as_u64().ok_or(format!("config has no repeat.{k}"));
        let list = |k: &str| -> Result<Vec<u64>, String> {
            Ok(config[k]
                .as_array()
                .ok_or(format!("config has no {k}"))?
                .iter()
                .filter_map(Value::as_u64)
                .collect())
        };
        Ok(Gepard {
            layers: u("layers")? as usize,
            start_of_text: u("start_of_text")? as u32,
            end_of_text: u("end_of_text")? as u32,
            start_of_speech: u("start_of_speech")? as u32,
            repeat: Repeat {
                enabled: r["enabled"].as_bool().unwrap_or(false),
                target_text_tokens: ru("target_text_tokens")? as usize,
                apply_below: ru("apply_below")? as usize,
                max_repeats: ru("max_repeats")? as usize,
            },
            vocabs: list("vocabs")?.into_iter().map(|v| v as usize).collect(),
            levels: list("levels")?.into_iter().map(|v| v as u32).collect(),
            max_context: u("max_context")? as usize,
            max_frames: u("max_frames")? as usize,
            min_frames: u("min_frames")? as usize,
            frame_rate: f("frame_rate")?,
            stop_threshold: f("stop_threshold")?,
            temperature: f("temperature")?,
            seed: u("seed")?,
            repetition_penalty: f("repetition_penalty")?,
            repetition_window: u("repetition_window")? as usize,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().unwrap_or(""),
            _ => return Err(format!("Gepard answers synthesize and run, not {task}")),
        };
        // `options.frames`: frames `[n][32]` to decode, the backbone skipped.
        let frames: Vec<Vec<u32>> = match o["frames"].as_array() {
            Some(rows) => rows
                .iter()
                .map(|r| {
                    r.as_array()
                        .map(|c| {
                            c.iter()
                                .filter_map(|v| v.as_u64().map(|v| v as u32))
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect(),
            None => self.frames(text, o)?,
        };
        if frames.is_empty() {
            return Err("Gepard produced zero frames".into());
        }
        if task == "run" {
            let flat: Vec<i32> = frames.iter().flatten().map(|&c| c as i32).collect();
            return Ok(Message::tensors(vec![Tensor::i32(
                "frames",
                vec![frames.len(), self.vocabs.len()],
                &flat,
            )]));
        }
        let wav = self.decode(&frames)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Gepard);
