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

//! Moshi's host logic as an RLX program, answering `synthesize` (and `run`,
//! the frames' codes): `MoshiSession::generate_one_way` over the package's
//! graphs — the prompt's tokens a frame each (`prompt_frame_tokens`: the
//! start token, the prompt, then pads); each frame's delayed codes (the
//! semantic one a frame back, the acoustic ones `acoustic_delay + 1` back,
//! pads before), `embed` and `step` (its caches in slots), the text sampled
//! (`sampling.rs`, its seed); the DepFormer's slices `dep_{k}` each sampled,
//! the first frames' forced pads fed on; each frame past the acoustic delay
//! with no pad to Mimi (`mimi_lookup`, `mimi_decode`). Options:
//! `max_steps`, `text_temperature`, `audio_temperature`, `text_seed`,
//! `audio_seed`.

#[path = "../../src/sampling.rs"]
#[allow(dead_code)]
mod sampling;

use ndarray::ArrayView1;
use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Map, Value, json};

const UNGENERATED: u32 = u32::MAX;

struct Moshi {
    tokenizer: String,
    layers: usize,
    heads: usize,
    head_dim: usize,
    rope_theta: f32,
    dep_layers: usize,
    slices: usize,
    books: usize,
    generated: usize,
    delay: usize,
    audio_pad: u32,
    text_pad: u32,
    text_start: u32,
    max_steps: usize,
    text_temperature: f64,
    text_top_k: usize,
    audio_temperature: f64,
    audio_top_k: usize,
    text_seed: u64,
    audio_seed: u64,
    mimi_codebooks: usize,
    sample_rate: u32,
}

fn output(outs: &[Tensor], n: &str) -> Result<Vec<f32>, String> {
    Ok(outs
        .iter()
        .find(|t| t.name == n)
        .ok_or(format!("no output {n}"))?
        .to_f32())
}

impl Moshi {
    fn speak(&self, prompt: &str, o: &Value) -> Result<(Vec<Vec<u32>>, Vec<f32>), String> {
        let max_steps = o["max_steps"]
            .as_u64()
            .map_or(self.max_steps, |v| v as usize);
        let f = |k: &str, d: f64| o[k].as_f64().unwrap_or(d);
        let u = |k: &str, d: u64| o[k].as_u64().unwrap_or(d);
        let mut text_lp = sampling::LogitsProcessor::new(
            f("text_temperature", self.text_temperature),
            self.text_top_k,
            u("text_seed", self.text_seed),
        );
        let mut audio_lp = sampling::LogitsProcessor::new(
            f("audio_temperature", self.audio_temperature),
            self.audio_top_k,
            u("audio_seed", self.audio_seed),
        );
        // The prompt a token a frame.
        let r = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer, "text": prompt, "special": false})),
        )?;
        let mut ids: Vec<u32> = r.body["ids"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as u32)
                    .collect()
            })
            .unwrap_or_default();
        if ids.is_empty() {
            ids.push(self.text_pad);
        }
        let frames_in: Vec<u32> = (0..max_steps)
            .map(|fi| match fi {
                0 => self.text_start,
                _ if fi - 1 < ids.len() => ids[fi - 1],
                _ => self.text_pad,
            })
            .collect();

        // The temporal caches: zeros, a power of two past every step.
        let cap = (max_steps + 16).next_power_of_two();
        let width = self.heads * self.head_dim;
        let zeros = vec![0f32; cap * width];
        let mut from = Map::new();
        let mut keep = Map::new();
        from.insert("inputs_embeds".into(), json!("m.x"));
        for l in 0..self.layers {
            for s in ["k", "v"] {
                host(
                    "slot.set",
                    Message::new(
                        json!({"slot": format!("m.{s}{l}")}),
                        vec![Tensor::f32(
                            format!("past_{s}_{l}"),
                            vec![1, cap, self.heads, self.head_dim],
                            &zeros,
                        )],
                    ),
                )?;
                from.insert(format!("past_{s}_{l}"), json!(format!("m.{s}{l}")));
                keep.insert(format!("past_{s}_{l}.next"), json!(format!("m.{s}{l}")));
            }
        }

        let buf = max_steps + self.delay;
        let mut audio = vec![vec![UNGENERATED; self.books]; buf];
        let mut frames = Vec::new();
        let half = self.head_dim / 2;
        for step in 0..max_steps {
            // The delayed codes.
            let delayed: Vec<i32> = (0..self.books)
                .map(|cb| {
                    let t = if cb == 0 || cb == self.generated {
                        if step == 0 {
                            self.audio_pad
                        } else {
                            audio[step - 1][cb]
                        }
                    } else if step <= self.delay {
                        self.audio_pad
                    } else {
                        audio[step - self.delay - 1][cb]
                    };
                    t as i32
                })
                .collect();
            if delayed.iter().any(|&t| t as u32 == UNGENERATED) {
                return Err(format!("ungenerated audio at step {step}"));
            }
            run_body(
                "embed",
                vec![
                    Tensor::i32("text", vec![1], &[frames_in[step] as i32]),
                    Tensor::i32("audio", vec![self.books], &delayed),
                ],
                json!({"keep": {"inputs_embeds": "m.x"}}),
            )?;
            let (mut cos, mut sin) = (vec![0f32; half], vec![0f32; half]);
            for i in 0..half {
                let inv = 1.0f32 / self.rope_theta.powf(i as f32 / half as f32);
                let a = step as f32 * inv;
                cos[i] = a.cos();
                sin[i] = a.sin();
            }
            let mut mask = vec![0f32; cap + 1];
            mask[..step].fill(1.0);
            mask[cap] = 1.0;
            let outs = run_body(
                "step",
                vec![
                    Tensor::f32("position", vec![1], &[step as f32]),
                    Tensor::f32("rope_cos", vec![1, half], &cos),
                    Tensor::f32("rope_sin", vec![1, half], &sin),
                    Tensor::f32("mask", vec![1, cap + 1], &mask),
                ],
                json!({"from": from, "keep": keep}),
            )?;
            let logits = output(&outs, "logits")?;
            let hidden = output(&outs, "hidden")?;
            let text = text_lp
                .sample(ArrayView1::from(&logits))
                .map_err(|e| format!("{e:#}"))?;

            // The DepFormer, each slice on the last's code (the first
            // frames' forced pads instead).
            let forced: Vec<Option<u32>> = if step >= self.delay {
                vec![None; 2]
            } else {
                vec![Some(self.audio_pad); 2]
            };
            let mut last = text;
            for k in 0..self.slices {
                let mut dfrom = Map::new();
                let mut dkeep = Map::new();
                for l in 0..self.dep_layers {
                    for s in ["k", "v"] {
                        if k > 0 {
                            dfrom.insert(format!("{s}_{l}"), json!(format!("m.d{s}{l}")));
                        }
                        dkeep.insert(format!("{s}_{l}.next"), json!(format!("m.d{s}{l}")));
                    }
                }
                let d = hidden.len();
                let outs = run_body(
                    &format!("dep_{k}"),
                    vec![
                        Tensor::f32("hidden", vec![1, 1, d], &hidden),
                        Tensor::i32("token", vec![1], &[last as i32]),
                    ],
                    json!({"from": dfrom, "keep": dkeep}),
                )?;
                let tok = audio_lp
                    .sample(ArrayView1::from(&output(&outs, "logits")?))
                    .map_err(|e| format!("{e:#}"))?;
                let pos = if k == 0 {
                    step
                } else {
                    step.saturating_sub(self.delay)
                };
                audio[pos][k] = tok;
                last = forced.get(k).copied().flatten().unwrap_or(tok);
            }
            // A frame past the acoustic delay, with no pad, to Mimi.
            if step + 1 > self.delay {
                let frame = &audio[step - self.delay][..self.generated];
                if frame.iter().all(|&t| t < self.audio_pad) {
                    frames.push(frame.to_vec());
                }
            }
        }

        // Mimi: codes `[codebooks, frames]`.
        let (books, t) = (self.mimi_codebooks.min(self.generated), frames.len());
        if t == 0 {
            return Err("no audio frames (try a larger max_steps)".into());
        }
        let mut codes = vec![0i32; books * t];
        for (fi, frame) in frames.iter().enumerate() {
            for q in 0..books {
                codes[q * t + fi] = frame[q] as i32;
            }
        }
        run_body(
            "mimi_lookup",
            vec![Tensor::i32("codes", vec![books, t], &codes)],
            json!({"keep": {"z": "m.z"}}),
        )?;
        let outs = run_body("mimi_decode", vec![], json!({"from": {"z": "m.z"}}))?;
        let pcm = outs.last().ok_or("no audio")?.to_f32();
        let mut slots: Vec<String> = vec!["m.x".into(), "m.z".into()];
        for l in 0..self.layers {
            slots.push(format!("m.k{l}"));
            slots.push(format!("m.v{l}"));
        }
        for l in 0..self.dep_layers {
            slots.push(format!("m.dk{l}"));
            slots.push(format!("m.dv{l}"));
        }
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok((frames, pcm))
    }
}

impl Program for Moshi {
    fn init(c: Value) -> Result<Self, String> {
        let u = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let f = |k: &str| c[k].as_f64().ok_or(format!("config has no {k}"));
        Ok(Moshi {
            tokenizer: c["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .to_string(),
            layers: u("layers")?,
            heads: u("heads")?,
            head_dim: u("head_dim")?,
            rope_theta: f("rope_theta")? as f32,
            dep_layers: u("dep_layers")?,
            slices: u("slices")?,
            books: u("books")?,
            generated: u("generated")?,
            delay: u("acoustic_delay")?,
            audio_pad: u("audio_pad")? as u32,
            text_pad: u("text_pad")? as u32,
            text_start: u("text_start")? as u32,
            max_steps: u("max_steps")?,
            text_temperature: f("text_temperature")?,
            text_top_k: u("text_top_k")?,
            audio_temperature: f("audio_temperature")?,
            audio_top_k: u("audio_top_k")?,
            text_seed: u("text_seed")? as u64,
            audio_seed: u("audio_seed")? as u64,
            mimi_codebooks: u("mimi_codebooks")?,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = req.body["text"]
            .as_str()
            .or_else(|| o["text"].as_str())
            .ok_or(format!("{task} takes `text`"))?;
        let (frames, audio) = self.speak(text, o)?;
        match task {
            "synthesize" => Ok(Message::new(
                json!({"sample_rate": self.sample_rate}),
                vec![Tensor::f32("audio", vec![audio.len()], &audio)],
            )),
            _ => {
                let (t, books) = (frames.len(), frames[0].len());
                let codes: Vec<i32> = frames.iter().flatten().map(|&c| c as i32).collect();
                Ok(Message::tensors(vec![
                    Tensor::i32("codes", vec![t, books], &codes),
                    Tensor::f32("audio", vec![audio.len()], &audio),
                ]))
            }
        }
    }
}

rlx_program::program!(Moshi);
