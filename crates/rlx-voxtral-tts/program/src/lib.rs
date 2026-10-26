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

//! Voxtral-4B-TTS's host logic as an RLX program, answering `synthesize`:
//! what `NativeTtsEngine::synthesize` does — the speech prompt (`[BOS]
//! [BEGIN_AUDIO] [AUDIO]×n [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT]
//! [BEGIN_AUDIO]`, the text Tekken-encoded), its rows (`embed`) with the
//! voice's rows at the audio slots; the backbone over them (`prefill`, its
//! caches kept in slots, one length a request); each frame the semantic
//! code (the head's argmax, the empty code and the padding masked) and the
//! acoustic ones (`acoustic` from the frame's noise, the runner's `StdRng`
//! stream, quantized to 21 levels), until the end of audio; the frame fed
//! back (`step`); the codec over the frames (`codec`). `run` answers the
//! frames' codes (`options.frames` decodes given ones).

#[path = "../../src/rng.rs"]
mod rng;
mod tekken;

use rlx_program::{Message, Program, Tensor, host, run, run_body, run_with};
use serde_json::{Value, json};
use tekken::Tekken;

struct Voxtral {
    files: String,
    tok: Tekken,
    voices: serde_json::Map<String, Value>,
    voice: String,
    layers: usize,
    hidden: usize,
    audio_token: u32,
    begin_audio: u32,
    text_to_audio: u32,
    audio_to_text: u32,
    bos: u32,
    end_audio: u32,
    empty_audio: u32,
    special: u32,
    semantic_limit: usize,
    acoustic_levels: usize,
    n_acoustic: usize,
    cfg_alpha: f32,
    seed: u64,
    max_frames: usize,
    max_context: usize,
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

/// The lowest index of the largest value (`rlx_search::argmax`).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

impl Voxtral {
    fn read_f32(&self, rel: &str) -> Result<Vec<f32>, String> {
        let p = format!("{}/{rel}", self.files);
        Ok(std::fs::read(&p)
            .map_err(|e| format!("read {p}: {e}"))?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// The speech prompt for `text` with `n` audio slots.
    fn prompt(&self, text: &str, n: usize) -> Result<Vec<u32>, String> {
        let ids = self.tok.encode(text)?;
        let mut out = vec![self.bos, self.begin_audio];
        out.extend(std::iter::repeat_n(self.audio_token, n));
        out.push(self.text_to_audio);
        out.extend(ids);
        out.push(self.audio_to_text);
        out.push(self.begin_audio);
        Ok(out)
    }

    /// The frames' codes (`[37]` each, the special tokens counted).
    fn frames(&self, text: &str, o: &Value) -> Result<Vec<Vec<u32>>, String> {
        let voice = o["voice"].as_str().unwrap_or(&self.voice);
        let n = self.voices[voice]
            .as_u64()
            .ok_or_else(|| format!("no voice {voice:?}"))? as usize;
        let rows = self.read_f32(&format!("voices/{voice}.f32"))?;
        let prompt = self.prompt(text, n)?;
        let max_frames = o["max_frames"]
            .as_u64()
            .map_or(self.max_frames, |v| v as usize);
        let alpha = o["cfg_alpha"].as_f64().map_or(self.cfg_alpha, |v| v as f32);
        let seed = o["seed"].as_u64().unwrap_or(self.seed);

        // The prompt's rows, the voice's at its audio slots.
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let t = ids.len();
        let mut x = first(
            run("embed", vec![Tensor::i32("ids", vec![t], &ids)])?,
            "embed",
        )?
        .to_f32();
        let h = self.hidden;
        let mut vi = 0;
        for (i, &tid) in prompt.iter().enumerate() {
            if tid == self.audio_token && vi < n {
                x[i * h..(i + 1) * h].copy_from_slice(&rows[vi * h..(vi + 1) * h]);
                vi += 1;
            }
        }

        let cap = (t + max_frames + 1).div_ceil(256) * 256;
        let cap = cap.min(self.max_context);
        let max_frames = max_frames.min(cap - t);
        let names: Vec<(String, String)> = (0..self.layers)
            .flat_map(|i| [format!("k{i}"), format!("v{i}")])
            .map(|n| (n.clone(), format!("voxtral.{n}")))
            .collect();
        let nexts: Vec<(String, String)> = names
            .iter()
            .map(|(n, s)| (format!("{n}.next"), s.clone()))
            .collect();
        let keep: serde_json::Map<String, Value> = names
            .iter()
            .map(|(a, b)| (a.clone(), Value::String(b.clone())))
            .collect();
        let mut hidden = first(
            run_body(
                "prefill",
                vec![Tensor::f32("x", vec![1, t, h], &x)],
                json!({"keep": keep, "dims": {"cap": cap}}),
            )?,
            "prefill",
        )?
        .to_f32();

        let mut frames = Vec::new();
        let mut mask = vec![0.0f32; cap];
        mask[..t].fill(1.0);
        for fi in 0..max_frames {
            let mut x0 = vec![0f32; self.n_acoustic];
            rng::fill_standard_normal(&mut x0, rng::frame_seed(seed, fi));
            let outs = run(
                "acoustic",
                vec![
                    Tensor::f32("h", vec![1, h], &hidden),
                    Tensor::f32("x0", vec![1, self.n_acoustic], &x0),
                    Tensor::f32("cfg_alpha", vec![1], &[alpha]),
                ],
            )?;
            let get = |n: &str| {
                outs.iter()
                    .find(|t| t.name == n)
                    .map(Tensor::to_f32)
                    .ok_or_else(|| format!("acoustic gave no {n}"))
            };
            let mut sem = get("semantic")?;
            sem[self.empty_audio as usize] = f32::NEG_INFINITY;
            for v in sem.iter_mut().skip(self.semantic_limit) {
                *v = f32::NEG_INFINITY;
            }
            let semantic = argmax(&sem) as u32;
            if semantic == self.end_audio {
                break;
            }
            let sampled = get("sampled")?;
            let mut frame = vec![semantic];
            for &v in &sampled {
                let v = v.clamp(-1.0, 1.0);
                let scaled = ((v + 1.0) / 2.0) * (self.acoustic_levels as f32 - 1.0);
                frame.push(scaled.round() as u32 + self.special);
            }
            let pos = t + fi;
            mask[pos] = 1.0;
            let codes: Vec<i32> = frame.iter().map(|&c| c as i32).collect();
            hidden = first(
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
            frames.push(frame);
        }
        let slots: Vec<&str> = names.iter().map(|(_, s)| s.as_str()).collect();
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok(frames)
    }

    /// Frames → 24 kHz audio, the codec over the codebook ids.
    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        let f = frames.len();
        let sem: Vec<i32> = frames
            .iter()
            .map(|fr| fr[0].saturating_sub(self.special) as i32)
            .collect();
        let ac: Vec<f32> = frames
            .iter()
            .flat_map(|fr| {
                fr[1..]
                    .iter()
                    .map(|&c| c.saturating_sub(self.special) as f32)
            })
            .collect();
        Ok(first(
            run(
                "codec",
                vec![
                    Tensor::i32("semantic", vec![f], &sem),
                    Tensor::f32("acoustic", vec![f, self.n_acoustic], &ac),
                ],
            )?,
            "codec",
        )?
        .to_f32())
    }
}

impl Program for Voxtral {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let files = config["files"].as_str().unwrap_or("/package").to_string();
        let tok = Tekken::load(&format!("{files}/tekken.json"))?;
        Ok(Voxtral {
            tok,
            voices: config["voices"].as_object().cloned().unwrap_or_default(),
            voice: config["voice"].as_str().unwrap_or("neutral_female").into(),
            layers: u("layers")? as usize,
            hidden: u("hidden")? as usize,
            audio_token: u("audio_token")? as u32,
            begin_audio: u("begin_audio")? as u32,
            text_to_audio: u("text_to_audio")? as u32,
            audio_to_text: u("audio_to_text")? as u32,
            bos: u("bos")? as u32,
            end_audio: u("end_audio")? as u32,
            empty_audio: u("empty_audio")? as u32,
            special: u("special")? as u32,
            semantic_limit: u("semantic_limit")? as usize,
            acoustic_levels: u("acoustic_levels")? as usize,
            n_acoustic: u("n_acoustic")? as usize,
            cfg_alpha: config["cfg_alpha"].as_f64().unwrap_or(1.2) as f32,
            seed: u("seed")?,
            max_frames: u("max_frames")? as usize,
            max_context: u("max_context")? as usize,
            sample_rate: u("sample_rate")? as u32,
            files,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().unwrap_or(""),
            _ => {
                return Err(format!(
                    "Voxtral TTS answers synthesize and run, not {task}"
                ));
            }
        };
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
            None => {
                if text.is_empty() {
                    return Err("speech text must be non-empty".into());
                }
                self.frames(text, o)?
            }
        };
        if frames.is_empty() {
            return Err("Voxtral produced zero audio frames".into());
        }
        if task == "run" {
            let flat: Vec<i32> = frames.iter().flatten().map(|&c| c as i32).collect();
            return Ok(Message::tensors(vec![Tensor::i32(
                "frames",
                vec![frames.len(), 37],
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

rlx_program::program!(Voxtral);
