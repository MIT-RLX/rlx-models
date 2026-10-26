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

//! Qwen3-TTS's host logic as an RLX program, answering `synthesize` (and
//! `run`, the codec frames): `synthesize_custom_voice_greedy` over the
//! package's graphs — the assistant text tokenized, its rows and the codec
//! rows (`text_rows`, `codec_rows`) laid out as
//! `build_custom_voice_prompt` does (the think tokens and language, the
//! speaker, the text body over codec pads, `tts_eos`, the bos row); the
//! talker over it (`prompt`); then each frame as `codec_frame_step`: the
//! codec head (`head`), the repetition penalty on the first codes so far,
//! the greedy first code with the reserved range masked but end of speech
//! (end of speech before `min_new_tokens` frames emits nothing and goes on),
//! the code predictor (`cp_prompt` on the talker's hidden state and the first
//! code's row, then `cp_head_{k}` greedy and `cp_embed_{k}` / `cp_step` a
//! group), the frame's embedding sum (the first code's row, `tts_pad`, each
//! group's row) through the talker (`step`); and the frames through the
//! speech decoder (`speech`). Options: `speaker`, `language`,
//! `max_new_tokens`, `repetition_penalty`.

use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Map, Value, json};

struct Tts {
    tokenizer: String,
    layers: usize,
    cp_layers: usize,
    groups: usize,
    hidden: usize,
    vocab: usize,
    rows: usize,
    cp_rows: usize,
    eos: u32,
    codec_pad: u32,
    codec_bos: u32,
    codec_think: u32,
    codec_nothink: u32,
    codec_think_bos: u32,
    codec_think_eos: u32,
    speakers: Map<String, Value>,
    languages: Map<String, Value>,
    tts_bos: u32,
    tts_eos: u32,
    tts_pad: u32,
    speaker: String,
    language: String,
    repetition_penalty: f32,
    max_new_tokens: usize,
    min_new_tokens: usize,
    sample_rate: u32,
}

fn out(outs: &[Tensor], n: &str) -> Result<Vec<f32>, String> {
    Ok(outs
        .iter()
        .find(|t| t.name == n)
        .ok_or(format!("no output {n}"))?
        .to_f32())
}

/// The first largest of `logits[..n]`.
fn argmax(logits: &[f32], n: usize) -> u32 {
    let (mut best, mut v) = (0usize, f32::NEG_INFINITY);
    for (i, &x) in logits.iter().take(n).enumerate() {
        if x > v {
            v = x;
            best = i;
        }
    }
    best as u32
}

impl Tts {
    /// `ids`' rows through `function` (`text_rows` / `codec_rows`), `[n][h]`.
    fn rows(&self, function: &str, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let n = ids.len();
        let v: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let outs = run_body(function, vec![Tensor::i32("ids", vec![n], &v)], json!({}))?;
        let flat = out(&outs, "rows")?;
        Ok(flat.chunks(self.hidden).map(|c| c.to_vec()).collect())
    }

    fn speak(&self, text: &str, o: &Value) -> Result<(Vec<Vec<u32>>, Vec<f32>), String> {
        let speaker = o["speaker"]
            .as_str()
            .unwrap_or(&self.speaker)
            .to_lowercase();
        let language = o["language"]
            .as_str()
            .unwrap_or(&self.language)
            .to_lowercase();
        let rep = o["repetition_penalty"]
            .as_f64()
            .map_or(self.repetition_penalty, |v| v as f32);
        let max_steps = o["max_new_tokens"]
            .as_u64()
            .map_or(self.max_new_tokens, |v| v as usize)
            .max(2);
        let min_frames = self.min_new_tokens.max(1);

        // The prompt (`build_custom_voice_prompt`).
        let assistant = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let r = host(
            "tokenizer.encode",
            Message::body(
                json!({"tokenizer": self.tokenizer, "text": assistant, "special": false}),
            ),
        )?;
        let ids: Vec<u32> = r.body["ids"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as u32)
                    .collect()
            })
            .unwrap_or_default();
        if ids.len() < 8 {
            return Err(format!("prompt too short ({} tokens)", ids.len()));
        }
        let spk = self
            .speakers
            .get(&speaker)
            .and_then(Value::as_u64)
            .ok_or(format!("unknown speaker {speaker:?}"))? as u32;
        let lang = match language.as_str() {
            "auto" => None,
            l => Some(
                self.languages
                    .get(l)
                    .and_then(Value::as_u64)
                    .ok_or(format!("unknown language {l:?}"))? as u32,
            ),
        };
        let text_rows = self.rows("text_rows", &ids)?;
        let tts = self.rows("text_rows", &[self.tts_bos, self.tts_eos, self.tts_pad])?;
        let (tts_bos, tts_eos, tts_pad) = (&tts[0], &tts[1], &tts[2]);
        let mut codec_ids = match lang {
            None => vec![
                self.codec_nothink,
                self.codec_think_bos,
                self.codec_think_eos,
            ],
            Some(l) => vec![
                self.codec_think,
                self.codec_think_bos,
                l,
                self.codec_think_eos,
            ],
        };
        codec_ids.push(spk);
        codec_ids.extend([self.codec_pad, self.codec_bos]);
        let codec = self.rows("codec_rows", &codec_ids)?;
        let n_codec = codec.len();
        let pad_codec = &codec[n_codec - 2];
        let bos_codec = &codec[n_codec - 1];
        let add = |a: &[f32], b: &[f32]| -> Vec<f32> {
            let mut r = a.to_vec();
            for (x, y) in r.iter_mut().zip(b) {
                *x += y;
            }
            r
        };
        let mut seq: Vec<Vec<f32>> = text_rows.iter().take(3).cloned().collect();
        for (i, piece) in codec.iter().take(n_codec - 1).enumerate() {
            seq.push(add(piece, if i < n_codec - 2 { tts_pad } else { tts_bos }));
        }
        let text_end = ids.len().saturating_sub(5);
        for row in &text_rows[3..text_end] {
            seq.push(add(row, pad_codec));
        }
        seq.push(add(tts_eos, pad_codec));
        seq.push(add(tts_pad, bos_codec));

        // The talker over it.
        let t = seq.len();
        let h = self.hidden;
        let cap = (t + max_steps + 1).next_power_of_two().min(self.rows);
        if t + 1 >= cap {
            return Err(format!("the prompt's {t} rows reach past {cap} positions"));
        }
        let flat: Vec<f32> = seq.concat();
        let mut keep = Map::new();
        let mut from = Map::new();
        let mut step_keep = Map::new();
        for l in 0..self.layers {
            for s in ["k", "v"] {
                let slot = json!(format!("t.{s}{l}"));
                keep.insert(format!("past_{s}_{l}"), slot.clone());
                from.insert(format!("past_{s}_{l}"), slot.clone());
                step_keep.insert(format!("past_{s}_{l}.next"), slot);
            }
        }
        let outs = run_body(
            "prompt",
            vec![Tensor::f32("inputs_embeds", vec![1, t, h], &flat)],
            json!({"keep": keep, "dims": {"cap": cap}}),
        )?;
        let mut hidden = out(&outs, "hidden")?;

        let (mut cp_keep, mut cp_from, mut cp_step_keep) = (Map::new(), Map::new(), Map::new());
        for l in 0..self.cp_layers {
            for s in ["k", "v"] {
                let slot = json!(format!("t.c{s}{l}"));
                cp_keep.insert(format!("past_{s}_{l}"), slot.clone());
                cp_from.insert(format!("past_{s}_{l}"), slot.clone());
                cp_step_keep.insert(format!("past_{s}_{l}.next"), slot);
            }
        }

        // The codec loop.
        let reserved = self.vocab.saturating_sub(1024);
        let mut past_g0: Vec<u32> = Vec::new();
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut position = t;
        for _ in 0..max_steps {
            let outs = run_body(
                "head",
                vec![Tensor::f32("hidden", vec![1, 1, h], &hidden)],
                json!({}),
            )?;
            let mut logits = out(&outs, "logits")?;
            if (rep - 1.0).abs() >= 1e-6 {
                for &tok in &past_g0 {
                    let i = tok as usize;
                    if i < logits.len() {
                        if logits[i] > 0.0 {
                            logits[i] /= rep;
                        } else {
                            logits[i] *= rep;
                        }
                    }
                }
            }
            let (mut g0, mut best) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in logits.iter().enumerate() {
                if i >= reserved && i != self.eos as usize {
                    continue;
                }
                if v > best {
                    best = v;
                    g0 = i;
                }
            }
            let g0 = g0 as u32;
            if g0 == self.eos {
                if frames.len() >= min_frames {
                    break;
                }
                continue;
            }
            // The code predictor's groups.
            let g0_row = &self.rows("codec_rows", &[g0])?[0];
            let mut emb = g0_row.clone();
            for (x, y) in emb.iter_mut().zip(tts_pad) {
                *x += y;
            }
            let mut two = hidden.clone();
            two.extend_from_slice(g0_row);
            let outs = run_body(
                "cp_prompt",
                vec![Tensor::f32("inputs_embeds", vec![1, 2, h], &two)],
                json!({"keep": cp_keep, "dims": {"cp_cap": self.cp_rows}}),
            )?;
            let mut cp_hidden = out(&outs, "hidden")?;
            let mut codes = vec![g0];
            for k in 0..self.groups {
                let outs = run_body(
                    &format!("cp_head_{k}"),
                    vec![Tensor::f32("hidden", vec![1, 1, h], &cp_hidden)],
                    json!({}),
                )?;
                let logits = out(&outs, "logits")?;
                let tok = argmax(&logits, logits.len());
                codes.push(tok);
                let outs = run_body(
                    &format!("cp_embed_{k}"),
                    vec![Tensor::i32("token", vec![1], &[tok as i32])],
                    json!({}),
                )?;
                let row = out(&outs, "row")?;
                for (x, y) in emb.iter_mut().zip(&row) {
                    *x += y;
                }
                if k + 1 < self.groups {
                    let outs = run_body(
                        "cp_step",
                        vec![
                            Tensor::f32("inputs_embeds", vec![1, 1, h], &row),
                            Tensor::f32("position", vec![1], &[(k + 2) as f32]),
                        ],
                        json!({"from": cp_from, "keep": cp_step_keep}),
                    )?;
                    cp_hidden = out(&outs, "hidden")?;
                }
            }
            past_g0.push(g0);
            frames.push(codes);
            // The talker on the frame's embedding.
            let outs = run_body(
                "step",
                vec![
                    Tensor::f32("inputs_embeds", vec![1, 1, h], &emb),
                    Tensor::f32("position", vec![1], &[position as f32]),
                ],
                json!({"from": from, "keep": step_keep}),
            )?;
            hidden = out(&outs, "hidden")?;
            position += 1;
            if position + 1 >= cap {
                break;
            }
        }

        // The speech decoder.
        let n = frames.len();
        if n == 0 {
            return Err("no codec frames".into());
        }
        let mut codes = vec![0i32; 16 * n];
        for (fi, frame) in frames.iter().enumerate() {
            for (k, &c) in frame.iter().enumerate() {
                codes[k * n + fi] = c as i32;
            }
        }
        let positions: Vec<i32> = (0..n as i32).collect();
        let outs = run_body(
            "speech",
            vec![
                Tensor::i32("codes", vec![16, n], &codes),
                Tensor::i32("positions", vec![n], &positions),
            ],
            json!({}),
        )?;
        let audio = out(&outs, "audio")?;
        let mut slots: Vec<String> = Vec::new();
        for l in 0..self.layers {
            slots.push(format!("t.k{l}"));
            slots.push(format!("t.v{l}"));
        }
        for l in 0..self.cp_layers {
            slots.push(format!("t.ck{l}"));
            slots.push(format!("t.cv{l}"));
        }
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok((frames, audio))
    }
}

impl Program for Tts {
    fn init(c: Value) -> Result<Self, String> {
        let u = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let s = |k: &str| c[k].as_str().unwrap_or("").to_string();
        Ok(Tts {
            tokenizer: s("tokenizer"),
            layers: u("layers")?,
            cp_layers: u("cp_layers")?,
            groups: u("groups")?,
            hidden: u("hidden")?,
            vocab: u("vocab")?,
            rows: u("rows")?,
            cp_rows: u("cp_rows")?,
            eos: u("eos")? as u32,
            codec_pad: u("codec_pad")? as u32,
            codec_bos: u("codec_bos")? as u32,
            codec_think: u("codec_think")? as u32,
            codec_nothink: u("codec_nothink")? as u32,
            codec_think_bos: u("codec_think_bos")? as u32,
            codec_think_eos: u("codec_think_eos")? as u32,
            speakers: c["speakers"].as_object().cloned().unwrap_or_default(),
            languages: c["languages"].as_object().cloned().unwrap_or_default(),
            tts_bos: u("tts_bos")? as u32,
            tts_eos: u("tts_eos")? as u32,
            tts_pad: u("tts_pad")? as u32,
            speaker: s("speaker"),
            language: s("language"),
            repetition_penalty: c["repetition_penalty"].as_f64().unwrap_or(1.05) as f32,
            max_new_tokens: u("max_new_tokens")?,
            min_new_tokens: u("min_new_tokens")?,
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
                let n = frames.len();
                let codes: Vec<i32> = frames.iter().flatten().map(|&c| c as i32).collect();
                Ok(Message::tensors(vec![
                    Tensor::i32("codes", vec![n, 16], &codes),
                    Tensor::f32("audio", vec![audio.len()], &audio),
                ]))
            }
        }
    }
}

rlx_program::program!(Tts);
