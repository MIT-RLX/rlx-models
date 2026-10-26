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

//! Nemotron ASR's host logic as an RLX program, answering `transcribe`:
//! what `NemotronAsr::transcribe` does — the features and the encoder
//! (`features`, `encoder`, kept in slots), every frame through the language
//! prompt kernel and the joint's projection (`enc_proj`, with the language
//! one-hot: `options.language`, else the config's), then the greedy RNN-T
//! search: from the prediction net's start-of-sequence output (its state
//! left at zero), each frame's argmax (`joint`) a blank that moves to the
//! next frame or a label that steps the prediction net (`pred_step`),
//! at most `max_symbols_per_step` labels a frame (a TDT model's duration
//! head moves the frame by `durations[argmax]` instead); the labels' pieces
//! joined (`▁` a space, `<…>` dropped).

use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Map, Value, json};

struct Asr {
    sample_rate: u64,
    hop: usize,
    subsampling: Vec<usize>,
    languages: usize,
    language_map: Map<String, Value>,
    language: String,
    pred_layers: usize,
    pred_hidden: usize,
    blank: i64,
    /// TDT durations (empty: RNN-T, a label stays on its frame).
    durations: Vec<i64>,
    /// Keep the start-of-sequence step's state.
    sos_carry: bool,
    max_symbols: usize,
    pieces: Vec<String>,
    space: String,
}

fn named(outs: &[Tensor], n: &str) -> Result<Tensor, String> {
    outs.iter()
        .find(|t| t.name == n)
        .cloned()
        .ok_or_else(|| format!("no output {n}"))
}

impl Asr {
    fn text(&self, ids: &[i64]) -> String {
        let mut s = String::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            if p.starts_with('<') && p.ends_with('>') {
                continue;
            }
            s.push_str(&p.replace(&self.space, " "));
        }
        s.trim().to_string()
    }

    /// The prediction net on `id` (`None`: the start of the sequence),
    /// its output in `a.pred`; its state carried when `carry`.
    fn predict(&self, id: Option<i64>, carry: bool) -> Result<(), String> {
        let mut from = Map::new();
        let mut keep = Map::new();
        keep.insert("out".into(), json!("a.pred"));
        for l in 0..self.pred_layers {
            for s in ["h", "c"] {
                from.insert(format!("{s}{l}"), json!(format!("a.{s}{l}")));
                if carry {
                    keep.insert(format!("{s}{l}.next"), json!(format!("a.{s}{l}")));
                }
            }
        }
        let outs = run_body(
            "pred_step",
            vec![
                Tensor::i32("id", vec![1], &[id.unwrap_or(0) as i32]),
                Tensor::f32("on", vec![1, 1], &[if id.is_some() { 1.0 } else { 0.0 }]),
            ],
            json!({"from": from, "keep": keep}),
        )?;
        drop(outs);
        Ok(())
    }
}

impl Program for Asr {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        Ok(Asr {
            sample_rate: u("sample_rate")?,
            hop: u("hop")? as usize,
            subsampling: config["subsampling"]
                .as_array()
                .ok_or("config has no subsampling")?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect(),
            languages: u("languages")? as usize,
            language_map: config["language_map"]
                .as_object()
                .cloned()
                .unwrap_or_default(),
            language: config["language"].as_str().unwrap_or("en-US").to_string(),
            pred_layers: u("pred_layers")? as usize,
            pred_hidden: u("pred_hidden")? as usize,
            blank: u("blank")? as i64,
            durations: config["durations"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_i64).collect())
                .unwrap_or_default(),
            sos_carry: config["sos_carry"].as_bool().unwrap_or(false),
            max_symbols: u("max_symbols_per_step")?.max(1) as usize,
            pieces: config["pieces"]
                .as_array()
                .ok_or("config has no pieces")?
                .iter()
                .map(|p| p.as_str().unwrap_or("").to_string())
                .collect(),
            space: config["space"].as_str().unwrap_or("\u{2581}").to_string(),
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "transcribe" {
            return Err(format!("Nemotron ASR answers transcribe, not {task}"));
        }
        let rate = req.body["sample_rate"].as_u64().unwrap_or(self.sample_rate);
        if rate != self.sample_rate {
            return Err(format!(
                "the audio is at {rate} Hz; this model takes {} Hz",
                self.sample_rate
            ));
        }
        let audio = req
            .tensor("audio")
            .ok_or("transcribe takes `audio`")?
            .to_f32();
        if audio.is_empty() {
            return Err("the audio is empty".into());
        }
        let o = &req.body["options"];
        let mut proj_inputs = Vec::new();
        if self.languages > 0 {
            let code = o["language"].as_str().unwrap_or(&self.language);
            let idx = self
                .language_map
                .get(code)
                .and_then(Value::as_u64)
                .ok_or(format!("no language {code}"))? as usize;
            let mut lang = vec![0f32; self.languages];
            lang[idx.min(self.languages - 1)] = 1.0;
            proj_inputs.push(Tensor::f32("lang", vec![1, self.languages], &lang));
        }

        let n = audio.len();
        run_body(
            "features",
            vec![Tensor::f32("audio", vec![n], &audio)],
            json!({"keep": {"mel": "a.mel"}}),
        )?;
        run_body(
            "subsampling",
            vec![],
            json!({"from": {"mel": "a.mel"}, "keep": {"x": "a.x"}}),
        )?;
        run_body(
            "encoder",
            vec![],
            json!({"from": {"x": "a.x"}, "keep": {"enc": "a.enc"}}),
        )?;
        run_body(
            "enc_proj",
            proj_inputs,
            json!({"from": {"enc": "a.enc"}, "keep": {"proj": "a.proj"}}),
        )?;
        let mut frames = n / self.hop + 1;
        for &pad in &self.subsampling {
            frames = (frames + pad - 3) / 2 + 1;
        }
        let zeros = vec![0f32; self.pred_hidden];
        for l in 0..self.pred_layers {
            for s in ["h", "c"] {
                host(
                    "slot.set",
                    Message::new(
                        json!({"slot": format!("a.{s}{l}")}),
                        vec![Tensor::f32(
                            format!("{s}{l}"),
                            vec![1, self.pred_hidden],
                            &zeros,
                        )],
                    ),
                )?;
            }
        }
        // The start of the sequence, its state carried or not.
        self.predict(None, self.sos_carry)?;
        let mut ids = Vec::new();
        let (mut t, mut last, mut at_last) = (0usize, None, 0usize);
        while t < frames {
            let outs = run_body(
                "joint",
                vec![Tensor::i32("at", vec![1], &[t as i32])],
                json!({"from": {"proj": "a.proj", "pred": "a.pred"}}),
            )?;
            let label = named(&outs, "label")?.to_i64()[0];
            let advance = match self.durations.is_empty() {
                true => 0,
                false => {
                    let di = named(&outs, "duration")?.to_i64()[0] as usize;
                    *self
                        .durations
                        .get(di)
                        .ok_or("duration index out of range")? as usize
                }
            };
            if label == self.blank {
                // A blank that would stall moves on a frame.
                t += advance.max(1);
                continue;
            }
            ids.push(label);
            self.predict(Some(label), true)?;
            if last == Some(t) {
                at_last += 1;
            } else {
                last = Some(t);
                at_last = 1;
            }
            t += advance;
            if at_last >= self.max_symbols && Some(t) == last {
                t += 1;
            }
        }
        let mut slots: Vec<String> = ["a.mel", "a.x", "a.enc", "a.proj", "a.pred"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for l in 0..self.pred_layers {
            slots.push(format!("a.h{l}"));
            slots.push(format!("a.c{l}"));
        }
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok(Message::body(json!({"text": self.text(&ids), "ids": ids})))
    }
}

rlx_program::program!(Asr);
