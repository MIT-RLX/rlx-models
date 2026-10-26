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

//! Inflect-Nano's host logic as an RLX program, answering `synthesize`:
//! what `InflectNano::synthesize` does around its graphs — the crate's text
//! frontend (cleaning, normalization, CMUdict, g2p, the POS tagger for
//! homographs, blanks) over the assets under `frontend`; `encode`; the
//! durations (`expm1`, clamped, times `length_scale / speed`, rounded half
//! to even, at least `min_duration`); each frame's token, neighbours,
//! position bin and meta rows; `frames` and `vocoder`; the waveform's mean
//! removed, gained to −20 dB RMS, limited to −1 dBFS peak. Options:
//! `speaker` (a name), `speed`, `length_scale`, `pitch_scale`,
//! `energy_scale`.

#![allow(dead_code, unused_imports)]

#[path = "../../src/frontend/mod.rs"]
mod frontend;
#[path = "../../src/ops.rs"]
mod ops;
#[path = "../../src/weights.rs"]
mod weights;

use rlx_program::{Message, Program, Tensor, run_body};
use serde_json::{Map, Value, json};

struct Inflect {
    frontend: Option<frontend::English>,
    dir: String,
    add_blank: bool,
    speakers: Map<String, Value>,
    speaker: i32,
    tone_size: usize,
    lang_size: usize,
    max_frames: usize,
    abs_frame_bins: usize,
    min_duration: i64,
    max_duration: i64,
    sample_rate: u32,
}

/// `audio::normalize_audio`.
fn normalize(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return vec![0.0];
    }
    let n = audio.len() as f64;
    let mean = audio.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut out: Vec<f32> = audio.iter().map(|&v| (v as f64 - mean) as f32).collect();
    let ms = out.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n;
    let rms_db = 20.0 * (ms.sqrt() + 1e-9).log10();
    let gain = 10f64.powf((-20.0 - rms_db) / 20.0) as f32;
    for v in out.iter_mut() {
        *v *= gain;
    }
    let peak = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max) + 1e-9;
    let limit = 10f32.powf(-1.0 / 20.0);
    if peak > limit {
        let g = limit / peak;
        for v in out.iter_mut() {
            *v *= g;
        }
    }
    for v in out.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
    out
}

impl Inflect {
    fn english(&mut self) -> Result<&frontend::English, String> {
        if self.frontend.is_none() {
            let e = frontend::English::load(std::path::Path::new(&self.dir))
                .map_err(|e| format!("the text frontend: {e:#}"))?;
            self.frontend = Some(e);
        }
        Ok(self.frontend.as_ref().expect("loaded above"))
    }
}

impl Program for Inflect {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Inflect {
            frontend: None,
            dir: config["frontend"]
                .as_str()
                .ok_or("config has no frontend")?
                .to_string(),
            add_blank: config["add_blank"].as_bool().unwrap_or(true),
            speakers: config["speakers"].as_object().cloned().unwrap_or_default(),
            speaker: config["speaker"].as_i64().unwrap_or(0) as i32,
            tone_size: u("tone_size")?,
            lang_size: u("lang_size")?,
            max_frames: u("max_frames")?,
            abs_frame_bins: u("abs_frame_bins")?,
            min_duration: config["min_duration"].as_i64().unwrap_or(1),
            max_duration: config["max_duration"].as_i64().unwrap_or(80),
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("Inflect-Nano answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes `text`")?;
        let o = req.body["options"].clone();
        let f = |k: &str, d: f32| o[k].as_f64().map_or(d, |v| v as f32);
        let (speed, length_scale) = (f("speed", 1.0), f("length_scale", 1.0));
        let (pitch_scale, energy_scale) = (f("pitch_scale", 1.0), f("energy_scale", 1.0));
        let speaker = match o["speaker"].as_str() {
            Some(name) => {
                self.speakers
                    .get(name)
                    .and_then(Value::as_i64)
                    .ok_or(format!("Inflect-Nano has no speaker {name}"))? as i32
            }
            None => self.speaker,
        };
        let add_blank = self.add_blank;
        let (phone, tone, lang) = self
            .english()?
            .text_to_ids(text, add_blank)
            .map_err(|e| format!("{e:#}"))?;
        let s = phone.len();
        if s == 0 {
            return Err("the text has no phonemes".into());
        }
        let i32s = |v: &[i64], cap: usize| -> Vec<i32> {
            v.iter().map(|&x| (x as usize).min(cap) as i32).collect()
        };
        let phone_ids: Vec<i32> = phone.iter().map(|&x| x as i32).collect();
        let outs = run_body(
            "encode",
            vec![
                Tensor::i32("phone", vec![s], &phone_ids),
                Tensor::i32("tone", vec![s], &i32s(&tone, self.tone_size - 1)),
                Tensor::i32("lang", vec![s], &i32s(&lang, self.lang_size - 1)),
                Tensor::i32("speaker", vec![1], &[speaker]),
                Tensor::f32("energy_scale", vec![1], &[energy_scale]),
                Tensor::f32("pitch_scale", vec![1], &[pitch_scale]),
            ],
            json!({"keep": {"conditioned": "i.cond", "pitch": "i.pitch"}}),
        )?;
        let log_dur = outs
            .iter()
            .find(|t| t.name == "log_duration")
            .ok_or("no log_duration")?
            .to_f32();
        let scale = length_scale / speed.max(1e-3);
        let durations: Vec<usize> = log_dur
            .iter()
            .map(|&v| {
                let pd = v.exp_m1().clamp(0.0, self.max_duration as f32) * scale;
                (pd.round_ties_even() as i64).max(self.min_duration) as usize
            })
            .collect();

        // Each frame's token, its neighbours, its position bin and meta rows.
        let total: usize = durations.iter().sum();
        let t = total.min(self.max_frames).max(1);
        let token_count = s.max(1);
        let (mut token, mut prev, mut next, mut abs) =
            (vec![0i32; t], vec![0i32; t], vec![0i32; t], vec![0i32; t]);
        let mut meta = vec![0f32; t * 8];
        let mut fi = 0usize;
        for (i, &dur) in durations.iter().enumerate() {
            if dur == 0 {
                continue;
            }
            let token_pos = i as f32 / (token_count.max(2) - 1) as f32;
            let log1p_dur = (dur as f32).ln_1p() / 6.0;
            for j in 0..dur {
                if fi >= t {
                    break;
                }
                let rel = if dur <= 1 {
                    0.0
                } else {
                    j as f32 / (dur - 1) as f32
                };
                let m = &mut meta[fi * 8..fi * 8 + 8];
                m[0] = rel;
                m[1] = 1.0 - rel;
                m[2] = 1.0 - (rel * 2.0 - 1.0).abs();
                m[3] = (rel * std::f32::consts::PI).sin();
                m[4] = (rel * std::f32::consts::PI).cos();
                m[5] = token_pos;
                m[6] = log1p_dur;
                m[7] = dur as f32 / 40.0;
                token[fi] = i as i32;
                prev[fi] = i.saturating_sub(1) as i32;
                next[fi] = (i + 1).min(s - 1) as i32;
                fi += 1;
            }
        }
        for (f, a) in abs.iter_mut().enumerate() {
            *a = ((f * self.abs_frame_bins) / self.max_frames.max(1)).min(self.abs_frame_bins - 1)
                as i32;
        }
        run_body(
            "frames",
            vec![
                Tensor::i32("token", vec![t], &token),
                Tensor::i32("prev", vec![t], &prev),
                Tensor::i32("next", vec![t], &next),
                Tensor::i32("abs", vec![t], &abs),
                Tensor::f32("meta", vec![t, 8], &meta),
            ],
            json!({"from": {"conditioned": "i.cond", "pitch": "i.pitch"}, "keep": {"mel": "i.mel"}}),
        )?;
        let outs = run_body("vocoder", vec![], json!({"from": {"mel": "i.mel"}}))?;
        let raw = outs.first().ok_or("no audio")?.to_f32();
        rlx_program::host(
            "slot.drop",
            Message::body(json!({"slots": ["i.cond", "i.pitch", "i.mel"]})),
        )?;
        let wav = normalize(&raw);
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Inflect);
