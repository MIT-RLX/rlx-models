// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Greedy CTC transcription as an RLX program, answering `transcribe`: the
//! package's `features` function (audio → `[1, n_mels, frames]`), the
//! frames padded to the encoder's bucket (as the runner pads them), the
//! encoder (its last output the argmax id of each encoder frame), the ids
//! past the audio's own frames dropped, repeats collapsed, blanks removed,
//! and the pieces joined (`space` read as a space; `<…>` pieces dropped
//! when `strip_specials`).
//!
//! `config`: `sample_rate`; `features` (a function); `encoders`
//! (`[{frames, function}]`, smallest first; a longer input takes the
//! longest); `pad_value`; `subsampling` (`[[kernel, stride, pad], …]`: the
//! encoder's frames from the mel's); `blank`; `pieces`; `space`;
//! `strip_specials`.

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

struct Ctc {
    sample_rate: u64,
    features: String,
    encoders: Vec<(usize, String)>,
    pad_value: f32,
    subsampling: Vec<(usize, usize, usize)>,
    blank: u32,
    pieces: Vec<String>,
    space: String,
    strip_specials: bool,
}

impl Ctc {
    fn encoder_frames(&self, mut t: usize) -> usize {
        for &(k, s, p) in &self.subsampling {
            t = (t + 2 * p).saturating_sub(k) / s + 1;
        }
        t
    }

    fn text(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            if self.strip_specials && p.starts_with('<') && p.ends_with('>') {
                continue;
            }
            s.push_str(&p.replace(&self.space, " "));
        }
        s.trim().to_string()
    }
}

impl Program for Ctc {
    fn init(config: Value) -> Result<Self, String> {
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        let encoders: Vec<(usize, String)> = config["encoders"]
            .as_array()
            .ok_or("config has no encoders")?
            .iter()
            .filter_map(|e| {
                Some((
                    e["frames"].as_u64()? as usize,
                    e["function"].as_str()?.to_string(),
                ))
            })
            .collect();
        if encoders.is_empty() {
            return Err("config lists no encoder".into());
        }
        let subsampling = config["subsampling"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|c| {
                        let c = c.as_array()?;
                        Some((
                            c.first()?.as_u64()? as usize,
                            c.get(1)?.as_u64()? as usize,
                            c.get(2)?.as_u64()? as usize,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Ctc {
            sample_rate: config["sample_rate"]
                .as_u64()
                .ok_or("config has no sample_rate")?,
            features: s("features")?,
            encoders,
            pad_value: config["pad_value"].as_f64().unwrap_or(0.0) as f32,
            subsampling,
            blank: config["blank"].as_u64().ok_or("config has no blank")? as u32,
            pieces: config["pieces"]
                .as_array()
                .ok_or("config has no pieces")?
                .iter()
                .map(|p| p.as_str().unwrap_or("").to_string())
                .collect(),
            space: config["space"].as_str().unwrap_or("\u{2581}").to_string(),
            strip_specials: config["strip_specials"].as_bool().unwrap_or(true),
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "transcribe" {
            return Err(format!("a CTC model answers transcribe, not {task}"));
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
        let n = audio.len();
        let mel = run(&self.features, vec![Tensor::f32("audio", vec![n], &audio)])?;
        let mel = mel.first().ok_or("the features returned nothing")?;
        let (n_mels, frames) = match mel.shape.as_slice() {
            [1, m, f] => (*m, *f),
            s => return Err(format!("features are {s:?}, not [1, n_mels, frames]")),
        };
        let data = mel.to_f32();
        let (bucket, function) = self
            .encoders
            .iter()
            .find(|(b, _)| *b >= frames)
            .or(self.encoders.last())
            .cloned()
            .expect("non-empty");
        if frames > bucket {
            return Err(format!(
                "{frames} feature frames: longer than the longest encoder ({bucket})"
            ));
        }
        let mut padded = vec![self.pad_value; n_mels * bucket];
        for m in 0..n_mels {
            padded[m * bucket..m * bucket + frames]
                .copy_from_slice(&data[m * frames..(m + 1) * frames]);
        }
        let out = run(
            &function,
            vec![Tensor::f32("mel", vec![1, n_mels, bucket], &padded)],
        )?;
        let ids = out.last().ok_or("the encoder returned nothing")?.to_i64();
        let keep = self.encoder_frames(frames).min(ids.len());
        let mut tokens = Vec::new();
        let mut prev = None;
        for &id in &ids[..keep] {
            let id = id as u32;
            if Some(id) != prev && id != self.blank {
                tokens.push(id);
            }
            prev = Some(id);
        }
        let text = self.text(&tokens);
        Ok(Message::body(json!({"text": text, "ids": tokens})))
    }
}

rlx_program::program!(Ctc);
