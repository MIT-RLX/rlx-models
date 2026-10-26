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

//! Voxtral's host logic as an RLX program, answering `transcribe`: what
//! `VoxtralRunner::transcribe_wav` does — the signal zero-padded to whole
//! 30 s windows and reflect-padded by `n_fft / 2` (`mel`), the mel laid out
//! a window a row (`encoder`, `projector`, kept in slots); the transcription
//! prompt (`<s>[INST][BEGIN_AUDIO]`, a placeholder per audio row,
//! `[/INST]lang:<code>[TRANSCRIBE]`, `options.language` the code, `auto`
//! none), its embedding with the audio's rows (`embed`), the decoder over it
//! (`prompt`, its caches kept), then a greedy step a token (`step`) to `</s>`;
//! the ids rendered through `tekken.json` (control tokens dropped).

use base64::Engine;
use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Map, Value, json};

const BOS: i32 = 1;
const INST: i32 = 3;
const END_INST: i32 = 4;
const BEGIN_AUDIO: i32 = 25;
const TRANSCRIBE: i32 = 34;
const LANG: i32 = 9909;
const COLON: i32 = 1058;

struct Asr {
    tekken: String,
    sample_rate: u64,
    hop: usize,
    n_fft: usize,
    n_mels: usize,
    window_frames: usize,
    rows_per_window: usize,
    layers: usize,
    rows: usize,
    max_tokens: usize,
    audio: i32,
    eos: i64,
    languages: Map<String, Value>,
    language: String,
    /// Each id's bytes past the control tokens, read on first use.
    vocab: Option<(usize, Vec<Vec<u8>>)>,
}

fn named(outs: &[Tensor], n: &str) -> Result<Tensor, String> {
    outs.iter()
        .find(|t| t.name == n)
        .cloned()
        .ok_or_else(|| format!("no output {n}"))
}

impl Asr {
    fn sample(&self, logits: &Tensor) -> Result<i64, String> {
        let out = host("sample", Message::new(json!({}), vec![logits.clone()]))?;
        out.body["id"].as_i64().ok_or("sample gave no id".into())
    }

    /// `ids` as text: control tokens dropped, the rest's bytes joined.
    fn text(&mut self, ids: &[i64]) -> Result<String, String> {
        if self.vocab.is_none() {
            let file = host("file", Message::body(json!({"name": self.tekken})))?;
            let bytes = &file.tensors.first().ok_or("no tekken.json")?.data;
            let v: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
            let special = v["config"]["default_num_special_tokens"]
                .as_u64()
                .unwrap_or(1000) as usize;
            let b64 = base64::engine::general_purpose::STANDARD;
            let vocab = v["vocab"]
                .as_array()
                .ok_or("tekken.json has no vocab")?
                .iter()
                .map(|e| {
                    b64.decode(e["token_bytes"].as_str().unwrap_or(""))
                        .unwrap_or_default()
                })
                .collect();
            self.vocab = Some((special, vocab));
        }
        let (special, vocab) = self.vocab.as_ref().expect("read above");
        let mut out = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id < *special {
                continue;
            }
            out.extend(
                vocab
                    .get(id - special)
                    .ok_or(format!("token {id} past the vocabulary"))?,
            );
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

impl Program for Asr {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Asr {
            tekken: config["tekken"]
                .as_str()
                .ok_or("config has no tekken")?
                .to_string(),
            sample_rate: u("sample_rate")? as u64,
            hop: u("hop")?,
            n_fft: u("n_fft")?,
            n_mels: u("n_mels")?,
            window_frames: u("window_frames")?,
            rows_per_window: u("rows_per_window")?,
            layers: u("layers")?,
            rows: u("rows")?,
            max_tokens: u("max_tokens")?,
            audio: u("audio")? as i32,
            eos: u("eos")? as i64,
            languages: config["languages"].as_object().cloned().unwrap_or_default(),
            language: config["language"].as_str().unwrap_or("en").to_string(),
            vocab: None,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "transcribe" {
            return Err(format!("Voxtral answers transcribe, not {task}"));
        }
        let rate = req.body["sample_rate"].as_u64().unwrap_or(self.sample_rate);
        if rate != self.sample_rate {
            return Err(format!(
                "the audio is at {rate} Hz; this model takes {} Hz",
                self.sample_rate
            ));
        }
        let pcm = req
            .tensor("audio")
            .ok_or("transcribe takes `audio`")?
            .to_f32();
        if pcm.is_empty() {
            return Err("the audio is empty".into());
        }
        let o = &req.body["options"];

        // Whole windows of zeros past the speech, reflect-padded
        // (`center=True`), cut at the last frame's end.
        let per = self.window_frames * self.hop;
        let windows = pcm.len().div_ceil(per).max(1);
        let n = windows * per;
        let frames = windows * self.window_frames;
        let pad = self.n_fft / 2;
        let at = |j: usize| pcm.get(j).copied().unwrap_or(0.0);
        let len = self.hop * (frames + self.n_fft.div_ceil(self.hop) - 1);
        let signal: Vec<f32> = (0..len)
            .map(|i| {
                let j = i as isize - pad as isize;
                let j = if j < 0 { -j } else { j } as usize;
                let j = if j >= n { 2 * (n - 1) - j } else { j };
                at(j)
            })
            .collect();
        let outs = run_body(
            "mel",
            vec![Tensor::f32("signal", vec![len], &signal)],
            json!({}),
        )?;
        let mel = named(&outs, "mel")?.to_f32();
        // `[n_mels, frames]` → `[windows, n_mels, window_frames]`.
        let w = self.window_frames;
        let mut laid = Vec::with_capacity(mel.len());
        for c in 0..windows {
            for m in 0..self.n_mels {
                laid.extend_from_slice(&mel[m * frames + c * w..m * frames + (c + 1) * w]);
            }
        }
        run_body(
            "encoder",
            vec![Tensor::f32("mel", vec![windows, self.n_mels, w], &laid)],
            json!({"keep": {"hidden": "v.hidden"}}),
        )?;
        run_body(
            "projector",
            vec![],
            json!({"from": {"encoder_hidden": "v.hidden"}, "keep": {"audio": "v.audio"}}),
        )?;

        // The transcription prompt, its embedding, and the decoder over it.
        let tokens = windows * self.rows_per_window;
        let code = o["language"].as_str().unwrap_or(&self.language);
        let mut ids = vec![BOS, INST, BEGIN_AUDIO];
        let first = ids.len();
        ids.extend(std::iter::repeat_n(self.audio, tokens));
        ids.push(END_INST);
        if code != "auto" {
            let id = self
                .languages
                .get(code)
                .and_then(Value::as_i64)
                .ok_or(format!("Voxtral's prompt names no language {code}"))?;
            ids.extend([LANG, COLON, id as i32]);
        }
        ids.push(TRANSCRIBE);
        let t = ids.len();
        let max_tokens = o["max_tokens"]
            .as_u64()
            .map_or(self.max_tokens, |v| v as usize);
        // The caches' rows: a power of two past the prompt and transcript,
        // so lengths near each other share a compile.
        let cap = (t + max_tokens + 1).next_power_of_two().min(self.rows);
        if t >= cap {
            return Err(format!("the prompt's {t} rows reach past {cap} positions"));
        }
        run_body(
            "embed",
            vec![
                Tensor::i32("ids", vec![t], &ids),
                Tensor::f32("at", vec![1], &[first as f32]),
            ],
            json!({"from": {"audio": "v.audio"}, "keep": {"embeds": "v.embeds"}}),
        )?;
        let mut keep = Map::new();
        let mut from = Map::new();
        let mut step_keep = Map::new();
        for l in 0..self.layers {
            for s in ["k", "v"] {
                let slot = json!(format!("v.{s}{l}"));
                keep.insert(format!("past_{s}_{l}"), slot.clone());
                from.insert(format!("past_{s}_{l}"), slot.clone());
                step_keep.insert(format!("past_{s}_{l}.next"), slot);
            }
        }
        let outs = run_body(
            "prompt",
            vec![],
            json!({"from": {"inputs_embeds": "v.embeds"}, "keep": keep, "dims": {"cap": cap}}),
        )?;
        let mut next = self.sample(&named(&outs, "logits")?)?;
        let mut out = Vec::new();
        let mut position = t;
        // The step's caches hold `cap` rows, its last the slot the token's
        // own key and value are appended in (the decode's in-place spare).
        while next != self.eos && next != 0 && position + 1 < cap && out.len() < max_tokens {
            out.push(next);
            // The cached rows and the token's own (the last).
            let mut mask = vec![0f32; cap];
            mask[..position].fill(1.0);
            mask[cap - 1] = 1.0;
            let outs = run_body(
                "step",
                vec![
                    Tensor::f32("input_ids", vec![1, 1], &[next as f32]),
                    Tensor::f32("position", vec![1], &[position as f32]),
                    Tensor::f32("mask", vec![1, cap], &mask),
                ],
                json!({"from": from, "keep": step_keep}),
            )?;
            next = self.sample(&named(&outs, "logits")?)?;
            position += 1;
        }
        let mut slots = vec![
            "v.hidden".to_string(),
            "v.audio".to_string(),
            "v.embeds".to_string(),
        ];
        for l in 0..self.layers {
            slots.push(format!("v.k{l}"));
            slots.push(format!("v.v{l}"));
        }
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        let text = self.text(&out)?;
        Ok(Message::body(json!({"text": text.trim(), "ids": out})))
    }
}

rlx_program::program!(Asr);
