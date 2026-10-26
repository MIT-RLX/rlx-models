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

//! Qwen3-ASR's host logic as an RLX program, answering `transcribe`: what
//! `AsrRunner::transcribe_pcm` does — the signal reflect-padded by `n_fft /
//! 2` (`mel`), the mel cut into chunks of `chunk` frames, the last
//! zero-padded, each audio token's attention window (`encoder`, kept in a
//! slot); the chat prompt (`options.context` the system text) with a
//! placeholder per audio token (`embed` writes the audio's rows there),
//! `options.language` forcing the language as `transformers` does
//! (`language <Name><asr_text>` after the assistant's turn); then the
//! decoder over the prompt (`prompt`, its caches kept) and a greedy step a
//! token (`step`) until `<|im_end|>`. The text is what follows `<asr_text>`.

use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Map, Value, json};

/// The languages Qwen3-ASR names, by ISO code.
const LANGUAGES: [(&str, &str); 30] = [
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

struct Asr {
    tokenizer: String,
    sample_rate: u64,
    hop: usize,
    n_fft: usize,
    n_mels: usize,
    chunk: usize,
    window: usize,
    layers: usize,
    rows: usize,
    max_tokens: usize,
    im_start: i32,
    im_end: i32,
    audio_start: i32,
    audio_pad: i32,
    audio_end: i32,
    asr_text: i64,
    eos: Vec<i64>,
}

fn named(outs: &[Tensor], n: &str) -> Result<Tensor, String> {
    outs.iter()
        .find(|t| t.name == n)
        .cloned()
        .ok_or_else(|| format!("no output {n}"))
}

/// Output frames of one stride-2, padding-1, kernel-3 convolution.
fn conv_len(l: usize) -> usize {
    if l == 0 { 0 } else { (l - 1) / 2 + 1 }
}

fn downsample(l: usize) -> usize {
    conv_len(conv_len(conv_len(l)))
}

impl Asr {
    fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        let out = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer, "text": text, "special": false})),
        )?;
        Ok(out.body["ids"]
            .as_array()
            .ok_or("tokenizer.encode gave no ids")?
            .iter()
            .filter_map(Value::as_i64)
            .map(|v| v as i32)
            .collect())
    }

    /// `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n
    /// <|audio_start|>`, the placeholders, `<|audio_end|><|im_end|>\n
    /// <|im_start|>assistant\n` (then the forced language, if any); and the
    /// first placeholder's row.
    fn prompt(
        &self,
        context: &str,
        tokens: usize,
        language: Option<&str>,
    ) -> Result<(Vec<i32>, usize), String> {
        let mut ids = vec![self.im_start];
        ids.extend(self.encode(&format!("system\n{context}"))?);
        ids.push(self.im_end);
        ids.extend(self.encode("\n")?);
        ids.push(self.im_start);
        ids.extend(self.encode("user\n")?);
        ids.push(self.audio_start);
        let at = ids.len();
        ids.extend(std::iter::repeat_n(self.audio_pad, tokens));
        ids.push(self.audio_end);
        ids.push(self.im_end);
        ids.extend(self.encode("\n")?);
        ids.push(self.im_start);
        ids.extend(self.encode("assistant\n")?);
        if let Some(lang) = language {
            let name = LANGUAGES
                .iter()
                .find(|(c, n)| c.eq_ignore_ascii_case(lang) || n.eq_ignore_ascii_case(lang))
                .map(|(_, n)| *n)
                .ok_or(format!("Qwen3-ASR names no language {lang}"))?;
            ids.extend(self.encode(&format!("language {name}"))?);
            ids.push(self.asr_text as i32);
        }
        Ok((ids, at))
    }

    fn sample(&self, logits: &Tensor) -> Result<i64, String> {
        let out = host("sample", Message::new(json!({}), vec![logits.clone()]))?;
        out.body["id"].as_i64().ok_or("sample gave no id".into())
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
            tokenizer: config["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .to_string(),
            sample_rate: u("sample_rate")? as u64,
            hop: u("hop")?,
            n_fft: u("n_fft")?,
            n_mels: u("n_mels")?,
            chunk: u("chunk")?,
            window: u("window")?.max(1),
            layers: u("layers")?,
            rows: u("rows")?,
            max_tokens: u("max_tokens")?,
            im_start: u("im_start")? as i32,
            im_end: u("im_end")? as i32,
            audio_start: u("audio_start")? as i32,
            audio_pad: u("audio_pad")? as i32,
            audio_end: u("audio_end")? as i32,
            asr_text: u("asr_text")? as i64,
            eos: config["eos"]
                .as_array()
                .ok_or("config has no eos")?
                .iter()
                .filter_map(Value::as_i64)
                .collect(),
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "transcribe" {
            return Err(format!("Qwen3-ASR answers transcribe, not {task}"));
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
        let n = pcm.len();
        let frames = n / self.hop;
        if n <= self.n_fft / 2 || frames == 0 {
            return Err(format!("the audio is too short ({n} samples)"));
        }
        let o = &req.body["options"];

        // The signal reflect-padded (`np.pad(mode="reflect")`), cut at the
        // last frame's end.
        let pad = self.n_fft / 2;
        let len = self.hop * (frames + self.n_fft.div_ceil(self.hop) - 1);
        let signal: Vec<f32> = (0..len)
            .map(|i| {
                let j = i as isize - pad as isize;
                let j = if j < 0 { -j } else { j };
                let j = if j as usize >= n {
                    2 * (n as isize - 1) - j
                } else {
                    j
                };
                if j >= 0 && (j as usize) < n {
                    pcm[j as usize]
                } else {
                    0.0
                }
            })
            .collect();
        let outs = run_body(
            "mel",
            vec![Tensor::f32("signal", vec![len], &signal)],
            json!({}),
        )?;
        let mel = named(&outs, "mel")?.to_f32();

        // Chunks of `chunk` frames, the last zero-padded; each token's
        // window.
        let chunks = frames.div_ceil(self.chunk);
        let width = chunks * self.chunk;
        let mut laid = vec![0f32; self.n_mels * width];
        for m in 0..self.n_mels {
            laid[m * width..m * width + frames].copy_from_slice(&mel[m * frames..(m + 1) * frames]);
        }
        let tokens = (frames / self.chunk) * downsample(self.chunk)
            + match frames % self.chunk {
                0 => 0,
                r => downsample(r),
            };
        let window: Vec<f32> = (0..tokens).map(|i| (i / self.window) as f32).collect();
        run_body(
            "encoder",
            vec![
                Tensor::f32("mel", vec![1, self.n_mels, width], &laid),
                Tensor::f32("window", vec![tokens], &window),
            ],
            json!({"keep": {"audio": "q.audio"}}),
        )?;

        // The prompt, its embedding, and the decoder over it.
        let context = o["context"].as_str().unwrap_or("");
        let language = o["language"]
            .as_str()
            .filter(|l| !l.is_empty() && *l != "auto");
        let (ids, at) = self.prompt(context, tokens, language)?;
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
                Tensor::f32("at", vec![1], &[at as f32]),
            ],
            json!({"from": {"audio": "q.audio"}, "keep": {"embeds": "q.embeds"}}),
        )?;
        let mut keep = Map::new();
        let mut from = Map::new();
        let mut step_keep = Map::new();
        for l in 0..self.layers {
            for s in ["k", "v"] {
                let slot = json!(format!("q.{s}{l}"));
                keep.insert(format!("past_{s}_{l}"), slot.clone());
                from.insert(format!("past_{s}_{l}"), slot.clone());
                step_keep.insert(format!("past_{s}_{l}.next"), slot);
            }
        }
        let outs = run_body(
            "prompt",
            vec![],
            json!({"from": {"inputs_embeds": "q.embeds"}, "keep": keep, "dims": {"cap": cap}}),
        )?;
        let mut next = self.sample(&named(&outs, "logits")?)?;
        let mut out = Vec::new();
        let mut position = t;
        while !self.eos.contains(&next) && position + 1 < cap && out.len() < max_tokens {
            out.push(next);
            let outs = run_body(
                "step",
                vec![
                    Tensor::f32("input_ids", vec![1, 1], &[next as f32]),
                    Tensor::f32("position", vec![1], &[position as f32]),
                ],
                json!({"from": from, "keep": step_keep}),
            )?;
            next = self.sample(&named(&outs, "logits")?)?;
            position += 1;
        }
        let mut slots = vec!["q.audio".to_string(), "q.embeds".to_string()];
        for l in 0..self.layers {
            slots.push(format!("q.k{l}"));
            slots.push(format!("q.v{l}"));
        }
        host("slot.drop", Message::body(json!({ "slots": slots })))?;

        // The transcript: what follows `<asr_text>` (all of it, if none);
        // the language the model named before it.
        let split = out.iter().position(|&i| i == self.asr_text);
        let (lang_ids, text_ids) = match split {
            Some(p) => (&out[..p], &out[p + 1..]),
            None => (&out[..0], &out[..]),
        };
        let decode = |ids: &[i64]| -> Result<String, String> {
            let r = host(
                "tokenizer.decode",
                Message::body(json!({"tokenizer": self.tokenizer, "ids": ids})),
            )?;
            Ok(r.body["text"].as_str().unwrap_or("").to_string())
        };
        let text = decode(text_ids)?;
        let detected = decode(lang_ids)?;
        let language = language.map(str::to_string).unwrap_or_else(|| {
            detected
                .trim()
                .strip_prefix("language ")
                .unwrap_or(detected.trim())
                .to_string()
        });
        Ok(Message::body(
            json!({"text": text.trim(), "ids": out, "language": language}),
        ))
    }
}

rlx_program::program!(Asr);
