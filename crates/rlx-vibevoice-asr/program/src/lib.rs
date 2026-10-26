// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! VibeVoice-ASR-Streaming as an RLX program, answering `transcribe`: what
//! the official `streaming_generate` does with its defaults — the audio cut
//! into segments of `segment` samples every `chunk` (the last zero-padded),
//! each segment's features (`acoustic` + its connector plus `semantic` + its
//! connector), the prompt stepped a token at a time, then per segment
//! `<|speech_start|>`, the features and `<|speech_end|>`, greedy tokens
//! until a stop id, `<|text_chunk_end|>` fed after them; each chunk's text
//! decoded and cleaned of the special tokens. The transcript is the chunks'
//! texts, trimmed and joined by spaces (the runner's), the chunks too
//! (`chunks`). Options: `max_tokens` (per chunk), `context` (hotwords).

use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Map, Value, json};

struct Stream {
    tokenizer: String,
    hidden: usize,
    layers: usize,
    kv: usize,
    rows: usize,
    sample_rate: u64,
    chunk: usize,
    segment: usize,
    frames: usize,
    prompt: String,
    prompt_prefix: String,
    speech_start: i32,
    speech_end: i32,
    chunk_end: i32,
    stop: Vec<i32>,
    max_tokens: usize,
}

fn named(outs: &[Tensor], n: &str) -> Result<Vec<f32>, String> {
    Ok(outs
        .iter()
        .find(|t| t.name == n)
        .ok_or(format!("no output {n}"))?
        .to_f32())
}

fn argmax(v: &[f32]) -> i32 {
    let (mut best, mut at) = (f32::NEG_INFINITY, 0);
    for (i, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            at = i;
        }
    }
    at as i32
}

/// The decoder's caches and where the next row goes.
struct Decoder<'a> {
    s: &'a Stream,
    from: Map<String, Value>,
    keep: Map<String, Value>,
    position: usize,
    cap: usize,
}

impl Decoder<'_> {
    /// One row through the step; its logits.
    fn row(&mut self, row: &[f32]) -> Result<Vec<f32>, String> {
        if self.position + 1 >= self.cap {
            return Err(format!("past the caches' {} rows", self.cap));
        }
        let outs = run_body(
            "step",
            vec![
                Tensor::f32("inputs_embeds", vec![1, 1, self.s.hidden], row),
                Tensor::f32("position", vec![1], &[self.position as f32]),
            ],
            json!({"from": self.from, "keep": self.keep}),
        )?;
        self.position += 1;
        named(&outs, "logits")
    }

    /// A token's row through the step.
    fn token(&mut self, id: i32) -> Result<Vec<f32>, String> {
        let outs = run_body(
            "embed",
            vec![Tensor::i32("token", vec![1], &[id])],
            json!({}),
        )?;
        let row = named(&outs, "row")?;
        self.row(&row)
    }
}

impl Stream {
    fn decode(&self, ids: &[i32]) -> Result<String, String> {
        let r = host(
            "tokenizer.decode",
            Message::body(json!({"tokenizer": self.tokenizer, "ids": ids})),
        )?;
        let mut text = r.body["text"].as_str().unwrap_or("").to_string();
        for st in [
            "<|text_chunk_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|speech_start|>",
            "<|speech_end|>",
            "<|speech_pad|>",
        ] {
            text = text.replace(st, "");
        }
        Ok(text)
    }

    /// One segment's features, `[frames · hidden]`.
    fn features(&self, seg: &[f32]) -> Result<Vec<f32>, String> {
        let mut sum = vec![0f32; self.frames * self.hidden];
        for name in ["acoustic", "semantic"] {
            let outs = run_body(
                name,
                vec![Tensor::f32("audio", vec![1, self.segment], seg)],
                json!({}),
            )?;
            let latent = &outs.first().ok_or("no latent")?;
            let outs = run_body(
                &format!("{name}_connector"),
                vec![Tensor::f32(
                    "latent",
                    latent.shape.clone(),
                    &latent.to_f32(),
                )],
                json!({}),
            )?;
            let f = named(&outs, "features")?;
            if f.len() < sum.len() {
                return Err(format!("{name} gave {} values", f.len()));
            }
            for (s, v) in sum.iter_mut().zip(&f) {
                *s += v;
            }
        }
        Ok(sum)
    }

    fn transcribe(&self, pcm: &[f32], o: &Value) -> Result<(String, Vec<String>), String> {
        if pcm.is_empty() {
            return Err("empty audio".into());
        }
        // `split_then_encode`: a segment every chunk, the last zero-padded.
        let mut segs: Vec<Vec<f32>> = Vec::new();
        let mut start = 0;
        while start < pcm.len() {
            let end = (start + self.segment).min(pcm.len());
            let mut seg = pcm[start..end].to_vec();
            seg.resize(self.segment, 0.0);
            segs.push(seg);
            start = (start + self.chunk).min(pcm.len());
        }
        let prompt = match o["context"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(info) => format!("{} and extra info: {info}\n", self.prompt_prefix),
            None => self.prompt.clone(),
        };
        let r = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer, "text": prompt, "special": false})),
        )?;
        let ids: Vec<i32> = r.body["ids"]
            .as_array()
            .ok_or("tokenizer.encode returned no ids")?
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as i32)
            .collect();
        let max_tokens = o["max_tokens"]
            .as_u64()
            .map_or(self.max_tokens, |v| v as usize);

        // The caches: a power of two past every row there may be.
        let need = ids.len() + segs.len() * (self.frames + 3 + max_tokens) + 1;
        let cap = need.next_power_of_two().min(self.rows);
        let zeros = vec![0f32; cap * self.kv];
        let (mut from, mut keep) = (Map::new(), Map::new());
        for l in 0..self.layers {
            for s in ["k", "v"] {
                let slot = format!("v.{s}{l}");
                host(
                    "slot.set",
                    Message::new(
                        json!({"slot": slot}),
                        vec![Tensor::f32(
                            format!("past_{s}_{l}"),
                            vec![1, cap, self.kv],
                            &zeros,
                        )],
                    ),
                )?;
                from.insert(format!("past_{s}_{l}"), json!(slot));
                keep.insert(format!("past_{s}_{l}.next"), json!(slot));
            }
        }
        let mut dec = Decoder {
            s: self,
            from,
            keep,
            position: 0,
            cap,
        };
        for &id in &ids {
            dec.token(id)?;
        }
        let mut texts = Vec::new();
        for seg in &segs {
            let feats = self.features(seg)?;
            dec.token(self.speech_start)?;
            for r in 0..self.frames {
                dec.row(&feats[r * self.hidden..(r + 1) * self.hidden])?;
            }
            let mut logits = dec.token(self.speech_end)?;
            let mut out = Vec::new();
            for _ in 0..max_tokens {
                let next = argmax(&logits);
                if self.stop.contains(&next) {
                    break;
                }
                out.push(next);
                logits = dec.token(next)?;
            }
            dec.token(self.chunk_end)?;
            texts.push(self.decode(&out)?);
        }
        let mut slots = Vec::new();
        for l in 0..self.layers {
            slots.push(format!("v.k{l}"));
            slots.push(format!("v.v{l}"));
        }
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        let text = texts
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Ok((text, texts))
    }
}

impl Program for Stream {
    fn init(c: Value) -> Result<Self, String> {
        let n = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let s = |k: &str| -> Result<String, String> {
            Ok(c[k]
                .as_str()
                .ok_or(format!("config has no {k}"))?
                .to_string())
        };
        Ok(Stream {
            tokenizer: s("tokenizer")?,
            hidden: n("hidden")?,
            layers: n("layers")?,
            kv: n("kv")?,
            rows: n("rows")?,
            sample_rate: n("sample_rate")? as u64,
            chunk: n("chunk")?,
            segment: n("segment")?,
            frames: n("frames")?,
            prompt: s("prompt")?,
            prompt_prefix: s("prompt_prefix")?,
            speech_start: n("speech_start")? as i32,
            speech_end: n("speech_end")? as i32,
            chunk_end: n("chunk_end")? as i32,
            stop: serde_json::from_value(c["stop"].clone())
                .map_err(|e| format!("config stop: {e}"))?,
            max_tokens: n("max_tokens")?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "transcribe" {
            return Err(format!("VibeVoice-ASR answers transcribe, not {task}"));
        }
        let o = &req.body["options"];
        let rate = req.body["sample_rate"]
            .as_u64()
            .or_else(|| o["sample_rate"].as_u64())
            .unwrap_or(self.sample_rate);
        if rate != self.sample_rate {
            return Err(format!(
                "the audio is at {rate} Hz; this model takes {} Hz",
                self.sample_rate
            ));
        }
        let pcm = req
            .tensor("audio")
            .or_else(|| req.tensors.first())
            .ok_or("transcribe takes `audio`")?
            .to_f32();
        let (text, chunks) = self.transcribe(&pcm, o)?;
        Ok(Message::body(json!({"text": text, "chunks": chunks})))
    }
}

rlx_program::program!(Stream);
