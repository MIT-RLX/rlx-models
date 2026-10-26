// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! FireRedAudio's understanding path as an RLX program, answering
//! `transcribe` and `run` (a question about the clip, the `prompt` option):
//! what `FireRedRunner::asr_pcm` / `understand_pcm` do — the signal
//! reflect-padded by `n_fft / 2` (`mel`), the mel laid in chunks of `chunk`
//! frames with zeros after it, the masks the reference's padding gives
//! (`encoder`), the chatml prompt (the crate's `prompt.rs`) with one
//! `<|AUDIO|>` per audio token, then the prompt and reply through the
//! package's `step` (`rlx-qwen35`'s `decode.rs`), each placeholder's
//! embedding the encoder's row. Options: `max_tokens`, `thinking` (`run`),
//! and the sampling ones (greedy unless given).

#[allow(dead_code)]
#[path = "../../../rlx-qwen35/program/src/decode.rs"]
mod decode;
#[allow(dead_code)]
#[path = "../../src/prompt.rs"]
mod prompt;

use rlx_program::{Message, Program, Tensor, run_body};
use serde_json::{Value, json};

struct FireRed {
    decoder: decode::Decoder,
    sample_rate: u64,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    chunk: usize,
    audio_token: String,
    audio_token_id: u32,
    eos: Vec<u32>,
    max_tokens: usize,
}

/// Output length of one kernel-3, stride-2, padding-1 convolution.
fn conv_len(l: usize) -> usize {
    (l.saturating_sub(1)) / 2 + 1
}

fn named<'a>(outs: &'a [Tensor], n: &str) -> Result<&'a Tensor, String> {
    outs.iter()
        .find(|t| t.name == n)
        .ok_or(format!("no output {n}"))
}

impl FireRed {
    /// The clip's audio tokens, `[tokens · hidden]`.
    fn encode(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize), String> {
        let n = pcm.len();
        let frames = n / self.hop;
        if n <= self.n_fft / 2 || frames == 0 {
            return Err(format!("the audio is too short ({n} samples)"));
        }
        // The signal reflect-padded (`center=True`), cut at the last
        // frame's end.
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

        // Whole chunks, zeros after the clip, and the masks.
        let chunks = frames.div_ceil(self.chunk);
        let width = chunks * self.chunk;
        let mut laid = vec![0f32; self.n_mels * width];
        for m in 0..self.n_mels {
            laid[m * width..m * width + frames].copy_from_slice(&mel[m * frames..(m + 1) * frames]);
        }
        let on = |k: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| f32::from(u8::from(i < k))).collect()
        };
        let t_pc = conv_len(self.chunk);
        let seq = chunks * t_pc;
        let mut window = vec![chunks as f32; seq];
        let mut after_cnn = 0;
        for c in 0..chunks {
            let this = (frames - c * self.chunk).min(self.chunk);
            let w = conv_len(this);
            for r in 0..w {
                window[c * t_pc + r] = c as f32;
            }
            after_cnn += w;
        }
        let s3 = conv_len(after_cnn);
        let tokens = conv_len(s3);
        let outs = run_body(
            "encoder",
            vec![
                Tensor::f32("mel", vec![1, self.n_mels, width], &laid),
                Tensor::f32("frames", vec![width], &on(frames, width)),
                Tensor::f32("window", vec![seq], &window),
                Tensor::f32("valid", vec![seq], &on(after_cnn, seq)),
                Tensor::f32("valid3", vec![conv_len(seq)], &on(s3, conv_len(seq))),
            ],
            json!({}),
        )?;
        let audio = named(&outs, "audio")?.to_f32();
        let h = self.decoder.hidden;
        if audio.len() < tokens * h {
            return Err(format!(
                "the encoder returned {} values for {tokens} tokens",
                audio.len()
            ));
        }
        Ok((audio[..tokens * h].to_vec(), tokens))
    }

    /// The reply to `chatml` (one placeholder) about the clip.
    fn answer(&self, pcm: &[f32], chatml: &str, o: &Value) -> anyhow::Result<(String, Vec<u32>)> {
        let (audio, tokens) = self.encode(pcm).map_err(anyhow::Error::msg)?;
        let text = chatml.replacen(&self.audio_token, &self.audio_token.repeat(tokens), 1);
        let ids = self.decoder.encode(&text)?;
        let h = self.decoder.hidden;
        let mut k = 0;
        let rows: Vec<Option<&[f32]>> = ids
            .iter()
            .map(|&t| {
                (t == self.audio_token_id).then(|| {
                    let r = &audio[k * h..(k + 1) * h];
                    k += 1;
                    r
                })
            })
            .collect();
        anyhow::ensure!(
            k == tokens,
            "the prompt has {k} audio placeholders for {tokens} audio tokens"
        );
        let positions: Vec<[usize; 4]> = (0..ids.len()).map(|i| [i, i, i, 0]).collect();
        let max_new = o["max_tokens"]
            .as_u64()
            .map_or(self.max_tokens, |v| v as usize);
        let out = self
            .decoder
            .generate(&ids, &rows, &positions, max_new, &self.eos, o)?;
        Ok((self.decoder.decode(&out)?, out))
    }
}

impl Program for FireRed {
    fn init(c: Value) -> Result<Self, String> {
        let n = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        Ok(FireRed {
            decoder: decode::Decoder::from_config(&c)?,
            sample_rate: n("sample_rate")? as u64,
            n_fft: n("n_fft")?,
            hop: n("hop")?,
            n_mels: n("n_mels")?,
            chunk: n("chunk")?,
            audio_token: c["audio_token"]
                .as_str()
                .ok_or("config has no audio_token")?
                .to_string(),
            audio_token_id: n("audio_token_id")? as u32,
            eos: serde_json::from_value(c["eos"].clone())
                .map_err(|e| format!("config eos: {e}"))?,
            max_tokens: n("max_tokens")?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
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
            .ok_or(format!("{task} takes `audio`"))?
            .to_f32();
        let chatml = match task {
            "transcribe" => prompt::build_asr_prompt(1, &self.audio_token),
            _ => {
                let question = o["prompt"]
                    .as_str()
                    .ok_or("run takes the `prompt` option")?;
                let thinking = o["thinking"].as_bool().unwrap_or(false);
                prompt::build_understand_prompt(question, 1, &self.audio_token, thinking)
            }
        }
        .map_err(|e| format!("{e:#}"))?;
        let (text, ids) = self
            .answer(&pcm, &chatml, o)
            .map_err(|e| format!("{e:#}"))?;
        let text = text.trim().to_string();
        match task {
            "transcribe" => Ok(Message::body(json!({"text": text, "ids": ids}))),
            _ => {
                let bytes = text.into_bytes();
                let n = bytes.len();
                Ok(Message::tensors(vec![Tensor::new(
                    "text",
                    "u8",
                    vec![n],
                    bytes,
                )]))
            }
        }
    }
}

rlx_program::program!(FireRed);
