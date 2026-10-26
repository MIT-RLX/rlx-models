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

//! Pocket TTS's host logic as an RLX program, answering `synthesize`: what
//! `PocketTts::generate` does — the text prepared and split into chunks
//! (the crate's `text.rs`, the host's tokenizer); each chunk's tokens, then
//! its frames, through `step` from the voice's key / value cache (kept in
//! slots, a multiple of 64 rows), each frame fed the last one's latent and
//! the runner's noise (`StdRng`, `Normal` at the temperature's square root),
//! until a few frames after the EOS head fires; each chunk's latents
//! de-normalized and decoded by `decode` in 8-frame blocks from a fresh
//! state, faded in over 5 ms. `run` answers the latents.

#[path = "../../src/text.rs"]
mod text;

/// What `text.rs` reads of a model config.
mod spec {
    use std::collections::BTreeMap;

    pub struct ModelSpec {
        pub pad_with_spaces_for_short_inputs: bool,
        pub remove_semicolons: bool,
        pub append_terminal_punctuation: bool,
        pub capitalize_first_letter: bool,
        pub replace_characters: BTreeMap<String, String>,
    }
}

/// The host's tokenizer, as `text.rs` calls the runner's.
mod tokenizer {
    use anyhow::{Result, anyhow};
    use rlx_program::{Message, host};
    use serde_json::json;

    pub struct PocketTokenizer {
        pub tokenizer: String,
    }

    impl PocketTokenizer {
        pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
            let m = host(
                "tokenizer.encode",
                Message::body(json!({"tokenizer": self.tokenizer, "text": text, "special": false})),
            )
            .map_err(|e| anyhow!(e))?;
            Ok(m.body["ids"]
                .as_array()
                .ok_or_else(|| anyhow!("no ids"))?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect())
        }

        pub fn decode(&self, ids: &[u32]) -> Result<String> {
            let m = host(
                "tokenizer.decode",
                Message::body(json!({"tokenizer": self.tokenizer, "ids": ids})),
            )
            .map_err(|e| anyhow!(e))?;
            Ok(m.body["text"].as_str().unwrap_or_default().to_string())
        }
    }
}

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rlx_program::{Message, Program, Tensor, host, run_with};
use serde_json::{Value, json};

use text::{TextOptions, prepare_text_prompt, split_into_chunks};
use tokenizer::PocketTokenizer;

struct Pocket {
    files: String,
    tok: PocketTokenizer,
    text: TextOptions,
    voices: serde_json::Map<String, Value>,
    voice: String,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    latent_dim: usize,
    embed_width: usize,
    embed: Vec<f32>,
    embed_proj: Option<Vec<f32>>,
    input_linear: Vec<f32>,
    bos: Vec<f32>,
    emb_mean: Vec<f32>,
    emb_std: Vec<f32>,
    kv_bucket: usize,
    min_frames_before_eos: usize,
    tokens_per_second: f32,
    gen_seconds_padding: f32,
    max_tokens_per_chunk: usize,
    temperature: f32,
    eos_threshold: f32,
    frames_after_eos: Option<usize>,
    seed: u64,
    frame_rate: f32,
    decoder_frames: usize,
    steps_per_frame: usize,
    frame_size: usize,
    mimi_context: usize,
    states: Vec<Vec<usize>>,
    sample_rate: u32,
}

/// `W x` for row-major `W [rows, cols]` (the runner's `matvec`).
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    Ok(std::fs::read(path)
        .map_err(|e| format!("read {path}: {e}"))?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn first(outs: Vec<Tensor>, what: &str) -> Result<Tensor, String> {
    outs.into_iter()
        .next()
        .ok_or_else(|| format!("{what} gave nothing"))
}

fn set_slot(slot: &str, t: Tensor) -> Result<(), String> {
    host("slot.set", Message::new(json!({ "slot": slot }), vec![t]))?;
    Ok(())
}

fn pairs(v: &[(String, String)]) -> Vec<(&str, &str)> {
    v.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

/// One chunk's output.
struct Chunk {
    latents: Vec<f32>,
    audio: Vec<f32>,
}

impl Pocket {
    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The text embedding of `token` (`[d_model]`).
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let w = self.embed_width;
        let row = &self.embed[token as usize * w..(token as usize + 1) * w];
        match &self.embed_proj {
            Some(p) => matvec(p, self.d_model, w, row),
            None => row.to_vec(),
        }
    }

    /// `input_linear` of a latent; `None` is the BOS position.
    fn embed_latent(&self, latent: Option<&[f32]>) -> Vec<f32> {
        matvec(
            &self.input_linear,
            self.d_model,
            self.latent_dim,
            latent.unwrap_or(&self.bos),
        )
    }

    /// The voice's caches in slots, `cap` rows each; its rows.
    fn load_voice(&self, voice: &str, cap: usize) -> Result<usize, String> {
        let offset = self.voices[voice]
            .as_u64()
            .ok_or_else(|| format!("no voice {voice:?}"))? as usize;
        let data = read_f32(&format!("{}/voices/{voice}.f32", self.files))?;
        let n = offset * self.kv_dim();
        for l in 0..self.layers {
            for (j, kind) in ["k", "v"].iter().enumerate() {
                let rows = &data[(2 * l + j) * n..(2 * l + j + 1) * n];
                let mut b = vec![0.0f32; cap * self.kv_dim()];
                b[..n].copy_from_slice(rows);
                set_slot(
                    &format!("pocket.past_{kind}{l}"),
                    Tensor::f32(
                        format!("past_{kind}{l}"),
                        vec![1, cap, self.kv_heads, self.head_dim],
                        &b,
                    ),
                )?;
            }
        }
        Ok(offset)
    }

    /// One backbone row at `len` of `cap`: its latent and EOS logit.
    fn step(
        &self,
        x: &[f32],
        noise: &[f32],
        len: usize,
        cap: usize,
    ) -> Result<(Vec<f32>, f32), String> {
        if len >= cap {
            return Err(format!("KV cache full ({cap} rows)"));
        }
        let mut mask = vec![0.0f32; cap + 1];
        mask[..len].fill(1.0);
        mask[cap] = 1.0;
        let names: Vec<(String, String)> = (0..self.layers)
            .flat_map(|l| [format!("past_k{l}"), format!("past_v{l}")])
            .map(|n| (n.clone(), format!("pocket.{n}")))
            .collect();
        let nexts: Vec<(String, String)> = names
            .iter()
            .map(|(n, s)| (format!("{n}.next"), s.clone()))
            .collect();
        let outs = run_with(
            "step",
            vec![
                Tensor::f32("x", vec![1, 1, self.d_model], x),
                Tensor::f32("noise", vec![1, self.latent_dim], noise),
                Tensor::f32("mask", vec![1, cap + 1], &mask),
                Tensor::f32("pos", vec![1], &[len as f32]),
            ],
            &pairs(&names),
            &pairs(&nexts),
        )?;
        let find = |n: &str| {
            outs.iter()
                .find(|t| t.name == n)
                .map(Tensor::to_f32)
                .ok_or_else(|| format!("step gave no {n}"))
        };
        Ok((find("latent")?, find("eos")?[0]))
    }

    /// `PocketTts::chunk`: one prepared chunk's latents and audio.
    fn chunk(
        &self,
        text: &str,
        voice: &str,
        frames_after_eos: usize,
        noise: &mut dyn FnMut() -> Vec<f32>,
    ) -> Result<Chunk, String> {
        let tokens = self.tok.encode(text).map_err(|e| e.to_string())?;
        if tokens.is_empty() {
            return Err(format!("chunk {text:?} has no tokens"));
        }
        let secs = tokens.len() as f32 / self.tokens_per_second + self.gen_seconds_padding;
        let max_gen = (secs * self.frame_rate).ceil() as usize;
        let offset = self.voices[voice]
            .as_u64()
            .ok_or_else(|| format!("no voice {voice:?}"))? as usize;
        let cap = (offset + tokens.len() + max_gen)
            .max(1)
            .div_ceil(self.kv_bucket)
            * self.kv_bucket;
        let mut len = self.load_voice(voice, cap)?;
        let zeros = vec![0.0; self.latent_dim];
        for &t in &tokens {
            self.step(&self.embed_token(t), &zeros, len, cap)?;
            len += 1;
        }
        let mut latents = Vec::new();
        let mut prev: Option<Vec<f32>> = None;
        let mut eos_step: Option<usize> = None;
        for step in 0..max_gen {
            let x = self.embed_latent(prev.as_deref());
            let z = noise();
            let (latent, eos) = self.step(&x, &z, len, cap)?;
            len += 1;
            if eos > self.eos_threshold && eos_step.is_none() && step >= self.min_frames_before_eos
            {
                eos_step = Some(step);
            }
            if eos_step.is_some_and(|e| step >= e + frames_after_eos) {
                break;
            }
            latents.extend_from_slice(&latent);
            prev = Some(latent);
        }
        let mut audio = self.decode(&latents)?;
        let fade = self.sample_rate as usize / 200;
        let n = fade.min(audio.len());
        for (i, s) in audio[..n].iter_mut().enumerate() {
            *s *= i as f32 / (fade - 1) as f32;
        }
        Ok(Chunk { latents, audio })
    }

    /// `decode_latents`: de-normalized, decoded from a fresh state in
    /// blocks of `decoder_frames`.
    fn decode(&self, latents: &[f32]) -> Result<Vec<f32>, String> {
        let ldim = self.latent_dim;
        let denorm: Vec<f32> = latents
            .chunks(ldim)
            .flat_map(|f| {
                f.iter()
                    .zip(&self.emb_std)
                    .zip(&self.emb_mean)
                    .map(|((x, s), m)| x * s + m)
            })
            .collect();
        if denorm.is_empty() {
            return Ok(Vec::new());
        }
        let names: Vec<(String, String)> = (0..self.states.len())
            .map(|i| (format!("state{i}"), format!("pocket.state{i}")))
            .collect();
        let nexts: Vec<(String, String)> = names
            .iter()
            .map(|(n, s)| (format!("{n}.next"), s.clone()))
            .collect();
        for ((n, s), shape) in names.iter().zip(&self.states) {
            let len = shape.iter().product();
            set_slot(s, Tensor::f32(n.clone(), shape.clone(), &vec![0.0; len]))?;
        }
        let f = self.decoder_frames;
        let t = f * self.steps_per_frame;
        let p = self.mimi_context - 1;
        let mut position = 0usize;
        let mut out = Vec::new();
        for block in denorm.chunks(f * ldim) {
            let n = block.len() / ldim;
            let mut x = vec![0.0f32; ldim * f];
            for i in 0..n {
                for c in 0..ldim {
                    x[c * f + i] = block[i * ldim + c];
                }
            }
            let p0 = position;
            let positions: Vec<f32> = (p0..p0 + t).map(|i| i as f32).collect();
            let mut mask = vec![f32::NEG_INFINITY; t * (p + t)];
            for qi in 0..t {
                let qpos = (p0 + qi) as i64;
                for s in 0..p + t {
                    let kpos = p0 as i64 - p as i64 + s as i64;
                    if kpos >= 0 && kpos <= qpos && qpos - kpos < self.mimi_context as i64 {
                        mask[qi * (p + t) + s] = 0.0;
                    }
                }
            }
            let mut audio = first(
                run_with(
                    "decode",
                    vec![
                        Tensor::f32("x", vec![1, ldim, f, 1], &x),
                        Tensor::f32("attn_mask", vec![1, t, p + t], &mask),
                        Tensor::f32("positions", vec![t], &positions),
                    ],
                    &pairs(&names),
                    &pairs(&nexts),
                )?,
                "decode",
            )?
            .to_f32();
            audio.truncate(n * self.frame_size);
            out.extend(audio);
            position += t;
        }
        let slots: Vec<&str> = names.iter().map(|(_, s)| s.as_str()).collect();
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        Ok(out)
    }

    /// `PocketTts::generate`: every chunk, the noise one stream.
    fn generate(&self, text: &str, o: &Value) -> Result<Vec<Chunk>, String> {
        let voice = o["voice"].as_str().unwrap_or(&self.voice).to_string();
        let temperature = o["temperature"]
            .as_f64()
            .map_or(self.temperature, |v| v as f32);
        let seed = o["seed"].as_u64().unwrap_or(self.seed);
        let fae_opt = o["frames_after_eos"]
            .as_u64()
            .map(|v| v as usize)
            .or(self.frames_after_eos);
        let chunks = split_into_chunks(&self.tok, text, self.max_tokens_per_chunk, &self.text)
            .map_err(|e| e.to_string())?;
        let mut rng = StdRng::seed_from_u64(seed);
        let std = temperature.max(0.0).sqrt();
        let normal = Normal::new(0.0f32, std.max(f32::MIN_POSITIVE)).map_err(|e| e.to_string())?;
        let ldim = self.latent_dim;
        let mut noise = move || -> Vec<f32> {
            (0..ldim)
                .map(|_| {
                    if std == 0.0 {
                        return 0.0;
                    }
                    normal.sample(&mut rng)
                })
                .collect()
        };
        let mut out = Vec::new();
        for chunk in chunks {
            let (prepared, guess) =
                prepare_text_prompt(&chunk, &self.text).map_err(|e| e.to_string())?;
            let fae = fae_opt.unwrap_or(guess + 2);
            out.push(self.chunk(&prepared, &voice, fae, &mut noise)?);
        }
        Ok(out)
    }
}

impl Program for Pocket {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        let floats = |k: &str| -> Result<Vec<f32>, String> {
            Ok(config[k]
                .as_array()
                .ok_or(format!("config has no {k}"))?
                .iter()
                .filter_map(|v| v.as_f64().map(|v| v as f32))
                .collect())
        };
        let t = &config["text"];
        let replace = t["replace_characters"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let spec = spec::ModelSpec {
            pad_with_spaces_for_short_inputs: t["pad_with_spaces_for_short_inputs"]
                .as_bool()
                .unwrap_or(false),
            remove_semicolons: t["remove_semicolons"].as_bool().unwrap_or(false),
            append_terminal_punctuation: t["append_terminal_punctuation"]
                .as_bool()
                .unwrap_or(false),
            capitalize_first_letter: t["capitalize_first_letter"].as_bool().unwrap_or(false),
            replace_characters: replace,
        };
        let files = config["files"].as_str().unwrap_or("/package").to_string();
        let embed_proj = if config["embed_proj"].as_bool().unwrap_or(false) {
            Some(read_f32(&format!("{files}/embed_proj.f32"))?)
        } else {
            None
        };
        Ok(Pocket {
            tok: PocketTokenizer {
                tokenizer: config["tokenizer"]
                    .as_str()
                    .ok_or("config has no tokenizer")?
                    .into(),
            },
            text: TextOptions::from_spec(&spec),
            voices: config["voices"].as_object().cloned().unwrap_or_default(),
            voice: config["voice"].as_str().unwrap_or_default().to_string(),
            layers: u("layers")? as usize,
            kv_heads: u("kv_heads")? as usize,
            head_dim: u("head_dim")? as usize,
            d_model: u("d_model")? as usize,
            latent_dim: u("latent_dim")? as usize,
            embed_width: u("embed_width")? as usize,
            embed: read_f32(&format!("{files}/embed.f32"))?,
            embed_proj,
            input_linear: read_f32(&format!("{files}/input_linear.f32"))?,
            bos: floats("bos")?,
            emb_mean: floats("emb_mean")?,
            emb_std: floats("emb_std")?,
            kv_bucket: u("kv_bucket")? as usize,
            min_frames_before_eos: u("min_frames_before_eos")? as usize,
            tokens_per_second: f("tokens_per_second")?,
            gen_seconds_padding: f("gen_seconds_padding")?,
            max_tokens_per_chunk: config["max_tokens_per_chunk"]
                .as_u64()
                .map_or(text::MAX_TOKENS_PER_CHUNK, |v| v as usize),
            temperature: f("temperature")?,
            eos_threshold: f("eos_threshold")?,
            frames_after_eos: config["frames_after_eos"].as_u64().map(|v| v as usize),
            seed: u("seed")?,
            frame_rate: f("frame_rate")?,
            decoder_frames: u("decoder_frames")? as usize,
            steps_per_frame: u("steps_per_frame")? as usize,
            frame_size: u("frame_size")? as usize,
            mimi_context: u("mimi_context")? as usize,
            states: config["states"]
                .as_array()
                .ok_or("config has no states")?
                .iter()
                .map(|s| {
                    s.as_array()
                        .map(|d| {
                            d.iter()
                                .filter_map(|x| x.as_u64().map(|x| x as usize))
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect(),
            sample_rate: u("sample_rate")? as u32,
            files,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().ok_or("run takes options.text")?,
            _ => return Err(format!("Pocket TTS answers synthesize and run, not {task}")),
        };
        let chunks = self.generate(text, o)?;
        if task == "run" {
            let latents: Vec<f32> = chunks.iter().flat_map(|c| c.latents.clone()).collect();
            let n = latents.len() / self.latent_dim;
            return Ok(Message::tensors(vec![Tensor::f32(
                "latents",
                vec![n, self.latent_dim],
                &latents,
            )]));
        }
        let wav: Vec<f32> = chunks.into_iter().flat_map(|c| c.audio).collect();
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Pocket);
