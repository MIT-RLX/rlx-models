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

//! Kyutai TTS's host logic as an RLX program, answering `synthesize` (and
//! `run`, the frames' codes): the crate's own `generate.rs` (the
//! delay-multiplexed stream, the state machine feeding the script's words,
//! the end padding, the frames trimmed for Mimi), `state_machine.rs`,
//! `delays.rs` and `sampling.rs` (the runner's seeded `StdRng`, top-k and
//! Gumbel draws) — over a [`model::KyutaiLm`] whose steps are the
//! package's graphs: the voice's cross-attention context once (`speaker`),
//! then each step's embedding (`embed`), temporal row (`step`, its caches
//! kept in slots) and the 32 DepFormer codebooks (`dep_{k}`); then Mimi
//! (`mimi_lookup`, `mimi_decode`) and the peak normalized to 0.95.
//! Options: `cfg_alpha`, `max_steps`, `seed`, `text_temperature`,
//! `audio_temperature`.

#![allow(dead_code)]

#[path = "../../src/delays.rs"]
mod delays;
#[path = "../../src/generate.rs"]
mod generate;
#[path = "../../src/sampling.rs"]
mod sampling;
#[path = "../../src/state_machine.rs"]
mod state_machine;

use ndarray::{Array1, Array2};
use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Map, Value, json};

/// What `generate.rs`, `delays.rs` and `state_machine.rs` read of the
/// model's config.
mod config {
    pub struct TtsConfig {
        pub second_stream_ahead: usize,
    }

    pub struct KyutaiTtsConfig {
        pub text_card: usize,
        pub card: usize,
        pub dep_q: usize,
        pub delays: Vec<i32>,
        pub audio_delay: f32,
        pub tts_config: TtsConfig,
    }

    impl KyutaiTtsConfig {
        pub fn audio_delay_frames(&self) -> usize {
            (self.audio_delay * 12.5).round() as usize
        }

        pub fn audio_pad_token(&self) -> u32 {
            self.card as u32
        }
    }
}

mod session {
    pub struct GenerationConfig {
        pub max_steps: usize,
        pub text_temperature: f64,
        pub audio_temperature: f64,
        pub cfg_alpha: f32,
        pub seed: u64,
        pub mimi_codebooks: usize,
    }
}

/// The script's words through the package's tokenizer.
mod tokenizer {
    use rlx_program::{Message, host};
    use serde_json::json;

    pub struct KyutaiTokenizer {
        pub name: String,
    }

    impl KyutaiTokenizer {
        pub fn encode_prompt_words(&self, prompt: &str) -> anyhow::Result<Vec<Vec<u32>>> {
            let mut out = Vec::new();
            for word in prompt.split_whitespace() {
                let r = host(
                    "tokenizer.encode",
                    Message::body(json!({"tokenizer": self.name, "text": word, "special": false})),
                )
                .map_err(anyhow::Error::msg)?;
                out.push(
                    r.body["ids"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64())
                                .map(|v| v as u32)
                                .collect()
                        })
                        .unwrap_or_default(),
                );
            }
            Ok(out)
        }
    }
}

/// The crate's `KyutaiLm`, as `generate.rs` calls it.
mod model {
    use crate::config::KyutaiTtsConfig;
    use crate::sampling::StreamSampler;
    use anyhow::Result;
    use ndarray::{Array1, Array2};

    pub trait KyutaiLm {
        fn config(&self) -> &KyutaiTtsConfig;
        fn reset_state(&mut self);
        fn set_generation_conditions(
            &mut self,
            cfg_alpha: f32,
            speaker: Option<&Array2<f32>>,
        ) -> Result<()>;
        fn forward_step(
            &mut self,
            text_token: u32,
            audio_delayed: &[u32],
            sampler: &mut StreamSampler,
        ) -> Result<(u32, Array1<f32>)>;
        fn depformer_step(
            &mut self,
            hidden: &Array1<f32>,
            depformer_text: u32,
            sampler: &mut StreamSampler,
        ) -> Result<Vec<u32>>;
        fn audio_pad_token(&self) -> u32;
        fn lm_zero_token(&self) -> u32;
    }
}

fn err(e: String) -> anyhow::Error {
    anyhow::Error::msg(e)
}

fn first(outs: &[Tensor], n: &str) -> anyhow::Result<Vec<f32>> {
    Ok(outs
        .iter()
        .find(|t| t.name == n)
        .ok_or_else(|| anyhow::anyhow!("no output {n}"))?
        .to_f32())
}

/// The package's graphs as a `KyutaiLm`.
struct PackageLm {
    cfg: config::KyutaiTtsConfig,
    cfg_values: Vec<String>,
    cfg_bin: i32,
    layers: usize,
    dep_layers: usize,
    rope_theta: f32,
    half: usize,
    cap: usize,
    position: usize,
}

impl PackageLm {
    /// A text token's two `card + 1`-strided ids and whether each is there.
    fn demux(token: u32, card: u32) -> ([i32; 2], [f32; 2]) {
        if token == u32::MAX {
            return ([0, 0], [0.0, 0.0]);
        }
        let main = token % card;
        let second = token / card;
        if second > 0 && second - 1 < card {
            ([main as i32, (second - 1) as i32], [1.0, 1.0])
        } else {
            ([main as i32, 0], [1.0, 0.0])
        }
    }
}

impl model::KyutaiLm for PackageLm {
    fn config(&self) -> &config::KyutaiTtsConfig {
        &self.cfg
    }

    fn reset_state(&mut self) {
        self.position = 0;
    }

    fn set_generation_conditions(
        &mut self,
        cfg_alpha: f32,
        _speaker: Option<&Array2<f32>>,
    ) -> anyhow::Result<()> {
        let key = format!("{cfg_alpha:.1}");
        self.cfg_bin = self
            .cfg_values
            .iter()
            .position(|v| *v == key)
            .ok_or_else(|| anyhow::anyhow!("no CFG bin {key}"))? as i32;
        // The default voice's cross-attention context.
        run_body(
            "speaker",
            vec![],
            json!({"weights": {"speaker_wavs": "voice.default"}, "keep": {"cross_ctx": "k.ctx"}}),
        )
        .map_err(err)?;
        // Zero caches.
        let d = self.half * 2;
        let nh = 16;
        let zeros = vec![0f32; self.cap * nh * d];
        for l in 0..self.layers {
            for s in ["k", "v"] {
                host(
                    "slot.set",
                    Message::new(
                        json!({"slot": format!("k.{s}{l}")}),
                        vec![Tensor::f32(
                            format!("past_{s}_{l}"),
                            vec![1, self.cap, nh, d],
                            &zeros,
                        )],
                    ),
                )
                .map_err(err)?;
            }
        }
        Ok(())
    }

    fn forward_step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut sampling::StreamSampler,
    ) -> anyhow::Result<(u32, Array1<f32>)> {
        let (text, text_on) = Self::demux(text_token, self.cfg.text_card as u32 + 1);
        let audio: Vec<i32> = audio_delayed
            .iter()
            .map(|&t| if t == u32::MAX { 0 } else { t as i32 })
            .collect();
        let audio_on: Vec<f32> = audio_delayed
            .iter()
            .map(|&t| if t == u32::MAX { 0.0 } else { 1.0 })
            .collect();
        let n = audio.len();
        run_body(
            "embed",
            vec![
                Tensor::i32("text", vec![2], &text),
                Tensor::f32("text_on", vec![2], &text_on),
                Tensor::i32("audio", vec![n], &audio),
                Tensor::f32("audio_on", vec![n], &audio_on),
                Tensor::i32("cfg", vec![1], &[self.cfg_bin]),
            ],
            json!({"keep": {"inputs_embeds": "k.x"}}),
        )
        .map_err(err)?;
        let p = self.position;
        if p >= self.cap {
            anyhow::bail!("the temporal caches hold {} rows", self.cap);
        }
        let (mut cos, mut sin) = (vec![0f32; self.half], vec![0f32; self.half]);
        for i in 0..self.half {
            let inv = 1.0f32 / self.rope_theta.powf(i as f32 / self.half as f32);
            let f = p as f32 * inv;
            cos[i] = f.cos();
            sin[i] = f.sin();
        }
        let mut mask = vec![0f32; self.cap + 1];
        mask[..p].fill(1.0);
        mask[self.cap] = 1.0;
        let t_cross = 625;
        let mut from = Map::new();
        let mut keep = Map::new();
        from.insert("inputs_embeds".into(), json!("k.x"));
        from.insert("cross_ctx".into(), json!("k.ctx"));
        for l in 0..self.layers {
            for s in ["k", "v"] {
                from.insert(format!("past_{s}_{l}"), json!(format!("k.{s}{l}")));
                keep.insert(format!("past_{s}_{l}.next"), json!(format!("k.{s}{l}")));
            }
        }
        let outs = run_body(
            "step",
            vec![
                Tensor::f32("position", vec![1], &[p as f32]),
                Tensor::f32("rope_cos", vec![1, self.half], &cos),
                Tensor::f32("rope_sin", vec![1, self.half], &sin),
                Tensor::f32("mask", vec![1, self.cap + 1], &mask),
                Tensor::f32("cross_mask", vec![1, t_cross], &vec![1f32; t_cross]),
            ],
            json!({"from": from, "keep": keep}),
        )
        .map_err(err)?;
        self.position += 1;
        let logits = first(&outs, "logits")?;
        let hidden = first(&outs, "hidden")?;
        let text = sampler.sample_text(&Array1::from_vec(logits));
        Ok((text, Array1::from_vec(hidden)))
    }

    fn depformer_step(
        &mut self,
        hidden: &Array1<f32>,
        depformer_text: u32,
        sampler: &mut sampling::StreamSampler,
    ) -> anyhow::Result<Vec<u32>> {
        let h = hidden.to_vec();
        let d = h.len();
        let mut out = Vec::with_capacity(self.cfg.dep_q);
        let mut prev = depformer_text;
        for k in 0..self.cfg.dep_q {
            let mut inputs = vec![Tensor::f32("hidden", vec![1, 1, d], &h)];
            if k == 0 {
                let (text, on) = Self::demux(prev, self.cfg.text_card as u32 + 1);
                inputs.push(Tensor::i32("text", vec![2], &text));
                inputs.push(Tensor::f32("text_on", vec![2], &on));
            } else {
                inputs.push(Tensor::i32("token", vec![1], &[prev as i32]));
            }
            let mut from = Map::new();
            let mut keep = Map::new();
            for l in 0..self.dep_layers {
                for s in ["k", "v"] {
                    if k > 0 {
                        from.insert(format!("{s}_{l}"), json!(format!("k.d{s}{l}")));
                    }
                    keep.insert(format!("{s}_{l}.next"), json!(format!("k.d{s}{l}")));
                }
            }
            let outs = run_body(
                &format!("dep_{k}"),
                inputs,
                json!({"from": from, "keep": keep}),
            )
            .map_err(err)?;
            let tok = sampler.sample_audio(&Array1::from_vec(first(&outs, "logits")?));
            out.push(tok);
            prev = tok;
        }
        Ok(out)
    }

    fn audio_pad_token(&self) -> u32 {
        self.cfg.audio_pad_token()
    }

    fn lm_zero_token(&self) -> u32 {
        u32::MAX
    }
}

struct Kyutai {
    tokenizer: String,
    cfg_values: Vec<String>,
    cfg_alpha: f32,
    max_steps: usize,
    text_temperature: f64,
    audio_temperature: f64,
    seed: u64,
    rope_theta: f32,
    sample_rate: u32,
}

impl Kyutai {
    fn speak(&self, text: &str, o: &Value) -> anyhow::Result<(Vec<Vec<u32>>, Vec<f32>)> {
        let max_steps = o["max_steps"]
            .as_u64()
            .map_or(self.max_steps, |v| v as usize);
        let gen_cfg = session::GenerationConfig {
            max_steps,
            text_temperature: o["text_temperature"]
                .as_f64()
                .unwrap_or(self.text_temperature),
            audio_temperature: o["audio_temperature"]
                .as_f64()
                .unwrap_or(self.audio_temperature),
            cfg_alpha: o["cfg_alpha"].as_f64().map_or(self.cfg_alpha, |v| v as f32),
            seed: o["seed"].as_u64().unwrap_or(self.seed),
            mimi_codebooks: 32,
        };
        let mut delays = vec![0, 0];
        delays.extend(std::iter::repeat_n(2, 31));
        let cfg = config::KyutaiTtsConfig {
            text_card: 8000,
            card: 2048,
            dep_q: 32,
            delays,
            audio_delay: 1.28,
            tts_config: config::TtsConfig {
                second_stream_ahead: 2,
            },
        };
        let gc = generate::GenerateConfig::from_session(&gen_cfg, &cfg);
        // The caches' rows: a power of two past every step.
        let cap = (max_steps + 16).next_power_of_two();
        let mut lm = PackageLm {
            cfg,
            cfg_values: self.cfg_values.clone(),
            cfg_bin: 0,
            layers: 16,
            dep_layers: 4,
            rope_theta: self.rope_theta,
            half: 64,
            cap,
            position: 0,
        };
        let tok = tokenizer::KyutaiTokenizer {
            name: self.tokenizer.clone(),
        };
        let (frames, _, _) = generate::generate_codes(&mut lm, &tok, text, gc, None)?;
        if frames.is_empty() {
            anyhow::bail!("generation produced no audio frames (try a larger max_steps)");
        }
        // Mimi: codes `[codebooks, frames]`.
        let (books, t) = (frames[0].len(), frames.len());
        let mut codes = vec![0i32; books * t];
        for (f, frame) in frames.iter().enumerate() {
            for (q, &c) in frame.iter().enumerate() {
                codes[q * t + f] = c as i32;
            }
        }
        run_body(
            "mimi_lookup",
            vec![Tensor::i32("codes", vec![books, t], &codes)],
            json!({"keep": {"z": "k.z"}}),
        )
        .map_err(err)?;
        let outs = run_body("mimi_decode", vec![], json!({"from": {"z": "k.z"}})).map_err(err)?;
        let mut audio = outs
            .last()
            .ok_or_else(|| anyhow::anyhow!("no audio"))?
            .to_f32();
        let mut slots: Vec<String> = vec!["k.x".into(), "k.ctx".into(), "k.z".into()];
        for l in 0..16 {
            slots.push(format!("k.k{l}"));
            slots.push(format!("k.v{l}"));
        }
        for l in 0..4 {
            slots.push(format!("k.dk{l}"));
            slots.push(format!("k.dv{l}"));
        }
        host("slot.drop", Message::body(json!({ "slots": slots }))).map_err(err)?;
        // `session.rs`'s `normalize_peak`.
        let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1e-7 && peak < 0.95 {
            let scale = 0.95 / peak;
            for s in &mut audio {
                *s *= scale;
            }
        }
        Ok((frames, audio))
    }
}

impl Program for Kyutai {
    fn init(config: Value) -> Result<Self, String> {
        Ok(Kyutai {
            tokenizer: config["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .to_string(),
            cfg_values: config["cfg_values"]
                .as_array()
                .ok_or("config has no cfg_values")?
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            cfg_alpha: config["cfg_alpha"].as_f64().unwrap_or(2.0) as f32,
            max_steps: config["max_steps"].as_u64().unwrap_or(100) as usize,
            text_temperature: config["text_temperature"].as_f64().unwrap_or(0.6),
            audio_temperature: config["audio_temperature"].as_f64().unwrap_or(0.6),
            seed: config["seed"].as_u64().unwrap_or(42),
            rope_theta: config["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            sample_rate: config["sample_rate"].as_u64().unwrap_or(24000) as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        // `synthesize` carries the text; `run` its options do.
        let text = req.body["text"]
            .as_str()
            .or_else(|| req.body["options"]["text"].as_str())
            .ok_or(format!("{task} takes `text`"))?;
        let (frames, audio) = self
            .speak(text, &req.body["options"])
            .map_err(|e| format!("{e:#}"))?;
        match task {
            "synthesize" => {
                let n = audio.len();
                Ok(Message::new(
                    json!({"sample_rate": self.sample_rate}),
                    vec![Tensor::f32("audio", vec![n], &audio)],
                ))
            }
            _ => {
                let (t, books) = (frames.len(), frames[0].len());
                let codes: Vec<i32> = frames.iter().flatten().map(|&c| c as i32).collect();
                Ok(Message::tensors(vec![
                    Tensor::i32("codes", vec![t, books], &codes),
                    Tensor::f32("audio", vec![audio.len()], &audio),
                ]))
            }
        }
    }
}

rlx_program::program!(Kyutai);
