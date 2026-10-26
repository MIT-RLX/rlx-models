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

//! Supertonic-3's host logic as an RLX program, answering `synthesize`:
//! what `Supertonic::synthesize` does around its graphs, the same code —
//! the text preprocessor and unicode tokenizer, the config and the voice
//! styles (`rlx-supertonic`'s own modules, reading the package's files),
//! the seeded noise and the flow-matching loop. The duration predictor,
//! the text encoder, the vector estimator and the vocoder run in rlx
//! (`rlx_program::run`).

use std::path::Path;

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

#[allow(dead_code, clippy::all)]
#[path = "../../src/config.rs"]
mod config;
#[allow(dead_code, clippy::all)]
#[path = "../../src/tokenize.rs"]
mod tokenize;
#[allow(dead_code, clippy::all)]
#[path = "../../src/voices.rs"]
mod voices;

/// Where the package's files are.
const PACKAGE: &str = "/package";
const MIN_AUDIBLE_PEAK: f32 = 1e-3;

/// The runner's deterministic Gaussian source (xorshift128+ → Box–Muller).
struct Rng {
    s0: u64,
    s1: u64,
    spare: Option<f32>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s0: next() | 1,
            s1: next() | 1,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        self.s0 = s0;
        s1 ^= s1 << 23;
        self.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.s1.wrapping_add(s0)
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn randn(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let mut u1 = self.next_unit();
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_unit();
        }
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }
}

struct Supertonic {
    cfg: config::StConfig,
    indexer: tokenize::UnicodeIndexer,
    default_voice: String,
    total_step: usize,
    speed: f32,
}

fn first(out: Vec<Tensor>, what: &str) -> Result<Vec<f32>, String> {
    Ok(out
        .first()
        .ok_or(format!("the {what} gave no output"))?
        .to_f32())
}

impl Supertonic {
    fn speak(
        &self,
        text: &str,
        lang: &str,
        voice: &voices::Voice,
        total: usize,
        speed: f32,
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        let ids = self.indexer.encode(text, lang).map_err(|e| e.to_string())?;
        if ids.is_empty() {
            return Err("the text tokenized to nothing".into());
        }
        let t = ids.len();
        let idb: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let text_ids = || Tensor::new("text_ids", "i64", vec![1, t], idb.clone());
        let text_mask = || Tensor::f32("text_mask", vec![1, 1, t], &vec![1.0; t]);
        let (dp, ttl) = (&voice.dp, &voice.ttl);
        let style_ttl = || Tensor::f32("style_ttl", vec![1, ttl.rows, ttl.cols], &ttl.data);

        let duration = first(
            run(
                "duration_predictor",
                vec![
                    text_ids(),
                    Tensor::f32("style_dp", vec![1, dp.rows, dp.cols], &dp.data),
                    text_mask(),
                ],
            )?,
            "duration predictor",
        )?;
        let duration = (duration.first().ok_or("no duration")? / speed).max(0.05);
        let text_emb = first(
            run("text_encoder", vec![text_ids(), style_ttl(), text_mask()])?,
            "text encoder",
        )?;

        let l = self.cfg.latent_len(duration);
        let ch = self.cfg.latent_channels();
        let mut rng = Rng::new(seed);
        let mut xt: Vec<f32> = (0..ch * l).map(|_| rng.randn()).collect();
        let total = total.max(1);
        for step in 0..total {
            xt = first(
                run(
                    "vector_estimator",
                    vec![
                        Tensor::f32("noisy_latent", vec![1, ch, l], &xt),
                        Tensor::f32("text_emb", vec![1, 256, t], &text_emb),
                        style_ttl(),
                        Tensor::f32("latent_mask", vec![1, 1, l], &vec![1.0; l]),
                        text_mask(),
                        Tensor::f32("current_step", vec![1], &[step as f32]),
                        Tensor::f32("total_step", vec![1], &[total as f32]),
                    ],
                )?,
                "vector estimator",
            )?;
        }
        let wav = first(
            run("vocoder", vec![Tensor::f32("latent", vec![1, ch, l], &xt)])?,
            "vocoder",
        )?;
        let n = ((duration * self.cfg.sample_rate as f32) as usize).min(wav.len());
        let audio = wav[..n.max(1)].to_vec();
        let peak = audio
            .iter()
            .fold(0f32, |a, s| if s.is_finite() { a.max(s.abs()) } else { a });
        if peak < MIN_AUDIBLE_PEAK {
            return Err(format!("the audio is silent (peak {peak:.2e})"));
        }
        Ok(audio)
    }
}

impl Program for Supertonic {
    fn init(config: Value) -> Result<Self, String> {
        let dir = Path::new(PACKAGE);
        Ok(Supertonic {
            cfg: config::StConfig::load(dir).map_err(|e| e.to_string())?,
            indexer: tokenize::UnicodeIndexer::load(dir).map_err(|e| e.to_string())?,
            default_voice: config["default_voice"]
                .as_str()
                .ok_or("config has no default_voice")?
                .into(),
            total_step: config["total_step"]
                .as_u64()
                .ok_or("config has no total_step")? as usize,
            speed: config["speed"].as_f64().ok_or("config has no speed")? as f32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("Supertonic answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let name = o["voice"].as_str().unwrap_or(&self.default_voice);
        if name.contains(['/', '\\']) || name.starts_with('.') {
            return Err(format!("no voice {name}"));
        }
        let voice =
            voices::Voice::load(&Path::new(PACKAGE).join(format!("voice_styles/{name}.json")))
                .map_err(|_| format!("no voice {name}"))?;
        let lang = o["lang"].as_str().unwrap_or("en");
        let steps = o["steps"].as_u64().map_or(self.total_step, |v| v as usize);
        let speed = o["speed"].as_f64().map_or(self.speed, |v| v as f32);
        let seed = o["seed"].as_u64().unwrap_or(0);
        let wav = self.speak(text, lang, &voice, steps, speed, seed)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.cfg.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Supertonic);
