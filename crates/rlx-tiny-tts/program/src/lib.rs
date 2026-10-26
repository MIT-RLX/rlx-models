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

//! TinyTTS / MeloTTS's host logic as an RLX program, answering
//! `synthesize`: what `TinyTts::synthesize` does around its graphs — the
//! English frontend (rlx-inflect-nano's: cleaning, numbers, WordPiece groups,
//! CMUdict and homographs, the neural G2P's search) and the VITS glue (the
//! durations, the alignment, the prior expanded and sampled; rlx-tiny-tts'
//! `glue`). The text encoder, the duration predictor, the flow, the
//! decoder and the G2P's GRU run in rlx.

use rlx_program::{Message, Program, Tensor};
use serde_json::{Value, json};

#[allow(dead_code, clippy::all)]
mod frontend;
#[allow(dead_code, clippy::all)]
#[path = "../../src/glue.rs"]
mod glue;
mod voice;

pub use voice::TinyTts;

impl Program for TinyTts {
    fn init(config: Value) -> Result<Self, String> {
        TinyTts::load(&config)
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "synthesize" {
            return Err(format!("TinyTTS answers synthesize, not {task}"));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        let length_scale = o["length_scale"]
            .as_f64()
            .map_or(self.length_scale, |v| v as f32);
        let noise_scale = o["noise_scale"]
            .as_f64()
            .map_or(self.noise_scale, |v| v as f32);
        let seed = o["seed"].as_u64().unwrap_or(self.seed);
        let wav = self.speak(text, length_scale, noise_scale, seed)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(TinyTts);
