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

//! TinyTTS / MeloTTS speaking, as the program does it: the frontend, the
//! package's graphs and the VITS glue between them (`TinyTts::speak`).
//! Shared with programs built on it (OpenVoice's).

use std::path::Path;

use rlx_program::{Tensor, run};
use serde_json::Value;

use crate::{frontend, glue};

pub struct TinyTts {
    frontend: frontend::English,
    add_blank: bool,
    speaker: i64,
    pub sample_rate: u32,
    channels: usize,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub seed: u64,
}

fn i64s(name: &str, shape: Vec<usize>, v: &[i64]) -> Tensor {
    Tensor::new(
        name,
        "i64",
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

impl TinyTts {
    pub fn speak(
        &self,
        text: &str,
        length_scale: f32,
        noise_scale: f32,
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        let (phone, tone, lang) = self
            .frontend
            .text_to_ids(text, self.add_blank)
            .map_err(|e| e.to_string())?;
        let t = phone.len();
        if t == 0 {
            return Err("the text has no phonemes".into());
        }
        let c = self.channels;
        let enc = run(
            "text_encoder",
            vec![
                i64s("phone_ids", vec![1, t], &phone),
                i64s("phone_lengths", vec![1], &[t as i64]),
                i64s("tone_ids", vec![1, t], &tone),
                i64s("language_ids", vec![1, t], &lang),
                Tensor::f32("bert", vec![1, 1024, t], &vec![0.0; 1024 * t]),
                Tensor::f32("ja_bert", vec![1, 768, t], &vec![0.0; 768 * t]),
                i64s("speaker_id", vec![1], &[self.speaker]),
            ],
        )?;
        if enc.len() < 5 {
            return Err(format!("the text encoder gave {} outputs", enc.len()));
        }
        let (m_p, logs_p) = (enc[1].to_f32(), enc[2].to_f32());
        let mut x_enc = enc[0].clone();
        x_enc.name = "x".into();
        let mut g = enc[4].clone();
        g.name = "g".into();
        let x_mask = vec![1.0f32; t];
        let dp = run(
            "duration_predictor",
            vec![
                x_enc,
                Tensor::f32("x_mask", vec![1, 1, t], &x_mask),
                g.clone(),
            ],
        )?;
        let logw = dp
            .first()
            .ok_or("the duration predictor gave no output")?
            .to_f32();

        let (w_ceil, y_len) = glue::durations(&logw, &x_mask, length_scale);
        let attn = glue::alignment_path(&w_ceil, y_len);
        let m_exp = glue::expand_prior(&attn, &m_p, c, t, y_len);
        let logs_exp = glue::expand_prior(&attn, &logs_p, c, t, y_len);
        let mut rng = glue::Rng::new(seed);
        let z_p = glue::sample_z_p(&m_exp, &logs_exp, noise_scale, &mut rng);

        let flow = run(
            "flow",
            vec![
                Tensor::f32("z_p", vec![1, c, y_len], &z_p),
                Tensor::f32("y_mask", vec![1, 1, y_len], &vec![1.0; y_len]),
                g.clone(),
            ],
        )?;
        let mut z = flow.into_iter().next().ok_or("the flow gave no output")?;
        z.name = "z".into();
        let dec = run("decoder", vec![z, g])?;
        let wav = dec.first().ok_or("the decoder gave no output")?.to_f32();
        let peak = wav
            .iter()
            .fold(0f32, |a, s| if s.is_finite() { a.max(s.abs()) } else { a });
        if peak < 1e-3 {
            return Err(format!("the audio is silent (peak {peak:.2e})"));
        }
        Ok(wav)
    }

    /// From the entry's config (the frontend's files at `/package/frontend`).
    pub fn load(config: &Value) -> Result<Self, String> {
        let f = |k: &str| config[k].as_f64().ok_or(format!("config has no {k}"));
        Ok(TinyTts {
            frontend: frontend::English::load(Path::new("/package/frontend"))
                .map_err(|e| e.to_string())?,
            add_blank: config["add_blank"].as_bool().unwrap_or(true),
            speaker: config["speaker"].as_i64().unwrap_or(0),
            sample_rate: f("sample_rate")? as u32,
            channels: f("channels")? as usize,
            noise_scale: f("noise_scale")? as f32,
            length_scale: f("length_scale")? as f32,
            seed: config["seed"].as_u64().unwrap_or(1234),
        })
    }
}
