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

//! MOSS-TTS-Nano's host logic as an RLX program, answering `synthesize`:
//! what `MossNative::synthesize` does around its graphs — the prompt rows
//! (the manifest's templates, the voice's codes, the host's tokenizer), the
//! frame loop (the prefill re-run at buckets of 32, the runner's random
//! draws for the sampled frame, the repetition mask), the peak limit and
//! the pause trimming (`rlx-moss-nano`'s `dsp`). The prefill, the sampled
//! frame and the codec run in rlx.

use std::path::Path;

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

#[allow(dead_code, clippy::all)]
#[path = "../../src/config.rs"]
mod config;
#[allow(dead_code, clippy::all)]
#[path = "../../src/dsp.rs"]
mod dsp;

const TOKENIZER: &str = "sidecar:tokenizer.json";
const HIDDEN: usize = 768;
const BUCKET: usize = 32;
const CODEBOOK: usize = 1024;

/// The runner's splitmix64.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Moss {
    manifest: config::Manifest,
    sample_rate: u32,
    channels: usize,
    default_voice: String,
    max_frames: usize,
}

fn i32s(name: &str, shape: Vec<usize>, v: &[i32]) -> Tensor {
    Tensor::i32(name, shape, v)
}

impl Moss {
    fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": text})),
        )?;
        Ok(m.body["ids"]
            .as_array()
            .ok_or("no ids")?
            .iter()
            .filter_map(|v| v.as_i64().map(|v| v as i32))
            .collect())
    }

    /// The prompt as rows of `n_vq + 1`: text tokens padded, the voice's
    /// frames behind their slot (`build_input_ids`).
    fn rows(&self, text: &str, voice: &[Vec<i32>]) -> Result<Vec<i32>, String> {
        let c = &self.manifest.tts_config;
        let t = &self.manifest.prompt_templates;
        let rw = c.row_width();
        let pad = c.audio_pad_token_id;
        let mut rows: Vec<i32> = Vec::new();
        let push_text = |tok: i32, rows: &mut Vec<i32>| {
            rows.push(tok);
            rows.extend(std::iter::repeat_n(pad, rw - 1));
        };
        for &tok in &t.user_prompt_prefix_token_ids {
            push_text(tok, &mut rows);
        }
        push_text(c.audio_start_token_id, &mut rows);
        for frame in voice {
            if frame.len() != c.n_vq {
                return Err(format!(
                    "a voice frame of {} codes, not {}",
                    frame.len(),
                    c.n_vq
                ));
            }
            rows.push(c.audio_user_slot_token_id);
            rows.extend_from_slice(frame);
        }
        push_text(c.audio_end_token_id, &mut rows);
        for &tok in &t.user_prompt_after_reference_token_ids {
            push_text(tok, &mut rows);
        }
        for tok in self.encode(text)? {
            push_text(tok, &mut rows);
        }
        for &tok in &t.assistant_prompt_prefix_token_ids {
            push_text(tok, &mut rows);
        }
        push_text(c.audio_start_token_id, &mut rows);
        Ok(rows)
    }

    /// The frames the model speaks `text` in, in `voice`.
    fn generate(
        &self,
        text: &str,
        voice: &str,
        seed: u64,
        max_frames: usize,
    ) -> Result<Vec<Vec<i32>>, String> {
        let c = &self.manifest.tts_config;
        let (rw, nvq, pad) = (c.row_width(), c.n_vq, c.audio_pad_token_id);
        let v = self
            .manifest
            .voice(voice)
            .ok_or_else(|| format!("unknown voice '{voice}'"))?;
        let mut rows = self.rows(text, &v.prompt_audio_codes)?;
        let hard_cap = rows.len() / rw + max_frames;
        let mut rng = Rng::new(seed);
        let mut seen = vec![0i32; nvq * CODEBOOK];
        let mut frames: Vec<Vec<i32>> = Vec::new();
        for _ in 0..max_frames {
            // The prefill over the rows, padded to a bucket of 32 (as the
            // runner runs it), its last real position's hidden state.
            let seq = rows.len() / rw;
            let bucket = (seq.max(1).div_ceil(BUCKET) * BUCKET).min(hard_cap);
            let mut ids = vec![pad; bucket * rw];
            ids[..rows.len()].copy_from_slice(&rows);
            let mask: Vec<i32> = (0..bucket).map(|i| i32::from(i < seq)).collect();
            let out = run(
                "moss_tts_prefill",
                vec![
                    i32s("input_ids", vec![1, bucket, rw], &ids),
                    i32s("attention_mask", vec![1, bucket], &mask),
                ],
            )?;
            let hidden = out.first().ok_or("the prefill gave no output")?.to_f32();
            let last = &hidden[(seq - 1) * HIDDEN..seq * HIDDEN];
            let aru = rng.uniform();
            let aut: Vec<f32> = (0..nvq).map(|_| rng.uniform()).collect();
            let out = run(
                "moss_tts_local_fixed_sampled_frame",
                vec![
                    Tensor::f32("global_hidden", vec![1, HIDDEN], last),
                    i32s("repetition_seen_mask", vec![1, nvq, CODEBOOK], &seen),
                    Tensor::f32("assistant_random_u", vec![1], &[aru]),
                    Tensor::f32("audio_random_u", vec![1, nvq], &aut),
                ],
            )?;
            if out.len() < 2 {
                return Err("the sampled frame gave no frame".into());
            }
            if out[0].to_i64().first().copied().unwrap_or(0) == 0 {
                break;
            }
            let frame: Vec<i32> = out[1].to_i64().iter().map(|&x| x as i32).collect();
            for (ch, &code) in frame.iter().enumerate() {
                if (code as usize) < CODEBOOK {
                    seen[ch * CODEBOOK + code as usize] = 1;
                }
            }
            rows.push(c.audio_assistant_slot_token_id);
            rows.extend_from_slice(&frame);
            frames.push(frame);
        }
        Ok(frames)
    }

    /// `frames` through the codec, peak-limited, its pauses trimmed.
    fn render(&self, frames: &[Vec<i32>]) -> Result<Vec<f32>, String> {
        let nvq = self.manifest.tts_config.n_vq;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let n = frames.len();
        let flat: Vec<i32> = frames.iter().flatten().copied().collect();
        let out = run(
            "moss_audio_tokenizer_decode_full",
            vec![
                i32s("audio_codes", vec![1, n, nvq], &flat),
                i32s("audio_code_lengths", vec![1], &[n as i32]),
            ],
        )?;
        let audio = out.first().ok_or("the codec gave no audio")?.to_f32();
        let ch = self.channels;
        let samples = audio.len() / ch;
        let mut inter = vec![0f32; audio.len()];
        for c in 0..ch {
            for i in 0..samples {
                inter[i * ch + c] = audio[c * samples + i];
            }
        }
        let peak = inter.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if peak > 0.98 {
            let g = 0.98 / peak;
            for s in &mut inter {
                *s *= g;
            }
        }
        Ok(dsp::tighten_pauses(
            &inter,
            self.sample_rate,
            ch,
            dsp::TightenOpts::default(),
        ))
    }
}

impl Moss {
    /// [`Moss::generate`] with the options a call sets.
    fn generate_from(&self, o: &Value, text: &str) -> Result<Vec<Vec<i32>>, String> {
        let voice = o["voice"]
            .as_str()
            .unwrap_or(&self.default_voice)
            .to_string();
        let seed = o["seed"].as_u64().unwrap_or(0);
        let max_frames = o["max_frames"]
            .as_u64()
            .map_or(self.max_frames, |v| v as usize);
        self.generate(text, &voice, seed, max_frames)
    }
}

impl Program for Moss {
    fn init(config: Value) -> Result<Self, String> {
        let manifest = config::Manifest::load(Path::new("/package/browser_poc_manifest.json"))
            .map_err(|e| e.to_string())?;
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        Ok(Moss {
            default_voice: config["voice"]
                .as_str()
                .ok_or("config has no voice")?
                .into(),
            sample_rate: u("sample_rate")? as u32,
            channels: u("channels")? as usize,
            max_frames: u("max_frames")? as usize,
            manifest,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        // `run`: the frames alone (`options.text`, `voice`, `seed`,
        // `max_frames`), `codes` `[frames, n_vq]`.
        if task == "run" {
            let o = &req.body["options"];
            let text = o["text"].as_str().ok_or("run takes options.text")?;
            let frames = self.generate_from(o, text)?;
            let n = frames.len();
            let nvq = self.manifest.tts_config.n_vq;
            let flat: Vec<i32> = frames.into_iter().flatten().collect();
            return Ok(Message::tensors(vec![Tensor::i32(
                "codes",
                vec![n, nvq],
                &flat,
            )]));
        }
        if task != "synthesize" {
            return Err(format!(
                "MOSS-TTS-Nano answers synthesize and run, not {task}"
            ));
        }
        let text = req.body["text"].as_str().ok_or("synthesize takes text")?;
        let o = &req.body["options"];
        // `options.codes`: frames to decode, generation skipped.
        let frames = match o["codes"].as_array() {
            Some(rows) => rows
                .iter()
                .map(|r| {
                    r.as_array()
                        .map(|c| {
                            c.iter()
                                .filter_map(|v| v.as_i64().map(|v| v as i32))
                                .collect()
                        })
                        .ok_or("options.codes is frames of codes")
                })
                .collect::<Result<Vec<Vec<i32>>, _>>()?,
            None => self.generate_from(o, text)?,
        };
        let wav = self.render(&frames)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate, "channels": self.channels}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Moss);
