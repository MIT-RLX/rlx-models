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

//! HumeAI TADA's host logic as an RLX program, answering `synthesize`:
//! what `synth::Synthesizer::synthesize` does — the text normalized
//! (`text.rs`) and tokenized (the host's tokenizer), the prompt planned
//! around the voice (`voice.json`, or `options.voice`: its tokens' latents
//! and frame gaps, its transcript masked out); the backbone over the
//! prompt's rows (`embed`, `prefill`, both guidance branches, caches kept
//! in slots); each step the next token's row fed back (`embed`, `step`),
//! the flow-matching head solved from the two hidden states and seeded
//! noise (`solver`, `rng.rs`), the latent and its frame gaps (`gray.rs`)
//! fed to the next; then the generated latents laid on the frame grid,
//! decoded (`codec_attn`, `codec_conv`), the leading gap trimmed and a
//! runaway cut (`voice_clone.rs`). `run` answers the latents and gaps.

#[allow(dead_code)]
#[path = "../../src/gray.rs"]
mod gray;
#[allow(dead_code)]
#[path = "../../src/rng.rs"]
mod rng;
#[allow(dead_code)]
#[path = "../../src/text.rs"]
mod text;
#[allow(dead_code)]
#[path = "../../../rlx-models-core/src/voice_clone.rs"]
mod voice_clone;

use rlx_program::{Message, Program, Tensor, host, run_body};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";
/// Frames of the prompt handed back to the model before it takes over.
const NUM_TRANSITION_STEPS: usize = 5;
/// KV cache rows, a multiple of this (`backbone::decode_bucket`).
const DECODE_BUCKET: usize = 64;

/// A voice prompt: its token ids, their latents (`[tokens, acoustic]`) and
/// frame gaps.
struct Voice {
    token_ids: Vec<u32>,
    token_values: Vec<f32>,
    before: Vec<u32>,
    after: Vec<u32>,
}

impl Voice {
    fn parse(v: &Value) -> Result<Self, String> {
        let u32s = |k: &str| -> Result<Vec<u32>, String> {
            Ok(v[k]
                .as_array()
                .ok_or(format!("the voice has no {k}"))?
                .iter()
                .filter_map(|x| x.as_u64().map(|x| x as u32))
                .collect())
        };
        Ok(Voice {
            token_ids: u32s("token_ids")?,
            token_values: v["token_values"]
                .as_array()
                .ok_or("the voice has no token_values")?
                .iter()
                .filter_map(|x| x.as_f64().map(|x| x as f32))
                .collect(),
            before: u32s("before")?,
            after: u32s("after")?,
        })
    }
}

struct Tada {
    voice: Voice,
    /// The backbone's layers, for its caches' names.
    layers: usize,
    prefix: Vec<u32>,
    bos: u32,
    eos: u32,
    eot: u32,
    start_header: u32,
    end_header: u32,
    pad: u32,
    hidden: usize,
    adim: usize,
    bits: usize,
    classes: u32,
    latent: usize,
    shift: usize,
    num_eos: usize,
    mean: f32,
    std: f32,
    noise_temperature: f32,
    seed: u64,
    sample_rate: u32,
    frame_rate: usize,
}

/// What `Synthesizer::plan` derives before the first forward pass.
struct Plan {
    input_ids: Vec<u32>,
    prompt_rows: usize,
    acoustic: Vec<f32>,
    masks: Vec<u8>,
    before: Vec<u32>,
    after: Vec<u32>,
    prefill_len: usize,
    num_steps: usize,
}

fn named(outs: &[Tensor], n: &str) -> Result<Vec<f32>, String> {
    outs.iter()
        .find(|t| t.name == n)
        .map(Tensor::to_f32)
        .ok_or_else(|| format!("no output {n}"))
}

fn encode(text: &str) -> Result<Vec<u32>, String> {
    let m = host(
        "tokenizer.encode",
        Message::body(json!({"tokenizer": TOKENIZER, "text": text, "special": false})),
    )?;
    Ok(m.body["ids"]
        .as_array()
        .ok_or("no ids")?
        .iter()
        .filter_map(|v| v.as_u64().map(|v| v as u32))
        .collect())
}

impl Tada {
    fn is_structural(&self, t: u32) -> bool {
        t == self.start_header || t == self.end_header || t == self.eot
    }

    fn plan(&self, voice: &Voice, text: &str) -> Result<Plan, String> {
        let target = encode(&text::normalize_text(text))?;
        if target.is_empty() {
            return Err("target text tokenized to nothing".into());
        }
        let n = voice.token_ids.len();
        let mut input_ids = vec![self.bos];
        input_ids.extend(&self.prefix);
        input_ids.extend(&voice.token_ids);
        input_ids.extend(&target);
        input_ids.extend(std::iter::repeat_n(self.eot, self.num_eos));

        let prefix_len = self.prefix.len();
        let total_rows = prefix_len + n;
        if total_rows <= NUM_TRANSITION_STEPS {
            return Err(format!(
                "reference audio is too short: {total_rows} prompt rows"
            ));
        }
        let rows = total_rows - NUM_TRANSITION_STEPS;
        let a = self.adim;
        let mut acoustic = vec![0f32; rows * a];
        let mut masks = vec![0u8; rows];
        let mut before = vec![0u32; rows];
        let mut after = vec![0u32; rows];
        for r in prefix_len..rows {
            let i = r - prefix_len;
            acoustic[r * a..(r + 1) * a].copy_from_slice(&voice.token_values[i * a..(i + 1) * a]);
            masks[r] = 1;
            before[r] = voice.before[i];
            after[r] = voice.after[i];
        }
        let mut shifted = masks[1..].to_vec();
        shifted.push(1);

        // Hide the reference transcript: only chat structure survives.
        let mut in_header = false;
        for slot in input_ids.iter_mut().take(rows) {
            let tok = *slot;
            let keep = if tok == self.start_header {
                in_header = true;
                true
            } else if tok == self.end_header {
                in_header = false;
                true
            } else {
                in_header || tok == self.eot || tok == self.bos || tok == self.eos
            };
            if !keep {
                *slot = self.pad;
            }
        }

        let num_steps = input_ids.len();
        let shift = self.shift;
        let cap = num_steps.saturating_sub(shift + 1);
        let n_ac = cap.min(rows.saturating_sub(1));
        let n_frames_cap = rows.saturating_sub(2);
        let n_prefill = if n_ac > 0 { n_ac.min(n_frames_cap) } else { 0 };
        if n_prefill == 0 {
            return Err(format!(
                "reference audio is too short to prefill ({rows} prompt rows)"
            ));
        }
        let prefill_len = num_steps.min(shift + n_prefill + 1);
        if prefill_len >= num_steps {
            return Err("the reference prompt fills every step: no room to generate".into());
        }
        Ok(Plan {
            input_ids,
            prompt_rows: rows,
            acoustic,
            masks: shifted,
            before,
            after,
            prefill_len,
            num_steps,
        })
    }

    /// Both branches' embeddings of rows with `tokens` (`[2][t]`), latents
    /// and gaps, kept in the slot `t.x`.
    fn embed(
        &self,
        tokens: [&[u32]; 2],
        latent: &[f32],
        mask: &[u8],
        before: &[u32],
        after: &[u32],
    ) -> Result<(), String> {
        let t = mask.len();
        let toks: Vec<i32> = tokens
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as i32))
            .collect();
        let i32s = |v: &[u32]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
        let m: Vec<i32> = mask.iter().map(|&x| x as i32).collect();
        run_body(
            "embed",
            vec![
                Tensor::i32("tokens", vec![2, t], &toks),
                Tensor::f32("latent", vec![t, self.adim], latent),
                Tensor::i32("mask", vec![t], &m),
                Tensor::i32("before", vec![t], &i32s(before)),
                Tensor::i32("after", vec![t], &i32s(after)),
            ],
            json!({"keep": {"embeds": "t.x"}}),
        )?;
        Ok(())
    }

    /// `generate()`'s latents (`[rows, acoustic]`, the encoder's
    /// normalization undone) and frame gaps (`rows + 1`).
    fn run_loop(
        &self,
        plan: &Plan,
        seed: u64,
        temperature: f32,
    ) -> Result<(Vec<f32>, Vec<u32>), String> {
        let layers = self.layers;
        let (a, shift) = (self.adim, self.shift);
        let pl = plan.prefill_len;
        let row = |i: usize| &plan.acoustic[i * a..(i + 1) * a];

        // Prefill: position t carries the latent of token t - shift - 1 and
        // the gaps of token t - shift.
        let n_ac = pl.saturating_sub(shift + 1).min(plan.prompt_rows);
        let n_t = pl.saturating_sub(shift + 1).min(plan.before.len() - 1);
        let mut latent = vec![0f32; pl * a];
        let (mut mask, mut before, mut after) = (vec![0u8; pl], vec![0u32; pl], vec![0u32; pl]);
        for t in 0..pl {
            if t <= shift {
                continue;
            }
            let i = t - shift - 1;
            if i < n_ac {
                latent[t * a..(t + 1) * a].copy_from_slice(row(i));
                mask[t] = plan.masks[i];
            }
            if i < n_t {
                before[t] = plan.before[i + 1];
                after[t] = plan.after[i + 1];
            }
        }
        let ids = &plan.input_ids[..pl];
        self.embed([ids, ids], &latent, &mask, &before, &after)?;
        let upper = plan.num_steps.div_ceil(DECODE_BUCKET) * DECODE_BUCKET;
        let names: Vec<String> = (0..layers)
            .flat_map(|l| [format!("past_k_{l}"), format!("past_v_{l}")])
            .collect();
        let keep: serde_json::Map<String, Value> = names
            .iter()
            .map(|n| (n.clone(), Value::String(format!("t.{n}"))))
            .collect();
        run_body(
            "prefill",
            vec![],
            json!({"from": {"inputs_embeds": "t.x"}, "keep": keep, "dims": {"cap": upper}}),
        )?;

        let n_prefill = pl - shift;
        let mut all_latents: Vec<f32> = Vec::with_capacity(plan.num_steps * a);
        let mut all_gaps: Vec<u32> = Vec::with_capacity(plan.num_steps);
        for i in 0..n_prefill {
            all_latents.extend_from_slice(row(i));
            all_gaps.push(plan.before[i + 1]);
        }
        let mut cur_latent = row(n_prefill - 1).to_vec();
        let mut cur_mask = plan.masks[n_prefill - 1];
        let mut cur_before = plan.before[n_prefill];
        let mut cur_after = plan.after[n_prefill];
        let mut last_gap = None;
        let mut rng = rng::Normal::new(seed);
        let mut noise = vec![0f32; self.latent];
        let from: serde_json::Map<String, Value> =
            std::iter::once(("inputs_embeds".to_string(), Value::String("t.x".into())))
                .chain(
                    names
                        .iter()
                        .map(|n| (n.clone(), Value::String(format!("t.{n}")))),
                )
                .collect();
        let keep: serde_json::Map<String, Value> = names
            .iter()
            .map(|n| (format!("{n}.next"), Value::String(format!("t.{n}"))))
            .collect();
        let h = self.hidden;
        for step in pl..plan.num_steps {
            let token = plan.input_ids[step.min(plan.input_ids.len() - 1)];
            let neg = if self.is_structural(token) {
                token
            } else {
                self.pad
            };
            self.embed(
                [&[token], &[neg]],
                &cur_latent,
                &[cur_mask],
                &[cur_before],
                &[cur_after],
            )?;
            let mut mask = vec![0f32; 2 * (upper + 1)];
            for b in 0..2 {
                let m = &mut mask[b * (upper + 1)..(b + 1) * (upper + 1)];
                m[..step].fill(1.0);
                m[upper] = 1.0;
            }
            let outs = run_body(
                "step",
                vec![
                    Tensor::f32("mask", vec![2, upper + 1], &mask),
                    Tensor::f32("position", vec![1], &[step as f32]),
                ],
                json!({"from": from, "keep": keep}),
            )?;
            let hidden = named(&outs, "hidden")?;
            rng.fill(&mut noise, temperature);
            let outs = run_body(
                "solver",
                vec![
                    Tensor::f32("noise", vec![1, self.latent], &noise),
                    Tensor::f32("cond", vec![1, h], &hidden[..h]),
                    Tensor::f32("neg_cond", vec![1, h], &hidden[h..2 * h]),
                ],
                json!({}),
            )?;
            let speech = named(&outs, "speech")?;
            let bits = self.bits;
            let pred_before = gray::decode(&speech[a..a + bits], bits);
            let pred_after = gray::decode(&speech[a + bits..], bits);
            let i = step - shift;
            if i < plan.prompt_rows {
                cur_latent.copy_from_slice(row(i));
                cur_mask = plan.masks[i];
            } else {
                cur_latent.copy_from_slice(&speech[..a]);
                cur_mask = 1;
            }
            all_latents.extend_from_slice(&cur_latent);
            if i + 1 < plan.prompt_rows {
                cur_before = plan.before[i + 1];
                cur_after = plan.after[i + 1];
            } else {
                cur_before = pred_before.min(self.classes - 1);
                cur_after = pred_after.min(self.classes - 1);
            }
            all_gaps.push(cur_before);
            last_gap = Some(cur_before);
        }
        let mut slots: Vec<String> = names.iter().map(|n| format!("t.{n}")).collect();
        slots.push("t.x".into());
        host("slot.drop", Message::body(json!({ "slots": slots })))?;
        if let Some(g) = last_gap {
            all_gaps.push(g);
        }
        for v in &mut all_latents {
            *v = *v * self.std + self.mean;
        }
        Ok((all_latents, all_gaps))
    }

    /// `synthesize_latents`: the generated span only.
    fn latents(
        &self,
        voice: &Voice,
        text: &str,
        seed: u64,
        temperature: f32,
    ) -> Result<(Vec<f32>, Vec<u32>), String> {
        let plan = self.plan(voice, text)?;
        let (latents, gaps) = self.run_loop(&plan, seed, temperature)?;
        let skip = plan.prompt_rows + NUM_TRANSITION_STEPS - 1;
        let rows = latents.len() / self.adim;
        if rows <= skip || gaps.len() <= skip {
            return Err(format!(
                "synthesis produced {rows} latents but the prompt alone occupies {skip}: \
                 the target text is too short to leave any generated audio"
            ));
        }
        Ok((latents[skip * self.adim..].to_vec(), gaps[skip..].to_vec()))
    }

    /// Latents on the frame grid (`expand_to_frames`), decoded, trimmed.
    fn decode(&self, latents: &[f32], gaps: &[u32]) -> Result<Vec<f32>, String> {
        let a = self.adim;
        let n = latents.len() / a;
        if gaps.len() < n + 1 {
            return Err(format!("need {} frame gaps for {n} latents", n + 1));
        }
        let total: usize =
            (0..n).map(|i| gaps[i].max(1) as usize).sum::<usize>() + gaps[n] as usize;
        let mut z = vec![0f32; total * a];
        let mut r = 0usize;
        for i in 0..n {
            r += gaps[i].saturating_sub(1) as usize;
            z[r * a..(r + 1) * a].copy_from_slice(&latents[i * a..(i + 1) * a]);
            r += 1;
        }
        let mask: Vec<f32> = z
            .chunks(a)
            .map(|row| {
                if row.iter().any(|v| *v != 0.0) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        run_body(
            "codec_attn",
            vec![
                Tensor::f32("z", vec![1, total, a], &z),
                Tensor::f32("mask", vec![total], &mask),
            ],
            json!({"keep": {"x": "t.codec"}}),
        )?;
        let outs = run_body("codec_conv", vec![], json!({"from": {"x": "t.codec"}}))?;
        host("slot.drop", Message::body(json!({"slots": ["t.codec"]})))?;
        let wav = named(&outs, "audio")?;
        let trim = self.sample_rate as usize * gaps[0] as usize / self.frame_rate;
        let wav = wav.get(trim..).unwrap_or(&[]).to_vec();
        let cut = voice_clone::OutputTrim {
            trim_leading: false,
            ..voice_clone::OutputTrim::default()
        };
        if !voice_clone::has_tts_runaway(&wav, self.sample_rate, &cut) {
            return Ok(wav);
        }
        Ok(voice_clone::trim_tts_output(&wav, self.sample_rate, &cut))
    }
}

impl Program for Tada {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        let voice = config["voice"].as_str().ok_or("config has no voice")?;
        let bytes = host(
            "file",
            Message::body(json!({"name": format!("sidecar:{voice}")})),
        )?;
        let bytes = &bytes.tensors.first().ok_or("no voice bytes")?.data;
        let voice: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        Ok(Tada {
            voice: Voice::parse(&voice)?,
            layers: u("layers")? as usize,
            prefix: config["prefix"]
                .as_array()
                .ok_or("config has no prefix")?
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect(),
            bos: u("bos")? as u32,
            eos: u("eos")? as u32,
            eot: u("eot")? as u32,
            start_header: u("start_header")? as u32,
            end_header: u("end_header")? as u32,
            pad: u("pad")? as u32,
            hidden: u("hidden")? as usize,
            adim: u("acoustic_dim")? as usize,
            bits: u("time_bits")? as usize,
            classes: u("time_classes")? as u32,
            latent: u("latent")? as usize,
            shift: u("shift")? as usize,
            num_eos: u("num_eos")? as usize,
            mean: f("acoustic_mean")?,
            std: f("acoustic_std")?,
            noise_temperature: f("noise_temperature")?,
            seed: u("seed")?,
            sample_rate: u("sample_rate")? as u32,
            frame_rate: u("frame_rate")? as usize,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().unwrap_or(""),
            _ => return Err(format!("TADA answers synthesize and run, not {task}")),
        };
        let given;
        let voice = match o.get("voice") {
            Some(v) if v.is_object() => {
                given = Voice::parse(v)?;
                &given
            }
            _ => &self.voice,
        };
        let seed = o["seed"].as_u64().unwrap_or(self.seed);
        let temperature = o["noise_temperature"]
            .as_f64()
            .map_or(self.noise_temperature, |t| t as f32);
        let (latents, gaps) = self.latents(voice, text, seed, temperature)?;
        if task == "run" {
            let n = latents.len() / self.adim;
            let g: Vec<i32> = gaps.iter().map(|&v| v as i32).collect();
            return Ok(Message::tensors(vec![
                Tensor::f32("latents", vec![n, self.adim], &latents),
                Tensor::i32("gaps", vec![gaps.len()], &g),
            ]));
        }
        let wav = self.decode(&latents, &gaps)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Tada);
