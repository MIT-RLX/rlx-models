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

//! FunASR's host logic as an RLX program, answering `transcribe` for
//! SenseVoiceSmall and Paraformer (`config.kind`), what each runner does: the
//! crate's frontend (`frontend.rs`: Kaldi fbank, LFR, the checkpoint's CMVN);
//! then for SenseVoice the package's `encoder` on the features after the
//! prompt (`options.language`, else `auto`; `options.itn`), each frame's CTC
//! id collapsed and the blank dropped; for Paraformer the `encoder` (its
//! states and the CIF alphas), the CIF fired (`cif.rs`) and the `decoder`
//! on its embeddings, the blank dropped; the pieces joined (`<|…|>` tags
//! apart, `▁` a space, `@@` a joined word piece).

#[allow(dead_code)]
#[path = "../../src/cif.rs"]
mod cif;
#[allow(dead_code)]
#[path = "../../src/frontend.rs"]
mod frontend;

/// What `cif.rs` reads of the crate's config.
#[allow(dead_code)]
mod config {
    pub struct CifConfig {
        pub idim: usize,
        pub l_order: usize,
        pub r_order: usize,
        pub threshold: f32,
        pub tail_threshold: f32,
        pub smooth_factor: f32,
        pub noise_threshold: f32,
    }
}

use frontend::{Cmvn, FrontendConfig, WavFrontend};
use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Map, Value, json};

const SP_SPACE: char = '\u{2581}';

struct SenseVoice {
    /// Paraformer's CIF and encoder width, when it is one.
    paraformer: Option<(config::CifConfig, usize)>,
    frontend: WavFrontend,
    sample_rate: u64,
    languages: Map<String, Value>,
    textnorm: Map<String, Value>,
    blank: u32,
    pieces: Vec<String>,
}

fn is_tag(p: &str) -> bool {
    p.starts_with("<|") && p.ends_with("|>")
}

fn is_structural(p: &str) -> bool {
    matches!(
        p,
        "<blank>" | "<s>" | "</s>" | "<sos/eos>" | "<sos>" | "<eos>" | "" | "<pad>"
    )
}

impl SenseVoice {
    /// `Tokenizer::decode(ids, true)` and `Tokenizer::tags`.
    fn decode(&self, ids: &[u32]) -> (String, Vec<String>) {
        let mut kept = String::new();
        let mut tags = Vec::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            if is_tag(p) {
                tags.push(p.clone());
            }
            if is_structural(p) || is_tag(p) {
                continue;
            }
            kept.push_str(p);
        }
        let text = if kept.contains(SP_SPACE) {
            kept.replace(SP_SPACE, " ").trim().to_string()
        } else {
            kept.replace("@@", "")
        };
        (text, tags)
    }
}

impl Program for SenseVoice {
    fn init(config: Value) -> Result<Self, String> {
        let fe = &config["frontend"];
        let u = |k: &str| fe[k].as_u64().ok_or(format!("frontend has no {k}"));
        let f = |k: &str| {
            fe[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("frontend has no {k}"))
        };
        let cfg = FrontendConfig {
            sample_rate: u("sample_rate")? as usize,
            n_mels: u("n_mels")? as usize,
            frame_length_ms: f("frame_length_ms")?,
            frame_shift_ms: f("frame_shift_ms")?,
            low_freq: f("low_freq")?,
            high_freq: f("high_freq")?,
            preemphasis: f("preemphasis")?,
            lfr_m: u("lfr_m")? as usize,
            lfr_n: u("lfr_n")? as usize,
        };
        let floats = |v: &Value| -> Vec<f32> {
            v.as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|x| x as f32))
                        .collect()
                })
                .unwrap_or_default()
        };
        let cmvn = config["cmvn"].as_object().map(|c| Cmvn {
            neg_mean: floats(&c["neg_mean"]),
            inv_std: floats(&c["inv_std"]),
        });
        let paraformer = match config["kind"].as_str() {
            Some("paraformer") => {
                let c = &config["cif"];
                let u = |k: &str| c[k].as_u64().ok_or(format!("cif has no {k}"));
                let f = |k: &str| {
                    c[k].as_f64()
                        .map(|v| v as f32)
                        .ok_or(format!("cif has no {k}"))
                };
                let cif = config::CifConfig {
                    idim: u("idim")? as usize,
                    l_order: u("l_order")? as usize,
                    r_order: u("r_order")? as usize,
                    threshold: f("threshold")?,
                    tail_threshold: f("tail_threshold")?,
                    smooth_factor: f("smooth_factor")?,
                    noise_threshold: f("noise_threshold")?,
                };
                let dim = config["dim"].as_u64().ok_or("config has no dim")? as usize;
                Some((cif, dim))
            }
            _ => None,
        };
        Ok(SenseVoice {
            paraformer,
            sample_rate: cfg.sample_rate as u64,
            frontend: WavFrontend::new(cfg, cmvn),
            languages: config["languages"].as_object().cloned().unwrap_or_default(),
            textnorm: config["textnorm"].as_object().cloned().unwrap_or_default(),
            blank: config["blank"].as_u64().ok_or("config has no blank")? as u32,
            pieces: config["pieces"]
                .as_array()
                .ok_or("config has no pieces")?
                .iter()
                .map(|p| p.as_str().unwrap_or("").to_string())
                .collect(),
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "transcribe" {
            return Err(format!("SenseVoice answers transcribe, not {task}"));
        }
        let rate = req.body["sample_rate"].as_u64().unwrap_or(self.sample_rate);
        if rate != self.sample_rate {
            return Err(format!(
                "the audio is at {rate} Hz; this model takes {} Hz",
                self.sample_rate
            ));
        }
        let audio = req
            .tensor("audio")
            .ok_or("transcribe takes `audio`")?
            .to_f32();
        let o = &req.body["options"];
        let lang = o["language"].as_str().unwrap_or("auto");
        let lid = self
            .languages
            .get(lang)
            .or_else(|| self.languages.get("auto"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as i32;
        let norm = if o["itn"].as_bool().unwrap_or(false) {
            "withitn"
        } else {
            "woitn"
        };
        let tn = self
            .textnorm
            .get(norm)
            .and_then(Value::as_u64)
            .unwrap_or(15) as i32;
        let feats = self.frontend.extract(&audio);
        if feats.n_frames == 0 {
            return Err("the audio is too short".into());
        }
        if let Some((cif, d)) = &self.paraformer {
            let t = feats.n_frames;
            let outs = run(
                "encoder",
                vec![Tensor::f32(
                    "feats",
                    vec![1, t, feats.feat_dim],
                    &feats.data,
                )],
            )?;
            let field = |n: &str| {
                outs.iter()
                    .find(|o| o.name == n)
                    .map(Tensor::to_f32)
                    .ok_or(format!("the encoder gave no {n}"))
            };
            let (enc, alphas) = (field("enc")?, field("alphas")?);
            let (acoustic, l) = cif::integrate_and_fire(&enc, t, *d, &alphas, cif);
            let mut ids = Vec::new();
            if l > 0 {
                let outs = run(
                    "decoder",
                    vec![
                        Tensor::f32("memory", vec![1, t, *d], &enc),
                        Tensor::f32("tgt", vec![1, l, *d], &acoustic),
                    ],
                )?;
                let tokens = outs.first().ok_or("the decoder returned nothing")?.to_i64();
                ids = tokens
                    .into_iter()
                    .map(|id| id as u32)
                    .filter(|&id| id != self.blank)
                    .collect();
            }
            let (text, tags) = self.decode(&ids);
            return Ok(Message::body(
                json!({"text": text, "tags": tags, "ids": ids}),
            ));
        }
        let outs = run(
            "encoder",
            vec![
                Tensor::i32("prompt", vec![4], &[lid, 1, 2, tn]),
                Tensor::f32(
                    "feats",
                    vec![1, feats.n_frames, feats.feat_dim],
                    &feats.data,
                ),
            ],
        )?;
        let frames = outs.first().ok_or("the encoder returned nothing")?.to_i64();
        let mut ids = Vec::new();
        let mut prev = None;
        for &id in &frames {
            let id = id as u32;
            if Some(id) != prev && id != self.blank {
                ids.push(id);
            }
            prev = Some(id);
        }
        let (text, tags) = self.decode(&ids);
        Ok(Message::body(
            json!({"text": text, "tags": tags, "ids": ids}),
        ))
    }
}

rlx_program::program!(SenseVoice);
