// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! The decider's protocol as an RLX program, answering `run`: what
//! `PplxDecider::decide` does around the network, from the crate's own
//! `protocol.rs` — the request validated, the prompt rendered and tokenized
//! (by the package's tokenizer), split into question chunks when it is past
//! the longest bucket, each chunk through the package's `decide_{bucket}`
//! graph, each answer's distribution.
//!
//! A request's images (`data:` URLs) are decoded and preprocessed as the
//! runner does (`vision.rs`), run through the package's `vision` graph with
//! the patch positions and RoPE rows Qwen3.5's own `preprocess.rs` gives,
//! and their rows placed over the image tokens, at their M-RoPE positions.
//!
//! In: option `request`, the request JSON as text (so its questions keep
//! their order); `temperature` overrides the checkpoint's. Out:
//! `decision`, the `{"answers": …, "usage": …}` JSON as UTF-8 bytes.

#[allow(dead_code)]
#[path = "../../../rlx-qwen35/src/vision/preprocess.rs"]
mod preprocess;
#[allow(dead_code)]
#[path = "../../src/protocol.rs"]
mod protocol;
#[allow(dead_code)]
#[path = "../../src/vision.rs"]
mod vision;

/// The mmproj settings Qwen3.5's `preprocess.rs` reads.
mod config {
    pub struct MmProjConfig {
        pub patch_size: usize,
        pub n_merge: usize,
        pub image_min_pixels: usize,
        pub image_max_pixels: usize,
        pub image_mean: [f32; 3],
        pub image_std: [f32; 3],
    }

    impl MmProjConfig {
        pub fn align_size(&self) -> usize {
            self.patch_size * self.n_merge
        }
    }
}

use protocol::{Escaper, MAX_OPTIONS, Request, answer, probabilities, prompt_text};
use rlx_core::ordered_json::Json;
use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};
use vision::{MERGE, PATCH, Preprocessed, RgbImage};

struct Decider {
    tokenizer: String,
    marker_id: u32,
    codes: Vec<String>,
    escaper: Escaper,
    temperature: f64,
    key_mask: String,
    buckets: Vec<usize>,
    hidden: usize,
    image_pad_id: Option<u32>,
    /// The vision tower's settings, when the package has one.
    vision: Option<Vision>,
}

struct Vision {
    /// Side of the square position table.
    side: usize,
    head_dim: usize,
    min_pixels: usize,
    max_pixels: usize,
}

/// One tokenized sequence (`Encoded` in the runner).
struct Encoded {
    ids: Vec<u32>,
    is_image: Vec<bool>,
    grids: Vec<(usize, usize)>,
}

/// Bilinear weights `[dst, side]` onto a `side`-long axis, as the runner
/// resamples the position table (`linspace`, then two taps).
fn bilinear(dst: usize, side: usize) -> Vec<f32> {
    let mut r = vec![0f32; dst * side];
    let s = side as f32;
    for t in 0..dst {
        let g = if dst == 1 {
            0.0
        } else {
            t as f32 * (s - 1.0) / (dst as f32 - 1.0)
        };
        let y0 = g.floor() as usize;
        let y1 = (y0 + 1).min(side - 1);
        let f = g - y0 as f32;
        r[t * side + y0] += 1.0 - f;
        r[t * side + y1] += f;
    }
    r
}

/// `PplxDecider`'s bucket: a power of two from 64 to 512, then a multiple of 512.
fn seq_bucket(n: usize) -> usize {
    if n <= 512 {
        n.max(64).next_power_of_two()
    } else {
        n.div_ceil(512) * 512
    }
}

impl Decider {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer, "text": text})),
        )?;
        Ok(m.body["ids"]
            .as_array()
            .ok_or("tokenizer.encode returned no ids")?
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as u32)
            .collect())
    }

    /// `PplxDecider::encode_prepared`: each `<|image_pad|>` as many tokens
    /// as its image has merged patches.
    fn encode_with(&self, req: &Request, images: &[Preprocessed]) -> Result<Encoded, String> {
        let raw = self.encode(&prompt_text(req, &self.codes, &self.escaper))?;
        let (mut ids, mut is_image) = (Vec::new(), Vec::new());
        let mut next = images.iter();
        for t in raw {
            if Some(t) == self.image_pad_id {
                let img = next.next().ok_or("more image placeholders than images")?;
                ids.extend(std::iter::repeat_n(t, img.n_tokens()));
                is_image.extend(std::iter::repeat_n(true, img.n_tokens()));
            } else {
                ids.push(t);
                is_image.push(false);
            }
        }
        if next.next().is_some() {
            return Err("fewer image placeholders than images".into());
        }
        Ok(Encoded {
            ids,
            is_image,
            grids: images.iter().map(|i| (i.grid_h, i.grid_w)).collect(),
        })
    }

    /// The vision tower's rows `[tokens, hidden]` for one image.
    fn vision_rows(&self, v: &Vision, img: &Preprocessed) -> Result<Vec<f32>, String> {
        let (tw, th) = (img.resized.width, img.resized.height);
        let (hp, wp) = (th / PATCH, tw / PATCH);
        let n_pos = hp * wp;
        let cfg = config::MmProjConfig {
            patch_size: PATCH,
            n_merge: MERGE,
            image_min_pixels: v.min_pixels,
            image_max_pixels: v.max_pixels,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        };
        let idx = preprocess::build_spatial_merge_gather_idx(hp, wp, MERGE);
        let (cos, sin) = preprocess::vision_rope_feeds(
            &preprocess::build_vision_position_hw(tw, th, &cfg),
            v.head_dim,
        );
        let outs = run(
            "vision",
            vec![
                Tensor::f32("image", vec![1, 3, th, tw], &img.nchw),
                Tensor::f32("ry", vec![hp, v.side], &bilinear(hp, v.side)),
                Tensor::f32("rx", vec![wp, v.side], &bilinear(wp, v.side)),
                Tensor::f32("spatial_merge_idx", vec![n_pos], &idx),
                Tensor::f32("attn_mask", vec![1, n_pos], &vec![1.0; n_pos]),
                Tensor::f32("vision_rope_cos", vec![n_pos, v.head_dim], &cos),
                Tensor::f32("vision_rope_sin", vec![n_pos, v.head_dim], &sin),
            ],
        )?;
        let rows = outs
            .first()
            .ok_or("the vision tower returned nothing")?
            .to_f32();
        if rows.len() != img.n_tokens() * self.hidden {
            return Err(format!(
                "the vision tower returned {} values for {} tokens",
                rows.len(),
                img.n_tokens()
            ));
        }
        Ok(rows)
    }

    /// `PplxDecider::plan`: one sequence, or question halves until each fits.
    fn plan(
        &self,
        req: &Request,
        images: &[Preprocessed],
        out: &mut Vec<(Request, Encoded)>,
    ) -> Result<(), String> {
        let enc = self.encode_with(req, images)?;
        let longest = *self.buckets.last().ok_or("the package has no bucket")?;
        if enc.ids.len() <= longest {
            out.push((req.clone(), enc));
            return Ok(());
        }
        let n = req.questions.len();
        if n == 1 {
            return Err(format!(
                "Question {:?} needs {} tokens with the state and images, over the \
                 {longest}-token limit; shorten the state, lower the image resolution or \
                 raise max_length",
                req.questions[0].id,
                enc.ids.len()
            ));
        }
        self.plan(&req.subset(0..n / 2), images, out)?;
        self.plan(&req.subset(n / 2..n), images, out)
    }

    /// Readout logits of each question of `req`, tokenized as `enc`, its
    /// images' rows `vision` (in order).
    fn logits(
        &self,
        req: &Request,
        enc: &Encoded,
        vision: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let ids = &enc.ids;
        let n = ids.len();
        let markers: Vec<i32> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.marker_id).then_some(i as i32))
            .collect();
        if markers.len() != req.questions.len() {
            return Err(format!(
                "{} readout markers for {} questions after tokenization",
                markers.len(),
                req.questions.len()
            ));
        }
        let seq = seq_bucket(n);
        if !self.buckets.contains(&seq) {
            return Err(format!("the package has no graph for {seq} tokens"));
        }
        let mut tok = vec![0i32; seq];
        let mut tok_f = vec![0f32; seq];
        let mut mask = vec![0f32; seq];
        for (i, &t) in ids.iter().enumerate() {
            tok[i] = t as i32;
            tok_f[i] = t as f32;
            mask[i] = 1.0;
        }
        // Each token's M-RoPE position (`[t, t, t, 0]` for text, and for the
        // pads past it), and its row: its own embedding, or an image's.
        let at = vision::mrope_positions(&enc.is_image, &enc.grids).map_err(|e| e.to_string())?;
        let mut positions = vec![0i32; 4 * seq];
        let mut rows: Vec<i32> = (0..seq as i32).collect();
        let mut k = 0;
        for t in 0..seq {
            let p = at.get(t).copied().unwrap_or([t, t, t, 0]);
            for (s, v) in p.iter().enumerate() {
                positions[s * seq + t] = *v as i32;
            }
            if enc.is_image.get(t).copied().unwrap_or(false) {
                rows[t] = (seq + k) as i32;
                k += 1;
            }
        }
        let mut image_rows: Vec<f32> = vision.concat();
        if image_rows.len() != k * self.hidden {
            return Err(format!(
                "{k} image tokens for {} image rows",
                image_rows.len() / self.hidden
            ));
        }
        if image_rows.is_empty() {
            image_rows = vec![0.0; self.hidden];
        }
        let m_img = image_rows.len() / self.hidden;
        let m = markers.len();
        let outs = run(
            &format!("decide_{seq}"),
            vec![
                Tensor::i32("ids", vec![seq], &tok),
                Tensor::i32("rows", vec![seq], &rows),
                Tensor::i32("positions", vec![4, seq], &positions),
                Tensor::f32("image_rows", vec![m_img, self.hidden], &image_rows),
                Tensor::f32("input_ids", vec![1, seq], &tok_f),
                Tensor::f32(self.key_mask.as_str(), vec![1, seq], &mask),
                Tensor::i32("markers", vec![m], &markers),
            ],
        )?;
        let logits = outs.first().ok_or("decide returned nothing")?.to_f32();
        Ok(req
            .questions
            .iter()
            .enumerate()
            .map(|(k, q)| logits[k * MAX_OPTIONS..k * MAX_OPTIONS + q.keys.len()].to_vec())
            .collect())
    }
}

impl Program for Decider {
    fn init(config: Value) -> Result<Self, String> {
        let strings = |k: &str| -> Result<Vec<String>, String> {
            serde_json::from_value(config[k].clone()).map_err(|e| format!("config {k}: {e}"))
        };
        Ok(Decider {
            tokenizer: config["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .to_string(),
            marker_id: config["marker_id"]
                .as_u64()
                .ok_or("config has no marker_id")? as u32,
            codes: strings("codes")?,
            escaper: Escaper::new(&strings("escape_tokens")?),
            temperature: config["temperature"]
                .as_f64()
                .ok_or("config has no temperature")?,
            key_mask: config["key_mask"]
                .as_str()
                .ok_or("config has no key_mask")?
                .to_string(),
            buckets: serde_json::from_value(config["buckets"].clone())
                .map_err(|e| format!("config buckets: {e}"))?,
            hidden: config["hidden"].as_u64().ok_or("config has no hidden")? as usize,
            image_pad_id: config["image_pad_id"].as_u64().map(|v| v as u32),
            vision: config["vision"].is_object().then(|| {
                let v = &config["vision"];
                let n = |k: &str| v[k].as_u64().unwrap_or(0) as usize;
                Vision {
                    side: n("side"),
                    head_dim: n("head_dim"),
                    min_pixels: n("min_pixels"),
                    max_pixels: n("max_pixels"),
                }
            }),
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("the decider answers run, not {task}"));
        }
        let o = &req.body["options"];
        let text = o["request"]
            .as_str()
            .ok_or("run takes the `request` option: the request JSON, as text")?;
        let value = Json::parse(text).map_err(|e| format!("the request: {e:?}"))?;
        let request = Request::from_json(&value, self.codes.len()).map_err(|e| e.to_string())?;
        let mut images = Vec::new();
        for entry in &request.images {
            let v = self.vision.as_ref().ok_or(
                "this package has no vision tower (export it with the mmproj beside the GGUF)",
            )?;
            let img = RgbImage::open(entry, None).map_err(|e| format!("{e:#}"))?;
            images.push(
                vision::preprocess(&img, v.min_pixels, v.max_pixels)
                    .map_err(|e| format!("{e:#}"))?,
            );
        }
        let rows = match &self.vision {
            Some(v) => images
                .iter()
                .map(|img| self.vision_rows(v, img))
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        let t = o["temperature"].as_f64().unwrap_or(self.temperature);
        if !t.is_finite() || t <= 0.0 {
            return Err("temperature must be positive and finite".into());
        }
        let mut chunks = Vec::new();
        self.plan(&request, &images, &mut chunks)?;
        let mut by_id = std::collections::HashMap::new();
        let mut tokens = 0usize;
        for (chunk, enc) in &chunks {
            tokens += enc.ids.len();
            for (q, l) in chunk.questions.iter().zip(self.logits(chunk, enc, &rows)?) {
                by_id.insert(q.id.clone(), l);
            }
        }
        let answers = request
            .questions
            .iter()
            .map(|q| {
                let p = probabilities(&by_id[&q.id], t);
                (q.id.clone(), answer(q, &p))
            })
            .collect();
        let decision = Json::Object(vec![
            ("answers".into(), Json::Object(answers)),
            (
                "usage".into(),
                Json::Object(vec![
                    ("input_tokens".into(), Json::UInt(tokens as u64)),
                    ("output_tokens".into(), Json::UInt(0)),
                    ("sequences".into(), Json::UInt(chunks.len() as u64)),
                ]),
            ),
        ]);
        let bytes = decision.to_python_json(None).into_bytes();
        let n = bytes.len();
        Ok(Message::tensors(vec![Tensor::new(
            "decision",
            "u8",
            vec![n],
            bytes,
        )]))
    }
}

rlx_program::program!(Decider);
