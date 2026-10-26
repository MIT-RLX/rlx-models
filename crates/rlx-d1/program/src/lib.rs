// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! d1's protocol as an RLX program, answering `run`: what `D1::system_one`
//! does for a text state, from the crate's own `prompt.rs` — the questions
//! validated, the prompt rendered and tokenized (by the package's
//! tokenizer, no special tokens added), one pass per question through the
//! package's `pass_{bucket}` graph, each answer's distribution.
//!
//! Pictures (`images`: `data:` URLs) are capped and processed as the
//! runner does (`processor.rs`), each tile through the package's `tower`
//! (its position grid resampled by SigLIP2's `pos_grid.rs`) and
//! `projector`, and their features placed over the image tokens.
//!
//! In: option `request`, `{"state": …, "questions": {name: question},
//! "images": […]}` as text (so its questions keep their order). Out:
//! `decision`, the `{"answers": …, "usage": …}` JSON as UTF-8 bytes.

#[allow(dead_code)]
#[path = "../../../rlx-siglip2/src/pos_grid.rs"]
mod pos_grid;
#[allow(dead_code)]
#[path = "../../src/processor.rs"]
mod processor;
#[allow(dead_code)]
#[path = "../../src/prompt.rs"]
mod prompt;

use processor::{IMAGE_TOKEN, ProcessedImage, ProcessorConfig, RgbImage, Tile};

use anyhow::anyhow;
use prompt::{
    PromptStyle, Question, TextTokenizer, answer, prefix_text, readout, readout_ids, suffix_text,
};
use rlx_core::ordered_json::Json;
use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

struct Host {
    tokenizer: String,
}

impl TextTokenizer for Host {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer, "text": text, "special": false})),
        )
        .map_err(|e| anyhow!(e))?;
        Ok(m.body["ids"]
            .as_array()
            .ok_or_else(|| anyhow!("tokenizer.encode returned no ids"))?
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as u32)
            .collect())
    }
}

struct D1 {
    tok: Host,
    style: PromptStyle,
    buckets: Vec<usize>,
    hidden: usize,
    image_token_id: u32,
    vision: Vision,
}

/// The vision half's settings (the package's `tower` and `projector`).
struct Vision {
    proc: ProcessorConfig,
    hidden: usize,
    patch: usize,
    pos_side: usize,
    seq: usize,
    proj_rows: usize,
    /// The learned `[pos_side², hidden]` position grid.
    grid: Vec<f32>,
}

/// `data:image/…;base64,…` to its bytes.
fn data_url(entry: &str) -> Result<Vec<u8>, String> {
    let payload = entry
        .strip_prefix("data:image/")
        .and_then(|r| r.split_once(','))
        .map(|(_, p)| p)
        .ok_or("images are `data:image/…;base64,` URLs (a package reads no paths)")?;
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let bytes = payload.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c).ok_or("a data URL that is not base64")? << (18 - 6 * i);
        }
        for i in 0..chunk.len().saturating_sub(1) {
            out.push((n >> (16 - 8 * i)) as u8);
        }
    }
    Ok(out)
}

impl D1 {
    /// `VisionEncoder::encode_tile`: one tile's projected features.
    fn encode_tile(&self, tile: &Tile) -> Result<Vec<f32>, String> {
        let v = &self.vision;
        let pd = 3 * v.patch * v.patch;
        let n = tile.ph * tile.pw;
        let ds = v.proc.downsample_factor;
        if n > v.seq {
            return Err(format!(
                "tile of {n} patches exceeds the {}-patch tower",
                v.seq
            ));
        }
        let mut pv = vec![0f32; v.seq * pd];
        pv[..n * pd].copy_from_slice(&tile.pixel_values);
        // SigLIP2 pads past the real patches with the resized grid's first row.
        let pos = pos_grid::resize_position_grid(&v.grid, v.pos_side, v.hidden, tile.ph, tile.pw);
        let mut pos_embed = vec![0f32; v.seq * v.hidden];
        for p in 0..v.seq {
            let src = if p < n { p } else { 0 };
            pos_embed[p * v.hidden..(p + 1) * v.hidden]
                .copy_from_slice(&pos[src * v.hidden..(src + 1) * v.hidden]);
        }
        let mut mask = vec![0f32; v.seq];
        mask[..n].fill(1.0);
        let feats = run(
            "tower",
            vec![
                Tensor::f32("pixel_values", vec![1, v.seq, pd], &pv),
                Tensor::f32("pos_embed", vec![1, v.seq, v.hidden], &pos_embed),
                Tensor::f32("key_mask", vec![1, v.seq], &mask),
            ],
        )?;
        let feats = feats.first().ok_or("the tower returned nothing")?.to_f32();
        let unshuffled =
            processor::pixel_unshuffle(&feats[..n * v.hidden], tile.ph, tile.pw, v.hidden, ds);
        let rows = n / (ds * ds);
        let width = v.hidden * ds * ds;
        let mut padded = vec![0f32; v.proj_rows * width];
        padded[..rows * width].copy_from_slice(&unshuffled);
        let proj = run(
            "projector",
            vec![Tensor::f32(
                "image_features",
                vec![1, v.proj_rows, width],
                &padded,
            )],
        )?;
        let proj = proj
            .first()
            .ok_or("the projector returned nothing")?
            .to_f32();
        Ok(proj[..rows * self.hidden].to_vec())
    }

    /// `D1::expand_and_encode`: each `<image>` expanded to its picture's
    /// token layout, then tokenized as one string.
    fn expand_and_encode(&self, text: &str, pics: &[ProcessedImage]) -> anyhow::Result<Vec<u32>> {
        let parts: Vec<&str> = text.split(IMAGE_TOKEN).collect();
        anyhow::ensure!(
            parts.len() == pics.len() + 1,
            "the prompt's `<image>`s are not its pictures"
        );
        let mut expanded = String::from(parts[0]);
        for (p, rest) in pics.iter().zip(&parts[1..]) {
            expanded.push_str(&p.placeholder(&self.vision.proc));
            expanded.push_str(rest);
        }
        self.tok.encode(&expanded)
    }

    /// `D1::read_row`: one causal pass over `ids` (its image tokens taking
    /// `features` in order), scored at the last position against the
    /// options' tokens of the tied head.
    fn read_row(
        &self,
        ids: &[u32],
        features: &[f32],
        groups: &[Vec<u32>],
    ) -> Result<Vec<f64>, String> {
        let n = ids.len();
        if n == 0 {
            return Err("an empty prompt".into());
        }
        let seq = n.max(64).next_power_of_two();
        if !self.buckets.contains(&seq) {
            return Err(format!(
                "{n} tokens: the package's passes go to {} tokens",
                self.buckets.last().copied().unwrap_or(0)
            ));
        }
        let mut options: Vec<u32> = Vec::new();
        for &id in groups.iter().flatten() {
            if !options.contains(&id) {
                options.push(id);
            }
        }
        let mut padded = vec![0i32; seq];
        let mut rows: Vec<i32> = (0..seq as i32).collect();
        let mut k = 0;
        for (i, &t) in ids.iter().enumerate() {
            padded[i] = t as i32;
            if t == self.image_token_id {
                rows[i] = (seq + k) as i32;
                k += 1;
            }
        }
        if k * self.hidden != features.len() {
            return Err(format!(
                "{k} image tokens for {} image features",
                features.len() / self.hidden
            ));
        }
        let image_rows = if features.is_empty() {
            vec![0f32; self.hidden]
        } else {
            features.to_vec()
        };
        let opts: Vec<i32> = options.iter().map(|&t| t as i32).collect();
        let outs = run(
            &format!("pass_{seq}"),
            vec![
                Tensor::i32("ids", vec![seq], &padded),
                Tensor::i32("rows", vec![seq], &rows),
                Tensor::f32(
                    "image_rows",
                    vec![image_rows.len() / self.hidden, self.hidden],
                    &image_rows,
                ),
                Tensor::i32("last", vec![1], &[n as i32 - 1]),
                Tensor::i32("options", vec![opts.len()], &opts),
            ],
        )?;
        let scores = outs.first().ok_or("the pass returned nothing")?.to_f32();
        Ok(readout(groups, |id| {
            let i = options
                .iter()
                .position(|&o| o == id)
                .expect("every option scored");
            f64::from(scores[i])
        }))
    }

    /// `D1::decide`: `(probabilities, input tokens)`.
    fn decide(
        &self,
        state: &Json,
        questions: &[Question],
        images: &[RgbImage],
    ) -> anyhow::Result<(Vec<Vec<f64>>, usize)> {
        let groups: Vec<Vec<Vec<u32>>> = questions
            .iter()
            .map(|q| readout_ids(&self.tok, q))
            .collect::<anyhow::Result<_>>()?;
        let suffixes: Vec<String> = questions
            .iter()
            .map(|q| suffix_text(&self.tok, q, &self.style))
            .collect::<anyhow::Result<_>>()?;
        let pics: Vec<ProcessedImage> = images
            .iter()
            .map(|im| {
                processor::process_image(
                    &processor::cap_pixels(im, processor::VISION_MAX_PIXELS),
                    &self.vision.proc,
                )
            })
            .collect::<anyhow::Result<_>>()?;
        // What the chat template writes for each picture at the head of the turn.
        let prefix = prefix_text(state, &self.style, &IMAGE_TOKEN.repeat(pics.len()));
        let mut features = Vec::new();
        for p in &pics {
            for t in &p.tiles {
                features.extend(self.encode_tile(t).map_err(|e| anyhow!(e))?);
            }
        }
        let encode = |text: &str| -> anyhow::Result<Vec<u32>> {
            if pics.is_empty() {
                self.tok.encode(text)
            } else {
                self.expand_and_encode(text, &pics)
            }
        };
        if questions.len() == 1 {
            let ids = encode(&(prefix + &suffixes[0]))?;
            let p = self
                .read_row(&ids, &features, &groups[0])
                .map_err(|e| anyhow!(e))?;
            return Ok((vec![p], ids.len()));
        }
        // The state tokenized once, each question on its own: d1's tree.
        let trunk = encode(&prefix)?;
        let mut tokens = trunk.len();
        let mut out = Vec::with_capacity(questions.len());
        for (suffix, g) in suffixes.iter().zip(&groups) {
            let ids = self.tok.encode(suffix)?;
            tokens += ids.len();
            let mut row = trunk.clone();
            row.extend(ids);
            out.push(self.read_row(&row, &features, g).map_err(|e| anyhow!(e))?);
        }
        Ok((out, tokens))
    }
}

impl Program for D1 {
    fn init(config: Value) -> Result<Self, String> {
        Ok(D1 {
            tok: Host {
                tokenizer: config["tokenizer"]
                    .as_str()
                    .ok_or("config has no tokenizer")?
                    .to_string(),
            },
            style: PromptStyle {
                bos: config["bos"].as_str().unwrap_or_default().to_string(),
                ..PromptStyle::default()
            },
            buckets: serde_json::from_value(config["buckets"].clone())
                .map_err(|e| format!("config buckets: {e}"))?,
            hidden: config["hidden"].as_u64().ok_or("config has no hidden")? as usize,
            image_token_id: config["image_token_id"]
                .as_u64()
                .ok_or("config has no image_token_id")? as u32,
            vision: {
                let v = &config["vision"];
                let n = |k: &str| {
                    v[k].as_u64()
                        .map(|x| x as usize)
                        .ok_or(format!("config vision has no {k}"))
                };
                let p = &v["processor"];
                let pn = |k: &str| {
                    p[k].as_u64()
                        .map(|x| x as usize)
                        .ok_or(format!("config processor has no {k}"))
                };
                let pb = |k: &str| p[k].as_bool().ok_or(format!("config processor has no {k}"));
                let proc = ProcessorConfig {
                    downsample_factor: pn("downsample_factor")?,
                    do_image_splitting: pb("do_image_splitting")?,
                    min_tiles: pn("min_tiles")?,
                    max_tiles: pn("max_tiles")?,
                    use_thumbnail: pb("use_thumbnail")?,
                    min_image_tokens: pn("min_image_tokens")?,
                    max_image_tokens: pn("max_image_tokens")?,
                    encoder_patch_size: pn("encoder_patch_size")?,
                    tile_size: pn("tile_size")?,
                    max_pixels_tolerance: p["max_pixels_tolerance"]
                        .as_f64()
                        .ok_or("config processor has no max_pixels_tolerance")?,
                    use_image_special_tokens: pb("use_image_special_tokens")?,
                };
                let file = config["position_grid"]
                    .as_str()
                    .ok_or("config has no position_grid")?;
                let bytes = host("file", Message::body(json!({"name": file})))?
                    .tensor("bytes")
                    .ok_or("the position grid came back empty")?
                    .data
                    .clone();
                let grid = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Vision {
                    proc,
                    hidden: n("hidden")?,
                    patch: n("patch")?,
                    pos_side: n("pos_side")?,
                    seq: n("seq")?,
                    proj_rows: n("proj_rows")?,
                    grid,
                }
            },
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("d1 answers run, not {task}"));
        }
        let text = req.body["options"]["request"]
            .as_str()
            .ok_or("run takes the `request` option: {\"state\": …, \"questions\": {…}} as text")?;
        let value = Json::parse(text).map_err(|e| format!("the request: {e:?}"))?;
        let state = value.get("state").ok_or("the request has no state")?;
        let named = value
            .get("questions")
            .and_then(Json::as_object)
            .ok_or("questions must be an object of name -> question")?;
        let qs: Vec<Question> = named
            .iter()
            .map(|(n, q)| Question::from_json(q).map_err(|e| format!("question `{n}`: {e:#}")))
            .collect::<Result<_, _>>()?;
        if qs.is_empty() {
            return Err("no questions".into());
        }
        let mut images = Vec::new();
        if let Some(list) = value.get("images").and_then(Json::as_array) {
            for entry in list {
                let bytes = data_url(entry.as_str().ok_or("an image entry is a string")?)?;
                let rgb = image::load_from_memory(&bytes)
                    .map_err(|e| format!("decoding an image: {e}"))?
                    .to_rgb8();
                let (w, h) = rgb.dimensions();
                images.push(
                    RgbImage::new(w as usize, h as usize, rgb.into_raw())
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        let (probs, tokens) = self
            .decide(state, &qs, &images)
            .map_err(|e| format!("{e:#}"))?;
        let answers = named
            .iter()
            .zip(&qs)
            .zip(&probs)
            .map(|(((n, _), q), p)| (n.clone(), answer(q, p)))
            .collect();
        let decision = Json::Object(vec![
            ("answers".into(), Json::Object(answers)),
            (
                "usage".into(),
                Json::Object(vec![
                    ("input_tokens".into(), Json::UInt(tokens as u64)),
                    ("output_tokens".into(), Json::UInt(0)),
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

rlx_program::program!(D1);
