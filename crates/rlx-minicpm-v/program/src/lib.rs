// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! MiniCPM-V's chat turn as an RLX program, answering `run`: what
//! `MiniCpmVRunner::chat` does, from the crate's own files — the pictures
//! sliced (`preprocess.rs`), the prompt and placeholders (`prompt.rs`),
//! each crop's feeds (`feeds.rs`) through the package's `vision_{bucket}`
//! graph, the canvas positions (`positions.rs`) — then the prompt through
//! the package's `step` one token at a time and the reply decoded, the
//! caches staying in the runtime's slots.
//!
//! In: pictures `u8 [h, w, 3]` (in name order); options `prompt`,
//! `max_tokens` (default 256), `thinking`. Out: `text`, UTF-8 bytes.

#[allow(dead_code)]
#[path = "../../src/config.rs"]
mod config;
/// The Qwen3.5 trunk's decode loop, shared with the other programs on it.
#[path = "../../../rlx-qwen35/program/src/decode.rs"]
mod decode;
#[allow(dead_code)]
#[path = "../../src/feeds.rs"]
mod feeds;
#[allow(dead_code)]
#[path = "../../src/positions.rs"]
mod positions;
#[allow(dead_code)]
#[path = "../../src/preprocess.rs"]
mod preprocess;
#[allow(dead_code)]
#[path = "../../src/prompt.rs"]
mod prompt;

use config::{MiniCpmVConfig, MropeMode};
use preprocess::SlicedImage;
use prompt::Media;
use rlx_program::{Message, Program, Tensor, run};
use serde_json::Value;

struct MiniCpmV {
    cfg: MiniCpmVConfig,
    decoder: decode::Decoder,
    eos: Vec<u32>,
    hidden: usize,
    buckets: Vec<usize>,
}

fn host_err(e: String) -> anyhow::Error {
    anyhow::anyhow!(e)
}

impl MiniCpmV {
    /// `MiniCpmVRunner::tokens_for`: tokens per crop of `grid` patches.
    fn tokens_for(&self, grid: (usize, usize)) -> usize {
        let f = self.cfg.downsample.merge_factor();
        grid.0 * grid.1 / (f * f)
    }

    /// `MiniCpmVRunner::layout`: `(thumb tokens, slice tokens, grid)`.
    fn layout(&self, im: &SlicedImage) -> (usize, usize, Option<(usize, usize)>) {
        let th = self.tokens_for((im.thumbnail().grid_h, im.thumbnail().grid_w));
        let sl = im
            .slices()
            .first()
            .map_or(0, |c| self.tokens_for((c.grid_h, c.grid_w)));
        (th, sl, im.grid)
    }

    /// `MiniCpmVRunner::encode_crop`: one crop's LM rows.
    fn encode_crop(&self, c: &preprocess::Crop) -> anyhow::Result<Vec<f32>> {
        let n = c.num_patches();
        let bucket = feeds::bucket_for(n);
        if !self.buckets.contains(&bucket) {
            anyhow::bail!(
                "a crop of {n} patches: the package's vision graphs go to {:?}",
                self.buckets
            );
        }
        let f = feeds::crop_feeds(&self.cfg, c.grid_h, c.grid_w, &c.patches, bucket)?;
        let dim = 3 * self.cfg.vision.patch_size * self.cfg.vision.patch_size;
        let inputs = f
            .as_inputs()
            .into_iter()
            .map(|(name, data)| {
                let shape = if name == "patches" {
                    vec![1, bucket, dim]
                } else {
                    vec![1, data.len()]
                };
                Tensor::f32(name, shape, data)
            })
            .collect();
        let out = run(&format!("vision_{bucket}"), inputs).map_err(host_err)?;
        let rows = out
            .first()
            .ok_or_else(|| anyhow::anyhow!("the vision graph returned nothing"))?
            .to_f32();
        anyhow::ensure!(
            rows.len() >= f.n_tokens * self.hidden,
            "the vision graph returned too few rows"
        );
        Ok(rows[..f.n_tokens * self.hidden].to_vec())
    }

    fn chat(
        &self,
        images: &[SlicedImage],
        question: &str,
        options: &Value,
    ) -> anyhow::Result<String> {
        let max_new = options["max_tokens"].as_u64().unwrap_or(256) as usize;
        let thinking = options["thinking"].as_bool().unwrap_or(false);
        let media: Vec<Media> = images.iter().map(|_| Media::Image).collect();
        let text = prompt::chat_prompt_media(question, &media, thinking);
        // `MiniCpmVRunner::assemble_media`.
        let use_id = self.cfg.slice.use_image_id;
        let placeholders: Vec<(Media, String)> = images
            .iter()
            .enumerate()
            .map(|(i, im)| {
                let (th, sl, grid) = self.layout(im);
                (
                    Media::Image,
                    prompt::image_placeholder(use_id.then_some(i), th, sl, grid),
                )
            })
            .collect();
        let text = prompt::expand_media(&text, &placeholders)?;
        let ids = self.decoder.encode(&text)?;
        let modality: Vec<u8> = ids
            .iter()
            .map(|&t| u8::from(t == self.cfg.image_token_id))
            .collect();
        let crops: Vec<&preprocess::Crop> = images.iter().flat_map(|im| im.crops.iter()).collect();
        let grids: Vec<(usize, usize)> = crops.iter().map(|c| (c.grid_h, c.grid_w)).collect();
        let mut features = Vec::new();
        for c in &crops {
            features.extend(self.encode_crop(c)?);
        }
        let positions = match self.cfg.mrope {
            MropeMode::Linear => positions::linear_positions(ids.len()),
            MropeMode::Canvas => {
                let markers = self.cfg.markers.ok_or_else(|| {
                    anyhow::anyhow!("canvas M-RoPE needs the marker ids in config.json")
                })?;
                positions::canvas_positions(
                    &ids,
                    &modality,
                    &grids,
                    self.cfg.downsample.merge_factor(),
                    &markers,
                )?
            }
        };
        let h = self.hidden;
        let mut k = 0;
        let rows: Vec<Option<&[f32]>> = modality
            .iter()
            .map(|&m| {
                (m == 1).then(|| {
                    let r = &features[k * h..(k + 1) * h];
                    k += 1;
                    r
                })
            })
            .collect();
        anyhow::ensure!(
            k * h == features.len(),
            "{k} image tokens for {} image rows",
            features.len() / h
        );
        let out = self
            .decoder
            .generate(&ids, &rows, &positions, max_new, &self.eos, options)?;
        self.decoder.decode(&out)
    }
}

impl Program for MiniCpmV {
    fn init(config: Value) -> Result<Self, String> {
        let pp = config["preprocessor"].clone();
        let cfg = MiniCpmVConfig::from_values(&config["config"], (!pp.is_null()).then_some(&pp))
            .map_err(|e| format!("{e:#}"))?;
        let n = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        Ok(MiniCpmV {
            cfg,
            decoder: decode::Decoder::from_config(&config)?,
            eos: serde_json::from_value(config["eos"].clone())
                .map_err(|e| format!("config eos: {e}"))?,
            hidden: n("hidden")?,
            buckets: serde_json::from_value(config["buckets"].clone())
                .map_err(|e| format!("config buckets: {e}"))?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("MiniCPM-V answers run, not {task}"));
        }
        let o = &req.body["options"];
        let question = o["prompt"]
            .as_str()
            .ok_or("run takes the `prompt` option")?;
        let mut tensors: Vec<&Tensor> = req.tensors.iter().collect();
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let mut images = Vec::new();
        for t in tensors {
            if t.dtype != "u8" || t.shape.len() != 3 || t.shape[2] != 3 {
                return Err(format!("`{}` is not a u8 [h, w, 3] picture", t.name));
            }
            let (h, w) = (t.shape[0], t.shape[1]);
            images.push(
                preprocess::slice_image(&t.data, w, h, &self.cfg.slice, self.cfg.vision.patch_size)
                    .map_err(|e| format!("{e:#}"))?,
            );
        }
        let text = self
            .chat(&images, question, o)
            .map_err(|e| format!("{e:#}"))?;
        let bytes = text.into_bytes();
        let n = bytes.len();
        Ok(Message::tensors(vec![Tensor::new(
            "text",
            "u8",
            vec![n],
            bytes,
        )]))
    }
}

rlx_program::program!(MiniCpmV);
