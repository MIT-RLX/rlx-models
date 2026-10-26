// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Qwen3.5's multimodal turn as an RLX program, answering `run`: what
//! `Qwen35Runner::generate_multimodal_with_opts` does, from the crate's own
//! files — each picture preprocessed (`vision/preprocess.rs`), through the
//! package's `vision` graph, the prompt assembled around it
//! (`vision/multimodal.rs`) — then the prompt and the reply through the
//! package's `step` ([`decode`]).
//!
//! In: pictures `u8 [h, w, 3]` (in name order); options `prompt` (one
//! `<__media__>` per picture), `max_tokens` (default 256) and the sampling
//! options. Out: `text`, UTF-8 bytes.

mod decode;
mod tower;

/// The mmproj settings `preprocess.rs` reads.
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

/// The vision tower's output, as `multimodal.rs` takes it.
mod encoder {
    pub struct VisionEncodeOutput {
        pub embeddings: Vec<f32>,
        pub grid_x: usize,
        pub grid_y: usize,
        pub n_tokens: usize,
    }
}

#[allow(dead_code)]
#[path = "../../src/vision/preprocess.rs"]
mod preprocess;
#[allow(dead_code)]
#[path = "../../src/vision/multimodal.rs"]
mod multimodal;

use rlx_program::{Message, Program, Tensor};
use serde_json::Value;

struct Qwen35 {
    decoder: decode::Decoder,
    tower: tower::Tower,
    stop: Vec<u32>,
}

impl Program for Qwen35 {
    fn init(config: Value) -> Result<Self, String> {
        Ok(Qwen35 {
            decoder: decode::Decoder::from_config(&config)?,
            tower: tower::Tower::from_config(&config["vision"])?,
            stop: serde_json::from_value(config["stop_ids"].clone()).map_err(|e| format!("config stop_ids: {e}"))?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("Qwen3.5 answers run, not {task}"));
        }
        let o = &req.body["options"];
        let prompt = o["prompt"].as_str().ok_or("run takes the `prompt` option")?;
        let max_new = o["max_tokens"].as_u64().unwrap_or(256) as usize;
        let mut tensors: Vec<&Tensor> = req.tensors.iter().collect();
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let text = (|| -> anyhow::Result<String> {
            let mut images = Vec::new();
            for t in tensors {
                anyhow::ensure!(
                    t.dtype == "u8" && t.shape.len() == 3 && t.shape[2] == 3,
                    "`{}` is not a u8 [h, w, 3] picture",
                    t.name
                );
                images.push(self.tower.encode(&t.data, t.shape[1], t.shape[0], self.decoder.hidden)?);
            }
            let h = self.decoder.hidden;
            let (prefill, is_image) = multimodal::MultiImagePrompt {
                prompt,
                images: &images,
            }
            .assemble_with_mask(|s| self.decoder.encode(s), |ids| Ok(vec![0.0; ids.len() * h]), h, 0)?;
            let rows: Vec<Option<&[f32]>> = is_image
                .iter()
                .enumerate()
                .map(|(i, &img)| img.then(|| &prefill.hidden[i * h..(i + 1) * h]))
                .collect();
            let ids = self
                .decoder
                .generate(&prefill.seq, &rows, &prefill.mrope_sections, max_new, &self.stop, o)?;
            self.decoder.decode(&ids)
        })()
        .map_err(|e| format!("{e:#}"))?;
        let bytes = text.into_bytes();
        let n = bytes.len();
        Ok(Message::tensors(vec![Tensor::new("text", "u8", vec![n], bytes)]))
    }
}

rlx_program::program!(Qwen35);
