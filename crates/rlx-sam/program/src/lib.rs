// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! SAM v1's host logic as an RLX program, answering `run`: what
//! `Sam::forward` does around the package's graphs, from the crate's own
//! files — the image resized, normalized and padded (`image.rs`), the
//! points and boxes embedded (`points.rs`) — then `encode`, `mask_embed`
//! for a mask prompt, and `decode`, the image embedding kept on the host
//! side between them.
//!
//! In: `image`, u8 `[h, w, 3]`; `points` f32 `[N, 2]` with `labels` f32
//! `[N]` (1 foreground, 0 background), `boxes` f32 `[M, 4]`, `mask` f32
//! `[256, 256]` — coordinates in the 1024-pixel input's. Option
//! `multimask` (default on). Out: `masks` f32 `[k, 256, 256]` logits,
//! `iou` f32 `[k]`, `size` i32 `[2]` (the resized image's height, width).

#[allow(dead_code)]
#[path = "../../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../../src/image.rs"]
mod image;
#[allow(dead_code)]
#[path = "../../src/points.rs"]
mod points;

use rlx_program::{Message, Program, Tensor, run_with};
use serde_json::Value;

struct Sam {
    embed_dim: usize,
    num_masks: usize,
    pe_gaussian: Vec<f32>,
    not_a_point_embed: Vec<f32>,
    point_embeddings: Vec<f32>,
    /// The IoU token, then the mask tokens.
    output_tokens: Vec<f32>,
}

fn floats(config: &Value, key: &str) -> Result<Vec<f32>, String> {
    serde_json::from_value(config[key].clone()).map_err(|e| format!("config {key}: {e}"))
}

impl Program for Sam {
    fn init(config: Value) -> Result<Self, String> {
        let n = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Sam {
            embed_dim: n("embed_dim")?,
            num_masks: n("num_mask_tokens")?,
            pe_gaussian: floats(&config, "pe_gaussian")?,
            not_a_point_embed: floats(&config, "not_a_point_embed")?,
            point_embeddings: floats(&config, "point_embeddings")?,
            output_tokens: floats(&config, "output_tokens")?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("SAM answers run, not {task}"));
        }
        let image = req
            .tensor("image")
            .ok_or("run takes an `image` tensor, u8 [h, w, 3]")?;
        if image.dtype != "u8" || image.shape.len() != 3 || image.shape[2] != 3 {
            return Err("`image` is u8 [h, w, 3]".into());
        }
        let (h, w) = (image.shape[0], image.shape[1]);
        let points = req.tensor("points").map(Tensor::to_f32);
        let labels = match (&points, req.tensor("labels")) {
            (Some(p), Some(l)) => Some(l.to_f32()).filter(|l| l.len() * 2 == p.len()),
            (Some(p), None) => Some(vec![1.0; p.len() / 2]),
            _ => None,
        };
        if points.is_some() && labels.is_none() {
            return Err("`labels` has one label per point".into());
        }
        let boxes = req.tensor("boxes").map(Tensor::to_f32);
        let weights = points::PointWeights {
            embed_dim: self.embed_dim,
            pe_gaussian: &self.pe_gaussian,
            not_a_point_embed: &self.not_a_point_embed,
            point_embeddings: &self.point_embeddings,
        };
        let pts = points.as_deref().zip(labels.as_deref());
        let sparse = points::sparse_embeddings(&weights, pts, boxes.as_deref())
            .map_err(|e| e.to_string())?;
        let mut tokens = self.output_tokens.clone();
        tokens.extend_from_slice(&sparse);
        let q = tokens.len() / self.embed_dim;

        let (nchw, (rh, rw)) = image::preprocess_image(&image.data, h, w);
        let side = config::SAM_IMG_SIZE;
        run_with(
            "encode",
            vec![Tensor::f32("image", vec![1, 3, side, side], &nchw)],
            &[],
            &[("embeddings", "embeddings")],
        )?;
        let mut from = vec![("embeddings", "embeddings")];
        let mut body = serde_json::json!({});
        if let Some(m) = req.tensor("mask") {
            let m = m.to_f32();
            let s = 4 * config::SAM_EMBED_HW;
            if m.len() != s * s {
                return Err(format!("`mask` is f32 [{s}, {s}]"));
            }
            run_with(
                "mask_embed",
                vec![Tensor::f32("mask", vec![1, 1, s, s], &m)],
                &[],
                &[("dense", "dense")],
            )?;
            from.push(("dense", "dense"));
        } else {
            body["weights"] = serde_json::json!({"dense": "no_mask_dense"});
        }
        body["from"] = from
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect::<serde_json::Map<_, _>>()
            .into();
        let outs = rlx_program::run_body(
            "decode",
            vec![Tensor::f32("tokens", vec![1, q, self.embed_dim], &tokens)],
            body,
        )?;
        let get = |n: &str| {
            outs.iter()
                .find(|t| t.name == n)
                .map(Tensor::to_f32)
                .ok_or(format!("decode returned no {n}"))
        };
        let (masks, iou) = (get("masks")?, get("iou")?);
        let nm = self.num_masks;
        let plane = masks.len() / nm;
        let s = (plane as f64).sqrt() as usize;
        let multimask = req.body["options"]["multimask"].as_bool().unwrap_or(true);
        let (masks, iou) = if multimask {
            (&masks[plane..], &iou[1..])
        } else {
            (&masks[..plane], &iou[..1])
        };
        let k = iou.len();
        Ok(Message::tensors(vec![
            Tensor::f32("masks", vec![k, s, s], masks),
            Tensor::f32("iou", vec![k], iou),
            Tensor::i32("size", vec![2], &[rh as i32, rw as i32]),
        ]))
    }
}

rlx_program::program!(Sam);
