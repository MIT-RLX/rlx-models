// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! A Qwen3.5 trunk's prompt and reply through a package's stateful `step`
//! (`rlx_qwen35::export_vl::host_embed_step`): the states zeroed in the
//! runtime's slots, each prompt token (or image row) stepped at its M-RoPE
//! positions with the caches kept there, then tokens sampled and stepped
//! until a stop id. The programs of the packages built on this trunk
//! include this file.

use rlx_program::{Message, Tensor, host, run_with};
use serde_json::{Value, json};

/// One state of the step: its input, and its dims (`-1`: the capacity).
pub struct State {
    pub name: String,
    pub dims: Vec<i64>,
}

/// The step's settings, from a program's config.
pub struct Decoder {
    pub tokenizer: String,
    pub hidden: usize,
    pub context: usize,
    pub states: Vec<State>,
}

fn err(e: String) -> anyhow::Error {
    anyhow::anyhow!(e)
}

impl Decoder {
    /// From a config's `tokenizer`, `hidden`, `context` and `states`.
    pub fn from_config(config: &Value) -> Result<Self, String> {
        let n = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        Ok(Decoder {
            tokenizer: config["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .to_string(),
            hidden: n("hidden")?,
            context: n("context")?,
            states: config["states"]
                .as_array()
                .ok_or("config has no states")?
                .iter()
                .map(|s| State {
                    name: s["name"].as_str().unwrap_or_default().to_string(),
                    dims: serde_json::from_value(s["dims"].clone()).unwrap_or_default(),
                })
                .collect(),
        })
    }

    /// The text's ids, no special tokens added.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": self.tokenizer, "text": text, "special": false})),
        )
        .map_err(err)?;
        Ok(m.body["ids"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tokenizer.encode returned no ids"))?
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as u32)
            .collect())
    }

    /// The ids' text, special tokens skipped.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        let m = host(
            "tokenizer.decode",
            Message::body(json!({"tokenizer": self.tokenizer, "ids": ids})),
        )
        .map_err(err)?;
        Ok(m.body["text"].as_str().unwrap_or_default().to_string())
    }

    /// Zeroed states at capacity `cap`, in their slots.
    fn reset(&self, cap: usize) -> anyhow::Result<()> {
        for st in &self.states {
            let dims: Vec<usize> = st
                .dims
                .iter()
                .map(|&d| if d < 0 { cap } else { d as usize })
                .collect();
            let n: usize = dims.iter().product();
            host(
                "slot.set",
                Message::new(
                    json!({"slot": st.name}),
                    vec![Tensor::f32(st.name.as_str(), dims, &vec![0.0; n])],
                ),
            )
            .map_err(err)?;
        }
        Ok(())
    }

    /// One step at cache row `row`: a token, or an image row, at `pos`. The
    /// logits come back with `logits`, else stay in their slot.
    fn step(
        &self,
        token: u32,
        image: Option<&[f32]>,
        pos: [usize; 4],
        row: usize,
        logits: bool,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        let zero = vec![0f32; self.hidden];
        let p: Vec<i32> = pos.iter().map(|&v| v as i32).collect();
        let inputs = vec![
            Tensor::i32("token", vec![1], &[token as i32]),
            Tensor::f32("image_row", vec![1, self.hidden], image.unwrap_or(&zero)),
            Tensor::f32(
                "use_image",
                vec![1],
                &[if image.is_some() { 1.0 } else { 0.0 }],
            ),
            Tensor::i32("positions", vec![4], &p),
            Tensor::f32("input_ids", vec![1, 1], &[token as f32]),
            Tensor::f32("position", vec![1], &[row as f32]),
        ];
        let from: Vec<(&str, &str)> = self
            .states
            .iter()
            .map(|s| (s.name.as_str(), s.name.as_str()))
            .collect();
        let nexts: Vec<String> = self
            .states
            .iter()
            .map(|s| format!("{}.next", s.name))
            .collect();
        let mut keep: Vec<(&str, &str)> = nexts
            .iter()
            .zip(&self.states)
            .map(|(n, s)| (n.as_str(), s.name.as_str()))
            .collect();
        if !logits {
            keep.push(("logits", "logits"));
        }
        let outs = run_with("step", inputs, &from, &keep).map_err(err)?;
        if !logits {
            return Ok(None);
        }
        Ok(Some(
            outs.iter()
                .find(|t| t.name == "logits")
                .ok_or_else(|| anyhow::anyhow!("the step returned no logits"))?
                .to_f32(),
        ))
    }

    /// The next id from `logits`, by the request's sampling options.
    fn sample(&self, logits: &[f32], options: &Value, step: usize) -> anyhow::Result<u32> {
        let mut body = if options.is_object() {
            options.clone()
        } else {
            json!({})
        };
        body["step"] = json!(step);
        let m = host(
            "sample",
            Message::new(
                body,
                vec![Tensor::f32("logits", vec![logits.len()], logits)],
            ),
        )
        .map_err(err)?;
        Ok(m.body["id"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("sample returned no id"))? as u32)
    }

    /// The reply's ids to a prompt of `ids` (`rows[i]`: an image row in
    /// place of token `i`'s embedding) at `positions`: up to `max_new`,
    /// ending before the first of `stop` (when `stop` holds any). Positions
    /// past the prompt go on from its last token's time position, one per
    /// token (the runner's `mrope_decode_rope_at_past`).
    pub fn generate(
        &self,
        ids: &[u32],
        rows: &[Option<&[f32]>],
        positions: &[[usize; 4]],
        max_new: usize,
        stop: &[u32],
        options: &Value,
    ) -> anyhow::Result<Vec<u32>> {
        let len = ids.len();
        anyhow::ensure!(len > 0, "an empty prompt");
        anyhow::ensure!(
            len + max_new < self.context,
            "{len} prompt tokens and {max_new} more: past the {}-token context",
            self.context
        );
        // The capacity rounded up, so a few compiled shapes serve every turn.
        let cap = ((len + max_new + 1).div_ceil(256) * 256).min(self.context + 1);
        self.reset(cap)?;
        let mut logits = None;
        for i in 0..len {
            logits = self.step(ids[i], rows[i], positions[i], i, i + 1 == len)?;
        }
        let mut logits = logits.expect("the last prompt step returns its logits");
        let last_t = positions[len - 1][0];
        let mut out = Vec::new();
        for i in 0..max_new {
            let tok = self.sample(&logits, options, i)?;
            if stop.contains(&tok) {
                break;
            }
            out.push(tok);
            if i + 1 == max_new {
                break;
            }
            let past = len + i;
            let t = last_t + (past - len + 1);
            logits = self
                .step(tok, None, [t, t, t, 0], past, true)?
                .expect("asked for logits");
        }
        Ok(out)
    }
}
