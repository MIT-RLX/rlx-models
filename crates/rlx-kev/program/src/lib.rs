// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! kev's protocol as an RLX program, answering `run`: what
//! `DecisionModel::answer` does around the backbone, from the crate's own
//! files — the request read in order (`api.rs`, `json.rs`), encoded
//! (`encode.rs`, tokenized by the package's tokenizer), one causal row per
//! question shaped as the runner buckets it, the package's `rows` graph,
//! then the pointer head (`pointer.rs`) over the states it gathered.
//!
//! In: option `request`, the `/v1/systemone` body as text (so questions and
//! options keep their order). Out: `decision`, the response JSON as UTF-8
//! bytes, and `probabilities`, every question's distribution concatenated.

#[allow(dead_code)]
#[path = "../../src/api.rs"]
mod api;
#[allow(dead_code)]
#[path = "../../src/encode.rs"]
mod encode;
#[allow(dead_code)]
#[path = "../../src/json.rs"]
mod json;
#[allow(dead_code)]
#[path = "../../src/pointer.rs"]
mod pointer;

use std::collections::HashMap;

use anyhow::anyhow;
use api::{SystemOneRequest, answers_json, to_answers, to_record};
use encode::{EncodeOpts, Encoder, TokenizerLike, rows_of};
use json::Json;
use pointer::PointerHead;
use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

struct Host {
    tokenizer: String,
    specials: HashMap<String, u32>,
}

impl TokenizerLike for Host {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.specials.get(token).copied()
    }

    fn encode_no_special(&self, text: &str) -> anyhow::Result<Vec<u32>> {
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

struct Kev {
    encoder: Encoder<Host>,
    head: PointerHead,
    opts: EncodeOpts,
    bucket: usize,
    max_rows: usize,
    max_seq: usize,
}

impl Kev {
    /// `DecisionModel::resolve_shape` under its bucketed policy.
    fn shape(&self, rows: usize, longest: usize) -> Result<(usize, usize), String> {
        let r = rows.next_power_of_two().min(self.max_rows).max(rows);
        let s = longest.div_ceil(self.bucket) * self.bucket;
        if r > self.max_rows || s > self.max_seq {
            return Err(format!(
                "request needs {r}×{s} which exceeds the {}×{} limit ({rows} questions, \
                 {longest} tokens in the longest branch)",
                self.max_rows, self.max_seq
            ));
        }
        Ok((r, s.max(self.bucket)))
    }

    fn answer(&self, text: &str) -> anyhow::Result<(Json, Vec<f32>)> {
        let req = SystemOneRequest::parse(text)?;
        let (rec, meta) = to_record(&req)?;
        let enc = self.encoder.encode(&rec, self.opts)?;
        let (state_ids, _, rows) = rows_of(&enc)?;
        if rows.is_empty() {
            anyhow::bail!("record has no questions");
        }
        let ls = state_ids.len();
        let longest = rows.iter().map(|r| ls + r.ids.len()).max().unwrap_or(0);
        let (r, s) = self.shape(rows.len(), longest).map_err(|e| anyhow!(e))?;

        // Each question one row: the state, then its branch; the reads are
        // its `<decide>`, then its options' `</opt>`.
        let mut ids = vec![0f32; r * s];
        let mut reads = Vec::new();
        let mut spans = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            for (j, &t) in state_ids.iter().chain(&row.ids).enumerate() {
                ids[i * s + j] = t as f32;
            }
            let at = |p: usize| (i * s + ls + p) as i32;
            spans.push((reads.len(), row.opts.len()));
            reads.push(at(row.decide));
            reads.extend(row.opts.iter().map(|&o| at(o)));
        }
        let positions: Vec<i32> = (0..s as i32).collect();
        let outs = run(
            "rows",
            vec![
                Tensor::f32("input_ids", vec![r, s], &ids),
                Tensor::i32("positions", vec![s], &positions),
                Tensor::i32("reads", vec![reads.len()], &reads),
            ],
        )
        .map_err(|e| anyhow!(e))?;
        let states = outs
            .first()
            .ok_or_else(|| anyhow!("the rows graph returned nothing"))?
            .to_f32();
        let d = self.head.hidden;
        let mut probs = Vec::with_capacity(spans.len());
        for &(at, n) in &spans {
            let decide = &states[at * d..(at + 1) * d];
            let opts = &states[(at + 1) * d..(at + 1 + n) * d];
            probs.push(self.head.probs(decide, opts)?);
        }
        let answers = to_answers(&probs, &meta)?;
        let body = Json::Object(vec![
            ("model".into(), Json::Str(req.model.clone())),
            ("answers".into(), answers_json(&answers)),
            (
                "usage".into(),
                Json::Object(vec![
                    ("input_tokens".into(), Json::UInt(enc.ids.len() as u64)),
                    ("state_tokens".into(), Json::UInt(enc.state_len() as u64)),
                ]),
            ),
        ]);
        Ok((body, probs.concat()))
    }
}

impl Program for Kev {
    fn init(config: Value) -> Result<Self, String> {
        let n = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let specials: HashMap<String, u32> = serde_json::from_value(config["specials"].clone())
            .map_err(|e| format!("config specials: {e}"))?;
        let tok = Host {
            tokenizer: config["tokenizer"]
                .as_str()
                .ok_or("config has no tokenizer")?
                .to_string(),
            specials,
        };
        let encoder = Encoder::new(tok).map_err(|e| format!("{e:#}"))?;
        let file = config["head"].as_str().ok_or("config has no head")?;
        let bytes = host("file", Message::body(json!({"name": file})))?
            .tensor("bytes")
            .ok_or("the head file came back empty")?
            .data
            .clone();
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (hd, h) = (n("head_dim")?, n("hidden")?);
        if v.len() != 2 * (hd * h + hd) {
            return Err(format!(
                "the head file holds {} values, not {}",
                v.len(),
                2 * (hd * h + hd)
            ));
        }
        let (qw, rest) = v.split_at(hd * h);
        let (qb, rest) = rest.split_at(hd);
        let (kw, kb) = rest.split_at(hd * h);
        let head = PointerHead::new(
            qw.to_vec(),
            qb.to_vec(),
            kw.to_vec(),
            kb.to_vec(),
            h,
            hd,
            config["temperature"]
                .as_f64()
                .ok_or("config has no temperature")? as f32,
        )
        .map_err(|e| format!("{e:#}"))?;
        Ok(Kev {
            encoder,
            head,
            opts: EncodeOpts {
                max_state: n("max_state")?,
                max_branch: n("max_branch")?,
                strict: false,
                option_isolation: false,
            },
            bucket: n("shape_bucket")?,
            max_rows: n("max_rows")?,
            max_seq: n("max_seq")?,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("kev answers run, not {task}"));
        }
        let text = req.body["options"]["request"]
            .as_str()
            .ok_or("run takes the `request` option: the /v1/systemone body, as text")?;
        let (body, probs) = self.answer(text).map_err(|e| format!("{e:#}"))?;
        let bytes = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
        let n = bytes.len();
        Ok(Message::tensors(vec![
            Tensor::new("decision", "u8", vec![n], bytes),
            Tensor::f32("probabilities", vec![probs.len()], &probs),
        ]))
    }
}

rlx_program::program!(Kev);
