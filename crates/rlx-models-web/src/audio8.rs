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

//! Audio8-TTS-Realistic-Air (`rlx-fish`) across Web Workers. `web/audio8.js`
//! drives them:
//!
//! - **slow stages** ([`Audio8Slow`]): a run of the 24 slow layers each; the
//!   first embeds the rows, the last ends in the final norm and the band
//!   head;
//! - **frames** ([`Audio8Frames`]): the fast decoder and the sampler — the
//!   last stage's hidden state and band logits in, the next frame's row out;
//! - **codec** ([`Audio8Codec`]): codes to 44.1 kHz speech, and a reference
//!   clip to codes for voice cloning.
//!
//! Each fetches only its own tensors with HTTP `Range` requests
//! ([`Audio8Fetch`]), into one buffer that is already a valid safetensors file
//! of them. The pipeline draws the frames `S2Model::generate` draws from the
//! same seed (`rlx_fish::audio8` tests the split against it).

use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_fish::audio8::{FrameMaker, SlowStage, StageInput};
use rlx_fish::config::S2Config;
use rlx_fish::s2::{Reference, S2Model, S2Options, Stage};
use rlx_fish::sampling::SampleParams;
use rlx_fish::{FishCodec, codec::CodecWeights};
use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use crate::{Files, js_err, parse_backend};

fn config(config_json: &str) -> Result<S2Config, JsError> {
    let v: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| JsError::new(&format!("config.json: {e}")))?;
    S2Config::from_value(&v).map_err(js_err)
}

/// The slow layers of stage `index` of `count`, split as evenly as they go.
fn stage_layers(
    cfg: &S2Config,
    index: usize,
    count: usize,
) -> Result<std::ops::Range<usize>, JsError> {
    let n = cfg.slow.n_layer;
    if count == 0 || index >= count || count > n {
        return Err(JsError::new(&format!(
            "stage {index} of {count}: the model has {n} slow layers"
        )));
    }
    Ok(index * n / count..(index + 1) * n / count)
}

/// One tensor of the source file: name, dtype, shape, absolute byte range.
struct Entry {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    start: u64,
    end: u64,
    /// Where its bytes go in [`Audio8Fetch::buffer`].
    at: usize,
}

/// The tensors of one Audio8 worker, read from `model.safetensors` (or
/// `codec_*.safetensors`) by HTTP `Range`:
///
/// ```js
/// const f = new Audio8Fetch(prefix, config, "slow", index, count); // prefix: the file's first bytes
/// const r = f.ranges();                                            // [start, end, start, end, …]
/// for (let i = 0; i < r.length; i += 2) f.add(r[i], await fetchRange(url, r[i], r[i + 1]));
/// const stage = Audio8Slow.from_fetch(f, config, index, count, "cpu");
/// ```
///
/// `role` is `"slow"` (stage `index` of `count`), `"frames"` (the fast
/// decoder) or `"all"` (every tensor — the codec's files). The constructor
/// fails if `prefix` ends inside the header; fetch more and retry.
#[wasm_bindgen]
pub struct Audio8Fetch {
    entries: Vec<Entry>,
    /// `[8-byte length | header | data]`: the subset as a safetensors file,
    /// filled in place as pieces arrive.
    buffer: Option<Vec<u8>>,
    ranges: Vec<f64>,
    added: usize,
}

#[wasm_bindgen]
impl Audio8Fetch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        prefix: &[u8],
        config_json: &str,
        role: &str,
        index: usize,
        count: usize,
    ) -> Result<Audio8Fetch, JsError> {
        let header_err = || JsError::new("safetensors header: prefix too short");
        let n = u64::from_le_bytes(prefix.get(..8).ok_or_else(header_err)?.try_into().unwrap());
        let head = prefix.get(8..8 + n as usize).ok_or_else(header_err)?;
        let json: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(head)
            .map_err(|e| JsError::new(&format!("safetensors header: {e}")))?;
        let keep: Box<dyn Fn(&str) -> bool> = match role {
            "all" => Box::new(|_| true),
            "frames" => Box::new(|k: &str| k.starts_with("fast_")),
            "slow" => {
                let cfg = config(config_json)?;
                let layers = stage_layers(&cfg, index, count)?;
                let (first, last) = (index == 0, index + 1 == count);
                Box::new(move |k: &str| {
                    if let Some(rest) = k.strip_prefix("layers.") {
                        let l: usize = rest
                            .split('.')
                            .next()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(usize::MAX);
                        return layers.contains(&l);
                    }
                    (first && (k == "embeddings.weight" || k == "codebook_embeddings.weight"))
                        // The band head is the tied embedding's semantic rows.
                        || (last && (k == "norm.weight" || k == "embeddings.weight"))
                })
            }
            _ => return Err(JsError::new(&format!("role {role:?}: slow | frames | all"))),
        };
        let data_start = 8 + n;
        let mut entries: Vec<Entry> = Vec::new();
        for (name, v) in &json {
            if name == "__metadata__" || !keep(name) {
                continue;
            }
            let get = |k: &str| {
                v.get(k)
                    .ok_or_else(|| JsError::new(&format!("{name}: no {k}")))
            };
            let offsets = get("data_offsets")?
                .as_array()
                .ok_or_else(|| JsError::new("data_offsets"))?;
            let (a, b) = (
                offsets[0].as_u64().unwrap_or(0),
                offsets[1].as_u64().unwrap_or(0),
            );
            entries.push(Entry {
                name: name.clone(),
                dtype: get("dtype")?.as_str().unwrap_or("").to_string(),
                shape: get("shape")?
                    .as_array()
                    .map(|s| s.iter().filter_map(serde_json::Value::as_u64).collect())
                    .unwrap_or_default(),
                start: data_start + a,
                end: data_start + b,
                at: 0,
            });
        }
        if entries.is_empty() {
            return Err(JsError::new(&format!("role {role:?} keeps no tensors")));
        }
        entries.sort_by_key(|e| e.start);
        // The subset's own header: the same tensors, offsets into its data.
        let mut header = serde_json::Map::new();
        let mut off = 0u64;
        for e in &entries {
            header.insert(
                e.name.clone(),
                serde_json::json!({"dtype": e.dtype, "shape": e.shape, "data_offsets": [off, off + (e.end - e.start)]}),
            );
            off += e.end - e.start;
        }
        let mut head = serde_json::to_vec(&header).map_err(|e| JsError::new(&e.to_string()))?;
        while (8 + head.len()) % 8 != 0 {
            head.push(b' ');
        }
        let base = 8 + head.len();
        let mut buffer = Vec::with_capacity(base + off as usize);
        buffer.extend_from_slice(&(head.len() as u64).to_le_bytes());
        buffer.extend_from_slice(&head);
        buffer.resize(base + off as usize, 0);
        let mut at = base;
        for e in &mut entries {
            e.at = at;
            at += (e.end - e.start) as usize;
        }
        // File ranges to fetch; neighbours under 1 MB apart merged.
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        for e in &entries {
            match ranges.last_mut() {
                Some(r) if e.start <= r.1 + (1 << 20) => r.1 = r.1.max(e.end),
                _ => ranges.push((e.start, e.end)),
            }
        }
        Ok(Audio8Fetch {
            entries,
            buffer: Some(buffer),
            ranges: ranges
                .into_iter()
                .flat_map(|(a, b)| [a as f64, b as f64])
                .collect(),
            added: 0,
        })
    }

    /// File byte ranges to fetch, flattened `[start, end, …]` (end exclusive).
    pub fn ranges(&self) -> Vec<f64> {
        self.ranges.clone()
    }

    /// The bytes of one piece starting at file offset `start`; only this
    /// worker's tensor bytes in it are kept.
    pub fn add(&mut self, start: f64, bytes: &[u8]) -> Result<(), JsError> {
        let buf = self
            .buffer
            .as_mut()
            .ok_or_else(|| JsError::new("Audio8Fetch already used"))?;
        let (lo, hi) = (start as u64, start as u64 + bytes.len() as u64);
        for e in &self.entries {
            let (a, b) = (e.start.max(lo), e.end.min(hi));
            if a < b {
                let dst = e.at + (a - e.start) as usize;
                buf[dst..dst + (b - a) as usize]
                    .copy_from_slice(&bytes[(a - lo) as usize..(b - lo) as usize]);
            }
        }
        self.added += bytes.len();
        Ok(())
    }

    /// Bytes this worker keeps.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.buffer.as_ref().map_or(0, Vec::len)
    }
}

impl Audio8Fetch {
    fn checkpoint(&mut self) -> Result<SafetensorsCheckpoint, JsError> {
        let buf = self
            .buffer
            .take()
            .ok_or_else(|| JsError::new("Audio8Fetch already used"))?;
        SafetensorsCheckpoint::from_bytes(vec![("model.safetensors".into(), buf)]).map_err(js_err)
    }
}

/// What a slow stage was handed, decoded from JS: rows (`Uint32Array`) on
/// the first stage, hidden rows (`Float32Array`) after.
enum Owned {
    Rows(Vec<Vec<u32>>),
    Hidden(Vec<f32>),
}

impl Owned {
    fn view(&self) -> StageInput<'_> {
        match self {
            Owned::Rows(r) => StageInput::Rows(r),
            Owned::Hidden(h) => StageInput::Hidden(h),
        }
    }
}

/// One stage of the slow stack. The first takes rows (`Uint32Array`,
/// `1 + num_codebooks` per row), every other the previous stage's
/// `Float32Array`; each answers with its output rows — on the last stage,
/// the newest row's hidden state followed by its band logits. `*_async`
/// for WebGPU, which cannot block on a read-back.
#[wasm_bindgen]
pub struct Audio8Slow {
    stage: Rc<RefCell<SlowStage>>,
    first: bool,
    width: usize,
}

#[wasm_bindgen]
impl Audio8Slow {
    pub fn from_fetch(
        fetch: &mut Audio8Fetch,
        config_json: &str,
        index: usize,
        count: usize,
        backend: Option<String>,
    ) -> Result<Audio8Slow, JsError> {
        let cfg = config(config_json)?;
        let layers = stage_layers(&cfg, index, count)?;
        let width = 1 + cfg.num_codebooks;
        let opts = S2Options {
            stage: Some(Stage {
                layers,
                last: index + 1 == count,
            }),
            ..S2Options::default()
        };
        let device = parse_backend(backend.as_deref())?;
        let model =
            S2Model::from_checkpoint(cfg, fetch.checkpoint()?, "audio8-web".into(), device, opts)
                .map_err(js_err)?;
        Ok(Audio8Slow {
            stage: Rc::new(RefCell::new(SlowStage::new(model))),
            first: index == 0,
            width,
        })
    }

    /// Rows the cache must hold for a `prompt`-row prompt and `max_new` frames.
    pub fn capacity(&self, prompt: usize, max_new: usize) -> usize {
        rlx_fish::audio8::cache_capacity(&self.stage.borrow().model.cfg, prompt, max_new)
    }

    /// Start an utterance (`rows` on the first stage, hidden rows after).
    pub fn prefill(&mut self, input: JsValue, cap: usize) -> Result<Vec<f32>, JsError> {
        let input = self.owned(&input)?;
        let mut st = self.stage.borrow_mut();
        let out = st.prefill(input.view(), cap).map_err(js_err)?;
        Ok(answer(&st, out))
    }

    /// One row more.
    pub fn step(&mut self, input: JsValue) -> Result<Vec<f32>, JsError> {
        let input = self.owned(&input)?;
        let mut st = self.stage.borrow_mut();
        let out = st.step(input.view()).map_err(js_err)?;
        Ok(answer(&st, out))
    }

    /// Prefill one sentence of the next batch (every stage queues the same
    /// sentences in the same order). Returns this stage's output rows; the
    /// last stage returns nothing until the batch steps.
    pub fn queue(&mut self, input: JsValue, cap: usize) -> Result<Vec<f32>, JsError> {
        let input = self.owned(&input)?;
        let mut st = self.stage.borrow_mut();
        let out = st.queue(input.view(), cap).map_err(js_err)?;
        Ok(if st.model.is_last() { Vec::new() } else { out })
    }

    /// Decode the queued sentences together from here on. The last stage
    /// answers with every sentence's hidden state then every sentence's
    /// band logits.
    pub fn join(&mut self) -> Result<Vec<f32>, JsError> {
        let mut st = self.stage.borrow_mut();
        st.join().map_err(js_err)?;
        Ok(batch_answer(&st, Vec::new()))
    }

    /// One row more for every sentence (`n` rows on the first stage).
    pub fn step_batch(&mut self, input: JsValue) -> Result<Vec<f32>, JsError> {
        let input = self.owned(&input)?;
        let mut st = self.stage.borrow_mut();
        let out = st.step_batch(input.view()).map_err(js_err)?;
        Ok(batch_answer(&st, out))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Audio8Slow {
    /// `prefill` as a Promise (a `Float32Array`) — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn prefill_async(&self, input: JsValue, cap: usize) -> Result<js_sys::Promise, JsError> {
        let input = self.owned(&input)?;
        let stage = self.stage.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            // Calls are awaited one at a time, so this borrow is never contended.
            let mut st = stage.borrow_mut();
            let out = st.prefill_async(input.view(), cap).await.map_err(to_js)?;
            Ok(js_sys::Float32Array::from(answer(&st, out).as_slice()).into())
        }))
    }

    /// `step` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn step_async(&self, input: JsValue) -> Result<js_sys::Promise, JsError> {
        let input = self.owned(&input)?;
        let stage = self.stage.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let mut st = stage.borrow_mut();
            let out = st.step_async(input.view()).await.map_err(to_js)?;
            Ok(js_sys::Float32Array::from(answer(&st, out).as_slice()).into())
        }))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Audio8Slow {
    /// `queue` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn queue_async(&self, input: JsValue, cap: usize) -> Result<js_sys::Promise, JsError> {
        let input = self.owned(&input)?;
        let stage = self.stage.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let mut st = stage.borrow_mut();
            let out = st.queue_async(input.view(), cap).await.map_err(to_js)?;
            let out = if st.model.is_last() { Vec::new() } else { out };
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        }))
    }

    /// `step_batch` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn step_batch_async(&self, input: JsValue) -> Result<js_sys::Promise, JsError> {
        let input = self.owned(&input)?;
        let stage = self.stage.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let mut st = stage.borrow_mut();
            let out = st.step_batch_async(input.view()).await.map_err(to_js)?;
            Ok(js_sys::Float32Array::from(batch_answer(&st, out).as_slice()).into())
        }))
    }
}

#[cfg(target_arch = "wasm32")]
fn to_js(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

impl Audio8Slow {
    fn owned(&self, input: &JsValue) -> Result<Owned, JsError> {
        if self.first {
            let flat: Vec<u32> = js_sys::Uint32Array::new(input).to_vec();
            if flat.is_empty() || !flat.len().is_multiple_of(self.width) {
                return Err(JsError::new("rows: a Uint32Array of whole rows"));
            }
            Ok(Owned::Rows(
                flat.chunks(self.width).map(<[u32]>::to_vec).collect(),
            ))
        } else {
            Ok(Owned::Hidden(js_sys::Float32Array::new(input).to_vec()))
        }
    }
}

/// The last stage answers with the newest row's hidden state and logits.
fn answer(stage: &SlowStage, out: Vec<f32>) -> Vec<f32> {
    if !stage.model.is_last() {
        return out;
    }
    let mut v = stage.hidden().to_vec();
    v.extend_from_slice(stage.logits());
    v
}

/// The last stage of a batch answers with `[n, dim]` hidden rows then the
/// `[n, band]` logits.
fn batch_answer(stage: &SlowStage, out: Vec<f32>) -> Vec<f32> {
    if !stage.model.is_last() {
        return out;
    }
    let mut v = stage.batch_hidden().to_vec();
    v.extend_from_slice(stage.batch_logits());
    v
}

/// The fast decoder and the sampler: `next(hidden ‖ logits)` is the next
/// frame's row (`Uint32Array`), or `undefined` at the end of the utterance.
#[wasm_bindgen]
pub struct Audio8Frames {
    maker: Rc<RefCell<FrameMaker>>,
    dim: usize,
    /// Sentences of the current batch (0 = not batching).
    batch: usize,
}

#[wasm_bindgen]
impl Audio8Frames {
    pub fn from_fetch(
        fetch: &mut Audio8Fetch,
        config_json: &str,
        backend: Option<String>,
    ) -> Result<Audio8Frames, JsError> {
        let cfg = config(config_json)?;
        let n = cfg.slow.n_layer;
        let dim = cfg.slow.dim;
        let opts = S2Options {
            stage: Some(Stage {
                layers: n..n,
                last: false,
            }),
            ..S2Options::default()
        };
        let device = parse_backend(backend.as_deref())?;
        let model =
            S2Model::from_checkpoint(cfg, fetch.checkpoint()?, "audio8-web".into(), device, opts)
                .map_err(js_err)?;
        Ok(Audio8Frames {
            maker: Rc::new(RefCell::new(FrameMaker::new(
                model,
                SampleParams::default(),
                0,
            ))),
            dim,
            batch: 0,
        })
    }

    /// Sampling for the next utterance, from `seed`.
    pub fn restart(&mut self, temperature: f32, top_p: f32, top_k: usize, seed: f64) {
        let params = SampleParams {
            temperature,
            top_p,
            top_k,
        };
        self.maker.borrow_mut().restart(params, seed as u64);
    }

    pub fn next(&mut self, hidden_logits: &[f32]) -> Result<Option<Vec<u32>>, JsError> {
        let (hidden, logits) = self.split(hidden_logits)?;
        self.maker.borrow_mut().next(hidden, logits).map_err(js_err)
    }

    /// Sampling for a batch: sentence `i` from `seeds[i]`.
    pub fn restart_batch(&mut self, temperature: f32, top_p: f32, top_k: usize, seeds: Vec<f64>) {
        let params = SampleParams {
            temperature,
            top_p,
            top_k,
        };
        let seeds: Vec<u64> = seeds.iter().map(|&s| s as u64).collect();
        self.maker.borrow_mut().restart_batch(params, &seeds);
        self.batch = seeds.len();
    }

    /// Every sentence's next row, flattened (`n` rows), from the last
    /// stage's batch answer; `undefined` once all have ended. Rows of ended
    /// sentences are filler: see `ended`.
    pub fn next_batch(&mut self, hidden_logits: &[f32]) -> Result<Option<Vec<u32>>, JsError> {
        let (hidden, logits) = self.split_batch(hidden_logits)?;
        let rows = self
            .maker
            .borrow_mut()
            .next_batch(hidden, logits)
            .map_err(js_err)?;
        Ok(rows.map(|r| r.concat()))
    }

    /// Whether sentence `i` of the batch has ended.
    pub fn ended(&self, i: usize) -> bool {
        self.maker.borrow().ended(i)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Audio8Frames {
    /// `next` as a Promise (a `Uint32Array`, or `undefined`) — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn next_async(&self, hidden_logits: Vec<f32>) -> Result<js_sys::Promise, JsError> {
        self.split(&hidden_logits)?;
        let (maker, dim) = (self.maker.clone(), self.dim);
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let (hidden, logits) = hidden_logits.split_at(dim);
            let row = maker
                .borrow_mut()
                .next_async(hidden, logits)
                .await
                .map_err(to_js)?;
            Ok(row.map_or(JsValue::UNDEFINED, |r| {
                js_sys::Uint32Array::from(r.as_slice()).into()
            }))
        }))
    }

    /// `next_batch` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn next_batch_async(&self, hidden_logits: Vec<f32>) -> Result<js_sys::Promise, JsError> {
        self.split_batch(&hidden_logits)?;
        let (maker, rows) = (self.maker.clone(), self.dim * self.batch);
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let (hidden, logits) = hidden_logits.split_at(rows);
            let out = maker
                .borrow_mut()
                .next_batch_async(hidden, logits)
                .await
                .map_err(to_js)?;
            Ok(out.map_or(JsValue::UNDEFINED, |r| {
                js_sys::Uint32Array::from(r.concat().as_slice()).into()
            }))
        }))
    }
}

impl Audio8Frames {
    fn split_batch<'a>(&self, hidden_logits: &'a [f32]) -> Result<(&'a [f32], &'a [f32]), JsError> {
        let rows = self.dim * self.batch;
        if self.batch == 0 || hidden_logits.len() <= rows {
            return Err(JsError::new(
                "next_batch: restart_batch, then hidden rows and band logits",
            ));
        }
        Ok(hidden_logits.split_at(rows))
    }

    fn split<'a>(&self, hidden_logits: &'a [f32]) -> Result<(&'a [f32], &'a [f32]), JsError> {
        if hidden_logits.len() <= self.dim {
            return Err(JsError::new("next: hidden state then band logits"));
        }
        Ok(hidden_logits.split_at(self.dim))
    }
}

/// The codec: `decode(codes)` (codebook-major `[10][frames]`, flattened) to
/// 44.1 kHz mono, `encode(pcm)` back to codes (for a reference voice).
#[wasm_bindgen]
pub struct Audio8Codec {
    codec: Rc<RefCell<FishCodec>>,
}

fn code_rows(codes: &[u32]) -> Result<Vec<Vec<u32>>, JsError> {
    let nc = rlx_fish::codec::NUM_CODEBOOKS;
    if codes.is_empty() || !codes.len().is_multiple_of(nc) {
        return Err(JsError::new("codes: [10][frames], flattened"));
    }
    let frames = codes.len() / nc;
    Ok(codes.chunks(frames).map(<[u32]>::to_vec).collect())
}

#[wasm_bindgen]
impl Audio8Codec {
    /// `files`: `codec_decoder.safetensors`, and `codec_encoder.safetensors`
    /// to clone a voice (`rlx-fish export-web`).
    #[wasm_bindgen(constructor)]
    pub fn new(files: &mut Files, backend: Option<String>) -> Result<Audio8Codec, JsError> {
        let parts: Vec<(String, Vec<u8>)> =
            ["codec_decoder.safetensors", "codec_encoder.safetensors"]
                .into_iter()
                .filter_map(|n| files.take_file(n).ok().map(|b| (n.to_string(), b)))
                .collect();
        if parts.is_empty() {
            return Err(JsError::new("codec: no codec_*.safetensors"));
        }
        let ckpt = SafetensorsCheckpoint::from_bytes(parts).map_err(js_err)?;
        let device = parse_backend(backend.as_deref())?;
        let codec =
            FishCodec::from_weights(CodecWeights::Safetensors(ckpt), device).map_err(js_err)?;
        Ok(Audio8Codec {
            codec: Rc::new(RefCell::new(codec)),
        })
    }

    pub fn decode(&mut self, codes: &[u32]) -> Result<Vec<f32>, JsError> {
        let rows = code_rows(codes)?;
        self.codec.borrow_mut().decode(&rows).map_err(js_err)
    }

    /// The samples of `decode(codes)` from sample `from` on, decoding only
    /// those: streaming, as the frames arrive.
    pub fn decode_from(&mut self, codes: &[u32], from: usize) -> Result<Vec<f32>, JsError> {
        let rows = code_rows(codes)?;
        self.codec
            .borrow_mut()
            .decode_from(&rows, from)
            .map_err(js_err)
    }

    /// 44.1 kHz mono → codes `[10][frames]`, flattened.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u32>, JsError> {
        let codes = self.codec.borrow_mut().encode(pcm).map_err(js_err)?;
        Ok(codes.concat())
    }

    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        rlx_fish::codec::SAMPLE_RATE
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Audio8Codec {
    /// `decode` as a Promise (a `Float32Array`) — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn decode_async(&self, codes: Vec<u32>) -> Result<js_sys::Promise, JsError> {
        let rows = code_rows(&codes)?;
        let codec = self.codec.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let pcm = codec
                .borrow_mut()
                .decode_async(&rows)
                .await
                .map_err(to_js)?;
            Ok(js_sys::Float32Array::from(pcm.as_slice()).into())
        }))
    }

    /// `decode_from` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn decode_from_async(
        &self,
        codes: Vec<u32>,
        from: usize,
    ) -> Result<js_sys::Promise, JsError> {
        let rows = code_rows(&codes)?;
        let codec = self.codec.clone();
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let pcm = codec
                .borrow_mut()
                .decode_from_async(&rows, from)
                .await
                .map_err(to_js)?;
            Ok(js_sys::Float32Array::from(pcm.as_slice()).into())
        }))
    }

    /// `encode` as a Promise (a `Uint32Array`) — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn encode_async(&self, pcm: Vec<f32>) -> js_sys::Promise {
        let codec = self.codec.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let codes = codec.borrow_mut().encode_async(&pcm).await.map_err(to_js)?;
            Ok(js_sys::Uint32Array::from(codes.concat().as_slice()).into())
        })
    }
}

/// The prompt rows for `text` (`Uint32Array`, `1 + num_codebooks` per row),
/// as `ArkttsProcessor` builds them; with `reference_codes` (`[10][frames]`,
/// flattened) and its transcript, for a cloned voice.
/// `text` cut into sentences for `speak({ batch })`, as the native
/// `rlx_fish::audio8::synthesize_batch` cuts it.
#[wasm_bindgen]
pub fn audio8_sentences(text: &str) -> Vec<String> {
    rlx_fish::audio8::split_sentences(text, rlx_fish::audio8::SENTENCE_MIN_CHARS)
}

#[wasm_bindgen]
pub fn audio8_prompt(
    tokenizer: &crate::TextTokenizer,
    config_json: &str,
    text: &str,
    reference_text: Option<String>,
    reference_codes: Option<Vec<u32>>,
) -> Result<Vec<u32>, JsError> {
    let cfg = config(config_json)?;
    let reference = match (reference_text, reference_codes) {
        (Some(t), Some(c)) if !c.is_empty() => {
            let nc = cfg.num_codebooks;
            if !c.len().is_multiple_of(nc) {
                return Err(JsError::new("reference codes: [10][frames], flattened"));
            }
            let frames = c.len() / nc;
            Some(Reference {
                text: t,
                codes: c.chunks(frames).map(<[u32]>::to_vec).collect(),
            })
        }
        _ => None,
    };
    let rows = rlx_fish::audio8::prompt_rows(tokenizer.inner(), &cfg, text, reference.as_ref())
        .map_err(js_err)?;
    Ok(rows.concat())
}
