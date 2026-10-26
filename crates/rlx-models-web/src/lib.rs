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

//! Browser bindings (`wasm-bindgen`) for small RLX models.
//!
//! Weights arrive as bytes the page fetched — there is no filesystem on
//! wasm32 — collected in a [`Files`] bundle under the checkpoint's own file
//! names (`config.json`, `model.safetensors`, `tokenizer.json`, …):
//!
//! ```js
//! const files = new Files();
//! for (const name of ["config.json", "model.safetensors", "tokenizer.json"])
//!   files.add(name, new Uint8Array(await (await fetch(base + name)).arrayBuffer()));
//! const asr = new Transcriber(files);
//! console.log(asr.transcribe(pcm16k));
//! ```
//!
//! Everything runs on the CPU backend, single-threaded.
//!
//! The same types work natively (the crate is also an `rlib`), which is how
//! `tests/memory_source.rs` checks that in-memory loading matches loading
//! from a directory.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use rlx_core::AssetSource;
use rlx_runtime::Device;
use wasm_bindgen::prelude::*;

/// Runs when the module loads: route Rust panics to `console.error`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

/// Set an `RLX_*` setting for this module (e.g. `"RLX_DISABLE_GPU_KV", "1"`);
/// the browser has no process environment, and every `RLX_*` read goes
/// through `rlx_ir::env`. Set it before building the model it affects.
#[wasm_bindgen]
pub fn set_setting(key: &str, value: &str) {
    rlx_ir::env::set(key, value);
}

/// Set up WebGPU; resolves to whether a GPU adapter was found. Await it once
/// before creating a model with `backend: "webgpu"`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn init_webgpu() -> bool {
    // Some browsers expose `navigator.gpu` but grant no adapter (headless
    // Firefox: `requestAdapter()` resolves to null). wgpu then reads `.info`
    // off the null inside a callback, so its future never settles and the
    // page hangs. Ask first, and only hand over to wgpu when there is one.
    if !adapter_available().await {
        return false;
    }
    rlx_runtime::init_webgpu().await
}

/// `await navigator.gpu?.requestAdapter()` is non-null (window or worker).
#[cfg(target_arch = "wasm32")]
async fn adapter_available() -> bool {
    use js_sys::{Function, Promise, Reflect};
    let get = |obj: &JsValue, key: &str| Reflect::get(obj, &JsValue::from_str(key)).ok();
    let Some(gpu) = get(&js_sys::global(), "navigator")
        .and_then(|n| get(&n, "gpu"))
        .filter(|g| !g.is_undefined() && !g.is_null())
    else {
        return false;
    };
    let Some(request) = get(&gpu, "requestAdapter").and_then(|f| f.dyn_into::<Function>().ok())
    else {
        return false;
    };
    let Ok(promise) = request.call0(&gpu).and_then(|p| p.dyn_into::<Promise>()) else {
        return false;
    };
    matches!(
        wasm_bindgen_futures::JsFuture::from(promise).await,
        Ok(adapter) if !adapter.is_null() && !adapter.is_undefined()
    )
}

/// `"cpu"` (default), `"webgpu"` — wgpu on the browser's WebGPU, which needs
/// the `webgpu` build feature and `init_webgpu()` — or `"webgl"` (WebGL2,
/// the `webgl` build feature; runs synchronously, so the `*_async` methods
/// simply resolve once it is done).
fn parse_backend(backend: Option<&str>) -> Result<Device, JsError> {
    match backend.unwrap_or("cpu") {
        "cpu" => Ok(Device::Cpu),
        "webgl" | "opengl" => {
            if cfg!(feature = "webgl") {
                Ok(Device::OpenGl)
            } else {
                Err(JsError::new(
                    "this build has no WebGL (build with the `webgl` feature)",
                ))
            }
        }
        "webgpu" | "gpu" => {
            if cfg!(feature = "webgpu") {
                Ok(Device::Gpu)
            } else {
                Err(JsError::new(
                    "this build has no WebGPU (build with the `webgpu` feature)",
                ))
            }
        }
        other => Err(JsError::new(&format!(
            "unknown backend {other:?}: cpu or webgpu"
        ))),
    }
}

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

/// A checkpoint's files, by name, as the page fetched them.
#[wasm_bindgen]
#[derive(Default, Clone)]
pub struct Files {
    map: HashMap<String, Vec<u8>>,
}

#[wasm_bindgen]
impl Files {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Files {
        Files::default()
    }

    /// Add one file (`name` relative to the checkpoint root, e.g.
    /// `adapters/retrieval/adapter_model.safetensors`).
    pub fn add(&mut self, name: &str, bytes: Vec<u8>) {
        self.map.insert(name.to_string(), bytes);
    }

    /// Total bytes held.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.map.values().map(Vec::len).sum()
    }
}

impl Files {
    /// Remove and return one file's bytes (no copy).
    fn take_file(&mut self, name: &str) -> Result<Vec<u8>, JsError> {
        self.map
            .remove(name)
            .ok_or_else(|| JsError::new(&format!("missing file {name:?}")))
    }

    /// Take the files out as an [`AssetSource`]; the bundle is left empty.
    pub fn take_source(&mut self) -> AssetSource {
        AssetSource::memory(std::mem::take(&mut self.map))
    }
}

/// Moonshine v2 streaming speech recognition (tiny / small / medium).
#[wasm_bindgen]
pub struct Transcriber {
    // Shared so the async method can hold it across an await.
    model: Rc<RefCell<rlx_moonshine_streaming::MoonshineStreaming>>,
}

#[wasm_bindgen]
impl Transcriber {
    /// Consumes the files in `files`. `backend`: `"cpu"` (default), `"webgpu"` or `"webgl"`.
    #[wasm_bindgen(constructor)]
    pub fn new(files: &mut Files, backend: Option<String>) -> Result<Transcriber, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let model =
            rlx_moonshine_streaming::MoonshineStreaming::open_source(&files.take_source(), device)
                .map_err(js_err)?;
        Ok(Transcriber {
            model: Rc::new(RefCell::new(model)),
        })
    }

    /// Transcribe 16 kHz mono PCM in `[-1, 1]` (CPU or WebGL backend).
    pub fn transcribe(
        &mut self,
        pcm: Vec<f32>,
        max_new_tokens: Option<usize>,
    ) -> Result<String, JsError> {
        self.model
            .borrow_mut()
            .transcribe(&pcm, max_new_tokens)
            .map_err(js_err)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Transcriber {
    /// Like `transcribe`, as a Promise — required on WebGPU. Works on every backend.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn transcribe_async(
        &self,
        pcm: Vec<f32>,
        max_new_tokens: Option<usize>,
    ) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut m = model
                .try_borrow_mut()
                .map_err(|_| JsValue::from_str("a transcription is already in flight"))?;
            let text = m
                .transcribe_async(&pcm, max_new_tokens)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(JsValue::from_str(&text))
        })
    }
}

enum EmbedModel {
    Gemma(rlx_embeddinggemma::EmbeddingGemma),
    Jina(rlx_jina_v5::JinaV5),
}

/// Sentence embeddings: `"embeddinggemma"` or `"jina-v5"`.
#[wasm_bindgen]
pub struct Embedder {
    // Shared so the async method can hold it across an await (wasm-bindgen
    // futures cannot borrow `self`).
    model: Rc<RefCell<EmbedModel>>,
}

#[wasm_bindgen]
impl Embedder {
    /// `task` selects the jina-v5 adapter (`retrieval`, `text-matching`,
    /// `clustering`, `classification`); EmbeddingGemma ignores it.
    /// `backend`: `"cpu"` (default), `"webgpu"` or `"webgl"`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        files: &mut Files,
        model: &str,
        task: Option<String>,
        backend: Option<String>,
    ) -> Result<Embedder, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let src = files.take_source();
        let model = match model {
            "embeddinggemma" => EmbedModel::Gemma(
                rlx_embeddinggemma::EmbeddingGemma::open_source(&src, device).map_err(js_err)?,
            ),
            "jina-v5" => EmbedModel::Jina(
                rlx_jina_v5::JinaV5::open_source(
                    &src,
                    task.as_deref().unwrap_or("retrieval"),
                    device,
                )
                .map_err(js_err)?,
            ),
            other => {
                return Err(JsError::new(&format!(
                    "unknown embedding model {other:?}: expected \"embeddinggemma\" or \"jina-v5\""
                )));
            }
        };
        Ok(Embedder {
            model: Rc::new(RefCell::new(model)),
        })
    }

    /// Embedding width.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        match &*self.model.borrow() {
            EmbedModel::Gemma(m) => m.dim(),
            EmbedModel::Jina(m) => m.dim(),
        }
    }

    /// Unit-length embedding of `text` with the named prompt (`"query"`,
    /// `"document"`, …; empty for none). `dim` cuts to a Matryoshka prefix.
    pub fn embed(
        &mut self,
        text: &str,
        prompt: &str,
        dim: Option<usize>,
    ) -> Result<Vec<f32>, JsError> {
        let v = match &mut *self.model.borrow_mut() {
            EmbedModel::Gemma(m) => {
                let p = m.prompt(prompt).unwrap_or("").to_string();
                m.embed_text(&p, text).map_err(js_err)?
            }
            EmbedModel::Jina(m) => {
                let p = m.prompt(prompt).unwrap_or("").to_string();
                m.embed_text(&p, text).map_err(js_err)?
            }
        };
        Ok(match dim {
            Some(d) => rlx_jina_v5::truncate(&v, d),
            None => v,
        })
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Embedder {
    /// Like `embed`, as a Promise — required on WebGPU, where reading
    /// results back from the GPU cannot block. Works on every backend.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn embed_async(&self, text: String, prompt: String, dim: Option<usize>) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            // The borrow spans the await: a second call before the first
            // resolves must be an error, not a panic.
            let mut guard = model
                .try_borrow_mut()
                .map_err(|_| JsValue::from_str("an embedding is already in flight"))?;
            let v = match &mut *guard {
                EmbedModel::Jina(m) => {
                    let p = m.prompt(&prompt).unwrap_or("").to_string();
                    let ids = m
                        .tokenize(&format!("{p}{text}"))
                        .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
                    m.embed_ids_async(&ids)
                        .await
                        .map_err(|e| JsValue::from_str(&format!("{e:#}")))?
                }
                EmbedModel::Gemma(m) => {
                    let p = m.prompt(&prompt).unwrap_or("").to_string();
                    let ids = m
                        .tokenize(&format!("{p}{text}"))
                        .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
                    m.embed_ids_async(&ids)
                        .await
                        .map_err(|e| JsValue::from_str(&format!("{e:#}")))?
                }
            };
            let v = match dim {
                Some(d) => rlx_jina_v5::truncate(&v, d),
                None => v,
            };
            Ok(js_sys::Float32Array::from(v.as_slice()).into())
        })
    }
}

/// A BERT-family sentence encoder (bge, MiniLM, MedEmbed, MedCPT, …) from an
/// HF checkpoint — f32, or int8-packed by `rlx_embed::packed` (int8 matmuls,
/// a quarter of the memory) — for one text of up to `seq` tokens, compiled
/// per length bucket.
#[wasm_bindgen]
pub struct BertEmbedder {
    inner: Rc<RefCell<rlx_embed::BertTextEncoder>>,
}

#[wasm_bindgen]
impl BertEmbedder {
    /// Consumes `config.json`, `model.safetensors`, `tokenizer.json` and (when
    /// present) `tokenizer_config.json` / `special_tokens_map.json` from
    /// `files`. `pooling`: `"cls"` (bge, MedCPT) or `"mean"` (MiniLM). `seq`:
    /// the token budget per text (default 32). `backend`: `"cpu"` (default),
    /// `"webgpu"` or `"webgl"`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        files: &mut Files,
        pooling: &str,
        seq: Option<usize>,
        backend: Option<String>,
    ) -> Result<BertEmbedder, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let pooling = match pooling {
            "cls" => rlx_embed::Pooling::Cls,
            "mean" => rlx_embed::Pooling::Mean,
            other => {
                return Err(JsError::new(&format!(
                    "unknown pooling {other:?}: cls or mean"
                )));
            }
        };
        let mut optional = |name: &str| files.take_file(name).unwrap_or_else(|_| b"{}".to_vec());
        let (tokenizer_config, special_tokens_map) = (
            optional("tokenizer_config.json"),
            optional("special_tokens_map.json"),
        );
        let enc = rlx_embed::BertTextEncoder::from_bytes(
            rlx_embed::BertFiles {
                config: &files.take_file("config.json")?,
                weights: files.take_file("model.safetensors")?,
                tokenizer: &files.take_file("tokenizer.json")?,
                tokenizer_config: &tokenizer_config,
                special_tokens_map: &special_tokens_map,
            },
            pooling,
            seq.unwrap_or(32),
            device,
        )
        .map_err(js_err)?;
        Ok(BertEmbedder {
            inner: Rc::new(RefCell::new(enc)),
        })
    }

    /// Embedding width.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.inner.borrow().dim()
    }

    /// Unit-length embedding of `text` (CPU or WebGL backend). Prefix any
    /// query instruction (bge's) yourself.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, JsError> {
        self.inner.borrow_mut().embed(text).map_err(js_err)
    }

    /// Compile every length bucket now (a text runs in the smallest of 8, 16,
    /// 32, … that holds it), so no later call waits for one.
    pub fn warm_buckets(&mut self) -> Result<(), JsError> {
        self.inner.borrow_mut().warm_buckets().map_err(js_err)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl BertEmbedder {
    /// Like `embed`, as a Promise — required on WebGPU. Works on every backend.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn embed_async(&self, text: String) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut enc = inner
                .try_borrow_mut()
                .map_err(|_| JsValue::from_str("an embedding is already in flight"))?;
            let v = enc
                .embed_async(&text)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(js_sys::Float32Array::from(v.as_slice()).into())
        })
    }
}

/// LFM2 / LFM2.5 text generation (greedy) from an HF checkpoint.
#[wasm_bindgen]
pub struct TextGenerator {
    // Shared so the async method can keep them across an await.
    runner: Rc<rlx_lfm::Lfm2GgufRunner>,
    tokenizer: Rc<tokenizers::Tokenizer>,
    /// A run is in flight. The runner's decode cache sits behind a mutex, and
    /// on wasm's single thread a second call would wait on it forever.
    busy: Rc<std::cell::Cell<bool>>,
}

#[wasm_bindgen]
impl TextGenerator {
    /// `backend`: `"cpu"` (default), `"webgl"` (WebGL2, synchronous) or
    /// `"webgpu"` (use `generate_async`).
    #[wasm_bindgen(constructor)]
    pub fn new(files: &mut Files, backend: Option<String>) -> Result<TextGenerator, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let src = files.take_source();
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(src.read("tokenizer.json").map_err(js_err)?)
                .map_err(|e| JsError::new(&format!("tokenizer.json: {e}")))?;
        let runner = rlx_lfm::Lfm2GgufRunner::open_source(&src, device).map_err(js_err)?;
        Ok(TextGenerator {
            runner: Rc::new(runner),
            tokenizer: Rc::new(tokenizer),
            busy: Rc::default(),
        })
    }

    /// Greedy continuation of token ids (for callers that tokenize themselves).
    /// CPU and WebGL.
    pub fn generate_ids(&mut self, ids: Vec<u32>, max_tokens: usize) -> Result<Vec<u32>, JsError> {
        let _guard = Busy::claim(&self.busy)?;
        self.runner
            .generate(&ids, max_tokens, |_| true)
            .map_err(js_err)
    }

    /// Greedy continuation of `prompt` (special tokens such as
    /// `<|im_start|>` are parsed, so a chat template can be passed as text).
    /// CPU and WebGL.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, JsError> {
        let _guard = Busy::claim(&self.busy)?;
        let ids = encode(&self.tokenizer, prompt).map_err(|e| JsError::new(&e))?;
        let out = self
            .runner
            .generate(&ids, max_tokens, |_| true)
            .map_err(js_err)?;
        decode(&self.tokenizer, &out).map_err(|e| JsError::new(&e))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl TextGenerator {
    /// Like `generate`, as a Promise — required on WebGPU. Works on every backend.
    pub fn generate_async(&self, prompt: String, max_tokens: usize) -> js_sys::Promise {
        let (runner, tokenizer, busy) = (
            self.runner.clone(),
            self.tokenizer.clone(),
            self.busy.clone(),
        );
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let ids = encode(&tokenizer, &prompt).map_err(|e| JsValue::from_str(&e))?;
            let out = runner
                .generate_async(&ids, max_tokens, |_| true)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            let text = decode(&tokenizer, &out).map_err(|e| JsValue::from_str(&e))?;
            Ok(JsValue::from_str(&text))
        })
    }
}

/// Qwen3.5 text generation (greedy) from a GGUF — `model.gguf` plus the HF
/// `tokenizer.json`. Quantized weights stay packed end to end (the GPU
/// dequantizes inside the matmul), and a BF16 file stays bf16; nothing is
/// widened to f32 on the host, so Qwen3.5-0.8B fits a wasm32 page (Q4_K_M
/// ~0.5 GB of weights, BF16 ~1.5 GB) instead of needing > 4 GiB.
#[wasm_bindgen]
pub struct Qwen35Generator {
    runner: Rc<RefCell<rlx_qwen35::Qwen35Runner>>,
    tokenizer: Rc<tokenizers::Tokenizer>,
    busy: Rc<std::cell::Cell<bool>>,
}

#[wasm_bindgen]
impl Qwen35Generator {
    /// `backend`: `"webgpu"` (use `generate_async`) or `"cpu"`. `max_seq`
    /// bounds prompt + generated tokens (default 512); prompts are limited to
    /// 128 tokens.
    #[wasm_bindgen(constructor)]
    pub fn new(
        files: &mut Files,
        backend: Option<String>,
        max_seq: Option<usize>,
    ) -> Result<Qwen35Generator, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(files.take_file("tokenizer.json")?)
            .map_err(|e| JsError::new(&format!("tokenizer.json: {e}")))?;
        // Parsed in place: the download becomes the data segment, no copy.
        let file = rlx_gguf::GgufFile::from_bytes(files.take_file("model.gguf")?)
            .map_err(|e| JsError::new(&format!("model.gguf: {e}")))?;
        // Gather token embeddings on the host from the packed table, one row
        // at a time — never the 1 GB f32 copy.
        rlx_ir::env::set("RLX_QWEN35_HOST_EMBED", "1");
        // A BF16 GGUF stays bf16 (2 bytes/weight) instead of widening to f32.
        rlx_ir::env::set("RLX_QWEN35_KEEP_BF16", "1");
        // Q4_1 stays packed too: CPU and wgpu both read it (a Q4_0 GGUF
        // stores a few `ffn_down` tensors as Q4_1).
        rlx_ir::env::set("RLX_GGUF_PACKED_Q4_1", "1");
        // Upload packed weights straight from the GGUF bytes rather than via a
        // second, cached copy of every tensor.
        rlx_ir::env::set("RLX_LOW_MEM_COMPILE", "1");
        keep_prefill_by_default(&file, 0, 1);
        let runner = rlx_qwen35::Qwen35Runner::builder()
            .gguf_loader(rlx_core::weight_loader::GgufLoader::from_gguf_file(file))
            .device(device)
            .max_seq(max_seq.unwrap_or(512))
            // Prompts are prefilled in one graph padded to this length. On
            // wasm32 a wgpu arena must stay under 2 GiB (bit 31 tags weight
            // offsets); for Qwen3.5-0.8B that is ~7 MB per position on top of
            // 0.55 GB of packed weights, so 128 → 1.44 GiB.
            .prefill_seq(128)
            .skip_auto_mmproj(true)
            // Warming runs a decode step, which blocks — not possible on WebGPU.
            .skip_warm(true)
            .build()
            .map_err(js_err)?;
        Ok(Qwen35Generator {
            runner: Rc::new(RefCell::new(runner)),
            tokenizer: Rc::new(tokenizer),
            busy: Rc::default(),
        })
    }

    /// Greedy continuation of `prompt` (CPU backend).
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, JsError> {
        let _guard = Busy::claim(&self.busy)?;
        let ids = encode(&self.tokenizer, prompt).map_err(|e| JsError::new(&e))?;
        let out = self
            .runner
            .borrow_mut()
            .generate_stepwise(&ids, max_tokens, |_| true)
            .map_err(js_err)?;
        decode(&self.tokenizer, &out).map_err(|e| JsError::new(&e))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Qwen35Generator {
    /// Like `generate`, as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn generate_async(&self, prompt: String, max_tokens: usize) -> js_sys::Promise {
        let (runner, tokenizer, busy) = (
            self.runner.clone(),
            self.tokenizer.clone(),
            self.busy.clone(),
        );
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let ids = encode(&tokenizer, &prompt).map_err(|e| JsValue::from_str(&e))?;
            // `busy` keeps a second call out, so this borrow is never contended.
            let out = runner
                .borrow_mut()
                .generate_async(&ids, max_tokens, |_| true)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            let text = decode(&tokenizer, &out).map_err(|e| JsValue::from_str(&e))?;
            Ok(JsValue::from_str(&text))
        })
    }
}

/// One stage of a Qwen3.5 layer pipeline: this instance loads and runs only
/// its share of the trunk layers (`index` of `count`), so N Web Workers — each
/// with its own wasm32 address space and WebGPU device — can hold a model
/// none of them could alone. `web/pipeline.js` (`Qwen35Pipeline`) drives a
/// set of them; the hand-off between stages is one `Float32Array` of hidden
/// rows (`[n, hidden]`; 4 KB per decode step for Qwen3.5-0.8B).
///
/// The first stage takes token ids (`Uint32Array`), every later one the
/// previous stage's `Float32Array`; the last stage answers with the next
/// token id (a number), every other stage with its `Float32Array`.
#[wasm_bindgen]
pub struct Qwen35Stage {
    runner: Rc<RefCell<rlx_qwen35::Qwen35Runner>>,
    busy: Rc<std::cell::Cell<bool>>,
}

#[wasm_bindgen]
impl Qwen35Stage {
    /// `files` needs only `model.gguf`; this stage keeps its own layers and
    /// frees the rest. `backend` and `max_seq` as for `Qwen35Generator`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        files: &mut Files,
        backend: Option<String>,
        max_seq: Option<usize>,
        index: usize,
        count: usize,
        rows: Option<usize>,
        prefill: Option<usize>,
    ) -> Result<Qwen35Stage, JsError> {
        let file = rlx_gguf::GgufFile::from_bytes(files.take_file("model.gguf")?)
            .map_err(|e| JsError::new(&format!("model.gguf: {e}")))?;
        Self::build(file, backend, max_seq, index, count, rows, prefill)
    }

    /// A stage from the parts of `model.gguf` it reads ([`Qwen35StageFetch`])
    /// rather than the whole file. Empties `fetch`. `rows` > 1 runs decode
    /// as fixed multi-row steps (`rows` / `rows_async`).
    pub fn from_fetch(
        fetch: &mut Qwen35StageFetch,
        backend: Option<String>,
        max_seq: Option<usize>,
        rows: Option<usize>,
        prefill: Option<usize>,
    ) -> Result<Qwen35Stage, JsError> {
        let partial = fetch
            .partial
            .take()
            .ok_or_else(|| JsError::new("Qwen35StageFetch already used"))?;
        let file = partial
            .finish()
            .map_err(|e| JsError::new(&format!("model.gguf: {e}")))?;
        Self::build(
            file,
            backend,
            max_seq,
            fetch.index,
            fetch.count,
            rows,
            prefill,
        )
    }
}

impl Qwen35Stage {
    fn build(
        file: rlx_gguf::GgufFile,
        backend: Option<String>,
        max_seq: Option<usize>,
        index: usize,
        count: usize,
        rows: Option<usize>,
        prefill: Option<usize>,
    ) -> Result<Qwen35Stage, JsError> {
        let device = parse_backend(backend.as_deref())?;
        // Prompt tokens the prefill graph takes in one pass (default 128). A
        // wider one keeps only the last position's logits: all of them would
        // be [prefill, 248320] floats (1 GB at 1024).
        let prefill = prefill.unwrap_or(128);
        rlx_ir::env::set("RLX_QWEN35_HOST_EMBED", "1");
        rlx_ir::env::set("RLX_QWEN35_KEEP_BF16", "1");
        // Q4_1 stays packed too: CPU and wgpu both read it (a Q4_0 GGUF
        // stores a few `ffn_down` tensors as Q4_1).
        rlx_ir::env::set("RLX_GGUF_PACKED_Q4_1", "1");
        rlx_ir::env::set("RLX_LOW_MEM_COMPILE", "1");
        keep_prefill_by_default(&file, index, count);
        let runner = rlx_qwen35::Qwen35Runner::builder()
            .gguf_loader(rlx_core::weight_loader::GgufLoader::from_gguf_file(file))
            .device(device)
            .max_seq(max_seq.unwrap_or(512))
            .prefill_seq(prefill)
            .last_logits_only(prefill > 128)
            .skip_auto_mmproj(true)
            .skip_warm(true)
            .pipeline_stage(index, count)
            .decode_rows(rows.unwrap_or(1))
            .build()
            .map_err(js_err)?;
        Ok(Qwen35Stage {
            runner: Rc::new(RefCell::new(runner)),
            busy: Rc::default(),
        })
    }
}

#[wasm_bindgen]
impl Qwen35Stage {
    /// Sample this stage's tokens (the last stage of a pipeline, or a lone
    /// stage) instead of taking the most likely one: `temperature`, `top_p`
    /// (1 = off), `top_k` (0 = off) and an OpenAI-style `presence_penalty`
    /// over the tokens generated so far, from `seed`. Greedy decoding makes a
    /// small model like Qwen3.5-0.8B repeat itself and never end its turn;
    /// Qwen suggests 0.7 / 0.8 / 20 / 1.5 for non-thinking chat. Applies from
    /// the next `prefill`.
    pub fn set_sampling(
        &mut self,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        presence_penalty: f32,
        seed: f64,
    ) -> Result<(), JsError> {
        let mut o = rlx_runtime::SampleOpts::nucleus(temperature, top_p);
        o.top_k = (top_k > 0).then_some(top_k);
        o.presence_penalty = presence_penalty;
        self.runner
            .borrow_mut()
            .set_sampling(Some((o, seed as u64)))
            .map_err(js_err)
    }

    /// Logits of `ids` after the last `prefill` on the last stage (the whole
    /// prompt must have gone through `prefill`, not decode steps): for
    /// scoring a few candidate next tokens.
    pub fn scores(&self, ids: Vec<u32>) -> Result<Vec<f32>, JsError> {
        self.runner.borrow().stage_scores(&ids).map_err(js_err)
    }

    /// Back to greedy decoding (the most likely token every step).
    pub fn set_greedy(&mut self) {
        let _ = self.runner.borrow_mut().set_sampling(None);
    }

    /// Start a sequence: the prompt's token ids (first stage) or the previous
    /// stage's hidden rows. CPU backend; see `prefill_async`.
    pub fn prefill(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let _guard = Busy::claim(&self.busy)?;
        let input = StageIn::from_js(&input)?;
        let out = self
            .runner
            .borrow_mut()
            .stage_prefill(input.as_input())
            .map_err(js_err)?;
        Ok(stage_out_to_js(out))
    }

    /// Rows per decode step (`rows` at construction; 1 = one-token decode).
    pub fn decode_rows(&self) -> usize {
        self.runner.borrow().decode_rows()
    }

    /// One multi-row step: up to `decode_rows()` new positions — token ids
    /// (first stage) or the previous stage's hidden rows. The last stage
    /// answers with the next token after every row (a `Uint32Array`). With
    /// `undo`, `rows_undo` can cut the step back afterwards. `guesses`: the
    /// token ids rows 1.. stand for when they are guesses to check (the last
    /// stage judges them; see `Qwen35Runner::stage_rows`). CPU backend; see
    /// `rows_async`.
    pub fn rows(
        &mut self,
        input: JsValue,
        undo: bool,
        guesses: Option<Vec<u32>>,
    ) -> Result<JsValue, JsError> {
        let _guard = Busy::claim(&self.busy)?;
        let input = StageIn::from_js(&input)?;
        let out = self
            .runner
            .borrow_mut()
            .stage_rows(input.as_input(), undo, guesses.as_deref())
            .map_err(js_err)?;
        Ok(stage_out_to_js(out))
    }

    /// Keep only the first `keep` rows of the last `rows` step (run with
    /// `undo`). CPU backend; see `rows_undo_async`.
    pub fn rows_undo(&mut self, keep: usize) -> Result<(), JsError> {
        let _guard = Busy::claim(&self.busy)?;
        self.runner
            .borrow_mut()
            .stage_rows_undo(keep)
            .map_err(js_err)
    }

    /// One decode step: the newest token id (first stage) or the previous
    /// stage's one hidden row. CPU backend; see `decode_async`.
    pub fn decode(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let _guard = Busy::claim(&self.busy)?;
        let input = StageIn::from_js(&input)?;
        let out = self
            .runner
            .borrow_mut()
            .stage_decode(input.as_input())
            .map_err(js_err)?;
        Ok(stage_out_to_js(out))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Qwen35Stage {
    /// `prefill` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn prefill_async(&self, input: JsValue) -> js_sys::Promise {
        let (runner, busy) = (self.runner.clone(), self.busy.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let input = StageIn::from_js(&input).map_err(JsValue::from)?;
            // `busy` keeps a second call out, so this borrow is never contended.
            let out = runner
                .borrow_mut()
                .stage_prefill_async(input.as_input())
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(stage_out_to_js(out))
        })
    }

    /// `rows` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn rows_async(
        &self,
        input: JsValue,
        undo: bool,
        guesses: Option<Vec<u32>>,
    ) -> js_sys::Promise {
        let (runner, busy) = (self.runner.clone(), self.busy.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let input = StageIn::from_js(&input).map_err(JsValue::from)?;
            let out = runner
                .borrow_mut()
                .stage_rows_async(input.as_input(), undo, guesses.as_deref())
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(stage_out_to_js(out))
        })
    }

    /// `rows_undo` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn rows_undo_async(&self, keep: usize) -> js_sys::Promise {
        let (runner, busy) = (self.runner.clone(), self.busy.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            runner
                .borrow_mut()
                .stage_rows_undo_async(keep)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(JsValue::UNDEFINED)
        })
    }

    /// `decode` as a Promise — required on WebGPU.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn decode_async(&self, input: JsValue) -> js_sys::Promise {
        let (runner, busy) = (self.runner.clone(), self.busy.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let input = StageIn::from_js(&input).map_err(JsValue::from)?;
            let out = runner
                .borrow_mut()
                .stage_decode_async(input.as_input())
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(stage_out_to_js(out))
        })
    }
}

/// The parts of a Qwen3.5 `model.gguf` one pipeline stage reads, gathered
/// with HTTP `Range` requests so a worker never downloads (or holds) the rest:
///
/// ```js
/// const f = new Qwen35StageFetch(prefix, index, count); // prefix: the file's first bytes
/// const r = f.ranges();                                  // [start, end, start, end, …]
/// for (let i = 0; i < r.length; i += 2) f.add(r[i], await fetchRange(url, r[i], r[i + 1]));
/// const stage = Qwen35Stage.from_fetch(f, "webgpu", 256);
/// ```
///
/// The constructor fails if `prefix` ends inside the header (metadata plus
/// tensor table — about 10 MB for Qwen3.5, most of it the tokenizer
/// vocabulary); fetch more and retry.
#[wasm_bindgen]
pub struct Qwen35StageFetch {
    /// The stage's tensors, assembled as their bytes arrive.
    partial: Option<rlx_gguf::PartialGguf>,
    ranges: Vec<f64>,
    layers: Vec<u32>,
    added: usize,
    index: usize,
    count: usize,
}

#[wasm_bindgen]
impl Qwen35StageFetch {
    #[wasm_bindgen(constructor)]
    pub fn new(prefix: &[u8], index: usize, count: usize) -> Result<Qwen35StageFetch, JsError> {
        let header = rlx_gguf::GgufFile::header_from_bytes(prefix)
            .map_err(|e| JsError::new(&format!("model.gguf header: {e}")))?;
        // Fails here, not after the download, on a stage the model cannot have.
        let keep = rlx_qwen35::pipeline_stage_keeps(&header, index, count).map_err(js_err)?;
        let cfg = rlx_qwen35::Qwen35Config::from_gguf(&header).map_err(js_err)?;
        let layers = rlx_qwen35::pipeline_stage_layers(&cfg, index, count).map_err(js_err)?;
        let ranges = header
            .tensor_ranges(&keep, 1 << 20)
            .map_err(js_err)?
            .into_iter()
            .flat_map(|r| [r.start as f64, r.end as f64])
            .collect();
        // The stage's compact data segment, allocated once: downloaded pieces
        // are copied straight into it (`add`), so the worker holds its weights
        // once rather than once per piece plus once assembled.
        let partial = header.begin_partial(&keep).map_err(js_err)?;
        Ok(Qwen35StageFetch {
            partial: Some(partial),
            ranges,
            layers: vec![layers.start as u32, layers.end as u32],
            added: 0,
            index,
            count,
        })
    }

    /// File byte ranges to fetch, flattened `[start, end, …]` (end exclusive;
    /// an HTTP `Range` header wants `end - 1`). Neighbours less than 1 MB apart
    /// are merged. Fetch them in pieces of any size: `add` takes each piece.
    pub fn ranges(&self) -> Vec<f64> {
        self.ranges.clone()
    }

    /// The trunk layers this stage runs, `[start, end)`.
    pub fn layers(&self) -> Vec<u32> {
        self.layers.clone()
    }

    /// Hand over the bytes of one piece, starting at file offset `start`; only
    /// the stage's tensor bytes in it are kept.
    pub fn add(&mut self, start: f64, bytes: &[u8]) -> Result<(), JsError> {
        let p = self
            .partial
            .as_mut()
            .ok_or_else(|| JsError::new("Qwen35StageFetch already used"))?;
        p.feed(start as u64, bytes);
        self.added += bytes.len();
        Ok(())
    }

    /// Bytes added so far.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.added
    }
}

/// The parts of the FLUX.1 web GGUF (`rlx-genmoji --export-web`) one
/// denoiser stage reads, assembled as they arrive — `Qwen35StageFetch` for
/// FLUX: fetch `ranges()` with HTTP `Range`, `add` each piece, then
/// `FluxStage.from_fetch`.
#[wasm_bindgen]
pub struct FluxStageFetch {
    partial: Option<rlx_gguf::PartialGguf>,
    cfg: rlx_flux1::Flux1Config,
    parts: Vec<rlx_flux1::DitPart>,
    ranges: Vec<f64>,
    added: usize,
}

#[wasm_bindgen]
impl FluxStageFetch {
    /// `prefix`: the file's first bytes, enough to hold its header.
    #[wasm_bindgen(constructor)]
    pub fn new(prefix: &[u8], index: usize, count: usize) -> Result<FluxStageFetch, JsError> {
        let header = rlx_gguf::GgufFile::header_from_bytes(prefix)
            .map_err(|e| JsError::new(&format!("flux GGUF header: {e}")))?;
        let cfg = rlx_flux1::web::config_from_meta(&header).map_err(js_err)?;
        let parts = rlx_flux1::web::stage_parts(&cfg, index, count).map_err(js_err)?;
        let keep = |name: &str| rlx_flux1::web::stage_keeps(&parts, name);
        let ranges = header
            .tensor_ranges(keep, 1 << 20)
            .map_err(js_err)?
            .into_iter()
            .flat_map(|r| [r.start as f64, r.end as f64])
            .collect();
        let partial = header.begin_partial(&keep).map_err(js_err)?;
        Ok(FluxStageFetch {
            partial: Some(partial),
            cfg,
            parts,
            ranges,
            added: 0,
        })
    }

    /// File byte ranges to fetch, flattened `[start, end, …]` (end exclusive).
    pub fn ranges(&self) -> Vec<f64> {
        self.ranges.clone()
    }

    /// The blocks this stage runs, e.g. `Embed,Double(0),…`.
    pub fn parts(&self) -> String {
        self.parts
            .iter()
            .map(|p| format!("{p:?}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Hand over one fetched piece starting at file offset `start`.
    pub fn add(&mut self, start: f64, bytes: &[u8]) -> Result<(), JsError> {
        let p = self
            .partial
            .as_mut()
            .ok_or_else(|| JsError::new("FluxStageFetch already used"))?;
        p.feed(start as u64, bytes);
        self.added += bytes.len();
        Ok(())
    }

    /// Bytes added so far.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.added
    }
}

/// One stage of the FLUX.1 denoiser, compiled from its `FluxStageFetch`.
/// The first stage takes packed latents, T5 states and the conditioning
/// inputs and computes the conditioning vector; every stage hands its hidden
/// rows (and that vector) to the next; the last returns the velocity.
#[wasm_bindgen]
pub struct FluxStage {
    stage: Rc<RefCell<rlx_flux1::web::Flux1DitStage>>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))] // read by the async methods only
    busy: Rc<std::cell::Cell<bool>>,
    part_ms: Rc<RefCell<Vec<f64>>>,
}

#[wasm_bindgen]
impl FluxStage {
    /// Empties `fetch`. `txt_len` T5 positions, a `grid_h × grid_w` grid of
    /// 2×2 latent patches (160 px: 10 × 10); `group` blocks per graph (1).
    pub fn from_fetch(
        fetch: &mut FluxStageFetch,
        backend: Option<String>,
        txt_len: usize,
        grid_h: usize,
        grid_w: usize,
        group: Option<usize>,
    ) -> Result<FluxStage, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let file = fetch
            .partial
            .take()
            .ok_or_else(|| JsError::new("FluxStageFetch already used"))?
            .finish()
            .map_err(js_err)?;
        let geo = rlx_flux1::DitGeometry {
            txt_len,
            grid_h,
            grid_w,
        };
        let mut replay = rlx_flux1::web::Replay::new(file);
        let stage = rlx_flux1::web::Flux1DitStage::compile(
            &fetch.cfg,
            geo,
            &mut replay,
            device,
            fetch.parts.clone(),
            group.unwrap_or(1),
        )
        .map_err(js_err)?;
        Ok(FluxStage {
            stage: Rc::new(RefCell::new(stage)),
            busy: Rc::default(),
            part_ms: Rc::default(),
        })
    }

    /// Whether this stage starts the denoiser.
    pub fn is_first(&self) -> bool {
        self.stage.borrow().is_first()
    }

    /// The conditioning vector (first stage only).
    pub fn vec(&self, t: f32, guidance: f32, pooled: &[f32]) -> Result<Vec<f32>, JsError> {
        self.stage.borrow().vec(t, guidance, pooled).map_err(js_err)
    }

    /// Run the stage: `input` is the packed latents (first stage, with `txt`)
    /// or the previous stage's hidden rows. Resolves to a `Float32Array`;
    /// `part_ms()` then holds each part's time.
    #[allow(clippy::await_holding_refcell_ref)]
    #[cfg(target_arch = "wasm32")]
    pub fn forward_async(&self, input: Vec<f32>, txt: Vec<f32>, vec: Vec<f32>) -> js_sys::Promise {
        let (stage, busy, times) = (self.stage.clone(), self.busy.clone(), self.part_ms.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let mut last = js_sys::Date::now();
            let mut ms = Vec::new();
            // `busy` keeps a second call out, so this borrow is never contended.
            let out = stage
                .borrow_mut()
                .forward_async(&input, &txt, &vec, &mut |_| {
                    let now = js_sys::Date::now();
                    ms.push(now - last);
                    last = now;
                })
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            *times.borrow_mut() = ms;
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        })
    }

    /// Milliseconds each part took in the last `forward_async`.
    pub fn part_ms(&self) -> Vec<f64> {
        self.part_ms.borrow().clone()
    }
}

/// The parts of the web T5 GGUF one encoder stage reads (`FluxStageFetch`
/// for T5): fetch `ranges()`, `add` each piece, then `T5Stage.from_fetch`.
#[wasm_bindgen]
pub struct T5StageFetch {
    partial: Option<rlx_gguf::PartialGguf>,
    ranges: Vec<f64>,
    index: usize,
    count: usize,
}

#[wasm_bindgen]
impl T5StageFetch {
    #[wasm_bindgen(constructor)]
    pub fn new(prefix: &[u8], index: usize, count: usize) -> Result<T5StageFetch, JsError> {
        let header = rlx_gguf::GgufFile::header_from_bytes(prefix)
            .map_err(|e| JsError::new(&format!("T5 GGUF header: {e}")))?;
        let (cfg, embed_key) = rlx_flux1::web::t5_config_from_meta(&header).map_err(js_err)?;
        let layers = rlx_flux1::web::t5_stage_layers(&cfg, index, count).map_err(js_err)?;
        let keep = |n: &str| rlx_flux1::web::t5_stage_keeps(&cfg, &embed_key, &layers, n);
        let ranges = header
            .tensor_ranges(keep, 1 << 20)
            .map_err(js_err)?
            .into_iter()
            .flat_map(|r| [r.start as f64, r.end as f64])
            .collect();
        let partial = header.begin_partial(&keep).map_err(js_err)?;
        Ok(T5StageFetch {
            partial: Some(partial),
            ranges,
            index,
            count,
        })
    }

    /// File byte ranges to fetch, flattened `[start, end, …]` (end exclusive).
    pub fn ranges(&self) -> Vec<f64> {
        self.ranges.clone()
    }

    /// Hand over one fetched piece starting at file offset `start`.
    pub fn add(&mut self, start: f64, bytes: &[u8]) -> Result<(), JsError> {
        self.partial
            .as_mut()
            .ok_or_else(|| JsError::new("T5StageFetch already used"))?
            .feed(start as u64, bytes);
        Ok(())
    }
}

/// One stage of the T5 encoder. The first takes token ids, every stage hands
/// its hidden rows to the next; the last's output is the T5 states.
#[wasm_bindgen]
pub struct T5Stage {
    stage: Rc<RefCell<rlx_flux1::web::T5WebStage>>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))] // read by the async methods only
    busy: Rc<std::cell::Cell<bool>>,
}

#[wasm_bindgen]
impl T5Stage {
    /// Empties `fetch`; `len` token positions.
    pub fn from_fetch(
        fetch: &mut T5StageFetch,
        backend: Option<String>,
        len: usize,
    ) -> Result<T5Stage, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let file = fetch
            .partial
            .take()
            .ok_or_else(|| JsError::new("T5StageFetch already used"))?
            .finish()
            .map_err(js_err)?;
        let stage =
            rlx_flux1::web::T5WebStage::compile(file, len, fetch.index, fetch.count, device)
                .map_err(js_err)?;
        Ok(T5Stage {
            stage: Rc::new(RefCell::new(stage)),
            busy: Rc::default(),
        })
    }

    pub fn is_first(&self) -> bool {
        self.stage.borrow().is_first()
    }

    /// Embedding rows for `ids` (first stage only).
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>, JsError> {
        self.stage.borrow().embed(ids).map_err(js_err)
    }

    /// Run the stage's layers on hidden rows; resolves to a `Float32Array`.
    #[allow(clippy::await_holding_refcell_ref)]
    #[cfg(target_arch = "wasm32")]
    pub fn forward_async(&self, x: Vec<f32>) -> js_sys::Promise {
        let (stage, busy) = (self.stage.clone(), self.busy.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let out = stage
                .borrow_mut()
                .forward_async(&x)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        })
    }
}

/// CLIP (pooled embedding) and the VAE decoder from the web `clip-vae.gguf`.
#[wasm_bindgen]
pub struct AuxStage {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))] // read by the async methods only
    stage: Rc<RefCell<rlx_flux1::web::AuxStage>>,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))] // read by the async methods only
    busy: Rc<std::cell::Cell<bool>>,
}

#[wasm_bindgen]
impl AuxStage {
    /// `files` holds `clip-vae.gguf`; the VAE decodes `latent_h × latent_w`
    /// latents (160 px: 20 × 20).
    #[wasm_bindgen(constructor)]
    pub fn new(
        files: &mut Files,
        backend: Option<String>,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<AuxStage, JsError> {
        let device = parse_backend(backend.as_deref())?;
        let file = rlx_gguf::GgufFile::from_bytes(files.take_file("clip-vae.gguf")?)
            .map_err(|e| JsError::new(&format!("clip-vae.gguf: {e}")))?;
        let stage =
            rlx_flux1::web::AuxStage::compile(file, latent_h, latent_w, device).map_err(js_err)?;
        Ok(AuxStage {
            stage: Rc::new(RefCell::new(stage)),
            busy: Rc::default(),
        })
    }

    /// Pooled CLIP embedding for 77 ids; resolves to a `Float32Array`.
    #[allow(clippy::await_holding_refcell_ref)]
    #[cfg(target_arch = "wasm32")]
    pub fn pooled_async(&self, ids: Vec<u32>) -> js_sys::Promise {
        let (stage, busy) = (self.stage.clone(), self.busy.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let out = stage
                .borrow_mut()
                .pooled_async(&ids)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        })
    }

    /// Unpacked latents `[16, h, w]` → RGB8 (row-major, 8× the latent size);
    /// resolves to a `Uint8Array`.
    #[allow(clippy::await_holding_refcell_ref)]
    #[cfg(target_arch = "wasm32")]
    pub fn decode_async(&self, latents: Vec<f32>) -> js_sys::Promise {
        let (stage, busy) = (self.stage.clone(), self.busy.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let _guard = Busy::claim(&busy).map_err(JsValue::from)?;
            let out = stage
                .borrow_mut()
                .decode_rgb8_async(&latents)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            Ok(js_sys::Uint8Array::from(out.as_slice()).into())
        })
    }
}

/// FLUX.1's two tokenizers, as `rlx-flux1` pads and truncates their ids.
#[wasm_bindgen]
pub struct FluxTokenizers {
    t5: tokenizers::Tokenizer,
    clip: tokenizers::Tokenizer,
}

#[wasm_bindgen]
impl FluxTokenizers {
    /// The two `tokenizer.json` files' bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(t5_json: &[u8], clip_json: &[u8]) -> Result<FluxTokenizers, JsError> {
        let load = |b: &[u8]| {
            tokenizers::Tokenizer::from_bytes(b).map_err(|e| JsError::new(&format!("{e}")))
        };
        Ok(FluxTokenizers {
            t5: load(t5_json)?,
            clip: load(clip_json)?,
        })
    }

    /// T5 ids: `</s>`-terminated, padded or truncated to `len`.
    pub fn t5_ids(&self, text: &str, len: usize) -> Result<Vec<u32>, JsError> {
        rlx_flux1::t5::t5_ids(&self.t5, text, len).map_err(js_err)
    }

    /// CLIP ids: 77 positions, padded with `<|endoftext|>` (49407).
    pub fn clip_ids(&self, text: &str) -> Result<Vec<u32>, JsError> {
        rlx_flux1::clip::clip_ids(&self.clip, text, 77, 49407).map_err(js_err)
    }
}

/// Gaussian noise for a FLUX.1 latent (`rlx_flux1::sampler::gaussian_noise`).
#[wasm_bindgen]
pub fn flux1_noise(n: usize, seed: f64) -> Vec<f32> {
    rlx_flux1::sampler::gaussian_noise(n, seed as u64)
}

/// `[c, h, w]` latents → the denoiser's packed `[(h/2)(w/2), 4c]` rows.
#[wasm_bindgen]
pub fn flux1_pack(latents: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    rlx_flux1::sampler::pack(latents, c, h, w)
}

/// Packed rows → `[c, h, w]` latents.
#[wasm_bindgen]
pub fn flux1_unpack(packed: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    rlx_flux1::sampler::unpack(packed, c, h, w)
}

/// The sampler's `steps + 1` timesteps for `img_len` image positions (`shift`
/// for FLUX.1 [dev]).
#[wasm_bindgen]
pub fn flux1_schedule(steps: usize, img_len: usize, shift: bool) -> Vec<f32> {
    rlx_flux1::sampler::schedule(steps, img_len, shift)
}

/// A stage input copied out of its JS typed array.
enum StageIn {
    Tokens(Vec<u32>),
    Hidden(Vec<f32>),
}

impl StageIn {
    fn from_js(v: &JsValue) -> Result<Self, JsError> {
        if let Some(a) = v.dyn_ref::<js_sys::Uint32Array>() {
            Ok(StageIn::Tokens(a.to_vec()))
        } else if let Some(a) = v.dyn_ref::<js_sys::Float32Array>() {
            Ok(StageIn::Hidden(a.to_vec()))
        } else {
            Err(JsError::new(
                "stage input: a Uint32Array of token ids or a Float32Array of hidden rows",
            ))
        }
    }

    fn as_input(&self) -> rlx_qwen35::StageInput<'_> {
        match self {
            StageIn::Tokens(t) => rlx_qwen35::StageInput::Tokens(t),
            StageIn::Hidden(h) => rlx_qwen35::StageInput::Hidden(h),
        }
    }
}

fn stage_out_to_js(out: rlx_qwen35::StageOutput) -> JsValue {
    match out {
        rlx_qwen35::StageOutput::Token(t) => JsValue::from_f64(f64::from(t)),
        rlx_qwen35::StageOutput::Tokens(t) => js_sys::Uint32Array::from(t.as_slice()).into(),
        rlx_qwen35::StageOutput::Hidden(h) => js_sys::Float32Array::from(h.as_slice()).into(),
    }
}

/// A Hugging Face `tokenizer.json`, for a page that drives a model in pieces
/// (`Qwen35Pipeline` tokenizes here and passes ids to its stage workers).
#[wasm_bindgen]
pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
}

#[wasm_bindgen]
impl TextTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(tokenizer_json: Vec<u8>) -> Result<TextTokenizer, JsError> {
        let inner = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| JsError::new(&format!("tokenizer.json: {e}")))?;
        Ok(TextTokenizer { inner })
    }

    /// Token ids of `text`, special tokens as written (no BOS added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, JsError> {
        encode(&self.inner, text).map_err(|e| JsError::new(&e))
    }

    /// Text of `ids`, special tokens skipped.
    pub fn decode(&self, ids: &[u32]) -> Result<String, JsError> {
        decode(&self.inner, ids).map_err(|e| JsError::new(&e))
    }

    /// The id of a token string such as `"<|im_end|>"`, if it has one.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// Keep the Qwen3.5 prefill graph between prompts (`RLX_QWEN35_KEEP_PREFILL`)
/// unless the page set it, when this worker's weights are under 1 GiB. Dropped,
/// it is rebuilt on every prompt: warm prefill measured 1.28 s against 0.63 s
/// kept (Q4_K_M, one worker), and a pipeline rebuilds stage after stage. Each
/// compiled graph holds its own copy of the packed weights on the GPU, so a
/// larger checkpoint (BF16, 1.5 GB) keeps the default and frees it.
fn keep_prefill_by_default(file: &rlx_gguf::GgufFile, index: usize, count: usize) {
    if rlx_ir::env::var("RLX_QWEN35_KEEP_PREFILL").is_some() {
        return;
    }
    let Ok(keep) = rlx_qwen35::pipeline_stage_keeps(file, index, count) else {
        return;
    };
    let bytes: usize = file
        .tensors
        .values()
        .filter(|t| keep(&t.name))
        .filter_map(|t| rlx_gguf::bytes_for_public(t.dtype, t.n_elements()))
        .sum();
    if bytes < 1 << 30 {
        rlx_ir::env::set("RLX_QWEN35_KEEP_PREFILL", "1");
    }
}

fn encode(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    tok.encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| format!("tokenize: {e}"))
}

fn decode(tok: &tokenizers::Tokenizer, ids: &[u32]) -> Result<String, String> {
    tok.decode(ids, true)
        .map_err(|e| format!("detokenize: {e}"))
}

/// Holds the in-flight flag for the length of one run.
struct Busy(Rc<std::cell::Cell<bool>>);

impl Busy {
    fn claim(flag: &Rc<std::cell::Cell<bool>>) -> Result<Self, JsError> {
        if flag.replace(true) {
            return Err(JsError::new("a generation is already in flight"));
        }
        Ok(Busy(flag.clone()))
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        self.0.set(false);
    }
}
