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

//! rlx-models as an rlx **model plugin**: one provider, `rlx-models`, that
//! opens any checkpoint `rlx_models::run::auto_runner` can sniff (GGUF or
//! safetensors, every registered language-model family) as an
//! `rlx_runtime::model::Model`. `librlx_capi` loads the plugin at run time,
//! so the model is open to every rlx language binding:
//!
//! ```text
//! cargo build -p rlx-models-plugin --release
//! export RLX_MODEL_PLUGINS=$PWD/target/release/librlx_models_plugin.dylib
//! python -c 'import pyrlx; m = pyrlx.Model.open("Qwen3-0.6B-Q4_K_M.gguf"); print(m.generate_text("Hi", max_tokens=16))'
//! ```
//!
//! A new family needs nothing here: register its `LmRunner` constructor with
//! `rlx_core::lm_registry::register_lm_runner` (as every family in
//! `rlx_models::run` does) and `auto_runner` — and so this plugin — opens it.
//! A task beyond text generation (speech, vision, embeddings) is one more
//! [`ModelProvider`] in this crate, registered the same way.
//!
//! Open options (`Model.open(path, {…})`): `device`, `max_seq` (context in
//! tokens), `tokenizer` (a `tokenizer.json`; default: next to the weights),
//! `mmproj` (a vision encoder, for families that take one).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_models::run::{AutoOptions, auto_runner_with};
use rlx_runtime::model::{
    ChatMessage, Model, ModelInfo, ModelProvider, ModelRequest, TextCodec, lm_model,
};
use serde_json::{Value, json};

/// Special tokens that end a turn in the families rlx-models runs.
const STOP_TOKENS: &[&str] = &[
    "<|endoftext|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|end|>",
    "<end_of_turn>",
    "<eos>",
    "</s>",
];

/// A `tokenizer.json` as the [`TextCodec`] of a language model.
struct Tokenizer {
    handle: rlx_text::TokenizerHandle,
    stops: Vec<u32>,
    /// The model's chat template, for `chat`.
    template: Option<rlx_text::chat::ChatTemplate>,
}

impl Tokenizer {
    fn load(path: &Path, weights: &Path) -> Result<Self> {
        let handle = rlx_text::load_tokenizer(path)?;
        let stops = STOP_TOKENS
            .iter()
            .filter_map(|t| handle.raw().token_to_id(t))
            .collect();
        Ok(Self {
            handle,
            stops,
            template: chat_template(path, weights),
        })
    }
}

/// The chat template: a GGUF's `tokenizer.chat_template`, else the
/// `chat_template` of a `tokenizer_config.json` beside the tokenizer.
fn chat_template(tokenizer: &Path, weights: &Path) -> Option<rlx_text::chat::ChatTemplate> {
    let is_gguf = weights
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    if is_gguf && let Ok(t) = rlx_text::chat::ChatTemplate::from_gguf(weights) {
        return Some(t);
    }
    let config = tokenizer.with_file_name("tokenizer_config.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(config).ok()?).ok()?;
    rlx_text::chat::ChatTemplate::from_source(v.get("chat_template")?.as_str()?).ok()
}

impl TextCodec for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.handle.encode(text, true)
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        self.handle.encode(text, false)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.handle.decode(ids, true)
    }

    fn stop_ids(&self) -> Vec<u32> {
        self.stops.clone()
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let Some(t) = &self.template else {
            anyhow::bail!(
                "this model has no chat template (none in the GGUF, and no tokenizer_config.json \
                 beside the tokenizer); use generate_text with a prompt"
            );
        };
        let turns: Vec<rlx_text::chat::ChatMessage> = messages
            .iter()
            .map(|m| rlx_text::chat::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        t.render(&turns, true)
    }

    fn apply_chat_template_json(&self, messages: &[Value], options: &Value) -> Result<String> {
        let Some(t) = &self.template else {
            anyhow::bail!("this model has no chat template; use generate_text with a prompt");
        };
        rlx_text::model_codec::render_chat_json(t, messages, options)
    }
}

/// `options.tokenizer`, else a `tokenizer.json` in the weights directory or
/// beside the weights file.
fn tokenizer_path(weights: &Path, options: &Value) -> Option<PathBuf> {
    if let Some(p) = options.get("tokenizer").and_then(Value::as_str) {
        return Some(PathBuf::from(p));
    }
    let dir = if weights.is_dir() {
        weights
    } else {
        weights.parent()?
    };
    let p = dir.join("tokenizer.json");
    p.is_file().then_some(p)
}

/// Opens the checkpoints `rlx_models::run::auto_runner` knows.
pub struct RlxModelsProvider;

impl ModelProvider for RlxModelsProvider {
    fn name(&self) -> &str {
        "rlx-models"
    }

    fn description(&self) -> &str {
        "rlx-models' language-model families (GGUF / safetensors, sniffed by architecture)"
    }

    fn tasks(&self) -> Vec<String> {
        vec!["text-generation".into()]
    }

    fn families(&self) -> Vec<String> {
        rlx_models::run::register_builtin_lm_runners();
        rlx_core::lm_registry::registered_lm_runners()
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn probe(&self, request: &ModelRequest) -> u32 {
        let path = request.path();
        if !path.exists() {
            return 0;
        }
        rlx_models::run::register_builtin_lm_runners();
        match rlx_cli::auto_sniff(path) {
            // A parsed architecture with a constructor: more specific than any
            // extension match.
            Ok(s) if rlx_core::lm_registry::lm_runner_is_registered(s.runner_name) => 80,
            _ => 0,
        }
    }

    fn open(&self, request: &ModelRequest) -> Result<Box<dyn Model>> {
        let path = request.path();
        let o = &request.options;
        let opts = AutoOptions {
            // `device` absent: let the family pick for this machine.
            device: o.get("device").map(|_| request.device),
            max_seq: o.get("max_seq").and_then(Value::as_u64).map(|n| n as usize),
            mmproj: o.get("mmproj").and_then(Value::as_str).map(PathBuf::from),
            ..AutoOptions::default()
        };
        let runner = auto_runner_with(path, &opts)
            .with_context(|| format!("rlx-models: opening {}", path.display()))?;
        let codec = match tokenizer_path(path, o) {
            Some(t) => Some(Box::new(
                Tokenizer::load(&t, path).with_context(|| format!("tokenizer {}", t.display()))?,
            ) as Box<dyn TextCodec>),
            None => None,
        };
        let device = opts
            .device
            .map(rlx_runtime::device_label)
            .unwrap_or("auto")
            .to_string();
        Ok(lm_model(
            runner,
            codec,
            ModelInfo {
                provider: "rlx-models".into(),
                device,
                metadata: json!({"weights": path.display().to_string()}),
                ..Default::default()
            },
        ))
    }
}

mod tasks;

rlx_runtime::export_model_plugin!(
    "rlx-models",
    [
        RlxModelsProvider,
        #[cfg(feature = "whisper")]
        tasks::whisper::WHISPER,
        #[cfg(feature = "embed")]
        tasks::embed::EMBED,
    ]
);

#[cfg(test)]
mod tests {
    use rlx_runtime::model::{self, plugin};
    use serde_json::json;

    #[test]
    fn the_plugin_exports_the_rlx_models_provider() {
        let mut names = unsafe { plugin::register_table(super::rlx_model_plugin_v1()) }.unwrap();
        names.sort();
        assert_eq!(
            names,
            ["rlx-models", "rlx-models-embed", "rlx-models-whisper"]
        );
        let list = model::providers();
        let p = list
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "rlx-models")
            .unwrap();
        assert!(
            p["families"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == "qwen3"),
            "{p}"
        );
        // A file no family claims is refused by name.
        let e = model::open_source("no-such-model.gguf", json!({}))
            .err()
            .unwrap();
        assert!(e.to_string().contains("no model provider claims"), "{e}");
    }

    /// A real checkpoint end to end: `RLX_TEST_LM=<gguf or model dir>`.
    #[test]
    fn a_checkpoint_generates_through_the_model_api() {
        let Ok(path) = rlx_ir::env::var_result("RLX_TEST_LM") else {
            eprintln!("skip: set RLX_TEST_LM to a checkpoint");
            return;
        };
        let mut m = model::open_source(&path, json!({"device": "cpu"})).unwrap();
        assert_eq!(m.info().task, "text-generation");
        let toks = m
            .run(
                &[rlx_runtime::model::Tensor::new(
                    "input_ids",
                    rlx_runtime::DType::I32,
                    vec![2],
                    [1i32, 42].iter().flat_map(|v| v.to_le_bytes()).collect(),
                )],
                &json!({"max_tokens": 4}),
            )
            .unwrap();
        assert_eq!(toks[0].shape, vec![4]);
    }

    /// Whisper end to end: `RLX_TEST_ASR=<checkpoint dir>`, on the JFK clip.
    #[test]
    fn whisper_transcribes_through_the_model_api() {
        let Ok(dir) = rlx_ir::env::var_result("RLX_TEST_ASR") else {
            eprintln!("skip: set RLX_TEST_ASR to a Whisper checkpoint directory");
            return;
        };
        let wav = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/harness/jfk_16k.wav"
        );
        let pcm = rlx_models::whisper::load_wav_mono_f32(std::path::Path::new(wav)).unwrap();
        let mut m = model::open_source(&dir, json!({"language": "en"})).unwrap();
        assert_eq!(m.info().provider, "rlx-models-whisper");
        let text = m.transcribe(&pcm, 16_000, &json!({})).unwrap();
        assert!(text.contains("what you can do for your country"), "{text}");
    }

    /// Sentence embeddings: `RLX_TEST_EMBED=<checkpoint dir>`.
    #[test]
    fn embeddings_rank_a_paraphrase_above_an_unrelated_text() {
        let Ok(dir) = rlx_ir::env::var_result("RLX_TEST_EMBED") else {
            eprintln!("skip: set RLX_TEST_EMBED to a BERT / XLM-R checkpoint directory");
            return;
        };
        let mut m = model::open_source(&dir, json!({})).unwrap();
        let v = m
            .embed(
                &[
                    "how do I bake bread",
                    "a recipe for baking a loaf",
                    "the stock market fell",
                ],
                &json!({}),
            )
            .unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&v[0], &v[1]) > dot(&v[0], &v[2]), "{v:?}");
    }
}
