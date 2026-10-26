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

//! What a model is exported from: a Hugging Face checkpoint directory or a
//! GGUF, and everything a package takes from it besides its graphs.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::package_export::Referencer;
use crate::weight_loader::WeightLoader;

/// How a checkpoint stores its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// A Hugging Face directory: `config.json`, `model.safetensors` (or its
    /// shards), `tokenizer.json`, `tokenizer_config.json`, ….
    Safetensors,
    /// One `.gguf`, its config and tokenizer in its metadata (a
    /// `tokenizer.json` beside it wins).
    Gguf,
}

/// A checkpoint to export from.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// The directory, the GGUF file, or a lone `.safetensors`.
    pub path: PathBuf,
    pub format: Format,
    /// `config.json`, or the GGUF's metadata (`general.*`, `<arch>.*`).
    pub config: Value,
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(path).with_context(|| format!("{}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

fn ids(v: Option<&Value>) -> Vec<u64> {
    match v {
        Some(Value::Number(n)) => n.as_u64().into_iter().collect(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).collect(),
        _ => Vec::new(),
    }
}

/// A GGUF metadata value as JSON; `None` for a long array (the vocabulary,
/// its merges and scores), which a config has no use for.
fn meta_json(v: &rlx_gguf::MetaValue) -> Option<Value> {
    use rlx_gguf::MetaValue as M;
    Some(match v {
        M::U8(x) => json!(x),
        M::I8(x) => json!(x),
        M::U16(x) => json!(x),
        M::I16(x) => json!(x),
        M::U32(x) => json!(x),
        M::I32(x) => json!(x),
        M::U64(x) => json!(x),
        M::I64(x) => json!(x),
        M::F32(x) => json!(x),
        M::F64(x) => json!(x),
        M::Bool(x) => json!(x),
        M::String(s) => json!(s),
        M::Array(a) if a.len() <= 64 => Value::Array(a.iter().filter_map(meta_json).collect()),
        M::Array(_) => return None,
    })
}

/// A `tokenizer_config.json` special token: a string, or `{content: …}`.
fn token(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content")?.as_str().map(String::from),
        _ => None,
    }
}

impl Checkpoint {
    /// A Hugging Face directory (or a file in one; an OpenCLIP directory),
    /// or a `.gguf`.
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_dir() {
            // OpenCLIP's layout names its config `open_clip_config.json`.
            let open_clip = path.join("open_clip_config.json");
            let config = if !path.join("config.json").is_file() && open_clip.is_file() {
                read_json(&open_clip)?
            } else {
                read_json(&path.join("config.json"))?
            };
            return Ok(Self {
                path: path.to_path_buf(),
                format: Format::Safetensors,
                config,
            });
        }
        if path.extension().is_some_and(|e| e == "gguf") {
            let raw = rlx_gguf::GgufFile::from_path(path)
                .with_context(|| format!("open {}", path.display()))?;
            let config = Value::Object(
                raw.metadata
                    .iter()
                    .filter_map(|(k, v)| meta_json(v).map(|v| (k.clone(), v)))
                    .collect(),
            );
            return Ok(Self {
                path: path.to_path_buf(),
                format: Format::Gguf,
                config,
            });
        }
        match path.parent() {
            Some(dir) if path.is_file() && dir.join("config.json").is_file() => Self::open(dir),
            // One safetensors file and no config (a codec's weights): what
            // the tensors say is all there is.
            _ if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") => {
                Ok(Self {
                    path: path.to_path_buf(),
                    format: Format::Safetensors,
                    config: serde_json::json!({}),
                })
            }
            _ => bail!(
                "{}: not a checkpoint directory (no config.json) or a .gguf",
                path.display()
            ),
        }
    }

    /// The directory beside which its files are: the checkpoint's own, or
    /// the GGUF's.
    pub fn dir(&self) -> &Path {
        match self.format {
            Format::Safetensors if self.path.is_dir() => &self.path,
            _ => self.path.parent().unwrap_or(Path::new(".")),
        }
    }

    /// `config.json`'s `model_type`, or the GGUF's `general.architecture`.
    pub fn model_type(&self) -> Option<&str> {
        match self.format {
            Format::Safetensors => self.config.get("model_type"),
            Format::Gguf => self.config.get("general.architecture"),
        }
        .and_then(Value::as_str)
    }

    /// `config.json`'s `architectures`.
    pub fn architectures(&self) -> Vec<&str> {
        self.config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default()
    }

    /// A JSON file of the checkpoint's (`tokenizer_config.json`, …); `Null`
    /// when it has none.
    pub fn json(&self, file: &str) -> Value {
        read_json(&self.dir().join(file)).unwrap_or(Value::Null)
    }

    /// The package's name: the directory's, or the GGUF's `general.name`.
    pub fn name(&self) -> String {
        if self.format == Format::Gguf
            && let Some(n) = self.config.get("general.name").and_then(Value::as_str)
        {
            return n.to_string();
        }
        let p = match self.format {
            Format::Safetensors if self.path.is_dir() => &self.path,
            _ => &self.path.with_extension(""),
        };
        p.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model")
            .to_string()
    }

    /// The safetensors files: `model.safetensors`, or the shards
    /// `model.safetensors.index.json` lists. A GGUF has none.
    pub fn weight_files(&self) -> Result<Vec<PathBuf>> {
        if self.format == Format::Gguf {
            return Ok(Vec::new());
        }
        if self.path.is_file() {
            return Ok(vec![self.path.clone()]);
        }
        let dir = &self.path;
        let single = dir.join("model.safetensors");
        if single.is_file() {
            return Ok(vec![single]);
        }
        let open_clip = dir.join("open_clip_model.safetensors");
        if open_clip.is_file() {
            return Ok(vec![open_clip]);
        }
        let index = read_json(&dir.join("model.safetensors.index.json"))
            .context("no model.safetensors, and no shard index")?;
        let mut files: Vec<String> = index
            .get("weight_map")
            .and_then(Value::as_object)
            .context("model.safetensors.index.json has no weight_map")?
            .values()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect();
        files.sort();
        files.dedup();
        Ok(files.into_iter().map(|f| dir.join(f)).collect())
    }

    /// The weights as a builder takes them, each tensor read from its file
    /// when taken: call once per graph, so a build holds only the params it
    /// makes. A GGUF's tensors are named as the Hugging Face checkpoint
    /// names them.
    pub fn loader(&self) -> Result<Box<dyn WeightLoader>> {
        Ok(match self.format {
            Format::Safetensors if self.path.is_file() => Box::new(
                crate::safetensors_checkpoint::SafetensorsMmapLoader::open_file(&self.path)?,
            ),
            Format::Safetensors => Box::new(
                crate::safetensors_checkpoint::SafetensorsMmapLoader::open(&self.path)?,
            ),
            Format::Gguf => Box::new(crate::weight_loader::GgufLoader::from_file(
                self.path.to_str().context("GGUF path is not UTF-8")?,
            )?),
        })
    }

    /// The referencer for this checkpoint's weights: a param equal to one
    /// of its tensors (as stored, or transposed) is read from it at open;
    /// `hold_all` keeps every param in the package instead.
    pub fn referencer(&self, hold_all: bool) -> Result<Referencer> {
        match self.format {
            Format::Safetensors => Referencer::open(&self.weight_files()?, hold_all),
            Format::Gguf => Referencer::open_gguf(&self.path, hold_all),
        }
    }

    /// `tokenizer.json`'s bytes: beside the checkpoint, or (a GGUF) built
    /// from its metadata by `from_gguf`, which a family whose GGUFs carry a
    /// tokenizer supplies.
    pub fn tokenizer(
        &self,
        from_gguf: Option<&dyn Fn(&Path) -> Result<Vec<u8>>>,
    ) -> Result<Vec<u8>> {
        let beside = self.dir().join("tokenizer.json");
        if beside.is_file() {
            return std::fs::read(&beside).with_context(|| format!("{}", beside.display()));
        }
        match (self.format, from_gguf) {
            (Format::Gguf, Some(build)) => build(&self.path),
            (Format::Gguf, None) => bail!(
                "{}: no tokenizer.json beside the GGUF, and this family builds none from it",
                self.path.display()
            ),
            (Format::Safetensors, _) => bail!("{}: no tokenizer.json", self.path.display()),
        }
    }

    /// The chat template: `tokenizer_config.json`'s, `chat_template.jinja`,
    /// or the GGUF's `tokenizer.chat_template`.
    pub fn chat_template(&self) -> Option<String> {
        self.json("tokenizer_config.json")
            .get("chat_template")
            .and_then(Value::as_str)
            .map(String::from)
            .or_else(|| std::fs::read_to_string(self.dir().join("chat_template.jinja")).ok())
            .or_else(|| {
                self.config
                    .get("tokenizer.chat_template")
                    .and_then(Value::as_str)
                    .map(String::from)
            })
    }

    /// The chat block of a `generate` entry (template, BOS and EOS tokens)
    /// and the sidecars it names.
    pub fn chat(&self) -> (Value, Vec<(String, String, Vec<u8>)>) {
        let mut chat = json!({});
        let mut sidecars = Vec::new();
        if let Some(t) = self.chat_template() {
            sidecars.push((
                "chat_template.jinja".to_string(),
                "text/plain".to_string(),
                t.into_bytes(),
            ));
            chat["chat_template"] = json!("sidecar:chat_template.jinja");
        }
        let tok_cfg = self.json("tokenizer_config.json");
        for k in ["bos_token", "eos_token"] {
            if let Some(t) = token(tok_cfg.get(k)) {
                chat[k] = json!(t);
            }
        }
        (chat, sidecars)
    }

    /// The ids generation stops at: `generation_config.json`'s and
    /// `config.json`'s `eos_token_id`, or the GGUF's EOS and EOT.
    pub fn stop_ids(&self) -> Vec<u64> {
        let mut stop = ids(self.json("generation_config.json").get("eos_token_id"));
        let mut more = ids(self.config.get("eos_token_id"));
        more.extend(ids(self.config.get("tokenizer.ggml.eos_token_id")));
        more.extend(ids(self.config.get("tokenizer.ggml.eot_token_id")));
        if let Some(text) = self.config.get("text_config") {
            more.extend(ids(text.get("eos_token_id")));
        }
        for id in more {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
        stop
    }

    /// The checkpoint's sampling defaults (`generation_config.json`), for a
    /// `generate` entry's `generation`; `Null` when it has none.
    pub fn generation(&self) -> Value {
        let gen_cfg = self.json("generation_config.json");
        let g: serde_json::Map<String, Value> = [
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "max_new_tokens",
        ]
        .iter()
        .filter_map(|k| gen_cfg.get(*k).map(|v| (k.to_string(), v.clone())))
        .collect();
        if g.is_empty() {
            Value::Null
        } else {
            Value::Object(g)
        }
    }

    /// The model card: `README.md` beside the checkpoint.
    pub fn readme(&self) -> Option<String> {
        std::fs::read_to_string(self.dir().join("README.md")).ok()
    }

    /// The license a GGUF states (`general.license`).
    pub fn license(&self) -> Option<&str> {
        self.config.get("general.license").and_then(Value::as_str)
    }
}
