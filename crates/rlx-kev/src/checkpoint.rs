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

//! A kev run directory: a LoRA adapter, `head.pt`, and the tokenizer.
//!
//! This is the one place that knows the on-disk layout. A released
//! checkpoint (`jaredpalmer/kev-4b` and friends) holds:
//!
//! ```text
//! adapter_config.json      peft LoRA config (r, alpha, target_modules)
//! adapter_model.safetensors
//! head.pt                  torch.save of the Meta dict + pointer head
//! tokenizer.json
//! training_config.json     the run's arguments (informational)
//! ```
//!
//! `head.pt` is read natively — it is a STORED zip, so the pickle root comes
//! out without libtorch or Python.
//!
//! A checkpoint trained by `crate::train` (feature `train`) writes `head.json` +
//! `head.safetensors` instead (round-tripping through a torch pickle would be
//! pointless), and both layouts load here.
//!
//! # Temperature is never guessed
//!
//! The calibration temperature exists only in `head.pt`. If it cannot be read
//! we fail instead of falling back to `1.0`: on new sources that fallback
//! moves Kev-9B's calibration error from 0.042 to 0.106 and more than doubles
//! its confident errors, while leaving every argmax unchanged — which is
//! exactly the kind of wrong that looks right in a smoke test.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use rlx_torch_ckpt::pickle::Value as PkValue;
use rlx_torch_ckpt::{PtModel, read_torch_pickle_root};

use crate::pointer::{DEFAULT_HEAD_DIM, PointerHead};

/// The contents of `head.pt` minus the tensors.
#[derive(Debug, Clone)]
pub struct Meta {
    pub base: String,
    pub base_revision: Option<String>,
    pub lora: usize,
    pub head_dim: usize,
    pub option_isolation: bool,
    pub special_embeddings: bool,
    pub weights_dtype: String,
    pub temperature: f32,
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            base: String::new(),
            base_revision: None,
            lora: 16,
            head_dim: DEFAULT_HEAD_DIM,
            option_isolation: false,
            special_embeddings: false,
            weights_dtype: "fp32".into(),
            temperature: 1.0,
        }
    }
}

/// An opened run directory.
pub struct Checkpoint {
    dir: PathBuf,
    meta: Meta,
}

impl Checkpoint {
    /// Open `dir`, reading `head.pt` for the metadata.
    pub fn open(dir: &Path) -> Result<Self> {
        let head_pt = dir.join("head.pt");
        let head_json = dir.join("head.json");
        let meta = if head_pt.exists() {
            let root = read_torch_pickle_root(&head_pt)
                .with_context(|| format!("reading {}", head_pt.display()))?;
            meta_from_pickle(&root)
                .with_context(|| format!("reading metadata from {}", head_pt.display()))?
        } else if head_json.exists() {
            let text = std::fs::read_to_string(&head_json)
                .with_context(|| format!("reading {}", head_json.display()))?;
            meta_from_json(&text)
                .with_context(|| format!("reading metadata from {}", head_json.display()))?
        } else {
            bail!(
                "{} is not a kev run directory (no head.pt or head.json)",
                dir.display()
            );
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            meta,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Override the calibration temperature (`KEV_TEMPERATURE` / `--temperature`).
    ///
    /// `1.0` gives the raw logits. Never changes an answer.
    pub fn set_temperature(&mut self, t: f32) -> Result<()> {
        if !(t.is_finite() && t > 0.0) {
            bail!("temperature must be finite and positive, got {t}");
        }
        self.meta.temperature = t;
        Ok(())
    }

    /// Path to `tokenizer.json`.
    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.join("tokenizer.json")
    }

    /// `adapter_config.json`, parsed.
    pub fn adapter_config(&self) -> Result<serde_json::Value> {
        let p = self.dir.join("adapter_config.json");
        let text =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", p.display()))
    }

    /// Load the pointer head, inferring `hidden` from the stored shapes.
    pub fn pointer_head(&self) -> Result<PointerHead> {
        let pt = self.dir.join("head.pt");
        let st = self.dir.join("head.safetensors");
        if !pt.exists() && st.exists() {
            return self.pointer_head_safetensors(&st);
        }
        let p = pt;
        let m = PtModel::open(&p).with_context(|| format!("opening {}", p.display()))?;

        let get = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let t = m
                .tensor(name)
                .with_context(|| format!("{}: missing tensor {name}", p.display()))?;
            Ok((t.data, t.shape))
        };
        let (qw, qs) = get("head.q.weight")?;
        let (qb, _) = get("head.q.bias")?;
        let (kw, ks) = get("head.k.weight")?;
        let (kb, _) = get("head.k.bias")?;

        if qs.len() != 2 || ks != qs {
            bail!(
                "pointer head weights should be two identical rank-2 tensors; \
                 got q={qs:?} k={ks:?}"
            );
        }
        let (head_dim, hidden) = (qs[0], qs[1]);
        if head_dim != self.meta.head_dim {
            bail!(
                "head.pt says head_dim={} but q.weight is [{head_dim}, {hidden}]",
                self.meta.head_dim
            );
        }
        PointerHead::new(qw, qb, kw, kb, hidden, head_dim, self.meta.temperature)
    }
}

fn entries(v: &PkValue) -> Option<Vec<(String, PkValue)>> {
    match v {
        PkValue::Dict(d) => Some(
            d.borrow()
                .iter()
                .filter_map(|(k, val)| match k {
                    PkValue::Str(s) => Some((s.to_string(), val.clone())),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

fn find<'a>(items: &'a [(String, PkValue)], key: &str) -> Option<&'a PkValue> {
    items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn as_string(v: &PkValue) -> Option<String> {
    match v {
        PkValue::Str(s) => Some(s.to_string()),
        _ => None,
    }
}

fn as_bool(v: &PkValue) -> Option<bool> {
    match v {
        PkValue::Bool(b) => Some(*b),
        PkValue::Int(i) => Some(*i != 0),
        _ => None,
    }
}

fn as_usize(v: &PkValue) -> Option<usize> {
    match v {
        PkValue::Int(i) => usize::try_from(*i).ok(),
        _ => None,
    }
}

fn as_f32(v: &PkValue) -> Option<f32> {
    match v {
        PkValue::Float(f) => Some(*f as f32),
        PkValue::Int(i) => Some(*i as f32),
        _ => None,
    }
}

/// Pull the scalar fields out of an unpickled `head.pt` root.
fn meta_from_pickle(root: &PkValue) -> Result<Meta> {
    let items = entries(root)
        .ok_or_else(|| anyhow!("head.pt root is not a dict (is this a kev checkpoint?)"))?;

    let mut meta = Meta {
        base: find(&items, "base")
            .and_then(as_string)
            .ok_or_else(|| anyhow!("head.pt has no `base`"))?,
        base_revision: find(&items, "base_revision").and_then(as_string),
        ..Meta::default()
    };
    if let Some(v) = find(&items, "lora").and_then(as_usize) {
        meta.lora = v;
    }
    if let Some(v) = find(&items, "head_dim").and_then(as_usize) {
        meta.head_dim = v;
    }
    if let Some(v) = find(&items, "option_isolation").and_then(as_bool) {
        meta.option_isolation = v;
    }
    if let Some(v) = find(&items, "special_embeddings").and_then(as_bool) {
        meta.special_embeddings = v;
    }
    if let Some(v) = find(&items, "weights_dtype").and_then(as_string) {
        meta.weights_dtype = v;
    }
    meta.temperature = find(&items, "temperature")
        .and_then(as_f32)
        .ok_or_else(|| {
            anyhow!(
                "head.pt has no `temperature`. Refusing to default to 1.0: the fitted \
             value is what makes the probabilities calibrated, and substituting 1.0 \
             leaves every argmax unchanged while roughly doubling the confident-error \
             rate. Pass an explicit temperature if this checkpoint predates calibration."
            )
        })?;
    if !(meta.temperature.is_finite() && meta.temperature > 0.0) {
        bail!("head.pt temperature is {}", meta.temperature);
    }
    Ok(meta)
}

impl Checkpoint {
    /// The `head.safetensors` form written by [`crate::train::export_head`].
    fn pointer_head_safetensors(&self, path: &Path) -> Result<PointerHead> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let get = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let v = st
                .tensor(name)
                .map_err(|e| anyhow!("{}: {name}: {e}", path.display()))?;
            let shape = v.shape().to_vec();
            let data = rlx_core::safetensors_checkpoint::tensor_view_to_f32(name, v)?;
            Ok((data, shape))
        };
        let (qw, qs) = get("head.q.weight")?;
        let (qb, _) = get("head.q.bias")?;
        let (kw, ks) = get("head.k.weight")?;
        let (kb, _) = get("head.k.bias")?;
        if qs.len() != 2 || ks != qs {
            bail!("head.safetensors: q/k must be identical rank-2 tensors, got {qs:?} / {ks:?}");
        }
        PointerHead::new(qw, qb, kw, kb, qs[1], qs[0], self.meta.temperature)
    }
}

/// Parse the `head.json` sidecar written by [`crate::train::export_head`].
fn meta_from_json(text: &str) -> Result<Meta> {
    let v: serde_json::Value = serde_json::from_str(text)?;
    let mut meta = Meta {
        base: v["base"].as_str().unwrap_or_default().to_string(),
        base_revision: v["base_revision"].as_str().map(str::to_string),
        ..Meta::default()
    };
    if let Some(n) = v["lora"].as_u64() {
        meta.lora = n as usize;
    }
    if let Some(n) = v["head_dim"].as_u64() {
        meta.head_dim = n as usize;
    }
    if let Some(b) = v["option_isolation"].as_bool() {
        meta.option_isolation = b;
    }
    if let Some(b) = v["special_embeddings"].as_bool() {
        meta.special_embeddings = b;
    }
    if let Some(s) = v["weights_dtype"].as_str() {
        meta.weights_dtype = s.to_string();
    }
    meta.temperature = v["temperature"]
        .as_f64()
        .ok_or_else(|| anyhow!("head.json has no `temperature`; refusing to default to 1.0"))?
        as f32;
    if !(meta.temperature.is_finite() && meta.temperature > 0.0) {
        bail!("head.json temperature is {}", meta.temperature);
    }
    Ok(meta)
}
