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

//! What an `.rlxp` exporter of any family shares: which of a built graph's
//! params are a checkpoint's tensors (referenced, so the package holds the
//! model without its weights) and which were derived at build time (held
//! by the package).

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use rlx_ir::{Dim, Graph, Op};
use serde_json::{Value, json};

use crate::weight_map::WeightMap;

/// A graph's params' dims, by name.
pub fn param_dims(graph: &Graph) -> HashMap<String, Vec<usize>> {
    graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some((
                name.clone(),
                n.shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        Dim::Static(n) => *n,
                        Dim::Dynamic(_) => 0,
                    })
                    .collect(),
            )),
            _ => None,
        })
        .collect()
}

/// A built graph made fit to read: the builder's provenance (`origin`) and
/// any node name more than one node shares (a builder's `"mir"` on every
/// node) are dropped. A name only one node has stays: it labels the value.
pub fn tidy(graph: &mut Graph) {
    let mut count: HashMap<String, usize> = HashMap::new();
    for n in graph.nodes() {
        if let Some(name) = &n.name {
            *count.entry(name.clone()).or_default() += 1;
        }
    }
    for n in graph.nodes_mut() {
        n.origin = None;
        if n.name.as_ref().is_some_and(|name| count[name] > 1) {
            n.name = None;
        }
    }
}

/// Params split into references and held tensors.
pub struct Referenced {
    /// `param → checkpoint tensor` (a name, or `{tensor, transpose: true}`);
    /// a param stored under its own name is left out.
    pub map: serde_json::Map<String, Value>,
    /// Params no checkpoint tensor equals: `(name, dims, values)`.
    pub held: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub referenced: usize,
}

fn fingerprint(v: &[f32]) -> (usize, u64) {
    // Finds candidates; equality is then checked in full.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v.iter().step_by((v.len() / 64).max(1)) {
        h = (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3);
    }
    (v.len(), h)
}

/// Declare `graph`'s matmul weights `dtype`: `rlx_ir::retarget`'s, which a
/// package build for a device also uses.
pub use rlx_ir::retarget::matmul_weights_as;

/// `v` (`rows × cols`) transposed.
pub fn transposed(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

/// Which `params` equal a tensor of `source` (the checkpoint, as loaded),
/// as stored or transposed. `hold_all` keeps every param in the package.
pub fn reference_params(
    params: &HashMap<String, Vec<f32>>,
    dims: &HashMap<String, Vec<usize>>,
    source: &WeightMap,
    hold_all: bool,
) -> Referenced {
    let mut by_print: HashMap<(usize, u64), Vec<(String, bool)>> = HashMap::new();
    if !hold_all {
        for key in source.keys() {
            let (data, d) = source.get(key).expect("listed key");
            by_print
                .entry(fingerprint(data))
                .or_default()
                .push((key.to_string(), false));
            if let [r, c] = d[..] {
                by_print
                    .entry(fingerprint(&transposed(data, r, c)))
                    .or_default()
                    .push((key.to_string(), true));
            }
        }
    }
    let mut out = Referenced {
        map: serde_json::Map::new(),
        held: Vec::new(),
        referenced: 0,
    };
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for name in names {
        let data = &params[name];
        let found = by_print.get(&fingerprint(data)).and_then(|cands| {
            // The tensor of the same name first: a tied weight equals two.
            let mut cands: Vec<&(String, bool)> = cands.iter().collect();
            cands.sort_by_key(|(k, t)| (k != name, *t));
            cands.into_iter().find(|(key, t)| {
                let (src, sdims) = source.get(key).expect("listed key");
                match (t, sdims) {
                    (false, _) => src == &data[..],
                    (true, [r, c]) => transposed(src, *r, *c) == *data,
                    _ => false,
                }
            })
        });
        match found {
            Some((key, false)) => {
                if key != name {
                    out.map.insert(name.clone(), json!(key));
                }
                out.referenced += 1;
            }
            Some((key, true)) => {
                // The tensor is named like the param unless it says otherwise.
                let entry = if key == name {
                    json!({"transpose": true})
                } else {
                    json!({"tensor": key, "transpose": true})
                };
                out.map.insert(name.clone(), entry);
                out.referenced += 1;
            }
            None => {
                let d = dims.get(name).cloned().unwrap_or_else(|| vec![data.len()]);
                out.held.push((name.clone(), d, data.clone()));
            }
        }
    }
    out
}

/// [`reference_params`] for a checkpoint too large to hold twice: the
/// checkpoint's tensors fingerprinted one at a time from its files (as
/// stored and transposed), then each graph's params decided as they are
/// built — a param equal to a checkpoint tensor becomes a reference and its
/// values are dropped there and then, so an export holds the derived
/// params, not a second copy of the weights.
pub struct Referencer {
    files: Vec<(PathBuf, TensorFile)>,
    by_print: HashMap<(usize, u64), Vec<(usize, String, bool)>>,
    hold_all: bool,
    /// Per file: `param → tensor` as [`Referenced::map`] has it.
    maps: Vec<serde_json::Map<String, Value>>,
    /// Per file: the params it holds.
    hits: Vec<usize>,
    referenced: usize,
    /// Of `referenced`, those read as stored ([`Self::packed`]).
    packed: usize,
    held: HashMap<String, Vec<f32>>,
    decided: std::collections::HashSet<String>,
}

/// A checkpoint file the [`Referencer`] reads tensors from.
enum TensorFile {
    Safetensors(crate::safetensors_checkpoint::SafetensorsCheckpoint),
    Gguf(rlx_gguf::GgufFile),
}

impl TensorFile {
    fn keys(&self) -> Vec<String> {
        match self {
            Self::Safetensors(ck) => ck.keys().map(String::from).collect(),
            Self::Gguf(g) => g.tensors.keys().cloned().collect(),
        }
    }

    /// Its element count, where the header says without a read (a GGUF's;
    /// 0 for safetensors, which no filter asks about).
    fn len(&self, key: &str) -> usize {
        match self {
            Self::Safetensors(_) => 0,
            Self::Gguf(g) => g.get(key).map_or(0, |t| t.shape.iter().product()),
        }
    }

    /// The tensor as the runtime binds it: a GGUF's dequantized (what a
    /// `gguf://` reference reads), its dims outermost first.
    fn load(&self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Safetensors(ck) => ck.load_tensor_f32(key),
            Self::Gguf(g) => {
                let (data, _) = g
                    .dequant_f32(key)
                    .map_err(|e| anyhow::anyhow!("{key}: {e}"))?;
                let t = g
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("{key}: no tensor"))?;
                Ok((data, t.shape.iter().rev().copied().collect()))
            }
        }
    }

    /// Its `weights` source scheme.
    fn scheme(&self) -> &'static str {
        match self {
            Self::Safetensors(_) => "safetensors",
            Self::Gguf(_) => "gguf",
        }
    }
}

impl Referencer {
    /// Fingerprint the tensors of `files` (safetensors); `hold_all` keeps
    /// every param instead.
    pub fn open(files: &[PathBuf], hold_all: bool) -> anyhow::Result<Self> {
        let mut opened = Vec::new();
        for file in files {
            let ck = crate::safetensors_checkpoint::SafetensorsCheckpoint::open_file(file)?;
            opened.push((file.clone(), TensorFile::Safetensors(ck)));
        }
        Self::with_files(opened, hold_all)
    }

    /// [`Self::open`] for a GGUF: a param equal to one of its tensors as
    /// the runtime dequantizes it becomes a `gguf://` reference.
    pub fn open_gguf(file: &Path, hold_all: bool) -> anyhow::Result<Self> {
        Self::open_gguf_only(file, hold_all, &|_, _| true)
    }

    /// [`Self::open_gguf`] matching params against only the tensors `only`
    /// takes (by name and element count): dequantizing every packed matrix
    /// of a large GGUF to look for one no param equals costs minutes.
    pub fn open_gguf_only(
        file: &Path,
        hold_all: bool,
        only: &dyn Fn(&str, usize) -> bool,
    ) -> anyhow::Result<Self> {
        let g = rlx_gguf::GgufFile::from_path(file)
            .map_err(|e| anyhow::anyhow!("open {}: {e}", file.display()))?;
        Self::with_files_only(
            vec![(file.to_path_buf(), TensorFile::Gguf(g))],
            hold_all,
            only,
        )
    }

    fn with_files(files: Vec<(PathBuf, TensorFile)>, hold_all: bool) -> anyhow::Result<Self> {
        Self::with_files_only(files, hold_all, &|_, _| true)
    }

    fn with_files_only(
        files: Vec<(PathBuf, TensorFile)>,
        hold_all: bool,
        only: &dyn Fn(&str, usize) -> bool,
    ) -> anyhow::Result<Self> {
        let n = files.len();
        let mut out = Self {
            files: Vec::new(),
            by_print: HashMap::new(),
            hold_all,
            maps: vec![serde_json::Map::new(); n],
            hits: vec![0; n],
            referenced: 0,
            packed: 0,
            held: HashMap::new(),
            decided: Default::default(),
        };
        for (i, (_, file)) in files.iter().enumerate() {
            if !hold_all {
                for key in file.keys() {
                    if !only(&key, file.len(&key)) {
                        continue;
                    }
                    let (data, d) = file.load(&key)?;
                    out.by_print.entry(fingerprint(&data)).or_default().push((
                        i,
                        key.clone(),
                        false,
                    ));
                    if let [r, c] = d[..] {
                        out.by_print
                            .entry(fingerprint(&transposed(&data, r, c)))
                            .or_default()
                            .push((i, key, true));
                    }
                }
            }
        }
        out.files = files;
        Ok(out)
    }

    /// Decide each of `params` not decided yet: referenced (its values
    /// dropped) or held.
    pub fn take(&mut self, params: HashMap<String, Vec<f32>>) -> anyhow::Result<()> {
        for (name, data) in params {
            if !self.decided.insert(name.clone()) {
                continue;
            }
            let mut cands: Vec<(usize, String, bool)> = self
                .by_print
                .get(&fingerprint(&data))
                .cloned()
                .unwrap_or_default();
            // The tensor of the same name first: a tied weight equals two.
            cands.sort_by_key(|(_, k, t)| (*k != name, *t));
            let mut found = None;
            for (i, key, t) in cands {
                let (src, d) = self.files[i].1.load(&key)?;
                let same = match (t, &d[..]) {
                    (false, _) => src == data,
                    (true, [r, c]) => transposed(&src, *r, *c) == data,
                    _ => false,
                };
                if same {
                    found = Some((i, key, t));
                    break;
                }
            }
            match found {
                Some((i, key, false)) => {
                    self.hits[i] += 1;
                    if key != name {
                        self.maps[i].insert(name, json!(key));
                    }
                    self.referenced += 1;
                }
                Some((i, key, true)) => {
                    self.hits[i] += 1;
                    let entry = if key == name {
                        json!({"transpose": true})
                    } else {
                        json!({"tensor": key, "transpose": true})
                    };
                    self.maps[i].insert(name, entry);
                    self.referenced += 1;
                }
                None => {
                    self.held.insert(name, data);
                }
            }
        }
        Ok(())
    }

    /// `param` read from the first file's tensor `key` as stored (its
    /// quantized blocks, for a packed param), not matched by value.
    pub fn packed(&mut self, param: &str, key: &str) {
        if self.decided.insert(param.to_string()) {
            self.held.remove(param);
            self.maps[0].insert(param.to_string(), json!({"tensor": key}));
            self.hits[0] += 1;
            self.referenced += 1;
            self.packed += 1;
        }
    }

    /// Params read as stored ([`Self::packed`]).
    pub fn packed_count(&self) -> usize {
        self.packed
    }

    /// Each file that holds a referenced param (its path, `weights` scheme
    /// and map), the held params (dims from `dims`), and how many were
    /// referenced.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        self,
        dims: &HashMap<String, Vec<usize>>,
    ) -> (
        Vec<(PathBuf, &'static str, serde_json::Map<String, Value>)>,
        Vec<(String, Vec<usize>, Vec<f32>)>,
        usize,
    ) {
        let _ = self.hold_all;
        let mut sources = Vec::new();
        for ((path, file), (map, hits)) in self
            .files
            .into_iter()
            .zip(self.maps.into_iter().zip(self.hits))
        {
            if hits > 0 {
                sources.push((path, file.scheme(), map));
            }
        }
        let mut held: Vec<(String, Vec<usize>, Vec<f32>)> = self
            .held
            .into_iter()
            .map(|(n, v)| {
                let d = dims.get(&n).cloned().unwrap_or_else(|| vec![v.len()]);
                (n, d, v)
            })
            .collect();
        held.sort_by(|a, b| a.0.cmp(&b.0));
        (sources, held, self.referenced)
    }
}

/// `to` relative to the directory `from`, when they share a root; else
/// `to`. Both are resolved first: `..` from inside a symlinked directory
/// (macOS's `/var` is `/private/var`) climbs the real path.
pub fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let (Ok(from), Ok(to)) = (std::fs::canonicalize(from), std::fs::canonicalize(to)) else {
        return to.to_path_buf();
    };
    let a: Vec<Component> = from.components().collect();
    let b: Vec<Component> = to.components().collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    if common == 0 {
        return to;
    }
    let mut out = PathBuf::new();
    for _ in common..a.len() {
        out.push("..");
    }
    for c in &b[common..] {
        out.push(c);
    }
    out
}

/// A Hugging Face model card (`README.md`) split into its metadata, the
/// YAML front matter as JSON (`license`, `language`, `tags`, `datasets`,
/// `base_model`, `model-index`, …, under Hugging Face's keys), and its
/// Markdown body. A card with no front matter, or front matter that is not
/// a YAML map, is all body.
pub fn split_huggingface_card(readme: &str) -> (Option<Value>, String) {
    let text = readme.strip_prefix('\u{feff}').unwrap_or(readme);
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (None, readme.to_string());
    };
    let Some(end) = rest.find("\n---") else {
        return (None, readme.to_string());
    };
    let (front, after) = rest.split_at(end);
    let body = after["\n---".len()..]
        .trim_start_matches(['\r', ' '])
        .strip_prefix('\n')
        .unwrap_or("");
    match serde_yaml::from_str::<Value>(front) {
        Ok(v @ Value::Object(_)) => (Some(v), body.to_string()),
        _ => (None, readme.to_string()),
    }
}

/// Put a Hugging Face `README.md` into a model block: `card` its body,
/// `card_data` its metadata, `license` and `tags` from it, and its
/// `pipeline_tag` when the block names no task.
pub fn add_huggingface_card(model: &mut Value, readme: &str) {
    let (data, body) = split_huggingface_card(readme);
    if let Some(data) = data {
        for key in ["license", "tags"] {
            if let Some(v) = data.get(key)
                && model.get(key).is_none()
            {
                model[key] = v.clone();
            }
        }
        model["card_data"] = data;
    }
    model["card"] = json!(body);
}

/// The directory a package at `out` resolves relative references from.
pub fn package_dir(out: &Path) -> &Path {
    out.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_huggingface_card_splits_into_its_metadata_and_body() {
        let readme = "---\nlicense: apache-2.0\nlanguage:\n- en\n- zh\npipeline_tag: text-generation\n\
                      base_model: Qwen/Qwen3-0.6B-Base\nmodel-index:\n- name: m\n  results:\n  - task:\n      \
                      type: text-generation\n    metrics:\n    - type: acc\n      value: 0.5\n---\n\n# Model\n\nBody.\n";
        let (data, body) = split_huggingface_card(readme);
        let data = data.unwrap();
        assert_eq!(data["license"], "apache-2.0");
        assert_eq!(data["language"], json!(["en", "zh"]));
        assert_eq!(
            data["model-index"][0]["results"][0]["metrics"][0]["value"],
            0.5
        );
        assert_eq!(body, "\n# Model\n\nBody.\n");

        let mut model = json!({"name": "m"});
        add_huggingface_card(&mut model, readme);
        assert_eq!(model["license"], "apache-2.0");
        assert_eq!(model["card"], "\n# Model\n\nBody.\n");
        assert_eq!(model["card_data"]["base_model"], "Qwen/Qwen3-0.6B-Base");

        // No front matter, or not a map: all body. CRLF front matter reads.
        assert_eq!(
            split_huggingface_card("# Just text\n"),
            (None, "# Just text\n".into())
        );
        assert!(split_huggingface_card("---\n- a list\n---\nx").0.is_none());
        let (d, b) = split_huggingface_card("---\r\nlicense: mit\r\n---\r\nhi\r\n");
        assert_eq!(
            (d.unwrap()["license"].clone(), b.as_str()),
            (json!("mit"), "hi\r\n")
        );
    }
}
