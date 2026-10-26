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

//! [`Package`]: the one place an exporter writes a package.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rlx_ir::Graph;
use rlx_runtime::model::{Tensor, rlxp};
use serde_json::{Value, json};

use super::{Checkpoint, ExportOptions};
use crate::package_export::{
    Referencer, add_huggingface_card, matmul_weights_as, package_dir, param_dims, relative_path,
    tidy,
};

/// A package being exported: its functions (the main one first), symbol
/// names, model block, held tensors and sidecars.
#[derive(Debug, Clone)]
pub struct Package {
    pub graphs: Vec<Graph>,
    /// Names of the graphs' symbolic dims (`dims {seq: 1}`).
    pub dims: Vec<(String, u32)>,
    /// The model block: `name`, `version`, `family`, `task`, `entries`,
    /// `weights`, `metadata`, the card.
    pub model: Value,
    /// Tensors the package holds.
    pub held: Vec<Tensor>,
    /// Files the model block names as `sidecar:<id>`: `(id, media type,
    /// bytes)`.
    pub sidecars: Vec<(String, String, Vec<u8>)>,
    /// Params read from the checkpoint, and read as stored.
    pub referenced: usize,
    pub packed: usize,
    /// Declare the weights read from half-precision tensors as stored
    /// ([`Self::weights_as_stored`]).
    pub as_stored: bool,
    /// Each referenced param whose checkpoint tensor is stored bf16 / f16.
    pub stored: HashMap<String, rlx_ir::DType>,
}

impl Package {
    /// An empty package of `family` for `task`, by `exported_by` (a crate).
    pub fn new(name: &str, version: &str, family: &str, task: &str, exported_by: &str) -> Self {
        Self {
            graphs: Vec::new(),
            dims: Vec::new(),
            model: json!({
                "name": name,
                "version": version,
                "family": family,
                "task": task,
                "entries": {},
                "metadata": {"exported_by": exported_by},
            }),
            held: Vec::new(),
            sidecars: Vec::new(),
            referenced: 0,
            packed: 0,
            as_stored: false,
            stored: HashMap::new(),
        }
    }

    /// [`Self::new`] named after `ck`, with its card (`README.md`) or the
    /// license a GGUF states.
    pub fn for_checkpoint(
        ck: &Checkpoint,
        opts: &ExportOptions,
        family: &str,
        task: &str,
        exported_by: &str,
    ) -> Self {
        let mut p = Self::new(&ck.name(), &opts.version, family, task, exported_by);
        if let Some(readme) = ck.readme() {
            add_huggingface_card(&mut p.model, &readme);
        }
        if let Some(l) = ck.license()
            && p.model.get("license").is_none()
        {
            p.model["license"] = json!(l);
        }
        p
    }

    /// Declare each matmul weight and embedding table read from a bf16 /
    /// f16 checkpoint tensor in that dtype, unless the options name one: its
    /// values are exactly that dtype's, so the package computes what it did
    /// from half the bytes — as the runners keep a bf16 checkpoint.
    pub fn weights_as_stored(&mut self) {
        self.as_stored = true;
    }

    /// Add a function.
    pub fn function(&mut self, graph: Graph) {
        self.graphs.push(graph);
    }

    /// Set the entry `kind` (`generate`, `embed`, `pipeline`, …).
    pub fn entry(&mut self, kind: &str, entry: Value) {
        self.model["entries"][kind] = entry;
    }

    /// Name the entry `from` `to`, its kind kept (`"image": {"kind":
    /// "pipeline", …}`): a package may hold several entries of a kind, and
    /// a caller picks one by name (the open option `entry`).
    pub fn rename_entry(&mut self, from: &str, to: &str) -> anyhow::Result<()> {
        let entries = self.model["entries"]
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("the package has no entries"))?;
        let mut e = entries
            .remove(from)
            .ok_or_else(|| anyhow::anyhow!("the package has no entry '{from}'"))?;
        if e.get("kind").is_none() {
            e["kind"] = Value::String(from.to_string());
        }
        entries.insert(to.to_string(), e);
        Ok(())
    }

    /// Set `metadata.<key>`.
    pub fn metadata(&mut self, key: &str, value: Value) {
        self.model["metadata"][key] = value;
    }

    /// Hold a tensor in the package.
    pub fn hold(&mut self, tensor: Tensor) {
        self.held.push(tensor);
    }

    /// Hold `values` as the f32 tensor `name`.
    pub fn hold_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, values: &[f32]) {
        self.held.push(Tensor::f32(name, shape, values));
    }

    /// Add the file `id`; what the model block names it by
    /// (`sidecar:<id>`).
    pub fn sidecar(&mut self, id: &str, media_type: &str, bytes: Vec<u8>) -> String {
        self.sidecars.retain(|(i, _, _)| i != id);
        self.sidecars
            .push((id.to_string(), media_type.to_string(), bytes));
        format!("sidecar:{id}")
    }

    /// `ck`'s tokenizer as the sidecar `tokenizer.json`; what an entry's
    /// `tokenizer` names it by.
    pub fn tokenizer(
        &mut self,
        ck: &Checkpoint,
        from_gguf: Option<&dyn Fn(&Path) -> Result<Vec<u8>>>,
    ) -> Result<String> {
        let bytes = ck.tokenizer(from_gguf)?;
        Ok(self.sidecar("tokenizer.json", "application/json", bytes))
    }

    /// `ck`'s chat block (template, BOS / EOS), its template a sidecar.
    pub fn chat(&mut self, ck: &Checkpoint) -> Value {
        let (chat, sidecars) = ck.chat();
        for (id, media, bytes) in sidecars {
            self.sidecar(&id, &media, bytes);
        }
        chat
    }

    /// Every function's params' dims, by name (the first function that has
    /// a param gives its dims).
    pub fn param_dims(&self) -> HashMap<String, Vec<usize>> {
        let mut dims = HashMap::new();
        for g in &self.graphs {
            for (name, d) in param_dims(g) {
                dims.entry(name).or_insert(d);
            }
        }
        dims
    }

    /// A `weights` source of the model block.
    pub fn weight_source(&mut self, source: Value) {
        match self.model.get_mut("weights").and_then(Value::as_array_mut) {
            Some(a) => a.push(source),
            None => self.model["weights"] = json!([source]),
        }
    }

    /// The params `refr` decided: each file it references a `weights`
    /// source (its path relative to the package at `out`), the rest held.
    pub fn reference(&mut self, refr: Referencer, out: &Path) -> Result<()> {
        let dims = self.param_dims();
        let packed = refr.packed_count();
        let (files, held, referenced) = refr.finish(&dims);
        for (file, scheme, map) in &files {
            if *scheme != "safetensors" {
                continue;
            }
            let Ok(loader) = crate::safetensors_checkpoint::SafetensorsMmapLoader::open_file(file)
            else {
                continue;
            };
            for (param, entry) in map {
                let key = entry
                    .as_str()
                    .or_else(|| entry.get("tensor").and_then(Value::as_str))
                    .unwrap_or(param);
                if let Some(d @ (rlx_ir::DType::BF16 | rlx_ir::DType::F16)) =
                    crate::weight_loader::WeightLoader::stored_dtype(&loader, key)
                {
                    self.stored.insert(param.clone(), d);
                }
            }
        }
        for (file, scheme, map) in files {
            let uri = relative_path(package_dir(out), &file);
            let source = rlxp::weight_source(
                scheme,
                &uri.display().to_string(),
                &file,
                Value::Object(map),
            )?;
            self.weight_source(source);
        }
        for (n, d, v) in held {
            self.hold_f32(n, d, &v);
        }
        self.referenced += referenced - packed;
        self.packed += packed;
        Ok(())
    }

    /// The params of `params` no checkpoint tensor equals, held; the rest
    /// referenced from the single file `source` (`scheme://` it, e.g.
    /// `safetensors`) — for an exporter that builds its graphs from a
    /// checkpoint already in memory.
    pub fn reference_map(
        &mut self,
        params: &HashMap<String, Vec<f32>>,
        source: &crate::weight_map::WeightMap,
        scheme: &str,
        file: &Path,
        out: &Path,
        hold_all: bool,
    ) -> Result<()> {
        let dims = self.param_dims();
        let refs = crate::package_export::reference_params(params, &dims, source, hold_all);
        if refs.referenced > 0 {
            let uri = relative_path(package_dir(out), file);
            self.weight_source(rlxp::weight_source(
                scheme,
                &uri.display().to_string(),
                file,
                Value::Object(refs.map),
            )?);
        }
        for (n, d, v) in refs.held {
            self.hold_f32(n, d, &v);
        }
        self.referenced += refs.referenced;
        Ok(())
    }

    /// `file` a `weights` source (`scheme://` it) whose tensors bind the
    /// params named as they are — `count` of them — with no value read: for
    /// a checkpoint too large to fingerprint whose exporter names its params
    /// by its tensors. Half-precision tensors are recorded for
    /// [`Self::weights_as_stored`].
    pub fn reference_named(
        &mut self,
        scheme: &str,
        file: &Path,
        out: &Path,
        count: usize,
    ) -> Result<()> {
        // Each graph param the file holds in half precision, for
        // `weights_as_stored`.
        if scheme == "safetensors"
            && let Ok(loader) =
                crate::safetensors_checkpoint::SafetensorsMmapLoader::open_file(file)
        {
            for name in self.param_dims().keys() {
                if let Some(d @ (rlx_ir::DType::BF16 | rlx_ir::DType::F16)) =
                    crate::weight_loader::WeightLoader::stored_dtype(&loader, name)
                {
                    self.stored.insert(name.clone(), d);
                }
            }
        }
        let uri = relative_path(package_dir(out), file);
        self.weight_source(rlxp::weight_source(
            scheme,
            &uri.display().to_string(),
            file,
            json!({}),
        )?);
        self.referenced += count;
        Ok(())
    }

    /// Write the package to `out`: each function tidied (`tidy`), its
    /// matmul weights declared `opts.weight_dtype` (unset: as stored, see
    /// [`Self::weights_as_stored`], else f32). What it wrote.
    pub fn write(mut self, out: &Path, opts: &ExportOptions) -> Result<super::ExportReport> {
        for g in &mut self.graphs {
            if let Some(merged) = merge_duplicate_params(g)? {
                *g = merged;
            }
        }
        self.graphs.iter_mut().for_each(tidy);
        self.pretranspose()?;
        let mut narrowed = 0;
        match opts.weight_dtype {
            Some(rlx_ir::DType::F32) => {}
            Some(dtype) => {
                for g in &mut self.graphs {
                    narrowed += matmul_weights_as(g, dtype);
                }
            }
            None if self.as_stored => {
                for g in &mut self.graphs {
                    narrowed += as_stored(g, &self.stored);
                }
            }
            None => {}
        }
        let functions: Vec<String> = self.graphs.iter().map(|g| g.name.clone()).collect();
        let held = self.held.len();
        let mut module = rlx_ir::rlxmodel::Module::new(self.graphs);
        module.dims = self.dims;
        module.model = Some(self.model);
        let sidecars: Vec<(&str, &str, Vec<u8>)> = self
            .sidecars
            .iter()
            .map(|(id, media, bytes)| (id.as_str(), media.as_str(), bytes.clone()))
            .collect();
        rlxp::save_package_module(out, &module, &self.held, &sidecars)?;
        Ok(super::ExportReport {
            functions,
            referenced: self.referenced,
            packed: self.packed,
            held,
            narrowed,
            ..Default::default()
        })
    }
}

impl Package {
    /// Each 2-D f32 param that every graph reads only through a `[1, 0]`
    /// transpose (a linear layer's `[out, in]` weight under `x · Wᵀ`),
    /// declared transposed and read the other way round: its `weights` map
    /// entries' `transpose` flipped, or a held tensor's values transposed.
    /// The graphs then read it directly — no backend transposes a weight
    /// per call (a matmul of rank-3 activations folds no transpose on the
    /// CPU: Moshi's 7B step spent 139 of 140 s copying weights), and the
    /// matmul reads it as stored. The params rewritten.
    fn pretranspose(&mut self) -> Result<usize> {
        use rlx_ir::{Op, Shape};
        let mut ok: HashMap<String, bool> = HashMap::new();
        for g in &self.graphs {
            let mut readers: HashMap<rlx_ir::NodeId, Vec<&rlx_ir::Node>> = HashMap::new();
            for n in g.nodes() {
                for &i in &n.inputs {
                    readers.entry(i).or_default().push(n);
                }
            }
            for n in g.nodes() {
                let Op::Param { name } = &n.op else { continue };
                let fit = n.shape.rank() == 2
                    && n.shape.dtype() == rlx_ir::DType::F32
                    && n.shape
                        .dims()
                        .iter()
                        .all(|d| matches!(d, rlx_ir::Dim::Static(_)))
                    && !g.outputs.contains(&n.id)
                    && readers.get(&n.id).is_some_and(|r| {
                        r.iter().all(|u| {
                            matches!(&u.op, Op::Transpose { perm } if perm == &[1, 0])
                                && u.inputs == [n.id]
                        })
                    });
                let e = ok.entry(name.clone()).or_insert(true);
                *e &= fit;
            }
        }
        ok.retain(|_, v| *v);
        if ok.is_empty() {
            return Ok(0);
        }
        for g in &mut self.graphs {
            let mut swapped: HashMap<rlx_ir::NodeId, rlx_ir::NodeId> = HashMap::new();
            *g = rlx_ir::stateful::rebuild(g, |out, _, node, inputs| match &node.op {
                Op::Param { name } if ok.contains_key(name) => {
                    let d: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| match d {
                            rlx_ir::Dim::Static(x) => *x,
                            rlx_ir::Dim::Dynamic(_) => 0,
                        })
                        .collect();
                    let t = Shape::new(&[d[1], d[0]], node.shape.dtype());
                    let id = out.param(name.clone(), t);
                    out.node_mut(id).name = node.name.clone();
                    swapped.insert(node.id, id);
                    Ok(Some(id))
                }
                Op::Transpose { .. }
                    if node.inputs.len() == 1 && swapped.contains_key(&node.inputs[0]) =>
                {
                    let id = inputs[0];
                    if node.name.is_some() && out.node(id).name.is_none() {
                        out.node_mut(id).name = node.name.clone();
                    }
                    Ok(Some(id))
                }
                _ => Ok(None),
            })
            .map_err(anyhow::Error::msg)?;
        }
        // Their values: a held tensor transposed; a referenced one read
        // transposed from every source.
        let mut held = std::collections::HashSet::new();
        for t in &mut self.held {
            if !ok.contains_key(&t.name) {
                continue;
            }
            let [r, c] = t.shape[..] else {
                anyhow::bail!("held '{}' is {:?}, not 2-D", t.name, t.shape);
            };
            let w = t.dtype.size_bytes();
            let mut data = vec![0u8; t.data.len()];
            for i in 0..r {
                for j in 0..c {
                    let (src, dst) = ((i * c + j) * w, (j * r + i) * w);
                    data[dst..dst + w].copy_from_slice(&t.data[src..src + w]);
                }
            }
            t.data = data;
            t.shape = vec![c, r];
            held.insert(t.name.clone());
        }
        if let Some(sources) = self.model.get_mut("weights").and_then(Value::as_array_mut) {
            for src in sources {
                let map = src
                    .as_object_mut()
                    .map(|o| o.entry("map").or_insert_with(|| json!({})));
                let Some(Value::Object(map)) = map else {
                    continue;
                };
                for name in ok.keys().filter(|n| !held.contains(*n)) {
                    let entry = match map.remove(name) {
                        None => json!({"transpose": true}),
                        Some(Value::String(t)) => json!({"tensor": t, "transpose": true}),
                        Some(Value::Object(mut o)) => {
                            let t = o.get("transpose").and_then(Value::as_bool).unwrap_or(false);
                            o.insert("transpose".into(), json!(!t));
                            Value::Object(o)
                        }
                        Some(other) => other,
                    };
                    map.insert(name.clone(), entry);
                }
            }
        }
        Ok(ok.len())
    }
}

/// `g` with each param it declares more than once (a tied head written as a
/// second param of the embedding's name) read from its first node, when it
/// has any: a runtime binds a name to one node, and the others would read
/// nothing. Params of one name and different shapes are an error.
fn merge_duplicate_params(g: &Graph) -> Result<Option<Graph>> {
    let mut first: HashMap<&str, (rlx_ir::NodeId, &rlx_ir::Shape)> = HashMap::new();
    let mut dup = false;
    for n in g.nodes() {
        if let rlx_ir::Op::Param { name } = &n.op {
            match first.get(name.as_str()) {
                Some((_, shape)) if *shape != &n.shape => anyhow::bail!(
                    "function '{}' declares the param '{name}' twice, as {:?} and {:?}",
                    g.name,
                    shape.dims(),
                    n.shape.dims()
                ),
                Some(_) => dup = true,
                None => {
                    first.insert(name, (n.id, &n.shape));
                }
            }
        }
    }
    if !dup {
        return Ok(None);
    }
    let mut seen: HashMap<String, rlx_ir::NodeId> = HashMap::new();
    let merged = rlx_ir::stateful::rebuild(g, |out, _, node, _| {
        let rlx_ir::Op::Param { name } = &node.op else {
            return Ok(None);
        };
        if let Some(&id) = seen.get(name) {
            return Ok(Some(id));
        }
        let id = out.param(name.clone(), node.shape.clone());
        out.node_mut(id).name = node.name.clone();
        seen.insert(name.clone(), id);
        Ok(Some(id))
    })
    .map_err(anyhow::Error::msg)?;
    Ok(Some(merged))
}

/// [`Package::weights_as_stored`] on one graph: each 2-D f32 param that
/// only a `MatMul` reads (as its right operand) and whose tensor is stored
/// half precision declared so; the embedding tables (each only an axis-0
/// `Gather`'s table) too, when every one of them is stored in one dtype.
/// The count changed.
fn as_stored(g: &mut Graph, stored: &HashMap<String, rlx_ir::DType>) -> usize {
    use rlx_ir::{DType, Op, Shape};
    let mut uses: HashMap<rlx_ir::NodeId, (bool, bool)> = HashMap::new();
    for n in g.nodes() {
        for (i, input) in n.inputs.iter().enumerate() {
            let rhs = matches!(n.op, Op::MatMul) && i == 1;
            let table = matches!(n.op, Op::Gather { axis: 0 }) && i == 0;
            let e = uses.entry(*input).or_insert((true, true));
            e.0 &= rhs;
            e.1 &= table;
        }
    }
    let mut weights = Vec::new();
    let mut tables: Vec<Option<DType>> = Vec::new();
    for n in g.nodes() {
        let Op::Param { name } = &n.op else { continue };
        if n.shape.rank() != 2 || n.shape.dtype() != DType::F32 {
            continue;
        }
        match uses.get(&n.id) {
            Some((true, _)) => {
                if let Some(d) = stored.get(name) {
                    weights.push((n.id, *d));
                }
            }
            Some((_, true)) => tables.push(stored.get(name).copied()),
            _ => {}
        }
    }
    let mut changed = weights.len();
    for (id, d) in weights {
        let node = g.node_mut(id);
        node.shape = Shape::from_dims(node.shape.dims(), d);
    }
    if let Some(Some(d)) = tables.first()
        && tables.iter().all(|t| *t == Some(*d))
    {
        changed += rlx_ir::retarget::embedding_tables_as(g, *d);
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, GraphExt, Shape};

    /// A param declared twice (a tied head) reads the same values both times
    /// once merged, as the runtime binds it.
    #[test]
    fn a_param_declared_twice_is_read_from_one_node() {
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let mut g = Graph::new("tied");
        let x = g.input("x", f(&[1, 3]));
        let a = g.param("w", f(&[3, 3]));
        let y = g.mm(x, a);
        let b = g.param("w", f(&[3, 3]));
        let y = g.mm(y, b);
        g.set_outputs(vec![y]);
        let merged = merge_duplicate_params(&g).unwrap().expect("a duplicate");
        let params = merged
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, rlx_ir::Op::Param { .. }))
            .count();
        assert_eq!(params, 1);
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(merged);
        let w = [1.0f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        c.set_param("w", &w);
        c.finalize_params();
        let out = c.run(&[("x", &[1.0, 1.0, 1.0])]).remove(0);
        assert_eq!(out, vec![1.0, 4.0, 9.0]);
        assert!(
            merge_duplicate_params(&Graph::new("none"))
                .unwrap()
                .is_none()
        );
    }
}
