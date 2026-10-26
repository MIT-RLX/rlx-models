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

//! Any graph as a package: tensors in, tensors out (a `pipeline` entry),
//! so whatever a family builds — an encoder, a vision tower, a codec, a
//! whole model in one graph — runs later from the package alone,
//! `Model.run` in every language.
//!
//! ```no_run
//! # use std::collections::HashMap;
//! # use rlx_core::export::{ExportOptions, graph::{Stage, export_stages}};
//! # fn build() -> (rlx_ir::Graph, HashMap<String, Vec<f32>>) { unimplemented!() }
//! let (g, params) = build();
//! let stage = Stage::new(g); // its inputs and outputs by their own names
//! export_stages(vec![stage], params, None, "my-model", "feature-extraction",
//!               "out.rlxp".as_ref(), &ExportOptions::default())?;
//! # anyhow::Ok(())
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_ir::{Graph, GraphExt, Op};
use serde_json::{Value, json};

use super::{Checkpoint, ExportOptions, ExportReport, Package};

/// One function of a pipeline: what it reads and what it names its
/// outputs.
#[derive(Debug, Clone)]
pub struct Stage {
    pub graph: Graph,
    /// Each of the graph's inputs and the value it reads: a pipeline input
    /// or an earlier stage's output.
    pub inputs: Vec<(String, String)>,
    /// The names its outputs take, in order.
    pub outputs: Vec<String>,
    /// The device it runs on, where not the model's (`"cpu"` for a
    /// preprocessing stage that must round as the CPU does).
    pub device: Option<String>,
}

impl Stage {
    /// `graph` reading each input from the value of its own name, its
    /// outputs named as their nodes are (`output_<i>` for one with no name
    /// or a name another output has).
    pub fn new(graph: Graph) -> Self {
        let inputs = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some((name.clone(), name.clone())),
                _ => None,
            })
            .collect();
        let mut outputs: Vec<String> = Vec::new();
        for (i, &o) in graph.outputs.iter().enumerate() {
            match &graph.node(o).name {
                Some(n) if !outputs.contains(n) => outputs.push(n.clone()),
                _ => outputs.push(format!("output_{i}")),
            }
        }
        Self {
            graph,
            inputs,
            outputs,
            device: None,
        }
    }

    /// Run it on `device`.
    pub fn on(mut self, device: &str) -> Self {
        self.device = Some(device.to_string());
        self
    }

    /// Name the outputs.
    pub fn outputs(mut self, names: &[&str]) -> Self {
        self.outputs = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Feed the graph's input `input` from the value `from`.
    pub fn input(mut self, input: &str, from: &str) -> Self {
        for (i, v) in &mut self.inputs {
            if i == input {
                *v = from.to_string();
            }
        }
        self
    }
}

/// A `pipeline` package being written: its stages, run in order, and what
/// its entry says around them.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub family: String,
    pub task: String,
    pub stages: Vec<Stage>,
    /// The entry's inputs: each a name, or a contract (`{name, kind,
    /// dtype, shape, …}`). Empty: the values no stage makes, by name.
    pub inputs: Vec<Value>,
    /// The values it returns. Empty: the last stage's outputs.
    pub outputs: Vec<String>,
    /// Names of the graphs' symbolic dims (`[("height", 0), …]`).
    pub dims: Vec<(String, u32)>,
    /// `metadata.<key>` of the model block.
    pub metadata: Vec<(String, Value)>,
    /// Tensors it holds besides params (constants a stage reads).
    pub held: Vec<rlx_runtime::model::Tensor>,
    /// Functions besides the stages', for entries added to the package
    /// [`Self::package`] returns (their params among `params`).
    pub functions: Vec<Graph>,
    /// The crate that exported.
    pub exported_by: String,
}

impl Pipeline {
    pub fn new(family: &str, task: &str, stages: Vec<Stage>) -> Self {
        Self {
            family: family.to_string(),
            task: task.to_string(),
            stages,
            inputs: Vec::new(),
            outputs: Vec::new(),
            dims: Vec::new(),
            metadata: Vec::new(),
            held: Vec::new(),
            functions: Vec::new(),
            exported_by: "rlx-models".to_string(),
        }
    }

    /// Write it as the package `out`. `params` found in `checkpoint` are
    /// referenced from it; the rest are held.
    pub fn write(
        self,
        params: HashMap<String, Vec<f32>>,
        checkpoint: Option<&Checkpoint>,
        out: &Path,
        opts: &ExportOptions,
    ) -> Result<ExportReport> {
        self.package(params, checkpoint, out, opts)?
            .write(out, opts)
    }

    /// The package [`Self::write`] writes, not yet written: an exporter
    /// adds its other entries (over [`Self::functions`]) and writes it.
    pub fn package(
        self,
        params: HashMap<String, Vec<f32>>,
        checkpoint: Option<&Checkpoint>,
        out: &Path,
        opts: &ExportOptions,
    ) -> Result<Package> {
        let stages = self.stages;
        let Some(last) = stages.last() else {
            bail!("a pipeline needs a stage");
        };
        let outputs = if self.outputs.is_empty() {
            last.outputs.clone()
        } else {
            self.outputs
        };
        let mut made: Vec<&str> = Vec::new();
        let mut inputs: Vec<String> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for (i, s) in stages.iter().enumerate() {
            if s.outputs.len() != s.graph.outputs.len() {
                bail!(
                    "stage {i} ({}) names {} outputs, its graph has {}",
                    s.graph.name,
                    s.outputs.len(),
                    s.graph.outputs.len()
                );
            }
            for (_, from) in &s.inputs {
                // `image#shape`: the dims of a value, not one of its own.
                let from = from.split('#').next().unwrap_or(from).to_string();
                if !made.contains(&from.as_str()) && !inputs.contains(&from) {
                    inputs.push(from);
                }
            }
            made.extend(s.outputs.iter().map(String::as_str));
            let mut name = s.graph.name.clone();
            if name.is_empty() || names.contains(&name) {
                name = format!("stage_{i}");
            }
            names.push(name);
        }
        for o in &outputs {
            if !made.contains(&o.as_str()) {
                bail!("the pipeline returns '{o}', which no stage makes");
            }
        }
        let inputs: Vec<Value> = if self.inputs.is_empty() {
            inputs.into_iter().map(Value::from).collect()
        } else {
            self.inputs
        };

        let (family, task) = (self.family.as_str(), self.task.as_str());
        let mut pkg = match checkpoint {
            Some(ck) => Package::for_checkpoint(ck, opts, family, task, &self.exported_by),
            None => Package::new(family, &opts.version, family, task, &self.exported_by),
        };
        let entry_stages: Vec<Value> = stages
            .iter()
            .zip(&names)
            .map(|(s, name)| {
                let ins: serde_json::Map<String, Value> = s
                    .inputs
                    .iter()
                    .map(|(i, v)| (i.clone(), json!(v)))
                    .collect();
                let mut st = json!({"function": name, "inputs": ins, "outputs": s.outputs});
                if let Some(d) = &s.device {
                    st["device"] = json!(d);
                }
                st
            })
            .collect();
        for (s, name) in stages.into_iter().zip(names) {
            let mut g = s.graph;
            g.name = name;
            pkg.function(g);
        }
        for g in self.functions {
            pkg.function(g);
        }
        pkg.dims = self.dims;
        pkg.entry(
            "pipeline",
            json!({"task": task, "inputs": inputs, "stages": entry_stages, "outputs": outputs}),
        );
        for (k, v) in self.metadata {
            pkg.metadata(&k, v);
        }
        pkg.held.extend(self.held);
        match checkpoint {
            Some(ck) => {
                let mut refr = ck.referencer(opts.embed_weights)?;
                refr.take(params)?;
                pkg.reference(refr, out)?;
            }
            None => {
                let dims = pkg.param_dims();
                let mut names: Vec<&String> = params.keys().collect();
                names.sort();
                for n in names {
                    let d = dims
                        .get(n)
                        .cloned()
                        .unwrap_or_else(|| vec![params[n].len()]);
                    pkg.hold_f32(n.clone(), d, &params[n]);
                }
            }
        }
        Ok(pkg)
    }
}

/// Write `stages`, run in order, as the package `out` of `family` for
/// `task`: its inputs the values no stage makes, its outputs the last
/// stage's. Params found in `checkpoint` are referenced from it; the rest
/// are held.
#[allow(clippy::too_many_arguments)]
pub fn export_stages(
    stages: Vec<Stage>,
    params: HashMap<String, Vec<f32>>,
    checkpoint: Option<&Checkpoint>,
    family: &str,
    task: &str,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    Pipeline::new(family, task, stages).write(params, checkpoint, out, opts)
}

/// [`export_stages`] for one graph.
pub fn export_graph(
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    checkpoint: Option<&Checkpoint>,
    family: &str,
    task: &str,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    export_stages(
        vec![Stage::new(graph)],
        params,
        checkpoint,
        family,
        task,
        out,
        opts,
    )
}

/// One graph per size, as `builds` gives them (`(size, graph, params)`):
/// a param whose values differ between sizes (a table as long as the
/// sequence) is one per size, `name@<size>`, in its graph and in the params
/// returned; the rest are shared, as their names say.
pub fn per_size(
    builds: Vec<(usize, Graph, HashMap<String, Vec<f32>>)>,
) -> (Vec<Graph>, HashMap<String, Vec<f32>>) {
    let mut first: HashMap<&str, &Vec<f32>> = HashMap::new();
    let mut varies: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_, _, params) in &builds {
        for (k, v) in params {
            match first.get(k.as_str()) {
                Some(seen) if *seen != v => {
                    varies.insert(k.clone());
                }
                Some(_) => {}
                None => {
                    first.insert(k, v);
                }
            }
        }
    }
    // A param some sizes lack is theirs alone too.
    for (_, _, params) in &builds {
        for k in first.keys() {
            if !params.contains_key(*k) {
                varies.insert(k.to_string());
            }
        }
    }
    drop(first);
    let mut graphs = Vec::new();
    let mut all = HashMap::new();
    for (size, mut g, params) in builds {
        for node in g.nodes_mut() {
            if let Op::Param { name } = &mut node.op
                && varies.contains(name)
            {
                *name = format!("{name}@{size}");
            }
        }
        for (k, v) in params {
            let k = if varies.contains(&k) {
                format!("{k}@{size}")
            } else {
                k
            };
            all.entry(k).or_insert(v);
        }
        graphs.push(g);
    }
    (graphs, all)
}

/// Each param of the symbolic `graph` whose shape holds a symbol, as data a
/// package can hold:
///
/// - a block a builder materialized at the length (a convolution's zero
///   padding, a bias repeated over time) — the same along those axes — as
///   its one slice, held, and an `Expand` to the length;
/// - a table as long as the length (a RoPE table), symbolic in one axis
///   with only unit axes before it — the first rows of one of `tables`
///   (`(name, rows, row width)`) — as that table, held whole, and its first
///   rows read.
///
/// `at` is the graph built at a size with at least two positions on every
/// such axis (one position is the same along any axis), `params` its
/// values. Any other param depends on the length, and is an error.
pub fn broadcast_params(
    graph: &Graph,
    at: &Graph,
    params: &mut HashMap<String, Vec<f32>>,
    tables: &[(&str, &[f32], usize)],
) -> Result<Graph> {
    use rlx_ir::Dim;
    let concrete: HashMap<String, Vec<usize>> = at
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some((
                name.clone(),
                n.shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        Dim::Static(x) => *x,
                        Dim::Dynamic(_) => 0,
                    })
                    .collect(),
            )),
            _ => None,
        })
        .collect();
    let mut failure = None;
    let mut held_tables: Vec<String> = Vec::new();
    let g = rlx_ir::stateful::rebuild(graph, |out, attrs, node, _| {
        let Op::Param { name } = &node.op else {
            return Ok(None);
        };
        let dims = node.shape.dims();
        let symbolic: Vec<bool> = dims.iter().map(|d| matches!(d, Dim::Dynamic(_))).collect();
        if !symbolic.contains(&true) {
            return Ok(None);
        }
        let (Some(shape), Some(data)) = (concrete.get(name).cloned(), params.get(name).cloned())
        else {
            failure = Some(format!("param '{name}' has no values at the size built"));
            return Ok(None);
        };
        if shape.iter().zip(&symbolic).any(|(&d, &s)| s && d < 2) {
            failure = Some(format!(
                "param '{name}' was built one long on a symbolic axis: build `at` longer"
            ));
            return Ok(None);
        }
        // The slice at index 0 of every symbolic axis, and whether every
        // other index along them repeats it.
        let strides: Vec<usize> = (0..shape.len())
            .map(|i| shape[i + 1..].iter().product())
            .collect();
        let mut slice = Vec::new();
        let total: usize = shape.iter().product();
        for flat in 0..total {
            let mut first = 0;
            let mut at_zero = true;
            for (i, &st) in strides.iter().enumerate() {
                let idx = flat / st % shape[i];
                if symbolic[i] {
                    at_zero &= idx == 0;
                } else {
                    first += idx * st;
                }
            }
            if at_zero {
                slice.push(data[flat]);
            } else if data[flat] != data[first] {
                // Not a repeated block: the first rows of a table?
                let axis = symbolic.iter().position(|&s| s).unwrap_or(0);
                let width: usize = shape[axis + 1..].iter().product();
                let table = (shape[..axis].iter().all(|&d| d == 1)
                    && !symbolic[axis + 1..].contains(&true))
                .then(|| {
                    tables
                        .iter()
                        .find(|(_, rows, w)| *w == width && rows.starts_with(&data))
                })
                .flatten();
                let Some((tname, rows, w)) = table else {
                    failure = Some(format!(
                        "param '{name}' differs along its length: it depends on it, and \
                         no package holds it"
                    ));
                    return Ok(None);
                };
                params.remove(name);
                if !held_tables.iter().any(|t| t == tname) {
                    held_tables.push(tname.to_string());
                    params.insert(tname.to_string(), rows.to_vec());
                }
                let Dim::Dynamic(sym) = dims[axis] else {
                    unreachable!("axis {axis} is symbolic")
                };
                let p = out.param(
                    tname.to_string(),
                    rlx_ir::Shape::new(&[rows.len() / w, *w], node.shape.dtype()),
                );
                let lead =
                    rlx_ir::Shape::from_dims(&[dims[axis], Dim::Static(*w)], node.shape.dtype());
                let first = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: 0,
                        len: shape[axis],
                    },
                    vec![p],
                    lead,
                );
                attrs.push(rlx_ir::dynamic::SymAttr {
                    node: first,
                    path: vec!["len".into()],
                    sym,
                    scale: 1,
                    offset: 0,
                });
                let target = dims
                    .iter()
                    .map(|d| match d {
                        Dim::Static(x) => *x as i64,
                        Dim::Dynamic(_) => -1,
                    })
                    .collect();
                return Ok(Some(out.add_node(
                    Op::Reshape { new_shape: target },
                    vec![first],
                    node.shape.clone(),
                )));
            }
        }
        let small: Vec<usize> = shape
            .iter()
            .zip(&symbolic)
            .map(|(&d, &s)| if s { 1 } else { d })
            .collect();
        params.insert(name.clone(), slice);
        let p = out.param(name.clone(), rlx_ir::Shape::new(&small, node.shape.dtype()));
        let target = dims
            .iter()
            .map(|d| match d {
                Dim::Static(x) => *x as i64,
                Dim::Dynamic(_) => 1,
            })
            .collect();
        Ok(Some(out.add_node(
            Op::Expand {
                target_shape: target,
            },
            vec![p],
            node.shape.clone(),
        )))
    })
    .map_err(anyhow::Error::msg)?;
    match failure {
        Some(e) => bail!(e),
        None => Ok(g),
    }
}

/// `build` at sizes `a` and `b` of the symbol `sym` as one graph symbolic
/// in it — a dim may be any whole multiple of the size plus a constant (a
/// convolution's padding), or a square's (an attention's relative
/// positions), the third size `c` telling which fits — and checked against
/// `build` at `c`.
pub fn symbolic_in(
    sym: u32,
    (a, b, c): (usize, usize, usize),
    build: impl Fn(usize) -> Result<Graph>,
) -> Result<Graph> {
    let at_c = build(c)?;
    let g = rlx_ir::dynamic::generalize_checked(
        &build(a)?,
        &build(b)?,
        &at_c,
        &[(sym, a, b, c)],
        Some(1 << 16),
    )
    .map_err(|e| anyhow::anyhow!("not symbolic in its length: {e}"))?;
    let mut bind = rlx_ir::DimBinding::new();
    bind.set(sym, c);
    let (got, want) = (
        format!("{}", rlx_ir::dynamic::specialize(&g, &bind)),
        format!("{at_c}"),
    );
    if got != want {
        bail!(
            "the graph generalized from {a} and {b} is not the one built at {c}: {}",
            first_difference(&got, &want)
        );
    }
    Ok(g)
}

/// The first line two graphs' text differs in: `got` (generalized, then
/// specialized) against `want` (built).
fn first_difference(got: &str, want: &str) -> String {
    match got.lines().zip(want.lines()).find(|(x, y)| x != y) {
        Some((x, y)) => format!("`{}` where the build has `{}`", x.trim(), y.trim()),
        None => "one has more lines".into(),
    }
}

/// `v` as its runs of one value (`3×0, 1, 2×5`), the first dozen and the
/// last few.
fn sketch(v: &[f64]) -> String {
    let mut runs: Vec<(f64, usize)> = Vec::new();
    for &x in v {
        match runs.last_mut() {
            Some((y, n)) if *y == x => *n += 1,
            _ => runs.push((x, 1)),
        }
    }
    let show = |r: &[(f64, usize)]| {
        r.iter()
            .map(|(x, n)| {
                if *n == 1 {
                    format!("{x}")
                } else {
                    format!("{n}×{x}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    if runs.len() <= 16 {
        format!("[{}]", show(&runs))
    } else {
        format!(
            "[{}, … {}]",
            show(&runs[..12]),
            show(&runs[runs.len() - 4..])
        )
    }
}

/// A constant's elements as f64, for the element types a builder bakes.
fn elements(data: &[u8], dtype: rlx_ir::DType) -> Option<Vec<f64>> {
    use rlx_ir::DType as D;
    Some(match dtype {
        D::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        D::F64 => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect(),
        D::I64 => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as f64)
            .collect(),
        D::I32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        D::U32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        D::U8 | D::Bool => data.iter().map(|&b| f64::from(b)).collect(),
        D::I8 => data.iter().map(|&b| f64::from(b as i8)).collect(),
        _ => return None,
    })
}

/// `values` as a constant's bytes of `dtype` ([`elements`] back).
fn bytes_of(values: &[f64], dtype: rlx_ir::DType) -> Vec<u8> {
    use rlx_ir::DType as D;
    match dtype {
        D::F64 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        D::I64 => values
            .iter()
            .flat_map(|&v| (v as i64).to_le_bytes())
            .collect(),
        D::I32 => values
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect(),
        D::U32 => values
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect(),
        D::U8 | D::Bool => values.iter().map(|&v| v as u8).collect(),
        D::I8 => values.iter().map(|&v| v as i8 as u8).collect(),
        _ => values
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect(),
    }
}

/// How a constant baked at the length depends on it.
enum Lift {
    /// A head, a run of one value as long as the length makes it, a tail
    /// (an overlap-add's coverage count): rebuilt from those.
    Run {
        head: Vec<f64>,
        value: f64,
        tail: Vec<f64>,
    },
    /// A head, then a block of `block.len()` elements repeated, each repeat
    /// `step` more than the last (an `arange`, a `repeat_interleave`'s
    /// indices, an upsampler's staircase or its interpolation weights),
    /// then a tail: rebuilt in the graph from the length, so nothing grows
    /// with it.
    Periodic {
        head: Vec<f64>,
        block: Vec<f64>,
        step: f64,
        tail: Tail,
        /// The constant is these reversed (a countdown: `n − 1, …, 0`).
        reversed: bool,
    },
    /// The first elements of one longer sequence: the param `name`, a
    /// table built at the longest length.
    Table(String),
    /// As many elements at each length, those that differ whole numbers
    /// (a length baked as a scalar, a shape's dims): each such element
    /// counted out in the graph — `sum(ones[v + shift]) − shift`, the
    /// length a symbol's — the rest kept. `None` keeps an element.
    Sizes(Vec<Option<i64>>),
    /// A mask over its last two axes (the rest one wide): `hi` where the
    /// column is at least `k` past the row, `lo` elsewhere (a causal
    /// attention mask, `triu`): rebuilt from the row and column indices.
    Triangle { k: i64, lo: f64, hi: f64 },
}

/// `v` (`[…1, rows, cols]`) as [`Lift::Triangle`]: its `k`, `lo` and `hi`.
fn triangle(v: &[f64], shape: &rlx_ir::Shape) -> Option<(i64, f64, f64)> {
    let dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
    let r = dims.len();
    if r < 2 || dims[..r - 2].iter().any(|d| *d != 1) {
        return None;
    }
    let (rows, cols) = (dims[r - 2], dims[r - 1]);
    if rows < 2 || cols < 2 {
        return None;
    }
    // Row 0's first value is `lo` or `hi`; the first column that changes is `k`.
    let first = &v[..cols];
    let k = first.iter().position(|x| *x != first[0])? as i64;
    let (lo, hi) = (first[0], first[k as usize]);
    if first[k as usize..].iter().any(|x| *x != hi) {
        return None;
    }
    (0..rows)
        .all(|i| {
            (0..cols).all(|j| {
                let want = if j as i64 - i as i64 >= k { hi } else { lo };
                v[i * cols + j] == want
            })
        })
        .then_some((k, lo, hi))
}

/// What follows a [`Lift::Periodic`]'s repeats.
#[derive(PartialEq)]
enum Tail {
    /// These values.
    Fixed(Vec<f64>),
    /// The last repeated element, this many times more (an index clamped
    /// to the end).
    Edge(usize),
}

/// Element `i` of a [`Lift::Periodic`]'s repeats, as the graph computes it
/// (in f32: the repeat's index times `step`, plus the block's element).
fn periodic_at(block: &[f64], step: f64, i: usize) -> f64 {
    let k = (i / block.len()) as f32;
    f64::from(block[i % block.len()] as f32 + k * step as f32)
}

/// `v` as a head, repeats of a block of `p` elements (each `step` more
/// than the last) and a tail, the repeats around the middle of `v`.
fn periodic_fit(v: &[f64], p: usize, dtype: rlx_ir::DType) -> Option<(usize, f64, usize, Tail)> {
    let n = v.len();
    if n < 2 * p + 1 {
        return None;
    }
    let d = |i: usize| v[i] - v[i - p];
    let mid = n / 2;
    let step = d(mid);
    let mut s = mid;
    while s > p && d(s - 1) == step {
        s -= 1;
    }
    let mut e = mid + 1;
    while e < n && d(e) == step {
        e += 1;
    }
    let h = s - p;
    if e - h < 2 * p {
        return None;
    }
    let block = &v[h..h + p];
    let exact = |want: f64, got: f64| bytes_of(&[want], dtype) == bytes_of(&[got], dtype);
    if !(h..e).all(|i| exact(v[i], periodic_at(block, step, i - h))) {
        return None;
    }
    let tail = &v[e..];
    let tail = if !tail.is_empty() && tail.iter().all(|x| *x == v[e - 1]) {
        Tail::Edge(tail.len())
    } else {
        Tail::Fixed(tail.to_vec())
    };
    Some((h, step, p, tail))
}

/// [`Lift::Periodic`] for a constant `va` at one length and `vb` at
/// another, if both are the same head, block, step and tail.
fn periodic(va: &[f64], vb: &[f64], dtype: rlx_ir::DType) -> Option<Lift> {
    periodic_from(va, vb, dtype, false).or_else(|| {
        let rev = |v: &[f64]| v.iter().rev().copied().collect::<Vec<_>>();
        periodic_from(&rev(va), &rev(vb), dtype, true)
    })
}

fn periodic_from(va: &[f64], vb: &[f64], dtype: rlx_ir::DType, reversed: bool) -> Option<Lift> {
    (1..=(va.len() / 2).min(4096)).find_map(|p| {
        let (ha, sa, _, ta) = periodic_fit(va, p, dtype)?;
        let (hb, sb, _, tb) = periodic_fit(vb, p, dtype)?;
        (ha == hb && sa == sb && ta == tb && va[..ha + p] == vb[..hb + p]).then(|| Lift::Periodic {
            head: va[..ha].to_vec(),
            block: va[ha..ha + p].to_vec(),
            step: sa,
            tail: ta,
            reversed,
        })
    })
}

/// [`symbolic_in`] for a builder that bakes constants from the length —
/// what an ONNX import does (index tables, `arange`s, an overlap-add's
/// coverage counts). Each constant whose data differs between the builds
/// at `a` and `b` (after the unused ones are pruned) is made symbolic:
///
/// - a head, a run of one value and a tail (the head and tail the same in
///   both): rebuilt as their concatenation, the run expanded to its length;
/// - a head, a block repeated (each repeat a constant step more than the
///   last) and a tail — fixed, or the last element repeated: rebuilt in
///   the graph, the repeats' offsets an `arange` (`Op::Cumsum` of ones);
/// - the smaller build's data the start of the larger's: a param (f32,
///   cast back to the constant's type), held as the table `build` makes at
///   `longest`, its first elements read ([`broadcast_params`]).
///
/// Returns the graph and those tables, its params besides `build`'s own.
pub fn symbolic_in_lifted(
    sym: u32,
    (a, b, c): (usize, usize, usize),
    longest: usize,
    build: impl Fn(usize) -> Result<Graph>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    symbolic_in_lifted_sizes(&[(sym, a, b, c)], &[longest], |n| build(n[0]))
}

/// [`symbolic_in_lifted`] in several symbols: `sizes` lists each with its
/// size in the three builds, `longest` each one's for the tables, and
/// `build` takes one size per symbol, in that order (a flow decoder's
/// text and latent lengths).
pub fn symbolic_in_lifted_sizes(
    sizes: &[(u32, usize, usize, usize)],
    longest: &[usize],
    build: impl Fn(&[usize]) -> Result<Graph>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let sizes: Vec<(u32, Vec<usize>)> = sizes
        .iter()
        .map(|&(s, a, b, c)| (s, vec![a, b, c]))
        .collect();
    symbolic_in_lifted_many(&sizes, longest, build)
}

/// [`symbolic_in_lifted_sizes`] built at any number of sizes past two
/// (each symbol's size in each build): the first two fit, every one checks
/// — three can fit `⌊n / 7⌋` where a strided conv's dim is `⌈n / 8⌉`.
pub fn symbolic_in_lifted_many(
    sizes: &[(u32, Vec<usize>)],
    longest: &[usize],
    build: impl Fn(&[usize]) -> Result<Graph>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let builds = sizes.first().map_or(0, |(_, v)| v.len());
    if builds < 3 {
        bail!("a graph is generalized from three sizes or more");
    }
    let at = |k: usize| -> Vec<usize> { sizes.iter().map(|(_, v)| v[k]).collect() };
    let (a, b) = (at(0), at(1));
    let prep = |n: &[usize]| -> Result<Graph> { Ok(rlx_ir::retarget::prune(&build(n)?)) };
    let (ga, gb) = (prep(&a)?, prep(&b)?);
    if ga.nodes().len() != gb.nodes().len() {
        bail!(
            "the graph has {} nodes at {a:?} and {} at {b:?}: not one graph at two lengths",
            ga.nodes().len(),
            gb.nodes().len()
        );
    }
    let mut lifts: HashMap<usize, Lift> = HashMap::new();
    for (i, (x, y)) in ga.nodes().iter().zip(gb.nodes()).enumerate() {
        let (Op::Constant { data: da }, Op::Constant { data: db }) = (&x.op, &y.op) else {
            continue;
        };
        if da == db {
            continue;
        }
        let dtype = x.shape.dtype();
        let (Some(va), Some(vb)) = (elements(da, dtype), elements(db, dtype)) else {
            bail!("constant {} ({dtype:?}) depends on the length", x.id);
        };
        if let (Some(ta), Some(tb)) = (triangle(&va, &x.shape), triangle(&vb, &y.shape))
            && ta == tb
        {
            let (k, lo, hi) = ta;
            lifts.insert(i, Lift::Triangle { k, lo, hi });
            continue;
        }
        let whole = |v: &f64| v.fract() == 0.0 && v.abs() < f64::from(1 << 24);
        if va.len() == vb.len() && va.len() <= 64 && va.iter().chain(&vb).all(whole) {
            // Each element that differs, shifted so its count is at least
            // one at either length.
            let shifts = va
                .iter()
                .zip(&vb)
                .map(|(x, y)| (x != y).then(|| (1 - x.min(*y) as i64).max(0)))
                .collect();
            lifts.insert(i, Lift::Sizes(shifts));
            continue;
        }
        if va.len() >= vb.len() {
            let head = |v: &[f64]| v.iter().take(8).copied().collect::<Vec<_>>();
            bail!(
                "constant {} differs between the lengths but does not grow with them \
                 ({:?} vs {:?})",
                x.id,
                head(&va),
                head(&vb)
            );
        }
        let p = va.iter().zip(&vb).take_while(|(u, v)| u == v).count();
        let s = va
            .iter()
            .rev()
            .zip(vb.iter().rev())
            .take_while(|(u, v)| u == v)
            .count()
            .min(va.len() - p.min(va.len()));
        let mid_a = &va[p.min(va.len())..va.len() - s];
        let mid_b = &vb[p..vb.len() - s];
        let one = |m: &[f64]| m.windows(2).all(|w| w[0] == w[1]);
        let run = !mid_b.is_empty()
            && one(mid_b)
            && one(mid_a)
            && mid_a.first().is_none_or(|v| *v == mid_b[0]);
        let lift = if run {
            // The run's value belongs to the run: the head and the tail
            // give back what they share of it, so a shorter length still
            // fits.
            let value = mid_b[0];
            let mut head = va[..p.min(va.len())].to_vec();
            while head.last() == Some(&value) {
                head.pop();
            }
            let mut tail = va[va.len() - s..].to_vec();
            while tail.first() == Some(&value) {
                tail.remove(0);
            }
            Lift::Run { head, value, tail }
        } else if let Some(lift) = periodic(&va, &vb, dtype) {
            lift
        } else if vb.starts_with(&va) {
            Lift::Table(format!("{}.table.{i}", ga.name))
        } else {
            bail!(
                "constant {} ({} and {}) depends on the length otherwise than as a run, \
                 repeats or a table: {} and {}",
                x.id,
                x.shape,
                y.shape,
                sketch(&va),
                sketch(&vb)
            );
        };
        lifts.insert(i, lift);
    }
    let rewrite = |g: &Graph| -> Result<Graph> {
        rlx_ir::stateful::rebuild(g, |out, _, node, _| {
            let Some(lift) = lifts.get(&(node.id.0 as usize)) else {
                return Ok(None);
            };
            let dtype = node.shape.dtype();
            let n = node
                .shape
                .num_elements()
                .ok_or("a lifted constant is static")?;
            let flat = rlx_ir::Shape::new(&[n], dtype);
            let id = match lift {
                Lift::Run { head, value, tail } if head.is_empty() && tail.is_empty() => {
                    // One value throughout: expanded to the constant's own
                    // shape, each dim its own (`[1, frames − prompt, 100]`,
                    // not `100·(frames − prompt)`).
                    let one = out.add_node(
                        Op::Constant {
                            data: bytes_of(&[*value], dtype),
                        },
                        vec![],
                        rlx_ir::Shape::new(&[1], dtype),
                    );
                    let dims: Vec<i64> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static() as i64)
                        .collect();
                    return Ok(Some(out.add_node(
                        Op::Expand { target_shape: dims },
                        vec![one],
                        node.shape.clone(),
                    )));
                }
                Lift::Run { head, value, tail } => {
                    let mid = n - head.len() - tail.len();
                    let mut parts = Vec::new();
                    let constant = |out: &mut Graph, v: &[f64]| {
                        out.add_node(
                            Op::Constant {
                                data: bytes_of(v, dtype),
                            },
                            vec![],
                            rlx_ir::Shape::new(&[v.len()], dtype),
                        )
                    };
                    if !head.is_empty() {
                        parts.push(constant(out, head));
                    }
                    let one = constant(out, &[*value]);
                    parts.push(out.add_node(
                        Op::Expand {
                            target_shape: vec![mid as i64],
                        },
                        vec![one],
                        rlx_ir::Shape::new(&[mid], dtype),
                    ));
                    if !tail.is_empty() {
                        parts.push(constant(out, tail));
                    }
                    if parts.len() == 1 {
                        parts[0]
                    } else {
                        out.add_node(Op::Concat { axis: 0 }, parts, flat.clone())
                    }
                }
                Lift::Periodic {
                    head,
                    block,
                    step,
                    tail,
                    reversed,
                } => {
                    let f = rlx_ir::DType::F32;
                    let fixed = match tail {
                        Tail::Fixed(t) => t.len(),
                        Tail::Edge(t) => *t,
                    };
                    let (p, m) = (block.len(), n - head.len() - fixed);
                    let k = m.div_ceil(p);
                    let constant = |out: &mut Graph, v: &[f64], dtype| {
                        out.add_node(
                            Op::Constant {
                                data: bytes_of(v, dtype),
                            },
                            vec![],
                            rlx_ir::Shape::new(&[v.len()], dtype),
                        )
                    };
                    let b = constant(out, block, f);
                    let b = out.reshape(b, vec![1, p as i64], rlx_ir::Shape::new(&[1, p], f));
                    let mut body = out.add_node(
                        Op::Expand {
                            target_shape: vec![k as i64, p as i64],
                        },
                        vec![b],
                        rlx_ir::Shape::new(&[k, p], f),
                    );
                    if *step != 0.0 {
                        // The repeat's index, `arange(k)`, times the step.
                        let one = constant(out, &[1.0], f);
                        let ones = out.add_node(
                            Op::Expand {
                                target_shape: vec![k as i64],
                            },
                            vec![one],
                            rlx_ir::Shape::new(&[k], f),
                        );
                        let idx = out.add_node(
                            Op::Cumsum {
                                axis: 0,
                                exclusive: true,
                            },
                            vec![ones],
                            rlx_ir::Shape::new(&[k], f),
                        );
                        let st = constant(out, &[*step], f);
                        let off = out.binary(
                            rlx_ir::op::BinaryOp::Mul,
                            idx,
                            st,
                            rlx_ir::Shape::new(&[k], f),
                        );
                        let off =
                            out.reshape(off, vec![k as i64, 1], rlx_ir::Shape::new(&[k, 1], f));
                        body = out.binary(
                            rlx_ir::op::BinaryOp::Add,
                            body,
                            off,
                            rlx_ir::Shape::new(&[k, p], f),
                        );
                    }
                    let body =
                        out.reshape(body, vec![(k * p) as i64], rlx_ir::Shape::new(&[k * p], f));
                    let body = if k * p == m {
                        body
                    } else {
                        out.narrow_(body, 0, 0, m)
                    };
                    let body = if dtype == f {
                        body
                    } else {
                        out.add_node(
                            Op::Cast { to: dtype },
                            vec![body],
                            rlx_ir::Shape::new(&[m], dtype),
                        )
                    };
                    let mut parts = Vec::new();
                    if !head.is_empty() {
                        parts.push(constant(out, head, dtype));
                    }
                    parts.push(body);
                    match tail {
                        Tail::Fixed(t) if !t.is_empty() => parts.push(constant(out, t, dtype)),
                        Tail::Edge(t) => {
                            let last = out.narrow_(body, 0, m - 1, 1);
                            parts.push(out.add_node(
                                Op::Expand {
                                    target_shape: vec![*t as i64],
                                },
                                vec![last],
                                rlx_ir::Shape::new(&[*t], dtype),
                            ));
                        }
                        Tail::Fixed(_) => {}
                    }
                    let all = if parts.len() == 1 {
                        parts[0]
                    } else {
                        out.add_node(Op::Concat { axis: 0 }, parts, flat.clone())
                    };
                    if *reversed {
                        out.slice_(all, 0, n - 1, n, -1)
                    } else {
                        all
                    }
                }
                Lift::Sizes(shifts) => {
                    let Op::Constant { data } = &node.op else {
                        return Err("a lifted size is a constant".into());
                    };
                    let values = elements(data, dtype).ok_or("a size's type")?;
                    let f = rlx_ir::DType::F32;
                    let one = rlx_ir::Shape::new(&[1], f);
                    let parts: Vec<_> = values
                        .iter()
                        .zip(shifts)
                        .map(|(v, shift)| match shift {
                            None => out.add_node(
                                Op::Constant {
                                    data: bytes_of(&[*v], f),
                                },
                                vec![],
                                one.clone(),
                            ),
                            Some(shift) => {
                                let n = (*v as i64 + shift) as usize;
                                let c = out.add_node(
                                    Op::Constant {
                                        data: bytes_of(&[1.0], f),
                                    },
                                    vec![],
                                    one.clone(),
                                );
                                let ones = out.add_node(
                                    Op::Expand {
                                        target_shape: vec![n as i64],
                                    },
                                    vec![c],
                                    rlx_ir::Shape::new(&[n], f),
                                );
                                let count = out.sum(ones, vec![0], true);
                                if *shift == 0 {
                                    count
                                } else {
                                    let k = out.add_node(
                                        Op::Constant {
                                            data: bytes_of(&[*shift as f64], f),
                                        },
                                        vec![],
                                        one.clone(),
                                    );
                                    out.sub(count, k)
                                }
                            }
                        })
                        .collect();
                    let all = if parts.len() == 1 {
                        parts[0]
                    } else {
                        out.add_node(Op::Concat { axis: 0 }, parts, rlx_ir::Shape::new(&[n], f))
                    };
                    if dtype == f {
                        all
                    } else {
                        out.add_node(Op::Cast { to: dtype }, vec![all], flat.clone())
                    }
                }
                Lift::Triangle { k, lo, hi } => {
                    let dims: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let r = dims.len();
                    let (rows, cols) = (dims[r - 2], dims[r - 1]);
                    let f = rlx_ir::DType::F32;
                    let c = |out: &mut Graph, v: f64| {
                        out.add_node(
                            Op::Constant {
                                data: bytes_of(&[v], f),
                            },
                            vec![],
                            rlx_ir::Shape::new(&[1], f),
                        )
                    };
                    // `arange(n)` shaped `shape`.
                    let iota = |out: &mut Graph, n: usize, shape: &[usize]| {
                        let one = c(out, 1.0);
                        let ones = out.add_node(
                            Op::Expand {
                                target_shape: vec![n as i64],
                            },
                            vec![one],
                            rlx_ir::Shape::new(&[n], f),
                        );
                        let idx = out.add_node(
                            Op::Cumsum {
                                axis: 0,
                                exclusive: true,
                            },
                            vec![ones],
                            rlx_ir::Shape::new(&[n], f),
                        );
                        let to: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
                        out.reshape(idx, to, rlx_ir::Shape::new(shape, f))
                    };
                    let row = iota(out, rows, &[rows, 1]);
                    let col = iota(out, cols, &[1, cols]);
                    let grid = rlx_ir::Shape::new(&[rows, cols], f);
                    let diff = out.binary(rlx_ir::op::BinaryOp::Sub, col, row, grid.clone());
                    let kk = c(out, *k as f64);
                    let on = out.add_node(
                        Op::Compare(rlx_ir::op::CmpOp::Ge),
                        vec![diff, kk],
                        rlx_ir::Shape::new(&[rows, cols], rlx_ir::DType::Bool),
                    );
                    let on = out.add_node(Op::Cast { to: f }, vec![on], grid.clone());
                    let span = c(out, hi - lo);
                    let m = out.binary(rlx_ir::op::BinaryOp::Mul, on, span, grid.clone());
                    let base = c(out, *lo);
                    let m = out.binary(rlx_ir::op::BinaryOp::Add, m, base, grid.clone());
                    let m = if dtype == f {
                        m
                    } else {
                        out.add_node(
                            Op::Cast { to: dtype },
                            vec![m],
                            rlx_ir::Shape::new(&[rows, cols], dtype),
                        )
                    };
                    // Straight to its own shape: flattened, a dim would be
                    // `rows · cols`, a square.
                    let to: Vec<i64> = dims.iter().map(|d| *d as i64).collect();
                    return Ok(Some(out.add_node(
                        Op::Reshape { new_shape: to },
                        vec![m],
                        node.shape.clone(),
                    )));
                }
                Lift::Table(name) => {
                    let p = out.param(name.clone(), rlx_ir::Shape::new(&[n], rlx_ir::DType::F32));
                    if dtype == rlx_ir::DType::F32 {
                        p
                    } else {
                        out.add_node(Op::Cast { to: dtype }, vec![p], flat.clone())
                    }
                }
            };
            let dims: Vec<i64> = node
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static() as i64)
                .collect();
            Ok(Some(out.add_node(
                Op::Reshape { new_shape: dims },
                vec![id],
                node.shape.clone(),
            )))
        })
        .map_err(anyhow::Error::msg)
    };
    let (ra, rb) = (rewrite(&ga)?, rewrite(&gb)?);
    let rest: Vec<Graph> = (2..builds)
        .map(|k| rewrite(&prep(&at(k))?))
        .collect::<Result<_>>()?;
    let mut all: Vec<&Graph> = vec![&ra, &rb];
    all.extend(rest.iter());
    let g = rlx_ir::dynamic::generalize_many(&all, sizes, Some(1 << 16))
        .map_err(|e| anyhow::anyhow!("not symbolic in its length: {e}"))?;
    // A package holds a param at one shape: one as long as the length (a
    // fill the import made a param) would be wrong at every other.
    // (A table the lifts made is held at the longest length below.)
    let lifted = |name: &str| {
        lifts
            .values()
            .any(|l| matches!(l, Lift::Table(t) if t == name))
    };
    if let Some(n) = g.nodes().iter().find(|n| {
        matches!(&n.op, Op::Param { name } if !lifted(name))
            && n.shape.dims().iter().any(|d| !d.is_static())
    }) {
        bail!(
            "param {} ({}) is as long as the length: bake it as a constant",
            n.op,
            n.shape
        );
    }
    for (k, rc) in (2..builds).zip(&rest) {
        let mut bind = rlx_ir::DimBinding::new();
        for (sym, v) in sizes {
            bind.set(*sym, v[k]);
        }
        let (got, want) = (
            format!("{}", rlx_ir::dynamic::specialize(&g, &bind)),
            format!("{rc}"),
        );
        if got != want {
            bail!(
                "the graph generalized from {a:?} and {b:?} is not the one built at {:?}: {}",
                at(k),
                first_difference(&got, &want)
            );
        }
    }
    let tables: Vec<(usize, String)> = lifts
        .iter()
        .filter_map(|(i, l)| match l {
            Lift::Table(name) => Some((*i, name.clone())),
            Lift::Run { .. } | Lift::Periodic { .. } | Lift::Sizes(..) | Lift::Triangle { .. } => {
                None
            }
        })
        .collect();
    if tables.is_empty() {
        return Ok((g, HashMap::new()));
    }
    // Each table at the longest length, and its values at `b` for the check.
    let gl = prep(longest)?;
    let mut at_b = HashMap::new();
    let mut full: Vec<(String, Vec<f32>)> = Vec::new();
    for (i, name) in &tables {
        let values = |g: &Graph| -> Result<Vec<f32>> {
            match &g.nodes()[*i].op {
                Op::Constant { data } => Ok(elements(data, g.nodes()[*i].shape.dtype())
                    .context("a table's type")?
                    .into_iter()
                    .map(|v| v as f32)
                    .collect()),
                _ => bail!("node {i} of the longest build is not the table"),
            }
        };
        at_b.insert(name.clone(), values(&gb)?);
        full.push((name.clone(), values(&gl)?));
    }
    let table_refs: Vec<(&str, &[f32], usize)> = full
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice(), 1))
        .collect();
    let g = broadcast_params(&g, &rb, &mut at_b, &table_refs)?;
    Ok((g, at_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, GraphExt, Shape};
    use rlx_runtime::model::Tensor;

    /// `y = relu(x · w) + b`, then `z = y · 2` in a second stage.
    fn stages() -> (Vec<Stage>, HashMap<String, Vec<f32>>) {
        let f = DType::F32;
        let mut a = Graph::new("layer");
        let x = a.input("x", Shape::new(&[1, 3], f));
        let w = a.param("w", Shape::new(&[3, 2], f));
        let b = a.param("b", Shape::new(&[1, 2], f));
        let xw = a.mm(x, w);
        let r = a.relu(xw);
        let y = a.add(r, b);
        a.set_outputs(vec![y]);
        let mut c = Graph::new("double");
        let h = c.input("h", Shape::new(&[1, 2], f));
        let two = c.constant(2.0, f);
        let z = c.mul(h, two);
        c.set_outputs(vec![z]);
        let params = HashMap::from([
            ("w".to_string(), vec![1.0, -1.0, 2.0, 0.5, -3.0, 1.0]),
            ("b".to_string(), vec![0.25, -0.5]),
        ]);
        (
            vec![
                Stage::new(a).outputs(&["hidden"]),
                Stage::new(c).input("h", "hidden").outputs(&["z"]),
            ],
            params,
        )
    }

    #[test]
    fn stages_run_from_the_package_alone() {
        let (stages, params) = stages();
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("m.rlxp");
        let report = export_stages(
            stages,
            params,
            None,
            "toy",
            "feature-extraction",
            &out,
            &ExportOptions::default(),
        )
        .unwrap();
        assert_eq!(report.functions, ["layer", "double"]);
        assert_eq!(report.held, 2);

        let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
        let x = Tensor::f32("x", vec![1, 3], &[1.0, 2.0, 3.0]);
        let z = m.run(&[x], &json!({})).unwrap()[0].to_f32().unwrap();
        // x·w = [1·1 + 2·2 − 3·3, −1 + 2·0.5 + 3·1] = [−4, 3]; relu → [0, 3];
        // + b → [0.25, 2.5]; · 2.
        assert_eq!(z, [0.5, 5.0]);
    }

    /// A table `[1, t, 4]` (unit axes before the length, as a RoPE table
    /// shaped for broadcasting) held once and its first `t` rows read.
    #[test]
    fn a_table_after_unit_axes_is_held_once_and_read_to_the_length() {
        let f = DType::F32;
        let rows: Vec<f32> = (0..40).map(|i| i as f32 * 0.5).collect();
        let build = |t: usize| {
            let mut g = Graph::new("rot");
            let x = g.input("x", Shape::new(&[1, t, 4], f));
            let c = g.param("cos", Shape::new(&[1, t, 4], f));
            let y = g.mul(x, c);
            g.set_outputs(vec![y]);
            g
        };
        let g = symbolic_in(0, (5, 7, 3), |t| Ok(build(t))).unwrap();
        let mut params = HashMap::from([("cos".to_string(), rows[..12].to_vec())]);
        let g = broadcast_params(&g, &build(3), &mut params, &[("table", &rows, 4)]).unwrap();
        assert_eq!(params.keys().collect::<Vec<_>>(), ["table"]);
        let mut b = rlx_ir::DimBinding::new();
        b.set(0, 6);
        let at6 = rlx_ir::dynamic::specialize(&g, &b);
        let narrow = at6
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Narrow { .. }))
            .expect("the table read");
        assert_eq!(
            narrow.op,
            Op::Narrow {
                axis: 0,
                start: 0,
                len: 6
            }
        );
        assert_eq!(at6.outputs.len(), 1);
        let out = at6.outputs[0];
        assert_eq!(at6.node(out).shape, Shape::new(&[1, 6, 4], f));
    }

    /// A builder baking two constants from the length `n`: an index table
    /// (`0, 1, 4, …, (n−1)²`, as i64) and a coverage count (`1, 2, 2, …, 2, 1`).
    /// Lifted, the graph is one graph symbolic in `n`: specialized at 9 it
    /// is the builder's at 9, the table read from the one built at the
    /// longest length.
    #[test]
    fn baked_tables_and_runs_are_lifted() {
        let f = DType::F32;
        let build = |n: usize| -> Result<Graph> {
            let mut g = Graph::new("baked");
            let x = g.input("x", Shape::new(&[n], f));
            let idx: Vec<u8> = (0..n as i64).flat_map(|v| (v * v).to_le_bytes()).collect();
            let idx = g.add_node(
                Op::Constant { data: idx },
                vec![],
                Shape::new(&[n], DType::I64),
            );
            let idx = g.add_node(Op::Cast { to: f }, vec![idx], Shape::new(&[n], f));
            let mut cover = vec![2.0f32; n];
            cover[0] = 1.0;
            cover[n - 1] = 1.0;
            let cover: Vec<u8> = cover.iter().flat_map(|v| v.to_le_bytes()).collect();
            let cover = g.add_node(Op::Constant { data: cover }, vec![], Shape::new(&[n], f));
            let y = g.mul(x, idx);
            let y = g.add(y, cover);
            g.set_outputs(vec![y]);
            Ok(g)
        };
        let (g, tables) = symbolic_in_lifted(0, (5, 7, 4), 64, build).unwrap();
        assert_eq!(tables.len(), 1, "{:?}", tables.keys());
        let table = tables.values().next().unwrap();
        assert_eq!(table.len(), 64);
        assert_eq!(table[63], 3969.0);
        let mut b = rlx_ir::DimBinding::new();
        b.set(0, 9);
        let at9 = rlx_ir::dynamic::specialize(&g, &b);
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(at9);
        for (k, v) in &tables {
            c.set_param(k, v);
        }
        c.finalize_params();
        let x: Vec<f32> = (0..9).map(|i| i as f32 * 0.5).collect();
        let got = c.run(&[("x", &x)]).remove(0);
        let want: Vec<f32> = (0..9)
            .map(|i| x[i] * (i * i) as f32 + if i == 0 || i == 8 { 1.0 } else { 2.0 })
            .collect();
        assert_eq!(got, want);
    }

    /// A causal mask baked at the length (`[t, t + 1]`, 1 where the column
    /// is past the row): rebuilt from the indices, no table, and the graph
    /// at a length past every build is the builder's there.
    #[test]
    fn a_baked_causal_mask_is_rebuilt_from_the_length() {
        let f = DType::F32;
        let mask = |t: usize| -> Vec<f32> {
            (0..t * (t + 1))
                .map(|e| if e % (t + 1) > e / (t + 1) { 1.0 } else { 0.0 })
                .collect()
        };
        let build = move |t: usize| -> Result<Graph> {
            let mut g = Graph::new("masked");
            let x = g.input("x", Shape::new(&[1, t, t + 1], f));
            let data = mask(t).iter().flat_map(|v| v.to_le_bytes()).collect();
            let m = g.add_node(Op::Constant { data }, vec![], Shape::new(&[1, t, t + 1], f));
            let y = g.mul(x, m);
            g.set_outputs(vec![y]);
            Ok(g)
        };
        let (g, tables) = symbolic_in_lifted(0, (5, 7, 11), 64, build).unwrap();
        assert!(tables.is_empty(), "{:?}", tables.keys());
        let t = 30;
        let mut b = rlx_ir::DimBinding::new();
        b.set(0, t);
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu)
            .compile(rlx_ir::dynamic::specialize(&g, &b));
        c.finalize_params();
        let x: Vec<f32> = (0..t * (t + 1)).map(|i| i as f32 + 1.0).collect();
        let got = c.run(&[("x", &x)]).remove(0);
        let want: Vec<f32> = x.iter().zip(mask(t)).map(|(a, m)| a * m).collect();
        assert_eq!(got, want);
    }

    /// What an upsampler's import bakes at `3·t` samples: each sample's
    /// frame, clamped to the last (`min((i + 1) / 3, t − 1)`, as i64), its
    /// interpolation weight (`(i mod 3) / 4`) and its index. Lifted, they
    /// are built in the graph from the length — no table — and the graph
    /// specialized at a length beyond any built is the builder's there.
    #[test]
    fn repeats_are_rebuilt_from_the_length() {
        let f = DType::F32;
        let frame = |t: usize, i: usize| ((i + 1) / 3).min(t - 1);
        let build = move |t: usize| -> Result<Graph> {
            let n = 3 * t;
            let mut g = Graph::new("upsampled");
            let x = g.input("x", Shape::new(&[n], f));
            let frames: Vec<u8> = (0..n)
                .flat_map(|i| (frame(t, i) as i64).to_le_bytes())
                .collect();
            let frames = g.add_node(
                Op::Constant { data: frames },
                vec![],
                Shape::new(&[n], DType::I64),
            );
            let frames = g.add_node(Op::Cast { to: f }, vec![frames], Shape::new(&[n], f));
            let w: Vec<u8> = (0..n)
                .flat_map(|i| ((i % 3) as f32 * 0.25).to_le_bytes())
                .collect();
            let w = g.add_node(Op::Constant { data: w }, vec![], Shape::new(&[n], f));
            let idx: Vec<u8> = (0..n as i64).flat_map(|v| v.to_le_bytes()).collect();
            let idx = g.add_node(
                Op::Constant { data: idx },
                vec![],
                Shape::new(&[n], DType::I64),
            );
            let idx = g.add_node(Op::Cast { to: f }, vec![idx], Shape::new(&[n], f));
            let y = g.mul(x, frames);
            let y = g.add(y, w);
            let y = g.add(y, idx);
            g.set_outputs(vec![y]);
            Ok(g)
        };
        let (g, tables) = symbolic_in_lifted(0, (5, 7, 4), 64, build).unwrap();
        assert!(tables.is_empty(), "{:?}", tables.keys());
        let t = 80;
        let mut b = rlx_ir::DimBinding::new();
        b.set(0, t);
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu)
            .compile(rlx_ir::dynamic::specialize(&g, &b));
        c.finalize_params();
        let x: Vec<f32> = (0..3 * t).map(|i| i as f32 * 0.5).collect();
        let got = c.run(&[("x", &x)]).remove(0);
        let want: Vec<f32> = (0..3 * t)
            .map(|i| x[i] * frame(t, i) as f32 + (i % 3) as f32 * 0.25 + i as f32)
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn a_stage_naming_the_wrong_number_of_outputs_is_refused() {
        let (mut stages, params) = stages();
        stages[0] = stages[0].clone().outputs(&["a", "b"]);
        let tmp = tempfile::tempdir().unwrap();
        let err = export_stages(
            stages,
            params,
            None,
            "toy",
            "feature-extraction",
            &tmp.path().join("m.rlxp"),
            &ExportOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("names 2 outputs, its graph has 1"), "{err}");
    }
}
