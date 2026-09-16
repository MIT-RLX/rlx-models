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

//! Reader for the Espresso graph triple the NMT actually ships in:
//! `<name>.espresso.net` (JSON layer list), `.espresso.shape` (per-blob
//! geometry) and a `.espresso.weights` blob table.
//!
//! The weights file is named by the net's own `storage` field, **not** by the
//! net's file name: `readout.espresso.net` declares
//! `"storage": "embedding.espresso.weights"` and ships no container of its own,
//! because the readout gathers the tied embedding table.
//!
//! [`crate::espresso::Manifest`] names which graphs exist; this loads one.
//!
//! # Weights container
//!
//! ```text
//!   u64                 blob_count
//!   blob_count ×        { u64 index, u64 size }   // table, in payload order
//!   payloads            concatenated, in table order
//! ```
//!
//! Identical to the container `rlx-neuralhash` reads, so the two ports agree on
//! the format.
//!
//! # Shapes are an oracle, not decoration
//!
//! `.espresso.shape` lists `{_rank, w, h, k, n}` for **every** intermediate
//! blob. That makes it a per-layer correctness check for anything that executes
//! this graph: a wiring or reshape mistake shows up as a shape mismatch at the
//! offending layer instead of as bad output many layers later. See
//! [`Graph::declared_shape`].

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// Geometry of one blob, in Espresso's `(n, k, h, w)` order with `w` innermost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rank: usize,
    pub w: usize,
    pub h: usize,
    pub k: usize,
    pub n: usize,
}

impl Shape {
    /// Element count.
    pub fn len(&self) -> usize {
        self.w * self.h * self.k * self.n
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimensions outermost-first, trimmed to `rank`. Espresso stores `w`
    /// innermost, so a rank-2 `[h, w]` blob is a sequence of `h` rows of `w`.
    pub fn dims(&self) -> Vec<usize> {
        let all = [self.n, self.k, self.h, self.w];
        all[4 - self.rank.clamp(1, 4)..].to_vec()
    }
}

/// One layer of an `.espresso.net`.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    /// Layer `type`, e.g. `inner_product`.
    pub kind: String,
    /// Input blob names, in order.
    pub bottoms: Vec<String>,
    /// Output blob names, in order — several layers emit two (a quantized
    /// tensor and its scale).
    pub tops: Vec<String>,
    /// Everything else, verbatim, so unknown fields survive.
    pub attrs: Map<String, Value>,
}

impl Layer {
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.attrs.get(key)? {
            Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|v| v as i64)),
            Value::String(s) => s.parse().ok(),
            Value::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }

    pub fn float(&self, key: &str) -> Option<f64> {
        match self.attrs.get(key)? {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    pub fn flag(&self, key: &str) -> bool {
        match self.attrs.get(key) {
            Some(Value::Bool(b)) => *b,
            _ => self.int(key).unwrap_or(0) != 0,
        }
    }

    /// Blob index for a named weight, e.g. `W_int8` or `wGamma`.
    pub fn blob(&self, key: &str) -> Option<u64> {
        self.attrs.get("weights")?.as_object()?.get(key)?.as_u64()
    }

    /// Required blob index, erroring with the layer name when absent.
    pub fn req_blob(&self, key: &str) -> Result<u64> {
        self.blob(key).ok_or_else(|| {
            anyhow!(
                "layer {:?} ({}) has no {key:?} weight",
                self.name,
                self.kind
            )
        })
    }
}

/// The binary blob table of an `.espresso.weights`.
#[derive(Debug)]
pub struct Weights {
    bytes: Vec<u8>,
    /// Blob index → (offset, size) into `bytes`.
    spans: BTreeMap<u64, (usize, usize)>,
    /// Decoded f32 blobs, kept because the hot path re-reads them.
    ///
    /// `i8s` hands out a slice of `bytes`, but f32 blobs are little-endian and
    /// have to be decoded, so `f32s` allocates. Biases are read once per
    /// `dynamic_dequantize` and there are 2826 of those per translation — the
    /// same few hundred kilobytes decoded thousands of times. A `Mutex` costs
    /// ~20ns against an allocation and a decode loop.
    decoded: std::sync::Mutex<BTreeMap<u64, std::sync::Arc<[f32]>>>,
}

impl Weights {
    /// Parses the container.
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        ensure!(
            bytes.len() >= 8,
            "weights file is too short for a blob count"
        );
        let count = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) as usize;
        let table_end = 8 + count * 16;
        ensure!(
            table_end <= bytes.len(),
            "weights blob table of {count} entries overruns the file"
        );
        // Payloads follow the table in table order.
        let mut spans = BTreeMap::new();
        let mut order = Vec::with_capacity(count);
        for i in 0..count {
            let at = 8 + i * 16;
            let idx = u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"));
            let size = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().expect("eight bytes"))
                as usize;
            order.push((idx, size));
        }
        let mut pos = table_end;
        for (idx, size) in order {
            ensure!(
                pos + size <= bytes.len(),
                "weights blob {idx} of {size} bytes overruns the file"
            );
            spans.insert(idx, (pos, size));
            pos += size;
        }
        Ok(Self {
            bytes,
            spans,
            decoded: std::sync::Mutex::new(BTreeMap::new()),
        })
    }

    /// Reads a container from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).with_context(|| format!("reading weights {}", path.display()))?;
        Self::parse(bytes).with_context(|| format!("parsing weights {}", path.display()))
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Raw bytes of a blob.
    pub fn raw(&self, idx: u64) -> Result<&[u8]> {
        let (off, size) = *self
            .spans
            .get(&idx)
            .ok_or_else(|| anyhow!("weights have no blob {idx}"))?;
        Ok(&self.bytes[off..off + size])
    }

    /// Blob as signed bytes (`W_int8`).
    pub fn i8s(&self, idx: u64) -> Result<&[i8]> {
        let raw = self.raw(idx)?;
        // SAFETY-free reinterpretation: i8 and u8 have identical layout.
        Ok(unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<i8>(), raw.len()) })
    }

    /// Blob as little-endian f32, decoded once and shared thereafter.
    ///
    /// Prefer this on any path that runs per layer per step; [`Weights::f32s`]
    /// still exists for the one-shot callers that want an owned `Vec`.
    pub fn f32s_shared(&self, idx: u64) -> Result<std::sync::Arc<[f32]>> {
        if let Some(hit) = self
            .decoded
            .lock()
            .expect("weight cache mutex")
            .get(&idx)
            .cloned()
        {
            return Ok(hit);
        }
        let v: std::sync::Arc<[f32]> = self.f32s(idx)?.into();
        self.decoded
            .lock()
            .expect("weight cache mutex")
            .insert(idx, std::sync::Arc::clone(&v));
        Ok(v)
    }

    /// Blob as little-endian f32 (`biases`, `wGamma`, `Q_meta`).
    pub fn f32s(&self, idx: u64) -> Result<Vec<f32>> {
        let raw = self.raw(idx)?;
        ensure!(
            raw.len() % 4 == 0,
            "blob {idx} is {} bytes, not a whole number of f32",
            raw.len()
        );
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
            .collect())
    }
}

/// A loaded Espresso graph: layers, declared shapes and weights.
#[derive(Debug)]
pub struct Graph {
    pub name: String,
    pub layers: Vec<Layer>,
    /// Blob name → geometry, from `.espresso.shape`.
    pub shapes: BTreeMap<String, Shape>,
    pub weights: Weights,
    /// The container `weights` came from, per the net's `storage` field.
    pub storage: String,
}

impl Graph {
    /// Loads `<dir>/<stem>.espresso.{net,shape,weights}`.
    pub fn load(dir: impl AsRef<Path>, net_file: &str) -> Result<Self> {
        let dir = dir.as_ref();
        let stem = net_file
            .strip_suffix(".espresso.net")
            .ok_or_else(|| anyhow!("{net_file:?} is not an .espresso.net"))?;
        let net_path = dir.join(net_file);
        let net: Value = serde_json::from_slice(
            &std::fs::read(&net_path).with_context(|| format!("reading {}", net_path.display()))?,
        )
        .with_context(|| format!("parsing {}", net_path.display()))?;

        let layers = Self::layers_from_value(&net)
            .with_context(|| format!("reading layers of {}", net_path.display()))?;

        let shape_path = dir.join(format!("{stem}.espresso.shape"));
        let shapes = if shape_path.exists() {
            parse_shapes(&std::fs::read(&shape_path)?)
                .with_context(|| format!("parsing {}", shape_path.display()))?
        } else {
            BTreeMap::new()
        };

        // `storage` names the container; several nets share one (the readout
        // reads the embedding table). Fall back to the conventional name only
        // if the field is absent.
        let storage = net
            .get("storage")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{stem}.espresso.weights"));
        let weights = Weights::load(dir.join(&storage))
            .with_context(|| format!("{net_file} declares storage {storage:?}"))?;

        Ok(Self {
            name: stem.to_string(),
            layers,
            shapes,
            weights,
            storage,
        })
    }

    /// Parses the `layers` array of an `.espresso.net` value.
    pub fn layers_from_value(net: &Value) -> Result<Vec<Layer>> {
        net.get("layers")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("net has no layers array"))?
            .iter()
            .map(|l| {
                let o = l
                    .as_object()
                    .ok_or_else(|| anyhow!("layer is not an object"))?;
                let split = |key: &str| -> Vec<String> {
                    o.get(key)
                        .and_then(Value::as_str)
                        .map(|s| {
                            s.split(',')
                                .map(str::trim)
                                .filter(|p| !p.is_empty())
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default()
                };
                Ok(Layer {
                    name: o
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    kind: o
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    bottoms: split("bottom"),
                    tops: split("top"),
                    attrs: o.clone(),
                })
            })
            .collect()
    }

    /// Blobs consumed but never produced — the graph's inputs, in first-use
    /// order.
    pub fn inputs(&self) -> Vec<String> {
        let produced: std::collections::BTreeSet<&str> = self
            .layers
            .iter()
            .flat_map(|l| l.tops.iter().map(String::as_str))
            .collect();
        let mut seen = Vec::new();
        for l in &self.layers {
            for b in &l.bottoms {
                if !produced.contains(b.as_str()) && !seen.contains(b) {
                    seen.push(b.clone());
                }
            }
        }
        seen
    }

    /// The graph's terminal blob.
    pub fn output(&self) -> Result<&str> {
        self.layers
            .last()
            .and_then(|l| l.tops.first())
            .map(String::as_str)
            .ok_or_else(|| anyhow!("graph {:?} has no layers", self.name))
    }

    /// Blobs a layer explicitly marks `is_output` — the decoder marks its
    /// next-state tensors this way.
    pub fn marked_outputs(&self) -> Vec<String> {
        self.layers
            .iter()
            .filter(|l| {
                l.attrs
                    .get("attributes")
                    .and_then(Value::as_object)
                    .and_then(|a| a.get("is_output"))
                    .and_then(|v| v.as_i64().or_else(|| v.as_bool().map(i64::from)))
                    .unwrap_or(0)
                    != 0
            })
            .filter_map(|l| l.tops.first().cloned())
            .collect()
    }

    /// Declared geometry of a blob, when `.espresso.shape` covers it.
    pub fn declared_shape(&self, blob: &str) -> Option<Shape> {
        self.shapes.get(blob).copied()
    }

    /// Distinct layer types present, for coverage checks.
    pub fn layer_kinds(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for l in &self.layers {
            *out.entry(l.kind.clone()).or_insert(0) += 1;
        }
        out
    }

    /// Every layer whose type is not in `known`.
    pub fn unsupported(&self, known: &[&str]) -> Vec<&Layer> {
        self.layers
            .iter()
            .filter(|l| !known.contains(&l.kind.as_str()))
            .collect()
    }
}

fn parse_shapes(bytes: &[u8]) -> Result<BTreeMap<String, Shape>> {
    let v: Value = serde_json::from_slice(bytes)?;
    let table = v
        .get("layer_shapes")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("shape file has no layer_shapes"))?;
    let mut out = BTreeMap::new();
    for (name, s) in table {
        let o = s
            .as_object()
            .ok_or_else(|| anyhow!("shape entry {name:?} is not an object"))?;
        let get = |k: &str| o.get(k).and_then(Value::as_u64).unwrap_or(1) as usize;
        let rank = o.get("_rank").and_then(Value::as_u64).unwrap_or(4) as usize;
        if rank == 0 || rank > 4 {
            bail!("shape entry {name:?} has unsupported rank {rank}");
        }
        out.insert(
            name.clone(),
            Shape {
                rank,
                w: get("w"),
                h: get("h"),
                k: get("k"),
                n: get("n"),
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {

    #[test]
    fn decoded_f32_blobs_are_shared_not_re_decoded() {
        // One blob, index 7, holding [1.0, -2.0].
        let mut bytes = 1u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(&8u64.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.0f32).to_le_bytes());
        let w = super::Weights::parse(bytes).expect("parse");

        let a = w.f32s_shared(7).expect("first");
        let b = w.f32s_shared(7).expect("second");
        assert_eq!(&*a, &[1.0, -2.0]);
        // The point of the cache: the second call decodes nothing.
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        // And it agrees with the uncached reader it replaced.
        assert_eq!(&*a, w.f32s(7).expect("owned").as_slice());
        assert!(w.f32s_shared(8).is_err(), "unknown blob should still error");
    }
    use super::*;

    fn container(blobs: &[(u64, Vec<u8>)]) -> Vec<u8> {
        let mut out = (blobs.len() as u64).to_le_bytes().to_vec();
        for (idx, payload) in blobs {
            out.extend_from_slice(&idx.to_le_bytes());
            out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        }
        for (_, payload) in blobs {
            out.extend_from_slice(payload);
        }
        out
    }

    #[test]
    fn weights_table_indexes_payloads_in_table_order() {
        let w = Weights::parse(container(&[
            (0, vec![1, 2, 3]),
            (7, vec![9; 4]),
            (3, vec![]),
        ]))
        .expect("parses");
        assert_eq!(w.len(), 3);
        assert_eq!(w.raw(0).expect("blob 0"), &[1, 2, 3]);
        assert_eq!(w.raw(7).expect("blob 7"), &[9, 9, 9, 9]);
        assert!(w.raw(3).expect("blob 3").is_empty());
        assert!(w.raw(99).is_err());
    }

    #[test]
    fn f32_blobs_decode_little_endian() {
        let payload: Vec<u8> = [1.5f32, -2.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = Weights::parse(container(&[(1, payload)])).expect("parses");
        assert_eq!(w.f32s(1).expect("f32s"), vec![1.5, -2.25]);
    }

    #[test]
    fn odd_sized_f32_blob_is_rejected() {
        let w = Weights::parse(container(&[(1, vec![0, 1, 2])])).expect("parses");
        assert!(w.f32s(1).is_err(), "3 bytes is not a whole number of f32");
    }

    #[test]
    fn i8_blobs_reinterpret_the_bytes() {
        let w = Weights::parse(container(&[(1, vec![0, 127, 128, 255])])).expect("parses");
        assert_eq!(w.i8s(1).expect("i8s"), &[0i8, 127, -128, -1]);
    }

    #[test]
    fn a_truncated_container_is_rejected() {
        let mut bytes = container(&[(0, vec![1, 2, 3])]);
        bytes.truncate(bytes.len() - 2);
        assert!(Weights::parse(bytes).is_err());
        assert!(Weights::parse(vec![0, 1]).is_err());
    }

    #[test]
    fn shape_dims_are_outermost_first_with_w_innermost() {
        let s = Shape {
            rank: 2,
            w: 512,
            h: 64,
            k: 1,
            n: 1,
        };
        assert_eq!(s.dims(), vec![64, 512]);
        assert_eq!(s.len(), 512 * 64);
        let s3 = Shape {
            rank: 3,
            w: 64,
            h: 8,
            k: 2,
            n: 1,
        };
        assert_eq!(s3.dims(), vec![2, 8, 64]);
    }

    #[test]
    fn shapes_parse_from_the_shipped_json_form() {
        let json = br#"{"layer_shapes":{"embedding":{"_rank":2,"h":64,"k":1,"n":1,"w":512}}}"#;
        let s = parse_shapes(json).expect("parses");
        assert_eq!(
            s["embedding"],
            Shape {
                rank: 2,
                w: 512,
                h: 64,
                k: 1,
                n: 1
            }
        );
    }
}
