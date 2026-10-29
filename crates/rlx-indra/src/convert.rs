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

//! Export step: the upstream release (`datalab-to/indra-ocr`) → safetensors.
//!
//! Only the *weights* leave the `.onnx` files: every initializer is renamed
//! after the module that consumes it (`/rec/blocks.0/dw/Conv` input 1 →
//! `rec.blocks.0.dw.weight`), so the hand-written graphs in [`crate::graph`]
//! address parameters by module path, not by exporter-generated ids such as
//! `onnx::Conv_504`. The n-gram tables (`lm*.npz`) become safetensors too, and
//! `meta.json` is copied. Nothing here runs at inference time.

use anyhow::{Context, Result, bail};
use safetensors::Dtype;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::Path;

/// Networks the default pipeline reads (`math_*` is not ported).
pub const NETWORKS: &[&str] = &[
    "det",
    "rec",
    "rec_cjk",
    "rec_rtl",
    "rec_world",
    "script",
    "orient",
    "order",
];

/// One tensor bound for safetensors.
pub struct OwnedTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

// ── minimal protobuf reader (ModelProto → nodes + initializers) ─────────────

struct Pb<'a> {
    buf: &'a [u8],
    pos: usize,
}

enum Field<'a> {
    Varint(u64),
    Fixed64,
    Fixed32,
    Len(&'a [u8]),
}

impl<'a> Pb<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).context("truncated varint")?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("varint too long")
    }

    fn next(&mut self) -> Result<Option<(u32, Field<'a>)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = (key >> 3) as u32;
        let f = match key & 7 {
            0 => Field::Varint(self.varint()?),
            1 => {
                self.pos += 8;
                Field::Fixed64
            }
            2 => {
                let n = self.varint()? as usize;
                let s = self
                    .buf
                    .get(self.pos..self.pos + n)
                    .context("truncated field")?;
                self.pos += n;
                Field::Len(s)
            }
            5 => {
                self.pos += 4;
                Field::Fixed32
            }
            w => bail!("unsupported protobuf wire type {w}"),
        };
        Ok(Some((field, f)))
    }
}

struct Node {
    name: String,
    op: String,
    inputs: Vec<String>,
}

struct Initializer {
    name: String,
    dims: Vec<usize>,
    data_type: i32,
    raw: Vec<u8>,
}

fn utf8(b: &[u8]) -> Result<String> {
    Ok(std::str::from_utf8(b)?.to_string())
}

fn parse_node(b: &[u8]) -> Result<Node> {
    let mut n = Node {
        name: String::new(),
        op: String::new(),
        inputs: Vec::new(),
    };
    let mut pb = Pb::new(b);
    while let Some((f, v)) = pb.next()? {
        if let Field::Len(s) = v {
            match f {
                1 => n.inputs.push(utf8(s)?),
                3 => n.name = utf8(s)?,
                4 => n.op = utf8(s)?,
                _ => {}
            }
        }
    }
    Ok(n)
}

fn parse_tensor(b: &[u8]) -> Result<Initializer> {
    let mut t = Initializer {
        name: String::new(),
        dims: Vec::new(),
        data_type: 0,
        raw: Vec::new(),
    };
    let mut floats: Vec<u8> = Vec::new();
    let mut pb = Pb::new(b);
    while let Some((f, v)) = pb.next()? {
        match (f, v) {
            (1, Field::Varint(d)) => t.dims.push(d as usize),
            (1, Field::Len(s)) => {
                let mut p = Pb::new(s);
                while p.pos < s.len() {
                    t.dims.push(p.varint()? as usize);
                }
            }
            (2, Field::Varint(d)) => t.data_type = d as i32,
            (4, Field::Len(s)) => floats.extend_from_slice(s),
            (8, Field::Len(s)) => t.name = utf8(s)?,
            (9, Field::Len(s)) => t.raw = s.to_vec(),
            (14, _) => bail!("external tensor data is not supported"),
            _ => {}
        }
    }
    if t.raw.is_empty() {
        t.raw = floats;
    }
    Ok(t)
}

fn parse_model(bytes: &[u8]) -> Result<(Vec<Node>, Vec<Initializer>)> {
    let mut graph = None;
    let mut pb = Pb::new(bytes);
    while let Some((f, v)) = pb.next()? {
        if let (7, Field::Len(s)) = (f, v) {
            graph = Some(s);
        }
    }
    let graph = graph.context("ONNX model has no graph")?;
    let (mut nodes, mut inits) = (Vec::new(), Vec::new());
    let mut pb = Pb::new(graph);
    while let Some((f, v)) = pb.next()? {
        match (f, v) {
            (1, Field::Len(s)) => nodes.push(parse_node(s)?),
            (5, Field::Len(s)) => inits.push(parse_tensor(s)?),
            _ => {}
        }
    }
    Ok((nodes, inits))
}

// ── naming ──────────────────────────────────────────────────────────────────

/// Module path of an exporter node name: `/stages.0/stages.0.0/Conv` →
/// `stages.0.0` (a scope that extends its parent replaces it), `/rec/blocks.0/dw/Conv`
/// → `rec.blocks.0.dw`.
pub fn module_path(node_name: &str) -> (String, String) {
    let segs: Vec<&str> = node_name.trim_start_matches('/').split('/').collect();
    let (last, scopes) = segs.split_last().map_or(("", &[][..]), |(l, s)| (*l, s));
    let mut path: Vec<&str> = Vec::new();
    for s in scopes {
        match path.last() {
            Some(p) if s.starts_with(&format!("{p}.")) => *path.last_mut().unwrap() = s,
            _ => path.push(s),
        }
    }
    (path.join("."), last.to_string())
}

fn role(op: &str, index: usize, has_matmul: bool, last: &str) -> Option<String> {
    let r = match (op, index) {
        ("Conv" | "Gemm" | "MatMul" | "LayerNormalization", 1) => "weight",
        ("Conv" | "Gemm" | "LayerNormalization", 2) => "bias",
        ("BatchNormalization", 1) => "weight",
        ("BatchNormalization", 2) => "bias",
        ("BatchNormalization", 3) => "running_mean",
        ("BatchNormalization", 4) => "running_var",
        ("Add", _) if has_matmul => "bias",
        ("Add", _) => return Some(last.to_ascii_lowercase()),
        _ => return None,
    };
    Some(r.to_string())
}

/// Every initializer of one `.onnx` file under its module-path name.
pub fn onnx_weights(path: &Path) -> Result<BTreeMap<String, OwnedTensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (nodes, inits) = parse_model(&bytes)?;
    let matmul_paths: std::collections::HashSet<String> = nodes
        .iter()
        .filter(|n| n.op == "MatMul")
        .map(|n| module_path(&n.name).0)
        .collect();
    let mut consumer: HashMap<&str, (&Node, usize)> = HashMap::new();
    for n in &nodes {
        for (i, x) in n.inputs.iter().enumerate() {
            consumer.entry(x.as_str()).or_insert((n, i));
        }
    }
    let mut out = BTreeMap::new();
    for t in &inits {
        let Some((node, idx)) = consumer.get(t.name.as_str()) else {
            continue;
        };
        let (mpath, last) = module_path(&node.name);
        let role =
            role(&node.op, *idx, matmul_paths.contains(&mpath), &last).with_context(|| {
                format!(
                    "{}: no name rule for {} input {idx}",
                    path.display(),
                    node.op
                )
            })?;
        let key = if mpath.is_empty() {
            role
        } else {
            format!("{mpath}.{role}")
        };
        let dtype = match t.data_type {
            1 => Dtype::F32,
            7 => Dtype::I64,
            d => bail!("{}: initializer {} has dtype {d}", path.display(), t.name),
        };
        let n: usize = t.dims.iter().product();
        if t.raw.len() != n * (dtype.bitsize() / 8) {
            bail!(
                "{}: initializer {} has a short payload",
                path.display(),
                t.name
            );
        }
        let tensor = OwnedTensor {
            dtype,
            shape: t.dims.clone(),
            bytes: t.raw.clone(),
        };
        if out.insert(key.clone(), tensor).is_some() {
            bail!("{}: two initializers map to {key}", path.display());
        }
    }
    Ok(out)
}

// ── n-gram tables (`lm*.npz`) ───────────────────────────────────────────────

fn parse_npy(bytes: &[u8]) -> Result<OwnedTensor> {
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        bail!("not an .npy payload");
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen])?;
    let field = |k: &str| -> Result<&str> {
        let at = header
            .find(k)
            .with_context(|| format!("npy header lacks {k}"))?;
        Ok(header[at + k.len()..].trim_start_matches([':', ' ']))
    };
    if field("'fortran_order'")?.starts_with("True") {
        bail!("Fortran-order arrays are not supported");
    }
    let descr = field("'descr'")?;
    let dtype = if descr.starts_with("'<u8'") {
        Dtype::U64
    } else if descr.starts_with("'<f4'") {
        Dtype::F32
    } else if descr.starts_with("'<i8'") {
        Dtype::I64
    } else {
        bail!("unsupported npy dtype {descr}")
    };
    let shape_s = field("'shape'")?;
    let shape_s = &shape_s[1..shape_s.find(')').context("npy shape")?];
    let shape: Vec<usize> = shape_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let data = bytes[start + hlen..].to_vec();
    if data.len() != shape.iter().product::<usize>() * (dtype.bitsize() / 8) {
        bail!("npy payload size does not match its shape");
    }
    Ok(OwnedTensor {
        dtype,
        shape,
        bytes: data,
    })
}

/// Every array of an `.npz` archive, by name (without `.npy`).
pub fn npz_tensors(path: &Path) -> Result<BTreeMap<String, OwnedTensor>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)?;
    let mut out = BTreeMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name()?.trim_end_matches(".npy").to_string();
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        out.insert(
            name.clone(),
            parse_npy(&buf).with_context(|| format!("{}: {name}", path.display()))?,
        );
    }
    Ok(out)
}

pub fn write_safetensors(path: &Path, tensors: &BTreeMap<String, OwnedTensor>) -> Result<()> {
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(k, t)| {
            Ok((
                k.clone(),
                safetensors::tensor::TensorView::new(t.dtype, t.shape.clone(), &t.bytes)?,
            ))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, None, path)
        .with_context(|| format!("writing {}", path.display()))
}

/// Convert an upstream model folder into a native one: `<net>.safetensors`
/// per network, `lm*.safetensors` per n-gram table, and `meta.json`.
pub fn convert_dir(src: &Path, dst: &Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dst)?;
    let mut written = Vec::new();
    for net in NETWORKS {
        let onnx = src.join(format!("{net}.onnx"));
        if !onnx.exists() {
            continue;
        }
        let out = dst.join(format!("{net}.safetensors"));
        write_safetensors(&out, &onnx_weights(&onnx)?)?;
        written.push(out.display().to_string());
    }
    for entry in std::fs::read_dir(src)? {
        let p = entry?.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        if name.starts_with("lm") && name.ends_with(".npz") {
            let out = dst.join(name.replace(".npz", ".safetensors"));
            write_safetensors(&out, &npz_tensors(&p)?)?;
            written.push(out.display().to_string());
        }
    }
    let meta = src.join("meta.json");
    if !meta.exists() {
        bail!("{} lacks meta.json", src.display());
    }
    std::fs::copy(&meta, dst.join("meta.json"))?;
    written.push(dst.join("meta.json").display().to_string());
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::module_path;

    #[test]
    fn module_paths() {
        assert_eq!(
            module_path("/stages.0/stages.0.0/stages.0.0.0/stages.0.0.0.0/Conv"),
            ("stages.0.0.0.0".into(), "Conv".into())
        );
        assert_eq!(
            module_path("/rec/blocks.0/se/fc1/Conv"),
            ("rec.blocks.0.se.fc1".into(), "Conv".into())
        );
        assert_eq!(
            module_path("/m/body/body.2/body/body.0/body.0.0/Conv"),
            ("m.body.2.body.0.0".into(), "Conv".into())
        );
        assert_eq!(module_path("/cls/MatMul"), ("cls".into(), "MatMul".into()));
    }
}
