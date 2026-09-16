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

//! Reader for Apple's Espresso network container — the format the NeuralHash
//! model ships in (`NeuralHashv3b_fp16-current.espresso.net`, `.shape`,
//! `.weights`).
//!
//! Reading it directly is what makes the port native: the architecture comes
//! from the vendor's own layer description rather than a third-party ONNX
//! conversion, and every op is then emitted as an rlx-ir op by [`crate::flow`].
//!
//! # Container layout
//!
//! * **`.net`** — JSON. `{"storage": ..., "format_version": ..., "layers": [...]}`.
//!   Each layer carries `type`, `name`, `top`, `bottom` and type-specific
//!   fields; weights are referenced as integer indices into the blob table.
//! * **`.shape`** — JSON. Per-blob `{k, w, h, n}` (channels, width, height,
//!   batch) for every intermediate tensor. Optional, but when present it is
//!   the authoritative activation geometry and we validate against it.
//! * **`.weights`** — binary blob table:
//!
//!   ```text
//!     u64            blob_count
//!     blob_count ×   { u64 index, u64 size }      // table, in payload order
//!     payloads       concatenated, in table order
//!   ```
//!
//! Convolution kernels are stored f16 as `[C_out, C_in/groups, kH, kW]`;
//! biases are f32 `[C_out]`.

use anyhow::{Context, Result, ensure};
use half::f16;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// One layer of a `.espresso.net`, kept as raw JSON so unknown fields survive
/// for [`EspressoNet::layer_report`] rather than being silently dropped.
#[derive(Debug, Clone)]
pub struct EspressoLayer {
    pub name: String,
    pub kind: String,
    /// Output blob names (Espresso writes a single `top`, occasionally a list).
    pub top: Vec<String>,
    /// Input blob names, in order.
    pub bottom: Vec<String>,
    /// All remaining fields, verbatim.
    pub attrs: HashMap<String, Value>,
}

impl EspressoLayer {
    /// First present integer-valued field among `keys`.
    pub fn int(&self, keys: &[&str]) -> Option<i64> {
        for k in keys {
            match self.attrs.get(*k) {
                Some(Value::Number(n)) => {
                    if let Some(v) = n.as_i64() {
                        return Some(v);
                    }
                    if let Some(v) = n.as_f64() {
                        return Some(v as i64);
                    }
                }
                Some(Value::String(s)) => {
                    if let Ok(v) = s.parse::<i64>() {
                        return Some(v);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// [`Self::int`] with a default.
    pub fn int_or(&self, keys: &[&str], default: i64) -> i64 {
        self.int(keys).unwrap_or(default)
    }

    /// Required integer field; errors naming the layer when absent.
    pub fn req_int(&self, keys: &[&str]) -> Result<i64> {
        self.int(keys).ok_or_else(|| {
            anyhow::anyhow!(
                "espresso layer {:?} ({}) is missing any of {keys:?}",
                self.name,
                self.kind
            )
        })
    }

    /// Truthy test for Espresso's 0/1 flags.
    pub fn flag(&self, keys: &[&str]) -> bool {
        match self.attrs.get(keys[0]) {
            Some(Value::Bool(b)) => return *b,
            Some(_) => {}
            None => {
                for k in &keys[1..] {
                    if let Some(Value::Bool(b)) = self.attrs.get(*k) {
                        return *b;
                    }
                }
            }
        }
        self.int(keys).unwrap_or(0) != 0
    }

    /// Blob index referenced by any of `keys`, looking in both the layer body
    /// and its nested `weights` object (Espresso uses both spellings).
    pub fn blob(&self, keys: &[&str]) -> Option<u64> {
        if let Some(v) = self.int(keys) {
            return u64::try_from(v).ok();
        }
        let nested = self.attrs.get("weights")?.as_object()?;
        for k in keys {
            if let Some(v) = nested.get(*k).and_then(|v| v.as_u64()) {
                return Some(v);
            }
        }
        None
    }
}

/// Activation geometry of one blob, from the `.shape` sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

/// A parsed Espresso network: layers, blob shapes and the weight blob table.
pub struct EspressoNet {
    pub format_version: i64,
    pub layers: Vec<EspressoLayer>,
    /// Blob name → activation shape, from the `.shape` file (may be empty).
    pub shapes: HashMap<String, BlobShape>,
    blobs: BlobTable,
}

impl EspressoNet {
    /// Read `<stem>.espresso.net` / `.shape` / `.weights` from disk.
    ///
    /// `net` may point at any of the three siblings; the others are located by
    /// swapping the extension, so `--net NeuralHashv3b.espresso.net` is enough.
    pub fn open(net: impl AsRef<Path>) -> Result<Self> {
        let net = net.as_ref();
        let stem = net
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("espresso path is not UTF-8: {}", net.display()))?
            .trim_end_matches(".net")
            .trim_end_matches(".shape")
            .trim_end_matches(".weights")
            .to_string();
        let net_path = format!("{stem}.net");
        let shape_path = format!("{stem}.shape");
        let weights_path = format!("{stem}.weights");
        Self::open_parts(
            Path::new(&net_path),
            Path::new(&weights_path),
            Some(Path::new(&shape_path)),
        )
    }

    /// Read explicit `.net` / `.weights` (+ optional `.shape`) paths.
    pub fn open_parts(net: &Path, weights: &Path, shape: Option<&Path>) -> Result<Self> {
        let net_json = read_maybe_compressed(net)
            .with_context(|| format!("reading espresso net {}", net.display()))?;
        let net_json = String::from_utf8(net_json)
            .with_context(|| format!("espresso net {} is not UTF-8 JSON", net.display()))?;
        let raw = std::fs::read(weights)
            .with_context(|| format!("reading espresso weights {}", weights.display()))?;
        let shapes = match shape {
            Some(p) if p.is_file() => {
                let s = read_maybe_compressed(p)
                    .with_context(|| format!("reading espresso shapes {}", p.display()))?;
                let s = String::from_utf8(s).with_context(|| {
                    format!("espresso shapes {} is not UTF-8 JSON", p.display())
                })?;
                parse_shapes(&s).with_context(|| format!("parsing {}", p.display()))?
            }
            _ => HashMap::new(),
        };
        Self::from_parts(&net_json, &raw, shapes)
            .with_context(|| format!("parsing espresso network {}", net.display()))
    }

    /// Parse from in-memory `.net` JSON + `.weights` bytes.
    pub fn from_parts(
        net_json: &str,
        weights: &[u8],
        shapes: HashMap<String, BlobShape>,
    ) -> Result<Self> {
        let v: Value = serde_json::from_str(net_json).context("espresso .net is not valid JSON")?;
        let format_version = v
            .get("format_version")
            .and_then(|x| x.as_i64())
            .unwrap_or(0);
        let raw_layers = v
            .get("layers")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("espresso .net has no `layers` array"))?;

        let mut layers = Vec::with_capacity(raw_layers.len());
        for (i, l) in raw_layers.iter().enumerate() {
            let obj = l
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("espresso layer {i} is not an object"))?;
            let kind = obj
                .get("type")
                .and_then(|x| x.as_str())
                .ok_or_else(|| anyhow::anyhow!("espresso layer {i} has no `type`"))?
                .to_string();
            let name = obj
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or(&format!("{kind}_{i}"))
                .to_string();
            let attrs: HashMap<String, Value> = obj
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "type" | "name" | "top" | "bottom"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            layers.push(EspressoLayer {
                name,
                kind,
                top: blob_names(obj.get("top")),
                bottom: blob_names(obj.get("bottom")),
                attrs,
            });
        }

        Ok(Self {
            format_version,
            layers,
            shapes,
            blobs: BlobTable::parse(weights)?,
        })
    }

    /// Layer-type histogram, for `--inspect` and for error messages that need
    /// to say what an unsupported network actually contains.
    pub fn op_histogram(&self) -> Vec<(String, usize)> {
        let mut h: HashMap<&str, usize> = HashMap::new();
        for l in &self.layers {
            *h.entry(l.kind.as_str()).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = h.into_iter().map(|(k, n)| (k.to_string(), n)).collect();
        v.sort();
        v
    }

    /// Human-readable one-line-per-layer dump.
    pub fn layer_report(&self) -> String {
        let mut s = String::new();
        for (i, l) in self.layers.iter().enumerate() {
            s.push_str(&format!(
                "{i:>4}  {:<20} {:<32} {:?} -> {:?}\n",
                l.kind, l.name, l.bottom, l.top
            ));
        }
        s
    }

    /// Decode blob `idx` as f16 → f32.
    pub fn blob_f16(&self, idx: u64) -> Result<Vec<f32>> {
        let raw = self.blobs.payload(idx)?;
        ensure!(
            raw.len() % 2 == 0,
            "espresso blob {idx}: {} bytes is not a whole number of f16",
            raw.len()
        );
        Ok(raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect())
    }

    /// Decode blob `idx` as f32.
    pub fn blob_f32(&self, idx: u64) -> Result<Vec<f32>> {
        let raw = self.blobs.payload(idx)?;
        ensure!(
            raw.len() % 4 == 0,
            "espresso blob {idx}: {} bytes is not a whole number of f32",
            raw.len()
        );
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Byte length of blob `idx`.
    pub fn blob_len(&self, idx: u64) -> Option<usize> {
        self.blobs.sizes.get(&idx).copied()
    }

    /// Number of weight blobs in the container.
    pub fn blob_count(&self) -> usize {
        self.blobs.sizes.len()
    }
}

/// `{u64 count} {count × (u64 index, u64 size)} {payloads…}`.
struct BlobTable {
    raw: Vec<u8>,
    sizes: HashMap<u64, usize>,
    offsets: HashMap<u64, usize>,
}

impl BlobTable {
    fn parse(raw: &[u8]) -> Result<Self> {
        ensure!(
            raw.len() >= 8,
            "espresso .weights is {} bytes — too short for the blob count header",
            raw.len()
        );
        let count = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let table_bytes = (count as usize)
            .checked_mul(16)
            .ok_or_else(|| anyhow::anyhow!("espresso .weights: absurd blob count {count}"))?;
        ensure!(
            raw.len() >= 8 + table_bytes,
            "espresso .weights: {count} blobs need {} header bytes but the file is {}",
            8 + table_bytes,
            raw.len()
        );

        let mut sizes = HashMap::new();
        let mut order = Vec::with_capacity(count as usize);
        let mut o = 8usize;
        for _ in 0..count {
            let idx = u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
            let size = u64::from_le_bytes(raw[o + 8..o + 16].try_into().unwrap()) as usize;
            o += 16;
            sizes.insert(idx, size);
            order.push((idx, size));
        }
        let mut offsets = HashMap::new();
        let mut cur = o;
        for (idx, size) in order {
            offsets.insert(idx, cur);
            cur = cur
                .checked_add(size)
                .ok_or_else(|| anyhow::anyhow!("espresso .weights: blob {idx} size overflows"))?;
        }
        ensure!(
            cur == raw.len(),
            "espresso .weights: payloads end at {cur} but the file is {} bytes \
             (count={count}) — container layout mismatch",
            raw.len()
        );
        Ok(Self {
            raw: raw.to_vec(),
            sizes,
            offsets,
        })
    }

    fn payload(&self, idx: u64) -> Result<&[u8]> {
        let off = *self
            .offsets
            .get(&idx)
            .ok_or_else(|| anyhow::anyhow!("espresso .weights has no blob {idx}"))?;
        let size = self.sizes[&idx];
        Ok(&self.raw[off..off + size])
    }
}

/// Espresso writes `top`/`bottom` as a single name, a comma-separated list, or
/// an array. Normalize all three.
fn blob_names(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => s
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.to_string())
            .collect(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str())
            .map(|s| s.to_string())
            .collect(),
        _ => Vec::new(),
    }
}

/// Parse `.espresso.shape`: `{"layer_shapes": {"blob": {"k":C,"w":W,"h":H,"n":N}}}`.
fn parse_shapes(json: &str) -> Result<HashMap<String, BlobShape>> {
    let v: Value = serde_json::from_str(json).context("espresso .shape is not valid JSON")?;
    let map = v
        .get("layer_shapes")
        .and_then(|x| x.as_object())
        .or_else(|| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("espresso .shape has no `layer_shapes` object"))?;
    let mut out = HashMap::new();
    for (name, s) in map {
        let Some(o) = s.as_object() else { continue };
        let dim = |k: &str, d: u64| o.get(k).and_then(|x| x.as_u64()).unwrap_or(d) as usize;
        out.insert(
            name.clone(),
            BlobShape {
                n: dim("n", 1),
                c: dim("k", 1),
                h: dim("h", 1),
                w: dim("w", 1),
            },
        );
    }
    Ok(out)
}

/// Convenience: is this file plausibly an Espresso `.net`?
pub fn looks_like_espresso_net(path: &Path) -> bool {
    path.to_str().is_some_and(|s| s.ends_with(".net"))
}

// ---------------------------------------------------------------------------
// `pbze` container (LZFSE-compressed `.net` / `.shape`)
// ---------------------------------------------------------------------------

/// Magic of the compressed Espresso text container.
const PBZE_MAGIC: &[u8; 4] = b"pbze";
/// Header size; the LZFSE stream starts here.
const PBZE_HEADER: usize = 0x1c;

/// Read a `.net` / `.shape` sidecar, transparently decompressing the `pbze`
/// container.
///
/// Earlier OS versions shipped these as plain JSON. Current macOS (26.x) wraps
/// them in a `pbze` header around an LZFSE stream:
///
/// ```text
///   0x00  "pbze"
///   0x08  u64 le   block size (0x40)
///   0x10  u32 be   decompressed size
///   0x18  u32 be   compressed size
///   0x1c  LZFSE stream ("bvx2" …)
/// ```
///
/// Both forms are accepted, so one code path handles the older layout and the
/// container `Vision.framework` installs today.
pub fn read_maybe_compressed(path: &Path) -> Result<Vec<u8>> {
    let raw = std::fs::read(path)?;
    if raw.len() < PBZE_HEADER || &raw[..4] != PBZE_MAGIC {
        return Ok(raw); // plain JSON
    }
    let declared_out = u32::from_be_bytes(raw[0x10..0x14].try_into().unwrap()) as usize;
    let declared_in = u32::from_be_bytes(raw[0x18..0x1c].try_into().unwrap()) as usize;
    let payload = &raw[PBZE_HEADER..];
    ensure!(
        declared_in == payload.len(),
        "espresso pbze container: header declares {declared_in} compressed bytes \
         but {} follow the header",
        payload.len()
    );
    let out = lzfse_decode(payload, declared_out)?;
    ensure!(
        out.len() == declared_out,
        "espresso pbze container: decompressed {} bytes, header declares {declared_out}",
        out.len()
    );
    Ok(out)
}

/// Decode an LZFSE stream to exactly `capacity` bytes.
///
/// Uses the system `libcompression` (the same decoder Espresso itself calls),
/// so there is no third-party codec to keep in sync. These containers only
/// exist inside Apple OS images, so the non-Apple path just explains how to
/// pre-decompress.
#[cfg(target_os = "macos")]
fn lzfse_decode(src: &[u8], capacity: usize) -> Result<Vec<u8>> {
    /// `compression_algorithm` value for LZFSE.
    const COMPRESSION_LZFSE: i32 = 0x801;

    #[link(name = "compression")]
    unsafe extern "C" {
        fn compression_decode_buffer(
            dst: *mut u8,
            dst_size: usize,
            src: *const u8,
            src_size: usize,
            scratch: *mut core::ffi::c_void,
            algorithm: i32,
        ) -> usize;
    }

    // `compression_decode_buffer` reports truncation by returning `dst_size`,
    // so allocate one spare byte and treat a full buffer as a failure.
    let mut out = vec![0u8; capacity + 1];
    let n = unsafe {
        compression_decode_buffer(
            out.as_mut_ptr(),
            out.len(),
            src.as_ptr(),
            src.len(),
            std::ptr::null_mut(),
            COMPRESSION_LZFSE,
        )
    };
    ensure!(n > 0, "LZFSE decode failed (libcompression returned 0)");
    ensure!(
        n <= capacity,
        "LZFSE decode overran the declared size ({n} > {capacity})"
    );
    out.truncate(n);
    Ok(out)
}

#[cfg(not(target_os = "macos"))]
fn lzfse_decode(_src: &[u8], _capacity: usize) -> Result<Vec<u8>> {
    anyhow::bail!(
        "this Espresso file uses the LZFSE-compressed `pbze` container, which is \
         decoded here via Apple's libcompression and so needs macOS. Decompress the \
         `.net` / `.shape` on a Mac first — the plain-JSON form is accepted on every \
         platform."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `.weights` container the way Espresso lays one out.
    fn weights_blob(entries: &[(u64, Vec<u8>)]) -> Vec<u8> {
        let mut out = (entries.len() as u64).to_le_bytes().to_vec();
        for (idx, data) in entries {
            out.extend(idx.to_le_bytes());
            out.extend((data.len() as u64).to_le_bytes());
        }
        for (_, data) in entries {
            out.extend(data);
        }
        out
    }

    #[test]
    fn blob_table_indexes_payloads_in_table_order() {
        // Non-contiguous, out-of-order indices — offsets follow the table, not
        // the index value.
        let a: Vec<u8> = vec![1, 2, 3, 4];
        let b: Vec<u8> = vec![9, 9];
        let raw = weights_blob(&[(7, a.clone()), (2, b.clone())]);
        let t = BlobTable::parse(&raw).unwrap();
        assert_eq!(t.payload(7).unwrap(), a.as_slice());
        assert_eq!(t.payload(2).unwrap(), b.as_slice());
        assert!(t.payload(99).is_err());
    }

    #[test]
    fn truncated_or_padded_containers_are_rejected() {
        let raw = weights_blob(&[(0, vec![1, 2, 3, 4])]);
        assert!(BlobTable::parse(&raw).is_ok());
        assert!(BlobTable::parse(&raw[..raw.len() - 1]).is_err());
        let mut padded = raw.clone();
        padded.push(0);
        assert!(BlobTable::parse(&padded).is_err());
        assert!(BlobTable::parse(&[0u8; 4]).is_err());
        // A bogus count must not panic or allocate wildly.
        let mut bogus = u64::MAX.to_le_bytes().to_vec();
        bogus.extend([0u8; 8]);
        assert!(BlobTable::parse(&bogus).is_err());
    }

    #[test]
    fn f16_and_f32_blob_decoding() {
        let f16s: Vec<u8> = [1.0f32, -2.5, 0.5]
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let f32s: Vec<u8> = [3.0f32, -4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let raw = weights_blob(&[(0, f16s), (1, f32s)]);
        let net = EspressoNet::from_parts(
            r#"{"format_version": 200, "layers": []}"#,
            &raw,
            HashMap::new(),
        )
        .unwrap();
        assert_eq!(net.blob_f16(0).unwrap(), vec![1.0, -2.5, 0.5]);
        assert_eq!(net.blob_f32(1).unwrap(), vec![3.0, -4.0]);
        // Wrong-width decode is caught, not silently truncated.
        assert!(net.blob_f32(0).is_err()); // 6 bytes is not a multiple of 4
    }

    #[test]
    fn parses_layers_tops_and_bottoms() {
        let net_json = r#"{
          "format_version": 200,
          "layers": [
            {"type": "input", "name": "image", "top": "image", "bottom": "", "weights": {}},
            {"type": "convolution", "name": "c1", "top": "y", "bottom": "image",
             "C": 16, "K": 3, "Nx": 3, "Ny": 3, "n_groups": 1, "has_biases": 1,
             "blob_weights_f16": 0, "blob_biases": 1},
            {"type": "elementwise", "name": "add", "top": "z", "bottom": "y,image"}
          ]
        }"#;
        let raw = weights_blob(&[(0, vec![0; 2]), (1, vec![0; 4])]);
        let net = EspressoNet::from_parts(net_json, &raw, HashMap::new()).unwrap();
        assert_eq!(net.format_version, 200);
        assert_eq!(net.layers.len(), 3);
        assert_eq!(net.layers[0].bottom, Vec::<String>::new());
        assert_eq!(net.layers[1].top, vec!["y"]);
        // Comma-separated multi-input bottoms must split.
        assert_eq!(net.layers[2].bottom, vec!["y", "image"]);
        let conv = &net.layers[1];
        assert_eq!(conv.req_int(&["C"]).unwrap(), 16);
        assert_eq!(conv.int_or(&["stride_x", "stride"], 1), 1);
        assert!(conv.flag(&["has_biases"]));
        assert_eq!(conv.blob(&["blob_weights_f16"]), Some(0));
        assert_eq!(
            net.op_histogram(),
            vec![
                ("convolution".into(), 1),
                ("elementwise".into(), 1),
                ("input".into(), 1)
            ]
        );
    }

    #[test]
    fn nested_weights_object_blob_lookup() {
        let net_json = r#"{"layers": [
            {"type": "inner_product", "name": "ip", "top": "o", "bottom": "i",
             "nB": 4, "nC": 2, "weights": {"w_f16_t": 5, "b_f32": 6}}
        ]}"#;
        let raw = weights_blob(&[(5, vec![0; 16]), (6, vec![0; 8])]);
        let net = EspressoNet::from_parts(net_json, &raw, HashMap::new()).unwrap();
        let ip = &net.layers[0];
        assert_eq!(ip.blob(&["blob_weights_f16", "w_f16_t"]), Some(5));
        assert_eq!(ip.blob(&["blob_biases", "b_f32"]), Some(6));
    }

    #[test]
    fn shape_sidecar_parsing() {
        let s = r#"{"layer_shapes": {"image": {"k": 3, "w": 360, "h": 360, "n": 1}}}"#;
        let shapes = parse_shapes(s).unwrap();
        assert_eq!(
            shapes["image"],
            BlobShape {
                n: 1,
                c: 3,
                h: 360,
                w: 360
            }
        );
    }

    #[test]
    fn missing_layers_array_is_an_error() {
        assert!(
            EspressoNet::from_parts(
                r#"{"format_version": 200}"#,
                &weights_blob(&[]),
                HashMap::new()
            )
            .is_err()
        );
        assert!(EspressoNet::from_parts("not json", &weights_blob(&[]), HashMap::new()).is_err());
    }
}
