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

//! Minimal NPZ / NPY loader.
//!
//! Supports the subset of the NumPy array format actually used by KittenTTS:
//!   - NPY format version 1.0 and 2.0
//!   - `float32` dtype (`<f4`, `=f4`)
//!   - C-contiguous (row-major) layout
//!   - Arbitrary number of dimensions (we use 2-D voice matrices)
//!
//! NPZ files are simply ZIP archives whose members are `.npy` files.
//! Each member name without its `.npy` extension is the array name.

use anyhow::{Context, Result, bail};
use std::{collections::HashMap, io::Read, path::Path};
use zip::ZipArchive;

// ─────────────────────────────────────────────────────────────────────────────
// NPY header parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a raw `.npy` byte buffer and return the f32 data as a flat `Vec<f32>`
/// together with the shape.
pub fn parse_npy(data: &[u8]) -> Result<(Vec<usize>, Vec<f32>)> {
    // Magic: 6 bytes "\x93NUMPY"
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        bail!("Not a valid NPY file (bad magic)");
    }

    let major = data[6];
    let minor = data[7];

    // Header length: 2 bytes (v1) or 4 bytes (v2), little-endian.
    let (header_len, header_start) = match (major, minor) {
        (1, _) => {
            let len = u16::from_le_bytes([data[8], data[9]]) as usize;
            (len, 10)
        }
        (2, _) => {
            if data.len() < 12 {
                bail!("NPY v2 file too short");
            }
            let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            (len, 12)
        }
        _ => bail!("Unsupported NPY version {}.{}", major, minor),
    };

    let header_end = header_start + header_len;
    if data.len() < header_end {
        bail!("NPY file truncated in header");
    }
    let header = std::str::from_utf8(&data[header_start..header_end])
        .context("NPY header is not valid UTF-8")?;

    // Parse dtype
    let dtype = extract_header_field(header, "descr").context("NPY header missing 'descr'")?;
    let dtype = dtype.trim().trim_matches('\'').trim_matches('"');

    // We accept little-endian and native-endian float32 only.
    let is_f32 = matches!(dtype, "<f4" | "=f4" | "|f4" | ">f4");
    if !is_f32 {
        bail!("Unsupported dtype '{}' — only float32 is supported", dtype);
    }
    let big_endian = dtype.starts_with('>');

    // Parse fortran_order
    let fortran = extract_header_field(header, "fortran_order")
        .unwrap_or("False")
        .trim()
        .to_ascii_lowercase();
    if fortran == "true" {
        bail!("Fortran-order arrays are not supported");
    }

    // Parse shape — e.g. "(256, 512, )" or "(100,)"
    let shape_str = extract_header_field(header, "shape").context("NPY header missing 'shape'")?;
    let shape = parse_shape(shape_str.trim())?;

    // Total number of elements
    let n_elements: usize = shape.iter().product();

    // Raw bytes start right after the header
    let data_bytes = &data[header_end..];
    if data_bytes.len() < n_elements * 4 {
        bail!(
            "NPY data section too short: expected {} bytes, got {}",
            n_elements * 4,
            data_bytes.len()
        );
    }

    // Read f32 values
    let values: Vec<f32> = data_bytes[..n_elements * 4]
        .chunks_exact(4)
        .map(|b| {
            let arr = [b[0], b[1], b[2], b[3]];
            if big_endian {
                f32::from_be_bytes(arr)
            } else {
                f32::from_le_bytes(arr)
            }
        })
        .collect();

    Ok((shape, values))
}

/// Extract the value of a field from a Python-literal dict header string.
///
/// e.g. `extract_header_field("{'descr': '<f4', 'shape': (3,)}", "descr")`
/// returns `Some("<f4")`.
fn extract_header_field<'a>(header: &'a str, field: &str) -> Option<&'a str> {
    // Look for `'field':` or `"field":`.
    let key_sq = format!("'{}':", field);
    let key_dq = format!("\"{}\":", field);

    let start = header
        .find(key_sq.as_str())
        .map(|p| p + key_sq.len())
        .or_else(|| header.find(key_dq.as_str()).map(|p| p + key_dq.len()))?;

    let rest = header[start..].trim_start();

    // Value is either a Python string (quoted), tuple (parentheses), or a bare word.
    if rest.starts_with('(') {
        // Tuple — find the matching closing paren
        let end = rest.find(')')?;
        Some(&rest[..end + 1])
    } else if rest.starts_with('\'') || rest.starts_with('"') {
        let quote = rest.chars().next()?;
        let inner = &rest[1..];
        let end = inner.find(quote)?;
        Some(&inner[..end])
    } else {
        // Bare value (True, False, or a number) — read until comma or }
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        Some(rest[..end].trim())
    }
}

/// Parse a Python-style shape tuple like `(256, 512, )` or `(100,)` or `()`.
fn parse_shape(s: &str) -> Result<Vec<usize>> {
    let inner = s.trim_start_matches('(').trim_end_matches(')');
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .with_context(|| format!("Bad shape dim: '{}'", t))
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// NPZ loader — returns flat f32 data per array name
// ─────────────────────────────────────────────────────────────────────────────

/// A loaded NPZ entry: shape + flat f32 data in row-major (C) order.
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NpyArray {
    /// Number of rows (first dimension).
    pub fn nrows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Number of columns (second dimension).
    pub fn ncols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }

    /// Get row `i` as a slice of f32 values.
    pub fn row(&self, i: usize) -> &[f32] {
        let ncols = self.ncols();
        &self.data[i * ncols..(i + 1) * ncols]
    }
}

/// Load an NPZ file and return all arrays indexed by name (`.npy` extension stripped).
pub fn load_npz(path: &Path) -> Result<HashMap<String, NpyArray>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open NPZ file: {}", path.display()))?;
    let mut archive = ZipArchive::new(file)
        .with_context(|| format!("Cannot open ZIP archive: {}", path.display()))?;

    let mut arrays = HashMap::new();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).context("Failed to read ZIP entry")?;
        let name = entry.name().trim_end_matches(".npy").to_string();

        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut buf)
            .context("Failed to read NPY entry")?;

        let (shape, data) =
            parse_npy(&buf).with_context(|| format!("Failed to parse NPY entry '{}'", name))?;

        arrays.insert(name, NpyArray { shape, data });
    }

    Ok(arrays)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal v1.0 NPY byte buffer for testing.
    fn make_npy(shape: &[usize], values: &[f32]) -> Vec<u8> {
        let header_str = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        // Pad to multiple of 64 bytes (total header block = 10 + header_len, padded)
        let raw_len = header_str.len() + 1; // +1 for trailing \n
        let padded_len = raw_len.div_ceil(64) * 64;
        let _header_len = padded_len - 1; // without the final \n counted separately
        // Actually NPY spec: padding is spaces, last char is \n, total header_len bytes
        let pad_needed = padded_len - raw_len;
        let mut header = header_str;
        for _ in 0..pad_needed {
            header.push(' ');
        }
        header.push('\n');

        let header_len_u16 = header.len() as u16;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1); // major
        buf.push(0); // minor
        buf.extend_from_slice(&header_len_u16.to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        for &v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_parse_npy_1d() {
        let values = vec![1.0f32, 2.0, 3.0];
        let buf = make_npy(&[3], &values);
        let (shape, data) = parse_npy(&buf).unwrap();
        assert_eq!(shape, vec![3]);
        assert_eq!(data, values);
    }

    #[test]
    fn test_parse_npy_2d() {
        let values: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let buf = make_npy(&[2, 3], &values);
        let (shape, data) = parse_npy(&buf).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, values);
    }

    #[test]
    fn test_npy_array_row() {
        let values: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let buf = make_npy(&[2, 3], &values);
        let (shape, data) = parse_npy(&buf).unwrap();
        let arr = NpyArray { shape, data };
        assert_eq!(arr.row(0), &[0.0, 1.0, 2.0]);
        assert_eq!(arr.row(1), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_bad_magic() {
        let result = parse_npy(b"NOTANPY");
        assert!(result.is_err());
    }
}
