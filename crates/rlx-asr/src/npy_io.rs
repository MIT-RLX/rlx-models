// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Minimal `.npy` / `.npz` readers for ASR weight packing.

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
}

#[derive(Debug)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub dtype: NpyDType,
    pub data: Vec<u8>,
}

impl NpyArray {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Parse a complete `.npy` buffer (header + payload).
pub fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    let major = bytes[6];
    let (header_len, header_off) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10usize)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated npy v2 header");
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12usize)
        }
        other => bail!("unsupported npy version {other}"),
    };
    let header_end = header_off + header_len;
    if bytes.len() < header_end {
        bail!("truncated npy header");
    }
    let header = std::str::from_utf8(&bytes[header_off..header_end]).context("npy header utf8")?;
    let dtype = parse_descr(header)?;
    let shape = parse_shape(header)?;
    let fortran = parse_fortran_order(header);
    let n = shape.iter().product::<usize>();
    let elem = dtype_bytes(dtype);
    let need = n.saturating_mul(elem);
    let payload = &bytes[header_end..];
    if payload.len() < need {
        bail!(
            "npy payload short: have {} need {need} (shape={shape:?})",
            payload.len()
        );
    }
    let mut data = payload[..need].to_vec();
    if fortran {
        data = transpose_fortran_to_c(&data, &shape, elem)?;
    }
    Ok(NpyArray { shape, dtype, data })
}

fn dtype_bytes(d: NpyDType) -> usize {
    match d {
        NpyDType::F32 | NpyDType::I32 => 4,
        NpyDType::F64 | NpyDType::I64 => 8,
        NpyDType::I8 | NpyDType::U8 => 1,
        NpyDType::I16 => 2,
    }
}

fn parse_descr(header: &str) -> Result<NpyDType> {
    let descr = extract_quoted_field(header, "descr")
        .ok_or_else(|| anyhow::anyhow!("npy header missing descr"))?;
    let d = descr.trim_start_matches(['<', '>', '=', '|']);
    Ok(match d {
        "f4" | "f32" => NpyDType::F32,
        "f8" | "f64" => NpyDType::F64,
        "i1" | "int8" => NpyDType::I8,
        "i2" => NpyDType::I16,
        "i4" => NpyDType::I32,
        "i8" => NpyDType::I64,
        "u1" | "uint8" => NpyDType::U8,
        other => bail!("unsupported npy dtype {other}"),
    })
}

fn extract_quoted_field<'a>(header: &'a str, field: &str) -> Option<&'a str> {
    for q1 in ['\'', '"'] {
        for q2 in ['\'', '"'] {
            let pat = format!("{q1}{field}{q1}: {q2}");
            if let Some(i) = header.find(&pat) {
                let rest = &header[i + pat.len()..];
                let end = rest.find(q2)?;
                return Some(&rest[..end]);
            }
        }
    }
    None
}

fn parse_fortran_order(header: &str) -> bool {
    header.contains("'fortran_order': True")
        || header.contains("\"fortran_order\": True")
        || header.contains("'fortran_order':True")
        || header.contains("\"fortran_order\":True")
}

fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let Some(i) = header.find("shape") else {
        bail!("npy header missing shape");
    };
    let rest = &header[i..];
    let Some(open) = rest.find('(') else {
        bail!("npy shape '('");
    };
    let rest = &rest[open + 1..];
    let Some(close) = rest.find(')') else {
        bail!("npy shape ')'");
    };
    let inner = rest[..close].trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(
            p.parse::<usize>()
                .with_context(|| format!("shape dim {p}"))?,
        );
    }
    Ok(shape)
}

fn transpose_fortran_to_c(src: &[u8], shape: &[usize], elem: usize) -> Result<Vec<u8>> {
    match shape.len() {
        0 | 1 => Ok(src.to_vec()),
        2 => {
            let rows = shape[0];
            let cols = shape[1];
            let mut dst = vec![0u8; src.len()];
            for i in 0..rows {
                for j in 0..cols {
                    let s = (j * rows + i) * elem;
                    let d = (i * cols + j) * elem;
                    dst[d..d + elem].copy_from_slice(&src[s..s + elem]);
                }
            }
            Ok(dst)
        }
        3 => {
            let a = shape[0];
            let b = shape[1];
            let c = shape[2];
            let mut dst = vec![0u8; src.len()];
            for i in 0..a {
                for j in 0..b {
                    for k in 0..c {
                        let s = (i + a * j + a * b * k) * elem;
                        let d = ((i * b + j) * c + k) * elem;
                        dst[d..d + elem].copy_from_slice(&src[s..s + elem]);
                    }
                }
            }
            Ok(dst)
        }
        n => bail!("fortran-order rank-{n} not supported"),
    }
}

/// Load every array from an `.npz` (skips object / unsupported dtypes).
pub fn load_npz(path: &Path) -> Result<Vec<(String, NpyArray)>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut zip = ZipArchive::new(f).with_context(|| format!("npz {}", path.display()))?;
    let mut out = Vec::new();
    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        let name = file.name().to_string();
        if name.ends_with('/') {
            continue;
        }
        let stem = name
            .strip_suffix(".npy")
            .unwrap_or(name.as_str())
            .to_string();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        match parse_npy(&buf) {
            Ok(arr) => out.push((stem, arr)),
            Err(e) => {
                eprintln!(
                    "[rlx-asr] skip npz entry {} in {}: {e:#}",
                    name,
                    path.display()
                );
            }
        }
    }
    Ok(out)
}

/// Read a raw little-endian `.npy` file from disk.
pub fn load_npy(path: &Path) -> Result<NpyArray> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_npy(&bytes)
}

/// Read silence-fbank / similar whitespace-separated float text.
pub fn read_f32_txt(path: &Path) -> Result<Vec<f32>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(s.split_whitespace()
        .filter_map(|t| t.parse::<f32>().ok())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_descr_from_numpy_header() {
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (512, 512), }";
        assert_eq!(parse_descr(h).unwrap(), NpyDType::F32);
        assert!(!parse_fortran_order(h));
        assert_eq!(parse_shape(h).unwrap(), vec![512, 512]);
    }
}
