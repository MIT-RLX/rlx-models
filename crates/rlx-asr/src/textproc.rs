// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Hammer OpenFST spelling chains + etiquette profanity map.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Minimal OpenFST (tropical) acceptor/transducer loader — enough for `TP/*.fst`.
#[derive(Debug, Clone)]
pub struct Fst {
    pub start: u32,
    pub finals: HashMap<u32, f32>,
    /// (state, ilabel) -> Vec<(next, olabel, weight)>
    pub arcs: HashMap<(u32, u32), Vec<(u32, u32, f32)>>,
}

impl Fst {
    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_bytes(&data).with_context(|| format!("parse {}", path.display()))
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        parse_openfst_binary(data)
    }

    /// Compose a string (UTF-8 as byte labels) through the FST; returns best output string.
    pub fn apply_bytes(&self, input: &str) -> String {
        let labels: Vec<u32> = input.bytes().map(|b| b as u32).collect();
        match self.best_path(&labels) {
            Some(out) => {
                let bytes: Vec<u8> = out.into_iter().filter(|&c| c > 0 && c < 256).map(|c| c as u8).collect();
                String::from_utf8_lossy(&bytes).into_owned()
            }
            None => input.to_string(),
        }
    }

    fn best_path(&self, input: &[u32]) -> Option<Vec<u32>> {
        // Beam-1 Viterbi over (pos, state).
        let mut beam: HashMap<(usize, u32), (f32, Vec<u32>)> = HashMap::new();
        beam.insert((0, self.start), (0.0, Vec::new()));
        for pos in 0..=input.len() {
            let keys: Vec<_> = beam.keys().filter(|(p, _)| *p == pos).cloned().collect();
            for (p, st) in keys {
                let (cost, outs) = beam.get(&(p, st)).cloned()?;
                // epsilon arcs
                if let Some(arcs) = self.arcs.get(&(st, 0)) {
                    for &(nxt, ol, w) in arcs {
                        let mut no = outs.clone();
                        if ol != 0 {
                            no.push(ol);
                        }
                        let nc = cost + w;
                        let e = beam.entry((p, nxt)).or_insert((f32::INFINITY, Vec::new()));
                        if nc < e.0 {
                            *e = (nc, no);
                        }
                    }
                }
                if pos < input.len() {
                    let lab = input[pos];
                    if let Some(arcs) = self.arcs.get(&(st, lab)) {
                        for &(nxt, ol, w) in arcs {
                            let mut no = outs.clone();
                            if ol != 0 {
                                no.push(ol);
                            }
                            let nc = cost + w;
                            let e = beam.entry((pos + 1, nxt)).or_insert((f32::INFINITY, Vec::new()));
                            if nc < e.0 {
                                *e = (nc, no);
                            }
                        }
                    }
                }
            }
        }
        let mut best: Option<(f32, Vec<u32>)> = None;
        for ((pos, st), (cost, outs)) in &beam {
            if *pos != input.len() {
                continue;
            }
            let fc = self.finals.get(st).copied().unwrap_or(f32::INFINITY);
            if !fc.is_finite() {
                continue;
            }
            let total = cost + fc;
            if best.as_ref().map(|(c, _)| total < *c).unwrap_or(true) {
                best = Some((total, outs.clone()));
            }
        }
        best.map(|(_, o)| o)
    }
}

/// OpenFST ConstFst (standard/tropical) binary + AT&T text.
fn parse_openfst_binary(data: &[u8]) -> Result<Fst> {
    if data.len() < 40 {
        bail!("fst too small");
    }
    // AT&T text format
    if data.starts_with(b"0\t") || data.starts_with(b"#") || data.iter().take(64).all(|&b| b < 128) {
        return parse_att_text(std::str::from_utf8(data)?);
    }
    parse_const_fst(data)
}

fn read_u32(data: &[u8], off: &mut usize) -> Result<u32> {
    if *off + 4 > data.len() {
        bail!("truncated u32");
    }
    let v = u32::from_le_bytes(data[*off..*off + 4].try_into()?);
    *off += 4;
    Ok(v)
}

fn read_i64(data: &[u8], off: &mut usize) -> Result<i64> {
    if *off + 8 > data.len() {
        bail!("truncated i64");
    }
    let v = i64::from_le_bytes(data[*off..*off + 8].try_into()?);
    *off += 8;
    Ok(v)
}

fn read_u64(data: &[u8], off: &mut usize) -> Result<u64> {
    if *off + 8 > data.len() {
        bail!("truncated u64");
    }
    let v = u64::from_le_bytes(data[*off..*off + 8].try_into()?);
    *off += 8;
    Ok(v)
}

fn read_f32(data: &[u8], off: &mut usize) -> Result<f32> {
    if *off + 4 > data.len() {
        bail!("truncated f32");
    }
    let v = f32::from_le_bytes(data[*off..*off + 4].try_into()?);
    *off += 4;
    Ok(v)
}

fn read_len_str(data: &[u8], off: &mut usize) -> Result<String> {
    let n = read_u32(data, off)? as usize;
    if *off + n > data.len() {
        bail!("truncated string");
    }
    let s = String::from_utf8_lossy(&data[*off..*off + n]).into_owned();
    *off += n;
    Ok(s)
}

fn align16(off: usize) -> usize {
    (off + 15) & !15
}

/// ConstFst mmap layout (flags & IS_ALIGNED): header → pad16 → ConstState[n] → pad16 → Arc[m].
/// ConstState = {f32 final, u32 pos, u32 narcs, u32 nieps, u32 noeps} (20 B).
/// Arc = {i32 ilabel, i32 olabel, f32 weight, i32 next} (16 B).
fn parse_const_fst(data: &[u8]) -> Result<Fst> {
    const MAGIC: u32 = 0x7EB2_FDD6;
    let mut off = 0usize;
    let magic = read_u32(data, &mut off)?;
    if magic != MAGIC {
        bail!("bad OpenFST magic {magic:#x}");
    }
    let fsttype = read_len_str(data, &mut off)?;
    let arctype = read_len_str(data, &mut off)?;
    let _version = read_u32(data, &mut off)?;
    let flags = read_u32(data, &mut off)?;
    let _props = read_u64(data, &mut off)?;
    let start = read_i64(data, &mut off)?;
    let nstates = read_u64(data, &mut off)? as usize;
    let narcs = read_u64(data, &mut off)? as usize;
    if fsttype != "const" {
        bail!("unsupported fsttype {fsttype}");
    }
    if arctype != "standard" {
        bail!("unsupported arctype {arctype}");
    }
    let aligned = (flags & 0x4) != 0;
    if aligned {
        off = align16(off);
    }
    if off + nstates * 20 + narcs * 16 > data.len() + 32 {
        bail!("const fst truncated: states={nstates} arcs={narcs} off={off} len={}", data.len());
    }
    let mut finals = HashMap::new();
    let mut state_meta = Vec::with_capacity(nstates);
    for s in 0..nstates {
        let fw = read_f32(data, &mut off)?;
        let pos = read_u32(data, &mut off)? as usize;
        let na = read_u32(data, &mut off)? as usize;
        let _nie = read_u32(data, &mut off)?;
        let _noe = read_u32(data, &mut off)?;
        if fw.is_finite() {
            finals.insert(s as u32, fw);
        }
        state_meta.push((pos, na));
    }
    if aligned {
        off = align16(off);
    }
    let mut raw_arcs = Vec::with_capacity(narcs);
    for _ in 0..narcs {
        let il = i32::from_le_bytes(data[off..off + 4].try_into()?) as u32;
        off += 4;
        let ol = i32::from_le_bytes(data[off..off + 4].try_into()?) as u32;
        off += 4;
        let w = read_f32(data, &mut off)?;
        let nxt = i32::from_le_bytes(data[off..off + 4].try_into()?) as u32;
        off += 4;
        raw_arcs.push((il, ol, w, nxt));
    }
    let mut arcs: HashMap<(u32, u32), Vec<(u32, u32, f32)>> = HashMap::new();
    for (s, &(pos, na)) in state_meta.iter().enumerate() {
        for a in &raw_arcs[pos..pos + na] {
            arcs.entry((s as u32, a.0)).or_default().push((a.3, a.1, a.2));
        }
    }
    Ok(Fst {
        start: start as u32,
        finals,
        arcs,
    })
}

fn parse_att_text(text: &str) -> Result<Fst> {
    let mut finals = HashMap::new();
    let mut arcs: HashMap<(u32, u32), Vec<(u32, u32, f32)>> = HashMap::new();
    let mut start = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.len() {
            1 | 2 => {
                let s: u32 = parts[0].parse()?;
                let w = parts.get(1).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                finals.insert(s, w);
                if start.is_none() {
                    start = Some(s);
                }
            }
            4 | 5 => {
                let s: u32 = parts[0].parse()?;
                let n: u32 = parts[1].parse()?;
                let il: u32 = parts[2].parse().unwrap_or_else(|_| parts[2].bytes().next().unwrap_or(0) as u32);
                let ol: u32 = parts[3].parse().unwrap_or_else(|_| parts[3].bytes().next().unwrap_or(0) as u32);
                let w = parts.get(4).and_then(|x| x.parse().ok()).unwrap_or(0.0);
                arcs.entry((s, il)).or_default().push((n, ol, w));
                if start.is_none() {
                    start = Some(s);
                }
            }
            _ => {}
        }
    }
    Ok(Fst {
        start: start.unwrap_or(0),
        finals,
        arcs,
    })
}

/// Locale Hammer chain: `us2common → common2*`.
pub struct Hammer {
    pub fsts: Vec<Fst>,
}

impl Hammer {
    /// Load FSTs named for `locale` via `fetch(stem)` → raw OpenFST bytes.
    ///
    /// Stems omit the `.fst` suffix (`us2common`, `common2ca`, …).
    pub fn load_from_blobs<F>(mut fetch: F, locale: &str) -> Result<Self>
    where
        F: FnMut(&str) -> Option<Vec<u8>>,
    {
        let mut stems = vec!["us2common".to_string()];
        match locale {
            "en_GB" | "en-GB" => {
                stems.push("common2au".into());
                stems.push("au2uk".into());
            }
            "en_CA" | "en-CA" => stems.push("common2ca".into()),
            "en_AU" | "en-AU" => stems.push("common2au".into()),
            _ => {}
        }
        let mut fsts = Vec::new();
        for stem in stems {
            if let Some(data) = fetch(&stem) {
                match Fst::from_bytes(&data) {
                    Ok(f) => fsts.push(f),
                    Err(e) => {
                        if std::env::var_os("RLX_ASR_FST_DEBUG").is_some() {
                            eprintln!("[rlx-asr] skip FST {stem}: {e:#}");
                        }
                    }
                }
            }
        }
        Ok(Self { fsts })
    }

    pub fn load_dir(tp: &Path, locale: &str) -> Result<Self> {
        Self::load_from_blobs(
            |stem| {
                let p = tp.join(format!("{stem}.fst"));
                fs::read(&p).ok()
            },
            locale,
        )
    }

    pub fn apply(&self, text: &str) -> String {
        let mut s = text.to_string();
        for f in &self.fsts {
            s = f.apply_bytes(&s);
        }
        s
    }
}

pub struct Etiquette {
    map: HashMap<String, String>,
}

impl Etiquette {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(raw)?;
        let mut map = HashMap::new();
        if let Some(obj) = v.as_object() {
            for (k, val) in obj {
                if let Some(s) = val.as_str() {
                    map.insert(k.to_lowercase(), s.to_string());
                } else if let Some(s) = val.get("replacement").and_then(|x| x.as_str()) {
                    map.insert(k.to_lowercase(), s.to_string());
                }
            }
        }
        Ok(Self { map })
    }

    pub fn apply(&self, text: &str) -> String {
        text.split_whitespace()
            .map(|w| {
                self.map
                    .get(&w.to_lowercase())
                    .cloned()
                    .unwrap_or_else(|| w.to_string())
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn att_identity() {
        let text = "0 0 97 97 0\n0\n";
        let f = parse_att_text(text).unwrap();
        assert_eq!(f.apply_bytes("a"), "a");
    }

    #[test]
    fn const_fst_loads_from_gguf() {
        let Some(path) = crate::AsrPaths::resolve().pack() else {
            eprintln!("skip: no model.rlxp / model.gguf");
            return;
        };
        let g = crate::gguf_io::AsrPack::open(&path).expect("open weight pack");
        let data = g.blob("tp.common2ca").expect("tp.common2ca");
        let f = Fst::from_bytes(&data).expect("parse fst");
        assert_eq!(f.start, 86);
        assert!(f.finals.contains_key(&86));
        let out = f.apply_bytes("hello");
        assert!(!out.is_empty(), "empty apply");
        let hammer = g.load_hammer("en_US").expect("hammer");
        assert!(!hammer.fsts.is_empty(), "expected us2common from GGUF");
    }
}
