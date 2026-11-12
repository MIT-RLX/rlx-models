// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Distillation records: one `K`-chunk segment each, `fbank
//! [segment_frames(K), 80]` then the teacher's encoder output `[K·8, 512]`,
//! f32 little-endian, back to back in `segments.f32`; `meta.json` holds `K`
//! and the count. Read through `mmap`, so a multi-GB set never sits in RAM.
//!
//! An optional `contexts.u16` beside it holds, per record and output frame,
//! the teacher's predictor context (two token ids, [`NO_TOKEN`] for none)
//! from greedy decoding of the target ([`crate::joiner::Joiner::contexts`]);
//! with it the student trains on the teacher's token distributions.
//!
//! [`Records::open_all`] reads several such directories (same `K`) as one
//! set, in the order given.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::config::{N_MELS, OUT_FRAMES, StudentConfig};
use crate::joiner::NO_TOKEN;

/// f32 values of one record's features.
pub fn feat_len(k: usize) -> usize {
    StudentConfig::segment_frames(k) * N_MELS
}

/// f32 values of one record's target.
pub fn target_len(k: usize) -> usize {
    k * OUT_FRAMES * 512
}

/// Appends records.
pub struct Writer {
    file: std::io::BufWriter<std::fs::File>,
    k: usize,
    n: usize,
}

impl Writer {
    /// Start `dir/segments.f32`.
    pub fn create(dir: &Path, k: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let file = std::fs::File::create(dir.join("segments.f32"))?;
        Ok(Self {
            file: std::io::BufWriter::with_capacity(1 << 24, file),
            k,
            n: 0,
        })
    }

    /// One record.
    pub fn push(&mut self, feats: &[f32], target: &[f32]) -> Result<()> {
        ensure!(feats.len() == feat_len(self.k) && target.len() == target_len(self.k));
        for v in feats.iter().chain(target) {
            self.file.write_all(&v.to_le_bytes())?;
        }
        self.n += 1;
        Ok(())
    }

    /// Flush and write `meta.json`; returns the record count.
    pub fn finish(mut self, dir: &Path, extra: serde_json::Value) -> Result<usize> {
        self.file.flush()?;
        let mut meta = serde_json::json!({"chunks": self.k, "records": self.n});
        if let (Some(m), Some(e)) = (meta.as_object_mut(), extra.as_object()) {
            m.extend(e.clone());
        }
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
        Ok(self.n)
    }
}

/// Output frames of one record (= its predictor contexts).
pub fn frames_len(k: usize) -> usize {
    k * OUT_FRAMES
}

/// Write `dir/contexts.u16` (one entry per output frame of every record).
pub fn write_contexts(dir: &Path, contexts: &[[u16; 2]]) -> Result<()> {
    let bytes: Vec<u8> = contexts
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(dir.join("contexts.u16"), bytes)?;
    Ok(())
}

/// One directory of records, memory-mapped.
struct Part {
    map: memmap2::Mmap,
    contexts: Option<memmap2::Mmap>,
    /// Index of its first record in the whole set.
    start: usize,
}

/// A record set, memory-mapped, from one or more directories.
pub struct Records {
    parts: Vec<Part>,
    /// Chunks per segment.
    pub k: usize,
    /// Records.
    pub n: usize,
}

impl Records {
    /// Open `dir` (`segments.f32` + `meta.json`).
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_all(&[dir])
    }

    /// Open several directories as one set, records numbered in the order
    /// given; all must share `K`. Contexts count only if every one has them.
    pub fn open_all<P: AsRef<Path>>(dirs: &[P]) -> Result<Self> {
        ensure!(!dirs.is_empty(), "no record directories");
        let mut parts = Vec::with_capacity(dirs.len());
        let (mut k, mut n) = (0, 0);
        for dir in dirs {
            let (part, pk, pn) = Self::open_part(dir.as_ref(), n)?;
            ensure!(
                k == 0 || pk == k,
                "{}: {pk} chunks per record, others have {k}",
                dir.as_ref().display()
            );
            k = pk;
            n += pn;
            parts.push(part);
        }
        Ok(Self { parts, k, n })
    }

    fn open_part(dir: &Path, start: usize) -> Result<(Part, usize, usize)> {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
        let k = meta["chunks"].as_u64().context("meta chunks")? as usize;
        let n = meta["records"].as_u64().context("meta records")? as usize;
        let file = std::fs::File::open(dir.join("segments.f32"))?;
        // SAFETY: read-only map of a file this process does not modify.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        ensure!(
            map.len() == n * (feat_len(k) + target_len(k)) * 4,
            "segments.f32 size does not match meta.json"
        );
        let path = dir.join("contexts.u16");
        let contexts = if path.exists() {
            let file = std::fs::File::open(&path)?;
            // SAFETY: as above.
            let map = unsafe { memmap2::Mmap::map(&file)? };
            ensure!(
                map.len() == n * frames_len(k) * 4,
                "contexts.u16 size does not match meta.json"
            );
            Some(map)
        } else {
            None
        };
        Ok((
            Part {
                map,
                contexts,
                start,
            },
            k,
            n,
        ))
    }

    /// The part holding record `i`, and `i` within it.
    fn locate(&self, i: usize) -> (&Part, usize) {
        let p = self.parts.partition_point(|p| p.start <= i) - 1;
        let part = &self.parts[p];
        (part, i - part.start)
    }

    /// Whether every directory has `contexts.u16`.
    pub fn has_contexts(&self) -> bool {
        self.parts.iter().all(|p| p.contexts.is_some())
    }

    /// Record `i`'s predictor context per output frame
    /// (`[NO_TOKEN, NO_TOKEN]` throughout without `contexts.u16`).
    pub fn contexts(&self, i: usize) -> Vec<[u16; 2]> {
        let f = frames_len(self.k);
        let (part, i) = self.locate(i);
        match &part.contexts {
            Some(m) if self.has_contexts() => m[i * f * 4..(i + 1) * f * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| {
                    [
                        u16::from_le_bytes([b[0], b[1]]),
                        u16::from_le_bytes([b[2], b[3]]),
                    ]
                })
                .collect(),
            _ => vec![[NO_TOKEN; 2]; f],
        }
    }

    /// Append record `i`'s features and target to the batch buffers.
    pub fn read_into(&self, i: usize, feats: &mut Vec<f32>, target: &mut Vec<f32>) {
        let (fl, tl) = (feat_len(self.k), target_len(self.k));
        let (part, i) = self.locate(i);
        let base = i * (fl + tl) * 4;
        let rec = &part.map[base..base + (fl + tl) * 4];
        let mut it = rec
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b));
        feats.extend(it.by_ref().take(fl));
        target.extend(it.take(tl));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(seed: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
        let f = (0..feat_len(k)).map(|j| (seed * 7 + j) as f32).collect();
        let t = (0..target_len(k)).map(|j| (seed * 11 + j) as f32).collect();
        (f, t)
    }

    #[test]
    fn records_span_directories() -> Result<()> {
        let k = 1;
        let root = std::env::temp_dir().join(format!("rzd-records-{}", std::process::id()));
        let dirs = [root.join("a"), root.join("b")];
        let mut seed = 0;
        for (dir, count) in dirs.iter().zip([3, 2]) {
            let mut w = Writer::create(dir, k)?;
            let mut ctx = Vec::new();
            for _ in 0..count {
                let (f, t) = record(seed, k);
                w.push(&f, &t)?;
                ctx.extend((0..frames_len(k)).map(|j| [seed as u16, j as u16]));
                seed += 1;
            }
            w.finish(dir, serde_json::json!({}))?;
            write_contexts(dir, &ctx)?;
        }
        let all = Records::open_all(&dirs)?;
        assert_eq!((all.n, all.k), (5, k));
        assert!(all.has_contexts());
        for i in 0..all.n {
            let (mut f, mut t) = (Vec::new(), Vec::new());
            all.read_into(i, &mut f, &mut t);
            assert_eq!((f, t), record(i, k), "record {i}");
            assert_eq!(all.contexts(i)[1], [i as u16, 1]);
        }
        std::fs::remove_file(dirs[1].join("contexts.u16"))?;
        assert!(!Records::open_all(&dirs)?.has_contexts());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }
}
