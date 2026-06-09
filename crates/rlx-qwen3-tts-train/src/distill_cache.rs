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

//! Precomputed (inputs, teacher targets) — GPU teacher, optional disk cache.

use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use rlx_qwen3_tts::config::{Qwen3TtsConfig, TalkerConfig};
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::prompt::{build_assistant_text, load_text_tokenizer};
use rlx_qwen3_tts::text_embed::TextEmbedder;
use rlx_runtime::Device;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::codec_table::CodecEmbeddingTable;
use crate::dataset::CodesDataset;
use crate::teacher::TalkerTeacher;

const CACHE_MAGIC: &[u8; 8] = b"RLXQ3JFK";

#[derive(Clone)]
pub struct DistillBatch {
    pub seq: usize,
    pub inputs: Vec<f32>,
    pub targets: Vec<f32>,
}

pub struct DistillCache {
    batches: Vec<DistillBatch>,
}

impl DistillCache {
    pub fn open_or_build(
        cfg: &Qwen3TtsConfig,
        talker: &TalkerConfig,
        store: &Qwen3TtsWeightStore,
        data: &CodesDataset,
        train_device: Device,
        max_seq: usize,
        max_clips: usize,
        cache_path: Option<&Path>,
        verbose: bool,
    ) -> Result<Self> {
        let hidden = talker.hidden_size;
        if let Some(path) = cache_path {
            if path.is_file() {
                match Self::load(path, max_seq, hidden) {
                    Ok(c) => {
                        if verbose {
                            eprintln!("[jfk-lora] loaded distill cache {}", path.display());
                        }
                        return Ok(c);
                    }
                    Err(e) => {
                        eprintln!("[jfk-lora] cache read failed ({e}), rebuilding");
                    }
                }
            }
        }
        let cache = Self::build(
            cfg,
            talker,
            store,
            data,
            train_device,
            max_seq,
            max_clips,
            verbose,
        )?;
        if let Some(path) = cache_path {
            if let Err(e) = cache.save(path, max_seq, hidden) {
                eprintln!("[jfk-lora] cache write failed: {e}");
            } else if verbose {
                eprintln!("[jfk-lora] saved distill cache {}", path.display());
            }
        }
        Ok(cache)
    }

    pub fn build(
        cfg: &Qwen3TtsConfig,
        talker: &TalkerConfig,
        store: &Qwen3TtsWeightStore,
        data: &CodesDataset,
        train_device: Device,
        max_seq: usize,
        max_clips: usize,
        verbose: bool,
    ) -> Result<Self> {
        let hidden = talker.hidden_size;
        let buf_len = max_seq * hidden;
        let n = data.len().min(max_clips.max(1));
        let tokenizer = load_text_tokenizer(store.model_dir())?;
        let text_embedder = TextEmbedder::open(store)?;
        let codec = CodecEmbeddingTable::open(store)?;
        let mut teacher = TalkerTeacher::open(store, talker, train_device)?;

        let mut batches = Vec::with_capacity(n);
        let started = Instant::now();
        let mut embed_scratch = vec![0f32; hidden];
        for (idx, rec) in data.records.iter().take(n).enumerate() {
            let embeds = build_train_embeds(
                cfg,
                talker,
                &codec,
                &text_embedder,
                &tokenizer,
                &rec.text,
                &rec.audio_codes,
                max_seq,
                &mut embed_scratch,
            )?;
            let seq = embeds.nrows().min(max_seq);
            let flat_teacher = teacher.prefill_hidden(embeds.view())?;
            let mut inputs = vec![0f32; buf_len];
            let mut targets = vec![0f32; buf_len];
            let copy = seq * hidden;
            inputs[..copy].copy_from_slice(&embeds.as_slice().unwrap()[..copy]);
            targets[..copy].copy_from_slice(&flat_teacher[..copy]);
            batches.push(DistillBatch {
                seq,
                inputs,
                targets,
            });
            if verbose && (idx + 1) % 10 == 0 {
                eprintln!("[jfk-lora] precompute {}/{n}", idx + 1);
            }
        }
        if verbose {
            eprintln!(
                "[jfk-lora] precomputed {n} batches in {:.1}s",
                started.elapsed().as_secs_f64()
            );
        }
        Ok(Self { batches })
    }

    fn load(path: &Path, max_seq: usize, hidden: usize) -> Result<Self> {
        let buf_len = max_seq * hidden;
        let mut f = BufReader::new(File::open(path).context("open cache")?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        ensure!(magic == *CACHE_MAGIC, "bad distill cache magic");
        let n = read_u32(&mut f)? as usize;
        let file_max_seq = read_u32(&mut f)? as usize;
        let file_hidden = read_u32(&mut f)? as usize;
        ensure!(
            file_max_seq == max_seq && file_hidden == hidden,
            "cache shape mismatch (file {file_max_seq}x{file_hidden}, want {max_seq}x{hidden})"
        );
        let mut batches = Vec::with_capacity(n);
        for _ in 0..n {
            let seq = read_u32(&mut f)? as usize;
            ensure!(seq <= max_seq, "cache seq {seq} > max_seq {max_seq}");
            let mut inputs = vec![0f32; buf_len];
            let mut targets = vec![0f32; buf_len];
            let nfloats = seq * hidden;
            read_f32_slice(&mut f, &mut inputs[..nfloats])?;
            read_f32_slice(&mut f, &mut targets[..nfloats])?;
            batches.push(DistillBatch {
                seq,
                inputs,
                targets,
            });
        }
        Ok(Self { batches })
    }

    pub fn save(&self, path: &Path, max_seq: usize, hidden: usize) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = BufWriter::new(File::create(path).context("create cache")?);
        f.write_all(CACHE_MAGIC)?;
        write_u32(&mut f, self.batches.len() as u32)?;
        write_u32(&mut f, max_seq as u32)?;
        write_u32(&mut f, hidden as u32)?;
        for b in &self.batches {
            write_u32(&mut f, b.seq as u32)?;
            let nfloats = b.seq * hidden;
            write_f32_slice(&mut f, &b.inputs[..nfloats])?;
            write_f32_slice(&mut f, &b.targets[..nfloats])?;
        }
        f.flush()?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub fn get(&self, step: usize) -> Result<&DistillBatch> {
        ensure!(!self.batches.is_empty(), "empty distill cache");
        Ok(&self.batches[step % self.batches.len()])
    }
}

pub fn default_cache_path(
    jsonl: &Path,
    max_seq: usize,
    max_clips: usize,
    device: Device,
) -> PathBuf {
    let tag = format!("{device:?}").to_lowercase();
    jsonl
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("distill_cache_{tag}_s{max_seq}_n{max_clips}.bin"))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_u32(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_f32_slice(r: &mut impl Read, out: &mut [f32]) -> Result<()> {
    let mut bytes = vec![0u8; out.len() * 4];
    r.read_exact(&mut bytes)?;
    for (o, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *o = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

fn write_f32_slice(w: &mut impl Write, data: &[f32]) -> Result<()> {
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn build_train_embeds(
    cfg: &Qwen3TtsConfig,
    talker: &TalkerConfig,
    codec: &CodecEmbeddingTable,
    text_embedder: &TextEmbedder,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    audio_codes: &[Vec<u32>],
    max_seq: usize,
    scratch: &mut [f32],
) -> Result<Array2<f32>> {
    let assistant = build_assistant_text(text);
    let encoding = tokenizer
        .encode(assistant.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.len() < 8 {
        ids.resize(8, cfg.tts_pad_token_id);
    }
    let text_rows = text_embedder.embed_project_ids(&ids)?;
    let hidden = talker.hidden_size;

    let g0: Vec<u32> = audio_codes
        .iter()
        .map(|row| row.first().copied().unwrap_or(0))
        .collect();

    let seq_len = (3 + 5 + g0.len() + 1).min(max_seq);
    let mut out = Array2::<f32>::zeros((seq_len, hidden));
    let mut ti = 0usize;

    for row in text_rows.iter().take(3.min(seq_len)) {
        if ti >= seq_len {
            break;
        }
        for (j, &v) in row.iter().enumerate().take(hidden) {
            out[[ti, j]] = v;
        }
        ti += 1;
    }

    let codec_ids = [
        talker.codec_nothink_id,
        talker.codec_think_bos_id,
        talker.codec_think_eos_id,
        0u32,
        talker.codec_pad_id,
    ];
    for &cid in &codec_ids {
        if ti >= seq_len {
            break;
        }
        scratch.fill(0.0);
        if cid != 0 {
            codec.copy_row(cid, scratch);
        }
        for (j, &v) in scratch.iter().enumerate().take(hidden) {
            out[[ti, j]] = v;
        }
        ti += 1;
    }

    for &c in &g0 {
        if ti >= seq_len.saturating_sub(1) {
            break;
        }
        codec.copy_row(c, scratch);
        for (j, &v) in scratch.iter().enumerate().take(hidden) {
            out[[ti, j]] = v;
        }
        ti += 1;
    }
    if ti < seq_len {
        codec.copy_row(talker.codec_eos_token_id, scratch);
        for (j, &v) in scratch.iter().enumerate().take(hidden) {
            out[[ti, j]] = v;
        }
    }
    Ok(out)
}
