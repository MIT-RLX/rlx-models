//! A minimal self-describing checkpoint: magic + architecture header + every
//! parameter as `[name, f32 data]`. `rlx-tensor` ships no serializer, so we
//! roll a tiny one — enough to save trained weights and reload them for
//! generation.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::bpe::Bpe;
use crate::config::GptConfig;

// V2 appends a trailing tokenizer section (`[u32 len][bytes]`, len 0 = byte
// level) after the params. V1 files (no section) still load as byte-level.
const MAGIC_V1: &[u8; 8] = b"RLXTS1\0\0";
const MAGIC: &[u8; 8] = b"RLXTS2\0\0";

/// Write the config header, every `(name, data)` parameter, and the tokenizer
/// (BPE merges, or empty for byte-level) to `path`.
pub fn save(
    path: &Path,
    cfg: &GptConfig,
    params: &[(String, Vec<f32>)],
    bpe: Option<&Bpe>,
) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("create dir {}", dir.display()))?;
    }
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    for v in [
        cfg.vocab,
        cfg.block_size,
        cfg.n_layer,
        cfg.n_head,
        cfg.n_embd,
    ] {
        w.write_all(&(v as u32).to_le_bytes())?;
    }
    w.write_all(&(params.len() as u32).to_le_bytes())?;
    for (name, data) in params {
        w.write_all(&(name.len() as u32).to_le_bytes())?;
        w.write_all(name.as_bytes())?;
        w.write_all(&(data.len() as u32).to_le_bytes())?;
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        w.write_all(&bytes)?;
    }
    // Tokenizer section: BPE merge table, or length 0 for byte-level.
    let tok = bpe.map(Bpe::to_bytes).unwrap_or_default();
    w.write_all(&(tok.len() as u32).to_le_bytes())?;
    w.write_all(&tok)?;
    w.flush()?;
    Ok(())
}

/// Read a checkpoint back. The returned config has `batch = 1` (a generation
/// default) and `label_smoothing = 0.0` — override as needed. The `Option<Bpe>`
/// is the embedded tokenizer (`None` = byte-level, incl. all V1 checkpoints).
pub fn load(path: &Path) -> Result<(GptConfig, Vec<(String, Vec<f32>)>, Option<Bpe>)> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let v2 = &magic == MAGIC;
    if !v2 && &magic != MAGIC_V1 {
        bail!("{}: not an rlx-tinystories checkpoint", path.display());
    }
    let vocab = read_u32(&mut r)? as usize;
    let block_size = read_u32(&mut r)? as usize;
    let n_layer = read_u32(&mut r)? as usize;
    let n_head = read_u32(&mut r)? as usize;
    let n_embd = read_u32(&mut r)? as usize;
    let cfg = GptConfig {
        vocab,
        block_size,
        n_layer,
        n_head,
        n_embd,
        batch: 1,
        label_smoothing: 0.0,
    };

    let n_params = read_u32(&mut r)? as usize;
    let mut params = Vec::with_capacity(n_params);
    for _ in 0..n_params {
        let name_len = read_u32(&mut r)? as usize;
        let mut name_buf = vec![0u8; name_len];
        r.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf).context("param name utf8")?;
        let data_len = read_u32(&mut r)? as usize;
        let mut data_buf = vec![0u8; data_len * 4];
        r.read_exact(&mut data_buf)?;
        let data: Vec<f32> = data_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        params.push((name, data));
    }

    // V2 tokenizer section (absent in V1 → byte-level).
    let bpe = if v2 {
        let tok_len = read_u32(&mut r)? as usize;
        if tok_len == 0 {
            None
        } else {
            let mut buf = vec![0u8; tok_len];
            r.read_exact(&mut buf)?;
            Some(Bpe::from_bytes(&buf))
        }
    } else {
        None
    };
    Ok((cfg, params, bpe))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
