//! TinyStories data: an `mmap`-ed byte corpus (so a multi-GB split never lives
//! in RAM) plus a one-hot minibatch sampler. Byte-level tokens are the raw
//! file bytes, so a training window is just a slice of the mapping.

use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::config::GptConfig;
use crate::rng::Rng;

/// A memory-mapped byte corpus.
pub struct Corpus {
    _file: std::fs::File,
    mmap: Mmap,
}

impl Corpus {
    /// Memory-map a local UTF-8 text file as the token stream.
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("open corpus {}", path.display()))?;
        // SAFETY: we only read the mapping; the file is held open for its life.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap corpus {}", path.display()))?;
        if mmap.len() < 4 {
            bail!("corpus {} is too small", path.display());
        }
        Ok(Self { _file: file, mmap })
    }

    /// The raw corpus bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Corpus length in bytes (== number of byte-level tokens).
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

/// Fetch a TinyStories split from the Hugging Face Hub (dataset
/// `roneneldan/TinyStories`) into the local cache and return its path. `split`
/// is `"train"` (~2 GB) or `"valid"` (~20 MB).
#[cfg(feature = "download")]
pub fn download(split: &str) -> Result<std::path::PathBuf> {
    let file = match split {
        "train" => "TinyStoriesV2-GPT4-train.txt",
        "valid" | "validation" | "val" => "TinyStoriesV2-GPT4-valid.txt",
        other => bail!("unknown split {other:?} (expected train|valid)"),
    };
    let api = hf_hub::api::sync::ApiBuilder::new()
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.dataset("roneneldan/TinyStories".to_string());
    let path = repo
        .get(file)
        .with_context(|| format!("download {file} from roneneldan/TinyStories"))?;
    Ok(path)
}

/// A token stream the batcher samples windows from: either raw corpus bytes
/// (byte-level tokenizer, id == byte, sliced straight from the mmap) or a
/// precomputed id array (BPE). One code path feeds both.
#[derive(Clone, Copy)]
pub enum Tokens<'a> {
    Bytes(&'a [u8]),
    Ids(&'a [u32]),
}

impl<'a> Tokens<'a> {
    pub fn len(&self) -> usize {
        match self {
            Tokens::Bytes(b) => b.len(),
            Tokens::Ids(i) => i.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Token id at position `i`, as the f32 the model consumes (then casts→i64).
    #[inline]
    fn at(&self, i: usize) -> f32 {
        match self {
            Tokens::Bytes(b) => f32::from(b[i]),
            Tokens::Ids(ids) => ids[i] as f32,
        }
    }

    /// Sub-range view (for the train/val split).
    pub fn range(&self, r: std::ops::Range<usize>) -> Tokens<'a> {
        match self {
            Tokens::Bytes(b) => Tokens::Bytes(&b[r]),
            Tokens::Ids(i) => Tokens::Ids(&i[r]),
        }
    }
}

/// Samples token-id minibatches. Emits `[B*T]` integer ids (as f32; the model
/// casts to i64 and gathers) rather than `[B*T, V]` one-hot — the ~V× smaller
/// per-step payload is the point, since training is I/O/dispatch-bound. Label
/// smoothing now lives in the model's on-device target table, so the batcher
/// carries no vocab/smoothing state.
pub struct Batcher {
    block: usize,
    batch: usize,
}

impl Batcher {
    pub fn new(cfg: &GptConfig) -> Self {
        Self {
            block: cfg.block_size,
            batch: cfg.batch,
        }
    }

    /// Sample a random minibatch from `region` (a slice of the token stream): a
    /// batch of `T+1`-token windows, returned as input ids and next-token target
    /// ids, each `[B*T]` (values in `0..vocab`, as f32).
    pub fn sample(&self, region: &Tokens, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
        let (b, t) = (self.batch, self.block);
        let span = t + 1;
        assert!(
            region.len() > span,
            "corpus region shorter than block_size + 1"
        );
        let mut tok = vec![0f32; b * t];
        let mut tgt = vec![0f32; b * t];
        for bi in 0..b {
            let start = rng.below(region.len() - span);
            for p in 0..t {
                tok[bi * t + p] = region.at(start + p);
                tgt[bi * t + p] = region.at(start + p + 1);
            }
        }
        (tok, tgt)
    }
}
