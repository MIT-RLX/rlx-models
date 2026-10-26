//! A BERT-family sentence encoder end to end — text in, unit vector out — for
//! one text at a time. What a search box needs; built from in-memory files, as
//! a browser has them.
//!
//! A query of a few words runs at a few tokens: the graph is compiled per
//! length bucket (8, 16, 32, … up to the budget) and a text runs in the
//! smallest that holds it — a forward pass costs about its length, so padding
//! every query to the budget would cost 2–4× for nothing. Padding is masked,
//! so the bucket does not change the vector. Buckets compile when first
//! needed, or all at once with [`BertTextEncoder::warm_buckets`].

use anyhow::Result;
use rlx_runtime::Device;

use crate::{BertTokenizer, Pooling, RlxBertModel, pool_embeddings};

pub struct BertTextEncoder {
    /// one per bucket, `None` until compiled; `models[0]` always is
    models: Vec<Option<RlxBertModel>>,
    buckets: Vec<usize>,
    tok: BertTokenizer,
    pooling: Pooling,
    seq: usize,
}

/// 8, 16, 32, … below `seq`, then `seq`.
fn ladder(seq: usize) -> Vec<usize> {
    let mut b: Vec<usize> = std::iter::successors(Some(8usize), |x| Some(x * 2))
        .take_while(|&x| x < seq)
        .collect();
    b.push(seq);
    b
}

/// A checkpoint's files, as bytes. The tokenizer configs may be `b"{}"` when
/// the checkpoint has none.
pub struct BertFiles<'a> {
    pub config: &'a [u8],
    /// f32, or int8-packed ([`crate::packed`]).
    pub weights: Vec<u8>,
    pub tokenizer: &'a [u8],
    pub tokenizer_config: &'a [u8],
    pub special_tokens_map: &'a [u8],
}

impl BertTextEncoder {
    /// `seq`: the token budget per text; longer texts are cut, keeping the
    /// closing `[SEP]`.
    pub fn from_bytes(
        files: BertFiles,
        pooling: Pooling,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        Self::from_bytes_with(files, pooling, seq, device, true)
    }

    /// [`Self::from_bytes`], choosing for an int8-packed checkpoint whether
    /// its projections run as int8 matmuls or widened to f32
    /// ([`RlxBertModel::set_int8_matmuls`]).
    pub fn from_bytes_with(
        files: BertFiles,
        pooling: Pooling,
        seq: usize,
        device: Device,
        int8_matmuls: bool,
    ) -> Result<Self> {
        let tok = BertTokenizer::from_bytes(
            files.tokenizer,
            files.config,
            files.special_tokens_map,
            files.tokenizer_config,
            seq.max(2),
        )?;
        let buckets = ladder(seq);
        let mut first =
            RlxBertModel::from_bytes(files.config, files.weights, 1, buckets[0], device)?;
        first.set_int8_matmuls(int8_matmuls)?;
        let mut models: Vec<Option<RlxBertModel>> = buckets.iter().map(|_| None).collect();
        models[0] = Some(first);
        Ok(Self {
            models,
            buckets,
            tok,
            pooling,
            seq,
        })
    }

    /// Embedding width.
    pub fn dim(&self) -> usize {
        self.models[0].as_ref().map_or(0, |m| m.hidden_size())
    }

    /// The token lengths a graph is compiled for.
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// Compile every bucket now (e.g. while the page is idle), so no query
    /// waits for one.
    pub fn warm_buckets(&mut self) -> Result<()> {
        for i in 0..self.buckets.len() {
            self.model(i)?;
        }
        Ok(())
    }

    /// Bucket `i`'s model, compiled on first use.
    fn model(&mut self, i: usize) -> Result<&mut RlxBertModel> {
        if self.models[i].is_none() {
            let m = self.models[0]
                .as_ref()
                .expect("bucket 0 is built")
                .with_shape(1, self.buckets[i])?;
            self.models[i] = Some(m);
        }
        Ok(self.models[i].as_mut().unwrap())
    }

    /// Unit-length embedding of `text`. Prefix any query instruction yourself.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let (b, [ids, mask, tt, pos], m) = self.inputs(text)?;
        let hidden = self.model(b)?.forward(&ids, &mask, &tt, &pos);
        Ok(self.pool(&hidden, &m))
    }

    /// [`Self::embed`] for WebGPU in a browser, where reading back cannot block.
    #[cfg(target_arch = "wasm32")]
    pub async fn embed_async(&mut self, text: &str) -> Result<Vec<f32>> {
        let (b, [ids, mask, tt, pos], m) = self.inputs(text)?;
        let hidden = self.model(b)?.forward_async(&ids, &mask, &tt, &pos).await?;
        Ok(self.pool(&hidden, &m))
    }

    /// The bucket for `text`, the graph's four inputs cut or padded to its
    /// length, and the attention mask.
    fn inputs(&self, text: &str) -> Result<(usize, [Vec<f32>; 4], Vec<u32>)> {
        let enc = self.tok.encode_batch(&[text])?;
        let row = &enc.input_ids[0];
        let n = row.len().min(self.seq);
        let b = self
            .buckets
            .iter()
            .position(|&s| s >= n)
            .unwrap_or(self.buckets.len() - 1);
        let seq = self.buckets[b];
        let mut x = [
            vec![0f32; seq],
            vec![0f32; seq],
            vec![0f32; seq],
            vec![0f32; seq],
        ];
        let mut mask = vec![0u32; seq];
        for s in 0..n {
            x[0][s] = (if n < row.len() && s == n - 1 {
                row[row.len() - 1]
            } else {
                row[s]
            }) as f32;
            mask[s] = enc.attention_mask[0][s];
            x[1][s] = mask[s] as f32;
            x[2][s] = enc.token_type_ids[0][s] as f32;
        }
        let off = self.models[0].as_ref().map_or(0, |m| m.position_offset());
        for (s, p) in x[3].iter_mut().enumerate() {
            *p = (s + off) as f32;
        }
        Ok((b, x, mask))
    }

    fn pool(&self, hidden: &[f32], mask: &[u32]) -> Vec<f32> {
        pool_embeddings(hidden, &[mask], 1, mask.len(), self.dim(), self.pooling).remove(0)
    }
}
