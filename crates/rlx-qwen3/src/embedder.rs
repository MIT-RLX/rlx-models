//! Pluggable block **content embedders** for the disk-tiered KV context store's
//! semantic retrieval index (the dual-encoder path).
//!
//! The generation model's K vectors are a projection tuned for next-token
//! attention, not content matching, so K·K retrieval is not selective (see
//! `docs/dual-encoder-retrieval-plan.md`). A [`BlockEmbedder`] instead maps a
//! block's token span → a retrieval-grade vector whose similarity puts a question
//! next to its answer — the signal needed for selective (small-top-k) recall at
//! 1M-token scale, and one that HNSW navigates with near-exact recall.
//!
//! Two implementations, both selectable at runtime:
//! - [`TokenMeanEmbedder`] — self-contained (no download, no graph change): the
//!   L2-normalized mean of the model's own static token-embedding rows over the
//!   span. Position-invariant and content-based (a bag-of-static-word-vectors
//!   embedding), so far more selective than K·K while costing a table gather.
//! - the dedicated-encoder impl (rlx-embed MiniLM/BGE) lives behind the
//!   `dual-encoder` feature for maximum selectivity.

use std::sync::Arc;

/// Which block embedder the KV store uses for semantic retrieval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EmbedderKind {
    /// No semantic index (K·K / lexical only).
    #[default]
    None,
    /// Self-contained mean-of-static-token-embeddings ([`TokenMeanEmbedder`]).
    TokenMean,
    /// Dedicated retrieval encoder (rlx-embed MiniLM/BGE) — needs the
    /// `dual-encoder` feature; falls back to `TokenMean` when unavailable.
    Encoder,
}

/// Maps a block's token-id span to a fixed-dim retrieval embedding. Document and
/// query embeddings are separate methods so asymmetric encoders (task-prefixed,
/// e.g. nomic/E5) fit; symmetric embedders return the same for both.
pub trait BlockEmbedder: Send + Sync {
    /// Embedding dimensionality (the store's semantic-index dim).
    fn dim(&self) -> usize;
    /// Embed an offloaded block's token span (the "document" side).
    fn embed_document(&self, token_ids: &[u32]) -> Vec<f32>;
    /// Embed the current query window's tokens (the "query" side).
    fn embed_query(&self, token_ids: &[u32]) -> Vec<f32>;
    /// Embed a raw query STRING (the actual question), when the caller has it —
    /// avoids the noisy decode-position token window. `None` if this embedder can't
    /// embed text directly (e.g. the token-table embedder has no detokenizer).
    fn embed_query_str(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

/// Self-contained embedder: the L2-normalized mean of the model's static token
/// embeddings (`model.embed_tokens.weight`, shape `[vocab, hidden]`) over the
/// span. No second model, no download, no graph change — just a table gather.
/// Static word vectors are content-based and position-invariant, so a question
/// and its answer share the informative content words and land close, unlike the
/// position-entangled K space. Symmetric (query == document encoding).
pub struct TokenMeanEmbedder {
    /// Row-major `[vocab, hidden]` embedding table (shared with the weight cache).
    table: Arc<Vec<f32>>,
    vocab: usize,
    hidden: usize,
    /// Optional per-dim inverse-document-frequency-ish down-weight of the most
    /// frequent dims is NOT applied here; kept intentionally simple.
    _reserved: (),
}

impl TokenMeanEmbedder {
    /// Build from the raw `[vocab, hidden]` embedding table. Returns `None` if the
    /// table length doesn't match `vocab * hidden`.
    pub fn new(table: Arc<Vec<f32>>, vocab: usize, hidden: usize) -> Option<Self> {
        if hidden == 0 || table.len() < vocab * hidden {
            return None;
        }
        Some(Self {
            table,
            vocab,
            hidden,
            _reserved: (),
        })
    }

    fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let mut acc = vec![0.0f32; self.hidden];
        let mut n = 0usize;
        for &t in ids {
            let t = t as usize;
            if t < self.vocab {
                let base = t * self.hidden;
                for j in 0..self.hidden {
                    acc[j] += self.table[base + j];
                }
                n += 1;
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            for x in acc.iter_mut() {
                *x *= inv;
            }
        }
        // L2-normalize so dot == cosine (the store's HNSW ranks by dot).
        let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in acc.iter_mut() {
            *x /= norm;
        }
        acc
    }
}

impl BlockEmbedder for TokenMeanEmbedder {
    fn dim(&self) -> usize {
        self.hidden
    }
    fn embed_document(&self, token_ids: &[u32]) -> Vec<f32> {
        self.embed(token_ids)
    }
    fn embed_query(&self, token_ids: &[u32]) -> Vec<f32> {
        self.embed(token_ids)
    }
}

/// Dedicated retrieval encoder (rlx-embed BERT/MiniLM/BGE) — the high-selectivity
/// dual-encoder path. Detokenizes a block's qwen3 token span back to text, then
/// runs a contrastively-trained sentence encoder over it. Its embedding space is
/// built for retrieval (a question sits next to its answer), so small-top-k recall
/// lands the right block — unlike K·K or bag-of-static-embeddings. Downloads the
/// encoder weights via hf-hub on construction.
#[cfg(feature = "dual-encoder")]
pub struct RlxEmbedEmbedder {
    /// The generation model's tokenizer, used to detokenize block token ids → text.
    qwen_tok: tokenizers::Tokenizer,
    /// The sentence encoder (recompiles per (batch,seq); guarded for `&self` embed).
    model: std::sync::Mutex<rlx_embed::RlxBertModel>,
    btok: rlx_embed::BertTokenizer,
    pooling: rlx_embed::Pooling,
    dim: usize,
    /// Instruction prepended to QUERY text (asymmetric retrieval encoders like BGE
    /// need it; documents get none). Empty for symmetric encoders.
    query_prefix: String,
}

#[cfg(feature = "dual-encoder")]
impl RlxEmbedEmbedder {
    /// Download `repo` (e.g. `sentence-transformers/all-MiniLM-L6-v2`) and build a
    /// text encoder on `device`. `qwen_tok` is the generation model's tokenizer
    /// (for detokenizing block spans back to text).
    pub fn from_pretrained(
        qwen_tok: tokenizers::Tokenizer,
        repo: &str,
        device: rlx_runtime::Device,
    ) -> anyhow::Result<Self> {
        use anyhow::Context;
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()
            .context("hf-hub ApiBuilder::build")?;
        let r = api.model(repo.to_string());
        let config = r
            .get("config.json")
            .with_context(|| format!("fetch {repo} config.json"))?;
        let weights = r
            .get("model.safetensors")
            .with_context(|| format!("fetch {repo} model.safetensors"))?;
        let _tok_json = r
            .get("tokenizer.json")
            .with_context(|| format!("fetch {repo} tokenizer.json"))?;
        // BertTokenizer::from_dir needs these two as well (into the same dir).
        let _stm = r
            .get("special_tokens_map.json")
            .with_context(|| format!("fetch {repo} special_tokens_map.json"))?;
        let _tcfg = r
            .get("tokenizer_config.json")
            .with_context(|| format!("fetch {repo} tokenizer_config.json"))?;
        let dir = config
            .parent()
            .ok_or_else(|| anyhow::anyhow!("embed config has no parent dir"))?;
        let weights_str = weights
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 weights path"))?;
        let model = rlx_embed::RlxBertModel::load_sized_on(&config, weights_str, 1, 64, device)
            .context("RlxBertModel::load_sized_on")?;
        let dim = model.hidden_size();
        let btok = rlx_embed::BertTokenizer::from_dir(dir, 256)
            .with_context(|| format!("BertTokenizer::from_dir {dir:?}"))?;
        // BGE-family retrieval encoders are asymmetric: the query needs an
        // instruction prefix (documents don't). Missing it badly hurts recall.
        let query_prefix = if repo.to_ascii_lowercase().contains("bge") {
            "Represent this sentence for searching relevant passages: ".to_string()
        } else {
            String::new()
        };
        Ok(Self {
            qwen_tok,
            model: std::sync::Mutex::new(model),
            btok,
            pooling: rlx_embed::default_pooling(repo),
            dim,
            query_prefix,
        })
    }

    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut m = self.model.lock().expect("embed model lock");
        rlx_embed::embed_with_rlx(&mut m, &self.btok, &[text], self.pooling)
            .ok()
            .and_then(|mut v| v.drain(..).next())
            .unwrap_or_else(|| vec![0.0; self.dim])
    }

    /// Embed raw document text (no query prefix). For harnesses that hold the text
    /// directly rather than token ids.
    pub fn embed_document_text(&self, text: &str) -> Vec<f32> {
        self.embed_text(text)
    }

    /// Embed raw query text (with the encoder's query instruction prefix).
    pub fn embed_query_text(&self, text: &str) -> Vec<f32> {
        self.embed_text(&format!("{}{}", self.query_prefix, text))
    }

    fn detok(&self, ids: &[u32]) -> String {
        self.qwen_tok.decode(ids, true).unwrap_or_default()
    }
}

#[cfg(feature = "dual-encoder")]
impl BlockEmbedder for RlxEmbedEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed_document(&self, token_ids: &[u32]) -> Vec<f32> {
        self.embed_text(&self.detok(token_ids))
    }
    fn embed_query(&self, token_ids: &[u32]) -> Vec<f32> {
        let text = format!("{}{}", self.query_prefix, self.detok(token_ids));
        self.embed_text(&text)
    }
    fn embed_query_str(&self, text: &str) -> Option<Vec<f32>> {
        Some(self.embed_text(&format!("{}{}", self.query_prefix, text)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_mean_is_content_based_and_normalized() {
        // vocab 4, hidden 2. Distinct token vectors.
        let table = Arc::new(vec![
            1.0, 0.0, // tok 0
            0.0, 1.0, // tok 1
            1.0, 1.0, // tok 2
            -1.0, 0.0, // tok 3
        ]);
        let e = TokenMeanEmbedder::new(table, 4, 2).unwrap();
        // A span sharing tokens with the query embeds closer than a disjoint span.
        let q = e.embed_query(&[0, 2]); // mean of (1,0),(1,1) → (1,0.5) norm
        let near = e.embed_document(&[0, 2]);
        let far = e.embed_document(&[3]); // (-1,0) norm → (-1,0)
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &near) > dot(&q, &far),
            "shared-content span is nearer"
        );
        // L2-normalized.
        assert!((dot(&q, &q) - 1.0).abs() < 1e-5);
    }
}
