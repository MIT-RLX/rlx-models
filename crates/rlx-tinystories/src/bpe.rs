//! A from-scratch **byte-level BPE** tokenizer — trained on the corpus itself,
//! no `tokenizer.json`, no external crate. Byte-level tokens (1 byte = 1 token)
//! are information-sparse: a fixed-length sequence covers only that many bytes
//! of text. BPE merges frequent adjacent pairs into new tokens, so each token
//! carries ~3–4 bytes on English → a fixed-length sequence covers *more text*,
//! and the model reaches a given bits-per-byte in fewer steps. Gather-based
//! embedding (see [`crate::model`]) is what makes a >256 vocab affordable: the
//! per-step payload is `[B*T]` ids regardless of vocab, not a `[B*T, V]` one-hot.
//!
//! Training is the standard word-frequency BPE (à la minbpe/GPT-2): pre-split
//! into whitespace-led chunks, count adjacent pairs weighted by chunk frequency,
//! merge the most frequent, repeat to the target vocab. Encoding applies the
//! learned merges greedily (lowest merge-rank first) per chunk, with a cache.

use std::collections::HashMap;

/// A trained byte-level BPE tokenizer.
#[derive(Clone)]
pub struct Bpe {
    /// Learned merges: adjacent `(a, b)` → the new token id it produces. The id
    /// doubles as the merge *rank* (earlier merges get smaller ids ≥ 256), so
    /// `min` by id reproduces training order at encode time.
    ranks: HashMap<(u32, u32), u32>,
    /// `id → its byte expansion`, for decoding. Ids `0..256` are the raw bytes.
    vocab_bytes: Vec<Vec<u8>>,
}

#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

/// Split into chunks of `[leading whitespace run][non-whitespace run]`, so a
/// leading space stays attached to its word (GPT-2 style: `" the"`). Bounds
/// merge work per chunk and keeps word boundaries stable.
fn chunks(text: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let start = i;
        while i < text.len() && is_ws(text[i]) {
            i += 1;
        }
        while i < text.len() && !is_ws(text[i]) {
            i += 1;
        }
        out.push(&text[start..i]);
    }
    out
}

/// Replace every adjacent occurrence of `pair` in `seq` with `new_id`.
fn merge_seq(seq: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == pair.0 && seq[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    out
}

impl Bpe {
    /// Train BPE on `corpus` up to `vocab_size` tokens (≥ 256). Merges are
    /// counted over unique whitespace-led chunks weighted by frequency, so cost
    /// scales with the *vocabulary* of the text, not its length.
    pub fn train(corpus: &[u8], vocab_size: usize) -> Self {
        assert!(
            vocab_size >= 256,
            "BPE vocab must be ≥ 256 (the byte alphabet)"
        );
        let mut freq: HashMap<&[u8], u32> = HashMap::new();
        for c in chunks(corpus) {
            *freq.entry(c).or_default() += 1;
        }
        // Each unique chunk as a token sequence (starts as raw bytes) + its count.
        let mut seqs: Vec<(Vec<u32>, u32)> = freq
            .into_iter()
            .map(|(w, c)| (w.iter().map(|&b| u32::from(b)).collect(), c))
            .collect();

        let mut ranks: HashMap<(u32, u32), u32> = HashMap::new();
        let mut vocab_bytes: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();

        let mut next_id = 256u32;
        while (next_id as usize) < vocab_size {
            // Count adjacent pairs, weighted by chunk frequency.
            let mut pairs: HashMap<(u32, u32), u64> = HashMap::new();
            for (seq, c) in &seqs {
                for w in seq.windows(2) {
                    *pairs.entry((w[0], w[1])).or_default() += u64::from(*c);
                }
            }
            // Most frequent pair (ties broken by pair value for determinism).
            let best = pairs
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(p, _)| p);
            let Some(pair) = best else { break };

            let new_id = next_id;
            next_id += 1;
            ranks.insert(pair, new_id);
            let mut bytes = vocab_bytes[pair.0 as usize].clone();
            bytes.extend_from_slice(&vocab_bytes[pair.1 as usize]);
            vocab_bytes.push(bytes);

            for (seq, _) in &mut seqs {
                if seq.len() >= 2 {
                    *seq = merge_seq(seq, pair, new_id);
                }
            }
        }

        Self { ranks, vocab_bytes }
    }

    /// Number of tokens in the vocabulary (256 + learned merges).
    pub fn vocab_size(&self) -> usize {
        self.vocab_bytes.len()
    }

    /// Apply the learned merges to one chunk, lowest merge-rank (smallest new id)
    /// first — reproducing training order.
    fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        let mut seq: Vec<u32> = chunk.iter().map(|&b| u32::from(b)).collect();
        loop {
            let mut best: Option<(usize, u32)> = None; // (position, new id)
            for i in 0..seq.len().saturating_sub(1) {
                if let Some(&nid) = self.ranks.get(&(seq[i], seq[i + 1])) {
                    if best.is_none_or(|(_, b)| nid < b) {
                        best = Some((i, nid));
                    }
                }
            }
            let Some((i, nid)) = best else { break };
            seq[i] = nid;
            seq.remove(i + 1);
        }
        seq
    }

    /// Encode text to BPE token ids (chunk-cached for corpus-scale speed).
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let mut cache: HashMap<&[u8], Vec<u32>> = HashMap::new();
        let mut out = Vec::new();
        for c in chunks(text) {
            out.extend_from_slice(cache.entry(c).or_insert_with(|| self.encode_chunk(c)));
        }
        out
    }

    /// Decode token ids back to bytes → lossy UTF-8.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(b) = self.vocab_bytes.get(id as usize) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Serialize the merge table for a checkpoint: `[u32 count]` then, per merge
    /// in id order, `[u32 a][u32 b]`. The byte expansions are rederived on load.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Merges ordered by their produced id (256, 257, …).
        let mut merges: Vec<((u32, u32), u32)> =
            self.ranks.iter().map(|(&p, &id)| (p, id)).collect();
        merges.sort_by_key(|&(_, id)| id);
        let mut out = Vec::with_capacity(4 + merges.len() * 8);
        out.extend_from_slice(&(merges.len() as u32).to_le_bytes());
        for ((a, b), _) in merges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(data: &[u8]) -> Self {
        let rd = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        let n = rd(0) as usize;
        let mut ranks = HashMap::new();
        let mut vocab_bytes: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
        for i in 0..n {
            let a = rd(4 + i * 8);
            let b = rd(4 + i * 8 + 4);
            let new_id = 256 + i as u32;
            ranks.insert((a, b), new_id);
            let mut bytes = vocab_bytes[a as usize].clone();
            bytes.extend_from_slice(&vocab_bytes[b as usize]);
            vocab_bytes.push(bytes);
        }
        Self { ranks, vocab_bytes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_grow_vocab_and_roundtrip() {
        let text = b"the cat sat on the mat. the cat ran. the mat sat.";
        let bpe = Bpe::train(text, 300);
        assert!(bpe.vocab_size() > 256 && bpe.vocab_size() <= 300);
        // Lossless round-trip.
        let ids = bpe.encode(text);
        assert_eq!(bpe.decode(&ids), String::from_utf8_lossy(text));
        // BPE is denser than byte-level (fewer tokens than bytes).
        assert!(ids.len() < text.len());
    }

    #[test]
    fn serialization_roundtrips() {
        let text = b"hello hello world world world foo bar hello world";
        let bpe = Bpe::train(text, 280);
        let restored = Bpe::from_bytes(&bpe.to_bytes());
        assert_eq!(restored.vocab_size(), bpe.vocab_size());
        assert_eq!(restored.encode(text), bpe.encode(text));
        assert_eq!(
            restored.decode(&bpe.encode(text)),
            bpe.decode(&bpe.encode(text))
        );
    }

    #[test]
    fn frequent_pair_becomes_one_token() {
        // "ab" is the most frequent pair → the first merge (id 256).
        let bpe = Bpe::train(b"ababababab", 257);
        assert_eq!(bpe.encode(b"ab"), vec![256]);
    }
}
