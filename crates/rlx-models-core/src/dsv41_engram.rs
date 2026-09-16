// RLX — versatile ML compiler + runtime. GPLv3.
//! **DeepSeek-V4.1 Engram** — the n-gram conditional-memory subsystem.
//!
//! At a few backbone layers (`engram_layer_ids`, GA 4.1 = `[1, 14]`) the residual
//! stream picks up a lookup from a very large hash table: each position hashes the
//! `max_ngram_size - 1` n-grams ending there, fetches one row per `(n-gram size,
//! head)` pair, and the block writes a value into the stream gated by how well the
//! fetched key matches it.
//!
//! Three parts, and all three have to agree exactly with training or the table is
//! read at the wrong rows:
//!
//! * **The compressed token map** ([`compress_token_map`]) — n-grams are hashed
//!   over a *normalized* token space, so `" The"`, `"the"` and `"THE"` collide.
//! * **The bucket layout** ([`EngramHashPlan`]) — every `(n-gram size, head)` pair
//!   owns a disjoint prime-sized range of the layer's table, the primes drawn in
//!   order from `engram_vocab_size - 1` and never reused.
//! * **The multipliers** — one per `(layer, look-back)` from
//!   `np.random.default_rng(10007 · layer_id)`, which means reproducing numpy's
//!   `SeedSequence` → PCG64 → Lemire-bounded `integers()` chain bit for bit
//!   ([`np_rng`]).
//!
//! Reference: `deepseek-ai/DeepSeek-V4.1-Flash` `inference/engram.py` and the
//! `Engram` module in `inference/model.py`.

use crate::dsv41::EngramSpec;
use crate::standard_decoder::{load_p, synth_const, synth_zero};
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::GraphExt;
use rlx_ir::graph::{Graph, NodeId};
use std::collections::HashMap;

/// Bit-exact re-implementation of the pieces of `numpy.random` that
/// `compute_hash_multipliers` depends on: `SeedSequence(seed).generate_state(4,
/// uint64)`, PCG64 (XSL-RR 128/64) seeded from it, and
/// `Generator.integers(0, high, dtype=int64)` — which is Lemire's bounded
/// sampler, not masked rejection, for `Generator` (unlike legacy `RandomState`).
///
/// A wrong multiplier does not fail loudly: it rehashes every n-gram into a
/// different bucket, so the model reads uniformly random rows of a 384-million-row
/// table and simply degrades. Hence the verbatim port plus test vectors.
pub mod np_rng {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;
    const POOL_SIZE: usize = 4;
    /// PCG64 multiplier (`0x2360ED051FC65DA44385DF649FCCF645`).
    const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    /// `SeedSequence.mix_entropy` — fills the 4-word pool from the entropy words.
    fn mix_entropy(entropy: &[u32]) -> [u32; POOL_SIZE] {
        let mut hash_const = INIT_A;
        let hashmix = |value: u32, hash_const: &mut u32| -> u32 {
            let mut v = value ^ *hash_const;
            *hash_const = hash_const.wrapping_mul(MULT_A);
            v = v.wrapping_mul(*hash_const);
            v ^= v >> XSHIFT;
            v
        };
        let mix = |x: u32, y: u32| -> u32 {
            let mut r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            r ^= r >> XSHIFT;
            r
        };
        let mut pool = [0u32; POOL_SIZE];
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for &e in entropy.iter().skip(POOL_SIZE) {
            for i_dst in 0..POOL_SIZE {
                let h = hashmix(e, &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
        pool
    }

    /// `SeedSequence.generate_state(n_words, uint32)`.
    fn generate_state_u32(pool: &[u32; POOL_SIZE], n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut v = pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^= v >> XSHIFT;
                v
            })
            .collect()
    }

    /// numpy's `_int_to_uint32_array`: little-endian 32-bit words of a
    /// non-negative integer seed (`0` yields a single zero word).
    fn seed_words(mut seed: u64) -> Vec<u32> {
        if seed == 0 {
            return vec![0];
        }
        let mut out = Vec::new();
        while seed > 0 {
            out.push((seed & 0xFFFF_FFFF) as u32);
            seed >>= 32;
        }
        out
    }

    /// numpy's PCG64 (`pcg_setseq_128_xsl_rr_64`) as seeded by `default_rng`.
    pub struct Pcg64 {
        state: u128,
        inc: u128,
    }

    impl Pcg64 {
        /// `np.random.default_rng(seed)`.
        pub fn new(seed: u64) -> Self {
            let pool = mix_entropy(&seed_words(seed));
            // generate_state(4, uint64) == 8 uint32 words viewed little-endian
            let w = generate_state_u32(&pool, 8);
            let u64w = |i: usize| (w[2 * i] as u64) | ((w[2 * i + 1] as u64) << 32);
            let init_state = ((u64w(0) as u128) << 64) | u64w(1) as u128;
            let init_seq = ((u64w(2) as u128) << 64) | u64w(3) as u128;
            let mut g = Pcg64 {
                state: 0,
                inc: (init_seq << 1) | 1,
            };
            g.step();
            g.state = g.state.wrapping_add(init_state);
            g.step();
            g
        }

        fn step(&mut self) {
            self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        }

        /// `next_uint64` — step, then XSL-RR output.
        pub fn next_u64(&mut self) -> u64 {
            self.step();
            let s = self.state;
            let xored = ((s >> 64) as u64) ^ (s as u64);
            xored.rotate_right((s >> 122) as u32)
        }

        /// `Generator.integers(0, high, dtype=int64)` for `high > u32::MAX + 1`
        /// — Lemire's bounded sampler with the rejection fallback. `high` is
        /// exclusive.
        pub fn bounded_u64(&mut self, high: u64) -> u64 {
            let rng = high - 1;
            if rng == 0 {
                return 0;
            }
            let rng_excl = (rng as u128) + 1;
            let mut m = (self.next_u64() as u128) * rng_excl;
            let mut leftover = m as u64;
            if (leftover as u128) < rng_excl {
                let threshold = ((u64::MAX as u128 - rng_excl + 1) % rng_excl) as u64;
                while leftover < threshold {
                    m = (self.next_u64() as u128) * rng_excl;
                    leftover = m as u64;
                }
            }
            (m >> 64) as u64
        }
    }
}

/// Deterministic primality for the bucket moduli. `engram_vocab_size` is ~1.6e7
/// in the GA checkpoint, so trial division is both exact and instant.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    if n.is_multiple_of(3) {
        return n == 3;
    }
    let mut f = 5u64;
    while f.saturating_mul(f) <= n {
        if n.is_multiple_of(f) || n.is_multiple_of(f + 2) {
            return false;
        }
        f += 6;
    }
    true
}

/// `find_next_prime` — the smallest prime above `start` not already handed out.
fn next_prime(start: u64, seen: &mut std::collections::HashSet<u64>) -> u64 {
    let mut c = start + 1;
    while !is_prime(c) || seen.contains(&c) {
        c += 1;
    }
    c
}

/// The fully-resolved hash layout of one checkpoint's Engram tables: bucket
/// moduli, their offsets inside each layer's table, and the per-layer multipliers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramHashPlan {
    /// `[layer][n-gram size - 2][head]` bucket modulus.
    pub primes: Vec<Vec<Vec<u64>>>,
    /// `[layer][col]` offset of each `(n-gram size, head)` range within the
    /// layer's table, `col` running n-gram-major then head.
    pub offsets: Vec<Vec<u64>>,
    /// `[layer][look-back]` odd multipliers, `look-back` in `0..max_ngram_size`.
    pub multipliers: Vec<Vec<i64>>,
    max_ngram_size: usize,
    n_heads: usize,
    /// Compressed id of the pad token — fills look-back slots with no history.
    pad_id: u32,
}

impl EngramHashPlan {
    /// Build the layout from the spec. `token_map` supplies the compressed id of
    /// the pad token (see [`compress_token_map`]).
    pub fn new(spec: &EngramSpec, token_map: &[u32]) -> Result<Self> {
        if spec.max_ngram_size < 2 {
            return Err(anyhow!("engram: max_ngram_size must be >= 2"));
        }
        if spec.compressed_vocab_size == 0 {
            return Err(anyhow!("engram: compressed_vocab_size must be > 0"));
        }
        let pad_id = *token_map.get(spec.pad_token_id).ok_or_else(|| {
            anyhow!(
                "engram: pad token id {} outside the {}-entry token map",
                spec.pad_token_id,
                token_map.len()
            )
        })?;

        // Primes are drawn in one global sequence across layers, so a layer's
        // ranges depend on how many were consumed before it.
        let mut seen = std::collections::HashSet::new();
        let mut primes = Vec::with_capacity(spec.layer_ids.len());
        for _ in &spec.layer_ids {
            let mut per_ngram = Vec::with_capacity(spec.max_ngram_size - 1);
            for _ in 0..spec.max_ngram_size - 1 {
                let mut current = spec.vocab_size as u64 - 1;
                let mut sizes = Vec::with_capacity(spec.n_heads);
                for _ in 0..spec.n_heads {
                    current = next_prime(current, &mut seen);
                    seen.insert(current);
                    sizes.push(current);
                }
                per_ngram.push(sizes);
            }
            primes.push(per_ngram);
        }

        let offsets = primes
            .iter()
            .map(|layer| {
                let flat: Vec<u64> = layer.iter().flatten().copied().collect();
                let mut acc = 0u64;
                flat.iter()
                    .map(|&p| {
                        let o = acc;
                        acc += p;
                        o
                    })
                    .collect()
            })
            .collect();

        // bound = max(1, (i64::MAX // compressed_vocab) // 2); values are kept odd
        // and small enough that `token_id · multiplier` cannot overflow int64.
        let bound = ((i64::MAX as u64 / spec.compressed_vocab_size as u64) / 2).max(1);
        let multipliers = spec
            .layer_ids
            .iter()
            .map(|&lid| {
                let mut g = np_rng::Pcg64::new(10007u64.wrapping_mul(lid as u64));
                (0..spec.max_ngram_size)
                    .map(|_| g.bounded_u64(bound) as i64 * 2 + 1)
                    .collect()
            })
            .collect();

        Ok(EngramHashPlan {
            primes,
            offsets,
            multipliers,
            max_ngram_size: spec.max_ngram_size,
            n_heads: spec.n_heads,
            pad_id,
        })
    }

    /// Columns of hash ids per position.
    pub fn n_hash_cols(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_heads
    }

    /// Row indices for every `(position, engram layer, column)`, flattened
    /// `[seq · n_layers · n_hash_cols]` in that order.
    ///
    /// `compressed` is the sequence's token ids already mapped through
    /// [`compress_token_map`]; `alive[i] == false` marks a position that takes no
    /// part in an n-gram (an image span), which both blanks its own hash and stops
    /// look-back for later positions — an n-gram never spans a dead token.
    /// `start_pos` shifts the window for a continuation; `history` holds the
    /// compressed ids of everything before it (empty for a fresh prefill).
    pub fn hash_ids(&self, compressed: &[u32], alive: Option<&[bool]>, history: &[u32]) -> Vec<i64> {
        let seq = compressed.len();
        let n_layers = self.multipliers.len();
        let cols = self.n_hash_cols();
        let mut out = vec![0i64; seq * n_layers * cols];
        // The look-back walks the concatenation of `history` and this chunk; a
        // dead token is recorded as DEAD so it blocks every n-gram crossing it.
        const DEAD: i64 = -1;
        let mut tape: Vec<i64> = Vec::with_capacity(history.len() + seq);
        tape.extend(history.iter().map(|&c| c as i64));
        let base = tape.len();
        for (i, &c) in compressed.iter().enumerate() {
            let live = alive.map(|a| a[i]).unwrap_or(true);
            tape.push(if live { c as i64 } else { DEAD });
        }

        for i in 0..seq {
            let pos = base + i;
            // tokens[s] = the id `s` steps back, pad once blocked
            let mut tokens = vec![0i64; self.max_ngram_size];
            let mut blocked = false;
            for (shift, slot) in tokens.iter_mut().enumerate() {
                let src = if shift > pos { DEAD } else { tape[pos - shift] };
                blocked = blocked || shift > pos || src == DEAD;
                *slot = if blocked { self.pad_id as i64 } else { src };
            }
            for l in 0..n_layers {
                let mult = &self.multipliers[l];
                // rolling XOR of the multiplied ids: the value after step i is the
                // hash of the (i+1)-gram
                let mut rolling = tokens[0].wrapping_mul(mult[0]);
                let mut col = 0usize;
                for k in 1..self.max_ngram_size {
                    rolling ^= tokens[k].wrapping_mul(mult[k]);
                    for h in 0..self.n_heads {
                        let p = self.primes[l][k - 1][h] as i64;
                        let off = self.offsets[l][col] as i64;
                        out[(i * n_layers + l) * cols + col] = rolling.rem_euclid(p) + off;
                        col += 1;
                    }
                }
            }
        }
        out
    }
}

/// Normalize one decoded token exactly as `build_compressed_token_map`'s
/// `normalizers.Sequence` does: NFKC → NFD → strip accents → lowercase →
/// collapse `[ \t\r\n]+` to one space → protect a lone space → strip → restore.
fn normalize_token(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    use unicode_normalization::char::is_combining_mark;
    // NFKC then NFD; strip accents drops the combining marks NFD exposed.
    let decomposed: String = text.nfkc().collect::<String>().nfd().collect();
    let stripped: String = decomposed.chars().filter(|c| !is_combining_mark(*c)).collect();
    // `tokenizers::Lowercase` lowercases char by char.
    let lowered: String = stripped.chars().flat_map(char::to_lowercase).collect();
    // collapse runs of the four ASCII space characters
    let mut collapsed = String::with_capacity(lowered.len());
    let mut in_ws = false;
    for c in lowered.chars() {
        if matches!(c, ' ' | '\t' | '\r' | '\n') {
            if !in_ws {
                collapsed.push(' ');
            }
            in_ws = true;
        } else {
            collapsed.push(c);
            in_ws = false;
        }
    }
    // A token that is exactly one space must survive Strip() instead of
    // collapsing to "" and merging with unrelated tokens, so it is swapped for a
    // private-use sentinel across the strip and swapped back after.
    if collapsed == " " {
        return " ".to_string();
    }
    collapsed.trim_matches(char::is_whitespace).to_string()
}

/// Build the compressed token map: tokens that normalize alike collapse onto one
/// id. Returns `(map, compressed_vocab_size)`; the size is not a bound check but
/// the seed of every hash multiplier, so it must match
/// `engram_compressed_vocab_size`.
///
/// `decoded[i]` is what the tokenizer decodes id `i` to with
/// `skip_special_tokens=False`, and `pieces[i]` its raw piece. A decoded string
/// containing `U+FFFD` is a partial UTF-8 byte token with nothing to normalize,
/// so it is keyed by its raw piece instead.
pub fn compress_token_map(decoded: &[String], pieces: &[String]) -> (Vec<u32>, usize) {
    let mut key_to_new: HashMap<String, u32> = HashMap::new();
    let mut map = Vec::with_capacity(decoded.len());
    for (i, text) in decoded.iter().enumerate() {
        let key = if text.contains('\u{FFFD}') {
            pieces.get(i).cloned().unwrap_or_else(|| text.clone())
        } else {
            let n = normalize_token(text);
            if n.is_empty() { text.clone() } else { n }
        };
        let next = key_to_new.len() as u32;
        let id = *key_to_new.entry(key).or_insert(next);
        map.push(id);
    }
    let n = key_to_new.len();
    (map, n)
}

/// Emit one **Engram** block: `x + gate · value`, where the gate measures how well
/// the fetched key matches the stream.
///
/// `x` is the Hyper-Connection stream `[rows, hc, dim]`; `rows_ids` is an
/// `[rows, n_hash_cols]` I32 node holding this layer's slice of
/// [`EngramHashPlan::hash_ids`]. `table` is the layer's dequantized
/// `[num_embeddings, head_dim]` lookup. `alive_mask`, when given, is `[rows, 1]`
/// with `0` at positions that must pass through untouched.
///
/// Mirrors `Engram.forward`: `wkv` turns the flattened rows into one key per hc
/// copy plus a shared value; the gate is a per-(token, copy) normalized dot
/// product of stream against key, passed through a signed square root and a
/// sigmoid. The normalization is per `(token, copy)` over `dim` — **not** jointly
/// over the copies.
#[allow(clippy::too_many_arguments)]
pub fn build_v41_engram(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    x: NodeId,
    row_ids: NodeId,
    table: NodeId,
    wkv_t: NodeId,
    q_weight: NodeId,
    k_weight: NodeId,
    alive_mask: Option<NodeId>,
    rows: usize,
    hc: usize,
    dim: usize,
    head_dim: usize,
    n_hash_cols: usize,
    eps: f32,
    tag: &str,
) -> NodeId {
    let (r, h, d) = (rows as i64, hc as i64, dim as i64);
    // embed(hash_ids).flatten(-2) → [rows, n_hash_cols·head_dim]
    let fetched = g.gather_(table, row_ids, 0); // [rows, n_hash_cols, head_dim]
    let fetched = g.reshape_(fetched, vec![r, (n_hash_cols * head_dim) as i64]);
    let kv = g.mm(fetched, wkv_t); // [rows, (hc+1)·dim]
    let key = g.narrow_(kv, 1, 0, hc * dim);
    let value = g.narrow_(kv, 1, hc * dim, dim);
    let key = g.reshape_(key, vec![r, h, d]);

    // weight = q_weight · k_weight, only ever used as a product → [1, hc, dim]
    let weight = g.mul(q_weight, k_weight);
    let weight = g.reshape_(weight, vec![1, h, d]);

    let eps_c = synth_const(g, params, &format!("{tag}.eng.eps"), vec![eps], &[1, 1, 1]);
    let msq = |g: &mut Graph, t: NodeId| -> NodeId {
        let sq = g.mul(t, t);
        let m = g.mean(sq, vec![2], true); // [rows, hc, 1]
        let m = g.add(m, eps_c);
        g.rsqrt(m)
    };
    let rstd_x = msq(g, x);
    let rstd_k = msq(g, key);
    let rstd = g.mul(rstd_x, rstd_k);

    let xw = g.mul(x, weight);
    let dot = g.mul(xw, key);
    let dot = g.sum(dot, vec![2], true); // [rows, hc, 1]
    let dot = g.mul(dot, rstd);
    let inv_sqrt_d = synth_const(
        g,
        params,
        &format!("{tag}.eng.dscale"),
        vec![(dim as f32).powf(-0.5)],
        &[1, 1, 1],
    );
    let dot = g.mul(dot, inv_sqrt_d);

    // gate = sigmoid(copysign(sqrt(clamp_min(|dot|, 1e-6)), dot)) — the signed
    // square root the training kernel applies before the sigmoid.
    let adot = g.abs(dot);
    let adot = g.clamp_(adot, 1e-6, f32::MAX);
    let root = g.sqrt(adot);
    // `copysign(root, dot)`. `Activation::Sign` returns 0 at exactly zero while
    // `copysign` treats +0 as positive, so fold that one case back to +1:
    // `s + (1 - |s|)` is `s` when |s| == 1 and `1` when s == 0.
    let dot_shape = g.shape(dot).clone();
    let s_raw = g.activation(rlx_ir::op::Activation::Sign, dot, dot_shape);
    let s_abs = g.abs(s_raw);
    let one = synth_const(g, params, &format!("{tag}.eng.one"), vec![1.0], &[1, 1, 1]);
    let gap = g.sub(one, s_abs);
    let sign = g.add(s_raw, gap);
    let signed = g.mul(root, sign);
    let mut gate = g.sigmoid(signed); // [rows, hc, 1]
    if let Some(mask) = alive_mask {
        let m3 = g.reshape_(mask, vec![r, 1, 1]);
        gate = g.mul(gate, m3);
    }
    let value3 = g.reshape_(value, vec![r, 1, d]);
    let add = g.mul(gate, value3);
    g.add(x, add)
}

/// Load one layer's Engram weights and emit the block. `lp` is the layer prefix
/// (`layers.{N}`); the checkpoint stores the table as `engram.embed.weight` with a
/// companion `engram.embed.scale`, which the loader is expected to have already
/// folded in (see [`crate::dsv41_quant`]).
#[allow(clippy::too_many_arguments)]
pub fn load_and_build_v41_engram(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    lp: &str,
    x: NodeId,
    row_ids: NodeId,
    alive_mask: Option<NodeId>,
    rows: usize,
    hc: usize,
    dim: usize,
    spec: &EngramSpec,
    eps: f32,
) -> Result<NodeId> {
    let table = load_p(g, params, weights, &format!("{lp}.engram.embed.weight"), false)?;
    let wkv_t = load_p(g, params, weights, &format!("{lp}.engram.wkv.weight"), true)?;
    let q_weight = load_p(g, params, weights, &format!("{lp}.engram.q_weight"), false)?;
    let k_weight = load_p(g, params, weights, &format!("{lp}.engram.k_weight"), false)?;
    let _ = synth_zero(g, params, &format!("{lp}.engram.unused"), 0);
    Ok(build_v41_engram(
        g,
        params,
        x,
        row_ids,
        table,
        wkv_t,
        q_weight,
        k_weight,
        alive_mask,
        rows,
        hc,
        dim,
        spec.head_dim,
        spec.n_hash_cols(),
        eps,
        lp,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::EngramSpec;

    /// `np.random.default_rng(10007·layer).integers(0, bound, size=4, dtype=int64)`
    /// for the released `engram_compressed_vocab_size = 99092`, captured from
    /// numpy 2.4.3. These are the multipliers the GA checkpoint was trained with.
    #[test]
    fn numpy_pcg64_reproduces_reference_multipliers() {
        let bound = ((i64::MAX as u64 / 99092) / 2).max(1);
        assert_eq!(bound, 46_539_438_283_891);
        let cases: [(u64, [i64; 4]); 3] = [
            (1, [38316048023122, 2419938046656, 17979836159674, 36993668729195]),
            (14, [33858405369630, 25755403400457, 15460673601360, 41309613242795]),
            (3, [13763488024408, 40829620617460, 5438965595765, 35148471906864]),
        ];
        for (layer, want) in cases {
            let mut g = np_rng::Pcg64::new(10007 * layer);
            let got: Vec<i64> = (0..4).map(|_| g.bounded_u64(bound) as i64).collect();
            assert_eq!(got, want.to_vec(), "layer {layer}");
        }
    }

    /// A different `bound` exercises a different Lemire rejection threshold.
    #[test]
    fn numpy_pcg64_reproduces_small_vocab_multipliers() {
        let bound = ((i64::MAX as u64 / 64) / 2).max(1);
        assert_eq!(bound, 72_057_594_037_927_935);
        let cases: [(u64, [i64; 4]); 2] = [
            (
                1,
                [59325216104801821, 3746820326864244, 27838405073976278, 57277759714273634],
            ),
            (
                14,
                [52423392263866304, 39877413027471626, 23937954195406833, 63960190553987139],
            ),
        ];
        for (layer, want) in cases {
            let mut g = np_rng::Pcg64::new(10007 * layer);
            let got: Vec<i64> = (0..4).map(|_| g.bounded_u64(bound) as i64).collect();
            assert_eq!(got, want.to_vec(), "layer {layer}");
        }
    }

    fn ga_engram() -> EngramSpec {
        EngramSpec {
            layer_ids: vec![1, 14],
            num_embeddings: vec![384006168, 384016682],
            max_ngram_size: 4,
            vocab_size: 16_000_000,
            n_heads: 8,
            head_dim: 256,
            pad_token_id: 2,
            compressed_vocab_size: 99092,
        }
    }

    /// The prime ranges must tile each layer's table exactly — the checkpoint's
    /// `engram_num_embeddings` is the sum of that layer's 24 primes, so a wrong
    /// prime sequence shows up as a row-count mismatch against the real shapes.
    #[test]
    fn prime_layout_tiles_the_released_table_sizes() {
        let spec = ga_engram();
        let map: Vec<u32> = (0..129280).map(|i| i as u32 % 99092).collect();
        let plan = EngramHashPlan::new(&spec, &map).unwrap();
        assert_eq!(plan.primes.len(), 2);
        for (li, layer) in plan.primes.iter().enumerate() {
            assert_eq!(layer.len(), 3); // 2-gram, 3-gram, 4-gram
            let flat: Vec<u64> = layer.iter().flatten().copied().collect();
            assert_eq!(flat.len(), 24);
            let total: u64 = flat.iter().sum();
            assert_eq!(
                total, spec.num_embeddings[li] as u64,
                "layer {li} primes must sum to the checkpoint's table rows"
            );
        }
        // primes are drawn in one global, strictly increasing, never-reused order
        let all: Vec<u64> = plan.primes.iter().flatten().flatten().copied().collect();
        assert_eq!(all.len(), 48);
        let mut sorted = all.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 48, "no prime is handed out twice");
        assert!(all.iter().all(|&p| p >= 16_000_000));
        // offsets are the running sum of the layer's own primes
        for (li, offs) in plan.offsets.iter().enumerate() {
            let flat: Vec<u64> = plan.primes[li].iter().flatten().copied().collect();
            let mut acc = 0;
            for (c, &o) in offs.iter().enumerate() {
                assert_eq!(o, acc, "layer {li} col {c}");
                acc += flat[c];
            }
        }
    }

    /// Every emitted row index must land inside its own `(n-gram, head)` bucket
    /// range — that disjointness is what keeps the three n-gram sizes from
    /// aliasing each other in one table.
    #[test]
    fn hash_ids_stay_inside_their_bucket_ranges() {
        let spec = ga_engram();
        let map: Vec<u32> = (0..129280).map(|i| i as u32 % 99092).collect();
        let plan = EngramHashPlan::new(&spec, &map).unwrap();
        let ids: Vec<u32> = (0..32).map(|i| map[(i * 977 + 13) % map.len()]).collect();
        let out = plan.hash_ids(&ids, None, &[]);
        let cols = plan.n_hash_cols();
        assert_eq!(out.len(), 32 * 2 * cols);
        for i in 0..32 {
            for l in 0..2 {
                let flat: Vec<u64> = plan.primes[l].iter().flatten().copied().collect();
                for c in 0..cols {
                    let v = out[(i * 2 + l) * cols + c];
                    let lo = plan.offsets[l][c] as i64;
                    let hi = lo + flat[c] as i64;
                    assert!(v >= lo && v < hi, "pos {i} layer {l} col {c}: {v} ∉ [{lo},{hi})");
                    assert!(v < spec.num_embeddings[l] as i64);
                }
            }
        }
    }

    /// Look-back stops at the start of the sequence and at any dead (image-span)
    /// token, so an n-gram never spans one.
    #[test]
    fn dead_tokens_and_sequence_start_block_lookback() {
        let spec = EngramSpec {
            max_ngram_size: 3,
            n_heads: 1,
            vocab_size: 1009,
            num_embeddings: vec![0],
            layer_ids: vec![0],
            head_dim: 8,
            pad_token_id: 2,
            compressed_vocab_size: 64,
        };
        let map: Vec<u32> = (0..64).collect();
        let plan = EngramHashPlan::new(&spec, &map).unwrap();
        let ids: Vec<u32> = vec![10, 11, 12, 13, 14];
        let all_live = plan.hash_ids(&ids, None, &[]);
        // Position 0 has no history: both look-backs are pad, so its 2-gram hash
        // equals what a sequence of [pad, 10] would give at position 1.
        let padded = plan.hash_ids(&[2, 10], None, &[]);
        let cols = plan.n_hash_cols();
        assert_eq!(all_live[0..cols], padded[cols..2 * cols]);
        // Killing position 2 must change positions 2, 3 and 4 (the 3-gram at 4
        // still reaches back to 2) but leave 0 and 1 untouched.
        let alive = [true, true, false, true, true];
        let masked = plan.hash_ids(&ids, Some(&alive), &[]);
        assert_eq!(all_live[0..2 * cols], masked[0..2 * cols]);
        assert_ne!(all_live[2 * cols..3 * cols], masked[2 * cols..3 * cols]);
        assert_ne!(all_live[3 * cols..4 * cols], masked[3 * cols..4 * cols]);
        assert_ne!(all_live[4 * cols..5 * cols], masked[4 * cols..5 * cols]);
    }

    /// A continuation fed its history must hash identically to the one-shot
    /// prefill of the whole sequence — the property decode depends on.
    #[test]
    fn history_makes_continuation_match_full_prefill() {
        let spec = EngramSpec {
            max_ngram_size: 4,
            n_heads: 2,
            vocab_size: 1009,
            num_embeddings: vec![0],
            layer_ids: vec![5],
            head_dim: 8,
            pad_token_id: 2,
            compressed_vocab_size: 64,
        };
        let map: Vec<u32> = (0..64).collect();
        let plan = EngramHashPlan::new(&spec, &map).unwrap();
        let ids: Vec<u32> = (0..12).map(|i| (i * 7 + 3) % 64).collect();
        let full = plan.hash_ids(&ids, None, &[]);
        let cols = plan.n_hash_cols();
        for split in 1..12 {
            let cont = plan.hash_ids(&ids[split..], None, &ids[..split]);
            assert_eq!(
                cont,
                full[split * cols..].to_vec(),
                "split at {split} must match the full prefill"
            );
        }
    }

    #[test]
    fn token_map_collapses_case_space_and_accents() {
        let decoded: Vec<String> = ["The", " the", "THE", "  the\t", "café", "cafe", "x", " "]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let pieces: Vec<String> = decoded.clone();
        let (map, n) = compress_token_map(&decoded, &pieces);
        assert_eq!(map[0], map[1]);
        assert_eq!(map[0], map[2]);
        assert_eq!(map[0], map[3]);
        assert_eq!(map[4], map[5], "accents are stripped");
        assert_ne!(map[0], map[6]);
        assert_ne!(map[0], map[7], "a lone space is its own id, not the empty key");
        assert_eq!(n, 4); // {the, cafe, x, " "}
    }

    #[test]
    fn byte_fallback_tokens_key_on_their_raw_piece() {
        let decoded: Vec<String> = vec!["\u{FFFD}".into(), "\u{FFFD}".into(), "a".into()];
        let pieces: Vec<String> = vec!["<0x80>".into(), "<0x81>".into(), "a".into()];
        let (map, n) = compress_token_map(&decoded, &pieces);
        assert_ne!(map[0], map[1], "two byte tokens must not collapse together");
        assert_eq!(n, 3);
    }

    #[test]
    fn plan_rejects_pad_token_outside_the_map() {
        let mut spec = ga_engram();
        spec.pad_token_id = 999;
        let err = EngramHashPlan::new(&spec, &[0, 1, 2]).unwrap_err().to_string();
        assert!(err.contains("pad token id 999"), "{err}");
    }
}
