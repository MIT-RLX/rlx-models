//! Model + training hyper-parameters for the TinyStories GPT.

/// A nanoGPT / GPT-2-style decoder-only transformer config. Byte-level, so the
/// vocabulary is fixed at 256.
#[derive(Clone, Copy, Debug)]
pub struct GptConfig {
    /// Vocabulary size (256 for the byte-level tokenizer).
    pub vocab: usize,
    /// Context length (number of tokens per training window), `T`.
    pub block_size: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads (must divide `n_embd`).
    pub n_head: usize,
    /// Embedding / residual width, `D`.
    pub n_embd: usize,
    /// Training micro-batch size, `B`.
    pub batch: usize,
    /// Label-smoothing epsilon for the cross-entropy target (0.0 = off).
    pub label_smoothing: f32,
}

impl GptConfig {
    /// Per-head dimension, `D / n_head`.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Feed-forward inner width (4× expansion, GPT-2 convention).
    pub fn ffn(&self) -> usize {
        4 * self.n_embd
    }

    /// The default showcase config — ~2.8M parameters, trains to coherent
    /// TinyStories text on Apple GPU (Metal) in minutes.
    pub fn default_metal() -> Self {
        Self {
            vocab: 256,
            block_size: 256,
            n_layer: 6,
            n_head: 6,
            n_embd: 192,
            batch: 16,
            label_smoothing: 0.0,
        }
    }

    /// A tiny config for the CPU smoke test / CI — a few hundred K params, a
    /// handful of seconds.
    pub fn smoke() -> Self {
        Self {
            vocab: 256,
            block_size: 32,
            n_layer: 2,
            n_head: 2,
            n_embd: 64,
            batch: 8,
            label_smoothing: 0.0,
        }
    }

    /// Approximate trainable parameter count (embeddings tied, so counted once).
    pub fn n_params(&self) -> usize {
        let (v, d, t, ff, l) = (
            self.vocab,
            self.n_embd,
            self.block_size,
            self.ffn(),
            self.n_layer,
        );
        let embed = v * d + t * d; // wte + wpe (wte reused for the head)
        let per_layer = 2 * d      // ln1 gain+bias
            + 4 * d * d            // wq wk wv wo
            + 2 * d                // ln2 gain+bias
            + d * ff + ff          // w1 + b1
            + ff * d + d; // w2 + b2
        embed + l * per_layer + 2 * d // + final layernorm
    }

    /// Validate invariants (returns an error string if the config is malformed).
    pub fn check(&self) -> Result<(), String> {
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(format!(
                "n_embd ({}) must be divisible by n_head ({})",
                self.n_embd, self.n_head
            ));
        }
        if self.vocab != 256 {
            return Err("byte-level tokenizer requires vocab == 256".into());
        }
        Ok(())
    }
}

/// LayerNorm epsilon (GPT-2 uses 1e-5).
pub const LN_EPS: f32 = 1e-5;
