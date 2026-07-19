//! Autoregressive CSM generation loop (eager CPU backbone + depth).

use crate::depth::generate_frame_codes;
use crate::tokenize::{Frame, SesameTokenizer, frames_from_audio_codes, tokenize_text_prompt};
use crate::weights::{CsmWeights, KvCache, backbone_prefill, transformer_step};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct GenerateOpts {
    pub speaker: u32,
    pub max_audio_frames: usize,
    pub temperature: f32,
    pub topk: usize,
    pub seed: u64,
    pub greedy: bool,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        Self {
            speaker: 0,
            max_audio_frames: 1_125, // ~90s @ 12.5 Hz
            temperature: 0.9,
            topk: 50,
            seed: 42,
            greedy: false,
        }
    }
}

/// Generate Mimi codebook frames for `text` (optional context codes).
pub fn generate_codes(
    weights: &CsmWeights,
    tokenizer: &SesameTokenizer,
    text: &str,
    context_codes: Option<&[Vec<u32>]>,
    opts: &GenerateOpts,
) -> Result<Vec<Vec<u32>>> {
    let cfg = &weights.cfg;
    let mut frames: Vec<Frame> = Vec::new();
    if let Some(ctx) = context_codes {
        // Context as prior turn: text prompt of context is caller-owned; here we
        // only append encoded audio frames when provided alone.
        frames.extend(frames_from_audio_codes(cfg, ctx));
    }
    frames.extend(tokenize_text_prompt(tokenizer, cfg, text, opts.speaker)?);

    let embeds: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| weights.embed_frame(&f.tokens, &f.mask))
        .collect();

    let mut kv = KvCache::new(weights.backbone.layers.len());
    let mut last_h = backbone_prefill(&embeds, weights, &mut kv);
    let mut rng = fastrand::Rng::with_seed(opts.seed);
    let mut samples: Vec<Vec<u32>> = Vec::new();

    for _ in 0..opts.max_audio_frames {
        let codes = generate_frame_codes(
            weights,
            &last_h,
            opts.temperature,
            opts.topk,
            opts.greedy,
            &mut rng,
        );
        if codes.iter().all(|&c| c == 0) {
            break;
        }
        samples.push(codes.clone());

        // Feed frame back (audio slots active, text masked).
        let feedback = Frame::audio_with_empty_text(cfg, &codes);
        let emb = weights.embed_frame(&feedback.tokens, &feedback.mask);
        last_h = transformer_step(&emb, &weights.backbone, &mut kv);
    }
    Ok(samples)
}
