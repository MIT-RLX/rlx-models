//! CSM depth decoder — fills codebooks 1..31 from backbone hidden + c0.

use crate::nn::{argmax, sample_topk};
use crate::weights::{CsmWeights, KvCache, transformer_step};

/// Generate one full RVQ frame (32 codes) from the backbone last hidden state.
pub fn generate_frame_codes(
    weights: &CsmWeights,
    backbone_h: &[f32],
    temperature: f32,
    topk: usize,
    greedy: bool,
    rng: &mut fastrand::Rng,
) -> Vec<u32> {
    let k = weights.cfg.num_codebooks;
    let mut codes = Vec::with_capacity(k);

    // Codebook 0 from backbone lm_head.
    let c0_logits = weights.c0_logits(backbone_h);
    let c0 = if greedy {
        argmax(&c0_logits)
    } else {
        sample_topk(&c0_logits, topk, temperature, rng)
    };
    codes.push(c0);

    // Depth: first step processes [backbone_h, c0_embed] (both at backbone dim → projected).
    let mut depth_kv = KvCache::new(weights.depth.layers.len());
    let h0 = weights.project_to_depth(backbone_h);
    let c0_emb = weights.embed_audio(0, c0);
    let c0_proj = weights.project_to_depth(&c0_emb);

    let _ = transformer_step(&h0, &weights.depth, &mut depth_kv);
    let mut last_h = transformer_step(&c0_proj, &weights.depth, &mut depth_kv);

    for cb in 1..k {
        let logits = weights.codebook_logits(&last_h, cb);
        let ci = if greedy {
            argmax(&logits)
        } else {
            sample_topk(&logits, topk, temperature, rng)
        };
        codes.push(ci);
        if cb + 1 < k {
            let emb = weights.embed_audio(cb, ci);
            let proj = weights.project_to_depth(&emb);
            last_h = transformer_step(&proj, &weights.depth, &mut depth_kv);
        }
    }
    codes
}
