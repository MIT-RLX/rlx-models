//! Prefill diagnostics: print top-k c0 logits after text prompt.
use anyhow::Result;
use rlx_sesame::generate::GenerateOpts;
use rlx_sesame::tokenize::{SesameTokenizer, tokenize_text_prompt};
use rlx_sesame::weights::{CsmWeights, KvCache, backbone_prefill};

fn main() -> Result<()> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/tts/sesame".into());
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".into());
    let weights = CsmWeights::load(&model)?;
    let tok = SesameTokenizer::load(&model)?;
    let frames = tokenize_text_prompt(&tok, &weights.cfg, &text, 0)?;
    eprintln!("prompt frames={}", frames.len());
    let embeds: Vec<_> = frames
        .iter()
        .map(|f| weights.embed_frame(&f.tokens, &f.mask))
        .collect();
    let mut kv = KvCache::new(weights.backbone.layers.len());
    let last = backbone_prefill(&embeds, &weights, &mut kv);
    let logits = weights.c0_logits(&last);
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("c0 top10:");
    for (i, (id, v)) in indexed.iter().take(10).enumerate() {
        eprintln!("  {i}: id={id} logit={v:.4}");
    }
    let finite = logits.iter().filter(|v| v.is_finite()).count();
    eprintln!(
        "logits: len={} finite={} max={:.4} min={:.4}",
        logits.len(),
        finite,
        indexed[0].1,
        indexed.last().unwrap().1
    );
    // One greedy frame
    let mut rng = fastrand::Rng::with_seed(42);
    let codes = rlx_sesame::depth::generate_frame_codes(&weights, &last, 0.9, 50, true, &mut rng);
    eprintln!(
        "greedy frame codes[0..8]={:?} ... max={}",
        &codes[..8],
        codes.iter().max().unwrap()
    );
    let _ = GenerateOpts::default();
    Ok(())
}
