//! What does the cross-attention actually look like, and does it align?
//!
//! The manifest names `AlignmentLayerStr: decoder.1.encoder_attn.attn_probs`
//! with `AlignmentHeads: 1` and `ShiftedAlignments: true`, so the alignment the
//! unimplemented blocks need — `PDecForceAlign`, `AlignmentProcessor`,
//! `LinkAlternatives`, and `unk-replace` — is a tensor the decoder already
//! produces and `Nmt::step` throws away.
//!
//! Before building on it: what shape is it, does each target token peak on a
//! plausible source token, and what does `ShiftedAlignments` mean?

use anyhow::Result;
use rlx_translate::assets::Assets;
use rlx_translate::decode::{DecoderState, Nmt};
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

fn main() -> Result<()> {
    let assets = Assets::discover();
    let pair = LangPair::parse(&std::env::args().nth(1).unwrap_or("en_US-fr_FR".into()))?;
    let home = assets
        .model_home(&pair)
        .ok_or_else(|| anyhow::anyhow!("no model"))?;
    let (_, config) = assets.best_config(&pair)?;
    let params = rlx_translate::pdec::PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;
    let vocab = Vocab::load(home.join("spm.model"))?;
    let two = |l: &str| l.chars().take(2).collect::<String>().to_lowercase();
    let nmt = Nmt::load_for_pair(
        &home,
        &[home.as_path()],
        &two(&pair.source),
        &two(&pair.target),
        &params.shortlist.lang_pair,
    )?;

    let text = std::env::var("TEXT").unwrap_or("the small grey cat slept".into());
    let bos = vocab.id("<s>").expect("bos");
    let mut src: Vec<u32> = params
        .source_token_pieces()
        .iter()
        .chain(params.target_token_pieces().iter())
        .filter_map(|p| vocab.id(p))
        .collect();
    let tags = src.len();
    src.extend(vocab.encode(&text));
    src.push(bos);
    let src_pieces: Vec<&str> = src.iter().filter_map(|t| vocab.piece(*t)).collect();
    println!("source ({} pieces, {tags} tags):", src.len());
    println!("  {}\n", src_pieces.join(" "));

    let ho = nmt.encode(&src)?;
    let layer = nmt
        .manifest()
        .str("AlignmentLayerStr")
        .unwrap_or("decoder.1.encoder_attn.attn_probs")
        .to_string();

    let cands: Vec<u32> = (0..vocab.len() as u32).collect();
    let mut state = DecoderState::new(nmt.layers(), nmt.width);
    let mut prev = bos;
    for i in 0..14usize {
        let env = nmt.step_env(prev, i + 1, &state, &ho)?;
        let Some(v) = env.get(&layer) else {
            println!("no {layer} in the decoder environment");
            println!("produced: {:?}", env.keys().take(20).collect::<Vec<_>>());
            return Ok(());
        };
        let probs = v.f32()?;
        if i == 0 {
            println!("{layer}: dims {:?}\n", probs.dims());
        }
        // Average the heads, then take the peak.
        let n = probs.len() / src.len().max(1);
        let mut best = (0usize, f32::MIN);
        let mut acc = vec![0.0f32; src.len()];
        for h in 0..n {
            for (j, a) in acc.iter_mut().enumerate() {
                *a += probs.data()[h * src.len() + j];
            }
        }
        for (j, a) in acc.iter().enumerate() {
            if *a > best.1 {
                best = (j, *a);
            }
        }
        // Advance greedily.
        let (h, next) = nmt.step(prev, i + 1, &state, &ho)?;
        state = next;
        let scores = nmt.logits(&h, &cands)?;
        let pick = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j as u32)
            .unwrap_or(bos);
        println!(
            "  step {:>2}  emit {:<14} peaks on source[{:>2}] {:?}  ({:.2})",
            i + 1,
            format!("{:?}", vocab.piece(pick).unwrap_or("?")),
            best.0,
            src_pieces.get(best.0).unwrap_or(&"?"),
            best.1 / n.max(1) as f32
        );
        prev = pick;
        if pick == bos {
            break;
        }
    }
    Ok(())
}
