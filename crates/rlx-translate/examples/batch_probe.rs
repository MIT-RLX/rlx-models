//! Does the decoder graph tolerate more than one row?
//!
//! Beam search runs `beam` independent single-token steps per position, and
//! each one re-reads every decoder weight. Stacking them into one call would
//! amortise that — but only if the graph's reshapes are polymorphic in the row
//! count the way the encoder's are in sequence length. Cheaper to ask than to
//! reason about.

use anyhow::Result;
use rlx_translate::assets::Assets;
use rlx_translate::decode::{DecoderState, Nmt};
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

fn main() -> Result<()> {
    let assets = Assets::discover();
    let pair = LangPair::parse("en_US-fr_FR")?;
    let home = assets
        .model_home(&pair)
        .ok_or_else(|| anyhow::anyhow!("no fr model installed"))?;
    let vocab = Vocab::load(home.join("spm.model"))?;
    let nmt = Nmt::load(&home, &[home.as_path()], "fr")?;
    let (_, config) = assets.best_config(&pair)?;
    let params = rlx_translate::pdec::PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;

    let bos = vocab.id("<s>").expect("bos");
    let mut src: Vec<u32> = params
        .source_token_pieces()
        .iter()
        .chain(params.target_token_pieces().iter())
        .filter_map(|p| vocab.id(p))
        .collect();
    src.extend(vocab.encode("the woman of my dreams"));
    src.push(bos);
    let handover = nmt.encode(&src)?;

    let state = DecoderState::new(nmt.layers(), nmt.width);
    let (h1, _) = nmt.step(bos, 1, &state, &handover)?;
    println!("single row: hidden {:?}", h1.dims());

    match nmt.step_batch(&[bos, bos], 1, &[state.clone(), state], &handover) {
        Ok((h, next)) => {
            println!("two rows:   hidden {:?}, {} states", h.dims(), next.len());
            let w = nmt.width;
            let row0 = &h.data()[..w];
            let d = row0
                .iter()
                .zip(h1.data())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("batched row 0 vs single-step: max abs diff {d:.3e}");
        }
        Err(e) => println!("two rows:   REFUSED — {e:#}"),
    }
    Ok(())
}
