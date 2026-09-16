//! Which language does each per-language graph follow — the source or the
//! target?
//!
//! A bundle ships `input_<lang>`, `handover_<lang>` and `decoder_<lang>`. This
//! port keys all three by the *target*, which produces correct French, Hindi
//! and Arabic — but `input_<lang>` is the first four encoder blocks and reads
//! the *source*, so the assumption is worth testing rather than inheriting.
//! `en_US-tr_TR` is the direction that fails, so it is the one that should
//! discriminate.

use anyhow::Result;
use rlx_translate::assets::Assets;
use rlx_translate::decode::{Nmt, Parts};
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

fn main() -> Result<()> {
    let assets = Assets::discover();
    let text = std::env::var("TEXT")
        .unwrap_or_else(|_| "we walked along the river until it started raining".to_string());
    // Default to a direction every installation has, so running with no
    // arguments shows something rather than nothing.
    let mut names: Vec<String> = std::env::args().skip(1).collect();
    if names.is_empty() {
        names = vec!["en_US-fr_FR".to_string()];
    }
    for name in names {
        let pair = LangPair::parse(&name)?;
        let Some(home) = assets.model_home(&pair) else {
            continue;
        };
        let (_, config) = assets.best_config(&pair)?;
        let params =
            rlx_translate::pdec::PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;
        let vocab = Vocab::load(home.join("spm.model"))?;
        let two = |l: &str| l.chars().take(2).collect::<String>().to_lowercase();
        let (src_l, tgt_l) = (two(&pair.source), two(&pair.target));
        println!("{name}  (source {src_l}, target {tgt_l})");

        for (label, parts) in [
            ("all target", Parts::uniform(&tgt_l)),
            (
                "input=source",
                Parts {
                    input: src_l.clone(),
                    handover: tgt_l.clone(),
                    decoder: tgt_l.clone(),
                },
            ),
            (
                "input+handover=source",
                Parts {
                    input: src_l.clone(),
                    handover: src_l.clone(),
                    decoder: tgt_l.clone(),
                },
            ),
            (
                "handover=source",
                Parts {
                    input: tgt_l.clone(),
                    handover: src_l.clone(),
                    decoder: tgt_l.clone(),
                },
            ),
        ] {
            let nmt = match Nmt::load_parts(
                &home,
                &[home.as_path()],
                &parts,
                &params.shortlist.lang_pair,
            ) {
                Ok(n) => n,
                Err(e) => {
                    println!("  {label:<22} load failed: {e:#}");
                    continue;
                }
            };
            match nmt.translate_nbest(&vocab, &params, &text, 1) {
                Ok(v) if !v.is_empty() => println!("  {label:<22} {}", v[0].text),
                Ok(_) => println!("  {label:<22} (no hypothesis)"),
                Err(e) => println!("  {label:<22} error: {e:#}"),
            }
        }
    }
    Ok(())
}
