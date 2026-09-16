//! Picks candidate sentences the phrasebook does not cover.
//!
//! The reference dump was built by sampling the phrasebook, so it cannot
//! exercise the NMT. A replacement corpus has to be verified *against* the
//! phrasebook first, or it will have exactly the same problem.

use rlx_translate::assets::Assets;
use rlx_translate::phrasebook::Phrasebook;
use rlx_translate::quasar::LangPair;

fn main() -> anyhow::Result<()> {
    let candidates = [
        "the small grey cat slept on the warm windowsill",
        "she posted the letter before the office closed",
        "we walked along the river until it started raining",
        "his brother repairs bicycles in a shed behind the house",
        "the meeting was moved to Thursday afternoon",
        "there were seventeen empty chairs in the hall",
        "I forgot to bring my umbrella again",
        "the bread had gone stale by the weekend",
        "they argued quietly about the price of the tickets",
        "a lorry blocked the narrow lane for an hour",
        "the museum closes early on public holidays",
        "he learned to swim when he was seven",
        "the soup needs more salt and a little pepper",
        "our neighbours planted cherry trees last spring",
        "the train was delayed because of the storm",
        "she reads the newspaper every morning with coffee",
        "the keys were under a pile of old magazines",
        "nobody remembered to lock the back door",
        "the film lasted almost three hours",
        "we should leave before the traffic gets worse",
    ];
    let assets = Assets::discover();
    let pair = LangPair::parse("en_US-fr_FR")?;
    let (_, config) = assets.best_config(&pair)?;
    let pb = Phrasebook::load_all(&assets.phrasebook_files(&config, &pair))?;
    println!("phrasebook: {} sources", pb.len());
    let mut kept = Vec::new();
    for c in candidates {
        match pb.translate(c) {
            Some(t) => println!("  HIT  {c:?} -> {t:?}"),
            None => kept.push(c),
        }
    }
    println!("{} of {} miss the phrasebook", kept.len(), candidates.len());
    for k in &kept {
        println!("{k}");
    }
    Ok(())
}
