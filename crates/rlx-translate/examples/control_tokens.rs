//! What control tokens a direction resolves to, and whether the vocabulary has
//! them.
//!
//! `en_US-zh_TW` and `en_US-zh_CN` share one model and differ only in these
//! tokens, so when the two produce the same Simplified output the question is
//! whether the Traditional tag is reaching the encoder at all.

use anyhow::Result;
use rlx_translate::assets::Assets;
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

fn main() -> Result<()> {
    let assets = Assets::discover();
    // Default to a direction every installation has, so running with no
    // arguments shows something rather than nothing.
    let mut names: Vec<String> = std::env::args().skip(1).collect();
    if names.is_empty() {
        names = vec!["en_US-fr_FR".to_string()];
    }
    for name in names {
        let pair = LangPair::parse(&name)?;
        let Some(home) = assets.model_home(&pair) else {
            println!("{name}: no single-hop model");
            continue;
        };
        let (_, config) = assets.best_config(&pair)?;
        let params =
            rlx_translate::pdec::PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;
        let vocab = Vocab::load(home.join("spm.model"))?;
        println!("{name}  ({} pieces)", vocab.len());
        let nmt = rlx_translate::decode::Nmt::load_with_shortlist(
            &home,
            &[home.as_path()],
            &pair
                .target
                .chars()
                .take(2)
                .collect::<String>()
                .to_lowercase(),
            &params.shortlist.lang_pair,
        )?;
        let table = home
            .join("shortlists")
            .join(format!("{}.shortlist", params.shortlist.lang_pair));
        if table.exists()
            && let Err(e) = rlx_translate::shortlist::Shortlist::load(&table)
        {
            println!("  shortlist load error: {e:#}");
        }
        println!(
            "  shortlist {:<22} file {}  loaded {}",
            params.shortlist.lang_pair,
            if table.exists() { "present" } else { "ABSENT" },
            match nmt.shortlist.as_ref() {
                Some(s) => format!("{} source entries", s.len()),
                None => "NO — falls back to the whole vocabulary".to_string(),
            }
        );
        if let Ok(text) = std::env::var("DUMP_SEGMENT") {
            let ids = vocab.encode(&text);
            let pieces: Vec<&str> = ids.iter().map(|i| vocab.piece(*i).unwrap_or("?")).collect();
            println!("  segment ({} pieces): {}", ids.len(), pieces.join("|"));
        }
        if std::env::var("DUMP_CONTROL").is_ok() {
            let all: Vec<String> = (0..vocab.len() as u32)
                .filter_map(|i| vocab.piece(i).map(str::to_string))
                .filter(|p| p.starts_with('<') && p.ends_with('>'))
                .collect();
            println!("  {} control pieces: {}", all.len(), all.join(" "));
        }
        for (role, pieces) in [
            ("source", params.source_token_pieces()),
            ("target", params.target_token_pieces()),
        ] {
            for p in &pieces {
                match vocab.id(p) {
                    Some(id) => {
                        // A tag that resolves to an id whose piece is something
                        // else would look exactly like a model ignoring it.
                        let back = vocab.piece(id).unwrap_or("<none>");
                        let ok = if back == p { "" } else { "  <-- MISMATCH" };
                        println!("  {role:<6} {p:<28} id {id:<6} piece({id}) = {back}{ok}");
                    }
                    None => println!("  {role:<6} {p:<28} MISSING"),
                }
            }
        }
    }
    Ok(())
}
