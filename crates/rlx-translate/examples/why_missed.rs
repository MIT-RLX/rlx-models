//! Are the remaining misses search failures or model differences?
//!
//! When our translation differs from the OS's, exactly one of two things is
//! true: either our model scores the OS's string *higher* than the one we
//! emitted — in which case the search failed and a wider beam or a different
//! length penalty would recover it — or it scores it lower, and no amount of
//! searching helps because the disagreement is in the weights or in a pipeline
//! stage this port does not implement.
//!
//! Those two call for completely different work, so it is worth knowing which
//! before doing either. Teacher-force both strings and compare.

use anyhow::Result;
use rlx_translate::assets::Assets;
use rlx_translate::decode::{DecoderState, Nmt};
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

/// Total and per-token log-probability the model assigns to `text` as the
/// translation of `handover`'s source.
fn force(
    nmt: &Nmt,
    vocab: &Vocab,
    ho: &rlx_translate::exec::Env,
    text: &str,
) -> Result<(f64, f64)> {
    let bos = vocab.id("<s>").expect("bos");
    let mut ids = vocab.encode(text);
    ids.push(bos); // the terminator is part of the sequence's cost
    let all: Vec<u32> = (0..vocab.len() as u32).collect();
    let mut state = DecoderState::new(nmt.layers(), nmt.width);
    let mut prev = bos;
    let mut total = 0.0f64;
    for (i, want) in ids.iter().enumerate() {
        let (h, next) = nmt.step(prev, i + 1, &state, ho)?;
        state = next;
        let scores = nmt.logits(&h, &all)?;
        total += f64::from(scores[*want as usize]);
        prev = *want;
    }
    Ok((total, total / ids.len() as f64))
}

fn main() -> Result<()> {
    let dir = std::env::var("RLX_TRANSLATE_REFERENCE")
        .map_err(|_| anyhow::anyhow!("set RLX_TRANSLATE_REFERENCE"))?;
    let assets = Assets::discover();
    let (mut searchable, mut model, mut checked) = (0usize, 0usize, 0usize);
    let (mut unreachable, mut unreachable_tokens) = (0usize, 0usize);

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
        let nmt = Nmt::load_for_pair(
            &home,
            &[home.as_path()],
            &two(&pair.source),
            &two(&pair.target),
            &params.shortlist.lang_pair,
        )?;
        let path = std::path::Path::new(&dir).join(format!("{}-{}.tsv", pair.source, pair.target));
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        println!("{name}");
        for line in text.lines() {
            let Some((src, want)) = line.split_once('\t') else {
                continue;
            };
            let mut got = nmt.translate_nbest(&vocab, &params, src, 1)?;
            let Some(best) = got.drain(..).next() else {
                continue;
            };
            if best.text.trim() == want.trim() {
                continue;
            }
            checked += 1;
            // One encode, reused for both candidates.
            let mut ids: Vec<u32> = Vec::new();
            let bos = vocab.id("<s>").expect("bos");
            for p in params
                .source_token_pieces()
                .iter()
                .chain(params.target_token_pieces().iter())
            {
                ids.push(vocab.id(p).expect("control token"));
            }
            ids.extend(vocab.encode(src));
            ids.push(bos);
            let ho = nmt.encode(&ids)?;
            // Could the search have produced the OS's answer at all? The
            // shortlist restricts the softmax to ~700 candidates, and the
            // config asks for two tables — `cond-n` per source token, which
            // this port reads, and `freq-n`, which it does not. If the OS's
            // tokens are missing from the union, no search or score change
            // reaches them and the missing table is the thing to fix.
            let cands: std::collections::BTreeSet<u32> = nmt
                .candidates(&ids, &[bos], vocab.len())
                .into_iter()
                .collect();
            let missing: Vec<&str> = vocab
                .encode(want)
                .iter()
                .filter(|t| !cands.contains(t))
                .filter_map(|t| vocab.piece(*t))
                .collect();
            if !missing.is_empty() {
                unreachable_tokens += missing.len();
                unreachable += 1;
            }
            let (ours, ours_n) = force(&nmt, &vocab, &ho, &best.text)?;
            let (theirs, theirs_n) = force(&nmt, &vocab, &ho, want)?;
            let verdict = if theirs > ours {
                searchable += 1;
                "SEARCH  "
            } else {
                model += 1;
                "MODEL   "
            };
            println!(
                "  {verdict} ours {ours:>8.2} ({ours_n:>6.3}/tok)  os {theirs:>8.2} ({theirs_n:>6.3}/tok)"
            );
            println!("           ours  {}", best.text);
            println!("           os    {want}");
            if !missing.is_empty() {
                println!("           UNREACHABLE {}", missing.join(" "));
            }
        }
    }
    println!(
        "\n  {checked} misses: {searchable} the model prefers the OS's answer (search), \
         {model} it does not (model or pipeline)"
    );
    println!(
        "  {unreachable} of them contain {unreachable_tokens} token(s) the shortlist \
         cannot emit at all"
    );
    Ok(())
}
