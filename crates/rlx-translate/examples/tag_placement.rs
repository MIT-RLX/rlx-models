//! Where do the direction tags belong in the source sequence?
//!
//! `en_US-tr_TR` degenerates: with the shortlist off it reproduces the English
//! source verbatim, as though the target language were English, while
//! `en_US-hi_IN` and `en_US-ar_AE` — same bundle shape, same manifest flags,
//! same tag inventory — translate fluently. The tags resolve to real ids, so
//! the remaining variable is *where* they sit relative to the sentence.
//!
//! Greedy, no shortlist, so what comes out is the model's own preference rather
//! than the candidate table's.

use anyhow::Result;
use rlx_translate::assets::Assets;
use rlx_translate::decode::{DecoderState, Nmt};
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

fn greedy(nmt: &Nmt, vocab: &Vocab, src: &[u32], bos: u32, limit: usize) -> Result<String> {
    let ho = nmt.encode(src)?;
    let cands: Vec<u32> = (0..vocab.len() as u32).collect();
    let mut state = DecoderState::new(nmt.layers(), nmt.width);
    let mut prev = bos;
    let mut out: Vec<u32> = Vec::new();
    for i in 0..limit {
        let (h, next) = nmt.step(prev, i + 1, &state, &ho)?;
        state = next;
        let scores = nmt.logits(&h, &cands)?;
        let best = scores
            .iter()
            .enumerate()
            .filter(|(j, _)| {
                let p = vocab.piece(*j as u32).unwrap_or("");
                *j as u32 == bos || !(p.starts_with('<') && p.ends_with('>'))
            })
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j as u32)
            .unwrap_or(bos);
        if best == bos {
            break;
        }
        out.push(best);
        prev = best;
    }
    Ok(out
        .iter()
        .map(|t| vocab.piece(*t).unwrap_or(""))
        .collect::<String>()
        .replace('\u{2581}', " ")
        .trim()
        .to_string())
}

fn main() -> Result<()> {
    let assets = Assets::discover();
    let text = "we walked along the river until it started raining";
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
        let tgt: String = pair
            .target
            .chars()
            .take(2)
            .collect::<String>()
            .to_lowercase();
        let nmt = Nmt::load(&home, &[home.as_path()], &tgt)?;
        let bos = vocab.id("<s>").expect("bos");
        let id = |p: &String| vocab.id(p).expect("control token");
        let s: Vec<u32> = params.source_token_pieces().iter().map(id).collect();
        let t: Vec<u32> = params.target_token_pieces().iter().map(id).collect();
        let body = vocab.encode(text);

        println!("{name}");
        for (label, src) in [
            (
                "src,tar + text",
                [s.clone(), t.clone(), body.clone(), vec![bos]].concat(),
            ),
            (
                "tar,src + text",
                [t.clone(), s.clone(), body.clone(), vec![bos]].concat(),
            ),
            (
                "text + src,tar",
                [body.clone(), s.clone(), t.clone(), vec![bos]].concat(),
            ),
            (
                "src + text + tar",
                [s.clone(), body.clone(), t.clone(), vec![bos]].concat(),
            ),
            (
                "tar only + text",
                [t.clone(), body.clone(), vec![bos]].concat(),
            ),
            ("no tags", [body.clone(), vec![bos]].concat()),
            // The bundle carries en_GB tags as well as en_US ones. If the model
            // was trained with the other variant, the configured tag would be a
            // real id that the model has never usefully seen.
            // Maybe the domain tag is not a prefix at all.
            (
                "src,tar + text + opt",
                if t.len() > 1 {
                    [
                        s.clone(),
                        t[..1].to_vec(),
                        body.clone(),
                        t[1..].to_vec(),
                        vec![bos],
                    ]
                    .concat()
                } else {
                    Vec::new()
                },
            ),
            (
                "src + opt,tar + text",
                if t.len() > 1 {
                    [
                        s.clone(),
                        t[1..].to_vec(),
                        t[..1].to_vec(),
                        body.clone(),
                        vec![bos],
                    ]
                    .concat()
                } else {
                    Vec::new()
                },
            ),
            // The target side is two tags — `<tar-xx>` and `<...-optimal>`.
            // Which of the three the model actually wants is worth isolating.
            (
                "src + tar (no opt)",
                [s.clone(), t[..1].to_vec(), body.clone(), vec![bos]].concat(),
            ),
            (
                "src + opt (no tar)",
                if t.len() > 1 {
                    [s.clone(), t[1..].to_vec(), body.clone(), vec![bos]].concat()
                } else {
                    Vec::new()
                },
            ),
            (
                "tar only (no opt)",
                [t[..1].to_vec(), body.clone(), vec![bos]].concat(),
            ),
            // `AddSrcEos=true`, and this port terminates the source with `<s>`
            // because that is what the French-family bundle wants. `</s>` is
            // the other candidate and is otherwise unused.
            (
                "src,tar + text </s>",
                match vocab.id("</s>") {
                    Some(e) => [s.clone(), t.clone(), body.clone(), vec![e]].concat(),
                    None => Vec::new(),
                },
            ),
            (
                "text only + </s>",
                match vocab.id("</s>") {
                    Some(e) => [body.clone(), vec![e]].concat(),
                    None => Vec::new(),
                },
            ),
            (
                "en_GB src + tar",
                match vocab.id("<src-en_GB>") {
                    Some(gb) => [vec![gb], t.clone(), body.clone(), vec![bos]].concat(),
                    None => Vec::new(),
                },
            ),
        ] {
            if src.is_empty() {
                continue;
            }
            match greedy(&nmt, &vocab, &src, bos, 40) {
                Ok(o) => println!("  {label:<18} {o}"),
                Err(e) => println!("  {label:<18} error: {e:#}"),
            }
        }
    }
    Ok(())
}
