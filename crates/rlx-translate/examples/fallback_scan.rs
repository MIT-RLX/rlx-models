//! Which languages fall back to bytes, and how often.
//!
//! Byte fallback is how SentencePiece spells a character the model has no piece
//! for. That is fine in itself; what was not fine is that this port rendered
//! those pieces back to text as the literal string `<0xNN>`, and the stage graph
//! feeds `spm_encode`'s re-rendered text to the translator. Thai lost 0.15 chrF
//! to it.
//!
//! The fix is in, so this answers the follow-up: how much else was affected, and
//! is any language worse hit than Thai?

use anyhow::Result;
use rlx_translate::assets::Assets;
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

fn main() -> Result<()> {
    let dir = std::env::var("RLX_TRANSLATE_FLORES")
        .map_err(|_| anyhow::anyhow!("set RLX_TRANSLATE_FLORES"))?;
    let assets = Assets::discover();
    let mut rows: Vec<(String, f64, f64, usize)> = Vec::new();

    for entry in std::fs::read_dir(&dir)?.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "txt") {
            continue;
        }
        let Some(locale) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // Any direction out of this locale will do: the source side is what
        // gets encoded, and every bundle covering it shares a vocabulary.
        let Some(pair) = ["en_US", "fr_FR", "de_DE"]
            .iter()
            .filter(|t| **t != locale)
            .find_map(|t| {
                let p = LangPair::parse(&format!("{locale}-{t}")).ok()?;
                assets.model_home(&p).map(|h| (p, h))
            })
        else {
            continue;
        };
        let Ok(vocab) = Vocab::load(pair.1.join("spm.model")) else {
            continue;
        };
        let text = std::fs::read_to_string(&path)?;
        let (mut sents, mut hit, mut pieces, mut fb) = (0usize, 0usize, 0usize, 0usize);
        let mut chars: std::collections::BTreeMap<char, usize> = std::collections::BTreeMap::new();
        for line in text.lines().take(200) {
            let ids = vocab.encode(line);
            let n = ids
                .iter()
                .filter(|i| vocab.piece(**i).is_some_and(|p| p.starts_with("<0x")))
                .count();
            sents += 1;
            pieces += ids.len();
            fb += n;
            if n > 0 {
                hit += 1;
                // Which characters, not just how many: a fallback that is
                // really a typographic variant points at a normalizer, not at a
                // vocabulary gap.
                let mut bytes: Vec<u8> = Vec::new();
                for i in &ids {
                    let Some(pc) = vocab.piece(*i) else { continue };
                    if let Some(h) = pc.strip_prefix("<0x").and_then(|r| r.strip_suffix('>'))
                        && let Ok(b) = u8::from_str_radix(h, 16)
                    {
                        bytes.push(b);
                    } else if !bytes.is_empty() {
                        for ch in String::from_utf8_lossy(&bytes).chars() {
                            *chars.entry(ch).or_insert(0usize) += 1;
                        }
                        bytes.clear();
                    }
                }
                for ch in String::from_utf8_lossy(&bytes).chars() {
                    *chars.entry(ch).or_insert(0usize) += 1;
                }
            }
        }
        if sents > 0 {
            let mut top: Vec<(char, usize)> = chars.into_iter().collect();
            top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            let shown: Vec<String> = top
                .iter()
                .take(5)
                .map(|(c, n)| format!("{c:?}x{n}"))
                .collect();
            rows.push((
                locale.to_string(),
                100.0 * hit as f64 / sents as f64,
                100.0 * fb as f64 / pieces.max(1) as f64,
                fb,
            ));
            if !shown.is_empty() {
                eprintln!("  {locale:<7} {}", shown.join("  "));
            }
        }
    }
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "{:<8}{:>12}{:>12}{:>10}",
        "locale", "sents w/ fb", "% pieces", "fb pieces"
    );
    for (l, s, p, n) in &rows {
        if *n == 0 {
            continue;
        }
        println!("{l:<8}{s:>11.1}%{p:>11.2}%{n:>10}");
    }
    let clean: Vec<&str> = rows
        .iter()
        .filter(|r| r.3 == 0)
        .map(|r| r.0.as_str())
        .collect();
    println!("\nno byte fallback at all: {}", clean.join(" "));
    Ok(())
}
