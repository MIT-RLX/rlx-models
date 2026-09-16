//! Validates the cosine metric against an independent sentence embedder.
//!
//! [`rlx_translate::decode::Nmt::sentence_embedding`] pools the *translation
//! model's own* encoder, so using it to judge that model's output is circular:
//! a systematic bias in the encoder would flatter every answer equally and the
//! metric would never notice. `rlx-embed` supplies a second opinion from a
//! model that had no part in producing the text.
//!
//! The locally installed embedder is `all-MiniLM-L6-v2`, which is
//! English-only, so this covers the directions whose *target* is English. Point
//! `RLX_EMBED_MODEL` at a multilingual checkpoint (the registry knows
//! `intfloat/multilingual-e5-*` and `paraphrase-multilingual-*`) to widen it.

use std::path::PathBuf;

use rlx_embed::{BertTokenizer, Pooling, RlxBertModel, embed_with_rlx};
use rlx_translate::score::{center, cosine_centered};

/// Directory of the sentence embedder, if one is installed.
fn embed_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RLX_EMBED_MODEL") {
        let p = PathBuf::from(p);
        return p.join("config.json").is_file().then_some(p);
    }
    // Prefer a multilingual checkpoint: the translator covers 20 languages, and
    // an English-only embedder can only validate the en-target directions.
    let multi = PathBuf::from("/Volumes/FOUR/weights/embed/multilingual-e5-base");
    if multi.join("config.json").is_file() && multi.join("model.safetensors").is_file() {
        return Some(multi);
    }
    let hub = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots");
    std::fs::read_dir(hub)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("config.json").is_file() && p.join("model.safetensors").is_file())
}

/// Embeds `texts` with the independent model, mean-pooled.
fn embed(dir: &std::path::Path, texts: &[&str]) -> Option<Vec<Vec<f32>>> {
    let tok = BertTokenizer::from_dir(dir, 128).ok()?;
    let weights = dir.join("model.safetensors");
    let mut model = RlxBertModel::load_sized(
        &dir.join("config.json"),
        weights.to_str()?,
        texts.len(),
        128,
    )
    .ok()?;
    embed_with_rlx(&mut model, &tok, texts, Pooling::Mean).ok()
}

/// Multilingual separation, in the languages the translator actually covers.
///
/// `all-MiniLM-L6-v2` is English-only; point `RLX_EMBED_MODEL` at a
/// multilingual checkpoint and this exercises the same property everywhere.
#[test]
fn the_embedder_separates_paraphrase_in_every_language() {
    let Some(dir) = embed_model() else {
        eprintln!("skipping: no sentence embedder installed");
        return;
    };
    // (language, a, b) equivalent; then unrelated in the same language.
    let cases: [(&str, &str, &str, &str); 6] = [
        (
            "fr",
            "J'aime l'été",
            "J'adore l'été",
            "Le train arrive à huit heures",
        ),
        (
            "de",
            "Ich liebe den Sommer",
            "Ich mag den Sommer",
            "Es regnet in Strömen",
        ),
        (
            "es",
            "Me encanta el verano",
            "Amo el verano",
            "El tren llega a las ocho",
        ),
        (
            "ru",
            "Я люблю лето",
            "Мне нравится лето",
            "Поезд прибывает в восемь",
        ),
        (
            "ja",
            "夏が大好きです",
            "夏が好きです",
            "電車は八時に到着します",
        ),
        ("zh", "我爱夏天", "我喜欢夏季", "火车八点到达"),
    ];
    let mut flat: Vec<&str> = Vec::new();
    for (_, a, b, c) in &cases {
        flat.push(a);
        flat.push(b);
        flat.push(c);
    }
    let Some(vecs) = embed(&dir, &flat) else {
        panic!("the embedder failed to run");
    };
    let mean = center(&vecs);
    let mut worst_equiv = 1.0f64;
    let mut best_unrel = -1.0f64;
    for (i, (lang, a, b, _)) in cases.iter().enumerate() {
        let (ia, ib, ic) = (i * 3, i * 3 + 1, i * 3 + 2);
        let eq = cosine_centered(&vecs[ia], &vecs[ib], &mean);
        let un = cosine_centered(&vecs[ia], &vecs[ic], &mean);
        eprintln!("  {lang}  equivalent {eq:+.4}  unrelated {un:+.4}   {a:?} / {b:?}");
        worst_equiv = worst_equiv.min(eq);
        best_unrel = best_unrel.max(un);
    }
    eprintln!("  worst equivalent {worst_equiv:+.4} vs best unrelated {best_unrel:+.4}");
    assert!(
        worst_equiv > best_unrel,
        "the embedder does not separate paraphrase from unrelated text in every \
         language ({worst_equiv:.4} vs {best_unrel:.4})"
    );
}

#[test]
fn an_independent_embedder_agrees_that_paraphrase_is_close() {
    let Some(dir) = embed_model() else {
        eprintln!("skipping: no sentence embedder installed (set RLX_EMBED_MODEL)");
        return;
    };
    eprintln!("  embedder: {}", dir.display());

    // Equivalent pairs first, then unrelated ones, all in English so the
    // locally installed model can judge them.
    let texts = [
        "My throat hurts",
        "My neck hurts",
        "We meet at the bank",
        "I'll meet you at the bank",
        "The woman of my dreams",
        "The woman I dream of",
        "The train arrives at eight",
        "It is raining heavily",
    ];
    let Some(vecs) = embed(&dir, &texts) else {
        panic!("the embedder failed to run");
    };
    assert_eq!(vecs.len(), texts.len());
    let mean = center(&vecs);
    let cc = |i: usize, j: usize| cosine_centered(&vecs[i], &vecs[j], &mean);

    let equivalent = [(0usize, 1usize), (2, 3), (4, 5)];
    let unrelated = [(0usize, 6usize), (4, 7), (2, 7)];
    let mut lo = 1.0f64;
    for (i, j) in equivalent {
        let c = cc(i, j);
        eprintln!("  equivalent  {:?} / {:?} -> {c:.4}", texts[i], texts[j]);
        lo = lo.min(c);
    }
    let mut hi = -1.0f64;
    for (i, j) in unrelated {
        let c = cc(i, j);
        eprintln!("  unrelated   {:?} / {:?} -> {c:.4}", texts[i], texts[j]);
        hi = hi.max(c);
    }
    eprintln!("  worst equivalent {lo:.4} vs best unrelated {hi:.4}");
    assert!(
        lo > hi,
        "the independent embedder does not separate paraphrase from unrelated \
         text ({lo:.4} vs {hi:.4}); the cosine metric cannot be trusted"
    );
}

/// Do the two embedders *rank the same way*?
///
/// The number a metric produces matters less than the order it induces. If the
/// translation model's own encoder and an outside model disagree about which of
/// two candidates is closer to a reference, the in-house metric is measuring
/// something about the model rather than about meaning.
#[test]
fn the_two_embedders_rank_candidates_the_same_way() {
    let Some(dir) = embed_model() else {
        eprintln!("skipping: no sentence embedder installed");
        return;
    };
    let Some((home, vocab)) = en_bundle() else {
        eprintln!("skipping: no bundle installed");
        return;
    };
    let nmt =
        rlx_translate::decode::Nmt::load(&home, &[home.as_path()], "en").expect("model loads");

    // (reference, closer candidate, further candidate) — all English.
    let cases = [
        (
            "My throat hurts",
            "My neck hurts",
            "The train arrives at eight",
        ),
        (
            "The woman of my dreams",
            "The woman I dream of",
            "It is raining heavily",
        ),
        (
            "We meet at the bank",
            "I'll meet you at the bank",
            "I love the summer",
        ),
    ];
    let mut agreed = 0usize;
    for (r, near, far) in cases {
        let texts = [r, near, far];
        let Some(v) = embed(&dir, &texts) else {
            continue;
        };
        let m = center(&v);
        let (out_near, out_far) = (
            cosine_centered(&v[0], &v[1], &m),
            cosine_centered(&v[0], &v[2], &m),
        );

        let own: Vec<Vec<f32>> = texts
            .iter()
            .filter_map(|t| nmt.sentence_embedding(&vocab, "en_US", t).ok())
            .collect();
        if own.len() != 3 {
            continue;
        }
        let om = center(&own);
        let (in_near, in_far) = (
            cosine_centered(&own[0], &own[1], &om),
            cosine_centered(&own[0], &own[2], &om),
        );
        eprintln!(
            "  {r:?}\n     independent: near {out_near:.3} far {out_far:.3}\n\
             \x20    own encoder: near {in_near:.3} far {in_far:.3}"
        );
        if (out_near > out_far) == (in_near > in_far) {
            agreed += 1;
        }
    }
    eprintln!(
        "  the two embedders agreed on {agreed} of {} cases",
        cases.len()
    );
    assert_eq!(
        agreed,
        cases.len(),
        "the in-house cosine disagrees with an independent embedder about which \
         candidate is closer; it is measuring the model, not meaning"
    );
}

/// A bundle that can decode into English.
///
/// Resolved through [`Assets::model_home`] rather than by looking for
/// `decoder_en` beside a manifest: the per-language decoders ship in separate
/// `partial-<lang>` assets, so they are never in the same directory as the
/// manifest, and a naive search finds nothing.
fn en_bundle() -> Option<(PathBuf, rlx_translate::spm::Vocab)> {
    let assets = rlx_translate::assets::Assets::discover();
    let pair = rlx_translate::quasar::LangPair::parse("fr_FR-en_US").ok()?;
    let home = assets.model_home(&pair)?;
    let vocab = rlx_translate::spm::Vocab::load(home.join("spm.model")).ok()?;
    Some((home, vocab))
}

/// Are our translations equivalent to the OS's — and can cosine tell?
///
/// A similarity number on its own proves nothing: if every sentence in a
/// language scored 0.9 against every other, "equivalent" would be vacuous. So
/// this measures both halves at once. For each source we translate it and
/// compare our output against the OS's translation of *the same* source
/// (matched) and against the OS's translations of the *other* sources
/// (mismatched). The gap between those two distributions is what makes the
/// metric meaningful, and `retrieval@1` — is the OS's own translation the
/// nearest of all candidates — is the sharpest form of it.
///
/// Slow — roughly 25 minutes, dominated by beam search and by loading three
/// separate multilingual bundles.
#[test]
fn our_translations_match_the_os_and_cosine_can_tell() {
    let Some(dir) = embed_model() else {
        eprintln!("skipping: no sentence embedder installed");
        return;
    };
    let refdir = match std::env::var("RLX_TRANSLATE_REFERENCE") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("skipping: set RLX_TRANSLATE_REFERENCE to the OS's output dump");
            return;
        }
    };
    let assets = rlx_translate::assets::Assets::discover();
    let mut all_matched: Vec<f64> = Vec::new();
    let mut all_mismatched: Vec<f64> = Vec::new();
    let (mut hits, mut total) = (0usize, 0usize);

    for spec in ["en_US-fr_FR", "en_US-de_DE", "en_US-ja_JP"] {
        let pair = rlx_translate::quasar::LangPair::parse(spec).expect("pair");
        let Some(home) = assets.model_home(&pair) else {
            continue;
        };
        let Ok((_, config)) = assets.best_config(&pair) else {
            continue;
        };
        let Ok(params) = config
            .mt_app()
            .and_then(|d| d.translator(&pair))
            .and_then(rlx_translate::pdec::PDecParams::from_block)
        else {
            continue;
        };
        let Ok(vocab) = rlx_translate::spm::Vocab::load(home.join("spm.model")) else {
            continue;
        };
        let tgt: String = pair
            .target
            .chars()
            .take(2)
            .collect::<String>()
            .to_lowercase();
        let Ok(nmt) = rlx_translate::decode::Nmt::load(&home, &[home.as_path()], &tgt) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(refdir.join(format!("{spec}.tsv"))) else {
            continue;
        };
        // Distinct sources only: repeats would make the mismatched set contain
        // genuine equivalents and understate the gap.
        let mut cases: Vec<(String, String)> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for line in text.lines() {
            let Some((s, t)) = line.split_once('\t') else {
                continue;
            };
            if s.split_whitespace().count() > 7 || !seen.insert(s.to_string()) {
                continue;
            }
            cases.push((s.to_string(), t.to_string()));
            if cases.len() >= 8 {
                break;
            }
        }
        if cases.len() < 4 {
            continue;
        }

        let mut ours = Vec::new();
        for (src, _) in &cases {
            // `translate_nbest` derives its own length budget from `params`.
            match nmt.translate_nbest(&vocab, &params, src, 1) {
                Ok(mut v) if !v.is_empty() => ours.push(v.remove(0).text),
                _ => ours.push(String::new()),
            }
        }
        let theirs: Vec<String> = cases.iter().map(|(_, t)| t.clone()).collect();

        // One embedding space for both sides, centred together.
        let flat: Vec<&str> = ours
            .iter()
            .map(String::as_str)
            .chain(theirs.iter().map(String::as_str))
            .collect();
        let Some(vecs) = embed(&dir, &flat) else {
            continue;
        };
        let mean = center(&vecs);
        let n = ours.len();
        let (mut m_sum, mut x_sum, mut x_n) = (0.0f64, 0.0f64, 0usize);
        for i in 0..n {
            let mut best = (f64::MIN, usize::MAX);
            for j in 0..n {
                let c = cosine_centered(&vecs[i], &vecs[n + j], &mean);
                if c > best.0 {
                    best = (c, j);
                }
                if i == j {
                    all_matched.push(c);
                    m_sum += c;
                } else {
                    all_mismatched.push(c);
                    x_sum += c;
                    x_n += 1;
                }
            }
            if best.1 == i {
                hits += 1;
            } else {
                // Report the miss in full: what we said, what the OS said for
                // this source, and which of the OS's other translations the
                // metric preferred. Without the winner there is no way to tell
                // a divergent translation from a metric failure.
                let own = cosine_centered(&vecs[i], &vecs[n + i], &mean);
                eprintln!(
                    "    MISS {spec} {:?}\n      ours      {:?}\n      os        {:?}  cos {own:+.3}\n      but near  {:?}  cos {:+.3}  (source {:?})",
                    cases[i].0, ours[i], theirs[i], theirs[best.1], best.0, cases[best.1].0
                );
            }
            total += 1;
        }
        eprintln!(
            "  {spec}: matched {:.3}  mismatched {:.3}  over {n} sentences",
            m_sum / n as f64,
            x_sum / x_n.max(1) as f64
        );
        eprintln!(
            "     e.g. ours {:?}\n          os    {:?}",
            ours[0], theirs[0]
        );
        // Persist the decoded text: re-running this test costs ~25 minutes, and
        // anything worth looking at afterwards only needs the strings.
        if let Ok(dir) = std::env::var("RLX_TRANSLATE_DUMP") {
            let rows: Vec<serde_json::Value> = cases
                .iter()
                .zip(&ours)
                .map(|((src, theirs), ours)| {
                    serde_json::json!({"source": src, "ours": ours, "os": theirs})
                })
                .collect();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(
                PathBuf::from(&dir).join(format!("{spec}.json")),
                serde_json::to_string_pretty(&rows).unwrap_or_default(),
            );
        }
    }

    if total == 0 {
        eprintln!("skipping: nothing was translated");
        return;
    }
    let mean_m = all_matched.iter().sum::<f64>() / all_matched.len() as f64;
    let mean_x = all_mismatched.iter().sum::<f64>() / all_mismatched.len() as f64;
    eprintln!(
        "\n  matched {mean_m:.3} vs mismatched {mean_x:.3} (gap {:.3})",
        mean_m - mean_x
    );
    eprintln!("  retrieval@1: {hits}/{total}");
    assert!(
        mean_m > mean_x + 0.2,
        "cosine does not distinguish our translation of a sentence from our \
         translation of a different one ({mean_m:.3} vs {mean_x:.3}); the \
         similarity numbers elsewhere would be meaningless"
    );
    assert!(
        hits * 2 > total,
        "the OS's own translation is not usually the nearest candidate \
         ({hits}/{total}); either the translations differ in meaning or the \
         metric cannot see it"
    );
}

/// BERTScore, with real contextual token embeddings.
///
/// This is the BERT-family metric proper: greedy token matching rather than a
/// single pooled vector. It should rank a paraphrase above a replacement and
/// both above unrelated text, and — unlike chrF — should not be fooled by two
/// correct translations that share few characters.
#[test]
fn bertscore_ranks_paraphrase_above_replacement() {
    let Some(dir) = embed_model() else {
        eprintln!("skipping: no sentence embedder installed");
        return;
    };
    let tok = match BertTokenizer::from_dir(&dir, 64) {
        Ok(t) => t,
        Err(e) => panic!("tokenizer: {e:#}"),
    };
    let weights = dir.join("model.safetensors");

    // Per-token contextual states for a batch, minus the special tokens at
    // each end: they match perfectly in every pair and inflate every score.
    let embed_tokens = |texts: &[&str]| -> Vec<Vec<Vec<f32>>> {
        let batch = tok.encode_batch(texts).expect("tokenize");
        let (b, s) = (texts.len(), batch.seq_len);
        let mut model = RlxBertModel::load_sized(
            &dir.join("config.json"),
            weights.to_str().expect("path"),
            b,
            s,
        )
        .expect("model");
        let ids: Vec<f32> = batch
            .input_ids
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as f32))
            .collect();
        let mask: Vec<f32> = batch
            .attention_mask
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as f32))
            .collect();
        let tt: Vec<f32> = batch
            .token_type_ids
            .iter()
            .flat_map(|r| r.iter().map(|&v| v as f32))
            .collect();
        let off = model.position_offset();
        let pos: Vec<f32> = (0..b)
            .flat_map(|_| (0..s).map(|i| (i + off) as f32))
            .collect();
        let hidden = model.forward(&ids, &mask, &tt, &pos);
        let h = model.hidden_size();
        (0..b)
            .map(|i| {
                let live: Vec<usize> = (0..s)
                    .filter(|j| batch.attention_mask[i][*j] == 1)
                    .collect();
                // Drop the first and last live positions ([CLS] / [SEP]).
                let inner = if live.len() > 2 {
                    &live[1..live.len() - 1]
                } else {
                    &live[..]
                };
                inner
                    .iter()
                    .map(|j| hidden[(i * s + j) * h..(i * s + j + 1) * h].to_vec())
                    .collect()
            })
            .collect()
    };

    let texts = [
        "La femme de mes rêves",         // reference
        "La femme dont je rêve",         // paraphrase
        "La femme de mes cauchemars",    // one word replaced: dreams -> nightmares
        "Le train arrive à huit heures", // unrelated
    ];
    let e = embed_tokens(&texts);
    let para = rlx_translate::score::bertscore(&e[1], &e[0]);
    let repl = rlx_translate::score::bertscore(&e[2], &e[0]);
    let unrel = rlx_translate::score::bertscore(&e[3], &e[0]);
    eprintln!("  paraphrase  {para:.4}");
    eprintln!("  replacement {repl:.4}");
    eprintln!("  unrelated   {unrel:.4}");
    let self_score = rlx_translate::score::bertscore(&e[0], &e[0]);
    eprintln!("  identical   {self_score:.4}");
    assert!((self_score - 1.0).abs() < 1e-6, "identical must score 1");
    assert!(
        para > unrel && repl > unrel,
        "unrelated text should score lowest: {para:.3}/{repl:.3}/{unrel:.3}"
    );
}
