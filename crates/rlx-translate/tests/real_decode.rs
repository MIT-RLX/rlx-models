// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Runs the decoder on the real installed weights: encode, step, score.

use rlx_translate::assets::Assets;
use rlx_translate::decode::{DecoderState, Nmt};
use rlx_translate::espresso::Manifest;
use rlx_translate::spm::Vocab;
use std::path::PathBuf;

/// The bundle covering `lang`, plus every other MT dir for per-language graphs.
fn bundle(lang: &str) -> Option<(PathBuf, Vec<PathBuf>, Vocab)> {
    let dirs: Vec<PathBuf> = Assets::discover()
        .roots
        .iter()
        .map(|r| r.join("MT"))
        .filter(|p| p.is_dir())
        .collect();
    let home = dirs.iter().find(|d| {
        d.join("pyespresso.mdl.bin").exists()
            && Manifest::load(d.join("pyespresso.mdl.bin"))
                .map(|m| m.languages().contains(&lang))
                .unwrap_or(false)
    })?;
    let vocab = Vocab::load(home.join("spm.model")).ok()?;
    Some((home.clone(), dirs, vocab))
}

/// Source ids: the source-language control token, the words, then EOS.
///
/// The config's `source-token: "en_US"` is the inside of `<src-en_US>` (id 718
/// in the shipped vocabulary); there is no bare `<en_US>`. `AddSrcEos` is set,
/// so the source is terminated.
fn ids_for(v: &Vocab, src_locale: &str, words: &[&str]) -> Vec<u32> {
    ids_for_placement(v, src_locale, words, TagAt::Start)
}

/// Where the source control token goes. The manifest says `AddTag: end` with
/// `TagFormat: bothSeparate`, which argues for appending, but "end" could also
/// describe the tag pair rather than its position — so both are measured.
#[derive(Clone, Copy, PartialEq)]
enum TagAt {
    Start,
    BeforeEos,
    AfterEos,
    None,
}

fn ids_for_placement(v: &Vocab, src_locale: &str, words: &[&str], at: TagAt) -> Vec<u32> {
    let tag = v.id(&format!("<src-{src_locale}>"));
    let eos = v.id("</s>");
    let mut out = Vec::new();
    if at == TagAt::Start
        && let Some(t) = tag
    {
        out.push(t);
    }
    out.extend(words.iter().filter_map(|w| v.word_id(w)));
    if at == TagAt::BeforeEos
        && let Some(t) = tag
    {
        out.push(t);
    }
    if let Some(e) = eos {
        out.push(e);
    }
    if at == TagAt::AfterEos
        && let Some(t) = tag
    {
        out.push(t);
    }
    out
}

/// The token the decoder is primed with: the pair/variant tag from
/// `target-token`, wrapped in angle brackets (`<en_US-fr_FR-optimal>` = 252).
/// The leading `fr_FR` of that field selects the decoder *graph*, not a token —
/// no bare `<fr_FR>` exists in the vocabulary.
fn prime_id(v: &Vocab, src_locale: &str, tgt_locale: &str) -> Option<u32> {
    v.id(&format!("<{src_locale}-{tgt_locale}-optimal>"))
}

#[test]
fn encode_produces_every_cross_attention_tensor() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    eprintln!(
        "loaded fr: source_len {} width {} layers {}",
        nmt.source_len,
        nmt.width,
        nmt.layers()
    );

    let src = ids_for(&vocab, "en_US", &["i", "love", "the", "summer"]);
    assert!(src.len() >= 3, "source ids did not resolve: {src:?}");
    let ho = nmt.encode(&src).expect("encode runs");
    for name in nmt.manifest.csv("HandoverStrings") {
        let t = ho[&name].f32().expect("handover tensor");
        assert!(
            t.data().iter().all(|v| v.is_finite()),
            "{name} has non-finite values"
        );
    }
    eprintln!(
        "encoded {} source ids -> {} handover tensors",
        src.len(),
        ho.len()
    );
}

#[test]
fn a_decoder_step_advances_state_and_scores_candidates() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");

    let src = ids_for(&vocab, "en_US", &["i", "love", "the", "summer"]);
    let ho = nmt.encode(&src).expect("encode runs");

    let bos = vocab
        .id("</s>")
        .or_else(|| vocab.id("<s>"))
        .expect("a start token");
    let state = DecoderState::new(nmt.layers(), nmt.width);
    let (hidden, next) = nmt.step(bos, 1, &state, &ho).expect("decoder step");

    assert_eq!(hidden.width(), nmt.width, "hidden state width");
    assert!(
        hidden.data().iter().all(|v| v.is_finite()),
        "hidden state has non-finite values"
    );
    assert_eq!(next.accum.len(), nmt.layers());
    // The accumulator must actually accumulate — a zero next-state would mean
    // the average-attention carry is not being written back.
    let moved = next
        .accum
        .iter()
        .any(|t| t.data().iter().any(|v| v.abs() > 1e-6));
    assert!(moved, "decoder state did not advance from zero");

    // Score a candidate set through the tied readout table.
    let candidates: Vec<u32> = (0..64).map(|i| (i * 977 + 1000) as u32).collect();
    let scores = nmt.logits(&hidden, &candidates).expect("logits");
    assert_eq!(scores.len(), candidates.len());
    assert!(scores.iter().all(|v| v.is_finite()), "non-finite log-probs");
    assert!(scores.iter().all(|v| *v <= 1e-4), "log-probs must be <= 0");
    let total: f32 = scores.iter().map(|v| v.exp()).sum();
    assert!(
        (total - 1.0).abs() < 1e-3,
        "candidate probabilities sum to {total}, not 1"
    );
    eprintln!(
        "step 1: hidden {:?}, state advanced, {} candidates scored (sum {total:.4})",
        hidden.dims(),
        scores.len()
    );
}

#[test]
fn greedy_decoding_over_the_full_vocabulary_emits_plausible_tokens() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let mut nmt = Nmt::load(&home, &extra, "fr").expect("model loads");

    let src = ids_for(&vocab, "en_US", &["i", "love", "the", "summer"]);
    let eos = vocab.id("</s>").expect("eos");

    // Score against the whole vocabulary: the shortlist is a speed and quality
    // restriction, and the unrestricted argmax is the honest first check.
    let all: Vec<u32> = (0..vocab.len() as u32).collect();
    let prime = prime_id(&vocab, "en_US", "fr_FR").expect("pair control token");

    // The source is encoded at its true length, so there is no padding token
    // for the maskless cross-attention to attend to.
    let mut any = false;
    for name in ["unpadded"] {
        let with_pos = true;
        nmt.set_decoder_uses_positions(with_pos);
        let _ = name;
        let ho = nmt.encode(&src).expect("encode runs");
        let mut state = DecoderState::new(nmt.layers(), nmt.width);
        let mut prev = prime;
        let mut pieces = Vec::new();
        for pos in 1..=8usize {
            let (hidden, next) = nmt.step(prev, pos, &state, &ho).expect("decoder step");
            let scores = nmt.logits(&hidden, &all).expect("logits");
            let (best, _) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("a best candidate");
            let id = all[best];
            state = next;
            prev = id;
            if id == eos {
                break;
            }
            pieces.push(vocab.piece(id).unwrap_or("?").to_string());
        }
        eprintln!(
            "greedy fr (decoder positions={with_pos}): {:?}",
            pieces.join("")
        );
        any |= !pieces.is_empty();
    }
    assert!(
        any,
        "decoder emitted nothing before EOS under either reading"
    );
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Localizes the degenerate-output fault: is the encoder discriminating between
/// sentences at all, and does the decoder's hidden state move between steps?
///
/// If two different sources produce near-identical cross-attention tensors, the
/// fault is upstream of the decoder. If they differ but the decoder's hidden
/// state is constant across steps, the fault is the recurrence.
#[test]
fn diagnose_where_the_signal_is_lost() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");

    let a = ids_for(&vocab, "en_US", &["i", "love", "the", "summer"]);
    let b = ids_for(&vocab, "en_US", &["the", "dog", "is", "black"]);
    let ha = nmt.encode(&a).expect("encode a");
    let hb = nmt.encode(&b).expect("encode b");
    let ha2 = nmt.encode(&a).expect("encode a again");

    let key = nmt.manifest.csv("HandoverStrings")[0].clone();
    let ta = ha[&key].f32().expect("f32");
    let tb = hb[&key].f32().expect("f32");
    let ta2 = ha2[&key].f32().expect("f32");

    let same = cosine(ta.data(), ta2.data());
    let diff = cosine(ta.data(), tb.data());
    eprintln!("encoder determinism  cos(a,a) = {same:.6}");
    eprintln!("encoder discrimination cos(a,b) = {diff:.6}");
    assert!((same - 1.0).abs() < 1e-5, "encoding is not deterministic");

    // Decoder hidden state across steps, on one source.
    let prime = prime_id(&vocab, "en_US", "fr_FR").expect("control token");
    let mut state = DecoderState::new(nmt.layers(), nmt.width);
    let mut prev = prime;
    let mut hs: Vec<Vec<f32>> = Vec::new();
    for pos in 1..=4usize {
        let (h, next) = nmt.step(prev, pos, &state, &ha).expect("step");
        hs.push(h.data().to_vec());
        state = next;
        // Feed a fixed token so only position and state vary.
        prev = vocab.id("</s>").expect("eos");
    }
    for i in 1..hs.len() {
        eprintln!(
            "decoder hidden cos(step1, step{}) = {:.6}",
            i + 1,
            cosine(&hs[0], &hs[i])
        );
    }
    eprintln!(
        "encoder rows: cos(row0,row40) on source a = {:.6}  (row40 is padding)",
        cosine(ta.row(0), ta.row(40.min(ta.rows() - 1)))
    );
}

/// Corrected discrimination check: compare only the *real* token positions.
///
/// Whole-tensor comparison is confounded because 60 of the 64 positions are
/// identical padding in both sources and dominate the cosine.
#[test]
fn encoder_discriminates_on_real_token_positions() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");

    let a = ids_for(&vocab, "en_US", &["i", "love", "the", "summer"]);
    let b = ids_for(&vocab, "en_US", &["the", "dog", "is", "black"]);
    let ea = nmt.encoder_states(&a).expect("encode a");
    let eb = nmt.encoder_states(&b).expect("encode b");

    let n = a.len().min(b.len());
    let mut real = Vec::new();
    for i in 0..n {
        real.push(cosine(ea.row(i), eb.row(i)));
    }
    let pad: Vec<f32> = (n + 2..(n + 8).min(ea.rows()))
        .map(|i| cosine(ea.row(i), eb.row(i)))
        .collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    eprintln!(
        "real token positions  cos(a,b) mean {:.4}  {:?}",
        mean(&real),
        real.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
    );
    eprintln!("padded positions      cos(a,b) mean {:.4}", mean(&pad));
    // Within one sentence, distinct tokens should give distinct states.
    let within = cosine(ea.row(1), ea.row(2));
    eprintln!("within sentence a     cos(pos1,pos2) {within:.4}");

    assert!(
        mean(&real) < 0.999,
        "the encoder gives identical states to different sentences"
    );
}

/// Walks the source path stage by stage, reporting how much two *different*
/// token positions still differ. Whichever stage drives the cosine to ~1 is
/// where the token signal dies.
#[test]
fn find_the_stage_that_collapses_the_signal() {
    use rlx_translate::exec::{Env, Value, run};
    use rlx_translate::net::Graph;
    use rlx_translate::tensor::Tensor;

    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let find = |net: &str| -> Graph {
        let d = std::iter::once(&home)
            .chain(dirs.iter())
            .find(|d| d.join(net).exists())
            .expect("graph installed");
        Graph::load(d, net).expect("loads")
    };

    // Four distinct, unrelated words in the first four positions.
    let words = ["dog", "summer", "water", "king"];
    let mut ids: Vec<u32> = words.iter().filter_map(|w| vocab.word_id(w)).collect();
    assert!(ids.len() == 4, "words did not resolve");
    ids.resize(64, vocab.id("</s>").unwrap_or(2));
    let pos: Vec<f32> = (0..64).map(|i| i as f32).collect();

    let report = |label: &str, t: &Tensor| {
        let c01 = cosine(t.row(0), t.row(1));
        let c02 = cosine(t.row(0), t.row(2));
        let c03 = cosine(t.row(0), t.row(3));
        let mean = (c01 + c02 + c03) / 3.0;
        let norm: f32 = (t.row(0).iter().map(|v| v * v).sum::<f32>() / t.width() as f32).sqrt();
        eprintln!(
            "{label:<22} cos(0,1)={c01:.4} cos(0,2)={c02:.4} cos(0,3)={c03:.4}  mean={mean:.4}  rms(row0)={norm:.3}"
        );
    };

    // Stage 0 — raw table rows, before any scaling or positional add.
    let emb_graph = find("embedding.espresso.net");
    let gather = emb_graph
        .layers
        .iter()
        .filter(|l| l.kind == "quantized_gather")
        .max_by_key(|l| l.int("nRow").unwrap_or(0))
        .expect("vocab gather");
    let cols = gather.int("nCol").expect("nCol") as usize;
    let table = emb_graph
        .weights
        .raw(gather.blob("weights_u8").expect("w"))
        .expect("raw");
    let meta = emb_graph
        .weights
        .f32s(gather.blob("Q_meta").expect("m"))
        .expect("meta");
    let mut raw = Vec::new();
    for id in ids.iter().take(4) {
        let base = *id as usize * cols;
        for c in 0..cols {
            raw.push(rlx_translate::exec::dequant_gather_row(
                table[base + c],
                &meta[c * 4..c * 4 + 4],
            ));
        }
    }
    report("0 raw table", &Tensor::new(vec![4, cols], raw).expect("t"));

    // Stage 1 — embedding graph output (scale + positional add).
    let mut env = Env::new();
    env.insert(
        "src_tokens".into(),
        Value::F32(Tensor::new(vec![64], ids.iter().map(|v| *v as f32).collect()).expect("t")),
    );
    env.insert(
        "positions".into(),
        Value::F32(Tensor::new(vec![64], pos).expect("t")),
    );
    let e1 = run(&emb_graph, env).expect("embedding runs");
    let emb = e1["embedding"].f32().expect("f32").clone();
    report("1 embedding graph", &emb);

    // Stage 2 — input_<lang>.
    let input_net = nmt
        .manifest
        .lang_graphs
        .get("InputLangGraph")
        .and_then(|m| m.get("fr"))
        .expect("input graph");
    let mut env = Env::new();
    env.insert("embedding".into(), Value::F32(emb));
    let e2 = run(&find(input_net), env).expect("input net runs");
    let bridge = nmt
        .manifest
        .str("InputNetValuesStr")
        .unwrap_or("encoder.3.output");
    let mid = e2[bridge].f32().expect("f32").clone();
    report("2 input net", &mid);

    // Stage 3 — the 12 encoder blocks.
    let mut env = Env::new();
    env.insert(bridge.to_string(), Value::F32(mid));
    let e3 = run(
        &find(
            nmt.manifest
                .str("EncoderGraph")
                .unwrap_or("encoder.espresso.net"),
        ),
        env,
    )
    .expect("encoder runs");
    let enc = nmt
        .manifest
        .str("EncoderValuesStr")
        .unwrap_or("encoder.15.output");
    report("3 encoder", e3[enc].f32().expect("f32"));
}

/// Is the attention actually attending, or averaging everything?
///
/// Uniform attention makes every output position the mean of all input
/// positions, which is precisely how the token signal would vanish.
#[test]
fn attention_is_peaked_not_uniform() {
    use rlx_translate::exec::{Env, Value, run};
    use rlx_translate::net::Graph;
    use rlx_translate::tensor::Tensor;

    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let find = |net: &str| -> Graph {
        let d = std::iter::once(&home)
            .chain(dirs.iter())
            .find(|d| d.join(net).exists())
            .expect("installed");
        Graph::load(d, net).expect("loads")
    };

    let words = ["dog", "summer", "water", "king"];
    let mut ids: Vec<u32> = words.iter().filter_map(|w| vocab.word_id(w)).collect();
    ids.resize(64, vocab.id("</s>").unwrap_or(2));
    let pos: Vec<f32> = (0..64).map(|i| i as f32).collect();

    let emb_graph = find("embedding.espresso.net");
    let mut env = Env::new();
    env.insert(
        "src_tokens".into(),
        Value::F32(Tensor::new(vec![64], ids.iter().map(|v| *v as f32).collect()).expect("t")),
    );
    env.insert(
        "positions".into(),
        Value::F32(Tensor::new(vec![64], pos).expect("t")),
    );
    let emb = run(&emb_graph, env).expect("embedding")["embedding"]
        .f32()
        .expect("f32")
        .clone();

    let input_net = nmt
        .manifest
        .lang_graphs
        .get("InputLangGraph")
        .and_then(|m| m.get("fr"))
        .expect("input graph");
    let g = find(input_net);
    let mut env = Env::new();
    env.insert("embedding".into(), Value::F32(emb));
    let out = run(&g, env).expect("input net runs");

    let uniform = 1.0f32 / 64.0;
    let mut reported = 0;
    for l in &g.layers {
        if l.kind != "softmax" {
            continue;
        }
        let w = out[&l.tops[0]].f32().expect("f32");
        let logits = out[&l.bottoms[0]].f32().expect("f32");
        // Row 0 of head 0.
        let probs = w.row(0);
        let max = probs.iter().copied().fold(0.0f32, f32::max);
        let ent: f32 = -probs
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f32>();
        let lrms: f32 =
            (logits.row(0).iter().map(|v| v * v).sum::<f32>() / logits.width() as f32).sqrt();
        eprintln!(
            "{:<18} logit rms {lrms:8.4}  max prob {max:.4} (uniform {uniform:.4})  entropy {ent:.3} (max {:.3})",
            l.name,
            (64.0f32).ln()
        );
        reported += 1;
        if reported >= 4 {
            break;
        }
    }
    assert!(reported > 0, "input net has no softmax");
}

/// Discrimination after every block of the input net, to see whether the
/// collapse is one bad block or a gradual smear.
#[test]
fn locate_the_collapsing_block_inside_the_input_net() {
    use rlx_translate::exec::{Env, Value, run};
    use rlx_translate::net::Graph;
    use rlx_translate::tensor::Tensor;

    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let find = |net: &str| -> Graph {
        let d = std::iter::once(&home)
            .chain(dirs.iter())
            .find(|d| d.join(net).exists())
            .expect("installed");
        Graph::load(d, net).expect("loads")
    };

    let words = ["dog", "summer", "water", "king"];
    let mut ids: Vec<u32> = words.iter().filter_map(|w| vocab.word_id(w)).collect();
    ids.resize(64, vocab.id("</s>").unwrap_or(2));
    let pos: Vec<f32> = (0..64).map(|i| i as f32).collect();

    let mut env = Env::new();
    env.insert(
        "src_tokens".into(),
        Value::F32(Tensor::new(vec![64], ids.iter().map(|v| *v as f32).collect()).expect("t")),
    );
    env.insert(
        "positions".into(),
        Value::F32(Tensor::new(vec![64], pos).expect("t")),
    );
    let emb = run(&find("embedding.espresso.net"), env).expect("embedding")["embedding"]
        .f32()
        .expect("f32")
        .clone();

    let net = nmt
        .manifest
        .lang_graphs
        .get("InputLangGraph")
        .and_then(|m| m.get("fr"))
        .expect("input graph");
    let g = find(net);
    let mut env = Env::new();
    env.insert("embedding".into(), Value::F32(emb));
    let out = run(&g, env).expect("input net runs");

    let disc = |t: &Tensor| {
        let c: f32 = (1..4).map(|i| cosine(t.row(0), t.row(i))).sum::<f32>() / 3.0;
        let rms: f32 = (t.row(0).iter().map(|v| v * v).sum::<f32>() / t.width() as f32).sqrt();
        (c, rms)
    };

    // Every blob that a LayerNorm or an attention produces, in graph order.
    for l in &g.layers {
        if !matches!(l.kind.as_str(), "instancenorm_1d" | "softmax")
            && !l.name.starts_with("batch_matmul")
        {
            continue;
        }
        let Some(Value::F32(t)) = out.get(&l.tops[0]) else {
            continue;
        };
        let (c, rms) = disc(t);
        eprintln!("{:<20} {:<18} cos={c:.4}  rms={rms:.3}", l.name, l.kind);
    }
}

/// `AddTag: end` in the manifest suggests the control tokens are appended, not
/// prepended. Measures every placement by how much the encoder still
/// discriminates between the real source tokens.
#[test]
fn source_tag_placement_changes_encoder_discrimination() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let words = ["dog", "summer", "water", "king"];

    for (label, at) in [
        ("prepend", TagAt::Start),
        ("before eos", TagAt::BeforeEos),
        ("after eos", TagAt::AfterEos),
        ("no tag", TagAt::None),
    ] {
        let ids = ids_for_placement(&vocab, "en_US", &words, at);
        let e = nmt.encoder_states(&ids).expect("encode");
        // Compare the four content words wherever they landed.
        let off = if at == TagAt::Start { 1 } else { 0 };
        let mut c = 0.0f32;
        let mut n = 0;
        for i in 0..3 {
            for j in i + 1..4 {
                c += cosine(e.row(off + i), e.row(off + j));
                n += 1;
            }
        }
        eprintln!(
            "source tag {label:<11} ids={:?}  mean cos between content tokens = {:.4}",
            ids.len(),
            c / n as f32
        );
    }
}

/// For every dequantize in the input net, compare the matmul term against the
/// bias term. If the bias dominates, the layer emits a nearly position-
/// independent vector — which is exactly how a position-wise FFN could raise
/// cross-position similarity.
#[test]
fn matmul_term_is_not_swamped_by_the_bias() {
    use rlx_translate::exec::{Env, Value, run};
    use rlx_translate::net::Graph;
    use rlx_translate::tensor::Tensor;

    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let find = |net: &str| -> Graph {
        let d = std::iter::once(&home)
            .chain(dirs.iter())
            .find(|d| d.join(net).exists())
            .expect("installed");
        Graph::load(d, net).expect("loads")
    };

    let words = ["dog", "summer", "water", "king"];
    let mut ids: Vec<u32> = words.iter().filter_map(|w| vocab.word_id(w)).collect();
    ids.resize(64, vocab.id("</s>").unwrap_or(2));
    let pos: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let mut env = Env::new();
    env.insert(
        "src_tokens".into(),
        Value::F32(Tensor::new(vec![64], ids.iter().map(|v| *v as f32).collect()).expect("t")),
    );
    env.insert(
        "positions".into(),
        Value::F32(Tensor::new(vec![64], pos).expect("t")),
    );
    let emb = run(&find("embedding.espresso.net"), env).expect("emb")["embedding"]
        .f32()
        .expect("f32")
        .clone();

    let net = nmt
        .manifest
        .lang_graphs
        .get("InputLangGraph")
        .and_then(|m| m.get("fr"))
        .expect("input graph");
    let g = find(net);
    let mut env = Env::new();
    env.insert("embedding".into(), Value::F32(emb));
    let out = run(&g, env).expect("input net runs");

    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
    let mut shown = 0;
    for l in &g.layers {
        if l.kind != "dynamic_dequantize" {
            continue;
        }
        let Some(Value::F32(t)) = out.get(&l.tops[0]) else {
            continue;
        };
        let bias = match l.blob("biases") {
            Some(b) => g.weights.f32s(b).expect("bias"),
            None => Vec::new(),
        };
        if bias.is_empty() {
            continue;
        }
        // Output minus bias is the matmul contribution (pre-ReLU it would be
        // exact; with ReLU this is a lower bound, which is enough to see a
        // swamped term).
        let w = t.width();
        let mut mm = Vec::with_capacity(t.len());
        for (i, v) in t.data().iter().enumerate() {
            mm.push(v - bias[i % w]);
        }
        // Cross-position variation of the output.
        let c: f32 = (1..4).map(|i| cosine(t.row(0), t.row(i))).sum::<f32>() / 3.0;
        eprintln!(
            "{:<22} relu={} out rms {:8.4}  bias rms {:8.4}  matmul-ish rms {:8.4}  cos {c:.4}",
            l.name,
            l.flag("has_relu"),
            rms(t.data()),
            rms(&bias),
            rms(&mm)
        );
        shown += 1;
        if shown >= 10 {
            break;
        }
    }
    assert!(shown > 0, "no dequantize layers with biases");
}

/// Decoder priming conventions. Multilingual NMT of this family feeds
/// `[EOS] [lang tag] …` — EOS acting as BOS — which is not what a bare tag
/// prime does. Each convention is run as a forced prefix, then greedy.
#[test]
fn decoder_priming_conventions() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let src = ids_for(&vocab, "en_US", &["i", "love", "the", "summer"]);
    let ho = nmt.encode(&src).expect("encode");

    let eos = vocab.id("</s>").expect("eos");
    let bos = vocab.id("<s>").expect("bos");
    let tag = prime_id(&vocab, "en_US", "fr_FR").expect("tag");
    let all: Vec<u32> = (0..vocab.len() as u32).collect();

    let conventions: [(&str, Vec<u32>); 4] = [
        ("tag only", vec![tag]),
        ("eos then tag", vec![eos, tag]),
        ("bos then tag", vec![bos, tag]),
        ("eos only", vec![eos]),
    ];

    for (label, prefix) in conventions {
        let mut state = DecoderState::new(nmt.layers(), nmt.width);
        let mut pieces = Vec::new();
        let mut prev = prefix[0];
        // Force the prefix, then decode greedily. Positions are 1-based.
        for step in 0..6usize {
            let (hidden, next) = nmt.step(prev, step + 1, &state, &ho).expect("step");
            state = next;
            if step + 1 < prefix.len() {
                prev = prefix[step + 1];
                continue;
            }
            let scores = nmt.logits(&hidden, &all).expect("logits");
            let (best, _) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("best");
            let id = all[best];
            if id == eos {
                break;
            }
            pieces.push(vocab.piece(id).unwrap_or("?").to_string());
            prev = id;
        }
        eprintln!("prime {label:<14} -> {:?}", pieces.join(""));
    }
}

/// Greedy decoding restricted to the shipped shortlist, which is what the
/// device actually does — the config sets `enable_shortlist` and names
/// `all-fr`. Scoring the whole 168k vocabulary lets the argmax wander onto
/// generic high-frequency tokens the model was never allowed to emit here.
#[test]
fn greedy_decoding_with_the_shortlist() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let Some(sl) = nmt.shortlist.as_ref() else {
        eprintln!("skipping: no shortlist installed");
        return;
    };
    eprintln!("shortlist covers {} source tokens", sl.len());

    let src = ids_for(&vocab, "en_US", &["i", "love", "the", "summer"]);
    let ho = nmt.encode(&src).expect("encode");
    let eos = vocab.id("</s>").expect("eos");
    let tag = prime_id(&vocab, "en_US", "fr_FR").expect("tag");

    let cands = nmt.candidates(&src, &[eos], vocab.len());
    eprintln!("candidates for this source: {}", cands.len());
    assert!(
        cands.len() < vocab.len() / 10,
        "shortlist should be far smaller than the vocabulary"
    );

    let mut state = DecoderState::new(nmt.layers(), nmt.width);
    let mut prev = tag;
    let mut pieces = Vec::new();
    for pos in 1..=12usize {
        let (hidden, next) = nmt.step(prev, pos, &state, &ho).expect("step");
        let scores = nmt.logits(&hidden, &cands).expect("logits");
        let (best, _) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("best");
        let id = cands[best];
        state = next;
        prev = id;
        if id == eos {
            break;
        }
        pieces.push(vocab.piece(id).unwrap_or("?").to_string());
    }
    let text = pieces.join("").replace('\u{2581}', " ");
    eprintln!("greedy fr (shortlisted): {text:?}");
    assert!(!pieces.is_empty(), "decoder emitted nothing");
}

/// Does the output actually track the input? The decisive test of whether the
/// model is translating at all, as opposed to emitting a fixed favourite token.
#[test]
fn output_tracks_the_source_sentence() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let tag = prime_id(&vocab, "en_US", "fr_FR").expect("tag");

    for words in [
        vec!["i", "love", "the", "summer"],
        vec!["the", "dog", "is", "black"],
        vec!["i", "want", "to", "drink", "water"],
    ] {
        let src = ids_for(&vocab, "en_US", &words);
        let ho = nmt.encode(&src).expect("encode");
        let cands = nmt.candidates(&src, &[eos], vocab.len());
        let mut state = DecoderState::new(nmt.layers(), nmt.width);
        let mut prev = tag;
        let mut pieces = Vec::new();
        for pos in 1..=10usize {
            let (hidden, next) = nmt.step(prev, pos, &state, &ho).expect("step");
            let scores = nmt.logits(&hidden, &cands).expect("logits");
            let (best, _) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("best");
            let id = cands[best];
            state = next;
            prev = id;
            if id == eos {
                break;
            }
            pieces.push(vocab.piece(id).unwrap_or("?").to_string());
        }
        eprintln!(
            "{:<32} -> {:?}",
            words.join(" "),
            pieces.join("").replace('\u{2581}', " ")
        );
    }
}

/// Scores decoded output against the OS's own translations.
///
/// Eyeballing single sentences has repeatedly misled this investigation, and
/// cosine has misled it three times. This is the objective function: token
/// overlap (F1) against the live-framework reference dump, over several
/// sentences, for whichever gather curve `RLX_TRANSLATE_GATHER` selects.
#[test]
fn score_decoding_against_the_os_reference() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let refpath = format!(
        "{}/reference/en_US-fr_FR.tsv",
        "/private/tmp/claude-501/-Users-Shared-rlx-models/08f4eb12-43ad-4c0c-a2c5-39438cf10ea7/scratchpad"
    );
    let Ok(text) = std::fs::read_to_string(&refpath) else {
        eprintln!("skipping: no reference dump at {refpath}");
        return;
    };
    // Short sentences only: long ones need beam search and more steps.
    let mut cases: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        let Some((s, t)) = line.split_once('\t') else {
            continue;
        };
        let w = s.split_whitespace().count();
        if (2..=5).contains(&w) && s.is_ascii() {
            cases.push((s.to_string(), t.to_string()));
        }
        if cases.len()
            >= std::env::var("RLX_TEST_N")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6)
        {
            break;
        }
    }
    if cases.is_empty() {
        eprintln!("skipping: no usable reference cases");
        return;
    }

    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    // The config's main PDecTranslatorBlock says `source-token: "en_US"` and
    // `target-token: "fr_FR> <en_US-fr_FR-optimal"`. A raw value is wrapped as
    // `<src-{v}>` / `<tar-{v}>`; the `> <` join exists so that wrapping yields
    // *two* valid pieces. So the target prefix is `<tar-fr_FR>` followed by
    // `<en_US-fr_FR-optimal>` — two tags, and the first one was missing.
    // Source carries all three control tokens; the decoder opens with `<s>`,
    // the conventional beginning-of-sequence piece. Measured by teacher-forced
    // log-rank, `<s>` beats every other opening by a wide margin
    // (0.554 vs 2.251 for the language tag).
    let src_tag = vocab.id("<src-en_US>").expect("<src-en_US>");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("<tar-fr_FR>");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("<optimal>");
    let prefix: Vec<u32> = vec![vocab.id("<s>").expect("<s>")];

    let norm = |s: &str| -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect()
    };

    let (mut tot_f1, mut n) = (0.0f64, 0usize);
    for (src_text, want) in &cases {
        let mut src = vec![src_tag, tar_tag, opt];
        src.extend(vocab.encode(src_text));
        // `AddSrcEos T`, and the terminator is `<s>` on *both* sides: measured,
        // ending the source with `</s>` instead costs a factor of ~25 in rank
        // (0.014 -> 0.574 mean log-rank on held-out reference sentences).
        src.push(prefix[0]);
        let Ok(ho) = nmt.encode(&src) else { continue };
        // `<s>` must stay scoreable: it opens the sequence *and* closes it
        // (measured - it ranks top-1 after a complete gold translation), so
        // excluding it left the model unable to stop and it looped instead.
        // The config sets `enable_shortlist`, so score only the shortlist union
        // rather than all 168 000 pieces.
        let stop = prefix[0];
        let cands = nmt.candidates(&src, &[stop, eos], vocab.len());
        let mut state = DecoderState::new(nmt.layers(), nmt.width);
        let mut prev = prefix[0];
        let mut pieces = Vec::new();
        for pos in 1..=18usize {
            let Ok((hidden, next)) = nmt.step(prev, pos, &state, &ho) else {
                break;
            };
            // Consume the rest of the target prefix before scoring anything.
            if pos < prefix.len() {
                state = next;
                prev = prefix[pos];
                continue;
            }
            let Ok(scores) = nmt.logits(&hidden, &cands) else {
                break;
            };
            let (best, _) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("best");
            let id = cands[best];
            state = next;
            prev = id;
            if id == eos || id == stop {
                break;
            }
            pieces.push(vocab.piece(id).unwrap_or("").to_string());
        }
        let got = pieces.join("").replace('\u{2581}', " ");
        let (g, w) = (norm(&got), norm(want));
        let hits = w.iter().filter(|t| g.contains(t)).count();
        let f1 = if g.is_empty() || w.is_empty() {
            0.0
        } else {
            let p = hits as f64 / g.len() as f64;
            let r = hits as f64 / w.len() as f64;
            if p + r == 0.0 {
                0.0
            } else {
                2.0 * p * r / (p + r)
            }
        };
        tot_f1 += f1;
        n += 1;
        eprintln!("  {src_text:?}\n     got  {got:?}\n     want {want:?}   F1 {f1:.3}");
    }
    let mean = tot_f1 / n.max(1) as f64;
    eprintln!(
        "GATHER={} mean F1 = {mean:.4} over {n} sentences",
        std::env::var("RLX_TRANSLATE_GATHER").unwrap_or_else(|_| "quartile".into()),
    );
    // Locks in the working protocol. Measured at 0.82 over 40 reference
    // sentences; this floor is low enough to tolerate sentence-set noise but
    // would catch any of the regressions that produced 0.07 before.
    assert!(
        mean > 0.55,
        "translation quality collapsed: mean F1 {mean:.4} over {n} sentences"
    );
}

/// Sweeps the plausible control-token layouts against the OS's own output.
///
/// The config gives `source-token: "en_US"` and
/// `target-token: "fr_FR> <en_US-fr_FR-optimal"`, and a raw value is wrapped as
/// `<src-{v}>` / `<tar-{v}>` — the `> <` join exists so the wrap yields two
/// valid pieces. What it does *not* say is where each tag goes: source side,
/// decoder prefix, or both. Rather than guess, score every layout.
#[test]
fn control_token_protocol_sweep() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let refpath = format!(
        "{}/reference/en_US-fr_FR.tsv",
        "/private/tmp/claude-501/-Users-Shared-rlx-models/08f4eb12-43ad-4c0c-a2c5-39438cf10ea7/scratchpad"
    );
    let Ok(text) = std::fs::read_to_string(&refpath) else {
        eprintln!("skipping: no reference dump");
        return;
    };
    let mut cases: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        let Some((s, t)) = line.split_once('\t') else {
            continue;
        };
        let w = s.split_whitespace().count();
        if (2..=5).contains(&w) && s.is_ascii() {
            cases.push((s.to_string(), t.to_string()));
        }
        if cases.len() >= 3 {
            break;
        }
    }
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let src_tag = vocab.id("<src-en_US>").expect("<src-en_US>");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("<tar-fr_FR>");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("<optimal>");

    // (name, source-side prefix, decoder prefix)
    let layouts: Vec<(&str, Vec<u32>, Vec<u32>)> = vec![
        (
            "src=[s]        dec=[t,o]",
            vec![src_tag],
            vec![tar_tag, opt],
        ),
        ("src=[s]        dec=[o]", vec![src_tag], vec![opt]),
        (
            "src=[s]        dec=[e,t,o]",
            vec![src_tag],
            vec![eos, tar_tag, opt],
        ),
        (
            "src=[s,t,o]    dec=[e]",
            vec![src_tag, tar_tag, opt],
            vec![eos],
        ),
        (
            "src=[s,t,o]    dec=[o]",
            vec![src_tag, tar_tag, opt],
            vec![opt],
        ),
        ("src=[]         dec=[t,o]", vec![], vec![tar_tag, opt]),
        ("src=[s]        dec=[e,o]", vec![src_tag], vec![eos, opt]),
    ];
    let norm = |s: &str| -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect()
    };
    for (name, spre, prefix) in &layouts {
        let (mut tot, mut n) = (0.0f64, 0usize);
        let mut sample = String::new();
        for (src_text, want) in &cases {
            let mut src = spre.clone();
            src.extend(src_text.split_whitespace().filter_map(|w| vocab.word_id(w)));
            src.push(eos);
            let Ok(ho) = nmt.encode(&src) else { continue };
            let cands = nmt.candidates(&src, &[eos], vocab.len());
            let mut state = DecoderState::new(nmt.layers(), nmt.width);
            let mut prev = prefix[0];
            let mut pieces = Vec::new();
            for pos in 1..=14usize {
                let Ok((hidden, next)) = nmt.step(prev, pos, &state, &ho) else {
                    break;
                };
                state = next;
                if pos < prefix.len() {
                    prev = prefix[pos];
                    continue;
                }
                let Ok(scores) = nmt.logits(&hidden, &cands) else {
                    break;
                };
                let (best, _) = scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("best");
                let id = cands[best];
                prev = id;
                if id == eos {
                    break;
                }
                pieces.push(vocab.piece(id).unwrap_or("").to_string());
            }
            let got = pieces.join("").replace('\u{2581}', " ");
            let (g, w) = (norm(&got), norm(want));
            let hits = w.iter().filter(|t| g.contains(t)).count();
            if !g.is_empty() && !w.is_empty() {
                let (p, r) = (hits as f64 / g.len() as f64, hits as f64 / w.len() as f64);
                if p + r > 0.0 {
                    tot += 2.0 * p * r / (p + r);
                }
            }
            n += 1;
            if sample.is_empty() {
                sample = got;
            }
        }
        eprintln!(
            "  {name}  F1 {:.4}   e.g. {sample:?}",
            tot / n.max(1) as f64
        );
    }
}

/// Is the *first* predicted token right?
///
/// Repetition after a plausible start and a wrong start are different faults:
/// the first fault is in how history is carried between steps, the second in
/// the encoder/handover. This separates them by printing the top candidates at
/// step 1, where no recurrent state has accumulated yet.
#[test]
fn first_token_against_the_os() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let src_tag = vocab.id("<src-en_US>").expect("src");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");

    // the OS's own output for these three, from the reference dump.
    let cases = [
        ("i love the summer", "J'adore l'été"),
        ("i want to hug you", "Je veux vous serrer dans mes bras"),
        ("the woman of my dreams", "La femme de mes rêves"),
    ];
    for (text, want) in cases {
        let mut src = vec![src_tag, tar_tag, opt];
        src.extend(text.split_whitespace().filter_map(|w| vocab.word_id(w)));
        src.push(eos);
        let ho = nmt.encode(&src).expect("encode");
        let all: Vec<u32> = (0..vocab.len() as u32).collect();
        let state = DecoderState::new(nmt.layers(), nmt.width);
        let (hidden, _) = nmt.step(opt, 1, &state, &ho).expect("step");
        let scores = nmt.logits(&hidden, &all).expect("logits");
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top: Vec<String> = order
            .iter()
            .take(8)
            .map(|i| format!("{:?}", vocab.piece(all[*i]).unwrap_or("")))
            .collect();
        eprintln!(
            "  {text:?}\n     want first piece of {want:?}\n     top8 {}",
            top.join(" ")
        );
    }
}

/// Is the source attended to at all?
///
/// The first predicted token is wrong while the top candidates still carry
/// source semantics, which is what weak cross-attention looks like. Uniform
/// `attn_probs` would mean the query/key contraction is wrong; peaked probs
/// would clear cross-attention and move the fault downstream.
#[test]
fn cross_attention_is_peaked() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let src_tag = vocab.id("<src-en_US>").expect("src");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");

    let mut src = vec![src_tag, tar_tag, opt];
    src.extend(
        "the woman of my dreams"
            .split_whitespace()
            .filter_map(|w| vocab.word_id(w)),
    );
    src.push(eos);
    eprintln!("  source is {} tokens", src.len());
    let ho = nmt.encode(&src).expect("encode");
    let state = DecoderState::new(nmt.layers(), nmt.width);
    let env = nmt.step_env(opt, 1, &state, &ho).expect("step");

    for l in 0..nmt.layers() {
        let name = format!("decoder.{l}.encoder_attn.attn_probs");
        let Some(v) = env.get(&name) else {
            eprintln!("  {name}: absent");
            continue;
        };
        let t = v.f32().expect("f32");
        let d = t.data();
        let s = t.dims().len();
        let keys = *t.dims().last().expect("width");
        // Uniform over `keys` would be 1/keys; report the peak and the entropy
        // of head 0 so a flat distribution is unmistakable.
        let head0 = &d[..keys];
        let max = head0.iter().cloned().fold(f32::MIN, f32::max);
        let ent: f32 = -head0
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f32>();
        eprintln!(
            "  {name}: dims {:?} (rank {s}) head0 max {max:.3} (uniform {:.3}) entropy {ent:.3} (uniform {:.3})",
            t.dims(),
            1.0 / keys as f32,
            (keys as f32).ln()
        );
        eprintln!(
            "     head0 = {:?}",
            head0
                .iter()
                .map(|p| (p * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
    }
}

/// Teacher forcing against the OS's own target.
///
/// Free-running decode compounds its own errors, so a wrong first token tells
/// us little about the rest. Feeding the *correct* prefix at every position
/// separates a broken model from a broken search: if each next token is
/// predicted correctly here, the wiring is right and only the start/priming is
/// wrong; if it fails here too, the fault is in the model path itself.
#[test]
fn teacher_forced_next_token() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let src_tag = vocab.id("<src-en_US>").expect("src");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");
    // Real decoding suppresses the control vocabulary (the config carries
    // `shortlist-suppress-tokens`); scoring against it lets `<s>` win
    // positions no decoder would ever emit it at.
    let all: Vec<u32> = (0..vocab.len() as u32)
        .filter(|i| {
            let p = vocab.piece(*i).unwrap_or("");
            *i == eos || !(p.starts_with('<') && p.ends_with('>'))
        })
        .collect();

    let cases = [
        ("the woman of my dreams", "La femme de mes rêves"),
        ("i want to hug you", "Je veux vous serrer dans mes bras"),
    ];
    for (text, gold) in cases {
        let mut src = vec![src_tag, tar_tag, opt];
        src.extend(text.split_whitespace().filter_map(|w| vocab.word_id(w)));
        src.push(eos);
        let ho = nmt.encode(&src).expect("encode");

        let gold_ids: Vec<u32> = gold
            .split_whitespace()
            .filter_map(|w| vocab.word_id(w))
            .collect();
        if gold_ids.len() != gold.split_whitespace().count() {
            eprintln!("  {text:?}: some gold words are not single pieces, skipping");
            continue;
        }
        eprintln!("  {text:?} -> {gold:?}");
        let mut state = DecoderState::new(nmt.layers(), nmt.width);
        let mut prev = opt;
        let mut hits = 0usize;
        for (i, want) in gold_ids.iter().enumerate() {
            let (hidden, next) = nmt.step(prev, i + 1, &state, &ho).expect("step");
            let scores = nmt.logits(&hidden, &all).expect("logits");
            let (best, _) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("best");
            let got = all[best];
            // Where does the correct token actually rank?
            let want_pos = all
                .iter()
                .position(|c| c == want)
                .expect("gold is a candidate");
            let want_score = scores[want_pos];
            let rank = scores.iter().filter(|s| **s > want_score).count();
            eprintln!(
                "     pos {} predict {:?} (want {:?}, its rank {rank})",
                i + 1,
                vocab.piece(got).unwrap_or(""),
                vocab.piece(*want).unwrap_or("")
            );
            if got == *want {
                hits += 1;
            }
            state = next;
            prev = *want; // teacher forcing
        }
        eprintln!("     {hits}/{} exact next-token", gold_ids.len());
    }
}

/// Has the encoder's self-attention saturated?
///
/// Espresso emits no explicit `1/sqrt(head_dim)` layer, so either the query
/// weights absorb it or it is missing. A saturated softmax (max-prob 1.0,
/// entropy 0) is a hard argmax that throws away the source mixing, and it
/// degrades output without destroying it — which is the symptom.
#[test]
fn encoder_self_attention_entropy() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    // Does layer 0 saturate because of the control token, or regardless of
    // content? Run the same sentence with and without the tag, and a wholly
    // different sentence, and compare where the queries look.
    for (label, tagged, sentence) in [
        ("tagged", true, "the woman of my dreams"),
        ("untagged", false, "the woman of my dreams"),
        ("untagged/other", false, "i love the summer"),
    ] {
        let mut src = Vec::new();
        if tagged {
            src.push(vocab.id("<src-en_US>").expect("src"));
        }
        src.extend(sentence.split_whitespace().filter_map(|w| vocab.word_id(w)));
        src.push(eos);
        eprintln!("--- {label}: {sentence:?}");
        layer0_report(&nmt, &src);
    }
}

fn layer0_report(nmt: &Nmt, src: &[u32]) {
    let env = nmt.encode_env(src).expect("encode");
    let n = src.len();
    eprintln!(
        "  {n} source tokens; uniform entropy would be {:.3}",
        (n as f32).ln()
    );
    for l in 0..4 {
        let name = format!("encoder.{l}.self_attn.attn_probs");
        let Some(v) = env.get(&name) else { continue };
        let t = v.f32().expect("f32");
        let keys = *t.dims().last().expect("w");
        let d = t.data();
        let rows = d.len() / keys;
        let mut ent = 0.0f32;
        let mut mx = 0.0f32;
        for r in 0..rows {
            let row = &d[r * keys..(r + 1) * keys];
            ent += -row
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| p * p.ln())
                .sum::<f32>();
            mx += row.iter().cloned().fold(f32::MIN, f32::max);
        }
        eprintln!(
            "  {name}: dims {:?} mean entropy {:.4} mean max-prob {:.4}",
            t.dims(),
            ent / rows as f32,
            mx / rows as f32
        );
        // A saturated layer that attends to the diagonal is benign: it is a
        // position-wise linear layer, not lost source mixing. Print where each
        // query actually looks so the two cases are distinguishable.
        if l == 0 {
            let argmax: Vec<usize> = (0..rows)
                .map(|r| {
                    let row = &d[r * keys..(r + 1) * keys];
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();
            eprintln!("     argmax per query (head-major, {keys} queries per head): {argmax:?}");
        }
    }
}

/// Does `transpose` actually split the heads?
///
/// All eight encoder heads producing identical attention is impossible for a
/// trained model, so the reshape/transpose pair that forms `[heads, seq, dim]`
/// from `[seq, heads*dim]` is suspect. This checks it elementwise against the
/// pre-transpose tensor: `query_transpose[h][s][d]` must equal
/// `query[s][h * 64 + d]`.
#[test]
fn head_split_is_elementwise_correct() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let mut src = vec![vocab.id("<src-en_US>").expect("src")];
    src.extend(
        "the woman of my dreams"
            .split_whitespace()
            .filter_map(|w| vocab.word_id(w)),
    );
    src.push(eos);
    let env = nmt.encode_env(&src).expect("encode");

    let flat = env
        .get("encoder.0.self_attn.query")
        .expect("query")
        .f32()
        .expect("f32");
    let split = env
        .get("encoder.0.self_attn.query_transpose")
        .expect("query_transpose")
        .f32()
        .expect("f32");
    eprintln!(
        "  query {:?}  query_transpose {:?}",
        flat.dims(),
        split.dims()
    );
    let s_len = src.len();
    let (heads, dim) = (8usize, 64usize);
    assert_eq!(split.dims(), &[heads, s_len, dim], "unexpected split shape");

    let f = flat.data();
    let g = split.data();
    let mut bad = 0usize;
    let mut first: Option<(usize, usize, usize, f32, f32)> = None;
    for h in 0..heads {
        for s in 0..s_len {
            for d in 0..dim {
                let want = f[s * heads * dim + h * dim + d];
                let got = g[(h * s_len + s) * dim + d];
                if (want - got).abs() > 1e-6 {
                    bad += 1;
                    first.get_or_insert((h, s, d, want, got));
                }
            }
        }
    }
    eprintln!("  mismatched elements: {bad} of {}", heads * s_len * dim);
    if let Some((h, s, d, want, got)) = first {
        eprintln!("  first mismatch at head {h} pos {s} dim {d}: want {want} got {got}");
    }
    assert_eq!(bad, 0, "transpose does not split heads correctly");
}

/// Mean log-rank of the OS's own next token, as a single comparable number.
///
/// Free-running output and teacher-forced rank have disagreed, and eyeballing
/// a handful of pieces has misled this investigation more than once. Mean
/// log10(rank+1) over many gold positions is stable enough to compare
/// structural hypotheses against: 0 is perfect, ~5 is chance over 168k pieces.
#[test]
fn teacher_forced_mean_log_rank() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let refpath = format!(
        "{}/reference/en_US-fr_FR.tsv",
        "/private/tmp/claude-501/-Users-Shared-rlx-models/08f4eb12-43ad-4c0c-a2c5-39438cf10ea7/scratchpad"
    );
    let Ok(text) = std::fs::read_to_string(&refpath) else {
        eprintln!("skipping: no reference dump");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let mut nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let src_tag = vocab.id("<src-en_US>").expect("src");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");
    // `<s>` opens the target sequence and also closes it.
    let bos = vocab.id("<s>").expect("bos");

    // Real SentencePiece segmentation on both sides, so teacher forcing feeds
    // exactly the pieces the OS's own decoder would have emitted. The earlier
    // whole-word lookup silently skipped anything it could not represent as one
    // piece, which threw away every sentence with an apostrophe.
    // `RLX_TEST_SKIP` holds out a different slice of the reference dump, so a
    // parameter tuned on one set can be checked on another.
    let skip: usize = std::env::var("RLX_TEST_SKIP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut cases: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    for line in text.lines().skip(skip) {
        let Some((s, t)) = line.split_once('\t') else {
            continue;
        };
        if s.split_whitespace().count() > 6 {
            continue;
        }
        let mut g = vocab.encode(t);
        if g.is_empty() {
            continue;
        }
        // Score end-of-sentence too: after a complete gold translation a working
        // model should rank `</s>` at the top, a sharp check that does not
        // depend on choosing among near-synonyms.
        g.push(bos);
        let mut src = vec![src_tag, tar_tag, opt];
        src.extend(vocab.encode(s));
        // `AddSrcEos T` says the source is terminated, but not with which piece.
        // The target side terminates with `<s>`, so try both.
        src.push(if std::env::var("RLX_SRC_EOS").as_deref() == Ok("bos") {
            bos
        } else {
            eos
        });
        cases.push((src, g));
        if cases.len() >= 8 {
            break;
        }
    }
    eprintln!("  {} gold sentences", cases.len());

    for with_pos in [true, false] {
        nmt.set_decoder_uses_positions(with_pos);
        let (mut tot, mut n, mut top1) = (0.0f64, 0usize, 0usize);
        let mut stop_ranks: Vec<usize> = Vec::new();
        for (src, gold) in &cases {
            let Ok(ho) = nmt.encode(src) else { continue };
            // Score against the shortlist the config asks for, plus the gold
            // pieces so a rank is always defined. ~100x fewer dot products.
            let mut always = vec![bos, eos];
            always.extend(gold.iter().copied());
            let cands = nmt.candidates(src, &always, vocab.len());
            let mut state = DecoderState::new(nmt.layers(), nmt.width);
            let mut prev = bos;
            for (i, want) in gold.iter().enumerate() {
                let Ok((hidden, next)) = nmt.step(prev, i + 1, &state, &ho) else {
                    break;
                };
                let Ok(scores) = nmt.logits(&hidden, &cands) else {
                    break;
                };
                let wp = cands
                    .iter()
                    .position(|c| c == want)
                    .expect("gold candidate");
                let ws = scores[wp];
                let rank = scores.iter().filter(|s| **s > ws).count();
                tot += ((rank + 1) as f64).log10();
                if rank == 0 {
                    top1 += 1;
                }
                if *want == bos {
                    stop_ranks.push(rank);
                    // What does the model think ends a sentence, if not `</s>`?
                    let mut ord: Vec<usize> = (0..scores.len()).collect();
                    ord.sort_by(|a, b| {
                        scores[*b]
                            .partial_cmp(&scores[*a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let top: Vec<&str> = ord
                        .iter()
                        .take(6)
                        .map(|k| vocab.piece(cands[*k]).unwrap_or(""))
                        .collect();
                    eprintln!(
                        "     at end: top {top:?}  stop score {ws:.4}  best {:.4}  worst {:.4}",
                        scores[ord[0]],
                        scores[ord[ord.len() - 1]]
                    );
                }
                n += 1;
                state = next;
                prev = *want;
            }
        }
        eprintln!(
            "  gather={:<10} decoder_positions={with_pos:<5} mean log-rank {:.3}  top-1 {top1}/{n}",
            std::env::var("RLX_TRANSLATE_GATHER").unwrap_or_else(|_| "quartile".into()),
            tot / n.max(1) as f64
        );
        eprintln!("     end-of-sentence ranks: {stop_ranks:?}");
    }
}

/// Where do the control tokens belong? Scored by mean log-rank.
///
/// The earlier free-running sweep preferred every tag on the source side, but
/// that was measured under the wrong dequantisation curve and with a metric
/// that compounds its own errors. Encoder layer 0 collapses entirely onto
/// `<src-en_US>` when it is present, which is reason to re-ask the question
/// against teacher-forced rank.
#[test]
fn control_token_placement_by_log_rank() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let refpath = format!(
        "{}/reference/en_US-fr_FR.tsv",
        "/private/tmp/claude-501/-Users-Shared-rlx-models/08f4eb12-43ad-4c0c-a2c5-39438cf10ea7/scratchpad"
    );
    let Ok(text) = std::fs::read_to_string(&refpath) else {
        eprintln!("skipping: no reference dump");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let s_tag = vocab.id("<src-en_US>").expect("src");
    let t_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");
    let cands: Vec<u32> = (0..vocab.len() as u32)
        .filter(|i| {
            let p = vocab.piece(*i).unwrap_or("");
            *i == eos || !(p.starts_with('<') && p.ends_with('>'))
        })
        .collect();

    let mut cases: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    for line in text.lines() {
        let Some((s, t)) = line.split_once('\t') else {
            continue;
        };
        if !s.is_ascii() || s.split_whitespace().count() > 6 {
            continue;
        }
        let g: Vec<u32> = t
            .split_whitespace()
            .filter_map(|w| vocab.word_id(w))
            .collect();
        if g.len() != t.split_whitespace().count() || g.is_empty() {
            continue;
        }
        let w: Vec<u32> = s
            .split_whitespace()
            .filter_map(|x| vocab.word_id(x))
            .collect();
        cases.push((w, g));
        if cases.len() >= 8 {
            break;
        }
    }

    let layouts: Vec<(&str, Vec<u32>, Vec<u32>)> = vec![
        ("src=[s,t,o]  dec=[o]", vec![s_tag, t_tag, opt], vec![opt]),
        ("src=[s,t,o]  dec=[e]", vec![s_tag, t_tag, opt], vec![eos]),
        ("src=[s]      dec=[t,o]", vec![s_tag], vec![t_tag, opt]),
        ("src=[s]      dec=[o]", vec![s_tag], vec![opt]),
        ("src=[]       dec=[t,o]", vec![], vec![t_tag, opt]),
        ("src=[]       dec=[o]", vec![], vec![opt]),
        ("src=[]       dec=[e,t,o]", vec![], vec![eos, t_tag, opt]),
        ("src=[t,o]    dec=[e]", vec![t_tag, opt], vec![eos]),
    ];
    for (name, spre, prefix) in &layouts {
        let (mut tot, mut n, mut top1) = (0.0f64, 0usize, 0usize);
        for (words, gold) in &cases {
            let mut src = spre.clone();
            src.extend(words.iter().copied());
            src.push(eos);
            let Ok(ho) = nmt.encode(&src) else { continue };
            let mut state = DecoderState::new(nmt.layers(), nmt.width);
            let mut prev = prefix[0];
            let mut step_i = 0usize;
            // Positions 1..prefix.len()-1 consume the prefix; scoring starts
            // at prefix.len(), so the last scored position is prefix.len()-1+gold.len().
            for pos in 1..=(prefix.len() - 1 + gold.len()) {
                let Ok((hidden, next)) = nmt.step(prev, pos, &state, &ho) else {
                    break;
                };
                state = next;
                if pos < prefix.len() {
                    prev = prefix[pos];
                    continue;
                }
                let want = gold[step_i];
                let Ok(scores) = nmt.logits(&hidden, &cands) else {
                    break;
                };
                let wp = cands
                    .iter()
                    .position(|c| *c == want)
                    .expect("gold candidate");
                let ws = scores[wp];
                let rank = scores.iter().filter(|s| **s > ws).count();
                tot += ((rank + 1) as f64).log10();
                if rank == 0 {
                    top1 += 1;
                }
                n += 1;
                prev = want;
                step_i += 1;
            }
        }
        eprintln!(
            "  {name}  mean log-rank {:.3}  top-1 {top1}/{n}",
            tot / n.max(1) as f64
        );
    }
}

/// Sanity-checks the SentencePiece segmentation on the shipped vocabulary.
#[test]
fn spm_segmentation_looks_right() {
    let Some((_, _, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    for text in [
        "the woman of my dreams",
        "La femme de mes rêves",
        "i love the summer",
        "J'adore l'été",
        "antidisestablishmentarianism",
        "Ωμέγα 漢字 🙂",
    ] {
        let ids = vocab.encode(text);
        let pieces: Vec<&str> = ids.iter().map(|i| vocab.piece(*i).unwrap_or("?")).collect();
        // Round-trip: the pieces must reassemble the input. `<0xNN>` pieces are
        // raw bytes (this bundle covers only en/es/de/it/fr/pt/nl, so anything
        // outside that legitimately falls back to bytes), so rebuild via bytes.
        let mut raw: Vec<u8> = Vec::new();
        for p in &pieces {
            if let Some(h) = p.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')) {
                raw.push(u8::from_str_radix(h, 16).expect("hex byte"));
            } else {
                raw.extend_from_slice(p.as_bytes());
            }
        }
        let back = String::from_utf8_lossy(&raw).replace('\u{2581}', " ");
        eprintln!("  {text:?}\n     {} pieces: {pieces:?}", ids.len());
        assert_eq!(back.trim(), text, "segmentation does not round-trip");
    }
}

/// Verifies `batch_matmul` against a hand-computed contraction on real tensors.
///
/// Attention is the one place where a wrong contraction axis cannot be caught by
/// a shape check when every dimension is 64, and it is central enough that it
/// deserves a value-level check rather than an inspection of the net JSON.
#[test]
fn batch_matmul_matches_manual_attention() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let mut src = vec![vocab.id("<src-en_US>").expect("src")];
    src.extend(vocab.encode("the woman of my dreams"));
    src.push(eos);
    let env = nmt.encode_env(&src).expect("encode");

    let q = env
        .get("encoder.0.self_attn.query_transpose")
        .expect("q")
        .f32()
        .expect("f32");
    let k = env
        .get("encoder.0.self_attn.key_transpose")
        .expect("k")
        .f32()
        .expect("f32");
    let w = env
        .get("encoder.0.self_attn.attn_weights")
        .expect("w")
        .f32()
        .expect("f32");
    eprintln!(
        "  q {:?} k {:?} -> weights {:?}",
        q.dims(),
        k.dims(),
        w.dims()
    );
    let (h, s, dim) = (q.dims()[0], q.dims()[1], q.dims()[2]);
    assert_eq!(w.dims(), &[h, s, s]);

    let (qd, kd, wd) = (q.data(), k.data(), w.data());
    let mut worst = 0.0f32;
    for head in 0..h {
        for i in 0..s {
            for j in 0..s {
                let mut acc = 0.0f32;
                for t in 0..dim {
                    acc += qd[(head * s + i) * dim + t] * kd[(head * s + j) * dim + t];
                }
                let got = wd[(head * s + i) * s + j];
                worst = worst.max((acc - got).abs() / acc.abs().max(1.0));
            }
        }
    }
    eprintln!("  worst relative error vs manual q.k^T: {worst:.3e}");
    assert!(worst < 1e-4, "batch_matmul does not match q.k^T");
}

/// What token actually terminates a target sentence?
///
/// `</s>` ranks near the bottom of the vocabulary after a complete gold
/// translation, which is worse than chance and so cannot be mere uncertainty.
/// Scoring against the *unfiltered* vocabulary reveals whether some control
/// piece is the real terminator.
#[test]
fn sentence_terminator_identity() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let src_tag = vocab.id("<src-en_US>").expect("src");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");
    let all: Vec<u32> = (0..vocab.len() as u32).collect();

    for (text, gold) in [
        ("the woman of my dreams", "La femme de mes rêves"),
        ("i love the summer", "J'adore l'été"),
    ] {
        let mut src = vec![src_tag, tar_tag, opt];
        src.extend(vocab.encode(text));
        src.push(eos);
        let ho = nmt.encode(&src).expect("encode");
        let g = vocab.encode(gold);
        let mut state = DecoderState::new(nmt.layers(), nmt.width);
        let mut prev = vocab.id("<s>").expect("bos");
        let mut hidden = None;
        for (i, t) in g.iter().enumerate() {
            let (hh, next) = nmt.step(prev, i + 1, &state, &ho).expect("step");
            state = next;
            prev = *t;
            hidden = Some(hh);
        }
        // One more step, consuming the last gold piece: this is where the model
        // must choose to end.
        let (hh, _) = nmt.step(prev, g.len() + 1, &state, &ho).expect("step");
        let _ = hidden;
        let scores = nmt.logits(&hh, &all).expect("logits");
        let mut ord: Vec<usize> = (0..scores.len()).collect();
        ord.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        eprintln!("  {text:?} -> {gold:?}");
        eprintln!(
            "     top12: {:?}",
            ord.iter()
                .take(12)
                .map(|k| vocab.piece(all[*k]).unwrap_or(""))
                .collect::<Vec<_>>()
        );
        for name in ["</s>", "<s>", "<unk>", "<STRUCT_END>", "<GENDER_TAG>"] {
            if let Some(id) = vocab.id(name) {
                let sc = scores[id as usize];
                let rank = scores.iter().filter(|s| **s > sc).count();
                eprintln!("     {name:<14} rank {rank}");
            }
        }
    }
}

/// Does the decoder's history actually reach its output?
///
/// The model wants to re-emit a word it has already produced, which is what a
/// decoder with no memory does. History travels only through the average
/// attention running mean, so this checks two things: that `accum` really is
/// the running sum across steps, and that a *different* history at the same
/// position changes the prediction.
#[test]
fn decoder_history_reaches_the_output() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");
    let mut src = vec![
        vocab.id("<src-en_US>").expect("src"),
        vocab.id("<tar-fr_FR>").expect("tar"),
        opt,
    ];
    src.extend(vocab.encode("the woman of my dreams"));
    src.push(eos);
    let ho = nmt.encode(&src).expect("encode");

    // 1. Is accum the running sum? Layer 0 accumulates the input embedding, so
    //    after t steps its norm should grow, not sit still.
    let mut state = DecoderState::new(nmt.layers(), nmt.width);
    let mut prev = opt;
    let hist = vocab.encode("La femme de mes");
    let mut norms = Vec::new();
    for (i, t) in std::iter::once(&opt).chain(hist.iter()).enumerate() {
        let env = nmt.step_env(*t, i + 1, &state, &ho).expect("step_env");
        let acc = env
            .get("decoder.0.self_attn.accum.next")
            .expect("accum.next")
            .f32()
            .expect("f32");
        let n: f32 = acc.data().iter().map(|v| v * v).sum::<f32>().sqrt();
        norms.push((n * 100.0).round() / 100.0);
        let (_, next) = nmt.step(*t, i + 1, &state, &ho).expect("step");
        state = next;
        prev = *t;
    }
    let _ = prev;
    eprintln!("  ||accum|| after each step: {norms:?}");
    assert!(
        norms.windows(2).all(|w| w[1] != w[0]),
        "accum is not changing between steps - history is not accumulating"
    );

    // 2. Same position, different history: do the predictions differ?
    let all: Vec<u32> = (0..vocab.len() as u32)
        .filter(|i| {
            let p = vocab.piece(*i).unwrap_or("");
            *i == eos || !(p.starts_with('<') && p.ends_with('>'))
        })
        .collect();
    let mut tops = Vec::new();
    for hist in ["La femme de mes", "Je veux vous serrer"] {
        let mut state = DecoderState::new(nmt.layers(), nmt.width);
        let mut prev = opt;
        let ids = vocab.encode(hist);
        for (i, t) in ids.iter().enumerate() {
            let (_, next) = nmt.step(prev, i + 1, &state, &ho).expect("step");
            state = next;
            prev = *t;
        }
        let (hh, _) = nmt.step(prev, ids.len() + 1, &state, &ho).expect("step");
        let scores = nmt.logits(&hh, &all).expect("logits");
        let (best, _) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("best");
        let piece = vocab.piece(all[best]).unwrap_or("").to_string();
        eprintln!("  history {hist:?} -> next {piece:?}");
        tops.push(piece);
    }
    assert_ne!(
        tops[0], tops[1],
        "different histories give the same prediction"
    );
}

/// Sweeps the decoder's opening tokens, including `<s>` as BOS.
///
/// Teacher-forced ranks are bad at the first positions and perfect at the last
/// ("La femme de mes" -> "rêves"), which is what a wrong *opening* looks like
/// rather than a broken model. The earlier placement sweep tried `</s>` and the
/// language tags as the decoder's first input but never `<s>`, the conventional
/// beginning-of-sequence piece.
#[test]
fn decoder_bos_sweep() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let refpath = format!(
        "{}/reference/en_US-fr_FR.tsv",
        "/private/tmp/claude-501/-Users-Shared-rlx-models/08f4eb12-43ad-4c0c-a2c5-39438cf10ea7/scratchpad"
    );
    let Ok(text) = std::fs::read_to_string(&refpath) else {
        eprintln!("skipping: no reference dump");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let bos = vocab.id("<s>").expect("bos");
    let s_tag = vocab.id("<src-en_US>").expect("src");
    let t_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");
    let cands: Vec<u32> = (0..vocab.len() as u32)
        .filter(|i| {
            let p = vocab.piece(*i).unwrap_or("");
            *i == eos || !(p.starts_with('<') && p.ends_with('>'))
        })
        .collect();

    let mut cases: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    for line in text.lines() {
        let Some((s, t)) = line.split_once('\t') else {
            continue;
        };
        if s.split_whitespace().count() > 6 {
            continue;
        }
        let g = vocab.encode(t);
        if g.is_empty() {
            continue;
        }
        let mut src = vec![s_tag, t_tag, opt];
        src.extend(vocab.encode(s));
        src.push(eos);
        cases.push((src, g));
        if cases.len() >= 4 {
            break;
        }
    }

    let prefixes: Vec<(&str, Vec<u32>)> = vec![
        ("[opt]", vec![opt]),
        ("[bos]", vec![bos]),
        ("[bos,opt]", vec![bos, opt]),
        ("[bos,tar,opt]", vec![bos, t_tag, opt]),
        ("[opt,bos]", vec![opt, bos]),
        ("[eos]", vec![eos]),
    ];
    for (name, prefix) in &prefixes {
        let (mut tot, mut n, mut top1) = (0.0f64, 0usize, 0usize);
        let mut firsts: Vec<usize> = Vec::new();
        for (src, gold) in &cases {
            let Ok(ho) = nmt.encode(src) else { continue };
            let mut state = DecoderState::new(nmt.layers(), nmt.width);
            let mut prev = prefix[0];
            let mut step_i = 0usize;
            for pos in 1..=(prefix.len() - 1 + gold.len()) {
                let Ok((hidden, next)) = nmt.step(prev, pos, &state, &ho) else {
                    break;
                };
                state = next;
                if pos < prefix.len() {
                    prev = prefix[pos];
                    continue;
                }
                let want = gold[step_i];
                let Ok(scores) = nmt.logits(&hidden, &cands) else {
                    break;
                };
                let wp = cands.iter().position(|c| *c == want).expect("candidate");
                let ws = scores[wp];
                let rank = scores.iter().filter(|s| **s > ws).count();
                tot += ((rank + 1) as f64).log10();
                if rank == 0 {
                    top1 += 1;
                }
                if step_i == 0 {
                    firsts.push(rank);
                }
                n += 1;
                prev = want;
                step_i += 1;
            }
        }
        eprintln!(
            "  dec={name:<15} mean log-rank {:.3}  top-1 {top1}/{n}  first-token ranks {firsts:?}",
            tot / n.max(1) as f64
        );
    }
}

/// End-to-end beam-search translation against the OS's own output.
///
/// Greedy decode loops because it cannot recover from one bad step; the config
/// asks for beam 3 with `norm-costs`, which is what the OS actually runs.
#[test]
fn beam_translation_against_the_os() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let eos = vocab.id("</s>").expect("eos");
    let bos = vocab.id("<s>").expect("bos");
    let src_tag = vocab.id("<src-en_US>").expect("src");
    let tar_tag = vocab.id("<tar-fr_FR>").expect("tar");
    let opt = vocab.id("<en_US-fr_FR-optimal>").expect("opt");

    let params = rlx_translate::PDecParams {
        beam: 3,
        nbest: 1,
        norm_costs: true,
        max_seq_length: 40,
        max_seq_length_floor: 12,
        max_seq_length_relative: 2.0,
        ..Default::default()
    };

    // The shortlist already restricts what can be scored, so only the control
    // pieces that survive into it need suppressing - never the terminator.
    let suppressed: std::collections::BTreeSet<u32> = (0..vocab.len() as u32)
        .filter(|i| {
            let p = vocab.piece(*i).unwrap_or("");
            *i != bos && *i != eos && p.starts_with('<') && p.ends_with('>')
        })
        .collect();
    let options = rlx_translate::SearchOptions {
        pruning: Default::default(),
        suppressed,
        no_repeat_ngram: 0,
        no_repeat_char_ngram: 0,
        length_penalty: None,
    };

    for (text, want) in [
        ("the woman of my dreams", "La femme de mes rêves"),
        ("i love the summer", "J'adore l'été"),
        ("1 million", "1 million"),
    ] {
        let mut src = vec![src_tag, tar_tag, opt];
        src.extend(vocab.encode(text));
        src.push(bos); // the terminator is `<s>` on both sides
        let ho = nmt.encode(&src).expect("encode");
        let cands = nmt.candidates(&src, &[bos, eos], vocab.len());
        eprintln!("     shortlist: {} candidates", cands.len());
        let mut dec = nmt.beam_decoder(&ho, vocab.len(), bos, cands);
        let hyps = rlx_translate::beam::search(&mut dec, &params, &[bos], src.len(), &options)
            .expect("beam search");
        let got = hyps
            .first()
            .map(|h| {
                h.tokens
                    .iter()
                    .map(|t| vocab.piece(*t).unwrap_or(""))
                    .collect::<String>()
                    .replace('\u{2581}', " ")
                    .trim()
                    .to_string()
            })
            .unwrap_or_default();
        eprintln!("  {text:?}\n     got  {got:?}\n     want {want:?}");
    }
}

/// Is cosine over sentence embeddings actually discriminative?
///
/// Cosine has misled this investigation three times, always when used to
/// compare *pipeline stages*. This is a different use — comparing two sentences
/// in one space — but it still has to earn its place: a metric that scores
/// paraphrase and nonsense alike is worse than useless. This prints the numbers
/// for equivalent, related and unrelated pairs so the separation is visible,
/// and fails if equivalent pairs do not out-score unrelated ones.
#[test]
fn sentence_cosine_separates_paraphrase_from_nonsense() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    let emb = |loc: &str, t: &str| nmt.sentence_embedding(&vocab, loc, t).expect("embed");
    let cos =
        |a: &str, b: &str, loc: &str| rlx_translate::score::cosine(&emb(loc, a), &emb(loc, b));

    // (label, a, b, locale, expected band)
    let equivalent = [
        ("paraphrase", "J'aime l'été", "J'adore l'été", "fr_FR"),
        (
            "word order",
            "La femme de mes rêves",
            "La femme dont je rêve",
            "fr_FR",
        ),
        (
            "politeness",
            "Ich liebe den Sommer",
            "Ich mag den Sommer",
            "de_DE",
        ),
    ];
    let unrelated = [
        (
            "unrelated",
            "J'adore l'été",
            "Le train arrive à huit heures",
            "fr_FR",
        ),
        (
            "unrelated",
            "La femme de mes rêves",
            "Il pleut des cordes",
            "fr_FR",
        ),
    ];
    let mut lo_equiv = 1.0f64;
    for (k, a, b, loc) in equivalent {
        let c = cos(a, b, loc);
        eprintln!("  {k:<12} {a:?} / {b:?} -> {c:.4}");
        lo_equiv = lo_equiv.min(c);
    }
    let mut hi_unrel = -1.0f64;
    for (k, a, b, loc) in unrelated {
        let c = cos(a, b, loc);
        eprintln!("  {k:<12} {a:?} / {b:?} -> {c:.4}");
        hi_unrel = hi_unrel.max(c);
    }
    // Everything sits near 1.0: post-norm encoder states are anisotropic, so a
    // large common direction dominates every pair. Subtracting the mean of a
    // sample removes it. Measure whether that actually widens the gap.
    let corpus = [
        "J'aime l'été",
        "J'adore l'été",
        "La femme de mes rêves",
        "La femme dont je rêve",
        "Le train arrive à huit heures",
        "Il pleut des cordes",
        "Où puis-je obtenir un hamburger",
        "Le chat dort sur le canapé",
    ];
    let vecs: Vec<Vec<f32>> = corpus.iter().map(|t| emb("fr_FR", t)).collect();
    let mut mean = vec![0.0f32; vecs[0].len()];
    for v in &vecs {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += *x / vecs.len() as f32;
        }
    }
    let _ = &mean;
    let mean = rlx_translate::score::center(&vecs);
    let cc = |i: usize, j: usize| rlx_translate::score::cosine_centered(&vecs[i], &vecs[j], &mean);
    eprintln!("  --- centered on an 8-sentence sample ---");
    eprintln!("  paraphrase   0/1 -> {:.4}", cc(0, 1));
    eprintln!("  word order   2/3 -> {:.4}", cc(2, 3));
    eprintln!("  unrelated    1/4 -> {:.4}", cc(1, 4));
    eprintln!("  unrelated    2/5 -> {:.4}", cc(2, 5));
    let c_equiv = cc(0, 1).min(cc(2, 3));
    let c_unrel = cc(1, 4).max(cc(2, 5));
    eprintln!("  centered: worst equivalent {c_equiv:.4} vs best unrelated {c_unrel:.4}");

    // chrF's blind spot, for contrast.
    let f = rlx_translate::score::chrf("J'aime l'été", "J'adore l'été");
    eprintln!("  chrF on the paraphrase pair: {f:.4}");
    eprintln!("  raw: worst equivalent {lo_equiv:.4} vs best unrelated {hi_unrel:.4}");
    assert!(
        lo_equiv > hi_unrel,
        "cosine does not separate paraphrase from unrelated text \
         ({lo_equiv:.4} vs {hi_unrel:.4}); it would be misleading as a metric"
    );
    // The point of centring: the ordering survives either way, but the margin
    // is what makes the number safe to report.
    assert!(
        c_equiv - c_unrel > 5.0 * (lo_equiv - hi_unrel),
        "centring should widen the margin substantially, got {:.4} against {:.4}",
        c_equiv - c_unrel,
        lo_equiv - hi_unrel
    );
    assert!(c_unrel < 0.3, "unrelated text should not read as similar");
}

/// The shipped quality estimator, on the shipped framework's own bad output.
///
/// `qualityEstimator/` is real data in the bundle, and the failures it is meant
/// to catch are ones the OS itself produced — a five-fold repetition and a
/// hallucinated vulgarity. If our reading of the regexes and term lists is
/// right, both are flagged and a clean translation is not.
#[test]
fn the_shipped_quality_estimator_catches_the_real_failures() {
    let Some((home, _, _)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let qe =
        rlx_translate::quality::QualityEstimator::load(&home.join("qualityEstimator"), "en", "fr")
            .expect("quality estimator loads");
    assert!(!qe.is_empty(), "no check was armed from the shipped files");

    // the OS's own output for these two sources, from the reference dump.
    let bad = qe.check("Avocado platypus i love the summer", "J'adore l'été merde");
    eprintln!("  hallucinated profanity -> {bad:?}");
    assert!(!bad.is_empty(), "the shipped vulgar-term list did not fire");

    let loop_flagged = qe.check(
        "the woman of my dreams",
        "La femme de mes rêves rêves rêves rêves rêves",
    );
    eprintln!("  repetition             -> {loop_flagged:?}");
    assert!(
        !loop_flagged.is_empty(),
        "the shipped repeat regex did not fire"
    );

    let clean = qe.check("i love the summer", "J'adore l'été");
    eprintln!("  clean                  -> {clean:?}");
    assert!(
        clean.is_empty(),
        "a correct translation was flagged: {clean:?}"
    );
}

/// Suppression follows the config, not a blanket rule on angle brackets.
///
/// `shortlist-suppress-tokens` is empty for en->fr, so the OS suppresses
/// nothing. Dropping everything angle-bracketed also drops the `<STRUCT_*>`
/// markers the model uses to offer gender alternatives, which is a decision the
/// config did not ask for. The direction tags stay suppressed because emitting
/// `<src-en_US>` mid-sentence is garbage, and `<s>` stays scoreable because it
/// terminates the sequence.
#[test]
fn only_the_configured_tokens_are_suppressed() {
    let Some((home, dirs, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    let assets = rlx_translate::assets::Assets::discover();
    let pair = rlx_translate::quasar::LangPair::parse("en_US-fr_FR").expect("pair");
    let Ok((_, config)) = assets.best_config(&pair) else {
        eprintln!("skipping: no config");
        return;
    };
    let params = rlx_translate::pdec::PDecParams::from_block(
        config
            .mt_app()
            .expect("mt_app")
            .translator(&pair)
            .expect("block"),
    )
    .expect("params");
    assert!(
        params.shortlist.suppress_tokens.is_empty(),
        "this pair is expected to suppress nothing; got {:?}",
        params.shortlist.suppress_tokens
    );

    let extra: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
    let nmt = Nmt::load(&home, &extra, "fr").expect("model loads");
    // Decoding still works, and the direction tags never appear in the output.
    let out = nmt
        .translate_nbest(&vocab, &params, "the accountant is here", 1)
        .expect("decode");
    let text = out.first().map(|v| v.text.clone()).unwrap_or_default();
    eprintln!("  {text:?}");
    assert!(!text.is_empty(), "decoding produced nothing");
    for bad in ["<src-", "<tar-", "-optimal>", "<unk>"] {
        assert!(
            !text.contains(bad),
            "control token {bad} leaked into {text:?}"
        );
    }
}

/// The SentencePiece stages carry tokens, not text.
///
/// `spm_encode` and `spm_decode` are separate stages in the shipped graph because
/// the edges between them carry token arrays. When the executor passed text
/// everywhere they were no-ops; this checks they now do real work and still
/// round-trip, which is the precondition for the alignment blocks.
#[test]
fn the_sentencepiece_stages_round_trip_through_tokens() {
    let Some((_, _, vocab)) = bundle("fr") else {
        eprintln!("skipping: no bundle covering fr");
        return;
    };
    use rlx_translate::execute::{Context, Val, run};
    use rlx_translate::pipeline::{Stage, StageKind, Support, TranslationPlan};
    use rlx_translate::quasar::{BlockKind, LangPair, SpmAction};

    let mk = |name: &str, kind: StageKind, input: &str, spm: Option<SpmAction>| Stage {
        name: name.into(),
        inputs: vec![input.into()],
        roles: vec!["in".into()],
        ports: vec![None],
        kind,
        block: None,
        files: vec![],
        spm,
        pdec: None,
        support: Support::Ready,
    };
    let plan = TranslationPlan {
        task: "mt_app".into(),
        pair: LangPair::parse("en_US-fr_FR").expect("pair"),
        stages: vec![
            mk(
                "enc",
                StageKind::Block(BlockKind::SentencePiece),
                "graph-input",
                Some(SpmAction::Encode),
            ),
            mk(
                "dec",
                StageKind::Block(BlockKind::SentencePiece),
                "enc",
                Some(SpmAction::Decode),
            ),
            mk("out", StageKind::Output, "dec", None),
        ],
        pdec: None,
    };
    let ctx = Context {
        phrasebook: None,
        nmt: None,
        case_locale: None,
        target: "fr_FR".into(),
        quality: None,
        vocab: Some(&vocab),
        source: Some("en_US".into()),
    };
    let text = "La femme de mes rêves";
    let out = run(&plan, &ctx, text).expect("run");
    assert_eq!(
        out.text.as_deref(),
        Some(text),
        "encode/decode must round-trip"
    );

    // And the encode stage really produced tokens, not the string back.
    let encoded = vocab.encode(text);
    assert!(
        encoded.len() > 3,
        "expected several pieces, got {encoded:?}"
    );
    assert_eq!(
        Val::Tokens(encoded.clone()).text(Some(&vocab)),
        text,
        "rendering the tokens must give the text back"
    );
    eprintln!("  {} pieces round-tripped: {text:?}", encoded.len());
}

/// The prefix-state cache must serve exactly what a full replay would.
///
/// `BeamAdapter` caches the decoder state each prefix leaves behind, which took
/// beam search from 10.9 s to 5.5 s. The failure mode of a cache like that is
/// not a crash: it is a stale state, quietly producing a different — and
/// wrong — translation. So run the same sentences both ways and require the
/// hypotheses to agree on text *and* score, not merely to look reasonable.
#[test]
fn the_prefix_cache_agrees_with_replaying_every_prefix() {
    let Some((home, extra, vocab)) = bundle("fr") else {
        eprintln!("skipping: no fr bundle installed");
        return;
    };
    let refs: Vec<&std::path::Path> = extra.iter().map(PathBuf::as_path).collect();
    let Ok(mut nmt) = Nmt::load(&home, &refs, "fr") else {
        eprintln!("skipping: fr model would not load");
        return;
    };
    let assets = Assets::discover();
    let pair = rlx_translate::quasar::LangPair::parse("en_US-fr_FR").expect("pair");
    let Ok((_, config)) = assets.best_config(&pair) else {
        eprintln!("skipping: no config installed");
        return;
    };
    let params = rlx_translate::pdec::PDecParams::from_block(
        config
            .mt_app()
            .expect("mt_app")
            .translator(&pair)
            .expect("block"),
    )
    .expect("params");

    // Short and long, one and three best: the cache is exercised hardest when
    // several live hypotheses share a prefix.
    for text in [
        "the woman of my dreams",
        "she posted the letter before the office closed",
    ] {
        for n in [1usize, 3] {
            nmt.tuning_mut().incremental = true;
            let fast = nmt
                .translate_nbest(&vocab, &params, text, n)
                .expect("cached");
            nmt.tuning_mut().incremental = false;
            let slow = nmt
                .translate_nbest(&vocab, &params, text, n)
                .expect("replayed");
            assert_eq!(
                fast.len(),
                slow.len(),
                "{text:?} n={n}: cache returned {} variants, replay {}",
                fast.len(),
                slow.len()
            );
            for (a, b) in fast.iter().zip(&slow) {
                assert_eq!(a.text, b.text, "{text:?} n={n}: cache diverged from replay");
                assert!(
                    (a.score - b.score).abs() < 1e-4,
                    "{text:?} n={n}: {:?} scored {} cached, {} replayed",
                    a.text,
                    a.score,
                    b.score
                );
            }
        }
    }
    nmt.tuning_mut().incremental = true;
}
