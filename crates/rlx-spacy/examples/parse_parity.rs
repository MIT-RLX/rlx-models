//! Dependency parse vs spaCy (`tribev2-ref/spacy/dump_t2v.py` parse.json):
//! class order, heads, labels, sentence starts and `doc.sents` char spans.
//! usage: parse_parity MODEL_DIR REF_DIR [LIMIT]
use serde_json::Value;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let md = std::path::Path::new(&a[1]);
    let tok = rlx_spacy::Tokenizer::load(&md.join("tokenizer"))?;
    let t2v = rlx_spacy::Tok2Vec::load(md)?;
    let parser = rlx_spacy::Parser::load(md)?;
    let classes: Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{}/parser_classes.json",
        a[2]
    ))?)?;
    let want: Vec<String> = classes["classes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let got = parser.class_names();
    println!(
        "classes: {} vs {} ({})",
        got.len(),
        want.len(),
        if got == want { "identical" } else { "DIFFER" }
    );
    let cases: Vec<(String, Value)> =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/tok.json", a[2]))?)?;
    let refs: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/parse.json", a[2]))?)?;
    let limit: usize = a.get(3).map_or(usize::MAX, |s| s.parse().unwrap());
    let (mut docs, mut ok, mut hbad, mut dbad, mut sbad, mut sent_bad, mut ntok) =
        (0, 0, 0, 0, 0, 0, 0);
    // per-step score bits (dump_parse_scores.py) for the first docs
    let f32s = |p: String| -> Vec<f32> {
        std::fs::read(p)
            .map(|b| {
                b.chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .unwrap_or_default()
    };
    let (rtv, rsc) = (
        f32s(format!("{}/scores/tokvecs.f32", a[2])),
        f32s(format!("{}/scores/scores.f32", a[2])),
    );
    let nsc: usize = std::fs::read_to_string(format!("{}/scores/steps.json", a[2]))
        .ok()
        .map_or(0, |s| serde_json::from_str::<Vec<usize>>(&s).unwrap().len());
    let (mut tv_at, mut sc_at, mut tv_bad, mut sc_bad) = (0, 0, 0, 0);
    for (text, _) in cases.iter().take(nsc) {
        let toks = tok.tokenize(text);
        let (_, tv, sc) = parser.parse_traced(&toks, &t2v.forward(&toks));
        tv_bad += tv
            .iter()
            .zip(&rtv[tv_at..])
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        sc_bad += sc
            .iter()
            .zip(&rsc[sc_at..])
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        tv_at += tv.len();
        sc_at += sc.len();
    }
    if nsc > 0 {
        println!(
            "{nsc} docs: tokvecs {tv_bad}/{tv_at} values differ ({} ref); step scores {sc_bad}/{sc_at} differ ({} ref)",
            rtv.len(),
            rsc.len()
        );
    }
    let t0 = std::time::Instant::now();
    for ((text, _), r) in cases.iter().zip(&refs).take(limit) {
        let toks = tok.tokenize(text);
        let p = parser.parse(&t2v, &toks);
        let heads: Vec<usize> = r["heads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let deps: Vec<&str> = r["deps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let ss: Vec<bool> = r["sent_starts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let sents: Vec<(usize, usize)> = r["sents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let my_sents: Vec<(usize, usize)> = p
            .sents()
            .iter()
            .map(|s| {
                let last = &toks[s.end - 1];
                (toks[s.start].idx, last.idx + last.text.chars().count())
            })
            .collect();
        let h = p.heads.iter().zip(&heads).filter(|(x, y)| x != y).count();
        let d = p.deps.iter().zip(&deps).filter(|(x, y)| x != y).count();
        let s = p
            .sent_starts
            .iter()
            .zip(&ss)
            .filter(|(x, y)| x != y)
            .count();
        let sb = (my_sents != sents) as usize;
        if h + d + s + sb > 0 && docs - ok < 5 {
            eprintln!(
                "doc {docs}: {h} heads, {d} deps, {s} sent starts differ; sents {my_sents:?} vs {sents:?}"
            );
        }
        ok += (h + d + s + sb == 0) as usize;
        hbad += h;
        dbad += d;
        sbad += s;
        sent_bad += sb;
        ntok += toks.len();
        docs += 1;
    }
    println!(
        "{docs} docs, {ntok} tokens in {:.2}s: {ok} identical; heads {hbad}, deps {dbad}, sent starts {sbad} differ; doc.sents differ in {sent_bad} docs",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
