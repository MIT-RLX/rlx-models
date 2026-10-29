//! Stage timings of tokenizer / tok2vec / parser over tok.json.
//! usage: spacy_bench MODEL_DIR TOK.json
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let md = std::path::Path::new(&a[1]);
    let t = std::time::Instant::now();
    let tok = rlx_spacy::Tokenizer::load(&md.join("tokenizer"))?;
    let t2v = rlx_spacy::Tok2Vec::load(md)?;
    let parser = rlx_spacy::Parser::load(md)?;
    println!("load {:.2}s", t.elapsed().as_secs_f64());
    let cases: Vec<(String, serde_json::Value)> =
        serde_json::from_str(&std::fs::read_to_string(&a[2])?)?;
    let (mut tt, mut tv, mut tp, mut n) = (0f64, 0f64, 0f64, 0);
    for (text, _) in &cases {
        let t = std::time::Instant::now();
        let toks = tok.tokenize(text);
        tt += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let v = t2v.forward(&toks);
        tv += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let p = parser.parse_tokvecs(&toks, &v);
        tp += t.elapsed().as_secs_f64();
        n += p.heads.len();
    }
    println!("{n} tokens: tokenizer {tt:.2}s, tok2vec {tv:.2}s, parser {tp:.2}s");
    Ok(())
}
