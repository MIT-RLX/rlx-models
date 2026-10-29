//! `Tokenizer::tokenize` vs spaCy (`tribev2-ref/spacy/dump_tok.py`).
//! usage: tok_parity MODEL_DIR TOK.json
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t = rlx_spacy::Tokenizer::load(&std::path::Path::new(&a[1]).join("tokenizer"))?;
    let cases: Vec<(String, Vec<(String, String)>)> =
        serde_json::from_str(&std::fs::read_to_string(&a[2])?)?;
    let (mut bad, mut toks) = (0, 0);
    for (text, want) in &cases {
        let got: Vec<(String, String)> = t
            .tokenize(text)
            .into_iter()
            .map(|x| {
                (
                    x.text,
                    if x.whitespace {
                        " ".into()
                    } else {
                        String::new()
                    },
                )
            })
            .collect();
        toks += want.len();
        if &got != want {
            bad += 1;
            if bad <= 6 {
                let k = got
                    .iter()
                    .zip(want)
                    .position(|(g, w)| g != w)
                    .unwrap_or(got.len().min(want.len()));
                eprintln!(
                    "{text:?}\n  at {k}: rust {:?}\n        spacy {:?}",
                    &got[k.saturating_sub(1)..(k + 3).min(got.len())],
                    &want[k.saturating_sub(1)..(k + 3).min(want.len())]
                );
            }
        }
    }
    println!(
        "{} / {} texts tokenized identically ({toks} tokens)",
        cases.len() - bad,
        cases.len()
    );
    Ok(())
}
