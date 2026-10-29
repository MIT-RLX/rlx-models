//! tok2vec stages vs spaCy (`tribev2-ref/spacy/dump_t2v.py`): feature ids
//! (exact), embed output and tok2vec output (bits / max |d|).
//! usage: t2v_parity MODEL_DIR REF_DIR
fn f32s(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let md = std::path::Path::new(&a[1]);
    let tok = rlx_spacy::Tokenizer::load(&md.join("tokenizer"))?;
    let t2v = rlx_spacy::Tok2Vec::load(md)?;
    let cases: Vec<(String, serde_json::Value)> =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/tok.json", a[2]))?)?;
    let feats: Vec<u64> = std::fs::read(format!("{}/feats.u64", a[2]))?
        .chunks(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let (emb, vec) = (
        f32s(&format!("{}/embed.f32", a[2])),
        f32s(&format!("{}/t2v.f32", a[2])),
    );
    let (mut at, mut fbad, mut ebits, mut vbits, mut n) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut emax, mut vmax) = (0f32, 0f32);
    let limit: usize = a.get(3).map_or(usize::MAX, |s| s.parse().unwrap());
    for (text, _) in cases.iter().take(limit) {
        let toks = tok.tokenize(text);
        let f = t2v.features(&toks);
        for (i, row) in f.iter().enumerate() {
            for c in 0..6 {
                if row[c] != feats[(at + i) * 6 + c] {
                    fbad += 1;
                    if fbad <= 5 {
                        eprintln!(
                            "feature {c} of {:?}: {} vs {}",
                            toks[i].text,
                            row[c],
                            feats[(at + i) * 6 + c]
                        );
                    }
                }
            }
        }
        let e = t2v.embed(&toks);
        let v = t2v.forward(&toks);
        for (j, (x, y)) in e
            .iter()
            .zip(&emb[at * 96..(at + toks.len()) * 96])
            .enumerate()
        {
            let _ = j;
            ebits += (x.to_bits() != y.to_bits()) as usize;
            emax = emax.max((x - y).abs());
        }
        for (x, y) in v.iter().zip(&vec[at * 96..(at + toks.len()) * 96]) {
            vbits += (x.to_bits() != y.to_bits()) as usize;
            vmax = vmax.max((x - y).abs());
        }
        at += toks.len();
        n += toks.len() * 96;
    }
    println!(
        "{at} tokens: feature mismatches {fbad}; embed {ebits}/{n} values differ (max |d| {emax:.2e}); tok2vec {vbits}/{n} differ (max |d| {vmax:.2e})"
    );
    Ok(())
}
