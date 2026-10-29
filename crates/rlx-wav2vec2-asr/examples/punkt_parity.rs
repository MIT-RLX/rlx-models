//! Compare `punkt::Punkt::span_tokenize` with nltk's spans.
//! usage: punkt_parity CASES.json   ([[text, [[start, end], ...]], ...])
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cases: Vec<(String, Vec<(usize, usize)>)> =
        serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let p = rlx_wav2vec2_asr::punkt::Punkt::english();
    let mut bad = 0;
    for (t, want) in &cases {
        let got = p.span_tokenize(t);
        if &got != want {
            bad += 1;
            if bad <= 8 {
                eprintln!("{t:?}\n  nltk {want:?}\n  rust {got:?}");
            }
        }
    }
    println!("{} / {} match", cases.len() - bad, cases.len());
}
