// Semantic sanity check for a BERT-style embedding model.
// Embeds a few sentences and prints the cosine-similarity matrix.
// Related pairs should score clearly higher than unrelated pairs.
//   cargo run --release -p rlx-embed --example semantic_check -- <model_dir>
use rlx_embed::{BertTokenizer, Pooling, RlxBertModel, embed_with_rlx};
use std::path::Path;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-8)
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: semantic_check <model_dir>");
    let dir = Path::new(&dir);
    let pooling = if dir.to_string_lossy().to_lowercase().contains("bge") {
        Pooling::Cls
    } else {
        Pooling::Mean
    };

    let tok = BertTokenizer::from_dir(dir, 64)?;
    let mut model = RlxBertModel::load(
        &dir.join("config.json"),
        dir.join("model.safetensors").to_str().unwrap(),
    )?;

    let texts = [
        "A man is playing a guitar.",                  // 0
        "Someone is strumming an acoustic guitar.",    // 1  ~ 0
        "It is freezing outside and snow is falling.", // 2
        "The weather is cold and it is snowing.",      // 3  ~ 2
    ];
    let vecs = embed_with_rlx(&mut model, &tok, &texts, pooling)?;
    println!("dim={} pooling={:?}", vecs[0].len(), pooling);

    println!("\ncosine matrix:");
    print!("      ");
    for j in 0..texts.len() {
        print!("  s{j:<5}");
    }
    println!();
    for i in 0..texts.len() {
        print!("s{i}  ");
        for j in 0..texts.len() {
            print!("  {:.3} ", cosine(&vecs[i], &vecs[j]));
        }
        println!();
    }

    let sim_related = (cosine(&vecs[0], &vecs[1]) + cosine(&vecs[2], &vecs[3])) / 2.0;
    let sim_unrelated = (cosine(&vecs[0], &vecs[2])
        + cosine(&vecs[0], &vecs[3])
        + cosine(&vecs[1], &vecs[2])
        + cosine(&vecs[1], &vecs[3]))
        / 4.0;
    println!("\nmean related  (0-1, 2-3): {sim_related:.3}");
    println!("mean unrelated (cross)  : {sim_unrelated:.3}");
    if sim_related > sim_unrelated + 0.05 {
        println!("PASS: related pairs are clearly more similar than unrelated pairs.");
    } else {
        println!("FAIL: semantic ordering not observed.");
        std::process::exit(1);
    }
    Ok(())
}
