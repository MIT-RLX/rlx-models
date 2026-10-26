//! `RlxBertModel::from_bytes` (a browser's files, in memory) gives the same
//! hidden states as loading the same checkpoint from disk.
//!
//!   RLX_BERT_DIR=<hf bert dir> cargo test -p rlx-embed --release --test bert_from_bytes
//!
//! Skips without RLX_BERT_DIR.
use rlx_embed::RlxBertModel;
use rlx_runtime::Device;

#[test]
fn from_bytes_matches_from_file() {
    let Ok(dir) = std::env::var("RLX_BERT_DIR") else {
        eprintln!("RLX_BERT_DIR unset: skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let (config, weights) = (dir.join("config.json"), dir.join("model.safetensors"));
    let seq = 8;
    let mut a =
        RlxBertModel::load_sized_on(&config, weights.to_str().unwrap(), 1, seq, Device::Cpu)
            .unwrap();
    let mut b = RlxBertModel::from_bytes(
        &std::fs::read(&config).unwrap(),
        std::fs::read(&weights).unwrap(),
        1,
        seq,
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(a.position_offset(), b.position_offset());
    // [CLS] a few word pieces [SEP], then padding
    let ids = [101.0, 7592.0, 2088.0, 3637.0, 102.0, 0.0, 0.0, 0.0];
    let mask = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let tt = [0.0; 8];
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let (ha, hb) = (
        a.forward(&ids, &mask, &tt, &pos),
        b.forward(&ids, &mask, &tt, &pos),
    );
    assert_eq!(ha.len(), seq * a.hidden_size());
    let max = ha
        .iter()
        .zip(&hb)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max < 1e-5, "max |Δ| {max}");
}

/// An int8-packed checkpoint (projections as `DequantMatMul`, embeddings
/// gathered on the host) embeds close to the f32 model, and a short text
/// gives the same vector in its 8-token bucket as padded to 32.
#[test]
fn packed_matches_f32_and_buckets_agree() {
    use rlx_embed::{BertFiles, BertTextEncoder, Pooling};
    let Ok(dir) = std::env::var("RLX_BERT_DIR") else {
        eprintln!("RLX_BERT_DIR unset: skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let read = |f: &str| std::fs::read(dir.join(f)).unwrap_or_else(|_| b"{}".to_vec());
    let (config, tok, tc, sp) = (
        read("config.json"),
        read("tokenizer.json"),
        read("tokenizer_config.json"),
        read("special_tokens_map.json"),
    );
    let f32_weights = std::fs::read(dir.join("model.safetensors")).unwrap();
    let packed = rlx_embed::packed::pack_int8(&f32_weights).unwrap();
    let open = |weights: Vec<u8>| {
        let files = BertFiles {
            config: &config,
            weights,
            tokenizer: &tok,
            tokenizer_config: &tc,
            special_tokens_map: &sp,
        };
        BertTextEncoder::from_bytes(files, Pooling::Cls, 32, Device::Cpu).unwrap()
    };
    let (mut full, mut int8) = (open(f32_weights), open(packed));
    assert_eq!(int8.buckets(), &[8, 16, 32]);
    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    for text in [
        "memory",
        "where is hearing processed in the brain",
        "the blood supply of the primary visual cortex and the arteries that feed it",
    ] {
        let (a, b) = (full.embed(text).unwrap(), int8.embed(text).unwrap());
        assert!(cos(&a, &b) > 0.995, "{text}: cos {}", cos(&a, &b));
    }
    // the bucket does not change the vector: the same tokens through the
    // 8-token graph and, padded, through the 32-token one
    let model8 = RlxBertModel::from_bytes(&config, packed_again(&dir), 1, 8, Device::Cpu).unwrap();
    let mut model32 = model8.with_shape(1, 32).unwrap();
    let mut model8 = model8;
    let ids = [101.0, 3638.0, 102.0];
    let run = |m: &mut RlxBertModel, seq: usize| {
        let mut x = [vec![0f32; seq], vec![0f32; seq], vec![0f32; seq]];
        for (i, &id) in ids.iter().enumerate() {
            x[0][i] = id;
            x[1][i] = 1.0;
        }
        let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        let h = m.forward(&x[0], &x[1], &x[2], &pos);
        h[..m.hidden_size() * ids.len()].to_vec()
    };
    let (a, b) = (run(&mut model8, 8), run(&mut model32, 32));
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max < 1e-4, "bucket 8 vs 32: max |Δ| {max}");
}

fn packed_again(dir: &std::path::Path) -> Vec<u8> {
    rlx_embed::packed::pack_int8(&std::fs::read(dir.join("model.safetensors")).unwrap()).unwrap()
}
