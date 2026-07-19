//! Detokenize argv token ids via the GGUF-embedded BPE (debug helper).
//!
//! ```text
//! cargo run -p rlx-qwen35 --example decode_ids --release -- \
//!   weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf 11751,25,271
//! ```

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf = args.next().expect("usage: decode_ids <gguf> <id,id,...>");
    let ids_raw = args.next().expect("token id list");
    let ids: Vec<u32> = ids_raw
        .split(',')
        .map(|s| s.trim().parse().expect("u32"))
        .collect();
    let text =
        rlx_qwen35::decode_ids_from_gguf(std::path::Path::new(&gguf), &ids, true).expect("decode");
    println!("{text}");
}
