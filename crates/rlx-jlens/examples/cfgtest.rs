fn main() {
    let p = std::path::Path::new("/Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct/config.json");
    match rlx_qwen3::Qwen3Config::from_file(p) {
        Ok(c) => println!(
            "OK hidden={} layers={} heads={} kv={} head_dim={} vocab={} tied={}",
            c.hidden_size,
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.vocab_size,
            c.tie_word_embeddings
        ),
        Err(e) => println!("ERR {e}"),
    }
}
