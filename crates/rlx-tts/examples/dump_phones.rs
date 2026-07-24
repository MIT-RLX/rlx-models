//! Print Hydra phone sequences for CLI args (debug G2P / lexicon).
//!
//! ```bash
//! cargo run -p rlx-tts --example dump_phones --release -- "The quick brown fox"
//! ```

use anyhow::Result;
use rlx_tts::RlxTts;
use rlx_tts::frontend::TextFrontend;

fn main() -> Result<()> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello from RLX.".into());
    let tts = RlxTts::open_default()?;
    let phones = tts.frontend().text_to_phones(&text)?;
    println!("text:   {text}");
    println!("phones: {}", phones.join(" "));
    Ok(())
}
