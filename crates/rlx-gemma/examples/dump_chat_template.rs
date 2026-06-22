use anyhow::Result;
use rlx_gguf::{GgufFile, MetaValue};
use std::path::PathBuf;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: dump_chat_template <model.gguf>")
        .into();
    let raw = GgufFile::from_path(&path)?;
    let Some(MetaValue::String(tmpl)) = raw.metadata.get("tokenizer.chat_template") else {
        anyhow::bail!("no tokenizer.chat_template");
    };
    let out = path.with_extension("chat_template.jinja");
    std::fs::write(&out, tmpl)?;
    eprintln!("wrote {} ({} bytes)", out.display(), tmpl.len());
    for (i, line) in tmpl.lines().enumerate().take(40) {
        println!("{:4}|{line}", i + 1);
    }
    Ok(())
}
