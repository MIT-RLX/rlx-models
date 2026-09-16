//! JSON stdin/stdout embedding helper for denial RAG.
//! Input:  {"texts":["..."], "model_dir":"/path/to/minilm"}
//! Output: {"dim":384, "embeddings":[[...],...]}
//!
//!   cargo run --release -p rlx-embed --example json_embed < req.json

use anyhow::{Context, Result, bail};
use rlx_embed::{BertTokenizer, Pooling, RlxBertModel, embed_with_rlx};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Request {
    texts: Vec<String>,
    model_dir: PathBuf,
    #[serde(default)]
    max_length: Option<usize>,
}

#[derive(Debug, Serialize)]
struct Response {
    dim: usize,
    embeddings: Vec<Vec<f32>>,
    model_dir: String,
}

fn main() -> Result<()> {
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("read stdin")?;
    let req: Request = serde_json::from_str(&buf).context("parse request JSON")?;
    if req.texts.is_empty() {
        serde_json::to_writer(
            std::io::stdout().lock(),
            &Response {
                dim: 0,
                embeddings: vec![],
                model_dir: req.model_dir.display().to_string(),
            },
        )?;
        println!();
        return Ok(());
    }

    let dir = &req.model_dir;
    if !dir.join("config.json").is_file() || !dir.join("model.safetensors").is_file() {
        bail!(
            "model_dir must contain config.json and model.safetensors: {}",
            dir.display()
        );
    }

    let max_len = req.max_length.unwrap_or(256);
    let pooling = if dir.to_string_lossy().to_lowercase().contains("bge") {
        Pooling::Cls
    } else {
        Pooling::Mean
    };

    let tok = BertTokenizer::from_dir(dir, max_len)?;
    let mut model = RlxBertModel::load(
        &dir.join("config.json"),
        dir.join("model.safetensors").to_str().context("utf8")?,
    )?;

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(req.texts.len());
    for chunk in req.texts.chunks(8) {
        let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        embeddings.extend(embed_with_rlx(&mut model, &tok, &refs, pooling)?);
    }

    let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(
        &mut stdout,
        &Response {
            dim,
            embeddings,
            model_dir: dir.display().to_string(),
        },
    )?;
    stdout.write_all(b"\n")?;
    Ok(())
}
