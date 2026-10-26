// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use anyhow::{Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_jina_v5::{JinaV5, TASKS, cosine, truncate};

const HELP: &str = "\
rlx-jina-v5 — jina-embeddings-v5-text-nano text embeddings

USAGE:
  rlx-jina-v5 --weights <DIR> --text \"…\" [--text \"…\" …] [options]

Prints one embedding per --text as a JSON array, then the cosine matrix
on stderr when more than one text is given.

OPTIONS:
  --weights <DIR>     jinaai/jina-embeddings-v5-text-nano checkpoint directory
  --task <NAME>       retrieval (default) | text-matching | clustering | classification
  --text <TEXT>       text to embed (repeatable)
  --prompt <NAME>     named prompt: query (default) | document
  --raw               no prompt prefix
  --dim <N>           Matryoshka width: 768 (default) | 512 | 256 | 128 | 64 | 32
  --device <NAME>     cpu | metal | mlx | cuda | rocm | gpu | vulkan (default cpu)
  -h, --help          show this help
";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights = None;
    let mut texts = Vec::new();
    let mut prompt_name = "query".to_string();
    let mut task = TASKS[0].to_string();
    let mut raw = false;
    let mut dim = None;
    let mut device = "cpu".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(&args, &mut i)?),
            "--text" => texts.push(req(&args, &mut i)?),
            "--prompt" => prompt_name = req(&args, &mut i)?,
            "--task" => task = req(&args, &mut i)?,
            "--raw" => {
                raw = true;
                i += 1;
            }
            "--dim" => dim = Some(req(&args, &mut i)?.parse::<usize>()?),
            "--device" => device = req(&args, &mut i)?,
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            other => bail!("unknown flag: {other} (try --help)"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    if texts.is_empty() {
        bail!("give at least one --text");
    }
    let device = parse_standard_device("jina-v5", &device)?;
    let mut model = JinaV5::open(&weights, &task, device)?;
    let prompt = if raw {
        String::new()
    } else {
        model
            .prompt(&prompt_name)
            .ok_or_else(|| anyhow!("no prompt named {prompt_name:?} in the checkpoint"))?
            .to_string()
    };
    let dim = dim.unwrap_or(model.dim());

    let mut out = Vec::with_capacity(texts.len());
    for t in &texts {
        let t0 = std::time::Instant::now();
        let e = truncate(&model.embed_text(&prompt, t)?, dim);
        eprintln!("[rlx-jina-v5] {:?} in {:.2?}", t, t0.elapsed());
        println!("{}", serde_json::to_string(&e)?);
        out.push(e);
    }
    if out.len() > 1 {
        for (i, a) in out.iter().enumerate() {
            let row: Vec<String> = out.iter().map(|b| format!("{:.4}", cosine(a, b))).collect();
            eprintln!("[{i}] {}", row.join("  "));
        }
    }
    Ok(())
}
