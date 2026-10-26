//! `rlx-gemini-nano` — locate and inspect Chrome's Gemini Nano `weights.bin`.
//!
//! ```text
//! rlx-gemini-nano locate
//! rlx-gemini-nano inspect [PATH] [--tensors N] [--json]
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rlx_gemini_nano::{TfliteModel, inspect, locate_chrome_weights};

const USAGE: &str =
    "usage: rlx-gemini-nano locate\n       rlx-gemini-nano inspect [PATH] [--tensors N] [--json]";

fn default_weights() -> Result<PathBuf> {
    locate_chrome_weights().into_iter().next().context(
        "no Gemini Nano weights.bin found; enable Chrome's built-in AI \
         (chrome://flags: optimization-guide-on-device-model, prompt-api-for-gemini-nano) \
         and wait for chrome://on-device-internals to report the model as ready",
    )
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("locate") => {
            let found = locate_chrome_weights();
            if found.is_empty() {
                bail!("{}", default_weights().unwrap_err());
            }
            for p in found {
                println!("{}", p.display());
            }
        }
        Some("inspect") => {
            let mut path = None;
            let mut tensors = 200usize;
            let mut json = false;
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--json" => json = true,
                    "--tensors" => {
                        tensors = it
                            .next()
                            .context("--tensors needs a value")?
                            .parse()
                            .context("--tensors")?
                    }
                    flag if flag.starts_with("--") => bail!("unknown flag {flag}\n{USAGE}"),
                    p => path = Some(PathBuf::from(p)),
                }
            }
            let path = match path {
                Some(p) => p,
                None => default_weights()?,
            };
            let model = TfliteModel::open(&path)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&inspect::manifest_json(&model))?
                );
            } else {
                println!("{}", path.display());
                print!("{}", inspect::report(&model, tensors)?);
            }
        }
        _ => bail!("{USAGE}"),
    }
    Ok(())
}
