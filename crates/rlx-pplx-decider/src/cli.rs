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

//! `rlx-pplx-decider` command line.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use rlx_runtime::Device;

use crate::{Json, OpenOptions, PplxDecider};

const USAGE: &str = "usage: rlx-pplx-decider --weights FILE.gguf|DIR [--device DEV] \
[--tokenizer tokenizer.json] [--mmproj FILE.gguf] [--max-length N] [--max-pixels N] [--temperature T] \
(--request FILE|- | --state TEXT | --state-json JSON) [--questions JSON | --questions-file PATH] \
[--image PATH]... [--logits] [--prompt] [--timing]

  --request FILE   one request object, or a list of them (`-` reads stdin)
  --state/--questions build a single request from the command line
  --image PATH     attach an image to that request (repeatable; PNG/JPEG/…)
  --mmproj FILE    vision projector (default: the mmproj-*.gguf beside the weights)
  --logits         print raw readout logits (valid options, before temperature)
  --prompt         print the rendered prompt and its token ids, then exit";

pub fn run(argv: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut opts = OpenOptions::default();
    let mut temperature: Option<f64> = None;
    let mut request: Option<Json> = None;
    let mut state: Option<Json> = None;
    let mut questions: Option<Json> = None;
    let mut images: Vec<Json> = Vec::new();
    let (mut logits, mut prompt, mut timing) = (false, false, false);
    let mut it = argv.iter();
    let value = |it: &mut std::slice::Iter<'_, String>, flag: &str| -> Result<String> {
        it.next()
            .cloned()
            .ok_or_else(|| anyhow!("{flag} needs a value\n{USAGE}"))
    };
    let parse =
        |text: &str, what: &str| Json::parse(text).with_context(|| format!("parsing {what}"));
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--weights" => weights = Some(value(&mut it, flag)?.into()),
            "--device" => device = rlx_cli::parse_device(&value(&mut it, flag)?)?,
            "--tokenizer" => opts.tokenizer = Some(value(&mut it, flag)?.into()),
            "--mmproj" => opts.mmproj = Some(value(&mut it, flag)?.into()),
            "--max-pixels" => opts.max_pixels = Some(value(&mut it, flag)?.parse()?),
            "--image" => images.push(Json::Str(value(&mut it, flag)?)),
            "--max-length" => opts.max_length = Some(value(&mut it, flag)?.parse()?),
            "--temperature" => temperature = Some(value(&mut it, flag)?.parse()?),
            "--request" => {
                let path = value(&mut it, flag)?;
                let text = if path == "-" {
                    std::io::read_to_string(std::io::stdin())?
                } else {
                    std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?
                };
                request = Some(parse(&text, "--request")?);
            }
            "--state" => state = Some(Json::Str(value(&mut it, flag)?)),
            "--state-json" => state = Some(parse(&value(&mut it, flag)?, "--state-json")?),
            "--questions" => questions = Some(parse(&value(&mut it, flag)?, "--questions")?),
            "--questions-file" => {
                let path = value(&mut it, flag)?;
                let text =
                    std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
                questions = Some(parse(&text, "--questions-file")?);
            }
            "--logits" => logits = true,
            "--prompt" => prompt = true,
            "--timing" => timing = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required\n{USAGE}"))?;
    let requests: Vec<Json> = match (request, state, questions) {
        (Some(Json::Array(list)), None, None) => list,
        (Some(r), None, None) => vec![r],
        (None, Some(s), Some(q)) => {
            let mut r = vec![("state".into(), s), ("questions".into(), q)];
            if !images.is_empty() {
                r.push(("images".into(), Json::Array(std::mem::take(&mut images))));
            }
            vec![Json::Object(r)]
        }
        _ => bail!("pass either --request, or --state/--state-json with --questions\n{USAGE}"),
    };
    if !images.is_empty() {
        bail!(
            "--image attaches to a --state request; put \"images\" in the --request JSON instead"
        );
    }
    let batch = requests.len() > 1;

    let t = Instant::now();
    let mut d = PplxDecider::open_with(&weights, device, &opts)?;
    if timing {
        eprintln!(
            "[pplx-decider] load {:.2?} on {:?}",
            t.elapsed(),
            d.device()
        );
    }
    let mut out = Vec::with_capacity(requests.len());
    for r in &requests {
        let req = d.request(r)?;
        if prompt {
            let enc = d.encode(&req)?;
            println!("{}", d.prompt_text(&req));
            println!("{:?}", enc.input_ids);
            println!("markers {:?}", enc.markers);
            continue;
        }
        let t = Instant::now();
        let res = if logits {
            Json::Object(
                d.logits(&req)?
                    .into_iter()
                    .map(|q| {
                        let v = q
                            .logits
                            .iter()
                            .map(|&x| Json::Float(f64::from(x)))
                            .collect();
                        (q.id, Json::Array(v))
                    })
                    .collect(),
            )
        } else {
            d.decide(&req, temperature)?
        };
        if timing {
            eprintln!(
                "[pplx-decider] request {:.2?} (compiled lengths {:?})",
                t.elapsed(),
                d.compiled_lengths()
            );
        }
        out.push(res);
    }
    if prompt {
        return Ok(());
    }
    let res = if batch {
        Json::Array(out)
    } else {
        out.remove(0)
    };
    println!("{}", res.to_python_json(Some(2)));
    Ok(())
}
