//! Compare greedy backbone audio-token IDs: ORT (KV cache) vs RLX full-recompute.
//!
//! ```bash
//! cargo run -p rlx-soprano --release --example backbone_tok_diag --features apple-silicon
//! ```

use std::path::PathBuf;
use std::process::Command;

use rlx_runtime::{Device, is_available};
use rlx_soprano::{DEFAULT_LOCAL_DIR, InferOpts, NativeSoprano};

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(DEFAULT_LOCAL_DIR);
    let text = std::env::var("RLX_TEXT")
        .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog.".into());
    let n: usize = std::env::var("RLX_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);

    println!("text: {text:?}");
    let ort = ort_tokens(&dir, &text, n)?;
    println!("ORT   n={}: {:?}", ort.len(), &ort[..ort.len().min(32)]);

    let opts = InferOpts {
        max_new_tokens: n.saturating_sub(1),
        greedy: true,
        ..Default::default()
    };

    for (dev, label) in [
        (Device::Cpu, "CPU"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
    ] {
        if dev != Device::Cpu && !is_available(dev) {
            println!("{label}: n/a");
            continue;
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let model = NativeSoprano::open(&dir, dev)?;
            model.generate_audio_tokens(&text, &opts)
        }));
        match r {
            Ok(Ok(toks)) => {
                let match_n = ort
                    .iter()
                    .zip(toks.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                let note = if match_n >= ort.len().min(toks.len()) && toks.len() == ort.len() {
                    "OK".to_string()
                } else if match_n < ort.len().min(toks.len()) {
                    format!(
                        "first_diff@{match_n} ort={} rlx={} (matched {match_n}/{})",
                        ort[match_n],
                        toks[match_n],
                        ort.len().min(toks.len())
                    )
                } else {
                    format!(
                        "len ort={} rlx={} prefix_ok={match_n}",
                        ort.len(),
                        toks.len()
                    )
                };
                println!(
                    "{label:5} n={}: {:?}  {note}",
                    toks.len(),
                    &toks[..toks.len().min(32)]
                );
            }
            Ok(Err(e)) => println!("{label}: err {e}"),
            Err(_) => println!("{label}: panic"),
        }
    }
    Ok(())
}

fn ort_tokens(dir: &PathBuf, text: &str, n: usize) -> anyhow::Result<Vec<i64>> {
    let dir_s = dir.display().to_string();
    let text_s = text.replace('\\', "\\\\").replace('\'', "\\'");
    let py = format!(
        r#"
import json, numpy as np, onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer
dir=Path('{dir_s}')
text='{text_s}'
n={n}
EOS=3
tok=Tokenizer.from_file(str(dir/'tokenizer.json'))
prompt=f'[STOP][TEXT]{{text.strip()}}[START]'
ids=tok.encode(prompt).ids
bb=ort.InferenceSession(str(dir/'onnx/soprano_backbone_kv_fp32.onnx'), providers=['CPUExecutionProvider'])
past={{i:(np.zeros((1,1,0,128),np.float32),np.zeros((1,1,0,128),np.float32)) for i in range(17)}}
feeds_past=lambda:{{**{{f'past_key_values.{{i}}.key':past[i][0] for i in range(17)}},**{{f'past_key_values.{{i}}.value':past[i][1] for i in range(17)}}}}
cur=np.array([ids],np.int64)
attn=np.ones((1,len(ids)),np.int64)
pos=np.arange(len(ids),dtype=np.int64)[None]
out_toks=[]
for step in range(n):
  outs=bb.run(None,{{'input_ids':cur,'attention_mask':attn,'position_ids':pos,**feeds_past()}})
  for i in range(17):
    past[i]=(outs[1+2*i],outs[2+2*i])
  next_id=int(np.argmax(outs[0][0,-1]))
  out_toks.append(next_id)
  if next_id==EOS: break
  total=past[0][0].shape[2]+1
  cur=np.array([[next_id]],np.int64)
  attn=np.ones((1,total),np.int64)
  pos=np.array([[total-1]],np.int64)
print(json.dumps(out_toks))
"#
    );
    let out = Command::new("python3").arg("-c").arg(py).output()?;
    if !out.status.success() {
        anyhow::bail!("ort failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let s = String::from_utf8(out.stdout)?;
    let v: Vec<i64> = serde_json::from_str(s.trim())?;
    Ok(v)
}
