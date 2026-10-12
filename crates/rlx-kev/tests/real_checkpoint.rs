//! Reading a **released** kev checkpoint: `head.pt`, `adapter_config.json`
//! and `adapter_model.safetensors` as jaredpalmer actually publishes them.
//!
//! Synthetic fixtures prove the code does what I wrote; only the real files
//! prove I read the format right. `head.pt` is a `torch.save` of a plain
//! dict, not a `state_dict`, and the calibration temperature lives nowhere
//! else — so this test exists mostly to prove that number comes out.
//!
//! Gated on `KEV_RUN_DIR`, pointing at a directory holding those three files
//! (~135 MB for kev-4b; no base model needed):
//!
//! ```text
//! KEV_RUN_DIR=/path/to/kev-4b cargo test -p rlx-kev --test real_checkpoint
//! ```

use std::path::PathBuf;

use rlx_kev::api::{SystemOneRequest, to_record};
use rlx_kev::checkpoint::Checkpoint;
use rlx_kev::encode::{EncodeOpts, Encoder, HfTokenizer};
use rlx_kev::lora::{LoraAdapter, module_path_for_gguf};

fn run_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("KEV_RUN_DIR")?);
    p.join("head.pt").exists().then_some(p)
}

macro_rules! run_dir_or_skip {
    () => {
        match run_dir() {
            Some(d) => d,
            None => {
                eprintln!("skipping: set KEV_RUN_DIR to a kev checkpoint directory");
                return;
            }
        }
    };
}

#[test]
fn head_pt_yields_the_metadata_and_a_fitted_temperature() {
    let dir = run_dir_or_skip!();
    let ckpt = Checkpoint::open(&dir).expect("opens head.pt");
    let m = ckpt.meta();

    assert!(
        m.base.starts_with("Qwen/"),
        "base should name a Hub model, got {:?}",
        m.base
    );
    assert_eq!(m.head_dim, 256, "released checkpoints use dp = 256");
    assert_eq!(m.lora, 16, "released checkpoints use rank 16");
    assert!(
        !m.special_embeddings,
        "released adapters do not train tokens"
    );

    // The whole reason head.pt is read natively. Released checkpoints fit a
    // temperature around 2.1-2.4; anything at exactly 1.0 would mean the
    // calibration silently did not load.
    assert!(
        (1.5..4.0).contains(&m.temperature),
        "temperature {} is outside the range a fitted checkpoint should carry",
        m.temperature
    );
}

#[test]
fn the_pointer_head_loads_at_the_backbone_width() {
    let dir = run_dir_or_skip!();
    let ckpt = Checkpoint::open(&dir).expect("opens");
    let head = ckpt.pointer_head().expect("reads the pointer head");

    assert_eq!(head.head_dim, 256);
    assert_eq!(head.q_weight.len(), head.head_dim * head.hidden);
    assert_eq!(head.k_weight.len(), head.head_dim * head.hidden);
    assert_eq!(head.q_bias.len(), head.head_dim);
    assert_eq!(head.temperature, ckpt.meta().temperature);

    // A head read at the wrong offset would come back all-zero or wild.
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let q_rms = rms(&head.q_weight);
    assert!(
        q_rms > 1e-4 && q_rms < 10.0,
        "q.weight RMS {q_rms} does not look like trained values"
    );
    assert!(
        head.q_weight.iter().all(|x| x.is_finite()),
        "q.weight has non-finite values"
    );
}

/// Every module the released adapter targets must map back from a GGUF
/// tensor name, or those layers would silently run un-adapted.
#[test]
fn every_adapter_module_is_reachable_from_a_gguf_name() {
    let dir = run_dir_or_skip!();
    let ad = LoraAdapter::open(&dir).expect("opens the adapter");
    assert_eq!(ad.r, 16);
    assert_eq!(ad.scale(), 2.0, "alpha 32 / r 16");

    let mut reachable = std::collections::HashSet::new();
    // The tensor names a Qwen3.5 build asks for, across enough layers to
    // cover both block kinds.
    for layer in 0..48 {
        for tail in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "attn_qkv",
            "attn_gate",
            "ssm_alpha",
            "ssm_beta",
            "ssm_out",
        ] {
            if let Some(m) = module_path_for_gguf(&format!("blk.{layer}.{tail}.weight")) {
                reachable.insert(m);
            }
        }
    }

    let missing: Vec<&str> = ad.modules().filter(|m| !reachable.contains(*m)).collect();
    assert!(
        missing.is_empty(),
        "{} adapter module(s) unreachable from any GGUF name, e.g. {:?}",
        missing.len(),
        &missing[..missing.len().min(5)]
    );
    assert!(
        ad.len() > 100,
        "expected a full-model adapter, got {}",
        ad.len()
    );
}

/// **Encoder parity against the reference.**
///
/// `tests/fixtures/golden_encoding.json` holds a request and the encoding
/// that `kev/model.py::encode` produces for it under the real Qwen3.5
/// tokenizer (see `scripts/kev_golden_encoding.py`). The request is chosen to
/// hit the parts that are easy to get subtly wrong: a `<|fim_prefix|>` buried
/// in the state, a `null` option description, an `instructions` field that is
/// a nested object rather than a string, and all three question types.
///
/// This is the strongest check available without downloading a base model —
/// if the ids drift, the model is being asked a different question than it
/// was trained on, and nothing downstream would notice.
#[test]
fn encoding_is_identical_to_the_reference() {
    let dir = run_dir_or_skip!();
    let tok_path = dir.join("tokenizer.json");
    if !tok_path.exists() {
        eprintln!("skipping: {} has no tokenizer.json", dir.display());
        return;
    }

    // Parse the fixture with the crate's own order-preserving value type.
    // Going through `serde_json::Value` here would sort `criteria` and the
    // nested `instructions` object before the encoder ever saw them, and the
    // resulting mismatch would look like a port bug rather than a harness
    // one — which is exactly what happened the first time this was written.
    let fixture = rlx_kev::json::Json::parse(include_str!("fixtures/golden_encoding.json"))
        .expect("fixture parses");
    let req_text =
        serde_json::to_string(fixture.get("request").expect("request field")).expect("request");
    let req = SystemOneRequest::parse(&req_text).expect("request parses");
    let want_json = fixture.get("expected").expect("expected field");

    let encoder = Encoder::new(HfTokenizer::from_file(&tok_path).expect("tokenizer"))
        .expect("delimiters resolve in the real Qwen3.5 vocabulary");
    let (rec, _meta) = to_record(&req).expect("record");
    let enc = encoder
        .encode(
            &rec,
            EncodeOpts {
                max_state: 8192,
                max_branch: 8192,
                strict: false,
                option_isolation: false,
            },
        )
        .expect("encodes");

    let want: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(want_json).expect("expected"))
            .expect("expected re-parses");
    let as_u = |v: &serde_json::Value| -> Vec<u64> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_u64().expect("uint"))
            .collect()
    };
    let as_i = |v: &serde_json::Value| -> Vec<i64> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_i64().expect("int"))
            .collect()
    };

    assert_eq!(
        enc.ids.iter().map(|x| *x as u64).collect::<Vec<_>>(),
        as_u(&want["ids"]),
        "token ids differ from the reference encoder"
    );
    assert_eq!(
        enc.seg.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        as_i(&want["seg"])
    );
    assert_eq!(
        enc.pos.iter().map(|x| *x as u64).collect::<Vec<_>>(),
        as_u(&want["pos"])
    );
    assert_eq!(
        enc.opt.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        as_i(&want["opt"])
    );
    assert_eq!(
        enc.decide_idx.iter().map(|x| *x as u64).collect::<Vec<_>>(),
        as_u(&want["decide_idx"])
    );
    let want_opt_idx: Vec<Vec<u64>> = want["opt_idx"]
        .as_array()
        .expect("array")
        .iter()
        .map(&as_u)
        .collect();
    assert_eq!(
        enc.opt_idx
            .iter()
            .map(|v| v.iter().map(|x| *x as u64).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        want_opt_idx
    );
    assert_eq!(
        enc.state_truncated,
        want["state_truncated"].as_bool().expect("bool")
    );
}
