// Prefill-logit parity: native rlx-qwen3 backbone vs the onnx-imported backbone.
// Milestone 1 of the ort-free soprano rewrite (see src/native_qwen3.rs).
//
//   SOPRANO_BACKBONE_ST=~/.cache/huggingface/.../model.safetensors \
//   cargo test -p rlx-soprano --test native_qwen3_parity -- --nocapture
//
// Skips (not fails) when the ekwek/Soprano-1.1-80M safetensors isn't provided.

use std::path::Path;

use rlx_soprano::native_qwen3::SopranoQwen3;
use rlx_soprano::{InferOpts, NativeSoprano, parse_device};

fn setup() -> Option<(NativeSoprano, SopranoQwen3)> {
    let st = std::env::var("SOPRANO_BACKBONE_ST").ok()?;
    let dir = std::env::var("RLX_SOPRANO_DIR").unwrap_or_else(|_| "weights/tts/soprano".into());
    let device =
        parse_device(&std::env::var("RLX_SOPRANO_DEVICE").unwrap_or_else(|_| "cpu".into())).unwrap();
    let onnx = NativeSoprano::open(&dir, device).expect("open onnx backbone");
    let native = SopranoQwen3::open(Path::new(&st), device).expect("open native backbone");
    Some((onnx, native))
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

#[test]
fn prefill_logit_parity() {
    let Some((onnx, native)) = setup() else {
        eprintln!("SKIP: set SOPRANO_BACKBONE_ST to ekwek/Soprano-1.1-80M model.safetensors");
        return;
    };
    for text in ["Hello from Soprano.", "The quick brown fox."] {
        let ids = onnx.encode_prompt(text).expect("encode");
        let lo = onnx.prefill_logits(text).expect("onnx prefill");
        let ln = native.prefill_logits(&ids).expect("native prefill");
        assert_eq!(lo.len(), ln.len(), "vocab mismatch");
        let (ao, an) = (argmax(&lo), argmax(&ln));
        let cos = cosine(&lo, &ln);
        println!("[{text}] onnx_argmax={ao} native_argmax={an} cos={cos:.5} n_tok={}", ids.len());
        assert!(cos > 0.99, "logit cosine {cos} too low for '{text}'");
        assert_eq!(ao, an, "argmax mismatch for '{text}'");
    }
}

/// M2: greedy AR token stream must match the ONNX backbone (bit-exact logits →
/// identical argmax at every step). ONNX caps ids at 128; compare the overlap.
#[test]
fn greedy_token_parity() {
    let Some((onnx, native)) = setup() else {
        eprintln!("SKIP: set SOPRANO_BACKBONE_ST");
        return;
    };
    let text = "Hello from Soprano.";
    let ids = onnx.encode_prompt(text).expect("encode");
    let opts = InferOpts { greedy: true, max_new_tokens: 40, ..Default::default() };
    let (ol, otoks) = onnx.generate_latents(text, &opts).expect("onnx generate");
    let (nl, ntoks) = native.generate_latents_greedy(&ids, 40).expect("native generate");
    let n = otoks.len().min(ntoks.len()).min(24);
    println!("onnx   toks[..{n}] = {:?}", &otoks[..n]);
    println!("native toks[..{n}] = {:?}", &ntoks[..n]);
    println!("onnx latents={} native latents={}", ol.len(), nl.len());
    assert!(n >= 4, "too few tokens to compare ({n})");
    assert_eq!(&otoks[..n], &ntoks[..n], "greedy token stream diverged");
}
