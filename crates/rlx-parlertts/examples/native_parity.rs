// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native (rlx-onnx-import) vs onnxruntime parity for the Parler T5 encoder +
//! decoder — proves the native path is numerically correct (esp. that the
//! decoder's best-effort `binary_infer` shape choice is right) BEFORE wiring the
//! full native synthesis. ort is a DEV-dependency here (validation only).
//!
//! ```text
//! cargo run -p rlx-parlertts --example native_parity --features native
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ort::value::Tensor;
use rlx_ir::DType;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_runtime::{AotCache, CompileOptions, Device};
use tokenizers::Tokenizer;

const ENC_LEN: usize = 128; // T5 relative-position-bias is baked at 128
const DEC_T: usize = 4; // decode length for the parity probe

fn i64_le(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

struct Native {
    cache: AotCache,
    dir: std::path::PathBuf,
}
impl Native {
    fn run(
        &self,
        component: &str,
        named: &[(&str, usize)],
        seq: usize,
        inputs: &[(&str, &[u8], DType)],
    ) -> Result<Vec<(Vec<u8>, DType)>> {
        let path = self.dir.join("onnx").join(format!("{component}.onnx"));
        let mut named_lengths: HashMap<String, usize> =
            named.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        named_lengths.entry("batch_size".into()).or_insert(1);
        let opts = ImportOptions {
            sequence_length: seq,
            named_lengths,
            strict: false,
            ..Default::default()
        };
        let (hir, mut params, _r, _m) = build_hir_from_onnx_file(&path, opts)?;
        let key = format!("parity_{component}_s{seq}");
        // Toggle passes off to isolate which one drops the T5 mask (RLX_NO_PASSES=1).
        let mut copts = CompileOptions::default();
        if std::env::var_os("RLX_NO_PASSES").is_some() {
            copts.dce = false;
            copts.constant_folding = false;
            copts.fusion_opts.skip_fusion = true; // disable the attention-block fusion
        }
        let dev = match std::env::var("RLX_DEVICE").as_deref() {
            Ok("metal") => Device::Metal,
            Ok("mlx") => Device::Mlx,
            Ok("wgpu") => Device::WebGpu,
            Ok("coreml") => Device::Ane,
            _ => Device::Cpu,
        };
        let mut g = self
            .cache
            .compile_hir_cached(
                &format!("{key}_np{}_{dev:?}", copts.dce as u8),
                dev,
                hir,
                &copts,
            )
            .map_err(|e| anyhow::anyhow!("compile {component}: {e}"))?;
        for (n, d) in params.drain() {
            g.set_param(&n, &d);
        }
        g.finalize_params();
        Ok(g.run_typed(inputs))
    }
}

fn main() -> Result<()> {
    // Decoder import trips one spurious binary_infer; best-effort-continue and let
    // THIS parity check decide whether that shape choice is numerically correct.
    unsafe { std::env::set_var("RLX_DBG_BINF", "1") };

    let dir = std::env::var("RLX_PARLER_DIR").unwrap_or_else(|_| "weights/tts/parlertts".into());
    let dir = Path::new(&dir).to_path_buf();
    let tok =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let nat = Native {
        cache: AotCache::new(std::env::temp_dir().join("rlx_parler_parity")),
        dir: dir.clone(),
    };

    // fixed input: pad transcript to ENC_LEN
    let mut ids: Vec<i64> = tok
        .encode("The quick brown fox.", false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .iter()
        .map(|&i| i as i64)
        .collect();
    ids.push(1); // T5 eos
    let real = ids.len();
    ids.resize(ENC_LEN, 0);
    let mask: Vec<i64> = (0..ENC_LEN).map(|i| if i < real { 1 } else { 0 }).collect();

    // ---- T5 encoder ----
    let nat_enc = nat.run(
        "text_encoder",
        &[("sequence_length", ENC_LEN), ("t", ENC_LEN)],
        ENC_LEN,
        &[
            ("input_ids", &i64_le(&ids), DType::I64),
            ("attention_mask", &i64_le(&mask), DType::I64),
        ],
    )?;
    let nat_hs = as_f32(&nat_enc[0].0);

    let mut enc =
        ort::session::Session::builder()?.commit_from_file(dir.join("onnx/text_encoder.onnx"))?;
    let o = enc.run(ort::inputs![
        "input_ids" => Tensor::<i64>::from_array(([1usize, ENC_LEN], ids.clone()))?,
        "attention_mask" => Tensor::<i64>::from_array(([1usize, ENC_LEN], mask.clone()))?
    ])?;
    let ort_hs = o[0].try_extract_tensor::<f32>()?.1.to_vec();
    let real_n = real * 1024; // only the non-padding rows matter (decoder masks padding)
    println!(
        "[encoder] full cosine={:.6}  REAL-tokens({real} tok) cosine={:.6}",
        cosine(&nat_hs, &ort_hs),
        cosine(&nat_hs[..real_n], &ort_hs[..real_n]),
    );
    // Non-confounded per-real-row breakdown: is each real row scale-off or
    // direction-off? (RMSNorm makes all rows unit-RMS, so a low per-row cosine ⇒
    // genuine direction error upstream, not a magnitude artifact.)
    for r in 0..real {
        let (a, b) = (
            &nat_hs[r * 1024..r * 1024 + 1024],
            &ort_hs[r * 1024..r * 1024 + 1024],
        );
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        println!(
            "   row {r}: cos={:.5}  |nat|={na:.3} |ort|={nb:.3}",
            cosine(a, b)
        );
    }

    // Isolating test: NO padding — fill all 128 with a real token, mask all-1.
    // High cosine here ⇒ the bug is padding-mask handling; low ⇒ attention/bias.
    {
        let ids2: Vec<i64> = vec![100i64; ENC_LEN];
        let mask2: Vec<i64> = vec![1i64; ENC_LEN];
        let n2 = nat.run(
            "text_encoder",
            &[("sequence_length", ENC_LEN), ("t", ENC_LEN)],
            ENC_LEN,
            &[
                ("input_ids", &i64_le(&ids2), DType::I64),
                ("attention_mask", &i64_le(&mask2), DType::I64),
            ],
        )?;
        let nat2 = as_f32(&n2[0].0);
        let mut enc2 = ort::session::Session::builder()?
            .commit_from_file(dir.join("onnx/text_encoder.onnx"))?;
        let o2 = enc2.run(ort::inputs![
            "input_ids" => Tensor::<i64>::from_array(([1usize, ENC_LEN], ids2))?,
            "attention_mask" => Tensor::<i64>::from_array(([1usize, ENC_LEN], mask2))?
        ])?;
        let ort2 = o2[0].try_extract_tensor::<f32>()?.1.to_vec();
        println!(
            "[encoder] NO-PADDING (128 real, mask all-1) cosine={:.6}",
            cosine(&nat2, &ort2)
        );
    }

    // Diagnostic: pad with a "safe" token (100) instead of 0, KEEP the mask. If
    // this is bit-exact, the padding-token *embedding* leaks past the mask (mask
    // bug); if it stays ~0.75, it's a numerical residual independent of pad id.
    {
        let mut ids3 = ids.clone();
        for i in real..ENC_LEN {
            ids3[i] = 100;
        }
        let n3 = nat.run(
            "text_encoder",
            &[("sequence_length", ENC_LEN), ("t", ENC_LEN)],
            ENC_LEN,
            &[
                ("input_ids", &i64_le(&ids3), DType::I64),
                ("attention_mask", &i64_le(&mask), DType::I64),
            ],
        )?;
        let nat3 = as_f32(&n3[0].0);
        let mut e3 = ort::session::Session::builder()?
            .commit_from_file(dir.join("onnx/text_encoder.onnx"))?;
        let o3 = e3.run(ort::inputs![
            "input_ids" => Tensor::<i64>::from_array(([1usize, ENC_LEN], ids3))?,
            "attention_mask" => Tensor::<i64>::from_array(([1usize, ENC_LEN], mask.clone()))?
        ])?;
        let ort3 = o3[0].try_extract_tensor::<f32>()?.1.to_vec();
        println!(
            "[encoder] PAD-WITH-100 (mask on) full={:.6} real={:.6}",
            cosine(&nat3, &ort3),
            cosine(&nat3[..real_n], &ort3[..real_n])
        );
    }

    // Real-length NO-PADDING (the actual single-utterance synthesis mode): import
    // + run the encoder at exactly `real` tokens, mask all-1. Bit-exact here means
    // native Parler synthesis needs no padding at all.
    {
        let idsr: Vec<i64> = ids[..real].to_vec();
        let maskr: Vec<i64> = vec![1i64; real];
        let nr = nat.run(
            "text_encoder",
            &[("sequence_length", real), ("t", real)],
            real,
            &[
                ("input_ids", &i64_le(&idsr), DType::I64),
                ("attention_mask", &i64_le(&maskr), DType::I64),
            ],
        )?;
        let natr = as_f32(&nr[0].0);
        let mut encr = ort::session::Session::builder()?
            .commit_from_file(dir.join("onnx/text_encoder.onnx"))?;
        let or = encr.run(ort::inputs![
            "input_ids" => Tensor::<i64>::from_array(([1usize, real], idsr))?,
            "attention_mask" => Tensor::<i64>::from_array(([1usize, real], maskr))?
        ])?;
        let ortr = or[0].try_extract_tensor::<f32>()?.1.to_vec();
        println!(
            "[encoder] REAL-LENGTH no-pad (N={real}) cosine={:.6}",
            cosine(&natr, &ortr)
        );
    }

    // Long real-length (N=64, no pad): exercises the relative-position LOG-bucket
    // branch (distances > 8), which the short cases don't. Confirms parity holds
    // for realistic transcripts.
    {
        let n = 64usize;
        let idsl: Vec<i64> = (0..n).map(|i| (100 + (i % 40)) as i64).collect();
        let maskl: Vec<i64> = vec![1i64; n];
        let nl = nat.run(
            "text_encoder",
            &[("sequence_length", n), ("t", n)],
            n,
            &[
                ("input_ids", &i64_le(&idsl), DType::I64),
                ("attention_mask", &i64_le(&maskl), DType::I64),
            ],
        )?;
        let natl = as_f32(&nl[0].0);
        let mut el = ort::session::Session::builder()?
            .commit_from_file(dir.join("onnx/text_encoder.onnx"))?;
        let ol = el.run(ort::inputs![
            "input_ids" => Tensor::<i64>::from_array(([1usize, n], idsl))?,
            "attention_mask" => Tensor::<i64>::from_array(([1usize, n], maskl))?
        ])?;
        let ortl = ol[0].try_extract_tensor::<f32>()?.1.to_vec();
        println!(
            "[encoder] LONG no-pad (N={n}, log-bucket) cosine={:.6}",
            cosine(&natl, &ortl)
        );
    }

    // ---- decoder (fixed decoder_input_ids of BOS + transcript prompt) ----
    // New 4-input export: prompt_input_ids (the transcript) → embed_prompts →
    // prompt_hidden prefix. Output logits are [9, PT+DEC_T, 1088].
    let prompt_ids: Vec<i64> = {
        let mut p: Vec<i64> = tok
            .encode("Hello there.", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .iter()
            .map(|&i| i as i64)
            .collect();
        p.push(1); // eos
        p
    };
    let pt = prompt_ids.len();
    let dids: Vec<i64> = vec![1025i64; 9 * DEC_T]; // [1,9,T] all BOS
    let nat_dec = nat.run(
        "decoder",
        &[
            ("t", DEC_T),
            ("et", ENC_LEN),
            ("pt", pt),
            ("sequence_length", DEC_T),
            ("encoder_sequence_length", ENC_LEN),
        ],
        DEC_T,
        &[
            ("decoder_input_ids", &i64_le(&dids), DType::I64),
            ("encoder_hidden_states", &f32_le(&ort_hs), DType::F32),
            ("encoder_attention_mask", &i64_le(&mask), DType::I64),
            ("prompt_input_ids", &i64_le(&prompt_ids), DType::I64),
        ],
    )?;
    let nat_lg = as_f32(&nat_dec[0].0);

    let mut dec =
        ort::session::Session::builder()?.commit_from_file(dir.join("onnx/decoder.onnx"))?;
    let od = dec.run(ort::inputs![
        "decoder_input_ids" => Tensor::<i64>::from_array(([1usize, 9, DEC_T], dids))?,
        "encoder_hidden_states" => Tensor::<f32>::from_array(([1usize, ENC_LEN, 1024], ort_hs))?,
        "encoder_attention_mask" => Tensor::<i64>::from_array(([1usize, ENC_LEN], mask))?,
        "prompt_input_ids" => Tensor::<i64>::from_array(([1usize, pt], prompt_ids))?
    ])?;
    let ort_lg = od[0].try_extract_tensor::<f32>()?.1.to_vec();
    println!("[decoder] pt={pt} DEC_T={DEC_T} → expect logits len 9*{}*1088={}", pt + DEC_T, 9 * (pt + DEC_T) * 1088);
    let c = cosine(&nat_lg, &ort_lg);
    println!(
        "[decoder] native {} vs ort {} → cosine={:.6}",
        nat_lg.len(),
        ort_lg.len(),
        c
    );

    if c > 0.999 {
        println!(
            "✅ native Parler encoder+decoder match onnxruntime — best-effort shape is correct"
        );
    } else {
        println!(
            "⚠️ decoder mismatch (cos {c:.4}) — the best-effort Mul shape is WRONG, needs an importer fix"
        );
    }
    Ok(())
}
