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

#![cfg(feature = "export")]

//! An exported Qwen3 package, opened by rlx alone, generates the tokens
//! `Qwen3Runner` does. Opt-in: `RLX_TEST_QWEN3_DIR=<a Hugging Face Qwen3
//! checkpoint directory>` (`model.safetensors`, `tokenizer.json`, …).

use rlx_qwen3::export::{ExportOptions, export_rlxp};
use rlx_qwen3::high_level_runner::Qwen3Runner;
use rlx_runtime::model::{ChatMessage, Tensor, ids_of};
use serde_json::json;

#[test]
fn an_exported_package_generates_what_qwen3_runner_does() {
    let Some(dir) = rlx_ir::env::var_os("RLX_TEST_QWEN3_DIR") else {
        eprintln!("skip: RLX_TEST_QWEN3_DIR is not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen3.rlxp");
    // `RLX_TEST_WEIGHT_DTYPE=f16` exports f16 matmul weights.
    let weight_dtype = match std::env::var("RLX_TEST_WEIGHT_DTYPE").as_deref() {
        Ok("f16") => Some(rlx_ir::DType::F16),
        Ok("bf16") => Some(rlx_ir::DType::BF16),
        _ => None,
    };
    // `RLX_TEST_GROW=1`: one short cache size, so the 45 tokens run on
    // steps grown past it (16 → 33 → 67) to `max_context`.
    let grow = std::env::var_os("RLX_TEST_GROW").is_some();
    let opts = ExportOptions {
        uppers: if grow { vec![16] } else { vec![32, 128] },
        max_context: Some(if grow { 128 } else { 512 }),
        weight_dtype,
        ..ExportOptions::default()
    };
    let t = std::time::Instant::now();
    let report = export_rlxp(&dir, &out, &opts).unwrap();
    eprintln!(
        "symbolic: {:?}, stateful: {:?}, prefill: {:?}, batch: {:?}, check: {:?}",
        report.symbolic_failure,
        report.stateful_failure,
        report.prefill_failure,
        report.batch_failure,
        report.check_failure
    );
    assert_eq!(
        report.functions,
        [
            "step",
            "prefill",
            "prefill_b2",
            "prefill_b4",
            "prefill_b8",
            "check"
        ],
        "{report:?}"
    );
    eprintln!(
        "export {:?}: {:?}, {} params referenced, {} held, {} narrowed, package {} bytes",
        t.elapsed(),
        report.functions,
        report.referenced,
        report.held,
        report.narrowed,
        std::fs::metadata(&out).unwrap().len()
    );

    // `RLX_TEST_BUILD=1`: run the package built for Metal and CPU
    // (`rlx_pkg::build`), each device its own function.
    let out = if std::env::var_os("RLX_TEST_BUILD").is_some() {
        let built = tmp.path().join("qwen3.built.rlxp");
        let targets: Vec<(&str, rlx_runtime::model::rlxp::TargetRecipe)> = ["metal", "cpu"]
            .into_iter()
            .map(|t| {
                (
                    t,
                    rlx_runtime::model::rlxp::TargetRecipe::for_target(t).unwrap(),
                )
            })
            .collect();
        let report = rlx_runtime::model::rlxp::build_for_targets(&out, &built, &targets).unwrap();
        eprintln!("build: {report:?}");
        built
    } else {
        out
    };
    let package = out.clone();
    // The package on `RLX_TEST_DEVICE` (`cpu` by default; `metal`, `cuda`, …
    // with that feature); the reference runner on the same device.
    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    eprintln!("package device: {device}");
    rlx_text::install_model_tokenizer();
    let t = std::time::Instant::now();
    let mut m = rlx_runtime::model::open_source(
        out.to_str().unwrap(),
        json!({
            "device": device,
            // `RLX_TEST_PREFILL=release`: one copy of the weights on the device.
            "prefill": std::env::var("RLX_TEST_PREFILL").unwrap_or_default(),
        }),
    )
    .unwrap();
    eprintln!("package open {:?}", t.elapsed());
    let info = m.info();
    assert_eq!(
        (info.task.as_str(), info.family.as_str()),
        ("text-generation", "qwen3")
    );
    // The checkpoint's model card: its front matter as card_data, its body.
    let data = &info.metadata["card_data"];
    assert_eq!(info.metadata["license"], data["license"], "{data}");
    assert_eq!(data["pipeline_tag"], "text-generation", "{data}");
    let card = info.metadata["card"].as_str().unwrap_or_default();
    assert!(!card.starts_with("---") && card.contains('#'), "{card}");

    // The same prompt ids into both.
    let prompt = "The capital of France is";
    let ids: Vec<u32> = rlx_text::load_tokenizer(&dir.join("tokenizer.json"))
        .unwrap()
        .encode(prompt, false)
        .unwrap();
    let n = 40;
    let t = std::time::Instant::now();
    let prompt_ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    let input = Tensor::new(
        "input_ids",
        rlx_ir::DType::I32,
        vec![prompt_ids.len()],
        prompt_ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
    );
    let out = m.run(&[input], &json!({"max_tokens": n})).unwrap();
    let got = ids_of(&out[0]).unwrap();
    let package_time = t.elapsed();

    let new_runner = || {
        Qwen3Runner::builder()
            .weights(dir.join("model.safetensors"))
            .device(rlx_runtime::parse_device(&device).unwrap())
            .max_seq(128)
            .build()
            .unwrap()
    };
    let chat = |m: &mut Box<dyn rlx_runtime::model::Model>| {
        let text = m
            .chat(
                &[ChatMessage::user(
                    "What is the capital of France? Answer in one word.",
                )],
                &json!({"max_tokens": 48}),
            )
            .unwrap();
        eprintln!("chat: {text:?}");
        assert!(text.contains("Paris"), "{text}");
    };
    // Beam search, the beams in the batched step against each extended on
    // its own (`batched: false`).
    let beam_opts = json!({"max_tokens": 12, "num_beams": 3, "temperature": 0.0});
    let beams: String;
    let plain_beams = |want_from_batched: &str| {
        let mut plain = rlx_runtime::model::open_source(
            package.to_str().unwrap(),
            json!({"device": device, "batched": false}),
        )
        .unwrap();
        let t = std::time::Instant::now();
        let want = plain
            .generate_text("The capital of France is", &beam_opts)
            .unwrap();
        eprintln!("3 beams one by one: {:?}", t.elapsed());
        assert_eq!(want_from_batched, want, "batched beams changed the text");
    };
    // Prompts of different lengths decoded together (the step's batch
    // symbol) give what each gives alone.
    {
        let prompts = [
            "The capital of France is",
            "One, two, three,",
            "Water boils at a temperature of",
        ];
        let opts = json!({"max_tokens": 12, "temperature": 0.0});
        let t = std::time::Instant::now();
        let alone: Vec<String> = prompts
            .iter()
            .map(|p| m.generate_text(p, &opts).unwrap())
            .collect();
        let alone_time = t.elapsed();
        m.generate_batch(&prompts, &opts).unwrap(); // compile the batch-3 step
        let t = std::time::Instant::now();
        let together = m.generate_batch(&prompts, &opts).unwrap();
        eprintln!(
            "batch of 3: {:?} together, {alone_time:?} one by one\n  {together:?}",
            t.elapsed()
        );
        assert_eq!(together, alone, "decoding together changed the text");
        // Beam search, the beams in the batched step against each extended
        // on its own (`batched: false`).
        let t = std::time::Instant::now();
        beams = m.generate_text(prompts[0], &beam_opts).unwrap();
        eprintln!("3 beams batched: {:?}, {beams:?}", t.elapsed());
        // On a shared GPU (`RLX_TEST_LOW_MEMORY`) the other copy loads once
        // this one is gone.
        if std::env::var_os("RLX_TEST_LOW_MEMORY").is_none() {
            plain_beams(&beams);
        }
    }
    // `RLX_TEST_LOW_MEMORY=1` (a GPU other work shares): the package's chat,
    // then the package freed before the runner loads; no timings.
    if std::env::var_os("RLX_TEST_LOW_MEMORY").is_some() {
        chat(&mut m);
        // `RLX_TEST_PACKAGE_ONLY=1`: the package alone (its memory).
        if std::env::var_os("RLX_TEST_PACKAGE_ONLY").is_some() {
            eprintln!("package ids: {got:?}");
            drop(m);
            plain_beams(&beams);
            return;
        }
        drop(m);
        plain_beams(&beams);
        let want = new_runner().generate(&ids, n, |_| {}).unwrap();
        eprintln!("package ids: {got:?}\nrunner ids:  {want:?}");
        assert_eq!(got, want[..got.len()], "greedy tokens differ");
        return;
    }
    // The checkpoint's sampling defaults; a reply kept to JSON; a chat
    // told about a tool.
    {
        assert!(
            info.metadata["generation"]["temperature"].is_number(),
            "{}",
            info.metadata
        );
        let reply = m
            .chat(
                &[rlx_runtime::model::ChatMessage::user(
                    "Give the capital of France as JSON with the key \"city\".",
                )],
                &json!({"max_tokens": 32, "response_format": "json", "temperature": 0.0}),
            )
            .unwrap();
        eprintln!("json: {reply}");
        let v: serde_json::Value = serde_json::from_str(reply.trim()).expect("one JSON value");
        assert_eq!(v["city"], "Paris", "{v}");
        let tools = json!([{"type": "function", "function": {
            "name": "get_weather",
            "description": "The weather in a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }}]);
        let call = m
            .chat(
                &[rlx_runtime::model::ChatMessage::user(
                    "What is the weather in Paris?",
                )],
                &json!({"tools": tools, "max_tokens": 48, "temperature": 0.0}),
            )
            .unwrap();
        eprintln!("tool call: {call}");
        assert!(call.contains("get_weather"), "{call}");
    }
    // `RLX_TEST_QUANTIZE=q8_0|q4_0`: the package with its matmul weights
    // quantized into it; how many of the greedy tokens it keeps.
    if let Ok(q) = std::env::var("RLX_TEST_QUANTIZE") {
        let scheme = match q.as_str() {
            "q4_0" => rlx_ir::quant::QuantScheme::GgufQ4_0,
            _ => rlx_ir::quant::QuantScheme::GgufQ8_0,
        };
        let quantized = tmp.path().join(format!("qwen3.{q}.rlxp"));
        let t = std::time::Instant::now();
        let report =
            rlx_runtime::model::rlxp::quantize_package(&package, &quantized, scheme).unwrap();
        eprintln!(
            "quantized {q} in {:?}: {report:?}, package {} MB",
            t.elapsed(),
            std::fs::metadata(&quantized).unwrap().len() / 1_000_000
        );
        let mut qm =
            rlx_runtime::model::open_source(quantized.to_str().unwrap(), json!({"device": device}))
                .unwrap();
        let input = Tensor::new(
            "input_ids",
            rlx_ir::DType::I32,
            vec![prompt_ids.len()],
            prompt_ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
        );
        let opts = json!({"max_tokens": n});
        ids_of(&qm.run(std::slice::from_ref(&input), &opts).unwrap()[0]).unwrap();
        let t = std::time::Instant::now();
        let q_ids = ids_of(&qm.run(&[input], &opts).unwrap()[0]).unwrap();
        let same = q_ids.iter().zip(&got).take_while(|(a, b)| a == b).count();
        eprintln!(
            "{q}: {n} tokens in {:?}, the first {same} as f32's",
            t.elapsed()
        );
        chat(&mut qm);
    }
    // Continuous batching: B joins A's batch ten steps in; A leaves when
    // done; the batch outgrows its 32-row cache and moves back. Each gets
    // the tokens it gets alone.
    {
        let tok = rlx_text::load_tokenizer(&dir.join("tokenizer.json")).unwrap();
        let mut alone = |text: &str, k: usize| {
            let ids: Vec<i32> = tok
                .encode(text, false)
                .unwrap()
                .iter()
                .map(|&i| i as i32)
                .collect();
            let input = Tensor::new(
                "input_ids",
                rlx_ir::DType::I32,
                vec![ids.len()],
                ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
            );
            (
                ids.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
                ids_of(&m.run(&[input], &json!({"max_tokens": k})).unwrap()[0]).unwrap(),
            )
        };
        let k = 24;
        let (a_ids, a_want) = alone("The capital of France is", k);
        let (b_ids, b_want) = alone("One, two, three,", k);
        let mut served = rlx_runtime::model::open_source(
            package.to_str().unwrap(),
            json!({"device": device, "max_context": 30}),
        )
        .unwrap();
        let lm = served.lm().unwrap();
        lm.serve_open(2).unwrap();
        let a = lm.serve_add(&a_ids).unwrap();
        let mut got: std::collections::HashMap<usize, Vec<u32>> = Default::default();
        let mut b = None;
        let t = std::time::Instant::now();
        for step in 0.. {
            if step == 10 {
                b = Some(lm.serve_add(&b_ids).unwrap());
            }
            for (slot, token) in lm.serve_step().unwrap() {
                let out = got.entry(slot).or_default();
                out.push(token);
                if out.len() == k {
                    lm.serve_remove(slot);
                }
            }
            if b.is_some_and(|b| got.get(&b).is_some_and(|o| o.len() == k)) {
                break;
            }
            assert!(step < 200, "served too long");
        }
        eprintln!("served two requests in {:?}", t.elapsed());
        assert_eq!(got[&a], a_want, "A in the batch");
        assert_eq!(got[&b.unwrap()], b_want, "B, joined later");
    }
    // Speculative decoding through the check: the same tokens.
    {
        let mut spec = rlx_runtime::model::open_source(
            package.to_str().unwrap(),
            json!({"device": device, "speculative": true}),
        )
        .unwrap();
        let input = || {
            Tensor::new(
                "input_ids",
                rlx_ir::DType::I32,
                vec![prompt_ids.len()],
                prompt_ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
            )
        };
        let opts = json!({"max_tokens": n});
        ids_of(&spec.run(&[input()], &opts).unwrap()[0]).unwrap(); // compile
        let t = std::time::Instant::now();
        let spec_ids = ids_of(&spec.run(&[input()], &opts).unwrap()[0]).unwrap();
        let spec_time = t.elapsed();
        let t = std::time::Instant::now();
        ids_of(&m.run(&[input()], &opts).unwrap()[0]).unwrap();
        eprintln!(
            "speculative: {n} tokens in {spec_time:?}, one at a time {:?}",
            t.elapsed()
        );
        assert_eq!(spec_ids, got, "speculative decoding changed the tokens");
        drop(spec);
        // A draft model's guesses (the package itself: every guess holds).
        let mut drafted = rlx_runtime::model::open_source(
            package.to_str().unwrap(),
            json!({"device": device, "speculative": true, "draft": package.to_str().unwrap()}),
        )
        .unwrap();
        ids_of(&drafted.run(&[input()], &opts).unwrap()[0]).unwrap(); // compile
        let t = std::time::Instant::now();
        let draft_ids = ids_of(&drafted.run(&[input()], &opts).unwrap()[0]).unwrap();
        eprintln!("drafted by a model: {n} tokens in {:?}", t.elapsed());
        assert_eq!(draft_ids, got, "a draft model changed the tokens");
    }
    let mut runner = new_runner();
    let t = std::time::Instant::now();
    let want = runner.generate(&ids, n, |_| {}).unwrap();
    let runner_time = t.elapsed();
    eprintln!(
        "package: {} tokens in {package_time:?} ({:.1} tok/s incl. prompt); \
         Qwen3Runner: {} in {runner_time:?} ({:.1} tok/s)",
        got.len(),
        got.len() as f64 / package_time.as_secs_f64(),
        want.len(),
        want.len() as f64 / runner_time.as_secs_f64(),
    );
    eprintln!("package ids: {got:?}\nrunner ids:  {want:?}");

    // Steady state, compile excluded: the first token (the prompt) and the
    // per-token cost after it, best of three.
    let run = |m: &mut Box<dyn rlx_runtime::model::Model>, k: usize| {
        let input = Tensor::new(
            "input_ids",
            rlx_ir::DType::I32,
            vec![prompt_ids.len()],
            prompt_ids.iter().flat_map(|i| i.to_le_bytes()).collect(),
        );
        let t = std::time::Instant::now();
        let ids = ids_of(&m.run(&[input], &json!({"max_tokens": k})).unwrap()[0]).unwrap();
        (t.elapsed().as_secs_f64(), ids)
    };
    // Each of the two runs at its best of three: a stall in one round
    // must not leak into the other's difference.
    let best = |f: &mut dyn FnMut() -> (f64, f64)| {
        let (first, all) = (0..3)
            .map(|_| f())
            .fold((f64::MAX, f64::MAX), |a, b| (a.0.min(b.0), a.1.min(b.1)));
        (first, (all - first) / (n - 1) as f64)
    };
    let (p_first, p_tok) = best(&mut || {
        let (t1, _) = run(&mut m, 1);
        let (tn, ids) = run(&mut m, n);
        assert_eq!(ids, got);
        (t1, tn)
    });
    let (r_first, r_tok) = best(&mut || {
        let t = std::time::Instant::now();
        let mut stamps = Vec::new();
        runner
            .generate(&ids, n, |_| stamps.push(t.elapsed().as_secs_f64()))
            .unwrap();
        (stamps[0], stamps[n - 1])
    });
    eprintln!(
        "warm first token: package {:.1} ms, Qwen3Runner {:.1} ms; \
         per token: package {:.1} ms ({:.1} tok/s), Qwen3Runner {:.1} ms ({:.1} tok/s)",
        p_first * 1e3,
        r_first * 1e3,
        p_tok * 1e3,
        1.0 / p_tok,
        r_tok * 1e3,
        1.0 / r_tok,
    );
    assert_eq!(got, want[..got.len()], "greedy tokens differ");
    chat(&mut m);
}
