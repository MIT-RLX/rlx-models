//! Compare CPU F32 vs packed GGUF prefill logits on Orpheus GGUF.
use rlx_core::weight_loader::load_from_path;
use rlx_gguf::GgufFile;
use rlx_llama32::{
    Llama32Generator, Llama32RunnerBuilder, MetalGgufPrefillMode, llama32_cfg_from_gguf,
};
use rlx_orpheus::DEFAULT_COMPILE_SEQ_CAP;
use rlx_orpheus::tokens::build_prompt;
use rlx_runtime::Device;

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() -> anyhow::Result<()> {
    let gguf = std::env::var("ORPHEUS_GGUF_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf")
        });
    let prompt = build_prompt(&gguf, "Hi.", Some("tara"))?;
    eprintln!("prompt len={}", prompt.len());

    let path = gguf.to_str().unwrap();
    let raw = GgufFile::from_path(&gguf)?;
    let cfg = llama32_cfg_from_gguf(&raw)?;
    let cap = DEFAULT_COMPILE_SEQ_CAP as usize;

    for (label, mode) in [
        ("cpu_f32", MetalGgufPrefillMode::CpuF32),
        ("packed", MetalGgufPrefillMode::PackedGguf),
    ] {
        let mut loader = load_from_path(path)?;
        let mut generator = Llama32Generator::from_loader_at_mode(
            cfg.clone(),
            loader.as_mut(),
            Device::Metal,
            &gguf,
            mode,
        )?
        .with_compile_seq_cap(cap)
        .with_prefill_cache(4)
        .with_decode_cache(cap + 8);
        let logits = generator.prefill_get_last_logits(&prompt)?;
        eprintln!(
            "{label}: logits len={} argmax={} top={:.4}",
            logits.len(),
            argmax(&logits),
            logits[argmax(&logits)]
        );
        if label == "packed" {
            generator.prefill(&prompt);
            let tok = generator
                .step_cached(rlx_qwen3::SampleOpts::greedy())
                .expect("step");
            eprintln!("{label}: step_cached first token={tok}");
        }
    }

    let upper = prompt.len().next_power_of_two().min(cap).max(prompt.len());
    eprintln!("runner bucket upper_seq={upper}");
    let mut runner = Llama32RunnerBuilder::default()
        .weights(&gguf)
        .max_seq(upper)
        .device(Device::Metal)
        .packed_weights(true)
        .build()?;
    let logits = runner.predict_logits(&prompt)?;
    eprintln!(
        "runner packed: logits len={} argmax={} top={:.4}",
        logits.len(),
        argmax(&logits),
        logits[argmax(&logits)]
    );
    Ok(())
}
