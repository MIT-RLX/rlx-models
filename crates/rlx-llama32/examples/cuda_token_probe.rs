//! Greedy token parity: CPU reference vs CUDA decode paths.
use rlx_core::weight_loader::GgufLoader;
use rlx_llama32::{Llama32Generator, MetalGgufPrefillMode, llama32_cfg_from_gguf};
use rlx_qwen35::encode_prompt_from_gguf;
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;

const PROMPT_START_ID: u32 = 128_259;
const BOS_TOKEN_ID: u32 = 128_000;
const PROMPT_END_IDS: [u32; 4] = [128_009, 128_260, 128_261, 128_257];

fn build_prompt_ids(gguf: &std::path::Path, body: &str) -> anyhow::Result<Vec<u32>> {
    let mut ids = encode_prompt_from_gguf(gguf, body)?;
    let mut out = Vec::with_capacity(ids.len() + 2 + PROMPT_END_IDS.len());
    out.push(PROMPT_START_ID);
    if ids.first().copied() != Some(BOS_TOKEN_ID) {
        out.push(BOS_TOKEN_ID);
    }
    out.append(&mut ids);
    out.extend_from_slice(&PROMPT_END_IDS);
    Ok(out)
}

fn generate_tokens(device: Device, prefill: MetalGgufPrefillMode, gguf: &str, prompt: &[u32], n: usize) -> anyhow::Result<Vec<u32>> {
    let mut loader = GgufLoader::from_file(gguf)?;
    let cfg = llama32_cfg_from_gguf(loader.file())?;
    let path = std::path::Path::new(gguf);
    let dynamic = std::env::var("ORPHEUS_DYNAMIC_DECODE").ok().as_deref() == Some("1");
    let mut g = Llama32Generator::from_loader_at_mode(cfg, &mut loader, device, path, prefill)?
        .with_compile_seq_cap(128);
    g = if dynamic {
        g.with_dynamic_decode_cache(2)
    } else {
        g.with_decode_cache(128)
    };
    g.prefill(prompt);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(g.step_cached(SampleOpts::greedy())?);
    }
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let gguf = std::env::var("ORPHEUS_GGUF_PATH")
        .unwrap_or_else(|_| "/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf".into());
    let text = std::env::var("ORPHEUS_TEXT").unwrap_or_else(|_| "The weather is nice today.".into());
    let voice = std::env::var("ORPHEUS_VOICE").unwrap_or_else(|_| "tara".into());
    let n: usize = std::env::var("ORPHEUS_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let prompt = build_prompt_ids(std::path::Path::new(&gguf), &format!("{voice}: {text}"))?;
    eprintln!("prompt len={}", prompt.len());

    let native_prefill = std::env::var("ORPHEUS_CUDA_NATIVE_PREFILL").ok().as_deref() == Some("1");
    let native_decode = std::env::var("ORPHEUS_CUDA_NATIVE_DECODE").ok().as_deref() == Some("1");
    let cuda_only = std::env::var("CUDA_PROBE_CUDA_ONLY").ok().as_deref() == Some("1");
    if native_decode {
        unsafe { std::env::set_var("ORPHEUS_CUDA_NATIVE_DECODE", "1") };
    }
    unsafe { std::env::set_var("RLX_CUDA_ARENA_POOL", "0") };

    let cpu = if cuda_only {
        Vec::new()
    } else {
        generate_tokens(Device::Cpu, MetalGgufPrefillMode::CpuF32, &gguf, &prompt, n)?
    };
    let cuda_prefill = if native_prefill {
        unsafe { std::env::set_var("ORPHEUS_CUDA_NATIVE_PREFILL", "1") };
        MetalGgufPrefillMode::PackedGguf
    } else {
        MetalGgufPrefillMode::CpuF32
    };
    let cuda = generate_tokens(Device::Cuda, cuda_prefill, &gguf, &prompt, n)?;
    eprintln!("CPU:  {cpu:?}");
    eprintln!("CUDA: {cuda:?}");
    if cuda_only {
        eprintln!("cuda-only ok for {n} tokens");
        return Ok(());
    }
    let div = cpu.iter().zip(cuda.iter()).position(|(a, b)| a != b);
    if let Some(i) = div {
        anyhow::bail!("diverge at step {i}: cpu={} cuda={}", cpu[i], cuda[i]);
    }
    eprintln!("parity ok for {n} tokens");
    Ok(())
}
