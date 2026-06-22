//! Long-phrase chunk OOB regression (full-graph infer is high RAM).
//!
//! **Default CI / local:** run `transpose_numel` (`l_sin_gen_expand_inputs_are_broadcastable`) —
//! import-only, catches the Expand_3 / Gather_4 shape bug without executing the graph.
//!
//! **Full infer (heavy):** `KITTEN_RLX_HEAVY=1 cargo test -p kitten_tts_mini_rlx --test chunk25_infer -- --ignored --test-threads=1`
//! Optional: `KITTEN_MAX_WAVE=367200` for production buffer size.

use kitten_tts_mini_rlx::bundle_compile::{
    SeqCompileCache, ensure_kernels_registered, run_parity_inputs_with_duration,
    shape_all_graphs_for_infer,
};
use kitten_tts_mini_rlx::compile_profile::{compile_slot_length, compile_waveform_cap};
use rlx_runtime::Device;

fn max_wave_for_test(runtime_tokens: usize) -> usize {
    if let Ok(v) = std::env::var("KITTEN_MAX_WAVE") {
        if let Ok(n) = v.parse::<usize>() {
            return n.max(1);
        }
    }
    if kitten_tts_mini_rlx::compile_profile::env_flag("KITTEN_RLX_HEAVY") {
        return 367_200;
    }
    compile_waveform_cap(runtime_tokens, usize::MAX)
}

fn compile_cache(
    bundle_dir: std::path::PathBuf,
    compile_seq: usize,
    max_wave: usize,
) -> SeqCompileCache {
    SeqCompileCache::new(
        Device::Cpu,
        bundle_dir,
        compile_seq.saturating_add(8),
        max_wave,
        1,
    )
}

#[test]
#[ignore = "full-graph infer; high RAM — use transpose_numel for default OOB check"]
fn hello_infer_smoke() {
    ensure_kernels_registered();
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("graph.json").is_file() {
        return;
    }
    let runtime_tokens = 8usize;
    let max_wave = max_wave_for_test(runtime_tokens);
    let compile_seq = compile_slot_length(runtime_tokens);
    let cache = compile_cache(bundle_dir, compile_seq, max_wave);
    let graphs = cache
        .cached_graphs_for_seq(runtime_tokens)
        .expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let mut ids_padded = ids.clone();
    ids_padded.resize(compile_seq, 0);
    let mut dur: Vec<i64> = vec![19, 2, 1, 2, 3, 2, 3, 2];
    dur.resize(compile_seq, 0);
    let style = vec![0.0f32; 256];
    shape_all_graphs_for_infer(&graphs, runtime_tokens, compile_seq).expect("shape");
    let mut g = graphs.full.lock().expect("graph");
    let _ = run_parity_inputs_with_duration(
        &mut g,
        compile_seq,
        runtime_tokens,
        &ids_padded,
        &style,
        Some(&dur),
    );
}

#[test]
#[ignore = "full-graph infer; high RAM — use transpose_numel for default OOB check"]
fn chunk25_parity_infer_does_not_oob() {
    ensure_kernels_registered();
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("graph.json").is_file() {
        eprintln!("skip: rlx_bundle missing");
        return;
    }

    let runtime_tokens = 25usize;
    let max_wave = max_wave_for_test(runtime_tokens);
    let compile_seq = compile_slot_length(runtime_tokens);
    assert_eq!(compile_seq, 31, "expected compile headroom 25+6");

    let cache = compile_cache(bundle_dir, compile_seq, max_wave);
    let graphs = cache
        .cached_graphs_for_seq(runtime_tokens)
        .expect("compile");

    let mut ids: Vec<i64> = vec![0];
    ids.extend(1..=23i64);
    ids.push(0);
    assert_eq!(ids.len(), runtime_tokens);

    let mut ids_padded = ids.clone();
    ids_padded.resize(compile_seq, 0);

    let mut dur: Vec<i64> = vec![
        4, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
    ];
    dur.resize(compile_seq, 0);

    let style = vec![0.0f32; 256];
    shape_all_graphs_for_infer(&graphs, runtime_tokens, compile_seq).expect("shape");

    let mut g = graphs.full.lock().expect("graph");
    let outs = run_parity_inputs_with_duration(
        &mut g,
        compile_seq,
        runtime_tokens,
        &ids_padded,
        &style,
        Some(&dur),
    );
    let wave = outs.first().expect("waveform");
    assert!(wave.0.len() > 4, "expected waveform bytes");
}
