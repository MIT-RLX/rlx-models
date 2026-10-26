// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export FireRedAudio's understanding path (ASR and audio questions) as an
//! `.rlxp` package rlx runs alone: `mel` (Whisper's log-mel, symbolic in the
//! frames), `encoder` (the audio encoder over whole chunks, the reference's
//! padding as masks the program gives: one graph for any clip within them,
//! symbolic in the chunks), and the Qwen3.5 backbone's decode `step` with
//! its state inside, taking a token or an audio row
//! ([`rlx_qwen35::export_vl::host_embed_step_with`]). The checkpoint's
//! tensors are referenced, bf16 as stored. A program answers `transcribe`
//! (and `run`, with a question): what [`crate::FireRedRunner`] does — the
//! signal reflect-padded, the mel laid in chunks, the masks, the chatml
//! prompt with one `<|AUDIO|>` per audio token, then the prompt and reply
//! through the step (`rlx-qwen35`'s shared `decode.rs`).
//!
//! Generation (RedAE, the DiT) is not exported.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::export::audio::{whisper_log_mel, whisper_mel_params};
use rlx_core::export::graph::symbolic_in_lifted_many;
use rlx_core::export::{Checkpoint, Package, program};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_ir::Graph;
use serde_json::json;

use crate::config::FireRedAudioConfig;
use crate::load::WeightStore;
use crate::weights::PREFIX_AUDIO;
use rlx_qwen35::export_vl::{CAP, PAST};

/// The mel's frames.
const FRAMES: u32 = 2;
/// The encoder's chunks.
const CHUNKS: u32 = 3;
const N_FFT: usize = 400;
const HOP: usize = 160;
const SAMPLE_RATE: usize = 16_000;
/// The longest context exported when the options name none: a 30 s clip's
/// 375 audio tokens, the prompt around them, and the runner's 300-token
/// reply, with room for a second chunk.
const DEFAULT_CONTEXT: usize = 2048;
/// The checkpoint's embedding table.
const EMBED_KEY: &str = "backbone_llm.model.language_model.embed_tokens.weight";

type Params = HashMap<String, Vec<f32>>;

/// The model directory: `dir`, or its `FireRedAudio/` (the Hugging Face
/// repository's layout).
fn model_dir(dir: &Path) -> PathBuf {
    if !dir.join("config.json").is_file() && dir.join("FireRedAudio/config.json").is_file() {
        dir.join("FireRedAudio")
    } else {
        dir.to_path_buf()
    }
}

/// The audio encoder over `chunks` chunks, named as the program feeds it.
fn encoder(
    cfg: &FireRedAudioConfig,
    store: &WeightStore,
    chunks: usize,
) -> Result<(Graph, Params)> {
    let mut wm = store.load_prefixes(&[PREFIX_AUDIO])?;
    let built = crate::encoder::build_encoder_masked(&cfg.audio_encoder, &mut wm, chunks)?;
    let (mut g, p) = rlx_core::flow_util::graph_from_built(built)?;
    g.name = "encoder".into();
    let out = g.outputs[0];
    g.node_mut(out).name = Some("audio".into());
    Ok((g, p))
}

/// Export the FireRedAudio directory `dir` (the Hugging Face repository, or
/// its `FireRedAudio/`: `config.json`, the shards, `tokenizer.json`) as the
/// package `out`. `opts.max_context` bounds prompt and reply together.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let dir = model_dir(dir);
    let cfg = FireRedAudioConfig::from_file(&dir.join("config.json"))?;
    cfg.validate()?;
    let qcfg = cfg.qwen35_config();
    let rows = opts.max_context.unwrap_or(DEFAULT_CONTEXT) + 1;
    let ck = Checkpoint::open(&dir)?;
    let mut refr = ck.referencer(opts.embed_weights)?;

    // The backbone's step, its weights given up as each build is taken.
    let store = WeightStore::open(&dir)?;
    let (step, states) = rlx_qwen35::export_vl::host_embed_step_with(
        &|past| {
            let (c, w) = crate::hf_config::load_qwen35_backbone(&dir, &cfg)?;
            rlx_qwen35::export_vl::raw_step_from(&c, w, past)
        },
        &qcfg,
        rows,
        &mut |p| refr.take(p),
    )?;
    let mut extra = rlx_qwen35::export_vl::step_tables(&qcfg, rows)?;
    let mut wm = store.load_keys(&[EMBED_KEY])?;
    extra.insert("embed_tokens".into(), wm.take(EMBED_KEY)?.0);
    drop(wm);
    refr.take(extra)?;

    // The front end.
    let n_mels = cfg.audio_encoder.num_mel_bins;
    let (mel, lifted) = symbolic_in_lifted_many(&[(FRAMES, vec![37, 53, 20])], &[0], |n| {
        Ok(whisper_log_mel(N_FFT, HOP, n_mels, n[0]))
    })?;
    anyhow::ensure!(lifted.is_empty(), "the mel lifted tables");
    refr.take(whisper_mel_params(
        N_FFT,
        &crate::audio::mel_filterbank(SAMPLE_RATE as f64, N_FFT, n_mels),
    ))?;

    // The audio encoder, symbolic in its chunks.
    let (enc, lifted) = symbolic_in_lifted_many(&[(CHUNKS, vec![3, 5, 2])], &[0], |n| {
        Ok(encoder(&cfg, &store, n[0])?.0)
    })?;
    anyhow::ensure!(lifted.is_empty(), "the encoder lifted tables");
    refr.take(encoder(&cfg, &store, 2)?.1)?;

    let mut pkg = Package::for_checkpoint(
        &ck,
        opts,
        "fireredaudio",
        "automatic-speech-recognition",
        "rlx-fireredaudio",
    );
    pkg.model["license"] = json!("apache-2.0");
    pkg.weights_as_stored();
    pkg.graphs = vec![mel, enc, step];
    pkg.dims = vec![
        ("cap".into(), CAP),
        ("past".into(), PAST),
        ("frames".into(), FRAMES),
        ("chunks".into(), CHUNKS),
    ];
    pkg.reference(refr, out)?;
    let tokenizer = pkg.tokenizer(&ck, None)?;

    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build FireRedAudio's program")?;
    let t = &cfg.tokens;
    let config = json!({
        "tokenizer": tokenizer,
        "hidden": qcfg.hidden_size,
        "context": rows - 1,
        "states": rlx_qwen35::export_vl::states_json(&states),
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop": HOP,
        "n_mels": n_mels,
        "chunk": cfg.audio_encoder.chunk_frames(),
        "audio_token": t.audio_special_token,
        "audio_token_id": t.audio_special_token_id,
        "eos": [crate::tokenizer::FireRedTokenizer::from_model_dir(&dir)?.eos_id()],
        "max_tokens": 300,
    });
    program::entry(
        &mut pkg,
        "understand",
        module,
        &["transcribe", "run"],
        "automatic-speech-recognition",
        config,
    );
    pkg.metadata("context_length", json!(rows - 1));
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "fireredaudio",
        "automatic-speech-recognition",
        export_rlxp,
    ));
}

#[cfg(test)]
mod stages {
    use super::*;
    use crate::audio::{AudioGeometry, conv_len, pcm_to_log_mel};

    fn run(g: Graph, params: &Params, inputs: &[(&str, &[f32])]) -> Vec<f32> {
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
        for (n, v) in params {
            c.set_param(n, v);
        }
        c.finalize_params();
        c.run(inputs).remove(0)
    }

    /// The masked encoder over whole chunks against the runner's, built at
    /// the clip's own length: the harness clip (one chunk, no padding) and
    /// seven of it back to back (two chunks, the last padded).
    #[test]
    fn masked_encoder_matches_the_runners() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = model_dir(&root.join("weights/asr/fireredaudio"));
        if !dir.join("config.json").is_file() {
            eprintln!("skip: no FireRedAudio checkpoint");
            return;
        }
        let cfg = FireRedAudioConfig::from_file(&dir.join("config.json")).unwrap();
        let store = WeightStore::open(&dir).unwrap();
        let clip =
            rlx_whisper::load_wav_mono_f32(&root.join("assets/harness/jfk_16k.wav")).unwrap();
        for (repeat, chunks) in [(1, 1), (7, 2)] {
            let pcm = clip.repeat(repeat);
            let mel = pcm_to_log_mel(&pcm, cfg.audio_encoder.num_mel_bins).unwrap();
            let geom = AudioGeometry::new(&cfg.audio_encoder, mel.n_frames).unwrap();
            assert_eq!(geom.num_chunks, chunks);
            let (n, mels) = (mel.n_frames, mel.n_mels);
            let chunk = cfg.audio_encoder.chunk_frames();
            let width = chunks * chunk;
            let mut laid = vec![0f32; mels * width];
            for m in 0..mels {
                laid[m * width..m * width + n].copy_from_slice(&mel.data[m * n..(m + 1) * n]);
            }

            // The runner's input: chunks of its own width, the last padded.
            let want = {
                let mcl = geom.max_chunk_len;
                let mut padded = vec![0f32; mels * chunks * mcl];
                for m in 0..mels {
                    padded[m * chunks * mcl..m * chunks * mcl + n]
                        .copy_from_slice(&mel.data[m * n..(m + 1) * n]);
                }
                let mut wm = store.load_prefixes(&[PREFIX_AUDIO]).unwrap();
                let built = crate::encoder::build_encoder_built(&cfg.audio_encoder, &mut wm, &geom)
                    .unwrap();
                let (g, p) = rlx_core::flow_util::graph_from_built(built).unwrap();
                run(g, &p, &[("mel", &padded)])
            };

            let on = |k: usize, n: usize| -> Vec<f32> {
                (0..n).map(|i| f32::from(u8::from(i < k))).collect()
            };
            let t_pc = conv_len(chunk);
            let seq = chunks * t_pc;
            let mut window = vec![chunks as f32; seq];
            let mut after = 0;
            for c in 0..chunks {
                let w = conv_len((n - c * chunk).min(chunk));
                for r in 0..w {
                    window[c * t_pc + r] = c as f32;
                }
                after += w;
            }
            let s3 = conv_len(after);
            let (g, p) = encoder(&cfg, &store, chunks).unwrap();
            let got = run(
                g,
                &p,
                &[
                    ("mel", &laid),
                    ("frames", &on(n, width)),
                    ("window", &window),
                    ("valid", &on(after, seq)),
                    ("valid3", &on(s3, conv_len(seq))),
                ],
            );
            let tokens = conv_len(s3);
            assert_eq!(tokens, geom.num_audio_tokens);
            let k = tokens * cfg.audio_encoder.output_dim;
            assert_eq!(want.len(), k);
            let d = want
                .iter()
                .zip(&got[..k])
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            eprintln!("{chunks} chunk(s), {tokens} tokens, max |d| {d} (peak {peak})");
            assert!(d < 1e-3 * peak.max(1.0), "max |d| {d}");
        }
    }
}
