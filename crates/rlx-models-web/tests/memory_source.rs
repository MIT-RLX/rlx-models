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

//! The browser path loads every model from bytes ([`Files`]); natively the
//! same models load from a directory. Both must give the same output, bit for
//! bit — they run the same graph on the same weights, so any difference is a
//! loading bug. The file lists double as what a page has to fetch.
//!
//! ```sh
//! RLX_WEB_WEIGHTS=$PWD/weights cargo test -p rlx-models-web --release --test memory_source
//! ```

use std::path::{Path, PathBuf};

use rlx_models_web::Files;
use rlx_runtime::Device;

fn root() -> Option<PathBuf> {
    rlx_ir::env::var_result("RLX_WEB_WEIGHTS")
        .ok()
        .map(PathBuf::from)
}

fn files(dir: &Path, names: &[&str]) -> Files {
    let mut f = Files::new();
    for n in names {
        f.add(
            n,
            std::fs::read(dir.join(n)).unwrap_or_else(|e| panic!("{n}: {e}")),
        );
    }
    f
}

#[test]
fn moonshine_streaming_from_memory_matches_directory() {
    let Some(root) = root() else { return };
    let dir = root.join("asr/moonshine-streaming-tiny");
    let wav = root.join("../assets/harness/jfk_16k.wav");
    let (pcm, _) = rlx_core::asr_bench::read_wav_mono(&wav).unwrap();

    let mut disk = rlx_moonshine_streaming::MoonshineStreaming::open(&dir, Device::Cpu).unwrap();
    let mut f = files(
        &dir,
        &["config.json", "model.safetensors", "tokenizer.json"],
    );
    let mut mem =
        rlx_moonshine_streaming::MoonshineStreaming::open_source(&f.take_source(), Device::Cpu)
            .unwrap();
    let (a, b) = (disk.encode(&pcm).unwrap(), mem.encode(&pcm).unwrap());
    assert_eq!(a.hidden, b.hidden);
    let text = mem.transcribe(&pcm, None).unwrap();
    assert_eq!(disk.transcribe(&pcm, None).unwrap(), text);
    eprintln!("{text}");
}

#[test]
fn embedders_from_memory_match_directory() {
    let Some(root) = root() else { return };
    let text = "Mars, often called the Red Planet, has a thin atmosphere.";

    let dir = root.join("embed/embeddinggemma-300m");
    let mut disk = rlx_embeddinggemma::EmbeddingGemma::open(&dir, Device::Cpu).unwrap();
    let mut f = files(
        &dir,
        &[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "2_Dense/model.safetensors",
            "3_Dense/model.safetensors",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
        ],
    );
    let mut mem =
        rlx_embeddinggemma::EmbeddingGemma::open_source(&f.take_source(), Device::Cpu).unwrap();
    let p = disk.prompt("document").unwrap().to_string();
    assert_eq!(mem.prompt("document"), Some(p.as_str()));
    assert_eq!(
        disk.embed_text(&p, text).unwrap(),
        mem.embed_text(&p, text).unwrap()
    );

    let dir = root.join("embed/jina-embeddings-v5-text-nano");
    let mut disk = rlx_jina_v5::JinaV5::open(&dir, "retrieval", Device::Cpu).unwrap();
    let mut f = files(
        &dir,
        &[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "config_sentence_transformers.json",
            "adapters/retrieval/adapter_config.json",
            "adapters/retrieval/adapter_model.safetensors",
        ],
    );
    let mut mem =
        rlx_jina_v5::JinaV5::open_source(&f.take_source(), "retrieval", Device::Cpu).unwrap();
    let p = disk.prompt("document").unwrap().to_string();
    assert_eq!(
        disk.embed_text(&p, text).unwrap(),
        mem.embed_text(&p, text).unwrap()
    );
}

#[test]
fn lfm_from_memory_matches_directory() {
    let Some(root) = root() else { return };
    let dir = root.join("lm/lfm2.5-230m");
    let disk = rlx_lfm::Lfm2GgufRunner::open(&dir, Device::Cpu).unwrap();
    let mut f = files(&dir, &["config.json", "model.safetensors"]);
    let mem = rlx_lfm::Lfm2GgufRunner::open_source(&f.take_source(), Device::Cpu).unwrap();
    let prompt = [1u32, 1098, 3374, 708, 803];
    assert_eq!(
        disk.predict_logits(&prompt).unwrap(),
        mem.predict_logits(&prompt).unwrap()
    );
    assert_eq!(
        disk.generate(&prompt, 8, |_| true).unwrap(),
        mem.generate(&prompt, 8, |_| true).unwrap()
    );
}
