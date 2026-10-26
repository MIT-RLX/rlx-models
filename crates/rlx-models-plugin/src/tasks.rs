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

//! The non-text families, one task trait and one `Provider` each: what a
//! family takes to become `Model.open(path).transcribe(…)` / `.embed(…)` in
//! every language.

/// A checkpoint directory, or the directory of a file in one.
#[allow(dead_code)]
fn checkpoint_dir(path: &std::path::Path) -> &std::path::Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    }
}

/// Whisper: speech recognition (`model_type: whisper`).
#[cfg(feature = "whisper")]
pub(crate) mod whisper {
    use anyhow::{Result, bail};
    use rlx_models::whisper::{WhisperRunner, WhisperRunnerBuilder};
    use rlx_runtime::model::{Claim, Provider, SpeechToText, speech_to_text_model};
    use serde_json::Value;

    struct Whisper(WhisperRunner);

    /// Linear resampling to Whisper's 16 kHz.
    pub(super) fn to_16k(audio: &[f32], rate: u32) -> Vec<f32> {
        if rate == 16_000 || audio.is_empty() {
            return audio.to_vec();
        }
        let step = rate as f64 / 16_000.0;
        let n = (audio.len() as f64 / step) as usize;
        (0..n)
            .map(|i| {
                let x = i as f64 * step;
                let (j, f) = (x as usize, (x.fract()) as f32);
                let a = audio[j];
                let b = *audio.get(j + 1).unwrap_or(&a);
                a + (b - a) * f
            })
            .collect()
    }

    impl SpeechToText for Whisper {
        fn family(&self) -> &str {
            "whisper"
        }

        fn transcribe(
            &mut self,
            audio: &[f32],
            sample_rate: u32,
            options: &Value,
        ) -> Result<String> {
            if sample_rate == 0 {
                bail!("whisper: sample_rate must be positive");
            }
            let pcm = to_16k(audio, sample_rate);
            let beam = rlx_runtime::model::option(options, "beam_size").and_then(Value::as_u64);
            let text = match beam {
                Some(b) if b > 1 => self.0.transcribe_beam(&pcm)?,
                _ => self.0.transcribe_greedy(&pcm)?,
            };
            Ok(text.trim().to_string())
        }
    }

    pub static WHISPER: Provider =
        Provider::new("rlx-models-whisper", "speech-recognition", |req| {
            let dir = super::checkpoint_dir(req.path());
            let weights = if req.path().is_file() {
                req.path().to_path_buf()
            } else {
                dir.join("model.safetensors")
            };
            let mut b = WhisperRunnerBuilder::default()
                .weights(weights)
                .config_path(dir.join("config.json"))
                .tokenizer_path(dir.join("tokenizer.json"))
                .device(req.device);
            if let Some(lang) = req.option("language").and_then(Value::as_str) {
                b = b.language(lang);
            }
            Ok(speech_to_text_model(Whisper(b.build()?), req.model_info()))
        })
        .description("Whisper speech recognition (Hugging Face checkpoints)")
        .families(&["whisper"])
        .claims(&[Claim::HfModelType("whisper")]);
}

/// BERT-family sentence embeddings (`model_type: bert` / `xlm-roberta`).
#[cfg(feature = "embed")]
pub(crate) mod embed {
    use anyhow::{Context, Result};
    use rlx_models::embed::{BertTokenizer, Pooling, RlxBertModel, embed_with_rlx};
    use rlx_runtime::model::{Claim, Provider, TextEmbedding, text_embedding_model};
    use serde_json::Value;

    struct Embedder {
        family: String,
        model: RlxBertModel,
        tokenizer: BertTokenizer,
        pooling: Pooling,
    }

    impl TextEmbedding for Embedder {
        fn family(&self) -> &str {
            &self.family
        }

        fn dim(&self) -> usize {
            self.model.hidden_size()
        }

        fn embed(&mut self, texts: &[&str], _: &Value) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(8) {
                out.extend(embed_with_rlx(
                    &mut self.model,
                    &self.tokenizer,
                    chunk,
                    self.pooling,
                )?);
            }
            Ok(out)
        }
    }

    pub static EMBED: Provider = Provider::new("rlx-models-embed", "embedding", |req| {
        let dir = super::checkpoint_dir(req.path());
        let max_len = req
            .option("max_length")
            .and_then(Value::as_u64)
            .unwrap_or(256) as usize;
        // `pooling`: "mean" (default) or "cls" (BGE-style checkpoints).
        let pooling = match req.option("pooling").and_then(Value::as_str) {
            Some("cls") => Pooling::Cls,
            Some(_) => Pooling::Mean,
            None if dir.to_string_lossy().to_lowercase().contains("bge") => Pooling::Cls,
            None => Pooling::Mean,
        };
        let weights = dir.join("model.safetensors");
        let model = RlxBertModel::load(
            &dir.join("config.json"),
            weights.to_str().context("weights path is not UTF-8")?,
        )?;
        let tokenizer = BertTokenizer::from_dir(dir, max_len)?;
        Ok(text_embedding_model(
            Embedder {
                family: rlx_runtime::model::tasks::hf_model_type(dir)
                    .unwrap_or_else(|| "bert".into()),
                model,
                tokenizer,
                pooling,
            },
            req.model_info(),
        ))
    })
    .description("BERT / XLM-R sentence embeddings (Hugging Face checkpoints)")
    .families(&["bert", "xlm-roberta"])
    .claims(&[
        Claim::HfModelType("bert"),
        Claim::HfModelType("xlm-roberta"),
    ]);
}

#[cfg(all(test, feature = "whisper"))]
mod tests {
    #[test]
    fn resampling_keeps_duration_and_shape() {
        let ramp: Vec<f32> = (0..44_100).map(|i| i as f32 / 44_100.0).collect();
        let out = super::whisper::to_16k(&ramp, 44_100);
        assert_eq!(out.len(), 16_000);
        assert!((out[8_000] - 0.5).abs() < 1e-3, "{}", out[8_000]);
        assert_eq!(super::whisper::to_16k(&ramp, 16_000).len(), ramp.len());
    }
}
