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

//! NeuTTS's host logic as an RLX program, answering `synthesize`: the text
//! and the voice's reference text phonemized (espeak-ng, its data read from
//! the package), the prompt around the voice's reference codes (the host's
//! tokenizer), the backbone a token at a time (the package's `generate`
//! entry, sampled by the host from the stream `seed + step`), its speech
//! tokens' codes — and NeuCodec (`neucodec_embed`, `neucodec_decode`,
//! `neucodec_istft`) in rlx. `run` answers the codes alone.

use std::collections::HashMap;

use rlx_program::{Message, Program, Tensor, host, run};
use serde_json::{Value, json};

const TOKENIZER: &str = "sidecar:tokenizer.json";

struct Voice {
    codes: Vec<i32>,
    text: String,
}

struct NeuTts {
    language: String,
    espeak_dir: String,
    voices: HashMap<String, Voice>,
    default_voice: Option<String>,
    speech_base: u32,
    speech_count: u32,
    stop: u32,
    max_new_tokens: usize,
    temperature: f64,
    top_k: u64,
    top_p: f64,
    seed: u64,
    sample_rate: u32,
}

impl NeuTts {
    /// `text` in IPA: espeak-ng over the spans between punctuation, the
    /// punctuation kept where it was (phonemizer's `preserve_punctuation`,
    /// as NeuTTS phonemizes), whitespace runs one space.
    fn phonemize(&self, text: &str) -> Result<String, String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }
        let engine = espeak_ng::EspeakNg::with_data_dir(
            &self.language,
            std::path::Path::new(&self.espeak_dir),
        )
        .map_err(|e| format!("espeak-ng init for '{}': {e}", self.language))?;
        const PUNCT: &[char] = &['.', ',', '!', '?', ';', ':'];
        let mut out = String::new();
        let mut span = String::new();
        let flush = |span: &mut String, out: &mut String| -> Result<(), String> {
            if span.trim().is_empty() {
                span.clear();
                return Ok(());
            }
            let ipa = engine
                .text_to_phonemes(span.trim())
                .map_err(|e| format!("espeak-ng phonemize: {e}"))?;
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(ipa.trim());
            span.clear();
            Ok(())
        };
        for ch in text.chars() {
            if PUNCT.contains(&ch) {
                flush(&mut span, &mut out)?;
                out.push(ch);
            } else {
                span.push(ch);
            }
        }
        flush(&mut span, &mut out)?;
        Ok(out.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    /// The prompt's ids: NeuTTS' instruction, the reference and the input
    /// phonemes, then the reference's codes as speech tokens — tokenized
    /// whole, as the runner tokenizes it.
    fn prompt(&self, ref_ipa: &str, ipa: &str, ref_codes: &[i32]) -> Result<Vec<u32>, String> {
        let mut text = format!(
            "user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_ipa} {ipa}\
             <|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>"
        );
        for c in ref_codes {
            text.push_str(&format!("<|speech_{c}|>"));
        }
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": text})),
        )?;
        Ok(m.body["ids"]
            .as_array()
            .ok_or("no ids")?
            .iter()
            .filter_map(|v| v.as_u64().map(|v| v as u32))
            .collect())
    }

    /// The codes the backbone speaks after `prompt`: each sampled token
    /// one of the speech tokens a code, until the end of speech.
    fn codes(&self, prompt: &[u32], seed: u64, max_new: usize) -> Result<Vec<i32>, String> {
        let mut context = prompt.to_vec();
        let mut codes = Vec::new();
        for step in 0..max_new {
            let ids: Vec<i32> = context.iter().map(|&t| t as i32).collect();
            let out = host(
                "entry.run",
                Message::new(
                    json!({"entry": "generate"}),
                    vec![Tensor::i32("input_ids", vec![1, ids.len()], &ids)],
                ),
            )?;
            let logits = out
                .tensors
                .first()
                .ok_or("the backbone gave no logits")?
                .to_f32();
            let n = logits.len();
            let opts = json!({"seed": seed, "step": step, "temperature": self.temperature,
                              "top_k": self.top_k, "top_p": self.top_p});
            let next = host(
                "sample",
                Message::new(opts, vec![Tensor::f32("logits", vec![n], &logits)]),
            )?
            .body["id"]
                .as_u64()
                .ok_or("the host sampled no id")? as u32;
            if next == self.stop {
                break;
            }
            if (self.speech_base..self.speech_base + self.speech_count).contains(&next) {
                codes.push((next - self.speech_base) as i32);
            }
            context.push(next);
        }
        Ok(codes)
    }

    /// `codes` through NeuCodec: the FSQ rows, the decoder, the inverse STFT.
    fn decode(&self, codes: &[i32]) -> Result<Vec<f32>, String> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        if codes
            .iter()
            .any(|c| !(0..self.speech_count as i32).contains(c))
        {
            return Err(format!("a code outside 0..{}", self.speech_count));
        }
        let emb = run(
            "neucodec_embed",
            vec![Tensor::i32("codes", vec![codes.len()], codes)],
        )?;
        let mut emb = emb.into_iter().next().ok_or("NeuCodec gave no embedding")?;
        emb.name = "emb".into();
        let spec = run("neucodec_decode", vec![emb])?;
        let mut spec = spec.into_iter().next().ok_or("NeuCodec gave no spectrum")?;
        spec.name = "spec".into();
        let audio = run("neucodec_istft", vec![spec])?;
        Ok(audio.first().ok_or("NeuCodec gave no audio")?.to_f32())
    }

    /// The reference: `options.ref_codes` with `ref_ipa` (or `ref_text`),
    /// else the voice `options.voice` names (or the default).
    fn reference(&self, o: &Value) -> Result<(Vec<i32>, String), String> {
        if let Some(c) = o["ref_codes"].as_array() {
            let codes = c
                .iter()
                .filter_map(|v| v.as_i64().map(|v| v as i32))
                .collect();
            let ipa = match (o["ref_ipa"].as_str(), o["ref_text"].as_str()) {
                (Some(ipa), _) => ipa.to_string(),
                (None, Some(text)) => self.phonemize(text)?,
                (None, None) => return Err("ref_codes need ref_ipa or ref_text".into()),
            };
            return Ok((codes, ipa));
        }
        let name = o["voice"]
            .as_str()
            .map(String::from)
            .or_else(|| self.default_voice.clone())
            .ok_or("no voice: give options.voice, or ref_codes and ref_text")?;
        let v = self.voices.get(&name).ok_or_else(|| {
            let mut have: Vec<&String> = self.voices.keys().collect();
            have.sort();
            format!("no voice '{name}' (have {have:?})")
        })?;
        Ok((v.codes.clone(), self.phonemize(&v.text)?))
    }
}

impl Program for NeuTts {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| config[k].as_f64().ok_or(format!("config has no {k}"));
        let s = |k: &str| {
            config[k]
                .as_str()
                .map(String::from)
                .ok_or(format!("config has no {k}"))
        };
        let mut voices = HashMap::new();
        for (name, v) in config["voices"].as_object().ok_or("config has no voices")? {
            let codes = v["codes"]
                .as_array()
                .ok_or("a voice has no codes")?
                .iter()
                .filter_map(|c| c.as_i64().map(|c| c as i32))
                .collect();
            let text = v["text"].as_str().ok_or("a voice has no text")?.into();
            voices.insert(name.clone(), Voice { codes, text });
        }
        Ok(NeuTts {
            language: s("language")?,
            espeak_dir: s("espeak")?,
            voices,
            default_voice: config["default_voice"].as_str().map(String::from),
            speech_base: u("speech_base")? as u32,
            speech_count: u("speech_count")? as u32,
            stop: u("stop")? as u32,
            max_new_tokens: u("max_new_tokens")? as usize,
            temperature: f("temperature")?,
            top_k: u("top_k")?,
            top_p: f("top_p")?,
            seed: u("seed")?,
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        // `run`: the codes for `options.text` (or `options.ipa`), as `[n]` i32.
        let text = match task {
            "synthesize" => req.body["text"].as_str().unwrap_or(""),
            "run" => o["text"].as_str().unwrap_or(""),
            _ => return Err(format!("NeuTTS answers synthesize and run, not {task}")),
        };
        // `options.codes`: codes to decode, the backbone skipped.
        let codes: Vec<i32> = match o["codes"].as_array() {
            Some(c) => c
                .iter()
                .filter_map(|v| v.as_i64().map(|v| v as i32))
                .collect(),
            None => {
                let ipa = match o["ipa"].as_str() {
                    Some(ipa) => ipa.to_string(),
                    None => self.phonemize(text)?,
                };
                if ipa.is_empty() {
                    return Err("nothing to say: give text (or options.ipa)".into());
                }
                let (ref_codes, ref_ipa) = self.reference(o)?;
                let prompt = self.prompt(&ref_ipa, &ipa, &ref_codes)?;
                let seed = o["seed"].as_u64().unwrap_or(self.seed);
                let max_new = o["max_new_tokens"]
                    .as_u64()
                    .map_or(self.max_new_tokens, |v| v as usize);
                self.codes(&prompt, seed, max_new)?
            }
        };
        if task == "run" {
            let n = codes.len();
            return Ok(Message::tensors(vec![Tensor::i32(
                "codes",
                vec![n],
                &codes,
            )]));
        }
        let wav = self.decode(&codes)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(NeuTts);
