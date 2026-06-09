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

use crate::generation::GenerationConfig;
use crate::options::VoxtralTtsRunnerBuilder;
use crate::speech_tokenizer::{SpeechTokenizer, default_prompt_tokens_path, resolve_voice};
use crate::tokens::{DEFAULT_CFG_ALPHA, PRESET_VOICES};
use crate::voice_pt::convert_preset_voices;
use anyhow::{Result, bail};
use rlx_cli::parse_standard_device;
use rlx_core::STANDARD_DEVICE_NAMES;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut text: Option<String> = None;
    let mut voice = "neutral_female".to_string();
    let mut out = PathBuf::from("voxtral_tts_out.wav");
    let mut gen_cfg = GenerationConfig {
        cfg_alpha: DEFAULT_CFG_ALPHA,
        ..Default::default()
    };
    let mut decode_codes: Option<PathBuf> = None;
    let mut prompt_tokens: Option<PathBuf> = None;
    let mut write_prompt_tokens: Option<PathBuf> = None;
    let mut list_voices = false;
    let mut convert_voices = false;
    let mut tokenize_only = false;
    let mut device = "cpu".to_string();
    let mut eager_lm = false;
    let mut eager_acoustic = false;
    let mut reference_wav: Option<PathBuf> = None;
    let mut voice_embedding: Option<PathBuf> = None;
    let mut encode_reference: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "--weights" => {
                model_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--text" => {
                text = Some(args[i + 1].clone());
                i += 2;
            }
            "--voice" | "-v" => {
                voice = args[i + 1].clone();
                i += 2;
            }
            "--output" | "-o" => {
                out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--cfg-alpha" => {
                gen_cfg.cfg_alpha = args[i + 1]
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--cfg-alpha"))?;
                i += 2;
            }
            "--seed" => {
                gen_cfg.seed = args[i + 1].parse().map_err(|_| anyhow::anyhow!("--seed"))?;
                i += 2;
            }
            "--max-frames" => {
                gen_cfg.max_frames = args[i + 1]
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--max-frames"))?;
                i += 2;
            }
            "--decode-codes" => {
                decode_codes = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--prompt-tokens" => {
                prompt_tokens = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--write-prompt-tokens" => {
                write_prompt_tokens = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--list-voices" => {
                list_voices = true;
                i += 1;
            }
            "--convert-voices" => {
                convert_voices = true;
                i += 1;
            }
            "--tokenize-only" => {
                tokenize_only = true;
                i += 1;
            }
            "--device" => {
                device = args[i + 1].clone();
                i += 2;
            }
            "--eager-lm" => {
                eager_lm = true;
                i += 1;
            }
            "--eager-acoustic" => {
                eager_acoustic = true;
                i += 1;
            }
            "--reference-wav" => {
                reference_wav = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--voice-embedding" => {
                voice_embedding = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--encode-reference" => {
                encode_reference = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--native" | "--rust-codec" => {
                bail!("removed: native synthesis is the default (--text or --prompt-tokens)");
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    if list_voices {
        for v in PRESET_VOICES {
            println!("{v}");
        }
        return Ok(());
    }

    let model_dir = model_dir.ok_or_else(|| anyhow::anyhow!("--model-dir is required"))?;

    if convert_voices {
        convert_preset_voices(&model_dir)?;
        return Ok(());
    }

    if tokenize_only && text.is_none() {
        bail!("--tokenize-only requires --text");
    }

    let device = parse_standard_device("voxtral-tts", &device)?;
    let mut runner = VoxtralTtsRunnerBuilder::default()
        .model_dir(&model_dir)
        .device(device)
        .eager_lm(eager_lm)
        .eager_acoustic(eager_acoustic)
        .build()?;

    if let Some(ref_path) = encode_reference {
        if reference_wav.is_none() {
            bail!("--encode-reference requires --reference-wav");
        }
        let ref_wav = reference_wav.as_ref().unwrap();
        eprintln!(
            "[rlx-voxtral-tts] encode reference {} -> {}",
            ref_wav.display(),
            ref_path.display()
        );
        let emb = runner.encode_reference_to_file(ref_wav, &ref_path, "cloned")?;
        eprintln!(
            "[rlx-voxtral-tts] wrote {} frames x hidden={}",
            emb.n_tokens, emb.hidden
        );
        return Ok(());
    }

    let synthesize_cloned_direct = reference_wav.is_some() && text.is_some();

    let prompt_ids = if synthesize_cloned_direct {
        None
    } else if let Some(ref t) = text {
        let tok = SpeechTokenizer::from_model_dir(&model_dir)?;
        let ids = if let Some(ref emb_path) = voice_embedding {
            let hidden = runner.config().text_config.hidden_size;
            let emb = crate::voice::VoiceEmbedding::load_f32(emb_path, "custom", hidden)?;
            tok.encode_speech_with_n_audio(t, emb.n_tokens as u32)?
        } else {
            let _ = resolve_voice(&model_dir, &voice)?;
            tok.encode_speech(t, &voice)?
        };
        if let Some(ref out_path) = write_prompt_tokens {
            SpeechTokenizer::write_prompt_tokens(out_path, &ids)?;
            eprintln!(
                "[rlx-voxtral-tts] wrote {} ({} tokens)",
                out_path.display(),
                ids.len()
            );
        }
        if tokenize_only {
            return Ok(());
        }
        Some(ids)
    } else if let Some(ref path) = prompt_tokens {
        if text.is_some() {
            eprintln!("[rlx-voxtral-tts] note: --text ignored; using --prompt-tokens");
        }
        Some(crate::prompt_tokens::load_prompt_tokens(path)?)
    } else {
        None
    };

    if let Some(codes_path) = decode_codes {
        eprintln!(
            "[rlx-voxtral-tts] decode codes={} -> {}",
            codes_path.display(),
            out.display()
        );
        runner.decode_codes_file(&codes_path, &out)?;
    } else if synthesize_cloned_direct {
        let t = text.as_ref().unwrap();
        let ref_wav = reference_wav.as_ref().unwrap();
        eprintln!(
            "[rlx-voxtral-tts] model={} device={device} clone-from={} cfg_alpha={} seed={}",
            model_dir.display(),
            ref_wav.display(),
            gen_cfg.cfg_alpha,
            gen_cfg.seed
        );
        runner.synthesize_cloned_with_text(t, ref_wav, &out, &gen_cfg)?;
    } else {
        let tokens = prompt_ids.ok_or_else(|| {
            anyhow::anyhow!(
                "synthesis requires --text or --prompt-tokens PATH.\n\
                 Example: just voxtral-tts -- --text \"Hello\" --voice {voice} -o out.wav"
            )
        })?;
        eprintln!(
            "[rlx-voxtral-tts] model={} device={device} voice={voice} tokens={} cfg_alpha={} seed={}",
            model_dir.display(),
            tokens.len(),
            gen_cfg.cfg_alpha,
            gen_cfg.seed
        );
        if let Some(ref_wav) = reference_wav {
            runner.synthesize_cloned_from_wav(&tokens, &ref_wav, &out, &gen_cfg)?;
        } else if let Some(emb_path) = voice_embedding {
            let hidden = runner.config().text_config.hidden_size;
            let voice_emb = crate::voice::VoiceEmbedding::load_f32(&emb_path, "custom", hidden)?;
            runner.synthesize_native_with_voice(&tokens, &voice_emb, &out, &gen_cfg)?;
        } else {
            runner.synthesize_native(&tokens, &voice, &out, &gen_cfg)?;
        }
    }

    eprintln!("[rlx-voxtral-tts] wrote {}", out.display());
    Ok(())
}

fn print_help() {
    eprintln!(
        "rlx-voxtral-tts — Mistral Voxtral-4B-TTS (native Rust)\n\
         \n\
         Flags:\n\
           --model-dir PATH      HF checkpoint dir (consolidated.safetensors)\n\
           --text STRING         Speech text (native Tekken tokenization)\n\
           --prompt-tokens PATH  Pre-tokenized speech prompt ids\n\
           --write-prompt-tokens PATH  Save token ids from --text\n\
           --tokenize-only       With --text: write tokens and exit\n\
           --convert-voices      Convert voice_embedding/*.pt -> .f32 and exit\n\
           --decode-codes PATH   Decode exported discrete codes with native codec\n\
           --voice NAME          Preset voice (default: neutral_female)\n\
           --reference-wav PATH  Clone from mono reference audio (needs encoder in checkpoint)\n\
           --voice-embedding PATH  Pre-encoded .f32 voice (from --encode-reference)\n\
           --encode-reference PATH  Encode --reference-wav to .f32 and exit\n\
           --output PATH         Output wav (default: voxtral_tts_out.wav)\n\
           --cfg-alpha F         Flow-matching CFG (default: 1.2)\n\
           --seed N              Flow-matching RNG seed (default: 42)\n\
           --max-frames N        Max audio frames (default: 2500)\n\
           --device NAME         Execution backend ({STANDARD_DEVICE_NAMES})\n\
           --eager-lm            Hand-ported CPU LM (debug / parity)\n\
           --eager-acoustic      Hand-ported CPU acoustic stack\n\
           --list-voices         Print preset voice names\n\
         \n\
         Native workflow:\n\
           just fetch-voxtral-tts\n\
           just voxtral-tts-prepare-voices\n\
           just voxtral-tts -- --model-dir $RLX_VOXTRAL_TTS_DIR \\\n\
             --text \"Hello world\" --voice neutral_female -o out.wav\n\
         \n\
         Optional vLLM parity (Docker ref only): just test-voxtral-tts-parity\n\
         Default prompt token dump path: {}",
        default_prompt_tokens_path().display()
    );
}

#[cfg(test)]
mod tests {
    use crate::prompt_tokens::load_prompt_tokens;
    use std::io::Write;

    #[test]
    fn load_prompt_tokens_parses_whitespace() {
        let dir = std::env::temp_dir().join(format!("rlx_voxtral_tts_tok_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tok.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "1 24 24 35").unwrap();
        let ids = load_prompt_tokens(&path).unwrap();
        assert_eq!(ids, vec![1, 24, 24, 35]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
