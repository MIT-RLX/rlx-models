// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard parity tests — validates codec ops, config parsing,
//! weight key layout, and synthesis pipeline against the local bundle.

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rlx_gepard::{
        codec_ops::{NUM_CHANNELS, PACKED_VOCAB, dequantize_frame, fold_channels, unfold_codes},
        config::{CodecConfig, GepardConfig},
        synthesis::GepardSynthesizer,
        weights::GepardOverlay,
    };

    fn bundle_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/gepard")
    }

    // ── codec ops ────────────────────────────────────────────────────────────

    #[test]
    fn test_codec_unfold_fold_roundtrip() {
        for seed in [0u32, 1, 42, 999, PACKED_VOCAB - 1] {
            let codes: Vec<u32> = (0..8).map(|i| (seed + i * 13) % PACKED_VOCAB).collect();
            let ch = unfold_codes(&codes).unwrap();
            let back = fold_channels(&ch).unwrap();
            assert_eq!(codes, back, "roundtrip failed for seed={seed}");
        }
    }

    #[test]
    fn test_codec_channel_count() {
        let codes = vec![0u32; 8];
        let ch = unfold_codes(&codes).unwrap();
        assert_eq!(ch.len(), NUM_CHANNELS);
    }

    #[test]
    fn test_codec_dequantize_range() {
        let codes = vec![0u32, 42, 100, 500, 1000, 1500, 2000, 2015];
        let ch = unfold_codes(&codes).unwrap();
        let dq = dequantize_frame(&ch);
        for &v in &dq {
            assert!(v >= -1.0 && v <= 1.0, "dequantised value {v} out of [-1,1]");
        }
    }

    // ── config ────────────────────────────────────────────────────────────────

    #[test]
    fn test_codec_config_defaults() {
        let c = CodecConfig::nanocodec_defaults();
        assert_eq!(c.num_channels(), 32);
        assert_eq!(c.channel_vocabs().len(), 32);
        assert_eq!(c.channel_vocabs()[0], 8);
        assert_eq!(c.frame_rate_hz, 21.5);
        assert_eq!(c.sample_rate, 22050);
    }

    #[test]
    fn test_gepard_config_from_bundle() {
        let p = bundle_path().join("gepard_config.json");
        if !p.is_file() {
            eprintln!("skipping: no gepard_config.json at {}", p.display());
            return;
        }
        let cfg = GepardConfig::from_path(&p).expect("parse gepard_config.json");
        assert_eq!(cfg.hidden_size(), 1024);
        assert!(cfg.num_audio_heads() > 0);
    }

    // ── weight keys ───────────────────────────────────────────────────────────

    #[test]
    fn test_weight_key_layout() {
        use rlx_gepard::weights::{
            audio_emb_key, backbone_embed_key, backbone_layer_key, codebook_head_key, stop_head_key,
        };
        assert_eq!(audio_emb_key(0), "audio_embeddings.0.weight");
        assert_eq!(audio_emb_key(31), "audio_embeddings.31.weight");
        assert_eq!(codebook_head_key(0, "weight"), "codebook_heads.0.weight");
        assert_eq!(codebook_head_key(31, "weight"), "codebook_heads.31.weight");
        assert_eq!(stop_head_key("bias"), "stop_head.bias");
        assert_eq!(backbone_embed_key(), "model.embed_tokens.weight");
        assert_eq!(
            backbone_layer_key(0, "self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn test_expected_overlay_key_count() {
        let keys = GepardOverlay::expected_keys(32);
        // 32 audio_emb + 4 proj + 1 scale + 64 codebook + 2 stop = 103
        assert_eq!(keys.len(), 103);
    }

    // ── bundle weights ────────────────────────────────────────────────────────

    #[test]
    fn test_bundle_weights_exist_and_parseable() {
        let p = bundle_path().join("weights.safetensors");
        if !p.is_file() {
            eprintln!("skipping: no bundle weights at {}", p.display());
            return;
        }
        let bytes = std::fs::read(&p).expect("read bundle weights");
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .expect("parse bundle weights safetensors");
        let names: Vec<_> = st.names().into_iter().collect();
        assert!(!names.is_empty(), "bundle weights has no tensors");
        // Must have at least audio embedding and codebook head tensors
        assert!(
            names.iter().any(|n| n.starts_with("audio_embeddings.")),
            "expected audio_embeddings.* in bundle"
        );
        assert!(
            names.iter().any(|n| n.starts_with("codebook_heads.")),
            "expected codebook_heads.* in bundle"
        );
    }

    // ── synthesis pipeline ────────────────────────────────────────────────────

    #[test]
    fn test_synthesis_produces_audio() {
        let bundle = bundle_path();
        if !bundle.join("model.safetensors").is_file() || !bundle.join("tokenizer.json").is_file() {
            return;
        }
        let synth = GepardSynthesizer::new(&bundle).expect("open");
        let audio = synth.synthesize("Hello from Gepard.", "").unwrap();
        assert!(!audio.is_empty(), "synthesize must return audio");
        assert!(
            audio.iter().any(|v| v.abs() > 1e-4),
            "synthesized audio must be non-trivially nonzero"
        );
    }

    #[test]
    fn test_synthesis_deterministic() {
        let bundle = bundle_path();
        if !bundle.join("model.safetensors").is_file() || !bundle.join("tokenizer.json").is_file() {
            return;
        }
        let synth = GepardSynthesizer::new(&bundle).expect("open");
        let a1 = synth.synthesize("same text", "").unwrap();
        let a2 = synth.synthesize("same text", "").unwrap();
        assert_eq!(a1, a2, "synthesize must be deterministic");
    }

    #[test]
    fn test_synthesis_varies_with_text() {
        let bundle = bundle_path();
        if !bundle.join("model.safetensors").is_file() || !bundle.join("tokenizer.json").is_file() {
            return;
        }
        let synth = GepardSynthesizer::new(&bundle).expect("open");
        let a = synth.synthesize("Hello there.", "").unwrap();
        let b = synth
            .synthesize("Completely different text content.", "")
            .unwrap();
        // The outputs should differ (even if both are fallback paths)
        let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(differs, "different texts should produce different audio");
    }

    #[test]
    fn test_bundle_weights_validated_per_model() {
        let p = bundle_path().join("weights.safetensors");
        if !p.is_file() {
            eprintln!("skipping: no bundle weights");
            return;
        }
        let bytes = std::fs::read(&p).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();

        // Validate the overlay loads without error
        let overlay = GepardOverlay::load(&st, 32).expect("load overlay from bundle");
        assert_eq!(overlay.audio_embeddings.len(), 32);
        assert_eq!(overlay.codebook_weights.len(), 32);
        assert_eq!(overlay.codebook_biases.len(), 32);
        assert!(
            overlay.audio_embed_scale > 0.0,
            "audio_embed_scale must be positive"
        );
    }

    #[test]
    #[ignore]
    fn test_gepard_real_weights_parity() {
        // Requires: RLX_GEPARD_DIR env pointing to a full checkpoint
        // python: from gepard.inference.runner import GepardRunner; ...
        eprintln!("Set RLX_GEPARD_DIR to test against real weights");
    }

    // ── numerical parity: Rust vs PyTorch overlay ─────────────────────────────

    /// Generate a numpy-style reference computation in Python and compare with
    /// the Rust implementation.  Checks audio-embed MLP and one codebook head.
    #[test]
    fn test_overlay_numerical_parity_vs_python() {
        use std::process::Command;

        let bundle = bundle_path();
        if !bundle.join("weights.safetensors").is_file() {
            eprintln!("skipping: no bundle weights");
            return;
        }

        // Run the Python reference computation
        let py_script = r#"
import sys, json
from safetensors.torch import load_file
import torch, torch.nn.functional as F

bundle = sys.argv[1]
st = load_file(bundle + "/weights.safetensors")

audio_embed_dim = 32
hidden = 1024
fsq_levels = [8,7,6,6]
# Use code = i % vocab_size so every code is in-range
vocabs = [fsq_levels[i % 4] for i in range(32)]
channels_in = [i % v for i, v in enumerate(vocabs)]

parts = []
for i, code in enumerate(channels_in):
    tbl = st[f"audio_embeddings.{i}.weight"]  # [L_i, 32]
    parts.append(tbl[code])
concat = torch.cat(parts).float()

h = F.linear(concat, st["audio_embed_proj.0.weight"].float(), st["audio_embed_proj.0.bias"].float())
h = F.gelu(h, approximate="tanh")
h = F.linear(h, st["audio_embed_proj.2.weight"].float(), st["audio_embed_proj.2.bias"].float())

# Affine-free LayerNorm (mean-center + RMS) — matches Rust embed_audio_frame.
mean = h.mean()
var = ((h - mean).pow(2)).mean()
h = (h - mean) / (var + 1e-5).sqrt()
scale = st["audio_embed_scale"].float().squeeze().item()
h = h * scale

logits = F.linear(h, st["codebook_heads.0.weight"].float(), st["codebook_heads.0.bias"].float())
pred = logits.argmax().item()

print(json.dumps({
    "embed_sample": h[:4].tolist(),
    "logits_sample": logits[:4].tolist(),
    "pred_ch0": int(pred),
    "channels_in": channels_in,
}))
"#;

        let output = Command::new("python3")
            .args(["-c", py_script, bundle.to_str().unwrap()])
            .output();

        let output = match output {
            Ok(o) if o.status.success() => o,
            Ok(o) => {
                eprintln!("python3 failed: {}", String::from_utf8_lossy(&o.stderr));
                return; // skip if Python unavailable
            }
            Err(e) => {
                eprintln!("python3 not available: {e}");
                return;
            }
        };

        let py_result: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("parse python output");

        // Run the Rust overlay forward on the same input
        let bytes = std::fs::read(bundle.join("weights.safetensors")).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let overlay = rlx_gepard::weights::GepardOverlay::load(&st, 32).unwrap();

        // Same valid codes as the Python script
        let fsq_levels = [8u32, 7, 6, 6];
        let channels_in: Vec<u32> = (0..32u32)
            .map(|i| i % fsq_levels[(i % 4) as usize])
            .collect();
        let emb = rlx_gepard::synthesis::embed_audio_frame(&channels_in, &overlay, 32, 1024);

        // Extract codebook head 0 prediction
        use rlx_gepard::backbone::matvec;
        let logits = matvec(
            &overlay.codebook_weights[0],
            &emb,
            Some(&overlay.codebook_biases[0]),
            1024,
            overlay.codebook_weights[0].len() / 1024,
        );
        let rust_pred = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        // Compare embed sample (first 4 values)
        let py_embed: Vec<f64> = py_result["embed_sample"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let rust_embed = &emb[..4];
        eprintln!("Python embed[:4]: {py_embed:?}");
        eprintln!("Rust   embed[:4]: {rust_embed:?}");

        for (py, rs) in py_embed.iter().zip(rust_embed) {
            let diff = (py - *rs as f64).abs();
            assert!(
                diff < 1e-4,
                "audio-embed MLP mismatch: py={py:.6} rust={rs:.6} diff={diff:.2e}"
            );
        }

        // Compare codebook head 0 prediction
        let py_pred = py_result["pred_ch0"].as_u64().unwrap() as u32;
        assert_eq!(
            rust_pred, py_pred,
            "codebook head 0 greedy prediction mismatch: py={py_pred} rust={rust_pred}"
        );

        eprintln!("✅ Overlay parity: Python ≡ Rust (diff < 1e-4, codebook head 0 matches)");
    }
}
