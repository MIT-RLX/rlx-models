# rlx-models — shortcuts for per-crate CLIs and common cargo tasks.
# Install: https://github.com/casey/just — `brew install just` or `cargo install just`
#
#   just                  # list recipes
#   just qwen3 -- --weights model.gguf --prompt-ids 1,2,3
#   just inspect weights/model.gguf

# `true` | `false` — e.g. `just release=false check`
release := "true"

# Bundled JFK reference audio (voice clone demos / eval WAV).
jfk_ref_wav := "assets/jfk/jfk_voice_clone.wav"

profile := if release == "true" { "--release" } else { "" }

# Optional cargo features (comma-separated), e.g. `just features=metal qwen3-metal -- …`
features := ""

feature_args := if features != "" { "--features " + features } else { "" }

# rlx-minicpm5 binary requires the `tokenizer` cargo feature (not a CLI flag).
minicpm5_feature_args := if features != "" { "--features " + features + ",tokenizer" } else { "--features tokenizer" }

# Run a per-crate binary (fast link). Pass CLI flags after `--`.
[private]
run-bin package bin *ARGS:
    cargo run -p {{package}} --bin {{bin}} {{profile}} {{feature_args}} -- {{ARGS}}

[private]
run-minicpm5 *ARGS:
    cargo run -p rlx-minicpm5 --bin rlx-minicpm5 {{profile}} {{minicpm5_feature_args}} -- {{ARGS}}

# Multiplexer (links all models). Subcommand is first arg after `--`.
[private]
run-rlx *ARGS:
    cargo run -p rlx-models --bin rlx-run {{profile}} {{feature_args}} -- {{ARGS}}

default:
    @just --list

# --- workspace ---

check:
    cargo check --workspace

publish-list:
    ./scripts/publish.sh --list

publish-dry-run:
    ./scripts/publish.sh --dry-run --yes

test *ARGS:
    cargo test -p rlx-models {{ARGS}}

test-quick:
    cargo test -p rlx-models --test qwen35_forward_check --test compile_profile_quick_check
    cargo test -p rlx-models --test gemma_backend_quick_check gemma_tiny --release
    cargo test -p rlx-gemma --lib gemma2_rms_ones --release

# Qwen3 synthetic prefill + generator on each backend.
#   just features=all-backends test-qwen3-backends
# Use `just release=true …` for faster GPU kernels.
test-qwen3-backends *ARGS:
    cargo test -p rlx-models --test qwen3_backend_quick_check --test qwen3_gpu_backend_parity {{profile}} {{feature_args}} {{ARGS}}

test-qwen35-backends *ARGS:
    cargo test -p rlx-models --test qwen35_backend_quick_check {{profile}} {{feature_args}} {{ARGS}}

# Gemma synthetic prefill + generator on each backend.
#   just features=all-backends test-gemma-backends
test-gemma-backends *ARGS:
    cargo test -p rlx-models --test gemma_backend_quick_check --test gemma_gpu_backend_parity {{profile}} {{feature_args}} {{ARGS}}

test-gemma-parity *ARGS:
    cargo test -p rlx-models --test gemma_parity --features parity-candle --release {{ARGS}}

test-parity *ARGS:
    cargo test -p rlx-models --features parity-candle {{ARGS}}

# RLX OCR vs upstream ocrs crate (`parity-ocrs` feature; pipeline needs models).
test-ocr-parity *ARGS:
    cargo test -p rlx-ocr --test ocr_parity --features parity-ocrs --release {{ARGS}}

test-ocr-parity-full *ARGS:
    OCR_PARITY_DOWNLOAD=1 just test-ocr-parity ocr_pipeline_matches_reference -- --nocapture {{ARGS}}

# Fail if rlx-ocr get_text is not ≥5% faster than ocrs on this machine (needs models).
test-ocr-perf-gate *ARGS:
    OCR_PERF_GATE=1 cargo test -p rlx-ocr --test ocr_perf_vs_reference --features parity-ocrs --release -- --nocapture {{ARGS}}

test-ocr-perf-gate-download *ARGS:
    OCR_PERF_GATE=1 OCR_PARITY_DOWNLOAD=1 just test-ocr-perf-gate {{ARGS}}

# Convert `.rten` → `.safetensors` in a model dir (or single file pair).
ocr-convert *ARGS:
    cargo run -p rlx-ocr --features convert-rten --bin rlx-ocr-convert --release -- {{ARGS}}

# Latency: native rlx-ocr vs upstream ocrs (`parity-ocrs` + `convert-rten`; needs models + test image).
bench-ocr *ARGS:
    cargo bench -p rlx-ocr --features "parity-ocrs,convert-rten" --bench ocr_vs_reference --release {{ARGS}}

bench-ocr-download *ARGS:
    OCR_PARITY_DOWNLOAD=1 just bench-ocr {{ARGS}}

# Native RLX detection + get_text latency (safetensors; auto-converts from .rten on download).
bench-ocr-rlx *ARGS:
    cargo test -p rlx-ocr --test ocr_backend_bench --features "rlx,convert-rten" --release ocr_bench_report -- --nocapture {{ARGS}}

# Same with Metal (`--features rlx-ocr/metal`).
bench-ocr-rlx-metal *ARGS:
    OCR_DEVICE=metal cargo test -p rlx-ocr --features "rlx,convert-rten,metal" --test ocr_backend_bench --release ocr_bench_report -- --nocapture {{ARGS}}

bench-ocr-rlx-cuda *ARGS:
    OCR_DEVICE=cuda cargo test -p rlx-ocr --features "rlx,convert-rten,cuda" --test ocr_backend_bench --release ocr_bench_report -- --nocapture {{ARGS}}

bench-ocr-rlx-download *ARGS:
    OCR_PARITY_DOWNLOAD=1 just bench-ocr-rlx {{ARGS}}

# Real-weight quick check (safetensors in OCR_MODEL_DIR).
test-ocr-batch *ARGS:
    cargo test -p rlx-ocr --test ocr_batch_quick_check --features rlx --release {{ARGS}}

# OCR detection + recognition + pipeline on each backend (needs OCR_MODEL_DIR).
#   just features=all-backends test-ocr-backends
test-ocr-backends *ARGS:
    cargo test -p rlx-ocr --test ocr_backend_quick_check --features "rlx,convert-rten" {{profile}} {{feature_args}} {{ARGS}}

test-ocr-backends-download *ARGS:
    OCR_PARITY_DOWNLOAD=1 just test-ocr-backends {{ARGS}}

bench-ocr-batch *ARGS:
    just bench-ocr-rlx {{ARGS}}

bench-ocr-batch-download *ARGS:
    just bench-ocr-rlx-download {{ARGS}}

# Resize test page to several WxH targets (`OCR_BENCH_SIZES`).
bench-ocr-sizes *ARGS:
    cargo test -p rlx-ocr --test ocr_backend_bench --features "rlx,convert-rten" --release ocr_bench_image_sizes -- --nocapture {{ARGS}}

bench-ocr-sizes-download *ARGS:
    OCR_PARITY_DOWNLOAD=1 just bench-ocr-sizes {{ARGS}}

# Legacy RTen graph inference bench (baseline only).
bench-ocr-rten *ARGS:
    cargo test -p rlx-ocr --test ocr_backend_bench --features "rlx,convert-rten,rten-inference" --release ocr_bench_report -- --nocapture {{ARGS}}

build:
    cargo build --workspace {{profile}}

# --- inspect ---

inspect PATH:
    cargo run -p rlx-cli --bin rlx-inspect {{profile}} -- {{PATH}}

# --- per-model CLIs (preferred) ---

qwen3 *ARGS:
    just run-bin rlx-qwen3 rlx-qwen3 {{ARGS}}

qwen35 *ARGS:
    just run-bin rlx-qwen35 rlx-qwen35 {{ARGS}}

llama32 *ARGS:
    just run-bin rlx-llama32 rlx-llama32 {{ARGS}}

minicpm5 *ARGS:
    just run-minicpm5 {{ARGS}}

# Chat inference: HF chat template → rlx-minicpm5 (CPU fastest/reliable on Apple Silicon today).
minicpm5-chat MESSAGE *ARGS:
    RLX_MODELS_ROOT={{justfile_directory()}} python3 crates/rlx-models/examples/minicpm5_chat.py "{{MESSAGE}}" {{ARGS}}

# Build release binary once, then chat (avoids cargo startup each message).
minicpm5-chat-fast MESSAGE *ARGS:
    cargo build -p rlx-minicpm5 --features tokenizer,mlx,metal --release
    just minicpm5-chat "{{MESSAGE}}" {{ARGS}}

test-minicpm5-backends *ARGS:
    cargo test -p rlx-models --test minicpm5_backend_parity {{profile}} {{feature_args}} {{ARGS}}

# Synthetic graph: CPU vs Metal/MLX/CUDA/WGPU/…
test-minicpm5-backends-all *ARGS:
    just features=all-backends test-minicpm5-backends {{ARGS}}

# Real GGUF Q4_K_M packed prefill on each backend.
test-minicpm5-gguf-backends *ARGS:
    RLX_MINICPM5_GGUF_DIR={{real_weights_dir}}/MiniCPM5-1B-GGUF \
        cargo test -p rlx-models --test minicpm5_backend_gguf_check {{profile}} {{feature_args}} {{ARGS}}

test-minicpm5-gguf-quants *ARGS:
    RLX_MINICPM5_GGUF_DIR={{real_weights_dir}}/MiniCPM5-1B-GGUF \
        cargo test -p rlx-models --test minicpm5_quant_matrix {{profile}} {{feature_args}} minicpm5_quant_matrix -- --nocapture {{ARGS}}

fetch-minicpm5-gguf QUANT="Q4_K_M":
    MINICPM5_MODEL_DIR={{real_weights_dir}}/MiniCPM5-1B-GGUF \
    RLX_MINICPM5_GGUF_DIR={{real_weights_dir}}/MiniCPM5-1B-GGUF \
        cargo run -p rlx-models --example minicpm5_gguf_download --features hf-download --release -- {{QUANT}}

fetch-minicpm5-gguf-all:
    just fetch-minicpm5-gguf all

test-minicpm5-parity *ARGS:
    cargo test -p rlx-models --test minicpm5_parity --features parity-pytorch --release {{ARGS}}

bench-minicpm5 *ARGS:
    cargo bench -p rlx-models --bench minicpm5_inference {{ARGS}}

bench-minicpm5-all-backends *ARGS:
    cargo bench -p rlx-models --bench minicpm5_inference --features all-backends {{ARGS}}
    cargo test -p rlx-models --test minicpm5_bench_report --features all-backends --release minicpm5_bench_report -- --nocapture {{ARGS}}

# Real MiniCPM5-1B prefill + decode (needs `just fetch-minicpm5`).
bench-minicpm5-real *ARGS:
    RLX_MINICPM5_WEIGHTS={{real_weights_dir}}/MiniCPM5-1B/model-00000-of-00001.safetensors \
    MINICPM5_MODEL_DIR={{real_weights_dir}}/MiniCPM5-1B \
        cargo run -p rlx-models --example minicpm5_forward_bench --features "metal,mlx,cuda,rocm,gpu,vulkan" --release -- {{ARGS}}

# Real 1B weights on every RLX backend available on this machine.
bench-minicpm5-real-all-backends *ARGS:
    just bench-minicpm5-real --all-backends {{ARGS}}

gemma *ARGS:
    just run-bin rlx-gemma rlx-gemma {{ARGS}}

# Gemma 4 synthetic e2e bench (sequential — safe for GPU).
gemma4-bench *ARGS:
    cargo test -p rlx-gemma --release --features apple-silicon \
        --test gemma4_e2e_bench bench_e2e_all_backends -- --nocapture --test-threads=1 {{ARGS}}

gemma4-bench-metal *ARGS:
    cargo test -p rlx-gemma --release --features apple-silicon \
        --test gemma4_e2e_bench bench_synthetic_metal -- --nocapture --test-threads=1 {{ARGS}}

# Fast Metal smoke (~short_chat + image_caption only).
gemma4-bench-lite *ARGS:
    RLX_GEMMA4_BENCH_LITE=1 just gemma4-bench-metal {{ARGS}}

# Real Gemma 4 12B weights (needs RLX_GEMMA4_FIXTURE with HF download).
gemma4-bench-real *ARGS:
    RLX_GEMMA4_FIXTURE={{env_var_or_default('RLX_GEMMA4_FIXTURE', '/Users/Shared/rlx-models/.cache/gemma4-12B-it')}} \
    cargo test -p rlx-gemma --release --features apple-silicon \
        --test gemma4_e2e_bench bench_real_weights -- --nocapture --test-threads=1 {{ARGS}}

# Download google/gemma-4-12B-it into .cache/gemma4-12B-it (≈24 GB).
fetch-gemma4-12b-it:
    mkdir -p .cache/gemma4-12B-it
    hf download google/gemma-4-12B-it config.json tokenizer.json tokenizer_config.json model.safetensors \
        --local-dir .cache/gemma4-12B-it

# Q4_K_M GGUF for 64 GB hosts (≈7 GB; packed inference).
fetch-gemma4-12b-it-gguf:
    mkdir -p .cache/gemma4-12B-it
    hf download unsloth/gemma-4-12B-it-GGUF gemma-4-12b-it-Q4_K_M.gguf \
        --local-dir .cache/gemma4-12B-it
    test -f .cache/gemma4-12B-it/config.json || \
        hf download google/gemma-4-12B-it config.json tokenizer.json tokenizer_config.json \
            --local-dir .cache/gemma4-12B-it

# Download Gemma 4 12B GGUF quants for sweep (env RLX_GEMMA4_QUANTS=Q4_K_M,Q5_K_M,...).
fetch-gemma4-quants:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .cache/gemma4-12B-it
    quants="${RLX_GEMMA4_QUANTS:-Q3_K_S,Q3_K_M,Q4_0,Q4_1,Q4_K_S,Q4_K_M,Q5_K_S,Q5_K_M,Q6_K,Q8_0}"
    IFS=',' read -ra qs <<< "$quants"
    for q in "${qs[@]}"; do
        q="$(echo "$q" | tr -d ' ')"
        hf download unsloth/gemma-4-12B-it-GGUF "gemma-4-12b-it-${q}.gguf" \
            --local-dir .cache/gemma4-12B-it
    done
    test -f .cache/gemma4-12B-it/config.json || \
        hf download google/gemma-4-12B-it config.json tokenizer.json tokenizer_config.json \
            --local-dir .cache/gemma4-12B-it

# Speed + precision sweep across local Gemma 4 GGUF quants (Metal).
test-gemma4-quant-sweep *ARGS:
    cargo test -p rlx-gemma --release --features apple-silicon \
        --test gemma4_gguf_quant_sweep bench_all_quants -- --nocapture --test-threads=1 {{ARGS}}

# Isolated Metal step_cached throughput (hidden=1024, bucketed decode).
gemma4-decode-bench-metal *ARGS:
    cargo test -p rlx-gemma --release --features apple-silicon \
        --test gemma4_decode_throughput bench_step_cached_metal_1024 -- --nocapture {{ARGS}}

dinov2 *ARGS:
    just run-bin rlx-dinov2 rlx-dinov2 {{ARGS}}

vjepa2 *ARGS:
    just run-bin rlx-vjepa2 rlx-vjepa2 {{ARGS}}

wav2vec2 *ARGS:
    just run-bin rlx-wav2vec2-bert rlx-wav2vec2-bert {{ARGS}}

whisper *ARGS:
    just run-bin rlx-whisper rlx-whisper {{ARGS}}

fetch-whisper:
    # Minimal RLX Whisper layout (safetensors + config + tokenizer).
    mkdir -p .cache/whisper-tiny
    test -s .cache/whisper-tiny/model.safetensors || \
        curl -L --create-dirs -C - -o .cache/whisper-tiny/model.safetensors \
            'https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors'
    test -s .cache/whisper-tiny/config.json || \
        curl -L --create-dirs -C - -o .cache/whisper-tiny/config.json \
            'https://huggingface.co/openai/whisper-tiny/resolve/main/config.json'
    test -s .cache/whisper-tiny/tokenizer.json || \
        curl -L --create-dirs -C - -o .cache/whisper-tiny/tokenizer.json \
            'https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json'

fetch-whisper-base:
    huggingface-cli download openai/whisper-base.en --local-dir .cache/whisper-base.en

test-whisper-parity *ARGS:
    cargo test -p rlx-models --test whisper_parity --features parity-candle whisper_synthetic --release {{ARGS}}

test-whisper-backend-parity *ARGS:
    cargo test -p rlx-models --test whisper_backend_parity --features "metal,mlx,gpu" --release {{ARGS}}

test-whisper-wgpu-gpu-kv *ARGS:
    cargo test -p rlx-models --test whisper_wgpu_gpu_kv --features gpu --release {{ARGS}}

bench-whisper *ARGS:
    cargo run -p rlx-models --example whisper_bench --features "metal,mlx,apple-silicon" --release -- {{ARGS}}

bench-whisper-precision *ARGS:
    just bench-whisper --precision --all-backends {{ARGS}}

bench-whisper-all-backends *ARGS:
    just bench-whisper --all-backends {{ARGS}}

# VAD (Earshot + Silero on assets/jfk)
test-vad *ARGS:
    cargo test -p rlx-vad --release {{ARGS}}
    cargo test -p rlx-vad --no-default-features --features earshot --release {{ARGS}}
    cargo test -p rlx-vad --no-default-features --features silero --release {{ARGS}}

test-vad-backends *ARGS:
    cargo test -p rlx-vad --test backend_quick_check --features all-backends --release {{ARGS}}

vad *ARGS:
    just run-bin rlx-vad rlx-vad {{ARGS}}

bench-vad-jfk *ARGS:
    cargo run -p rlx-vad --example jfk_bench --release -- {{ARGS}}

bench-vad-jfk-all-devices *ARGS:
    cargo run -p rlx-vad --example jfk_bench --release --features all-backends -- --devices all {{ARGS}}

voxtral *ARGS:
    just run-bin rlx-voxtral rlx-voxtral {{ARGS}}

locateanything *ARGS:
    just run-bin rlx-locateanything rlx-locateanything {{ARGS}}

# GPU build (Metal on Apple Silicon). Default `locateanything` is CPU-only → very slow for 3B.
locateanything-metal *ARGS:
    just features=metal run-bin rlx-locateanything rlx-locateanything {{ARGS}}

# Ground "person" — Apple Silicon: Metal + MLX; Linux: CUDA when available.
locateanything-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS=(--task ground-single --phrase person --device auto --max-image-side 480 --max-tokens 32)
    case "$(uname -s)" in
      Darwin) just features=apple-silicon locateanything "${ARGS[@]}" ;;
      *)      just features=nvidia-gpu locateanything "${ARGS[@]}" 2>/dev/null \
                || just locateanything "${ARGS[@]}" ;;
    esac

locateanything-demo-metal *ARGS:
    just features=metal locateanything \
      --task ground-single --phrase person --device metal \
      --max-image-side 480 --max-tokens 32 {{ARGS}}

locateanything-demo-cuda *ARGS:
    just features=cuda locateanything \
      --task ground-single --phrase person --device cuda \
      --max-image-side 480 --max-tokens 32 {{ARGS}}

locateanything-all-backends *ARGS:
    just features=all-backends run-bin rlx-locateanything rlx-locateanything {{ARGS}}

test-locateanything-backends *ARGS:
    cargo test -p rlx-locateanything --test backend_quick_check {{profile}} {{feature_args}} {{ARGS}}

test-locateanything-moonvit-backends *ARGS:
    cargo test -p rlx-locateanything --test moonvit_backends {{profile}} {{feature_args}} {{ARGS}}

# CPU vs each available backend — grounding tokens on real weights (needs RLX_LOCATEANYTHING_DIR).
test-locateanything-grounding-parity *ARGS:
    RLX_LOCATEANYTHING_DIR=${RLX_LOCATEANYTHING_DIR:-.cache/locateanything/LocateAnything-3B} \
    cargo test -p rlx-locateanything --test backend_grounding_parity --features apple-silicon,tokenizer {{profile}} -- --test-threads 1 {{ARGS}}

bench-locateanything-backends *ARGS:
    cargo run -p rlx-models --example locateanything_bench --release {{feature_args}} -- --all-backends {{ARGS}}

fetch-locateanything:
    cargo run -p rlx-models --example locateanything_download --features hf-download --release
    just fetch-locateanything-tokenizer

# HF AutoTokenizer export into the snapshot (processor prompts). Uses HF cache pointer from fetch.
fetch-locateanything-tokenizer:
    #!/usr/bin/env bash
    set -euo pipefail
    CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
    DIR="${RLX_LOCATEANYTHING_DIR:-}"
    if [ -z "$DIR" ] || [ ! -f "$DIR/config.json" ]; then
      if [ -f "$CACHE/.rlx_locateanything_snapshot" ]; then
        DIR="$(cat "$CACHE/.rlx_locateanything_snapshot")"
      fi
    fi
    if [ ! -f "$DIR/config.json" ]; then
      echo "error: no checkpoint — run \`just fetch-locateanything\` first" >&2
      exit 1
    fi
    python3 scripts/export_locateanything_tokenizer.py --model-dir "$DIR"

test-locateanything-checkpoint: fetch-locateanything
    RLX_LOCATEANYTHING_DIR=${RLX_LOCATEANYTHING_DIR:-.cache/locateanything/LocateAnything-3B} \
    cargo test -p rlx-models --test locateanything_checkpoint --release

test-locateanything-parity: fetch-locateanything
    RLX_LOCATEANYTHING_DIR=${RLX_LOCATEANYTHING_DIR:-.cache/locateanything/LocateAnything-3B} \
    cargo test -p rlx-models --test locateanything_hf_parity --release -- --test-threads 1

# Real-photo HF parity (fixture JPEG + optional RLX_LOCATEANYTHING_IMAGE override).
test-locateanything-parity-real: fetch-locateanything
    RLX_LOCATEANYTHING_DIR=${RLX_LOCATEANYTHING_DIR:-.cache/locateanything/LocateAnything-3B} \
    cargo test -p rlx-models --test locateanything_hf_parity --release -- _real --test-threads 1

voxtral-tts *ARGS:
    just run-bin rlx-voxtral-tts rlx-voxtral-tts {{ARGS}}

# Stage timing on one Voxtral-4B-TTS checkpoint (RLX_VOXTRAL_TTS_DIR). --compare A/B compiled vs eager.
bench-voxtral-tts *ARGS:
    cargo run -p rlx-models --example voxtral_tts_bench --features "metal,mlx,apple-silicon" --release -- {{ARGS}}

fetch-qwen3-tts:
    cargo run -p rlx-models --example qwen3_tts_download --features hf-download --release

fetch-qwen3-tts-base:
    cargo run -p rlx-models --example qwen3_tts_download_base --features hf-download --release

# JFK clips + train_raw.jsonl (default: reference transcript alignment; JFK_TRANSCRIPT_MODE=whisper|hybrid)
qwen3-tts-jfk-prep:
    bash scripts/qwen3_tts_prep_jfk.sh

# Metal (Apple MPS): HF SFT on JFK chunks → speaker `jfk`.
qwen3-tts-train-jfk-metal *ARGS:
    BACKEND=metal bash scripts/qwen3_tts_finetune_jfk.sh {{ARGS}}

# MLX: HF codec prepare + native RLX talker LoRA on JFK.
qwen3-tts-train-jfk-mlx *ARGS:
    BACKEND=mlx bash scripts/qwen3_tts_finetune_jfk.sh {{ARGS}}

qwen3-tts-train-jfk *ARGS:
    just qwen3-tts-train-jfk-metal {{ARGS}}

# Fetch Base + JFK prep (141 clips) + train. Quick subset: MAX_CLIPS=32 EPOCHS=1 …
qwen3-tts-train-jfk-go *ARGS:
    bash scripts/qwen3_tts_train_go.sh {{ARGS}}

# Flags on the recipe (no extra `--`; run-bin adds it for cargo). Multi-word text: RLX_QWEN3_TTS_TEXT='…'
qwen3-tts *ARGS:
    just features=all-backends run-bin rlx-qwen3-tts rlx-qwen3-tts {{ARGS}}

qwen3-tts-jfk-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-.cache/qwen3-tts/jfk-checkpoint/checkpoint-epoch-2}"
    export RLX_QWEN3_TTS_TEXT="${RLX_QWEN3_TTS_TEXT:-We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard.}"
    just qwen3-tts --model-dir "$RLX_QWEN3_TTS_DIR" --speaker jfk --language English --out-wav /tmp/jfk-trained.wav --device auto

# HF inference for finetuned CustomVoice (reference path until native codec matches HF on JFK weights).
qwen3-tts-jfk-hf-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    ROOT="{{justfile_directory()}}"
    VENV="${VENV:-$ROOT/.venv-qwen3-tts-train}"
    MODEL_DIR="${RLX_QWEN3_TTS_DIR:-$ROOT/.cache/qwen3-tts/jfk-checkpoint/checkpoint-epoch-2}"
    TEXT="${RLX_QWEN3_TTS_TEXT:-We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard.}"
    set +a
    test -x "$VENV/bin/python" || { echo "missing $VENV — run just qwen3-tts-train-jfk first"; exit 1; }
    "$VENV/bin/python" "$ROOT/scripts/qwen3_tts_hf_infer.py" \
      --model-dir "$MODEL_DIR" --text "$TEXT" --speaker jfk --language english \
      --out-wav /tmp/jfk-trained-hf.wav --device mps --max-new-tokens 128

# Stock CustomVoice (vivian) — native RLX, fast sanity check.
# Duplex voice chat: bundled question WAV → Whisper → Qwen3 LM → JFK TTS reply.
voice-chat-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="{{justfile_directory()}}"
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-$ROOT/.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base}"
    VECLIB_MAXIMUM_THREADS=1 cargo run --release -p rlx-qwen3-tts --features apple-silicon \
      --example bidirectional_voice_chat -- --turbo \
      --ref-wav "$ROOT/assets/jfk/jfk_voice_clone.wav" \
      --input-wav "$ROOT/crates/rlx-qwen3-tts/examples/audio/voice_chat_question.wav" \
      --out-dir /tmp/voice_chat_roundtrip

qwen3-tts-vivian-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
    export RLX_QWEN3_TTS_TEXT="${RLX_QWEN3_TTS_TEXT:-Hello from RLX Qwen3-TTS.}"
    just qwen3-tts --model-dir "$RLX_QWEN3_TTS_DIR" --speaker vivian --language english --out-wav /tmp/vivian-demo.wav --device auto --max-frames 32

# HF reference (audible CustomVoice) — compare when native sounds wrong.
qwen3-tts-vivian-hf-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="{{justfile_directory()}}"
    VENV="${VENV:-{{justfile_directory()}}/.venv-qwen3-tts-train}"
    MODEL_DIR="${RLX_QWEN3_TTS_DIR:-{{justfile_directory()}}/.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
    TEXT="${RLX_QWEN3_TTS_TEXT:-Hello from RLX Qwen3-TTS.}"
    test -x "$VENV/bin/python" || { echo "missing $VENV — run just qwen3-tts-train-jfk first or pip install qwen-tts"; exit 1; }
    "$VENV/bin/python" "$ROOT/scripts/qwen3_tts_hf_infer.py" \
      --model-dir "$MODEL_DIR" --text "$TEXT" --speaker vivian --language english \
      --out-wav /tmp/vivian-hf-demo.wav --device mps --max-new-tokens 64

bench-qwen3-tts *ARGS:
    cargo run -p rlx-models --example qwen3_tts_bench --features "metal,mlx,apple-silicon" --release -- {{ARGS}}

bench-qwen3-tts-cp-ab *ARGS:
    VECLIB_MAXIMUM_THREADS=1 cargo run -p rlx-models --example qwen3_tts_cp_ab --release -- {{ARGS}}

bench-qwen3-tts-fusion-ab *ARGS:
    cargo run -p rlx-models --example qwen3_tts_fusion_ab --release --features metal -- {{ARGS}}

bench-qwen3-tts-session *ARGS:
    RLX_QWEN3_TTS_TIMING=1 cargo run -p rlx-models --example qwen3_tts_session_bench --features metal --release -- {{ARGS}}

bench-qwen3-tts-rtf *ARGS:
    VECLIB_MAXIMUM_THREADS=1 RLX_QWEN3_TTS_TIMING=1 cargo run -p rlx-models --example qwen3_tts_rtf_bench --features metal --release -- {{ARGS}}

test-qwen3-tts-cp-metal-repro *ARGS:
    RLX_QWEN3_TTS_PARITY=1 cargo test -p rlx-models --test qwen3_tts_cp_metal_upstream_repro --release --features metal -- {{ARGS}}

# Greedy CustomVoice vs committed HF golden frames (RLX_QWEN3_TTS_PARITY=1, no Python).
# Qwen3-TTS synthesis → Whisper-base.en ASR round-trip (intelligibility; needs weights + whisper-base.en).
test-qwen3-tts-whisper *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
    export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
    cargo test -p rlx-models --test qwen3_tts_whisper_roundtrip --features metal --release -- {{ARGS}}

test-qwen3-tts-parity *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
    export RLX_QWEN3_TTS_PARITY=1
    export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
    unset RLX_QWEN3_TTS_METAL_DECODE_NATIVE
    cargo test -p rlx-models --test qwen3_tts_hf_parity --features metal --release -- {{ARGS}}
    cargo test -p rlx-models --test qwen3_tts_speech_decode_golden --features metal --release -- {{ARGS}}
    cargo test -p rlx-models --test qwen3_tts_whisper_roundtrip --features metal --release -- {{ARGS}}

# Core Qwen3-TTS tests (RLX_QWEN3_TTS_DIR; layer tests need HF JSON under .cache/qwen3-tts/).
test-qwen3-tts *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
    export RLX_QWEN3_TTS_PARITY=1
    tests=(qwen3_tts_hf_parity qwen3_tts_embeds_diff qwen3_tts_one_decode qwen3_tts_talker_layers qwen3_tts_talker_eager_vs_compiled qwen3_tts_debug_first)
    for t in "${tests[@]}"; do
      cargo test -p rlx-models --test "$t" --release -- "$@"
    done
    cargo test -p rlx-models --test qwen3_tts_mlx_greedy_parity --features apple-silicon --release -- "$@"
    cargo test -p rlx-models --test qwen3_tts_speech_pt_mlx_parity --features apple-silicon --release -- "$@"

# Talker prefill/decode on real 0.6B weights per backend (RLX_QWEN3_TTS_DIR).
test-qwen3-tts-backends *ARGS:
    cargo test -p rlx-models --test qwen3_tts_backend_quick_check --features all-backends --release -- {{ARGS}}

# Voice-clone streaming PCM (lossless chunking of generate()) per backend.
test-qwen3-tts-streaming *ARGS:
    cargo test -p rlx-qwen3-tts --test streaming_pcm_parity --features all-backends --release -- {{ARGS}}

fetch-kittentts:
    cargo run -p rlx-kittentts --features hf-download --release -- --download

# Re-export Kitten ONNX → RLX bundle (graph.json + weights.safetensors).
# Set KITTEN_ONNX_PATH to override the default HF cache snapshot.
export-kitten-rlx-bundle:
    #!/usr/bin/env bash
    set -euo pipefail
    ONNX="${KITTEN_ONNX_PATH:-$HOME/.cache/huggingface/hub/models--KittenML--kitten-tts-mini-0.8/snapshots/c02725660cea441db4c383af69f1f26f5cd00947/kitten_tts_mini_v0_8.onnx}"
    OUT="crates/kitten_tts_mini_rlx/weights/rlx_bundle"
    test -f "$ONNX" || { echo "missing ONNX: $ONNX (set KITTEN_ONNX_PATH)" >&2; exit 1; }
    python3 scripts/export_kitten_rlx_bundle.py "$ONNX" "$OUT"

kittentts *ARGS:
    RLX_KITTENTTS_DIR=${RLX_KITTENTTS_DIR:-.cache/kittentts-mini-0.8} \
    just run-bin rlx-kittentts rlx-kittentts {{ARGS}}

# One-command demo after `just fetch-kittentts`
kittentts-demo:
    just kittentts --ipa "həˈloʊ" --voice Jasper --out-wav /tmp/kittentts_demo.wav

# Long IPA sentence (~5 s) — catches silent/empty WAV regressions
kittentts-long-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    IPA='ðɪs ɪz ə lɔŋɡɚ sɛntəns fɔɹ tɛstɪŋ ðə kɪtən tɛkst tə spitʃ sɪstəm ɪn ɹʌst'
    RLX_KITTENTTS_DIR="${RLX_KITTENTTS_DIR:-.cache/kittentts-mini-0.8}" \
    cargo run -p rlx-kittentts --bin rlx-kittentts --release -- \
      --ipa "$IPA" --voice Jasper --out-wav /tmp/kittentts_long_demo.wav

kittentts-voices:
    just kittentts --list-voices

# Plain English via espeak-ng (build with --features espeak)
kittentts-text-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    TEXT='This is a longer sentence for testing the kitten text to speech system.'
    RLX_KITTENTTS_DIR="${RLX_KITTENTTS_DIR:-.cache/kittentts-mini-0.8}" \
    cargo run -p rlx-kittentts --features espeak --bin rlx-kittentts --release -- \
      --text "$TEXT" --voice Jasper --out-wav /tmp/kittentts_text_demo.wav

test-kittentts-espeak *ARGS:
    cargo test -p rlx-kittentts --features "espeak,onnx" --release --test e2e_text -- {{ARGS}}

test-kittentts-whisper *ARGS:
    # Prefer whisper-base.en when present (override with RLX_WHISPER_DIR).
    cargo test -p rlx-kittentts --features "espeak,onnx" --release --test e2e_whisper_roundtrip -- {{ARGS}}

fetch-kittentts-whisper:
    just fetch-whisper-base

test-kittentts-backends *ARGS:
    cargo test -p rlx-kittentts --test backend_quick_check --features all-backends --release -- {{ARGS}}

test-kittentts-native *ARGS:
    cargo test -p rlx-kittentts --features native --release native_infer_smoke -- {{ARGS}}

test-kittentts-native-parity *ARGS:
    cargo test -p rlx-kittentts --features "native,onnx" --release --test native_onnx_parity -- {{ARGS}}

# Fetch (if needed) + unit tests + ONNX/native E2E synthesis
test-kittentts-e2e *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f .cache/kittentts-mini-0.8/config.json ]]; then
      just fetch-kittentts
    fi
    cargo test -p rlx-kittentts --features "hf-download" --release -- {{ARGS}}
    cargo test -p rlx-kittentts --features "hf-download,onnx" --release --test e2e_onnx -- {{ARGS}}
    cargo test -p rlx-kittentts --features "hf-download,onnx" --release --test e2e_long_sentence -- {{ARGS}}
    just kittentts-demo
    just kittentts-long-demo
    python3 -c "import struct,sys; p='/tmp/kittentts_long_demo.wav'; f=open(p,'rb'); f.read(44); d=f.read(); n=len(d)//2; peak=max(abs(struct.unpack('<h',d[i:i+2])[0]/32768) for i in range(0,len(d)-1,2)); assert n>=80000 and peak>=1e-3, f'long demo failed: {n} samples peak={peak:.2e}'; print(f'long demo ok: {n} samples peak={peak:.3f}')"
    just kittentts-text-demo
    just test-kittentts-espeak
    if [[ -f .cache/whisper-base.en/model.safetensors || -f .cache/whisper-tiny/model.safetensors ]]; then
      just test-kittentts-whisper
    fi
    if [[ -f crates/kitten_tts_mini_rlx/weights/rlx_bundle/graph.json ]]; then
      cargo test -p rlx-kittentts --features native --release native_infer_smoke -- {{ARGS}}
      cargo test -p rlx-kittentts --features native --release native_long_sentence_smoke -- {{ARGS}}
      cargo run -p rlx-kittentts --features native --release -- \
        --native --ipa "ðɪs ɪz ə lɔŋɡɚ sɛntəns fɔɹ tɛstɪŋ ðə kɪtən tɛkst tə spitʃ sɪstəm ɪn ɹʌst" \
        --voice Jasper --seq-len 256 --max-waveform-samples 200000 \
        --out-wav /tmp/kittentts_native_e2e.wav
    fi

fetch-voxtral-tts:
    cargo run -p rlx-models --example voxtral_tts_download --features hf-download --release

# Docker-only vLLM-Omni reference (parity export). Inference/tokenization are native Rust.
voxtral-tts-docker-tools-build:
    bash docker/voxtral-tts/run-tools.sh build

voxtral-tts-docker-ref-build:
    bash docker/voxtral-tts/run-ref.sh build

voxtral-tts-prepare-voices:
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    just run-bin rlx-voxtral-tts rlx-voxtral-tts -- --model-dir ${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} --convert-voices

voxtral-tts-tokenize *ARGS:
    # Native Tekken tokenization. Env: RLX_VOXTRAL_TTS_TEXT, RLX_VOXTRAL_TTS_VOICE, RLX_VOXTRAL_TTS_OUT
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    just run-bin rlx-voxtral-tts rlx-voxtral-tts -- \
      --model-dir ${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
      --text "${RLX_VOXTRAL_TTS_TEXT:-Hello}" \
      --voice "${RLX_VOXTRAL_TTS_VOICE:-neutral_female}" \
      --write-prompt-tokens "${RLX_VOXTRAL_TTS_OUT:-.cache/voxtral/tts/prompt_tokens.txt}" \
      --tokenize-only {{ARGS}}

voxtral-tts-train-encoder *ARGS:
    cargo run -p rlx-voxtral-tts-train --bin rlx-voxtral-tts-train --release -- encoder {{ARGS}}

voxtral-tts-bench-encoder *ARGS:
    # Per-backend forward/backward compile+run matrix (writes .cache/voxtral/bench-out/encoder-matrix.json)
    cargo run -p rlx-voxtral-tts-train --bin bench-encoder --features metal --release {{ARGS}}

voxtral-tts-train-encoder-50ep *ARGS:
    # 50-epoch JFK encoder train: full metrics report, checkpoint every 5 epochs.
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    bash -lc '\
      set -euo pipefail; \
      OUT=".cache/voxtral/train/encoder-50ep"; \
      WAV=".cache/voxtral/jfk/wavs"; \
      MAN=".cache/voxtral/jfk/manifest.json"; \
      EVAL="${EVAL_WAV:-{{jfk_ref_wav}}}"; \
      EPOCHS="${EPOCHS:-50}"; \
      STEPS="${STEPS_PER_EPOCH:-40}"; \
      CKPT_EPOCH="${CHECKPOINT_EVERY_EPOCH:-5}"; \
      EARLY_STOP="${EARLY_STOP_PATIENCE:-5}"; \
      LOW_VRAM=1 USE_DISCRIMINATOR=0 USE_ASR=0 EARLY_STOP_PATIENCE="$EARLY_STOP" \
      cargo run -p rlx-voxtral-tts-train --bin rlx-voxtral-tts-train --features metal --release -- encoder \
        --model-dir "$RLX_VOXTRAL_TTS_DIR" --wav-dir "$WAV" --manifest "$MAN" --out-dir "$OUT" \
        --eval-wav "$EVAL" --checkpoint-every-epoch "$CKPT_EPOCH" \
        --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --device metal {{ARGS}} \
    '

voxtral-tts-train-encoder-bench *ARGS:
    # Timed encoder train on JFK clips with epoch checkpoints + JSON report (ablation-friendly).
    # Env: EPOCHS STEPS_PER_EPOCH CHECKPOINT_EVERY_EPOCH EVAL_WAV
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    bash -lc '\
      set -euo pipefail; \
      OUT=".cache/voxtral/train/encoder-bench"; \
      WAV=".cache/voxtral/jfk/wavs"; \
      MAN=".cache/voxtral/jfk/manifest.json"; \
      EVAL="${EVAL_WAV:-{{jfk_ref_wav}}}"; \
      EPOCHS="${EPOCHS:-5}"; \
      STEPS="${STEPS_PER_EPOCH:-40}"; \
      CKPT_EPOCH="${CHECKPOINT_EVERY_EPOCH:-1}"; \
      EARLY_STOP="${EARLY_STOP_PATIENCE:-0}"; \
      LOW_VRAM=1 USE_DISCRIMINATOR=0 USE_ASR=0 EARLY_STOP_PATIENCE="$EARLY_STOP" \
      cargo run -p rlx-voxtral-tts-train --bin rlx-voxtral-tts-train --features metal --release -- encoder \
        --model-dir "$RLX_VOXTRAL_TTS_DIR" --wav-dir "$WAV" --manifest "$MAN" --out-dir "$OUT" \
        --eval-wav "$EVAL" --checkpoint-every-epoch "$CKPT_EPOCH" \
        --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --device metal {{ARGS}} \
    '

voxtral-tts-train-encoder-low-vram *ARGS:
    LOW_VRAM=1 USE_DISCRIMINATOR=0 cargo run -p rlx-voxtral-tts-train --release -- encoder {{ARGS}}

voxtral-tts-clone-retrain *ARGS:
    # Full clone-quality retrain: encoder (early stop, wd=0) → LoRA (rank 16) → inject → rig.
    # Env: RLX_VOXTRAL_TTS_DIR EVAL_WAV (default jfk_0020)
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    bash -lc '\
      set -euo pipefail; \
      ROOT=".cache/voxtral/train/clone-v2"; \
      WAV=".cache/voxtral/jfk/wavs"; \
      MAN=".cache/voxtral/jfk/manifest.json"; \
      EVAL="${EVAL_WAV:-{{jfk_ref_wav}}}"; \
      CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/rlx-voxtral-build}"; \
      export CARGO_TARGET_DIR; \
      if [ "${SKIP_ENCODER:-0}" != "1" ]; then \
        echo "=== Phase 1: encoder ==="; \
        LOW_VRAM=1 USE_DISCRIMINATOR=0 USE_ASR=0 WEIGHT_DECAY=0 \
        EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}" EVAL_WAV="$EVAL" \
        EPOCHS="${ENCODER_EPOCHS:-20}" STEPS_PER_EPOCH="${ENCODER_STEPS:-40}" \
        cargo run -p rlx-voxtral-tts-train --bin rlx-voxtral-tts-train --features metal --release -- encoder \
          --model-dir "$RLX_VOXTRAL_TTS_DIR" --wav-dir "$WAV" --manifest "$MAN" \
          --out-dir "$ROOT/encoder" --eval-wav "$EVAL" \
          --checkpoint-every-epoch 5 --epochs "${ENCODER_EPOCHS:-20}" --steps-per-epoch "${ENCODER_STEPS:-40}" --device metal {{ARGS}}; \
      else \
        echo "=== Phase 1: encoder (skipped, SKIP_ENCODER=1) ==="; \
      fi; \
      echo "=== Phase 2: LoRA (26 layers, rank 16, cached teacher, CPU backward) ==="; \
      PRODUCTION=1 LORA_RANK=16 LORA_ALPHA=32 PRECOMPUTE_DISTILL=1 GRAD_ACCUM="${GRAD_ACCUM:-2}" \
      EPOCHS="${LORA_EPOCHS:-8}" STEPS_PER_EPOCH="${LORA_STEPS:-80}" \
      cargo run -p rlx-voxtral-tts-train --bin rlx-voxtral-tts-train --features metal --release -- lora \
        --model-dir "$RLX_VOXTRAL_TTS_DIR" --reference-wav-dir "$WAV" --manifest "$MAN" \
        --encoder-weights "$ROOT/encoder/best_encoder.safetensors" \
        --out-dir "$ROOT/lora" --epochs "${LORA_EPOCHS:-8}" --device metal {{ARGS}}; \
      echo "=== Inject encoder + LoRA ==="; \
      cargo run -p rlx-voxtral-tts-train --bin rlx-voxtral-tts-train --release -- inject \
        --model-dir "$RLX_VOXTRAL_TTS_DIR" \
        --encoder-weights "$ROOT/encoder/best_encoder.safetensors" \
        --lora-weights "$ROOT/lora/lora_adapters.safetensors"; \
      echo "=== Rig: synthesize + mel similarity ==="; \
      RLX_VOXTRAL_TTS_DIR="$RLX_VOXTRAL_TTS_DIR" RLX_VOXTRAL_TTS_TRAIN_RIG=1 \
        RLX_VOXTRAL_TTS_REF_WAV="$EVAL" \
        cargo test -p rlx-voxtral-tts-train --test synthesize_rig --features metal --release -- --nocapture; \
      echo "done — weights in $ROOT, merged into $RLX_VOXTRAL_TTS_DIR/consolidated.safetensors"; \
    '

voxtral-tts-train-lora *ARGS:
    cargo run -p rlx-voxtral-tts-train --release -- lora {{ARGS}}

voxtral-tts-inject-encoder *ARGS:
    cargo run -p rlx-voxtral-tts-train --release -- inject {{ARGS}}

test-voxtral-tts-train:
    cargo test -p rlx-voxtral-tts-train --release

test-voxtral-tts-train-backends:
    cargo test -p rlx-voxtral-tts-train --test train_backends --features all-backends --release

voxtral-tts-train-encoder-gpu *ARGS:
    just features=all-backends voxtral-tts-train-encoder -- --device auto {{ARGS}}

voxtral-tts-train-lora-gpu *ARGS:
    just features=all-backends voxtral-tts-train-lora -- --device auto {{ARGS}}

test-voxtral-tts-train-gpu-step:
    RLX_VOXTRAL_TTS_TRAIN_GPU_STEP=1 just features=all-backends test-voxtral-tts-train-backends

test-voxtral-tts-train-rig:
    RLX_VOXTRAL_TTS_TRAIN_RIG=1 cargo test -p rlx-voxtral-tts-train --test clone_pipeline_e2e rig_real_model_when_env_set --release -- --nocapture

test-voxtral-tts-train-synthesize-rig:
    RLX_VOXTRAL_TTS_TRAIN_RIG=1 cargo test -p rlx-voxtral-tts-train --test synthesize_rig --release -- --nocapture

voxtral-tts-train-manifest *ARGS:
    cargo run -p rlx-voxtral-tts-train --release -- manifest {{ARGS}}

voxtral-tts-train-all *ARGS:
    cargo run -p rlx-voxtral-tts-train --release -- all {{ARGS}}

voxtral-tts-jfk-prep:
    # Download JFK inaugural address (public domain) and chop into 24kHz mono WAV clips.
    bash scripts/voxtral_prep_jfk.sh

voxtral-tts-jfk-pretrain *ARGS:
    # Pretrain the Voxtral codec encoder on the JFK clips.
    # Env: RLX_VOXTRAL_TTS_DIR (defaults to `.cache/voxtral/Voxtral-4B-TTS-2603`)
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    bash -lc '\
      set -euo pipefail; \
      WAV_DIR=".cache/voxtral/jfk/wavs"; \
      OUT_DIR=".cache/voxtral/train/encoder-jfk"; \
      MANIFEST=".cache/voxtral/jfk/manifest.json"; \
      cargo run -p rlx-voxtral-tts-train --release -- manifest --wav-dir "$WAV_DIR" --out "$MANIFEST" --sample-rate 24000; \
      LOW_VRAM=1 USE_DISCRIMINATOR=0 cargo run -p rlx-voxtral-tts-train --release -- encoder \
        --model-dir "$RLX_VOXTRAL_TTS_DIR" --wav-dir "$WAV_DIR" --manifest "$MANIFEST" --out-dir "$OUT_DIR" {{ARGS}} \
    '

voxtral-tts-train-production *ARGS:
    PRODUCTION=1 just voxtral-tts-train-all -- --device auto {{ARGS}}

voxtral-tts-export-codes *ARGS:
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    bash docker/voxtral-tts/run-ref.sh export-codes {{ARGS}}

test-voxtral-tts-parity:
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    RLX_VOXTRAL_TTS_PARITY=1 \
    cargo test -p rlx-models --test voxtral_tts_parity --release -- --nocapture

test-voxtral-tts-native-parity:
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    RLX_VOXTRAL_TTS_NATIVE_PARITY=1 \
    cargo test -p rlx-models --test voxtral_tts_native_parity --release -- --nocapture

test-voxtral-tts-codec:
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    cargo test -p rlx-models --test voxtral_tts_codec --release -- --nocapture

test-voxtral-tts-compiled-lm:
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    cargo test -p rlx-voxtral-tts --test compiled_lm --features metal --release -- --nocapture --test-threads=1 --skip metal_compiled --skip wgpu_compiled
    RLX_VOXTRAL_TTS_DIR=${RLX_VOXTRAL_TTS_DIR:-.cache/voxtral/Voxtral-4B-TTS-2603} \
    cargo test -p rlx-voxtral-tts --test compiled_lm_wgpu_compile --features gpu --release -- --nocapture wgpu_decode_hir_builds
    cargo test -p rlx-voxtral-tts --test compiled_lm_cpu_decode --release -- --nocapture

fetch-voxtral:
    cargo run -p rlx-models --example voxtral_download --features hf-download --release

test-voxtral-parity: fetch-voxtral
    RLX_VOXTRAL_DIR=.cache/voxtral/Voxtral-Mini-3B-2507 \
    RLX_VOXTRAL_PYTHON=.venv-voxtral/bin/python \
    cargo test -p rlx-models --test voxtral_hf_parity --release -- --nocapture --test-threads 1

ocr *ARGS:
    just run-bin rlx-ocr rlx-ocr {{ARGS}}

sam1 *ARGS:
    just run-bin rlx-sam rlx-sam1 {{ARGS}}

sam2 *ARGS:
    just run-bin rlx-sam2 rlx-sam2 {{ARGS}}

sam3 *ARGS:
    just run-bin rlx-sam3 rlx-sam3 {{ARGS}}

flux2 *ARGS:
    just run-bin rlx-flux2 rlx-flux2 {{ARGS}}

flux2-serve *ARGS:
    just run-bin rlx-flux2 rlx-flux2-serve {{ARGS}}

# --- multiplexer (same flags, slower build) ---

run *ARGS:
    just run-rlx {{ARGS}}

qwen3-metal *ARGS:
    just features=metal run-bin rlx-qwen3 rlx-qwen3 {{ARGS}}

qwen3-all-backends *ARGS:
    just features=all-backends run-bin rlx-qwen3 rlx-qwen3 {{ARGS}}

qwen35-metal *ARGS:
    just features=metal run-bin rlx-qwen35 rlx-qwen35 {{ARGS}}

qwen35-all-backends *ARGS:
    just features=all-backends run-bin rlx-qwen35 rlx-qwen35 {{ARGS}}

minicpm5-metal *ARGS:
    just features=metal run-minicpm5 {{ARGS}}

minicpm5-all-backends *ARGS:
    just features=all-backends run-minicpm5 {{ARGS}}

llama32-metal *ARGS:
    just features=metal run-rlx llama32 {{ARGS}}

dinov2-metal *ARGS:
    just features=metal run-rlx dinov2 {{ARGS}}

sam3-metal *ARGS:
    just features=metal run-rlx sam3 {{ARGS}}

flux2-metal *ARGS:
    just features=metal,flux2-image run-rlx flux2 {{ARGS}}

# Diamond Maps on FLUX (env: FLUX_GGUF_PATH or FLUX_MODEL_ROOT; optional FLUX_DIAMOND=1).
test-flux2-diamond *ARGS:
    cargo test -p rlx-models --test flux2_diamond_guidance {{profile}} {{ARGS}}

# --- examples (facade templates) ---

example NAME *ARGS:
    cargo run -p rlx-models --example {{NAME}} {{profile}} {{feature_args}} -- {{ARGS}}

example-qwen3-gguf *ARGS:
    just example run_qwen3_gguf {{ARGS}}

example-qwen35 *ARGS:
    just example run_qwen35 {{ARGS}}

example-sam3 *ARGS:
    just example run_sam3 {{ARGS}}

example-minicpm5 *ARGS:
    just fetch-minicpm5
    RLX_MINICPM5_WEIGHTS={{real_weights_dir}}/MiniCPM5-1B/model-00000-of-00001.safetensors \
        just example run_minicpm5 {{ARGS}}

# --- docker weight fetch ---

fetch-qwen3 REPO="Qwen/Qwen3-0.6B":
    mkdir -p weights
    docker build -t rlx-qwen3-fetch docker/qwen3-fetch
    docker run --rm -v "$PWD/weights:/weights" rlx-qwen3-fetch {{REPO}}

# Q4_K_M GGUF for low-memory / non-MLX LM paths (voice chat defaults to safetensors on MLX).
fetch-qwen3-gguf:
    mkdir -p weights/Qwen3-0.6B-gguf
    test -s weights/Qwen3-0.6B-gguf/Qwen3-0.6B-Q4_K_M.gguf || \
        curl -L --create-dirs -C - -o weights/Qwen3-0.6B-gguf/Qwen3-0.6B-Q4_K_M.gguf \
            'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf'

# --- real-weight integration tests (PLAN.md M0–M3 verification) ---
#
# Downloads tiny public GGUFs into /tmp/rlx-weights and runs the
# env-gated real-weight tests against them. Idempotent: skips
# downloads when files already exist. ~1.5 GB total disk after the
# first run.

real_weights_dir := "/tmp/rlx-weights"

# Download all real-weight test artifacts (~1.5 GB after Q4_K_M).
# Re-runs are no-ops thanks to `curl -C - --create-dirs`.
fetch-real-weights:
    mkdir -p {{real_weights_dir}}
    test -s {{real_weights_dir}}/SmolLM2-135M.gguf || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/SmolLM2-135M.gguf \
            'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf'
    test -s {{real_weights_dir}}/tokenizer.json || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/tokenizer.json \
            'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json'
    test -s {{real_weights_dir}}/Qwen2.5-0.5B.gguf || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/Qwen2.5-0.5B.gguf \
            'https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'
    test -s {{real_weights_dir}}/qwen2.5-tokenizer.json || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/qwen2.5-tokenizer.json \
            'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json'
    test -s {{real_weights_dir}}/gemma-3-270m.gguf || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/gemma-3-270m.gguf \
            'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf'
    test -s {{real_weights_dir}}/Llama-3.2-1B.gguf || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/Llama-3.2-1B.gguf \
            'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'
    test -s {{real_weights_dir}}/llama-3.2-tokenizer.json || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/llama-3.2-tokenizer.json \
            'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json'
    @echo "real weights ready in {{real_weights_dir}}"

# MiniCPM5-1B via Hugging Face Hub (~2.1 GB safetensors + tokenizer).
fetch-minicpm5:
    MINICPM5_MODEL_DIR={{real_weights_dir}}/MiniCPM5-1B \
        cargo run -p rlx-models --example minicpm5_download --features hf-download --release

test-minicpm5-real: fetch-minicpm5
    RLX_MINICPM5_WEIGHTS={{real_weights_dir}}/MiniCPM5-1B/model-00000-of-00001.safetensors \
    RLX_MINICPM5_CONFIG={{real_weights_dir}}/MiniCPM5-1B/config.json \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_minicpm5 -- --nocapture

test-minicpm5-parity-full: fetch-minicpm5
    RLX_MINICPM5_WEIGHTS={{real_weights_dir}}/MiniCPM5-1B/model-00000-of-00001.safetensors \
    RLX_MINICPM5_CONFIG={{real_weights_dir}}/MiniCPM5-1B/config.json \
        cargo test -p rlx-models --test minicpm5_parity --features parity-pytorch --release \
            minicpm5_pytorch_last_logits minicpm5_config_matches_hf_card -- --nocapture

test-minicpm5-all: fetch-minicpm5
    just test-minicpm5-real
    just test-minicpm5-parity-full
    just test-minicpm5-backends
    just bench-minicpm5

test-minicpm5-full: fetch-minicpm5 fetch-minicpm5-gguf-all
    just test-minicpm5-all
    just features=all-backends test-minicpm5-backends-all
    just test-minicpm5-gguf-quants

# Run the no-inference real-weight tests (config + compat + chat template).
# Fast (~2 s per suite). Inference tests are env-gated separately.
test-real-weights: fetch-real-weights
    RLX_SMOLLM2_GGUF={{real_weights_dir}}/SmolLM2-135M.gguf \
    RLX_QWEN25_GGUF={{real_weights_dir}}/Qwen2.5-0.5B.gguf \
    RLX_GEMMA3_GGUF={{real_weights_dir}}/gemma-3-270m.gguf \
    RLX_LLAMA32_GGUF={{real_weights_dir}}/Llama-3.2-1B.gguf \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_smollm2 \
            --test real_weights_qwen25 \
            --test real_weights_gemma3 \
            --test real_weights_llama32_1b \
            -- --nocapture

# Same suites + actual end-to-end forward inference (SmolLM2 + Llama 3.2).
# Slow (~30 s wall clock at 135M, ~minutes at 1B on CPU).
test-real-weights-inference: fetch-real-weights
    RLX_SMOLLM2_GGUF={{real_weights_dir}}/SmolLM2-135M.gguf \
    RLX_SMOLLM2_TOKENIZER={{real_weights_dir}}/tokenizer.json \
    RLX_SMOLLM2_RUN_INFERENCE=1 \
    RLX_QWEN25_GGUF={{real_weights_dir}}/Qwen2.5-0.5B.gguf \
    RLX_QWEN25_TOKENIZER={{real_weights_dir}}/qwen2.5-tokenizer.json \
    RLX_QWEN25_RUN_INFERENCE=1 \
    RLX_GEMMA3_GGUF={{real_weights_dir}}/gemma-3-270m.gguf \
    RLX_LLAMA32_GGUF={{real_weights_dir}}/Llama-3.2-1B.gguf \
    RLX_LLAMA32_TOKENIZER={{real_weights_dir}}/llama-3.2-tokenizer.json \
    RLX_LLAMA32_RUN_INFERENCE=1 \
        cargo test -p rlx-models {{profile}} --features qwen35-tokenizer \
            --test real_weights_smollm2 \
            --test real_weights_qwen25 \
            --test real_weights_gemma3 \
            --test real_weights_llama32_1b \
            -- --nocapture

# Live HuggingFace Hub compat-check (HTTPS): config.json fetch + GGUF range-GET.
test-net-hf:
    RLX_NET_TESTS=1 cargo test -p rlx-cli --features compat-net \
        --test hf_repo_check_live {{profile}} -- --nocapture
