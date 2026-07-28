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

# rlx-nanbeige binary requires the `tokenizer` cargo feature (not a CLI flag).
nanbeige_feature_args := if features != "" { "--features " + features + ",tokenizer" } else { "--features tokenizer" }

# rlx-tinyllama binary requires the `tokenizer` cargo feature (not a CLI flag).
tinyllama_feature_args := if features != "" { "--features " + features + ",tokenizer" } else { "--features tokenizer" }

# Run a per-crate binary (fast link). Pass CLI flags after `--`.
[private]
run-bin package bin *ARGS:
    cargo run -p {{package}} --bin {{bin}} {{profile}} {{feature_args}} -- {{ARGS}}

[private]
run-minicpm5 *ARGS:
    cargo run -p rlx-minicpm5 --bin rlx-minicpm5 {{profile}} {{minicpm5_feature_args}} -- {{ARGS}}

[private]
run-nanbeige *ARGS:
    cargo run -p rlx-nanbeige --bin rlx-nanbeige {{profile}} {{nanbeige_feature_args}} -- {{ARGS}}

[private]
run-tinyllama *ARGS:
    cargo run -p rlx-tinyllama --bin rlx-tinyllama {{profile}} {{tinyllama_feature_args}} -- {{ARGS}}

# Multiplexer (links all models). Subcommand is first arg after `--`.
[private]
run-rlx *ARGS:
    cargo run -p rlx-models --bin rlx-run {{profile}} {{feature_args}} -- {{ARGS}}

default:
    @just --list

# --- workspace ---

check:
    cargo check --workspace

# Auto-format the workspace.
fmt:
    cargo fmt --all

# Fail if formatting drifts (same bar as CI / publish).
fmt-check:
    cargo fmt --all -- --check

# Clippy with warnings as errors (same bar as CI / publish).
lint:
    ./scripts/rust-lint-gate.sh --workspace

# fmt-check + clippy (-D warnings).
lint-all: fmt-check lint

# Point this clone at committed hooks under `.githooks/` (local git config only).
install-hooks:
    git config core.hooksPath .githooks
    @echo "core.hooksPath=.githooks (pre-commit runs scripts/rust-lint-gate.sh --staged)"

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

# Synthetic tiny Qwen3.5 on each backend (cheap RAM). Prefer this for matrix coverage.
#   just features=all-backends,qwen35,qwen3 test-qwen35-backends
test-qwen35-backends *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    feats="${features:-all-backends}"
    case ",$feats," in
      *,qwen35,*) ;;
      *) feats="${feats},qwen35,qwen3" ;;
    esac
    echo "==> qwen35_backend_quick_check (features=$feats)"
    cargo test -p rlx-models --test qwen35_backend_quick_check --release --features "$feats" {{ARGS}}

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

# Scan LM Studio / Ollama / HF / Lemonade / RLX local caches for weight files.
# Extra args after `--`: e.g. `just weights-scan -- --query qwen --json`
weights-scan *ARGS:
    cargo run -p rlx-cli --bin rlx-inspect {{profile}} -- scan {{ARGS}}

# --- per-model CLIs (preferred) ---

qwen3 *ARGS:
    just run-bin rlx-qwen3 rlx-qwen3 {{ARGS}}

qwen35 *ARGS:
    just run-bin rlx-qwen35 rlx-qwen35 {{ARGS}}

# Microsoft Fara1.5 computer-use agent (Qwen3.5 multimodal safetensors).
fara *ARGS:
    just features={{features}} run-bin rlx-fara rlx-fara {{ARGS}}

fetch-fara-4b:
    #!/usr/bin/env bash
    set -euo pipefail
    local=".cache/fara/4b"
    mkdir -p "$local" .cache/fara
    if [ -f "$local/config.json" ] && ls "$local"/*.safetensors >/dev/null 2>&1; then
      echo ">> Fara1.5-4B already at $local"
      echo "$local" > .cache/fara/.rlx_fara_4b_snapshot
      exit 0
    fi
    if ! command -v hf >/dev/null 2>&1; then
      echo "error: need \`hf\` (pip install -U huggingface_hub[cli])" >&2
      exit 1
    fi
    echo ">> downloading microsoft/Fara1.5-4B → $local"
    hf download microsoft/Fara1.5-4B --local-dir "$local"
    echo "$local" > .cache/fara/.rlx_fara_4b_snapshot
    echo ">> ready: $local"

fetch-fara-9b:
    #!/usr/bin/env bash
    set -euo pipefail
    local=".cache/fara/9b"
    mkdir -p "$local" .cache/fara
    if [ -f "$local/config.json" ] && ls "$local"/*.safetensors >/dev/null 2>&1; then
      echo ">> Fara1.5-9B already at $local"
      echo "$local" > .cache/fara/.rlx_fara_9b_snapshot
      exit 0
    fi
    if ! command -v hf >/dev/null 2>&1; then
      echo "error: need \`hf\` (pip install -U huggingface_hub[cli])" >&2
      exit 1
    fi
    echo ">> downloading microsoft/Fara1.5-9B → $local"
    hf download microsoft/Fara1.5-9B --local-dir "$local"
    echo "$local" > .cache/fara/.rlx_fara_9b_snapshot
    echo ">> ready: $local"

fara-demo *ARGS:
    just fara --model-dir .cache/fara/4b --size 4b {{ARGS}}

# Fara-4B text ChatML probe (top-1 must be HF id 760 "The").
# RAM-heavy: default DEVICES=cpu. One GPU at a time, e.g. `DEVICES=metal just test-fara-backends`.
# Full backend matrix without Fara weights: `just test-qwen35-backends`.
# Needs `.cache/fara/4b` (`just fetch-fara-4b`). Skips unavailable backends.
test-fara-backends *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    model="${FARA_MODEL_DIR:-.cache/fara/4b}"
    if [ ! -f "$model/config.json" ]; then
      echo "error: missing $model (just fetch-fara-4b)" >&2
      exit 1
    fi
    IFS=',' read -r -a devices <<< "${DEVICES:-cpu}"
    expect_id="${FARA_EXPECT_TOP1:-760}"
    fail=0
    ran=0
    feats="${features:-apple-silicon}"
    echo "==> building fara_text_probe (features=$feats)"
    echo "    note: Fara-4B is RAM-heavy; run one GPU device per invocation"
    cargo build -p rlx-qwen35 --example fara_text_probe --release --features "$feats"
    bin="target/release/examples/fara_text_probe"
    for d in "${devices[@]}"; do
      d="$(echo "$d" | tr -d '[:space:]')"
      [ -n "$d" ] || continue
      echo ""
      echo "==> fara text probe --device $d"
      log="/tmp/fara_probe_${d}.log"
      /usr/bin/purge >/dev/null 2>&1 || true
      set +e
      "$bin" --model-dir "$model" --device "$d" {{ARGS}} >"$log" 2>&1
      rc=$?
      set -e
      if [ "$rc" -ne 0 ]; then
        if rg -q "backend not available|not available" "$log"; then
          echo "  skip $d (unavailable)"
          continue
        fi
        if rg -q "Killed|Cannot allocate|out of memory|OOM" "$log" || [ "$rc" -eq 137 ]; then
          echo "  FAIL $d (OOM — free RAM and retry alone: DEVICES=$d just test-fara-backends)"
          fail=1
          continue
        fi
        echo "  FAIL $d (probe exited $rc)"
        tail -40 "$log"
        fail=1
        continue
      fi
      ran=$((ran + 1))
      top1="$(rg -o 'id=[0-9]+' "$log" | head -1 | cut -d= -f2 || true)"
      echo "  top1 id=${top1:-?} (expect $expect_id)"
      rg -n "top5:|id=" "$log" | head -8
      if [ "$top1" != "$expect_id" ]; then
        echo "  FAIL $d: top1=$top1 want $expect_id"
        fail=1
      else
        echo "  ok $d"
      fi
    done
    echo ""
    echo "==> fara backends: ran=$ran fail=$fail"
    [ "$ran" -gt 0 ] || { echo "error: no backends ran" >&2; exit 1; }
    [ "$fail" -eq 0 ]

# prism-ml/Bonsai-27B (qwen35 arch, Q1_0 packed). Downloads to weights/Bonsai-27B-gguf/.
fetch-bonsai27b:
    #!/usr/bin/env bash
    set -euo pipefail
    dest="{{real_weights_dir}}/Bonsai-27B-gguf"
    mkdir -p "$dest"
    gguf="$dest/Bonsai-27B-Q1_0.gguf"
    if [ ! -s "$gguf" ]; then
      echo ">> downloading Bonsai-27B-Q1_0.gguf → $gguf"
      curl -L -C - --retry 5 -o "$gguf" \
        "https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-Q1_0.gguf"
    fi
    ls -lh "$gguf"

# prism-ml/Ternary-Bonsai-27B (qwen35 arch, Q2_0_g128 packed). ~7.2 GB.
fetch-ternary-bonsai27b:
    #!/usr/bin/env bash
    set -euo pipefail
    dest="{{real_weights_dir}}/Ternary-Bonsai-27B-gguf"
    mkdir -p "$dest"
    gguf="$dest/Ternary-Bonsai-27B-Q2_0.gguf"
    if [ ! -s "$gguf" ]; then
      echo ">> downloading Ternary-Bonsai-27B-Q2_0.gguf → $gguf"
      curl -L -C - --retry 5 -o "$gguf" \
        "https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_0.gguf"
    fi
    ls -lh "$gguf"

# Dispatch CLI: sniffs GGUF arch (qwen35 → Qwen35Runner). Always pass --packed for 27B.
bonsai *ARGS:
    cargo run -p rlx-models --bin rlx-run {{profile}} \
      {{ if features != "" { "--features " + features + ",bonsai,qwen35,llama32" } else { "--features bonsai,qwen35,llama32" } }} \
      -- bonsai {{ARGS}}

# Real-weight header/config check (env RLX_BONSAI27B_GGUF or default path).
test-bonsai27b *ARGS:
    RLX_BONSAI27B_GGUF="${RLX_BONSAI27B_GGUF:-{{real_weights_dir}}/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf}" \
      cargo test -p rlx-models --features "bonsai,qwen35" --test real_weights_bonsai27b {{profile}} {{ARGS}}

# Ternary Bonsai-27B header/config check (env RLX_TERNARY_BONSAI27B_GGUF or default).
test-ternary-bonsai27b *ARGS:
    RLX_TERNARY_BONSAI27B_GGUF="${RLX_TERNARY_BONSAI27B_GGUF:-{{real_weights_dir}}/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf}" \
      cargo test -p rlx-models --features "bonsai,qwen35" --test real_weights_ternary_bonsai27b {{profile}} {{ARGS}}

llama32 *ARGS:
    just run-bin rlx-llama32 rlx-llama32 {{ARGS}}

minicpm5 *ARGS:
    just run-minicpm5 {{ARGS}}

nanbeige *ARGS:
    just run-nanbeige {{ARGS}}

tinyllama *ARGS:
    just run-tinyllama {{ARGS}}

# Transformers-style one-liner: text in / text out, auto-download, chat template.
#   just tinyllama-pipeline -- --prompt "What is the capital of France?"
#   just features=metal tinyllama-pipeline -- --model /path/to/ckpt --device metal --prompt "Hi"
tinyllama-pipeline *ARGS:
    cargo run -p rlx-tinyllama --bin rlx-tinyllama-pipeline {{profile}} {{ if features != "" { "--features " + features + ",pipeline" } else { "--features pipeline" } }} -- {{ARGS}}

phi *ARGS:
    just run-bin rlx-phi rlx-phi {{ARGS}}

# thinkingmachines/Inkling — multimodal MoE scaffold (config / weight map / synth text forward).
inkling *ARGS:
    just run-bin rlx-inkling rlx-inkling {{ARGS}}

test-inkling *ARGS:
    cargo test -p rlx-inkling {{profile}} {{ARGS}}

# Eager text forward vs checked-in transformers tiny dump (no Hub download).
test-inkling-parity *ARGS:
    cargo test -p rlx-inkling --test hf_tiny_parity {{profile}} {{ARGS}}

# Header-only shape probe against the Hub (config+index + Range GETs; no shard payload).
inkling-probe-remote *ARGS:
    cargo run -p rlx-inkling --features hf-probe {{profile}} -- --probe-remote {{ARGS}}

# Unsloth GGUF header sniff (meta + first weight shard; Range only — no IQ payload).
inkling-probe-gguf *ARGS:
    cargo run -p rlx-inkling --features hf-probe {{profile}} -- --probe-gguf-remote {{ARGS}}

# Unsloth Laguna-S GGUF (nested UD-Q4_K_M shards) + Poolside tokenizer → .cache/laguna-s
# Override quant folder: `just fetch-laguna QUANT=UD-Q4_K_XL`
# RAM: S UD-Q4_K_M ≈73 GB packed — needs ≫64 GB host RAM (mmap + KV + OS).
# On ≤64 GB machines prefer `just fetch-laguna-xs` (~20 GB Q4_K_M).
fetch-laguna QUANT="UD-Q4_K_M":
    #!/usr/bin/env bash
    set -euo pipefail
    dest=".cache/laguna-s"
    mkdir -p "$dest"
    echo ">> note: Laguna-S {{QUANT}} is large (~50–75+ GB depending on quant); ≤64 GB hosts should use \`just fetch-laguna-xs\`"
    hf download unsloth/Laguna-S-2.1-GGUF --include "{{QUANT}}/*" --local-dir "$dest"
    hf download poolside/Laguna-S-2.1 \
        config.json tokenizer.json tokenizer_config.json \
        --local-dir "$dest"
    # Optional chat template sidecars (name varies by revision).
    hf download poolside/Laguna-S-2.1 --include "chat_template*" --include "*.jinja" \
        --local-dir "$dest" 2>/dev/null || true
    shard=$(find "$dest/{{QUANT}}" -name '*-00001-of-*.gguf' -type f 2>/dev/null | head -n1 || true)
    if [[ -z "$shard" ]]; then
        shard=$(find "$dest/{{QUANT}}" -name '*.gguf' -type f 2>/dev/null | sort | head -n1 || true)
    fi
    echo ">> Laguna-S ready under $dest"
    if [[ -n "$shard" ]]; then
        echo ">> first shard: $shard"
        echo ">> sniff:  just laguna -- --weights $dest --prefer Q4_K_M"
        echo ">> generate: just features=apple-silicon laguna -- --weights $dest --prefer Q4_K_M --packed-load --device metal --tokenizer-dir $dest --prompt \"Say hello\" --max-tokens 8"
    fi

fetch-laguna-s QUANT="UD-Q4_K_M":
    just fetch-laguna QUANT={{QUANT}}

# Optional XS single-file GGUF + tokenizer (lighter bring-up; docs historically used .cache/laguna-xs).
fetch-laguna-xs QUANT="Q4_K_M":
    #!/usr/bin/env bash
    set -euo pipefail
    dest=".cache/laguna-xs"
    mkdir -p "$dest"
    hf download poolside/Laguna-XS-2.1-GGUF --include "*{{QUANT}}*" --local-dir "$dest"
    hf download poolside/Laguna-XS-2.1 \
        config.json tokenizer.json tokenizer_config.json \
        --local-dir "$dest"
    hf download poolside/Laguna-XS-2.1 --include "chat_template*" --include "*.jinja" \
        --local-dir "$dest" 2>/dev/null || true
    gguf=$(find "$dest" -maxdepth 1 -name '*.gguf' -type f | sort | head -n1 || true)
    echo ">> Laguna-XS ready under $dest${gguf:+ ($gguf)}"

# poolside Laguna MoE — packed GGUF generate (KV cache); Metal: `just features=apple-silicon laguna -- … --device metal`
laguna *ARGS:
    just run-bin rlx-laguna rlx-laguna {{ARGS}}

test-laguna *ARGS:
    cargo test -p rlx-laguna {{profile}} {{ARGS}}

laguna-probe-gguf *ARGS:
    cargo run -p rlx-laguna --features hf-probe {{profile}} -- --probe-gguf-remote {{ARGS}}

# OpenAI-compatible HTTP (`--serve`); greedy decode. Prefer central multi-model:
#   just features=apple-silicon,laguna openai-serve -- \
#     --engine laguna --weights .cache/laguna-s --prefer Q4_K_M --tokenizer-dir .cache/laguna-s --device metal
# Example (single-model convenience):
#   just features=apple-silicon laguna-serve -- \
#     --weights .cache/laguna-s --prefer Q4_K_M \
#     --tokenizer-dir .cache/laguna-s --device metal --host 127.0.0.1 --port 8080
laguna-serve *ARGS:
    cargo run -p rlx-laguna --bin rlx-laguna {{profile}} {{feature_args}} --features serve -- --serve {{ARGS}}

# Central OpenAI server (multi-model RegistryBackend). Example:
#   just features=apple-silicon,laguna openai-serve -- \
#     --engine laguna --weights .cache/laguna-s --prefer Q4_K_M \
#     --tokenizer-dir .cache/laguna-s --device metal --model-id laguna
openai-serve *ARGS:
    cargo run -p rlx-openai --bin rlx-openai {{profile}} {{feature_args}} -- {{ARGS}}

# Packed DequantMatMul backend speed + parity (CPU / Metal / MLX / …).
laguna-backend-bench *ARGS:
    cargo run -p rlx-laguna --example backend_bench --features apple-silicon {{profile}} -- {{ARGS}}

minicpm5-chat MESSAGE *ARGS:
    RLX_MODELS_ROOT={{justfile_directory()}} python3 crates/rlx-models/examples/minicpm5_chat.py "{{MESSAGE}}" {{ARGS}}

# Build release binary once, then chat (avoids cargo startup each message).
minicpm5-chat-fast MESSAGE *ARGS:
    cargo build -p rlx-minicpm5 --features tokenizer,mlx,metal --release
    just minicpm5-chat "{{MESSAGE}}" {{ARGS}}

test-minicpm5-backends *ARGS:
    cargo test -p rlx-models --test minicpm5_backend_parity {{profile}} {{feature_args}} {{ARGS}}

test-nanbeige-backends *ARGS:
    cargo test -p rlx-models --test nanbeige_backend_parity --features nanbeige,llama32 {{profile}} {{feature_args}} {{ARGS}}

test-nanbeige-backends-all *ARGS:
    just features=all-backends test-nanbeige-backends {{ARGS}}

nanbeige-backend-matrix *ARGS:
    cargo run -p rlx-nanbeige --example backend_matrix --features all-backends --release -- {{ARGS}}

# Synth (default) or real weights: `just bench-nanbeige-backends -- --weights /tmp/rlx-weights/Nanbeige4.2-3B`
bench-nanbeige-backends *ARGS:
    cargo run -p rlx-nanbeige --example backend_bench {{profile}} {{feature_args}} -- {{ARGS}}

bench-nanbeige-backends-all *ARGS:
    just features=all-backends bench-nanbeige-backends {{ARGS}}

test-tinyllama-backends *ARGS:
    cargo test -p rlx-models --test tinyllama_backend_parity --features tinyllama,llama32 {{profile}} {{feature_args}} {{ARGS}}

test-tinyllama-backends-all *ARGS:
    just features=all-backends test-tinyllama-backends {{ARGS}}

test-tinyllama-gguf-backends *ARGS:
    RLX_TINYLLAMA_GGUF_DIR={{real_weights_dir}}/TinyLlama-1.1B-GGUF \
        cargo test -p rlx-models --test tinyllama_backend_gguf_check --features all-backends,tinyllama,llama32 {{profile}} {{feature_args}} {{ARGS}}

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

# Gemma 4 12B Coder GGUF (Composer 2.5 × Fable 5; ~7 GB Q4_K_M packed).
fetch-gemma4-12b-coder-gguf:
    mkdir -p .cache/gemma4-12b-coder
    hf download yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF gemma4-coding-Q4_K_M.gguf \
        --local-dir .cache/gemma4-12b-coder
    test -f .cache/gemma4-12b-coder/tokenizer.json || \
        hf download yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1 \
            config.json tokenizer.json tokenizer_config.json chat_template.jinja \
            --local-dir .cache/gemma4-12b-coder

# All coder quants (env RLX_GEMMA4_CODER_QUANTS=Q2_K,Q3_K_M,...).
fetch-gemma4-12b-coder-quants:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .cache/gemma4-12b-coder
    quants="${RLX_GEMMA4_CODER_QUANTS:-Q2_K,Q3_K_M,Q4_K_M,Q6_K,Q8_0}"
    IFS=',' read -ra qs <<< "$quants"
    for q in "${qs[@]}"; do
        q="$(echo "$q" | tr -d ' ')"
        hf download yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF "gemma4-coding-${q}.gguf" \
            --local-dir .cache/gemma4-12b-coder
    done
    test -f .cache/gemma4-12b-coder/tokenizer.json || \
        hf download yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1 \
            config.json tokenizer.json tokenizer_config.json chat_template.jinja \
            --local-dir .cache/gemma4-12b-coder

# Packed Q4_K_M coding demo (thinking chat template; temp 1.0 / top-p 0.95 per model card).
gemma4-coder-demo *ARGS:
    just gemma -- --weights .cache/gemma4-12b-coder/gemma4-coding-Q4_K_M.gguf \
        --device auto --packed --max-seq 2048 --max-tokens 128 \
        --temperature 1.0 --top-p 0.95 \
        --prompt "Write a Python function that checks whether a string is a palindrome." {{ARGS}}

test-gemma4-coder *ARGS:
    RLX_GEMMA4_CODER_FIXTURE={{env_var_or_default('RLX_GEMMA4_CODER_FIXTURE', justfile_directory() + '/.cache/gemma4-12b-coder')}} \
    cargo test -p rlx-gemma --release --features apple-silicon \
        --test gemma4_coder_gguf -- --nocapture --test-threads=1 {{ARGS}}

# Recursive llama.cpp checkout (submodules) for local parity / vendor builds.
fetch-llama-cpp:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{justfile_directory()}}/.cache/llama.cpp"
    if [[ -d "$dir/.git" ]]; then
        git -C "$dir" pull --ff-only
        git -C "$dir" submodule update --init --recursive
    else
        git clone --recursive --depth 1 https://github.com/ggml-org/llama.cpp.git "$dir"
    fi
    echo "llama.cpp ready at $dir"

# Gemma 4 12B Coder logits parity vs llama.cpp — run steps separately (lower peak RAM).
gemma4-coder-parity-llama *ARGS:
    just fetch-llama-cpp
    cargo run -p rlx-gemma --release --features "tokenizer parity-llama" \
        --example parity_check -- \
        {{justfile_directory()}}/.cache/gemma4-12b-coder/gemma4-coding-Q4_K_M.gguf llama {{ARGS}}

gemma4-coder-parity-rlx *ARGS:
    cargo run -p rlx-gemma --release --features "tokenizer apple-silicon" \
        --example parity_check -- \
        {{justfile_directory()}}/.cache/gemma4-12b-coder/gemma4-coding-Q4_K_M.gguf rlx {{ARGS}}

gemma4-coder-parity *ARGS:
    just gemma4-coder-parity-llama {{ARGS}}
    just gemma4-coder-parity-rlx {{ARGS}}

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

hoct *ARGS:
    just run-bin rlx-hoct rlx-hoct {{ARGS}}

fetch-hoct:
    #!/usr/bin/env bash
    set -euo pipefail
    CACHE="${HOCT_CACHE_DIR:-.cache/hoct}"
    mkdir -p "$CACHE"
    PT="$CACHE/general_v0.pt"
    ST="$CACHE/general_v0.safetensors"
    if [[ ! -s "$PT" ]]; then
      curl -fsSL -o "$PT" \
        "https://github.com/royerlab/hoct/releases/download/weights-v0/general_v0.pt"
    fi
    if [[ ! -s "$ST" ]]; then
      python3 crates/rlx-hoct/scripts/export_jit_safetensors.py "$PT" -o "$ST"
    fi
    echo "HOCT weights: $ST"

test-hoct-parity:
    #!/usr/bin/env bash
    set -euo pipefail
    just fetch-hoct
    export HOCT_WEIGHTS="${HOCT_CACHE_DIR:-.cache/hoct}/general_v0.safetensors"
    # Refresh JIT reference dumps when Torch is available (optional).
    if [[ -f "${HOCT_CACHE_DIR:-.cache/hoct}/general_v0.pt" ]] && python3 -c "import torch" 2>/dev/null; then
      python3 crates/rlx-hoct/scripts/dump_jit_reference.py \
        --pt "${HOCT_CACHE_DIR:-.cache/hoct}/general_v0.pt" \
        --out-prefix /tmp/hoct_ref_logits || true
    fi
    python3 crates/rlx-hoct/scripts/dump_pipeline_fixtures.py || true
    cargo test -p rlx-hoct --release -- --nocapture

test-hoct-backends *ARGS:
    cargo test -p rlx-hoct --test backend_parity --features all-backends --release -- --nocapture {{ARGS}}

vjepa2 *ARGS:
    just run-bin rlx-vjepa2 rlx-vjepa2 {{ARGS}}

wav2vec2 *ARGS:
    just run-bin rlx-wav2vec2-bert rlx-wav2vec2-bert {{ARGS}}

whisper *ARGS:
    just run-bin rlx-whisper rlx-whisper {{ARGS}}

# NVIDIA Conformer-CTC small (EncDecCTC / .nemo) — https://huggingface.co/nvidia/stt_en_conformer_ctc_small
fetch-conformer-ctc:
    mkdir -p .cache/conformer-ctc
    test -s .cache/conformer-ctc/stt_en_conformer_ctc_small.nemo || \
        hf download nvidia/stt_en_conformer_ctc_small --local-dir .cache/conformer-ctc

conformer-ctc *ARGS:
    just run-bin rlx-conformer-ctc rlx-conformer-ctc {{ARGS}}

test-conformer-ctc *ARGS:
    cargo test -p rlx-conformer-ctc --release {{ARGS}}

# Cross-backend Conformer-CTC transcription (skips unavailable devices).
test-conformer-ctc-backends *ARGS:
    cargo run -p rlx-conformer-ctc --release --example backend_matrix --features all-backends {{ARGS}}

# CUDA on the remote host — set RLX_CUDA_HOST (sync trees + nemo/wav, then cpu+cuda matrix).
conformer-ctc-cuda-remote:
    bash scripts/conformer_ctc_cuda_validate.sh --remote

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

fetch-tts-validation-bundles:
    # Lightweight placeholders for crates whose full weights are not yet fetched.
    # Real Parler / MeloTTS / OpenVoice / ChatterBox / MetaVoice live under weights/tts/
    # (gitignored). Prefer the per-model `just fetch-*` / Hugging Face downloads.
    mkdir -p weights/tts/melotts weights/tts/openvoice weights/tts/parlertts weights/tts/metavoice weights/tts/chatterbox
    for d in weights/tts/melotts weights/tts/openvoice weights/tts/parlertts weights/tts/metavoice weights/tts/chatterbox; do \
        if [ ! -f "$d/manifest.json" ]; then \
            printf '{"model_id":"placeholder","note":"use real HF weights under weights/tts"}\n' > "$d/manifest.json"; \
        fi; \
    done

# JFK clip + paired reference transcript for whisper bench / backend parity.
fetch-whisper-bench:
    bash scripts/fetch_whisper_bench.sh

test-whisper-jfk *ARGS:
    just fetch-whisper fetch-whisper-bench
    cargo test -p rlx-models --test whisper_jfk_e2e --release {{ARGS}}

test-whisper-parity *ARGS:
    cargo test -p rlx-models --test whisper_parity --features parity-candle whisper_synthetic --release {{ARGS}}

test-whisper-backend-parity *ARGS:
    just fetch-whisper fetch-whisper-bench
    cargo test -p rlx-models --test whisper_backend_parity --features "metal,mlx,gpu" --release {{ARGS}}

test-whisper-all-backends *ARGS:
    just fetch-whisper fetch-whisper-bench
    cargo test -p rlx-models --test whisper_all_backends_e2e --features "metal,mlx,gpu" --release {{ARGS}}

test-whisper-wgpu-gpu-kv *ARGS:
    cargo test -p rlx-models --test whisper_wgpu_gpu_kv --features gpu --release {{ARGS}}

test-whisper-timestamps *ARGS:
    cargo test -p rlx-models --test whisper_segment_timestamps --release {{ARGS}}
    cargo test -p rlx-models --test whisper_word_dtw --release {{ARGS}}
    cargo test -p rlx-whisper --features timestamps --release {{ARGS}}
    cargo test -p rlx-wav2vec2-asr --release {{ARGS}}
    cargo test -p rlx-diarize --release {{ARGS}}

whisper-subtitles *ARGS:
    just fetch-whisper fetch-whisper-bench
    cargo run -p rlx-whisper --features "timestamps,word-dtw,silero-vad" --release -- \
        --weights .cache/whisper-tiny/model.safetensors \
        --config .cache/whisper-tiny/config.json \
        --tokenizer .cache/whisper-tiny/tokenizer.json \
        --wav .cache/whisper-bench/jfk_16k.wav \
        --lang en --timestamps --word-align dtw --silero-vad \
        --output-format srt {{ARGS}}

bench-whisper *ARGS:
    cargo run -p rlx-models --example whisper_bench --features "metal,mlx,apple-silicon" --release -- {{ARGS}}

bench-whisper-precision *ARGS:
    just bench-whisper --precision --all-backends {{ARGS}}

bench-whisper-all-backends *ARGS:
    just bench-whisper --all-backends {{ARGS}}

bench-whisper-subtitles *ARGS:
    just fetch-whisper fetch-whisper-bench
    cargo run -p rlx-models --example whisper_subtitles_bench --features "whisper-subtitles,metal,apple-silicon" --release -- {{ARGS}}

bench-whisper-subtitles-all-backends *ARGS:
    just fetch-whisper fetch-whisper-bench
    cargo run -p rlx-models --example whisper_subtitles_bench --features "whisper-subtitles,all-backends" --release -- --all-backends {{ARGS}}

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

# --- Wake-word (openWakeWord / nanowakeword / porcupine / voxrt) ---

fetch-openwakeword:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .cache/openwakeword/onnx crates/rlx-openwakeword/weights
    python3 scripts/wake_export/export_openwakeword_weights.py \
      --onnx-dir .cache/openwakeword/onnx \
      --out-dir crates/rlx-openwakeword/weights

fetch-nanowakeword:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .cache/nanowakeword crates/rlx-nanowakeword/weights
    python3 scripts/wake_export/export_nanowakeword_weights.py \
      --out crates/rlx-nanowakeword/weights/model_lite.safetensors

openwakeword-demo *ARGS:
    just run-bin rlx-openwakeword rlx-openwakeword {{ARGS}}

nanowakeword-demo *ARGS:
    just run-bin rlx-nanowakeword rlx-nanowakeword {{ARGS}}

porcupine-demo *ARGS:
    just run-bin rlx-porcupine rlx-porcupine {{ARGS}}

voxrt-demo *ARGS:
    just run-bin rlx-voxrt rlx-voxrt {{ARGS}}

# Sweep every available RLX backend for all four engines (stub weights).
#   just features=all-backends wake-all-backends -- --wav clip.wav
wake-all-backends *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    args=({{ARGS}})
    if [[ ${#args[@]} -gt 0 && "${args[0]}" == "--" ]]; then
      args=("${args[@]:1}")
    fi
    cargo run -p rlx-openwakeword --bin rlx-openwakeword {{profile}} {{feature_args}} -- --device all "${args[@]}"
    cargo run -p rlx-nanowakeword --bin rlx-nanowakeword {{profile}} {{feature_args}} -- --device all "${args[@]}"
    cargo run -p rlx-porcupine --bin rlx-porcupine {{profile}} {{feature_args}} -- --device all "${args[@]}"
    cargo run -p rlx-voxrt --bin rlx-voxrt {{profile}} {{feature_args}} -- --device all "${args[@]}"

test-wake *ARGS:
    cargo test -p rlx-wake --release {{ARGS}}
    cargo test -p rlx-openwakeword --release {{ARGS}}
    cargo test -p rlx-nanowakeword --release {{ARGS}}
    cargo test -p rlx-porcupine --release {{ARGS}}
    cargo test -p rlx-voxrt --release {{ARGS}}

test-wake-backends *ARGS:
    cargo test -p rlx-wake --test backend_quick_check --features all-backends --release {{ARGS}}
    cargo test -p rlx-wake --test train_backends --features all-backends --release {{ARGS}}
    cargo test -p rlx-openwakeword --test backend_quick_check --features all-backends --release {{ARGS}}
    cargo test -p rlx-nanowakeword --test backend_quick_check --features all-backends --release {{ARGS}}
    cargo test -p rlx-porcupine --test backend_quick_check --features all-backends --release {{ARGS}}
    cargo test -p rlx-voxrt --test backend_quick_check --features all-backends --release {{ARGS}}
    just test-wake-parity {{ARGS}}

openwakeword-onnx-parity *ARGS:
    cargo test -p rlx-openwakeword --features onnx --test onnx_parity --release {{ARGS}}

nanowakeword-onnx-parity *ARGS:
    cargo test -p rlx-nanowakeword --features onnx --test onnx_parity --release {{ARGS}}

bench-wake *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo run -p rlx-wake --example wake_bench --release --features all-backends -- {{ARGS}}
    cargo run -p rlx-openwakeword --example engines_bench --release --features all-backends -- {{ARGS}}
    cargo run -p rlx-wakeword --example compare_bench --release --features all-backends -- {{ARGS}}

test-wake-parity *ARGS:
    cargo test -p rlx-wake --test backend_parity --features all-backends --release -- --nocapture {{ARGS}}
    cargo test -p rlx-openwakeword --test backend_parity --features all-backends --release -- --nocapture {{ARGS}}
    cargo test -p rlx-nanowakeword --test backend_parity --features all-backends --release -- --nocapture {{ARGS}}
    cargo test -p rlx-porcupine --test backend_parity --features all-backends --release -- --nocapture {{ARGS}}
    cargo test -p rlx-voxrt --test backend_parity --features all-backends --release -- --nocapture {{ARGS}}

# Train custom wake words in RLX only (no PyTorch / upstream trainers).
# Pass backends via features=, e.g. `just features=all-backends wake-train-cnn -- --device all …`
wake-train-synth *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    args=({{ARGS}})
    if [[ ${#args[@]} -gt 0 && "${args[0]}" == "--" ]]; then args=("${args[@]:1}"); fi
    cargo run -p rlx-wake --bin rlx-wake-train {{profile}} {{feature_args}} -- synth "${args[@]}"

wake-train-cnn *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    args=({{ARGS}})
    if [[ ${#args[@]} -gt 0 && "${args[0]}" == "--" ]]; then args=("${args[@]:1}"); fi
    cargo run -p rlx-wake --bin rlx-wake-train {{profile}} {{feature_args}} -- cnn "${args[@]}"

wake-train-mlp *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    args=({{ARGS}})
    if [[ ${#args[@]} -gt 0 && "${args[0]}" == "--" ]]; then args=("${args[@]:1}"); fi
    cargo run -p rlx-wake --bin rlx-wake-train {{profile}} {{feature_args}} -- mlp "${args[@]}"

wake-train-phrase *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    args=({{ARGS}})
    if [[ ${#args[@]} -gt 0 && "${args[0]}" == "--" ]]; then args=("${args[@]:1}"); fi
    cargo run -p rlx-openwakeword --bin rlx-openwakeword-train {{profile}} {{feature_args}} -- "${args[@]}"

test-wake-train *ARGS:
    cargo test -p rlx-wake --test train_quick --release {{ARGS}}
    cargo test -p rlx-wake --test train_backends --features all-backends --release {{ARGS}}

# CUDA on the remote host — set RLX_CUDA_HOST (sync trees, then cpu+cuda wake matrix).
wake-cuda-remote:
    bash scripts/wake_cuda_validate.sh --remote

# --- First-party wakeword product (event API, multi-phrase, VAD gate) ---

wakeword-demo *ARGS:
    just run-bin rlx-wakeword rlx-wakeword {{ARGS}}

wakeword-train *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    args=({{ARGS}})
    if [[ ${#args[@]} -gt 0 && "${args[0]}" == "--" ]]; then args=("${args[@]:1}"); fi
    cargo run -p rlx-wakeword --bin rlx-wakeword-train {{profile}} {{feature_args}} -- "${args[@]}"

# Scale bench: N=2..10 phrase heads (latency / size / RAM table)
wakeword-multi-bench *ARGS:
    cargo run -p rlx-wakeword --example multi_phrase_bench --release -- {{ARGS}}

# WASM (wasm-bindgen → Node): smoke + multi-phrase f32/ternary table
wakeword-wasm-bench *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    wasm-pack build crates/rlx-wakeword-wasm --target nodejs --release --out-dir pkg
    node crates/rlx-wakeword-wasm/run_bench.mjs {{ARGS}}

# Browser / Web Worker package (ES module, no window/DOM required)
wakeword-wasm-web:
    wasm-pack build crates/rlx-wakeword-wasm --target web --release --out-dir web/pkg-web

# Serve worker demo (build web target first if needed)
wakeword-wasm-worker-serve *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f crates/rlx-wakeword-wasm/web/pkg-web/rlx_wakeword_wasm.js ]]; then
      just wakeword-wasm-web
    fi
    echo "open http://127.0.0.1:8765/"
    python3 -m http.server 8765 --directory crates/rlx-wakeword-wasm/web {{ARGS}}

# Node worker_threads smoke (same protocol as browser module worker)
wakeword-wasm-worker-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f crates/rlx-wakeword-wasm/web/pkg-web/rlx_wakeword_wasm.js ]]; then
      just wakeword-wasm-web
    fi
    node crates/rlx-wakeword-wasm/run_worker_smoke.mjs

test-wakeword *ARGS:
    cargo test -p rlx-wakeword-core --release {{ARGS}}
    cargo test -p rlx-wakeword --release {{ARGS}}

test-wakeword-backends *ARGS:
    cargo test -p rlx-wakeword-core --test parity_wake --release {{ARGS}}
    cargo test -p rlx-wakeword --test backend_quick_check --features all-backends --release {{ARGS}}
    cargo test -p rlx-wakeword --test session_quick --release {{ARGS}}

# AEC (16 kHz FDAF-NLMS + residual)
test-aec *ARGS:
    cargo test -p rlx-aec --release {{ARGS}}

bench-aec *ARGS:
    cargo run -p rlx-aec --example echo_bench --release -- {{ARGS}}

bench-aec-parity *ARGS:
    cargo run -p rlx-aec --example echo_bench --release -- --json-out /tmp/aec_rust.json {{ARGS}}
    python3 scripts/aec_bench_speex.py --out /tmp/aec_python.json
    python3 scripts/aec_bench_compare.py --rust-json /tmp/aec_rust.json --python-json /tmp/aec_python.json --csv-out /tmp/aec_compare.csv

aec *ARGS:
    just run-bin rlx-aec rlx-aec {{ARGS}}

voxtral *ARGS:
    just run-bin rlx-voxtral rlx-voxtral {{ARGS}}

locateanything *ARGS:
    just run-bin rlx-locateanything rlx-locateanything {{ARGS}}

unlimited-ocr *ARGS:
    just run-bin rlx-unlimited-ocr rlx-unlimited-ocr {{ARGS}}

unlimited-ocr-metal *ARGS:
    just features=metal run-bin rlx-unlimited-ocr rlx-unlimited-ocr {{ARGS}}

fetch-unlimited-ocr:
    cargo run -p rlx-unlimited-ocr --features hf-download --release -- --download

test-unlimited-ocr-backends *ARGS:
    cargo test -p rlx-unlimited-ocr --test backend_quick_check {{profile}} {{feature_args}} {{ARGS}}
    cargo test -p rlx-unlimited-ocr --test backend_token_parity {{profile}} {{feature_args}} -- --test-threads 1 {{ARGS}}

# Full-checkpoint greedy token IDs vs CPU (needs ~tens of GB RAM + weights).
test-unlimited-ocr-token-parity *ARGS:
    RLX_UNLIMITED_OCR_TOKEN_PARITY=1 cargo test -p rlx-unlimited-ocr --test backend_token_parity --release {{feature_args}} -- --test-threads 1 {{ARGS}}

test-unlimited-ocr-parity *ARGS:
    # Uses HF cache / RLX_UNLIMITED_OCR_DIR via default_model_dir(); optional
    # RLX_UNLIMITED_OCR_PYTHON for exact HF e2e (needs transformers≈4.46 + addict/einops/…).
    cargo test -p rlx-unlimited-ocr --test hf_parity --release -- --test-threads 1 {{ARGS}}

# Pack + compile + prefill/decode across lm-precision modes (F32/F16/Q8/Q4).
# Tiny synthetic by default; `-- --full` uses the HF checkpoint LM only.
bench-unlimited-ocr-lm-precision *ARGS:
    cargo run -p rlx-unlimited-ocr --example bench_lm_precision --release {{feature_args}} -- {{ARGS}}

locateanything-metal *ARGS:
    just features=metal run-bin rlx-locateanything rlx-locateanything {{ARGS}}

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

# ---- Qwen2.5-VL (AIF / VLMEvalKit target) ----

qwen25_vl_hf_dir := env_var_or_default("RLX_QWEN25_VL_HF_DIR", ".cache/qwen25-vl/Qwen2.5-VL-7B-Instruct")
qwen25_vl_gguf_dir := env_var_or_default("RLX_QWEN25_VL_GGUF_DIR", ".cache/qwen25-vl/gguf")
qwen25_vl_sample := env_var_or_default("RLX_QWEN25_VL_IMAGE", "crates/rlx-locateanything/fixtures/sample.jpg")

# LM + mmproj GGUF for native RLX (~4.7 GB + ~1.4 GB mmproj at Q4_K_M).
fetch-qwen25-vl-gguf QUANT="Q4_K_M":
    mkdir -p {{qwen25_vl_gguf_dir}}
    hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
        --include "Qwen2.5-VL-7B-Instruct-{{QUANT}}.gguf" \
        --local-dir {{qwen25_vl_gguf_dir}}
    hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
        --include "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" \
        --local-dir {{qwen25_vl_gguf_dir}}
    @echo "GGUF ready: {{qwen25_vl_gguf_dir}}"

# HuggingFace safetensors checkpoint for Python reference dumps (~15 GB).
fetch-qwen25-vl-hf:
    hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir {{qwen25_vl_hf_dir}}
    @echo "HF checkpoint ready: {{qwen25_vl_hf_dir}}"

# GGUF + HF checkpoint (parity / AIF eval).
fetch-qwen25-vl: fetch-qwen25-vl-gguf fetch-qwen25-vl-hf

# Python venv for local HF reference dumps (alternative to Docker).
qwen25-vl-ref-venv:
    #!/usr/bin/env bash
    set -euo pipefail
    PY="${QWEN25_VL_PY:-python3}"
    "$PY" -m venv .venv-qwen25-vl
    .venv-qwen25-vl/bin/pip install --quiet --upgrade pip
    .venv-qwen25-vl/bin/pip install --quiet \
        "torch>=2.4" "torchvision>=0.19" "transformers>=4.49" "accelerate" \
        "numpy>=1.26,<2" "safetensors>=0.4" "huggingface_hub[cli]" \
        "pillow" "qwen-vl-utils"

# Download weights + venv, then run native AIF dynamics parity vs HF.
test-qwen25-vl-aif-native-parity-full: fetch-qwen25-vl qwen25-vl-ref-venv
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{justfile_directory()}}"
    export RLX_QWEN25_VL_HF_DIR="$root/{{qwen25_vl_hf_dir}}"
    export RLX_QWEN25_VL_PYTHON="$root/.venv-qwen25-vl/bin/python"
    export RLX_QWEN25_VL_GGUF_PATH="$root/{{qwen25_vl_gguf_dir}}/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    export RLX_QWEN25_VL_MMPROJ_PATH="$root/{{qwen25_vl_gguf_dir}}/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
    export RLX_QWEN25_VL_IMAGE="$root/{{qwen25_vl_sample}}"
    cd "$root"
    just test-qwen25-vl-aif-native-parity

test-qwen25-vl-aif-decode-step-parity-full: fetch-qwen25-vl qwen25-vl-ref-venv
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{justfile_directory()}}"
    export RLX_QWEN25_VL_HF_DIR="$root/{{qwen25_vl_hf_dir}}"
    export RLX_QWEN25_VL_PYTHON="$root/.venv-qwen25-vl/bin/python"
    export RLX_QWEN25_VL_GGUF_PATH="$root/{{qwen25_vl_gguf_dir}}/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    export RLX_QWEN25_VL_MMPROJ_PATH="$root/{{qwen25_vl_gguf_dir}}/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
    export RLX_QWEN25_VL_IMAGE="$root/{{qwen25_vl_sample}}"
    cd "$root"
    just test-qwen25-vl-aif-decode-step-parity

# Build HuggingFace reference Docker image for parity tests.
build-qwen25-vl-ref:
    bash crates/rlx-models/tests/qwen25_vl_parity_helpers/build.sh

# Synthetic VLM quick check (vision + mRoPE prefill + decode, no weights).
test-qwen25-vl-quick-check *ARGS:
    cargo test -p rlx-qwen25-vl --test vlm_quick_check --test aif_decode_quick_check {{ARGS}}
    cargo test -p rlx-models --test qwen25_vlm_quick_check --features qwen25-vl {{ARGS}}

# HF parity via Docker reference dump + RLX GGUF (needs weights + image).
test-qwen25-vl-parity *ARGS:
    cargo test -p rlx-models --test qwen25_vl_hf_parity --features qwen25-vl --release -- --nocapture {{ARGS}}

# Vision tower only — mmproj + image (no LM GGUF).
test-qwen25-vl-vision-parity *ARGS:
    cargo test -p rlx-models --test qwen25_vl_hf_parity qwen25_vl_vision_embed_parity --features qwen25-vl --release -- --nocapture {{ARGS}}

# AIF μ-guided decode vs HF reference (needs LM GGUF + docker/python reference with attentions).
test-qwen25-vl-aif-parity *ARGS:
    cargo test -p rlx-models --test qwen25_vl_hf_parity qwen25_vl_aif_mu_decode --features qwen25-vl --release -- --nocapture {{ARGS}}

# Paper AIF algorithm unit tests (Eq. 3–5, no weights).
test-qwen25-vl-aif-algo *ARGS:
    cargo test -p rlx-qwen25-vl --test aif_paper_algo --test aif_decode_quick_check {{ARGS}}

# Native AIF probe (graph Q/K vs CPU replay, synthetic).
test-qwen25-vl-native-probe *ARGS:
    cargo test -p rlx-qwen25-vl --test native_probe --test aif_paper_algo {{ARGS}}

# Native AIF probe + masked decode on GPU backends (needs --features all-backends).
test-qwen25-vl-aif-backends *ARGS:
    cargo test -p rlx-qwen25-vl --test aif_backend_probe {{ARGS}}

# Native AIF dynamics vs HF reference (needs LM GGUF + docker reference with attentions).
test-qwen25-vl-aif-native-parity *ARGS:
    cargo test -p rlx-models --test qwen25_vl_hf_parity qwen25_vl_aif_native_dynamics --features qwen25-vl --release -- --nocapture {{ARGS}}

# Decode-step AIF dynamics vs HF (RLX_AIF_DYNAMICS=decode_step in reference dump).
test-qwen25-vl-aif-decode-step-parity *ARGS:
    cargo test -p rlx-models --test qwen25_vl_hf_parity qwen25_vl_aif_native_decode_step_dynamics --features qwen25-vl --release -- --nocapture {{ARGS}}

# Export HF AIF probes for a VLMEvalKit JSONL (needs RLX_QWEN25_VL_HF_DIR).
export-qwen25-vl-aif-probes *ARGS:
    python3 scripts/aif_export_probes.py {{ARGS}}

# Baseline vs AIF eval on JSONL (see examples/aif_eval.rs).
eval-qwen25-vl-aif *ARGS:
    cargo run -p rlx-qwen25-vl --example aif_eval --release -- {{ARGS}}

# Native VLMEvalKit eval (RealWorldQA / TextVQA TSV or JSONL).
eval-qwen25-vl-vlmevalkit *ARGS:
    cargo run -p rlx-qwen25-vl --example vlmevalkit_eval --release -- {{ARGS}}

# AIF decode-step probe unit tests (synthetic).
test-qwen25-vl-aif-decode-step *ARGS:
    cargo test -p rlx-qwen25-vl --test aif_decode_step {{ARGS}}

# VLMEvalKit loader/scoring unit tests.
test-qwen25-vl-vlmevalkit *ARGS:
    cargo test -p rlx-qwen25-vl eval::vlmevalkit {{ARGS}}

# ---- Florence-2 (DaViT + BART vision-language) ----

florence2_dir := env_var_or_default("RLX_FLORENCE2_DIR", ".cache/florence2/Florence-2-large")

# Download the Florence-2-large checkpoint (weights + tokenizer + config).
fetch-florence2:
    hf download microsoft/Florence-2-large --local-dir {{florence2_dir}}

# Create the HF reference venv (.venv-florence2) for parity tests. Needs python3.11.
florence2-ref-venv:
    #!/usr/bin/env bash
    set -euo pipefail
    PY="${FLORENCE2_PY:-python3.11}"
    "$PY" -m venv .venv-florence2
    .venv-florence2/bin/pip install --quiet --upgrade pip
    .venv-florence2/bin/pip install --quiet "torch==2.4.1" "transformers==4.44.2" \
        timm einops pillow "numpy<2" safetensors "tokenizers<0.20"

# Run Florence-2 on an image. e.g. just florence2 -- --weights DIR --image img.jpg --task '<CAPTION>'
florence2 *ARGS:
    just run-bin rlx-florence2 rlx-florence2 {{ARGS}}

# Caption demo on the bundled sample image (CPU).
florence2-demo:
    cargo run -p rlx-florence2 --release -- \
        --weights {{florence2_dir}} \
        --image crates/rlx-locateanything/fixtures/sample.jpg --task '<CAPTION>'

florence2-all-backends *ARGS:
    cargo run -p rlx-florence2 --release --features all-backends -- {{ARGS}}

# Dump the HF reference fixtures (caption + OD) then run staged + e2e parity.
test-florence2-parity:
    #!/usr/bin/env bash
    set -euo pipefail
    DIR="$(cd {{florence2_dir}} && pwd)"
    PY="${RLX_FLORENCE2_PYTHON:-.venv-florence2/bin/python}"
    "$PY" scripts/florence2_hf_parity.py --model-dir "$DIR" --task '<CAPTION>' \
        --out .cache/florence2/parity_caption.json
    "$PY" scripts/florence2_hf_parity.py --model-dir "$DIR" --task '<OD>' \
        --image crates/rlx-locateanything/fixtures/sample.jpg --max-new-tokens 96 \
        --out .cache/florence2/parity_od.json
    RLX_FLORENCE2_DIR="$DIR" \
    RLX_FLORENCE2_FIXTURE="$(pwd)/.cache/florence2/parity_caption.json" \
    RLX_FLORENCE2_OD_FIXTURE="$(pwd)/.cache/florence2/parity_od.json" \
        cargo test -p rlx-florence2 --release --test florence2_hf_parity -- --nocapture --test-threads 1

# Cross-backend parity (CPU vs Metal/MLX) on the real checkpoint.
test-florence2-backends:
    RLX_FLORENCE2_DIR="$(cd {{florence2_dir}} && pwd)" \
    RLX_FLORENCE2_FIXTURE="$(pwd)/.cache/florence2/parity_caption.json" \
        cargo test -p rlx-florence2 --release --features apple-silicon \
        --test florence2_backend_parity -- --nocapture --test-threads 1

voxtral-tts *ARGS:
    just run-bin rlx-voxtral-tts rlx-voxtral-tts {{ARGS}}

# Stage timing on one Voxtral-4B-TTS checkpoint (RLX_VOXTRAL_TTS_DIR). --compare A/B compiled vs eager.
bench-voxtral-tts *ARGS:
    cargo run -p rlx-models --example voxtral_tts_bench --features "metal,mlx,apple-silicon" --release -- {{ARGS}}

fetch-qwen3-tts:
    cargo run -p rlx-models --example qwen3_tts_download --features "hf-download,qwen3-tts" --release

fetch-qwen3-tts-base:
    cargo run -p rlx-models --example qwen3_tts_download_base --features "hf-download,qwen3-tts" --release

# JFK clips + train_raw.jsonl (default: reference transcript alignment; JFK_TRANSCRIPT_MODE=whisper|hybrid)
qwen3-tts-jfk-prep:
    bash scripts/qwen3_tts_prep_jfk.sh

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

# All-Qwen duplex chat: question WAV → Qwen3-ASR → Qwen3 LM → Qwen3-TTS reply.
# Fastest RLX backends: Qwen3-ASR + TTS on Metal, Qwen3 LM on MLX (apple-silicon
# builds in both). The LM auto-uses the Q4_K_M GGUF sibling (weights/Qwen3-0.6B-gguf)
# when present. Prereqs:
#   just fetch-qwen3 fetch-qwen3-gguf fetch-qwen3-asr fetch-qwen3-tts-base
# Warm per-turn (M4 Pro): ASR ~0.5s · LM ~4s · TTS-TTFA ~1s → ~5.5s to first audio.
qwen-voice-chat-demo:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="{{justfile_directory()}}"
    export RLX_QWEN3_ASR_DIR="${RLX_QWEN3_ASR_DIR:-$ROOT/.cache/qwen3-asr/Qwen3-ASR-0.6B}"
    export RLX_QWEN3_WEIGHTS="${RLX_QWEN3_WEIGHTS:-$ROOT/weights/Qwen3-0.6B}"
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-$ROOT/.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base}"
    VECLIB_MAXIMUM_THREADS=1 cargo run --release -p rlx-qwen3-tts --features apple-silicon \
      --example qwen_voice_chat -- --fast \
      --device metal --qwen3-device mlx \
      --ref-wav "$ROOT/assets/jfk/jfk_voice_clone.wav" \
      --input-wav "$ROOT/crates/rlx-qwen3-tts/examples/audio/voice_chat_question.wav" \
      --out-dir /tmp/qwen_voice_chat

# Live talk-to-the-model: default microphone in, cloned-voice reply out the speaker.
# Adds the `mic` feature (pulls in cpal). Speak, pause to send, Ctrl-C to quit.
# Same prereqs as qwen-voice-chat-demo. Grant terminal mic permission on first run.
qwen-voice-chat-mic:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="{{justfile_directory()}}"
    export RLX_QWEN3_ASR_DIR="${RLX_QWEN3_ASR_DIR:-$ROOT/.cache/qwen3-asr/Qwen3-ASR-0.6B}"
    export RLX_QWEN3_WEIGHTS="${RLX_QWEN3_WEIGHTS:-$ROOT/weights/Qwen3-0.6B}"
    export RLX_QWEN3_TTS_DIR="${RLX_QWEN3_TTS_DIR:-$ROOT/.cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base}"
    VECLIB_MAXIMUM_THREADS=1 cargo run --release -p rlx-qwen3-tts --features apple-silicon,mic \
      --example qwen_voice_chat -- --fast --mic \
      --device metal --qwen3-device mlx \
      --ref-wav "$ROOT/assets/jfk/jfk_voice_clone.wav" \
      --out-dir /tmp/qwen_voice_chat

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

# Kokoro-82M ONNX bundle — https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
fetch-kokoro:
    cargo run -p rlx-kokoro --features hf-download --release -- --download

# Parler-TTS Mini v1 — https://huggingface.co/parler-tts/parler-tts-mini-v1
# Needs ONNX under weights/tts/parlertts/onnx (see crates/rlx-parlertts/scripts/export_onnx.py)
# and Descript DAC at weights/tts/parler-dac.
fetch-parler-dac:
    huggingface-cli download parler-tts/dac_44khz --local-dir weights/tts/parler-dac

# Soprano 1.1 ONNX (KV backbone + vocoder) — https://huggingface.co/KevinAHM/soprano-1.1-onnx
fetch-soprano:
    mkdir -p weights/tts/soprano
    .venv-hf/bin/hf download eugenehp/soprano soprano.rlxp --local-dir weights/tts/soprano
    @echo "Soprano in weights/tts/soprano (soprano.rlxp)"

export-soprano-rlxp DIR="weights/tts/soprano" OUT="weights/tts/soprano/soprano.rlxp":
    cargo run -p rlx-soprano --release -- --pack-rlxp "{{OUT}}" --model-dir "{{DIR}}"

export-soprano-gguf DIR="weights/tts/soprano" OUT="weights/tts/soprano/soprano.gguf":
    cargo run -p rlx-soprano --release -- --pack-gguf "{{OUT}}" --model-dir "{{DIR}}"

soprano-demo TEXT="The quick brown fox jumps over the lazy dog." DEVICE="metal":
    cargo run -p rlx-soprano --release --features apple-silicon -- \
        --text "{{TEXT}}" --device {{DEVICE}} --output /tmp/soprano_demo.wav

soprano-matrix:
    RLX_GREEDY=1 cargo run -p rlx-soprano --release --example backend_matrix --features apple-silicon

# Text-in → Soprano → Whisper text-out (CPU/Metal/MLX, …). Env: RLX_DEVICES,
# RLX_ORT_LATENTS=1 (ORT backbone latents → native Vocos only).
soprano-whisper TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" RLX_GREEDY=1 cargo run -p rlx-soprano --release --example whisper_roundtrip --features apple-silicon

# Brand/name check (longer phrase — short “Hello from Soprano.” → Whisper “Suprano”).
soprano-whisper-brand TEXT="Hello from the Soprano model.":
    RLX_TEXT="{{TEXT}}" RLX_GREEDY=1 cargo run -p rlx-soprano --release --example whisper_roundtrip --features apple-silicon

# Zonos v0.1 transformer (espeak → AR → DAC 44.1 kHz)
fetch-zonos:
    hf download Zyphra/Zonos-v0.1-transformer --local-dir weights/tts/zonos

zonos-demo TEXT="Hello from Zonos." OUT="zonos.wav" DEVICE="mlx":
    cargo run -p rlx-zonos --release --features "espeak,metal,mlx" -- \
        --text "{{TEXT}}" --device "{{DEVICE}}" --output "{{OUT}}"

zonos-whisper TEXT="Hello from Zonos.":
    RLX_TEXT="{{TEXT}}" RLX_MAX_TOKENS=256 cargo run -p rlx-zonos --release --example whisper_roundtrip --features "espeak,metal,mlx"

# Sentence across CPU/Metal/MLX (compiled backbone + DAC).
zonos-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" RLX_MAX_TOKENS=256 cargo run -p rlx-zonos --release --example backend_matrix --features "espeak,apple-silicon"

# Gepard (~556M) — Qwen3.5 FA backbone + NanoCodec FSQ @ 22.05 kHz
fetch-gepard:
    huggingface-cli download nineninesix/gepard-1.0 --local-dir weights/tts/gepard
    @echo "Also need nano_dec_1.89kbps.safetensors under weights/tts/gepard (see rlx-gepard README)."

gepard-demo TEXT="The quick brown fox jumps over the lazy dog." DEVICE="metal" OUT="/tmp/gepard_demo.wav":
    cargo run -p rlx-gepard --release --features apple-silicon -- \
        --weights weights/tts/gepard --text "{{TEXT}}" --device {{DEVICE}} --out "{{OUT}}"

gepard-whisper:
    cargo test -p rlx-gepard --release --features apple-silicon --test whisper_roundtrip test_gepard_whisper_fox -- --nocapture

gepard-whisper-long:
    cargo test -p rlx-gepard --release --features apple-silicon --test whisper_roundtrip test_gepard_whisper_long -- --nocapture

gepard-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" cargo run -p rlx-gepard --release --example backend_matrix --features all-backends

gepard-parity:
    cargo test -p rlx-gepard --release --features apple-silicon --test whisper_roundtrip test_gepard_compiled_cpu_prefill_parity -- --nocapture

gepard-bench DEVICE="metal" TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" cargo run -p rlx-gepard --release --example bench_timing --features all-backends -- --device {{DEVICE}}

gepard-validate-cuda:
    bash scripts/gepard_cuda_validate.sh

# Sesame CSM-1B — Llama-3.2-1B + depth → Mimi @ 24 kHz
fetch-sesame:
    mkdir -p weights/tts/sesame
    .venv-hf/bin/hf download unsloth/csm-1b --local-dir weights/tts/sesame
    @echo "CSM weights in weights/tts/sesame (also need: just fetch-mimi)"

sesame TEXT="The quick brown fox jumps over the lazy dog." DEVICE="cpu" OUT="/tmp/sesame.wav" SEED="42":
    cargo run -p rlx-sesame --release --features apple-silicon -- \
        --model-dir weights/tts/sesame --mimi-dir .cache/mimi \
        --text "{{TEXT}}" --device {{DEVICE}} --seed {{SEED}} --output "{{OUT}}"

sesame-whisper:
    cargo test -p rlx-sesame --release --features apple-silicon --test whisper_roundtrip test_sesame_whisper_fox -- --nocapture

sesame-whisper-long:
    cargo test -p rlx-sesame --release --features apple-silicon --test whisper_roundtrip test_sesame_whisper_long -- --nocapture

# Mimi decode + Whisper across available backends (LM codes cached after first run).
sesame-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" cargo run -p rlx-sesame --release --example backend_matrix --features all-backends

sesame-backends-long:
    RLX_TEXT="The quick brown fox jumps over the lazy dog. Courage and kindness matter more than cleverness alone when people face hard times together and choose to help each other without waiting for perfect conditions." \
    RLX_CODES_CACHE=/tmp/sesame_long_codes.json \
    RLX_SEED="${RLX_SESAME_LONG_SEED:-42}" \
    cargo run -p rlx-sesame --release --example backend_matrix --features all-backends

# Sesame CUDA on the remote host — set RLX_CUDA_HOST (sync + fetch weights on remote + fox/long + cpu,cuda matrix).
sesame-validate-cuda:
    bash scripts/sesame_cuda_validate.sh --remote

# MioTTS-0.6B — Qwen3 LM + MioCodec (ORT) @ 24 kHz
fetch-miotts:
    mkdir -p weights/tts/miotts weights/tts/miocodec
    .venv-hf/bin/hf download Aratako/MioTTS-0.6B --local-dir weights/tts/miotts
    .venv-hf/bin/hf download Aratako/MioCodec-25Hz-24kHz --local-dir weights/tts/miocodec
    @echo "MioTTS in weights/tts/miotts + MioCodec in weights/tts/miocodec"
    @echo "Then: .venv-miotts/bin/python crates/rlx-miotts/scripts/export_miocodec_decode.py"

export-miocodec:
    .venv-miotts/bin/python crates/rlx-miotts/scripts/export_miocodec_decode.py

miotts TEXT="The quick brown fox jumps over the lazy dog." DEVICE="cpu" OUT="/tmp/miotts.wav" SEED="42" PRESET="en_female":
    cargo run -p rlx-miotts --release --features apple-silicon -- \
        --model-dir weights/tts/miotts --codec-dir weights/tts/miocodec \
        --text "{{TEXT}}" --device {{DEVICE}} --seed {{SEED}} --preset {{PRESET}} --output "{{OUT}}"

miotts-whisper:
    cargo test -p rlx-miotts --release --features apple-silicon --test whisper_roundtrip test_miotts_whisper_fox -- --nocapture

miotts-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" cargo run -p rlx-miotts --release --example backend_matrix --features all-backends

gepard-whisper-cuda-long:
    cargo run -p rlx-gepard --release --features nvidia-gpu -- \
        --weights weights/tts/gepard \
        --text "The quick brown fox jumps over the lazy dog. Courage and kindness matter more than cleverness alone when people face hard times together and choose to help each other without waiting for perfect conditions." \
        --device cuda --seed 4 --out /tmp/gepard_long_cuda.wav

parlertts-demo TEXT="Hello from Parler.":
    cargo run -p rlx-parlertts --release -- \
        --text "{{TEXT}}" \
        --voice "A clear female voice speaks slowly." \
        --output /tmp/parlertts_demo.wav

# MetaVoice-1B — first/second-stage eager + EnCodec on --device
metavoice-demo TEXT="The quick brown fox jumps over the lazy dog." DEVICE="metal":
    cargo run -p rlx-metavoice --release --features apple-silicon -- \
        --text "{{TEXT}}" --max-tokens 448 --device {{DEVICE}} --output /tmp/metavoice_demo.wav

metavoice-matrix:
    cargo run -p rlx-metavoice --release --example backend_matrix --features apple-silicon

# One-command demo after `just fetch-kokoro`
kokoro-demo:
    cargo run -p rlx-kokoro --release -- --text "Hello from Kokoro." --voice af_heart --out /tmp/kokoro_demo.wav

# StyleTTS2-family = Kokoro-82M thin facade (`rlx-styletts2` → native Kokoro)
styletts2 TEXT="The quick brown fox jumps over the lazy dog." DEVICE="cpu" OUT="/tmp/styletts2.wav" VOICE="af_heart":
    cargo run -p rlx-styletts2 --release --features apple-silicon -- \
        --text "{{TEXT}}" --voice "{{VOICE}}" --device {{DEVICE}} --output "{{OUT}}"

styletts2-whisper:
    cargo test -p rlx-styletts2 --release --features apple-silicon --test whisper_roundtrip test_styletts2_whisper_fox -- --nocapture

styletts2-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" cargo run -p rlx-styletts2 --release --example backend_matrix --features apple-silicon

# Supertonic-3 flow-matching TTS — https://huggingface.co/Supertone/supertonic-3
fetch-supertonic:
    cargo run -p rlx-supertonic --features hf-download --release -- --download

# One-command demo after `just fetch-supertonic`
supertonic-demo:
    cargo run -p rlx-supertonic --release -- --text "Hello from Supertonic." --voice F1 --out /tmp/supertonic_demo.wav

# LuxTTS voice-cloning TTS — https://huggingface.co/YatharthS/LuxTTS
# After downloading, export the Vocos spectral head once (needs a venv with `vocos onnxscript`):
#   python crates/rlx-luxtts/scripts/export_vocoder.py weights/tts/luxtts/vocoder/vocos.bin weights/tts/luxtts/onnx/vocoder_spec.onnx
luxtts-demo PROMPT_WAV PROMPT_TEXT TEXT:
    cargo run -p rlx-luxtts --release -- --prompt-wav "{{PROMPT_WAV}}" --prompt-text "{{PROMPT_TEXT}}" --text "{{TEXT}}" --out /tmp/luxtts_demo.wav

# F5-TTS voice cloning — https://huggingface.co/huggingfacess/F5-TTS-ONNX (weights CC-BY-NC)
# Needs F5_{Preprocess,Transformer,Decode}.onnx + vocab.txt in weights/tts/f5tts
fetch-f5tts:
    mkdir -p weights/tts/f5tts
    .venv-hf/bin/hf download huggingfacess/F5-TTS-ONNX --local-dir weights/tts/f5tts
    @echo "F5-TTS ONNX in weights/tts/f5tts (add vocab.txt from SWivid/F5-TTS if missing)"

f5tts TEXT="The quick brown fox jumps over the lazy dog." DEVICE="cpu" NFE="32" OUT="/tmp/f5tts.wav" REF="crates/rlx-f5tts/tests/fixtures/prompt.wav" REF_TEXT="Hello from Kokoro. This is a test of speech synthesis in Rust.":
    cargo run -p rlx-f5tts --release --features apple-silicon -- \
      --ref-wav "{{REF}}" --ref-text "{{REF_TEXT}}" --text "{{TEXT}}" \
      --nfe "{{NFE}}" --device "{{DEVICE}}" --out "{{OUT}}"

f5tts-whisper OUT="tmp/f5tts_wavs/validated.wav":
    mkdir -p tmp/f5tts_wavs
    RLX_F5TTS_DIR=weights/tts/f5tts RLX_WHISPER_DIR=.cache/whisper-base.en NFE=32 OUT="{{OUT}}" \
      cargo run -p rlx-f5tts --release --features apple-silicon --quiet --example native_clone

f5tts-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" RLX_NFE=32 cargo run -p rlx-f5tts --release --example backend_matrix --features apple-silicon

f5tts-demo REF_WAV REF_TEXT TEXT:
    cargo run -p rlx-f5tts --release --features apple-silicon -- \
      --ref-wav "{{REF_WAV}}" --ref-text "{{REF_TEXT}}" --text "{{TEXT}}" --nfe 32 --out /tmp/f5tts_demo.wav

# Piper VITS TTS — voices at https://huggingface.co/rhasspy/piper-voices (MIT)
# Place <voice>.onnx + <voice>.onnx.json in weights/tts/piper/
piper-demo:
    cargo run -p rlx-piper --release -- --text "The quick brown fox jumps over the lazy dog." --out /tmp/piper_demo.wav

piper-backends:
    RLX_PIPER_DETERMINISTIC=1 RLX_ITERS=1 cargo run -p rlx-piper --release --example backend_matrix --features apple-silicon -- weights/tts/piper

# ZipVoice voice cloning — k2-fsa/ZipVoice (Apache); reuses the LuxTTS runtime.
# Download zipvoice_distill/ + export vocoder (scripts/export_vocoder.py) into weights/tts/zipvoice-distill
zipvoice-demo REF_WAV REF_TEXT TEXT:
    cargo run -p rlx-zipvoice --release -- --prompt-wav "{{REF_WAV}}" --prompt-text "{{REF_TEXT}}" --text "{{TEXT}}" --out /tmp/zipvoice_demo.wav

# MOSS-TTS-Nano — OpenMOSS hierarchical AR codec-LM (Apache, 48kHz). See crate README
# for weights setup (2 HF repos + scripts/convert_tokenizer.py). --list-voices for voices.
fetch-moss-nano:
    mkdir -p weights/tts/moss-nano/codec
    .venv-hf/bin/hf download eugenehp/moss-nano moss-nano.rlxp --local-dir weights/tts/moss-nano
    @echo "MOSS-TTS-Nano in weights/tts/moss-nano (moss-nano.rlxp)"

export-moss-nano-rlxp DIR="weights/tts/moss-nano" OUT="weights/tts/moss-nano/moss-nano.rlxp":
    cargo run -p rlx-moss-nano --release -- --pack-rlxp --data "{{DIR}}" --out "{{OUT}}"

export-moss-nano-gguf DIR="weights/tts/moss-nano" OUT="weights/tts/moss-nano/moss-nano.gguf":
    cargo run -p rlx-moss-nano --release -- --pack-gguf --data "{{DIR}}" --out "{{OUT}}"

moss-nano TEXT="The quick brown fox jumps over the lazy dog." VOICE="Trump" DEVICE="cpu" OUT="/tmp/moss_nano.wav":
    cargo run -p rlx-moss-nano --release --features apple-silicon -- \
      --text "{{TEXT}}" --voice "{{VOICE}}" --device "{{DEVICE}}" --out "{{OUT}}"

moss-nano-whisper:
    cargo run -p rlx-moss-nano --release --features apple-silicon --quiet --example native_whisper

moss-nano-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" cargo run -p rlx-moss-nano --release --example backend_matrix --features apple-silicon

moss-nano-demo TEXT VOICE="Trump":
    cargo run -p rlx-moss-nano --release --features apple-silicon -- --text "{{TEXT}}" --voice "{{VOICE}}" --out /tmp/moss_nano_demo.wav

# Maya1 — expressive voice-design TTS (Llama-3B + SNAC, Apache). Reuses rlx-orpheus.
# Needs a Maya1 GGUF in weights/tts/maya1 + ORPHEUS_SNAC_PATH (see crate README).
maya1-demo TEXT DESC="Realistic female voice in her 20s with a British accent. Warm timbre, conversational pacing.":
    cargo run -p rlx-maya1 --release -- --description "{{DESC}}" --text "{{TEXT}}" --out /tmp/maya1_demo.wav

# MeloTTS — VITS2 multilingual TTS (MIT). Real inference via rlx-tiny-tts engine.
melotts-demo TEXT:
    cargo run -p rlx-melotts --release --features apple-silicon -- --text "{{TEXT}}" --out /tmp/melotts_demo.wav

# TinyTTS/MeloTTS cross-backend parity (cos≥0.95). Override: RLX_DEVICES=cpu,gpu
melotts-backends *ARGS:
    cargo run -p rlx-tiny-tts --release --example backend_matrix --features apple-silicon -- weights/tts/melotts {{ARGS}}

tiny-tts-backends *ARGS:
    cargo run -p rlx-tiny-tts --release --example backend_matrix --features apple-silicon -- weights/tts/tiny-tts-rlx {{ARGS}}

fetch-tiny-tts:
    mkdir -p weights/tts/tiny-tts-rlx
    .venv-hf/bin/hf download eugenehp/tiny-tts-rlx tiny-tts.rlxp --local-dir weights/tts/tiny-tts-rlx
    @echo "TinyTTS in weights/tts/tiny-tts-rlx (tiny-tts.rlxp)"

# MeloTTS shares the TinyTTS Hub pack (local symlink).
fetch-melotts: fetch-tiny-tts
    mkdir -p weights/tts
    ln -sfn tiny-tts-rlx weights/tts/melotts
    @echo "MeloTTS alias → weights/tts/melotts → tiny-tts-rlx"

# Pack TinyTTS/MeloTTS dir → official flat `.rlxp` (RLXPFLAT sidecars).
export-tiny-tts-rlxp DIR="weights/tts/tiny-tts-rlx" OUT="weights/tts/tiny-tts-rlx/tiny-tts.rlxp":
    cargo run -p rlx-tiny-tts --release --example pack_rlxp -- "{{DIR}}" "{{OUT}}"

# Legacy alias (same as export-tiny-tts-rlxp).
export-tiny-tts-rlxpack DIR="weights/tts/tiny-tts-rlx" OUT="weights/tts/tiny-tts-rlx/tiny-tts.rlxp":
    just export-tiny-tts-rlxp "{{DIR}}" "{{OUT}}"

# OpenVoice v2 — zero-shot cloning (MIT). MeloTTS base + ONNX tone-color converter.
openvoice-demo REF_WAV TEXT:
    cargo run -p rlx-openvoice --release -- --ref-wav "{{REF_WAV}}" --text "{{TEXT}}" --out /tmp/openvoice_demo.wav

# ChatterBox — Resemble AI 0.5B Llama T3 + S3Gen zero-shot cloning (MIT, native RLX).
fetch-chatterbox:
    mkdir -p weights/tts/chatterbox
    .venv-hf/bin/hf download synath/chatterbox-ONNX --local-dir weights/tts/chatterbox
    @echo "ChatterBox in weights/tts/chatterbox"

chatterbox TEXT="The quick brown fox jumps over the lazy dog." DEVICE="cpu" OUT="/tmp/chatterbox.wav" REF="crates/rlx-luxtts/tests/fixtures/prompt.wav":
    cargo run -p rlx-chatterbox --release --features apple-silicon -- \
        --ref-wav "{{REF}}" --text "{{TEXT}}" --device "{{DEVICE}}" --greedy --out "{{OUT}}"

chatterbox-whisper:
    cargo test -p rlx-chatterbox --release --features apple-silicon --test whisper_roundtrip test_chatterbox_whisper_fox -- --nocapture

chatterbox-backends TEXT="The quick brown fox jumps over the lazy dog.":
    RLX_TEXT="{{TEXT}}" cargo run -p rlx-chatterbox --release --example backend_matrix --features apple-silicon

chatterbox-demo REF_WAV TEXT:
    cargo run -p rlx-chatterbox --release --features apple-silicon -- --ref-wav "{{REF_WAV}}" --text "{{TEXT}}" --greedy --out /tmp/chatterbox_demo.wav

# Kyutai Mimi codec — https://huggingface.co/kyutai/mimi
fetch-mimi:
    cargo run -p rlx-mimi --features hf-download --release -- --fetch

# Kyutai TTS 1.6B — https://huggingface.co/kyutai/tts-1.6b-en_fr
fetch-kyutai-tts:
    cargo run -p rlx-kyutai-tts --features hf-download --release -- --fetch

kyutai-tts PROMPT="Hello from Kyutai." DEVICE="cpu" OUT="/tmp/kyutai_tts.wav":
    cargo run -p rlx-kyutai-tts --release --features "hf-download,apple-silicon" -- \
        --prompt "{{PROMPT}}" --device "{{DEVICE}}" --out-wav "{{OUT}}"

kyutai-tts-e2e:
    RLX_KYUTAI_TTS_E2E=1 cargo test -p rlx-kyutai-tts --release --features "hf-download,apple-silicon" --test whisper_validate -- --nocapture

# Descript Audio Codec — https://github.com/descriptinc/descript-audio-codec
fetch-dac MODEL="24khz":
    cargo run -p rlx-dac --features hf-download --release -- --fetch --model-type {{MODEL}}

dac *ARGS:
    cargo run -p rlx-dac --features hf-download --release -- {{ARGS}}

test-dac *ARGS:
    export RLX_DAC_DIR="${RLX_DAC_DIR:-.cache/dac/24khz}"
    cargo test -p rlx-dac --release -- {{ARGS}}

mimi *ARGS:
    cargo run -p rlx-mimi --features hf-download --release -- {{ARGS}}

fetch-tsac:
    cargo run -p rlx-tsac --features fetch --release -- --fetch

tsac *ARGS:
    cargo run -p rlx-tsac --features fetch --release -- {{ARGS}}

test-tsac *ARGS:
    export RLX_TSAC_DIR="${RLX_TSAC_DIR:-.cache/tsac}"
    cargo test -p rlx-tsac --release -- {{ARGS}}

bench-tsac-parity *ARGS:
    just fetch-tsac
    export RLX_TSAC_DIR="${RLX_TSAC_DIR:-.cache/tsac}"
    export RLX_TSAC_PARITY=1
    cargo test -p rlx-tsac --test bellard_parity --release --features "fetch,native-codec" -- --nocapture {{ARGS}}

bench-tsac-parity-example *ARGS:
    just fetch-tsac
    export RLX_TSAC_DIR="${RLX_TSAC_DIR:-.cache/tsac}"
    cargo run -p rlx-tsac --example bellard_parity_bench --release --features "fetch,native-codec" -- {{ARGS}}

test-mimi *ARGS:
    export RLX_MIMI_DIR="${RLX_MIMI_DIR:-.cache/mimi}"
    cargo test -p rlx-mimi --release -- {{ARGS}}

# HF transformers encode/decode parity (baked fixture on 24 kHz ask_not.wav)
test-mimi-hf *ARGS:
    python3 scripts/mimi_hf_parity.py \
      --wav crates/rlx-qwen3-tts/examples/audio/ask_not.wav \
      --out crates/rlx-mimi/tests/fixtures/hf_ask_not.json
    export RLX_MIMI_DIR="${RLX_MIMI_DIR:-.cache/mimi}"
    cargo test -p rlx-mimi --test hf_parity --release -- --nocapture {{ARGS}}

test-mimi-whisper *ARGS:
    just fetch-mimi
    bash scripts/fetch_whisper_bench.sh
    export RLX_MIMI_DIR="${RLX_MIMI_DIR:-.cache/mimi}"
    cargo test -p rlx-mimi --test whisper_roundtrip --release -- --nocapture {{ARGS}}

# Kyutai Moshi speech-to-speech — https://huggingface.co/kyutai/moshiko-candle-bf16
fetch-moshi:
    cargo run -p rlx-moshi --features hf-download --release -- --fetch

fetch-moshi-q8:
    cargo run -p rlx-moshi --features hf-download --release -- --fetch --checkpoint q8

fetch-moshi-q4:
    cargo run -p rlx-moshi --features hf-download --release -- --fetch --checkpoint q4

fetch-moshi-mlx-bf16:
    cargo run -p rlx-moshi --features hf-download --release -- --fetch --checkpoint mlx-bf16

fetch-moshika:
    cargo run -p rlx-moshi --features hf-download --release -- --fetch --variant moshika-one-way

fetch-moshika-q4:
    cargo run -p rlx-moshi --features hf-download --release -- --fetch --variant moshika-one-way --checkpoint q4

fetch-moshika-mlx-bf16:
    cargo run -p rlx-moshi --features hf-download --release -- --fetch --variant moshika-one-way --checkpoint mlx-bf16

moshi *ARGS:
    cargo run -p rlx-moshi --features "hf-download,gpu-lm,compiled-lm,mlx-lm,metal" --release -- {{ARGS}}

# Audio-to-audio voice chat with Moshi (one full-duplex model — no ASR/LLM/TTS).
# Batch: drive Moshi from a WAV, write its spoken reply. Needs full-duplex weights
# + Mimi: `just fetch-mimi && just fetch-moshi` (or fetch-moshi-q8 + RLX_MOSHI_CHECKPOINT=q8).
moshi-voice-chat *ARGS:
    cargo run --release -p rlx-moshi --features "apple-silicon,hf-download" \
      --example moshi_voice_chat -- --device metal {{ARGS}}

# Live mic ↔ Moshi ↔ speaker, full-duplex. USE HEADPHONES (Moshi hears the mic
# continuously). Needs a GPU for real-time (7B @ 12.5 Hz).
moshi-voice-chat-mic *ARGS:
    cargo run --release -p rlx-moshi --features "apple-silicon,hf-download,mic" \
      --example moshi_voice_chat -- --device metal --mic {{ARGS}}

test-moshi *ARGS:
    export RLX_MOSHI_DIR="${RLX_MOSHI_DIR:-.cache/moshiko}"
    export RLX_MIMI_DIR="${RLX_MIMI_DIR:-.cache/mimi}"
    cargo test -p rlx-moshi --release -- {{ARGS}}

test-moshi-weights *ARGS:
    export RLX_MOSHI_DIR="${RLX_MOSHI_DIR:-.cache/moshiko}"
    cargo test -p rlx-moshi --test weight_keys --release -- --nocapture {{ARGS}}

test-moshi-e2e *ARGS:
    just fetch-moshi
    just fetch-mimi
    export RLX_MOSHI_DIR="${RLX_MOSHI_DIR:-.cache/moshiko}"
    export RLX_MIMI_DIR="${RLX_MIMI_DIR:-.cache/mimi}"
    cargo test -p rlx-moshi --release -- --nocapture {{ARGS}}

test-moshi-stream-whisper *ARGS:
    just fetch-moshi
    just fetch-mimi
    bash scripts/fetch_whisper_bench.sh 2>/dev/null || just fetch-whisper-base
    export RLX_MOSHI_DIR="${RLX_MOSHI_DIR:-.cache/moshiko}"
    export RLX_MIMI_DIR="${RLX_MIMI_DIR:-.cache/mimi}"
    export RLX_MOSHI_STREAM_E2E=1
    cargo test -p rlx-moshi --test stream_whisper_roundtrip --features all-backends --release -- --nocapture {{ARGS}}

moshi-ws *ARGS:
    cargo run -p rlx-moshi --example ws_server --features "ws-server,hf-download,all-backends" --release -- {{ARGS}}

# Orpheus TTS — unsloth/orpheus-3b-0.1-ft-GGUF (Q4_K_M default)
# Writes /tmp/rlx-weights/orpheus/… and links into weights/tts/orpheus/.
fetch-orpheus QUANT="Q4_K_M":
    cargo run -p rlx-orpheus --features "llama,hf-download" --release -- \
      --download-orpheus --quant {{QUANT}}
    mkdir -p weights/tts/orpheus
    ln -sfn /tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-{{QUANT}}.gguf \
        weights/tts/orpheus/orpheus-3b-0.1-ft-{{QUANT}}.gguf
    @echo "Orpheus GGUF → weights/tts/orpheus/orpheus-3b-0.1-ft-{{QUANT}}.gguf"

# Prefer already-exported decoder under weights/tts/snac_24khz when present.
fetch-orpheus-snac:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p weights/tts/snac /tmp/rlx-weights/snac
    if [[ -f weights/tts/snac_24khz/snac_24khz_decoder.safetensors ]]; then
      ln -sfn "$(pwd)/weights/tts/snac_24khz/snac_24khz_decoder.safetensors" \
          weights/tts/snac/snac_24khz_decoder.safetensors
      ln -sfn "$(pwd)/weights/tts/snac_24khz/snac_24khz_decoder.safetensors" \
          /tmp/rlx-weights/snac/snac_24khz_decoder.safetensors
      echo "SNAC decoder → weights/tts/snac/ (from snac_24khz)"
    else
      cargo run -p rlx-orpheus --features "llama,hf-download" --release -- --download-snac
      just export-orpheus-snac
      ln -sfn /tmp/rlx-weights/snac/snac_24khz_decoder.safetensors \
          weights/tts/snac/snac_24khz_decoder.safetensors
    fi

export-orpheus-snac OUT="/tmp/rlx-weights/snac":
    python3 scripts/export_snac_decoder.py --out {{OUT}}

export-orpheus-audio-assets:
    export ORPHEUS_SNAC_PATH="${ORPHEUS_SNAC_PATH:-/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors}"
    test -f "$ORPHEUS_SNAC_PATH" || { echo "missing SNAC — run \`just export-orpheus-snac\`" >&2; exit 1; }
    cargo run -p rlx-orpheus --example export_audio_assets --release --features "llama,coreml,metal"

orpheus-coreml-demo: fetch-orpheus fetch-orpheus-snac export-orpheus-snac
    export ORPHEUS_SNAC_PATH="/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors"
    cargo run -p rlx-orpheus --release --features "llama,coreml" -- \
      --weights /tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf \
      --device coreml \
      --text "Hello from RLX on CoreML." \
      --voice tara \
      --max-tokens 120 \
      --out /tmp/orpheus-coreml-demo.wav

orpheus *ARGS:
    cargo run -p rlx-orpheus --release -- {{ARGS}}

orpheus-demo: fetch-orpheus fetch-orpheus-snac
    #!/usr/bin/env bash
    set -euo pipefail
    export ORPHEUS_SNAC_PATH="/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors"
    just orpheus -- \
      --weights /tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf \
      --text "Hello from RLX Orpheus." \
      --voice tara \
      --device auto \
      --out /tmp/orpheus-demo.wav

orpheus-wgpu-demo: fetch-orpheus fetch-orpheus-snac
    #!/usr/bin/env bash
    set -euo pipefail
    export ORPHEUS_SNAC_PATH="/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors"
    cargo run -p rlx-orpheus --release --features "llama,gpu" -- \
      --weights /tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf \
      --text "Hello from RLX on wgpu." \
      --voice tara \
      --device gpu \
      --max-tokens 120 \
      --out /tmp/orpheus-wgpu-demo.wav

# Encode reference WAV -> JSON for Orpheus zero-shot clone (needs Python snac).
orpheus-encode-ref WAV TRANSCRIPT OUT="/tmp/jfk_orpheus_ref.json":
    python3 scripts/orpheus_encode_reference.py --wav "{{WAV}}" --transcript "{{TRANSCRIPT}}" --out "{{OUT}}"

# Voice clone walkthrough (pretrained GGUF — set ORPHEUS_PRETRAINED_GGUF).
orpheus-voice-clone REF_JSON="/tmp/jfk_orpheus_ref.json" *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export ORPHEUS_SNAC_PATH="${ORPHEUS_SNAC_PATH:-/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors}"
    test -f "$ORPHEUS_SNAC_PATH" || { echo "missing SNAC — run \`just fetch-orpheus-snac\`" >&2; exit 1; }
    cargo run -p rlx-orpheus --example voice_clone --release --features apple-silicon -- \
      --ref-json "{{REF_JSON}}" {{ARGS}}

test-orpheus *ARGS:
    cargo test -p rlx-orpheus --release -- {{ARGS}}

test-orpheus-whisper *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just fetch-whisper 2>/dev/null || true
    export ORPHEUS_SNAC_PATH="${ORPHEUS_SNAC_PATH:-/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors}"
    test -f "$ORPHEUS_SNAC_PATH" || { echo "missing SNAC — run \`just fetch-orpheus-snac\`" >&2; exit 1; }
    cargo test -p rlx-orpheus --test whisper_roundtrip --features llama --release golden_codec_intelligible_via_whisper -- --nocapture {{ARGS}}

test-orpheus-whisper-e2e *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just fetch-whisper 2>/dev/null || true
    export ORPHEUS_SNAC_PATH="${ORPHEUS_SNAC_PATH:-/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors}"
    export ORPHEUS_GGUF_PATH="${ORPHEUS_GGUF_PATH:-/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf}"
    export ORPHEUS_WHISPER_E2E=1
    test -f "$ORPHEUS_GGUF_PATH" || { echo "missing Orpheus GGUF — run \`just fetch-orpheus\`" >&2; exit 1; }
    test -f "$ORPHEUS_SNAC_PATH" || { echo "missing SNAC — run \`just fetch-orpheus-snac\`" >&2; exit 1; }
    cargo test -p rlx-orpheus --test whisper_roundtrip --features "llama,metal" --release roundtrip_text_via_whisper_e2e -- --ignored --nocapture {{ARGS}}

test-orpheus-backends-whisper *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just fetch-whisper 2>/dev/null || true
    export ORPHEUS_SNAC_PATH="${ORPHEUS_SNAC_PATH:-/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors}"
    export ORPHEUS_GGUF_PATH="${ORPHEUS_GGUF_PATH:-/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf}"
    test -f "$ORPHEUS_SNAC_PATH" || { echo "missing SNAC — run \`just fetch-orpheus-snac\`" >&2; exit 1; }
    test -f "$ORPHEUS_GGUF_PATH" || { echo "missing Orpheus GGUF — run \`just fetch-orpheus\`" >&2; exit 1; }
    cargo test -p rlx-orpheus --test backends_whisper --features all-backends --release -- --nocapture {{ARGS}}

bench-orpheus *ARGS:
    cargo run -p rlx-orpheus --example tts_bench --release --features apple-silicon -- {{ARGS}}

bench-orpheus-all-devices *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just fetch-whisper 2>/dev/null || true
    just bench-orpheus -- --devices all --whisper {{ARGS}}

bench-orpheus-voice-clone *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just fetch-whisper 2>/dev/null || true
    export ORPHEUS_SNAC_PATH="${ORPHEUS_SNAC_PATH:-/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors}"
    export ORPHEUS_CLONE_REF_JSON="${ORPHEUS_CLONE_REF_JSON:-/tmp/jfk_orpheus_ref.json}"
    test -f "$ORPHEUS_SNAC_PATH" || { echo "missing SNAC" >&2; exit 1; }
    test -f "$ORPHEUS_CLONE_REF_JSON" || { echo "missing ref JSON — run \`just orpheus-encode-ref …\`" >&2; exit 1; }
    just bench-orpheus -- --devices all --voice-clone --whisper --clone-ref "$ORPHEUS_CLONE_REF_JSON" {{ARGS}}

# Generate demo WAVs (short/long voices + optional clone from jfk_ref.json in ORPHEUS_DEMO_DIR).
orpheus-demos *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export ORPHEUS_GGUF_PATH="${ORPHEUS_GGUF_PATH:-/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf}"
    export ORPHEUS_SNAC_PATH="${ORPHEUS_SNAC_PATH:-/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors}"
    export ORPHEUS_DEMO_DIR="${ORPHEUS_DEMO_DIR:-/tmp/orpheus-demos}"
    test -f "$ORPHEUS_GGUF_PATH" || { echo "missing Orpheus GGUF — run \`just fetch-orpheus\`" >&2; exit 1; }
    test -f "$ORPHEUS_SNAC_PATH" || { echo "missing SNAC — export with scripts/export_snac_decoder.py" >&2; exit 1; }
    cargo run -p rlx-orpheus --example batch_demos --release --features apple-silicon -- {{ARGS}}

test-orpheus-clone-whisper *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    export ORPHEUS_CLONE_BENCH=1
    just test-orpheus-backends-whisper voice_clone_whisper -- --ignored --nocapture {{ARGS}}

# Re-export Kitten ONNX → RLX bundle (graph.json + weights.safetensors).
# Set KITTEN_ONNX_PATH to override the default HF cache snapshot.
export-kitten-rlx-bundle:
    #!/usr/bin/env bash
    set -euo pipefail
    ONNX="${KITTEN_ONNX_PATH:-$HOME/.cache/huggingface/hub/models--KittenML--kitten-tts-mini-0.8/snapshots/c02725660cea441db4c383af69f1f26f5cd00947/kitten_tts_mini_v0_8.onnx}"
    OUT="crates/kitten_tts_mini_rlx/weights/rlx_bundle"
    test -f "$ONNX" || { echo "missing ONNX: $ONNX (set KITTEN_ONNX_PATH)" >&2; exit 1; }
    python3 scripts/export_kitten_rlx_bundle.py "$ONNX" "$OUT"

# Decompose ONNX → native Rust graph + model.safetensors (no graph.json at runtime).
export-kitten-native-weights:
    #!/usr/bin/env bash
    set -euo pipefail
    just export-kitten-rlx-bundle
    cargo build -p rlx-onnx-decompose --release
    rlx-onnx-decompose --bundle crates/kitten_tts_mini_rlx/weights/rlx_bundle \
      -o crates/kitten_tts_mini_rlx --crate-name kitten_tts_mini_rlx \
      --seq-len 128 --max-samples 48000

# Optional GGUF weight container (requires: pip install gguf safetensors numpy in a venv).
export-kitten-gguf:
    #!/usr/bin/env bash
    set -euo pipefail
    test -f crates/kitten_tts_mini_rlx/weights/model.safetensors || just export-kitten-native-weights
    python3 -c "import gguf" 2>/dev/null || {
      echo "install: python3 -m venv .venv-kitten && .venv-kitten/bin/pip install gguf safetensors numpy" >&2
      exit 1
    }
    python3 scripts/onnx_decompose_to_gguf.py \
      crates/kitten_tts_mini_rlx/weights/model.safetensors \
      crates/kitten_tts_mini_rlx/weights/model.gguf

test-kitten-native-compile:
    KITTEN_RLX_WEIGHTS=crates/kitten_tts_mini_rlx/weights \
      cargo run -p kitten_tts_mini_rlx --example native_weights_compile_check --release --features native

# Ensure the private RLX TTS bundle lives under weights/tts/rlx-tts (gitignored).
# Optional: just tts-prepare SRC=/path/to/bundle
tts-prepare SRC="":
    #!/usr/bin/env bash
    set -euo pipefail
    dest="weights/tts/rlx-tts"
    mkdir -p "$dest"
    if [[ -n "{{SRC}}" ]]; then
      src="{{SRC}}"
      echo "syncing $src → $dest"
      rsync -a --delete "$src"/ "$dest"/
    fi
    if [[ -f "$dest/gryphon.cfg" && ! -f "$dest/post.cfg" ]]; then
      cp "$dest/gryphon.cfg" "$dest/post.cfg"
    fi
    rm -f "$dest/gryphon.cfg"
    # Drop Apple-compare fixtures and unused frontend blobs if present.
    rm -rf "$dest/fixtures"
    rm -f "$dest/.DS_Store" "$dest/frontend.cfg" "$dest/gryphon.cfg"
    rm -f "$dest/frontend/gprm" "$dest/frontend/g2p_seq2seq.bin" \
      "$dest/frontend/g2p_seq2seq.arch.json" "$dest/frontend/g2p_seq2seq.stack.json" \
      "$dest/frontend/g2p_seq2seq.inventory.json" "$dest/frontend/g2p_seq2seq.meta.json" \
      "$dest/frontend/phbk" "$dest/frontend/phonetic/to_xsampa.json" \
      "$dest/frontend/phonetic/symbols.json"
    rm -rf "$dest"/.rlx-extracted-*
    rm -rf .cache/simone-rlx
    if [[ -f "$dest/manifest.json" ]]; then
      cargo run -p rlx-tts --quiet -- --sanitize-manifest "$dest/manifest.json"
    fi
    mkdir -p .cache
    ln -sfn ../weights/tts/rlx-tts .cache/rlx-tts
    if [[ -f "$dest/rlx-tts.gguf" ]]; then
      echo "RLX TTS bundle ready (GGUF): $dest/rlx-tts.gguf ($(du -sh "$dest/rlx-tts.gguf" | awk '{print $1}'))"
    else
      test -f "$dest/manifest.json"
      test -f "$dest/encoder.safetensors"
      test -f "$dest/decoder.safetensors"
      test -f "$dest/wavernn.safetensors"
      echo "RLX TTS bundle ready: $dest ($(du -sh "$dest" | awk '{print $1}'))"
    fi

# Drop loose assets once rlx-tts.rlxp (or legacy gguf) exists.
tts-pack-only:
    #!/usr/bin/env bash
    set -euo pipefail
    dest="weights/tts/rlx-tts"
    if [[ -f "$dest/rlx-tts.rlxp" ]]; then
      keep="rlx-tts.rlxp"
    elif [[ -f "$dest/rlx-tts.gguf" ]]; then
      keep="rlx-tts.gguf"
    else
      echo "missing $dest/rlx-tts.rlxp — run just export-rlx-tts-rlxp first" >&2
      exit 1
    fi
    find "$dest" -mindepth 1 -maxdepth 1 ! -name "$keep" ! -name 'README.md' ! -name 'LICENSE' ! -name '.gitattributes' -exec rm -rf {} +
    echo "kept $dest/$keep ($(du -sh "$dest/$keep" | awk '{print $1}'))"

export-rlx-tts-rlxp BUNDLE="weights/tts/rlx-tts" OUT="":
    #!/usr/bin/env bash
    set -euo pipefail
    out="{{OUT}}"
    if [[ -z "$out" ]]; then
      cargo run -p rlx-tts --release -- --pack-rlxp --bundle "{{BUNDLE}}"
    else
      cargo run -p rlx-tts --release -- --pack-rlxp --bundle "{{BUNDLE}}" --out "$out"
    fi

fetch-rlx-tts:
    mkdir -p weights/tts/rlx-tts
    .venv-hf/bin/hf download eugenehp/rlx-tts rlx-tts.rlxp --local-dir weights/tts/rlx-tts
    @echo "RLX TTS in weights/tts/rlx-tts (rlx-tts.rlxp)"

# Pack directory bundle → single runnable GGUF (legacy). Pure Rust — no Python.
export-rlx-tts-gguf BUNDLE="weights/tts/rlx-tts" OUT="":
    #!/usr/bin/env bash
    set -euo pipefail
    out="{{OUT}}"
    if [[ -z "$out" ]]; then
      cargo run -p rlx-tts --release -- --pack-gguf --bundle "{{BUNDLE}}"
    else
      cargo run -p rlx-tts --release -- --pack-gguf --bundle "{{BUNDLE}}" --out "$out"
    fi

# Alias
tts-extract *ARGS:
    just tts-prepare {{ARGS}}

tts-native *ARGS:
    cargo run -p rlx-tts --release -- {{ARGS}}

tts-probe:
    cargo run -p rlx-tts --release -- --probe-bundle

tts-demo:
    cargo run -p rlx-tts --release -- --text "Hello from RLX." --out /tmp/rlx_tts_demo.wav

test-tts *ARGS:
    cargo test -p rlx-tts --release -- {{ARGS}}

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

fetch-kittentts-whisper:
    just fetch-whisper-base

test-kittentts-native *ARGS:
    cargo test -p rlx-kittentts --features native --release --test native_smoke -- {{ARGS}}

test-kittentts-native-speed *ARGS:
    cargo test -p rlx-kittentts --features "native-fast" --release --test native_infer_speed -- {{ARGS}}

# Short+long IPA × every available RLX backend (RTF = wall/audio).
# NVIDIA: `just features=native,gpu,cuda,vulkan bench-kittentts-backends`
bench-kittentts-backends *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    feats="{{features}}"
    if [[ -z "$feats" ]]; then
      case "$(uname -s)" in
        Darwin) feats="native,apple-silicon" ;;
        *)      feats="native,gpu,cuda,vulkan" ;;
      esac
    fi
    cargo test -p rlx-kittentts --features "$feats" --release \
      --test native_backend_bench -- --nocapture --test-threads=1 {{ARGS}}

# Production vs legacy native RAM/timing (macOS: `/usr/bin/time -l` peak RSS)
bench-kittentts-native-alloc PHRASE="hello":
    KITTEN_RLX_SKIP_FUSION=1 KITTEN_RLX_PREFER_METAL=0 ./scripts/bench_kitten_native_alloc.sh {{PHRASE}}

# Native weights-only vs RLX bundle
test-kittentts-native-weights-parity *ARGS:
    cargo test -p rlx-kittentts --features native --release --test native_weights_parity -- {{ARGS}}

# Fetch (if needed) + unit tests + native E2E synthesis
test-kittentts-e2e *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f .cache/kittentts-mini-0.8/config.json ]]; then
      just fetch-kittentts
    fi
    cargo test -p rlx-kittentts --features "hf-download" --release -- {{ARGS}}
    just kittentts-demo
    just kittentts-long-demo
    python3 -c "import struct,sys; p='/tmp/kittentts_long_demo.wav'; f=open(p,'rb'); f.read(44); d=f.read(); n=len(d)//2; peak=max(abs(struct.unpack('<h',d[i:i+2])[0])/32768) for i in range(0,len(d)-1,2)); assert n>=80000 and peak>=1e-3, f'long demo failed: {n} samples peak={peak:.2e}'; print(f'long demo ok: {n} samples peak={peak:.3f}')"
    just kittentts-text-demo
    if [[ -f crates/kitten_tts_mini_rlx/weights/model.safetensors || -f crates/kitten_tts_mini_rlx/weights/rlx_bundle/graph.json ]]; then
      just test-kittentts-native-weights-parity
      just test-kittentts-native {{ARGS}}
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

# Native (Rust/RLX) Voxtral frontend parity vs a pre-dumped HF reference.
# Skips cleanly if the reference JSON / wav are absent (no Python needed at test time).
test-voxtral-parity:
    RLX_VOXTRAL_REF=${RLX_VOXTRAL_REF:-$PWD/.cache/voxtral_ref_jfk.json} \
    RLX_VOXTRAL_WAV=${RLX_VOXTRAL_WAV:-$PWD/.cache/whisper-bench/jfk_16k.wav} \
    cargo test -p rlx-models --test voxtral_hf_parity --release -- --nocapture --test-threads 1

ocr *ARGS:
    just run-bin rlx-ocr rlx-ocr {{ARGS}}

# PP-OCRv6 tiny/small — download ONNX, rewrite, export safetensors for native RLX.
fetch-ppocrv6-tiny:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - <<'PY'
    from huggingface_hub import hf_hub_download, list_repo_files
    import os, subprocess, sys
    from pathlib import Path
    base = Path(".cache/ppocrv6/tiny")
    for repo, sub in [
        ("PaddlePaddle/PP-OCRv6_tiny_det_onnx", "det"),
        ("PaddlePaddle/PP-OCRv6_tiny_rec_onnx", "rec"),
    ]:
        dest = base / sub
        dest.mkdir(parents=True, exist_ok=True)
        for f in list_repo_files(repo):
            if f.endswith((".onnx", ".yml", ".json")):
                hf_hub_download(repo_id=repo, filename=f, local_dir=str(dest))
        subprocess.check_call([
            sys.executable, "scripts/export_ppocrv6_onnx_weights.py",
            str(dest / "inference.onnx"), str(dest),
            "--stem", f"ppocrv6_tiny_{sub}",
        ])
    # copy bundled dict
    import shutil
    shutil.copy("crates/rlx-ppocrv6/assets/dicts/tiny_keys.txt", base / "rec" / "keys.txt")
    print("ready", base)
    PY

fetch-ppocrv6-small:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - <<'PY'
    from huggingface_hub import hf_hub_download, list_repo_files
    import os, subprocess, sys, shutil
    from pathlib import Path
    base = Path(".cache/ppocrv6/small")
    for repo, sub in [
        ("PaddlePaddle/PP-OCRv6_small_det_onnx", "det"),
        ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "rec"),
    ]:
        dest = base / sub
        dest.mkdir(parents=True, exist_ok=True)
        for f in list_repo_files(repo):
            if f.endswith((".onnx", ".yml", ".json")):
                hf_hub_download(repo_id=repo, filename=f, local_dir=str(dest))
        subprocess.check_call([
            sys.executable, "scripts/export_ppocrv6_onnx_weights.py",
            str(dest / "inference.onnx"), str(dest),
            "--stem", f"ppocrv6_small_{sub}",
        ])
    shutil.copy("crates/rlx-ppocrv6/assets/dicts/small_keys.txt", base / "rec" / "keys.txt")
    print("ready", base)
    PY

ppocrv6 *ARGS:
    just run-bin rlx-ppocrv6 rlx-ppocrv6 {{ARGS}}

test-ppocrv6-backends *ARGS:
    cargo test -p rlx-ppocrv6 --test ppocrv6_backend_quick_check --release {{ARGS}}

asr *ARGS:
    just run-bin rlx-asr rlx-asr {{ARGS}}

fetch-rlx-asr:
    mkdir -p weights/asr
    .venv-hf/bin/hf download eugenehp/rlx-asr model.rlxp --local-dir weights/asr
    @echo "RLX ASR in weights/asr (model.rlxp)"

# Materialize pack-only weights/asr (prune sidecars; prefer model.rlxp)
asr-weights-sync *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    ASR="${RLX_ASR_DIR:-$PWD/weights/asr}"
    SRC="${RLX_ASR_PACK_SRC:-$PWD/.cache/asr}"
    mkdir -p "$ASR"
    prune() {
      local d="$1"
      find "$d" -mindepth 1 -maxdepth 1 ! -name 'model.gguf' ! -name 'manifest.json' -exec rm -rf {} +
      find "$d" -name '.DS_Store' -delete 2>/dev/null || true
    }
    prune "$ASR"
    if [[ -d "$SRC" ]] || [[ -f "$ASR/model.gguf" ]]; then
      cargo run -p rlx-asr --release --bin rlx-asr-pack-gguf -- --dir "$ASR" --out "$ASR/model.gguf" {{ARGS}} || true
    fi
    prune "$ASR"
    RLX_ASR_DIR="$ASR" RLX_ASR_PACK_SRC="$SRC" python3 scripts/asr_weights_manifest.py "$ASR"

asr-pack-gguf *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    ASR="${RLX_ASR_DIR:-$PWD/weights/asr}"
    export RLX_ASR_PACK_SRC="${RLX_ASR_PACK_SRC:-$PWD/.cache/asr}"
    cargo run -p rlx-asr --release --bin rlx-asr-pack-gguf -- --dir "$ASR" --out "$ASR/model.gguf" {{ARGS}}

asr-pack-rlxp *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    ASR="${RLX_ASR_DIR:-$PWD/weights/asr}"
    export RLX_ASR_PACK_SRC="${RLX_ASR_PACK_SRC:-$PWD/.cache/asr}"
    cargo run -p rlx-asr --release --bin rlx-asr-pack-gguf -- --rlxp --dir "$ASR" --out "$ASR/model.rlxp" {{ARGS}}

asr-check *ARGS:
    cargo test -p rlx-asr --release {{ARGS}}

test-asr *ARGS:
    cargo test -p rlx-asr --release {{ARGS}}

# Folded native batch E2E: mel → body residual R → CTC beam
# Example: just asr-e2e-native -- --mode folded --wav .cache/conformer-ctc/sample.wav
asr-e2e-native *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    PY="${PWD}/.venv-asr312/bin/python"
    [[ -x "$PY" ]] || PY="$(command -v python3)"
    ASR="${RLX_ASR_DIR:-$PWD/weights/asr}"
    OUT="$ASR/e2e_native"
    ARGS=({{ARGS}})
    if [[ ${#ARGS[@]} -gt 0 && "${ARGS[0]}" == "--" ]]; then ARGS=("${ARGS[@]:1}"); fi
    if [[ ! -f "$ASR/model.gguf" ]]; then
      echo "soft skip: missing $ASR/model.gguf (run: just asr-weights-sync && just asr-pack-gguf)" >&2
      exit 0
    fi
    if [[ ${#ARGS[@]} -eq 0 ]] || [[ "${ARGS[0]}" == --* ]]; then
      WAVS=()
      for w in \
        "$PWD/.cache/conformer-ctc/sample.wav" \
        "$PWD/.cache/rlx-tts/fixtures/hello_native.wav"
      do
        [[ -f "$w" ]] && WAVS+=("$w")
      done
      if [[ ${#WAVS[@]} -eq 0 ]]; then
        echo "soft skip: no default wavs found" >&2
        exit 0
      fi
      "$PY" crates/rlx-asr/tools/e2e_native_whole.py \
        --out "$OUT" \
        --mode folded \
        --wav "${WAVS[@]}" \
        ${ARGS[@]+"${ARGS[@]}"}
    else
      WAVS=(); FLAGS=()
      for a in "${ARGS[@]}"; do
        if [[ "$a" == --* ]]; then FLAGS+=("$a"); else WAVS+=("$a"); fi
      done
      has_mode=0
      for f in ${FLAGS[@]+"${FLAGS[@]}"}; do
        [[ "$f" == --mode ]] && has_mode=1
      done
      if [[ $has_mode -eq 0 ]]; then FLAGS+=(--mode folded); fi
      "$PY" crates/rlx-asr/tools/e2e_native_whole.py \
        --out "$OUT" \
        --wav "${WAVS[@]}" \
        ${FLAGS[@]+"${FLAGS[@]}"}
    fi

sam1 *ARGS:
    just run-bin rlx-sam rlx-sam1 {{ARGS}}

sam2 *ARGS:
    just run-bin rlx-sam2 rlx-sam2 {{ARGS}}

sam3 *ARGS:
    just run-bin rlx-sam3 rlx-sam3 {{ARGS}}

flux2 *ARGS:
    just run-bin rlx-flux2 rlx-flux2 {{ARGS}}

trellis2 *ARGS:
    just run-bin rlx-trellis2 rlx-trellis2 {{ARGS}}

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

tinyllama-metal *ARGS:
    just features=metal run-tinyllama {{ARGS}}

tinyllama-all-backends *ARGS:
    just features=all-backends run-tinyllama {{ARGS}}

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

# Qwen3-ASR-0.6B safetensors + tokenizer for the all-Qwen voice chat example.
fetch-qwen3-asr REPO="Qwen/Qwen3-ASR-0.6B":
    huggingface-cli download {{REPO}} --local-dir .cache/qwen3-asr/Qwen3-ASR-0.6B

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

# Gemma 3 270M chat → Inflect-Nano TTS (needs fetch-gemma3-270m + Inflect bundle).
gemma-inflect-speak *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="{{justfile_directory()}}"
    export RLX_GEMMA3_GGUF="${RLX_GEMMA3_GGUF:-{{real_weights_dir}}/gemma-3-270m.gguf}"
    export RLX_INFLECT_NANO_DATA="${RLX_INFLECT_NANO_DATA:-$ROOT/weights/inflect-nano-rlx}"
    just fetch-gemma3-270m
    test -f "$RLX_INFLECT_NANO_DATA/config.json" || {
        echo "missing Inflect bundle at $RLX_INFLECT_NANO_DATA — see crates/rlx-inflect-nano/README.md"
        exit 1
    }
    VECLIB_MAXIMUM_THREADS=1 cargo run --release -p rlx-gemma-inflect-nano --features apple-silicon \
      --example speak -- --device metal --tts-device auto {{ARGS}}

# Gemma 3 270M Q4_K_M (~250 MB) + HF tokenizer.json.
fetch-gemma3-270m:
    mkdir -p {{real_weights_dir}}
    test -s {{real_weights_dir}}/gemma-3-270m.gguf || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/gemma-3-270m.gguf \
            'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf'
    test -s {{real_weights_dir}}/gemma-3-270m.tokenizer.json || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/gemma-3-270m.tokenizer.json \
            'https://huggingface.co/unsloth/gemma-3-270m-it/resolve/main/tokenizer.json'

test-gemma3-real: fetch-gemma3-270m
    RLX_GEMMA3_GGUF={{real_weights_dir}}/gemma-3-270m.gguf \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_gemma3 --features gemma,runner -- --nocapture

test-gemma3-real-inference: fetch-gemma3-270m
    RLX_GEMMA3_GGUF={{real_weights_dir}}/gemma-3-270m.gguf \
    RLX_GEMMA3_RUN_INFERENCE=1 \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_gemma3 forward_inference_real_weights --features gemma,runner -- --nocapture

# Phi-3-mini-4k Q4_K_M (~2.3 GB).
fetch-phi3-mini:
    mkdir -p {{real_weights_dir}}
    test -s {{real_weights_dir}}/Phi-3-mini-4k-instruct.gguf || \
        curl -L --create-dirs -C - -o {{real_weights_dir}}/Phi-3-mini-4k-instruct.gguf \
            'https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf'

test-phi3-real: fetch-phi3-mini
    RLX_PHI3_GGUF={{real_weights_dir}}/Phi-3-mini-4k-instruct.gguf \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_phi3 --features phi,llama32,runner -- --nocapture

test-phi3-real-inference: fetch-phi3-mini
    RLX_PHI3_GGUF={{real_weights_dir}}/Phi-3-mini-4k-instruct.gguf \
    RLX_PHI3_RUN_INFERENCE=1 \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_phi3 forward_inference_real_weights --features phi,llama32,runner -- --nocapture

# MiniCPM5-1B via Hugging Face Hub (~2.1 GB safetensors + tokenizer).
fetch-minicpm5:
    MINICPM5_MODEL_DIR={{real_weights_dir}}/MiniCPM5-1B \
        cargo run -p rlx-models --example minicpm5_download --features hf-download --release

# Nanbeige4.2-3B (~8 GB BF16 safetensors). Override dest with NANBEIGE_MODEL_DIR=.
fetch-nanbeige:
    NANBEIGE_MODEL_DIR={{real_weights_dir}}/Nanbeige4.2-3B \
        cargo run -p rlx-nanbeige --example fetch_nanbeige --features hf-download --release

fetch-tinyllama:
    TINYLLAMA_MODEL_DIR={{real_weights_dir}}/TinyLlama-1.1B-Chat-v1.0 \
        cargo run -p rlx-models --example tinyllama_download --features hf-download,tinyllama --release

fetch-tinyllama-gguf QUANT="Q4_K_M":
    RLX_TINYLLAMA_GGUF_DIR={{real_weights_dir}}/TinyLlama-1.1B-GGUF \
        cargo run -p rlx-models --example tinyllama_gguf_download --features hf-download,tinyllama --release -- {{QUANT}}

fetch-tinyllama-gguf-all:
    just fetch-tinyllama-gguf all

test-tinyllama-real: fetch-tinyllama
    RLX_TINYLLAMA_WEIGHTS={{real_weights_dir}}/TinyLlama-1.1B-Chat-v1.0/model-00001-of-00003.safetensors \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_tinyllama --features tinyllama,llama32 -- --nocapture

test-tinyllama-real-inference: fetch-tinyllama
    RLX_TINYLLAMA_RUN_INFERENCE=1 \
    RLX_TINYLLAMA_WEIGHTS={{real_weights_dir}}/TinyLlama-1.1B-Chat-v1.0/model-00001-of-00003.safetensors \
        cargo test -p rlx-models {{profile}} \
            --test real_weights_tinyllama forward_inference_tinyllama_1_1b --features tinyllama,llama32 -- --nocapture

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

# --- rlx-fft (learned FFT + Welch peaks) ---

[private]
run-rlx-fft *ARGS:
    cargo run -p rlx-fft --bin rlx-fft {{profile}} {{feature_args}} -- {{ARGS}}

# IO-aware Welch peaks picker bench. macOS: `just features=apple-silicon bench-welch-peaks -- --device metal`
bench-welch-peaks *ARGS:
    just run-rlx-fft bench-welch-peaks {{ARGS}}

# Fusion phase bench (dev-only module). Override backends: `just features=dev,gpu bench-fusion-phases -- …`
bench-fusion-phases *ARGS:
    cargo run -p rlx-fft --bin rlx-fft {{profile}} --features dev,apple-silicon -- bench-fusion-phases {{ARGS}}

test-rlx-fft-welch-peaks *ARGS:
    cargo test -p rlx-fft welch_peaks {{profile}} {{feature_args}} {{ARGS}}

test-rlx-fft-fusion-gate *ARGS:
    cargo test -p rlx-fft --lib fusion_gate_batch_matrix io_gate --features apple-silicon {{profile}} {{ARGS}}

# --- cross-backend model harness (scripts/matrix) ---

# Run the harness on the remote host (set RLX_CUDA_HOST): sync both trees -> build+run per backend -> pull report back.
#   just matrix-remote                     # Tier-1, all host backends
#   just matrix-remote ONLY=qwen3-0.6b     # one model
#   just matrix-remote TIER=all ALL=1      # everything incl. Tier-2/NC/gated
#   just matrix-remote BACKENDS=cpu,cuda   # subset of backends
matrix-remote TIER="1" ONLY="" BACKENDS="" ALL="0":
    bash scripts/matrix/remote_run.sh "{{TIER}}" "{{ONLY}}" "{{BACKENDS}}" "{{ALL}}"

# Run the harness on THIS host (auto-detects local backends; no ssh/sync).
matrix TIER="1" ONLY="" BACKENDS="" ALL="0":
    TIER="{{TIER}}" ONLY="{{ONLY}}" BACKENDS="{{BACKENDS}}" ALL="{{ALL}}" python3 scripts/matrix/run_matrix.py

# Just sync the two working trees to the remote host (no run).
matrix-sync:
    bash scripts/matrix/sync_to_remote.sh

# --- unified TTS bench (crates/rlx-tts-bench) ---

# Full matrix → HTML under --out-dir. Isolation/resume are on by default for `run`.
#   just tts-bench run --models all --devices auto --phrases short,long --whisper --spectral --noise --clone
#   just tts-bench run --models all --devices auto --resume   # continue after kill/abort
tts-bench *ARGS:
    cargo run -p rlx-tts-bench --bin rlx-tts-bench {{profile}} {{feature_args}} -- {{ARGS}}

# Optional Piper reference for spectral: just tts-stress -- --ref-model piper --spectral
# Preflight: just tts-stress -- --n 20
# Full + resume: just tts-stress -- --n 1000 --resume
tts-stress *ARGS:
    cargo run -p rlx-tts-bench --bin rlx-tts-bench {{profile}} --features "rlx-tts,matrix-onnx" -- \
        stress --target rlx-tts --device cpu --whisper --write-corpus \
        --out-dir /tmp/rlx-tts-stress {{ARGS}}

# Probe every RLX device; product synth runs on host CPU (others skip until Device kernels).
#   just tts-backends
#   RLX_ITERS=5 just tts-backends
tts-backends *ARGS:
    cargo run -p rlx-tts --release --example backend_matrix --features apple-silicon -- {{ARGS}}

# Unified matrix: rlx-tts × available devices + Whisper (non-CPU cells skip).
tts-backends-whisper *ARGS:
    cargo run -p rlx-tts-bench --bin rlx-tts-bench {{profile}} --features "rlx-tts,matrix-onnx,apple-silicon" -- \
        run -m rlx-tts -d auto --phrases short,long --whisper --noise --no-isolate \
        --out-dir /tmp/rlx-tts-backends {{ARGS}}

# CPU + short phrase only (fast local preflight; fake adapter if no weights).
tts-bench-quick *ARGS:
    cargo run -p rlx-tts-bench --bin rlx-tts-bench {{profile}} --features matrix-onnx -- \
        run -m fake -d cpu --phrases short --noise --out-dir /tmp/tts-bench-quick {{ARGS}}

tts-bench-metal *ARGS:
    cargo run -p rlx-tts-bench --bin rlx-tts-bench {{profile}} --features "all-models,apple-silicon" -- {{ARGS}}

tts-bench-list:
    cargo run -p rlx-tts-bench --bin rlx-tts-bench {{profile}} {{feature_args}} -- list

# ── Local rlx development (sibling ../rlx checkout) ──────────────────────────
# Toggle the gitignored `.cargo/config.toml` [patch.crates-io] override that
# points every rlx-* crate at ../rlx. The committed Cargo.toml never carries a
# patch — this is the one, one-command way to build against local rlx edits.
#
#   just link-local   # edit rlx in ../rlx and build/test rlx-models against it
#   just unlink       # revert to the published =0.2.x crates (CI / publish state)
#   just relink       # refresh after editing config.toml.example

# Point rlx-* at the local sibling ../rlx working tree.
link-local:
    cp .cargo/config.toml.example .cargo/config.toml
    @echo "linked: rlx-* -> ../rlx via .cargo/config.toml (gitignored). 'just unlink' reverts."

# Revert to the published crates.io rlx-* (same as CI / `cargo publish`).
unlink:
    rm -f .cargo/config.toml
    @echo "unlinked: rlx-* now resolve to the published =0.2.x crates."

# Re-link from a freshly edited example (unlink, then link).
relink: unlink link-local
