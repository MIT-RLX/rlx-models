#!/usr/bin/env bash
# Reference environment for MiniCPM-SALA's InfLLM-v2 sparse attention: a uv
# venv with PyTorch, transformers 4.56 (the remote code predates 5.x) and
# OpenBMB's `infllm_v2` CUDA kernels built from source.
#
# Nothing about the machine is assumed. The script finds every CUDA toolkit
# installed ($CUDA_HOME, /usr/local/cuda*, /opt/cuda*, nvcc on PATH), and
# takes the newest one that
#   1. compiles a test kernel for this GPU's compute capability (from
#      nvidia-smi), with the default host compiler or, failing that, the
#      newest installed gcc-N it accepts (passed to nvcc as -ccbin), and
#   2. has a PyTorch wheel index of the same CUDA major (newest minor not
#      above the toolkit's) — torch refuses to build extensions across majors.
#
# Usage: scripts/infllm_reference_env.sh [VENV_DIR] [SRC_DIR]
#   defaults: .venv-infllm and infllmv2_cuda_impl under the current directory.
# Settings: CUDA_ARCH (e.g. 8.6) overrides the detected compute capability;
#   MAX_JOBS bounds the parallel nvcc jobs; by default it is half the cores,
#   capped at one job per 6 GB of available memory (each FlashAttention-style
#   kernel takes several GB to compile, and an unbounded build starves the
#   machine).
set -euo pipefail

VENV="${1:-$PWD/.venv-infllm}"
SRC="${2:-$PWD/infllmv2_cuda_impl}"
PY="${PYTHON_VERSION:-3.12}"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

# --- GPU compute capability --------------------------------------------------
arch="${CUDA_ARCH:-}"
if [[ -z "$arch" ]]; then
  arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
fi
[[ -n "$arch" ]] || { echo "no GPU compute capability (set CUDA_ARCH)" >&2; exit 1; }
sm="${arch/./}"
echo "==> GPU compute capability $arch (sm_$sm)"

# --- candidate toolkits, newest first ----------------------------------------
declare -A seen=()
cands=()
add() {
  local d="$1"
  [[ -x "$d/bin/nvcc" ]] || return 0
  d="$(cd "$d" && pwd -P)"
  [[ -n "${seen[$d]:-}" ]] && return 0
  seen[$d]=1
  cands+=("$d")
}
[[ -n "${CUDA_HOME:-}" ]] && add "$CUDA_HOME"
for d in /usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*; do add "$d"; done
if command -v nvcc >/dev/null; then add "$(dirname "$(dirname "$(command -v nvcc)")")"; fi
[[ ${#cands[@]} -gt 0 ]] || { echo "no CUDA toolkit found" >&2; exit 1; }

nvcc_version() { "$1/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p'; }
mapfile -t cands < <(for d in "${cands[@]}"; do echo "$(nvcc_version "$d") $d"; done | sort -t' ' -k1,1Vr | cut -d' ' -f2-)

# Host compilers to try with a toolkit: the default, then gcc-N newest first.
hosts=("")
for g in $(ls /usr/bin/gcc-[0-9]* 2>/dev/null | sed 's#.*/gcc-##' | sort -nr); do hosts+=("/usr/bin/gcc-$g"); done

probe_dir="$(mktemp -d)"
trap 'rm -rf "$probe_dir"' EXIT
cat > "$probe_dir/k.cu" <<'EOF'
#include <cuda_fp16.h>
__global__ void k(float *x) { x[threadIdx.x] = __half2float(__float2half(1.0f)); }
int main() { return 0; }
EOF

# Torch wheel indexes, newest first.
TORCH_INDEXES=(cu130 cu129 cu128 cu126 cu124 cu121 cu118)
pick_index() {  # toolkit version -> newest index with the same major, minor <= toolkit
  local major minor
  major="${1%%.*}"; minor="${1##*.}"
  for ix in "${TORCH_INDEXES[@]}"; do
    local v im in
    v="${ix#cu}"; im="${v:0:${#v}-1}"; in="${v: -1}"
    if [[ "$im" == "$major" && "$in" -le "$minor" ]]; then
      if curl -sfI "https://download.pytorch.org/whl/$ix/torch/" >/dev/null; then
        echo "$ix"; return 0
      fi
    fi
  done
  return 1
}

chosen="" ccbin="" index=""
for d in "${cands[@]}"; do
  ver="$(nvcc_version "$d")"
  if ! "$d/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_$sm"; then
    echo "    skip CUDA $ver ($d): no compute_$sm"; continue
  fi
  ix="$(pick_index "$ver" || true)"
  [[ -n "$ix" ]] || { echo "    skip CUDA $ver ($d): no PyTorch wheel for this major"; continue; }
  for h in "${hosts[@]}"; do
    flags=(-arch="sm_$sm" -o "$probe_dir/k" "$probe_dir/k.cu")
    [[ -n "$h" ]] && flags=(-ccbin "$h" "${flags[@]}")
    if "$d/bin/nvcc" "${flags[@]}" >/dev/null 2>"$probe_dir/err"; then
      chosen="$d"; ccbin="$h"; index="$ix"; break 2
    fi
  done
  echo "    skip CUDA $ver ($d): no host compiler it accepts ($(head -c 160 "$probe_dir/err" | tr '\n' ' '))"
done
[[ -n "$chosen" ]] || { echo "no usable CUDA toolkit / host compiler / PyTorch wheel combination" >&2; exit 1; }
echo "==> CUDA $(nvcc_version "$chosen") at $chosen, host compiler ${ccbin:-default}, torch index $index"

export CUDA_HOME="$chosen"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ -n "$ccbin" ]]; then
  export NVCC_PREPEND_FLAGS="-ccbin $ccbin"
  export CC="$ccbin" CXX="${ccbin/gcc/g++}"
fi
export TORCH_CUDA_ARCH_LIST="$arch"
export FLASH_ATTN_CUDA_ARCHS="$sm"
if [[ -z "${MAX_JOBS:-}" ]]; then
  avail_gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
  by_mem=$(( avail_gb / 6 ))
  by_cpu=$(( ($(nproc) + 1) / 2 ))
  MAX_JOBS=$(( by_mem < by_cpu ? by_mem : by_cpu ))
  (( MAX_JOBS >= 1 )) || MAX_JOBS=1
fi
export MAX_JOBS
echo "==> MAX_JOBS=$MAX_JOBS"

# --- venv ---------------------------------------------------------------------
[[ -d "$VENV" ]] || uv venv -p "$PY" "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
uv pip install torch --index-url "https://download.pytorch.org/whl/$index"
uv pip install "transformers==4.56.2" einops ninja packaging numpy safetensors tokenizers \
  sentencepiece psutil setuptools wheel
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
EOF

# --- infllm_v2 ------------------------------------------------------------------
[[ -d "$SRC" ]] || git clone --recursive https://github.com/OpenBMB/infllmv2_cuda_impl.git "$SRC"
git -C "$SRC" log -1 --format='infllmv2_cuda_impl %H'
# The vendored CUTLASS revision's cuda_host_adapter.hpp lost the `#` of one
# `#if` and the underscores of `__CUDACC_VER_MAJOR__` / `__CUDACC_VER_MINOR__`,
# so on every toolkit the version checks read undefined macros (0) and the
# CUDA >= 12.5 driver wrappers are never declared. Restore both if present.
adapter="$SRC/csrc/cutlass/include/cutlass/cuda_host_adapter.hpp"
if [[ -f "$adapter" ]]; then
  if grep -q '^if ((CUDACC_VER_MAJOR' "$adapter"; then
    sed -i 's/^if ((CUDACC_VER_MAJOR/#if ((CUDACC_VER_MAJOR/' "$adapter"
    echo "==> repaired a missing '#if' in $adapter"
  fi
  if grep -qP '(?<![_A-Za-z])CUDACC_VER_(MAJOR|MINOR)(?![_A-Za-z])' "$adapter"; then
    perl -pi -e 's/(?<![_A-Za-z])CUDACC_VER_(MAJOR|MINOR)(?![_A-Za-z])/__CUDACC_VER_$1__/g' "$adapter"
    echo "==> repaired the __CUDACC_VER_*__ macro names in $adapter"
  fi
fi
# The repo builds with -std=c++17; newer torch headers need C++20. Probe the
# installed torch at C++17 and raise the standard only if it refuses.
if ! python - <<'PROBE'
import os, subprocess, sys, tempfile
from torch.utils.cpp_extension import include_paths
src = os.path.join(tempfile.mkdtemp(), "p.cpp")
open(src, "w").write("#include <torch/extension.h>\nint main() { return 0; }\n")
inc = sum((["-I", p] for p in include_paths()), [])
r = subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-fsyntax-only", *inc, src],
                   capture_output=True)
sys.exit(r.returncode)
PROBE
then
  sed -i 's/-std=c++17/-std=c++20/g' "$SRC/setup.py"
  echo "==> torch headers need C++20: building with -std=c++20"
fi
uv pip install --no-build-isolation -e "$SRC"
python -c 'import infllm_v2; print("infllm_v2 ok")'
