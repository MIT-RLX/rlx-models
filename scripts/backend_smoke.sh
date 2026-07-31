#!/usr/bin/env bash
# Cross-backend smoke sweep for the newly-portable model crates.
#
# Runs each crate's device-parametrized finite-logits / parity smoke test on
# every requested backend. The same command works on the Mac (cpu metal mlx gpu
# coreml) and on the CUDA host (cpu gpu cuda vulkan) — pass the device list.
#
#   scripts/backend_smoke.sh                      # default: cpu
#   scripts/backend_smoke.sh cpu metal mlx gpu coreml
#   scripts/backend_smoke.sh cpu gpu cuda vulkan  # on the NVIDIA box
#
# Device string == cargo feature (gpu==wgpu, coreml==ANE). CPU uses no backend
# feature (except rlx-vibevoice-asr, whose cpu path is a feature).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEVICES=("${@:-cpu}")
[ $# -gt 0 ] && DEVICES=("$@")

# crate  :  extra cargo args (test target / feature quirks)
#   %DEV%   -> the backend feature (or empty on cpu)
#   %CPUONLY% used for crates whose cpu path needs a feature / no-default
declare -a CRATES=(
  "rlx-jamba|"
  "rlx-glm4moe|"
  "rlx-deepseek|"
  "rlx-llama4|"
  "rlx-mllama|"
  "rlx-mistral-vl|--test vision_flow_smoke"
  "rlx-vibevoice-asr|--test vae_encoder_smoke|cpufeat"
  "rlx-neutrino|--test fv5_dequant_smoke|nodefault"
  # Kimi-K3: MLA/MoE/vision run on every backend; the KDA tests use the
  # per-channel GatedDeltaNet, which now runs on ALL backends — native kernels
  # on cpu/wgpu/metal/cuda, CPU host-fallback on rocm, unfuse decomposition on
  # vulkan, and time-loop decomposition on mlx/coreml.
  "rlx-kimi-k3|--test mla_smoke --test moe_smoke --test vision_smoke --lib"
  "rlx-kimi-k3|--test gated_delta_net_pc --test kda_smoke --test flow_smoke|kdaonly"
)

fail=0
for dev in "${DEVICES[@]}"; do
  echo "==================== backend: $dev ===================="
  for entry in "${CRATES[@]}"; do
    IFS='|' read -r pkg targ quirk <<<"$entry"
    # KDA per-channel GatedDeltaNet runs on every backend now: native kernels
    # (cpu/wgpu/metal/cuda), rocm host-fallback, vulkan unfuse, mlx/coreml
    # time-loop decomposition. The (kda) label just isolates it in the report.
    label="$pkg"
    [ "$quirk" = "kdaonly" ] && label="$pkg (kda)"
    feats=""
    nodefault=""
    if [ "$dev" = "cpu" ]; then
      [ "$quirk" = "cpufeat" ] && feats="--features cpu"
      [ "$quirk" = "nodefault" ] && { nodefault="--no-default-features"; }
    else
      feats="--features $dev"
      [ "$quirk" = "nodefault" ] && nodefault="--no-default-features"
    fi
    printf '  %-22s ' "$label"
    logf="/tmp/bs_${label// /_}_${dev}.log"
    if RLX_TEST_DEVICE="$dev" cargo test -p "$pkg" $nodefault $feats $targ \
         -- --test-threads=1 >"$logf" 2>&1; then
      passed=$(grep -hE "test result: ok" "$logf" \
        | sed -E 's/.*ok\. ([0-9]+) passed.*/\1/' | awk '{s+=$1} END{print s+0}')
      echo "OK   ($passed passed)"
    else
      echo "FAIL (see $logf)"
      fail=1
    fi
  done

  # FV5/FV5B GPU dequant kernel parity (in the sibling ../rlx runtime repo).
  case "$dev" in
    metal)
      printf '  %-22s ' "rlx-metal fv5 parity"
      if (cd ../rlx && cargo test -p rlx-metal --test q8_q4_dequant_parity --test fv5_matmul_parity fv5 >/tmp/bs_metal_fv5.log 2>&1); then
        echo "OK"; else echo "FAIL (/tmp/bs_metal_fv5.log)"; fail=1; fi ;;
    gpu|vulkan)
      printf '  %-22s ' "rlx-wgpu fv5 parity"
      if (cd ../rlx && cargo test -p rlx-wgpu --test gguf_dequant_parity --test gguf_dequant_matmul_prefill_parity fv5 >/tmp/bs_wgpu_fv5.log 2>&1); then
        echo "OK"; else echo "FAIL (/tmp/bs_wgpu_fv5.log)"; fail=1; fi ;;
  esac
done

echo
[ $fail -eq 0 ] && echo "ALL GREEN" || echo "SOME FAILURES — inspect the /tmp/bs_*.log files"
exit $fail
