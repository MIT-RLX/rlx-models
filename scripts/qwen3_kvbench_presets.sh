#!/usr/bin/env bash
set -euo pipefail

# Run qwen_quant_bench presets and print a compact speed+quality comparison.
#
# Usage:
#   scripts/qwen3_kvbench_presets.sh
#
# Optional env:
#   RLX_QWEN_DIR   (default: /Users/Shared/weights/qwen3-0.6b)
#   DEVICE         (default: metal)
#   FEATURES       (default: quant-opt,metal)
#   EXAMPLE        (default: qwen_quant_bench)
#   PACKAGE        (default: rlx-qwen3)
#   OUT_CSV        (default: /tmp/qwen3_kvbench_presets.csv)
#   RUN_PBENCH     (default: 1; set 0 to skip quality-stress pbench)

RLX_QWEN_DIR="${RLX_QWEN_DIR:-/Users/Shared/weights/qwen3-0.6b}"
DEVICE="${DEVICE:-metal}"
FEATURES="${FEATURES:-quant-opt,metal}"
EXAMPLE="${EXAMPLE:-qwen_quant_bench}"
PACKAGE="${PACKAGE:-rlx-qwen3}"
OUT_CSV="${OUT_CSV:-/tmp/qwen3_kvbench_presets.csv}"
RUN_PBENCH="${RUN_PBENCH:-1}"

if [[ ! -f "${RLX_QWEN_DIR}/config.json" ]]; then
  echo "error: RLX_QWEN_DIR does not contain config.json: ${RLX_QWEN_DIR}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/qwen3-kvbench-presets.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

run_kvbench_preset() {
  local name="$1"
  shift
  local log_file="${TMP_DIR}/${name}.log"

  echo "[run] ${name}" >&2
  env RLX_QWEN_DIR="${RLX_QWEN_DIR}" "$@" \
    cargo run -p "${PACKAGE}" --example "${EXAMPLE}" --release --features "${FEATURES}" -- kvbench "${DEVICE}" \
    2>&1 | tee "${log_file}" >/dev/null

  local line
  line="$(grep 'CSVMEAN,' "${log_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo "error: did not find CSVMEAN line in ${name} run" >&2
    echo "log: ${log_file}" >&2
    exit 1
  fi

  local dev decode cold warm
  dev="$(awk -F',' '{print $2}' <<<"${line}")"
  decode="$(awk -F',' '{print $3}' <<<"${line}")"
  cold="$(awk -F',' '{print $4}' <<<"${line}")"
  warm="$(awk -F',' '{print $5}' <<<"${line}")"

  # profile,bench,device,forward_ms,decode_tps,prefill_cold_tps,prefill_warm_tps,cos,top1_pct,kl
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${name}" "kvbench" "${dev}" "" "${decode}" "${cold}" "${warm}" "" "" ""
}

run_pbench_quality() {
  local name="$1"
  shift
  local log_file="${TMP_DIR}/${name}.log"

  echo "[run] ${name}" >&2
  env RLX_QWEN_DIR="${RLX_QWEN_DIR}" "$@" \
    cargo run -p "${PACKAGE}" --example "${EXAMPLE}" --release --features "${FEATURES}" -- pbench "${DEVICE}" \
    2>&1 | tee "${log_file}" >/dev/null

  local line
  line="$(grep 'CSVMEAN,' "${log_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo "error: did not find CSVMEAN line in ${name} run" >&2
    echo "log: ${log_file}" >&2
    exit 1
  fi

  local dev forward decode prefill cos top1 kl
  dev="$(awk -F',' '{print $2}' <<<"${line}")"
  forward="$(awk -F',' '{print $3}' <<<"${line}")"
  decode="$(awk -F',' '{print $4}' <<<"${line}")"
  prefill="$(awk -F',' '{print $5}' <<<"${line}")"
  cos="$(awk -F',' '{print $6}' <<<"${line}")"
  top1="$(awk -F',' '{print $7}' <<<"${line}")"
  kl="$(awk -F',' '{print $8}' <<<"${line}")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${name}" "pbench" "${dev}" "${forward}" "${decode}" "${prefill}" "" "${cos}" "${top1}" "${kl}"
}

baseline="$(run_kvbench_preset safe_quality)"
tuned="$(run_kvbench_preset max_speed_decode RLX_QWEN3_F16_WEIGHTS=1 RLX_QWEN3_BAKE_WEIGHTS=1 RLX_QWEN3_GQA_NATIVE=1)"

pbench_row=""
if [[ "${RUN_PBENCH}" != "0" ]]; then
  pbench_row="$(run_pbench_quality quality_stress)"
fi

{
  echo 'profile,bench,device,forward_ms,decode_tps,prefill_cold_tps,prefill_warm_tps,cos,top1_pct,kl'
  echo "${baseline}"
  echo "${tuned}"
  if [[ -n "${pbench_row}" ]]; then
    echo "${pbench_row}"
  fi
} >"${OUT_CSV}"

# Pretty table to stdout.
printf '\n%-16s %-8s %-7s %10s %10s %12s %12s %8s %9s %8s\n' \
  'profile' 'bench' 'device' 'forward_ms' 'decode' 'prefill_cold' 'prefill_warm' 'cos' 'top1_%' 'kl'
printf '%-16s %-8s %-7s %10s %10s %12s %12s %8s %9s %8s\n' \
  '----------------' '--------' '-------' '----------' '----------' '------------' '------------' '--------' '---------' '--------'
while IFS=',' read -r profile bench device forward decode cold warm cos top1 kl; do
  if [[ "${profile}" == "profile" ]]; then
    continue
  fi
  printf '%-16s %-8s %-7s %10s %10s %12s %12s %8s %9s %8s\n' \
    "${profile}" "${bench}" "${device}" "${forward}" "${decode}" "${cold}" "${warm}" "${cos}" "${top1}" "${kl}"
done <"${OUT_CSV}"

printf '\nSaved CSV: %s\n' "${OUT_CSV}"