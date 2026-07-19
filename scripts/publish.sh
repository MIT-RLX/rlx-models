#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# scripts/publish.sh — workspace-wide publish driver for rlx-models.
#
# Walks the dep graph in tier order (leaves first), publishing one
# crate at a time with rate-limit-respecting delays AND active
# polling of the sparse index until each just-published version is
# resolvable. This combination is what keeps the script safe to
# leave running unattended:
#
#   * Hard rate-limit floor: a `MIN_INTERVAL` sleep (default 65 s,
#     above the documented "1 new crate per minute" + "1 version per
#     minute" crates.io throttle).
#   * Active readiness check: after every successful `cargo publish`,
#     poll `https://index.crates.io/<partition>/<crate>` until the
#     just-uploaded version appears. Until that happens, the next
#     crate's `cargo publish` would either 404 on dep resolution or
#     get stale metadata.
#   * Skip published: before each upload, check the sparse index for
#     the workspace version; if it is already there, skip (no
#     `cargo publish`, no rate-limit sleep). Use `--no-skip-published`
#     to force an upload attempt anyway.
#   * Registry HTTP errors (429, 408, 5xx, timeouts): parse cargo output as
#     it finishes, sleep until crates.io's "try again after <GMT>" when
#     present (+ pad), else status-specific backoff — then retry the same
#     crate automatically until it uploads. --max-retries 0 (default) never
#     gives up on retryable codes.
#   * Missing registry deps: poll the sparse index for the dependency,
#     then retry (no manual --start-crate for index lag).
#
# Crates.io documented limits (as of 2026-05):
#   * New crates:        1 per minute   (10 per 10 min)
#   * New versions:      1 per minute   (30 per 10 min)
#   * Sparse-index serve: typically <30 s, occasionally up to ~5 min
#
# Default flow:
#
#   1. Pre-flight: cargo fmt --check, cargo clippy -- -D warnings,
#      cargo test --workspace --release. Aborts if any fails.
#   2. Confirm prompt (skip with --yes).
#   3. Per tier, publish each crate sequentially. After each crate:
#        a. wait for the sparse index to report the new version
#           (poll interval `POLL_INTERVAL`, hard cap `POLL_TIMEOUT`),
#        b. sleep `MIN_INTERVAL` (rate-limit safety floor),
#      before issuing the next publish. After the last crate of a
#      tier the loop additionally sleeps `BETWEEN_DELAY` to let
#      downstream crates' dep resolution catch up.
#   4. Crates marked `publish = false` or in workspace.exclude (see SKIPPED)
#      are not published — cargo skips them; this script lists the rest.
#
# Prerequisite: publish upstream `rlx*` crates (crates.io 0.2.13) from the RLX repo
# before `rlx-models` path deps resolve on the registry.
#
# Publishable workspace crates in 7 tiers (tier 6 = facade `rlx-models` last).
# Notable ordering: `kitten_tts_mini_rlx` before `rlx-kittentts`; `rlx-whisper`
# before `rlx-kittentts` (dev-dep for roundtrip tests); `rlx-llama32` / `rlx-gemma`
# (tier 4) before `rlx-minicpm5` / `rlx-voxtral-tts-train` (tier 5–6);
# `rlx-guardrails` before `rlx-serve`; `rlx-kokoro` before `rlx-styletts2`;
# `rlx-luxtts` before `rlx-zipvoice`; `rlx-quant-calib` before `rlx-tune`;
# `rlx-distributed` before `rlx-qwen3`.
# Workspace / upstream pin: 0.2.13 — bump `[workspace.package].version` and
# `[workspace.dependencies]` path `version =` fields before publishing.
# Bump `[workspace.package].version`, per-crate `[package].version` when needed
# (e.g. `rlx-models-core`), and `[workspace.dependencies]` pins before publishing.
#
# Usage:
#
#   scripts/publish.sh --dry-run                    # safe — no upload
#   scripts/publish.sh --list                       # print tier order, exit
#   scripts/publish.sh --no-gate                    # skip fmt/clippy/test
#   scripts/publish.sh --yes                        # skip confirm prompt
#   scripts/publish.sh --start-tier 3               # resume from tier 3
#   scripts/publish.sh --start-crate rlx-runtime    # resume from a crate
#   scripts/publish.sh --min-interval 90            # rate-limit floor (sec)
#   scripts/publish.sh --between-delay 120          # between-tier extra (sec)
#   scripts/publish.sh --poll-interval 10           # index-poll interval
#   scripts/publish.sh --poll-timeout 600           # index-poll cap (sec)
#   scripts/publish.sh --max-retries 5              # cap 429 backoff retries (0 = unlimited)
#   scripts/publish.sh --rate-limit-pad 15          # pad after server retry-after
#   scripts/publish.sh --no-skip-published          # upload even if version exists
#   scripts/publish.sh --no-verify                  # skip cargo's local rebuild
#   scripts/publish.sh --no-poll                    # disable index polling
#
# Resuming:
#
#   If a publish fails (network blip, transient crates.io error),
#   re-run with `--start-crate <name>` (or `--start-tier N`) to pick
#   up where you stopped. Already-published crates aren't revisited.
#
# Dry-run limitations:
#
#   `cargo publish --dry-run` rebuilds each crate tarball and resolves
#   deps against crates.io (path deps become version requirements).
#   Crates that depend on sibling workspace crates not yet on the index
#   show as ⚠ (packaging OK, index resolution pending). Manifest errors,
#   missing `version` on path deps, and verify-time compile failures abort
#   the dry-run immediately. Publish tier 0 first (`rlx-models-core`, …),
#   then re-run dry-run as more leaves land on the index. Run
#   `scripts/publish.sh --list` for tier order + per-crate versions;
#   `validate_publish_order` checks path deps against that order.
#
# Environment:
#
#   CARGO_REGISTRY_TOKEN — must be set, or run `cargo login` first.
#                          Not validated here; cargo errors clearly.

set -euo pipefail

DRY_RUN=0
LIST_ONLY=0
NO_GATE=0
NO_VERIFY=0
NO_POLL=0
SKIP_PUBLISHED=1
ASSUME_YES=0
MIN_INTERVAL=65          # rate-limit safety floor (sec)
BETWEEN_DELAY=90         # extra sleep at tier boundaries (sec)
POLL_INTERVAL=10         # index poll cadence (sec)
POLL_TIMEOUT=600         # max time we'll wait for the index (sec)
MAX_RETRIES=0            # 429 backoff cap when retry-after unparsed (0 = unlimited)
RATE_LIMIT_PAD=15        # extra seconds after server "try again after" time
START_TIER=0
START_CRATE=""
LAST_PUBLISH_ERR=""      # temp log from the last failed publish attempt

# Crates with `publish = false` in their Cargo.toml (cargo skips them;
# listed here for tier-coverage validation only).
SKIPPED=(
    rlx-mamba-bench
    bench_matmul_rlx
    rlx-gemma-inflect-nano
    rlx-narma10
    rlx-tts-bench
    rlx-vision-bench
    rlx-termclean
)

# Tier definitions. Each array entry is a single tier; space-separated
# crate names within. Order within a tier matters when one member
# depends on another in the same tier (e.g. rlx-ir before rlx-flow).
# Publish order: `cargo publish` resolves every path dep in `[dependencies]`
# and `[dev-dependencies]` (including optional) against crates.io. Within
# a tier, list deps before dependents (e.g. rlx-cpu before rlx-splat).
TIERS=(
    "kitten_tts_mini_rlx rlx-assets rlx-diamond rlx-diarize rlx-distributed rlx-guardrails rlx-llama-base rlx-model-hub rlx-models-core rlx-inflect-nano rlx-onnx-decompose rlx-protocol rlx-quant-calib rlx-ssm rlx-vlm-base rlx-wav2vec2-asr"
    "rlx-bert rlx-cli rlx-encodec rlx-facodec rlx-llada2 rlx-mamba rlx-nanocodec rlx-nomic rlx-sam-ir rlx-snac rlx-speechtokenizer rlx-tiny-tts rlx-tune rlx-vibevoice rlx-vision rlx-wavtokenizer rlx-xcodec"
    "rlx-bioclip2 rlx-clinicalbert rlx-dac rlx-dinov2 rlx-dinov3 rlx-hoct rlx-embed rlx-fft rlx-florence2 rlx-funasr rlx-grounding-dino rlx-lfm rlx-lfm-vl rlx-minimax rlx-nemotron-asr rlx-ocr rlx-qwen3 rlx-qwen3-vl rlx-sam rlx-siglip2 rlx-uni2 rlx-vad rlx-vjepa2 rlx-wav2vec2-bert"
    "rlx-eval rlx-flux2 rlx-locateanything rlx-omnicoder rlx-qwen25-vl rlx-qwen35 rlx-sam2 rlx-sam3 rlx-serve rlx-trellis2 rlx-tsac rlx-vit-elastic rlx-whisper"
    "rlx-aec rlx-gemma rlx-kittentts rlx-llama32 rlx-mimi rlx-nemotron-omni rlx-pocket-tts rlx-qwen3-asr rlx-chatterbox rlx-f5tts rlx-gepard rlx-kokoro rlx-luxtts rlx-melotts rlx-metavoice rlx-miotts rlx-miratts rlx-moss-nano rlx-openvoice rlx-parlertts rlx-piper rlx-sesame rlx-soprano rlx-supertonic rlx-zonos"
    "rlx-bonsai rlx-cohere rlx-eagle3 rlx-glm rlx-gpt-oss rlx-granite rlx-inkling rlx-kyutai-tts rlx-minicpm5 rlx-mistral rlx-moshi rlx-nemotron rlx-neutts rlx-orpheus rlx-maya1 rlx-phi rlx-qwen3-tts rlx-styletts2 rlx-tinyllama rlx-voxtral rlx-voxtral-tts rlx-zipvoice"
    "rlx-models rlx-qwen3-tts-train rlx-voxtral-tts-train"
)

usage() {
    sed -n '2,80p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --list)           LIST_ONLY=1; shift ;;
        --no-gate)        NO_GATE=1; shift ;;
        --no-verify)      NO_VERIFY=1; shift ;;
        --no-poll)        NO_POLL=1; shift ;;
        --no-skip-published) SKIP_PUBLISHED=0; shift ;;
        --yes|-y)         ASSUME_YES=1; shift ;;
        --min-interval)   MIN_INTERVAL="$2"; shift 2 ;;
        --between-delay)  BETWEEN_DELAY="$2"; shift 2 ;;
        --poll-interval)  POLL_INTERVAL="$2"; shift 2 ;;
        --poll-timeout)   POLL_TIMEOUT="$2"; shift 2 ;;
        --max-retries)    MAX_RETRIES="$2"; shift 2 ;;
        --rate-limit-pad) RATE_LIMIT_PAD="$2"; shift 2 ;;
        --start-tier)     START_TIER="$2"; shift 2 ;;
        --start-crate)    START_CRATE="$2"; shift 2 ;;
        --help|-h)        usage ;;
        # Legacy aliases for the old --within-delay flag — map onto
        # --min-interval so older invocations don't silently no-op.
        --within-delay)   MIN_INTERVAL="$2"; shift 2 ;;
        *)
            echo "unknown arg: $1" >&2
            echo "run with --help for usage" >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/.."

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

# Fail fast if TIERS diverge from [workspace] members (needs `jq`).
validate_tier_coverage() {
    local -a listed=() missing=() extra=()
    local name tier c s

    for tier in "${TIERS[@]}"; do
        for c in $tier; do
            listed+=("$c")
        done
    done

    if ! command -v jq >/dev/null 2>&1; then
        yellow "jq not found — skipping tier coverage check (install jq to enable)."
        return 0
    fi

    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        for s in "${SKIPPED[@]}"; do
            if [[ "$name" == "$s" ]]; then
                continue 2
            fi
        done
        local found=0
        for c in "${listed[@]}"; do
            if [[ "$name" == "$c" ]]; then
                found=1
                break
            fi
        done
        if (( ! found )); then
            missing+=("$name")
        fi
    done < <(
        cargo metadata --no-deps --format-version 1 2>/dev/null \
            | jq -r '.workspace_members[] as $m | .packages[] | select(.id == $m) | .name'
    )

    for c in "${listed[@]}"; do
        local found=0
        while IFS= read -r name; do
            [[ "$name" == "$c" ]] && found=1 && break
        done < <(
            cargo metadata --no-deps --format-version 1 2>/dev/null \
                | jq -r '.workspace_members[] as $m | .packages[] | select(.id == $m) | .name'
        )
        if (( ! found )); then
            extra+=("$c")
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        red "publish.sh TIERS missing workspace crates: ${missing[*]}"
        exit 1
    fi
    if (( ${#extra[@]} > 0 )); then
        red "publish.sh TIERS list unknown workspace crates: ${extra[*]}"
        exit 1
    fi
}

validate_tier_coverage

# Every rlx-* path dep in [dependencies] / [dev-dependencies] must appear
# in an earlier tier (or the same tier, listed before this crate).
validate_publish_order() {
    if ! command -v python3 >/dev/null 2>&1; then
        yellow "python3 not found — skipping publish-order check (install python3 to enable)."
        return 0
    fi
    local err
    err="$(python3 - "$PWD" <<'PY'
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
script = (root / "scripts/publish.sh").read_text()
m = re.search(r'TIERS=\(\n((?:\s+"[^"]+"\n)+)\)', script)
if not m:
    print("could not parse TIERS from publish.sh", file=sys.stderr)
    sys.exit(2)
tier_lines = re.findall(r'"([^"]+)"', m.group(1))
crate_tier = {}
for i, line in enumerate(tier_lines):
    for j, c in enumerate(line.split()):
        crate_tier[c] = (i, j)

# Workspace dependency keys that differ from the published package name.
DEP_PACKAGE_ALIAS = {"rlx-core": "rlx-models-core"}

# Crate directory names that differ from the published package name
# (TIERS lists package names, since `cargo publish --package` takes them).
DIR_TO_PKG = {}

def workspace_internal_keys(root: Path) -> set[str]:
    ws = (root / "Cargo.toml").read_text()
    sm = re.search(r"\[workspace\.dependencies\](.*?)(?=\n\[|\Z)", ws, re.S)
    keys: set[str] = set()
    if not sm:
        return keys
    for line in sm.group(1).splitlines():
        m2 = re.match(r"^([a-zA-Z0-9_-]+)\s*=", line.strip())
        if not m2:
            continue
        key = m2.group(1)
        block = re.search(rf"^{re.escape(key)}\s*=\s*\{{([^}}]+)\}}", sm.group(1), re.M)
        if block and 'path = "crates/' in block.group(1):
            keys.add(key)
    return keys

INTERNAL_KEYS = workspace_internal_keys(root)

def is_internal_path_dep(line: str, key: str) -> bool:
    if key in INTERNAL_KEYS:
        return True
    return bool(
        re.search(
            rf"^{re.escape(key)}\s*=\s*\{{[^}}]*path\s*=\s*(\"\.\.?/|\"crates/)",
            line.strip(),
        )
    )

def parse_rlx_deps(toml_path: Path, crate_name: str) -> set[str]:
    text = toml_path.read_text()
    deps: set[str] = set()
    for sm in re.finditer(r"\[([^\]]+)\](.*?)(?=\n\[|\Z)", text, re.S):
        section = sm.group(1)
        if section != "dependencies" and "dependencies" not in section:
            continue
        for line in sm.group(2).splitlines():
            m2 = re.match(r"^([a-zA-Z0-9_-]+)\s*=", line.strip())
            if not m2:
                continue
            key = m2.group(1)
            if key == crate_name:
                continue
            if is_internal_path_dep(line, key):
                deps.add(key)
    return deps

violations: list[str] = []
for toml in sorted((root / "crates").glob("*/Cargo.toml")):
    name = DIR_TO_PKG.get(toml.parent.name, toml.parent.name)
    if re.search(r"^publish\s*=\s*false", toml.read_text(), re.M):
        continue
    if name not in crate_tier:
        violations.append(f"{name} is publishable but missing from TIERS")
        continue
    my_tier, my_pos = crate_tier[name]
    for dep in sorted(parse_rlx_deps(toml, name)):
        dep = DEP_PACKAGE_ALIAS.get(dep, dep)
        if dep == name:
            continue
        if dep not in crate_tier:
            violations.append(f"{name}: path dep {dep} is not listed in TIERS")
            continue
        dep_tier, dep_pos = crate_tier[dep]
        if dep_tier > my_tier or (dep_tier == my_tier and dep_pos >= my_pos):
            violations.append(
                f"{name} (tier {my_tier} pos {my_pos} needs {dep} at tier {dep_tier} pos {dep_pos}): "
                f"publish {dep} before {name}"
            )

if violations:
    for v in violations:
        print(v)
    sys.exit(1)
PY
)"
    if [[ -n "$err" ]]; then
        red "Publish tier order does not match Cargo.toml path dependencies:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && red "  $line"
        done <<< "$err"
        red "Fix scripts/publish.sh TIERS (or remove path deps from dev-dependencies)."
        exit 1
    fi
}

validate_publish_order

# Extract the workspace version once so the index-readiness check
# knows what to look for. Stops at the next `[…]` header so we don't
# accidentally read a version line from another table.
WORKSPACE_VERSION="$(awk '
    BEGIN              { in_block = 0 }
    /^\[workspace\.package\]/ { in_block = 1; next }
    /^\[/              { in_block = 0; next }
    in_block && $1 == "version" {
        # Line looks like:  version       = "0.2.0"
        match($0, /"[^"]+"/)
        if (RSTART > 0) {
            v = substr($0, RSTART + 1, RLENGTH - 2)
            print v
            exit
        }
    }
' Cargo.toml)"
if [[ -z "$WORKSPACE_VERSION" ]]; then
    echo "could not parse [workspace.package].version from Cargo.toml" >&2
    exit 1
fi

# `[package].version` in crates/<name>/Cargo.toml, or `version.workspace = true`.
crate_version() {
    local crate="$1"
    local toml="crates/${crate}/Cargo.toml"
    if [[ ! -f "$toml" ]]; then
        echo "$WORKSPACE_VERSION"
        return
    fi
    local parsed
    parsed="$(
        awk '
            BEGIN { in_pkg = 0 }
            /^\[package\]/ { in_pkg = 1; next }
            /^\[/ { in_pkg = 0 }
            in_pkg && $1 == "version" {
                if ($0 ~ /version\.workspace/) { exit 2 }
                match($0, /"[^"]+"/)
                if (RSTART > 0) {
                    print substr($0, RSTART + 1, RLENGTH - 2)
                    exit
                }
            }
        ' "$toml" 2>/dev/null || true
    )"
    if [[ -n "$parsed" ]]; then
        echo "$parsed"
    else
        echo "$WORKSPACE_VERSION"
    fi
}

list_tiers() {
    bold "Publish order (workspace default $WORKSPACE_VERSION):"
    local i=0
    for tier in "${TIERS[@]}"; do
        echo "  tier $i:"
        for c in $tier; do
            local ver
            ver="$(crate_version "$c")"
            if [[ "$ver" == "$WORKSPACE_VERSION" ]]; then
                echo "    - $c"
            else
                echo "    - $c ($ver)"
            fi
        done
        i=$((i + 1))
    done
    echo
    bold "Skipped (publish = false or not a workspace member):"
    for s in "${SKIPPED[@]}"; do
        case "$s" in
            rlx-mamba-bench)
                echo "  - rlx-mamba-bench          (workspace.exclude; burn bench)"
                ;;
            bench_matmul_rlx)
                echo "  - bench_matmul_rlx         (workspace.exclude; publish = false; matmul bench)"
                ;;
            rlx-gemma-inflect-nano)
                echo "  - rlx-gemma-inflect-nano   (publish = false; Gemma + Inflect demo)"
                ;;
            rlx-narma10)
                echo "  - rlx-narma10              (publish = false; NARMA-10 reference)"
                ;;
            rlx-tts-bench)
                echo "  - rlx-tts-bench            (publish = false; unified TTS bench harness)"
                ;;
            rlx-vision-bench)
                echo "  - rlx-vision-bench         (publish = false; vision train/bench harness)"
                ;;
            rlx-termclean)
                echo "  - rlx-termclean            (publish = false; terminal cleanup helper)"
                ;;
            *)
                echo "  - $s"
                ;;
        esac
    done
}

if (( LIST_ONLY )); then
    list_tiers
    exit 0
fi

# ── Pre-flight gates ────────────────────────────────────────────
if (( ! NO_GATE )); then
    bold "[1/3] cargo fmt --check"
    cargo fmt --all -- --check

    bold "[2/3] cargo clippy --workspace --all-targets -- -D warnings"
    cargo clippy --workspace --all-targets -- -D warnings

    bold "[3/3] cargo test --workspace --release --lib"
    # Publish gate runs LIB UNIT TESTS ONLY — the fast, deterministic checks of
    # core library logic. Integration tests (`tests/`) are intentionally NOT run
    # here: the bulk are real-weight / cross-backend / GPU / e2e parity suites that
    # need local checkpoints, GPUs, or network, so they only self-skip in a clean
    # publish env while adding large compile+run time. Run them separately with
    # `cargo test --workspace --release --tests` (with checkpoints/GPUs available).
    #
    # `kitten_tts_mini_rlx` is still excluded: its 16 lib unit tests need local
    # ONNX fixture paths (optional weights/) absent in a clean checkout.
    # Examples/benches are compile-checked by the clippy `--all-targets` gate above.
    cargo test --workspace --release --lib --exclude kitten_tts_mini_rlx
    green "Pre-flight gates passed."
fi

# ── Confirmation ────────────────────────────────────────────────
list_tiers
echo
if (( DRY_RUN )); then
    yellow "Mode: DRY RUN — no actual uploads, no rate-limit sleeps, no index polling."
else
    yellow "Mode: REAL PUBLISH — uploads to crates.io."
    yellow "Rate-limit floor:      ${MIN_INTERVAL}s between publishes."
    yellow "Between-tier extra:    ${BETWEEN_DELAY}s after each tier."
    if (( NO_POLL )); then
        yellow "Index polling:         DISABLED (--no-poll) — relying on fixed sleeps only."
    else
        yellow "Index polling:         every ${POLL_INTERVAL}s, hard cap ${POLL_TIMEOUT}s per crate."
    fi
    if (( MAX_RETRIES == 0 )); then
        yellow "Registry retries:        auto-wait + retry until success (unlimited)."
    else
        yellow "Registry retries:        up to ${MAX_RETRIES} backoff attempts per crate."
    fi
    yellow "Retry-after parsing:     HTTP-date + status-specific backoff (+${RATE_LIMIT_PAD}s pad)."
    if (( SKIP_PUBLISHED )); then
        yellow "Already on crates.io:    skip (no upload, no rate-limit wait)."
    else
        yellow "Already on crates.io:    still attempt upload (--no-skip-published)."
    fi
    if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
        # cargo will use ~/.cargo/credentials if no env var.
        yellow "CARGO_REGISTRY_TOKEN not set — relying on \`cargo login\` credentials."
    fi
fi
echo

if (( ! ASSUME_YES )); then
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        red "Aborted."
        exit 1
    fi
fi

# ── Sparse-index URL builder ────────────────────────────────────
#
# crates.io's sparse-index partitioning mirrors crates.io-index's
# git-side layout — encoded in the URL prefix so each shard stays
# small:
#
#   1 char:   /1/<crate>
#   2 chars:  /2/<crate>
#   3 chars:  /3/<first_char>/<crate>
#   4+ chars: /<first_2>/<next_2>/<crate>
#
# Names are always lower-case and hyphens are kept verbatim. See
# https://doc.rust-lang.org/cargo/reference/registry-index.html
sparse_index_path() {
    local name="$1"
    local n="${#name}"
    if   (( n == 1 )); then printf '1/%s\n'     "$name"
    elif (( n == 2 )); then printf '2/%s\n'     "$name"
    elif (( n == 3 )); then printf '3/%s/%s\n'  "${name:0:1}" "$name"
    else                    printf '%s/%s/%s\n' "${name:0:2}" "${name:2:2}" "$name"
    fi
}

# Returns 0 if <crate>@<version> is already on the sparse index.
version_on_crates_io() {
    check_index "$1" "$2"
}

# Returns 0 if the sparse index serves <version> for <crate>, 1 if
# the version isn't there yet, or 2 if the crate itself isn't in the
# index at all (pre-first-publish — perfectly fine, just means we
# should keep polling).
check_index() {
    local crate="$1"
    local version="$2"
    local url="https://index.crates.io/$(sparse_index_path "$crate")"
    local body
    # `curl -fsS` exits non-zero on 4xx/5xx; 404 means "crate doesn't
    # exist on the index yet" which is fine — keep polling.
    body="$(curl -fsS --max-time 10 "$url" 2>/dev/null || true)"
    if [[ -z "$body" ]]; then
        return 2
    fi
    # Index lines are NDJSON. Match `"vers":"<version>"` anywhere in
    # the file. `grep -F` keeps the dots from being regex-interpreted.
    if printf '%s' "$body" | grep -Fq "\"vers\":\"$version\""; then
        return 0
    fi
    return 1
}

# Block until check_index says we're good or the timeout trips.
wait_for_index() {
    local crate="$1"
    local version="$2"
    if (( NO_POLL )); then
        yellow "  (--no-poll set; skipping index readiness check)"
        return 0
    fi
    local elapsed=0
    yellow "  polling https://index.crates.io/.../$crate for $version (every ${POLL_INTERVAL}s, cap ${POLL_TIMEOUT}s)..."
    while (( elapsed < POLL_TIMEOUT )); do
        if check_index "$crate" "$version"; then
            green "  index ready after ${elapsed}s — $crate@$version resolvable."
            return 0
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$(( elapsed + POLL_INTERVAL ))
    done
    yellow "  index didn't report $crate@$version within ${POLL_TIMEOUT}s — continuing anyway,"
    yellow "    next crate's publish may fail dep resolution if it depends on this one."
    return 0
}

# True when a path dep was rewritten for publish but is not on crates.io yet.
is_missing_registry_dep_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    grep -qE 'no matching package named|failed to select a version for the requirement|location searched: crates\.io index' "$err_file"
}

# Manifest / verify failures are real blockers (not "dep not on index yet").
is_publish_verify_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    grep -qE 'failed to verify manifest|failed to verify package tarball|does not specify a version|could not compile' "$err_file"
}

missing_registry_dep_name() {
    local err_file="$1"
    local name
    name="$(
        grep -oE 'no matching package named `[^`]+`' "$err_file" 2>/dev/null \
            | head -1 \
            | sed -E 's/no matching package named `([^`]+)`/\1/'
    )"
    if [[ -n "$name" ]]; then
        echo "$name"
        return
    fi
    grep -oE 'requirement `[^ =`]+' "$err_file" 2>/dev/null \
        | head -1 \
        | sed -E 's/requirement `//'
}

# Version cargo wanted on the index (from the error, else crate_version).
missing_registry_dep_version() {
    local err_file="$1"
    local missing_crate="$2"
    local v
    v="$(
        grep -oE 'requirement `[^`]+`' "$err_file" 2>/dev/null \
            | head -1 \
            | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' \
            | head -1
    )"
    if [[ -n "$v" ]]; then
        echo "$v"
    else
        crate_version "$missing_crate"
    fi
}

# True when cargo says this version was uploaded already.
is_already_exists_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    grep -qiE 'already exists on crates\.io|already exists on the registry' "$err_file"
}

# Last HTTP status from a `status NNN` line in cargo/crates.io output (e.g. 429).
registry_http_status_from_log() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 0
    grep -oiE 'status [0-9]{3}' "$err_file" 2>/dev/null \
        | tail -1 \
        | grep -oE '[0-9]{3}' \
        || true
}

# Transient registry / network failures — retry after a computed wait.
is_retryable_registry_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    local status
    status="$(registry_http_status_from_log "$err_file")"
    case "$status" in
        408|429|500|502|503|504) return 0 ;;
    esac
    grep -qiE \
        'Too Many Requests|rate.?limit|published too many new crates|try again after|temporarily unavailable|service unavailable|bad gateway|gateway timeout|timed out|connection reset|connection refused|unexpected eof|broken pipe|error sending request|operation timed out|dns error' \
        "$err_file"
}

# Convert crates.io HTTP-date ("Wed, 27 May 2026 11:09:09 GMT") → epoch.
http_date_to_epoch() {
    local when="$1"
    if [[ "$(uname -s)" == Darwin ]]; then
        date -j -u -f '%a, %d %b %Y %H:%M:%S GMT' "$when" '+%s' 2>/dev/null
    else
        date -u -d "$when" '+%s' 2>/dev/null
    fi
}

# Default backoff when no explicit retry window is in the log.
registry_status_default_wait() {
    local status="$1"
    case "$status" in
        429) echo $(( MIN_INTERVAL * 2 )) ;;
        408) echo 60 ;;
        500|502|503|504) echo 90 ;;
        *) echo "$MIN_INTERVAL" ;;
    esac
}

# Seconds to wait before retrying a registry error.
# Prints: wait_seconds parsed_flag http_status
#   parsed_flag=1 → wait derived from server retry window or status default
registry_retry_wait_seconds() {
    local err_file="$1"
    local when epoch now wait parsed=0 status
    status="$(registry_http_status_from_log "$err_file")"
    wait="$(registry_status_default_wait "$status")"

    when="$(
        grep -oiE '(please )?try again after [A-Za-z]{3}, [0-9]+ [A-Za-z]+ [0-9]+ [0-9:]+ GMT' \
            "$err_file" 2>/dev/null \
            | tail -1 \
            | sed -E 's/^[Pp]lease [Tt]ry again after //; s/^[Tt]ry again after //'
    )"
    if [[ -n "$when" ]]; then
        epoch="$(http_date_to_epoch "$when" || true)"
        if [[ -n "${epoch:-}" ]]; then
            now="$(date -u '+%s')"
            wait=$(( epoch - now + RATE_LIMIT_PAD ))
            parsed=1
            if (( wait < MIN_INTERVAL )); then
                wait=$MIN_INTERVAL
            fi
            if (( wait > 86400 )); then
                wait=86400
            fi
            echo "$wait $parsed ${status:-0}"
            return 0
        fi
        if grep -qiE 'try again after|published too many new crates' "$err_file"; then
            wait=$(( MIN_INTERVAL * 2 ))
            parsed=1
            echo "$wait $parsed ${status:-0}"
            return 0
        fi
    fi

    local retry_after
    retry_after="$(
        grep -oiE 'retry-after: *[0-9]+' "$err_file" 2>/dev/null \
            | tail -1 \
            | grep -oE '[0-9]+' \
            || true
    )"
    if [[ -n "$retry_after" ]]; then
        wait=$(( retry_after + RATE_LIMIT_PAD ))
        parsed=1
        echo "$wait $parsed ${status:-0}"
        return 0
    fi

    if is_retryable_registry_error "$err_file"; then
        parsed=1
    fi
    echo "$wait $parsed ${status:-0}"
}

# Sleep with a single live countdown line — easier to watch than
# silent waits during long publishes.
sleep_with_countdown() {
    local seconds="$1"
    local label="$2"
    local remaining=$seconds
    while (( remaining > 0 )); do
        printf "\r  %s — %3ds remaining " "$label" "$remaining"
        sleep 1
        remaining=$(( remaining - 1 ))
    done
    printf "\r  %s — done.                  \n" "$label"
}

# ── Walk tiers ──────────────────────────────────────────────────
DRY_RUN_PASS=()
DRY_RUN_FAIL=()
DRY_RUN_BLOCKER_LOG="$(mktemp)"
ALREADY_ON_CRATES_IO=()
PUBLISHED_THIS_RUN=()

publish_one_attempt() {
    local crate="$1"
    local args=()
    args+=("--package" "$crate")
    if (( DRY_RUN )); then
        args+=("--dry-run")
    fi
    if (( NO_VERIFY )); then
        args+=("--no-verify")
    fi
    if (( DRY_RUN )); then
        args+=("--allow-dirty")
    fi
    local tmp_err
    tmp_err="$(mktemp)"
    LAST_PUBLISH_ERR=""
    # Do not trust cargo's exit code alone — crates.io 429 often returns 0.
    set +e
    if (( DRY_RUN )); then
        cargo publish "${args[@]}" >"$tmp_err" 2>&1
    else
        bold ">> cargo publish ${args[*]}"
        cargo publish "${args[@]}" 2>&1 | tee "$tmp_err"
    fi
    local rc=${PIPESTATUS[0]}
    set -e

    if is_missing_registry_dep_error "$tmp_err"; then
        LAST_PUBLISH_ERR="$tmp_err"
        return 44
    fi

    if is_already_exists_error "$tmp_err"; then
        rm -f "$tmp_err"
        return 43
    fi

    if is_retryable_registry_error "$tmp_err"; then
        LAST_PUBLISH_ERR="$tmp_err"
        return 42
    fi

    if (( rc == 0 )) && ! grep -qE 'error(\[[0-9]+\])?: failed|error(\[[0-9]+\])?: failed to publish' "$tmp_err"; then
        rm -f "$tmp_err"
        return 0
    fi

    LAST_PUBLISH_ERR="$tmp_err"
    return 1
}

# 0 = published (or dry-run ok), 1 = hard failure, 2 = already on crates.io
publish_one() {
    local crate="$1"
    local crate_ver
    crate_ver="$(crate_version "$crate")"
    local attempt=1
    local backoff=$MIN_INTERVAL
    local missing_dep_attempts=0
    while true; do
        # `|| rc=$?` keeps `set -e` from aborting on non-zero returns (42/44).
        local rc=0
        publish_one_attempt "$crate" || rc=$?
        if (( rc == 0 )); then
            rm -f "${LAST_PUBLISH_ERR:-}"
            LAST_PUBLISH_ERR=""
            if (( DRY_RUN )); then
                DRY_RUN_PASS+=("$crate")
                green "  ✓ $crate@$crate_ver — packaged + verify ok"
            fi
            return 0
        fi
        if (( rc == 43 )); then
            green "  ○ $crate@$crate_ver already on crates.io — skip"
            return 2
        fi
        if (( DRY_RUN )); then
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]] \
                && is_publish_verify_error "$LAST_PUBLISH_ERR"; then
                red "Dry-run verify failed for $crate (fix manifest/deps before publish):"
                sed -n '1,40p' "$LAST_PUBLISH_ERR" >&2
                red "  full log: $LAST_PUBLISH_ERR"
                exit 1
            fi
            DRY_RUN_FAIL+=("$crate")
            local missing="" miss_ver=""
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
                missing="$(missing_registry_dep_name "$LAST_PUBLISH_ERR")"
                if [[ -n "$missing" ]]; then
                    miss_ver="$(missing_registry_dep_version "$LAST_PUBLISH_ERR" "$missing")"
                    echo "$missing@$miss_ver $crate" >>"$DRY_RUN_BLOCKER_LOG"
                    yellow "  ⚠ $crate@$crate_ver — needs $missing@$miss_ver on crates.io"
                else
                    yellow "  ⚠ $crate@$crate_ver — index dep missing (see --list tier order)"
                fi
            else
                yellow "  ⚠ $crate@$crate_ver — index dep missing (see --list tier order)"
            fi
            rm -f "${LAST_PUBLISH_ERR:-}"
            LAST_PUBLISH_ERR=""
            return 0
        fi
        if (( rc == 44 )); then
            local missing=""
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
                missing="$(missing_registry_dep_name "$LAST_PUBLISH_ERR")"
            fi
            if [[ -n "$missing" ]]; then
                local miss_ver
                miss_ver="$(missing_registry_dep_version "$LAST_PUBLISH_ERR" "$missing")"
                missing_dep_attempts=$(( missing_dep_attempts + 1 ))
                yellow "  $crate: waiting for $missing@$miss_ver on sparse index (attempt $missing_dep_attempts), then retry."
                wait_for_index "$missing" "$miss_ver"
                rm -f "${LAST_PUBLISH_ERR:-}"
                LAST_PUBLISH_ERR=""
                sleep_with_countdown "$POLL_INTERVAL" "index settle after $missing"
                continue
            fi
            red "Publish blocked for $crate: dependency not on crates.io yet."
            red "  Check scripts/publish.sh tier order."
            exit 1
        fi
        if (( rc == 42 )); then
            local wait_sec=$MIN_INTERVAL parsed=0 http_status=0 when_human=""
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
                read -r wait_sec parsed http_status < <(registry_retry_wait_seconds "$LAST_PUBLISH_ERR")
                if grep -qiE 'try again after' "$LAST_PUBLISH_ERR"; then
                    when_human="$(
                        grep -oiE '(please )?try again after [A-Za-z]{3}, [0-9]+ [A-Za-z]+ [0-9]+ [0-9:]+ GMT' \
                            "$LAST_PUBLISH_ERR" 2>/dev/null \
                            | tail -1 \
                            | sed -E 's/^[Pp]lease [Tt]ry again after //; s/^[Tt]ry again after //'
                    )"
                fi
                rm -f "$LAST_PUBLISH_ERR"
                LAST_PUBLISH_ERR=""
            fi
            if (( parsed )); then
                if (( http_status > 0 )); then
                    yellow "  crates.io HTTP $http_status for $crate — waiting ${wait_sec}s then retry (automatic)."
                else
                    yellow "  registry transient error for $crate — waiting ${wait_sec}s then retry (automatic)."
                fi
                if [[ -n "$when_human" ]]; then
                    yellow "    server window ends: $when_human (+${RATE_LIMIT_PAD}s pad)"
                fi
                attempt=1
                backoff=$MIN_INTERVAL
            else
                if (( MAX_RETRIES > 0 && attempt > MAX_RETRIES )); then
                    red "Publish failed for $crate after $MAX_RETRIES registry retries."
                    red "Re-run with --start-crate $crate to resume later."
                    exit 1
                fi
                wait_sec=$backoff
                if (( MAX_RETRIES > 0 )); then
                    yellow "  registry retry for $crate (attempt $attempt/$MAX_RETRIES) — backoff ${wait_sec}s."
                else
                    yellow "  registry retry for $crate (attempt $attempt) — backoff ${wait_sec}s."
                fi
                backoff=$(( backoff * 2 ))
                if (( backoff > 600 )); then
                    backoff=600
                fi
                attempt=$(( attempt + 1 ))
            fi
            sleep_with_countdown "$wait_sec" "crates.io registry cooldown"
            continue
        fi
        if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
            red "  last cargo log: $LAST_PUBLISH_ERR"
        fi
        red "Publish failed for $crate (non-retryable, exit $rc)."
        red "Re-run with --start-crate $crate (or a later one) to resume."
        exit 1
    done
}

# Resolve start position.
skip_until_tier=$START_TIER
skip_until_crate="$START_CRATE"

for tier_idx in "${!TIERS[@]}"; do
    if (( tier_idx < skip_until_tier )); then
        continue
    fi
    tier="${TIERS[$tier_idx]}"
    bold "── Tier $tier_idx ────────────────────────────────────────"

    published_in_tier=0
    for crate in $tier; do
        if (( DRY_RUN )); then
            bold ">> $crate@$(crate_version "$crate")"
        fi
        # Skip past --start-crate if specified.
        if [[ -n "$skip_until_crate" ]]; then
            if [[ "$crate" != "$skip_until_crate" ]]; then
                yellow "  skip $crate (resume target: $skip_until_crate)"
                continue
            fi
            skip_until_crate=""
        fi

        # Already on crates.io — no upload, no rate-limit sleep.
        if (( SKIP_PUBLISHED )) && (( ! DRY_RUN )); then
            crate_ver="$(crate_version "$crate")"
            if version_on_crates_io "$crate" "$crate_ver"; then
                green "  skip $crate@$crate_ver (already on crates.io)"
                ALREADY_ON_CRATES_IO+=("$crate")
                continue
            fi
        fi

        # Rate-limit floor only between actual uploads (not after skips).
        if (( published_in_tier > 0 )) && (( ! DRY_RUN )); then
            sleep_with_countdown "$MIN_INTERVAL" "rate-limit floor"
        fi

        pub_rc=0
        publish_one "$crate" || pub_rc=$?
        if (( pub_rc == 2 )); then
            ALREADY_ON_CRATES_IO+=("$crate")
            continue
        fi
        if (( pub_rc != 0 )); then
            exit 1
        fi

        PUBLISHED_THIS_RUN+=("$crate")
        published_in_tier=$(( published_in_tier + 1 ))

        # Post-publish: poll the index until the new version is
        # actually queryable, so the *next* crate's dep resolution
        # doesn't 404. Dry runs skip this entirely.
        if (( ! DRY_RUN )); then
            pub_ver="$(crate_version "$crate")"
            wait_for_index "$crate" "$pub_ver"
        fi
    done

    # Between-tier extra delay only when this tier uploaded something.
    if (( ! DRY_RUN )) && (( tier_idx + 1 < ${#TIERS[@]} )) && (( published_in_tier > 0 )); then
        sleep_with_countdown "$BETWEEN_DELAY" "between-tier (crates.io index settle)"
    fi
done

if (( DRY_RUN )); then
    echo
    bold "Dry-run summary:"
    green "  passed (full dry-run incl. dep resolution): ${#DRY_RUN_PASS[@]}"
    if (( ${#DRY_RUN_PASS[@]} > 0 )); then
        for c in "${DRY_RUN_PASS[@]}"; do echo "    ✓ $c@$(crate_version "$c")"; done
    fi
    if (( ${#DRY_RUN_FAIL[@]} > 0 )); then
        yellow "  metadata + packaging ok, index dep missing (publish tiers in order): ${#DRY_RUN_FAIL[@]}"
        for c in "${DRY_RUN_FAIL[@]}"; do echo "    ⚠ $c@$(crate_version "$c")"; done
        if [[ -s "$DRY_RUN_BLOCKER_LOG" ]]; then
            echo
            yellow "  Publish these on crates.io first (unblocks dependents):"
            awk '
                { key = $1; $1 = ""; sub(/^ /, "", $0); crates[key] = crates[key] ? crates[key] ", " $0 : $0; n[key]++ }
                END {
                    for (k in n) printf "    → %s (%d crate%s)\n", k, n[k], (n[k] == 1 ? "" : "s")
                }
            ' "$DRY_RUN_BLOCKER_LOG" | sort
        fi
        echo
        yellow "  Not failures — publish earlier tiers (see --list), wait for index, re-run dry-run."
    fi
    rm -f "$DRY_RUN_BLOCKER_LOG"
else
    echo
    if (( ${#PUBLISHED_THIS_RUN[@]} > 0 )); then
        green "Published this run (${#PUBLISHED_THIS_RUN[@]}):"
        for c in "${PUBLISHED_THIS_RUN[@]}"; do echo "    ✓ $c"; done
    fi
    if (( ${#ALREADY_ON_CRATES_IO[@]} > 0 )); then
        yellow "Already on crates.io — skipped (${#ALREADY_ON_CRATES_IO[@]}):"
        for c in "${ALREADY_ON_CRATES_IO[@]}"; do echo "    ○ $c"; done
    fi
    if (( ${#PUBLISHED_THIS_RUN[@]} == 0 )) && (( ${#ALREADY_ON_CRATES_IO[@]} > 0 )); then
        green "Nothing left to publish — all listed crates already on crates.io at their target versions."
    elif (( ${#PUBLISHED_THIS_RUN[@]} > 0 )); then
        green "Publish run finished."
    else
        green "All tiers processed."
    fi
fi
