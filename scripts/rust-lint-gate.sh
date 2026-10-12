#!/usr/bin/env bash
# Shared cargo fmt + clippy + rustdoc gate (warnings as errors).
#
# Usage:
#   scripts/rust-lint-gate.sh --workspace
#   scripts/rust-lint-gate.sh --packages pkg1,pkg2
#   scripts/rust-lint-gate.sh --files path/a.rs path/b.rs
#   scripts/rust-lint-gate.sh --staged
#
# Exit 0 on success; non-zero on fmt/clippy failure.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE=""
PACKAGES=""
FILES=""

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \?//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) MODE=workspace; shift ;;
    --staged) MODE=staged; shift ;;
    --packages)
      MODE=packages
      # Usage documents a comma-separated list, but everything downstream
      # word-splits on whitespace — without this, `-p a,b` reaches cargo as one
      # package name and it refuses the comma.
      PACKAGES="$(printf '%s' "${2:-}" | tr ',' ' ')"
      shift 2
      ;;
    --files)
      MODE=files
      shift
      while [ $# -gt 0 ]; do
        case "$1" in
          --*) break ;;
          *) FILES="${FILES}${FILES:+ }$1"; shift ;;
        esac
      done
      ;;
    -h|--help) usage ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      ;;
  esac
done

if [ -z "$MODE" ]; then
  MODE=workspace
fi

package_for_file() {
  f="$1"
  case "$f" in
    /*) rel="${f#"$ROOT"/}" ;;
    *) rel="$f" ;;
  esac
  crate_dir="$(printf '%s\n' "$rel" | awk -F/ 'NF>=2 && $1=="crates" {print $1"/"$2; exit}')"
  if [ -z "$crate_dir" ]; then
    return 1
  fi
  manifest="$ROOT/$crate_dir/Cargo.toml"
  if [ ! -f "$manifest" ]; then
    return 1
  fi
  awk -F'"' '/^name[[:space:]]*=/ { print $2; exit }' "$manifest"
}

append_unique_pkg() {
  pkg="$1"
  case " $PACKAGES " in
    *" $pkg "*) ;;
    *) PACKAGES="${PACKAGES}${PACKAGES:+ }$pkg" ;;
  esac
}

collect_packages_from_files() {
  for f in "$@"; do
    case "$f" in
      *.rs) ;;
      *) continue ;;
    esac
    if pkg="$(package_for_file "$f")"; then
      append_unique_pkg "$pkg"
    fi
  done
}

case "$MODE" in
  staged)
    FILES="$(git diff --cached --name-only --diff-filter=ACMR -- '*.rs' | tr '\n' ' ')"
    FILES="$(printf '%s' "$FILES" | sed 's/[[:space:]]*$//')"
    if [ -z "$FILES" ]; then
      echo "rust-lint-gate: no staged .rs files; skip"
      exit 0
    fi
    # shellcheck disable=SC2086
    collect_packages_from_files $FILES
    ;;
  files)
    if [ -z "$FILES" ]; then
      echo "rust-lint-gate: no files; skip"
      exit 0
    fi
    # shellcheck disable=SC2086
    collect_packages_from_files $FILES
    ;;
  packages|workspace)
    ;;
esac

if [ "$MODE" != workspace ] && [ -z "$PACKAGES" ] && [ -n "${FILES:-}" ]; then
  echo "rust-lint-gate: .rs files present but no workspace packages resolved; falling back to workspace" >&2
  MODE=workspace
fi

# An explicit --packages with nothing in it is a caller bug (an unset variable
# in a wrapper), not a no-op: skipping there would make a gate that never ran
# indistinguishable from one that passed. --staged with no staged .rs files is
# the genuine no-op and still skips below.
if [ "$MODE" = packages ] && [ -z "$PACKAGES" ]; then
  echo "rust-lint-gate: --packages given with no package names" >&2
  exit 2
fi

if [ "$MODE" != workspace ] && [ -z "$PACKAGES" ]; then
  echo "rust-lint-gate: nothing to check; skip"
  exit 0
fi

pkg_args=""
for p in $PACKAGES; do
  pkg_args="$pkg_args -p $p"
done

echo "==> cargo fmt"
if [ "$MODE" = workspace ]; then
  cargo fmt --all -- --check
elif [ -n "$FILES" ]; then
  # shellcheck disable=SC2086
  rustfmt --edition 2024 --check $FILES
else
  # Scope this to the selected packages, for the same reason the clippy step
  # passes --no-deps: an unrelated crate mid-edit elsewhere in the workspace
  # must not fail a gate the caller scoped to its own packages.
  # shellcheck disable=SC2086
  cargo fmt $pkg_args -- --check
fi

echo "==> cargo clippy (-D warnings)"
if [ "$MODE" = workspace ]; then
  cargo clippy --workspace --all-targets -- -D warnings
else
  # --no-deps: only lint selected packages so workspace-member
  # dependencies with pre-existing warnings do not fail a scoped gate.
  # shellcheck disable=SC2086
  cargo clippy $pkg_args --all-targets --no-deps -- -D warnings
fi

# rustdoc's link lints are warn-by-default and nothing else runs rustdoc, so a
# broken `[`Thing`]` ships as literal brackets in the published docs. Cheap
# here: --no-deps means rustdoc runs only over the selected packages, reusing
# the dependency builds clippy just made. Default features only, so a
# feature-gated module's links go unchecked — the same blind spot the clippy
# step above has. --keep-going because rustdoc otherwise stops at the first
# failing crate, which means one broken crate per full run.
echo "==> cargo doc (-D warnings)"
if [ "$MODE" = workspace ]; then
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --keep-going --quiet
else
  # shellcheck disable=SC2086
  RUSTDOCFLAGS="-D warnings" cargo doc $pkg_args --no-deps --keep-going --quiet
fi

echo "rust-lint-gate: ok"
