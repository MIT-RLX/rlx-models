#!/usr/bin/env python3
"""Upload prepared weights/* dirs as Hugging Face model repos.

Requires auth:
  hf auth login
  # or: export HF_TOKEN=hf_...

Usage:
  python3 scripts/publish_weights_hf.py --prepare --cards-only
  python3 scripts/publish_weights_hf.py --only moss-nano,soprano,tiny-tts-rlx,rlx-tts,rlx-asr
  python3 scripts/publish_weights_hf.py --dry-run
  python3 scripts/publish_weights_hf.py              # all publishable dirs

Ignore patterns:
  - Always skip .DS_Store, fixtures, *.f16.gguf, *.rlxpack
  - Native RLXP repos (kind=rlx-native or primary *.rlxp): also skip local ONNX /
    legacy GGUF, then purge any leftover ONNX still on Hub (local pack-time ONNX
    is not removed by upload_folder delete_patterns alone)

ONNX-based redistribs (kokoro, chatterbox, …) still upload their .onnx trees.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
PREPARE = ROOT / "scripts" / "prepare_weights_hf.py"

# Always skip local junk / staging leftovers.
IGNORE_BASE = [
    ".DS_Store",
    "**/.DS_Store",
    "**/.cache/**",
    "**/__pycache__/**",
    "**/fixtures/**",
    "**/*.f16.gguf",
    "**/*.rlxpack",
]

# Native RLXP packs: Hub ships nested graphs/*.rlxp only — never re-upload pack-time ONNX
# or superseded local GGUF sidecars.
IGNORE_NATIVE_RLXP = [
    "codec-q4_k_m.gguf",
    "Soprano-1.1-80M.Q4_K_M.gguf",
    "moss-nano.gguf",
    "soprano.gguf",
    "rlx-tts.gguf",
    "model.gguf",
    "**/*.onnx",
    "**/*.onnx.data",
    "**/*.onnx_data",
    "**/onnx/**",
    "**/*_shared.data",
    "moss_tts_global_shared.data",
    "moss_tts_local_shared.data",
]

# Remote leftovers to drop when syncing native RLXP repos.
DELETE_NATIVE_RLXP = [
    "**/*.onnx",
    "**/*.onnx.data",
    "**/*.onnx_data",
    "**/onnx/**",
    "**/*_shared.data",
    "moss-nano.gguf",
    "soprano.gguf",
    "rlx-tts.gguf",
    "model.gguf",
    "**/*.f16.gguf",
    "**/*.rlxpack",
]


def is_native_rlxp(meta: dict) -> bool:
    if meta.get("kind") == "rlx-native":
        return True
    primary = meta.get("primary") or []
    return any(str(p).endswith(".rlxp") for p in primary)


def ignore_for(meta: dict) -> list[str]:
    pats = list(IGNORE_BASE)
    if is_native_rlxp(meta):
        pats.extend(IGNORE_NATIVE_RLXP)
    return pats


def remote_native_leftovers(filenames: list[str]) -> list[str]:
    """Remote paths that must not remain on native RLXP Hub repos."""
    legacy = {"moss-nano.gguf", "soprano.gguf", "rlx-tts.gguf", "model.gguf"}
    out = []
    for name in filenames:
        lower = name.lower()
        base = name.rsplit("/", 1)[-1]
        if base in legacy or lower.endswith((".f16.gguf", ".rlxpack")):
            out.append(name)
        elif lower.endswith((".onnx", ".onnx.data", ".onnx_data")):
            out.append(name)
        elif "/onnx/" in lower or lower.startswith("onnx/"):
            out.append(name)
        elif lower.endswith("_shared.data") or base in {
            "moss_tts_global_shared.data",
            "moss_tts_local_shared.data",
        }:
            out.append(name)
    return out


def purge_native_leftovers(api, repo_id: str, token: str) -> int:
    """Delete Hub leftovers that still exist locally (upload delete_patterns skips those)."""
    from huggingface_hub import CommitOperationDelete

    info = api.repo_info(repo_id, repo_type="model", token=token)
    names = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
    doomed = remote_native_leftovers(names)
    if not doomed:
        return 0
    ops = [CommitOperationDelete(path_in_repo=n) for n in doomed]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=ops,
        commit_message="Remove pack-time ONNX / legacy GGUF leftovers (Hub is .rlxp only)",
        token=token,
    )
    print(f"purged {len(doomed)} leftover file(s) from {repo_id}")
    return len(doomed)


def load_mod():
    spec = importlib.util.spec_from_file_location("prepare_weights_hf", PREPARE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--cards-only", action="store_true", help="upload README/LICENSE/.gitattributes only")
    ap.add_argument("--prepare", action="store_true", help="cleanup + regenerate cards first")
    ap.add_argument("--only", type=str, default="", help="comma-separated leaf names or rel paths")
    ap.add_argument("--org", type=str, default="")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    mod = load_mod()
    org, models = mod.ORG, mod.MODELS
    if args.org:
        org = args.org

    if args.prepare:
        mod.cleanup_weights_tree()
        for rel, meta in sorted(models.items()):
            mod.prepare_one(rel, meta)

    only = {x.strip() for x in args.only.split(",") if x.strip()}

    try:
        from huggingface_hub import HfApi, create_repo, get_token, whoami
    except ImportError:
        print("install huggingface_hub (e.g. .venv-hf)", file=sys.stderr)
        return 1

    token = get_token()
    if not token:
        print("Not logged in to Hugging Face.\n  hf auth login", file=sys.stderr)
        return 2

    me = whoami(token=token)
    print(f"logged in as {me.get('name')} (org target: {org})")

    api = HfApi(token=token)
    jobs = []
    for rel, meta in sorted(models.items()):
        leaf = Path(rel).name
        repo_name = meta.get("repo_name") or leaf
        if only and rel not in only and leaf not in only and repo_name not in only:
            continue
        path = WEIGHTS / rel
        card_only = bool(meta.get("card_only"))
        # Alias / redirect cards (e.g. melotts): still push README on full sync.
        if meta.get("skip_upload") or card_only:
            jobs.append((rel, f"{org}/{repo_name}", path, meta, True))
            continue
        if not path.is_dir() or path.is_symlink():
            print(f"skip missing/symlink {rel}")
            continue
        jobs.append((rel, f"{org}/{repo_name}", path, meta, False))

    print(f"{len(jobs)} repos to process")
    ok = fail = skipped = 0
    for rel, repo_id, path, meta, card_only in jobs:
        mode = "cards" if (args.cards_only or card_only) else "folder"
        print(f"\n=== {rel} -> {repo_id} ({meta['kind']}, {mode}) ===")
        if args.dry_run:
            print(f"dry-run: would sync {path} ({mode})")
            skipped += 1
            continue
        try:
            create_repo(
                repo_id,
                repo_type="model",
                private=args.private,
                exist_ok=True,
                token=token,
            )
            if args.skip_existing and mode == "folder":
                info = api.repo_info(repo_id, repo_type="model")
                siblings = getattr(info, "siblings", None) or []
                weightish = [
                    s
                    for s in siblings
                    if any(
                        str(getattr(s, "rfilename", "")).endswith(ext)
                        for ext in (
                            ".safetensors",
                            ".gguf",
                            ".onnx",
                            ".bin",
                            ".pt",
                            ".rlxp",
                            ".rlxpack",
                        )
                    )
                ]
                if weightish:
                    print(f"skip-existing: {len(weightish)} weight files already on Hub")
                    skipped += 1
                    continue

            t0 = time.time()
            if args.cards_only or card_only:
                scan = path
                if path.is_symlink() or not path.is_dir():
                    # Alias cards (melotts) point at tiny-tts assets for file discovery.
                    scan = WEIGHTS / "tts" / "tiny-tts-rlx"
                readme = mod.render_readme(rel, meta, scan)
                api.upload_file(
                    path_or_fileobj=readme.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                )
                if path.is_dir() and not path.is_symlink():
                    for name in ("LICENSE", ".gitattributes"):
                        src = path / name
                        if src.is_file():
                            api.upload_file(
                                path_or_fileobj=str(src),
                                path_in_repo=name,
                                repo_id=repo_id,
                                repo_type="model",
                                token=token,
                            )
            else:
                kwargs = {
                    "folder_path": str(path),
                    "repo_id": repo_id,
                    "repo_type": "model",
                    "token": token,
                    "ignore_patterns": ignore_for(meta),
                }
                if is_native_rlxp(meta):
                    kwargs["delete_patterns"] = DELETE_NATIVE_RLXP
                api.upload_folder(**kwargs)
                if is_native_rlxp(meta):
                    purge_native_leftovers(api, repo_id, token)
            dt = time.time() - t0
            print(f"synced in {dt:.1f}s → https://huggingface.co/{repo_id}")
            ok += 1
        except Exception as e:
            print(f"FAIL {repo_id}: {type(e).__name__}: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone: ok={ok} fail={fail} skipped={skipped}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
