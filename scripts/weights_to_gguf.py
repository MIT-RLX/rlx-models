#!/usr/bin/env python3
"""Add a GGUF sibling into each weights/* model dir, then optionally upload.

Strategy (per dir):
  1. Skip if a .gguf already exists
  2. If HUB_GGUF maps this model → download that file into the dir
  3. Else pack all local .safetensors (F32) into `<name>.f16.gguf`
  4. Else extract ONNX initializers → temp safetensors → GGUF (small/medium only)

Usage:
  python3 scripts/weights_to_gguf.py              # convert all
  python3 scripts/weights_to_gguf.py --only snac-24khz,dinov2
  python3 scripts/weights_to_gguf.py --upload     # also push new .gguf to Hub
  python3 scripts/weights_to_gguf.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

# Preferred community / official GGUF to fetch into the staging dir.
# Values: (hub_repo, filename_or_glob_substring)
HUB_GGUF: dict[str, tuple[str, str]] = {
    "lm/qwen3-0.6b": ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
    "tts/orpheus": ("unsloth/orpheus-3b-0.1-ft-GGUF", "orpheus-3b-0.1-ft-Q4_K_M.gguf"),
    "tts/sesame": ("ggml-org/sesame-csm-1b-GGUF", "sesame-csm-backbone.gguf"),
    "tts/miotts": ("Aratako/MioTTS-GGUF", "MioTTS-0.6B-Q4_K_M.gguf"),
    "tts/miratts": ("mradermacher/MiraTTS-GGUF", "MiraTTS.Q4_K_M.gguf"),
    "tts/chatterbox": ("hans00/Chatterbox-TTS-GGUF", "t3-q4_k_m.gguf"),
    "tts/parlertts": ("ecyht2/parler-tts-mini-v1-GGUF", "parler-tts-mini-v1-Q4_0.gguf"),
    "tts/kokoro-82m": ("cstr/kokoro-82m-GGUF", "kokoro-82m-q8_0.gguf"),
    "tts/f5tts": ("cstr/f5-tts-GGUF", "f5-tts-v1-base-f16.gguf"),
    # Prefer RLX packs via `just export-*-gguf` / crate gguf_bundle — not these
    # community LM/codec-only GGUFs (not loadable by rlx-moss-nano / rlx-soprano).
    # "tts/moss-nano": ("hans00/MOSS-TTS-Nano-GGUF", "codec-q4_k_m.gguf"),
    # "tts/soprano": ("hans00/Soprano-1.1-80M-GGUF", "Soprano-1.1-80M.Q4_K_M.gguf"),
    "tts/zonos": ("cstr/zonos-v0.1-transformer-GGUF", "zonos-v0.1-transformer-f16.gguf"),
    "tts/neutts": ("neuphonic/neutts-air-q4-gguf", "neutts-air-Q4_0.gguf"),
    "tts/inflect-nano-rlx": ("remixerdec/Inflect-Nano-v1-GGUF", "acoustic/inflect_acoustic_q4_k.gguf"),
    "tts/snac-24khz": ("cstr/snac-24khz-GGUF", "snac-24khz.gguf"),
}


def load_prepare():
    spec = importlib.util.spec_from_file_location(
        "prepare_weights_hf", ROOT / "scripts" / "prepare_weights_hf.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def existing_gguf(path: Path) -> list[Path]:
    return sorted(path.rglob("*.gguf"))


def _to_f16(v: np.ndarray) -> np.ndarray | None:
    """Normalize any numeric ndarray to float16."""
    dt = str(getattr(v.dtype, "name", v.dtype))
    if v.dtype == np.float16 or dt == "float16":
        return np.ascontiguousarray(v)
    if dt in ("bfloat16", "bf16"):
        # Prefer torch (handles bf16); fall back to ml_dtypes / uint16 bitcast.
        try:
            import torch

            t = torch.from_numpy(v.view(np.uint16)).view(torch.bfloat16)
            return t.float().numpy().astype(np.float16)
        except Exception:
            pass
        try:
            from ml_dtypes import bfloat16 as bfloat16_t

            return np.asarray(v, dtype=bfloat16_t).astype(np.float32).astype(np.float16)
        except Exception:
            # Interpret raw bits as bf16 → f32 via float32 bit trick is lossy; skip
            u = v.view(np.uint16).astype(np.uint32) << 16
            return np.ascontiguousarray(u.view(np.float32).astype(np.float16))
    if np.issubdtype(v.dtype, np.floating) or np.issubdtype(v.dtype, np.integer):
        return np.ascontiguousarray(v.astype(np.float16))
    return None


def load_safetensors(paths: list[Path]) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    out: dict[str, np.ndarray] = {}
    for p in paths:
        prefix = p.stem if len(paths) > 1 else ""
        loaded: dict[str, np.ndarray] = {}
        # Prefer torch so bfloat16 tensors convert cleanly.
        try:
            with safe_open(str(p), framework="pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    loaded[k] = t.detach().float().cpu().numpy()
        except Exception:
            with safe_open(str(p), framework="np") as f:
                for k in f.keys():
                    loaded[k] = np.asarray(f.get_tensor(k))
        for k, raw in loaded.items():
            arr = _to_f16(np.asarray(raw))
            if arr is None:
                continue
            key = f"{prefix}.{k}" if prefix else k
            out[key] = arr
        del loaded
    return out


def pack_arrays(out: Path, arch: str, tensors: dict[str, np.ndarray]) -> None:
    import gguf

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out), arch)
    writer.add_string("general.name", out.stem)
    writer.add_string("general.description", f"RLX staging GGUF ({arch})")
    for name, arr in tensors.items():
        if not isinstance(arr, np.ndarray):
            continue
        arr = _to_f16(arr)
        if arr is None:
            continue
        writer.add_tensor(name.replace("/", "."), arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

def onnx_to_arrays(onnx_paths: list[Path], max_bytes: int = 2_000_000_000) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    total = sum(p.stat().st_size for p in onnx_paths)
    if total > max_bytes:
        raise RuntimeError(f"onnx set too large for inline pack ({total/1e9:.1f}G > {max_bytes/1e9:.1f}G)")
    out: dict[str, np.ndarray] = {}
    for p in onnx_paths:
        # Skip external-data-only shells when .onnx_data sibling is huge handled via load
        model = onnx.load(str(p), load_external_data=True)
        for init in model.graph.initializer:
            name = init.name or f"unnamed_{len(out)}"
            key = f"{p.stem}.{name}"
            out[key] = numpy_helper.to_array(init)
    if not out:
        raise RuntimeError("no ONNX initializers found")
    return out


def try_hub_download(rel: str, dest_dir: Path, dry: bool) -> Path | None:
    if rel not in HUB_GGUF:
        return None
    repo, filename = HUB_GGUF[rel]
    dest = dest_dir / Path(filename).name
    nested = dest_dir / filename
    for candidate in (dest, nested):
        if candidate.exists() and candidate.stat().st_size > 0:
            print(f"  hub gguf already present: {candidate.relative_to(dest_dir)}")
            return candidate
    print(f"  fetch {repo}/{filename}")
    if dry:
        return dest
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    got = Path(path)
    # Flatten nested downloads to dest_dir/<basename>
    if got.exists() and got.resolve() != dest.resolve():
        if not dest.exists():
            dest.write_bytes(got.read_bytes())
        return dest
    return got if got.exists() else None


def convert_one(rel: str, path: Path, arch: str, dry: bool, force: bool = False) -> Path | None:
    have = existing_gguf(path)
    if have and not force:
        print(f"  already has GGUF: {have[0].relative_to(path)}")
        return have[0]
    if force:
        for p in path.glob("*.f16.gguf"):
            print(f"  remove {p.name} (--force)")
            if not dry:
                p.unlink()

    # 1) Hub
    try:
        hub = try_hub_download(rel, path, dry=dry)
        if hub and (dry or hub.exists()):
            return hub
    except Exception as e:
        print(f"  hub fetch failed ({e}); falling back to local pack")

    # 2) local pack: safetensors and/or onnx
    sts = sorted(path.rglob("*.safetensors"))
    sts = [p for p in sts if "fixtures" not in p.parts]
    onnxs = sorted(path.rglob("*.onnx"))
    onnxs = [p for p in onnxs if "fixtures" not in p.parts]

    tensors: dict[str, np.ndarray] = {}
    if sts:
        print(f"  load {len(sts)} safetensors")
        if not dry:
            tensors.update(load_safetensors(sts))
    if onnxs:
        # Skip huge ONNX sets when we already have a substantial safetensors pack
        st_bytes = sum(p.stat().st_size for p in sts) if sts else 0
        if st_bytes < 50_000_000 or not sts:
            print(f"  load {len(onnxs)} onnx")
            if not dry:
                tensors.update(onnx_to_arrays(onnxs))
        else:
            print(f"  skip onnx pack (safetensors already {st_bytes/1e6:.0f} MB)")

    if tensors or (dry and (sts or onnxs)):
        out = path / f"{path.name}.f16.gguf"
        print(f"  pack → {out.name}")
        if dry:
            return out
        if out.exists():
            out.unlink()
        pack_arrays(out, arch=arch, tensors=tensors)
        print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(tensors)} tensors)")
        return out

    print("  no safetensors/onnx/hub source — skip")
    return None


def upload_gguf(repo_id: str, gguf_path: Path, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"  upload {gguf_path.name} → {repo_id}")
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild local .f16.gguf even if a .gguf already exists",
    )
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    prep = load_prepare()
    org = prep.ORG
    token = None
    if args.upload:
        from huggingface_hub import get_token

        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            # keychain fallback
            import subprocess

            out = subprocess.check_output(
                ["git", "credential-osxkeychain", "get"],
                input=b"protocol=https\nhost=huggingface.co\n\n",
            ).decode()
            d = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
            token = d.get("password")
        if not token:
            print("need HF_TOKEN for --upload", file=sys.stderr)
            return 2

    ok = skip = fail = 0
    results = []
    for rel, meta in sorted(prep.MODELS.items()):
        leaf = Path(rel).name
        repo_name = meta.get("repo_name") or leaf
        if only and rel not in only and leaf not in only and repo_name not in only:
            continue
        path = WEIGHTS / rel
        if not path.is_dir() or path.is_symlink():
            print(f"skip missing {rel}")
            skip += 1
            continue
        print(f"\n=== {rel} ===")
        try:
            gguf = convert_one(rel, path, arch=f"rlx-{leaf}", dry=args.dry_run, force=args.force)
            if gguf is None:
                fail += 1
                continue
            ok += 1
            results.append((rel, str(gguf), f"{org}/{repo_name}"))
            if args.upload and not args.dry_run and gguf.exists():
                # only upload if this file is new-ish under the model dir (always push)
                upload_gguf(f"{org}/{repo_name}", gguf, token=token)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            fail += 1

    summary = ROOT / "weights" / "GGUF_STATUS.json"
    if not args.dry_run:
        summary.write_text(json.dumps([{"rel": r, "gguf": g, "repo": repo} for r, g, repo in results], indent=2))
        print(f"\nwrote {summary}")
    print(f"\nDone: ok={ok} fail={fail} skip={skip}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
