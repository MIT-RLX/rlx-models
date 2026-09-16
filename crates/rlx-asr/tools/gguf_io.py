#!/usr/bin/env python3
# RLX — GPLv3.
# Minimal GGUF v3 + flat `.rlxp` reader for rlx-asr (F32 tensors + metadata).
from __future__ import annotations

import json
import struct
import zstandard as zstd
from pathlib import Path
from typing import Any

import numpy as np

GGUF_MAGIC = 0x46554747  # GGUF little-endian
RLXP_MAGIC = b"RLXPFLAT"
GGML_F32 = 0
GGML_I8 = 24
DATA_ALIGN = 64

_VALUE_READERS = {
    0: ("B", 1),  # u8
    1: ("b", 1),  # i8
    2: ("H", 2),  # u16
    3: ("h", 2),  # i16
    4: ("I", 4),  # u32
    5: ("i", 4),  # i32
    6: ("f", 4),  # f32
    7: ("?", 1),  # bool
    10: ("Q", 8),  # u64
    11: ("q", 8),  # i64
    12: ("d", 8),  # f64
}


class RlxpFile:
    """Flat `.rlxp` (RLXPFLAT v2) — hot f32 tensors + zstd sidecars."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._data = self.path.read_bytes()
        if self._data[:8] != RLXP_MAGIC:
            raise ValueError(f"not RLXPFLAT: {self.path}")
        _ver, _flags, toc_len = struct.unpack_from("<IIQ", self._data, 8)
        self._toc: dict[str, Any] = json.loads(self._data[24 : 24 + toc_len])
        hdr = 24 + toc_len
        pad = (DATA_ALIGN - (hdr % DATA_ALIGN)) % DATA_ALIGN
        self._data_base = hdr + pad
        self._strings: list[str] = self._toc.get("strings") or []
        self._tensors: dict[str, dict[str, Any]] = {}
        for t in self._toc.get("tensors") or []:
            name = t.get("name") or ""
            if not name and "name_i" in t:
                name = self._strings[int(t["name_i"])]
            self._tensors[name] = t
        self._sidecars: dict[str, dict[str, Any]] = {
            s["id"]: s for s in self._toc.get("sidecars") or []
        }

    def has(self, name: str) -> bool:
        return name in self._tensors

    def tensor_f32(self, name: str) -> np.ndarray:
        t = self._tensors.get(name)
        if t is None:
            raise KeyError(name)
        if t.get("scheme", "f32") != "f32":
            raise ValueError(f"tensor {name}: scheme {t.get('scheme')}")
        off = int(t["offset"])
        ln = int(t["length"])
        raw = self._data[self._data_base + off : self._data_base + off + ln]
        shape = [int(x) for x in t.get("shape") or []]
        n = int(np.prod(shape)) if shape else ln // 4
        arr = np.frombuffer(raw, dtype="<f4", count=n)
        return np.array(arr, dtype=np.float32).reshape(shape) if shape else arr.copy()

    def sidecar_bytes(self, stem: str) -> bytes:
        key = stem if stem in self._sidecars else f"{stem}.json"
        sc = self._sidecars.get(stem) or self._sidecars.get(key)
        if sc is None:
            raise KeyError(stem)
        off = int(sc["offset"])
        ln = int(sc["length"])
        raw = bytes(self._data[self._data_base + off : self._data_base + off + ln])
        if raw[:4] == b"(\xb5/\xfd":
            return zstd.ZstdDecompressor().decompress(raw, max_output_size=10_000_000)
        return raw

    def units(self) -> list[str]:
        raw = self.sidecar_bytes("units.txt")
        return [
            ln.split()[0]
            for ln in raw.decode("utf-8", errors="replace").splitlines()
            if ln.strip()
        ]


class GgufFile:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, dict[str, Any]] = {}
        self._data: np.memmap | None = None
        self._data_offset = 0
        self._load()

    def _load(self) -> None:
        raw = self.path.read_bytes()
        if len(raw) < 24:
            raise ValueError(f"truncated GGUF: {self.path}")
        magic, version, n_tensors, n_kv = struct.unpack_from("<IIII", raw, 0)
        # Actually header is: magic u32, version u32, n_tensors u64, n_kv u64
        magic, version = struct.unpack_from("<II", raw, 0)
        n_tensors, n_kv = struct.unpack_from("<QQ", raw, 8)
        if magic != GGUF_MAGIC:
            raise ValueError(f"bad GGUF magic in {self.path}")
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        off = 24
        for _ in range(n_kv):
            key, off = self._read_string(raw, off)
            val, off = self._read_value(raw, off)
            self.metadata[key] = val
        alignment = int(self.metadata.get("general.alignment", 32))
        infos: list[tuple[str, list[int], int, int]] = []
        for _ in range(n_tensors):
            name, off = self._read_string(raw, off)
            n_dims = struct.unpack_from("<I", raw, off)[0]
            off += 4
            shape = []
            for _d in range(n_dims):
                shape.append(struct.unpack_from("<Q", raw, off)[0])
                off += 8
            dtype = struct.unpack_from("<I", raw, off)[0]
            off += 4
            data_off = struct.unpack_from("<Q", raw, off)[0]
            off += 8
            infos.append((name, [int(s) for s in shape], dtype, int(data_off)))
        # pad to alignment
        pad = (alignment - (off % alignment)) % alignment
        data_start = off + pad
        self._data_offset = data_start
        self._data = np.memmap(self.path, mode="r", dtype=np.uint8)
        for name, shape, dtype, rel in infos:
            self.tensors[name] = {
                "shape": shape,
                "dtype": dtype,
                "offset": data_start + rel,
            }

    @staticmethod
    def _read_string(buf: bytes, off: int) -> tuple[str, int]:
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        s = buf[off : off + n].decode("utf-8", errors="replace")
        return s, off + n

    def _read_value(self, buf: bytes, off: int) -> tuple[Any, int]:
        (ty,) = struct.unpack_from("<I", buf, off)
        off += 4
        if ty == 8:  # string
            return self._read_string(buf, off)
        if ty == 9:  # array
            (elem_ty,) = struct.unpack_from("<I", buf, off)
            off += 4
            (n,) = struct.unpack_from("<Q", buf, off)
            off += 8
            items = []
            for _ in range(n):
                if elem_ty == 8:
                    s, off = self._read_string(buf, off)
                    items.append(s)
                else:
                    v, off = self._read_scalar(buf, off, elem_ty)
                    items.append(v)
            return items, off
        return self._read_scalar(buf, off, ty)

    @staticmethod
    def _read_scalar(buf: bytes, off: int, ty: int) -> tuple[Any, int]:
        if ty not in _VALUE_READERS:
            raise ValueError(f"unsupported GGUF value type {ty}")
        fmt, nbytes = _VALUE_READERS[ty]
        (v,) = struct.unpack_from("<" + fmt, buf, off)
        return v, off + nbytes

    def tensor_f32(self, name: str) -> np.ndarray:
        info = self.tensors.get(name)
        if info is None:
            raise KeyError(name)
        shape = info["shape"]
        n = int(np.prod(shape)) if shape else 0
        dtype = info["dtype"]
        start = info["offset"]
        assert self._data is not None
        if dtype == GGML_F32:
            raw = np.frombuffer(self._data[start : start + n * 4], dtype="<f4")
            return np.array(raw, dtype=np.float32).reshape(shape)
        if dtype == GGML_I8:
            raw = np.frombuffer(self._data[start : start + n], dtype=np.int8)
            return np.array(raw, dtype=np.float32).reshape(shape)
        raise ValueError(f"tensor {name}: unsupported ggml dtype {dtype}")

    def has(self, name: str) -> bool:
        return name in self.tensors


def resolve_pack(root: Path | None = None) -> Path | None:
    """Prefer `model.rlxp`, then legacy GGUF under an ASR root."""
    from audio_io import asr_dir

    root = root or asr_dir()
    env = __import__("os").environ.get("RLX_ASR_GGUF")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for name in ("model.rlxp", "model.gguf", "asr.rlxp", "asr.gguf", "rlx-asr.gguf"):
        p = root / name
        if p.is_file():
            return p
    return None


def resolve_gguf(root: Path | None = None) -> Path | None:
    from audio_io import asr_dir

    root = root or asr_dir()
    env = __import__("os").environ.get("RLX_ASR_GGUF")
    if env:
        p = Path(env)
        if p.is_file() and p.suffix.lower() != ".rlxp":
            return p
    for name in ("model.gguf", "asr.gguf", "rlx-asr.gguf"):
        p = root / name
        if p.is_file():
            return p
    pack = resolve_pack(root)
    if pack is not None and pack.suffix.lower() == ".gguf":
        return pack
    return None


def open_pack(path: Path | None = None) -> GgufFile | RlxpFile:
    from audio_io import asr_dir

    path = path or resolve_pack(asr_dir())
    if path is None:
        raise FileNotFoundError("model.rlxp / model.gguf not found under ASR dir")
    if path.suffix.lower() == ".rlxp" or path.read_bytes()[:8] == RLXP_MAGIC:
        return RlxpFile(path)
    return GgufFile(path)


def load_encoder_pack(pack_path: Path | None = None) -> dict[str, np.ndarray]:
    """Load folded encoder tensors (`encoder.*` → unprefixed keys)."""
    path = pack_path or resolve_pack()
    if path is None:
        raise FileNotFoundError("model.rlxp / model.gguf not found (run: just fetch-rlx-asr)")
    pack = open_pack(path)
    out: dict[str, np.ndarray] = {}
    prefix = "encoder."
    if isinstance(pack, RlxpFile):
        for name in pack._tensors:
            if not name.startswith(prefix):
                continue
            try:
                out[name[len(prefix) :]] = pack.tensor_f32(name)
            except (ValueError, KeyError):
                continue
        return out
    for name in pack.tensors:
        if not name.startswith(prefix):
            continue
        try:
            out[name[len(prefix) :]] = pack.tensor_f32(name)
        except ValueError:
            continue
    return out
