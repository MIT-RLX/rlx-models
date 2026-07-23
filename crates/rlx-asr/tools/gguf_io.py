#!/usr/bin/env python3
# RLX — GPLv3.
# Minimal GGUF v3 reader for rlx-asr (F32 / I8 tensors + string metadata).
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

GGUF_MAGIC = 0x46554747  # GGUF little-endian
GGML_F32 = 0
GGML_I8 = 24

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


def resolve_gguf(root: Path | None = None) -> Path | None:
    from audio_io import asr_dir

    root = root or asr_dir()
    env = __import__("os").environ.get("RLX_ASR_GGUF")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for name in ("model.gguf", "asr.gguf", "rlx-asr.gguf"):
        p = root / name
        if p.is_file():
            return p
    return None


def load_encoder_pack(gguf_path: Path | None = None) -> dict[str, np.ndarray]:
    """Load folded encoder tensors from GGUF (`encoder.*` keys → unprefixed)."""
    path = gguf_path or resolve_gguf()
    if path is None:
        raise FileNotFoundError("model.gguf not found (run: just asr-pack-gguf)")
    g = GgufFile(path)
    out: dict[str, np.ndarray] = {}
    prefix = "encoder."
    for name in g.tensors:
        if not name.startswith(prefix):
            continue
        try:
            out[name[len(prefix) :]] = g.tensor_f32(name)
        except ValueError:
            continue
    return out
