"""Restricted `torch.save` reader for the HTR reference scripts.

PyLaia, DAN and Kraken training checkpoints pickle more than a state dict:
Lightning callbacks, jsonargparse types, numpy scalars. `torch.load(...,
weights_only=True)` rejects them and `weights_only=False` runs whatever the
pickle names. This loader rebuilds tensors and plain containers and turns
every other global into an inert `Stub` that records its constructor
arguments, so no code from the checkpoint ever runs.
"""

import collections
import io
import pickle
import zipfile

import torch

_DT = {
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16,
    "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}


class Stub:
    """Placeholder for a non-allowlisted global; keeps args and state."""

    def __new__(cls, *a, **k):
        o = object.__new__(cls)
        o.a = a
        o.st = None
        return o

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.st = st

    def __repr__(self):
        return f"{getattr(self, 'qualname', '?')}{self.a}"


def unwrap(v):
    """Strip jsonargparse-style wrappers (`PositiveInt(3)`) down to plain values."""
    if isinstance(v, Stub):
        return unwrap(v.a[0]) if len(v.a) == 1 else v
    if isinstance(v, list):
        return [unwrap(x) for x in v]
    if isinstance(v, tuple):
        return tuple(unwrap(x) for x in v)
    if isinstance(v, dict):
        return {k: unwrap(x) for k, x in v.items()}
    return v


def _rebuild(st, off, size, stride, *_):
    raw, dt = st
    flat = torch.frombuffer(bytearray(raw), dtype=dt) if raw else torch.empty(0, dtype=dt)
    if len(size) == 0:
        return flat[off].clone()
    return torch.as_strided(flat, size, stride, off).clone()


class _Unpickler(pickle.Unpickler):
    def __init__(self, data, zf, root):
        super().__init__(io.BytesIO(data))
        self.zf = zf
        self.root = root

    def find_class(self, mod, name):
        if (mod, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild
        if (mod, name) == ("torch._utils", "_rebuild_parameter"):
            return lambda data, *_: data
        if mod == "torch" and name in _DT:
            return name
        if (mod, name) == ("torch", "Size"):
            return torch.Size
        if mod == "builtins" and name in ("set", "slice", "frozenset"):
            return {"set": set, "slice": slice, "frozenset": frozenset}[name]
        return type(name, (Stub,), {"qualname": f"{mod}.{name}"})

    def persistent_load(self, pid):
        _, st, key, _loc, _n = pid
        dt = _DT.get(st, torch.float32) if isinstance(st, str) else torch.float32
        return (self.zf.read(f"{self.root}/data/{key}"), dt)


def load(path):
    """Unpickle a `torch.save` ZIP without executing any of its globals."""
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    root = pkl.rsplit("/", 1)[0]
    return _Unpickler(zf.read(pkl), zf, root).load()


def state_dict(obj, prefix):
    """Tensors under `prefix` (e.g. `state_dict.model.`), prefix stripped."""
    out = {}

    def walk(d, pre):
        for k, v in d.items():
            key = f"{pre}{k}"
            if isinstance(v, torch.Tensor):
                out[key] = v
            elif isinstance(v, dict):
                walk(v, key + ".")

    walk(obj, "")
    return {k[len(prefix):]: v for k, v in out.items() if k.startswith(prefix)}
